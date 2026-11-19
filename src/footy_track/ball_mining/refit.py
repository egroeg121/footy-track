"""Refit of a training round (design §3.2 "Adaptive within a round", §6 Loop 2 "Refit"): a **library plus CLI that run
outside the labeller server** (``ball_round refit --job <ref>``; Prefect flow ``ball-refit`` in the jobs repo).

The labeller only *triggers* it: ``round.ROUND_HOOKS`` attach a hook of action type ``prefect`` (every 50 verdicts) that
queues the ``ball-refit/ball-refit`` deployment with ``{job_id, hook, trigger, dataset, labeller_url?}``; the flow runs the CLI,
which talks back through the labeller's API (``GET /api/jobs/<ref>/results``, ``POST /api/jobs/<ref>/reorder``). Nothing
here is registered as an in-process hook action.

Each run, over **every training verdict of the dataset** (all ``ball-round/*`` jobs; cards carry ``sampling`` as a hidden
key, which the job results hand back as ``labels[].meta``):

1. per-bucket hit rates (share of decided cards that were the real ball), for reporting;
2. a Platt fit ``P(real) = sigmoid(a * logit(p_real) + b)`` against the verdicts, using **only R and A** cards (the
   bucket-U / M / N cards were chosen for being unusual, so they would bias it); U / M / N only feed the report;
3. Learn-then-Test (``threshold.py``) on the **A** verdicts of the current scorer version, globally and per match;
4. the gold check: if planted-wrong (G) cards are accepted more than 10 % of the time the round is flagged;
5. an immutable ``<marks>/ball_mining/state/vNNN.json`` (same data again reuses the latest version, as
   ``broadcast_calibrate`` does);
6. unless it is the close run (``close.py``): a new order for the unanswered cards, POSTed to ``/api/jobs/<ref>/reorder``
   with ``if_rev`` (the ``rev`` read together with the unanswered positions; a 409 rev conflict re-reads and retries, up
   to ``REORDER_TRIES`` times). Inside the slots the U cards already hold they are re-dealt by ``|p_cal - 0.5|``
   ascending (most uncertain first), and the M cards inside the M slots by ``p_cal`` descending (most plausible gap
   first). R / A / N / G cards never move, so audits and gold stay interleaved at random and answered cards are
   untouched (the API never moves answered cards).

Any other API error propagates (fail closed): the state file may exist, the job is simply not reordered.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from footy_track.ball_mining.threshold import (
    CORRECT,
    WRONG,
    audit_rows,
    learn_then_test,
    learn_then_test_per_match,
)
from footy_track.labeller import datasets, jobs, jobs_client
from footy_track.labeller.broadcast_review import fit_platt, logit, sigmoid

LOGGER = logging.getLogger(__name__)

STATE_SCHEMA = "ball-mining-state/v1"
ROUND_KEY_PREFIX = "ball-round/"
REORDER_TRIES = 3
REORDER_BY = "ball_round_refit"
LIVE = ("open", "in_progress")
CAL_MIN_PER_CLASS = 10
GOLD_ACCEPT_FLAG = 0.10
_STATE_NAME = re.compile(r"^v(\d{3,})\.json$")


def state_dir(marks_dir: str | Path) -> Path:
    return Path(marks_dir) / "ball_mining" / "state"


def _versions(marks_dir: str | Path) -> list[tuple[int, Path]]:
    d = state_dir(marks_dir)
    if not d.is_dir():
        return []
    return sorted(
        (int(m.group(1)), p) for p in d.iterdir() if (m := _STATE_NAME.match(p.name))
    )


def latest_state(marks_dir: str | Path) -> dict | None:
    for _, p in reversed(_versions(marks_dir)):
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("schema") == STATE_SCHEMA:
            return doc
    return None


# ---------------------------------------------------------------------------
# Verdict rows
# ---------------------------------------------------------------------------


def is_round_job(job: dict) -> bool:
    """A job the training miner created (``ball-round/<date>/<n>``); cancelled ones are skipped by the callers."""
    return job.get("kind") == "ball_check" and str(job.get("key") or "").startswith(
        ROUND_KEY_PREFIX
    )


def rows_from_results(doc: dict) -> list[dict]:
    """One row per live verdict on a training card of a ``labelling-job-results/v1`` document: ``bucket, band, match,
    p_real, pi, scorer_version, verdict, clip, frame, job, gold_kind`` (``clip`` is the pool's clip name: ``pool_clip`` when the
    card was linked under a safe name)."""
    rows = []
    for lab in doc.get("labels") or []:
        meta = lab.get("meta") or {}
        s = meta.get("sampling") or {}
        if s.get("regime") != "train":
            continue
        rows.append({
            "bucket": s.get("bucket"), "band": s.get("band"), "match": meta.get("match") or s.get("match"),
            "p_real": meta.get("p_real"), "pi": s.get("pi"), "scorer_version": meta.get("scorer_version") or s.get("scorer_version"),
            "verdict": lab.get("verdict"), "clip": meta.get("pool_clip") or lab.get("clip"), "frame": lab.get("frame_index"), "job": doc["job"]["id"],
            "gold_kind": (s.get("gold") or {}).get("kind"),
        })  # fmt: skip
    return rows


def collect_rows(client, dataset: str, *, current: dict | None = None) -> list[dict]:
    """Verdict rows of every live (not cancelled) round job of ``dataset``, read through the labeller API. ``current`` is
    the already fetched results document of one of them (not fetched twice)."""
    rows = []
    for j in client.list_jobs(kind="ball_check", dataset=dataset, limit="5000"):
        if not is_round_job(j) or j.get("status") == "cancelled":
            continue
        doc = (
            current
            if current is not None and current["job"]["id"] == j["id"]
            else client.results(j["id"])
        )
        rows.extend(rows_from_results(doc))
    if current is not None and not any(r["job"] == current["job"]["id"] for r in rows):
        rows.extend(
            rows_from_results(current)
        )  # listed under another status filter / just created
    return rows


def unanswered_from_results(doc: dict) -> list[tuple[int, dict]]:
    """``(position, item)`` of the cards not fully answered, in job order (the rule ``job_edit.answered_positions`` uses)."""
    return [
        (p["index"], p["item"])
        for p in doc.get("items") or []
        if not (p["total"] and p["done"] >= p["total"])
    ]


def hit_rates(rows: list[dict]) -> dict[str, dict]:
    """``{bucket: {n, correct, wrong, unsure, hit_rate}}``; hit_rate = correct / (correct + wrong)."""
    c: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        v = r.get("verdict")
        k = (
            "correct"
            if v in CORRECT
            else "wrong"
            if v in WRONG
            else "unsure"
            if v == "unsure"
            else None
        )
        if k:
            c[r.get("bucket") or "?"][k] += 1
    out = {}
    for b, cnt in sorted(c.items()):
        decided = cnt["correct"] + cnt["wrong"]
        out[b] = {"n": decided + cnt["unsure"], "correct": cnt["correct"], "wrong": cnt["wrong"], "unsure": cnt["unsure"],
                  "hit_rate": round(cnt["correct"] / decided, 4) if decided else None}  # fmt: skip
    return out


def gold_check(rows: list[dict]) -> dict:
    """Planted-wrong cards accepted as they were shown, i.e. a plain ``ball`` (an over-accepting labeller). Flags a round
    above 10 %. ``corrected`` is the RIGHT answer to a shifted gold box (the labeller moved it onto the ball), and ``box_off`` /
    ``not_ball`` reject it, so only ``ball`` counts as accepting the planted error."""
    g = [
        r
        for r in rows
        if r.get("bucket") == "G"
        and (r.get("verdict") in CORRECT or r.get("verdict") in WRONG)
    ]
    accepted = sum(1 for r in g if r["verdict"] == "ball")
    rate = accepted / len(g) if g else None
    return {"n": len(g), "accepted_wrong": accepted, "accept_rate": None if rate is None else round(rate, 4),
            "flagged": bool(g) and len(g) >= 5 and rate > GOLD_ACCEPT_FLAG}  # fmt: skip


def fit_calibration(
    rows: list[dict], *, min_per_class: int = CAL_MIN_PER_CLASS
) -> dict:
    """Platt on ``logit(p_real)`` from R and A verdicts only. ``{"fitted": False, "noop": why}`` when too few."""
    use = [
        r
        for r in rows
        if r.get("bucket") in ("R", "A")
        and r.get("p_real") is not None
        and (r["verdict"] in CORRECT or r["verdict"] in WRONG)
    ]
    y = [1 if r["verdict"] in CORRECT else 0 for r in use]
    n_yes, n_no = sum(y), len(y) - sum(y)
    base = {"buckets_used": ["R", "A"], "n": len(y), "n_yes": n_yes, "n_no": n_no}
    if n_yes < min_per_class or n_no < min_per_class:
        return {
            "fitted": False,
            "noop": f"need >= {min_per_class} correct and >= {min_per_class} wrong R+A verdicts, have {n_yes} / {n_no}",
            **base,
        }
    a, b = fit_platt([logit(r["p_real"]) for r in use], y)
    if not a > 0:
        return {
            "fitted": False,
            "noop": f"degenerate fit (slope a={a:.3g} <= 0)",
            **base,
        }
    return {"fitted": True, "method": "platt on logit(p_real), smoothed targets", "a": a, "b": b,
            "boundary_p_real": sigmoid(-b / a), **base}  # fmt: skip


def calibrated(p: float, cal: dict | None) -> float:
    return (
        sigmoid(cal["a"] * logit(p) + cal["b"])
        if cal and cal.get("fitted")
        else float(p)
    )


def priority_order(unanswered: list[tuple[int, dict]], cal: dict | None) -> list[int]:
    """New order of the unanswered item positions (see the module docstring); every position appears once."""
    slots: dict[str, list[int]] = defaultdict(list)
    cards: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for pos, it in unanswered:
        b = (it.get("sampling") or {}).get("bucket")
        p = it.get("p_real")
        if b in ("U", "M") and p is not None:
            slots[b].append(pos)
            pc = calibrated(float(p), cal)
            cards[b].append((abs(pc - 0.5) if b == "U" else -pc, pos))
    fill: dict[int, int] = {}
    for b in ("U", "M"):
        ranked = [pos for _, pos in sorted(cards[b])]
        fill.update(zip(slots[b], ranked, strict=True))
    return [fill.get(pos, pos) for pos, _ in unanswered]


# ---------------------------------------------------------------------------
# State files
# ---------------------------------------------------------------------------


def write_state(marks_dir: str | Path, doc: dict) -> dict:
    """Store ``doc`` as the next ``vNNN.json`` unless the latest version saw the same verdict count (then return that)."""
    prev = latest_state(marks_dir)
    if (
        prev is not None
        and prev.get("n_verdicts") == doc["n_verdicts"]
        and prev.get("closed") == doc.get("closed")
    ):
        return {**prev, "reused": True}
    vs = _versions(marks_dir)
    v = (vs[-1][0] if vs else 0) + 1
    while True:
        doc = {**doc, "version": v}
        if jobs._write_atomic(
            state_dir(marks_dir) / f"v{v:03d}.json", doc, exclusive=True
        ):
            return {**doc, "reused": False}
        v += 1


def current_scorer(job: dict) -> str | None:
    c = Counter(
        str(it["scorer_version"])
        for it in job.get("items") or []
        if it.get("scorer_version")
    )
    return c.most_common(1)[0][0] if c else None


def compute_state(
    doc: dict,
    rows: list[dict],
    *,
    marks: Path,
    closed: bool = False,
    params: dict | None = None,
) -> tuple[dict, dict]:
    """Compute and store the state for the round whose results are ``doc``; returns ``(state, calibration)``."""
    params = params or {}
    job = doc["job"]
    scorer = current_scorer({"items": [p["item"] for p in doc.get("items") or []]})
    same = [r for r in rows if r.get("scorer_version") in (scorer, None)]
    cal = fit_calibration(
        same, min_per_class=int(params.get("min_per_class", CAL_MIN_PER_CLASS))
    )
    aud = audit_rows(same)
    kw = {
        "target": float(params.get("target", 0.98)),
        "delta": float(params.get("delta", 0.1)),
    }
    ltt = learn_then_test(aud, **kw)
    per_match = learn_then_test_per_match(aud, **kw)
    decided = [r for r in rows if r.get("verdict") in (*CORRECT, *WRONG)]
    state = {
        "schema": STATE_SCHEMA, "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "job_id": job["id"], "round": job.get("key"),
        "trigger": "close" if closed else "refit", "closed": closed, "scorer_version": scorer, "n_verdicts": len(decided),
        "n_rows_other_scorer": len(rows) - len(same), "hit_rates": hit_rates(rows), "gold": gold_check(rows), "calibration": cal,
        "threshold": {"global": ltt.as_dict(), "per_match": {m: r.as_dict() for m, r in per_match.items()}, "grid": [0.99, 0.50, 0.01],
                      "method": "learn-then-test, fixed sequence from high t, exact binomial", **kw},
        "auto_accept": {"threshold": ltt.effective_threshold, "scale": ltt.effective_scale, "fallback": ltt.fallback},
    }  # fmt: skip
    return write_state(marks, state), cal


def resolve_marks(dataset: str, marks_dir: str | Path | None) -> Path:
    return Path(marks_dir) if marks_dir else datasets.get(dataset).marks_dir


def _apply_order(client, ref: str, doc: dict, cal: dict | None, state: dict) -> dict:
    """POST the new order with ``if_rev``; on a rev conflict (a reorder / append landed since the read) re-read the job and
    recompute the order from the fresh unanswered cards, up to ``REORDER_TRIES`` times."""
    note = f"ball_mining state v{state['version']} ({state['n_verdicts']} verdicts)"
    for attempt in range(1, REORDER_TRIES + 1):
        if doc["job"].get("status") not in LIVE:
            return {"reorder": f"skipped, job is {doc['job'].get('status')}"}
        order = priority_order(unanswered_from_results(doc), cal)
        try:
            res = client.reorder(
                ref,
                order,
                by=REORDER_BY,
                note=note,
                if_rev=int(doc["job"].get("rev") or 0),
            )
        except jobs_client.JobsError as e:
            if e.status == 409 and isinstance(e.body, dict) and "rev" in e.body:
                LOGGER.info(
                    "reorder of %s hit a rev conflict (attempt %d): re-reading",
                    ref,
                    attempt,
                )
                doc = client.results(ref)
                continue
            raise
        return {"reordered": res["unanswered"], "changed": res["changed"], "rev": res["rev"], "attempts": attempt}  # fmt: skip
    return {"reorder": "rev_conflict", "attempts": REORDER_TRIES}


def summary(state: dict, cal: dict | None) -> dict:
    return {"version": state["version"], "n_verdicts": state["n_verdicts"], "reused": state["reused"], "closed": state["closed"],
            "hit_rates": {b: v["hit_rate"] for b, v in state["hit_rates"].items()}, "calibrated": bool(cal and cal.get("fitted")),
            "auto_accept": state["auto_accept"], "gold": state["gold"]}  # fmt: skip


def refit_round(
    ref: str,
    *,
    client=None,
    marks_dir: str | Path | None = None,
    params: dict | None = None,
    reorder: bool = True,
) -> dict:
    """Refit from the verdicts of the dataset's rounds, store ``state/vNNN.json`` and (``reorder``) POST the new order of the
    unanswered U / M cards of job ``ref`` (id or exact title). Returns a small JSON-able summary."""
    client = client or jobs_client
    doc = client.results(ref)
    rows = collect_rows(client, doc["job"]["dataset"], current=doc)
    marks = resolve_marks(doc["job"]["dataset"], marks_dir)
    state, cal = compute_state(doc, rows, marks=marks, params=params)
    out = {"job": doc["job"]["id"], "key": doc["job"].get("key"), **summary(state, cal)}
    if reorder:
        out.update(_apply_order(client, ref, doc, cal, state))
    return out
