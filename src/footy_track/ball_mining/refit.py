"""The ``ball_mining_refit`` labelling hook (design §3.2 "Adaptive within a round", §6 Loop 2 "Refit" and "Close").

Attached to a round's ``ball_check`` job (``round.py`` does this) as two hooks of one action::

    {"name": "refit", "trigger": {"every_n": 50},  "action": {"type": "builtin", "name": "ball_mining_refit"}}
    {"name": "close", "trigger": {"on_done": true}, "action": {"type": "builtin", "name": "ball_mining_refit", "params": {"close": true}}}

Each run, over **every training verdict of the dataset** (all rounds; cards carry ``sampling`` as a hidden key, which
the job results hand back as ``labels[].meta``):

1. per-bucket hit rates (share of decided cards that were the real ball), for reporting;
2. a Platt fit ``P(real) = sigmoid(a * logit(p_real) + b)`` against the verdicts, using **only R and A** cards (the
   bucket-U / M / N cards were chosen for being unusual, so they would bias it); U / M / N only feed the report;
3. Learn-then-Test (``threshold.py``) on the **A** verdicts of the current scorer version, globally and per match;
4. the gold check: if planted-wrong (G) cards are accepted more than 10 % of the time the round is flagged;
5. an immutable ``<marks>/ball_mining/state/vNNN.json`` (same data again reuses the latest version, as
   ``broadcast_calibrate`` does);
6. unless it is the ``close`` run: a new order for the unanswered cards. Inside the slots the U cards already hold they
   are re-dealt by ``|p_cal - 0.5|`` ascending (most uncertain first), and the M cards inside the M slots by ``p_cal``
   descending (most plausible gap first). R / A / N / G cards never move, so audits and gold stay interleaved at random
   and answered cards are untouched (``job_edit.reorder_unanswered``).

``ball_mining_refit`` is registered with ``hooks.register_action`` when ``hooks._load_builtins`` imports this module.
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
from footy_track.labeller import datasets, hooks, job_edit, jobs
from footy_track.labeller.broadcast_review import fit_platt, logit, sigmoid

LOGGER = logging.getLogger(__name__)

STATE_SCHEMA = "ball-mining-state/v1"
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


def is_mining_job(job: dict) -> bool:
    return job.get("kind") == "ball_check" and any(
        ((it.get("sampling") or {}).get("regime") == "train")
        for it in job.get("items") or []
    )


def collect_rows(dataset: str) -> list[dict]:
    """One row per live verdict on a training card of any job of ``dataset``: ``bucket, band, match, p_real, pi,
    scorer_version, verdict, clip, frame, job``. Must run under ``datasets.using(dataset)`` (the hook does)."""
    rows = []
    for job in jobs.all_jobs():
        if (
            job.get("dataset") != dataset
            or job.get("status") == "cancelled"
            or not is_mining_job(job)
        ):
            continue
        for lab in jobs.compute(job)["labels"]:
            meta = lab.get("meta") or {}
            s = meta.get("sampling") or {}
            if s.get("regime") != "train":
                continue
            rows.append({
                "bucket": s.get("bucket"), "band": s.get("band"), "match": meta.get("match") or s.get("match"),
                "p_real": meta.get("p_real"), "pi": s.get("pi"), "scorer_version": meta.get("scorer_version") or s.get("scorer_version"),
                "verdict": lab.get("verdict"), "clip": lab.get("clip"), "frame": lab.get("frame_index"), "job": job["id"],
                "gold_kind": (s.get("gold") or {}).get("kind"),
            })  # fmt: skip
    return rows


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
    """Planted-wrong cards accepted as ball / corrected (an over-accepting labeller). Flags a round above 10 %."""
    g = [
        r
        for r in rows
        if r.get("bucket") == "G"
        and (r.get("verdict") in CORRECT or r.get("verdict") in WRONG)
    ]
    accepted = sum(1 for r in g if r["verdict"] in CORRECT)
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


def refit(
    job: dict,
    *,
    trigger: str = "manual",
    marks_dir: str | Path | None = None,
    params: dict | None = None,
) -> tuple[dict, dict | None]:
    """Compute and store the state for ``job``'s dataset; returns ``(state, calibration)``."""
    params = params or {}
    ds = datasets.get(job["dataset"])
    marks = Path(marks_dir) if marks_dir else ds.marks_dir
    rows = collect_rows(job["dataset"])
    scorer = current_scorer(job)
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
        "trigger": trigger, "closed": trigger == "on_done", "scorer_version": scorer, "n_verdicts": len(decided),
        "n_rows_other_scorer": len(rows) - len(same), "hit_rates": hit_rates(rows), "gold": gold_check(rows), "calibration": cal,
        "threshold": {"global": ltt.as_dict(), "per_match": {m: r.as_dict() for m, r in per_match.items()}, "grid": [0.99, 0.50, 0.01],
                      "method": "learn-then-test, fixed sequence from high t, exact binomial", **kw},
        "auto_accept": {"threshold": ltt.effective_threshold, "scale": ltt.effective_scale, "fallback": ltt.fallback},
    }  # fmt: skip
    return write_state(marks, state), cal


def _action(ctx: hooks.HookContext) -> dict:
    """Refit hit rates, Platt (R+A) and the LTT threshold from the round verdicts, store ``state/vNNN.json`` and re-prioritise
    the unanswered U / M cards (params: ``close``, ``reorder`` (true), ``min_per_class``, ``target``, ``delta``)."""
    state, cal = refit(ctx.job, trigger=ctx.trigger, params=ctx.params)
    out = {"version": state["version"], "n_verdicts": state["n_verdicts"], "reused": state["reused"], "hit_rates": {b: v["hit_rate"] for b, v in state["hit_rates"].items()},
           "calibrated": bool(cal and cal.get("fitted")), "auto_accept": state["auto_accept"], "gold": state["gold"]}  # fmt: skip
    if (
        ctx.params.get("close")
        or ctx.trigger == "on_done"
        or not ctx.params.get("reorder", True)
    ):
        return out
    order = priority_order(job_edit.unanswered_items(ctx.job), cal)
    res = ctx.reorder(
        order,
        note=f"ball_mining state v{state['version']}",
        meta={"state_version": state["version"], "n": state["n_verdicts"]},
    )
    return {**out, "reordered": res["unanswered"], "changed": res["changed"]}


hooks.register_action("ball_mining_refit", _action, kinds=("ball_check",))
