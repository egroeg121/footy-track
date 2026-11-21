"""Reconciler for the ball-anchor loop (docs/design/ball_anchor_iteration.md sections 7.0, 7.7). READ-ONLY.

Per job family (eval ``ball-eval/...``, train ``ball-round/...``; checked independently) it diffs three sets:

* **F**, the expected frames: ``selected_frame`` rows of the family's jobs (feature store);
* **cards / verdicts**: the labelling jobs' items and labels (``jobs_client.list_jobs`` + ``results``, the way
  ``refit.collect_rows`` reads them; the client is injectable);
* **terminal states**: ``frame_label_state`` ball rows (``FeatureStore.read_ball_terminal_states``).

Findings (every one is a violation): ``orphans`` (card / verdict for a frame not in the job's F, a job with cards and no F),
``missing`` (F frame with no card and no terminal state), ``stuck`` (open card, or non-terminal F frame, older than
``stuck_days``), ``store_lag`` (ball verdict present, no terminal ball state), ``split_violations`` (frame's ``split_of``
differs from its F row / family, or a frame is in both families' F) and ``duplicate_cards``.

``reconcile`` never writes anything. ``--apply`` in the CLI is a refusing stub.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime

from footy_track.feature_store.store import FeatureStore

REPORT_SCHEMA = "ball-reconcile/v1"
FAMILIES: dict[str, dict] = {
    "eval": {"prefix": "ball-eval/", "splits": ("eval_heldout_match", "eval_in_match")},
    "train": {"prefix": "ball-round/", "splits": ("train",)},
}
KINDS = ("orphans", "missing", "stuck", "store_lag", "split_violations", "duplicate_cards")
#: Verdicts that must leave a terminal ``ball`` state in the store (``ball_check.CLEAN_VERDICTS`` + ``box_off`` is not).
BALL_VERDICTS = ("ball", "corrected")
DEFAULT_STUCK_DAYS = 3.0


def family_of(key: str | None) -> str | None:
    for name, spec in FAMILIES.items():
        if str(key or "").startswith(spec["prefix"]):
            return name
    return None


def _default_split(clip: str) -> str:
    from footy_track import splits  # noqa: PLC0415

    return str(splits.split_of(splits.match_key_of(clip), clip))


def _age_days(ts: object, now: datetime) -> float | None:
    if ts is None:
        return None
    try:
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            dt = ts  # pandas Timestamp / datetime
        dt = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return (now - dt).total_seconds() / 86400.0
    except (TypeError, ValueError):
        return None


def _card_key(item: dict) -> tuple:
    b = item.get("bbox")
    ident = tuple(round(float(b[k]), 6) for k in ("x", "y", "w", "h")) if isinstance(b, dict) and all(k in b for k in "xywh") else item.get("box_index")
    return (item["clip"], int(item["frame_index"]), ident)


def reconcile(
    store: FeatureStore,
    client,
    *,
    families: tuple[str, ...] = tuple(FAMILIES),
    stuck_days: float = DEFAULT_STUCK_DAYS,
    now: datetime | None = None,
    split_fn: Callable[[str], str] | None = None,
    dataset: str | None = None,
) -> dict:
    """The report dict (schema ``ball-reconcile/v1``); ``report["ok"]`` is false iff any violation was found."""
    now = now or datetime.now(UTC)
    split_fn = split_fn or _default_split
    sel = store.read_selected_frames()
    sel_by_family: dict[str, list] = {f: [] for f in families}
    for r in sel.itertuples(index=False):
        fam = family_of(r.job_key)
        if fam in sel_by_family:
            sel_by_family[fam].append(r)

    filt = {"kind": "ball_check", "limit": "5000", **({"dataset": dataset} if dataset else {})}
    all_jobs = [j for j in client.list_jobs(**filt) if j.get("status") != "cancelled"]

    out: dict = {"schema": REPORT_SCHEMA, "generated_at": now.isoformat(), "stuck_days": stuck_days, "families": {}}
    f_by_family: dict[str, set] = {}
    terminal_cache: dict[str, dict[int, str]] = {}

    def terminal(game: str, frame: int) -> str | None:
        if game not in terminal_cache:
            terminal_cache[game] = store.read_ball_terminal_states(game)
        return terminal_cache[game].get(frame)

    for fam in families:
        spec = FAMILIES[fam]
        rep: dict = {k: [] for k in KINDS}
        # expected set per job key
        f_job: dict[str, dict[tuple[str, int], object]] = defaultdict(dict)
        for r in sel_by_family[fam]:
            f_job[r.job_key][(r.game_id, int(r.frame_index))] = r
        f_all = {fr for d in f_job.values() for fr in d}
        f_by_family[fam] = f_all
        jobs = [j for j in all_jobs if family_of(j.get("key")) == fam]
        job_keys = {j["key"] for j in jobs}
        n_cards = n_verdicts = 0
        carded: set[tuple[str, int]] = set()
        seen_cards: dict[tuple, str] = {}  # card identity -> first job key

        for j in jobs:
            doc = client.results(j["id"])
            key, jid = j["key"], j["id"]
            f = f_job.get(key, {})
            items = [it["item"] if "item" in it else it for it in doc.get("items") or []]
            answered = {(lab.get("clip"), lab.get("frame_index")) for lab in doc.get("labels") or [] if lab.get("verdict") is not None}
            job_age = _age_days(j.get("created"), now)
            if items and not f:
                rep["orphans"].append({"type": "job_without_selection", "job": key, "cards": len(items)})
            for it in items:
                clip, frame = it["clip"], int(it["frame_index"])
                n_cards += 1
                carded.add((clip, frame))
                ck = _card_key(it)
                if ck in seen_cards:
                    rep["duplicate_cards"].append({"job": key, "clip": clip, "frame": frame, "scope": "job" if seen_cards[ck] == key else "family", "first_job": seen_cards[ck]})
                else:
                    seen_cards[ck] = key
                if f and (clip, frame) not in f:
                    rep["orphans"].append({"type": "card", "job": key, "clip": clip, "frame": frame})
                if (clip, frame) not in answered and job_age is not None and job_age > stuck_days:
                    rep["stuck"].append({"type": "open_card", "job": key, "clip": clip, "frame": frame, "age_days": round(job_age, 2)})
                if f:
                    sp = split_fn(clip)
                    want = getattr(f.get((clip, frame)), "split", None)
                    if sp not in spec["splits"] or (want is not None and sp != want):
                        rep["split_violations"].append({"type": "card_split", "job": key, "clip": clip, "frame": frame, "split_of": sp, "job_split": want, "family": fam})
            for lab in doc.get("labels") or []:
                if lab.get("verdict") is None:
                    continue
                meta = lab.get("meta") or {}
                clip, frame = meta.get("pool_clip") or lab.get("clip"), int(lab["frame_index"])
                n_verdicts += 1
                if f and (clip, frame) not in f:
                    rep["orphans"].append({"type": "verdict", "job": key, "clip": clip, "frame": frame, "verdict": lab["verdict"]})
                elif lab["verdict"] in BALL_VERDICTS and terminal(clip, frame) != "ball":
                    rep["store_lag"].append({"job": key, "clip": clip, "frame": frame, "verdict": lab["verdict"], "terminal": terminal(clip, frame)})

        for key, f in f_job.items():
            if key not in job_keys:
                for (g, fr) in f:
                    if terminal(g, fr) is None:
                        rep["missing"].append({"job": key, "clip": g, "frame": fr, "why": "no_job"})
                continue
            for (g, fr), r in f.items():
                t = terminal(g, fr)
                if t is None and (g, fr) not in carded:
                    rep["missing"].append({"job": key, "clip": g, "frame": fr, "why": "no_card"})
                if t is None:
                    age = _age_days(getattr(r, "selected_at", None), now)
                    if age is not None and age > stuck_days:
                        rep["stuck"].append({"type": "non_terminal", "job": key, "clip": g, "frame": fr, "age_days": round(age, 2)})
                if r.split not in spec["splits"]:
                    rep["split_violations"].append({"type": "wrong_family", "job": key, "clip": g, "frame": fr, "split": r.split, "family": fam})
                else:
                    sp = split_fn(g)
                    if sp != r.split:
                        rep["split_violations"].append({"type": "selected_split", "job": key, "clip": g, "frame": fr, "split_of": sp, "job_split": r.split, "family": fam})

        out["families"][fam] = {
            "jobs": len(jobs), "frames_expected": len(f_all), "cards": n_cards, "verdicts": n_verdicts,
            "counts": {k: len(rep[k]) for k in KINDS}, **rep,
        }

    if len(families) > 1:  # a frame in both families' F is a hard split violation
        seen: dict[tuple[str, int], str] = {}
        for fam in families:
            for fr in f_by_family[fam]:
                if fr in seen and seen[fr] != fam:
                    out["families"][fam]["split_violations"].append({"type": "in_both_families", "clip": fr[0], "frame": fr[1], "families": [seen[fr], fam]})
                    out["families"][fam]["counts"]["split_violations"] += 1
                seen.setdefault(fr, fam)

    out["violations"] = sum(sum(f["counts"].values()) for f in out["families"].values())
    out["ok"] = out["violations"] == 0
    return out


def summary(report: dict) -> str:
    lines = [f"ball-reconcile: {'OK' if report['ok'] else str(report['violations']) + ' violation(s)'}"]
    for fam, f in report["families"].items():
        c = ", ".join(f"{k}={n}" for k, n in f["counts"].items() if n) or "clean"
        lines.append(f"  {fam}: {f['jobs']} jobs, {f['frames_expected']} frames expected, {f['cards']} cards, {f['verdicts']} verdicts; {c}")
    return "\n".join(lines)
