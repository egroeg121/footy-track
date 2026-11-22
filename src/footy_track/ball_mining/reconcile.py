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

The design's ``version skew`` finding (section 7.7) is not implemented: the doc names it but does not specify what is compared.

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


class _Ctx:
    """Per-run lookups shared by the per-check helpers."""

    def __init__(self, store: FeatureStore, now: datetime, split_fn: Callable[[str], str], stuck_days: float) -> None:
        self.store, self.now, self.split_fn, self.stuck_days = store, now, split_fn, stuck_days
        self._terminal: dict[str, dict[int, str]] = {}

    def terminal(self, game: str, frame: int) -> str | None:
        if game not in self._terminal:
            self._terminal[game] = self.store.read_ball_terminal_states(game)
        return self._terminal[game].get(frame)


def _items_of(doc: dict) -> list[dict]:
    return [it.get("item", it) for it in doc.get("items") or []]


def _verdict_labels(doc: dict) -> list[dict]:
    return [lab for lab in doc.get("labels") or [] if lab.get("verdict") is not None]


def _check_duplicates(rep: dict, key: str, items: list[dict], seen: dict[tuple, str]) -> None:
    for it in items:
        ck = _card_key(it)
        if ck in seen:
            rep["duplicate_cards"].append({
                "job": key, "clip": it["clip"], "frame": int(it["frame_index"]),
                "scope": "job" if seen[ck] == key else "family", "first_job": seen[ck],
            })  # fmt: skip
        else:
            seen[ck] = key


def _check_orphans(rep: dict, key: str, items: list[dict], labels: list[dict], f: dict) -> None:
    if items and not f:
        rep["orphans"].append({"type": "job_without_selection", "job": key, "cards": len(items)})
        return
    for it in items:
        if (it["clip"], int(it["frame_index"])) not in f:
            rep["orphans"].append({"type": "card", "job": key, "clip": it["clip"], "frame": int(it["frame_index"])})
    for lab in labels:
        clip = (lab.get("meta") or {}).get("pool_clip") or lab.get("clip")
        if (clip, int(lab["frame_index"])) not in f:
            rep["orphans"].append({"type": "verdict", "job": key, "clip": clip, "frame": int(lab["frame_index"]), "verdict": lab["verdict"]})


def _check_store_lag(rep: dict, ctx: _Ctx, key: str, labels: list[dict], f: dict) -> None:
    for lab in labels:
        clip = (lab.get("meta") or {}).get("pool_clip") or lab.get("clip")
        frame = int(lab["frame_index"])
        if (clip, frame) in f and lab["verdict"] in BALL_VERDICTS and ctx.terminal(clip, frame) != "ball":
            rep["store_lag"].append({"job": key, "clip": clip, "frame": frame, "verdict": lab["verdict"], "terminal": ctx.terminal(clip, frame)})


def _check_stuck_cards(rep: dict, ctx: _Ctx, job: dict, items: list[dict], labels: list[dict]) -> None:
    age = _age_days(job.get("created"), ctx.now)
    if age is None or age <= ctx.stuck_days:
        return
    answered = {(lab.get("clip"), lab.get("frame_index")) for lab in labels}
    for it in items:
        if (it["clip"], it["frame_index"]) not in answered:
            rep["stuck"].append({"type": "open_card", "job": job["key"], "clip": it["clip"], "frame": int(it["frame_index"]), "age_days": round(age, 2)})


def _check_card_splits(rep: dict, ctx: _Ctx, fam: str, key: str, items: list[dict], f: dict) -> None:
    if not f:
        return
    for it in items:
        clip, frame = it["clip"], int(it["frame_index"])
        sp = ctx.split_fn(clip)
        want = getattr(f.get((clip, frame)), "split", None)
        if sp not in FAMILIES[fam]["splits"] or (want is not None and sp != want):
            rep["split_violations"].append({"type": "card_split", "job": key, "clip": clip, "frame": frame, "split_of": sp, "job_split": want, "family": fam})


def _check_selected(rep: dict, ctx: _Ctx, fam: str, f_job: dict, job_keys: set[str], carded: set) -> None:
    """Missing, stuck non-terminal and split checks over the expected set F itself."""
    for key, f in f_job.items():
        for (g, fr), r in f.items():
            t = ctx.terminal(g, fr)
            if t is None and key not in job_keys:
                rep["missing"].append({"job": key, "clip": g, "frame": fr, "why": "no_job"})
            elif t is None and (g, fr) not in carded:
                rep["missing"].append({"job": key, "clip": g, "frame": fr, "why": "no_card"})
            if t is None:
                age = _age_days(getattr(r, "selected_at", None), ctx.now)
                if age is not None and age > ctx.stuck_days:
                    rep["stuck"].append({"type": "non_terminal", "job": key, "clip": g, "frame": fr, "age_days": round(age, 2)})
            if r.split not in FAMILIES[fam]["splits"]:
                rep["split_violations"].append({"type": "wrong_family", "job": key, "clip": g, "frame": fr, "split": r.split, "family": fam})
            elif ctx.split_fn(g) != r.split:
                rep["split_violations"].append({"type": "selected_split", "job": key, "clip": g, "frame": fr, "split_of": ctx.split_fn(g), "job_split": r.split, "family": fam})


def _check_family(ctx: _Ctx, fam: str, rows: list, jobs: list[dict], client) -> dict:
    rep: dict = {k: [] for k in KINDS}
    f_job: dict[str, dict[tuple[str, int], object]] = defaultdict(dict)
    for r in rows:
        f_job[r.job_key][(r.game_id, int(r.frame_index))] = r
    carded: set[tuple[str, int]] = set()
    seen: dict[tuple, str] = {}  # card identity -> first job key
    n_cards = n_verdicts = 0
    for j in jobs:
        doc = client.results(j["id"])
        items, labels, f = _items_of(doc), _verdict_labels(doc), f_job.get(j["key"], {})
        n_cards, n_verdicts = n_cards + len(items), n_verdicts + len(labels)
        carded.update((it["clip"], int(it["frame_index"])) for it in items)
        _check_duplicates(rep, j["key"], items, seen)
        _check_orphans(rep, j["key"], items, labels, f)
        _check_stuck_cards(rep, ctx, j, items, labels)
        _check_card_splits(rep, ctx, fam, j["key"], items, f)
        _check_store_lag(rep, ctx, j["key"], labels, f)
    _check_selected(rep, ctx, fam, f_job, {j["key"] for j in jobs}, carded)
    expected = {fr for d in f_job.values() for fr in d}
    return {
        "jobs": len(jobs), "frames_expected": len(expected), "cards": n_cards, "verdicts": n_verdicts,
        "counts": {k: len(rep[k]) for k in KINDS}, **rep, "_expected": expected,
    }  # fmt: skip


def _check_cross_family(fams: dict[str, dict]) -> None:
    """A frame in both families' F is a hard split violation."""
    seen: dict[tuple[str, int], str] = {}
    for fam, rep in fams.items():
        for fr in rep["_expected"]:
            if fr in seen and seen[fr] != fam:
                rep["split_violations"].append({"type": "in_both_families", "clip": fr[0], "frame": fr[1], "families": [seen[fr], fam]})
                rep["counts"]["split_violations"] += 1
            seen.setdefault(fr, fam)


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
    ctx = _Ctx(store, now or datetime.now(UTC), split_fn or _default_split, stuck_days)
    by_family: dict[str, list] = {f: [] for f in families}
    for r in store.read_selected_frames().itertuples(index=False):
        if family_of(r.job_key) in by_family:
            by_family[family_of(r.job_key)].append(r)
    filt = {"kind": "ball_check", "limit": "5000", **({"dataset": dataset} if dataset else {})}
    live = [j for j in client.list_jobs(**filt) if j.get("status") != "cancelled"]
    fams = {fam: _check_family(ctx, fam, by_family[fam], [j for j in live if family_of(j.get("key")) == fam], client) for fam in families}
    _check_cross_family(fams)
    for rep in fams.values():
        rep.pop("_expected")
    violations = sum(sum(f["counts"].values()) for f in fams.values())
    return {
        "schema": REPORT_SCHEMA, "generated_at": ctx.now.isoformat(), "stuck_days": stuck_days,
        "families": fams, "violations": violations, "ok": violations == 0,
    }  # fmt: skip


def summary(report: dict) -> str:
    lines = [f"ball-reconcile: {'OK' if report['ok'] else str(report['violations']) + ' violation(s)'}"]
    for fam, f in report["families"].items():
        c = ", ".join(f"{k}={n}" for k, n in f["counts"].items() if n) or "clean"
        lines.append(f"  {fam}: {f['jobs']} jobs, {f['frames_expected']} frames expected, {f['cards']} cards, {f['verdicts']} verdicts; {c}")
    return "\n".join(lines)
