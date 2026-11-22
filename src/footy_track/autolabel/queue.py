"""Targeting queue for human labelling (docs/design/full_frame_labelling.md section 4b).

Human time is only spent on *targeted* frames. Each candidate is ranked by expected teacher
improvement::

    score = priority                      (preverify per-frame priority: max(disagreement, 1 - conf))
          + AUDIT_FAIL_BONUS  if the frame is near / at a blind-audit failure
          + HARD_BALL_BONUS   if the ball is small or occluded
          + CROWD_WEIGHT * min(crowd_score / CROWD_NORM, 1)   (clusters.crowd_score of the boxes)

Every item carries a ``purpose``: ``measure_teacher`` (blind-audit frames), ``improve_teacher``
(flagged / audit-failure / hard-case / high priority), else ``student_data`` (teacher already does
well; a human confirming it makes a series). :func:`build_queue` only returns items of the
requested purpose and REJECTS (reported in ``rejected``, never queued):

* ``not_train``: frames of non-train matches, unless ``eval_panel`` (purpose ``measure_teacher``)
  AND the match is held-out (then ONLY held-out is allowed);
* ``already_gold``: frames with a gold state (hand_series / hand_frame / verdict);
* ``complete_window``: frames already ``frame_complete``.

CLI::

    python -m footy_track.autolabel.queue --purpose improve_teacher --top 50 --out queue.json
        --preverify-dir DIR [--db STORE.duckdb] [--audit-failures f.json] [--hard-cases f.json]
        [--eval-panel]

``audit-failures`` / ``hard-cases``: JSON lists of ``{"game_id", "frame_index"}`` (hard cases add
``"small_ball"`` / ``"occluded_ball"`` booleans and optional ``"boxes"`` [[x,y,w,h],...]).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from footy_track import splits
from footy_track.autolabel.clusters import crowd_score

PURPOSES = ("improve_teacher", "student_data", "measure_teacher")
AUDIT_FAIL_BONUS = 1.0
AUDIT_FAIL_RADIUS = 25  # frames either side of an audit failure that inherit the bonus
HARD_BALL_BONUS = 0.5
CROWD_WEIGHT = 0.5
CROWD_NORM = 10.0
IMPROVE_PRIORITY = 0.3  # preverify priority at/above which a frame is worth improving the teacher on
GOLD_METHODS = ("hand_series", "hand_frame", "verdict")


@dataclass
class Candidate:
    game_id: str
    frame_index: int
    priority: float = 0.0
    flagged: bool = False
    audit_frame: bool = False  # a blind-audit sample frame (measures the teacher)
    audit_failure: bool = False
    small_ball: bool = False
    occluded_ball: bool = False
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    window_id: str | None = None


@dataclass
class QueueItem:
    game_id: str
    frame_index: int
    purpose: str
    score: float
    reasons: list[str]
    window_id: str | None = None


def score_candidate(c: Candidate) -> tuple[float, list[str]]:
    reasons = []
    s = c.priority
    if c.priority > 0:
        reasons.append("priority")
    if c.audit_failure:
        s += AUDIT_FAIL_BONUS
        reasons.append("audit_failure")
    if c.small_ball or c.occluded_ball:
        s += HARD_BALL_BONUS
        reasons.append("small_ball" if c.small_ball else "occluded_ball")
    if c.boxes:
        cs = crowd_score(c.boxes)
        if cs.largest_cluster >= 3:
            s += CROWD_WEIGHT * min(cs.score / CROWD_NORM, 1.0)
            reasons.append("crowded")
    return s, reasons


def purpose_of(c: Candidate, reasons: Sequence[str]) -> str:
    if c.audit_frame:
        return "measure_teacher"
    hard = {"audit_failure", "small_ball", "occluded_ball", "crowded"}
    if c.flagged or c.priority >= IMPROVE_PRIORITY or hard & set(reasons):
        return "improve_teacher"
    return "student_data"


def _default_split(game_id: str) -> str:
    return str(splits.split_of(splits.match_key_of(game_id), game_id))


def build_queue(  # noqa: PLR0913
    candidates: Iterable[Candidate],
    purpose: str,
    top: int,
    *,
    eval_panel: bool = False,
    split_of: Callable[[str], str] = _default_split,
    gold_frames: Callable[[str], set[int]] = lambda _g: set(),
    complete_frames: Callable[[str], set[int]] = lambda _g: set(),
) -> tuple[list[QueueItem], dict[str, int]]:
    """``(items, rejected_counts)``: the top ``top`` items of ``purpose``, best first."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    if eval_panel and purpose != "measure_teacher":
        raise ValueError("eval_panel requests must have purpose measure_teacher")
    rejected: Counter[str] = Counter()
    gold_c: dict[str, set[int]] = {}
    done_c: dict[str, set[int]] = {}
    split_c: dict[str, str] = {}
    items: list[QueueItem] = []
    for c in candidates:
        g = c.game_id
        sp = split_c.setdefault(g, split_of(g))
        legal = sp == (splits.EVAL_HELDOUT if eval_panel else splits.TRAIN)
        if not legal:
            rejected["not_train" if not eval_panel else "not_heldout"] += 1
            continue
        if c.frame_index in gold_c.setdefault(g, set(gold_frames(g))):
            rejected["already_gold"] += 1
            continue
        if c.frame_index in done_c.setdefault(g, set(complete_frames(g))):
            rejected["complete_window"] += 1
            continue
        s, reasons = score_candidate(c)
        p = "measure_teacher" if eval_panel else purpose_of(c, reasons)
        if p != purpose:
            continue
        items.append(QueueItem(g, c.frame_index, p, round(s, 6), reasons, c.window_id))
    items.sort(key=lambda i: (-i.score, i.game_id, i.frame_index))
    return items[: max(top, 0)], dict(sorted(rejected.items()))


def spread_audit_failures(
    cands: dict[tuple[str, int], Candidate], failures: Iterable[tuple[str, int]]
) -> None:
    """Mark candidates within ``AUDIT_FAIL_RADIUS`` frames of a failure (same clip) in place."""
    by_game: dict[str, list[int]] = {}
    for g, f in failures:
        by_game.setdefault(g, []).append(f)
    for (g, f), c in cands.items():
        if any(abs(f - x) <= AUDIT_FAIL_RADIUS for x in by_game.get(g, ())):
            c.audit_failure = True


def candidates_from_preverify(doc: dict) -> list[Candidate]:
    """Candidates from one ``preverify/<window_id>.json`` document."""
    wid = doc["window_id"]
    game_id = wid.rsplit("_f", 1)[0]
    audit = set(doc.get("audit", ()))
    return [
        Candidate(
            game_id, int(f["frame_index"]), float(f.get("priority") or 0.0),
            flagged=f.get("verdict") == "flagged", audit_frame=int(f["frame_index"]) in audit,
            window_id=wid,
        )
        for f in doc["frames"]
    ]  # fmt: skip


def _store_sets(db: Path | None):
    if db is None:
        return (lambda _g: set()), (lambda _g: set()), None
    from footy_track.feature_store.lock import store_lock  # noqa: PLC0415
    from footy_track.feature_store.store import FeatureStore  # noqa: PLC0415

    lock = store_lock(db, exclusive=False)
    lock.__enter__()
    st = FeatureStore.open(db, create=False)

    def gold(g: str) -> set[int]:
        q = st.query(
            "SELECT DISTINCT frame_index FROM frame_label_state WHERE game_id = ? "
            f"AND method IN {GOLD_METHODS} AND state IN ('complete','absent') AND reason IS NULL", [g],
        )  # fmt: skip
        return {int(i) for i in q["frame_index"]}

    def done(g: str) -> set[int]:
        q = st.query("SELECT frame_index FROM frame_complete WHERE game_id = ?", [g])
        return {int(i) for i in q["frame_index"]}

    def close() -> None:
        st.close()
        lock.__exit__(None, None, None)

    return gold, done, close


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--purpose", choices=PURPOSES, required=True)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preverify-dir", type=Path, required=True)
    ap.add_argument("--db", type=Path)
    ap.add_argument("--audit-failures", type=Path)
    ap.add_argument("--hard-cases", type=Path)
    ap.add_argument("--eval-panel", action="store_true")
    args = ap.parse_args(argv)

    cands: dict[tuple[str, int], Candidate] = {}
    for p in sorted(args.preverify_dir.glob("*.json")):
        for c in candidates_from_preverify(json.loads(p.read_text())):
            cands[(c.game_id, c.frame_index)] = c
    if args.audit_failures:
        fails = [(d["game_id"], int(d["frame_index"])) for d in json.loads(args.audit_failures.read_text())]
        spread_audit_failures(cands, fails)
    if args.hard_cases:
        for d in json.loads(args.hard_cases.read_text()):
            c = cands.setdefault((d["game_id"], int(d["frame_index"])), Candidate(d["game_id"], int(d["frame_index"])))
            c.small_ball |= bool(d.get("small_ball"))
            c.occluded_ball |= bool(d.get("occluded_ball"))
            c.boxes = [tuple(b) for b in d.get("boxes", c.boxes)]
    gold, done, close = _store_sets(args.db)
    try:
        items, rejected = build_queue(
            cands.values(), args.purpose, args.top, eval_panel=args.eval_panel,
            gold_frames=gold, complete_frames=done,
        )  # fmt: skip
    finally:
        if close:
            close()
    args.out.write_text(
        json.dumps({"version": 1, "purpose": args.purpose, "eval_panel": args.eval_panel,
                    "rejected": rejected, "items": [asdict(i) for i in items]}, indent=1) + "\n"
    )  # fmt: skip
    print(f"{len(items)} items ({sum(rejected.values())} rejected: {rejected}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
