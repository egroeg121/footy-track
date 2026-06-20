"""Difficulty scoring: flag low-confidence and ambiguous detections for human review.

After frames are ingested into the feature store, this module identifies
``DetectionRow`` entries that are hard for the model and marks them as
``needs_review`` so human annotators can focus effort on the hardest frames.

Three criteria are applied per frame (any one suffices to flag a detection):

1. **Low confidence** — model confidence below ``conf_threshold`` (default 0.4).
   Hand-label rows (confidence = None) are never flagged by this criterion.
2. **Crowded frame** — bounding-box IoU with any other detection in the same
   frame exceeds ``iou_threshold`` (default 0.5). Both overlapping detections
   are flagged.
3. **Missing ball** — frame contains no detection labelled ``ball`` (or any of
   ``in_play_ball`` / ``out_of_play_ball``). Every detection in that frame is
   flagged so reviewers know to add the ball annotation.

Results are written back to the ``detection`` table via a new ``needs_review``
BOOLEAN column. The update is idempotent — re-running on the same
``(game_id, source, run_id)`` overwrites the flag cleanly.

Usage::

    from footy_track.feature_store import FeatureStore
    from footy_track.feature_store.difficulty import score_detections

    store = FeatureStore.open("data/feature_store.duckdb")
    report = score_detections(store, game_id="arsenal_mancity", source="yolo", run_id="r1")
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from footy_track.feature_store.store import FeatureStore

_BALL_LABELS = frozenset({"ball", "in_play_ball", "out_of_play_ball"})


@dataclass
class DifficultyReport:
    """Summary of one difficulty-scoring pass."""

    game_id: str
    source: str
    run_id: str
    total_detections: int = 0
    flagged_low_conf: int = 0
    flagged_crowded: int = 0
    flagged_no_ball: int = 0
    total_flagged: int = 0

    def __str__(self) -> str:
        pct = 100 * self.total_flagged / max(1, self.total_detections)
        return (
            f"DifficultyReport({self.game_id!r} src={self.source!r} run={self.run_id!r}): "
            f"{self.total_flagged}/{self.total_detections} flagged ({pct:.1f}%) — "
            f"low_conf={self.flagged_low_conf} crowded={self.flagged_crowded} "
            f"no_ball={self.flagged_no_ball}"
        )


def _iou(ax: float, ay: float, aw: float, ah: float,
          bx: float, by: float, bw: float, bh: float) -> float:
    """Intersection-over-Union for two top-left xywh boxes, all normalised."""
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def score_detections(
    store: FeatureStore,
    *,
    game_id: str,
    source: str,
    run_id: str,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
) -> DifficultyReport:
    """Score detections for one ``(game_id, source, run_id)`` and write
    ``needs_review`` flags back to the ``detection`` table.

    Only YOLO-style model detections (with numeric confidence values) are
    evaluated for low-confidence; hand-label rows are skipped for that
    criterion but may still be flagged for crowd / missing-ball.

    Returns a :class:`DifficultyReport` summarising what was flagged.
    """
    df = store.query(
        """
        SELECT frame_index, detection_id,
               label, confidence,
               bbox_x, bbox_y, bbox_w, bbox_h
        FROM detection
        WHERE game_id = ? AND source = ? AND run_id = ?
        ORDER BY frame_index, detection_id
        """,
        [game_id, source, run_id],
    )

    report = DifficultyReport(game_id=game_id, source=source, run_id=run_id)
    if df.empty:
        return report

    report.total_detections = len(df)

    # Collect flagged (frame_index, detection_id) pairs per criterion.
    flagged_low_conf: set[tuple[int, int]] = set()
    flagged_crowded: set[tuple[int, int]] = set()
    flagged_no_ball: set[tuple[int, int]] = set()

    for frame_idx, frame_df in df.groupby("frame_index"):
        rows = frame_df.to_dict("records")

        # Criterion 1: low confidence
        for r in rows:
            conf = r["confidence"]
            if conf is not None and conf < conf_threshold:
                flagged_low_conf.add((int(frame_idx), int(r["detection_id"])))

        # Criterion 2: crowded (pairwise IoU)
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if _iou(a["bbox_x"], a["bbox_y"], a["bbox_w"], a["bbox_h"],
                        b["bbox_x"], b["bbox_y"], b["bbox_w"], b["bbox_h"]) >= iou_threshold:
                    flagged_crowded.add((int(frame_idx), int(a["detection_id"])))
                    flagged_crowded.add((int(frame_idx), int(b["detection_id"])))

        # Criterion 3: no ball in frame
        has_ball = any(r["label"] in _BALL_LABELS for r in rows)
        if not has_ball:
            for r in rows:
                flagged_no_ball.add((int(frame_idx), int(r["detection_id"])))

    all_flagged = flagged_low_conf | flagged_crowded | flagged_no_ball
    report.flagged_low_conf = len(flagged_low_conf)
    report.flagged_crowded = len(flagged_crowded)
    report.flagged_no_ball = len(flagged_no_ball)
    report.total_flagged = len(all_flagged)

    if not all_flagged:
        # Write needs_review = FALSE for all rows in this run.
        store._conn.execute(
            """
            UPDATE detection SET needs_review = FALSE
            WHERE game_id = ? AND source = ? AND run_id = ?
            """,
            [game_id, source, run_id],
        )
        return report

    # Write needs_review flags: TRUE for flagged, FALSE for the rest.
    # Build a VALUES list to avoid one UPDATE per row.
    flagged_list = list(all_flagged)
    placeholders = ", ".join("(?, ?)" for _ in flagged_list)
    flat_params: list[object] = []
    for fi, di in flagged_list:
        flat_params.extend([fi, di])

    store._conn.execute(
        f"""
        UPDATE detection
        SET needs_review = CASE
            WHEN (frame_index, detection_id) IN (VALUES {placeholders}) THEN TRUE
            ELSE FALSE
        END
        WHERE game_id = ? AND source = ? AND run_id = ?
        """,
        flat_params + [game_id, source, run_id],
    )

    return report


def flag_for_review(
    store: FeatureStore,
    *,
    game_id: str,
    source: str,
    run_id: str,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
) -> DifficultyReport:
    """Alias for :func:`score_detections` — preferred public name."""
    return score_detections(
        store,
        game_id=game_id,
        source=source,
        run_id=run_id,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
