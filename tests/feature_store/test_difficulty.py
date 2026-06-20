"""Tests for feature_store.difficulty — difficulty scoring and needs_review flagging."""

from __future__ import annotations

import pytest

from footy_track.feature_store import (
    DetectionRow,
    FeatureStore,
    FrameRow,
    GameRow,
    RunRow,
    flag_for_review,  # noqa: F401
    score_detections,
)

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def store() -> FeatureStore:
    """In-memory store with one game, one run, and a handful of frames."""
    s = FeatureStore.open(":memory:")
    s.upsert_games([GameRow(game_id="g1", fps=25.0)])
    s.upsert_runs(
        [
            RunRow(
                run_id="run1", stage="detection", source="yolo", model_name="yolo.pt"
            ),
            RunRow(
                run_id="hl1", stage="detection", source="hand_label", model_name="human"
            ),
        ]
    )
    return s


def _frame(store: FeatureStore, frame_index: int) -> None:
    store.upsert_frames(
        [
            FrameRow(
                game_id="g1",
                frame_index=frame_index,
                frame_uri=f"f{frame_index}.jpg",
                width=1920,
                height=1080,
                continuous_time_s=float(frame_index) / 25.0,
            )
        ]
    )


def _det(
    frame_index: int,
    detection_id: int,
    *,
    label: str = "player",
    confidence: float | None = 0.8,
    bbox_x: float = 0.1,
    bbox_y: float = 0.1,
    bbox_w: float = 0.1,
    bbox_h: float = 0.1,
    source: str = "yolo",
    run_id: str = "run1",
) -> DetectionRow:
    return DetectionRow(
        game_id="g1",
        frame_index=frame_index,
        continuous_time_s=float(frame_index) / 25.0,
        detection_id=detection_id,
        source=source,
        run_id=run_id,
        label=label,
        confidence=confidence,
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_w=bbox_w,
        bbox_h=bbox_h,
    )


# --------------------------------------------------------------------------- #
# Low-confidence criterion                                                     #
# --------------------------------------------------------------------------- #


def test_low_confidence_flagged(store: FeatureStore) -> None:
    """Low-conf detection is flagged; other detection in same frame (with ball) is not."""
    _frame(store, 0)
    store.upsert_detections(
        [
            # detection 0: low confidence (flagged)
            _det(
                0,
                0,
                label="player",
                confidence=0.2,
                bbox_x=0.0,
                bbox_y=0.0,
                bbox_w=0.1,
                bbox_h=0.1,
            ),
            # detection 1: good confidence, no overlap with det 0
            _det(
                0,
                1,
                label="ball",
                confidence=0.9,
                bbox_x=0.5,
                bbox_y=0.5,
                bbox_w=0.1,
                bbox_h=0.1,
            ),
        ]
    )
    report = score_detections(store, game_id="g1", source="yolo", run_id="run1")
    assert report.flagged_low_conf == 1
    assert report.total_flagged == 1  # ball present; no overlap; only det 0

    df = store.query(
        "SELECT detection_id, needs_review FROM detection WHERE game_id='g1' ORDER BY detection_id"
    )
    assert bool(df.loc[df.detection_id == 0, "needs_review"].values[0])
    assert not bool(df.loc[df.detection_id == 1, "needs_review"].values[0])


def test_hand_label_not_flagged_for_confidence(store: FeatureStore) -> None:
    """Hand labels have confidence=None and must not trigger low-conf flag."""
    _frame(store, 0)
    store.upsert_detections(
        [_det(0, 0, confidence=None, source="hand_label", run_id="hl1")]
    )
    report = score_detections(store, game_id="g1", source="hand_label", run_id="hl1")
    assert report.flagged_low_conf == 0


# --------------------------------------------------------------------------- #
# Crowded-frame criterion                                                      #
# --------------------------------------------------------------------------- #


def test_overlapping_boxes_flagged(store: FeatureStore) -> None:
    """Two nearly-identical bounding boxes exceed IoU 0.5 — both are crowded-flagged."""
    _frame(store, 0)
    # Boxes: (0.0, 0.0, 0.3, 0.3) and (0.05, 0.05, 0.3, 0.3)
    # inter = 0.25*0.25=0.0625  union=0.09+0.09-0.0625=0.1175  IoU≈0.53
    store.upsert_detections(
        [
            _det(
                0,
                0,
                label="ball",
                bbox_x=0.0,
                bbox_y=0.0,
                bbox_w=0.3,
                bbox_h=0.3,
                confidence=0.9,
            ),
            _det(
                0,
                1,
                label="player",
                bbox_x=0.05,
                bbox_y=0.05,
                bbox_w=0.3,
                bbox_h=0.3,
                confidence=0.9,
            ),
        ]
    )
    report = score_detections(store, game_id="g1", source="yolo", run_id="run1")
    assert report.flagged_crowded == 2

    df = store.query(
        "SELECT detection_id, needs_review FROM detection WHERE game_id='g1' ORDER BY detection_id"
    )
    assert all(df["needs_review"])


def test_non_overlapping_boxes_not_flagged(store: FeatureStore) -> None:
    _frame(store, 0)
    # Two boxes far apart
    store.upsert_detections(
        [
            _det(0, 0, bbox_x=0.0, bbox_y=0.0, bbox_w=0.1, bbox_h=0.1, confidence=0.9),
            _det(0, 1, bbox_x=0.8, bbox_y=0.8, bbox_w=0.1, bbox_h=0.1, confidence=0.9),
        ]
    )
    report = score_detections(store, game_id="g1", source="yolo", run_id="run1")
    assert report.flagged_crowded == 0


# --------------------------------------------------------------------------- #
# Missing-ball criterion                                                       #
# --------------------------------------------------------------------------- #


def test_no_ball_in_frame_flags_all_detections(store: FeatureStore) -> None:
    _frame(store, 0)
    store.upsert_detections(
        [
            _det(0, 0, label="player", confidence=0.9),
            _det(0, 1, label="referee", confidence=0.85),
        ]
    )
    report = score_detections(store, game_id="g1", source="yolo", run_id="run1")
    assert report.flagged_no_ball == 2
    assert report.total_flagged == 2


def test_ball_present_no_missing_ball_flag(store: FeatureStore) -> None:
    _frame(store, 0)
    store.upsert_detections(
        [
            _det(0, 0, label="player", confidence=0.9),
            _det(0, 1, label="ball", confidence=0.8),
        ]
    )
    report = score_detections(store, game_id="g1", source="yolo", run_id="run1")
    assert report.flagged_no_ball == 0


@pytest.mark.parametrize("ball_label", ["ball", "in_play_ball", "out_of_play_ball"])
def test_all_ball_label_variants_recognised(
    store: FeatureStore, ball_label: str
) -> None:
    _frame(store, 0)
    store.upsert_detections(
        [
            _det(0, 0, label="player", confidence=0.9),
            _det(0, 1, label=ball_label, confidence=0.8),
        ]
    )
    report = score_detections(store, game_id="g1", source="yolo", run_id="run1")
    assert report.flagged_no_ball == 0


# --------------------------------------------------------------------------- #
# Multi-frame idempotency                                                      #
# --------------------------------------------------------------------------- #


def test_multi_frame_independent_scoring(store: FeatureStore) -> None:
    """Each frame is scored independently; one bad frame doesn't taint others."""
    _frame(store, 0)
    _frame(store, 1)
    store.upsert_detections(
        [
            # Frame 0: has ball, high confidence, no overlap → not flagged
            _det(
                0,
                0,
                label="player",
                confidence=0.9,
                bbox_x=0.0,
                bbox_y=0.0,
                bbox_w=0.1,
                bbox_h=0.1,
            ),
            _det(
                0,
                1,
                label="ball",
                confidence=0.85,
                bbox_x=0.5,
                bbox_y=0.5,
                bbox_w=0.1,
                bbox_h=0.1,
            ),
            # Frame 1: no ball → all flagged
            _det(
                1,
                0,
                label="player",
                confidence=0.9,
                bbox_x=0.0,
                bbox_y=0.0,
                bbox_w=0.1,
                bbox_h=0.1,
            ),
            _det(
                1,
                1,
                label="referee",
                confidence=0.9,
                bbox_x=0.5,
                bbox_y=0.5,
                bbox_w=0.1,
                bbox_h=0.1,
            ),
        ]
    )
    report = score_detections(store, game_id="g1", source="yolo", run_id="run1")
    assert report.flagged_no_ball == 2  # only frame 1's detections
    assert report.total_flagged == 2

    df = store.query(
        "SELECT frame_index, detection_id, needs_review FROM detection "
        "WHERE game_id='g1' ORDER BY frame_index, detection_id"
    )
    frame0 = df[df.frame_index == 0]
    frame1 = df[df.frame_index == 1]
    assert not any(frame0["needs_review"])
    assert all(frame1["needs_review"])


def test_idempotent_rescore(store: FeatureStore) -> None:
    """Re-running score_detections on the same data produces the same flags."""
    _frame(store, 0)
    store.upsert_detections([_det(0, 0, confidence=0.1)])

    report1 = score_detections(store, game_id="g1", source="yolo", run_id="run1")
    report2 = score_detections(store, game_id="g1", source="yolo", run_id="run1")
    assert report1.total_flagged == report2.total_flagged

    df = store.query("SELECT needs_review FROM detection WHERE game_id='g1'")
    assert all(df["needs_review"])


def test_empty_run_returns_empty_report(store: FeatureStore) -> None:
    report = score_detections(
        store, game_id="g1", source="yolo", run_id="run_nonexistent"
    )
    assert report.total_detections == 0
    assert report.total_flagged == 0


# --------------------------------------------------------------------------- #
# flag_for_review alias                                                        #
# --------------------------------------------------------------------------- #


def test_flag_for_review_alias(store: FeatureStore) -> None:
    _frame(store, 0)
    store.upsert_detections([_det(0, 0, confidence=0.05)])
    report = flag_for_review(store, game_id="g1", source="yolo", run_id="run1")
    assert report.total_flagged == 1


# --------------------------------------------------------------------------- #
# DifficultyReport string representation                                       #
# --------------------------------------------------------------------------- #


def test_report_str(store: FeatureStore) -> None:
    _frame(store, 0)
    store.upsert_detections([_det(0, 0, confidence=0.1)])
    report = score_detections(store, game_id="g1", source="yolo", run_id="run1")
    s = str(report)
    assert "g1" in s
    assert "yolo" in s
    assert "flagged" in s
