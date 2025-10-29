"""Unit tests for object detectors returning Pydantic outputs.

Runs both Ultralytics YOLO and Grounding DINO (ball-only) on a sample frame and
verifies schema, normalization, and basic serialization. Marked `slow` as they
invoke real models.
"""

from pathlib import Path

import pytest

from footy_track.object_detections import (
    Detection,
    FrameDetections,
)
from footy_track.object_detections.detectors import (
    BALL_TAG,
    GROUND_DINO_PROMPT_TO_CLASS,
    PERSON_TAG,
    GroundingDinoObjectDetector,
)


def assert_detection_schema(det: Detection):
    assert isinstance(det.label, str)
    assert 0.0 <= det.confidence <= 1.0
    assert 0.0 <= det.x <= 1.0
    assert 0.0 <= det.y <= 1.0
    assert 0.0 <= det.w <= 1.0
    assert 0.0 <= det.h <= 1.0


def assert_frame_schema(frame: FrameDetections, image_path: Path):
    assert isinstance(frame, FrameDetections)
    assert Path(frame.uri) == image_path
    assert frame.width > 0 and frame.height > 0
    assert isinstance(frame.detections, list)
    for d in frame.detections:
        assert_detection_schema(d)


@pytest.mark.slow
def test_ultralytics_detector_runs(image_path: Path, ultralytics_detector):
    frame = ultralytics_detector.predict_from_path(image_path)
    assert_frame_schema(frame, image_path)

    # If any detections, ensure labels are strings and normalization holds
    if frame.detections:
        for d in frame.detections:
            assert isinstance(d.label, str)
            assert 0.0 <= d.x <= 1.0
            assert 0.0 <= d.y <= 1.0
            assert 0.0 <= d.w <= 1.0
            assert 0.0 <= d.h <= 1.0

    # Test pydantic serialization roundtrip
    dumped = frame.model_dump()
    assert dumped["width"] == frame.width
    assert dumped["height"] == frame.height


def _iou_xywh(
    box_a: tuple[float, float, float, float] | list[float],
    box_b: tuple[float, float, float, float] | list[float],
) -> float:
    """IoU for normalized [x, y, w, h] boxes."""
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


@pytest.mark.slow
def test_grounding_dino_detector_ball_only(
    arsenal_mancity_image_path: Path,
    arsenal_mancity_test_detection_as_frame: FrameDetections,
):
    # Ground truth from curated FrameDetections fixture
    gt_frame = arsenal_mancity_test_detection_as_frame
    gt_ball_boxes = [
        [d.x, d.y, d.w, d.h]
        for d in gt_frame.detections
        if d.label in GROUND_DINO_PROMPT_TO_CLASS[BALL_TAG]
    ]
    assert gt_ball_boxes, "No ground-truth ball boxes in curated frame"

    detector = GroundingDinoObjectDetector()
    frame = detector.predict_from_path(arsenal_mancity_image_path)
    assert_frame_schema(frame, arsenal_mancity_image_path)

    # Predicted balls from Grounding DINO
    pred_ball_boxes = [[d.x, d.y, d.w, d.h] for d in frame.detections if d.label == BALL_TAG]
    assert pred_ball_boxes, "Detector did not return any 'ball' detections"

    iou_threshold = 0.5
    required_ratio = 0.5

    # Predicted -> GT: proportion of predicted boxes that match some GT box
    matched_pred = 0
    for pb in pred_ball_boxes:
        best_iou = max((_iou_xywh(pb, gb) for gb in gt_ball_boxes), default=0.0)
        if best_iou >= iou_threshold:
            matched_pred += 1
    pred_ratio = matched_pred / len(pred_ball_boxes)
    assert pred_ratio >= required_ratio, (
        f"Only {matched_pred}/{len(pred_ball_boxes)} predicted balls "
        f"({pred_ratio:.2%}) have IoU >= {iou_threshold}; required >= {required_ratio:.2%}"
    )

    # GT -> Predicted: proportion of GT boxes that are matched by some prediction
    matched_gt = 0
    for gb in gt_ball_boxes:
        best_iou = max((_iou_xywh(pb, gb) for pb in pred_ball_boxes), default=0.0)
        if best_iou >= iou_threshold:
            matched_gt += 1
    gt_ratio = matched_gt / len(gt_ball_boxes)
    assert gt_ratio >= required_ratio, (
        f"Only {matched_gt}/{len(gt_ball_boxes)} GT balls ({gt_ratio:.2%}) are matched "
        f"with IoU >= {iou_threshold}; required >= {required_ratio:.2%}"
    )


@pytest.mark.slow
def test_grounding_dino_detector_players_ref_coach_etc(
    arsenal_mancity_image_path: Path,
    arsenal_mancity_test_detection_as_frame: FrameDetections,
):
    ground_truth_person_tags = ["player_red", "player_blue", "coach", "referee", "goalkeeper"]

    # Ground truth from curated FrameDetections fixture (persons)
    gt_frame = arsenal_mancity_test_detection_as_frame
    gt_person_boxes = [
        [d.x, d.y, d.w, d.h] for d in gt_frame.detections if d.label in ground_truth_person_tags
    ]
    assert gt_person_boxes, "No ground-truth person boxes in curated frame"

    detector = GroundingDinoObjectDetector()
    frame = detector.predict_from_path(arsenal_mancity_image_path)
    assert_frame_schema(frame, arsenal_mancity_image_path)

    # Predicted persons from Grounding DINO
    pred_person_boxes = [[d.x, d.y, d.w, d.h] for d in frame.detections if d.label == PERSON_TAG]
    assert pred_person_boxes, "Detector did not return any 'person' detections"

    iou_threshold = 0.5
    required_gt_detection_ratio = 0.75

    # GT -> Predicted: proportion of GT person boxes that are matched by some prediction
    matched_gt = 0
    for gb in gt_person_boxes:
        best_iou = max((_iou_xywh(pb, gb) for pb in pred_person_boxes), default=0.0)
        if best_iou >= iou_threshold:
            matched_gt += 1
    gt_ratio = matched_gt / len(gt_person_boxes)
    assert gt_ratio >= required_gt_detection_ratio, (
        f"Only {matched_gt}/{len(gt_person_boxes)} GT persons ({gt_ratio:.2%}) are matched "
        f"with IoU >= {iou_threshold}; required >= {required_gt_detection_ratio:.2%}"
    )
