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


@pytest.mark.slow
def test_grounding_dino_detector_ball_only(image_path: Path, grounding_dino_detector):
    frame = grounding_dino_detector.predict_from_path(image_path)
    assert_frame_schema(frame, image_path)

    # Grounding DINO adapter maps all detections to the canonical label "football"
    for d in frame.detections:
        assert d.label == "football"

    # Test serialization
    dumped = frame.model_dump()
    assert Path(dumped["uri"]).name == image_path.name
