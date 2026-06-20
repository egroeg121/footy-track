"""Tests for per-class confidence threshold filtering in UltralyticsObjectDetector.

These tests mock YOLO inference to avoid requiring a checkpoint on disk.
"""

from __future__ import annotations
from pathlib import Path
from unittest.mock import patch

import pytest

from footy_track.detectors.ultralytics import UltralyticsObjectDetector
from footy_track.schema import ObjectDetection


def _make_mock_detector(
    raw_detections: list[ObjectDetection],
    min_confidence: float = 0.3,
    class_thresholds: dict[str, float] | None = None,
) -> UltralyticsObjectDetector:
    """Create a detector with mocked YOLO and pre-set raw detections."""
    with patch("footy_track.detectors.ultralytics.YOLO"):
        detector = UltralyticsObjectDetector(
            model_uri="dummy.pt",
            min_confidence=min_confidence,
            class_thresholds=class_thresholds,
        )

    # Patch predict_from_path to bypass YOLO and inject raw detections
    from footy_track.schema import FrameDetections

    def _fake_predict(image_path: Path) -> FrameDetections:
        # Replicate the post-filter logic from predict_from_path
        detections = list(raw_detections)
        if detector._class_thresholds:
            detections = [
                d
                for d in detections
                if d.confidence >= detector._class_thresholds.get(d.label, detector._min_confidence)
            ]
        return FrameDetections(uri=image_path, width=1920, height=1080, detections=detections)

    detector.predict_from_path = _fake_predict  # type: ignore[method-assign]
    return detector


class TestPerClassThresholds:
    def _raw_dets(self) -> list[ObjectDetection]:
        return [
            ObjectDetection(label="ball", confidence=0.22, x=0.5, y=0.5, w=0.02, h=0.02),
            ObjectDetection(label="person", confidence=0.35, x=0.1, y=0.3, w=0.05, h=0.1),
            ObjectDetection(label="person", confidence=0.18, x=0.2, y=0.3, w=0.05, h=0.1),
        ]

    def test_no_class_thresholds_conf_passed_to_yolo(self):
        """Without class_thresholds, the global conf is passed directly to YOLO."""
        with patch("footy_track.detectors.ultralytics.YOLO"):
            detector = UltralyticsObjectDetector(model_uri="dummy.pt", min_confidence=0.3)
        # No class_thresholds — YOLO itself enforces the threshold, not post-filter
        assert detector.predict_kwargs["conf"] == pytest.approx(0.3)
        assert detector._class_thresholds == {}

    def test_ball_threshold_lower_than_global(self):
        """Ball-specific threshold 0.2 passes the ball but not the 0.18 person."""
        detector = _make_mock_detector(
            self._raw_dets(),
            min_confidence=0.3,
            class_thresholds={"ball": 0.2},
        )
        result = detector.predict_from_path(Path("frame.png"))
        labels = [d.label for d in result.detections]
        assert "ball" in labels
        # Person at 0.35 passes global threshold; person at 0.18 does not
        persons = [l for l in labels if l == "person"]
        assert len(persons) == 1

    def test_ball_threshold_filters_low_conf_ball(self):
        """A ball at 0.15 is filtered even with ball threshold 0.2."""
        dets = [ObjectDetection(label="ball", confidence=0.15, x=0.5, y=0.5, w=0.02, h=0.02)]
        detector = _make_mock_detector(dets, min_confidence=0.3, class_thresholds={"ball": 0.2})
        result = detector.predict_from_path(Path("frame.png"))
        assert result.detections == []

    def test_effective_conf_is_min_of_global_and_class(self):
        """The YOLO predict_kwargs conf is set to min(global, min(class_thresholds))."""
        with patch("footy_track.detectors.ultralytics.YOLO"):
            detector = UltralyticsObjectDetector(
                model_uri="dummy.pt",
                min_confidence=0.3,
                class_thresholds={"ball": 0.15},
            )
        assert detector.predict_kwargs["conf"] == pytest.approx(0.15)

    def test_effective_conf_unchanged_when_class_threshold_higher(self):
        """If class threshold is higher than global, conf stays at global."""
        with patch("footy_track.detectors.ultralytics.YOLO"):
            detector = UltralyticsObjectDetector(
                model_uri="dummy.pt",
                min_confidence=0.3,
                class_thresholds={"ball": 0.5},
            )
        assert detector.predict_kwargs["conf"] == pytest.approx(0.3)

    def test_multiple_class_thresholds(self):
        """Multiple classes can have individual thresholds."""
        dets = [
            ObjectDetection(label="ball", confidence=0.2, x=0.5, y=0.5, w=0.02, h=0.02),
            ObjectDetection(label="in_play_ball", confidence=0.2, x=0.6, y=0.5, w=0.02, h=0.02),
            ObjectDetection(label="person", confidence=0.25, x=0.1, y=0.3, w=0.05, h=0.1),
        ]
        detector = _make_mock_detector(
            dets,
            min_confidence=0.3,
            class_thresholds={"ball": 0.15, "in_play_ball": 0.15},
        )
        result = detector.predict_from_path(Path("frame.png"))
        labels = [d.label for d in result.detections]
        assert "ball" in labels
        assert "in_play_ball" in labels
        # person (0.25) below global (0.3) — filtered
        assert "person" not in labels
