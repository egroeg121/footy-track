from pathlib import Path

from footy_track.detectors.ultralytics import UltralyticsObjectDetector
from footy_track.schema import FrameDetections


def test_ultralytics_object_detector_init():
    """Test that the UltralyticsObjectDetector can be initialized."""
    detector = UltralyticsObjectDetector()
    assert detector is not None


def test_ultralytics_object_detector_predict_from_path():
    """Test that the predict_from_path method returns a FrameDetections object."""
    detector = UltralyticsObjectDetector()
    test_image_path = Path("tests/data/arsenal_mancity_test_detection.jpg")
    frame_detections = detector.predict_from_path(test_image_path)
    assert isinstance(frame_detections, FrameDetections)
    assert frame_detections.uri == test_image_path
    assert len(frame_detections.detections) > 0
