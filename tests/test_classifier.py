"""Tests for the classifier module."""

from pathlib import Path

from footy_track.classifier import RandomClassifier
from footy_track.schema import EnumBroadcastClassification, FrameClassifications


def test_random_classifier(extracted_frames: list[Path]):
    """Test the RandomClassifier on a directory of frames.

    Args:
        extracted_frames (list[Path]): A list of paths to extracted frames.
    """

    classifier = RandomClassifier()
    assert len(extracted_frames) > 0, "No frames were extracted for the test"

    for image_path in extracted_frames:
        result = classifier.predict_from_path(image_path)
        assert isinstance(result, FrameClassifications)
        assert result.uri == image_path
        assert isinstance(result.classification.label, EnumBroadcastClassification)
        assert 0.5 <= result.classification.confidence <= 1.0
