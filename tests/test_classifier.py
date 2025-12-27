"""Tests for the classifier module."""

from pathlib import Path

import pytest

from footy_track.classifier import RandomClassifier, UltralyticsClassifier
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


@pytest.mark.slow
def test_ultralytics_classifier_loading(tmp_path: Path):
    """Test that the UltralyticsClassifier can be loaded."""
    # Use a temporary directory for model saves to avoid cluttering the project
    model_dir = tmp_path / "model_saves"
    model_name = "yolo11n-cls.pt"
    classifier = UltralyticsClassifier(model_name=model_name, model_dir=model_dir)
    assert classifier.model is not None
    assert (model_dir / model_name).exists()


@pytest.mark.slow
def test_ultralytics_classifier_inference(extracted_frames: list[Path], tmp_path: Path):
    """Test the UltralyticsClassifier on a directory of frames.

    Args:
        extracted_frames (list[Path]): A list of paths to extracted frames.
        tmp_path (Path): A temporary path provided by pytest.
    """
    model_dir = tmp_path / "model_saves"
    classifier = UltralyticsClassifier(model_dir=model_dir)
    assert len(extracted_frames) > 0, "No frames were extracted for the test"

    for image_path in extracted_frames:
        result = classifier.predict_from_path(image_path)
        assert isinstance(result, FrameClassifications)
        assert result.uri == image_path
        assert isinstance(result.classification.label, EnumBroadcastClassification)
        assert 0.0 <= result.classification.confidence <= 1.0
