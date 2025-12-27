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


def test_ultralytics_classifier_loading_default():
    """Test that the UltralyticsClassifier can be loaded."""
    # Use a temporary directory for model saves to avoid cluttering the project
    model_name = "yolo11n-cls.pt"
    classifier = UltralyticsClassifier(model_path=model_name)
    assert classifier.model is not None


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_path",
    [
        None,
        Path("model_saves/classifier/20251226-yolo11n-cls/0.987.pt"),
    ],
)
def test_ultralytics_classifier_inference(
    repo_root: Path, model_path: Path | None, extracted_frames: list[Path]
):
    """Test the UltralyticsClassifier on a directory of frames.

    Args:
        extracted_frames (list[Path]): A list of paths to extracted frames.
        tmp_path (Path): A temporary path provided by pytest.
    """

    assert len(extracted_frames) > 0, "No frames were extracted for the test"

    if model_path is not None:
        model_path = repo_root / model_path
    classifier = UltralyticsClassifier(model_path=model_path)

    for image_path in extracted_frames:
        result = classifier.predict_from_path(image_path)
        assert isinstance(result, FrameClassifications)
        assert result.uri == image_path
        assert isinstance(result.classification.label, EnumBroadcastClassification)
        assert 0.0 <= result.classification.confidence <= 1.0
