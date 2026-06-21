import logging
import random
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from ultralytics import YOLO

from footy_track.schema import (
    BroadcastClassification,
    EnumBroadcastClassification,
    FrameClassifications,
)

logger = logging.getLogger(__name__)


# This is set a the bottom of the file using the function after the class definitions and then we run the function, but this
# makes it easy to see and change when there is a better classifier available.
CURRENT_BEST_GUESS_CLASSIFIER_CLASS = None


def get_current_best_guess_classifier() -> "Classifier":
    """Returns the current best guess classifier class."""
    # Try to load the local model; fall back to random if unavailable
    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / "model_saves/classifier/20251226-yolo11n-cls/0.987.pt"

    try:
        return UltralyticsClassifier(model_path=model_path)
    except Exception as e:
        logger.warning(f"Failed to load classifier model ({e}), using fallback")
        return RandomClassifier()


class Classifier(ABC):
    """Base class fiftyoner classifiers."""

    @abstractmethod
    def predict_from_path(self, image_path: Path) -> FrameClassifications:
        """Predict the class of an image from its path."""
        raise NotImplementedError


class RandomClassifier(Classifier):
    """A classifier that randomly assigns classes."""

    def __init__(self):
        super().__init__()

        self.random = random
        self.classes = list(EnumBroadcastClassification)

    def predict_from_path(self, image_path: Path) -> FrameClassifications:
        label = self.random.choice(self.classes)
        confidence = self.random.uniform(0.5, 1.0)
        return FrameClassifications(
            uri=image_path,
            classification=BroadcastClassification(label=label, confidence=confidence),
        )


class UltralyticsClassifier(Classifier):
    """A classifier that uses the Ultralytics YOLO model."""

    def __init__(
        self,
        model_path: str | Path = None,
        model_dir: Path | None = None,
    ):
        """Load the Ultralytics YOLO classification model.
        model_path: str | Path, path to a local model or name of a model to download
        model_dir: Path if models are downloaded, this is where to save them
        """
        if model_path is None:
            model_path = "yolo11n-cls.pt"
        if model_dir is None:
            model_dir = Path(tempfile.gettempdir()) / "yolo_models"
        super().__init__()
        self.model = self._load_model(model_path=model_path, model_dir=model_dir)

    def _load_model(self, model_path: str | Path, model_dir: Path) -> YOLO:
        """Loads the YOLO model, downloading it if necessary.

        Args:
            model_path (str | Path): The path to a local model or the name of a model to download.
            model_dir (Path): The directory to save the model in.

        Returns:
            YOLO: The loaded YOLO model.
        """
        model_name_path = Path(model_path)
        if model_name_path.is_file():
            logger.info(f"Loading model directly from {model_name_path}")
            return YOLO(model_name_path)

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_download_path = model_dir / model_name_path.name

        if not model_download_path.exists():
            logger.info(f"Model not found at {model_download_path}, downloading...")
            # Load normally (downloads to cwd)
            YOLO(model_name_path.name)
            # Move the file
            downloaded_model = Path(model_name_path.name)
            if downloaded_model.exists():
                downloaded_model.rename(model_download_path)
            else:
                raise FileNotFoundError(
                    f"Failed to download model {model_name_path.name}, it should exist"
                )
        return YOLO(model_download_path)

    def _check_output(self, predicted_class_label: str) -> EnumBroadcastClassification:
        """Checks the predicted class label and returns a broadcast classification."""
        try:
            # Attempt to pass into classification, otherwise No
            return EnumBroadcastClassification(predicted_class_label)
        except ValueError:
            logger.debug(
                f"Predicted class label '{predicted_class_label}' is not a valid broadcast classification."
                " Returning NO."
            )
        return EnumBroadcastClassification.NO

    def _model_infer(self, image_path: Path):
        return self.model.predict(source=image_path, device="mps", verbose=False)

    def predict_from_path(self, image_path: Path) -> FrameClassifications:
        """Predict the class of an image from its path."""
        results = self._model_infer(image_path=image_path)
        result = results[0]
        top1_index = result.probs.top1
        top1_confidence = result.probs.top1conf.item()
        predicted_class_label = self.model.names[top1_index]

        classification = self._check_output(predicted_class_label)

        return FrameClassifications(
            uri=image_path,
            classification=BroadcastClassification(
                label=classification, confidence=top1_confidence
            ),
        )


CURRENT_BEST_GUESS_CLASSIFIER_CLASS = get_current_best_guess_classifier()
