import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path

from ultralytics import YOLO

from footy_track.schema import (
    BroadcastClassification,
    EnumBroadcastClassification,
    FrameClassifications,
)

logger = logging.getLogger(__name__)


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
        model_name: str = "yolo11n-cls.pt",
        model_dir: Path = Path("model_saves/classifier"),
    ):
        super().__init__()
        self.model = self._load_model(model_name, model_dir)

    def _load_model(self, model_name: str, model_dir: Path) -> YOLO:
        """Loads the YOLO model, downloading it if necessary.

        Args:
            model_name (str): The name of the model to load.
            model_dir (Path): The directory to save the model in.

        Returns:
            YOLO: The loaded YOLO model.
        """
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / model_name

        if not model_path.exists():
            logger.info(f"Model not found at {model_path}, downloading...")
            # Load normally (downloads to cwd)
            YOLO(model_name)
            # Move the file
            downloaded_model = Path(model_name)
            if downloaded_model.exists():
                downloaded_model.rename(model_path)
            else:
                raise FileNotFoundError(f"Failed to download model {model_name}, it should exist")
        return YOLO(model_path)

    def _check_output(self, predicted_class_label: str) -> EnumBroadcastClassification:
        """Checks the predicted class label and returns a broadcast classification.

        This method can be overridden by subclasses to implement custom logic.

        Args:
            predicted_class_label (str): The class label predicted by the model.

        Returns:
            EnumBroadcastClassification: The broadcast classification.
        """
        # HACK: a little fudging to get the desired output
        if predicted_class_label in ("sports ball", "soccer ball"):
            return EnumBroadcastClassification.No
        else:
            return EnumBroadcastClassification.Yes

    def predict_from_path(self, image_path: Path) -> FrameClassifications:
        """Predict the class of an image from its path."""
        results = self.model(image_path, verbose=False)
        result = results[0]
        top1_index = result.probs.top1
        top1_confidence = result.probs.top1conf.item()
        predicted_class_label = self.model.names[top1_index]

        label = self._check_output(predicted_class_label)

        return FrameClassifications(
            uri=image_path,
            classification=BroadcastClassification(label=label, confidence=top1_confidence),
        )
