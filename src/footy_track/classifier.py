import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path

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
