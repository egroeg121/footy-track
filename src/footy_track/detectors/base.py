from abc import ABC, abstractmethod
from pathlib import Path

from footy_track.schema import FrameDetections


class ObjectDetector(ABC):
    """Abstract base class for object detectors."""

    @abstractmethod
    def predict_from_path(self, image_path: Path) -> FrameDetections:
        """Run detection on an image from a path and return FrameDetections."""
        raise NotImplementedError
