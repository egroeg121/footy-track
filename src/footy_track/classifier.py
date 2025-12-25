import logging
import os
from enum import StrEnum
from pathlib import Path

import fiftyone as fo
from inference import get_model
from pydantic import BaseModel
from roboflow import Roboflow

logger = logging.getLogger(__name__)


class EnumBroadcastClassification(StrEnum):
    """Enumeration for broadcast classification."""

    YES = "Yes"
    NO = "No"
    UNLABELED = "Unlabeled"


class ClassificationResult(BaseModel):
    """Schema for a classification result."""

    label: EnumBroadcastClassification
    confidence: float

    def to_fiftyone_sample(self, image_path: Path, key: str = "prediction") -> fo.Sample:
        """Convert to FiftyOne format."""
        sample = fo.Sample(filepath=str(image_path))
        sample[key] = fo.Classification(label=str(self.label), confidence=self.confidence)
        return sample


class Classifier:
    """Base class for classifiers."""

    def predict_from_path(self, image_path: Path) -> ClassificationResult:
        """Predict the class of an image from its path."""
        raise NotImplementedError


class RoboflowClassifier(Classifier):
    """
    Classifier that uses a model from Roboflow for inference.
    """

    def __init__(
        self,
        workspace_name: str,
        project_name: str,
        version_number: int | None = None,
        api_key: str = None,
    ):
        """
        Args:
            workspace_name (str): The name of your Roboflow workspace.
            project_name (str): The name of your Roboflow project.
            version_number (int): The version number of the model to use.
            api_key (str, optional): Your Roboflow API key. Defaults to None,
                in which case it will be read from the ROBOFLOW_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable not set.")

        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(workspace_name).project(project_name)
        if version_number is None:
            versions = [v for v in project.versions() if v.model is not None]
            version_obj = max(versions, key=lambda v: v.created)
            version_number = int(version_obj.version)
        version = project.version(version_number)
        logger.info(f"Loading version {version.version}")
        model = version.model
        self.model = get_model(
            model_id=f"{model.name}/{version.version}",
            onnxruntime_execution_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )

    def predict_from_path(self, image_path: Path) -> ClassificationResult:
        """
        Predict the class of an image from its path using the Roboflow model.

        Args:
            image_path (Path): The path to the image file.

        Returns:
            ClassificationResult: The classification result.
        """

        predictions = self.model.infer(str(image_path))

        # Find the prediction with the highest confidence
        top_prediction = predictions[0]

        return ClassificationResult(label=top_prediction.top, confidence=top_prediction.confidence)
