import os
from pathlib import Path

from inference import get_model
from pydantic import BaseModel
from roboflow import Roboflow


class ClassificationResult(BaseModel):
    """Schema for a classification result."""

    label: str
    confidence: float


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
        version: int | None = None,
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
        # version_number = version or max(versions, key=lambda v: int(v.version))
        version_number = 1
        model = project.version(version_number).model
        self.model = get_model(
            model_id=f"{model.name}/{version_number}",
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
