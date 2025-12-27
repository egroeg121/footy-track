import logging
import os
import random
from pathlib import Path

from roboflow import Roboflow

from footy_track.classifier import CURRENT_BEST_GUESS_CLASSIFIER_CLASS, Classifier
from footy_track.constants import IMAGE_FORMAT
from footy_track.schema import FrameClassifications

_logger = logging.getLogger(__name__)


class BaseRoboflowHandler:
    """Base handler for Roboflow API interactions."""

    def __init__(self, workspace_name: str, api_key: str = None):
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable not set.")

        self.rf = Roboflow(api_key=self.api_key)
        self.workspace = self.rf.workspace(workspace_name)
        self._project = None

    def get_project(self, project_name: str):
        """Get a Roboflow project."""

        project = self.workspace.project(project_name)
        _logger.info(f"Using existing project: {project.name}")
        self._project = project
        return project

    def _download_roboflow_dataset(
        self, project_name: str, version_number: int, data_location: Path
    ) -> Path:
        """Download and set up a roboflow dataset for training"""
        project = self.workspace.project(project_name)
        version = project.version(version_number)
        dataset_folder = f"roboflow_dataset_{version_number}"
        dataset_path = data_location / dataset_folder
        dataset = version.download(model_format="folder", location=str(dataset_path))
        dataset_location = dataset.location
        _logger.info(f"Dataset downloaded to: {dataset_location}")
        return Path(dataset_location)


class RoboflowClassificationHandler(BaseRoboflowHandler):
    """Handler for Roboflow classification projects with yes/no classes."""

    def __init__(
        self,
        workspace_name: str,
        project_name: str,
        classifier: Classifier | None = None,
    ):
        super().__init__(workspace_name)
        self.project_name = project_name
        self.project = self.get_project(project_name)
        if classifier is None:
            classifier = CURRENT_BEST_GUESS_CLASSIFIER_CLASS
        self.classifier = classifier

    def download_dataset(self, version_number: int, data_location: Path) -> Path:
        """Downloads a dataset from Roboflow."""
        return self._download_roboflow_dataset(
            project_name=self.project_name,
            version_number=version_number,
            data_location=data_location,
        )

    def _classify_image(self, img_path: Path) -> FrameClassifications:
        """Runs the classifier on a local image path."""
        return self.classifier.predict_from_path(img_path)

    def upload_dir(
        self, image_dir: Path, sample_number: int = 0, batch_name: str = "uploads"
    ) -> None:
        """Uploads a directory of images to the project.

        Args:
            image_dir (Path): The directory of images to upload.
            sample_number (int, optional): The number of images to sample from the directory.
                Defaults to 0, which means all images will be uploaded.
            batch_name (str, optional): The name of the batch to upload the images to.
                Defaults to "uploads".
        """
        image_paths = self._load_local_images(image_dir, sample_number)
        return self.upload_images(image_paths, batch_name=batch_name)

    def upload_images(
        self,
        image_paths: list[Path],
        batch_name: str = "uploads",
        annotation_confidence_min: float = 0.8,
    ) -> None:
        """Upload images to the classification project with yes/no labels."""

        for img_path in image_paths:
            annotation_path = None
            is_prediction = False
            if self.classifier:
                frame_classification = self._classify_image(img_path)
                classification = frame_classification.classification

                if classification.confidence >= annotation_confidence_min:
                    # Create a temporary annotation file with the classification label
                    annotation_path = str(img_path.with_suffix(".txt"))
                    with open(annotation_path, "w") as f:
                        f.write(classification.label.value)
                is_prediction = True

            try:
                self.project.upload(
                    image_path=str(img_path),
                    annotation_path=annotation_path,
                    batch_name=batch_name,
                    is_prediction=is_prediction,
                )
            except Exception as e:
                _logger.error(f"Failed to upload {img_path}: {e}")
            finally:
                # Clean up the temporary annotation file
                if annotation_path and Path(annotation_path).exists():
                    os.remove(annotation_path)

    def _load_local_images(self, image_dir: Path, sample_number: int = 0) -> list[Path]:
        image_extensions = [f"*.{IMAGE_FORMAT}"]
        all_image_paths = []
        for ext in image_extensions:
            all_image_paths.extend(Path(image_dir).glob(ext))

        if (
            not sample_number
            or sample_number <= 0
            or sample_number >= len(all_image_paths)
        ):
            return sorted(all_image_paths)
        else:
            return sorted(random.sample(all_image_paths, k=sample_number))
