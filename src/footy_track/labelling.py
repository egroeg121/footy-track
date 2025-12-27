import logging
import os
import random
from pathlib import Path

from roboflow import Roboflow

from footy_track.classifier import Classifier, get_current_best_guess_classifier
from footy_track.constants import IMAGE_FORMAT
from footy_track.detectors.base import ObjectDetector
from footy_track.schema import FrameClassifications, FrameDetections

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
            classifier = get_current_best_guess_classifier()
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


class RoboflowObjectDetectionHandler(BaseRoboflowHandler):
    """Handler for Roboflow object detection projects."""

    def __init__(
        self,
        workspace_name: str,
        project_name: str,
        detector: ObjectDetector | None = None,
        classifier: "Classifier | None" = None,
    ):
        from footy_track.detectors.ultralytics import UltralyticsObjectDetector

        super().__init__(workspace_name)
        self.project_name = project_name
        self.project = self.get_project(project_name)
        self.detector = detector or UltralyticsObjectDetector()
        self.classifier = classifier or get_current_best_guess_classifier()

    def upload_dir(
        self,
        image_dir: Path,
        sample_number: int = 0,
        batch_name: str = "uploads",
        pre_annotate: bool = True,
        filter_by_broadcast_classifier: bool = True,
    ) -> None:
        """Uploads a directory of images to the project."""
        image_paths = self._load_local_images(image_dir, sample_number)
        return self.upload_images(
            image_paths,
            batch_name=batch_name,
            pre_annotate=pre_annotate,
            filter_by_broadcast_classifier=filter_by_broadcast_classifier,
        )

    def upload_images(
        self,
        image_paths: list[Path],
        batch_name: str = "uploads",
        pre_annotate: bool = True,
        filter_by_broadcast_classifier: bool = True,
    ) -> None:
        """Upload images to the dataset associated with this project."""
        from tqdm import tqdm

        path_to_detections: dict[Path, FrameDetections] = dict.fromkeys(image_paths)

        if filter_by_broadcast_classifier and self.classifier:
            filtered_image_paths = []
            for img_path in tqdm(
                image_paths, desc="Filtering images by broadcast classifier"
            ):
                classification = self.classifier.predict_from_path(img_path)
                if classification.classification.label == "Yes":
                    filtered_image_paths.append(img_path)
            path_to_detections = dict.fromkeys(filtered_image_paths)

        if pre_annotate:
            for img_path in tqdm(
                path_to_detections.keys(), desc="Running object detection"
            ):
                path_to_detections[img_path] = self.detector.predict_from_path(
                    image_path=img_path
                )

        for img_path, detections in tqdm(
            path_to_detections.items(), desc="Uploading images to Roboflow"
        ):
            if detections and detections.detections:
                annotation_path = img_path.with_suffix(".txt")
                with open(annotation_path, "w") as f:
                    for det in detections.detections:
                        if det.label not in self.project.classes:
                            _logger.warning(
                                f"Skipping detection with unknown label: {det.label}. "
                                f"Allowed classes are: {self.project.classes}"
                            )
                            continue
                        # darknet format
                        f.write(
                            f"{self.project.classes[det.label]} {det.x} {det.y} {det.w} {det.h}\n"
                        )

                self.project.upload(
                    image_path=str(img_path),
                    annotation_path=str(annotation_path),
                    batch_name=batch_name,
                    is_prediction=True,
                )
                annotation_path.unlink()
            else:
                self.project.upload(
                    image_path=str(img_path),
                    batch_name=batch_name,
                    is_prediction=False,
                )

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
