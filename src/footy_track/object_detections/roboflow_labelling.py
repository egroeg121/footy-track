import logging
import os
from pathlib import Path

from roboflow import Roboflow
from tqdm import tqdm

from footy_track.classifier import Classifier, RoboflowClassifier
from footy_track.object_detections.constants import BALL_TAG, PERSON_TAG
from footy_track.object_detections.detectors import ObjectDetector, UltralyticsObjectDetector
from footy_track.object_detections.schema import FrameDetections

logger = logging.getLogger(__name__)


def cache_path_from_img_path(img_path: Path) -> Path:
    return img_path.parent / "cached_detections" / (img_path.stem + ".json")


class BaseRoboflowHandler:
    """Base handler for Roboflow API interactions."""

    def __init__(self, workspace_name: str, api_key: str = None):
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY environment variable not set.")

        self.rf = Roboflow(api_key=self.api_key)
        self.workspace = self.rf.workspace(workspace_name)
        self._project = None

    def get_or_create_project(
        self, project_name: str, project_type: str, classes: list[str] = None
    ):
        """Get or create a Roboflow project."""
        try:
            project = self.workspace.project(project_name)
            logger.info(f"Using existing project: {project.name}")
        except Exception:
            project = self.workspace.create_project(
                project_name=project_name,
                project_type=project_type,
                classes=classes,
            )
            logger.info(f"Created new project: {project.name}")
        self._project = project
        return project


class RoboflowClassificationHandler(BaseRoboflowHandler):
    """Handler for Roboflow classification projects with yes/no classes."""

    def __init__(
        self, workspace_name: str, project_name: str, classifier: Classifier | None = None
    ):
        super().__init__(workspace_name)
        self.project_name = project_name
        self.project_type = "single-label-classification"
        self.classes = ["yes", "no"]
        self.project = self.get_or_create_project(project_name, self.project_type, self.classes)
        self.classifier = classifier or RoboflowClassifier(
            workspace_name=workspace_name, project_name=project_name, api_key=self.api_key
        )

    def _classify_image(self, img_path: Path) -> str:
        """Placeholder for a real classification model."""
        return self.classifier.predict_from_path(img_path).label

    def upload_images(
        self, image_dir: Path, batch_name: str = "uploads", sample_number: int = None
    ) -> None:
        """Upload images to the classification project with yes/no labels."""
        image_paths = self._load_local_images(image_dir, sample_number)

        for img_path in tqdm(image_paths, desc="Uploading images to Roboflow"):
            classification = self._classify_image(img_path)

            # Create a temporary annotation file with the classification label
            annotation_path = img_path.with_suffix(".txt")
            with open(annotation_path, "w") as f:
                f.write(classification)  # Write the class name (e.g., "yes" or "no")

            try:
                self.project.upload(
                    image_path=str(img_path),
                    annotation_path=str(annotation_path),
                    batch_name=batch_name,
                    is_prediction=True,
                )
            except Exception as e:
                logger.error(f"Failed to upload {img_path}: {e}")
            # finally:
            #     # Clean up the temporary annotation file
            #     if annotation_path.exists():
            #         os.remove(annotation_path)

    def _load_local_images(self, image_dir: Path, sample_number: int = None) -> list[Path]:
        all_image_paths = sorted(Path(image_dir).glob("*.jpg"))
        image_paths = all_image_paths
        if sample_number is not None:
            image_paths = all_image_paths[:: max(1, len(all_image_paths) // sample_number)]
        return image_paths


class RoboflowObjectDetectionHandler(BaseRoboflowHandler):
    """Handler for Roboflow object detection projects."""

    def __init__(
        self, workspace_name: str, project_name: str, detector: ObjectDetector | None = None
    ):
        super().__init__(workspace_name)
        self.project_name = project_name
        self.project_type = "object-detection"
        self.classes = [BALL_TAG, PERSON_TAG]
        self.project = self.get_or_create_project(project_name, self.project_type, self.classes)
        self.detector = detector or UltralyticsObjectDetector(model_uri="yolo11x.pt")

    def _load_local_images(self, image_dir: Path, sample_number: int = None) -> list[Path]:
        all_image_paths = sorted(Path(image_dir).glob("*.jpg"))
        image_paths = all_image_paths
        if sample_number is not None:
            image_paths = all_image_paths[:: max(1, len(all_image_paths) // sample_number)]
        return image_paths

    def _run_object_detection_on_images(
        self, path_to_detections: dict[Path, FrameDetections], cache: bool = True
    ) -> dict[Path, FrameDetections]:
        for img_path, det in tqdm(path_to_detections.items(), total=len(path_to_detections)):
            cache_path = cache_path_from_img_path(img_path)
            if det is not None:
                continue
            elif cache and cache_path.exists():
                loaded_det = FrameDetections.model_validate_json(cache_path.read_text())
            else:
                loaded_det = self.detector.predict_from_path(img_path)

            if cache and loaded_det:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    f.write(loaded_det.model_dump_json())

            path_to_detections[img_path] = loaded_det
        return path_to_detections

    def upload_images(
        self, image_dir: Path, pre_annotate: bool = True, batch_name: str = "uploads"
    ) -> None:
        """Upload images to the dataset associated with this project."""
        image_paths = self._load_local_images(image_dir)
        path_to_detections: dict[Path, FrameDetections] = dict.fromkeys(image_paths)

        if pre_annotate:
            path_to_detections = self._run_object_detection_on_images(path_to_detections)

        for img_path in tqdm(image_paths, desc="Uploading images to Roboflow"):
            annotation_path = None

            detections = path_to_detections.get(img_path)
            if detections and detections.detections:
                annotation_path = self._write_roboflow_annotations(detections, img_path)

            try:
                self.project.upload(
                    image_path=str(img_path),
                    annotation_path=annotation_path,
                    batch_name=batch_name,
                    is_prediction=True,
                )
            except Exception as e:
                logger.error(f"Failed to upload {img_path}: {e}")
            finally:
                if annotation_path:
                    os.remove(annotation_path)

    def _write_roboflow_annotations(self, det: FrameDetections, img_path: Path) -> str:
        """Converts FrameDetections to Roboflow's TXT format and writes to a temporary file."""
        annotation_path = img_path.with_suffix(".txt")
        with open(annotation_path, "w") as f:
            for d in det.detections:
                # Roboflow format: <class_index> <x_center> <y_center> <width> <height>
                class_index = 0 if d.label == BALL_TAG else 1  # Example class mapping
                x_center = d.x + d.w / 2
                y_center = d.y + d.h / 2
                f.write(f"{class_index} {x_center} {y_center} {d.w} {d.h}\n")
        return str(annotation_path)
