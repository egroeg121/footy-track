import json
import logging
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

from roboflow import Roboflow
from tqdm.auto import tqdm

from footy_track import constants
from footy_track.classifier import Classifier, get_current_best_guess_classifier
from footy_track.detectors.base import ObjectDetector
from footy_track.detectors.ultralytics import (
    UltralyticsSam3Detector,
)
from footy_track.schema import (
    EnumBroadcastClassification,
    FrameClassifications,
    FrameDetections,
)

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

    def _load_local_images(
        self, image_dir: Path, sample_number: int | None = None
    ) -> list[Path]:
        image_extensions = [f"*.{constants.IMAGE_FORMAT}"]
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

    @property
    def classifier_name(self) -> str:
        """Returns the name of the classifier."""
        return self.classifier.__class__.__name__

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


class RoboflowObjectDetectionHandler(BaseRoboflowHandler):
    """Handler for Roboflow object detection projects."""

    def __init__(
        self,
        workspace_name: str = constants.ROBOFLOW_WORKSPACE,
        project_name: str = constants.ROBOFLOW_DETECTION_PROJECT,
        detector: ObjectDetector | None = None,
        classifier: "Classifier | None" = None,
    ):
        super().__init__(workspace_name)
        self.project_name = project_name
        self.project = self.get_project(project_name)
        self.detector = detector or UltralyticsSam3Detector()
        self.classifier = classifier or get_current_best_guess_classifier()
        # Roboflow returns classes as {name: index}
        # Keep a local mapping for COCO categories
        # self.class_map: dict[str, int] = dict(self.project.classes) <- Think this is wrong
        self.class_map = [
            {"id": i, "name": name} for i, name in enumerate(self.project.classes)
        ]

    @property
    def classifier_name(self) -> str:
        """Returns the name of the classifier."""
        return self.classifier.__class__.__name__

    @property
    def detector_name(self) -> str:
        """Returns the name of the detector."""
        return self.detector.model_tag

    def upload_dir(
        self,
        image_dir: Path,
        sample_number: int = 0,
        batch_name: str = "uploads",
        pre_annotate: bool = True,
        filter_by_broadcast_classifier: bool = True,
    ) -> dict[str, int]:
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
    ) -> dict[str, int]:
        """Upload images to the dataset associated with this project."""

        result_counter: dict[str, int] = defaultdict(int)

        for img_path in tqdm(image_paths, desc="Processing and uploading images"):
            # Step 1: Filter by broadcast classifier
            if filter_by_broadcast_classifier and self.classifier:
                classification = self.classifier.predict_from_path(img_path)
                if (
                    classification.classification.label
                    == EnumBroadcastClassification.NO
                ):
                    result_counter["non_broadcast"] += 1
                    continue

            # Step 2: Pre-annotate with object detector
            detections = None
            if pre_annotate:
                detections = self.detector.predict_from_path(image_path=img_path)
                if not detections or not detections.detections:
                    result_counter["no_detections"] += 1
                    continue  # Don't upload if pre-annotating and no detections found

            # Step 3: Upload image with annotations (if any)
            annotation_path = None
            try:
                if detections:
                    # Write a temporary COCO JSON file for this single image
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".json", delete=False, encoding="utf-8"
                    ) as tmp_f:
                        annotation_path = Path(tmp_f.name)
                        self._write_coco_annotations(
                            {img_path: detections}, annotation_path
                        )

                self.project.upload(
                    image_path=str(img_path),
                    annotation_path=str(annotation_path) if annotation_path else None,
                    batch_name=batch_name,
                    is_prediction=True,
                    num_retry_uploads=2,
                    tag_names=[self.detector_name, batch_name],
                )
                result_counter["uploaded"] += 1
            except Exception as exc:
                _logger.error(f"Failed to upload {img_path}: {exc}")
                result_counter["error"] += 1
            finally:
                if annotation_path and annotation_path.exists():
                    os.remove(annotation_path)

        _logger.info(f"Stats: {result_counter}")
        return result_counter

    def _write_coco_annotations(
        self, path_to_detections: dict[Path, FrameDetections], output_path: Path
    ) -> dict:
        """Converts FrameDetections to COCO JSON and writes to `output_path`.

        Notes:
            - Our internal detections are normalized [0,1]. COCO expects pixels.
        """
        images: list[dict] = []
        annotations: list[dict] = []
        categories = [
            {"id": i, "name": cls["name"]} for i, cls in enumerate(self.class_map)
        ]
        categories_to_id = {cat["name"]: cat["id"] for cat in categories}

        annotation_id = 1
        for image_id, (img_path, frame_det) in enumerate(path_to_detections.items()):
            if not frame_det:
                continue

            images.append(
                {
                    "id": image_id,
                    "file_name": img_path.name,
                    "width": int(frame_det.width),
                    "height": int(frame_det.height),
                }
            )

            if frame_det.detections:
                for d in frame_det.detections:
                    category_id = categories_to_id[d.label]

                    # Convert normalized box to pixel COCO bbox [x_min, y_min, width, height]
                    x_min = float(d.x) * float(frame_det.width)
                    y_min = float(d.y) * float(frame_det.height)
                    w_px = float(d.w) * float(frame_det.width)
                    h_px = float(d.h) * float(frame_det.height)
                    bbox = [x_min, y_min, w_px, h_px]
                    area = w_px * h_px

                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(category_id),
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1

        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        with open(output_path, "w") as f:
            json.dump(coco_data, f, indent=2)
        return coco_data
