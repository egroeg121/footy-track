import enum
import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from roboflow import Roboflow
from tqdm import tqdm

from footy_track.classifier import (
    ClassificationResult,
    Classifier,
    EnumBroadcastClassification,
    RoboflowClassifier,
)
from footy_track.object_detections import detectors
from footy_track.object_detections.detectors import ObjectDetector
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

    def get_or_create_project(self, project_name: str, project_type: str):
        """Get or create a Roboflow project."""

        project = self.workspace.project(project_name)
        logger.info(f"Using existing project: {project.name}")
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
        self.project = self.get_or_create_project(project_name, self.project_type)
        self.classifier = classifier or RoboflowClassifier(
            workspace_name=workspace_name, project_name=project_name, api_key=self.api_key
        )

    def _classify_image(self, img_path: Path) -> ClassificationResult:
        """Placeholder for a real classification model."""
        return self.classifier.predict_from_path(img_path)

    def _get_all_project_image_paths(self):
        pass

    def upload_dir(
        self, image_dir: Path, sample_number: int = None, batch_name: str = "uploads"
    ) -> None:
        image_paths = self._load_local_images(image_dir, sample_number)
        return self.upload_images(image_paths, batch_name=batch_name)

    def upload_images(
        self,
        image_paths: list[Path],
        batch_name: str = "uploads",
        annotation_confidence_min: float = 0.8,
    ) -> None:
        """Upload images to the classification project with yes/no labels."""

        for img_path in tqdm(image_paths, desc="Uploading images to Roboflow"):
            classification = self._classify_image(img_path)

            annotation_path = None
            if classification.confidence >= annotation_confidence_min:
                # Create a temporary annotation file with the classification label
                annotation_path = str(img_path.with_suffix(".txt"))
                with open(annotation_path, "w") as f:
                    f.write(classification.label)

            try:
                self.project.upload(
                    image_path=str(img_path),
                    annotation_path=annotation_path,
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

    class ObjectDetectorEnum(enum.Enum):
        UltralyticsYolo11 = enum.auto()
        GroundingDino = enum.auto()
        UltralyticsSam3 = enum.auto()

    def __init__(
        self,
        workspace_name: str,
        project_name: str,
        detector: ObjectDetector | None = None,
        classifier: Classifier | None = None,
    ):
        super().__init__(workspace_name)
        self.project_name = project_name
        self.project_type = "object-detection"
        self.project = self.get_or_create_project(project_name, self.project_type)
        self.class_map = {c: i for i, c in enumerate(self.project.classes)}
        logger.info(f"Roboflow project classes: {self.class_map}")
        self.detector = detector or {
            self.ObjectDetectorEnum.UltralyticsYolo11: detectors.UltralyticsObjectDetector(
                model_uri="yolo11x.pt"
            ),
            self.ObjectDetectorEnum.UltralyticsSam3: detectors.UltralyticsSam3ObjectDetector(),
            self.ObjectDetectorEnum.GroundingDino: detectors.GroundingDinoObjectDetector(),
        }

        self.classifier = None
        if classifier is None:
            self.classifier = RoboflowClassifier(
                workspace_name=workspace_name,
                project_name="footy-track-broadcast-frame",
                api_key=self.api_key,
            )

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

    def upload_dir(
        self,
        image_dir: Path,
        sample_number: int = None,
        batch_name: str = "uploads",
        pre_annotate: bool = True,
        filter_by_broadcast_classifier: bool = True,
    ) -> None:
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
        pre_annotate: bool = True,
        filter_by_broadcast_classifier: bool = True,
        batch_name: str = "uploads",
    ) -> None:
        """Upload images to the dataset associated with this project."""
        result_counter = defaultdict(int)
        path_to_detections: dict[Path, FrameDetections] = dict.fromkeys(image_paths)

        if filter_by_broadcast_classifier and self.classifier is not None:
            filtered_image_paths = []
            for img_path in tqdm(image_paths, desc="Filtering images by broadcast classifier"):
                try:
                    classification = self.classifier.predict_from_path(img_path)
                    if classification.label == EnumBroadcastClassification.YES:
                        filtered_image_paths.append(img_path)
                    else:
                        continue
                except Exception:
                    result_counter["error"] += 1
            path_to_detections = dict.fromkeys(filtered_image_paths)
            logger.info(
                f"Filtered images: {len(filtered_image_paths)} / {len(image_paths)} kept after classification"
            )

        if pre_annotate:
            path_to_detections = self._run_object_detection_on_images(path_to_detections)

        # Filter out images with no detections if we pre-annotated
        if pre_annotate:
            path_to_detections_with_annos = {
                p: d for p, d in path_to_detections.items() if d and d.detections
            }
        else:
            path_to_detections_with_annos = path_to_detections

        if not path_to_detections_with_annos:
            logger.info("No images with detections to upload.")
            return

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            annotation_path = tmp_file.name
            self._write_coco_annotations(path_to_detections_with_annos, Path(annotation_path))

        try:
            for img_path in tqdm(
                path_to_detections_with_annos.keys(), desc="Uploading images to Roboflow"
            ):
                try:
                    self.project.upload(
                        image_path=str(img_path),
                        annotation_path=annotation_path,
                        batch_name=batch_name,
                        is_prediction=True,
                    )
                    result_counter["uploaded"] += 1
                except Exception as e:
                    logger.error(f"Failed to upload {img_path}: {e}")
                    result_counter["error"] += 1
        finally:
            if annotation_path:
                os.remove(annotation_path)
        logger.info(f"Stats: {result_counter}")

    def _write_coco_annotations(
        self, path_to_detections: dict[Path, FrameDetections], output_path: Path
    ):
        """Converts FrameDetections to COCO JSON format and writes to a file."""
        images = []
        annotations = []
        categories = [{"id": i, "name": name} for name, i in self.class_map.items()]

        annotation_id = 1
        for image_id, (img_path, frame_det) in enumerate(path_to_detections.items()):
            if not frame_det:
                continue

            images.append(
                {
                    "id": image_id,
                    "file_name": img_path.name,
                    "width": frame_det.width,
                    "height": frame_det.height,
                }
            )

            if frame_det.detections:
                for d in frame_det.detections:
                    if d.label not in self.class_map:
                        logger.warning(
                            f"Label '{d.label}' not in Roboflow project class map. Skipping."
                        )
                        continue

                    category_id = self.class_map[d.label]

                    # COCO bbox is [x_min, y_min, width, height]
                    bbox = [d.x, d.y, d.w, d.h]
                    area = d.w * d.h

                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
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
