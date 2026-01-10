from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results as UltralyticsResults
from ultralytics.models.sam import SAM3SemanticPredictor

from footy_track.constants import (
    BALL_TAG,
    COACH_TAG,
    IN_PLAY_BALL_TAG,
    OUT_OF_PLAY_BALL_TAG,
    PERSON_TAG,
    PLAYER_SUB_TAG,
    PLAYER_TAG,
    REFEREE_TAG,
)
from footy_track.schema import FrameDetections, ObjectDetection
from footy_track.utils import get_project_root

from .base import ObjectDetector
from .utils import _available_device, ultralytics_result_to_detections


class UltralyticsObjectDetector(ObjectDetector):
    """YOLO-based object detector returning Pydantic outputs.

    Uses the ultralytics YOLO models and returns a FrameDetections instance
    with normalized [x, y, w, h] boxes in [0, 1].
    """

    model_tag: str = "yolo"

    def __init__(
        self,
        model_uri: str = "yolo11n.pt",
        verbose: bool = False,
        compile: bool = False,
        min_confidence: float = 0.3,
        iou_threshold: float = 0.90,
    ):
        # Use shared device selection util (prefers MPS on Apple, then CUDA, then CPU)
        dev = _available_device()
        self.device = dev.type if isinstance(dev, torch.device) else str(dev)
        self.model = YOLO(model_uri)
        self.model_uri = model_uri
        self.predict_kwargs = {
            "verbose": verbose,
            "compile": compile,
            "conf": min_confidence,
            "iou": iou_threshold,
        }

    @property
    def classes(self) -> list[str]:
        """Get the list of class names the model can detect."""
        # return self.model.names
        return {
            0: PERSON_TAG,
            32: BALL_TAG,
        }

    @torch.no_grad()
    def predict_from_path(
        self,
        image_path: Path,
    ) -> FrameDetections:
        """Run detection and return FrameDetections."""
        result: UltralyticsResults = self.model.predict(image_path, device=self.device)[
            0
        ]

        # Image size
        h, w = result.orig_shape[:2]

        # Build detections via modular converter
        detections = ultralytics_result_to_detections(result, self.classes)

        return FrameDetections(
            uri=Path(image_path),
            width=int(w),
            height=int(h),
            detections=detections,
        )


class UltralyticsSam3DetectorBase:
    """Base class for text-prompted segmentation with SAM 3."""

    def __init__(
        self,
        model_uri: str | None = None,
        min_confidence: float = 0.25,
        bbox_padding_percent: float = 0.075,
        verbose: bool = False,
    ) -> None:
        dev = _available_device()
        model_uri = model_uri or get_project_root() / "model_saves" / "sam3" / "sam3.pt"

        self.device = dev.type if isinstance(dev, torch.device) else str(dev)
        self._prompt_specs: list[tuple[str, str, float]] = [
            ("soccer ball outside the pitch boundaries", OUT_OF_PLAY_BALL_TAG, 0.3),
            (
                "soccer ball stopped off the side of the pitch",
                OUT_OF_PLAY_BALL_TAG,
                0.3,
            ),
            ("soccer ball on the pitch", IN_PLAY_BALL_TAG, 0.45),
            ("soccer goalkeeper", PLAYER_TAG, 0.5),
            ("sports player on a team", PLAYER_TAG, 0.45),
            ("substitute player waiting off the pitch", PLAYER_SUB_TAG, 0.60),
            ("referee", REFEREE_TAG, 0.50),
            ("coach on sideline", COACH_TAG, 0.50),
        ]

        global_conf = min(min_confidence, *(thr for _, _, thr in self._prompt_specs))
        overrides = {
            "conf": global_conf,
            "task": "segment",
            "mode": "predict",
            "model": model_uri,
            "verbose": verbose,
            "device": self.device,
            "agnostic_nms": True,
            "iou": 0.7,
            "save": False,
            "save_txt": False,
            "save_json": False,
            "save_conf": False,
            "save_crop": False,
            "show": False,
        }
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.prompt_label_map: list[tuple[str, str]] = [
            (p, lbl) for (p, lbl, _thr) in self._prompt_specs
        ]
        self.bbox_padding_percent = bbox_padding_percent

    @property
    def model_tag(self) -> str:
        return "sam3"

    @property
    def output_classes(self) -> list[str]:
        return [lbl for (_p, lbl, _thr) in self._prompt_specs]

    @property
    def prompts(self) -> list[str]:
        return [p for (p, _lbl, _thr) in self._prompt_specs]

    def _filter_detections_by_distance(
        self, detections: list[ObjectDetection]
    ) -> list[ObjectDetection]:
        detections.sort(key=lambda d: d.confidence, reverse=True)
        filtered_detections: list[ObjectDetection] = []
        for det in detections:
            is_too_close = False
            for existing_det in filtered_detections:
                center_x1 = det.x + det.w / 2
                center_y1 = det.y + det.h / 2
                center_x2 = existing_det.x + existing_det.w / 2
                center_y2 = existing_det.y + existing_det.h / 2
                dist_sq = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2
                if dist_sq < 0.0001:
                    is_too_close = True
                    break
            if not is_too_close:
                filtered_detections.append(det)
        return filtered_detections

    def _process_masks_and_add_detections(
        self,
        mask_polys,
        scores,
        cls_indices,
        detections: list[ObjectDetection],
    ) -> None:
        for j, poly in enumerate(mask_polys):
            if poly is None or len(poly) == 0:
                continue
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            x1n, y1n = min(xs), min(ys)
            x2n, y2n = max(xs), max(ys)
            w_n_unpadded = x2n - x1n
            h_n_unpadded = y2n - y1n
            x_padding = w_n_unpadded * self.bbox_padding_percent
            y_padding = h_n_unpadded * (self.bbox_padding_percent / 2)
            x1n -= x_padding
            y1n -= y_padding
            x2n += x_padding
            y2n += y_padding
            x = max(0.0, min(1.0, x1n))
            y = max(0.0, min(1.0, y1n))
            w_n = max(0.0, min(1.0, x2n - x1n))
            h_n = max(0.0, min(1.0, y2n - y1n))
            conf = float(scores[j]) if j < len(scores) else 1.0
            cls_id = int(cls_indices[j]) if j < len(cls_indices) else 0
            _, label, min_thr = self._prompt_specs[cls_id]
            if conf < float(min_thr):
                continue
            detections.append(
                ObjectDetection(
                    label=label,
                    confidence=conf,
                    x=x,
                    y=y,
                    w=w_n,
                    h=h_n,
                    model=self.model_tag,
                )
            )


class UltralyticsSam3Detector(ObjectDetector, UltralyticsSam3DetectorBase):
    """Text-prompted segmentation with SAM 3 for single images."""

    @torch.no_grad()
    def predict_from_path(self, image_path: Path) -> FrameDetections:
        img_path = Path(image_path)
        results, width, height = self._run_predictor(img_path)
        if not results:
            return FrameDetections(
                uri=img_path, width=width, height=height, detections=[]
            )
        detections = self._process_results(results[0])
        return FrameDetections(
            uri=img_path, width=width, height=height, detections=detections
        )

    def _run_predictor(
        self, img_path: Path
    ) -> tuple[UltralyticsResults | None, int, int]:
        self.predictor.set_image(str(img_path))
        width, height = 0, 0
        prompt_list = [p for (p, _lbl, _thr) in self._prompt_specs]
        results = self.predictor(text=prompt_list)
        if not results:
            return None, width, height
        if width == 0 and height == 0:
            h, w = results[0].orig_shape[:2]
            width, height = int(w), int(h)
        return results, width, height

    def _process_results(self, result: UltralyticsResults) -> list[ObjectDetection]:
        detections: list[ObjectDetection] = []
        if getattr(result, "boxes", None) is not None:
            boxes = result.boxes
            scores = (
                boxes.conf.tolist()
                if hasattr(boxes, "conf") and boxes.conf is not None
                else []
            )
            cls_indices = (
                boxes.cls.int().tolist()
                if hasattr(boxes, "cls") and boxes.cls is not None
                else [0] * len(scores)
            )
            mask_polys = None
            if getattr(result, "masks", None) is not None and hasattr(
                result.masks, "xyn"
            ):
                mask_polys = result.masks.xyn
            if not (mask_polys is not None and len(mask_polys) == len(boxes)):
                raise ValueError(
                    "SAM3 results missing expected masks for detected boxes."
                )
            self._process_masks_and_add_detections(
                mask_polys, scores, cls_indices, detections
            )
        if detections:
            detections = self._filter_detections_by_distance(detections)
        return detections


class UltralyticsSam3VideoDetector(UltralyticsSam3DetectorBase):
    """Text-prompted segmentation with SAM 3 for videos."""

    @torch.no_grad()
    def predict_from_video_path(self, video_path: Path) -> list[FrameDetections]:
        video_path_str = str(video_path)
        results_generator = self.predictor(source=video_path_str, stream=True)
        frame_detections_list = []
        for frame_idx, results in enumerate(results_generator):
            try:
                h, w = results.orig_shape[:2]
                detections = []
                if (
                    getattr(results, "boxes", None) is not None
                    and len(results.boxes) > 0
                ):
                    detections = self._process_results(results)
                frame_detections = FrameDetections(
                    uri=Path(f"{video_path_str}_frame_{frame_idx + 1}"),
                    width=int(w),
                    height=int(h),
                    detections=detections,
                )
                frame_detections_list.append(frame_detections)
            except RuntimeError as e:
                if "torch.cat(): expected a non-empty list of Tensors" in str(e):
                    # Skip frames with no detections to avoid the error
                    continue
                else:
                    raise
        return frame_detections_list
