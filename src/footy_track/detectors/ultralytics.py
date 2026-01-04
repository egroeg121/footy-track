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

from .base import ObjectDetector
from .utils import _available_device, ultralytics_result_to_detections


class UltralyticsObjectDetector(ObjectDetector):
    """YOLO-based object detector returning Pydantic outputs.

    Uses the ultralytics YOLO models and returns a FrameDetections instance
    with normalized [x, y, w, h] boxes in [0, 1].
    """

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


class UltralyticsSam3Detector(ObjectDetector):
    """Text-prompted segmentation with SAM 3 returning bounding-box detections.

    This detector uses Ultralytics SAM 3 with text prompts to segment concepts
    and converts the resulting masks' bounding boxes into normalized detections. Pad these by a small amount fo raccuracy

    Prompts used:
    - "soccer ball" -> label "ball"
    - "sports player" -> label "person"
    """

    model_tag: str = "sam3"

    def __init__(
        self,
        model_uri: str = f"model_saves/{model_tag}/sam3.pt",
        min_confidence: float = 0.25,
        bbox_padding_percent: float = 0.075,
        verbose: bool = False,
    ) -> None:
        # Use shared device selection util for consistency across detectors
        dev = _available_device()
        self.device = dev.type if isinstance(dev, torch.device) else str(dev)
        # Prompt-specific thresholds (confidence) per concept
        # Defaults per request: ball -> 0.10, sports player -> 0.50
        # Store as (prompt, label, min_conf_threshold)
        self._prompt_specs: list[tuple[str, str, float]] = [
            (
                "soccer ball off the pitch",
                OUT_OF_PLAY_BALL_TAG,
                0.3,
            ),
            ("soccer ball near a player", IN_PLAY_BALL_TAG, 0.5),
            ("soccer ball on the pitch", IN_PLAY_BALL_TAG, 0.5),
            ("sports player on team", PLAYER_TAG, 0.50),
            ("substitute player on the sideline", PLAYER_SUB_TAG, 0.50),
            ("referee", REFEREE_TAG, 0.40),
            ("coach on sideline", COACH_TAG, 0.40),
        ]

        # Use the lowest threshold globally in predictor to avoid premature filtering,
        # then apply prompt-specific thresholds post-predict
        global_conf = min(min_confidence, *(thr for _, _, thr in self._prompt_specs))

        overrides = {
            "conf": global_conf,
            "task": "segment",
            "mode": "predict",
            "model": model_uri,
            "verbose": verbose,
            "device": self.device,
            "agnostic_nms": True,
            "iou": 0.9,
            # disable all Ultralytics auto-saving; we handle visualization ourselves
            "save": False,
            "save_txt": False,
            "save_json": False,
            "save_conf": False,
            "save_crop": False,
            "show": False,
        }
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

        # Back-compat mapping if needed elsewhere
        self.prompt_label_map: list[tuple[str, str]] = [
            (p, lbl) for (p, lbl, _thr) in self._prompt_specs
        ]
        self.bbox_padding_percent = bbox_padding_percent

    @property
    def output_classes(self) -> list[str]:
        """Get the list of class names the model can detect."""
        return [lbl for (_p, lbl, _thr) in self._prompt_specs]

    @property
    def prompts(self) -> list[str]:
        """Get the list of text prompts used."""
        return [p for (p, _lbl, _thr) in self._prompt_specs]

    @torch.no_grad()
    def predict_from_path(self, image_path: Path) -> FrameDetections:
        """Run SAM3 with text prompts and return combined FrameDetections."""
        img_path = Path(image_path)
        self.predictor.set_image(str(img_path))

        detections: list[ObjectDetection] = []
        width, height = 0, 0

        # Run all prompts at once for efficiency
        prompt_list = [p for (p, _lbl, _thr) in self._prompt_specs]
        results = self.predictor(text=prompt_list)

        if not results:
            return FrameDetections(
                uri=img_path, width=width, height=height, detections=detections
            )

        # Use the original image size for normalization to avoid any
        # letterbox/resize offsets that SAM3 may introduce internally.
        if width == 0 and height == 0:
            h, w = results[0].orig_shape[:2]
            width, height = int(w), int(h)

        # SAM3SemanticPredictor returns a single Results for multiple prompts.
        result = results[0]
        if getattr(result, "boxes", None) is not None:
            boxes = result.boxes
            scores = (
                boxes.conf.tolist()
                if hasattr(boxes, "conf") and boxes.conf is not None
                else []
            )
            # Class indices correspond to the prompt order we passed in
            cls_indices = (
                boxes.cls.int().tolist()
                if hasattr(boxes, "cls") and boxes.cls is not None
                else [0] * len(scores)
            )

            # Prefer deriving tight boxes from the segmentation masks when
            # they are available. Ultralytics' Boxes are stride-quantized and
            # may be a few pixels short on the bottom/right due to internal
            # rounding and clipping. Using the mask polygons ensures we
            # capture the full extent of the segmentation.
            mask_polys = None
            if getattr(result, "masks", None) is not None and hasattr(
                result.masks, "xyn"
            ):
                mask_polys = result.masks.xyn  # list[np.ndarray] normalized [0,1]

            if not (mask_polys is not None and len(mask_polys) == len(boxes)):
                raise ValueError(
                    "SAM3 results missing expected masks for detected boxes."
                )

            for j, poly in enumerate(mask_polys):
                if poly is None or len(poly) == 0:
                    continue
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                x1n, y1n = min(xs), min(ys)
                x2n, y2n = max(xs), max(ys)

                # Add a small padding to the bounding box
                w_n_unpadded = x2n - x1n
                h_n_unpadded = y2n - y1n
                x_padding = w_n_unpadded * self.bbox_padding_percent
                y_padding = h_n_unpadded * self.bbox_padding_percent

                x1n -= x_padding
                y1n -= y_padding
                x2n += x_padding
                y2n += y_padding

                # Convert to top-left + width/height with clamping
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

        return FrameDetections(
            uri=img_path, width=width, height=height, detections=detections
        )
