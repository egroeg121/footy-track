import os
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
from .utils import (
    _available_device,
    mask_poly_to_norm_xywh,
    ultralytics_result_to_detections,
)

# Path to the current best-performing object-detection checkpoint, relative to
# the project root. Update this when a better model is trained. Mirrors the
# classifier's get_current_best_guess_classifier() convention.
# Must always name a checkpoint trained at DEFAULT_IMGSZ. The previous value was a
# 2026-01 yolo11s on dataset v3, served at 640px where ball recall is 0.000 - so every
# experiment using it measured a detector effectively blind to the ball (bead ft-90o).
# The metric is in the comment so a defeated checkpoint cannot quietly stay the default.
CURRENT_BEST_DETECTOR_CHECKPOINT = (
    "model_saves/detector/rtdetr_1920_v11/best.pt"  # v11 @1280: ball mAP50 .221 / recall .209
)

# Resolution is the single biggest measured lever for ball detection (640 -> 1280 took
# ball recall from 0.000). Keep it explicit; never inherit the library default.
DEFAULT_IMGSZ = 1280


def debug_mode() -> bool:
    """True when FOOTY_DEBUG is set to a truthy value.

    Debug mode exists so the labeller can be developed on a machine with no GPU
    and little RAM: it must never load a checkpoint, download weights, or touch
    CUDA. See ``python -m footy_track.labeller --debug``.
    """
    return os.environ.get("FOOTY_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def get_current_best_detector(
    min_confidence: float = 0.3,
    iou_threshold: float = 0.90,
    verbose: bool = False,
) -> "ObjectDetector":
    """Return the current best-guess object detector.

    Resolves CURRENT_BEST_DETECTOR_CHECKPOINT against the project root so the
    same constant works regardless of the caller's working directory.

    Under ``FOOTY_DEBUG`` returns a :class:`StubObjectDetector` instead, so no
    weights are loaded on a CPU-only development box.
    """
    if debug_mode():
        return StubObjectDetector(min_confidence=min_confidence)
    model_uri = str(get_project_root() / CURRENT_BEST_DETECTOR_CHECKPOINT)
    return UltralyticsObjectDetector(
        model_uri=model_uri,
        min_confidence=min_confidence,
        iou_threshold=iou_threshold,
        verbose=verbose,
        use_model_names=True,
    )


class StubObjectDetector(ObjectDetector):
    """Zero-cost detector used by ``FOOTY_DEBUG``.

    Loads nothing, downloads nothing and never touches CUDA, so the labeller
    can be run and developed on a laptop with no GPU and little free RAM. It
    reads image dimensions only, and returns no detections.

    It deliberately returns an EMPTY detection list rather than fake boxes:
    invented boxes could be saved into a sidecar and become indistinguishable
    from real machine labels, which would corrupt the provenance tiers. An
    empty result is honest and harmless.
    """

    model_tag: str = "stub"

    def __init__(self, min_confidence: float = 0.3) -> None:
        self.min_confidence = min_confidence

    def predict_from_path(self, image_path: Path) -> FrameDetections:
        width = height = 0
        try:
            from PIL import Image  # noqa: PLC0415

            with Image.open(image_path) as im:
                width, height = im.size
        except Exception:  # noqa: BLE001 - dimensions are best-effort in debug
            pass
        return FrameDetections(
            uri=Path(image_path), width=width, height=height, detections=[]
        )


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
        use_model_names: bool = False,
        imgsz: int = DEFAULT_IMGSZ,
    ):
        # Use shared device selection util (prefers MPS on Apple, then CUDA, then CPU)
        dev = _available_device()
        self.device = dev.type if isinstance(dev, torch.device) else str(dev)
        self.model = YOLO(model_uri)
        self.model_uri = model_uri
        # When True, `classes` reflects the checkpoint's own label set rather than
        # the COCO person/ball subset used for stock weights.
        self.use_model_names = use_model_names
        self.imgsz = imgsz
        self.predict_kwargs = {
            "verbose": verbose,
            "compile": compile,
            "conf": min_confidence,
            "iou": iou_threshold,
            "imgsz": imgsz,
        }

    @property
    def classes(self) -> dict[int, str]:
        """Class-index → name mapping the model can detect.

        Defaults to the COCO person/ball subset that stock YOLO weights are used
        for here. Custom-trained checkpoints (e.g. the footy detector) should pass
        ``use_model_names=True`` so their own ``model.names`` labels flow through.
        """
        if self.use_model_names:
            names = getattr(self.model, "names", None)
            if names:
                return dict(names)
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

    def __init__(
        self,
        model_uri: str | None = None,
        min_confidence: float = 0.25,
        bbox_padding_percent: float = 0.075,
        verbose: bool = False,
    ) -> None:
        # Use shared device selection util for consistency across detectors
        dev = _available_device()
        model_uri = model_uri or get_project_root() / "model_saves" / "sam3" / "sam3.pt"

        self.device = dev.type if isinstance(dev, torch.device) else str(dev)
        # Prompt-specific thresholds (confidence) per concept
        # Defaults per request: ball -> 0.10, sports player -> 0.50
        # Store as (prompt, label, min_conf_threshold)
        self._prompt_specs: list[tuple[str, str, float]] = [
            (
                "soccer ball outside the pitch boundaries",
                OUT_OF_PLAY_BALL_TAG,
                0.3,
            ),
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
            # ("person", PERSON_TAG, 0.30),
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
            "iou": 0.7,
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
    def model_tag(self) -> str:
        return "sam3"

    @property
    def output_classes(self) -> list[str]:
        """Get the list of class names the model can detect."""
        return [lbl for (_p, lbl, _thr) in self._prompt_specs]

    @property
    def prompts(self) -> list[str]:
        """Get the list of text prompts used."""
        return [p for (p, _lbl, _thr) in self._prompt_specs]

    @torch.no_grad()
    def predict_from_path(self, image_path: Path) -> FrameDetections:  # noqa: PLR0912, PLR0915
        """Run SAM3 with text prompts and return combined FrameDetections."""
        img_path = Path(image_path)
        results, width, height = self._run_predictor(img_path)

        if not results:
            return FrameDetections(
                uri=img_path, width=width, height=height, detections=[]
            )

        detections: list[ObjectDetection] = []
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

            self._process_masks_and_add_detections(
                mask_polys,
                boxes,
                scores,
                cls_indices,
                detections,
                width,
                height,
            )

        if detections:
            detections = self._filter_detections_by_distance(detections)

        return FrameDetections(
            uri=img_path, width=width, height=height, detections=detections
        )

    def _run_predictor(
        self, img_path: Path
    ) -> tuple[UltralyticsResults | None, int, int]:
        """Run the SAM3 predictor and return the results, width, and height."""
        self.predictor.set_image(str(img_path))

        width, height = 0, 0

        # Run all prompts at once for efficiency
        prompt_list = [p for (p, _lbl, _thr) in self._prompt_specs]
        results = self.predictor(text=prompt_list)

        if not results:
            return None, width, height

        # Use the original image size for normalization to avoid any
        # letterbox/resize offsets that SAM3 may introduce internally.
        if width == 0 and height == 0:
            h, w = results[0].orig_shape[:2]
            width, height = int(w), int(h)

        return results, width, height

    def _filter_detections_by_distance(
        self, detections: list[ObjectDetection]
    ) -> list[ObjectDetection]:
        """
        Filter out detections that are too close to each other, keeping the one with the highest confidence.
        This is a custom Non-Maximum Suppression (NMS) based on center point distance.
        """
        detections.sort(key=lambda d: d.confidence, reverse=True)

        filtered_detections: list[ObjectDetection] = []
        for det in detections:
            is_too_close = False
            for existing_det in filtered_detections:
                # Calculate squared Euclidean distance between the centers of the two boxes
                center_x1 = det.x + det.w / 2
                center_y1 = det.y + det.h / 2
                center_x2 = existing_det.x + existing_det.w / 2
                center_y2 = existing_det.y + existing_det.h / 2

                dist_sq = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2

                # If the distance is less than 1% of the image dimension, consider them the same detection.
                # We use squared distance to avoid a sqrt operation. 0.01^2 = 0.0001
                if dist_sq < 0.0001:
                    is_too_close = True
                    break

            if not is_too_close:
                filtered_detections.append(det)

        return filtered_detections

    def _process_masks_and_add_detections(  # noqa: PLR0913  # noqa: PLR0913  # noqa: PLR0913
        self,
        mask_polys,
        boxes,
        scores,
        cls_indices,
        detections: list[ObjectDetection],
        width: int,
        height: int,
    ) -> None:
        for j, poly in enumerate(mask_polys):
            # Pad horizontally by the full percent and vertically by half (players
            # are taller than wide, so a tight vertical fit looks better).
            box = mask_poly_to_norm_xywh(
                poly,
                x_padding_percent=self.bbox_padding_percent,
                y_padding_percent=self.bbox_padding_percent / 2,
            )
            if box is None:
                continue
            x, y, w_n, h_n = box

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
