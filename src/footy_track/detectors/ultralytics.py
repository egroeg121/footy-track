from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results as UltralyticsResults
from ultralytics.models.sam import SAM3SemanticPredictor

from footy_track.schema import FrameDetections, ObjectDetection

from .base import ObjectDetector
from .constants import BALL_TAG, PERSON_TAG
from .utils import ultralytics_result_to_detections, _available_device


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
    and converts the resulting masks' bounding boxes into normalized detections.

    Prompts used:
    - "soccer ball" -> label "ball"
    - "sports player" -> label "person"
    """

    def __init__(
        self,
        model_uri: str = "model_saves/sam3/sam3.pt",
        min_confidence: float = 0.25,
        verbose: bool = False,
    ) -> None:
        # Use shared device selection util for consistency across detectors
        dev = _available_device()
        self.device = dev.type if isinstance(dev, torch.device) else str(dev)
        # Prompt-specific thresholds (confidence) per concept
        # Defaults per request: ball -> 0.10, sports player -> 0.50
        # Store as (prompt, label, min_conf_threshold)
        self.prompt_specs: list[tuple[str, str, float]] = [
            ("soccer ball", BALL_TAG, 0.5),
            ("sports player", PERSON_TAG, 0.50),
        ]

        # Use the lowest threshold globally in predictor to avoid premature filtering,
        # then apply prompt-specific thresholds post-predict
        global_conf = min(min_confidence, *(thr for _, _, thr in self.prompt_specs))

        overrides = dict(
            conf=global_conf,
            task="segment",
            mode="predict",
            model=model_uri,
            verbose=verbose,
            device=self.device,
            # disable all Ultralytics auto-saving; we handle visualization ourselves
            save=False,
            save_txt=False,
            save_json=False,
            save_conf=False,
            save_crop=False,
            show=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

        # Back-compat mapping if needed elsewhere
        self.prompt_label_map: list[tuple[str, str]] = [
            (p, lbl) for (p, lbl, _thr) in self.prompt_specs
        ]

    @torch.no_grad()
    def predict_from_path(self, image_path: Path) -> FrameDetections:
        """Run SAM3 with text prompts and return combined FrameDetections."""
        img_path = Path(image_path)
        self.predictor.set_image(str(img_path))

        detections: list[ObjectDetection] = []
        width, height = 0, 0

        for prompt, label, min_thr in self.prompt_specs:
            results = self.predictor(text=[prompt])

            if not results:
                continue

            if width == 0 and height == 0:
                h, w = results[0].orig_shape[:2]
                width, height = int(w), int(h)

            for result in results:
                if getattr(result, "boxes", None) is None:
                    continue

                xyxyn = (
                    result.boxes.xyxyn.tolist()
                    if hasattr(result.boxes, "xyxyn")
                    else []
                )
                scores = (
                    result.boxes.conf.tolist()
                    if hasattr(result.boxes, "conf") and result.boxes.conf is not None
                    else []
                )

                for j, b in enumerate(xyxyn):
                    x1, y1, x2, y2 = (
                        float(b[0]),
                        float(b[1]),
                        float(b[2]),
                        float(b[3]),
                    )
                    x = max(0.0, min(1.0, x1))
                    y = max(0.0, min(1.0, y1))
                    w_n = max(0.0, min(1.0, x2 - x1))
                    h_n = max(0.0, min(1.0, y2 - y1))

                    conf = float(scores[j]) if j < len(scores) else 1.0
                    # Apply prompt-specific confidence threshold
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
                            model="sam3",
                        )
                    )

        return FrameDetections(
            uri=img_path, width=width, height=height, detections=detections
        )
