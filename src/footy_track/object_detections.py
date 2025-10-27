"""Detectors and schema for object detection outputs.

This module defines small Pydantic models for normalized detections (`Detection`) and per-image results
(`FrameDetections`), plus two detectors that return this schema:
- `UltralyticsObjectDetector` wraps YOLO from ultralytics
- `GroundingDinoObjectDetector` uses Hugging Face Grounding DINO (ball-only -> label "football")
"""

# embed_folder.py
from abc import ABC
import pathlib
from typing import Iterable, List

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results as UltralyticsResults
import fiftyone as fo

# New imports for schema + HF model
from pydantic import BaseModel, Field
from PIL import Image

try:
    # Defer HF imports to runtime to avoid hard dependency at import time
    from transformers import (
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
        infer_device,
    )
except Exception:  # pragma: no cover - optional dependency
    AutoProcessor = None  # type: ignore
    AutoModelForZeroShotObjectDetection = None  # type: ignore
    infer_device = None  # type: ignore


# ------------------------------
# Pydantic data schema
# ------------------------------
class Detection(BaseModel):
    label: str = Field(..., description="Class name")
    confidence: float = Field(..., ge=0.0, le=1.0)
    x: float = Field(..., ge=0.0, le=1.0, description="Top-left x (normalized)")
    y: float = Field(..., ge=0.0, le=1.0, description="Top-left y (normalized)")
    w: float = Field(..., ge=0.0, le=1.0, description="Width (normalized)")
    h: float = Field(..., ge=0.0, le=1.0, description="Height (normalized)")


class FrameDetections(BaseModel):
    uri: pathlib.Path = Field(..., description="Path to the image file or identifier")
    width: int
    height: int
    detections: List[Detection]


# ------------------------------
# Utilities
# ------------------------------

def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


# Existing utility retained for potential downstream users
# (kept untouched for backward-compat elsewhere in the repo)
def ultralytics_detection_to_fiftyone_detection(
    detection_result: UltralyticsResults, classes: list[str]
) -> list[fo.Detection]:
    """Convert an Ultralytics YOLO detection result to FiftyOne detections.

    Notes
    -----
    - FiftyOne expects bounding_box as [x, y, w, h] with x,y being the
      top-left corner, all normalized to [0, 1].
    - Ultralytics provides several box formats. We use `xyxyn` which is
      already normalized (x1, y1, x2, y2), to avoid center->corner mistakes.
    """
    # Tensors on device -> Python lists
    labels = detection_result.boxes.cls.int().tolist()
    scores = detection_result.boxes.conf.tolist()
    # Normalized top-left/bottom-right
    xyxyn = detection_result.boxes.xyxyn.tolist()

    detections: list[fo.Detection] = []
    for label_idx, score, (x1, y1, x2, y2) in zip(labels, scores, xyxyn):
        # Convert (x1, y1, x2, y2) -> (x, y, w, h), clamp to [0, 1]
        x = _clamp01(x1)
        y = _clamp01(y1)
        w = _clamp01(max(0.0, x2 - x1))
        h = _clamp01(max(0.0, y2 - y1))

        # Class name can come from list or dict
        try:
            label_name = classes[int(label_idx)]
        except Exception:
            # Fallback if classes is a dict-like
            label_name = classes.get(int(label_idx), str(int(label_idx)))  # type: ignore[attr-defined]

        detections.append(
            fo.Detection(
                label=str(label_name),
                bounding_box=[x, y, w, h],
                confidence=float(score),
            )
        )

    return detections


# ------------------------------
# Detector Interfaces
# ------------------------------
class ObjectDetector(ABC):
    pass


class UltralyticsObjectDetector(ObjectDetector):
    """YOLO-based object detector returning Pydantic outputs.

    Uses the ultralytics YOLO models and returns a FrameDetections instance
    with normalized [x, y, w, h] boxes in [0, 1].
    """

    def __init__(self, model_uri: str = "yolo11n.pt", verbose=False, compile=False):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = YOLO(model_uri)
        self.predict_kwargs = {
            "verbose": verbose,
            "compile": compile,
        }

    @property
    def classes(self) -> list[str]:
        """Get the list of class names the model can detect."""
        return self.model.names

    def predict_from_path(
        self, image_path: pathlib.Path, *args, **kwargs
    ) -> FrameDetections:
        """Run detection and return FrameDetections.

        Parameters
        ----------
        image_path: pathlib.Path
            Path to an image file

        Returns
        -------
        FrameDetections
            Pydantic model containing all detections
        """
        result: UltralyticsResults = self.model.predict(
            image_path, device=self.device, *args, **kwargs, **self.predict_kwargs
        )[0]

        # Image size
        h, w = result.orig_shape[:2]

        # Build detections list from normalized xyxyn
        labels = result.boxes.cls.int().tolist() if result.boxes is not None else []
        scores = result.boxes.conf.tolist() if result.boxes is not None else []
        xyxyn = result.boxes.xyxyn.tolist() if result.boxes is not None else []

        detections: list[Detection] = []
        for label_idx, score, (x1, y1, x2, y2) in zip(labels, scores, xyxyn):
            x = _clamp01(float(x1))
            y = _clamp01(float(y1))
            w_n = _clamp01(max(0.0, float(x2) - float(x1)))
            h_n = _clamp01(max(0.0, float(y2) - float(y1)))

            # Map class index to name
            try:
                label_name = self.classes[int(label_idx)]
            except Exception:
                label_name = str(int(label_idx))

            detections.append(
                Detection(
                    label=label_name,
                    confidence=float(score),
                    x=x,
                    y=y,
                    w=w_n,
                    h=h_n,
                )
            )

        return FrameDetections(
            uri=pathlib.Path(image_path),
            width=int(w),
            height=int(h),
            detections=detections,
        )


class GroundingDinoObjectDetector(ObjectDetector):
    """Grounding DINO zero-shot detector for soccer balls only.

    - Uses Hugging Face transformers implementation
    - Detects only ball-like prompts and outputs label fixed to "football"
    - Returns FrameDetections with normalized [x, y, w, h]
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.40,
        text_threshold: float = 0.30,
        prompts: Iterable[str] | None = None,
    ) -> None:
        if AutoProcessor is None or AutoModelForZeroShotObjectDetection is None:
            raise ImportError(
                "transformers is required for GroundingDinoObjectDetector. Install via `pip install transformers`"
            )

        # device inference prefers CUDA/MPS when available
        self.device = infer_device() if infer_device is not None else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

        # Prompts focused on ball detection; can be customized
        default_prompts = ["soccer ball", "football", "ball"]
        self.text_labels = [list(prompts) if prompts else default_prompts]
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)

    @torch.no_grad()
    def predict_from_path(self, image_path: pathlib.Path) -> FrameDetections:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        inputs = self.processor(images=img, text=self.text_labels, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)

        # Post-process to get absolute boxes; then normalize
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],  # (height, width)
        )
        result = results[0]

        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])

        detections: list[Detection] = []
        for box, score, label in zip(boxes, scores, labels):
            # Only output a single canonical label "football"
            # The model returns label as the matched prompt string
            # We accept any of our prompt variants and map to "football"
            if str(label).strip().lower() not in {"soccer ball", "football", "ball"}:
                continue

            x1, y1, x2, y2 = [float(x) for x in box.tolist()]
            # Normalize
            x = _clamp01(x1 / w)
            y = _clamp01(y1 / h)
            w_n = _clamp01(max(0.0, (x2 - x1) / w))
            h_n = _clamp01(max(0.0, (y2 - y1) / h))

            detections.append(
                Detection(
                    label="football",
                    confidence=float(score),
                    x=x,
                    y=y,
                    w=w_n,
                    h=h_n,
                )
            )

        return FrameDetections(
            uri=pathlib.Path(image_path),
            width=int(w),
            height=int(h),
            detections=detections,
        )