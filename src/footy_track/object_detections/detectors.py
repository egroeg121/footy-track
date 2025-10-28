import json
import os
import re
from base64 import b64encode
from collections.abc import Iterable
from mimetypes import guess_type
from pathlib import Path
from typing import Any

import openai  # type: ignore
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    infer_device,
)

# Optional heavy imports are placed here to avoid import-time costs when unused
from ultralytics import YOLO
from ultralytics.engine.results import Results as UltralyticsResults

from .schema import Detection, FrameDetections, FrameDetectionsWithMeta
from .utils import _clamp01, ultralytics_result_to_detections


class ObjectDetector:
    pass


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
    ):
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

    @torch.no_grad()
    def predict_from_path(self, image_path: Path, *args, **kwargs) -> FrameDetections:
        """Run detection and return FrameDetections."""
        pred_kwargs = {**self.predict_kwargs, **kwargs, "device": self.device}
        result: UltralyticsResults = self.model.predict(image_path, *args, **pred_kwargs)[0]

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


# Multiple prompts map to a single class label
GROUND_DINO_PROMPT_TO_CLASS: dict[str, list[str]] = {
    "ball": ["soccer ball", "football", "ball"],
    "person": ["player", "referee", "coach"],
}


class GroundingDinoObjectDetector(ObjectDetector):
    """Grounding DINO zero-shot detector for ball and person.

    - Uses Hugging Face transformers implementation
    - Detects ball/person prompts and outputs canonical labels 'ball' and 'person'
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
        self.device = (
            infer_device()
            if infer_device is not None
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

        # Prompts focused on ball/person detection; can be customized
        self.class_prompts_mapping = GROUND_DINO_PROMPT_TO_CLASS
        self._lower_synonyms = {
            key: {s.lower() for s in labels} for key, labels in self.class_prompts_mapping.items()
        }
        self.text_labels = [list(labels) for labels in self.class_prompts_mapping.values()]
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)

    def _internal_predict(self, image: Image.Image) -> FrameDetections:  # placeholder
        raise NotImplementedError

    @torch.no_grad()
    def predict_from_path(self, image_path: Path) -> FrameDetections:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        detections: list[Detection] = []

        # Treat each prompt group independently to avoid cross-group interaction
        for canonical_label, synonyms in self.class_prompts_mapping.items():
            inputs = self.processor(images=img, text=list(synonyms), return_tensors="pt").to(
                self.model.device
            )
            outputs = self.model(**inputs)

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

            allowed = self._lower_synonyms.get(canonical_label, set())

            for box, score, label in zip(boxes, scores, labels, strict=False):
                lbl = str(label).strip().lower()
                if allowed and lbl not in allowed:
                    continue

                x1, y1, x2, y2 = [float(x) for x in box.tolist()]
                # Normalize
                x = _clamp01(x1 / w)
                y = _clamp01(y1 / h)
                w_n = _clamp01(max(0.0, (x2 - x1) / w))
                h_n = _clamp01(max(0.0, (y2 - y1) / h))

                detections.append(
                    Detection(
                        label="ball" if canonical_label == "ball" else "person",
                        confidence=float(score),
                        x=x,
                        y=y,
                        w=w_n,
                        h=h_n,
                    )
                )

        return FrameDetections(
            uri=Path(image_path),
            width=int(w),
            height=int(h),
            detections=detections,
        )


class ChatGPTObjectDetector(ObjectDetector):
    """LLM-based detector using an OpenAI vision model to return boxes and clock.

    Returns a FrameDetectionsWithMeta instance containing detections and optional clock.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert soccer (football) broadcast analysis assistant.\n"
        "Given a single broadcast frame, return relative bounding boxes for:\n"
        "- players (either team),\n- referee,\n- the ball (in play),\n- any ball out of play,\n"
        "and the on-screen game clock text if visible.\n"
        "Respond ONLY with JSON matching this schema:\n"
        '{\n  "clock": string|null,\n  "objects": [\n    { "label": one of ["player", "referee", "ball", "ball_out_of_play"],\n      "bbox": [x, y, w, h] with each in [0,1] }\n  ]\n}\n'
        "Ensure bboxes are normalized to [0,1] relative to the full image."
    )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        system_prompt: str | None = None,
    ):
        if openai is None:
            raise ImportError(
                "openai client is required for ChatGPTObjectDetector. Install via `uv add openai`."
            )
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set. Provide api_key or set env var.")
        # New-style client
        self.client = openai.OpenAI(api_key=key)  # type: ignore[attr-defined]
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    @torch.no_grad()
    def predict_from_path(self, image_path: Path) -> FrameDetectionsWithMeta:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        mime, _ = guess_type(str(image_path))
        mime = mime or "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = b64encode(f.read()).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"

        try:
            resp = self.client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this frame and respond with JSON only.",
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content  # type: ignore[index]
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"OpenAI request failed: {e}") from e

        result_json: dict = {"clock": None, "objects": []}
        if content:
            parsed = self._extract_json(content)
            if isinstance(parsed, dict):
                result_json = parsed

        detections: list[Detection] = []
        for obj in result_json.get("objects", []) or []:
            try:
                label = str(obj.get("label", "")).strip().lower()
                x, y, bw, bh = [float(v) for v in obj.get("bbox", [0, 0, 0, 0])]
                detections.append(
                    Detection(
                        label=label,
                        confidence=1.0,
                        x=_clamp01(x),
                        y=_clamp01(y),
                        w=_clamp01(bw),
                        h=_clamp01(bh),
                    )
                )
            except Exception:
                continue

        return FrameDetectionsWithMeta(
            uri=Path(image_path),
            width=int(w),
            height=int(h),
            detections=detections,
            clock=result_json.get("clock"),
        )

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Best-effort JSON object extraction from LLM responses."""

        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        cand = fence.group(1) if fence else None
        if not cand:
            brace = re.search(r"(\{.*\})", text, re.DOTALL)
            cand = brace.group(1) if brace else None
        if not cand:
            return None
        try:
            return json.loads(cand)
        except Exception:
            return None
