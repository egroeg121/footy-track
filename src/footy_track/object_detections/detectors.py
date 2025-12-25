import json
import os
import re
from base64 import b64encode
from mimetypes import guess_type
from pathlib import Path
from typing import Any

import openai  # type: ignore
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

try:  # optional accelerated NMS
    from torchvision.ops import nms as tv_nms  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    tv_nms = None  # type: ignore[assignment]

# Optional heavy imports are placed here to avoid import-time costs when unused
from ultralytics import SAM, YOLO
from ultralytics.engine.results import Results as UltralyticsResults

from footy_track.object_detections.constants import BALL_TAG, PERSON_TAG

from .schema import Detection, FrameDetections, FrameDetectionsWithMeta
from .utils import _clamp01, ultralytics_result_to_detections


class ObjectDetector:
    def predict_from_path(self, image_path: Path) -> FrameDetections:
        """Predict objects in an image from its path."""
        raise NotImplementedError


# --- Simple IoU and NMS helpers ---
def _iou_xywh_norm(a: "Detection", b: "Detection") -> float:
    """IoU for normalized [x, y, w, h] boxes in [0,1]."""
    ax, ay, aw, ah = float(a.x), float(a.y), float(a.w), float(a.h)
    bx, by, bw, bh = float(b.x), float(b.y), float(b.w), float(b.h)

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0.0 else inter / union


def _nms_single_label(dets: list["Detection"], iou_threshold: float) -> list["Detection"]:
    """Greedy NMS for a single label using confidence descending order."""
    if not dets:
        return dets
    order = sorted(range(len(dets)), key=lambda i: float(dets[i].confidence), reverse=True)
    keep: list[int] = []
    suppressed = [False] * len(dets)

    for i_idx in order:
        if suppressed[i_idx]:
            continue
        keep.append(i_idx)
        for j_idx in order:
            if j_idx == i_idx or suppressed[j_idx]:
                continue
            if _iou_xywh_norm(dets[i_idx], dets[j_idx]) >= iou_threshold:
                suppressed[j_idx] = True

    return [dets[i] for i in keep]


def nms_by_label(detections: list["Detection"], iou_threshold: float = 0.5) -> list["Detection"]:
    """Run NMS independently per label and return filtered detections.

    - detections: list of Detection with normalized boxes [x,y,w,h]
    - iou_threshold: IoU threshold for suppression
    """
    if not detections:
        return detections

    by_label: dict[str, list[Detection]] = {}
    for d in detections:
        by_label.setdefault(d.label, []).append(d)

    filtered: list[Detection] = []
    for _, dets in by_label.items():
        if tv_nms is not None and dets:
            # Convert to xyxy normalized for torchvision
            boxes_xyxy = []
            scores = []
            for d in dets:
                x1 = _clamp01(float(d.x))
                y1 = _clamp01(float(d.y))
                x2 = _clamp01(float(d.x) + float(d.w))
                y2 = _clamp01(float(d.y) + float(d.h))
                boxes_xyxy.append([x1, y1, x2, y2])
                scores.append(float(d.confidence))

            boxes_t = torch.tensor(boxes_xyxy, dtype=torch.float32)
            scores_t = torch.tensor(scores, dtype=torch.float32)
            keep_idx = tv_nms(boxes_t, scores_t, float(iou_threshold)).tolist()  # type: ignore[operator]
            filtered.extend([dets[i] for i in keep_idx])
        else:
            filtered.extend(_nms_single_label(dets, iou_threshold))

    return filtered


def _available_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


class UltralyticsObjectDetector(ObjectDetector):
    """YOLO-based object detector returning Pydantic outputs.

    Uses the ultralytics YOLO models and returns a FrameDetections instance
    with normalized [x, y, w, h] boxes in [0, 1].
    """

    def __init__(
        self,
        model_uri: str = "yolo11x.pt",
        verbose: bool = False,
        compile: bool = False,
        min_confidence: float = 0.2,
        iou_threshold: float = 0.90,
    ):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
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


class UltralyticsSam3ObjectDetector(ObjectDetector):
    """YOLO-based object detector returning Pydantic outputs.

    Uses the ultralytics YOLO models and returns a FrameDetections instance
    with normalized [x, y, w, h] boxes in [0, 1].
    """

    def __init__(
        self,
        model_uri: str = "sam3.pt",
        verbose: bool = False,
        compile: bool = False,
        min_confidence: float = 0.3,
        iou_threshold: float = 0.90,
    ):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SAM(model_uri)
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
            1: BALL_TAG,
        }

    @torch.no_grad()
    def predict_from_path(self, image_path: Path, *args, **kwargs) -> FrameDetections:
        """Run detection and return FrameDetections."""
        pred_kwargs = {**self.predict_kwargs, **kwargs, "device": self.device}

        # Run prediction for players
        player_results: UltralyticsResults = self.model.predict(
            image_path, *args, **pred_kwargs, prompt=PERSON_TAG
        )[0]

        # Run prediction for the ball
        ball_results: UltralyticsResults = self.model.predict(
            image_path, *args, **pred_kwargs, prompt=BALL_TAG
        )[0]

        # Image size
        h, w = player_results.orig_shape[:2]

        # Build detections via modular converter
        player_detections = ultralytics_result_to_detections(player_results, {0: PERSON_TAG})
        ball_detections = ultralytics_result_to_detections(ball_results, {0: BALL_TAG})

        detections = player_detections + ball_detections

        return FrameDetections(
            uri=Path(image_path),
            width=int(w),
            height=int(h),
            detections=detections,
        )


# Multiple prompts map to a single class label
GROUND_DINO_PROMPT_TO_CLASS: dict[str, list[str]] = {
    BALL_TAG: ["soccer ball", "football", "ball"],
    PERSON_TAG: [
        "player",
        "referee",
        "coach",
    ],
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
        nms_iou_threshold: float | None = 0.95,
    ) -> None:
        if AutoProcessor is None or AutoModelForZeroShotObjectDetection is None:
            raise ImportError(
                "transformers is required for GroundingDinoObjectDetector. Install via `pip install transformers`"
            )

        # device inference prefers CUDA/MPS when available
        self.device = _available_device()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model_id = model_id
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("mps")

        # Prompts focused on ball/person detection; can be customized
        self.class_prompts_mapping = GROUND_DINO_PROMPT_TO_CLASS
        self._lower_synonyms = {
            key: {s.lower() for s in labels} for key, labels in self.class_prompts_mapping.items()
        }
        self.text_labels = [list(labels) for labels in self.class_prompts_mapping.values()]
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold) if nms_iou_threshold is not None else None

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
                        label=BALL_TAG if canonical_label == BALL_TAG else PERSON_TAG,
                        confidence=float(score),
                        x=x,
                        y=y,
                        w=w_n,
                        h=h_n,
                    )
                )

        deduplicated_detections = (
            nms_by_label(detections, self.nms_iou_threshold)
            if self.nms_iou_threshold is not None
            else detections
        )
        return FrameDetections(
            uri=Path(image_path),
            width=int(w),
            height=int(h),
            detections=deduplicated_detections,
            model=self.model_id,
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
            model=self.model,
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
