from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import fiftyone as fo
from PIL import Image, ImageDraw, ImageFont

from .schema import Detection, FrameDetections


# ------------------------------
# Utilities
# ------------------------------


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def visualise_detections_on_image(
    frame_detections: FrameDetections, save_path: Path | None = None, show: bool = True
) -> Optional[Path]:
    """Draw detections on the source image and optionally save/show it.

    - Uses normalized bbox coordinates in `frame_detections.detections`
    - If `save_path` is provided, the annotated image is saved there and the path is returned
    - If `show` is True, the image is opened with the default image viewer
    """
    img_path = Path(frame_detections.uri)
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    color_map = {
        "person": (0, 255, 0),
        "player": (0, 200, 255),
        "referee": (255, 165, 0),
        "coach": (138, 43, 226),
        "ball": (255, 255, 0),
        "ball_in_play": (255, 255, 0),
        "ball_out_of_play": (255, 0, 0),
    }

    thickness = max(2, min(6, int(round(min(w, h) * 0.003))))

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, int(min(w, h) * 0.02)))
    except Exception:
        font = ImageFont.load_default()

    for det in frame_detections.detections:
        x1 = float(det.x) * w
        y1 = float(det.y) * h
        x2 = x1 + float(det.w) * w
        y2 = y1 + float(det.h) * h

        color = color_map.get(det.label.lower(), (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        label_text = det.label
        try:
            label_text = f"{det.label} {det.confidence:.2f}"
        except Exception:
            pass

        try:
            text_w, text_h = draw.textbbox((0, 0), label_text, font=font)[2:]
        except Exception:
            text_w, text_h = draw.textsize(label_text, font=font)
        pad = 2
        bg_coords = [x1, max(0, y1 - text_h - 2 * pad), x1 + text_w + 2 * pad, y1]
        draw.rectangle(bg_coords, fill=(0, 0, 0))
        draw.text(
            (x1 + pad, max(0, y1 - text_h - pad)),
            label_text,
            fill=(255, 255, 255),
            font=font,
        )

    out_path: Optional[Path] = None
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

    if show:
        img.show()

    return out_path


# ------------------------------
# Converters to/from vendor outputs
# ------------------------------


def ultralytics_result_to_detections(
    result: Any, classes: List[str] | Dict[int, str]
) -> List[Detection]:
    """Convert an Ultralytics YOLO `Results` object to a list of `Detection`.

    Parameters
    ----------
    result: UltralyticsResults-like
        Single-image prediction result from ultralytics
    classes: list[str] | dict[int, str]
        Mapping of class indices to names
    """
    if getattr(result, "boxes", None) is None:
        return []

    labels = result.boxes.cls.int().tolist()
    scores = result.boxes.conf.tolist()
    xyxyn = result.boxes.xyxyn.tolist()  # normalized x1,y1,x2,y2

    out: List[Detection] = []
    for label_idx, score, (x1, y1, x2, y2) in zip(labels, scores, xyxyn):
        x = _clamp01(float(x1))
        y = _clamp01(float(y1))
        w_n = _clamp01(max(0.0, float(x2) - float(x1)))
        h_n = _clamp01(max(0.0, float(y2) - float(y1)))

        try:
            label_name = classes[int(label_idx)]  # type: ignore[index]
        except Exception:
            label_name = str(int(label_idx))

        out.append(
            Detection(
                label=str(label_name),
                confidence=float(score),
                x=x,
                y=y,
                w=w_n,
                h=h_n,
            )
        )
    return out


# ------------------------------
# Converters to FiftyOne
# ------------------------------


def detection_to_fiftyone(d: Detection) -> fo.Detection:
    """Convert a single Detection to a FiftyOne Detection.

    Returns
    -------
    fiftyone.core.labels.Detection
        With bounding_box as [x, y, w, h] and confidence
    """
    return fo.Detection(
        label=d.label,
        bounding_box=[float(d.x), float(d.y), float(d.w), float(d.h)],
        confidence=float(d.confidence),
    )


def frame_to_fiftyone_detections(frame: FrameDetections) -> List[fo.Detection]:
    """Convert FrameDetections to a list of FiftyOne Detection objects."""
    return [detection_to_fiftyone(d) for d in frame.detections]


# ------------------------------
# LLM helpers
# ------------------------------


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON object extraction from LLM responses."""
    import json
    import re

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
