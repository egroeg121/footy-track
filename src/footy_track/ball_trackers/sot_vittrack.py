"""Method A (ft-ztw): Single-object tracker using VitTrack ONNX.

Implements the BallTracker protocol for the bake-off harness (ft-1my).

Architecture: ViT-based SOT from OpenCV Zoo. Template (128×128) set on first
valid bbox; search region (256×256) extracted around previous position each
frame using a motion-guided ROI crop.

Model weights are downloaded from HuggingFace Hub on first use and cached at
~/.cache/huggingface/hub/.

Usage::

    from footy_track.ball_trackers.sot_vittrack import VitTrackSOT

    tracker = VitTrackSOT()
    bbox = tracker.track(None, first_frame)           # cold-start, returns None
    bbox = tracker.track(prev_bbox, second_frame)     # warms template + tracks
    tracker.reset()                                   # between clips
"""

from __future__ import annotations

import math
import pathlib
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

# HuggingFace Hub repo / filename for the VitTrack ONNX model
_HF_REPO = "opencv/object_tracking_vittrack"
_HF_FILENAME = "object_tracking_vittrack_2023sep.onnx"

# Model constants
_TEMPLATE_SIZE = (128, 128)
_SEARCH_SIZE = (256, 256)
_TEMPLATE_FACTOR = 2.0  # crop = sqrt(w*h) * factor
_SEARCH_FACTOR = 4.0
_GRID = 16  # output heatmap grid size
_SCORE_THRESHOLD = 0.20

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _hann1d(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float32)
    return 0.5 * (1.0 - np.cos(2 * np.pi * (i + 1) / (n + 1)))


def _hann2d(h: int, w: int) -> np.ndarray:
    return _hann1d(h).reshape(-1, 1) * _hann1d(w).reshape(1, -1)


_HANN_WINDOW = _hann2d(_GRID, _GRID)


def _download_model() -> pathlib.Path:
    """Return path to the ONNX model, downloading from HF Hub if needed."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME)
        return pathlib.Path(path)
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the VitTrack model. "
            "Install it with: uv add huggingface-hub"
        ) from exc


def _make_session(model_path: pathlib.Path) -> ort.InferenceSession:
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    try:
        # Use CoreML on macOS if available (MPS-backed)
        from onnxruntime.capi import _pybind_state as C  # noqa: N812

        available = C.get_available_providers()
        if "CoreMLExecutionProvider" in available:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass

    return ort.InferenceSession(str(model_path), providers=providers)


def _crop_region(
    frame: np.ndarray,
    bbox_px: tuple[float, float, float, float],
    factor: float,
) -> tuple[np.ndarray, int]:
    """Extract a square crop centred on *bbox_px* with context factor.

    Args:
        frame: uint8 RGB (H, W, 3).
        bbox_px: (x, y, w, h) in *pixel* coordinates.
        factor: Crop size = ceil(sqrt(w * h) * factor).

    Returns:
        (crop_rgb, crop_sz) — crop_sz is the square side length in pixels.
    """
    x, y, w, h = bbox_px
    crop_sz = int(math.ceil(math.sqrt(w * h) * factor))
    crop_sz = max(crop_sz, 1)

    cx = x + w / 2
    cy = y + h / 2
    x1 = int(round(cx - crop_sz / 2))
    y1 = int(round(cy - crop_sz / 2))
    x2 = x1 + crop_sz
    y2 = y1 + crop_sz

    fh, fw = frame.shape[:2]
    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - fw)
    pad_b = max(0, y2 - fh)

    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(fw, x2)
    y2c = min(fh, y2)

    roi = frame[y1c:y2c, x1c:x2c]
    if pad_l or pad_t or pad_r or pad_b:
        roi = cv2.copyMakeBorder(roi, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)

    return roi, crop_sz


def _preprocess(crop_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize, normalize and convert crop to NCHW float32 blob."""
    resized = cv2.resize(crop_rgb, size)
    img = resized.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    blob = img.transpose(2, 0, 1)[np.newaxis]  # HWC → NCHW
    return blob.astype(np.float32)


def _decode_outputs(
    outputs: list[np.ndarray],
    crop_sz: int,
    bbox_px: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], float]:
    """Decode ONNX outputs into a pixel-space bbox and confidence score.

    Args:
        outputs: [conf_raw, size_raw, offset_raw] from session.run().
        crop_sz: Square crop size used for the search region.
        bbox_px: Previous bbox (x, y, w, h) in pixel coords used to define crop origin.

    Returns:
        (new_bbox_px, score)
    """
    conf_map = outputs[0].reshape(_GRID, _GRID) * _HANN_WINDOW
    size_map = outputs[1].reshape(2, _GRID, _GRID)
    offset_map = outputs[2].reshape(2, _GRID, _GRID)

    flat_idx = int(np.argmax(conf_map))
    gy, gx = divmod(flat_idx, _GRID)
    score = float(conf_map[gy, gx])

    # cx, cy normalized in crop space [0..1]
    cx = (gx + float(offset_map[0, gy, gx])) / _GRID
    cy = (gy + float(offset_map[1, gy, gx])) / _GRID

    # w, h normalized in crop space
    w_norm = float(size_map[0, gy, gx])
    h_norm = float(size_map[1, gy, gx])

    # Convert to pixel coords in crop space
    w_px = w_norm * crop_sz
    h_px = h_norm * crop_sz

    # Top-left of crop in frame coords
    x_prev, y_prev, w_prev, h_prev = bbox_px
    cx_prev = x_prev + w_prev / 2
    cy_prev = y_prev + h_prev / 2
    crop_x0 = cx_prev - crop_sz / 2
    crop_y0 = cy_prev - crop_sz / 2

    # Center in frame coords
    cx_frame = crop_x0 + cx * crop_sz
    cy_frame = crop_y0 + cy * crop_sz

    new_x = cx_frame - w_px / 2
    new_y = cy_frame - h_px / 2

    return (new_x, new_y, w_px, h_px), score


class VitTrackSOT:
    """ViT-based single-object ball tracker (method A, bake-off ft-ztw).

    Implements the BallTracker protocol. Downloads the VitTrack ONNX model
    from HuggingFace Hub on first instantiation.

    The tracker uses a motion-guided ROI approach:
    1. On first valid prev_bbox, crops a 128×128 template patch and caches it.
    2. On subsequent frames, crops a 256×256 search region around the previous
       position and runs the ViT to locate the ball.
    3. If score < threshold, returns None (ball lost) without updating state.
    """

    def __init__(self, model_path: str | pathlib.Path | None = None) -> None:
        """
        Args:
            model_path: Path to a local .onnx file. If None, downloads from
                HuggingFace Hub automatically.
        """
        if model_path is None:
            model_path = _download_model()
        self._session = _make_session(pathlib.Path(model_path))
        self._template_blob: np.ndarray | None = None
        self._last_bbox_px: tuple[float, float, float, float] | None = None
        # Exposed to harness via getattr — populated after each track() call
        self._last_crop_height: int | None = None

    def reset(self) -> None:
        """Reset all state. Called by the harness between clips."""
        self._template_blob = None
        self._last_bbox_px = None
        self._last_crop_height = None

    def track(
        self,
        prev_bbox: tuple[float, float, float, float] | None,
        frame: np.ndarray,
    ) -> tuple[float, float, float, float] | None:
        """Locate the ball in *frame* given its previous normalised bbox.

        Args:
            prev_bbox: Normalised (x, y, w, h) [0..1] or None on cold-start.
            frame: uint8 RGB array (H, W, 3).

        Returns:
            Normalised (x, y, w, h) if ball located, else None.
        """
        fh, fw = frame.shape[:2]

        if prev_bbox is None:
            # No anchor — cannot initialize or track
            self._last_crop_height = None
            return None

        # Convert normalised to pixel coords
        bbox_px = (
            prev_bbox[0] * fw,
            prev_bbox[1] * fh,
            prev_bbox[2] * fw,
            prev_bbox[3] * fh,
        )

        # Warm template on first valid bbox
        if self._template_blob is None:
            self._template_blob = self._init_template(frame, bbox_px)
            self._last_bbox_px = bbox_px

        # Extract search region around last known position
        search_bbox_px = self._last_bbox_px  # use internal state, not caller-supplied
        search_crop, crop_sz = _crop_region(frame, search_bbox_px, _SEARCH_FACTOR)
        search_blob = _preprocess(search_crop, _SEARCH_SIZE)
        self._last_crop_height = search_crop.shape[0]

        # Run the ONNX model
        outputs = self._session.run(
            ["output1", "output2", "output3"],
            {"template": self._template_blob, "search": search_blob},
        )

        new_bbox_px, score = _decode_outputs(outputs, crop_sz, search_bbox_px)

        if score < _SCORE_THRESHOLD:
            # Ball lost — keep last known position but don't update state
            return None

        self._last_bbox_px = new_bbox_px

        # Convert back to normalised coords
        nx = new_bbox_px[0] / fw
        ny = new_bbox_px[1] / fh
        nw = new_bbox_px[2] / fw
        nh = new_bbox_px[3] / fh

        # Clamp to [0, 1]
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        nw = max(0.0, min(1.0 - nx, nw))
        nh = max(0.0, min(1.0 - ny, nh))

        if nw <= 0 or nh <= 0:
            return None

        return (nx, ny, nw, nh)

    def _init_template(
        self,
        frame: np.ndarray,
        bbox_px: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Crop and preprocess the template patch."""
        crop, _ = _crop_region(frame, bbox_px, _TEMPLATE_FACTOR)
        return _preprocess(crop, _TEMPLATE_SIZE)
