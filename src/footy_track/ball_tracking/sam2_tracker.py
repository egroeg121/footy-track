"""Bake-off method B: box-prompted SAM2 video predictor on ROI.

Strategy
--------
Each frame, a motion-guided ROI is cropped from the full frame using the
Kalman-predicted ball position.  SAM2 (Ultralytics SAM2 image predictor) is
then prompted with the propagated bounding box, converted to crop-local coords.
The resulting mask is converted back to a normalised full-frame bbox.

Why SAM2 image predictor (not video predictor)?
SAM2's *video* predictor requires the full clip upfront and runs as a pipeline,
which doesn't fit the frame-by-frame protocol.  The *image* predictor gives us
mask+box for a single frame given a box prompt — lighter, CPU-compatible, and
composable with our own Kalman-based propagation between frames.

Interface: implements BallTracker from footy_track.ball_eval.interface.
"""

from __future__ import annotations

import numpy as np

from footy_track.ball_eval.dataset import BBox

# Kalman state: [cx, cy, vx, vy] in normalised coords.
_KF_F = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
)
_KF_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)


def _kf_init(cx: float, cy: float) -> dict:
    return {
        "x": np.array([cx, cy, 0.0, 0.0], dtype=np.float64),
        "P": np.eye(4, dtype=np.float64) * 0.1,
        "Q": np.eye(4, dtype=np.float64) * 5e-5,
        "R": np.eye(2, dtype=np.float64) * 1e-3,
    }


def _kf_predict(kf: dict) -> np.ndarray:
    kf["x"] = _KF_F @ kf["x"]
    kf["P"] = _KF_F @ kf["P"] @ _KF_F.T + kf["Q"]
    return kf["x"][:2].copy()


def _kf_update(kf: dict, cx: float, cy: float) -> None:
    z = np.array([cx, cy], dtype=np.float64)
    y = z - _KF_H @ kf["x"]
    S = _KF_H @ kf["P"] @ _KF_H.T + kf["R"]
    K = kf["P"] @ _KF_H.T @ np.linalg.inv(S)
    kf["x"] = kf["x"] + K @ y
    kf["P"] = (np.eye(4) - K @ _KF_H) @ kf["P"]


def _bbox_centre(bbox: BBox) -> tuple[float, float]:
    """Centre of a normalised (x, y, w, h) bbox."""
    return bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0


def _compute_roi(
    pred_cx: float,
    pred_cy: float,
    bbox_w: float,
    bbox_h: float,
    scale: float = 3.0,
) -> tuple[float, float, float, float]:
    """Return a normalised (x1, y1, x2, y2) ROI centred on the prediction."""
    roi_w = min(bbox_w * scale, 1.0)
    roi_h = min(bbox_h * scale, 1.0)
    x1 = max(0.0, pred_cx - roi_w / 2.0)
    y1 = max(0.0, pred_cy - roi_h / 2.0)
    x2 = min(1.0, x1 + roi_w)
    y2 = min(1.0, y1 + roi_h)
    # re-anchor left/top after clamping
    x1 = max(0.0, x2 - roi_w)
    y1 = max(0.0, y2 - roi_h)
    return x1, y1, x2, y2


def _mask_to_bbox(mask: np.ndarray) -> BBox | None:
    """Convert a binary mask to a normalised (x, y, w, h) bbox."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    cmin, cmax = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    H, W = mask.shape
    return (cmin / W, rmin / H, (cmax - cmin + 1) / W, (rmax - rmin + 1) / H)


class Sam2BallTracker:
    """SAM2 image predictor, box-prompted on a Kalman-guided ROI crop.

    Parameters
    ----------
    model_name:
        Ultralytics model tag. Default "sam2_b.pt" (SAM2 Base — 80M params).
        Use "sam2_t.pt" for tiny (38M, faster) or "sam2_l.pt" for large.
    roi_scale:
        How many ball-widths/heights around the predicted centre to crop.
    device:
        "cpu", "mps", or "cuda".  Auto-detected if None.
    conf_threshold:
        Minimum SAM2 mask confidence to accept.  Masks below this are treated
        as "ball not found".
    """

    def __init__(
        self,
        model_name: str = "sam2_b.pt",
        roi_scale: float = 3.0,
        device: str | None = None,
        conf_threshold: float = 0.5,
    ) -> None:
        self._model_name = model_name
        self._roi_scale = roi_scale
        self._conf_threshold = conf_threshold
        self._device = device or _auto_device()

        self._predictor = None  # lazy-loaded on first call
        self._kf: dict | None = None
        self._last_bbox: BBox | None = None
        # Exposed for harness to measure effective resolution
        self._last_crop_height: int | None = None

    def _ensure_model(self) -> None:
        if self._predictor is not None:
            return
        from ultralytics.models.sam import SAM  # noqa: PLC0415

        self._predictor = SAM(self._model_name)

    def track(self, prev_bbox: BBox | None, frame: np.ndarray) -> BBox | None:
        """Locate the ball in *frame* given the previous bounding box.

        Args:
            prev_bbox: Normalised (x, y, w, h) from the previous frame, or None.
            frame: uint8 RGB array (H, W, 3).

        Returns:
            Normalised (x, y, w, h) if ball found, else None.
        """
        self._ensure_model()
        H, W = frame.shape[:2]

        # --- 1. Kalman predict ---
        if prev_bbox is None:
            # No prior: run SAM2 on full frame with no box (point-free fallback)
            return self._full_frame_search(frame, H, W)

        cx, cy = _bbox_centre(prev_bbox)
        bw, bh = prev_bbox[2], prev_bbox[3]

        if self._kf is None:
            self._kf = _kf_init(cx, cy)

        pred_pos = _kf_predict(self._kf)
        pred_cx, pred_cy = float(pred_pos[0]), float(pred_pos[1])
        pred_cx = max(bw / 2.0, min(1.0 - bw / 2.0, pred_cx))
        pred_cy = max(bh / 2.0, min(1.0 - bh / 2.0, pred_cy))

        # --- 2. Crop ROI ---
        roi_x1, roi_y1, roi_x2, roi_y2 = _compute_roi(pred_cx, pred_cy, bw, bh, self._roi_scale)

        # Pixel coords of ROI
        px1 = int(roi_x1 * W)
        py1 = int(roi_y1 * H)
        px2 = int(roi_x2 * W)
        py2 = int(roi_y2 * H)

        if px2 - px1 < 4 or py2 - py1 < 4:
            return None

        crop = frame[py1:py2, px1:px2]
        self._last_crop_height = crop.shape[0]

        # --- 3. Compute box prompt in crop-local normalised coords ---
        # Project prev_bbox into the crop
        crop_w = px2 - px1
        crop_h = py2 - py1

        # prev_bbox in absolute pixels
        abs_bx1 = prev_bbox[0] * W - px1
        abs_by1 = prev_bbox[1] * H - py1
        abs_bx2 = abs_bx1 + prev_bbox[2] * W
        abs_by2 = abs_by1 + prev_bbox[3] * H

        # Clamp to crop and convert to xyxy for SAM2
        box_x1 = max(0.0, abs_bx1)
        box_y1 = max(0.0, abs_by1)
        box_x2 = min(float(crop_w), abs_bx2)
        box_y2 = min(float(crop_h), abs_by2)

        if box_x2 - box_x1 < 1.0 or box_y2 - box_y1 < 1.0:
            # Prev box is outside this crop — do a free-search on the crop
            box_prompt = None
        else:
            box_prompt = [box_x1, box_y1, box_x2, box_y2]

        # --- 4. SAM2 inference on crop ---
        result_bbox = self._run_sam2(crop, box_prompt)
        if result_bbox is None:
            _kf_predict(self._kf)  # advance Kalman without update
            return None

        # --- 5. Map crop-local normalised bbox back to full frame ---
        local_x, local_y, local_w, local_h = result_bbox
        full_x = roi_x1 + local_x * (roi_x2 - roi_x1)
        full_y = roi_y1 + local_y * (roi_y2 - roi_y1)
        full_w = local_w * (roi_x2 - roi_x1)
        full_h = local_h * (roi_y2 - roi_y1)

        full_cx = full_x + full_w / 2.0
        full_cy = full_y + full_h / 2.0
        _kf_update(self._kf, full_cx, full_cy)

        self._last_bbox = (full_x, full_y, full_w, full_h)
        return self._last_bbox

    def _run_sam2(self, crop: np.ndarray, box_prompt: list[float] | None) -> BBox | None:
        """Run SAM2 on the crop, return normalised (x, y, w, h) or None."""
        import torch  # noqa: PLC0415

        import cv2  # noqa: PLC0415

        # SAM2 predictor expects BGR
        bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        kwargs: dict = {"conf": self._conf_threshold, "verbose": False}
        if box_prompt is not None:
            kwargs["bboxes"] = [box_prompt]

        with torch.inference_mode():
            results = self._predictor(bgr_crop, **kwargs)

        if not results:
            return None

        result = results[0]
        masks = getattr(result, "masks", None)
        if masks is None or len(masks) == 0:
            # Fall back to boxes if available
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy[0].cpu().numpy()
                h, w = crop.shape[:2]
                x1, y1, x2, y2 = xyxy
                return (float(x1 / w), float(y1 / h), float((x2 - x1) / w), float((y2 - y1) / h))
            return None

        # Use the highest-confidence mask
        best_mask = masks.data[0].cpu().numpy().astype(bool)
        return _mask_to_bbox(best_mask)

    def _full_frame_search(self, frame: np.ndarray, H: int, W: int) -> BBox | None:
        """Cold-start: run SAM2 on the full frame without a box prompt."""
        self._last_crop_height = H
        result_bbox = self._run_sam2(frame, box_prompt=None)
        if result_bbox is None:
            return None
        cx, cy = _bbox_centre(result_bbox)
        self._kf = _kf_init(cx, cy)
        self._last_bbox = result_bbox
        return result_bbox

    def reset(self) -> None:
        """Reset all state between eval clips."""
        self._kf = None
        self._last_bbox = None
        self._last_crop_height = None


def _auto_device() -> str:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
