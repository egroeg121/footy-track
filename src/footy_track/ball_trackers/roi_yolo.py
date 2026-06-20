"""Method C: tiny ROI-YOLO ball detector on a motion-predicted crop.

Algorithm
---------
1. Kalman filter (constant-velocity, 2-D centre) predicts where the ball
   should be in the current frame.
2. A square ROI crop is cut around that prediction (size = prev_bbox diagonal
   × `roi_scale`, clamped to the frame).
3. A tiny YOLO model (default: yolo11n.pt, COCO class 32 = "sports ball")
   runs inference on only the cropped pixels.
4. The highest-confidence sports-ball detection inside the crop is mapped back
   to full-frame normalised (x, y, w, h) coordinates and returned.

Why this wins on "effective resolution": the ball is small (≈10-30 px on a
1920-wide broadcast frame ≈ 0.5–1.5% of width).  Running YOLO on a tight
160×160 crop around the predicted location gives the model a ball that fills
5–15% of the input — 10× more signal per pixel than running on the full frame
at the same imgsz.  Speed follows: inference on 160×160 is dramatically faster
than on 640×640.

Interface compatibility
-----------------------
Implements BallTracker protocol (footy_track.ball_eval.interface):
  - track(prev_bbox, frame) -> BBox | None
  - reset() -> None

The runner also reads ``self._last_crop_height`` (int | None) to record the
effective resolution in ClipMetrics.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from footy_track.ball_eval.dataset import BBox

# COCO class index for "sports ball"
_COCO_SPORTS_BALL = 32

# Kalman state: [cx, cy, vx, vy] — normalised screen coords
_F = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
)
_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)


def _kf_init(cx: float, cy: float, q: float, r: float) -> dict:
    return {
        "x": np.array([cx, cy, 0.0, 0.0]),
        "P": np.eye(4) * 0.1,
        "Q": np.eye(4) * q,
        "R": np.eye(2) * r,
    }


def _kf_predict(kf: dict) -> tuple[float, float]:
    kf["x"] = _F @ kf["x"]
    kf["P"] = _F @ kf["P"] @ _F.T + kf["Q"]
    return float(kf["x"][0]), float(kf["x"][1])


def _kf_update(kf: dict, cx: float, cy: float) -> None:
    z = np.array([cx, cy])
    y = z - _H @ kf["x"]
    S = _H @ kf["P"] @ _H.T + kf["R"]
    K = kf["P"] @ _H.T @ np.linalg.inv(S)
    kf["x"] = kf["x"] + K @ y
    kf["P"] = (np.eye(4) - K @ _H) @ kf["P"]


class RoiYoloTracker:
    """Bake-off Method C: tiny YOLO inference on a Kalman-predicted ROI crop.

    Parameters
    ----------
    model_uri:
        Path or Ultralytics hub name for the YOLO checkpoint (default: yolo11n.pt
        which includes COCO "sports ball" class 32).
    roi_scale:
        ROI side length as a multiple of the ball's bounding-box diagonal from
        the previous frame.  1.5 → 50% padding around the predicted ball centre.
    min_roi_frac:
        Minimum ROI size as a fraction of frame height (prevents tiny crops when
        the ball bounding box is unreliably small on first detection).
    min_confidence:
        Detection confidence threshold.
    process_noise / measurement_noise:
        Kalman filter tuning.
    """

    def __init__(
        self,
        model_uri: str = "yolo11n.pt",
        roi_scale: float = 3.0,
        min_roi_frac: float = 0.10,
        min_confidence: float = 0.20,
        process_noise: float = 5e-5,
        measurement_noise: float = 5e-4,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_uri)
        self._device = _pick_device()
        self._roi_scale = roi_scale
        self._min_roi_frac = min_roi_frac
        self._min_confidence = min_confidence
        self._q = process_noise
        self._r = measurement_noise

        self._kf: dict | None = None
        self._last_bbox: BBox | None = None  # prev frame's tracked bbox (normalised)
        self._last_crop_height: int | None = None  # read by harness runner

    # ------------------------------------------------------------------
    # BallTracker protocol
    # ------------------------------------------------------------------

    def track(self, prev_bbox: BBox | None, frame: np.ndarray) -> BBox | None:
        """Locate the ball in *frame* using motion-guided ROI + YOLO.

        Args:
            prev_bbox: Normalised (x, y, w, h) from previous frame, or None.
            frame: uint8 RGB array (H, W, 3).

        Returns:
            Normalised (x, y, w, h) if ball found, else None.
        """
        H, W = frame.shape[:2]

        if prev_bbox is not None:
            cx_n = prev_bbox[0] + prev_bbox[2] / 2.0
            cy_n = prev_bbox[1] + prev_bbox[3] / 2.0
            if self._kf is None:
                self._kf = _kf_init(cx_n, cy_n, self._q, self._r)
                pred_cx, pred_cy = cx_n, cy_n
            else:
                pred_cx, pred_cy = _kf_predict(self._kf)
            # Correct immediately with the prev bbox centre (we already have it)
            _kf_update(self._kf, cx_n, cy_n)
            # Predict again for the *current* frame
            pred_cx, pred_cy = _kf_predict(self._kf)
        elif self._kf is not None:
            # Ball was lost last frame — keep predicting without update
            pred_cx, pred_cy = _kf_predict(self._kf)
        else:
            # Cold start — no prior info; search the full frame
            pred_cx, pred_cy = 0.5, 0.5

        # Determine ROI size
        if prev_bbox is not None:
            bw, bh = prev_bbox[2], prev_bbox[3]
            diag = math.hypot(bw * W, bh * H)
            roi_side = int(diag * self._roi_scale)
        else:
            roi_side = None  # full frame cold start

        roi_side_min = int(H * self._min_roi_frac)

        if roi_side is None:
            # Cold start — use full frame
            crop = frame
            roi_x0, roi_y0, roi_x1, roi_y1 = 0, 0, W, H
        else:
            roi_side = max(roi_side, roi_side_min)
            # Crop centred on prediction
            px = int(pred_cx * W)
            py = int(pred_cy * H)
            half = roi_side // 2
            roi_x0 = max(0, px - half)
            roi_y0 = max(0, py - half)
            roi_x1 = min(W, roi_x0 + roi_side)
            roi_y1 = min(H, roi_y0 + roi_side)
            # Shift if clipped
            if roi_x1 - roi_x0 < roi_side:
                roi_x0 = max(0, roi_x1 - roi_side)
            if roi_y1 - roi_y0 < roi_side:
                roi_y0 = max(0, roi_y1 - roi_side)
            crop = frame[roi_y0:roi_y1, roi_x0:roi_x1]

        self._last_crop_height = crop.shape[0]

        # YOLO expects BGR; frame is RGB
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            results = self._model.predict(
                crop_bgr,
                device=self._device,
                conf=self._min_confidence,
                classes=[_COCO_SPORTS_BALL],
                verbose=False,
            )

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        # Pick highest-confidence sports-ball detection
        confs = result.boxes.conf.tolist()
        best_idx = int(max(range(len(confs)), key=lambda i: confs[i]))
        if confs[best_idx] < self._min_confidence:
            return None

        # Extract crop-normalised top-left box
        xyxyn = result.boxes.xyxyn[best_idx].tolist()
        cx1, cy1, cx2, cy2 = xyxyn
        cw_crop = crop.shape[1]
        ch_crop = crop.shape[0]

        # Map from crop-normalised to pixel in crop, then to full-frame normalised
        px1 = cx1 * cw_crop + roi_x0
        py1 = cy1 * ch_crop + roi_y0
        px2 = cx2 * cw_crop + roi_x0
        py2 = cy2 * ch_crop + roi_y0

        x_n = float(max(0.0, px1 / W))
        y_n = float(max(0.0, py1 / H))
        w_n = float(min(1.0, (px2 - px1) / W))
        h_n = float(min(1.0, (py2 - py1) / H))

        if w_n <= 0 or h_n <= 0:
            return None

        detected: BBox = (x_n, y_n, w_n, h_n)

        # Update Kalman with actual detection
        det_cx = x_n + w_n / 2.0
        det_cy = y_n + h_n / 2.0
        if self._kf is None:
            self._kf = _kf_init(det_cx, det_cy, self._q, self._r)
        else:
            _kf_update(self._kf, det_cx, det_cy)

        self._last_bbox = detected
        return detected

    def reset(self) -> None:
        """Reset all internal state (called between eval clips)."""
        self._kf = None
        self._last_bbox = None
        self._last_crop_height = None


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
