"""Method C: tiny ROI-YOLO ball detector on a motion-predicted crop.

Algorithm
---------
1. Kalman filter (constant-velocity, 2-D centre) predicts where the ball
   should be in the current frame.
2. A square ROI crop is cut around that prediction (size = prev_bbox diagonal
   × `roi_scale`, clamped to the frame).
3. A YOLO model runs inference on only the cropped pixels.
4. The highest-confidence ball detection inside the crop is mapped back
   to full-frame normalised (x, y, w, h) coordinates and returned.

Model
-----
The default model is the project-trained detector (yolo11s fine-tuned on
broadcast footage, classes 0=ball, 2=in_play_ball). Stock COCO models have
0% recall on broadcast footage because "sports ball" (class 32) was not
trained on football footage at broadcast resolution.

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
import pathlib

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from footy_track.ball_eval.dataset import BBox

# Default trained detector: yolo11s fine-tuned on broadcast footage.
# Classes: 0=ball, 1=coach, 2=in_play_ball, 3=person, 4=player, 5=player_sub, 6=referee
_DEFAULT_MODEL_PATH = str(
    pathlib.Path(__file__).parents[3]
    / "model_saves"
    / "detector"
    / "optuna_trial_1_2026-01-18_17-51_model_name=yolo11s_dataset_version=3_epochs=2226_freeze_layers=3"
    / "best.pt"
)
# Classes in the trained model that represent the ball
_BALL_CLASSES = [0, 2]  # 0=ball, 2=in_play_ball

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


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _kalman_predict(
    kf: dict | None,
    prev_bbox: BBox | None,
    q: float,
    r: float,
) -> tuple[dict, float, float]:
    """Advance Kalman state; return (kf, pred_cx, pred_cy)."""
    if prev_bbox is not None:
        cx_n = prev_bbox[0] + prev_bbox[2] / 2.0
        cy_n = prev_bbox[1] + prev_bbox[3] / 2.0
        if kf is None:
            kf = _kf_init(cx_n, cy_n, q, r)
            pred_cx, pred_cy = cx_n, cy_n
        else:
            pred_cx, pred_cy = _kf_predict(kf)
        _kf_update(kf, cx_n, cy_n)
        pred_cx, pred_cy = _kf_predict(kf)
    elif kf is not None:
        pred_cx, pred_cy = _kf_predict(kf)
    else:
        pred_cx, pred_cy = 0.5, 0.5
    if kf is None:
        kf = _kf_init(pred_cx, pred_cy, q, r)
    return kf, pred_cx, pred_cy


def _compute_roi(
    prev_bbox: BBox | None,
    pred_cx: float,
    pred_cy: float,
    H: int,
    W: int,
    roi_scale: float,
    min_roi_frac: float,
    min_roi_px: int = 512,
) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) pixel ROI for the current frame."""
    roi_side_min = max(int(H * min_roi_frac), min_roi_px)

    if prev_bbox is None:
        return 0, 0, W, H

    bw, bh = prev_bbox[2], prev_bbox[3]
    roi_side = max(int(math.hypot(bw * W, bh * H) * roi_scale), roi_side_min)

    px, py = int(pred_cx * W), int(pred_cy * H)
    half = roi_side // 2
    x0 = max(0, px - half)
    y0 = max(0, py - half)
    x1 = min(W, x0 + roi_side)
    y1 = min(H, y0 + roi_side)
    if x1 - x0 < roi_side:
        x0 = max(0, x1 - roi_side)
    if y1 - y0 < roi_side:
        y0 = max(0, y1 - roi_side)
    return x0, y0, x1, y1


def _best_detection(result, min_confidence: float) -> BBox | None:
    """Extract the highest-confidence sports-ball bbox from a YOLO result.

    Returns crop-normalised (x1, y1, x2, y2) or None.
    """
    if result.boxes is None or result.boxes.conf is None or len(result.boxes.conf) == 0:
        return None
    confs = result.boxes.conf.tolist()
    best_idx = int(max(range(len(confs)), key=lambda i: confs[i]))
    if confs[best_idx] < min_confidence:
        return None
    return tuple(result.boxes.xyxyn[best_idx].tolist())  # type: ignore[return-value]


class RoiYoloTracker:
    """Bake-off Method C: tiny YOLO inference on a Kalman-predicted ROI crop.

    Parameters
    ----------
    model_path:
        Path to the YOLO checkpoint. Defaults to the project-trained detector
        (yolo11s fine-tuned on broadcast footage). Override to use a different model.
    ball_classes:
        YOLO class indices that represent the ball. Defaults to [0, 2] for the
        trained model (0=ball, 2=in_play_ball). Set to [32] for stock COCO models.
    roi_scale:
        ROI side length as a multiple of the ball's bounding-box diagonal from
        the previous frame.  3.0 → 3× diagonal padding around the predicted centre.
    min_roi_px:
        Minimum ROI side in pixels. Defaults to 512 — empirically 512px crops yield
        11.6% detection vs 0% at 160px on broadcast footage (ft-019 bake-off finding).
    min_roi_frac:
        Minimum ROI size as a fraction of frame height (secondary floor). The actual
        minimum is max(min_roi_px, frame_height * min_roi_frac).
    min_confidence:
        Detection confidence threshold.
    process_noise / measurement_noise:
        Kalman filter tuning.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        ball_classes: list[int] | None = None,
        roi_scale: float = 3.0,
        min_roi_px: int = 512,
        min_roi_frac: float = 0.10,
        min_confidence: float = 0.20,
        process_noise: float = 5e-5,
        measurement_noise: float = 5e-4,
    ) -> None:
        self._model = YOLO(model_path)
        self._ball_classes = ball_classes if ball_classes is not None else _BALL_CLASSES
        self._device = _pick_device()
        self._roi_scale = roi_scale
        self._min_roi_px = min_roi_px
        self._min_roi_frac = min_roi_frac
        self._min_confidence = min_confidence
        self._q = process_noise
        self._r = measurement_noise

        self._kf: dict | None = None
        self._last_bbox: BBox | None = None
        self._last_crop_height: int | None = None  # read by harness runner

    # ------------------------------------------------------------------
    # BallTracker protocol
    # ------------------------------------------------------------------

    def track(self, prev_bbox: BBox | None, frame: np.ndarray) -> BBox | None:  # noqa: PLR0912
        """Locate the ball in *frame* using motion-guided ROI + YOLO.

        Args:
            prev_bbox: Normalised (x, y, w, h) from previous frame, or None.
            frame: uint8 RGB array (H, W, 3).

        Returns:
            Normalised (x, y, w, h) if ball found, else None.
        """
        H, W = frame.shape[:2]

        self._kf, pred_cx, pred_cy = _kalman_predict(
            self._kf, prev_bbox, self._q, self._r
        )

        x0, y0, x1, y1 = _compute_roi(
            prev_bbox,
            pred_cx,
            pred_cy,
            H,
            W,
            self._roi_scale,
            self._min_roi_frac,
            self._min_roi_px,
        )
        crop = frame[y0:y1, x0:x1]
        self._last_crop_height = crop.shape[0]

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            results = self._model.predict(
                crop_bgr,
                device=self._device,
                conf=self._min_confidence,
                classes=self._ball_classes,
                verbose=False,
            )

        xyxyn_crop = _best_detection(results[0], self._min_confidence)
        if xyxyn_crop is None:
            return None

        # Map from crop-normalised x1y1x2y2 → full-frame normalised xywh
        cx1, cy1, cx2, cy2 = xyxyn_crop
        cw = x1 - x0
        ch = y1 - y0
        px1 = cx1 * cw + x0
        py1 = cy1 * ch + y0
        px2 = cx2 * cw + x0
        py2 = cy2 * ch + y0

        x_n = float(max(0.0, px1 / W))
        y_n = float(max(0.0, py1 / H))
        w_n = float(min(1.0, (px2 - px1) / W))
        h_n = float(min(1.0, (py2 - py1) / H))

        if w_n <= 0 or h_n <= 0:
            return None

        detected: BBox = (x_n, y_n, w_n, h_n)

        det_cx = x_n + w_n / 2.0
        det_cy = y_n + h_n / 2.0
        _kf_update(self._kf, det_cx, det_cy)

        self._last_bbox = detected
        return detected

    def reset(self) -> None:
        """Reset all internal state (called between eval clips)."""
        self._kf = None
        self._last_bbox = None
        self._last_crop_height = None
