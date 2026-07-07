"""Motion-guided box-prompt tracker + ``FrameTracker``/``CropRunner`` protocol.

Implements ft-v43, the core new component of the SAM3 ball-labelling pipeline
(see ``docs/design/sam3_ball_labelling_gpu.md`` §2, §4, §7.2).

Per-step loop (design doc §2.1)::

    state: KalmanBoxState  (cx, cy, w, h, vx, vy)
    for each frame f:
        pred_box  = kalman.predict()                     # where do we expect the ball?
        roi       = expand(pred_box, margin)              # tight crop window
        crop      = frame[roi]                            # high effective resolution
        det       = runner.detect(crop, prior=local(pred_box))  # swappable backend
        if det.confidence >= thresh:
            box_full = map_to_frame(det.box, roi)
            kalman.update(box_full)
        else:
            box_full = reacquire(frame)                   # full-frame fallback
            ...

Two protocols decouple the loop from the model (design doc §2.4):

- ``CropRunner`` — a swappable per-step detector/segmenter/SOT backend that
  operates on a crop and returns a box in CROP-local coordinates.
- ``FrameTracker`` — the stateful, per-clip tracker exposed to callers
  (``reset`` / ``step``), operating in FRAME-absolute coordinates.

``MotionGuidedTracker`` implements ``FrameTracker``: it owns the Kalman state,
the crop sizing/extraction, the crop<->frame coordinate round-trip, and the
full-frame re-acquire fallback (§2.3) used when the per-step runner misses.

Box convention throughout this module: absolute pixel ``(x1, y1, x2, y2)``,
matching ``LabelledObject.bbox_xyxy_abs`` elsewhere in the labeller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# Absolute-pixel xyxy box, consistent with LabelledObject.bbox_xyxy_abs.
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """A single detection returned by a ``CropRunner`` or ``FrameTracker``.

    ``box`` coordinates are in whatever frame the producing method documents:
    ``CropRunner.detect`` returns CROP-local coordinates; ``FrameTracker.step``
    (and ``MotionGuidedTracker``'s internal use of a runner) returns
    FRAME-absolute coordinates.
    """

    box: Box
    confidence: float
    label: str = ""


@runtime_checkable
class CropRunner(Protocol):
    """A swappable per-step backend: detect/segment/track ONE box in a crop."""

    name: str

    def warmup(self) -> None:
        """Load weights / JIT-warm the model. Safe to call multiple times."""
        ...

    def detect(self, crop: np.ndarray, prior: Box | None) -> Detection | None:
        """Locate the tracked object in ``crop``.

        Args:
            crop: uint8 array (H, W, 3), pixels of the cropped ROI (or the
                full frame, when used for full-frame re-acquire).
            prior: Optional predicted box in CROP-local coordinates, usable as
                a prompt/hint by the backend.

        Returns:
            A ``Detection`` with ``box`` in CROP-local coordinates, or
            ``None`` if nothing was found.
        """
        ...


@runtime_checkable
class FrameTracker(Protocol):
    """The stateful, per-clip tracker: FRAME-absolute coordinates in/out."""

    def reset(self, frame: np.ndarray, seed: Box) -> None:
        """(Re)initialise tracking state from a known-good seed box on ``frame``."""
        ...

    def step(self, frame: np.ndarray) -> Detection | None:
        """Advance one frame and return the current detection, in FRAME coords."""
        ...


# ---------------------------------------------------------------------------
# Coordinate helpers: crop <-> frame round trip
# ---------------------------------------------------------------------------


def box_to_cxcywh(box: Box) -> tuple[float, float, float, float]:
    """Convert an xyxy box to ``(cx, cy, w, h)``."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def cxcywh_to_box(cx: float, cy: float, w: float, h: float) -> Box:
    """Convert ``(cx, cy, w, h)`` to an xyxy box."""
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def compute_roi(
    pred_box: Box,
    frame_w: int,
    frame_h: int,
    velocity: tuple[float, float] = (0.0, 0.0),
    margin_scale: float = 2.5,
    velocity_scale: float = 1.5,
    min_size: float = 64.0,
) -> Box:
    """Return a tight, frame-clamped square ROI (xyxy, absolute pixels).

    Sized from the predicted box's own dimensions and expanded further when
    the ball is moving fast (design doc §2.2: "Drive margin from predicted
    velocity — bigger crop when the ball is moving fast") so a fast ball is
    less likely to escape the crop before the next prediction.

    Args:
        pred_box: The Kalman-predicted box in FRAME-absolute coordinates.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.
        velocity: ``(vx, vy)`` in pixels/frame, from the Kalman state.
        margin_scale: ROI side = ``max(w, h, min_size) * margin_scale``,
            before the velocity term.
        velocity_scale: Additional ROI side growth per unit speed.
        min_size: Minimum ROI side length in pixels.

    Returns:
        ``(x1, y1, x2, y2)`` clamped to ``[0, frame_w] x [0, frame_h]``.
    """
    cx, cy, w, h = box_to_cxcywh(pred_box)
    vx, vy = velocity
    speed = float(np.hypot(vx, vy))
    side = max(w, h, min_size) * margin_scale + speed * velocity_scale
    side = max(side, min_size)
    # Never larger than the frame itself.
    side = min(side, float(max(frame_w, frame_h)))
    half = side / 2.0

    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half

    # Shift (not shrink) to stay in-bounds, preserving ROI size where possible.
    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0:
        y2 -= y1
        y1 = 0.0
    if x2 > frame_w:
        x1 -= x2 - frame_w
        x2 = float(frame_w)
    if y2 > frame_h:
        y1 -= y2 - frame_h
        y2 = float(frame_h)

    # Final clamp in case the frame is smaller than the ROI itself.
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(frame_w), x2)
    y2 = min(float(frame_h), y2)
    return (x1, y1, x2, y2)


def crop_frame(frame: np.ndarray, roi: Box) -> np.ndarray:
    """Slice ``frame`` to the integer-pixel ``roi`` (xyxy, absolute)."""
    x1, y1, x2, y2 = (int(round(v)) for v in roi)
    return frame[y1:y2, x1:x2]


def map_crop_to_frame(box_in_crop: Box, roi: Box) -> Box:
    """Map a box in CROP-local pixel coords to FRAME-absolute coords."""
    rx1, ry1, _, _ = roi
    bx1, by1, bx2, by2 = box_in_crop
    return (bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1)


def map_frame_to_crop(box_in_frame: Box, roi: Box) -> Box:
    """Map a box in FRAME-absolute coords to CROP-local coords (roi's origin)."""
    rx1, ry1, _, _ = roi
    bx1, by1, bx2, by2 = box_in_frame
    return (bx1 - rx1, by1 - ry1, bx2 - rx1, by2 - ry1)


# ---------------------------------------------------------------------------
# Kalman constant-velocity box filter: state (cx, cy, w, h, vx, vy)
# ---------------------------------------------------------------------------


class KalmanBoxTracker:
    """Constant-velocity Kalman filter over a box's centre and size.

    State vector ``x = [cx, cy, w, h, vx, vy]`` (design doc §2.1). Position
    (``cx``, ``cy``) evolves under velocity each step; width/height are
    modelled as constant-but-noisy (a small process-noise term lets them
    drift slowly as the ball moves toward/away from camera). Measurement is
    ``z = [cx, cy, w, h]``.

    This is a small hand-rolled KF (no external dependency) — plain numpy,
    CPU-only, works identically on any device since it never touches torch.
    """

    _F = np.array(
        [
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    _H = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
        velocity_process_noise: float = 5.0,
        initial_covariance: float = 10.0,
    ) -> None:
        self._q = process_noise
        self._qv = velocity_process_noise
        self._initial_covariance = initial_covariance
        self._Q = np.diag(
            [process_noise, process_noise, process_noise, process_noise, self._qv, self._qv]
        ).astype(np.float64)
        self._R = np.eye(4, dtype=np.float64) * measurement_noise
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None

    def init(self, box: Box) -> None:
        """(Re)initialise state from an observed box, with zero velocity."""
        cx, cy, w, h = box_to_cxcywh(box)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * self._initial_covariance

    @property
    def initialized(self) -> bool:
        return self.x is not None

    @property
    def velocity(self) -> tuple[float, float]:
        """Current ``(vx, vy)`` estimate in pixels/frame, or ``(0, 0)`` if unset."""
        if self.x is None:
            return (0.0, 0.0)
        return float(self.x[4]), float(self.x[5])

    @property
    def box(self) -> Box:
        """The current state's box, in xyxy absolute pixels."""
        if self.x is None:
            raise RuntimeError("KalmanBoxTracker has no state — call init() first.")
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        return cxcywh_to_box(float(cx), float(cy), float(w), float(h))

    def predict(self) -> Box:
        """Advance the state by one step (constant-velocity) and return the box.

        Must be called after ``init()``.
        """
        if self.x is None:
            raise RuntimeError("KalmanBoxTracker.predict() called before init().")
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        return self.box

    def update(self, box: Box) -> None:
        """Correct the state with an observed box (standard KF update).

        If the filter has no state yet, this behaves like ``init()``.
        """
        if self.x is None:
            self.init(box)
            return
        cx, cy, w, h = box_to_cxcywh(box)
        z = np.array([cx, cy, w, h], dtype=np.float64)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self._H) @ self.P


# ---------------------------------------------------------------------------
# MotionGuidedTracker: the FrameTracker implementation (§2, §4)
# ---------------------------------------------------------------------------


class MotionGuidedTracker:
    """``FrameTracker`` that drives a Kalman-predicted crop through a ``CropRunner``.

    Per design doc §2: predict -> crop -> swappable runner -> refine -> map
    back to frame -> update state; full-frame re-acquire on miss.
    """

    def __init__(
        self,
        runner: CropRunner,
        reacquire_runner: CropRunner | None = None,
        miss_confidence_thresh: float = 0.3,
        margin_scale: float = 2.5,
        velocity_scale: float = 1.5,
        min_roi_size: float = 64.0,
        kalman_kwargs: dict | None = None,
    ) -> None:
        """
        Args:
            runner: The cheap per-step ``CropRunner`` (SOT tracker / ROI-YOLO /
                cropped SAM variant — a benchmark variable per §5).
            reacquire_runner: The full-frame fallback backend used on a miss
                (§2.3), e.g. a ``Sam3VideoLabeller`` in ``CropRunner`` mode or
                YOLO full-frame. Defaults to ``runner`` itself (called on the
                whole frame instead of a crop) if not given.
            miss_confidence_thresh: Detections below this confidence are
                treated as a miss, triggering full-frame re-acquire.
            margin_scale / velocity_scale / min_roi_size: passed to
                ``compute_roi``.
            kalman_kwargs: Extra kwargs forwarded to ``KalmanBoxTracker``.
        """
        self.runner = runner
        self.reacquire_runner = reacquire_runner or runner
        self.miss_confidence_thresh = miss_confidence_thresh
        self.margin_scale = margin_scale
        self.velocity_scale = velocity_scale
        self.min_roi_size = min_roi_size
        self.kalman = KalmanBoxTracker(**(kalman_kwargs or {}))
        self._frame_w = 0
        self._frame_h = 0
        # Provenance tag of the source of the last returned Detection, e.g.
        # the runner's `.name`, or "<name>-reacquire" on full-frame fallback.
        self.last_provenance: str | None = None

    def reset(self, frame: np.ndarray, seed: Box) -> None:
        """Seed the tracker with a known-good box (the user's marked frame)."""
        self._frame_h, self._frame_w = frame.shape[:2]
        self.kalman.init(seed)
        self.runner.warmup()
        if self.reacquire_runner is not self.runner:
            self.reacquire_runner.warmup()
        self.last_provenance = None

    def step(self, frame: np.ndarray) -> Detection | None:
        """Advance one frame: predict -> crop -> detect -> refine, or re-acquire."""
        if not self.kalman.initialized:
            raise RuntimeError("MotionGuidedTracker.step() called before reset().")
        self._frame_h, self._frame_w = frame.shape[:2]

        pred_box = self.kalman.predict()
        roi = compute_roi(
            pred_box,
            self._frame_w,
            self._frame_h,
            velocity=self.kalman.velocity,
            margin_scale=self.margin_scale,
            velocity_scale=self.velocity_scale,
            min_size=self.min_roi_size,
        )
        crop = crop_frame(frame, roi)

        det = None
        if crop.size > 0:
            prior_in_crop = map_frame_to_crop(pred_box, roi)
            det = self.runner.detect(crop, prior_in_crop)

        if det is not None and det.confidence >= self.miss_confidence_thresh:
            box_full = map_crop_to_frame(det.box, roi)
            self.kalman.update(box_full)
            self.last_provenance = self.runner.name
            return Detection(box=box_full, confidence=det.confidence, label=det.label)

        return self._reacquire(frame)

    def _reacquire(self, frame: np.ndarray) -> Detection | None:
        """Full-frame re-acquire fallback (§2.3): relocate the ball, reinit state."""
        det = self.reacquire_runner.detect(frame, None)
        if det is None or det.confidence < self.miss_confidence_thresh:
            self.last_provenance = None
            return None
        self.kalman.init(det.box)
        self.last_provenance = f"{self.reacquire_runner.name}-reacquire"
        return Detection(box=det.box, confidence=det.confidence, label=det.label)
