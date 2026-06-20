"""Kalman-filter ball tracker for reliable broadcast-view ball tracking.

Designed for the football ball: small (10-30px), fast, motion-blurred,
frequently occluded. The standard IoU-based LapTracker fails because IoU
between two 15px boxes separated by even 20px is zero.

Key design choices:
- Centre-distance matching (not IoU) — works even when the ball isn't detected
  every frame.
- Constant-velocity Kalman filter predicts ball position during detection gaps,
  giving the tracker a "where should the ball be now?" estimate to match against
  the next detection.
- Gap-filling: during missed frames, emits interpolated TrackedDetections so
  downstream consumers see a continuous trajectory.
- Singleton by design: there is at most one ball in play, so this tracker keeps
  only one active track at a time. If multiple balls are detected we pick the
  one closest to the predicted position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from footy_track.constants import BALL_TAG, IN_PLAY_BALL_TAG, OUT_OF_PLAY_BALL_TAG
from footy_track.schema import FrameDetections, ObjectDetection
from footy_track.trackers.base import TrackMeta, TrackedDetection

BALL_LABELS = {BALL_TAG, IN_PLAY_BALL_TAG, OUT_OF_PLAY_BALL_TAG}

# --- Kalman filter (numpy-only, no filterpy) ---
# State: [cx, cy, vx, vy] — centre + velocity, all normalised screen coords.
# Measurement: [cx, cy]


def _build_kf(
    cx: float, cy: float, process_noise: float = 1e-4, measurement_noise: float = 1e-3
) -> dict:
    """Initialise Kalman filter state for a new ball track."""
    x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
    # State transition: position += velocity
    F = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    # Measurement matrix: observe cx, cy only
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    Q = np.eye(4, dtype=np.float64) * process_noise
    R = np.eye(2, dtype=np.float64) * measurement_noise
    P = np.eye(4, dtype=np.float64) * 0.1
    return {"x": x, "F": F, "H": H, "Q": Q, "R": R, "P": P}


def _kf_predict(kf: dict) -> np.ndarray:
    """Predict step — advances state by one frame, returns predicted [cx, cy]."""
    kf["x"] = kf["F"] @ kf["x"]
    kf["P"] = kf["F"] @ kf["P"] @ kf["F"].T + kf["Q"]
    return kf["x"][:2].copy()


def _kf_update(kf: dict, cx: float, cy: float) -> None:
    """Update step — incorporate measurement [cx, cy]."""
    z = np.array([cx, cy], dtype=np.float64)
    y = z - kf["H"] @ kf["x"]
    S = kf["H"] @ kf["P"] @ kf["H"].T + kf["R"]
    K = kf["P"] @ kf["H"].T @ np.linalg.inv(S)
    kf["x"] = kf["x"] + K @ y
    kf["P"] = (np.eye(4, dtype=np.float64) - K @ kf["H"]) @ kf["P"]


def _centre(det: ObjectDetection) -> tuple[float, float]:
    return det.x + det.w / 2.0, det.y + det.h / 2.0


@dataclass
class _BallTrack:
    track_id: int
    label: str
    kf: dict
    last_det: ObjectDetection  # last raw detection (used for w/h/conf/model)
    start_frame: int
    start_time: float
    last_frame: int
    last_time: float
    age: int = 0  # frames since last matched detection


@dataclass
class _PendingFrame:
    """Frame buffered during a gap; flushed when the ball is re-detected."""

    frame_index: int
    frame_t: float
    predicted_cx: float
    predicted_cy: float


class KalmanBallTracker:
    """Single-object Kalman-filter tracker for the football.

    Parameters
    ----------
    max_age:
        Frames without a detection before the track is finalised.
    max_match_dist:
        Maximum normalised-coordinate centre distance to accept a detection
        as matching the current ball track (vs spawning a new one).
    gap_fill:
        If True, emit interpolated TrackedDetections for frames where the ball
        was not detected (using Kalman-predicted position).
    process_noise / measurement_noise:
        Kalman filter tuning knobs.
    """

    def __init__(
        self,
        max_age: int = 10,
        max_match_dist: float = 0.15,
        gap_fill: bool = True,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-3,
    ) -> None:
        self._max_age = max_age
        self._max_match_dist = max_match_dist
        self._gap_fill = gap_fill
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise

        self._next_id: int = 1
        self._track: _BallTrack | None = None
        self._finalised: list[TrackMeta] = []
        self._frame_counter: int = 0
        self._pending: list[_PendingFrame] = []  # buffered gap frames

    # ------------------------------------------------------------------
    # Public interface (ObjectTracker protocol)
    # ------------------------------------------------------------------

    def update(self, frame_detections: FrameDetections, frame_t: float) -> list[TrackedDetection]:
        frame_idx = self._frame_counter
        self._frame_counter += 1

        ball_dets = [d for d in frame_detections.detections if d.label in BALL_LABELS]

        if self._track is None:
            if ball_dets:
                self._spawn(ball_dets[0], frame_idx, frame_t)
                return [self._make_tracked(ball_dets[0], frame_idx, frame_t, interpolated=False)]
            return []

        # Predict where the ball should be this frame
        pred = _kf_predict(self._track.kf)
        pred_cx, pred_cy = pred[0], pred[1]

        if ball_dets:
            # Find detection closest to prediction
            best_det, best_dist = self._best_match(ball_dets, pred_cx, pred_cy)

            if best_dist <= self._max_match_dist:
                # Matched — update Kalman, flush any buffered gap frames
                cx, cy = _centre(best_det)
                _kf_update(self._track.kf, cx, cy)

                results: list[TrackedDetection] = []
                if self._gap_fill:
                    results.extend(self._flush_pending())
                self._track.last_det = best_det
                self._track.last_frame = frame_idx
                self._track.last_time = frame_t
                self._track.age = 0
                results.append(self._make_tracked(best_det, frame_idx, frame_t, interpolated=False))
                return results
            else:
                # Detection too far from prediction — age existing track, maybe spawn new
                self._track.age += 1
                if self._track.age > self._max_age:
                    self._finalise_track()
                    self._pending.clear()
                    # Spawn fresh track for this detection
                    self._spawn(best_det, frame_idx, frame_t)
                    return [self._make_tracked(best_det, frame_idx, frame_t, interpolated=False)]
                # Buffer predicted position for potential later gap-fill
                self._buffer_gap(frame_idx, frame_t, pred_cx, pred_cy)
                return []
        else:
            # No detection this frame — age and buffer
            self._track.age += 1
            if self._track.age > self._max_age:
                self._finalise_track()
                self._pending.clear()
                return []
            self._buffer_gap(frame_idx, frame_t, pred_cx, pred_cy)
            return []

    def finalise(self) -> list[TrackMeta]:
        if self._track is not None:
            self._finalised.append(self._to_meta(self._track))
            self._track = None
        self._pending.clear()
        return list(self._finalised)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _spawn(self, det: ObjectDetection, frame_idx: int, frame_t: float) -> None:
        cx, cy = _centre(det)
        self._track = _BallTrack(
            track_id=self._next_id,
            label=det.label,
            kf=_build_kf(cx, cy, self._process_noise, self._measurement_noise),
            last_det=det,
            start_frame=frame_idx,
            start_time=frame_t,
            last_frame=frame_idx,
            last_time=frame_t,
        )
        self._next_id += 1
        self._pending.clear()

    def _best_match(
        self, dets: list[ObjectDetection], pred_cx: float, pred_cy: float
    ) -> tuple[ObjectDetection, float]:
        best_det = dets[0]
        cx, cy = _centre(dets[0])
        best_dist = float(np.hypot(cx - pred_cx, cy - pred_cy))
        for det in dets[1:]:
            cx, cy = _centre(det)
            d = float(np.hypot(cx - pred_cx, cy - pred_cy))
            if d < best_dist:
                best_dist = d
                best_det = det
        return best_det, best_dist

    def _buffer_gap(self, frame_idx: int, frame_t: float, cx: float, cy: float) -> None:
        if self._gap_fill:
            self._pending.append(
                _PendingFrame(
                    frame_index=frame_idx, frame_t=frame_t, predicted_cx=cx, predicted_cy=cy
                )
            )

    def _flush_pending(self) -> list[TrackedDetection]:
        """Emit interpolated detections for buffered gap frames."""
        assert self._track is not None
        results = []
        for pf in self._pending:
            det = self._track.last_det
            w, h = det.w, det.h
            interp = TrackedDetection(
                label=det.label,
                confidence=det.confidence,
                x=max(0.0, pf.predicted_cx - w / 2.0),
                y=max(0.0, pf.predicted_cy - h / 2.0),
                w=w,
                h=h,
                model=det.model,
                track_id=self._track.track_id,
                frame_index=pf.frame_index,
                continuous_time_s=pf.frame_t,
                is_interpolated=True,
            )
            results.append(interp)
        self._pending.clear()
        return results

    def _make_tracked(
        self, det: ObjectDetection, frame_idx: int, frame_t: float, *, interpolated: bool
    ) -> TrackedDetection:
        assert self._track is not None
        return TrackedDetection(
            label=det.label,
            confidence=det.confidence,
            x=det.x,
            y=det.y,
            w=det.w,
            h=det.h,
            model=det.model,
            track_id=self._track.track_id,
            frame_index=frame_idx,
            continuous_time_s=frame_t,
            is_interpolated=interpolated,
        )

    def _finalise_track(self) -> None:
        assert self._track is not None
        self._finalised.append(self._to_meta(self._track))
        self._track = None

    @staticmethod
    def _to_meta(trk: _BallTrack) -> TrackMeta:
        return TrackMeta(
            track_id=trk.track_id,
            label=trk.label,
            start_frame=trk.start_frame,
            end_frame=trk.last_frame,
            start_continuous_time_s=trk.start_time,
            end_continuous_time_s=trk.last_time,
        )
