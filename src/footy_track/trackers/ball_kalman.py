"""Kalman-filter-augmented ball tracker.

Wraps LapTracker for non-ball classes. For ball detections specifically, a
constant-velocity Kalman filter predicts position during detection gaps and
re-associates returning detections against the predicted position rather than
requiring IoU overlap.

State vector: [cx, cy, vx, vy] (normalised centre coordinates + velocity).
Measurement: [cx, cy].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from footy_track.constants import BALL_TAG
from footy_track.schema import FrameDetections, ObjectDetection
from footy_track.trackers.base import TrackMeta, TrackedDetection


# ---------------------------------------------------------------------------
# Kalman filter helpers
# ---------------------------------------------------------------------------

def _make_kalman(cx: float, cy: float, dt: float = 1.0 / 25.0) -> dict:
    """Initialise Kalman filter state for a ball track.

    Uses a constant-velocity model. ``dt`` is used for initial F; it is
    updated on each predict step with the actual elapsed time.
    """
    # State: [cx, cy, vx, vy]
    x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)

    # State transition (updated dynamically with real dt)
    F = np.eye(4, dtype=np.float64)
    F[0, 2] = dt
    F[1, 3] = dt

    # Measurement matrix: we observe cx, cy
    H = np.zeros((2, 4), dtype=np.float64)
    H[0, 0] = 1.0
    H[1, 1] = 1.0

    # Process noise — tuned for football ball kinematics in normalised coords.
    # Higher position noise than velocity noise to allow for rapid direction changes.
    Q = np.diag([1e-4, 1e-4, 5e-3, 5e-3])

    # Measurement noise — detection bbox centres are fairly reliable
    R = np.diag([1e-4, 1e-4])

    # Initial covariance — high uncertainty in velocity
    P = np.diag([1e-3, 1e-3, 1.0, 1.0])

    return {"x": x, "F": F, "H": H, "Q": Q, "R": R, "P": P}


def _kalman_predict(kf: dict, dt: float) -> np.ndarray:
    """Advance state by dt seconds; return predicted [cx, cy]."""
    F = kf["F"].copy()
    F[0, 2] = dt
    F[1, 3] = dt
    kf["F"] = F
    kf["x"] = F @ kf["x"]
    kf["P"] = F @ kf["P"] @ F.T + kf["Q"]
    return kf["H"] @ kf["x"]


def _kalman_update(kf: dict, cx: float, cy: float) -> None:
    """Measurement update with observed centre [cx, cy]."""
    z = np.array([cx, cy], dtype=np.float64)
    H, P, R, x = kf["H"], kf["P"], kf["R"], kf["x"]
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    kf["x"] = x + K @ (z - H @ x)
    kf["P"] = (np.eye(4) - K @ H) @ P


# ---------------------------------------------------------------------------
# Track dataclass
# ---------------------------------------------------------------------------

@dataclass
class _BallTrack:
    track_id: int
    last_det: ObjectDetection
    start_frame: int
    start_time: float
    last_frame: int
    last_time: float
    kf: dict = field(repr=False)
    age: int = 0  # frames since last matched detection
    # accumulated interpolated detections to emit when ball reappears
    gap_frames: list[tuple[int, float, ObjectDetection]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BallKalmanTracker
# ---------------------------------------------------------------------------

class BallKalmanTracker:
    """Kalman-filter tracker for the ball class.

    Non-ball detections are passed through with simple nearest-predicted-box
    assignment (no Kalman). Only a single ball track is maintained at any
    time (the ball is a singleton on the pitch).

    Parameters
    ----------
    max_gap:
        Maximum consecutive frames without a detection before the ball track
        is dropped entirely. Default 5 (configurable via constructor).
    max_dist:
        Maximum normalised centre distance for re-associating a detection with
        the predicted ball position. Default 0.15 (~15 % of frame width).
    fps:
        Assumed frame rate used to compute dt between frames. Default 25.
    """

    def __init__(
        self,
        max_gap: int = 5,
        max_dist: float = 0.15,
        fps: float = 25.0,
    ) -> None:
        self._max_gap = max_gap
        self._max_dist = max_dist
        self._fps = fps
        self._dt = 1.0 / fps

        self._next_id = 1
        self._ball_track: _BallTrack | None = None
        self._finalised: list[TrackMeta] = []
        self._frame_counter = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self, frame_detections: FrameDetections, frame_t: float
    ) -> list[TrackedDetection]:
        """Process one frame; return tracked detections for ball objects.

        Non-ball detections in ``frame_detections`` are ignored here (callers
        should use LapTracker for those separately, or filter before calling).
        """
        frame_idx = self._frame_counter
        self._frame_counter += 1

        ball_dets = [d for d in frame_detections.detections if d.label == BALL_TAG]
        results: list[TrackedDetection] = []

        if not ball_dets:
            results.extend(self._handle_no_detection(frame_idx, frame_t))
        else:
            # Use the highest-confidence ball detection
            best_det = max(ball_dets, key=lambda d: d.confidence)
            results.extend(self._handle_detection(best_det, frame_idx, frame_t))

        return results

    def finalise(self) -> list[TrackMeta]:
        if self._ball_track is not None:
            self._finalised.append(self._to_meta(self._ball_track))
            self._ball_track = None
        return list(self._finalised)

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _handle_no_detection(self, frame_idx: int, frame_t: float) -> list[TrackedDetection]:
        if self._ball_track is None:
            return []

        trk = self._ball_track
        trk.age += 1

        if trk.age > self._max_gap:
            self._finalised.append(self._to_meta(trk))
            self._ball_track = None
            return []

        # Kalman predict-only — produce an interpolated detection
        dt = frame_t - trk.last_time if frame_t > trk.last_time else self._dt
        pred_cx, pred_cy = _kalman_predict(trk.kf, dt)
        pred_cx = float(np.clip(pred_cx, 0.0, 1.0))
        pred_cy = float(np.clip(pred_cy, 0.0, 1.0))

        # Carry through bbox size from last detection
        w, h = trk.last_det.w, trk.last_det.h
        x = max(0.0, pred_cx - w / 2)
        y = max(0.0, pred_cy - h / 2)

        interp_det = ObjectDetection(
            label=BALL_TAG,
            confidence=trk.last_det.confidence,
            x=x,
            y=y,
            w=w,
            h=h,
            model=trk.last_det.model,
        )
        trk.last_time = frame_t
        trk.gap_frames.append((frame_idx, frame_t, interp_det))

        return [
            TrackedDetection(
                **interp_det.model_dump(),
                track_id=trk.track_id,
                frame_index=frame_idx,
                continuous_time_s=frame_t,
                is_interpolated=True,
            )
        ]

    def _handle_detection(
        self, det: ObjectDetection, frame_idx: int, frame_t: float
    ) -> list[TrackedDetection]:
        cx = det.x + det.w / 2
        cy = det.y + det.h / 2

        if self._ball_track is None:
            # Spawn new track
            kf = _make_kalman(cx, cy, dt=self._dt)
            _kalman_update(kf, cx, cy)
            self._ball_track = _BallTrack(
                track_id=self._next_id,
                last_det=det,
                start_frame=frame_idx,
                start_time=frame_t,
                last_frame=frame_idx,
                last_time=frame_t,
                kf=kf,
            )
            self._next_id += 1
        else:
            trk = self._ball_track
            # Check if detection is close to predicted position
            dt = frame_t - trk.last_time if frame_t > trk.last_time else self._dt
            pred = _kalman_predict(trk.kf, dt)
            dist = float(np.linalg.norm(pred - np.array([cx, cy])))

            if dist <= self._max_dist:
                # Re-associate
                _kalman_update(trk.kf, cx, cy)
                trk.last_det = det
                trk.last_frame = frame_idx
                trk.last_time = frame_t
                trk.age = 0
                trk.gap_frames.clear()
            else:
                # Too far — finalise old track, spawn new one
                self._finalised.append(self._to_meta(trk))
                kf = _make_kalman(cx, cy, dt=self._dt)
                _kalman_update(kf, cx, cy)
                self._ball_track = _BallTrack(
                    track_id=self._next_id,
                    last_det=det,
                    start_frame=frame_idx,
                    start_time=frame_t,
                    last_frame=frame_idx,
                    last_time=frame_t,
                    kf=kf,
                )
                self._next_id += 1

        return [
            TrackedDetection(
                **det.model_dump(),
                track_id=self._ball_track.track_id,
                frame_index=frame_idx,
                continuous_time_s=frame_t,
                is_interpolated=False,
            )
        ]

    @staticmethod
    def _to_meta(trk: _BallTrack) -> TrackMeta:
        return TrackMeta(
            track_id=trk.track_id,
            label=BALL_TAG,
            start_frame=trk.start_frame,
            end_frame=trk.last_frame,
            start_continuous_time_s=trk.start_time,
            end_continuous_time_s=trk.last_time,
        )
