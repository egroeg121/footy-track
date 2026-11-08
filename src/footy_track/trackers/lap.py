"""Hungarian-assignment tracker backed by lap.lapjv.

Re-ID is out of scope for v1 — see player_tracking_format.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from footy_track.schema import FrameDetections, ObjectDetection
from footy_track.trackers.base import TrackedDetection, TrackMeta


def _iou(a: ObjectDetection, b: ObjectDetection) -> float:
    ax2 = a.x + a.w
    ay2 = a.y + a.h
    bx2 = b.x + b.w
    by2 = b.y + b.h
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    label: str
    last_det: ObjectDetection
    start_frame: int
    start_time: float
    last_frame: int
    last_time: float
    age: int = 0  # frames since last matched detection


class LapTracker:
    """IoU Hungarian-assignment tracker.

    Parameters
    ----------
    max_age:
        Frames a track survives without a matching detection before being finalised.
    iou_threshold:
        Minimum IoU for a detection to be considered a match for an existing track.
    """

    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3) -> None:
        self._max_age = max_age
        self._iou_threshold = iou_threshold
        self._next_id = 1
        self._active: list[_Track] = []
        self._finalised: list[TrackMeta] = []
        self._frame_counter = 0

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def update(
        self, frame_detections: FrameDetections, frame_t: float
    ) -> list[TrackedDetection]:
        import lap  # noqa: PLC0415 — defer import; lap is an optional heavy dep

        dets = frame_detections.detections
        frame_idx = self._frame_counter
        self._frame_counter += 1

        if not self._active:
            results = self._spawn_new(dets, frame_idx, frame_t)
            return results

        if not dets:
            self._age_out(frame_idx)
            return []

        n_tracks = len(self._active)
        n_dets = len(dets)

        # Build cost matrix: 1 - IoU, shape (n_tracks, n_dets)
        cost = np.ones((n_tracks, n_dets), dtype=np.float64)
        for i, trk in enumerate(self._active):
            for j, det in enumerate(dets):
                iou_val = _iou(trk.last_det, det)
                cost[i, j] = 1.0 - iou_val

        # Pad to square for lapjv
        size = max(n_tracks, n_dets)
        padded = np.full((size, size), fill_value=1.0, dtype=np.float64)
        padded[:n_tracks, :n_dets] = cost

        _, row_ind, col_ind = lap.lapjv(
            padded, extend_cost=True, cost_limit=1.0 - self._iou_threshold
        )

        matched_track_idx: set[int] = set()
        matched_det_idx: set[int] = set()
        results: list[TrackedDetection] = []

        for trk_i, det_j in enumerate(row_ind):
            if trk_i >= n_tracks or det_j < 0 or det_j >= n_dets:
                continue
            if cost[trk_i, det_j] > 1.0 - self._iou_threshold:
                continue
            trk = self._active[trk_i]
            det = dets[det_j]
            trk.last_det = det
            trk.last_frame = frame_idx
            trk.last_time = frame_t
            trk.age = 0
            matched_track_idx.add(trk_i)
            matched_det_idx.add(det_j)
            results.append(self._make_tracked(det, trk.track_id, frame_idx, frame_t))

        # Age unmatched tracks
        for trk_i, trk in enumerate(self._active):
            if trk_i not in matched_track_idx:
                trk.age += 1

        # Spawn new tracks for unmatched detections
        for det_j, det in enumerate(dets):
            if det_j not in matched_det_idx:
                trk = self._new_track(det, frame_idx, frame_t)
                results.append(
                    self._make_tracked(det, trk.track_id, frame_idx, frame_t)
                )

        self._age_out(frame_idx)
        return results

    def finalise(self) -> list[TrackMeta]:
        for trk in self._active:
            self._finalised.append(self._to_meta(trk))
        self._active = []
        return list(self._finalised)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _spawn_new(
        self, dets: list[ObjectDetection], frame_idx: int, frame_t: float
    ) -> list[TrackedDetection]:
        results = []
        for det in dets:
            trk = self._new_track(det, frame_idx, frame_t)
            results.append(self._make_tracked(det, trk.track_id, frame_idx, frame_t))
        return results

    def _new_track(
        self, det: ObjectDetection, frame_idx: int, frame_t: float
    ) -> _Track:
        trk = _Track(
            track_id=self._next_id,
            label=det.label,
            last_det=det,
            start_frame=frame_idx,
            start_time=frame_t,
            last_frame=frame_idx,
            last_time=frame_t,
        )
        self._next_id += 1
        self._active.append(trk)
        return trk

    def _age_out(self, _frame_idx: int) -> None:
        survivors: list[_Track] = []
        for trk in self._active:
            if trk.age > self._max_age:
                self._finalised.append(self._to_meta(trk))
            else:
                survivors.append(trk)
        self._active = survivors

    @staticmethod
    def _make_tracked(
        det: ObjectDetection, track_id: int, frame_idx: int, frame_t: float
    ) -> TrackedDetection:
        return TrackedDetection(
            label=det.label,
            confidence=det.confidence,
            x=det.x,
            y=det.y,
            w=det.w,
            h=det.h,
            model=det.model,
            track_id=track_id,
            frame_index=frame_idx,
            continuous_time_s=frame_t,
        )

    @staticmethod
    def _to_meta(trk: _Track) -> TrackMeta:
        return TrackMeta(
            track_id=trk.track_id,
            label=trk.label,
            start_frame=trk.start_frame,
            end_frame=trk.last_frame,
            start_continuous_time_s=trk.start_time,
            end_continuous_time_s=trk.last_time,
        )
