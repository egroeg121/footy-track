"""Ball tracker over per-frame detector candidates (tracking-by-detection).

Unlike the other modules in this package this does NOT implement the
``BallTracker`` protocol in ``ball_eval.interface``: that protocol hands the
tracker pixels, whereas this one consumes the ball boxes the detector already
produced, so it needs no model, no GPU and runs in microseconds per frame.

Why it exists: the RT-DETR ball detector is precise only at high confidence
(Ball Check verdicts: precision 0.04 at 0.1-0.2, 0.42 at 0.5-0.6, 0.79 at
0.6-0.7, 0.96 at 0.7-0.8, 1.0 at 0.8-0.9) and recalls about 0.6 on held-out
clips. A single confident detection is trustworthy, so it seeds a track; after
that, weaker candidates are accepted only if they sit near where the ball should
be, which recovers recall without letting the 0.1-0.5 false positives in.

Per frame:
  1. Predict the position from the last two accepted detections (constant
     velocity).
  2. Gate: a candidate must be within ``base_radius_px + growth_px * gap`` of the
     prediction, where ``gap`` is frames since the last accepted detection. The
     radius is scaled down by ``confidence / start_conf`` for candidates below
     ``start_conf``, so a weak candidate has to be close and a strong one can be
     further away.
  3. Pick the candidate with the best ``confidence * (1 - distance / radius)``.
  4. No candidate in the gate: the frame is a miss. A candidate of at least
     ``jump_conf`` outside the gate restarts the track there (a real jump, e.g. a
     camera cut, beats a stale prediction). More than ``max_missed`` consecutive
     misses ends the track.
  5. With no live track, the strongest candidate at or above ``start_conf``
     starts one.

When a track re-acquires the ball after a gap, the frames in between are filled
by linear interpolation and marked ``interpolated``. Frames with no track stay
``none``; the tracker never extrapolates into the future.

Boxes are normalised (x, y, w, h) with (x, y) top-left, as everywhere else.
Pixel distances use ``frame_size`` so thresholds mean the same on any resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DETECTED = "detected"
INTERPOLATED = "interpolated"
NONE = "none"


@dataclass(frozen=True)
class BallCandidate:
    """One ball box from the detector (normalised, top-left x/y)."""

    x: float
    y: float
    w: float
    h: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass
class BallPoint:
    """The tracker's answer for one frame."""

    frame_index: int
    source: str = NONE  # detected | interpolated | none
    track_id: int | None = None
    x: float | None = None
    y: float | None = None
    w: float | None = None
    h: float | None = None
    confidence: float | None = None  # None unless detected

    def to_row(self) -> dict:
        """One ``ball_track.jsonl`` row (bbox/center dicts like the sidecars)."""
        has_box = self.x is not None
        return {
            "frame_index": self.frame_index,
            "track_id": self.track_id,
            "source": self.source,
            "bbox": {"x": self.x, "y": self.y, "w": self.w, "h": self.h}
            if has_box
            else None,
            "center": {"x": self.x + self.w / 2, "y": self.y + self.h / 2}
            if has_box
            else None,
            "confidence": self.confidence,
        }


@dataclass
class _Live:
    track_id: int
    last_frame: int
    box: tuple[float, float, float, float]  # x, y, w, h
    vel: tuple[float, float] = (0.0, 0.0)  # normalised centre units per frame


class DetectionBallTracker:
    """Single-ball tracker; call :meth:`update` once per processed frame, in order."""

    def __init__(
        self,
        frame_size: tuple[int, int] = (1920, 1080),
        start_conf: float = 0.6,
        jump_conf: float = 0.8,
        min_conf: float = 0.1,
        base_radius_px: float = 40.0,
        growth_px: float = 25.0,
        max_missed: int = 12,
    ) -> None:
        self._w, self._h = frame_size
        self._start_conf = start_conf
        self._jump_conf = jump_conf
        self._min_conf = min_conf
        self._base_radius_px = base_radius_px
        self._growth_px = growth_px
        self._max_missed = max_missed
        self._next_id = 1
        self._live: _Live | None = None
        self._points: list[BallPoint] = []
        self._gap_idx: list[int] = []  # indices into _points missed since last accept

    def update(self, frame_index: int, candidates: list[BallCandidate]) -> BallPoint:
        cands = [c for c in candidates if c.confidence >= self._min_conf]
        live = self._live
        if live is not None and frame_index - live.last_frame - 1 > self._max_missed:
            self._end_track()
            live = None

        chosen: BallCandidate | None = None
        if live is None:
            chosen = self._strongest(cands, self._start_conf)
            if chosen is not None:
                return self._start_track(frame_index, chosen)
        else:
            chosen = self._best_in_gate(live, frame_index, cands)
            if chosen is not None:
                return self._accept(live, frame_index, chosen)
            jump = self._strongest(cands, self._jump_conf)
            if jump is not None:
                self._end_track()
                return self._start_track(frame_index, jump)

        point = BallPoint(frame_index)
        self._points.append(point)
        if self._live is not None:
            self._gap_idx.append(len(self._points) - 1)
        return point

    def points(self) -> list[BallPoint]:
        """Every frame seen so far, in order, with gaps already interpolated."""
        return list(self._points)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strongest(cands: list[BallCandidate], floor: float) -> BallCandidate | None:
        eligible = [c for c in cands if c.confidence >= floor]
        return max(eligible, key=lambda c: c.confidence, default=None)

    def _best_in_gate(
        self, live: _Live, frame_index: int, cands: list[BallCandidate]
    ) -> BallCandidate | None:
        gap = frame_index - live.last_frame
        bx, by, bw, bh = live.box
        px = bx + bw / 2 + live.vel[0] * gap
        py = by + bh / 2 + live.vel[1] * gap
        radius = self._base_radius_px + self._growth_px * gap
        best, best_score = None, 0.0
        for c in cands:
            cx, cy = c.center
            dist = math.hypot((cx - px) * self._w, (cy - py) * self._h)
            gate = radius * min(1.0, c.confidence / self._start_conf)
            if dist > gate:
                continue
            score = c.confidence * (1.0 - dist / radius)
            if best is None or score > best_score:
                best, best_score = c, score
        return best

    def _start_track(self, frame_index: int, cand: BallCandidate) -> BallPoint:
        self._live = _Live(
            track_id=self._next_id,
            last_frame=frame_index,
            box=(cand.x, cand.y, cand.w, cand.h),
        )
        self._next_id += 1
        return self._record(frame_index, self._live, cand)

    def _accept(self, live: _Live, frame_index: int, cand: BallCandidate) -> BallPoint:
        gap = frame_index - live.last_frame
        ox, oy = live.box[0] + live.box[2] / 2, live.box[1] + live.box[3] / 2
        cx, cy = cand.center
        vx, vy = (cx - ox) / gap, (cy - oy) / gap
        if live.vel == (0.0, 0.0):
            live.vel = (vx, vy)
        else:
            live.vel = ((live.vel[0] + vx) / 2, (live.vel[1] + vy) / 2)
        self._interpolate_gap(live, frame_index, cand)
        live.box = (cand.x, cand.y, cand.w, cand.h)
        live.last_frame = frame_index
        return self._record(frame_index, live, cand)

    def _record(self, frame_index: int, live: _Live, cand: BallCandidate) -> BallPoint:
        point = BallPoint(
            frame_index,
            DETECTED,
            live.track_id,
            cand.x,
            cand.y,
            cand.w,
            cand.h,
            cand.confidence,
        )
        self._points.append(point)
        self._gap_idx = []
        return point

    def _interpolate_gap(
        self, live: _Live, frame_index: int, cand: BallCandidate
    ) -> None:
        span = frame_index - live.last_frame
        end = (cand.x, cand.y, cand.w, cand.h)
        for i in self._gap_idx:
            p = self._points[i]
            t = (p.frame_index - live.last_frame) / span
            p.x, p.y, p.w, p.h = (
                a + (b - a) * t for a, b in zip(live.box, end, strict=True)
            )
            p.source, p.track_id = INTERPOLATED, live.track_id
        self._gap_idx = []

    def _end_track(self) -> None:
        self._live = None
        self._gap_idx = []  # the unbridged frames stay "none"
