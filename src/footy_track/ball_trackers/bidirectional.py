"""Forward + backward ball tracking over a whole clip (offline).

The online ``DetectionBallTracker`` can only start a track on a confident
candidate, so when the first confident candidate is a false positive (a crowd
blob, a head) it follows weak candidates near it for seconds before the real
ball is found. Run in reverse time the same tracker starts from the *other* end:
the real ball is usually found first there, and its track covers the early
frames the forward pass got wrong.

``BidirectionalBallTracker`` buffers the candidates (``update`` per frame, like
the online tracker), then on :meth:`points` runs ``DetectionBallTracker`` forward
and on the time-reversed sequence and merges the two answers per frame.

Merge rule, per frame:

* neither pass has a ball: ``none``.
* only one pass has a ball: take it.
* both have one and the centres are within ``agree_px`` (60 px): they agree. Keep the
  detected one (the forward one if both are, or both are interpolated).
* they disagree: keep the pass whose *track* containing this frame has more
  support. A track is the run of consecutive frames with one ``track_id`` in that
  pass, and its support is the sum of the confidences of its detected frames
  (``support="conf_sum"``) or the number of its detections at or above
  ``start_conf`` (``support="strong_count"``). Ties go to the forward pass.

Gap filling falls out of the same rule: with ``fill="any"`` an interpolated frame
from either pass is a prediction like any other, so a gap the forward pass left
empty is filled when the backward pass bridged it (and the other way round); with
``fill="agree"`` (the default) a bridge that only one pass believes in is dropped
and only detections stand alone; bridges both passes made are kept. A frame is never filled by both disagreeing.

The output is the usual ``BallPoint`` list; track ids are renumbered so that a
new id starts whenever the winning pass or its track changes.
"""

from __future__ import annotations

import math

from footy_track.ball_trackers.detection_tracker import (
    DETECTED,
    NONE,
    BallCandidate,
    BallPoint,
    DetectionBallTracker,
)

SUPPORTS = ("conf_sum", "strong_count")
FILLS = ("any", "agree")


def run_online(
    frames: list[tuple[int, list[BallCandidate]]], **tracker_kwargs
) -> list[BallPoint]:
    """Run the online tracker over ``frames`` (frame_index, candidates), in order."""
    tracker = DetectionBallTracker(**tracker_kwargs)
    for index, cands in frames:
        tracker.update(index, cands)
    return tracker.points()


def run_reversed(
    frames: list[tuple[int, list[BallCandidate]]], **tracker_kwargs
) -> list[BallPoint]:
    """Run the online tracker backwards in time; points come back in forward order.

    Frame indices are negated so they increase along the reversed sequence, which
    is what the online tracker's gap arithmetic assumes.
    """
    tracker = DetectionBallTracker(**tracker_kwargs)
    for index, cands in reversed(frames):
        tracker.update(-index, cands)
    points = tracker.points()
    for p in points:
        p.frame_index = -p.frame_index
    points.reverse()
    return points


def track_support(
    points: list[BallPoint], kind: str, start_conf: float
) -> dict[int, float]:
    """Support of the track containing each frame (``{frame_index: support}``)."""
    support: dict[int, float] = {}
    run: list[BallPoint] = []

    def close() -> None:
        if not run:
            return
        if kind == "strong_count":
            s = sum(
                1 for p in run if p.source == DETECTED and p.confidence >= start_conf
            )
        else:
            s = sum(p.confidence for p in run if p.source == DETECTED)
        for p in run:
            support[p.frame_index] = float(s)
        run.clear()

    prev_id = None
    for p in points:
        if p.track_id is None or p.source == NONE or p.track_id != prev_id:
            close()
        if p.track_id is not None and p.source != NONE:
            run.append(p)
        prev_id = p.track_id if p.source != NONE else None
    close()
    return support


def _centre(p: BallPoint) -> tuple[float, float]:
    return p.x + p.w / 2, p.y + p.h / 2


def merge_passes(
    fwd: list[BallPoint],
    bwd: list[BallPoint],
    frame_size: tuple[int, int],
    *,
    agree_px: float = 60.0,
    support: str = "conf_sum",
    start_conf: float = 0.6,
    fill: str = "agree",
) -> list[BallPoint]:
    """Merge a forward and a backward pass (same frames, same order)."""
    if support not in SUPPORTS:
        raise ValueError(f"support must be one of {SUPPORTS}")
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {FILLS}")
    sf = track_support(fwd, support, start_conf)
    sb = track_support(bwd, support, start_conf)
    w, h = frame_size
    out: list[BallPoint] = []
    last_key: tuple[str, int] | None = None
    new_id = 0
    for f, b in zip(fwd, bwd, strict=True):
        f_has, b_has = f.source != NONE, b.source != NONE
        if not f_has and not b_has:
            out.append(BallPoint(f.frame_index))
            last_key = None
            continue
        if f_has and b_has:
            cf, cb = _centre(f), _centre(b)
            if math.hypot((cf[0] - cb[0]) * w, (cf[1] - cb[1]) * h) <= agree_px:
                pick = b if (b.source == DETECTED and f.source != DETECTED) else f
                side = "f" if pick is f else "b"
            else:
                side = (
                    "f"
                    if sf.get(f.frame_index, 0.0) >= sb.get(b.frame_index, 0.0)
                    else "b"
                )
                pick = f if side == "f" else b
        else:
            side, pick = ("f", f) if f_has else ("b", b)
            if fill == "agree" and pick.source != DETECTED:
                # a bridge only one pass believes in: leave the frame empty
                out.append(BallPoint(f.frame_index))
                last_key = None
                continue
        key = (side, pick.track_id)
        if key != last_key:
            new_id += 1
            last_key = key
        out.append(
            BallPoint(
                f.frame_index, pick.source, new_id, pick.x, pick.y, pick.w, pick.h,
                pick.confidence,
            )
        )  # fmt: skip
    return out


class BidirectionalBallTracker:
    """Offline drop-in for ``DetectionBallTracker``: ``update`` per frame, ``points`` at the end."""

    def __init__(
        self,
        frame_size: tuple[int, int] = (1920, 1080),
        *,
        agree_px: float = 60.0,
        support: str = "conf_sum",
        fill: str = "agree",
        **tracker_kwargs,
    ) -> None:
        # Tuned on the seen GT clips (docs/ball-tracking-eval.md): the online
        # defaults (max_missed 12, growth 25) bridge too much when run both ways.
        tracker_kwargs.setdefault("max_missed", 6)
        tracker_kwargs.setdefault("growth_px", 15.0)
        self._frame_size = frame_size
        self._agree_px = agree_px
        self._support = support
        self._fill = fill
        self._kwargs = dict(frame_size=frame_size, **tracker_kwargs)
        self._start_conf = tracker_kwargs.get("start_conf", 0.6)
        self._frames: list[tuple[int, list[BallCandidate]]] = []

    def update(self, frame_index: int, candidates: list[BallCandidate]) -> BallPoint:
        self._frames.append((frame_index, list(candidates)))
        return BallPoint(frame_index)  # the real answer needs the future: see points()

    def points(self) -> list[BallPoint]:
        if not self._frames:
            return []
        fwd = run_online(self._frames, **self._kwargs)
        bwd = run_reversed(self._frames, **self._kwargs)
        return merge_passes(
            fwd, bwd, self._frame_size,
            agree_px=self._agree_px, support=self._support, start_conf=self._start_conf,
            fill=self._fill,
        )  # fmt: skip


__all__ = ["BidirectionalBallTracker", "merge_passes", "run_online", "run_reversed"]
