"""Stage 3 helper: the production bidirectional ball tracker over a clip's stride-sampled RT-DETR candidates.

Which tracker: ``footy_track.ball_trackers.bidirectional.BidirectionalBallTracker`` (``make_ball_tracker("bidirectional", ...)``),
the one the autolabel loop uses (``scripts/claude_verify_v2.py``, ``autolabel/verify_v2.py``). It runs the online
``DetectionBallTracker`` forward and on the time-reversed sequence and merges per frame with ``fill="agree"`` (a bridge only
one pass believes in is dropped). Not the Viterbi tracker (``ball_trackers/viterbi.py``) and not the experiment's "B1" solver
(``exp/temporal/pseudo``), which exists only for 35 + 69 clips and is not in ``src``.

This module re-runs the same three steps (``run_online``, ``run_reversed``, ``merge_passes``) instead of calling
``BidirectionalBallTracker.points()`` so it can also report whether the two passes AGREED at a frame, which the tracker
does not expose. The merged points are identical to ``points()`` (a test pins that).

Stride: the tracker's gap arithmetic is in frame indices (``max_missed`` frames, ``growth_px`` per frame of gap), tuned at
stride 1. Candidates are fed with their REAL frame indices and ``max_missed`` is raised to allow two consecutive missed
samples (``3 * stride - 1`` frames); nothing else is retuned. ``growth_px`` 15 px per frame of gap gives a 115 px gate (at
1080p) between consecutive samples, on top of the constant-velocity prediction.

Outputs per sample: ``state`` (0 none, 1 interpolated, 2 detected), ``track_id``, the box, the detection confidence,
``agree`` (both passes had a ball within ``agree_px`` of each other), ``support`` (summed confidence of the detected samples of
the merged track) and ``run_len`` (samples of that track). Interpolated samples are the ``gap_frame`` nodes of the pool
(tracker interpolation with no detection); ``gap_plausibility`` scores them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from footy_track.ball_mining.candidates import ClipCandidates
from footy_track.ball_trackers.bidirectional import (
    merge_passes,
    run_online,
    run_reversed,
    track_support,
)
from footy_track.ball_trackers.detection_tracker import (
    DETECTED,
    INTERPOLATED,
    BallCandidate,
)

NONE_, INTERP, DETECT = 0, 1, 2
AGREE_PX = 60.0  # BidirectionalBallTracker default
START_CONF = (
    0.6  # DetectionBallTracker default (also the strong-count threshold of the merge)
)


@dataclass
class TrackResult:
    frames: np.ndarray  # int32 [n]
    state: np.ndarray  # int8 [n]
    track_id: np.ndarray  # int64 [n], -1 where none
    box: np.ndarray  # float32 [n, 4] x, y, w, h normalised top-left (nan where none)
    conf: np.ndarray  # float32 [n] detection confidence (nan unless detected)
    agree: np.ndarray  # bool [n]
    support: np.ndarray  # float32 [n]
    run_len: np.ndarray  # int32 [n]
    gap_plaus: (
        np.ndarray
    )  # float32 [n] plausibility of an interpolated sample (nan elsewhere)

    def index_of(self, frame: int) -> int | None:
        i = int(np.searchsorted(self.frames, frame))
        return i if i < len(self.frames) and self.frames[i] == frame else None


def tracker_kwargs(stride: int) -> dict:
    """The bidirectional tracker's defaults plus ``max_missed`` widened for a sample every ``stride`` frames."""
    return {"max_missed": 3 * stride - 1}


def gap_plausibility(left_conf: float, right_conf: float, run: int) -> float:
    """Heuristic P(the ball is where the tracker interpolated) for ``run`` consecutive interpolated samples bridged by two
    detections of confidence ``left_conf`` and ``right_conf``: the geometric mean of the two anchor confidences, discounted
    0.85 per extra sample of the gap. Not calibrated; it orders gaps for the miner's M bucket (longest and most plausible
    first) and is never used as a detection score."""
    c = math.sqrt(max(left_conf, 0.0) * max(right_conf, 0.0))
    return float(min(1.0, c * 0.85 ** max(run - 1, 0)))


def run_tracker(  # noqa: PLR0912, PLR0915
    cand: ClipCandidates, *, stride: int | None = None, min_conf: float = 0.1
) -> TrackResult:
    stride = stride or cand.stride
    size = (cand.width or 1920, cand.height or 1080)
    frames = [int(f) for f in cand.frames]
    seq = [
        (f, [BallCandidate(*map(float, r[:5])) for r in cand.balls(f) if r[4] >= min_conf])
        for f in frames
    ]  # fmt: skip
    n = len(frames)
    if n == 0:
        z = np.zeros(0)
        return TrackResult(
            z.astype(np.int32),
            z.astype(np.int8),
            z.astype(np.int64),
            np.zeros((0, 4), np.float32),
            z.astype(np.float32),
            z.astype(bool),
            z.astype(np.float32),
            z.astype(np.int32),
            z.astype(np.float32),
        )
    kw = {"frame_size": size, **tracker_kwargs(stride)}
    fwd = run_online(seq, **kw)
    bwd = run_reversed(seq, **kw)
    merged = merge_passes(
        fwd,
        bwd,
        size,
        agree_px=AGREE_PX,
        support="conf_sum",
        start_conf=START_CONF,
        fill="agree",
    )
    support = track_support(merged, "conf_sum", START_CONF)

    state = np.zeros(n, np.int8)
    tid = np.full(n, -1, np.int64)
    box = np.full((n, 4), np.nan, np.float32)
    conf = np.full(n, np.nan, np.float32)
    agree = np.zeros(n, bool)
    sup = np.zeros(n, np.float32)
    for i, (p, f, b) in enumerate(zip(merged, fwd, bwd, strict=True)):
        if p.source == DETECTED:
            state[i] = DETECT
            conf[i] = p.confidence
        elif p.source == INTERPOLATED:
            state[i] = INTERP
        if p.x is not None:
            tid[i] = p.track_id if p.track_id is not None else -1
            box[i] = (p.x, p.y, p.w, p.h)
            sup[i] = support.get(p.frame_index, 0.0)
            if f.x is not None and b.x is not None:
                d = math.hypot(
                    (f.x + f.w / 2 - b.x - b.w / 2) * size[0],
                    (f.y + f.h / 2 - b.y - b.h / 2) * size[1],
                )
                agree[i] = d <= AGREE_PX
    run_len = np.zeros(n, np.int32)
    i = 0
    while i < n:
        if tid[i] < 0:
            i += 1
            continue
        j = i
        while j + 1 < n and tid[j + 1] == tid[i]:
            j += 1
        run_len[i : j + 1] = j - i + 1
        i = j + 1
    plaus = np.full(n, np.nan, np.float32)
    i = 0
    while (
        i < n
    ):  # each run of interpolated samples sits between two detections of the same track
        if state[i] != INTERP:
            i += 1
            continue
        j = i
        while j + 1 < n and state[j + 1] == INTERP and tid[j + 1] == tid[i]:
            j += 1
        left = conf[i - 1] if i > 0 and state[i - 1] == DETECT else np.nan
        right = conf[j + 1] if j + 1 < n and state[j + 1] == DETECT else np.nan
        if np.isnan(left) or np.isnan(
            right
        ):  # cannot happen for merge output, but stay safe: use what exists
            anchors = [c for c in (left, right) if not np.isnan(c)]
            left = right = float(anchors[0]) if anchors else 0.0
        plaus[i : j + 1] = gap_plausibility(float(left), float(right), j - i + 1)
        i = j + 1
    return TrackResult(
        np.asarray(frames, np.int32), state, tid, box, conf, agree, sup, run_len, plaus
    )
