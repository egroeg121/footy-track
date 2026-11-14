"""Baselines B0 (per-frame), B1 (offline solver), B2 (+ rescue), B3 (+ anchors)."""

from __future__ import annotations

import numpy as np

from footy_track.ball_offline.metrics import Pred
from footy_track.ball_offline.solver import (
    SolverParams,
    fill_gaps,
    rescue_on_trajectory,
    solve,
)


def b0_per_frame(clip, tau: float) -> Pred:
    """Top-confidence ball candidate per frame if conf >= tau."""
    pred: Pred = {}
    for t in range(clip.n_frames):
        xy, conf = clip.cands(t)
        if len(xy):
            k = int(np.argmax(conf))
            if conf[k] >= tau:
                pred[t] = (float(xy[k, 0]), float(xy[k, 1]))
    return pred


def b1_solver(clip, params: SolverParams, anchors=None):
    nodes = solve(clip, params, anchors)
    return fill_gaps(nodes, params, clip), nodes


def b2_rescue(
    clip, params: SolverParams, anchors=None, gate_px: float = 12.0, min_conf=0.1
):
    pred, nodes = b1_solver(clip, params, anchors)
    return rescue_on_trajectory(pred, nodes, clip, gate_px, min_conf), nodes


def anchors_from_gt(clip, every: int) -> dict[int, tuple[float, float]]:
    """Every ``every``-th labelled GT frame (in frame order) as a hard anchor."""
    frames = sorted(clip.gt)
    return {t: clip.gt[t].xy for t in frames[::every]}


def anchor_interp(clip, anchors, max_gap: int = 50) -> Pred:
    """Control: linear interpolation between consecutive anchors only (no detections)."""
    ts = sorted(anchors)
    pred: Pred = {t: anchors[t] for t in ts}
    for a, b in zip(ts, ts[1:], strict=False):
        if b - a <= max_gap:
            for t in range(a + 1, b):
                f = (t - a) / (b - a)
                pred[t] = (
                    anchors[a][0] + f * (anchors[b][0] - anchors[a][0]),
                    anchors[a][1] + f * (anchors[b][1] - anchors[a][1]),
                )
    return pred
