# ruff: noqa: PLR0912, PLR0915, PLR0124, PLR0917, PLW2901, C408
"""Non-causal global trajectory solver over per-frame ball candidates.

Model.  Every frame either *emits* one of its candidates or is *missing*
(occluded / undetected / ball out of view).  A solution is a chain of accepted
(frame, candidate) nodes; the frames between two consecutive nodes are the
missing state.  Maximise::

    sum_i  w(conf_i)                                  # emission
  + sum_(i->j linked) [ kappa' - lam_pos * dn^2 - lam_vel * dv^2 ]   # motion
  - mu * (skipped frames inside a link)

where ``w = logit(p(conf)) - bias`` and ``p`` is the Ball-Check calibrated
precision of a candidate at that confidence, ``dn = |dp| / (vmax * gap + r0)``
is the displacement scaled by the gap length, and ``dv`` is the deviation from
the incoming velocity of the best path into the previous node (an approximate
second-order term: the velocity is carried on the best-so-far path rather than
making the state a candidate pair).  Two consecutive chain nodes may also be
*unlinked* (a break, score 0), so a lone confident detection is accepted on its
own and a link is only taken when motion consistency pays for it.  Links never
span more than ``max_gap`` frames or a detected shot cut.

``kappa'`` is ``kappa`` scaled down for links shorter than ``static_px``:
a static distractor (penalty spot, logo, boot lying still) would otherwise earn
the motion-consistency reward for free on every frame.

Because the DP is run over the whole clip and the chain is read out by
backtracking from the global optimum, every output frame benefits from both past
and future evidence (forward-backward in the Viterbi sense).

``solve`` returns the accepted nodes; ``fill_gaps`` interpolates between
linked, confident nodes.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np

# Ball Check precision of a machine candidate by confidence (assumed
# piecewise-linear between these knots; only the band values are measured:
# <=0.42 below 0.6, 0.79 at 0.6-0.7, >=0.96 above 0.7).
CALIB_KNOTS_CONF = (0.0, 0.10, 0.25, 0.42, 0.60, 0.65, 0.70, 0.85, 1.0)
CALIB_KNOTS_P = (0.02, 0.08, 0.25, 0.50, 0.72, 0.80, 0.93, 0.97, 0.99)

BIG = 1e6


def calibrated_p(conf: np.ndarray) -> np.ndarray:
    return np.interp(conf, CALIB_KNOTS_CONF, CALIB_KNOTS_P)


@dataclasses.dataclass
class SolverParams:
    min_conf: float = 0.30  # candidates below this are not considered at all
    bias: float = 0.0  # logit offset: lone candidate accepted when p > sigmoid(bias)
    emit_scale: float = 1.0
    kappa: float = 1.5  # reward for a motion-consistent link
    lam_pos: float = 2.0
    lam_vel: float = 0.0  # 0 disables the velocity term
    vmax: float = 40.0  # px / frame
    r0: float = 12.0  # px slack (detector jitter)
    dmax: float = 2.0  # prune links with dn above this
    mu: float = 0.02  # per skipped frame
    static_px: float = 3.0  # links shorter than this earn a proportionally reduced kappa (static FPs); 0 = off
    max_gap: int = 25
    camera: bool = False  # compute motion in camera-stabilised coordinates
    # gap filling
    fill_gap: int = 25
    fill_min_p: float = 0.5  # both endpoints need calibrated p >= this
    fill_mode: str = "linear"  # "linear" | "kalman" | "none"

    def replace(self, **kw) -> SolverParams:
        return dataclasses.replace(self, **kw)


@dataclasses.dataclass
class Node:
    t: int
    xy: tuple[float, float]  # pixels
    conf: float
    linked: bool  # linked to the previous node in the chain
    anchor: bool = False


def _gather(clip, params: SolverParams, anchors: dict[int, tuple[float, float]] | None):
    """Flat node arrays after pruning (+ hard anchors replacing frame candidates)."""
    anchors = anchors or {}
    ts, xs, ys, cs, an = [], [], [], [], []
    for t in range(clip.n_frames):
        if t in anchors:
            ts.append(t)
            xs.append(anchors[t][0])
            ys.append(anchors[t][1])
            cs.append(1.0)
            an.append(True)
            continue
        xy, conf = clip.cands(t)
        keep = conf >= params.min_conf
        for (x, y), c in zip(xy[keep], conf[keep], strict=True):
            ts.append(t)
            xs.append(x)
            ys.append(y)
            cs.append(c)
            an.append(False)
    return (
        np.array(ts, dtype=int),
        np.array(xs, dtype=float),
        np.array(ys, dtype=float),
        np.array(cs, dtype=float),
        np.array(an, dtype=bool),
    )


def solve(
    clip,
    params: SolverParams,
    anchors: dict[int, tuple[float, float]] | None = None,
) -> list[Node]:
    """Viterbi over candidates + missing state.  ``anchors`` are hard GT points."""
    ts, xs, ys, cs, an = _gather(clip, params, anchors)
    n = len(ts)
    if n == 0:
        return []
    # camera-stabilised coordinates for the motion model
    if params.camera:
        cum = np.cumsum(clip.cam_shift, axis=0)
        wx, wy = xs - cum[ts, 0], ys - cum[ts, 1]
    else:
        wx, wy = xs, ys
    p = calibrated_p(cs)
    p = np.clip(p, 1e-4, 1 - 1e-4)
    w = params.emit_scale * (np.log(p / (1 - p)) - params.bias)
    w[an] = BIG  # anchors must be on the path
    # cut markers: cumulative cut count, links may not cross a cut
    cutcum = np.cumsum(clip.cam_cut) if params.camera else np.zeros(clip.n_frames, int)

    # per-frame node ranges
    fptr = np.searchsorted(ts, np.arange(clip.n_frames + 1))
    B = np.full(n, -np.inf)
    pred = np.full(n, -1, dtype=np.int64)
    linked = np.zeros(n, dtype=bool)
    vel = np.full((n, 2), np.nan)
    # prefix best (value, index) over all nodes of earlier frames; index -1 = empty chain
    pref_val, pref_idx = 0.0, -1
    G = params.max_gap
    use_vel = params.lam_vel > 0
    for t in range(clip.n_frames):
        s, e = fptr[t], fptr[t + 1]
        if s == e:
            continue
        # break option: best of everything before this frame
        lo = fptr[max(0, t - G)]
        cand_j = np.arange(lo, s)
        for i in range(s, e):
            best_v, best_j, best_link = pref_val, pref_idx, False
            if len(cand_j):
                gap = (t - ts[cand_j]).astype(float)
                ok = cutcum[t] == cutcum[ts[cand_j]]
                dx = wx[i] - wx[cand_j]
                dy = wy[i] - wy[cand_j]
                d = np.hypot(dx, dy)
                dn = d / (params.vmax * gap + params.r0)
                ok &= dn <= params.dmax
                if np.any(ok):
                    if params.static_px > 0:
                        kap = params.kappa * np.minimum(1.0, d / params.static_px)
                    else:
                        kap = params.kappa
                    sc = B[cand_j] + kap - params.lam_pos * dn**2
                    sc = sc - params.mu * (gap - 1)
                    if use_vel:
                        vk = vel[cand_j]
                        has = ~np.isnan(vk[:, 0])
                        dvx = dx / gap - np.where(has, vk[:, 0], dx / gap)
                        dvy = dy / gap - np.where(has, vk[:, 1], dy / gap)
                        dvn = np.hypot(dvx, dvy) / (0.25 * params.vmax)
                        sc = sc - params.lam_vel * dvn**2
                    sc = np.where(ok, sc, -np.inf)
                    k = int(np.argmax(sc))
                    if sc[k] > best_v:
                        best_v, best_j, best_link = float(sc[k]), int(cand_j[k]), True
            if anchors is not None and an[i]:
                pass
            B[i] = best_v + w[i]
            pred[i] = best_j
            linked[i] = best_link
            if best_link:
                j = best_j
                g = t - ts[j]
                vel[i] = ((wx[i] - wx[j]) / g, (wy[i] - wy[j]) / g)
        # update prefix with this frame's nodes
        k = s + int(np.argmax(B[s:e]))
        if B[k] > pref_val:
            pref_val, pref_idx = float(B[k]), k
    if pref_idx < 0:
        return []
    chain = []
    i = pref_idx
    while i >= 0:
        chain.append(i)
        i = pred[i]
    chain.reverse()
    # a break to ``pred`` still means ``pred`` is in the chain (we extended its chain)
    return [
        Node(
            t=int(ts[i]),
            xy=(float(xs[i]), float(ys[i])),
            conf=float(cs[i]),
            linked=bool(linked[i]),
            anchor=bool(an[i]),
        )
        for i in chain
    ]


def kalman_rts(
    times: Sequence[int],
    pts: np.ndarray,
    t_query: np.ndarray,
    q: float = 1.0,
    r: float = 3.0,
) -> np.ndarray:
    """Constant-velocity Kalman filter + RTS smoother through ``pts`` observed at
    ``times``; returns the smoothed positions at ``t_query`` (all integer frames
    within the observed span).  ``q`` is accel. std (px/frame^2), ``r`` meas. std (px)."""
    t0, t1 = int(times[0]), int(times[-1])
    n = t1 - t0 + 1
    obs = {int(t) - t0: np.asarray(p) for t, p in zip(times, pts, strict=True)}
    F = np.array([[1, 1], [0, 1]], float)
    Q = q * q * np.array([[0.25, 0.5], [0.5, 1.0]])
    R = r * r
    out = np.zeros((n, 2))
    for ax in range(2):
        x = np.zeros((n, 2))
        P = np.zeros((n, 2, 2))
        xp = np.zeros((n, 2))
        Pp = np.zeros((n, 2, 2))
        xc = np.array([obs[0][ax], 0.0])
        Pc = np.diag([R, 100.0])
        for k in range(n):
            if k > 0:
                xc = F @ xc
                Pc = F @ Pc @ F.T + Q
            xp[k], Pp[k] = xc, Pc
            if k in obs:
                S = Pc[0, 0] + R
                K = Pc[:, 0] / S
                xc = xc + K * (obs[k][ax] - xc[0])
                Pc = Pc - np.outer(K, Pc[0])
            x[k], P[k] = xc, Pc
        xs_ = x.copy()
        for k in range(n - 2, -1, -1):
            C = P[k] @ F.T @ np.linalg.inv(Pp[k + 1])
            xs_[k] = x[k] + C @ (xs_[k + 1] - xp[k + 1])
        out[:, ax] = xs_[:, 0]
    return out[np.asarray(t_query, dtype=int) - t0]


def fill_gaps(
    nodes: list[Node],
    params: SolverParams,
    clip=None,
) -> dict[int, tuple[float, float]]:
    """Per-frame predictions: accepted nodes plus interpolation across short gaps
    between *confident* neighbours.  Gaps that are longer than ``fill_gap``, or
    whose endpoints are not confident, are left empty."""
    pred: dict[int, tuple[float, float]] = {n.t: n.xy for n in nodes}
    if params.fill_mode == "none" or len(nodes) < 2:
        return pred
    cum = (
        np.cumsum(clip.cam_shift, axis=0)
        if (params.camera and clip is not None)
        else None
    )
    pe = calibrated_p(np.array([n.conf for n in nodes]))
    # group maximal runs of nodes joined by fillable gaps (for kalman smoothing)
    runs: list[list[int]] = [[0]]
    for k in range(1, len(nodes)):
        a, b = nodes[k - 1], nodes[k]
        gap = b.t - a.t
        fillable = (
            gap > 1
            and gap <= params.fill_gap
            and (pe[k - 1] >= params.fill_min_p or a.anchor)
            and (pe[k] >= params.fill_min_p or b.anchor)
        )
        if fillable:
            runs[-1].append(k)
        else:
            runs.append([k])
    for run in runs:
        if len(run) < 2:
            continue
        ts = np.array([nodes[k].t for k in run])
        pts = np.array([nodes[k].xy for k in run], float)
        if cum is not None:
            pts = pts - cum[ts]
        missing = np.array(
            [t for t in range(ts[0], ts[-1] + 1) if t not in set(ts.tolist())]
        )
        if not len(missing):
            continue
        if params.fill_mode == "kalman":
            est = kalman_rts(ts, pts, missing)
        else:
            est = np.stack(
                [np.interp(missing, ts, pts[:, a]) for a in range(2)], axis=1
            )
        if cum is not None:
            est = est + cum[missing]
        for t, e in zip(missing, est, strict=True):
            pred[int(t)] = (float(e[0]), float(e[1]))
    return pred


def rescue_on_trajectory(
    pred: dict[int, tuple[float, float]],
    nodes: list[Node],
    clip,
    gate_px: float = 12.0,
    min_conf: float = 0.1,
) -> dict[int, tuple[float, float]]:
    """B2: in frames that the solved trajectory *interpolates* (not detections
    the solver accepted), replace the interpolated point with the highest
    confidence low-confidence candidate (>= ``min_conf``) within ``gate_px`` of it."""
    accepted = {n.t for n in nodes}
    out = dict(pred)
    for t, (px, py) in pred.items():
        if t in accepted:
            continue
        xy, conf = clip.cands(t)
        if not len(xy):
            continue
        d = np.hypot(xy[:, 0] - px, xy[:, 1] - py)
        ok = (d <= gate_px) & (conf >= min_conf)
        if np.any(ok):
            idx = np.where(ok)[0]
            k = idx[np.argmax(conf[idx])]
            out[t] = (float(xy[k, 0]), float(xy[k, 1]))
    return out


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
