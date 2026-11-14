"""Refine a pitch homography so the model lines land on the white-line mask.

All 8 DOF of ``H`` are optimised (numpy Levenberg-Marquardt, no scipy). The
cost is a robust (Cauchy) distance-transform of the white-line mask sampled at
points along every pitch line/arc that projects into the image and onto grass,
run coarse-to-fine (blurred/truncated DT with a ~40 px capture range first,
then sharper). The operator's/Claude's labelled points (and optional
labelled-line points) stay in as a *weak prior* so the solution cannot slide
along symmetric lines. The result is rejected if the cost does not improve or
the homography goes degenerate, in which case the input ``H`` is returned.

Convention as in :mod:`footy_track.calibration.geometry`: ``H`` maps normalised
image coords to pitch metres; the optimiser works on ``G = H^-1`` (metres ->
working pixels). Usable from the pre-labelling pipeline and, later, the manual
calibration page (it only needs the masks and an ``H``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import PitchModel

#: pitch metres are scaled by this inside the optimiser (conditioning only)
_PITCH_SCALE = 50.0


@dataclass
class LineLabel:
    """A named straight pitch line (key of ``PitchModel.segments()``) with >= 2
    normalised image points that lie on it."""

    line_id: str
    points: list[tuple[float, float]]
    confidence: float = 1.0


@dataclass
class RefineStage:
    trunc_px: float  # DT truncation == capture range at this stage
    blur_px: float  # Gaussian sigma applied to the truncated DT (0 = none)
    cauchy_px: float  # Cauchy scale
    iters: int = 25


#: coarse -> fine, in px at a 1568-wide working image (scaled with the width)
DEFAULT_STAGES = (
    RefineStage(trunc_px=40.0, blur_px=8.0, cauchy_px=20.0),
    RefineStage(trunc_px=20.0, blur_px=3.0, cauchy_px=8.0),
    RefineStage(trunc_px=8.0, blur_px=0.0, cauchy_px=3.0),
)
_REF_WIDTH = 1568.0


@dataclass
class RefineResult:
    H: np.ndarray  # refined (or the input when rejected)
    applied: bool
    reason: str | None = None
    cost_before: float | None = None  # mean truncated DT (px @ working res) on the sharp mask
    cost_after: float | None = None
    n_samples: int = 0
    stage_costs: list[float] = field(default_factory=list)
    max_shift_px: float | None = None  # largest movement of a model sample point
    k1: float = 0.0  # lens distortion (division model, see geometry) of ``H``; 0 = none
    k1_fitted: bool = False  # k1 was estimated here (not given)

    def to_dict(self) -> dict:
        d = {
            "applied": self.applied, "reason": self.reason,
            "cost_before": self.cost_before, "cost_after": self.cost_after,
            "n_samples": self.n_samples, "max_shift_px": self.max_shift_px,
        }
        if self.k1 or self.k1_fitted:
            d["k1"] = self.k1
            d["k1_fitted"] = self.k1_fitted
        return d


# --------------------------------------------------------------------------
# sampling the model
# --------------------------------------------------------------------------


def model_samples(pitch: PitchModel, step_m: float = 0.5) -> np.ndarray:
    """(N, 2) pitch points (metres) along every line and arc."""
    pts = []
    for poly in pitch.polylines().values():
        pts.append(geo._densify(np.asarray(poly, np.float64), step_m))
    return np.concatenate(pts)


def _truncated_dt(mask: np.ndarray, trunc: float, blur: float) -> np.ndarray:
    d = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    d = np.minimum(d, trunc)
    if blur > 0:
        d = cv2.GaussianBlur(d, (0, 0), blur)
    return d


def _bilinear(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """img sampled at (N,2) pixel coords (clamped to the border)."""
    h, w = img.shape
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (
        img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
        + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy
    )


def _distort_px(xy: np.ndarray, k1: float, size, aspect: float) -> np.ndarray:
    """undistorted working px -> observed working px (division model, see geometry)."""
    sz = np.asarray(size, float)
    return geo.distort_norm(xy / sz, k1, aspect=aspect) * sz


def _undistort_px(xy: np.ndarray, k1: float, size, aspect: float) -> np.ndarray:
    sz = np.asarray(size, float)
    return geo.undistort_norm(xy / sz, k1, aspect=aspect) * sz


def _project(G: np.ndarray, pts_m: np.ndarray, k1: float = 0.0, size=None, aspect: float = geo.DEFAULT_ASPECT):
    """metres -> observed working px through G (3x3) and the lens (k1); returns (xy, w)."""
    ph = np.concatenate([pts_m, np.ones((len(pts_m), 1))], axis=1) @ G.T
    w = ph[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = ph[:, :2] / w[:, None]
    if k1 != 0.0:
        xy = _distort_px(xy, k1, size, aspect)
    return xy, w


def _valid_mask(G, pts_m, size, grass, k1: float = 0.0, aspect: float = geo.DEFAULT_ASPECT) -> np.ndarray:
    w_img, h_img = size
    xy, w = _project(G, pts_m, k1, size, aspect)
    ok = (w > 1e-9) & np.all(np.isfinite(xy), axis=1)
    ok &= (xy[:, 0] >= 0) & (xy[:, 0] < w_img - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < h_img - 1)
    if ok.any():
        xi = np.round(xy[ok, 0]).astype(int)
        yi = np.round(xy[ok, 1]).astype(int)
        on = grass[yi, xi]
        idx = np.flatnonzero(ok)
        ok[idx[~on]] = False
    return ok


# --------------------------------------------------------------------------
# parametrisation and residuals
# --------------------------------------------------------------------------

_S = np.diag([1.0 / _PITCH_SCALE, 1.0 / _PITCH_SCALE, 1.0])


def _G_from_params(G0: np.ndarray, p: np.ndarray) -> np.ndarray:
    """G = G0 (I + Delta(p)) in the pitch-scaled frame; Delta[2,2] fixed to 0."""
    D = np.zeros(9)
    D[:8] = p
    D = D.reshape(3, 3)
    return G0 @ np.linalg.inv(_S) @ (np.eye(3) + D) @ _S


def _norm_G(G: np.ndarray) -> np.ndarray:
    return G / G[2, 2] if abs(G[2, 2]) > 1e-12 else G


def _line_dist_px(G: np.ndarray, line_pitch: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """Distance (px) of image points to the image of a pitch line.

    ``line_pitch`` is the homogeneous pitch-plane line (a, b, c): a*x+b*y+c=0.
    Image line = G^-T l (G maps metres -> px, so points obey l^T G^-1 p = 0).
    """
    li = np.linalg.inv(G).T @ line_pitch
    n = np.hypot(li[0], li[1])
    if n < 1e-15:
        return np.full(len(pts_px), 1e3)
    ph = np.concatenate([pts_px, np.ones((len(pts_px), 1))], axis=1)
    return np.abs(ph @ li) / n


def _pitch_line_h(seg: np.ndarray) -> np.ndarray:
    (x0, y0), (x1, y1) = seg
    l = np.cross([x0, y0, 1.0], [x1, y1, 1.0])
    return l / np.hypot(l[0], l[1])


class _Problem:
    def __init__(self, pitch, samples, size, prior_pts, prior_lines, prior_w, aspect=geo.DEFAULT_ASPECT):
        self.pitch, self.samples, self.size, self.aspect = pitch, samples, size, aspect
        self.prior_xy = np.array([p[0] for p in prior_pts]).reshape(-1, 2)  # metres
        self.prior_px = np.array([p[1] for p in prior_pts]).reshape(-1, 2)  # working px
        self.prior_conf = np.array([p[2] for p in prior_pts], float)
        self.prior_lines = prior_lines  # [(line3, pts_px (k,2), conf)]
        self.prior_w = prior_w

    def residual(self, G, idx, dt, trunc, cauchy, ref, k1=0.0):
        """-> (r, weights-free residual vector, with robust weights applied by caller)."""
        xy, w = _project(G, self.samples[idx], k1, self.size, self.aspect)
        bad = (w <= 1e-9) | ~np.all(np.isfinite(xy), axis=1)
        d = np.full(len(idx), trunc, np.float64)
        ok = ~bad
        d[ok] = np.where(
            (xy[ok, 0] < 0) | (xy[ok, 0] > self.size[0] - 1) | (xy[ok, 1] < 0) | (xy[ok, 1] > self.size[1] - 1),
            trunc, _bilinear(dt, xy[ok]),
        )
        parts = [d]
        # prior: labelled points as px reprojection error, robustified separately
        if len(self.prior_xy) and self.prior_w > 0:
            pxy, pw = _project(G, self.prior_xy, k1, self.size, self.aspect)
            e = pxy - self.prior_px
            e[(pw <= 1e-9) | ~np.all(np.isfinite(e), axis=1)] = 1e3
            parts.append((self.prior_w * np.sqrt(self.prior_conf)[:, None] * e).ravel())
        for line3, pts, conf in self.prior_lines:
            if self.prior_w > 0:
                if k1 != 0.0:
                    pts = _undistort_px(pts, k1, self.size, self.aspect)
                parts.append(self.prior_w * np.sqrt(conf) * _line_dist_px(G, line3, pts))
        return np.concatenate(parts), len(idx)


def _cauchy_w(r: np.ndarray, c: float) -> np.ndarray:
    return 1.0 / (1.0 + (r / c) ** 2)


#: weight (px of residual per unit k1) of the k1 prior; ``k1_prior`` scales it. A quadratic
#: (non-robust) penalty: with 150, k1 = 0.02 (~20 px of bow at the image edge) costs 3 px.
K1_PRIOR_PX = 150.0


def _lm(problem, G0, idx, dt, st: RefineStage, scale: float, k1=0.0, fit_k1=False, k1_prior=1.0):
    """LM on the 8 DOF of G (+ k1 when ``fit_k1``). Returns (G, k1, robust cost)."""
    trunc, cauchy = st.trunc_px * scale, st.cauchy_px * scale
    npar = 9 if fit_k1 else 8
    wk = K1_PRIOR_PX * k1_prior

    def cost_and_r(p):
        G = _G_from_params(G0, p[:8])
        k = k1 + (p[8] if fit_k1 else 0.0)
        r, n = problem.residual(G, idx, dt, trunc, cauchy, scale, k)
        w = _cauchy_w(r, cauchy)
        c = float(np.sum(0.5 * cauchy**2 * np.log1p((r / cauchy) ** 2)))
        if fit_k1:  # quadratic prior pulling k1 to 0, outside the robust loss
            r = np.append(r, wk * k)
            w = np.append(w, 1.0)
            c += 0.5 * (wk * k) ** 2
        return c, r, w

    p = np.zeros(npar)
    c, r, w = cost_and_r(p)
    lam = 1e-2
    for _ in range(st.iters):
        J = np.empty((len(r), npar))
        for k in range(npar):
            q = p.copy()
            q[k] += 1e-5
            _, rk, _ = cost_and_r(q)
            J[:, k] = (rk - r) / 1e-5
        Jw = J * np.sqrt(w)[:, None]
        A = Jw.T @ Jw
        g = Jw.T @ (r * np.sqrt(w))
        improved = False
        for _try in range(8):
            try:
                dp = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            c2, r2, w2 = cost_and_r(p + dp)
            if c2 < c:
                p, c, r, w = p + dp, c2, r2, w2
                lam = max(lam / 3, 1e-7)
                improved = True
                break
            lam *= 4
        if not improved or np.linalg.norm(dp) < 1e-7:
            break
    return _G_from_params(G0, p[:8]), k1 + (p[8] if fit_k1 else 0.0), c


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def dt_residual(G: np.ndarray, samples: np.ndarray, mask: np.ndarray, grass: np.ndarray, trunc_px: float = 20.0,
                k1: float = 0.0):
    """(mean truncated distance in px, n) of in-image, on-grass model samples to the mask."""
    h, w = mask.shape
    aspect = w / h
    ok = _valid_mask(G, samples, (w, h), grass, k1, aspect)
    if ok.sum() == 0:
        return float("inf"), 0
    d = _truncated_dt(mask, trunc_px, 0.0)
    xy, _ = _project(G, samples[ok], k1, (w, h), aspect)
    return float(_bilinear(d, xy).mean()), int(ok.sum())


def _degenerate(G: np.ndarray, size) -> str | None:
    if not np.all(np.isfinite(G)):
        return "non-finite H"
    Gn = G / np.linalg.norm(G)
    s = np.linalg.svd(Gn, compute_uv=False)
    if s[2] / s[0] < 1e-6:
        return "H is singular"
    # the ground in the lower image must stay in front of the camera: the bottom corners and
    # bottom-centre need w of one sign (the horizon may legitimately cross the top of the frame)
    w_img, h_img = size
    try:
        Hpx = np.linalg.inv(G)  # px -> metres
    except np.linalg.LinAlgError:
        return "H is singular"
    bottom = np.array([[0, h_img, 1.0], [w_img, h_img, 1.0], [w_img / 2, h_img, 1.0]])
    wb = (bottom @ Hpx.T)[:, 2]
    if not (np.all(wb > 0) or np.all(wb < 0)):
        return "the horizon crosses the bottom of the image"
    return None


def refine_homography(
    H: np.ndarray,
    pitch: PitchModel,
    lines_mask: np.ndarray,
    grass: np.ndarray,
    prior_points: list[tuple[str, tuple[float, float], float]] | None = None,
    line_labels: list[LineLabel] | None = None,
    prior_weight: float = 0.05,
    stages=DEFAULT_STAGES,
    starts: list[np.ndarray] | None = None,
    rank_label_weight: float = 0.05,
    label_tol_px: float = 45.0,
    max_shift_frac: float = 0.35,
    k1: float = 0.0,
    fit_k1: bool = False,
    k1_prior: float = 1.0,
    k1_limit: float = 0.15,
) -> RefineResult:
    """Refine ``H`` (normalised image -> pitch m) against the white-line mask.

    ``lines_mask``/``grass`` are boolean masks at working resolution.
    ``prior_points`` are ``(keypoint_id, (u, v), confidence)`` labels, held as a
    weak prior (``prior_weight`` px of residual per px of label error).
    ``starts`` are extra initial homographies (e.g. from subsets of the labels):
    every start is refined and the best by (all-sample DT cost + a small label
    term) wins, which makes the fit robust to a poor DLT from noisy labels.
    ``label_tol_px`` is the label error (px at 1568 wide) that is free in the ranking.
    ``max_shift_frac`` rejects a result that moved the in-view model further than
    this fraction of the image width from ``H`` (runaway; 90th percentile over samples in view under both).
    The returned ``H`` is the input when the result is rejected.

    Lens distortion (division model, :func:`footy_track.calibration.geometry.undistort_norm`):
    ``k1`` is the distortion ``H`` is given with (0 = plain homography). With ``fit_k1`` it
    is estimated jointly with ``H`` in the two finer stages (the coarse capture stage keeps
    it fixed), pulled to 0 by a quadratic prior of weight ``k1_prior`` and clamped to
    ``+-k1_limit``; the result's ``k1`` is then the new value. Without ``fit_k1`` the given
    ``k1`` is held fixed (a per-clip value) and returned unchanged.
    """
    H = np.asarray(H, np.float64)
    h_img, w_img = lines_mask.shape
    size = (w_img, h_img)
    scale = w_img / _REF_WIDTH
    aspect = w_img / h_img
    to_norm = np.diag([1.0 / w_img, 1.0 / h_img, 1.0])
    from_norm = np.diag([w_img, h_img, 1.0])

    def to_G(Hn):
        G = np.linalg.inv(np.asarray(Hn, np.float64) @ to_norm)
        return G / np.linalg.norm(G)

    samples = model_samples(pitch)
    kps = pitch.keypoints()
    px = np.array([w_img, h_img], float)
    pts = [(np.array(kps[k]), np.array(uv) * px, conf) for k, uv, conf in (prior_points or []) if k in kps]
    segs = pitch.segments()
    plines = []
    for ll in line_labels or []:
        if ll.line_id in segs and len(ll.points) >= 2:
            plines.append((_pitch_line_h(segs[ll.line_id]), np.array(ll.points, float) * px, ll.confidence))
    problem = _Problem(pitch, samples, size, pts, plines, prior_weight, aspect)
    dts = [_truncated_dt(lines_mask, st.trunc_px * scale, st.blur_px * scale) for st in stages]
    rank_trunc = 12.0 * scale
    rank_dt = _truncated_dt(lines_mask, rank_trunc, 0.0)

    def rank_cost(G, k=0.0) -> float:
        """Mean truncated DT over ALL model samples; off-image / behind-camera ones count as
        ``rank_trunc`` so hiding the model cannot win; plus a small label-reprojection term."""
        xy, w = _project(G, samples, k, size, aspect)
        inside = (w > 1e-9) & np.all(np.isfinite(xy), axis=1)
        inside &= (xy[:, 0] >= 0) & (xy[:, 0] <= w_img - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h_img - 1)
        d = np.full(len(samples), rank_trunc)
        if inside.any():
            xi, yi = np.round(xy[inside, 0]).astype(int), np.round(xy[inside, 1]).astype(int)
            on = grass[yi, xi]
            dd = _bilinear(rank_dt, xy[inside])
            dd[~on] = rank_trunc
            d[inside] = dd
        c = float(d.mean())
        if len(problem.prior_xy) and rank_label_weight > 0:
            pxy, pw = _project(G, problem.prior_xy, k, size, aspect)
            e = np.linalg.norm(pxy - problem.prior_px, axis=1)
            e[(pw <= 1e-9) | ~np.isfinite(e)] = 400.0 * scale
            # only label error beyond what imprecise labelling explains counts (px at 1568 wide)
            c += rank_label_weight * float(np.maximum(np.minimum(e, 400.0 * scale) - label_tol_px * scale, 0).mean())
        return c

    G_in = to_G(H)
    idx_in = np.flatnonzero(_valid_mask(G_in, samples, size, grass, k1, aspect))
    cost_in = rank_cost(G_in, k1)

    def run(G0):
        G, k, costs = G0, k1, []
        for si, (st, dt) in enumerate(zip(stages, dts, strict=True)):
            idx = np.flatnonzero(_valid_mask(G, samples, size, grass, k, aspect))
            if len(idx) < 30:
                return None
            G, k, c = _lm(problem, G, idx, dt, st, scale, k, fit_k1 and si >= 1, k1_prior)
            k = float(np.clip(k, -k1_limit, k1_limit))
            costs.append(c)
        return G, k, costs

    best = None  # (rank cost, G, stage costs)
    for Hs in [H, *(starts or [])]:
        try:
            G0 = to_G(Hs)
        except np.linalg.LinAlgError:
            continue
        if _degenerate(G0, size):
            continue
        out = run(G0)
        if out is None:
            continue
        G, k, costs = out
        if _degenerate(G, size):
            continue
        c = rank_cost(G, k)
        if best is None or c < best[0]:
            best = (c, G, costs, k)
    if best is None:
        return RefineResult(H, False, "no usable start (degenerate or nothing on grass)", cost_in, None,
                            len(idx_in), k1=k1)
    c_best, G, costs, k_f = best
    idx_f = np.flatnonzero(_valid_mask(G, samples, size, grass, k_f, aspect))
    # all-sample DT cost before/after (see rank_cost; label term excluded for reporting)
    def dt_only(Gx, kx):
        keep = problem.prior_xy, rank_label_weight
        problem.prior_xy = np.zeros((0, 2))
        try:
            return rank_cost(Gx, kx)
        finally:
            problem.prior_xy = keep[0]

    b, a = dt_only(G_in, k1), dt_only(G, k_f)
    both = np.intersect1d(idx_f, idx_in)  # in view under the input and the refined H
    shift = None
    if len(both):
        xy0, _ = _project(G_in, samples[both], k1, size, aspect)
        xy1, _ = _project(G, samples[both], k_f, size, aspect)
        shift = float(np.percentile(np.linalg.norm(xy1 - xy0, axis=1), 90))
    if len(idx_f) < 30:
        return RefineResult(H, False, "too little of the refined model on grass in the image", b, a, len(idx_f),
                            costs, shift, k1)
    if not c_best < cost_in - 1e-3:
        return RefineResult(H, False, "cost did not improve", b, a, len(idx_f), costs, shift, k1)
    if shift is not None and shift > max_shift_frac * w_img:
        return RefineResult(H, False, f"max model shift {shift:.0f}px exceeds limit", b, a, len(idx_f), costs, shift,
                            k1)
    Hn = geo.orient_homography(np.linalg.inv(G) @ from_norm, np.array([[0.5, 0.5]]))
    return RefineResult(Hn, True, None, b, a, len(idx_f), costs, shift, k_f, fit_k1)
