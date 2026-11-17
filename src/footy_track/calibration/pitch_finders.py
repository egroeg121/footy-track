# ruff: noqa: PLR0911, PLR0912, PLR0915, PLR2004
"""Two calibration point finders for the labelling editor: the centre and the penalty-box end.

George places image <-> pitch pairs by hand; these two finders place as many as they can, with known
pitch coordinates, from the white-line mask. Pitch *length and width vary per ground*, so nothing here
assumes a fixed image or pitch distance between structures: only the Laws-fixed shapes (centre circle
r 9.15; penalty area 16.5 x 40.32; goal area 5.5 x 18.32; D = circle r 9.15 about the spot 11 m out)
are used as templates, and pitch ids are mapped to metres with the *request's* ``PitchModel``.

A. ``find_centre`` (centre circle + halfway line -> a fitted homography and its points)
----------------------------------------------------------------------------------------
``centre_circle`` gives the halfway line ``l``, the circle's image ellipse ``E`` and ``V`` = pole of ``l``
w.r.t. ``E`` = the vanishing point of the pitch x (length) axis. ``centre_fit`` (see its docstring) solves the one
remaining degree of freedom (the vanishing point of the y axis on ``l``) from a touchline end, the ground's fixed
camera centre, or the square-pixel prior, builds ``H`` from the circle's four axis points, scores it on the whole
template against the mask and returns *points computed from that H* (circle top / bottom / left / right, spot,
halfway-line ends in frame, free circle points), so the editor reproduces the same ``H``. The touchline is
still found as a long line through ``V`` (angle test; the faint far-edge strip is part of the mask used to
verify it), its junction with ``l`` being ``(0, +-W/2)`` from the request's pitch dims. When no candidate is
accepted only the 2 circle x line points are returned. ``shape=False`` keeps the earlier derivation (touchline
end only, 3-4 known points on ``l`` -> circle centre -> left / right) for comparison.

B. ``find_d`` (penalty area template)
-------------------------------------
The whole penalty-area template (goal line, 18-yard box, 6-yard box, D) is fitted by brute force:
hypotheses = assignments of detected lines to template lines (a pair parallel to the goal line + a pair
along the pitch, which must have orthogonal vanishing points for a plausible focal length) and, when
the D ellipse is found, "D + front line + one more line" (3 points + 1 line, linear). Each hypothesis
gives a homography; the template is rendered under it and scored against the line mask (matched
length / (visible + detected), distance-transform lookups); the best few are refined by least squares
on the mask distance; the answer is kept only if it clearly beats the best *different* runner-up.
Returned points are the template keypoints that fall inside the frame; ``derived`` marks the ones with
no line evidence at their own location (spot, hidden corners). When the template fit abstains but the
D ellipse is clean, the 2 front-line x D points are returned as before.

Convention: the camera is on the ``-y`` side (far touchline = ``+y`` = "TOP", ``+x`` to the image
right), as everywhere else in the labeller; an improper (mirrored) fit is never accepted.

All geometry is in px of the undistorted working plane (``centre_circle.WORK_EDGE``), index
coordinates; outputs are observed normalised points, exactly like ``centre_circle``.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from footy_track.calibration import centre_circle as cc
from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import (
    CIRCLE_RADIUS,
    GOAL_AREA_DEPTH,
    GOAL_AREA_WIDTH,
    PEN_BOX_DEPTH,
    PEN_BOX_WIDTH,
    PEN_SPOT_DIST,
    PitchModel,
)
from footy_track.calibration.prelabel import candidates as cand

FINDER_KINDS = ("centre", "d")

# ---- tunables (px are at the working width ``w``; tolerances scale with ``w / 1280``) -------------
#: touchline: max angle between a candidate line and the direction to V
TOUCH_MAX_DEG = 2.5
#: pixel support needed around a junction (fraction of the image diagonal)
JUNCTION_R = 0.045
#: rendered-template precision needed to keep the derived centre points
CENTRE_MIN_PRECISION = 0.6
#: penalty-area fit acceptance
D_MIN_F1 = 0.50
D_MIN_PRECISION = 0.62
D_MIN_ELEMENTS = 3
#: a different fit must explain at least this much less of the mask (relative matched length) than the best one
D_MIN_MARGIN = 0.15
#: same-evidence fits this close in f1 are the best one's near-ties (their disagreement drops points)
D_TIE_F1 = 0.05
#: focal length range (f / width) allowed for a vanishing-point pair
F_RANGE = (0.6, 12.0)
MAX_LINES_D4 = 10
MAX_LINES_D = 12
N_REFINE = 24
SEG_CAP_PX = 30.0

# keypoints returned for one end of the pitch (ids exist in ``PitchModel.keypoints``)
_D_KEYS = (
    "PEN_BOX_TOP_OUTER",
    "PEN_BOX_BOTTOM_OUTER",
    "PEN_ARC_TOP",
    "PEN_ARC_BOTTOM",
    "PEN_ARC_MID",
    "PEN_SPOT",
    "PEN_BOX_TOP_GOAL",
    "PEN_BOX_BOTTOM_GOAL",
    "GOAL_AREA_TOP_OUTER",
    "GOAL_AREA_BOTTOM_OUTER",
    "GOAL_AREA_TOP_GOAL",
    "GOAL_AREA_BOTTOM_GOAL",
)
#: keypoints that are never a line feature (nothing to see at the point itself)
_ALWAYS_DERIVED = ("PEN_SPOT", "CENTRE_SPOT")


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass
class FoundPoint:
    #: keypoint id, or ``None`` for a free point (a derived point on the circle with no keypoint name)
    pitch_id: str | None
    uv: np.ndarray  # observed normalised image point
    confidence: float
    derived: bool
    pitch_xy: tuple[float, float] = (0.0, 0.0)  # metres, from the request's pitch dims


@dataclass
class FinderResult:
    kind: str  # "centre" | "d"
    points: list[FoundPoint]
    confidence: float
    info: dict = field(default_factory=dict)

    def correspondences(self) -> list[dict]:
        """Records in the labeller's ``correspondences`` shape plus ``confidence`` and ``derived``."""
        return [
            {
                "image": [float(p.uv[0]), float(p.uv[1])],
                "pitch_id": p.pitch_id,
                "pitch_xy": [float(p.pitch_xy[0]), float(p.pitch_xy[1])],
                "confidence": round(float(p.confidence), 3),
                "derived": bool(p.derived),
            }
            for p in self.points
        ]


@dataclass
class FinderDetection:
    result: FinderResult | None
    reason: str = ""
    notes: list[str] = field(default_factory=list)
    size: tuple[int, int] = (0, 0)
    seconds: float = 0.0
    #: drawing aid: ``H_px`` (pitch m -> undistorted px), template polylines, lines
    debug: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# small geometry helpers
# --------------------------------------------------------------------------


def _norm_t(size: tuple[int, int]) -> np.ndarray:
    """Undistorted index px -> centred normalised coordinates (``(p - c) / w``)."""
    w, h = size
    return np.array([[1 / w, 0, -0.5], [0, 1 / w, -0.5 * h / w], [0, 0, 1.0]])


def _hom(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    return np.append(p, 1.0)


def _dehom(v: np.ndarray) -> np.ndarray | None:
    return None if abs(v[2]) < 1e-12 * max(1.0, np.abs(v[:2]).max()) else v[:2] / v[2]


def _fp(pm: PitchModel, pid: str | None, uv, conf: float, derived: bool, xy=None) -> FoundPoint:
    """A found point; its metres come from the pitch model (``pid``) or ``xy`` (free point)."""
    x, y = pm.keypoints()[pid] if pid else xy
    return FoundPoint(pid, np.asarray(uv, float), float(conf), bool(derived), (float(x), float(y)))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _line_h(o: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.array([n[0], n[1], -(n @ o)])


def _to_obs(P_px: np.ndarray, size: tuple[int, int], dist: dict | None) -> np.ndarray:
    """Undistorted index px (N, 2) -> observed normalised (N, 2) (NaN where the lens has no inverse)."""
    w, h = size
    return geo.distort_points(geo.px_to_norm(np.atleast_2d(P_px) + 0.5, w, h), dist)


def _resample(poly: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(np.ceil(cum[-1] / step)) + 1)
    t = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(t, cum, poly[:, 0]), np.interp(t, cum, poly[:, 1])], 1)


# --------------------------------------------------------------------------
# distance field + template scoring
# --------------------------------------------------------------------------


class Field:
    """Euclidean distance to the nearest thin line pixel, bilinearly sampled."""

    def __init__(self, und: np.ndarray, size: tuple[int, int]):
        w, h = size
        img = np.full((h, w), 255, np.uint8)
        xi = np.clip(np.round(und[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(und[:, 1]).astype(int), 0, h - 1)
        img[yi, xi] = 0
        self.dt = cv2.distanceTransform(img, cv2.DIST_L2, 3).astype(np.float64)
        self.w, self.h = w, h
        self.und = und
        self._xi, self._yi = xi, yi

    def sample(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y = xy[..., 0], xy[..., 1]
        with np.errstate(invalid="ignore"):
            inb = (x >= 0) & (x <= self.w - 1.001) & (y >= 0) & (y <= self.h - 1.001)
        inb &= np.isfinite(x) & np.isfinite(y)
        xc = np.where(inb, x, 0.0)
        yc = np.where(inb, y, 0.0)
        x0 = np.floor(xc).astype(int)
        y0 = np.floor(yc).astype(int)
        fx, fy = xc - x0, yc - y0
        dt = self.dt
        d = (
            dt[y0, x0] * (1 - fx) * (1 - fy)
            + dt[y0, x0 + 1] * fx * (1 - fy)
            + dt[y0 + 1, x0] * (1 - fx) * fy
            + dt[y0 + 1, x0 + 1] * fx * fy
        )
        return d, inb

    def pixels_in(self, region: np.ndarray) -> int:
        """Number of thin line pixels inside a uint8 region mask."""
        return int(region[self._yi, self._xi].sum())


@dataclass
class Template:
    names: list[str]
    pts: np.ndarray  # (P, 2) metres
    elem: np.ndarray  # (P,) element index
    same_next: np.ndarray  # (P,) bool: the next sample belongs to the same element

    @classmethod
    def build(cls, polys: dict[str, np.ndarray], step: float) -> Template:
        names, pts, elem, nxt = [], [], [], []
        for i, (name, poly) in enumerate(polys.items()):
            r = _resample(np.asarray(poly, float), step)
            names.append(name)
            pts.append(r)
            elem.append(np.full(len(r), i))
            s = np.ones(len(r), bool)
            s[-1] = False
            nxt.append(s)
        return cls(names, np.concatenate(pts), np.concatenate(elem), np.concatenate(nxt))


def _project(Hs: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``Hs`` (K, 3, 3) pitch -> px over metre points (P, 2): ``xy (K, P, 2)``, ``ok (K, P)`` (in front)."""
    ph = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    pr = np.einsum("kij,pj->kpi", Hs, ph)
    wv = pr[..., 2]
    ok = wv > 1e-9
    xy = pr[..., :2] / np.where(ok, wv, 1.0)[..., None]
    return xy, ok


def _weights(xy: np.ndarray, inb: np.ndarray, tpl: Template) -> np.ndarray:
    """Per-sample image length (px) it stands for: half the px gap to each same-element neighbour."""
    seg = np.linalg.norm(xy[:, 1:] - xy[:, :-1], axis=2)
    good = tpl.same_next[None, :-1] & inb[:, 1:] & inb[:, :-1]
    seg = np.where(good, np.minimum(seg, SEG_CAP_PX), 0.0)
    wt = np.zeros(inb.shape)
    wt[:, :-1] += 0.5 * seg
    wt[:, 1:] += 0.5 * seg
    return np.where(inb, wt, 0.0)


def coarse_scores(Hs: np.ndarray, pts: np.ndarray, fld: Field, tolm: float, chunk: int = 1500) -> np.ndarray:
    """``2 matched - visible`` sample count per hypothesis (nearest-pixel lookups, uniform metre samples)."""
    out = np.empty(len(Hs))
    ph = np.concatenate([pts, np.ones((len(pts), 1))], 1).T  # (3, P)
    w, h = fld.w, fld.h
    for i in range(0, len(Hs), chunk):
        pr = np.matmul(Hs[i : i + chunk], ph)  # (K, 3, P)
        wv = pr[:, 2]
        ok = wv > 1e-9
        inv = 1.0 / np.where(ok, wv, 1.0)
        x = np.rint(pr[:, 0] * inv)
        y = np.rint(pr[:, 1] * inv)
        inb = ok & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        xi = np.where(inb, x, 0).astype(np.intp)
        yi = np.where(inb, y, 0).astype(np.intp)
        d = fld.dt[yi, xi]
        m = (inb & (d <= tolm)).sum(1)
        out[i : i + chunk] = 2.0 * m - inb.sum(1)
    return out


def evaluate(H: np.ndarray, tpl: Template, fld: Field, tolm: float) -> dict:
    """Detailed fit quality of one homography: precision, F1, per-element matched/visible length."""
    xy, ok = _project(H[None], tpl.pts)
    d, inb = fld.sample(xy)
    inb &= ok
    wt = _weights(xy, inb, tpl)[0]
    d, inb, xy = d[0], inb[0], xy[0]
    match = (d <= tolm) & inb
    m = float((wt * match).sum())
    v = float(wt.sum())
    elems = []
    for i, name in enumerate(tpl.names):
        sel = tpl.elem == i
        elems.append((name, float(wt[sel].sum()), float((wt * match)[sel].sum())))
    # detected line pixels in the hull of the visible template
    dlen = 0.0
    if inb.sum() >= 3:
        hull = cv2.convexHull(xy[inb].astype(np.float32))
        reg = np.zeros((fld.h, fld.w), np.uint8)
        cv2.fillConvexPoly(reg, np.round(hull).astype(np.int32).reshape(-1, 2), 1)
        k = int(2 * round(tolm) + 1)
        reg = cv2.dilate(reg, np.ones((k, k), np.uint8))
        dlen = float(fld.pixels_in(reg))
    f1 = 2.0 * m / max(v + max(dlen, m), 1.0)
    return {
        "m": m,
        "v": v,
        "dlen": dlen,
        "precision": m / v if v > 0 else 0.0,
        "f1": f1,
        "elems": elems,
        "xy": xy,
        "inb": inb,
        "match": match,
        "wt": wt,
    }


def _h_from_4(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    try:
        return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
    except cv2.error:
        return None


def _solve4(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Batched exact 4-point homography: ``src`` (K, 4, 2) -> ``dst`` (K, 4, 2), h33 = 1; NaN if singular."""
    K = len(src)
    A = np.zeros((K, 8, 8))
    b = np.zeros((K, 8))
    X, Y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    for i in range(4):
        A[:, 2 * i, 0], A[:, 2 * i, 1], A[:, 2 * i, 2] = X[:, i], Y[:, i], 1.0
        A[:, 2 * i, 6], A[:, 2 * i, 7] = -u[:, i] * X[:, i], -u[:, i] * Y[:, i]
        A[:, 2 * i + 1, 3], A[:, 2 * i + 1, 4], A[:, 2 * i + 1, 5] = X[:, i], Y[:, i], 1.0
        A[:, 2 * i + 1, 6], A[:, 2 * i + 1, 7] = -v[:, i] * X[:, i], -v[:, i] * Y[:, i]
        b[:, 2 * i], b[:, 2 * i + 1] = u[:, i], v[:, i]
    H = np.full((K, 3, 3), np.nan)
    det = np.linalg.det(A)
    good = np.abs(det) > 1e-14
    if good.any():
        sol = np.linalg.solve(A[good], b[good][..., None])[..., 0]
        h = np.concatenate([sol, np.ones((len(sol), 1))], 1).reshape(-1, 3, 3)
        H[good] = h
    return H


# --------------------------------------------------------------------------
# frame context shared by the finders
# --------------------------------------------------------------------------


@dataclass
class _Ctx:
    mask: np.ndarray
    dist: dict | None
    pitch: PitchModel
    size: tuple[int, int]
    obs: np.ndarray
    und: np.ndarray
    tol: float
    fld: Field | None = None
    #: line pixels used to look for the touchline (mask + faint far strip): ``(mask, obs, und)``
    tl: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    fld_tl: Field | None = None

    @property
    def diag(self) -> float:
        return float(np.hypot(*self.size))

    def field(self) -> Field:
        if self.fld is None:
            self.fld = Field(self.und, self.size)
        return self.fld

    def field_tl(self) -> Field:
        """Distance field of the mask *plus* the faint far touchline strip (used to verify a touchline)."""
        if self.tl is None:
            return self.field()
        if self.fld_tl is None:
            self.fld_tl = Field(self.tl[2], self.size)
        return self.fld_tl


def _ctx(mask: np.ndarray, dist: dict | None, pitch: PitchModel | None, prep=None) -> _Ctx:
    h, w = mask.shape
    obs, und = prep if prep is not None else cc._prep_points(mask, dist)
    return _Ctx(mask, dist, pitch or PitchModel(), (w, h), obs, und, max(2.5, w / 450))


# --------------------------------------------------------------------------
# A. centre finder
# --------------------------------------------------------------------------


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


def _line_pixels_near(und: np.ndarray, o: np.ndarray, d: np.ndarray, s_at: float, half_w: float, r: float) -> int:
    n = np.array([-d[1], d[0]])
    rel = und - o
    return int(((np.abs(rel @ n) < half_w) & (np.abs(rel @ d - s_at) < r)).sum())


def far_mask_for(bgr: np.ndarray) -> np.ndarray:
    """The faint far touchline strip (``candidates.far_edge_line``) of the working-size frame.

    Kept out of the ellipse detector's mask (it would feed it a pitch-boundary curve) and used only to
    find the touchline."""
    w = cc.resize_work(bgr)
    return cand.far_edge_line(w, cand.grass_mask(w))


def _find_touchlines(ctx: _Ctx, r: cc.CirclePoints, V: np.ndarray) -> list[dict]:
    """Long lines through the vanishing point ``V`` that cross the halfway line beyond the circle.

    ``V`` is homogeneous in undistorted index px. Returns candidates ``{end: "far"|"near", X, line, score}``
    where ``X`` is the touchline x halfway-line junction (index px).
    """
    w, h = ctx.size
    L = r.anchor
    tmask, tobs, tund = ctx.tl or (ctx.mask, ctx.obs, ctx.und)
    lines = cc._anchor_lines(tmask, tobs, tund, ctx.tol, 14, min_len_frac=0.05)
    sv = np.sort(np.array([(r.crossings_u[0] - L.o) @ L.d, (r.crossings_u[1] - L.o) @ L.d]))
    s_near, s_far = sv[0], sv[1]  # d points up the image: far = larger s
    out = []
    sin_max = np.sin(np.deg2rad(TOUCH_MAX_DEG))
    for T in lines:
        # not the halfway line itself, and not near-parallel to it
        if abs(T.d @ L.d) > np.cos(np.deg2rad(20)):
            continue
        mid = T.p(0.5 * (T.s0 + T.s1))
        vec = V[:2] - V[2] * mid
        if np.linalg.norm(vec) < 1e-9:
            continue
        sin_a = abs(T.n @ vec) / np.linalg.norm(vec)
        if sin_a > sin_max:
            continue
        # junction with the halfway line
        M = np.array([T.d, -L.d]).T
        if abs(np.linalg.det(M)) < 1e-6:
            continue
        sT, sL = np.linalg.solve(M, L.o - T.o)  # T.o + sT T.d = L.o + sL L.d
        X = L.o + sL * L.d
        if not (-0.02 * w <= X[0] <= 1.02 * w - 1 and -0.02 * h <= X[1] <= 1.02 * h - 1):
            continue
        margin = 0.03 * h
        if sL > s_far + margin:
            end = "far"
        elif sL < s_near - margin:
            end = "near"
        else:
            continue
        rr = JUNCTION_R * ctx.diag
        if _line_pixels_near(ctx.und, L.o, L.d, sL, 1.8 * ctx.tol, rr) < 12:
            continue
        if _line_pixels_near(tund, T.o, T.d, sT, 2.5 * ctx.tol, rr) < 12:
            continue
        out.append(
            {
                "end": end,
                "X": X,
                "sL": float(sL),
                "line": T,
                "sin_deg": float(np.degrees(np.arcsin(sin_a))),
                "score": T.length * min(1.0, T.support),
            }
        )
    return out


def _map_1d(ys: np.ndarray, ss: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit ``s(y) = (a y + b) / (c y + 1)``: returns ``((a, b, c), max abs residual)``."""
    A = np.stack([ys, np.ones_like(ys), -ss * ys], 1)
    sol, *_ = np.linalg.lstsq(A, ss, rcond=None)
    a, b, c = sol
    pred = (a * ys + b) / (c * ys + 1.0)
    return sol, float(np.abs(pred - ss).max())


def find_centre(
    mask: np.ndarray,
    dist: dict | None = None,
    pitch: PitchModel | None = None,
    prep: tuple[np.ndarray, np.ndarray] | None = None,
    min_confidence: float | None = None,
    far_mask: np.ndarray | None = None,
    rig=None,
    shape: bool = True,
    use: tuple[str, ...] = ("touchline", "rig", "intrinsic"),
    circle: cc.Detection | None = None,
    refine: bool = True,
) -> FinderDetection:
    """Centre circle x halfway line -> a fitted pitch homography and the points derived from it.

    ``shape=True`` (default): ``centre_fit`` solves ``H`` from the circle + line shape (touchline end / camera rig of
    the ground / square-pixel prior, whichever the whole template supports best) and returns the named keypoints
    and free circle points computed from it. ``circle`` is an already computed ``centre_circle`` detection (evaluation). ``shape=False`` is the earlier behaviour: the 2 circle x line points
    plus the points derived from a touchline end when one is found. See the module docstring."""
    t0 = time.perf_counter()
    ctx = _ctx(mask, dist, pitch, prep)
    w, h = ctx.size
    out = FinderDetection(None, size=ctx.size)
    if len(ctx.und) < 80:
        out.reason = "too few line pixels"
        return out
    if far_mask is not None and far_mask.any():
        m2 = mask | far_mask
        ctx.tl = (m2, *cc._prep_points(m2, dist))
    det = circle or cc.detect_from_mask(
        mask, dist, ("centre",), pitch=ctx.pitch, min_confidence=min_confidence, prep=(ctx.obs, ctx.und)
    )
    out.notes = det.notes
    r = det.result
    if r is None:
        out.reason = det.reason
        out.seconds = time.perf_counter() - t0
        return out
    pm = ctx.pitch
    base = [_fp(pm, pid, uv, r.confidence, False) for pid, uv in zip(r.pitch_ids, r.points_uv, strict=True)]
    info: dict = {"circle": dict(r.info), "touchline": None}
    res = FinderResult("centre", base, r.confidence, info)
    out.result = res
    out.debug = {"ellipse": r.ellipse, "anchor": r.anchor}
    try:
        if shape:
            _add_shape_points(ctx, r, res, out, rig, use, refine)
        else:
            _add_touchline_points(ctx, r, res, out)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:  # never lose the 2 base points
        info["touchline"] = f"failed: {exc}"
    out.seconds = time.perf_counter() - t0
    return out


def _add_shape_points(
    ctx: _Ctx, r: cc.CirclePoints, res: FinderResult, out: FinderDetection, rig, use: tuple[str, ...], refine: bool = True
) -> None:
    """Shape-first path (``calibration.centre_fit``): replaces the base points by the fit's points when accepted."""
    import footy_track.calibration.centre_fit as cf  # noqa: PLC0415  (centre_fit imports this module)

    L, E = r.anchor, r.ellipse
    V = np.linalg.solve(cc._conic(E), _line_h(L.o, L.n))
    V = V / np.abs(V).max()
    touch = _find_touchlines(ctx, r, V)
    sf, log = cf.fit_shape(ctx, r, ctx.field_tl(), touch, rig, use, refine)
    res.info["shape"] = {"candidates": log}
    if sf is None:
        res.info["shape"]["accepted"] = None
        return
    pts = cf.points_from_fit(sf, ctx, r.confidence)
    if not pts:
        res.info["shape"]["accepted"] = None
        res.info["shape"]["why"] = "fewer than 4 points in general position in frame"
        return
    res.points = pts
    res.info["shape"].update(
        accepted=sf.source,
        precision=sf.ev["core_precision"],
        template_f1=sf.ev["f1"],
        **{k: v for k, v in sf.info.items() if k in ("f_over_w", "jitter", "ground", "rms_px", "ends", "alternatives")},
    )
    res.info["touchline_ends"] = sorted({"HALFWAY_TOP" if e == "far" else "HALFWAY_BOTTOM" for e in sf.X})
    out.debug.update(H_px=sf.H, template=cf._core_tpl(ctx.pitch))


def _add_touchline_points(ctx: _Ctx, r: cc.CirclePoints, res: FinderResult, out: FinderDetection) -> None:
    w, h = ctx.size
    pm = ctx.pitch
    info = res.info
    L, E = r.anchor, r.ellipse
    C = cc._conic(E)
    V = np.linalg.solve(C, _line_h(L.o, L.n))  # pole of the halfway line: vanishing point of x
    V = V / np.abs(V).max()
    cands = _find_touchlines(ctx, r, V)
    if not cands:
        info["touchline"] = "no touchline end found through the vanishing point"
        return
    best = {}
    for end in ("far", "near"):
        c = [x for x in cands if x["end"] == end]
        if c:
            best[end] = max(c, key=lambda x: x["score"])
    sv = np.sort(np.array([(r.crossings_u[0] - L.o) @ L.d, (r.crossings_u[1] - L.o) @ L.d]))
    s_top, s_bot = sv[1], sv[0]  # y = +r, -r
    ys, ss = [CIRCLE_RADIUS, -CIRCLE_RADIUS], [s_top, s_bot]
    hw = pm.hw
    if "far" in best:
        ys.append(hw)
        ss.append(best["far"]["sL"])
    if "near" in best:
        ys.append(-hw)
        ss.append(best["near"]["sL"])
    ys_a, ss_a = np.array(ys), np.array(ss)
    (a, b, c), resid = _map_1d(ys_a, ss_a)
    info["touchline"] = {
        e: {"sin_deg": round(best[e]["sin_deg"], 2), "length": round(best[e]["line"].length)} for e in best
    }
    info["map_residual_px"] = resid
    # monotone map over the pitch width, finite on it
    yy = np.linspace(-hw, hw, 40)
    den = c * yy + 1.0
    if not (den > 1e-6).all() or resid > 3.0 * ctx.tol:
        info["touchline"]["rejected"] = "inconsistent 1-D map" if resid > 3.0 * ctx.tol else "map pole on the pitch"
        return
    # vanishing point of y (homogeneous, index px) and the vanishing line
    Winf = np.array([c * L.o[0] + a * L.d[0], c * L.o[1] + a * L.d[1], c])
    T = _norm_t(ctx.size)
    Tinv_T = np.linalg.inv(T).T
    Cn = Tinv_T @ C @ np.linalg.inv(T)  # conic in normalised coordinates
    Vn, Wn = T @ V, T @ Winf
    Vn, Wn = Vn / np.linalg.norm(Vn), Wn / np.linalg.norm(Wn)
    m = np.cross(Vn, Wn)
    On = np.linalg.solve(Cn, m)
    centre_h = np.linalg.inv(T) @ On
    Oxy = _dehom(centre_h)
    if Oxy is None:
        info["touchline"]["rejected"] = "circle centre at infinity"
        return
    # the centre must lie inside the ellipse and on the halfway line, between the circle ends
    q = E.matrix() @ (Oxy - np.asarray(E.centre))
    if q @ q >= 0.6:
        info["touchline"]["rejected"] = "derived centre not inside the ellipse"
        return
    d_ax = _unit(V[:2] - V[2] * Oxy)  # direction O -> V (sign irrelevant)
    sv2, _ = cc._line_ellipse(Oxy, d_ax, E)
    if sv2 is None:
        info["touchline"]["rejected"] = "x axis misses the ellipse"
        return
    P_lr = np.stack([Oxy + sv2[0] * d_ax, Oxy + sv2[1] * d_ax])
    P_left, P_right = (P_lr[0], P_lr[1]) if P_lr[0][0] < P_lr[1][0] else (P_lr[1], P_lr[0])

    pts_px = [(pid, p, True) for pid, p in (("CENTRE_CIRCLE_LEFT", P_left), ("CENTRE_CIRCLE_RIGHT", P_right), ("CENTRE_SPOT", Oxy))]
    base_conf = r.confidence
    far_px = {}
    for end, pid in (("far", "HALFWAY_TOP"), ("near", "HALFWAY_BOTTOM")):
        if end in best:
            far_px[pid] = best[end]["X"]
    # verification: homography from every point, template rendered against the mask
    corr_px = [("CENTRE_CIRCLE_TOP", r.crossings_u[0], False), ("CENTRE_CIRCLE_BOTTOM", r.crossings_u[1], False)]
    corr_px += [(pid, X, False) for pid, X in far_px.items()] + pts_px
    kp = pm.keypoints()
    src = np.array([kp[pid] for pid, _, _ in corr_px], float)
    dst = np.array([(p - np.array([w / 2, h / 2])) / w for _, p, _ in corr_px])
    Hn, _ = cv2.findHomography(src, dst, 0)
    if Hn is None or not np.isfinite(Hn).all():
        info["touchline"]["rejected"] = "homography failed"
        return
    Hpx = np.linalg.inv(T) @ Hn
    tpl = _centre_template(pm)
    ev = evaluate(Hpx, tpl, ctx.field(), 1.6 * ctx.tol)
    info["template_precision"] = ev["precision"]
    info["template_f1"] = ev["f1"]
    out.debug.update(H_px=Hpx, template=tpl)
    if ev["precision"] < CENTRE_MIN_PRECISION:
        info["touchline"]["rejected"] = f"template precision {ev['precision']:.2f}"
        return
    # accepted: add the touchline ends (observed) and the derived points
    for pid, X in far_px.items():
        uv = _to_obs(X, ctx.size, ctx.dist)[0]
        if np.isfinite(uv).all():
            res.points.append(_fp(pm, pid, uv, 0.9 * base_conf, False))
    # circle points off the two axes (free points): an exact homography from the circle's four axis points gives them
    # even when the circle is cut by the frame, so >= 4 non-collinear points survive a partly visible circle
    rr = CIRCLE_RADIUS
    axis_m = np.array([(0.0, rr), (0.0, -rr), (-rr, 0.0), (rr, 0.0)])
    axis_px = np.array([r.crossings_u[0], r.crossings_u[1], P_left, P_right])
    H4 = _h_from_4(axis_m, axis_px)
    extra = [(pid, p, None) for pid, p, _ in pts_px]
    if H4 is not None:
        k = rr / np.sqrt(2.0)
        for sx in (-1.0, 1.0):
            for sy in (1.0, -1.0):
                xy = (sx * k, sy * k)
                P, ok = _project(H4[None], np.array([xy]))
                if ok[0, 0]:
                    extra.append((None, P[0, 0], xy))
    for pid, p, xy in extra:
        uv = _to_obs(p, ctx.size, ctx.dist)[0]
        if np.isfinite(uv).all() and (uv >= 0.005).all() and (uv <= 0.995).all():
            res.points.append(_fp(pm, pid, uv, 0.8 * base_conf, True, xy))
    res.info["touchline_ends"] = sorted(far_px)


def _centre_template(pm: PitchModel) -> Template:
    segs, curves = pm.segments(), pm.curves()
    polys = {
        "HALFWAY_LINE": segs["HALFWAY_LINE"],
        "CENTRE_CIRCLE": curves["CENTRE_CIRCLE"],
        # touchlines near the halfway line only: their far ends depend on the pitch length
        "TOUCHLINE_TOP": np.array([[-25.0, pm.hw], [25.0, pm.hw]]),
        "TOUCHLINE_BOTTOM": np.array([[-25.0, -pm.hw], [25.0, -pm.hw]]),
    }
    return Template.build(polys, 0.3)


# --------------------------------------------------------------------------
# B. penalty-area template finder
# --------------------------------------------------------------------------


def _end_x(pm: PitchModel, end: str) -> dict[str, float]:
    s = -1.0 if end == "L" else 1.0
    return {
        "G": s * pm.hl,
        "6F": s * (pm.hl - GOAL_AREA_DEPTH),
        "F": s * (pm.hl - PEN_BOX_DEPTH),
        "spot": s * (pm.hl - PEN_SPOT_DIST),
    }


_Y_LABELS = {
    "S+": PEN_BOX_WIDTH / 2,
    "S-": -PEN_BOX_WIDTH / 2,
    "6S+": GOAL_AREA_WIDTH / 2,
    "6S-": -GOAL_AREA_WIDTH / 2,
}


def _d_template(pm: PitchModel, end: str, step: float) -> Template:
    seg, cur = pm.segments(), pm.curves()
    x = _end_x(pm, end)
    pad = PEN_BOX_WIDTH / 2 + 6.0  # goal line: only a little beyond the box (its ends depend on the width)
    polys = {
        "BOX_FRONT": seg[f"{end}_PEN_BOX_FRONT"],
        "BOX_TOP": seg[f"{end}_PEN_BOX_TOP"],
        "BOX_BOTTOM": seg[f"{end}_PEN_BOX_BOTTOM"],
        "SIX_FRONT": seg[f"{end}_GOAL_AREA_FRONT"],
        "SIX_TOP": seg[f"{end}_GOAL_AREA_TOP"],
        "SIX_BOTTOM": seg[f"{end}_GOAL_AREA_BOTTOM"],
        "GOAL_LINE": np.array([[x["G"], -pad], [x["G"], pad]]),
        "D_ARC": cur[f"{end}_PEN_ARC"],
    }
    return Template.build(polys, step)


def _anchor_pts(pm: PitchModel, end: str) -> np.ndarray:
    """4 box corners (pitch m): the LM parametrisation anchors."""
    x = _end_x(pm, end)
    bw = PEN_BOX_WIDTH / 2
    return np.array([[x["F"], bw], [x["F"], -bw], [x["G"], -bw], [x["G"], bw]])


def _orientation_ok(Hs: np.ndarray, pm: PitchModel, end: str, size: tuple[int, int]) -> np.ndarray:
    """Proper (non-mirrored), in-front, plausibly placed homographies (pitch m -> undistorted px)."""
    w, h = size
    x = _end_x(pm, end)
    c0 = np.array([x["spot"], 0.0])
    pts = np.array([c0, c0 + [5.0, 0.0], c0 + [0.0, 5.0]])
    xy, ok = _project(Hs, pts)
    ok = ok.all(1)
    a, b = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    c = xy[:, 0]
    plaus = (c[:, 0] > -0.6 * w) & (c[:, 0] < 1.6 * w) & (c[:, 1] > -0.6 * h) & (c[:, 1] < 1.6 * h)
    # camera on the -y side: +x runs to the image right and +y up the image (every GT frame does); this also removes
    # the 180-degree-rotated twin (the other end of the pitch with top/bottom swapped), which renders identically
    return ok & (cross < 0) & (a[:, 0] > 0) & (b[:, 1] < 0) & plaus & np.isfinite(xy).all((1, 2))


def _compose(Ht: np.ndarray, x0: float, size: tuple[int, int]) -> np.ndarray:
    """Batched ``Ht`` (K, 3, 3): shifted pitch (``((x - x0) / 10, y / 10)``) -> normalised  ==>  pitch m -> px."""
    w, h = size
    Tp = np.array([[0.1, 0, -0.1 * x0], [0, 0.1, 0], [0, 0, 1.0]])
    Ti = np.linalg.inv(_norm_t(size))
    return np.einsum("ij,kjl,lm->kim", Ti, Ht, Tp)


def _gen_from_lines(lines: list[cc._Line], ctx: _Ctx, end: str) -> np.ndarray:
    """4-line hypotheses: 2 lines parallel to the goal line + 2 along the pitch, all label assignments."""
    pm, (w, h) = ctx.pitch, ctx.size
    lines = lines[:MAX_LINES_D4]
    N = len(lines)
    if N < 4:
        return np.empty((0, 3, 3))
    ls = np.array([_line_h((L.o - np.array([w / 2, h / 2])) / w, L.n) for L in lines])
    seg0 = np.array([L.p(L.s0) for L in lines])
    seg1 = np.array([L.p(L.s1) for L in lines])
    pairs = np.array(list(itertools.combinations(range(N), 2)))
    P = np.cross(ls[pairs[:, 0]], ls[pairs[:, 1]])  # vanishing point of each pair
    dot = np.einsum("ik,jk->ij", P[:, :2], P[:, :2])
    den = P[:, 2][:, None] * P[:, 2][None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = -dot / den
    disjoint = (
        (pairs[:, None, 0] != pairs[None, :, 0])
        & (pairs[:, None, 0] != pairs[None, :, 1])
        & (pairs[:, None, 1] != pairs[None, :, 0])
        & (pairs[:, None, 1] != pairs[None, :, 1])
    )
    ok = disjoint & np.isfinite(f2) & (f2 >= F_RANGE[0] ** 2) & (f2 <= F_RANGE[1] ** 2)
    ia, ib = np.nonzero(ok)  # pair a (parallel to goal line -> meets at U), pair b (along pitch -> V)
    if len(ia) == 0:
        return np.empty((0, 3, 3))
    xs = _end_x(pm, end)
    xlab = [xs["G"], xs["6F"], xs["F"]]
    ylab = list(_Y_LABELS.values())
    xcombo = list(itertools.permutations(range(3), 2))  # ordered: line a1 gets label i, a2 gets label j
    ycombo = list(itertools.permutations(range(4), 2))
    # intersection points a_i x b_j (normalised, finite)
    a1, a2 = pairs[ia, 0], pairs[ia, 1]
    b1, b2 = pairs[ib, 0], pairs[ib, 1]
    Q = np.stack(
        [
            np.cross(ls[a1], ls[b1]),
            np.cross(ls[a1], ls[b2]),
            np.cross(ls[a2], ls[b1]),
            np.cross(ls[a2], ls[b2]),
        ],
        1,
    )  # (K, 4, 3)
    fin = (np.abs(Q[..., 2]) > 1e-9).all(1)
    Q, ia, ib = Q[fin], ia[fin], ib[fin]
    if len(Q) == 0:
        return np.empty((0, 3, 3))
    Qn = Q[..., :2] / Q[..., 2:3]  # (K, 4, 2)
    c = np.array([w / 2, h / 2])
    Qpx = Qn * w + c
    # each corner must sit near both of its lines' visible extents (the box corners are where the segments end)
    a1, a2 = pairs[ia, 0], pairs[ia, 1]
    b1, b2 = pairs[ib, 0], pairs[ib, 1]
    gap = 0.3 * ctx.diag
    near = np.ones(len(Qpx), bool)
    for q, (la, lb) in enumerate(((a1, b1), (a1, b2), (a2, b1), (a2, b2))):
        for li in (la, lb):
            near &= _seg_dist(Qpx[:, q], seg0[li], seg1[li]) < gap
    Qn, Qpx = Qn[near], Qpx[near]
    if len(Qn) == 0:
        return np.empty((0, 3, 3))
    # chirality: an image quad and its pitch quad must have opposite orientation (y-down image)
    p11, p12, p21 = Qpx[:, 0], Qpx[:, 1], Qpx[:, 2]
    c_img = (p12[:, 0] - p11[:, 0]) * (p21[:, 1] - p11[:, 1]) - (p12[:, 1] - p11[:, 1]) * (p21[:, 0] - p11[:, 0])
    x0 = xs["spot"]
    sc = lambda xv, yv: np.array([(xv - x0) / 10.0, yv / 10.0])  # noqa: E731
    out = []
    for xi, xj in xcombo:
        for yi, yj in ycombo:
            sgn = np.sign((ylab[yj] - ylab[yi]) * (xlab[xj] - xlab[xi]))
            sel = np.sign(c_img) == sgn
            if not sel.any():
                continue
            src = np.array(
                [sc(xlab[xi], ylab[yi]), sc(xlab[xi], ylab[yj]), sc(xlab[xj], ylab[yi]), sc(xlab[xj], ylab[yj])]
            )
            Ht = _solve4(np.broadcast_to(src, (int(sel.sum()), 4, 2)).copy(), Qn[sel])
            out.append(_compose(Ht, x0, ctx.size))
    if not out:
        return np.empty((0, 3, 3))
    Hs = np.concatenate(out)
    return Hs[np.isfinite(Hs).all((1, 2))]


def _seg_dist(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance of points ``P`` (K, 2) to the segment ``a``-``b`` (K, 2)."""
    ab = b - a
    t = np.clip(((P - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _gen_from_arc(arc: cc.CirclePoints, lines: list[cc._Line], ctx: _Ctx) -> tuple[str, np.ndarray]:
    """D hypotheses: 3 points (chord ends + pole of the front line) + 1 more line (linear, SVD)."""
    pm, (w, h) = ctx.pitch, ctx.size
    end = arc.pitch_ids[0][0]
    xs = _end_x(pm, end)
    s = -1.0 if end == "L" else 1.0
    dy = float(np.sqrt(CIRCLE_RADIUS**2 - (PEN_BOX_DEPTH - PEN_SPOT_DIST) ** 2))
    d = PEN_BOX_DEPTH - PEN_SPOT_DIST
    x_pole = xs["spot"] - s * CIRCLE_RADIUS**2 / d
    L, E = arc.anchor, arc.ellipse
    C = cc._conic(E)
    pole = np.linalg.solve(C, _line_h(L.o, L.n))
    pxy = _dehom(pole)
    if pxy is None:
        return end, np.empty((0, 3, 3))
    x0 = xs["spot"]
    c = np.array([w / 2, h / 2])
    sc = lambda x_, y_: np.array([(x_ - x0) / 10.0, y_ / 10.0])  # noqa: E731
    pitch_pts = np.array([sc(xs["F"], dy), sc(xs["F"], -dy), sc(x_pole, 0.0)])
    img_pts = (np.array([arc.crossings_u[0], arc.crossings_u[1], pxy]) - c) / w
    rows0 = []
    for X, x in zip(pitch_pts, img_pts, strict=True):
        Xh, z = np.append(X, 1.0), np.zeros(3)
        rows0.append(np.concatenate([Xh, z, -x[0] * Xh]))
        rows0.append(np.concatenate([z, Xh, -x[1] * Xh]))
    rows0 = np.array(rows0)
    labels = []
    for k, xv in (("G", xs["G"]), ("6F", xs["6F"])):
        labels.append((np.array([1.0, 0.0, -(xv - x0) / 10.0]), k))
    for k, yv in _Y_LABELS.items():
        labels.append((np.array([0.0, 1.0, -yv / 10.0]), k))
    Hs = []
    for L2 in lines:
        if L2 is L or (abs(L2.d @ L.d) > 0.998 and abs((L2.o - L.o) @ L.n) < 4 * ctx.tol):
            continue
        l_img = _line_h((L2.o - c) / w, L2.n)
        for lp, _name in labels:
            M = np.zeros((3, 9))
            for i in range(3):
                for j in range(3):
                    M[i, 3 * j + i] = l_img[j]
            skew = np.array([[0, -lp[2], lp[1]], [lp[2], 0, -lp[0]], [-lp[1], lp[0], 0]])
            rows = np.vstack([rows0, -skew @ M])
            _, sv, vt = np.linalg.svd(rows)
            if sv[-2] < 1e-9 * sv[0]:  # null space not unique
                continue
            Hs.append(vt[-1].reshape(3, 3))
    if not Hs:
        return end, np.empty((0, 3, 3))
    Hs = np.array(Hs)
    good = np.abs(Hs[:, 2, 2]) > 1e-12  # h33 is the spot's weight in the shifted frame
    Hs = Hs[good] / Hs[good][:, 2, 2][:, None, None]
    return end, _compose(Hs, x0, ctx.size)


def _refine(H0: np.ndarray, anchors: np.ndarray, tpl: Template, fld: Field, size: tuple[int, int]) -> np.ndarray:
    """Least squares on the mask distance; the 4 box corners' image positions are the parameters."""
    P0, ok0 = _project(H0[None], anchors)
    P0 = P0[0]
    if not ok0.all():
        return H0
    cap = 8.0
    w, h = size

    def hof(p: np.ndarray) -> np.ndarray | None:
        return _h_from_4(anchors, P0 + p.reshape(4, 2))

    def resid(p: np.ndarray) -> np.ndarray:
        H = hof(p)
        if H is None or not np.isfinite(H).all():
            return np.full(len(tpl.pts), cap)
        xy, ok = _project(H[None], tpl.pts)
        d, inb = fld.sample(xy[0])
        return np.where(inb & ok[0], np.minimum(d, cap), cap)

    sol = least_squares(resid, np.zeros(8), loss="soft_l1", f_scale=2.0, diff_step=0.05, max_nfev=30, x_scale=4.0)
    if np.abs(sol.x).max() > 0.06 * w:
        return H0
    H = hof(sol.x)
    return H if H is not None and np.isfinite(H).all() else H0


def find_d(
    mask: np.ndarray,
    dist: dict | None = None,
    pitch: PitchModel | None = None,
    prep: tuple[np.ndarray, np.ndarray] | None = None,
    min_confidence: float | None = None,
) -> FinderDetection:
    """Penalty area (box + 6-yard box + D) template fit; falls back to the 2 front-line x D points."""
    t0 = time.perf_counter()
    ctx = _ctx(mask, dist, pitch, prep)
    w, h = ctx.size
    out = FinderDetection(None, size=ctx.size)
    if len(ctx.und) < 80:
        out.reason = "too few line pixels"
        return out
    pm = ctx.pitch
    lines = cc._anchor_lines(mask, ctx.obs, ctx.und, ctx.tol, MAX_LINES_D, min_len_frac=0.04)
    arc_det = cc.detect_from_mask(
        mask, dist, ("pen_arc",), pitch=pm, min_confidence=min_confidence, prep=(ctx.obs, ctx.und)
    )
    arc = arc_det.result
    out.notes = list(arc_det.notes)
    fld = ctx.field()
    tolm = 1.6 * ctx.tol
    tpl_0 = {e: _d_template(pm, e, 1.5).pts for e in ("L", "R")}
    tpl_c = {e: _d_template(pm, e, 0.6) for e in ("L", "R")}
    tpl_f = {e: _d_template(pm, e, 0.2) for e in ("L", "R")}
    cands: list[tuple[str, np.ndarray, float, str]] = []  # (end, H, coarse, source)
    n_hyp = 0
    for end in ("L", "R"):
        parts = [(_gen_from_lines(lines, ctx, end), "lines")]
        out.debug.setdefault("n_lines", len(lines))
        if arc is not None and arc.pitch_ids[0][0] == end:
            _, Ha = _gen_from_arc(arc, lines, ctx)
            parts.append((Ha, "arc"))
        for Hs_raw, src in parts:
            if len(Hs_raw) == 0:
                continue
            Hs = Hs_raw[_orientation_ok(Hs_raw, pm, end, ctx.size)]
            if len(Hs) == 0:
                continue
            n_hyp += len(Hs)
            sc = coarse_scores(Hs, tpl_0[end], fld, tolm)
            top = np.argsort(-sc)[:400]
            sc2 = coarse_scores(Hs[top], tpl_c[end].pts, fld, tolm)
            for i in np.argsort(-sc2)[:60]:
                cands.append((end, Hs[top][i], float(sc2[i]), src))
    out.debug["n_hyp"] = n_hyp
    best = None
    if cands:
        cands.sort(key=lambda c: -c[2])
        # distinct coarse candidates, then refine
        ref: list[dict] = []
        seen: list[tuple[str, np.ndarray]] = []
        for end, H, sc, src in cands:
            kp = _project(H[None], _anchor_pts(pm, end))[0][0]
            if any(e == end and np.abs(kp - k).max() < 6.0 for e, k in seen):
                continue
            seen.append((end, kp))
            Hr = _refine(H, _anchor_pts(pm, end), tpl_c[end], fld, ctx.size)
            ev = evaluate(Hr, tpl_f[end], fld, tolm)
            ev.update(end=end, H=Hr, src=src, coarse=sc, key=_project(Hr[None], _anchor_pts(pm, end))[0][0])
            ref.append(ev)
            if len(ref) >= N_REFINE:
                break
        ref.sort(key=lambda e: -e["f1"])
        best = _accept_d(ref, ctx, out)
    if best is not None:
        out.result = _d_result(best, ctx, pm)
        out.debug.update(H_px=best["H"], template=_d_template(pm, best["end"], 0.2))
    elif arc is not None and arc.confidence > 0:
        out.result = FinderResult(
            "d",
            [_fp(pm, pid, uv, arc.confidence, False) for pid, uv in zip(arc.pitch_ids, arc.points_uv, strict=True)],
            arc.confidence,
            {"source": "arc", "template": out.reason or "abstained"},
        )
        out.reason = ""
    elif not out.reason:
        out.reason = arc_det.reason or "no penalty area found"
    out.seconds = time.perf_counter() - t0
    return out


def _distinct(a: dict, b: dict, w: int) -> bool:
    """Two fits are different answers when they disagree about the template parts that matched the mask.

    Fits that differ only where there is no evidence (a hidden box corner) are not rivals: they are the same
    answer with an uncertain extrapolation, handled per point by ``_d_result``.
    """
    if a["end"] != b["end"]:
        return True
    both = a["match"] & b["inb"]
    if both.sum() < 5:
        return True
    d = np.linalg.norm(a["xy"][both] - b["xy"][both], axis=1)
    return bool(np.percentile(d, 90) > 0.015 * w)


def _accept_d(ref: list[dict], ctx: _Ctx, out: FinderDetection) -> dict | None:
    if not ref:
        out.reason = "no penalty-area hypothesis"
        return None
    b = ref[0]
    w = ctx.size[0]
    n_el = sum(1 for _, v, m in b["elems"] if v >= 0.03 * w and m >= 0.6 * v)
    b["n_elements"] = n_el
    out.notes.append(
        f"d template end={b['end']} src={b['src']} f1={b['f1']:.2f} prec={b['precision']:.2f} "
        f"elements={n_el} vis={b['v']:.0f}px"
    )
    if b["f1"] < D_MIN_F1 or b["precision"] < D_MIN_PRECISION:
        out.reason = f"best template f1 {b['f1']:.2f} / precision {b['precision']:.2f} too low"
        return None
    if n_el < D_MIN_ELEMENTS:
        out.reason = f"only {n_el} template elements supported"
        return None
    runner = next((r for r in ref[1:] if _distinct(b, r, w)), None)
    margin = (b["m"] - runner["m"]) / max(b["m"], 1.0) if runner else 1.0
    b["margin"] = margin
    if margin < D_MIN_MARGIN:
        out.reason = f"ambiguous: a different fit explains all but {100 * margin:.0f}% of the same lines"
        return None
    # same-evidence fits as good as the best: their disagreement says which extrapolated points to distrust
    b["ties"] = [r for r in ref[1:] if r["end"] == b["end"] and r["f1"] >= b["f1"] - D_TIE_F1 and not _distinct(b, r, w)]
    return b


def _d_result(b: dict, ctx: _Ctx, pm: PitchModel) -> FinderResult:
    w, h = ctx.size
    end, H = b["end"], b["H"]
    kp = pm.keypoints()
    fld = ctx.field()
    tpl = _d_template(pm, end, 0.2)
    xy, ok = _project(H[None], tpl.pts)
    d, inb = fld.sample(xy[0])
    match = (d <= 1.6 * ctx.tol) & inb & ok[0]
    conf = float(min(1.0, b["f1"]))
    pts: list[FoundPoint] = []
    dropped = []
    for key in _D_KEYS:
        pid = f"{end}_{key}"
        p, okp = _project(H[None], np.array([kp[pid]]))
        p = p[0, 0]
        if not okp[0, 0] or not (0.01 * w <= p[0] <= 0.99 * w and 0.01 * h <= p[1] <= 0.99 * h):
            continue
        if b.get("ties"):
            others, ok_o = _project(np.array([t["H"] for t in b["ties"]]), np.array([kp[pid]]))
            if (np.linalg.norm(others[:, 0] - p, axis=1) > 0.02 * w).any() or not ok_o.all():
                dropped.append(key)  # equally good fits disagree about where this point is
                continue
        uv = _to_obs(p, ctx.size, ctx.dist)[0]
        if not np.isfinite(uv).all():
            continue
        r_loc = 0.05 * w
        near = (np.linalg.norm(xy[0] - p, axis=1) < r_loc) & inb
        derived = key in _ALWAYS_DERIVED or near.sum() < 8 or match[near].mean() < 0.5
        pts.append(_fp(pm, pid, uv, conf * (0.75 if derived else 1.0), derived))
    info = {
        "source": b["src"],
        "end": end,
        "f1": b["f1"],
        "precision": b["precision"],
        "margin": b.get("margin"),
        "n_elements": b.get("n_elements"),
        "ambiguous_dropped": dropped,
        "elements": {n: [round(v), round(m)] for n, v, m in b["elems"]},
    }
    return FinderResult("d", pts, conf, info)


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def find_points(
    kind: str, bgr: np.ndarray, dist: dict | None = None, pitch: PitchModel | None = None, **kw
) -> FinderDetection:
    """BGR frame -> finder result for ``kind`` ("centre" | "d")."""
    if kind not in FINDER_KINDS:
        raise ValueError(f"unknown finder kind {kind!r}")
    mask = cc.line_mask_for(bgr)
    if kind == "centre":
        return find_centre(mask, dist, pitch, far_mask=far_mask_for(bgr), **kw)
    kw.pop("rig", None)
    return find_d(mask, dist, pitch, **kw)


def draw_overlay(bgr: np.ndarray, det: FinderDetection, dist: dict | None = None, label: str = "") -> np.ndarray:
    """Debug overlay: found points (observed = green, derived = cyan ring), the template under the fit."""
    img = cc.resize_work(bgr).copy()
    h, w = img.shape[:2]
    r = det.result
    if r is None:
        cv2.putText(img, f"{label} ABSTAIN: {det.reason[:80]}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return img
    Hpx, tpl = det.debug.get("H_px"), det.debug.get("template")
    if Hpx is not None and tpl is not None:
        xy, ok = _project(Hpx[None], tpl.pts)
        o = geo.norm_to_px(geo.distort_points(geo.px_to_norm(xy[0] + 0.5, w, h), dist), w, h)
        for i in range(len(tpl.pts) - 1):
            if tpl.same_next[i] and ok[0, i] and ok[0, i + 1] and np.isfinite(o[i : i + 2]).all():
                a, b = np.round(o[i]).astype(int), np.round(o[i + 1]).astype(int)
                if max(abs(a).max(), abs(b).max()) < 4000:
                    cv2.line(img, tuple(a), tuple(b), (255, 160, 0), 1)
    for p in r.points:
        c = (int(round(p.uv[0] * w)), int(round(p.uv[1] * h)))
        if p.derived:
            cv2.circle(img, c, 7, (255, 255, 0), 2)
        else:
            cv2.circle(img, c, 6, (0, 255, 0), 2)
        cv2.putText(img, (p.pitch_id or "").replace("CENTRE_CIRCLE_", "CC_").replace("_PEN_", " "), (c[0] + 8, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(img, f"{label} {r.kind} conf={r.confidence:.2f} n={len(r.points)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return img
