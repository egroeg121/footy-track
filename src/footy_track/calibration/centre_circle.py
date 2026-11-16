# ruff: noqa: PLR0911, PLR0912, PLR0915
"""Find the two points where a pitch line crosses a pitch circle, for calibration labelling.

George labels camera -> pitch calibration GT by placing image <-> pitch point pairs. Many broadcast
frames show only the centre circle and the halfway line, or the penalty "D" and the front edge of the
penalty box. In both cases a long straight line cuts a circle in two points that have *known pitch
coordinates*, so the machine can place them:

* ``centre``: halfway line x circle(0, 0, 9.15)  -> ``CENTRE_CIRCLE_TOP`` (0, +9.15) and
  ``CENTRE_CIRCLE_BOTTOM`` (0, -9.15). The line is a diameter.
* ``pen_arc``: penalty-box front edge x circle(penalty spot, 9.15) -> ``{L,R}_PEN_ARC_{TOP,BOTTOM}``
  (+-(L/2 - 16.5), +-7.31). The line is a chord 5.5 m from the circle centre and the visible arc
  (the "D") lies on the goal-away side of it.

Method (classical, no ML; the line is the anchor, the ellipse is constrained by it)::

    grass mask -> white-line mask (``prelabel.candidates.line_mask``, saturation-relative gate)
    -> thin pixels, undistorted with the labeller's division model k1 when a lens is given
    -> long straight lines (Hough, merged) refined on the undistorted pixels
    -> per anchor line, RANSAC ellipse over the *other* line pixels, accepted only if the ellipse
       cuts the line in two points that sit on line pixels, with the line at a plausible offset from the
       ellipse centre (|d| < 0.4 of the radius = diameter -> centre circle; 0.4..0.9 with every arc
       pixel on the far side from the centre = D)
    -> refit on inliers, arc-coverage / residual / line-support confidence, in-frame checks
    -> intersect line and ellipse -> two points (far = smaller image y).

It abstains (``Detection.reason``) rather than guesses. The pole of the line with respect to the
ellipse is also returned: for the centre circle it is the vanishing point of the pitch length axis (the
tangents at the two points meet there); for the D it is a finite point on the length-axis through the
chord midpoint. Either way the direction ``midpoint -> pole`` is the image of the pitch x axis there.

Conventions match the labeller: points are returned as *observed* (distorted) coordinates normalised to
``[0, 1]^2`` (origin top-left), exactly what ``correspondences[].image`` stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import (
    CIRCLE_RADIUS,
    PEN_BOX_DEPTH,
    PEN_SPOT_DIST,
    PitchModel,
)
from footy_track.calibration.prelabel import candidates as cand

#: working-image long edge (px); masks and tolerances scale with the width
WORK_EDGE = 1280
#: |distance of the line from the ellipse centre| / radius (in the circle's own frame)
CENTRE_MAX_D = 0.40
ARC_MIN_D, ARC_MAX_D = 0.40, 0.92
#: minimum fraction of the (visible, correct-side) ellipse outline covered by line pixels
MIN_COVERAGE = {"centre": 0.45, "pen_arc": 0.55}
#: abstain below this confidence (per kind; the D is the harder, shorter arc)
MIN_CONFIDENCE = {"centre": 0.55, "pen_arc": 0.7}
#: the curve must run into both crossings: nearest arc pixel within this fraction of the chord (centre circle only; a
#: D cut by a player legitimately has gaps there)
MAX_END_GAP = {"centre": 0.15, "pen_arc": 1.0}
MIN_INLIERS = 25
#: distinct RANSAC hypotheses per (line, kind) taken to the full check
N_HYP = 4
KINDS = ("centre", "pen_arc")
# arc half-span of the D around its axis: points with x >= box_front - spot, cos = 5.5 / 9.15
_D_COS = (PEN_BOX_DEPTH - PEN_SPOT_DIST) / CIRCLE_RADIUS


#: set to a list to collect why hypotheses were dropped (debugging aid)
_trace: list[str] | None = None


def _why(msg: str) -> None:
    if _trace is not None:
        _trace.append(msg)


@dataclass
class Ellipse:
    """An ellipse in px of the (undistorted) working plane: centre, semi-axes (a, b), rotation (rad)."""

    centre: tuple[float, float]
    axes: tuple[float, float]
    angle: float

    def matrix(self) -> np.ndarray:
        """Unit-circle frame transform: ``q = M (p - c)``."""
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array(
            [
                [c / self.axes[0], s / self.axes[0]],
                [-s / self.axes[1], c / self.axes[1]],
            ]
        )

    def to_dict(self) -> dict:
        return {
            "centre": list(self.centre),
            "axes": list(self.axes),
            "angle_deg": float(np.rad2deg(self.angle)),
        }


@dataclass
class CirclePoints:
    kind: str  # "centre" | "pen_arc"
    #: observed normalised image points, ``[far, near]`` (far = smaller image y)
    points_uv: np.ndarray
    pitch_ids: tuple[str, str]
    confidence: float
    #: pole of the line wrt the ellipse and the chord midpoint, undistorted normalised image coords
    #: (homogeneous pole ``(x, y, w)``; w ~ 0 = vanishing point at infinity)
    pole_h: np.ndarray
    midpoint_u: np.ndarray
    ellipse: Ellipse
    line: tuple[np.ndarray, np.ndarray]  # endpoints, undistorted working px
    info: dict = field(default_factory=dict)
    #: the anchor line (undistorted working px) and where it crosses the ellipse, far first (for the finders)
    anchor: _Line | None = None
    crossings_u: np.ndarray | None = None
    #: sign of the arc side along the anchor normal (D only)
    arc_side: float = 0.0

    def correspondences(self) -> list[dict]:
        """Records in the labeller's ``correspondences`` shape (``image`` + ``pitch_id``)."""
        return [
            {"image": [float(u), float(v)], "pitch_id": pid}
            for (u, v), pid in zip(self.points_uv, self.pitch_ids, strict=False)
        ]

    def axis_direction(self, w: float = 16.0, h: float = 9.0) -> np.ndarray:
        """Unit image direction (px-scaled by ``(w, h)``) of the pitch length axis at the chord midpoint."""
        p = self.pole_h
        d = (p[:2] - self.midpoint_u * p[2]) * np.array([w, h])
        n = np.linalg.norm(d)
        return d / n if n > 0 else d


@dataclass
class Detection:
    result: CirclePoints | None
    reason: str = ""  # why it abstained
    #: best rejected/accepted hypotheses for debugging
    notes: list[str] = field(default_factory=list)
    size: tuple[int, int] = (0, 0)  # working (w, h)


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------


def line_mask_for(bgr: np.ndarray) -> np.ndarray:
    """White-line mask of the working-size frame (bool, h x w): what ``detect_from_mask`` consumes."""
    w = resize_work(bgr)
    grass = cand.grass_mask(w)
    return cand.line_mask(w, grass, with_far=False, sat_rel=cand.LINE_SAT_REL)


def resize_work(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = min(1.0, WORK_EDGE / max(h, w))
    return (
        bgr
        if s >= 1.0
        else cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    )


# --------------------------------------------------------------------------
# geometry helpers (all in px of the undistorted working plane)
# --------------------------------------------------------------------------


def _sampson(P: np.ndarray, E: Ellipse) -> np.ndarray:
    """First-order geometric distance (px) of points to the ellipse outline."""
    M = E.matrix()
    q = (P - np.asarray(E.centre)) @ M.T
    g = np.einsum("ij,ij->i", q, q) - 1.0
    grad = 2.0 * q @ M  # d g / d p
    return np.abs(g) / np.maximum(np.linalg.norm(grad, axis=1), 1e-9)


def _fit_ellipse(pts: np.ndarray) -> Ellipse | None:
    """``cv2.fitEllipse`` -> ``Ellipse`` (``None`` when degenerate)."""
    try:
        (cx, cy), (w, h), ang = cv2.fitEllipse(np.asarray(pts, np.float32))
    except cv2.error:
        return None
    if not (np.isfinite([cx, cy, w, h, ang]).all() and min(w, h) > 1.0):
        return None
    return Ellipse(
        (float(cx), float(cy)), (float(w) / 2.0, float(h) / 2.0), float(np.deg2rad(ang))
    )


def _conic(E: Ellipse) -> np.ndarray:
    M = E.matrix()
    c = np.asarray(E.centre)
    A = M.T @ M
    b = -A @ c
    k = c @ A @ c - 1.0
    return np.array(
        [[A[0, 0], A[0, 1], b[0]], [A[1, 0], A[1, 1], b[1]], [b[0], b[1], k]]
    )


def _line_ellipse(o: np.ndarray, d: np.ndarray, E: Ellipse):
    """Intersections of the line ``o + s d`` with the ellipse: ``(s values, d_n)`` or ``(None, d_n)``.

    ``d_n`` is the line's signed distance from the ellipse centre in the circle's own frame (radius 1),
    positive on the ``+n`` side where ``n`` = ``(-d_y, d_x)``.
    """
    M = E.matrix()
    qo = M @ (o - np.asarray(E.centre))
    qd = M @ d
    a = qd @ qd
    b = qo @ qd
    c = qo @ qo - 1.0
    disc = b * b - a * c
    # distance of the line from the unit-circle centre (signed along the image normal)
    cross = qo[0] * qd[1] - qo[1] * qd[0]
    d_n = cross / np.sqrt(a)
    if disc <= 0:
        return None, float(d_n)
    r = np.sqrt(disc)
    return np.array([(-b - r) / a, (-b + r) / a]), float(d_n)


def _phi_bins(P: np.ndarray, E: Ellipse, nbin: int) -> np.ndarray:
    q = (P - np.asarray(E.centre)) @ E.matrix().T
    ang = np.arctan2(q[:, 1], q[:, 0])
    return ((ang + np.pi) / (2 * np.pi) * nbin).astype(int) % nbin


# --------------------------------------------------------------------------
# the detector
# --------------------------------------------------------------------------


@dataclass
class _Line:
    o: np.ndarray  # a point on the line
    d: np.ndarray  # unit direction (pointing "up" the image: d_y <= 0)
    s0: float
    s1: float  # extent of its pixels along d
    support: float
    length: float

    @property
    def n(self) -> np.ndarray:
        return np.array([-self.d[1], self.d[0]])

    def p(self, s: float) -> np.ndarray:
        return self.o + s * self.d


def _prep_points(mask: np.ndarray, dist: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Thin line pixels as (observed px, undistorted px) arrays, both (N, 2) in the working plane."""
    h, w = mask.shape
    ys, xs = np.nonzero(cand._thin(mask))
    obs = np.stack([xs, ys], 1).astype(np.float64)
    if len(obs) == 0:
        return obs, obs
    if geo.distortion_params(dist) is None:
        return obs, obs
    norm = geo.px_to_norm(obs + 0.5, w, h)
    und = geo.norm_to_px(geo.undistort_points(norm, dist), w, h) - 0.5
    return obs, und


def _anchor_lines(
    mask: np.ndarray,
    obs: np.ndarray,
    und: np.ndarray,
    tol: float,
    max_lines: int,
    min_len_frac: float = 0.07,
) -> list[_Line]:
    """Long straight lines: Hough on the observed mask, re-fitted on the undistorted pixels."""
    h, w = mask.shape
    diag = float(np.hypot(h, w))
    segs = cand.detect_lines(mask, min_len_frac=min_len_frac)
    out: list[_Line] = []
    for L in segs:
        # pixels near the observed segment (generous band), then robust line fit on the undistorted ones
        d0 = (L.p1 - L.p0) / max(np.linalg.norm(L.p1 - L.p0), 1e-9)
        n0 = np.array([-d0[1], d0[0]])
        rel = obs - L.p0
        s = rel @ d0
        t = rel @ n0
        near = (np.abs(t) < 3.5 * tol) & (s > -0.05 * L.length) & (s < 1.05 * L.length)
        if near.sum() < 15:
            continue
        pts = und[near].astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        d = np.array([vx, vy], np.float64)
        if d[1] > 0 or (d[1] == 0 and d[0] < 0):
            d = -d
        o = np.array([x0, y0], np.float64)
        n = np.array([-d[1], d[0]])
        # refine: inliers of the refit line from the whole undistorted cloud near the segment
        rel = und - o
        t = rel @ n
        s = rel @ d
        sel = (
            (np.abs(t) < 1.6 * tol)
            & (s > (s[near].min() - 0.05 * L.length))
            & (s < (s[near].max() + 0.05 * L.length))
        )
        if sel.sum() < 15:
            continue
        vx, vy, x0, y0 = cv2.fitLine(
            und[sel].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
        ).ravel()
        d = np.array([vx, vy], np.float64)
        if d[1] > 0 or (d[1] == 0 and d[0] < 0):
            d = -d
        o = np.array([x0, y0], np.float64)
        n = np.array([-d[1], d[0]])
        t = (und - o) @ n
        s = (und - o) @ d
        sel = (np.abs(t) < 1.6 * tol) & (s > s[sel].min() - 1) & (s < s[sel].max() + 1)
        ss = s[sel]
        length = float(ss.max() - ss.min())
        if length < min_len_frac * diag:
            continue
        # support: fraction of the span (3 px steps) with a pixel within the band
        bins = np.unique(np.floor((ss - ss.min()) / 4.0).astype(int))
        support = float(len(bins) / max(1, int(length / 4.0) + 1))
        out.append(_Line(o, d, float(ss.min()), float(ss.max()), support, length))
    out.sort(key=lambda L: -(L.length * min(1.0, L.support)))
    # drop near-duplicates (same line found twice)
    keep: list[_Line] = []
    for L in out:
        dup = False
        for K in keep:
            ang = np.arccos(np.clip(abs(L.d @ K.d), 0, 1))
            mid = L.p(0.5 * (L.s0 + L.s1))
            if ang < np.deg2rad(4) and abs((mid - K.o) @ K.n) < 3 * tol:
                dup = True
                break
        if not dup:
            keep.append(L)
    return keep[:max_lines]


def _line_has_pixel_near(und: np.ndarray, L: _Line, s: float, tol: float) -> bool:
    p = L.p(s)
    return bool((np.linalg.norm(und - p, axis=1) < 3.0 * tol).any())


def _attached(
    und: np.ndarray, obs: np.ndarray, L: _Line, tol: float, size: tuple[int, int]
) -> np.ndarray:
    """Per pixel: id (> 0) of the connected stroke (off the line) it belongs to if that stroke comes within a few px of
    ``L``, else 0.

    The anchor line explains the pixels in its own band; whatever curve ends on it (the two halves of a
    circle, the D) is a stroke that touches that band, so ellipse samples are drawn from those strokes
    instead of from the whole line cloud.
    """
    w, h = size
    t = (und - L.o) @ L.n
    s = (und - L.o) @ L.d
    off = np.abs(t) >= 2.2 * tol
    img = np.zeros((h, w), np.uint8)
    xi = np.clip(np.round(obs[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(obs[:, 1]).astype(int), 0, h - 1)
    img[yi[off], xi[off]] = 1
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(img, connectivity=8)
    labs = lab[yi, xi]
    in_span = (s > L.s0 - 0.1 * L.length) & (s < L.s1 + 0.1 * L.length)
    touch = off & (np.abs(t) < 4.5 * tol) & in_span
    # a curve cut by a player is several strokes, only the end ones touch the line: also take strokes lying
    # within a quarter of the line's length of it (the D is at most ~0.35 of its chord deep)
    nearby = off & (np.abs(t) < 0.25 * L.length) & in_span
    good = np.unique(np.concatenate([labs[touch], labs[nearby]]))
    good = good[good > 0]
    return np.where(off & np.isin(labs, good), labs, 0)


def _fit_for_line(
    und: np.ndarray,
    obs: np.ndarray,
    L: _Line,
    tol: float,
    size: tuple[int, int],
    rng: np.random.Generator,
    n_iter: int = 600,
):
    """Best ellipse hypotheses constrained by the anchor line ``L``: list of dicts (one per kind)."""
    w, h = size
    t_all = (und - L.o) @ L.n
    off = np.abs(t_all) >= 2.2 * tol
    cloud = und[off]
    if len(cloud) < 40:
        return []
    att = _attached(und, obs, L, tol, size)[off]
    if (att > 0).sum() < 25:
        return []
    if len(cloud) > 3000:
        keep = rng.choice(len(cloud), 3000, replace=False)
        cloud, att = cloud[keep], att[keep]
    ct = (cloud - L.o) @ L.n
    seeds = cloud[att > 0]
    # strokes that touch the line, biggest first (>= 12 px: a stroke needs a few points to carry a curve)
    ids, counts = np.unique(att[att > 0], return_counts=True)
    strokes = [cloud[att == i] for i, c in zip(ids, counts, strict=False) if c >= 12]
    if len(seeds) < 25 or not strokes:
        return []
    sizes = np.array([len(x) for x in strokes], float)
    prob = np.sqrt(sizes) / np.sqrt(sizes).sum()
    best: dict[str, tuple] = {}
    for it in range(n_iter):
        if it % 4 == 0:  # anywhere on the nearby strokes
            pick = seeds[rng.choice(len(seeds), 6, replace=False)]
        elif (
            it % 4 == 2 and len(strokes) > 1
        ):  # two strokes: a curve cut by a player, or the two halves of a circle
            i, j = rng.choice(len(strokes), 2, replace=False, p=prob)
            a, b = strokes[i], strokes[j]
            pick = np.concatenate(
                [
                    a[rng.choice(len(a), min(3, len(a)), replace=False)],
                    b[rng.choice(len(b), min(3, len(b)), replace=False)],
                ]
            )
            if len(pick) < 6:
                continue
        else:  # points of ONE stroke: a curve
            st = strokes[rng.choice(len(strokes), p=prob)]
            if len(st) < 6:
                continue
            pick = st[rng.choice(len(st), 6, replace=False)]
        E = _fit_ellipse(pick)
        if E is None:
            continue
        lo, hi = sorted(E.axes)
        if lo < 0.012 * w or hi > 0.55 * w or lo < 0.08 * hi:
            continue
        sv, d_n = _line_ellipse(L.o, L.d, E)
        if sv is None:
            continue
        ad = abs(d_n)
        kind = (
            "centre" if ad < CENTRE_MAX_D else ("pen_arc" if ad < ARC_MAX_D else None)
        )
        if kind is None:
            continue
        # chord must land on the line's pixels
        if not (
            L.s0 - 0.25 * L.length <= sv.min() and sv.max() <= L.s1 + 0.25 * L.length
        ):
            continue
        inl = _sampson(cloud, E) < tol
        if kind == "pen_arc":
            # the arc is on the far side of the line from the centre: count pixels on that side only
            inl &= (ct * -np.sign(d_n)) > 0
        n_in = int(inl.sum())
        top = best.setdefault(kind, [])
        # keep the few best DISTINCT hypotheses per kind: the largest inlier count is not always the circle
        if any(
            np.linalg.norm(np.subtract(E.centre, o[1].centre)) < 0.03 * w
            and abs(sum(E.axes) - sum(o[1].axes)) < 0.05 * w
            for o in top
        ):
            for k_, o in enumerate(top):
                if (
                    np.linalg.norm(np.subtract(E.centre, o[1].centre)) < 0.03 * w
                    and n_in > o[0]
                ):
                    top[k_] = (n_in, E, d_n)
            continue
        top.append((n_in, E, d_n))
        top.sort(key=lambda o: -o[0])
        del top[N_HYP:]
    res = []
    for kind, tops in best.items():
        for n_in, E, _d_n in tops:
            if n_in < MIN_INLIERS:
                _why(f"{kind}: ransac hypothesis has {n_in} inliers (<{MIN_INLIERS})")
                continue
            res.append(_refine_and_check(und, L, kind, E, tol, size))
    return [r for r in res if r is not None]


def _refine_and_check(
    und, L: _Line, kind: str, E: Ellipse, tol: float, size: tuple[int, int]
):
    w, h = size
    off = np.abs((und - L.o) @ L.n) >= 2.2 * tol
    pool = und[off]
    for _ in range(3):
        sv, d_n = _line_ellipse(L.o, L.d, E)
        if sv is None:
            _why(f"{kind}: refit lost the line crossing")
            return None
        inl = _sampson(pool, E) < 1.3 * tol
        if kind == "pen_arc":
            side = -np.sign(d_n)
            inl &= ((pool - L.o) @ L.n) * side > 0
        if inl.sum() < MIN_INLIERS:
            _why(f"{kind}: only {int(inl.sum())} refit inliers")
            return None
        E = _fit_ellipse(pool[inl])
        if E is None:
            _why(f"{kind}: degenerate refit")
            return None
    sv, d_n = _line_ellipse(L.o, L.d, E)
    if sv is None:
        _why(f"{kind}: final fit misses the line")
        return None
    ad = abs(d_n)
    kind2 = "centre" if ad < CENTRE_MAX_D else ("pen_arc" if ad < ARC_MAX_D else None)
    if kind2 != kind:
        _why(f"{kind}: refit moved the line offset to {d_n:.2f}")
        return None
    lo, hi = sorted(E.axes)
    if lo < 0.012 * w or hi > 0.6 * w or lo < 0.08 * hi:
        _why(f"{kind}: refit ellipse implausible axes {lo:.0f}/{hi:.0f}")
        return None
    resid = _sampson(pool[inl], E)
    pts = pool[inl]
    # coverage of the ellipse outline in the circle's own angle, over bins that are in frame
    nbin = 72
    ang = (np.arange(nbin) + 0.5) / nbin * 2 * np.pi - np.pi
    unit = np.stack([np.cos(ang), np.sin(ang)], 1)
    outline = np.asarray(E.centre) + unit @ np.linalg.inv(E.matrix()).T
    in_frame = (
        (outline[:, 0] > 2)
        & (outline[:, 0] < w - 3)
        & (outline[:, 1] > 2)
        & (outline[:, 1] < h - 3)
    )
    # drop outline bins swallowed by the anchor line band (pixels there were removed)
    near_line = np.abs((outline - L.o) @ L.n) < 3.5 * tol
    relevant = in_frame & ~near_line
    if kind == "pen_arc":
        side = -np.sign(d_n)
        relevant &= ((outline - L.o) @ L.n) * side > 0
    covered = np.zeros(nbin, bool)
    b = _phi_bins(pts, E, nbin)
    covered[b] = True
    # tolerate single-bin gaps (kit/ball/lines painted thin)
    cov_g = covered | (np.roll(covered, 1) & np.roll(covered, -1))
    n_rel = int(relevant.sum())
    if n_rel < 6:
        _why(f"{kind}: only {n_rel} outline bins in frame/off the line")
        return None
    coverage = float((cov_g & relevant).sum() / n_rel)
    # the visible (in-frame) share of the whole circle that the hypothesis relies on
    span_share = n_rel / nbin
    # line support over the chord between the two crossings
    s_a, s_b = float(sv.min()), float(sv.max())
    line_cov = _chord_support(und, L, s_a, s_b, tol)
    # intersections must be inside the frame
    P = np.stack([L.p(s_a), L.p(s_b)])
    margin = 0.04
    inside = (
        (P[:, 0] > -margin * w)
        & (P[:, 0] < (1 + margin) * w)
        & (P[:, 1] > -margin * h)
        & (P[:, 1] < (1 + margin) * h)
    ).all()
    # the curve must actually run into both crossings: nearest arc pixel to each, relative to the chord
    chord = max(float(s_b - s_a), 1.0)
    end_gap = float(
        max(
            np.min(np.linalg.norm(pts - P[0], axis=1)),
            np.min(np.linalg.norm(pts - P[1], axis=1)),
        )
        / chord
    )
    return {
        "end_gap": end_gap,
        "kind": kind,
        "E": E,
        "d_n": d_n,
        "sv": (s_a, s_b),
        "coverage": coverage,
        "span_share": span_share,
        "resid": float(resid.mean()),
        "n_in": int(inl.sum()),
        "line_cov": line_cov,
        "inside": bool(inside),
        "line": L,
        "pts": pts,
    }


def _chord_support(
    und: np.ndarray, L: _Line, s_a: float, s_b: float, tol: float
) -> float:
    rel = und - L.o
    t = rel @ L.n
    s = rel @ L.d
    sel = (np.abs(t) < 1.8 * tol) & (s >= s_a - 2) & (s <= s_b + 2)
    if s_b - s_a < 4:
        return 0.0
    bins = np.unique(np.floor((s[sel] - s_a) / 4.0).astype(int))
    nb = int((s_b - s_a) / 4.0) + 1
    return float(min(1.0, len(bins) / nb))


def _score(h: dict) -> float:
    """Confidence in [0, 1]."""
    tol_resid = 1.0 - min(1.0, h["resid"] / 2.5)
    cov = min(1.0, h["coverage"] / 0.9)
    return float(
        cov
        * (0.5 + 0.5 * tol_resid)
        * min(1.0, h["line_cov"] / 0.85)
        * min(1.0, h["span_share"] / 0.25) ** 0.5
    )


def detect_from_mask(
    mask: np.ndarray,
    dist: dict | None = None,
    kinds: tuple[str, ...] = KINDS,
    seed: int = 0,
    max_lines: int = 5,
    pitch: PitchModel | None = None,
    min_confidence: float | None = None,
    prep: tuple[np.ndarray, np.ndarray] | None = None,
) -> Detection:
    """Detect from a white-line mask (``line_mask_for``); ``dist`` is the optional labeller lens dict.

    ``prep`` = ``_prep_points(mask, dist)`` when the caller already has it.
    """
    h, w = mask.shape
    det = Detection(None, size=(w, h))
    tol = max(2.5, w / 450)
    obs, und = prep if prep is not None else _prep_points(mask, dist)
    if len(und) < 80:
        det.reason = "too few line pixels"
        return det
    lines = _anchor_lines(mask, obs, und, tol, max_lines)
    if not lines:
        det.reason = "no long straight line"
        return det
    rng = np.random.default_rng(seed)
    hyps = []
    for L in lines:
        for hyp in _fit_for_line(und, obs, L, tol, (w, h), rng):
            hyp["conf"] = _score(hyp)
            hyps.append(hyp)
    hyps = [x for x in hyps if x["kind"] in kinds]
    if not hyps:
        det.reason = "no ellipse consistent with a long line"
        return det
    hyps.sort(key=lambda x: -x["conf"])
    for x in hyps[:6]:
        det.notes.append(
            f"{x['kind']} conf={x['conf']:.2f} cov={x['coverage']:.2f} resid={x['resid']:.2f} "
            f"line_cov={x['line_cov']:.2f} share={x['span_share']:.2f} inside={x['inside']} d_n={x['d_n']:.2f}"
        )
    best = None
    for x in hyps:
        if x["coverage"] < MIN_COVERAGE[x["kind"]]:
            continue
        if x["line_cov"] < 0.6 or not x["inside"]:
            continue
        if x["conf"] < (
            MIN_CONFIDENCE[x["kind"]] if min_confidence is None else min_confidence
        ):
            continue
        if x["end_gap"] > MAX_END_GAP[x["kind"]]:
            continue
        best = x
        break
    if best is None:
        top = hyps[0]
        why = []
        if top["coverage"] < MIN_COVERAGE[top["kind"]]:
            why.append(f"arc coverage {top['coverage']:.2f}")
        if top["line_cov"] < 0.6:
            why.append(f"line support {top['line_cov']:.2f}")
        if not top["inside"]:
            why.append("crossings outside frame")
        if not why:
            why.append(f"confidence {top['conf']:.2f} / end gap {top['end_gap']:.2f}")
        det.reason = f"best {top['kind']} hypothesis rejected: " + ", ".join(why)
        return det
    det.result = _finish(best, dist, (w, h), pitch or PitchModel())
    return det


def _finish(
    x: dict, dist: dict | None, size: tuple[int, int], pitch: PitchModel
) -> CirclePoints:
    w, h = size
    L: _Line = x["line"]
    E: Ellipse = x["E"]
    s_a, s_b = x["sv"]
    P = np.stack([L.p(s_a), L.p(s_b)])  # undistorted working px
    order = np.argsort(P[:, 1])  # far (smaller y) first
    P = P[order]
    kind = x["kind"]
    if kind == "centre":
        ids = ("CENTRE_CIRCLE_TOP", "CENTRE_CIRCLE_BOTTOM")
    else:
        # the arc lies on the side of the line away from the circle centre; with the far side up and
        # +x to the image right (camera on the -y side), an arc on the image-right means the L end
        side = -np.sign(x["d_n"])  # arc side along +n
        n = L.n
        arc_dir = side * n
        right = arc_dir[0] > 0 if abs(arc_dir[0]) > 1e-9 else True
        end = "L" if right else "R"
        ids = (f"{end}_PEN_ARC_TOP", f"{end}_PEN_ARC_BOTTOM")
    obs_norm = geo.distort_points(geo.px_to_norm(P + 0.5, w, h), dist)
    # pole of the line wrt the ellipse, in undistorted normalised coords
    C = _conic(E)
    ln = np.array([L.n[0], L.n[1], -(L.n @ L.o)])
    pole = np.linalg.solve(C, ln)
    S = np.array([[1.0 / w, 0, 0], [0, 1.0 / h, 0], [0, 0, 1]])
    pole_n = S @ pole
    pole_n = pole_n / (np.linalg.norm(pole_n) or 1.0)
    mid_n = geo.px_to_norm(P.mean(axis=0) + 0.5, w, h)
    pole_h = np.array([pole_n[0], pole_n[1], pole_n[2]])
    # homogeneous pole in normalised coords: (x, y, w) with x/w the image point
    return CirclePoints(
        kind=kind,
        points_uv=obs_norm,
        pitch_ids=ids,
        confidence=float(x["conf"]),
        pole_h=pole_h,
        midpoint_u=mid_n,
        ellipse=E,
        line=(L.p(L.s0), L.p(L.s1)),
        anchor=L,
        crossings_u=P,
        arc_side=float(-np.sign(x["d_n"])) if kind == "pen_arc" else 0.0,
        info={
            k: x[k]
            for k in (
                "coverage",
                "resid",
                "line_cov",
                "span_share",
                "n_in",
                "d_n",
                "end_gap",
            )
        },
    )


def detect_circle_points(
    bgr: np.ndarray, dist: dict | None = None, kinds: tuple[str, ...] = KINDS, **kw
) -> Detection:
    """BGR frame -> the two line x circle crossing points, or an abstention with a reason."""
    return detect_from_mask(line_mask_for(bgr), dist, kinds, **kw)


def detect_centre_circle(bgr: np.ndarray, dist: dict | None = None, **kw) -> Detection:
    return detect_circle_points(bgr, dist, ("centre",), **kw)


def detect_pen_arc(bgr: np.ndarray, dist: dict | None = None, **kw) -> Detection:
    return detect_circle_points(bgr, dist, ("pen_arc",), **kw)


# --------------------------------------------------------------------------
# drawing (debug)
# --------------------------------------------------------------------------


def draw_overlay(
    bgr: np.ndarray, det: Detection, dist: dict | None = None
) -> np.ndarray:
    """Debug overlay on a working-size copy: line, ellipse, the two points (far = green, near = orange)."""
    img = resize_work(bgr).copy()
    h, w = img.shape[:2]
    r = det.result
    if r is None:
        cv2.putText(
            img,
            "ABSTAIN: " + det.reason[:90],
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return img

    def to_obs(p):
        return geo.norm_to_px(
            geo.distort_points(geo.px_to_norm(np.atleast_2d(p) + 0.5, w, h), dist), w, h
        )

    t = np.linspace(0, 2 * np.pi, 180)
    E = r.ellipse
    outline = (
        np.asarray(E.centre)
        + np.stack([np.cos(t), np.sin(t)], 1) @ np.linalg.inv(E.matrix()).T
    )
    cv2.polylines(
        img,
        [np.round(to_obs(outline)).astype(np.int32).reshape(-1, 1, 2)],
        True,
        (255, 160, 0),
        1,
    )
    a, b = (to_obs(p)[0] for p in r.line)
    cv2.line(
        img,
        tuple(np.round(a).astype(int)),
        tuple(np.round(b).astype(int)),
        (0, 200, 255),
        1,
    )
    for (u, v), col in zip(r.points_uv, ((0, 255, 0), (0, 140, 255)), strict=False):
        cv2.circle(img, (int(round(u * w)), int(round(v * h))), 6, col, 2)
    cv2.putText(
        img,
        f"{r.kind} {r.pitch_ids[0]} conf={r.confidence:.2f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    return img
