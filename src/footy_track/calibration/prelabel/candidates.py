"""Classical (no ML) pitch-feature candidates for pre-labelling.

grass mask (HSV) -> white-line mask inside the grass (top-hat on brightness,
low saturation) -> Hough segments merged into long lines -> candidate points:

* ``line_x``  intersections of line pairs that land on line pixels,
* ``end_on``  a line's endpoint that ends on another line (T-junctions),
* ``ellipse_x`` intersections of fitted ellipses (circle / arcs) with lines,
* ``spot``    small compact bright blobs (penalty / centre spots).

Everything works on a working image of at most ``MAX_EDGE`` px on the long
edge; candidate coordinates are normalised ``[0,1]`` (u right, v down) so they
are resolution independent, and ``xy_px`` is in working-image pixels.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_EDGE = 1568  # Claude's image long-edge limit before downscaling


@dataclass
class Candidate:
    id: int
    kind: str  # line_x | end_on | ellipse_x | spot
    xy_px: tuple[float, float]  # working-image pixels
    uv: tuple[float, float]  # normalised
    support: float = 0.0  # 0..1 local line-pixel density
    refs: list[str] = field(default_factory=list)  # e.g. ["L3", "L5"] / ["E1","L2"]

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "xy_px": [round(v, 2) for v in self.xy_px],
            "uv": [round(v, 5) for v in self.uv], "support": round(self.support, 3),
            "refs": self.refs,
        }


@dataclass
class Line:
    id: int
    p0: np.ndarray  # endpoints, working px
    p1: np.ndarray
    support: float

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def dir(self) -> np.ndarray:
        d = self.p1 - self.p0
        return d / max(np.linalg.norm(d), 1e-9)


@dataclass
class Ellipse:
    id: int
    centre: tuple[float, float]
    axes: tuple[float, float]  # full axis lengths (w, h) as OpenCV
    angle: float
    residual: float
    n_pix: int

    def cv(self):
        return (self.centre, self.axes, self.angle)

    def polyline(self, n: int = 180) -> np.ndarray:
        return cv2.ellipse2Poly(
            (int(round(self.centre[0])), int(round(self.centre[1]))),
            (int(round(self.axes[0] / 2)), int(round(self.axes[1] / 2))),
            int(round(self.angle)), 0, 360, 360 // n,
        ).astype(np.float64)


@dataclass
class Detection:
    """Everything detected on one frame, in working-image pixels."""

    work_bgr: np.ndarray
    scale: float  # working px per original px
    grass: np.ndarray  # bool mask
    lines_mask: np.ndarray  # bool mask (white lines inside grass)
    lines: list[Line]
    ellipses: list[Ellipse]
    candidates: list[Candidate]
    far_line: np.ndarray | None = None  # bool mask of the pale far-touchline strip (subset of lines_mask)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.work_bgr.shape[:2]
        return w, h


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------


def resize_to_work(frame: np.ndarray, max_edge: int = MAX_EDGE) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    s = min(1.0, max_edge / max(h, w))
    if s < 1.0:
        frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return frame, s


#: Frames where the HSV grass mask leaks over dark stands / the advertising apron (their hue is
#: green-ish too) are re-masked with a saturation gate relative to the pitch's own colour.
GRASS_SAT_GATE = 0.58
FAR_MAX_Y_FRAC = 0.6
FAR_MAX_V_RATIO = 1.35  # median V of the strip / the grass's (faint touchline ~1.1-1.2, board base 1.6+)
GRASS_MIN_V = 0.5  # of the pitch's median V: loose, so shadows on the pitch survive


def _grass_pipeline(bgr: np.ndarray, gate: tuple[float, float] | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    g = ((h >= 30) & (h <= 95) & (s >= 40) & (v >= 35))
    if gate is not None:
        mv, ms = gate
        g &= (s >= GRASS_SAT_GATE * ms) & (v >= GRASS_MIN_V * mv)
    g = g.astype(np.uint8)
    k = max(3, bgr.shape[1] // 160)
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    big = max(15, bgr.shape[1] // 40)
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (big, big)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
    if n <= 1:
        return np.zeros(g.shape, bool)
    keep = np.zeros(g.shape, np.uint8)
    min_area = 0.02 * g.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    # fill holes (players, lines) without filling the stands: flood from border
    inv = (1 - keep).astype(np.uint8)
    ff = inv.copy()
    mask = np.zeros((g.shape[0] + 2, g.shape[1] + 2), np.uint8)
    for seed in [(0, 0), (g.shape[1] - 1, 0), (0, g.shape[0] - 1), (g.shape[1] - 1, g.shape[0] - 1)]:
        if ff[seed[1], seed[0]] == 1:
            cv2.floodFill(ff, mask, seed, 2)
    holes = ff == 1
    keep[holes] = 1
    return keep.astype(bool)


def pitch_colour(bgr: np.ndarray, grass: np.ndarray) -> tuple[float, float] | None:
    """(median V, median S) of the grass in the lower half of the frame (where it surely is)."""
    h = bgr.shape[0]
    low = grass.copy()
    low[: h // 2] = False
    ref = low if low.sum() > 0.01 * grass.size else grass
    if ref.sum() < 500:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[..., 2][ref])), float(np.median(hsv[..., 1][ref]))


def grass_mask(bgr: np.ndarray) -> np.ndarray:
    """Pitch region: green pixels, closed/filled so lines and players inside it
    belong to the pitch. Returns bool (h, w).

    The plain hue test also accepts the dark, green-tinted stands and the advertising apron
    behind the far touchline (on a high wide camera that pushes the "grass" boundary up past
    the hoardings and lets the far touchline fit onto them), so when the pitch colour is known
    the mask is redone with a saturation / brightness gate relative to it; frames the gate
    does not change keep the original mask."""
    g0 = _grass_pipeline(bgr)
    gate = pitch_colour(bgr, g0)
    if gate is None:
        return g0
    g1 = _grass_pipeline(bgr, gate)
    if g1.sum() < 0.3 * max(g0.sum(), 1) or g1.sum() > 0.995 * g0.sum():
        return g0  # gate removed (nearly) everything, or nothing worth it
    return g1


def line_mask(bgr: np.ndarray, grass: np.ndarray, with_far: bool = True) -> np.ndarray:
    """White pitch markings: bright, low-saturation, locally brighter than
    their surroundings (top-hat), inside the grass region."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    w = bgr.shape[1]
    kk = max(9, (w // 70) | 1)  # > line width, < blobs
    th = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    # adaptive: threshold relative to the grass's own top-hat noise
    ref = th[grass] if grass.any() else th.ravel()
    thr = max(16.0, float(np.percentile(ref, 90)) * 1.5)
    m = (th >= thr) & (s <= 100) & (v >= 85)
    # lines must sit inside the grass, but not at its very boundary (stands,
    # advertising boards) : erode the region a little
    e = max(3, w // 300)
    inner = cv2.erode(grass.astype(np.uint8), np.ones((2 * e + 1, 2 * e + 1), np.uint8)).astype(bool)
    m &= inner
    m8 = m.astype(np.uint8)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # players' white kit: blobs thicker than any line survive an opening with a
    # disk wider than the widest line; remove them (and a margin around them)
    d = max(11, w // 100) | 1
    blobs = cv2.morphologyEx(m8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    blobs = cv2.dilate(blobs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    m8[blobs > 0] = 0
    # drop specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    out = np.zeros_like(m8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= max(30, w // 40):
            out[lab == i] = 1
    out = out.astype(bool)
    return out | far_edge_line(bgr, grass) if with_far else out


def far_edge_line(bgr: np.ndarray, grass: np.ndarray) -> np.ndarray:
    """The faint far touchline: a thin pale strip hugging the pitch boundary.

    From a high wide camera the far touchline is 1-3 px wide and *not brighter than the grass*
    (it is a saturation dip, V ~ grass), so the brightness top-hat misses it and the brightest
    thing near the boundary is the advertising board, which then attracts the fit. The cue used
    here: pixels just outside the (colour-gated) grass that are nearly as bright as the grass
    but pale (low saturation); the dark apron / stands behind it fail the brightness test.
    Only long thin components survive (players' outlines are neither).

    The strip is also what a hoarding's white base looks like (Emirates), but that is much
    brighter than the grass, whereas the faint touchline is not: see ``FAR_MAX_V_RATIO``."""
    out = np.zeros(grass.shape, bool)
    gate = pitch_colour(bgr, grass)
    if gate is None:
        return out
    mv, ms = gate
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    w = bgr.shape[1]
    r = max(3, w // 200)  # ~8 px @1568: reach of the strip beyond the grass edge
    g8 = grass.astype(np.uint8)
    near = cv2.dilate(g8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    pale = (v >= 0.85 * mv) & (s < GRASS_SAT_GATE * ms) & (s >= 0.15 * ms) | (v >= 0.85 * mv) & (s < 0.15 * ms)
    # only the TOP edge of the pitch (grass below the strip), not the near/side edges
    below = np.zeros_like(grass)
    below[:-r] = grass[r:]
    cand = (near & pale & below).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    # keep runs that are long and straight at any angle within +-30 deg of horizontal (the far
    # touchline of a wide camera): an opening with a thin oriented line, which removes players'
    # legs and hoarding posts hanging off the strip
    L = max(15, w // 50)
    kept = np.zeros_like(cand)
    for ang in np.deg2rad(np.arange(-30, 31, 6)):
        k = np.zeros((2 * L + 1, 2 * L + 1), np.uint8)
        cv2.line(k, (int(round(L - L * np.cos(ang))), int(round(L - L * np.sin(ang)))),
                 (int(round(L + L * np.cos(ang))), int(round(L + L * np.sin(ang)))), 1, 1)
        kept |= cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
    kept = cv2.dilate(kept, np.ones((3, 3), np.uint8)) & cand
    n, lab, stats, _ = cv2.connectedComponentsWithStats(kept, connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < w / 12:
            continue
        out[lab == i] = True
    if out.any():
        ys, xs = np.nonzero(out)
        y_r, (x0, x1) = float(np.median(ys)), (int(xs.min()), int(xs.max()))
        h = out.shape[0]
        # a far touchline is in the upper part of the frame; and it is FAINT: a strip much brighter than
        # the grass is the white base of an advertising board, not a line
        if y_r > FAR_MAX_Y_FRAC * h or float(np.median(v[out])) > FAR_MAX_V_RATIO * mv:
            return np.zeros(grass.shape, bool)
    return out


def _thin(mask: np.ndarray) -> np.ndarray:
    """Thin the mask towards 1-px centre lines (Zhang-Suen if available)."""
    m8 = (mask * 255).astype(np.uint8)
    try:
        return cv2.ximgproc.thinning(m8) > 0  # type: ignore[attr-defined]
    except (AttributeError, cv2.error):
        # distance-transform ridge: keep pixels that are local maxima along x or y
        dist = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
        dil = cv2.dilate(dist, np.ones((3, 3), np.float32))
        return (dist >= dil - 1e-3) & (dist > 0.5)


# --------------------------------------------------------------------------
# lines
# --------------------------------------------------------------------------


def _line_params(p0: np.ndarray, p1: np.ndarray) -> tuple[float, float]:
    d = p1 - p0
    th = np.arctan2(d[1], d[0]) % np.pi
    n = np.array([-np.sin(th), np.cos(th)])
    return float(th), float(n @ p0)


def _dist_to_line(pts: np.ndarray, th: float, rho: float) -> np.ndarray:
    n = np.array([-np.sin(th), np.cos(th)])
    return np.abs(pts @ n - rho)


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def merge_segments(
    segs: np.ndarray, dist_tol: float, ang_tol: float = np.deg2rad(3.0)
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Merge collinear segments (any gap) into long lines."""
    items = [(np.array(s[:2], float), np.array(s[2:], float)) for s in segs]
    items.sort(key=lambda t: -np.linalg.norm(t[1] - t[0]))
    groups: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for p0, p1 in items:
        th, rho = _line_params(p0, p1)
        placed = False
        for g in groups:
            q = np.array([pt for s in g for pt in s])
            c = cv2.fitLine(q.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            gth = float(np.arctan2(c[1], c[0]) % np.pi)
            gn = np.array([-np.sin(gth), np.cos(gth)])
            grho = float(gn @ np.array([c[2], c[3]]))
            if _angle_diff(th, gth) < ang_tol and np.all(_dist_to_line(np.array([p0, p1]), gth, grho) < dist_tol):
                g.append((p0, p1))
                placed = True
                break
        if not placed:
            groups.append([(p0, p1)])
    out = []
    for g in groups:
        q = np.array([pt for s in g for pt in s])
        c = cv2.fitLine(q.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        d = c[:2].astype(float)
        o = c[2:].astype(float)
        t = (q - o) @ d
        out.append((o + d * t.min(), o + d * t.max()))
    return out


def _sample_support(mask_d: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    n = max(2, int(np.linalg.norm(p1 - p0) / 3))
    t = np.linspace(0, 1, n)[:, None]
    pts = p0 + (p1 - p0) * t
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, mask_d.shape[1] - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, mask_d.shape[0] - 1)
    return float(mask_d[y, x].mean())


def detect_lines(mask: np.ndarray, min_len_frac: float = 0.08) -> list[Line]:
    h, w = mask.shape
    thin = _thin(mask)
    diag = float(np.hypot(h, w))
    segs = cv2.HoughLinesP(
        (thin * 255).astype(np.uint8), 1, np.pi / 360, threshold=18,
        minLineLength=max(20, int(0.025 * diag)), maxLineGap=max(10, int(0.02 * diag)),
    )
    if segs is None:
        return []
    merged = merge_segments(np.asarray(segs).reshape(-1, 4), dist_tol=max(4.0, w / 250))
    md = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    lines: list[Line] = []
    for p0, p1 in merged:
        if np.linalg.norm(p1 - p0) < min_len_frac * diag:
            continue
        sup = _sample_support(md, p0, p1)
        if sup < 0.5:
            # a merged line bridging a gap: keep the best supported span only
            continue
        lines.append(Line(0, p0, p1, sup))
    lines.sort(key=lambda L: -L.length)
    for i, L in enumerate(lines, 1):
        L.id = i
    return lines


# --------------------------------------------------------------------------
# ellipses
# --------------------------------------------------------------------------


def _ellipse_dist(pts: np.ndarray, cen, axes, ang_deg: float) -> np.ndarray:
    """Approximate distance (px) of points to an ellipse outline (radial)."""
    t = np.deg2rad(ang_deg)
    c, s_ = np.cos(t), np.sin(t)
    d = pts - np.array(cen)
    x = d[:, 0] * c + d[:, 1] * s_
    y = -d[:, 0] * s_ + d[:, 1] * c
    a, b = axes[0] / 2.0, axes[1] / 2.0
    q = np.sqrt((x / a) ** 2 + (y / b) ** 2)
    r = np.hypot(x, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(r * (1 - 1 / np.maximum(q, 1e-6)))


def _arc_span_deg(pts: np.ndarray, cen) -> float:
    ang = np.sort(np.arctan2(pts[:, 1] - cen[1], pts[:, 0] - cen[0]))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    return float(np.rad2deg(2 * np.pi - gaps.max()))


def detect_ellipses(
    mask: np.ndarray, lines: list[Line], max_ellipses: int = 3, seed: int = 0
) -> list[Ellipse]:
    """RANSAC ellipses (centre circle, penalty arcs) over the thin line pixels
    left after removing straight lines."""
    h, w = mask.shape
    rest = mask.astype(np.uint8).copy()
    kill = np.zeros_like(rest)
    for L in lines:
        cv2.line(kill, tuple(np.round(L.p0).astype(int)), tuple(np.round(L.p1).astype(int)), 1,
                 thickness=max(7, w // 120))
    rest[kill > 0] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    keep = np.zeros_like(rest)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= max(25, w // 40):
            keep[lab == i] = 1
    ty, tx = np.nonzero(_thin(keep.astype(bool)))
    pts_all = np.stack([tx, ty], 1).astype(np.float32)
    rng = np.random.default_rng(seed)
    out: list[Ellipse] = []
    tol = max(2.5, w / 450)
    for k in range(max_ellipses):
        if len(pts_all) < 60:
            break
        best = None
        sub = pts_all if len(pts_all) <= 1500 else pts_all[rng.choice(len(pts_all), 1500, replace=False)]
        for _ in range(150):
            # seed a local patch so the 6 points likely share one curve
            c0 = sub[rng.integers(len(sub))]
            near = sub[np.linalg.norm(sub - c0, axis=1) < 0.35 * w]
            if len(near) < 12:
                continue
            samp = near[rng.choice(len(near), 8, replace=False)]
            try:
                (cx, cy), (a, b), ang = cv2.fitEllipse(samp)
            except cv2.error:
                continue
            lo, hi = min(a, b), max(a, b)
            if lo < 0.02 * w or hi > 1.2 * w or lo / hi < 0.12:
                continue
            d = _ellipse_dist(sub, (cx, cy), (a, b), ang)
            inl = d < tol
            n_in = int(inl.sum())
            if best is None or n_in > best[0]:
                best = (n_in, (cx, cy), (a, b), ang)
        if best is None:
            break
        _, cen, axes, ang = best
        d = _ellipse_dist(pts_all, cen, axes, ang)
        inl = d < tol
        if inl.sum() < 40:
            break
        (cx, cy), (a, b), ang = cv2.fitEllipse(pts_all[inl])
        d = _ellipse_dist(pts_all, (cx, cy), (a, b), ang)
        inl = d < tol
        if inl.sum() < 40:
            break
        span = _arc_span_deg(pts_all[inl], (cx, cy))
        n_in = int(inl.sum())
        # perimeter sanity: inliers must cover a decent part of the outline
        per = np.pi * (3 * (a + b) / 2 - np.sqrt((3 * a / 2 + b / 2) * (a / 2 + 3 * b / 2)))
        # partial arcs (penalty arcs) fit badly and give wrong intersections: full circles only
        plausible = span >= 150
        if plausible and n_in >= 40 and n_in >= 0.25 * per * (span / 360):
            out.append(Ellipse(len(out) + 1, (float(cx), float(cy)), (float(a), float(b)), float(ang),
                               float(d[inl].mean()), n_in))
        pts_all = pts_all[~inl]
    return out


def _drop_arc_lines(lines: list[Line], ells: list[Ellipse], mask: np.ndarray) -> list[Line]:
    """Hough happily fits a chord to a flat part of an arc: drop lines that lie
    along a fitted ellipse's outline."""
    if not ells:
        return lines
    keep = []
    for L in lines:
        n = max(4, int(L.length / 4))
        pts = L.p0 + (L.p1 - L.p0) * np.linspace(0, 1, n)[:, None]
        along = any(
            float((_ellipse_dist(pts, E.centre, E.axes, E.angle) < max(4.0, mask.shape[1] / 300)).mean()) > 0.5
            for E in ells
        )
        if not along:
            keep.append(L)
    for i, L in enumerate(keep, 1):
        L.id = i
    return keep


def _poly_line_intersections(poly: np.ndarray, L: Line) -> list[np.ndarray]:
    pts = []
    d = L.dir
    n = np.array([-d[1], d[0]])
    s = (poly - L.p0) @ n
    for i in range(len(poly) - 1):
        if s[i] == 0 or s[i] * s[i + 1] < 0:
            t = s[i] / (s[i] - s[i + 1]) if s[i] != s[i + 1] else 0.0
            p = poly[i] + t * (poly[i + 1] - poly[i])
            along = (p - L.p0) @ d
            if -0.15 * L.length <= along <= 1.15 * L.length:
                pts.append(p)
    return pts


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------


def _cluster(cands: list[tuple[str, np.ndarray, list[str]]], tol: float):
    used = [False] * len(cands)
    out = []
    for i, (k, p, r) in enumerate(cands):
        if used[i]:
            continue
        grp = [i]
        used[i] = True
        for j in range(i + 1, len(cands)):
            if not used[j] and np.linalg.norm(cands[j][1] - p) < tol:
                grp.append(j)
                used[j] = True
        pts = np.mean([cands[j][1] for j in grp], axis=0)
        kinds = [cands[j][0] for j in grp]
        refs = sorted({x for j in grp for x in cands[j][2]})
        # priority of the label
        kind = next((k for k in ("ellipse_x", "line_x", "end_on", "spot") if k in kinds), kinds[0])
        out.append((kind, pts, refs))
    return out


def find_candidates(
    bgr: np.ndarray, max_candidates: int = 70
) -> Detection:
    work, scale = resize_to_work(bgr)
    h, w = work.shape[:2]
    grass = grass_mask(work)
    far = far_edge_line(work, grass)
    lm = line_mask(work, grass, with_far=False) | far
    lines = detect_lines(lm)
    ells = detect_ellipses(lm, lines)
    lines = _drop_arc_lines(lines, ells, lm)
    md = cv2.dilate(lm.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    raw: list[tuple[str, np.ndarray, list[str]]] = []
    ext = 0.04 * np.hypot(h, w)

    def inside(p, margin=2.0) -> bool:
        return -margin <= p[0] <= w - 1 + margin and -margin <= p[1] <= h - 1 + margin

    def on_mask(p, r=6) -> float:
        x0, x1 = max(0, int(p[0]) - r), min(w, int(p[0]) + r + 1)
        y0, y1 = max(0, int(p[1]) - r), min(h, int(p[1]) + r + 1)
        if x0 >= x1 or y0 >= y1:
            return 0.0
        return float(md[y0:y1, x0:x1].mean())

    # line-line intersections
    for A, B in itertools.combinations(lines, 2):
        ang = _angle_diff(float(np.arctan2(*A.dir[::-1])), float(np.arctan2(*B.dir[::-1])))
        if ang < np.deg2rad(18):
            continue
        M = np.array([A.dir, -B.dir]).T
        try:
            t, s = np.linalg.solve(M, B.p0 - A.p0)
        except np.linalg.LinAlgError:
            continue
        p = A.p0 + A.dir * t
        if not inside(p):
            continue
        if not (-ext <= t <= A.length + ext and -ext <= s <= B.length + ext):
            continue
        if on_mask(p) < 0.12:
            continue
        # a through-junction vs a T: endpoint within ext of the point
        raw.append(("line_x", p, [f"L{A.id}", f"L{B.id}"]))
    # endpoints ending on another line
    for A in lines:
        for e in (A.p0, A.p1):
            for B in lines:
                if B.id == A.id:
                    continue
                s = (e - B.p0) @ B.dir
                if not (0 <= s <= B.length):
                    continue
                d_ = e - B.p0
                perp = abs(float(B.dir[0] * d_[1] - B.dir[1] * d_[0]))
                if perp < 0.012 * w and inside(e):
                    raw.append(("end_on", e.copy(), [f"L{A.id}", f"L{B.id}"]))
    # ellipse-line intersections
    for E in ells:
        poly = E.polyline()
        for L in lines:
            for p in _poly_line_intersections(poly, L):
                if inside(p) and on_mask(p) >= 0.1:
                    raw.append(("ellipse_x", p, [f"E{E.id}", f"L{L.id}"]))
    # spots: compact small blobs surrounded by plain grass
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 40)
    rest = lm.astype(np.uint8).copy()
    for L in lines:
        cv2.line(rest, tuple(np.round(L.p0).astype(int)), tuple(np.round(L.p1).astype(int)), 0,
                 thickness=max(7, w // 120))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(rest, connectivity=8)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if 12 <= a <= 0.0012 * w * h and 1.0 <= bw / max(bh, 1) <= 3.5 and max(bw, bh) <= w / 40 \
                and a / max(bw * bh, 1) > 0.65:
            cx, cy = cent[i]
            rr = int(max(bw, bh) * 2.5 + 4)
            ring = np.zeros(grass.shape, np.uint8)
            cv2.circle(ring, (int(cx), int(cy)), rr, 1, -1)
            cv2.circle(ring, (int(cx), int(cy)), int(max(bw, bh) / 2 + 3), 0, -1)
            ring_b = ring > 0
            if ring_b.sum() and green[ring_b].mean() > 0.9 and md[ring_b].mean() < 0.05:
                raw.append(("spot", cent[i].copy(), []))
    merged = _cluster(raw, tol=max(5.0, w / 200))
    cands: list[Candidate] = []
    for kind, p, refs in merged:
        cands.append(Candidate(0, kind, (float(p[0]), float(p[1])),
                               (float(p[0]) / w, float(p[1]) / h), on_mask(p), refs))
    cands.sort(key=lambda c: (-c.support, c.xy_px[0]))
    cands = cands[:max_candidates]
    # stable ids left->right, top->bottom reading order is confusing; keep support order
    for i, c in enumerate(cands, 1):
        c.id = i
    return Detection(work, scale, grass, lm, lines, ells, cands, far_line=far)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

KIND_COLOUR = {  # BGR
    "line_x": (0, 215, 255),  # amber
    "end_on": (255, 180, 0),  # sky blue
    "ellipse_x": (255, 0, 255),  # magenta
    "spot": (0, 255, 0),  # green
}


def render_candidates(det: Detection, tint: float = 0.35) -> np.ndarray:
    """Working frame with the white-line mask lightly tinted and numbered
    candidate markers (max edge <= 1568 px)."""
    img = det.work_bgr.copy()
    tint_col = np.array([255, 160, 40], np.float32)  # blue-ish, contrasts with grass + white
    m = det.lines_mask
    img[m] = (img[m].astype(np.float32) * (1 - tint) + tint_col * tint).astype(np.uint8)
    for E in det.ellipses:
        cv2.polylines(img, [np.round(E.polyline()).astype(np.int32)], False, (255, 0, 255), 1, cv2.LINE_AA)
    r = max(4, img.shape[1] // 260)
    fs = max(0.45, img.shape[1] / 2400)
    for c in det.candidates:
        col = KIND_COLOUR.get(c.kind, (255, 255, 255))
        x, y = int(round(c.xy_px[0])), int(round(c.xy_px[1]))
        cv2.circle(img, (x, y), r, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), r, col, 1, cv2.LINE_AA)
        txt = str(c.id)
        org = (x + r + 2, y - r - 1)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
    return img
