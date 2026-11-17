# ruff: noqa: PLR0911, PLR0912, PLR0915, PLR2004
"""Shape-first centre fit: solve the camera from the centre circle + halfway line, then derive the points.

``pitch_finders.find_centre`` used to return the two circle x halfway-line points and only derived more
when a touchline end was found. This module fits the *shape* (halfway line + circle ellipse, plus whatever
else is visible) to a pitch->image homography ``H`` and every returned point is that ``H`` applied to a pitch
coordinate, so the editor (which is point based) fits the same ``H`` back.

What the shape determines
-------------------------
Ellipse ``E`` (image of the circle) and the image line ``l`` of the halfway line give ``V`` = pole of ``l``
w.r.t. ``E`` = the vanishing point of the length (x) axis. ``H`` has 8 DOF, the circle uses 5 and the two
crossing points (0, +-r) 2, so ONE parameter is free: ``W`` = the vanishing point of the width (y) axis, a
point on ``l`` (parameter ``tau``: ``W = (tau P_far + d, tau)``, ``tau = 0`` is the point at infinity along
``l``). Given ``W``:

    vanishing line m = V x W;   circle centre O = pole of m w.r.t. E;   (+-r, 0) = line(O, V) x E

so ``H`` follows from the four axis points TOP, BOTTOM, LEFT, RIGHT.

Where ``W`` comes from, in order of strength (each candidate is scored against the whole template and the
best consistent one wins):

* ``touchline``: a touchline end ``X`` on ``l`` is the known point ``(0, +-W/2)``; three known points on ``l``
  fix its 1-D projective map, hence ``W`` (the previous derivation).
* ``rig``: the ground has a known fixed camera centre ``C`` (``camera_rigs/*.json``); pan / tilt / focal length
  are solved from the 2 crossing points + the circle outline (``ptz.solve_points``-like, ``C`` fixed).
* ``intrinsic``: square pixels, principal point at the image centre (``K = diag(f, f, 1)``). The vanishing line
  ``m`` meets ``E`` in the images of the circular points ``I, J`` (complex), which must lie on the image of the
  absolute conic ``omega = K^-T K^-1 = diag(1/f^2, 1/f^2, 1)``: ``I^T omega I = 0`` is one complex = two real
  equations in ``(tau, f)``. With ``I = V + lambda W`` (``lambda`` the complex root of ``E(V + lambda W) = 0``),
  ``alpha = I_x^2 + I_y^2``, ``beta = I_z^2``: ``g alpha + beta = 0`` with ``g = 1/f^2`` real, i.e.
  ``F(tau) = Im(beta conj(alpha)) / (|alpha||beta|) = 0`` and ``g = -Re(beta conj(alpha)) / |alpha|^2 > 0``.
  1-D root search over ``tau``. Near-affine (very long lens): ``f -> inf``, ``g -> 0``, the roots go to
  ``tau -> 0`` and ``O`` -> the ellipse centre, ``LEFT/RIGHT`` -> the ends of the ellipse diameter conjugate to
  ``l`` (checked in the tests). The solve is only as good as the ellipse: a symmetric frontal view makes ``F`` flat
  (no information on ``f``), so each root is perturbation-tested (ellipse / line jitter) and abstained on
  when the points move more than ``INTRINSIC_MAX_JITTER`` of the image width. Exact on synthetic pinhole renders, but
  on the broadcast GT set it is biased (focal length ~25% low; median editor-fit error 1.9 m, 5% within 0.5 m), so it is
  the weakest source: used only when neither a touchline end nor a ground rig answers, flagged approximate.

The winner is refined against the line mask (4 axis points as parameters, chamfer distance of the halfway
line + circle + touchlines), accepted only if the core template (halfway line + circle) precision is
>= ``CENTRE_MIN_PRECISION`` and the points it returns are the named keypoints (TOP, BOTTOM, LEFT, RIGHT, SPOT,
halfway-line ends in frame) plus free circle points until at least 4 are in general position.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares

from footy_track.calibration import centre_circle as cc
from footy_track.calibration import geometry as geo
from footy_track.calibration import pitch_finders as pf
from footy_track.calibration.pitch_model import (
    CIRCLE_RADIUS,
    PitchModel,
)

RIG_DIR = Path(__file__).parent / "camera_rigs"
#: ground -> clip-name prefixes (video stems). Unknown grounds get no rig: nothing is guessed.
GROUNDS: dict[str, tuple[str, ...]] = {
    "emirates": ("arsenal_mancity", "arsenal_astonvilla", "arsenal_nottinghamforest"),
    "vitality": ("arsenal_bournemouth",),
}

#: focal length / image width range accepted for an intrinsic root
F_OVER_W = (0.8, 25.0)
#: max jitter (fraction of the width) of LEFT/RIGHT/SPOT under ellipse/line perturbation for an intrinsic root
INTRINSIC_MAX_JITTER = 0.012
#: a rig solution must reproduce the two crossing points within this fraction of the width
RIG_MAX_END_ERR = 0.02
#: max shift (fraction of the width) the chamfer refinement may move an anchor
REFINE_MAX_SHIFT = 0.012
#: the chamfer cannot see an anchor sliding along the circle outline: a ridge penalty keeps the anchors put
REFINE_RIDGE = 4.0
#: a refinement must raise the template score by this fraction to be kept
REFINE_MIN_GAIN = 0.015
CANON = np.array([1920.0, 1080.0])
CENTRE = CANON / 2.0


# --------------------------------------------------------------------------
# rigs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rig:
    ground: str
    C: tuple[float, float, float]
    roll: float
    k1: float


@lru_cache(maxsize=8)
def load_rig(ground: str) -> Rig:
    d = json.loads((RIG_DIR / f"{ground}.json").read_text())
    return Rig(ground, tuple(float(c) for c in d["C"]), float(d["roll"]), float(d["k1"]))


def rig_for_clip(clip: str | None) -> Rig | None:
    """The rig of the ground a clip was played at (by video-stem prefix), ``None`` for an unknown ground."""
    if not clip:
        return None
    c = str(clip).lower()
    for ground, prefixes in GROUNDS.items():
        if c.startswith(prefixes):
            return load_rig(ground)
    return None


def _rotation(pan, tilt, roll=0.0) -> np.ndarray:
    """World -> camera rotation(s) (``pan``/``tilt`` scalars or (N,)): pan = azimuth of the optical axis from +x,
    tilt = depression below the horizon; camera axes x right / y down / z forward."""
    pan, tilt = np.asarray(pan, float), np.asarray(tilt, float)
    ct, st, cp, sp = np.cos(tilt), np.sin(tilt), np.cos(pan), np.sin(pan)
    zero = np.zeros_like(pan)
    fwd = np.stack([ct * cp, ct * sp, -st], axis=-1)
    right = np.stack([sp, -cp, zero], axis=-1)
    down = np.cross(fwd, right)
    cr, sr = np.cos(roll), np.sin(roll)
    r2 = cr * right - sr * down
    d2 = sr * right + cr * down
    return np.stack([r2, d2, fwd], axis=-2)


def solve_rig(
    rig: Rig,
    size: tuple[int, int],
    E: cc.Ellipse,
    ends_u: np.ndarray,
    r: float = CIRCLE_RADIUS,
    extra: list[tuple[tuple[float, float], np.ndarray]] | None = None,
    start: np.ndarray | None = None,
):
    """Pan / tilt / focal length with the rig's camera centre fixed: the two halfway-line x circle points
    (``ends_u`` far first = ``(0, +r)``, ``(0, -r)``) and the circle outline (24 samples of ``E`` back-projected to the ground must lie at radius ``r``).

    ``extra`` = more known points ``[((x, y) metres, undistorted index px)]`` (a touchline end), weighted like the
    two crossings.

    Everything is in canonical 1920 x 1080 px with the principal point at the centre. Returns ``(Hpx, info)``
    with ``Hpx`` pitch metres -> undistorted index px of the working frame, or ``None``.
    """
    w, h = size
    s = CANON[0] / w
    cen = np.array([w / 2.0, h / 2.0])
    Ec = cc.Ellipse(tuple((np.asarray(E.centre) - cen) * s + CENTRE), (E.axes[0] * s, E.axes[1] * s), E.angle)
    ends_c = (np.asarray(ends_u, float) - cen) * s + CENTRE
    ends_m = np.array([[0.0, r], [0.0, -r]] + [list(xy) for xy, _ in (extra or [])])
    if extra:
        ends_c = np.concatenate([ends_c, [(np.asarray(p, float) - cen) * s + CENTRE for _, p in extra]])
    th = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    # points on the observed ellipse outline (canonical px); a pose is good when they back-project to radius r
    M_inv = np.linalg.inv(Ec.matrix())
    ell_pts = np.asarray(Ec.centre) + (M_inv @ np.stack([np.cos(th), np.sin(th)])).T
    C = np.asarray(rig.C, float)
    sig = 2.0
    sig_m = 0.05  # metres of radius error ~ the same few px

    def resid3(x):
        pan, tilt, logf = x
        R = _rotation(pan, tilt, rig.roll)
        f = np.exp(logf)
        bad = np.full(2 * len(ends_m) + len(th), 1e3)
        P = np.column_stack([ends_m, np.zeros(len(ends_m))]) - C
        Xc = P @ R.T
        if not (Xc[:, 2] > 1e-6).all():
            return bad
        uv = f * Xc[:, :2] / Xc[:, 2:3] + CENTRE
        rp = (uv - ends_c).ravel() / sig
        d = np.column_stack([(ell_pts - CENTRE) / f, np.ones(len(ell_pts))])
        wv = d @ R  # R^T d, world directions
        if not (wv[:, 2] < -1e-6).all():
            return bad
        lam = -C[2] / wv[:, 2]
        X = C[:2] + lam[:, None] * wv[:, :2]
        re = (np.linalg.norm(X, axis=1) - r) / sig_m
        out = np.concatenate([rp, re])
        return out if np.isfinite(out).all() else bad

    # seeds: for each focal length, pan / tilt from the line of sight to the centre spot and the circle's aspect
    # ratio (sin elevation = minor / major axis), refined with f fixed; then the best few are refined with f free
    pan0 = float(np.arctan2(-C[1], -C[0]))
    ratio = float(np.clip(min(Ec.axes) / max(Ec.axes), 0.02, 0.98))
    tilt0 = float(np.arcsin(ratio))
    seeds = []
    for logf in () if start is not None else np.linspace(np.log(1300.0), np.log(9000.0), 9):
        try:
            sol = least_squares(
                lambda x, lf=logf: resid3(np.array([x[0], x[1], lf])), [pan0, tilt0], method="trf", x_scale="jac", max_nfev=400
            )
        except Exception:  # noqa: BLE001
            continue
        seeds.append((sol.cost, np.array([sol.x[0], sol.x[1], logf])))
    seeds.sort(key=lambda t: t[0])
    best = None
    for _, x0 in [(0.0, np.asarray(start, float))] if start is not None else seeds[:2]:
        try:
            sol = least_squares(
                resid3, x0, method="trf", x_scale="jac", max_nfev=400, bounds=([0.3, 0.03, np.log(900.0)], [2.8, 0.9, np.log(9500.0)])
            )
        except Exception:  # noqa: BLE001
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return None
    resid = resid3
    pan, tilt, logf = best.x
    f = float(np.exp(logf))
    R = _rotation(pan, tilt, rig.roll)
    K = np.array([[f, 0, CENTRE[0]], [0, f, CENTRE[1]], [0, 0, 1.0]])
    G = K @ np.stack([R[:, 0], R[:, 1], -R @ C], axis=1)  # metres -> canonical px
    k = w / CANON[0]
    A = np.array([[k, 0, cen[0] - CENTRE[0] * k], [0, k, cen[1] - CENTRE[1] * k], [0, 0, 1.0]])
    rr_ = resid(best.x)
    rms = float(np.sqrt(np.mean((rr_[: 2 * len(ends_m)] * sig) ** 2)))  # known points, canonical px
    ring = float(np.sqrt(np.mean((rr_[2 * len(ends_m) :] * sig_m) ** 2)))  # circle outline, metres of radius
    return A @ G, {"f_over_w": f / CANON[0], "pan_deg": float(np.degrees(pan)), "tilt_deg": float(np.degrees(tilt)), "rms_px": rms, "ring_rms_m": ring, "_x": best.x.copy()}


# --------------------------------------------------------------------------
# the circle / line geometry in normalised coordinates, W -> H
# --------------------------------------------------------------------------


@dataclass
class Geom:
    """Ellipse + halfway line in centred normalised coordinates (``(p - c) / w``)."""

    Cn: np.ndarray  # conic, interior negative
    V: np.ndarray  # unit homogeneous pole of the line (vanishing point of the length axis)
    Pf: np.ndarray  # far crossing (0, +r)
    Pn: np.ndarray  # near crossing (0, -r)
    d: np.ndarray  # unit direction near -> far


def make_geom(E: cc.Ellipse, Lo: np.ndarray, Ld: np.ndarray, size: tuple[int, int]) -> Geom | None:
    """``Lo`` a point on the halfway line, ``Ld`` its unit direction (undistorted index px)."""
    T = _T(size)
    Ti = np.linalg.inv(T)
    C = cc._conic(E)
    Cn = Ti.T @ C @ Ti
    n = np.array([-Ld[1], Ld[0]])
    sv, _ = cc._line_ellipse(Lo, Ld, E)
    if sv is None:
        return None
    ends = np.stack([Lo + sv[0] * Ld, Lo + sv[1] * Ld])
    far, near = (ends[0], ends[1]) if ends[0][1] < ends[1][1] else (ends[1], ends[0])
    l_n = Ti.T @ pf._line_h(Lo, n)
    V = np.linalg.solve(Cn, l_n)
    V = V / np.linalg.norm(V)
    Pf, Pn = T[:2] @ np.append(far, 1.0), T[:2] @ np.append(near, 1.0)
    d = Pf - Pn
    d = d / np.linalg.norm(d)
    return Geom(Cn, V, Pf, Pn, d)


def _T(size: tuple[int, int]) -> np.ndarray:
    return pf._norm_t(size)


def _W_of_tau(g: Geom, tau: np.ndarray) -> np.ndarray:
    W = np.stack([tau * g.Pf[0] + g.d[0], tau * g.Pf[1] + g.d[1], tau], axis=-1)
    return W / np.linalg.norm(W, axis=-1, keepdims=True)


def circular_scan(g: Geom, tau: np.ndarray):
    """``F(tau)`` and ``g_hat(tau)`` of the circular-point condition (NaN where the vanishing line cuts the ellipse)."""
    W = _W_of_tau(g, tau)
    a = np.einsum("ki,ij,kj->k", W, g.Cn, W)
    b = W @ g.Cn @ g.V
    c = float(g.V @ g.Cn @ g.V)
    disc = a * c - b * b
    ok = (disc > 0) & (a > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lam = (-b + 1j * np.sqrt(np.where(ok, disc, 0.0))) / a
    I = g.V[None, :] + lam[:, None] * W  # noqa: E741
    alpha = I[:, 0] ** 2 + I[:, 1] ** 2
    beta = I[:, 2] ** 2
    prod = beta * np.conj(alpha)
    with np.errstate(invalid="ignore", divide="ignore"):
        F = prod.imag / (np.abs(alpha) * np.abs(beta))
        gh = -prod.real / np.abs(alpha) ** 2
    F = np.where(ok, F, np.nan)
    gh = np.where(ok, gh, np.nan)
    return F, gh


TAU_GRID = np.concatenate([[0.0], np.geomspace(2e-3, 60.0, 360)])


def intrinsic_roots(g: Geom) -> list[tuple[float, float]]:
    """``[(tau, f/w)]`` where ``I^T omega I = 0`` with ``omega = diag(1/f^2, 1/f^2, 1)``, f/w in ``F_OVER_W``."""
    F, gh = circular_scan(g, TAU_GRID)
    out = []
    for i in range(len(TAU_GRID) - 1):
        if not (np.isfinite(F[i]) and np.isfinite(F[i + 1])) or F[i] * F[i + 1] > 0:
            continue
        if F[i] == 0.0:
            t = TAU_GRID[i]
        else:
            try:
                t = brentq(lambda x: float(circular_scan(g, np.array([x]))[0][0]), TAU_GRID[i], TAU_GRID[i + 1], xtol=1e-12)
            except (ValueError, RuntimeError):
                continue
        gg = float(circular_scan(g, np.array([t]))[1][0])
        if not np.isfinite(gg) or gg <= 0:
            continue
        f = 1.0 / np.sqrt(gg)
        if F_OVER_W[0] <= f <= F_OVER_W[1]:
            out.append((float(t), float(f)))
    return out


@dataclass
class Axis:
    """The four circle axis points + centre (normalised coordinates) implied by a vanishing point ``W``."""

    top: np.ndarray
    bottom: np.ndarray
    left: np.ndarray
    right: np.ndarray
    centre: np.ndarray


def axis_from_W(g: Geom, W: np.ndarray) -> Axis | None:
    m = np.cross(g.V, W)
    try:
        Oh = np.linalg.solve(g.Cn, m)
    except np.linalg.LinAlgError:
        return None
    if abs(Oh[2]) < 1e-9 * np.abs(Oh).max():
        return None
    O = Oh[:2] / Oh[2]  # noqa: E741
    Ph = np.append(O, 1.0)
    if Ph @ g.Cn @ Ph >= -1e-9:  # centre must be inside the ellipse
        return None
    dv = g.V[:2] - g.V[2] * O
    nv = np.linalg.norm(dv)
    if nv < 1e-12:
        return None
    dv = dv / nv
    D = np.append(dv, 0.0)
    a = D @ g.Cn @ D
    b = Ph @ g.Cn @ D
    c = Ph @ g.Cn @ Ph
    disc = b * b - a * c
    if a == 0 or disc <= 0:
        return None
    s = np.array([(-b - np.sqrt(disc)) / a, (-b + np.sqrt(disc)) / a])
    P = np.stack([O + s[0] * dv, O + s[1] * dv])
    left, right = (P[0], P[1]) if P[0][0] < P[1][0] else (P[1], P[0])
    # the centre lies between the crossings along the halfway line
    u = g.Pf - g.Pn
    t = ((O - g.Pn) @ u) / (u @ u)
    if not (0.05 < t < 0.95):
        return None
    if right[0] - left[0] < 1e-4:
        return None
    return Axis(g.Pf, g.Pn, left, right, O)


def homography_from_axis(ax: Axis, size: tuple[int, int], r: float = CIRCLE_RADIUS) -> np.ndarray | None:
    """Pitch metres -> undistorted index px from the four axis points."""
    T = _T(size)
    src = np.array([(0.0, r), (0.0, -r), (-r, 0.0), (r, 0.0)])
    dst = np.array([ax.top, ax.bottom, ax.left, ax.right])
    Hn = pf._h_from_4(src, dst)
    if Hn is None or not np.isfinite(Hn).all():
        return None
    return np.linalg.inv(T) @ Hn


def _jitter(E: cc.Ellipse, Lo: np.ndarray, Ld: np.ndarray, rng: np.random.Generator):
    """A perturbed ellipse / line pair (sub-pixel fit noise)."""
    c = np.asarray(E.centre) + rng.normal(0, 0.4, 2)
    ax = (E.axes[0] + rng.normal(0, 0.4), E.axes[1] + rng.normal(0, 0.4))
    E2 = cc.Ellipse((float(c[0]), float(c[1])), (float(ax[0]), float(ax[1])), float(E.angle + rng.normal(0, 0.002)))
    ang = np.arctan2(Ld[1], Ld[0]) + rng.normal(0, 0.0008)
    Ld2 = np.array([np.cos(ang), np.sin(ang)])
    n2 = np.array([-Ld2[1], Ld2[0]])
    Lo2 = np.asarray(Lo) + n2 * rng.normal(0, 0.3)
    return E2, Lo2, Ld2


def intrinsic_candidates(E: cc.Ellipse, Lo: np.ndarray, Ld: np.ndarray, size: tuple[int, int], n_jitter: int = 10, seed: int = 0):
    """Intrinsic-prior roots as ``[{tau, f_over_w, H, axis, jitter}]``; ``jitter`` = 90th-percentile displacement
    (fraction of the width) of LEFT / RIGHT / SPOT when the ellipse and line are jittered by sub-pixel noise."""
    g = make_geom(E, Lo, Ld, size)
    if g is None:
        return []
    out = []
    rng = np.random.default_rng(seed)
    for tau, f in intrinsic_roots(g):
        ax = axis_from_W(g, _W_of_tau(g, np.array([tau]))[0])
        if ax is None:
            continue
        H = homography_from_axis(ax, size)
        if H is None:
            continue
        nom = np.stack([ax.left, ax.right, ax.centre])
        disp = []
        for _ in range(n_jitter):
            g2 = make_geom(*_jitter(E, Lo, Ld, rng), size)
            best = np.inf
            if g2 is not None:
                for t2, _f2 in intrinsic_roots(g2):
                    a2 = axis_from_W(g2, _W_of_tau(g2, np.array([t2]))[0])
                    if a2 is not None:
                        best = min(best, float(np.abs(np.stack([a2.left, a2.right, a2.centre]) - nom).max()))
            disp.append(best)
        jit = float(np.percentile(disp, 90))
        out.append({"tau": tau, "f_over_w": f, "H": H, "axis": ax, "jitter": jit})
    return out


# --------------------------------------------------------------------------
# touchline candidate (3 known points on the halfway line -> W)
# --------------------------------------------------------------------------


def touchline_candidates(ctx, r: cc.CirclePoints, g: Geom, cands: list[dict]) -> list[dict]:
    """One candidate per touchline combination (far, near, both): the 1-D map of the halfway line gives ``W``."""
    pm = ctx.pitch
    L = r.anchor
    T = _T(ctx.size)
    sv = np.sort(np.array([(r.crossings_u[0] - L.o) @ L.d, (r.crossings_u[1] - L.o) @ L.d]))
    s_top, s_bot = sv[1], sv[0]
    best = {}
    for end in ("far", "near"):
        c = [x for x in cands if x["end"] == end]
        if c:
            best[end] = max(c, key=lambda x: x["score"])
    combos = [("far",), ("near",)] + ([("far", "near")] if len(best) == 2 else [])
    out = []
    for combo in combos:
        if not all(e in best for e in combo):
            continue
        ys, ss = [CIRCLE_RADIUS, -CIRCLE_RADIUS], [s_top, s_bot]
        for e in combo:
            ys.append(pm.hw if e == "far" else -pm.hw)
            ss.append(best[e]["sL"])
        (a, b, c), resid = pf._map_1d(np.array(ys), np.array(ss))
        yy = np.linspace(-pm.hw, pm.hw, 40)
        if not (c * yy + 1.0 > 1e-6).all() or resid > 3.0 * ctx.tol:
            continue
        Winf = np.array([c * L.o[0] + a * L.d[0], c * L.o[1] + a * L.d[1], c])
        W = T @ Winf
        W = W / np.linalg.norm(W)
        ax = axis_from_W(g, W)
        if ax is None:
            continue
        H = homography_from_axis(ax, ctx.size)
        if H is None:
            continue
        out.append({"ends": combo, "H": H, "axis": ax, "X": {e: best[e]["X"] for e in combo}, "resid": float(resid)})
    return out


# --------------------------------------------------------------------------
# template scoring
# --------------------------------------------------------------------------

CORE = ("HALFWAY_LINE", "CENTRE_CIRCLE")


def shape_template(pm: PitchModel) -> pf.Template:
    """Halfway line + circle (core) and everything else the centre view may show: touchlines near the halfway
    line and the penalty / goal area lines of both ends (extras: only their matched length counts)."""
    segs, curves = pm.segments(), pm.curves()
    polys = {"HALFWAY_LINE": segs["HALFWAY_LINE"], "CENTRE_CIRCLE": curves["CENTRE_CIRCLE"]}
    polys["TOUCHLINE_TOP"] = np.array([[-30.0, pm.hw], [30.0, pm.hw]])
    polys["TOUCHLINE_BOTTOM"] = np.array([[-30.0, -pm.hw], [30.0, -pm.hw]])
    for end in ("L", "R"):
        for k in ("PEN_BOX_TOP", "PEN_BOX_BOTTOM", "PEN_BOX_FRONT", "GOAL_AREA_TOP", "GOAL_AREA_BOTTOM", "GOAL_AREA_FRONT"):
            polys[f"{end}_{k}"] = segs[f"{end}_{k}"]
        polys[f"GOAL_LINE_{end}"] = segs[f"GOAL_LINE_{end}"]
        polys[f"{end}_PEN_ARC"] = curves[f"{end}_PEN_ARC"]
    return pf.Template.build(polys, 0.3)


@lru_cache(maxsize=16)
def _tpl_cached(length: float, width: float) -> pf.Template:
    return shape_template(PitchModel(length, width))


def score_H(H: np.ndarray, tpl: pf.Template, fld: pf.Field, tolm: float) -> dict:
    ev = pf.evaluate(H, tpl, fld, tolm)
    core_v = core_m = ex_v = ex_m = 0.0
    for name, v, m in ev["elems"]:
        if name in CORE:
            core_v += v
            core_m += m
        else:
            ex_v += v
            ex_m += m
    ev["core_v"], ev["core_m"] = core_v, core_m
    ev["core_precision"] = core_m / core_v if core_v > 0 else 0.0
    ev["extra_v"], ev["extra_m"] = ex_v, ex_m
    ev["score"] = core_m + ex_m - 0.5 * (ex_v - ex_m)
    return ev


def refine_ridge(H0: np.ndarray, tpl: pf.Template, fld: pf.Field, size: tuple[int, int]) -> np.ndarray:
    """Chamfer least squares over the 4 circle axis points' image positions with a ridge towards the start."""
    anchors = _anchors()
    P0, ok0 = pf._project(H0[None], anchors)
    P0 = P0[0]
    if not ok0.all():
        return H0
    cap = 8.0

    def hof(p: np.ndarray):
        return pf._h_from_4(anchors, P0 + p.reshape(4, 2))

    def resid(p: np.ndarray) -> np.ndarray:
        H = hof(p)
        if H is None or not np.isfinite(H).all():
            return np.full(len(tpl.pts) + 8, cap)
        xy, ok = pf._project(H[None], tpl.pts)
        d, inb = fld.sample(xy[0])
        return np.concatenate([np.where(inb & ok[0], np.minimum(d, cap), cap), REFINE_RIDGE * p])

    sol = least_squares(resid, np.zeros(8), loss="soft_l1", f_scale=2.0, diff_step=0.05, max_nfev=30, x_scale=4.0)
    H = hof(sol.x)
    return H if H is not None and np.isfinite(H).all() else H0


def _anchors(r: float = CIRCLE_RADIUS) -> np.ndarray:
    return np.array([(0.0, r), (0.0, -r), (-r, 0.0), (r, 0.0)])


def _core_tpl(pm: PitchModel) -> pf.Template:
    segs, curves = pm.segments(), pm.curves()
    polys = {
        "HALFWAY_LINE": segs["HALFWAY_LINE"],
        "CENTRE_CIRCLE": curves["CENTRE_CIRCLE"],
        "TOUCHLINE_TOP": np.array([[-30.0, pm.hw], [30.0, pm.hw]]),
        "TOUCHLINE_BOTTOM": np.array([[-30.0, -pm.hw], [30.0, -pm.hw]]),
    }
    return pf.Template.build(polys, 0.6)


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------


@dataclass
class ShapeFit:
    H: np.ndarray  # pitch m -> undistorted index px
    source: str  # "touchline" | "rig" | "intrinsic"
    ev: dict
    info: dict = field(default_factory=dict)
    X: dict = field(default_factory=dict)  # touchline junctions used (index px), by "far" / "near"


def _candidate_pool(ctx, r: cc.CirclePoints, g: Geom, touch: list[dict], rig: Rig | None, use: tuple[str, ...]):
    """All pitch->px candidates: ``(source, H, info, X)``."""
    pool = []
    if "touchline" in use:
        for tc in touchline_candidates(ctx, r, g, touch):
            pool.append(("touchline", tc["H"], {"ends": list(tc["ends"]), "map_residual_px": tc["resid"]}, tc["X"]))
    if "rig" in use and rig is not None:
        res = solve_rig(rig, ctx.size, r.ellipse, r.crossings_u)
        rig_x = None
        if res is not None:
            H, info = res
            rig_x = info.pop("_x")
            info = dict(info, ground=rig.ground)
            pool.append(("rig", H, info, {}))
        best = {}
        for end in ("far", "near"):
            c = [x for x in touch if x["end"] == end]
            if c:
                best[end] = max(c, key=lambda x: x["score"])
        if best and "touchline" in use:
            hw = ctx.pitch.hw
            extra = [((0.0, hw if e == "far" else -hw), b["X"]) for e, b in best.items()]
            res = solve_rig(rig, ctx.size, r.ellipse, r.crossings_u, extra=extra, start=rig_x)
            if res is not None:
                H, info = res
                info.pop("_x", None)
                pool.append(("rig+touchline", H, dict(info, ground=rig.ground, ends=sorted(best)), {e: b["X"] for e, b in best.items()}))
    if "intrinsic" in use:
        L = r.anchor
        for c in intrinsic_candidates(r.ellipse, L.o, L.d, ctx.size):
            pool.append(("intrinsic", c["H"], {"f_over_w": c["f_over_w"], "jitter": c["jitter"], "tau": c["tau"]}, {}))
    return pool


def fit_shape(
    ctx,
    r: cc.CirclePoints,
    fld: pf.Field,
    touch: list[dict],
    rig: Rig | None = None,
    use: tuple[str, ...] = ("touchline", "rig", "intrinsic"),
    refine: bool = True,
) -> tuple[ShapeFit | None, list[dict]]:
    """Best accepted candidate (or ``None``) and a per-candidate log for the diagnostics."""
    w, h = ctx.size
    pm = ctx.pitch
    g = make_geom(r.ellipse, r.anchor.o, r.anchor.d, ctx.size)
    log: list[dict] = []
    if g is None:
        return None, log
    tpl = _tpl_cached(pm.length, pm.width)
    tpl_c = _core_tpl(pm)
    tolm = 1.6 * ctx.tol
    ends_u = np.asarray(r.crossings_u, float)
    scored: list[ShapeFit] = []
    for source, H0, info, X in _candidate_pool(ctx, r, g, touch, rig, use):
        H = H0
        entry = {"source": source, **{k: v for k, v in info.items() if not isinstance(v, np.ndarray)}}
        log.append(entry)
        if source == "intrinsic" and info["jitter"] > INTRINSIC_MAX_JITTER:
            entry["rejected"] = f"unstable (jitter {info['jitter']:.3f} w)"
            continue
        # the crossings must be reproduced
        P, ok = pf._project(H[None], np.array([(0.0, CIRCLE_RADIUS), (0.0, -CIRCLE_RADIUS)]))
        if not ok.all() or np.abs(P[0] - ends_u).max() > RIG_MAX_END_ERR * w:
            entry["rejected"] = "does not reproduce the circle x halfway-line points"
            continue
        if refine:
            Hr = refine_ridge(H, tpl_c, fld, ctx.size)
            pa0 = pf._project(H[None], _anchors())[0][0]
            pa1 = pf._project(Hr[None], _anchors())[0][0]
            if np.abs(pa1 - pa0).max() <= REFINE_MAX_SHIFT * w:
                e0, e1 = score_H(H, tpl, fld, tolm), score_H(Hr, tpl, fld, tolm)
                if e1["score"] >= e0["score"] * (1.0 + REFINE_MIN_GAIN):
                    H = Hr
                    entry["refined"] = True
        ev = score_H(H, tpl, fld, tolm)
        entry.update(core_precision=round(ev["core_precision"], 3), score=round(ev["score"], 1), extra_m=round(ev["extra_m"]))
        if ev["core_precision"] < pf.CENTRE_MIN_PRECISION:
            entry["rejected"] = f"core precision {ev['core_precision']:.2f}"
            continue
        sf = ShapeFit(H, source, ev, dict(info), X)
        scored.append(sf)
    if not scored:
        return None, log
    prio = {"rig+touchline": 4, "rig": 3, "touchline": 2, "intrinsic": 1}
    # the square-pixel prior alone is the weakest source (systematically biased on broadcast cameras): it only
    # answers when nothing else does; otherwise the best matched evidence, near ties to the stronger source
    strong = [s for s in scored if s.source != "intrinsic"] or scored
    top = max(s.ev["score"] for s in strong)
    near = [s for s in strong if s.ev["score"] >= top - 0.08 * max(top, 1.0)]
    best = max(near, key=lambda s: (prio[s.source], s.ev["score"]))
    best.info["alternatives"] = [
        {"source": s.source, "score": round(s.ev["score"], 1)} for s in scored if s is not best
    ]
    return best, log


# --------------------------------------------------------------------------
# points from the fitted H
# --------------------------------------------------------------------------


def _in_frame(uv: np.ndarray) -> bool:
    return bool(np.isfinite(uv).all() and (uv >= 0.005).all() and (uv <= 0.995).all())


def points_from_fit(sf: ShapeFit, ctx, base_conf: float, n_circle_free: int = 4) -> list[pf.FoundPoint]:
    """Named keypoints + free points of the fitted ``H``; at least 4 points in general position when the fit
    shows enough of the pitch (the circle's four axis points, the spot, the halfway-line ends, 45-degree points,
    then wider circle / touchline / grid points)."""
    pm = ctx.pitch
    kp = pm.keypoints()
    H = sf.H
    ev = sf.ev
    rr = CIRCLE_RADIUS
    xy_s, ok_s = ev["xy"], ev["inb"]
    match = ev["match"]

    def seen(p_px) -> bool:
        near = (np.linalg.norm(xy_s - p_px, axis=1) < 0.05 * ctx.size[0]) & ok_s
        return bool(near.sum() >= 8 and match[near].mean() >= 0.5)

    out: list[pf.FoundPoint] = []

    def add(pid, xy, conf, derived):
        P, ok = pf._project(H[None], np.array([xy]))
        if not ok[0, 0]:
            return False
        uv = pf._to_obs(P[0, 0], ctx.size, ctx.dist)[0]
        if not _in_frame(uv):
            return False
        out.append(pf._fp(pm, pid, uv, conf, derived, None if pid else xy))
        return True

    s = sf.source
    k = {"rig+touchline": 1.0, "touchline": 1.0, "rig": 0.95, "intrinsic": 0.6}[s]
    for pid in ("CENTRE_CIRCLE_TOP", "CENTRE_CIRCLE_BOTTOM", "CENTRE_CIRCLE_LEFT", "CENTRE_CIRCLE_RIGHT"):
        xy = kp[pid]
        P = pf._project(H[None], np.array([xy]))[0][0, 0]
        d = not seen(P)
        add(pid, xy, base_conf * k * (0.9 if d else 1.0), d)
    add("CENTRE_SPOT", kp["CENTRE_SPOT"], base_conf * k * 0.8, True)
    for pid, end in (("HALFWAY_TOP", "far"), ("HALFWAY_BOTTOM", "near")):
        P = pf._project(H[None], np.array([kp[pid]]))[0][0, 0]
        d = not (end in sf.X or seen(P))
        add(pid, kp[pid], base_conf * k * (0.8 if d else 0.95), d)
    q = rr / np.sqrt(2.0)
    for sx in (-1.0, 1.0):
        for sy in (1.0, -1.0):
            add(None, (sx * q, sy * q), base_conf * k * 0.75, True)

    def general() -> bool:
        if len(out) < 4:
            return False
        uv = geo.undistort_points(np.array([p.uv for p in out]), ctx.dist)
        cfg = geo.analyse_configuration(uv, np.array([p.pitch_xy for p in out]))
        return bool(cfg["n_general_subsets"]) if cfg["n_general_subsets"] is not None else not cfg["groups"]

    if not general():
        extra = [(rr * np.cos(a), rr * np.sin(a)) for a in np.radians(np.arange(15, 360, 30))]
        extra += [(x, sy * pm.hw) for sy in (1.0, -1.0) for x in (-30.0, -20.0, -10.0, 10.0, 20.0, 30.0)]
        extra += [(x, y) for x in (-25.0, -15.0, 15.0, 25.0) for y in (-15.0, -5.0, 5.0, 15.0)]
        have = {tuple(np.round(p.pitch_xy, 3)) for p in out}
        for xy in extra:
            if tuple(np.round(xy, 3)) in have:
                continue
            if add(None, xy, base_conf * k * 0.6, True):
                have.add(tuple(np.round(xy, 3)))
            if len(out) >= 12 and general():
                break
        if not general():
            return []
    return out


def to_norm_H(Hpx: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Pitch m -> undistorted normalised image ([0, 1]) from a pitch m -> index-px homography."""
    w, h = size
    S = np.array([[1.0 / w, 0, 0.5 / w], [0, 1.0 / h, 0.5 / h], [0, 0, 1.0]])
    Hn = S @ Hpx
    return Hn / Hn[2, 2] if abs(Hn[2, 2]) > 1e-12 else Hn
