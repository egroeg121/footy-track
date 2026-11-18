"""PTZ rig model for a broadcast main camera (see ``docs/design/calibration_ptz.md``).

Hypothesis: the main camera has a (nearly) fixed centre of projection ``C`` per ground; a frame only varies by
pan, tilt, focal length (and, optionally, a small roll). The *rig* is ``C`` (+ shared roll, constant lens k1);
a frame is solved for ``(pan, tilt, log f)`` with ``C`` as a Gaussian prior (sigma > 0) or hard-fixed (sigma = 0).

Conventions
-----------
* Pitch frame as in :mod:`footy_track.calibration.geometry`: metres, origin at the centre spot, +x toward the R
  goal, +y toward the far touchline, z up.
* Image: normalised [0, 1]^2 like everything else; internally a canonical 1920 x 1080 pixel frame with the
  principal point at its centre (square pixels). ``f`` is in canonical pixels.
* Lens: division model of :mod:`geometry` (``k1 > 0`` barrel-ish for our feeds), applied between the observed
  image and the pinhole image; observations are undistorted before fitting points.
* Rotation (world -> camera, camera axes x right / y down / z forward): ``pan`` is the azimuth of the optical axis
  in the pitch plane (radians from +x, counter-clockwise seen from above), ``tilt`` the depression of the axis
  below the horizon, ``roll`` a rotation about the optical axis. A pan axis that is not exactly vertical shows up
  as a (mostly constant) roll and a tilt-dependent term that this model does not have; the rig fit reports
  whether the shared roll is adequate.

Per-frame solvers
-----------------
:func:`solve_points`  from point correspondences (any n >= 2, regularised by the rig prior);
:func:`search_lines`  from the white-line mask only: coarse search over pan x tilt x log f scored by a two-sided
semantic line score, multi-hypothesis, LM refinement, abstention on near-ties.
"""

# ruff: noqa: PLR0915, PLC0415
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import PitchModel

CANON_W, CANON_H = 1920.0, 1080.0
CANON = np.array([CANON_W, CANON_H])
CENTRE = CANON / 2.0
ASPECT = CANON_W / CANON_H

#: default per-frame prior on log f (canonical px): broad, only there to condition 2-3 point solves
LOGF_PRIOR = (float(np.log(3500.0)), 0.6)
PAN_RANGE = (np.deg2rad(20.0), np.deg2rad(160.0))  # azimuth of the optical axis; ~90 deg = looking across the pitch
TILT_RANGE = (np.deg2rad(2.0), np.deg2rad(50.0))
F_RANGE = (900.0, 9500.0)


# --------------------------------------------------------------------------
# camera geometry
# --------------------------------------------------------------------------


def rotation(pan, tilt, roll=0.0) -> np.ndarray:
    """World -> camera rotation(s). ``pan``/``tilt`` may be arrays (N,) -> (N, 3, 3); roll scalar or (N,)."""
    pan, tilt = np.asarray(pan, float), np.asarray(tilt, float)
    ct, st, cp, sp = np.cos(tilt), np.sin(tilt), np.cos(pan), np.sin(pan)
    zero = np.zeros_like(pan)
    fwd = np.stack([ct * cp, ct * sp, -st], axis=-1)
    right = np.stack([sp, -cp, zero], axis=-1)
    down = np.cross(fwd, right)
    cr, sr = np.cos(roll), np.sin(roll)
    cr, sr = np.broadcast_to(cr, pan.shape)[..., None], np.broadcast_to(sr, pan.shape)[..., None]
    r2 = cr * right - sr * down
    d2 = sr * right + cr * down
    return np.stack([r2, d2, fwd], axis=-2)


def project_ground(xy_m: np.ndarray, C, R, f, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Pitch-plane points (N,2) (z = 0) -> undistorted px (N,2) and depth (N,) for one camera. ``scale`` projects
    into a copy of the canonical frame scaled by it (``f`` stays in canonical px)."""
    P = np.column_stack([xy_m, np.zeros(len(xy_m))]) - np.asarray(C)
    Xc = P @ np.asarray(R).T
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = scale * f * Xc[:, :2] / z[:, None] + CENTRE * scale
    return uv, z


def backproject_ground(uv_px: np.ndarray, C, R, f) -> tuple[np.ndarray, np.ndarray]:
    """Undistorted canonical px (N,2) -> pitch (x, y) (N,2) and a validity mask (ray hits the ground ahead)."""
    d = np.column_stack([(uv_px - CENTRE) / f, np.ones(len(uv_px))])
    w = d @ np.asarray(R)  # R^T d
    C = np.asarray(C, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -C[2] / w[:, 2]
    ok = (w[:, 2] < -1e-9) & np.isfinite(lam) & (lam > 0)
    xy = C[:2] + lam[:, None] * w[:, :2]
    return xy, ok


def undistort_px(uv_px: np.ndarray, k1: float) -> np.ndarray:
    if not k1:
        return np.asarray(uv_px, float)
    return geo.undistort_norm(np.asarray(uv_px, float) / CANON, k1, aspect=ASPECT) * CANON


def distort_px(uv_px: np.ndarray, k1: float) -> np.ndarray:
    if not k1:
        return np.asarray(uv_px, float)
    return geo.distort_norm(np.asarray(uv_px, float) / CANON, k1, aspect=ASPECT) * CANON


@dataclass
class Camera:
    """A solved frame: pinhole at ``C`` with pan/tilt/roll/``f`` and lens ``k1``."""

    pan: float
    tilt: float
    logf: float
    roll: float
    C: np.ndarray
    k1: float = 0.0
    ok: bool = True
    cost: float = float("nan")
    info: dict = field(default_factory=dict)

    @property
    def f(self) -> float:
        return float(np.exp(self.logf))

    @property
    def R(self) -> np.ndarray:
        return rotation(self.pan, self.tilt, self.roll)

    def pitch_to_image(self, xy_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> observed normalised image coords (N,2) (NaN where undefined) and validity."""
        uv, z = project_ground(np.atleast_2d(xy_m), self.C, self.R, self.f)
        uv = distort_px(uv, self.k1)
        ok = (z > 1e-6) & np.all(np.isfinite(uv), axis=1)
        return uv / CANON, ok

    def image_to_pitch(self, uv_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Observed normalised image coords -> pitch metres and validity."""
        uv = undistort_px(np.atleast_2d(uv_norm) * CANON, self.k1)
        return backproject_ground(uv, self.C, self.R, self.f)

    def homography(self) -> np.ndarray:
        """H: *undistorted* normalised image -> pitch metres (the stored-GT convention, k1 kept separately)."""
        K = np.array([[self.f, 0, CENTRE[0]], [0, self.f, CENTRE[1]], [0, 0, 1.0]])
        R = self.R
        t = -R @ self.C
        G = K @ np.stack([R[:, 0], R[:, 1], t], axis=1)  # metres -> canonical px
        return np.linalg.inv(G) @ np.diag([CANON_W, CANON_H, 1.0])

    def params(self) -> dict:
        return {"pan_deg": float(np.degrees(self.pan)), "tilt_deg": float(np.degrees(self.tilt)), "f_px": self.f,
                "roll_deg": float(np.degrees(self.roll)), "C": [float(c) for c in self.C], "k1": self.k1}


@dataclass
class Rig:
    """Per-ground camera rig: centre of projection prior, shared roll and constant lens k1."""

    C: np.ndarray
    roll: float = 0.0
    k1: float = 0.0
    sigma_C: tuple[float, float, float] = (1.0, 3.0, 1.0)  # prior tolerance (m); 0 = hard fixed
    pitch: tuple[float, float] = (105.0, 68.0)
    info: dict = field(default_factory=dict)

    def with_sigma(self, sigma) -> Rig:
        return Rig(self.C, self.roll, self.k1, tuple(sigma), self.pitch, self.info)

    def with_k1(self, k1: float) -> Rig:
        return Rig(self.C, self.roll, k1, self.sigma_C, self.pitch, self.info)

    def to_json(self, path: str | Path) -> None:
        d = asdict(self)
        d["C"] = [float(c) for c in self.C]
        Path(path).write_text(json.dumps(d, indent=1, default=float))

    @classmethod
    def from_json(cls, path: str | Path) -> Rig:
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["C"]), d["roll"], d["k1"], tuple(d["sigma_C"]), tuple(d["pitch"]), d.get("info", {}))


# --------------------------------------------------------------------------
# solving from points
# --------------------------------------------------------------------------


def _grid(n_pan=36, n_tilt=18, n_f=14, pan_range=PAN_RANGE, tilt_range=TILT_RANGE, f_range=F_RANGE):
    pan = np.linspace(*pan_range, n_pan)
    tilt = np.linspace(*tilt_range, n_tilt)
    logf = np.linspace(np.log(f_range[0]), np.log(f_range[1]), n_f)
    P, T, F = np.meshgrid(pan, tilt, logf, indexing="ij")
    return np.stack([P.ravel(), T.ravel(), F.ravel()], axis=1)


_GRID = _grid()
#: grid for the lines-only search: ~2 deg steps (a step of 2 deg moves a line by ~50 px at SEARCH_W), scored with a
#: wide capture range (``TAU_COARSE``) that the LM refinement then tightens
_GRID_LINES = _grid(n_pan=53, n_tilt=19, n_f=17, pan_range=(np.deg2rad(40.0), np.deg2rad(145.0)),
                    tilt_range=(np.deg2rad(4.0), np.deg2rad(40.0)), f_range=(1400.0, 8000.0))
TAU_COARSE = 90.0


def _nms(grid: np.ndarray, order: np.ndarray, k: int, d_pan=0.06, d_tilt=0.04, d_logf=0.2) -> list[int]:
    keep: list[int] = []
    for i in order:
        g = grid[i]
        if all(not (abs(g[0] - grid[j][0]) < d_pan and abs(g[1] - grid[j][1]) < d_tilt
                    and abs(g[2] - grid[j][2]) < d_logf) for j in keep):
            keep.append(int(i))
            if len(keep) >= k:
                break
    return keep


def _batch_project(grid: np.ndarray, C: np.ndarray, roll: float, xy_m: np.ndarray, scale: float = 1.0):
    """Project (M,2) pitch points under every (pan,tilt,logf) of ``grid`` -> px (N,M,2) @ ``scale`` x canonical,
    depth (N,M). The principal point is the centre of the scaled image."""
    R = rotation(grid[:, 0], grid[:, 1], roll)  # (N,3,3)
    P = np.column_stack([xy_m, np.zeros(len(xy_m))]) - C
    Xc = np.einsum("nij,mj->nmi", R, P)
    z = Xc[..., 2]
    f = (np.exp(grid[:, 2]) * scale)[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = f * Xc[..., :2] / z[..., None] + CENTRE * scale
    return uv, z


def solve_points(
    img_norm: np.ndarray,
    pitch_m: np.ndarray,
    rig: Rig,
    *,
    px_sigma: float = 2.0,
    free_roll: bool = False,
    logf_prior: tuple[float, float] | None = LOGF_PRIOR,
    n_starts: int = 5,
    k1: float | None = None,
) -> Camera | None:
    """Solve a frame's (pan, tilt, log f) from >= 2 point correspondences with the rig's ``C`` as a prior.

    ``img_norm`` are the observed (distorted) normalised image points. ``rig.sigma_C``: Gaussian tolerance of C in
    metres (a zero component fixes that axis; all zero = hard fixed C). A broad Gaussian prior on log f
    (``logf_prior``) conditions the 2-3 point case. Multi-start: coarse grid over pan x tilt x log f, best
    ``n_starts`` distinct cells are refined. Returns None when fewer than 2 points are given.
    """
    img_norm = np.asarray(img_norm, float).reshape(-1, 2)
    pitch_m = np.asarray(pitch_m, float).reshape(-1, 2)
    if len(img_norm) < 2:
        return None
    k1 = rig.k1 if k1 is None else k1
    obs = undistort_px(img_norm * CANON, k1)
    C0 = np.asarray(rig.C, float)
    sig = np.asarray(rig.sigma_C, float)
    free_c = sig > 0

    # coarse grid with C at its prior mean
    uv, z = _batch_project(_GRID, C0, rig.roll, pitch_m)
    err = np.linalg.norm(uv - obs[None], axis=2)
    err = np.where((z > 0) & np.isfinite(err), err, 1e4)
    cost0 = (np.minimum(err, 400.0) ** 2).sum(1)
    if logf_prior:
        cost0 = cost0 + px_sigma**2 * ((_GRID[:, 2] - logf_prior[0]) / logf_prior[1]) ** 2
    starts = _nms(_GRID, np.argsort(cost0)[: 400], n_starts)

    def unpack(x):
        pan, tilt, logf = x[:3]
        i = 3
        roll = rig.roll
        if free_roll:
            roll = x[i]
            i += 1
        C = C0.copy()
        C[free_c] = x[i:i + int(free_c.sum())]
        return pan, tilt, logf, roll, C

    def resid(x):
        pan, tilt, logf, roll, C = unpack(x)
        R = rotation(pan, tilt, roll)
        uvp, zz = project_ground(pitch_m, C, R, np.exp(logf))
        e = (uvp - obs) / px_sigma
        e = np.where(np.isfinite(e) & (zz[:, None] > 0), e, 1e3).ravel()
        parts = [e]
        if free_c.any():
            parts.append((C[free_c] - C0[free_c]) / sig[free_c])
        if free_roll:
            parts.append(np.array([(roll - rig.roll) / np.deg2rad(1.0)]))
        if logf_prior:
            parts.append(np.array([(logf - logf_prior[0]) / logf_prior[1]]))
        return np.concatenate(parts)

    best = None
    for s in starts:
        x0 = list(_GRID[s])
        if free_roll:
            x0.append(rig.roll)
        x0 += list(C0[free_c])
        try:
            sol = least_squares(resid, np.array(x0), method="lm", max_nfev=200 * (len(x0) + 1))
        except Exception:  # noqa: BLE001 - degenerate start
            continue
        c = float(sol.cost)
        if best is None or c < best[0]:
            best = (c, sol.x)
    if best is None:
        return None
    pan, tilt, logf, roll, C = unpack(best[1])
    r = resid(best[1])[: 2 * len(pitch_m)] * px_sigma
    return Camera(pan, tilt, logf, roll, C, k1, True, best[0], {"rms_px": float(np.sqrt(np.mean(r**2)))})


# --------------------------------------------------------------------------
# rig fit
# --------------------------------------------------------------------------


@dataclass
class RigFrame:
    """Correspondences of one calibration frame (observed normalised image pts, pitch metres) + grouping keys."""

    img: np.ndarray
    pitch: np.ndarray
    key: str = ""
    shot: str = ""


def fit_rig(
    frames: list[RigFrame],
    *,
    C0=(0.0, -65.0, 15.0),
    fit_k1: bool = True,
    fit_roll: bool = True,
    robust_px: float = 4.0,
    sigma_C=(1.0, 3.0, 1.0),
    pitch=(105.0, 68.0),
    init: list[Camera] | None = None,
    fit_axis_slope: bool = False,
) -> tuple[Rig, list[Camera]]:
    """Joint fit of a shared rig (C, roll, k1) and per-frame (pan, tilt, log f) to point correspondences
    (bundle adjustment on undistorted pixel error, Cauchy loss). Returns the rig and the per-frame cameras.

    ``fit_axis_slope`` adds the zoom test of the proposal: the centre of projection of a frame is
    ``C + b * (log f - log 3500) * (optical axis)``, i.e. the entrance pupil slides along the axis with zoom; the
    fitted ``b`` (metres per unit log f) is returned in ``rig.info['axis_slope']``."""
    C0 = np.asarray(C0, float)
    cams0: list[Camera] = []
    for i, fr in enumerate(frames):
        if init is not None:
            cams0.append(init[i])
            continue
        c = solve_points(fr.img, fr.pitch, Rig(C0, 0.0, 0.0, (0.0, 0.0, 0.0)), n_starts=6, px_sigma=2.0)
        cams0.append(c)
    n = len(frames)
    n_shared = 3 + int(fit_roll) + int(fit_k1) + int(fit_axis_slope)

    def unpack_shared(x):
        C = x[:3]
        roll = x[3] if fit_roll else 0.0
        k1 = x[3 + int(fit_roll)] if fit_k1 else 0.0
        b = x[3 + int(fit_roll) + int(fit_k1)] if fit_axis_slope else 0.0
        return C, roll, k1, b

    # all points of all frames stacked: the residual is fully vectorised over frames
    fidx = np.concatenate([np.full(len(fr.pitch), i) for i, fr in enumerate(frames)])
    P_all = np.column_stack([np.concatenate([fr.pitch for fr in frames]), np.zeros(len(fidx))])
    obs_px = np.concatenate([fr.img * CANON for fr in frames])

    def resid(x):
        C, roll, k1, b = unpack_shared(x)
        fp = x[n_shared:].reshape(n, 3)
        R = rotation(fp[:, 0], fp[:, 1], roll)  # (n,3,3)
        Ci = np.broadcast_to(C, (n, 3)) + (b * (fp[:, 2] - LOGF_PRIOR[0])[:, None] * R[:, 2] if fit_axis_slope else 0.0)
        obs = undistort_px(obs_px, k1)
        Xc = np.einsum("pij,pj->pi", R[fidx], P_all - Ci[fidx])
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uvp = np.exp(fp[fidx, 2])[:, None] * Xc[:, :2] / z[:, None] + CENTRE
        e = uvp - obs
        e = np.where(np.isfinite(e) & (z[:, None] > 0), e, 1e3).ravel()
        pri = []
        if fit_k1:
            pri.append((k1 / 0.05) * robust_px)
        if fit_roll:
            pri.append((roll / np.deg2rad(3.0)) * robust_px)
        if fit_axis_slope:
            pri.append((b / 2.0) * robust_px)
        # weak per-frame priors keep f in range for frames with 4 points
        pri.extend(((fp[:, 2] - LOGF_PRIOR[0]) / 1.0) * 0.5)
        return np.concatenate([e, np.array(pri)])

    x0 = [*C0, *([0.0] if fit_roll else []), *([0.0] if fit_k1 else []), *([0.0] if fit_axis_slope else [])]
    for c in cams0:
        x0 += [c.pan, c.tilt, c.logf]
    x0 = np.array(x0)
    xs = np.concatenate([[2.0, 5.0, 2.0], [0.01] * int(fit_roll), [0.01] * int(fit_k1), [1.0] * int(fit_axis_slope),
                         np.tile([0.02, 0.02, 0.05], n)])
    # Jacobian by grouped forward differences: one evaluation per shared parameter plus three for the per-frame
    # parameters (frames do not share residual rows). Dense with tr_solver='exact': the sparse lsmr solver
    # stalls / creeps on the flat C_y - C_z - f ridge.
    n_pts = [len(fr.pitch) for fr in frames]
    n_pri = int(fit_k1) + int(fit_roll) + int(fit_axis_slope) + n
    row_frame = np.concatenate([np.repeat(np.arange(n), 2 * np.array(n_pts)), np.full(n_pri - n, -1), np.arange(n)])
    rows_of = [np.flatnonzero(row_frame == i) for i in range(n)]

    def jac(x):
        r0 = resid(x)
        J = np.zeros((len(r0), len(x)))
        for j in range(n_shared):
            h = 1e-6 * max(1.0, abs(x[j]))
            xp = x.copy()
            xp[j] += h
            J[:, j] = (resid(xp) - r0) / h
        for k in range(3):
            xp = x.copy()
            xp[n_shared + k::3] += 1e-6
            d = (resid(xp) - r0) / 1e-6
            cols = n_shared + k + 3 * np.arange(n)
            for i in range(n):
                J[rows_of[i], cols[i]] = d[rows_of[i]]
        return J

    sol = least_squares(resid, x0, jac=jac, loss="cauchy", f_scale=robust_px, method="trf", x_scale=xs,
                        max_nfev=100, tr_solver="exact")
    C, roll, k1, b = unpack_shared(sol.x)
    cams = []
    for i, fr in enumerate(frames):
        pan, tilt, logf = sol.x[n_shared + 3 * i: n_shared + 3 * i + 3]
        Ci = np.array(C) + (b * (logf - LOGF_PRIOR[0]) * rotation(pan, tilt, roll)[2] if fit_axis_slope else 0.0)
        r = _frame_resid_px(fr, Ci, roll, k1, pan, tilt, logf)
        cams.append(Camera(pan, tilt, logf, roll, Ci, k1, True, float("nan"),
                           {"rms_px": float(np.sqrt(np.mean(r**2))), "key": fr.key}))
    info = {"n_frames": n, "n_points": int(sum(len(f.pitch) for f in frames)), "cost": float(sol.cost)}
    if fit_axis_slope:
        info["axis_slope"] = float(b)
    rig = Rig(np.array(C), float(roll), float(k1), tuple(sigma_C), tuple(pitch), info)
    return rig, cams


def _frame_resid_px(fr: RigFrame, C, roll, k1, pan, tilt, logf) -> np.ndarray:
    obs = undistort_px(fr.img * CANON, k1)
    uvp, _ = project_ground(fr.pitch, C, rotation(pan, tilt, roll), np.exp(logf))
    return np.linalg.norm(uvp - obs, axis=1)


# --------------------------------------------------------------------------
# solving from white lines (semantic, multi-hypothesis, abstaining)
# --------------------------------------------------------------------------

#: search / scoring resolution (width px); the camera's f is scaled to it
SEARCH_W = 960
TAU_PX = 5.0  # capture / "explained" distance (px at SEARCH_W)


@dataclass
class LineEvidence:
    """The white-line evidence of one frame at SEARCH_W: truncated DT, grass mask, thin-line length."""

    w: int
    h: int
    dt: np.ndarray  # float32 (h, w) distance to the nearest line pixel, px
    grass: np.ndarray  # bool (h, w), dilated
    gdt: np.ndarray  # float32 (h, w): distance (px) from a non-grass pixel to the nearest grass pixel, 0 on grass
    L_mask: float  # px length of the thinned line mask
    mask_pts: np.ndarray  # (M, 2) subsample of thinned line pixels (x, y)
    scale: float = 1.0  # SEARCH_W / canonical width

    @classmethod
    def from_masks(cls, lines_mask: np.ndarray, grass: np.ndarray) -> LineEvidence:
        h0, w0 = lines_mask.shape
        s = SEARCH_W / w0
        w, h = SEARCH_W, int(round(h0 * s))
        lm = cv2.resize(lines_mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_AREA) > 40
        gm = cv2.resize(grass.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_AREA) > 127
        rg = max(2, w // 260)
        gm = cv2.dilate(gm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rg + 1,) * 2)) > 0
        dt = cv2.distanceTransform((~lm).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        gdt = cv2.distanceTransform((~gm).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        from footy_track.calibration.prelabel.candidates import _thin

        thin = _thin(lm)
        ys, xs = np.nonzero(thin)
        pts = np.column_stack([xs, ys]).astype(np.float64)
        if len(pts) > 3000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 3000, replace=False)]
        return cls(w, h, dt, gm, gdt, float(thin.sum()), pts, w / CANON_W)


def _samples(pitch: PitchModel, step_m: float = 1.5):
    """Densified model polylines -> points (N,2), consecutive pairs (a,b) within a polyline, weights."""
    pts, pa, pb = [], [], []
    off = 0
    for poly in pitch.polylines().values():
        d = geo._densify(np.asarray(poly, np.float64), step_m)
        pts.append(d)
        idx = np.arange(off, off + len(d) - 1)
        pa.append(idx)
        pb.append(idx + 1)
        off += len(d)
    return np.concatenate(pts), np.concatenate(pa), np.concatenate(pb)


@dataclass
class LineHypothesis:
    cam: Camera
    score: float  # 0 = every predicted line explained and every detected line explained; ~1 = nothing explained
    parts: dict = field(default_factory=dict)


def _line_scores(P: np.ndarray, vis: np.ndarray, pa, pb, ev: LineEvidence,
                 tau: float = TAU_PX) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided score of N hypotheses. ``P`` (N,M,2) projected+distorted samples at SEARCH_W px, ``vis`` (N,M).

    predicted-unexplained length: visible model segments whose endpoints are not on a detected line (or not on
    the pitch); detected-unexplained length: thinned mask length minus the length of model segments that land on
    it (never below 0). Normalised by the detected length: 0 = perfect, 1 = all of the evidence unexplained, > 1
    when many predicted lines have no evidence (zoomed-out or wrong hypotheses). Hiding the model costs 1.
    """
    ix = np.clip(np.rint(P[..., 0]).astype(np.int64), 0, ev.w - 1)
    iy = np.clip(np.rint(P[..., 1]).astype(np.int64), 0, ev.h - 1)
    flat = iy * ev.w + ix
    d = ev.dt.ravel()[flat]
    on_grass = np.clip(1.0 - ev.gdt.ravel()[flat] / tau, 0.0, 1.0)  # soft: a line just off the grass edge still counts
    m = np.clip(1.0 - d / tau, 0.0, 1.0) * on_grass * vis
    seg = np.linalg.norm(P[:, pb] - P[:, pa], axis=2)
    pv = vis[:, pa] & vis[:, pb]
    seg = np.where(pv & np.isfinite(seg), seg, 0.0)
    mseg = 0.5 * (m[:, pa] + m[:, pb])
    matched = (seg * mseg).sum(1)
    pred_unexpl = (seg * (1.0 - mseg)).sum(1)
    L = max(ev.L_mask, 1.0)
    score = (pred_unexpl + np.maximum(L - matched, 0.0)) / L
    return score, matched / L, pred_unexpl / L


def _project_obs(grid: np.ndarray, C, roll, k1, samples, ev: LineEvidence):
    uv, z = _batch_project(grid, np.asarray(C, float), roll, samples, ev.scale)
    if k1:
        sz = np.array([ev.w, ev.h], float)
        shp = uv.shape
        uv = geo.distort_norm((uv / sz).reshape(-1, 2), k1, aspect=ASPECT).reshape(shp) * sz
    vis = (z > 1.0) & np.all(np.isfinite(uv), axis=2)
    vis &= (uv[..., 0] >= 0) & (uv[..., 0] <= ev.w - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= ev.h - 1)
    uv = np.where(np.isfinite(uv), uv, -1e4)
    return uv, vis


def score_cameras(cams: list[Camera], ev: LineEvidence, pitch: PitchModel) -> np.ndarray:
    """Line score (see :func:`_line_scores`) of fully specified cameras (own C, roll, k1)."""
    samples, pa, pb = _samples(pitch, 1.0)
    out = []
    for c in cams:
        g = np.array([[c.pan, c.tilt, c.logf]])
        uv, vis = _project_obs(g, c.C, c.roll, c.k1, samples, ev)
        out.append(_line_scores(uv, vis, pa, pb, ev)[0][0])
    return np.array(out)


def explained_f1(matched: float, visible: float, detected: float) -> float:
    """Bounded agreement of a hypothesis with the evidence: F1 of recall (matched / expected visible model length)
    and precision (matched / detected line length) = ``2 matched / (visible + detected)``, in [0, 1].

    The raw score of :func:`_line_scores` divides the unexplained *predicted* length by the *detected* length, so a
    frame whose mask holds one clear line and misses the faint ones drives the support of a correct camera far below
    0 (down to -4 on the Vitality frames), while a mask rich in clutter looks mild. F1 normalises by both lengths
    (the expected visible length of the model as well as the detected one), so missing faint lines costs at most
    their share of the expected length and the value is comparable across grounds and mask qualities."""
    den = visible + detected
    return float(np.clip(2.0 * matched / den, 0.0, 1.0)) if den > 0 else 0.0


def line_support(cams: list[Camera], ev: LineEvidence, pitch: PitchModel) -> np.ndarray:
    """:func:`explained_f1` of fully specified cameras on the frame's line evidence."""
    samples, pa, pb = _samples(pitch, 1.0)
    L = max(ev.L_mask, 1.0)
    out = []
    for c in cams:
        uv, vis = _project_obs(np.array([[c.pan, c.tilt, c.logf]]), c.C, c.roll, c.k1, samples, ev)
        _, m, p = _line_scores(uv, vis, pa, pb, ev)
        out.append(explained_f1(m[0] * L, (m[0] + p[0]) * L, L))
    return np.array(out)


def search_lines(
    ev: LineEvidence,
    rig: Rig,
    pitch: PitchModel | None = None,
    *,
    n_top: int = 1500,
    n_cells: int = 60,
    n_refine: int = 10,
    refine: bool = True,
    margin: float = 0.08,
    grid: np.ndarray | None = None,
    min_support: float = 0.15,
    min_f1: float | None = None,
) -> tuple[Camera | None, list[LineHypothesis], dict]:
    """Lines-only solve: coarse pan x tilt x log f search (C at the rig prior, roll / k1 from the rig) scored by the
    two-sided line score, NMS to distinct hypotheses, robust LM on (pan, tilt, log f, C within its prior) for the
    best ones, re-scored. Abstains (returns ``None``) when the best two *distinct* hypotheses differ by less than
    ``margin`` in score (the line score is ~[0, 1]) or the best explains less than ``min_support`` of the evidence
    (``1 - score``; unbounded below). With ``min_f1`` the support test uses :func:`explained_f1` instead (``info["f1"]``
    is always reported), which does not collapse on frames whose mask misses faint lines.
    -> (camera or None, ranked hypotheses, info with the margin / support / f1 / reason)."""
    pitch = pitch or PitchModel(*rig.pitch)
    grid = _GRID_LINES if grid is None else grid
    samples, pa, pb = _samples(pitch, 1.5)
    C0 = np.asarray(rig.C, float)

    def eval_grid(g, tau):
        out = np.empty(len(g))
        for i in range(0, len(g), 400):
            uv, vis = _project_obs(g[i:i + 400], C0, rig.roll, rig.k1, samples, ev)
            out[i:i + 400] = _line_scores(uv, vis, pa, pb, ev, tau)[0]
        return out

    # level 1: coarse grid, wide capture range; level 2: local grids around the best distinct cells, tighter range
    scores = eval_grid(grid, TAU_COARSE)
    cells = _nms(grid, np.argsort(scores)[:n_top], n_cells)
    step = np.array([np.diff(np.unique(grid[:, 0]))[0], np.diff(np.unique(grid[:, 1]))[0],
                     np.diff(np.unique(grid[:, 2]))[0]])
    off = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij"), axis=-1).reshape(-1, 3) * step / 2.0
    local = np.concatenate([grid[c] + off for c in cells])
    ls = eval_grid(local, 0.4 * TAU_COARSE)
    starts_g = []
    for i in range(len(cells)):
        j = int(np.argmin(ls[i * len(off):(i + 1) * len(off)]))
        starts_g.append((ls[i * len(off) + j], local[i * len(off) + j]))
    starts_g.sort(key=lambda t: t[0])
    starts = [g for _, g in starts_g[:n_refine]]
    hyps: list[LineHypothesis] = []
    for g0 in starts:
        cam = Camera(g0[0], g0[1], g0[2], rig.roll, C0.copy(), rig.k1)
        if refine:
            cam = _refine_lines(cam, ev, rig, pitch)
        sc = score_cameras([cam], ev, pitch)[0]
        hyps.append(LineHypothesis(cam, float(sc)))
    hyps.sort(key=lambda h: h.score)
    # merge hypotheses that are the same camera in pitch space (refinement pulls neighbours together)
    distinct: list[LineHypothesis] = []
    for h in hyps:
        if all(_cam_distance(h.cam, d.cam) > 2.5 for d in distinct):
            distinct.append(h)
    info = {"n_distinct": len(distinct), "scores": [h.score for h in distinct[:3]]}
    if not distinct:
        return None, [], {**info, "reason": "no hypothesis"}
    best = distinct[0]
    gap = distinct[1].score - best.score if len(distinct) > 1 else 1.0
    f1 = float(line_support([best.cam], ev, pitch)[0])
    info.update(margin=float(gap), support=float(1.0 - best.score), best_score=best.score, f1=f1)
    if (f1 if min_f1 is not None else 1.0 - best.score) < (min_f1 if min_f1 is not None else min_support):
        return None, distinct, {**info, "reason": "low support"}
    if gap < margin:
        return None, distinct, {**info, "reason": "near-tie"}
    best.cam.ok = True
    return best.cam, distinct, {**info, "reason": None}


_PROBE = np.array([[x, y] for x in np.linspace(-45, 45, 7) for y in np.linspace(-28, 28, 5)])


def _cam_distance(a: Camera, b: Camera) -> float:
    """Mean pitch-plane displacement (m) of the image of a fixed pitch-point grid between two cameras."""
    ia, oka = a.pitch_to_image(_PROBE)
    ib, okb = b.pitch_to_image(_PROBE)
    ok = oka & okb & (np.abs(ia - 0.5).max(1) < 0.6)
    if ok.sum() < 5:
        return 99.0
    pa, vo = b.image_to_pitch(ia[ok])
    return float(np.mean(np.linalg.norm(pa - _PROBE[ok], axis=1))) if vo.all() else 99.0


def _refine_lines(cam: Camera, ev: LineEvidence, rig: Rig, pitch: PitchModel, fit_k1: bool = False,
                  k1_prior: float = 0.03) -> Camera:
    """Robust LM of (pan, tilt, log f, [k1], C within the prior) on the truncated DT of the white-line mask."""
    samples, pa, pb = _samples(pitch, 0.7)
    C0 = np.asarray(rig.C, float)
    sig = np.asarray(rig.sigma_C, float)
    free_c = sig > 0
    k = SEARCH_W / 1568.0  # refine.py's stages are px at a 1568-wide image
    stages = [(20.0 * k, 3.0 * k), (8.0 * k, 0.0)]  # (trunc px, blur) coarse -> fine @ SEARCH_W
    dts = []
    for trunc, blur in stages:
        d = np.minimum(ev.dt, trunc)
        if blur > 0:
            d = cv2.GaussianBlur(d, (0, 0), blur)
        dts.append((d, trunc))
    sz = np.array([ev.w, ev.h], float)

    nk = int(fit_k1)

    def make(x):
        C = C0.copy()
        C[free_c] = x[3 + nk:3 + nk + int(free_c.sum())]
        return Camera(x[0], x[1], x[2], rig.roll, C, x[3] if fit_k1 else rig.k1)

    def resid_factory(dt, trunc):
        def resid(x):
            c = make(x)
            uv, z = project_ground(samples, c.C, c.R, c.f, ev.scale)
            if c.k1:
                uv = geo.distort_norm(uv / sz, c.k1, aspect=ASPECT) * sz
            ok = (z > 1.0) & np.all(np.isfinite(uv), axis=1)
            ok &= (uv[:, 0] >= 0) & (uv[:, 0] <= ev.w - 1.001) & (uv[:, 1] >= 0) & (uv[:, 1] <= ev.h - 1.001)
            r = np.full(len(samples), trunc)
            if ok.any():
                xi, yi = uv[ok, 0], uv[ok, 1]
                x0, y0 = np.floor(xi).astype(int), np.floor(yi).astype(int)
                fx, fy = xi - x0, yi - y0
                v = (dt[y0, x0] * (1 - fx) * (1 - fy) + dt[y0, x0 + 1] * fx * (1 - fy)
                     + dt[y0 + 1, x0] * (1 - fx) * fy + dt[y0 + 1, x0 + 1] * fx * fy)
                gd = ev.gdt[np.rint(yi).astype(int), np.rint(xi).astype(int)]
                r[ok] = np.minimum(v + gd, trunc)
            parts = [r]
            if free_c.any():
                parts.append((c.C[free_c] - C0[free_c]) / sig[free_c] * 2.0)
            if fit_k1:
                parts.append(np.array([(c.k1 - rig.k1) / k1_prior * 3.0]))
            return np.concatenate(parts)

        return resid

    x = np.array([cam.pan, cam.tilt, cam.logf, *([cam.k1] if fit_k1 else []), *cam.C[free_c]])
    for dt, trunc in dts:
        try:
            lo = np.array([-np.inf, -np.inf, -np.inf, *([-np.inf] * nk), *(C0[free_c] - 4 * sig[free_c])])
            hi = np.array([np.inf, np.inf, np.inf, *([np.inf] * nk), *(C0[free_c] + 4 * sig[free_c])])
            x = np.clip(x, lo + 1e-9, hi - 1e-9)
            sol = least_squares(resid_factory(dt, trunc), x, bounds=(lo, hi), loss="cauchy", f_scale=trunc * 0.4,
                                method="trf",
                                x_scale=np.array([0.02, 0.02, 0.05, *([0.01] * nk), *([1.0] * int(free_c.sum()))]),
                                diff_step=1e-3, max_nfev=40)
            x = sol.x
        except Exception:  # noqa: BLE001
            break
    out = make(x)
    out.logf = float(np.clip(out.logf, np.log(F_RANGE[0]), np.log(F_RANGE[1])))
    return out


# --------------------------------------------------------------------------
# synthetic frames (tests and the V1 synthetic experiment)
# --------------------------------------------------------------------------


def render_synthetic(
    cam: Camera, pitch: PitchModel | None = None, width: int = 1568, thickness: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Line mask and grass mask (bool, ``width`` x ``width * 9/16``) of a known camera looking at the pitch:
    pitch markings drawn as thick polylines, grass = pitch rectangle plus a 3 m apron."""
    pitch = pitch or PitchModel()
    h = int(round(width / ASPECT))
    s = width / CANON_W
    sz = np.array([width, h], float)

    def proj(xy):
        uv, z = project_ground(xy, cam.C, cam.R, cam.f, s)
        if cam.k1:
            uv = geo.distort_norm(uv / sz, cam.k1, aspect=ASPECT) * sz
        return uv, (z > 0.5) & np.all(np.isfinite(uv), axis=1) & (np.abs(uv).max(axis=1) < 1e5)

    lines = np.zeros((h, width), np.uint8)
    for poly in pitch.polylines().values():
        d = geo._densify(np.asarray(poly, np.float64), 0.25)
        uv, ok = proj(d)
        for i in range(len(d) - 1):
            if ok[i] and ok[i + 1]:
                cv2.line(lines, tuple(np.rint(uv[i]).astype(int)), tuple(np.rint(uv[i + 1]).astype(int)), 1,
                         thickness)
    hl, hw = pitch.hl + 3.0, pitch.hw + 3.0
    rect = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]])
    edge = np.concatenate([geo._densify(rect[i:i + 2], 0.5)[:-1] for i in range(4)])
    uv, ok = proj(edge)
    grass = np.zeros((h, width), np.uint8)
    if ok.all():
        cv2.fillPoly(grass, [np.rint(uv).astype(np.int32)], 1)
    else:  # part of the apron is behind the camera / off-image: clip the polygon at the image border
        full = np.ones((h, width), np.uint8)
        pts = np.rint(np.clip(np.nan_to_num(uv, nan=0.0), -5e4, 5e4)).astype(np.int32)
        cv2.fillPoly(grass, [pts[ok]], 1)
        grass &= full
    return lines.astype(bool), grass.astype(bool)
