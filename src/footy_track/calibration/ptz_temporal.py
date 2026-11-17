"""PTZ V3: temporal calibration of a broadcast main camera (see ``reports/ptz_v3.md``).

A PTZ camera pans / tilts / zooms about a (nearly) fixed centre ``C`` (:mod:`ptz`), so consecutive frames of a shot
are related by a *rotation + zoom* homography ``H = K2 R K1^-1`` -- 3 + 1 degrees of freedom, whatever the scene
depth. This module

1. caches a clip (grey frames at ``WORK_W``, colour-histogram thumbnails, white-line / grass masks every few frames),
2. segments it into shots and classifies the main camera,
3. measures the inter-frame motion with a RANSAC on the *constrained* rotation + zoom model (:func:`rotzoom_ransac`;
   no free 8-DOF homography), tracking static-background features with players masked out,
4. solves all frames of a shot jointly (:func:`solve_shot`): per-frame ``(pan, tilt, log f)`` with residuals
   motion consistency (the raw inter-frame correspondences evaluated under the unknown poses), white-line evidence
   (distance transform, as :func:`ptz._refine_lines`), anchors (lines-only fits / finder points) and a mild
   smoothness prior; the rig (``C``, roll, ``k1``) is shared and may itself be estimated from the anchored frames of
   several shots (:func:`fit_shared`).

Conventions as in :mod:`ptz`: canonical 1920 x 1080 px with the principal point at the centre, ``f`` in canonical px,
division-model ``k1``. Work images are ``WORK_W`` wide (``S = WORK_W / 1920`` per canonical px).
"""

# ruff: noqa: PLR0915, PLR0912, PLR0913, PLC0415, N806, N803, E741
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from footy_track.calibration import geometry as geo
from footy_track.calibration import ptz
from footy_track.calibration import temporal as tp
from footy_track.calibration.pitch_model import PitchModel

Array = np.ndarray
WORK_W = 960
S = WORK_W / ptz.CANON_W  # work px per canonical px
CANON_CEN = ptz.CENTRE
#: default nominal focal length (canonical px) when no anchor is known (motion measurement only)
F_NOM = 3000.0

# --------------------------------------------------------------------------
# 1. clip cache
# --------------------------------------------------------------------------

STATIC_W = 320


def static_mask_small(bgr: Array) -> Array:
    """Usable static background of a ``STATIC_W``-wide frame (uint8 0/1): everything except the broadcast overlay and
    the non-green, non-white blobs inside the pitch region (players, referees, ball; opened so thin lines do not punch
    holes, dilated by a margin). Stands and advertising boards stay in: a pure rotation about C needs no scene depth."""
    from footy_track.calibration.prelabel.candidates import grass_mask

    grass = grass_mask(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = (h >= 30) & (h <= 95) & (s >= 40)
    white = (s <= 90) & (v >= 150)
    hole = (grass & ~(green | white)).astype(np.uint8)
    hole = cv2.morphologyEx(hole, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    k = max(3, bgr.shape[1] // 60) | 1
    hole = cv2.dilate(hole, np.ones((k, k), np.uint8))
    return (tp.overlay_mask(bgr.shape[:2]) & (hole == 0)).astype(np.uint8)


@dataclass
class ClipCache:
    """Per-frame data of one clip: grey frames at ``WORK_W``, histogram thumbnails and (every ``ev_step`` frames and
    on request) the white-line / grass masks of the pre-labelling pipeline."""

    video: Path
    root: Path
    n: int = 0
    size: tuple[int, int] = (1920, 1080)
    gray: Array | None = None  # (N, h, w) uint8
    hist: Array | None = None  # (N, 12*4*4) float32
    hd: Array | None = None  # (N,) Bhattacharyya distance to the previous frame
    static: Array | None = None  # (N, h, w) uint8 at STATIC_W: 1 = usable static background (no players, no overlay)
    ev_frames: list[int] = field(default_factory=list)

    def mask_path(self, t: int) -> Path:
        return self.root / "masks" / f"{t:05d}.npz"

    def build(self, ev_frames: set[int], log=print) -> ClipCache:
        """One sequential decode pass (cached): ``gray960.npy``, ``hist.npy``, masks for ``ev_frames``."""
        self.root.mkdir(parents=True, exist_ok=True)
        gp, hp, sp = self.root / "gray960.npy", self.root / "hist.npy", self.root / "static.npy"
        have = gp.exists() and hp.exists() and sp.exists()
        todo = {t for t in ev_frames if not self.mask_path(t).exists()}
        if have and not todo:
            self.gray = np.load(gp, mmap_mode="r")
            self.static = np.load(sp, mmap_mode="r")
            self._finish(np.load(hp))
            return self
        (self.root / "masks").mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(self.video))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (W, H)
        sh = int(round(H * WORK_W / W))
        gray = np.lib.format.open_memmap(gp.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8, shape=(n, sh, WORK_W)) \
            if not have else None
        hists = []
        statics = np.lib.format.open_memmap(sp.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8,
                                            shape=(n, int(round(H * STATIC_W / W)), STATIC_W)) if not have else None
        t0 = time.time()
        for t in range(n):
            ok, f = cap.read()
            if not ok:
                n = t
                break
            if gray is not None:
                small = cv2.resize(f, (WORK_W, sh), interpolation=cv2.INTER_AREA)
                gray[t] = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                hists.append(tp._hist_thumb(small))
                statics[t] = static_mask_small(cv2.resize(small, (STATIC_W, statics.shape[1]),
                                                          interpolation=cv2.INTER_AREA))
            if t in todo:
                self._save_masks(t, f)
            if t % 100 == 0:
                log(f"  decode {t}/{n} {time.time() - t0:.0f}s")
        cap.release()
        if gray is not None:
            gray.flush()
            del gray
            statics.flush()
            del statics
            gp.with_suffix(".tmp.npy").rename(gp)
            sp.with_suffix(".tmp.npy").rename(sp)
            np.save(hp, np.stack(hists).astype(np.float32))
        self.gray = np.load(gp, mmap_mode="r")
        self.static = np.load(sp, mmap_mode="r")
        self._finish(np.load(hp))
        return self

    def _finish(self, h: Array) -> None:
        self.hist = h
        self.n = len(h)
        self.hd = np.array([0.0] + [tp.hist_distance(h[i - 1], h[i]) for i in range(1, self.n)])
        self.ev_frames = sorted(int(p.stem) for p in (self.root / "masks").glob("*.npz"))

    def _save_masks(self, t: int, bgr: Array) -> None:
        from footy_track.calibration.prelabel.candidates import LINE_SAT_REL, grass_mask, line_mask, resize_to_work

        work, _ = resize_to_work(bgr)
        grass = grass_mask(work)
        lines = line_mask(work, grass, sat_rel=LINE_SAT_REL)
        from footy_track.calibration.ptz_propose import main_camera_p

        pm = main_camera_p(bgr)
        for _ in range(2):  # a full disk can truncate an npz: retry once
            try:
                np.savez_compressed(self.mask_path(t), lines=lines, grass=grass, pmain=pm)
                return
            except OSError:
                self.mask_path(t).unlink(missing_ok=True)

    def evidence(self, t: int) -> ptz.LineEvidence | None:
        p = self.mask_path(t)
        if not p.exists():
            return None
        try:
            d = np.load(p)
            return ptz.LineEvidence.from_masks(d["lines"], d["grass"])
        except (EOFError, OSError, ValueError):
            return None

    def read_full(self, t: int) -> Array | None:
        cap = cv2.VideoCapture(str(self.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, f = cap.read()
        cap.release()
        return f if ok else None


# --------------------------------------------------------------------------
# 2. shots + main camera
# --------------------------------------------------------------------------

CUT_HARD = tp.CUT_HIST_HARD
CUT_SOFT = tp.CUT_HIST_SOFT


def segment_shots(hd: Array, pair_ok: Array | None = None, min_len: int = 8) -> list[tuple[int, int]]:
    """Shots (start, end inclusive) from histogram jumps: ``hd[t]`` = distance between frame t-1 and t. A jump is a cut
    when >= ``CUT_HARD``, or >= ``CUT_SOFT`` and the motion pair (t-1, t) failed (``pair_ok[t]`` False)."""
    n = len(hd)
    cuts = [0]
    for t in range(1, n):
        if hd[t] >= CUT_HARD or (hd[t] >= CUT_SOFT and pair_ok is not None and not pair_ok[t]):
            cuts.append(t)
    cuts.append(n)
    shots = [(a, b - 1) for a, b in zip(cuts[:-1], cuts[1:], strict=True)]
    # merge slivers into their neighbour
    out: list[tuple[int, int]] = []
    for a, b in shots:
        if out and b - a + 1 < min_len:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def pmain_of(cache: ClipCache, t: int) -> float:
    """Cached main-camera probability of evidence frame ``t`` (NaN when there is none)."""
    try:
        return float(np.load(cache.mask_path(t))["pmain"])
    except (OSError, KeyError, ValueError, EOFError):
        return float("nan")


# --------------------------------------------------------------------------
# 3. the constrained inter-frame model: rotation (3) + zoom (1)
# --------------------------------------------------------------------------


def rodrigues(w: Array) -> Array:
    """Rotation matrices of rotation vectors ``w`` (..., 3) -> (..., 3, 3)."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w, axis=-1, keepdims=True)
    k = np.where(th > 1e-12, w / np.maximum(th, 1e-12), 0.0)
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def rotzoom_map(u: Array, f1: float, w: Array, s: Array | float) -> Array:
    """Map canonical-px offsets ``u`` (N,2) (from the principal point) of camera 1 (focal ``f1``) into camera 2, which
    is camera 1 rotated by the rotation vector ``w`` (camera-1 -> camera-2 axes) and zoomed by ``exp(s)``:
    ``u2 = K2 R K1^-1 u``. ``w`` (..., 3) / ``s`` (...) may be batched over hypotheses -> (..., N, 2)."""
    R = rodrigues(w)
    d = np.concatenate([u / f1, np.ones((len(u), 1))], axis=1)  # (N,3)
    v = np.einsum("...ij,nj->...ni", R, d)
    z = v[..., 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = v[..., :2] / z
    return p * (f1 * np.exp(np.asarray(s, float))[..., None, None])


def rotzoom_linear(u1: Array, u2: Array, f1: float) -> Array:
    """First-order (small angle) least-squares ``(wx, wy, wz, s)`` from correspondences (any n >= 2). Used as the
    RANSAC minimal solver (2 points) and the start of the exact fit."""
    x, y = u1[:, 0] / f1, u1[:, 1] / f1
    A = np.zeros((2 * len(u1), 4))
    A[0::2] = np.stack([-x * y, 1 + x * x, -y, x], 1)
    A[1::2] = np.stack([-(1 + y * y), x * y, x, y], 1)
    b = ((u2 - u1) / f1).reshape(-1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol


@dataclass
class PairMotion:
    ok: bool
    w: Array  # (3,) rotation vector
    s: float  # ln(f2 / f1)
    inl: Array  # bool (n,) over the input correspondences
    rms: float  # canonical px, over the inliers
    n_in: int
    reason: str | None = None


def rotzoom_fit(u1: Array, u2: Array, f1: float, w0=None, s0: float = 0.0) -> tuple[Array, float]:
    """Exact robust fit of (w, s) to correspondences (LM + Cauchy, scale 1.5 canonical px)."""
    from scipy.optimize import least_squares

    x0 = np.r_[np.zeros(3) if w0 is None else w0, s0]

    def res(x):
        r = (rotzoom_map(u1, f1, x[:3], x[3]) - u2).ravel()
        return np.where(np.isfinite(r), r, 1e3)

    sol = least_squares(res, x0, loss="cauchy", f_scale=1.5, x_scale=[1e-2, 1e-2, 1e-2, 1e-2], max_nfev=30)
    return sol.x[:3], float(sol.x[3])


def rotzoom_ransac(u1: Array, u2: Array, f1: float, *, thr: float = 2.0, n_iter: int = 300, min_inl: int = 25,
                   seed: int = 0) -> PairMotion:
    """RANSAC of the constrained rotation + zoom model on correspondences (canonical px offsets from the principal
    point, undistorted). Minimal sample: 2 points (4 equations, 4 unknowns), solved with the first-order model, scored
    with the exact one; the best consensus set is refitted exactly and the inliers re-evaluated (twice)."""
    n = len(u1)
    if n < max(min_inl, 4):
        return PairMotion(False, np.zeros(3), 0.0, np.zeros(n, bool), float("nan"), 0, "too few points")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=(n_iter, 2))
    i = i[i[:, 0] != i[:, 1]]
    x, y = u1[:, 0] / f1, u1[:, 1] / f1
    # batched 4x4 normal-free solve: rows of the two samples
    A = np.zeros((len(i), 4, 4))
    b = np.zeros((len(i), 4))
    for k in range(2):
        xx, yy = x[i[:, k]], y[i[:, k]]
        A[:, 2 * k] = np.stack([-xx * yy, 1 + xx * xx, -yy, xx], 1)
        A[:, 2 * k + 1] = np.stack([-(1 + yy * yy), xx * yy, xx, yy], 1)
        b[:, 2 * k] = (u2[i[:, k], 0] - u1[i[:, k], 0]) / f1
        b[:, 2 * k + 1] = (u2[i[:, k], 1] - u1[i[:, k], 1]) / f1
    A += 1e-9 * np.eye(4)
    try:
        H = np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return PairMotion(False, np.zeros(3), 0.0, np.zeros(n, bool), float("nan"), 0, "degenerate")
    H = np.where(np.isfinite(H), H, 0.0)
    # keep hypotheses with sane magnitude (< 20 deg, zoom < 2x)
    good = (np.abs(H[:, :3]).max(1) < 0.35) & (np.abs(H[:, 3]) < 0.7)
    H = H[good]
    if len(H) == 0:
        return PairMotion(False, np.zeros(3), 0.0, np.zeros(n, bool), float("nan"), 0, "no hypothesis")
    pred = rotzoom_map(u1, f1, H[:, :3], H[:, 3])  # (h, n, 2)
    err = np.linalg.norm(pred - u2[None], axis=2)
    cnt = (err < thr).sum(1)
    best = int(np.argmax(cnt))
    w, s = H[best, :3], float(H[best, 3])
    inl = err[best] < thr
    for _ in range(2):
        if inl.sum() < 6:
            break
        w, s = rotzoom_fit(u1[inl], u2[inl], f1, w, s)
        e = np.linalg.norm(rotzoom_map(u1, f1, w, s) - u2, axis=1)
        inl = e < thr
    e = np.linalg.norm(rotzoom_map(u1, f1, w, s) - u2, axis=1)
    rms = float(np.sqrt(np.mean(e[inl] ** 2))) if inl.any() else float("nan")
    ok = int(inl.sum()) >= min_inl
    return PairMotion(ok, np.asarray(w), float(s), inl, rms, int(inl.sum()), None if ok else "few inliers")


# --------------------------------------------------------------------------
# 4. tracking
# --------------------------------------------------------------------------

LK_PARAMS = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def detect_features(gray: Array, mask: Array | None, max_corners: int = 900) -> Array:
    m = None
    if mask is not None:
        m = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    pts = cv2.goodFeaturesToTrack(np.ascontiguousarray(gray), max_corners, 0.003, 6, mask=m, blockSize=5)
    return np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2)


def track(gA: Array, gB: Array, pts: Array, mask_b: Array | None = None, init: Array | None = None,
          fb_tol: float = 0.7) -> tuple[Array, Array]:
    """Forward-backward LK of ``pts`` (work px) from A to B -> (a, b) of the surviving correspondences; ``init`` is an
    initial guess of the B positions."""
    if len(pts) == 0:
        return pts, pts
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    gA, gB = np.ascontiguousarray(gA), np.ascontiguousarray(gB)
    if init is None:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(gA, gB, p0, None, **LK_PARAMS)
    else:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(gA, gB, p0, init.reshape(-1, 1, 2).astype(np.float32),
                                             flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **LK_PARAMS)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(gB, gA, p1, None, **LK_PARAMS)
    fb = np.linalg.norm(pb - p0, axis=2).ravel()
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_tol)
    a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    if mask_b is not None and len(a):
        h, w = mask_b.shape
        xi, yi = np.clip(np.rint(b[:, 0]).astype(int), 0, w - 1), np.clip(np.rint(b[:, 1]).astype(int), 0, h - 1)
        keep = mask_b[yi, xi] > 0
        a, b = a[keep], b[keep]
    return a, b


def work_to_canon(p: Array, k1: float) -> Array:
    """Work-px points (N,2) (observed, distorted) -> undistorted canonical-px offsets from the principal point."""
    uv = np.asarray(p, float) / S
    return ptz.undistort_px(uv, k1) - CANON_CEN


@dataclass
class Pair:
    """Inlier correspondences (observed work px) between frames ``i`` and ``j`` plus the constrained fit (``w``, ``s``
    given the nominal focal length that was used for the RANSAC)."""

    i: int
    j: int
    a: Array  # (n, 2) work px in frame i
    b: Array  # (n, 2) work px in frame j
    w: Array
    s: float
    rms: float
    n_in: int
    n_all: int
    ok: bool = True


def _mask_at(cache: ClipCache, t: int) -> Array:
    m = np.asarray(cache.static[t])
    return cv2.resize(m, (cache.gray.shape[2], cache.gray.shape[1]), interpolation=cv2.INTER_NEAREST)


def _distort_work(u_can: Array, k1: float) -> Array:
    """Undistorted canonical offsets -> observed work px."""
    return ptz.distort_px(u_can + CANON_CEN, k1) * S


def compute_motion(cache: ClipCache, start: int, end: int, f_nom: Array, k1: float, gaps=(1, 5), max_keep: int = 120,
                   thr: float = 2.0, log=None) -> tuple[list[Pair], dict]:
    """Constrained-model motion pairs over frames ``start..end`` (inclusive). ``f_nom[t - start]``: nominal focal length
    (canonical px) used to evaluate the model of the pair that starts at ``t``. Gap-1 pairs first (every frame);
    gap-g > 1 pairs (every g-th frame) are tracked with the composed consecutive motion as initial guess, so long
    baselines pin the accumulated drift down."""
    rng = np.random.default_rng(1)
    pairs: list[Pair] = []
    by_ij: dict[tuple[int, int], Pair] = {}
    t0 = time.time()
    stats = {"fail": 0}
    for g in sorted(gaps):
        for i in range(start, end - g + 1, g if g > 1 else 1):
            j = i + g
            ga, gb = cache.gray[i], cache.gray[j]
            ma, mb = _mask_at(cache, i), _mask_at(cache, j)
            pts = detect_features(ga, ma)
            init = None
            f1 = float(f_nom[i - start])
            if g > 1:  # compose the consecutive fits i -> j for an initial guess
                chain = [by_ij.get((k, k + 1)) for k in range(i, j)]
                if any(c is None or not c.ok for c in chain):
                    init = None
                else:
                    Rt, st, f = np.eye(3), 0.0, f1
                    for c in chain:
                        Rt = rodrigues(c.w) @ Rt
                        st += c.s
                    u = work_to_canon(pts, k1)
                    d = np.concatenate([u / f1, np.ones((len(u), 1))], 1) @ Rt.T
                    with np.errstate(divide="ignore", invalid="ignore"):
                        p = d[:, :2] / d[:, 2:3] * f1 * np.exp(st)
                    init = _distort_work(p, k1)
                    init = np.where(np.isfinite(init), init, pts)
            a, b = track(ga, gb, pts, mb, init)
            if len(a) < 25:
                stats["fail"] += 1
                pm = Pair(i, j, a, b, np.zeros(3), 0.0, float("nan"), 0, len(a), False)
            else:
                u1, u2 = work_to_canon(a, k1), work_to_canon(b, k1)
                # gap-g pairs may have large rotation: free the thresholds a little
                m = rotzoom_ransac(u1, u2, f1, thr=thr * (1 + 0.5 * (g > 1)), seed=i)
                if m.ok:
                    idx = np.nonzero(m.inl)[0]
                    if len(idx) > max_keep:
                        idx = np.sort(rng.choice(idx, max_keep, replace=False))
                    pm = Pair(i, j, a[idx], b[idx], m.w, m.s, m.rms, m.n_in, len(a), True)
                else:
                    stats["fail"] += 1
                    pm = Pair(i, j, a[:0], b[:0], np.zeros(3), 0.0, float("nan"), m.n_in, len(a), False)
            pairs.append(pm)
            by_ij[(i, j)] = pm
        if log:
            log(f"  motion gap {g}: {len(pairs)} pairs {time.time() - t0:.0f}s fail {stats['fail']}")
    stats["secs"] = time.time() - t0
    return pairs, stats


# --------------------------------------------------------------------------
# 5. joint solve
# --------------------------------------------------------------------------

#: shared parameters: name -> (FD step)
SHARED_NAMES = ("Cx", "Cy", "Cz", "roll", "k1")
_SHARED_STEP = {"Cx": 0.02, "Cy": 0.02, "Cz": 0.02, "roll": 2e-4, "k1": 3e-4}
_FRAME_STEP = np.array([2e-4, 2e-4, 2e-4])


@dataclass
class Shared:
    """The rig quantities shared by every frame: centre ``C``, roll and lens ``k1``."""

    C: Array
    roll: float
    k1: float

    def vec(self) -> Array:
        return np.array([*self.C, self.roll, self.k1], float)

    @classmethod
    def from_vec(cls, v: Array) -> Shared:
        return cls(np.array(v[:3], float), float(v[3]), float(v[4]))

    @classmethod
    def from_rig(cls, rig: ptz.Rig) -> Shared:
        return cls(np.array(rig.C, float), float(rig.roll), float(rig.k1))


def bilinear_stack(stack: Array, fi: Array, x: Array, y: Array, scale: float) -> Array:
    """Bilinear lookup in a (F, h, w) uint8 stack of ``scale``-scaled values at (fi, y, x) (all (n,))."""
    F, h, w = stack.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    flat = stack.reshape(-1)
    base = fi.astype(np.int64) * (h * w) + y0 * w + x0
    v = (flat[base] * (1 - fx) * (1 - fy) + flat[base + 1] * fx * (1 - fy)
         + flat[base + w] * (1 - fx) * fy + flat[base + w + 1] * fx * fy)
    return v / scale


DT_SCALE = 8.0
DT_CAP = 30.0


def build_dt_stack(evs: list[ptz.LineEvidence], blur: float) -> Array:
    """uint8 stack of ``min(dt, DT_CAP)`` (optionally Gaussian blurred) + distance-from-grass, x ``DT_SCALE``."""
    out = np.zeros((len(evs), evs[0].h, evs[0].w), np.uint8)
    for k, ev in enumerate(evs):
        d = np.minimum(ev.dt, DT_CAP)
        if blur > 0:
            d = cv2.GaussianBlur(d, (0, 0), blur)
        out[k] = np.clip((d + np.minimum(ev.gdt, DT_CAP)) * DT_SCALE, 0, 255).astype(np.uint8)
    return out


@dataclass
class ShotProblem:
    """Everything the joint solver needs for one shot (frames ``t0 .. t0 + F - 1``)."""

    t0: int
    F: int
    shared0: Shared
    shared_sigma: Array  # (5,) prior sigma of (Cx, Cy, Cz, roll, k1); 0 = hard fixed
    free_shared: Array  # (5,) bool
    pairs: list[Pair]
    line_frames: Array  # (L,) local frame indices with line evidence
    samples: Array  # (Ns, 2) pitch samples (m)
    anchors: dict[int, tuple[Array, Array]] = field(default_factory=dict)  # local frame -> (state(3), sigma(3))
    points: list[tuple[int, Array, Array, float]] = field(default_factory=list)  # (local frame, pitch (n,2), obs canon (n,2), sigma px)
    sigma_motion: float = 1.0  # canonical px
    sigma_line: float = 4.0  # work px
    sigma_acc: Array = field(default_factory=lambda: np.array([0.0015, 0.0010, 0.004]))  # rad / frame^2
    c_robust: float = 2.5
    #: local frame index -> segment id (frames of different segments / clips / shots are never smoothed together)
    seg_of: Array | None = None


def _pose_rot(pan, tilt, roll):
    return ptz.rotation(pan, tilt, roll)


class _Resid:
    """Residual blocks of a :class:`ShotProblem` with finite-difference Jacobians assembled sparsely."""

    def __init__(self, pb: ShotProblem, stack: Array, trunc: float):
        self.pb = pb
        self.stack = stack
        self.trunc = trunc
        self.ns = int(pb.free_shared.sum())
        self.fs_idx = np.nonzero(pb.free_shared)[0]
        self.nv = self.ns + 3 * pb.F
        ok = [p for p in pb.pairs if p.ok and len(p.a) >= 8]
        self.pairs = ok
        self.pI = np.array([p.i - pb.t0 for p in ok], int)
        self.pJ = np.array([p.j - pb.t0 for p in ok], int)
        self.pid = np.concatenate([np.full(len(p.a), k) for k, p in enumerate(ok)]) if ok else np.zeros(0, int)
        self.a = np.concatenate([p.a for p in ok]) / S if ok else np.zeros((0, 2))  # canonical observed
        self.b = np.concatenate([p.b for p in ok]) / S if ok else np.zeros((0, 2))
        self.nm = 2 * len(self.a)
        self.lf = np.asarray(pb.line_frames, int)
        self.nl = len(self.lf) * len(pb.samples)
        self.P3 = np.column_stack([pb.samples, np.zeros(len(pb.samples))])
        # anchors
        self.anc_f = np.array(sorted(pb.anchors), int)
        self.na = 3 * len(self.anc_f)
        # points
        self.npt = sum(2 * len(p[1]) for p in pb.points)
        seg = np.zeros(pb.F, int) if pb.seg_of is None else np.asarray(pb.seg_of)
        trip = [t for t in range(1, pb.F - 1) if seg[t - 1] == seg[t] == seg[t + 1]] if pb.sigma_acc is not None else []
        self.nacc = 3 * len(trip)
        self.nsh = self.ns
        self.n_res = self.nm + self.nl + self.na + self.npt + self.nacc + self.nsh
        import scipy.sparse as sp

        if trip:
            rows, cols, vals = [], [], []
            for n_, t in enumerate(trip):
                for k in range(3):
                    r = 3 * n_ + k
                    for dt_, v in ((-1, 1.0), (0, -2.0), (1, 1.0)):
                        rows.append(r)
                        cols.append(self.ns + 3 * (t + dt_) + k)
                        vals.append(v / pb.sigma_acc[k])
            self.J_acc = sp.csr_matrix((vals, (rows, cols)), shape=(self.nacc, self.nv))
        else:
            self.J_acc = sp.csr_matrix((0, self.nv))
        self.sh0 = pb.shared0.vec()
        self.sig = np.where(pb.shared_sigma > 0, pb.shared_sigma, 1.0)

    # -- parameter handling
    def split(self, x):
        sh = self.sh0.copy()
        sh[self.fs_idx] = x[: self.ns]
        return Shared.from_vec(sh), x[self.ns:].reshape(-1, 3)

    # -- blocks
    def motion(self, pi, pj, sh: Shared):
        pb = self.pb
        Ri = _pose_rot(pi[:, 0], pi[:, 1], sh.roll)
        Rj = _pose_rot(pj[:, 0], pj[:, 1], sh.roll)
        Rel = Rj @ Ri.transpose(0, 2, 1)
        fi, fj = np.exp(pi[:, 2]), np.exp(pj[:, 2])
        u1 = ptz.undistort_px(self.a, sh.k1) - CANON_CEN
        u2 = ptz.undistort_px(self.b, sh.k1) - CANON_CEN
        k = self.pid
        d = np.column_stack([u1 / fi[k][:, None], np.ones(len(u1))])
        v = np.einsum("nij,nj->ni", Rel[k], d)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = v[:, :2] / v[:, 2:3] * fj[k][:, None]
        r = (p - u2).ravel() / pb.sigma_motion
        return np.where(np.isfinite(r), r, 1e3)

    def lines(self, poses, sh: Shared):
        """(L*Ns,) residuals of the line frames (work px / sigma_line, truncated)."""
        pb = self.pb
        pl = poses[self.lf]
        R = _pose_rot(pl[:, 0], pl[:, 1], sh.roll)
        Xc = np.einsum("fij,sj->fsi", R, self.P3 - sh.C)
        z = Xc[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.exp(pl[:, 2])[:, None, None] * Xc[..., :2] / z[..., None] + CANON_CEN
        shp = uv.shape
        uvd = geo.distort_norm((uv / ptz.CANON).reshape(-1, 2), sh.k1, aspect=ptz.ASPECT).reshape(shp) * ptz.CANON * S
        h, w = self.stack.shape[1:]
        vis = (z > 1.0) & np.all(np.isfinite(uvd), axis=2)
        vis &= (uvd[..., 0] >= 0) & (uvd[..., 0] <= w - 1.001) & (uvd[..., 1] >= 0) & (uvd[..., 1] <= h - 1.001)
        uvd = np.where(vis[..., None], uvd, 0.0)
        fi = np.repeat(np.arange(len(self.lf)), uvd.shape[1])
        v = bilinear_stack(self.stack, fi, uvd[..., 0].ravel(), uvd[..., 1].ravel(), DT_SCALE).reshape(vis.shape)
        r = np.where(vis, np.minimum(v, self.trunc), self.trunc)
        return (r / pb.sigma_line).ravel()

    def points(self, poses, sh: Shared):
        out = []
        for (t, xy, obs, sig) in self.pb.points:
            R = _pose_rot(poses[t:t + 1, 0], poses[t:t + 1, 1], sh.roll)[0]
            Xc = (np.column_stack([xy, np.zeros(len(xy))]) - sh.C) @ R.T
            with np.errstate(divide="ignore", invalid="ignore"):
                uv = np.exp(poses[t, 2]) * Xc[:, :2] / Xc[:, 2:3] + CANON_CEN
            r = (uv - ptz.undistort_px(obs, sh.k1)) / sig
            out.append(np.where(np.isfinite(r), r, 1e3).ravel())
        return np.concatenate(out) if out else np.zeros(0)

    def anchors(self, poses):
        pb = self.pb
        r = [(poses[t] - pb.anchors[t][0]) / pb.anchors[t][1] for t in self.anc_f]
        return np.concatenate(r) if r else np.zeros(0)

    def shared_prior(self, sh: Shared):
        if not self.ns:
            return np.zeros(0)
        v = sh.vec()
        return ((v - self.sh0) / self.sig)[self.fs_idx]

    # -- everything
    def residual(self, x):
        sh, poses = self.split(x)
        pi, pj = poses[self.pI], poses[self.pJ]
        parts = [self.motion(pi, pj, sh) if len(self.pairs) else np.zeros(0), self.lines(poses, sh) if len(self.lf) else
                 np.zeros(0), self.anchors(poses), self.points(poses, sh),
                 (self.J_acc @ x) if self.nacc else np.zeros(0), self.shared_prior(sh)]
        return np.concatenate(parts)

    def jacobian(self, x):
        """Sparse Jacobian of :meth:`residual` (finite differences, vectorised over pairs / frames)."""
        import scipy.sparse as sp

        pb = self.pb
        sh, poses = self.split(x)
        F = pb.F
        rows, cols, vals = [], [], []
        o = 0

        def put(r0, J, c):  # J (n, len(c)) dense block
            rr, cc = np.nonzero(J)
            rows.append(rr + r0)
            cols.append(np.asarray(c)[cc])
            vals.append(J[rr, cc])

        # --- motion: columns = frame i params, frame j params, shared (roll, k1)
        if len(self.pairs):
            pi, pj = poses[self.pI], poses[self.pJ]
            r0 = self.motion(pi, pj, sh)
            npt = len(self.a)
            for k in range(3):
                for which in (0, 1):
                    pi2, pj2 = pi.copy(), pj.copy()
                    (pi2 if which == 0 else pj2)[:, k] += _FRAME_STEP[k]
                    dr = (self.motion(pi2, pj2, sh) - r0) / _FRAME_STEP[k]
                    fr = (self.pI if which == 0 else self.pJ)
                    col = self.ns + 3 * fr[self.pid] + k
                    cols.append(np.repeat(col, 2))
                    rows.append(np.arange(self.nm))
                    vals.append(dr)
            for q, name in enumerate(self.fs_idx):
                nm = SHARED_NAMES[name]
                if nm in ("roll", "k1"):
                    v = sh.vec()
                    v[name] += _SHARED_STEP[nm]
                    dr = (self.motion(pi, pj, Shared.from_vec(v)) - r0) / _SHARED_STEP[nm]
                    cols.append(np.full(self.nm, q))
                    rows.append(np.arange(self.nm))
                    vals.append(dr)
            o += self.nm
        # --- lines
        if len(self.lf):
            r0 = self.lines(poses, sh)
            Ns = len(pb.samples)
            for k in range(3):
                p2 = poses.copy()
                p2[:, k] += _FRAME_STEP[k]
                dr = (self.lines(p2, sh) - r0) / _FRAME_STEP[k]
                rows.append(o + np.arange(self.nl))
                cols.append(self.ns + 3 * np.repeat(self.lf, Ns) + k)
                vals.append(dr)
            for q, name in enumerate(self.fs_idx):
                nm = SHARED_NAMES[name]
                v = sh.vec()
                v[name] += _SHARED_STEP[nm]
                dr = (self.lines(poses, Shared.from_vec(v)) - r0) / _SHARED_STEP[nm]
                rows.append(o + np.arange(self.nl))
                cols.append(np.full(self.nl, q))
                vals.append(dr)
            o += self.nl
        # --- anchors (linear)
        for n_, t in enumerate(self.anc_f):
            for k in range(3):
                rows.append(np.array([o + 3 * n_ + k]))
                cols.append(np.array([self.ns + 3 * t + k]))
                vals.append(np.array([1.0 / pb.anchors[t][1][k]]))
        o += self.na
        # --- finder points: FD per point group
        if self.npt:
            r0 = self.points(poses, sh)
            oo = 0
            for (t, xy, obs, sig) in pb.points:
                n2 = 2 * len(xy)
                for k in range(3):
                    p2 = poses.copy()
                    p2[t, k] += _FRAME_STEP[k]
                    # only this group's rows change: recompute whole (cheap, few groups)
                    dr = (self.points(p2, sh) - r0)[oo:oo + n2] / _FRAME_STEP[k]
                    rows.append(o + oo + np.arange(n2))
                    cols.append(np.full(n2, self.ns + 3 * t + k))
                    vals.append(dr)
                for q, name in enumerate(self.fs_idx):
                    nm = SHARED_NAMES[name]
                    v = sh.vec()
                    v[name] += _SHARED_STEP[nm]
                    dr = (self.points(poses, Shared.from_vec(v)) - r0)[oo:oo + n2] / _SHARED_STEP[nm]
                    rows.append(o + oo + np.arange(n2))
                    cols.append(np.full(n2, q))
                    vals.append(dr)
                oo += n2
            o += self.npt
        # --- smoothness
        o_acc = o
        o += self.nacc
        # --- shared prior
        for q in range(self.ns):
            rows.append(np.array([o + q]))
            cols.append(np.array([q]))
            vals.append(np.array([1.0 / self.sig[self.fs_idx[q]]]))
        J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.n_res, self.nv))
        if self.nacc:
            J = J + sp.vstack([sp.csr_matrix((o_acc, self.nv)), self.J_acc,
                               sp.csr_matrix((self.n_res - o_acc - self.nacc, self.nv))]).tocsr()
        return J


def _robust_w(r: Array, c: float, quad: Array) -> Array:
    """IRLS weights of the Cauchy loss (1 on the quadratic rows)."""
    w = 1.0 / (1.0 + (r / c) ** 2)
    w[quad] = 1.0
    return w


def lm_solve(rs: _Resid, x0: Array, n_iter: int = 25, log=None, tag: str = "") -> tuple[Array, dict]:
    """Levenberg-Marquardt with IRLS Cauchy weights on all data rows (the smoothness / prior rows stay quadratic)."""
    quad = np.zeros(rs.n_res, bool)
    quad[rs.n_res - rs.nsh - rs.nacc: rs.n_res] = True
    c = rs.pb.c_robust
    x = x0.copy()

    def cost(r):
        return float(np.sum(np.where(quad, 0.5 * r * r, 0.5 * c * c * np.log1p((r / c) ** 2))))

    r = rs.residual(x)
    f0 = f = cost(r)
    lam = 1e-3
    hist = [f]
    for it in range(n_iter):
        J = rs.jacobian(x)
        w = _robust_w(r, c, quad)
        sw = np.sqrt(w)
        Jw = J.multiply(sw[:, None]).tocsr()
        A = (Jw.T @ Jw).toarray()
        g = Jw.T @ (sw * r)
        d = np.diag(A).copy()
        d[d < 1e-9] = 1e-9
        improved = False
        for _ in range(8):
            try:
                cf = cho_factor(A + lam * np.diag(d), lower=True, check_finite=False)
                step = -cho_solve(cf, g, check_finite=False)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            xn = x + step
            rn = rs.residual(xn)
            fn = cost(rn)
            if np.isfinite(fn) and fn < f:
                x, r, f, lam, improved = xn, rn, fn, max(lam / 5, 1e-7), True
                break
            lam *= 5
        hist.append(f)
        if log:
            log(f"    {tag} it {it}: cost {f:.1f} lam {lam:.1e} |step| {np.abs(step).max():.2e}")
        if not improved or (hist[-2] - f) < 1e-4 * max(hist[-2], 1.0):
            break
    return x, {"cost0": f0, "cost": f, "iters": len(hist) - 1}


# --------------------------------------------------------------------------
# 6. anchors, initial states, the per-segment solve
# --------------------------------------------------------------------------

MIN_F1 = 0.55
ANCHOR_TOL_M = 1.0  # an anchor farther than this (pitch metres) from the solved camera is an outlier
MARGIN = 0.08


def lines_anchor(ev: ptz.LineEvidence, rig: ptz.Rig, pitch: PitchModel, **kw) -> dict:
    """Per-frame lines-only fit (:func:`ptz.search_lines`, V2 settings) -> json-able record. ``ok`` = accepted (F1 >= 0.55
    and a clear margin); the best hypothesis is kept even when the search abstained."""
    t0 = time.time()
    cam, hyps, info = ptz.search_lines(ev, rig, pitch, margin=MARGIN, min_f1=MIN_F1, **kw)
    rec = {"ok": cam is not None, "reason": info.get("reason"), "f1": info.get("f1"), "margin": info.get("margin"),
           "secs": time.time() - t0}
    best = cam or (hyps[0].cam if hyps else None)
    if best is not None:
        rec.update(pan=best.pan, tilt=best.tilt, logf=best.logf)
    return rec


def cam_from_state(st: Array, sh: Shared) -> ptz.Camera:
    return ptz.Camera(float(st[0]), float(st[1]), float(st[2]), sh.roll, np.asarray(sh.C, float), sh.k1)


def state_from_R(R: Array, logf: float) -> Array:
    """(pan, tilt, logf) of a world -> camera rotation (the optical axis only; the roll stays the rig's)."""
    fwd = R[2]
    return np.array([np.arctan2(fwd[1], fwd[0]), -np.arcsin(np.clip(fwd[2], -1, 1)), logf])


def step_state(st: Array, pair: Pair, sh: Shared, forward: bool = True) -> Array | None:
    """Carry a pose across a motion pair with the *exact* focal length of ``st`` (re-fits the pair's inliers)."""
    if not pair.ok or len(pair.a) < 8:
        return None
    a, b = (pair.a, pair.b) if forward else (pair.b, pair.a)
    u1, u2 = work_to_canon(a, sh.k1), work_to_canon(b, sh.k1)
    f1 = float(np.exp(st[2]))
    w, s = rotzoom_fit(u1, u2, f1)
    R = rodrigues(w) @ ptz.rotation(st[0], st[1], sh.roll)
    return state_from_R(R, st[2] + s)


def chain(st: Array, pairs_by: dict[tuple[int, int], Pair], t_from: int, t_to: int, sh: Shared) -> dict[int, Array]:
    """States at every frame from ``t_from`` (inclusive) to ``t_to`` along consecutive pairs (stops at a failure)."""
    out = {t_from: st}
    step = 1 if t_to >= t_from else -1
    cur = st
    for t in range(t_from, t_to, step):
        pair = pairs_by.get((t, t + 1) if step == 1 else (t - 1, t))
        if pair is None:
            break
        nxt = step_state(cur, pair, sh, forward=step == 1)
        if nxt is None:
            break
        out[t + step] = nxt
        cur = nxt
    return out


def _state_dist(a: Array, b: Array, sh: Shared) -> float:
    return ptz._cam_distance(cam_from_state(a, sh), cam_from_state(b, sh))


def consistent_anchors(anchors: dict[int, Array], pairs_by, sh: Shared, tol_m: float = 2.0, max_gap: int = 30
                       ) -> tuple[dict[int, Array], list[int]]:
    """Motion-consistency vote: every anchor is carried along the motion chain to its neighbouring anchors
    (<= ``max_gap`` frames apart) and compared in pitch metres. An anchor whose every verifiable edge disagrees is
    dropped (a confident but wrong lines-only fit); -> (kept anchors, dropped frames)."""
    ts = sorted(anchors)
    edge: dict[tuple[int, int], bool] = {}
    for a, b in zip(ts[:-1], ts[1:], strict=True):
        if b - a > max_gap:
            continue
        ch = chain(anchors[a], pairs_by, a, b, sh)
        if b in ch:
            edge[(a, b)] = _state_dist(ch[b], anchors[b], sh) < tol_m
    dropped = []
    for k, t in enumerate(ts):
        e = [edge[x] for x in ((ts[k - 1], t) if k else None, (t, ts[k + 1]) if k + 1 < len(ts) else None)
             if x is not None and x in edge]
        if e and not any(e):
            dropped.append(t)
    return {t: v for t, v in anchors.items() if t not in dropped}, dropped


def init_states(t0: int, t1: int, anchors: dict[int, Array], pairs_by, sh: Shared) -> Array:
    """(F, 3) initial states by chaining the nearest anchor's pose over the exact-f motion fits; NaN where no anchor
    can reach (a failed pair in between)."""
    F = t1 - t0 + 1
    st = np.full((F, 3), np.nan)
    ts = sorted(anchors)
    if not ts:
        return st
    for k, t in enumerate(ts):
        lo = t0 if k == 0 else (ts[k - 1] + t) // 2 + 1
        hi = t1 if k == len(ts) - 1 else (t + ts[k + 1]) // 2
        for tt, v in chain(anchors[t], pairs_by, t, hi, sh).items():
            st[tt - t0] = v
        for tt, v in chain(anchors[t], pairs_by, t, lo, sh).items():
            st[tt - t0] = v
    return st


@dataclass
class SegmentResult:
    t0: int
    t1: int
    states: Array  # (F, 3) pan, tilt, logf (NaN = not calibrated)
    shared: Shared
    info: dict = field(default_factory=dict)
    dropped_anchors: list[int] = field(default_factory=list)
    n_anchors: int = 0

    def cam(self, t: int) -> ptz.Camera | None:
        st = self.states[t - self.t0]
        return None if not np.all(np.isfinite(st)) else cam_from_state(st, self.shared)


def solve_segment(cache: ClipCache, t0: int, t1: int, pairs: list[Pair], anchors: dict[int, Array], rig: ptz.Rig,
                  pitch: PitchModel, *, evidence_frames: list[int], shared: Shared | None = None,
                  free_shared: tuple[str, ...] = (), shared_sigma=None, points=None, sigma_line: float = 3.0,
                  anchor_sigma=(0.004, 0.004, 0.04), use_motion: bool = True, log=None) -> SegmentResult:
    """Joint solve of the frames ``t0 .. t1`` (one motion-connected main-camera segment). ``anchors``: frame -> accepted
    lines-only (pan, tilt, logf); ``evidence_frames``: frames whose line evidence enters the cost (the caller decides,
    e.g. withholds the scored frames)."""
    sh = shared or Shared.from_rig(rig)
    F = t1 - t0 + 1
    pairs_by = {(p.i, p.j): p for p in pairs}
    anchors_k, dropped = consistent_anchors(anchors, pairs_by, sh) if use_motion else (dict(anchors), [])
    x_init = init_states(t0, t1, anchors_k, pairs_by, sh)
    res = SegmentResult(t0, t1, np.full((F, 3), np.nan), sh, {}, dropped, len(anchors_k))
    if not anchors_k and not points:
        res.info["reason"] = "no anchor"
        return res
    ok = np.all(np.isfinite(x_init), axis=1)
    if not ok.any():
        res.info["reason"] = "no initial state"
        return res
    # fill NaN states (unreachable frames) by the nearest valid one so the smoothness block stays defined; they
    # are masked out of the answer afterwards
    idx = np.arange(F)
    for k in range(3):
        x_init[~ok, k] = np.interp(idx[~ok], idx[ok], x_init[ok, k])
    evs = [(t, cache.evidence(t)) for t in evidence_frames if t0 <= t <= t1]
    evs = [(t, e) for t, e in evs if e is not None]
    pts_l = [(t - t0, *rest) for t, *rest in (points or [])]
    ev_list, lf = [e for _, e in evs], np.array([t - t0 for t, _ in evs], int)

    def run(anc, x0):
        return joint_solve(F, sh, x0, ev_list, lf, pairs if use_motion else [],
                           {t - t0: (anc[t], np.array(anchor_sigma, float)) for t in anc}, pitch,
                           free_shared=free_shared, shared_sigma=shared_sigma, points=pts_l, sigma_line=sigma_line,
                           t0=t0, log=log)

    out_sh, poses, info = run(anchors_k, x_init)
    # pass 2: a confidently wrong lines-only anchor that no neighbouring anchor contradicts (the motion vote above
    # has nothing to compare it with) stands out against the solution of everything else: drop it and re-solve
    if use_motion and len(anchors_k) > 1:
        bad = [t for t in anchors_k if np.all(np.isfinite(poses[t - t0]))
               and _state_dist(anchors_k[t], poses[t - t0], out_sh) > ANCHOR_TOL_M]
        info["anchor_outliers"] = bad
        if bad and len(bad) < len(anchors_k):
            anchors_k = {t: v for t, v in anchors_k.items() if t not in bad}
            res.n_anchors = len(anchors_k)
            res.dropped_anchors = res.dropped_anchors + bad
            out_sh, poses, info2 = run(anchors_k, poses.copy())
            info["pass2"] = info2
    res.shared = out_sh
    out = poses.copy()
    out[~ok] = np.nan
    res.states = out
    res.info = info
    return res


def joint_solve(F: int, sh: Shared, x_init: Array, evs: list[ptz.LineEvidence], line_frames: Array, pairs: list[Pair],
                anchors: dict[int, tuple[Array, Array]], pitch: PitchModel, *, free_shared=(), shared_sigma=None,
                points=(), sigma_line: float = 3.0, t0: int = 0, seg_of=None, smooth: bool = True, log=None
                ) -> tuple[Shared, Array, dict]:
    """Two-stage (blurred / sharp line distance transform) LM of F frames' (pan, tilt, log f) + the free shared
    quantities. ``pairs`` carry frame indices ``t0 + local``; ``line_frames`` are local indices into ``evs``."""
    free = np.array([n in set(free_shared) for n in SHARED_NAMES])
    sig = np.array(shared_sigma if shared_sigma is not None else [1.0, 3.0, 1.0, np.deg2rad(0.5), 0.01], float)
    samples, _, _ = ptz._samples(pitch, 1.5)
    x = np.concatenate([sh.vec()[free], x_init.ravel()])
    info = {}
    shn = sh
    for stage, (blur, trunc) in enumerate(((1.8, 12.0), (0.0, 5.0))):
        stack = build_dt_stack(evs, blur) if evs else np.zeros((1, 2, 2), np.uint8)
        pb = ShotProblem(t0, F, sh, sig, free, pairs, np.asarray(line_frames, int), samples, dict(anchors),
                         list(points), sigma_line=sigma_line, seg_of=seg_of)
        if not smooth:
            pb.sigma_acc = None
        rs = _Resid(pb, stack, trunc)
        x, inf = lm_solve(rs, x, n_iter=20, log=log, tag=f"stage{stage + 1}")
        info[f"stage{stage + 1}"] = inf
        shn, poses = rs.split(x)
    return shn, poses, info


# --------------------------------------------------------------------------
# 7. shared (rig) parameters over many shots
# --------------------------------------------------------------------------


@dataclass
class Unit:
    """One shot / segment's contribution to the shared fit: frames with an initial pose, line evidence and (optionally)
    the segment's motion pairs. ``frames`` are absolute frame indices of the clip."""

    uid: int
    frames: list[int]
    states: Array  # (len(frames), 3)
    evidence: dict[int, ptz.LineEvidence]
    pairs: list[Pair] = field(default_factory=list)  # i/j are absolute frames, all inside ``frames``
    anchors: dict[int, Array] = field(default_factory=dict)


def fit_shared(units: list[Unit], rig: ptz.Rig, pitch: PitchModel, *, free=("Cx", "Cy", "Cz", "roll", "k1"),
               sigma=(5.0, 10.0, 3.0, np.deg2rad(2.0), 0.05), sigma_line: float = 3.0, log=None
               ) -> tuple[Shared, list[Array], dict]:
    """Bundle-adjustment style fit of the rig quantities (``C``, roll, ``k1``) over the frames of many shots at once:
    every frame has its own (pan, tilt, log f); the shared quantities see all the line evidence and all the motion
    pairs. The prior ``sigma`` is deliberately loose (the leave-out rig is only the starting value)."""
    frames_all: list[tuple[int, int]] = []
    x0, evs, lf, seg_of = [], [], [], []
    local: dict[tuple[int, int], int] = {}
    anchors = {}
    for u in units:
        for k, t in enumerate(u.frames):
            local[(u.uid, t)] = len(frames_all)
            frames_all.append((u.uid, t))
            x0.append(u.states[k])
            seg_of.append(u.uid)
            if t in u.evidence:
                lf.append(len(frames_all) - 1)
                evs.append(u.evidence[t])
            if t in u.anchors:
                anchors[len(frames_all) - 1] = (u.anchors[t], np.array([0.01, 0.01, 0.1]))
    pairs = []
    from dataclasses import replace

    for u in units:
        for p in u.pairs:
            if (u.uid, p.i) in local and (u.uid, p.j) in local:
                pairs.append(replace(p, i=local[(u.uid, p.i)], j=local[(u.uid, p.j)]))
    sh0 = Shared.from_rig(rig)
    shn, poses, info = joint_solve(len(frames_all), sh0, np.array(x0), evs, np.array(lf, int), pairs, anchors, pitch,
                                   free_shared=free, shared_sigma=sigma, sigma_line=sigma_line,
                                   seg_of=np.array(seg_of), smooth=False, log=log)
    out, k = [], 0
    for u in units:
        out.append(poses[k:k + len(u.frames)])
        k += len(u.frames)
    info.update(n_frames=len(frames_all), n_line_frames=len(lf), n_pairs=len(pairs))
    return shn, out, info
