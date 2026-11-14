"""Temporal calibration: image-to-image camera motion, chained to carry a calibration.

The broadcast main camera sits at a fixed position and only pans / tilts / zooms, so
two frames of one shot are related by a homography that does not depend on scene depth
(``H = K2 R K1^-1`` for a pure rotation), and for the ground plane in particular. This
module estimates that homography between frames, chains it, flags failures and shot
cuts, and carries a keyframe calibration to other frames.

Public API (all homographies map **frame a -> frame b**; ``H_norm`` is on normalised
``[0,1]^2`` image coords, the convention of :mod:`footy_track.calibration.geometry`;
``H_px`` is on full-resolution pixel coords of the frames passed in)::

    estimate_pair(frame_a, frame_b, mask_a=None, mask_b=None, scale=480, method="lk",
                  boxes_a=None, boxes_b=None) -> PairMotion
        PairMotion(H_px, H_norm, ok, inliers, n_tracked, residual, score, method, reason)
        ``ok`` is the quality gate; ``residual`` is the median inlier transfer error in
        px of the ``scale``-wide working image (multiply by ``full_width / scale`` for
        full-res px). Frames may be BGR or gray, any size; masks are bool, True = usable
        (static background), at any size. Without masks: the broadcast overlay is excluded
        and, unless ``boxes`` (normalised top-left xywh person/ball boxes) are given, the
        grass/line region minus its non-grass holes (players) is used.

    clip_motion(video, frames=None, ref=None, max_gap=4, scale=480, method="lk",
                boxes=None, mask_mode="auto", progress=None) -> ClipMotion
        One sequential decode; pairs (t-d -> t) for d = 1..max_gap, a shot-cut flag per
        frame, ``ClipMotion.save(path.npz)`` / ``ClipMotion.load``.

    ClipMotion.H(a, b)            normalised H a->b (direct pair if |a-b| <= max_gap and ok,
                                  else the chain with fewest hops); None across a cut/failure
    ClipMotion.to_reference(ref)  (N,3,3) H t->ref for every frame (NaN where unreachable)
    ClipMotion.cuts / .shot       frames that start a shot / shot id per frame
    ClipMotion.records()          one dict per frame (ok, inliers, residual, per gap) for logs

    propagate(H_key, motion, key, t)   carry a keyframe calibration (norm image -> pitch m)
    refine_frame(H, frame, pitch, ...) per-frame line refinement (refine_homography) of it

Frame ``t`` in every array means absolute video frame ``motion.start + i``. Methods:
``"lk"`` (corner features + pyramidal Lucas-Kanade + RANSAC, ~10-20 ms/pair at 480 px),
``"ecc"`` (``cv2.findTransformECC`` homography, coarse-to-fine on blurred gray),
``"lk_ecc"`` (LK, then ECC polishing). The constrained rotation + zoom model is
:func:`project_rot_zoom` (needs the focal length, :func:`focal_from_homography`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration.prelabel.candidates import (
    far_edge_line,
    grass_mask,
    line_mask,
)
from footy_track.calibration.refine import refine_homography

Array = np.ndarray

DEFAULT_SCALE = 480
#: normalised (x0, y0, x1, y1) boxes where broadcasters burn in graphics (scorebug,
#: channel logo, "Premier League" watermark): static in *screen* coords, so they would
#: vote for "no motion". Excluded from every mask.
OVERLAY_BOXES = ((0.0, 0.0, 0.30, 0.15), (0.84, 0.0, 1.0, 0.15), (0.80, 0.84, 1.0, 0.95))
#: ``ok`` gate of a pair
MIN_INLIERS = 25
MAX_RESIDUAL_PX = 1.5  # at the working scale
MIN_ECC_CC = 0.6
RANSAC_PX = 2.0
#: hist-distance (Bhattacharyya, thumbnails) above which t-1 -> t is a cut outright / if the pair also failed
CUT_HIST_HARD = 0.60
CUT_HIST_SOFT = 0.30


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass
class PairMotion:
    H_px: Array  # a -> b, full-res pixel coords
    H_norm: Array  # a -> b, normalised [0,1] coords
    ok: bool
    inliers: int = 0
    n_tracked: int = 0
    residual: float = float("nan")  # median inlier transfer error, px at the working scale
    score: float = 0.0  # inlier ratio (lk) or ECC correlation (ecc)
    method: str = "lk"
    reason: str | None = None

    def inverse(self) -> PairMotion:
        return PairMotion(np.linalg.inv(self.H_px), np.linalg.inv(self.H_norm), self.ok, self.inliers,
                          self.n_tracked, self.residual, self.score, self.method, self.reason)


def _unit(H: Array) -> Array:
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H / np.linalg.norm(H)


def _fail(reason: str, method: str, size=(1.0, 1.0), scale: float = 1.0) -> PairMotion:
    eye = np.eye(3)
    return PairMotion(eye.copy(), eye.copy(), False, 0, 0, float("nan"), 0.0, method, reason)


def _sane(H: Array) -> str | None:
    """Reject homographies no camera move produces (norm coords): flips, huge scale, strong perspective."""
    if not np.all(np.isfinite(H)):
        return "non-finite"
    A = H[:2, :2] / H[2, 2]
    det = float(np.linalg.det(A))
    if not 0.4 < det < 2.5:
        return f"scale change {det:.2f}"
    if np.hypot(H[2, 0], H[2, 1]) / abs(H[2, 2]) > 0.8:
        return "extreme perspective"
    if np.hypot(H[0, 2], H[1, 2]) / abs(H[2, 2]) > 1.2:
        return "motion larger than the frame"
    return None


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------


def overlay_mask(shape: tuple[int, int], boxes=OVERLAY_BOXES) -> Array:
    """bool (h, w), False inside the broadcast overlay boxes."""
    h, w = shape
    m = np.ones((h, w), bool)
    for x0, y0, x1, y1 in boxes:
        m[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))] = False
    return m


def boxes_mask(shape: tuple[int, int], boxes: Array | None, pad: float = 0.25) -> Array:
    """bool (h, w), False inside normalised top-left ``(x, y, w, h)`` boxes (padded by ``pad`` of their size)."""
    h, w = shape
    m = np.ones((h, w), bool)
    for x, y, bw, bh in np.asarray(boxes if boxes is not None else [], dtype=np.float64).reshape(-1, 4):
        x0, x1 = (x - pad * bw) * w, (x + (1 + pad) * bw) * w
        y0, y1 = (y - pad * bh) * h, (y + (1 + pad) * bh) * h
        m[max(0, int(y0)):max(0, int(np.ceil(y1))), max(0, int(x0)):max(0, int(np.ceil(x1)))] = False
    return m


def grass_static_mask(bgr: Array, max_width: int = 320) -> Array:
    """Pitch region without its non-grass, non-white holes (players, referee, ball): True = usable.

    Computed at ``max_width`` (the masks are blobby; this keeps it ~20 ms) and resized back. Holes are
    opened with a disk wider than a pitch line so antialiased line pixels do not punch holes.
    """
    full = bgr.shape[:2]
    small = bgr
    if bgr.shape[1] > max_width:
        small = cv2.resize(bgr, (max_width, round(bgr.shape[0] * max_width / bgr.shape[1])), interpolation=cv2.INTER_AREA)
    grass = grass_mask(small)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = (h >= 30) & (h <= 95) & (s >= 40)
    white = (s <= 90) & (v >= 150)
    hole = (grass & ~(green | white)).astype(np.uint8)
    hole = cv2.morphologyEx(hole, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    k = max(3, small.shape[1] // 100) | 1
    hole = cv2.dilate(hole, np.ones((k, k), np.uint8))
    m = (grass & ~hole.astype(bool)).astype(np.uint8)
    if small.shape[:2] != full:
        m = cv2.resize(m, (full[1], full[0]), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)


def static_mask(bgr_small: Array, boxes: Array | None = None, mode: str = "auto") -> Array:
    """Usable-background mask at the size of ``bgr_small``.

    ``mode``: ``"grass"`` (grass/line region minus holes), ``"all"`` (whole frame minus overlay and
    boxes), ``"auto"`` (``"all"`` when person boxes are given, else ``"grass"``).
    """
    if mode == "auto":
        mode = "all" if boxes is not None else "grass"
    shape = bgr_small.shape[:2]
    m = overlay_mask(shape)
    if boxes is not None:
        m &= boxes_mask(shape, boxes)
    if mode == "grass" and bgr_small.ndim == 3:
        m &= grass_static_mask(bgr_small)
    return m


def load_person_boxes(path: Path | str, tags=("player", "referee", "coach", "person", "player_sub", "ball", "in_play_ball"),
                      min_conf: float = 0.25) -> dict[int, Array]:
    """``frame_index -> (N, 4)`` normalised top-left xywh from a ``machine_labels/*.jsonl`` detections file."""
    out: dict[int, list] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("confidence", 1.0) < min_conf or not (set(r.get("tags", [])) & set(tags)):
            continue
        b = r["bbox"]
        out.setdefault(int(r["frame_index"]), []).append([b["x"], b["y"], b["w"], b["h"]])
    return {k: np.array(v) for k, v in out.items()}


# --------------------------------------------------------------------------
# a prepared frame
# --------------------------------------------------------------------------


@dataclass
class _Prep:
    gray: Array  # uint8 at the working scale
    mask: Array  # bool at the working scale
    size_full: tuple[int, int]  # (W, H) of the frame as given
    s: float  # working px per full px
    pts: Array | None = None
    thumb: Array | None = field(default=None, repr=False)

    @property
    def size_work(self) -> tuple[int, int]:
        return self.gray.shape[1], self.gray.shape[0]


def _to_work(img: Array, scale: int) -> tuple[Array, float]:
    w = img.shape[1]
    s = min(1.0, scale / w)
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return img, s


def prep_frame(frame: Array, mask: Array | None = None, boxes: Array | None = None, scale: int = DEFAULT_SCALE,
               mask_mode: str = "auto", size_full: tuple[int, int] | None = None, already_small: bool = False) -> _Prep:
    """Downscale ``frame`` and build its mask. ``already_small``: ``frame`` is at the working scale already
    (``size_full`` then gives the original size, used for the H_px rescale)."""
    if already_small:
        small, s = frame, frame.shape[1] / size_full[0]
    else:
        small, s = _to_work(frame, scale)
        size_full = (frame.shape[1], frame.shape[0])
    if mask is None:
        m = static_mask(small, boxes, mask_mode)
    else:
        m = np.asarray(mask, bool)
        if m.shape != small.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        m = m & overlay_mask(m.shape)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return _Prep(gray, m, tuple(size_full), s)


# --------------------------------------------------------------------------
# pair estimators
# --------------------------------------------------------------------------


def _features(p: _Prep, max_corners: int = 500) -> Array:
    if p.pts is None:
        m = cv2.erode(p.mask.astype(np.uint8), np.ones((5, 5), np.uint8))
        pts = cv2.goodFeaturesToTrack(p.gray, max_corners, 0.004, 6, mask=m, blockSize=5)
        p.pts = np.zeros((0, 1, 2), np.float32) if pts is None else pts
    return p.pts


def _lk(pa: _Prep, pb: _Prep, k1: float = 0.0) -> tuple[Array | None, Array | None, Array, Array, str | None]:
    """H (work px, a->b) via LK + RANSAC; also the correspondences. Returns (H, inlier mask, p0, p1, reason)."""
    p0 = _features(pa)
    if len(p0) < MIN_INLIERS:
        return None, None, p0.reshape(-1, 2), p0.reshape(-1, 2), "too few features"
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(pa.gray, pb.gray, p0, None, winSize=(21, 21), maxLevel=4, criteria=crit)
    pb0, st2, _ = cv2.calcOpticalFlowPyrLK(pb.gray, pa.gray, p1, None, winSize=(21, 21), maxLevel=4, criteria=crit)
    fb = np.linalg.norm(pb0 - p0, axis=2).ravel()
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.8)
    a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    if len(a):
        h, w = pb.mask.shape
        xi, yi = np.clip(np.round(b[:, 0]).astype(int), 0, w - 1), np.clip(np.round(b[:, 1]).astype(int), 0, h - 1)
        keep = pb.mask[yi, xi]
        a, b = a[keep], b[keep]
    if len(a) < MIN_INLIERS:
        return None, None, a, b, "too few tracked points"
    if k1:  # work in undistorted coords
        w, h = pa.size_work
        asp = w / h
        a = geo.undistort_norm(a / [w, h], k1, aspect=asp) * [w, h]
        b = geo.undistort_norm(b / [w, h], k1, aspect=asp) * [w, h]
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, RANSAC_PX, maxIters=2000, confidence=0.995)
    if H is None:
        return None, None, a, b, "RANSAC failed"
    inl = inl.ravel().astype(bool)
    for _ in range(2):  # least squares on inliers, then a tighter inlier set
        if inl.sum() < 8:
            break
        H2, _ = cv2.findHomography(a[inl], b[inl], 0)
        if H2 is None:
            break
        H = H2
        e = _transfer_err(H, a, b)
        inl = e < 1.0 * 1.5
    return H, inl, a, b, None


def _transfer_err(H: Array, a: Array, b: Array) -> Array:
    ph = np.concatenate([a, np.ones((len(a), 1))], axis=1) @ H.T
    return np.linalg.norm(ph[:, :2] / ph[:, 2:3] - b, axis=1)


def _blur(g: Array) -> Array:
    return cv2.GaussianBlur(g, (0, 0), 1.5)


def _ecc(pa: _Prep, pb: _Prep, init: Array | None = None, iters: int = 40) -> tuple[Array | None, float]:
    """H (work px, a->b) via ECC, coarse-to-fine (1/2 then full); ``init`` is a work-px homography."""
    ga, gb = _blur(pa.gray), _blur(pb.gray)
    mb = (pb.mask & pa.mask).astype(np.uint8)
    W = np.eye(3, dtype=np.float32) if init is None else (init / init[2, 2]).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5)
    cc = 0.0
    levels = (0.5, 1.0)
    for lv in levels:
        if lv != 1.0:
            ta = cv2.resize(ga, None, fx=lv, fy=lv, interpolation=cv2.INTER_AREA)
            tb = cv2.resize(gb, None, fx=lv, fy=lv, interpolation=cv2.INTER_AREA)
            tm = cv2.resize(mb, None, fx=lv, fy=lv, interpolation=cv2.INTER_NEAREST)
        else:
            ta, tb, tm = ga, gb, mb
        S = np.diag([lv, lv, 1.0]).astype(np.float32)
        Wl = (S @ W @ np.linalg.inv(S)).astype(np.float32)
        try:
            cc, Wl = cv2.findTransformECC(ta, tb, Wl, cv2.MOTION_HOMOGRAPHY, crit, tm, 5)
        except cv2.error:
            return None, 0.0
        W = (np.linalg.inv(S) @ Wl @ S).astype(np.float32)
    return W.astype(np.float64), float(cc)


def _finish(H_work: Array | None, pa: _Prep, pb: _Prep, method: str, inl=None, n_tracked=0, score=0.0, residual=float("nan"),
            reason: str | None = None) -> PairMotion:
    if H_work is None:
        return _fail(reason or "failed", method)
    wa, ha = pa.size_work
    wb, hb = pb.size_work
    Wf, Hf = pa.size_full
    Wfb, Hfb = pb.size_full
    Hn = np.diag([1 / wb, 1 / hb, 1.0]) @ H_work @ np.diag([wa, ha, 1.0])
    Hpx = np.diag([Wfb, Hfb, 1.0]) @ Hn @ np.diag([1 / Wf, 1 / Hf, 1.0])
    n_in = int(np.sum(inl)) if inl is not None else 0
    why = reason or _sane(Hn)
    if why is None and method != "ecc":
        if n_in < MIN_INLIERS:
            why = f"only {n_in} inliers"
        elif residual > MAX_RESIDUAL_PX:
            why = f"residual {residual:.2f}px"
    if why is None and method == "ecc" and score < MIN_ECC_CC:
        why = f"ecc correlation {score:.2f}"
    return PairMotion(_unit(Hpx), _unit(Hn), why is None, n_in, int(n_tracked), residual, float(score), method, why)


def _pair(pa: _Prep, pb: _Prep, method: str, k1: float = 0.0) -> PairMotion:
    if method in ("lk", "lk_ecc"):
        H, inl, a, b, why = _lk(pa, pb, k1)
        if H is None:
            return _fail(why or "failed", method)
        res = float(np.median(_transfer_err(H, a[inl], b[inl]))) if inl.any() else float("nan")
        score = float(inl.sum() / max(len(a), 1))
        if method == "lk_ecc":
            W2, cc = _ecc(pa, pb, H, iters=25)
            if W2 is not None and cc > 0:
                H = W2
                inl = _transfer_err(H, a, b) < 1.5
                res = float(np.median(_transfer_err(H, a[inl], b[inl]))) if inl.any() else float("nan")
        return _finish(H, pa, pb, method, inl, len(a), score, res, why)
    if method == "ecc":
        W, cc = _ecc(pa, pb, None)
        return _finish(W, pa, pb, method, None, 0, cc, float("nan"), None if W is not None else "ecc failed")
    raise ValueError(f"unknown method {method!r}")


def estimate_pair(frame_a: Array, frame_b: Array, mask_a: Array | None = None, mask_b: Array | None = None,
                  scale: int = DEFAULT_SCALE, method: str = "lk", boxes_a: Array | None = None,
                  boxes_b: Array | None = None, mask_mode: str = "auto", k1: float = 0.0) -> PairMotion:
    """Homography frame_a -> frame_b of the static background (see module docstring).

    ``k1`` (division-model distortion, see ``geometry``) makes the LK method fit in undistorted coords;
    the returned homography is then between *undistorted* normalised coords.
    """
    pa = prep_frame(frame_a, mask_a, boxes_a, scale, mask_mode)
    pb = prep_frame(frame_b, mask_b, boxes_b, scale, mask_mode)
    return _pair(pa, pb, method, k1)


# --------------------------------------------------------------------------
# clip level
# --------------------------------------------------------------------------


def _hist_thumb(bgr_small: Array) -> Array:
    t = cv2.resize(bgr_small, (96, 54), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).astype(np.float32)
    return cv2.normalize(h, None, 1.0, 0, cv2.NORM_L1).ravel()


def hist_distance(h0: Array, h1: Array) -> float:
    """Bhattacharyya distance of two thumbnail colour histograms: ~0 same scene, ~1 unrelated."""
    return float(cv2.compareHist(h0, h1, cv2.HISTCMP_BHATTACHARYYA))


@dataclass
class ClipMotion:
    """Pairwise motions (t-d -> t, d = 1..max_gap) of a clip plus shot structure; see module docstring."""

    start: int
    size: tuple[int, int]  # (W, H) full-res
    scale: int
    max_gap: int
    method: str
    Hn: Array  # (N, G, 3, 3) normalised, pair (t-d -> t) at [t, d-1]
    ok: Array  # (N, G) bool
    inliers: Array  # (N, G) int
    residual: Array  # (N, G) float
    score: Array  # (N, G) float
    hist_dist: Array  # (N,) distance between frame t-1 and t (0 for the first)
    cut: Array  # (N,) bool: frame t starts a new shot
    ms: Array = field(default_factory=lambda: np.zeros((0, 3)))  # (N, 3) ms per frame: decode, mask, motion

    @property
    def n(self) -> int:
        return len(self.cut)

    @property
    def shot(self) -> Array:
        return np.cumsum(self.cut)

    @property
    def cuts(self) -> list[int]:
        """Absolute frame indices that start a shot (excluding the clip's first frame)."""
        return [self.start + int(i) for i in np.flatnonzero(self.cut) if i > 0]

    def _i(self, t: int) -> int:
        i = int(t) - self.start
        if not 0 <= i < self.n:
            raise IndexError(f"frame {t} outside the motion range [{self.start}, {self.start + self.n})")
        return i

    def pair(self, a: int, b: int) -> PairMotion | None:
        """Direct stored pair for ``0 < |a-b| <= max_gap`` (a -> b), regardless of ``ok``."""
        lo, hi = sorted((a, b))
        d = hi - lo
        if not 1 <= d <= self.max_gap:
            return None
        i = self._i(hi)
        self._i(lo)
        Hn = self.Hn[i, d - 1]
        W, H = self.size
        Hpx = np.diag([W, H, 1.0]) @ Hn @ np.diag([1 / W, 1 / H, 1.0])
        pm = PairMotion(Hpx, Hn, bool(self.ok[i, d - 1]), int(self.inliers[i, d - 1]), 0, float(self.residual[i, d - 1]),
                        float(self.score[i, d - 1]), self.method)
        return pm if a < b else pm.inverse()

    def H(self, a: int, b: int) -> Array | None:
        """Normalised H a -> b, chained over ok pairs with the fewest hops; None if unreachable."""
        if a == b:
            return np.eye(3)
        lo, hi = sorted((a, b))
        self._i(lo)
        self._i(hi)
        if self.shot[hi - self.start] != self.shot[lo - self.start]:
            return None
        hops = {lo: (0, None)}
        for t in range(lo + 1, hi + 1):
            best = None
            for d in range(1, self.max_gap + 1):
                s = t - d
                if s < lo or s not in hops or not self.ok[t - self.start, d - 1]:
                    continue
                c = (hops[s][0] + 1, self.residual[t - self.start, d - 1] if np.isfinite(self.residual[t - self.start, d - 1]) else 9.0, s)
                if best is None or c < best:
                    best = c
            if best is not None:
                hops[t] = (best[0], best[2])
        if hi not in hops:
            return None
        t = hi
        path = [hi]
        while t != lo:
            t = hops[t][1]
            path.append(t)
        path = path[::-1]
        Hacc = np.eye(3)
        for s, t in zip(path[:-1], path[1:], strict=True):
            Hacc = self.Hn[t - self.start, t - s - 1] @ Hacc
            Hacc = Hacc / np.linalg.norm(Hacc)
        Hacc = _unit(Hacc)
        return Hacc if a < b else np.linalg.inv(Hacc)

    def to_reference(self, ref: int) -> Array:
        """``(N, 3, 3)`` normalised H t -> ref for every frame of the range; NaN where unreachable."""
        out = np.full((self.n, 3, 3), np.nan)
        out[self._i(ref)] = np.eye(3)
        cost = {ref: 0}
        for t in range(ref + 1, self.start + self.n):  # forward: H(t->ref) = H(s->ref) @ P(s->t)^-1
            best = None
            for d in range(1, self.max_gap + 1):
                s = t - d
                if s not in cost or not self.ok[t - self.start, d - 1] or self.shot[s - self.start] != self.shot[t - self.start]:
                    continue
                c = (cost[s] + 1, s, d)
                best = c if best is None or c < best else best
            if best is not None:
                _, s, d = best
                cost[t] = best[0]
                out[t - self.start] = _unit(out[s - self.start] @ np.linalg.inv(self.Hn[t - self.start, d - 1]))
        cost = {ref: 0}
        for t in range(ref - 1, self.start - 1, -1):  # backward: H(t->ref) = H(s->ref) @ P(t->s), s = t + d
            best = None
            for d in range(1, self.max_gap + 1):
                s = t + d
                if s not in cost or not self.ok[s - self.start, d - 1] or self.shot[s - self.start] != self.shot[t - self.start]:
                    continue
                c = (cost[s] + 1, s, d)
                best = c if best is None or c < best else best
            if best is not None:
                _, s, d = best
                cost[t] = best[0]
                out[t - self.start] = _unit(out[s - self.start] @ self.Hn[s - self.start, d - 1])
        return out

    def records(self) -> list[dict]:
        return [
            {"frame": self.start + i, "cut": bool(self.cut[i]), "shot": int(self.shot[i]), "hist_dist": float(self.hist_dist[i]),
             "ok": [bool(x) for x in self.ok[i]], "inliers": [int(x) for x in self.inliers[i]],
             "residual": [float(x) for x in self.residual[i]]}
            for i in range(self.n)
        ]

    def save(self, path: Path | str) -> None:
        np.savez_compressed(
            path, Hn=self.Hn, ok=self.ok, inliers=self.inliers, residual=self.residual, score=self.score,
            hist_dist=self.hist_dist, cut=self.cut, ms=self.ms,
            meta=json.dumps({"start": self.start, "size": list(self.size), "scale": self.scale, "max_gap": self.max_gap,
                             "method": self.method}))

    @classmethod
    def load(cls, path: Path | str) -> ClipMotion:
        z = np.load(path, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        return cls(m["start"], tuple(m["size"]), m["scale"], m["max_gap"], m["method"], z["Hn"], z["ok"], z["inliers"],
                   z["residual"], z["score"], z["hist_dist"], z["cut"], z["ms"])


def iter_video_small(video: Path | str, scale: int = DEFAULT_SCALE, start: int = 0, stop: int | None = None):
    """Yield ``(idx, bgr_small, (W, H))`` sequentially (one decode pass).

    cv2 first; if its first frame is all black (OpenCV 5 on interlaced clips, see ``calibration.video``) the
    ffmpeg CLI is piped instead.
    """
    video = Path(video)
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stop = n if stop is None else min(stop, n)
    sw = min(W, scale)
    sh = int(round(H * sw / W))
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ok, f = cap.read()
        if ok and f.max() > 8:
            i = start
            while ok and i < stop:
                yield i, cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA), (W, H)
                i += 1
                ok, f = cap.read()
            return
    finally:
        cap.release()
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(f"cannot decode {video}")
    cmd = [exe, "-v", "error", "-nostdin", "-i", str(video), "-vf", f"select=gte(n\\,{start}),scale={sw}:{sh}",
           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        i = start
        while i < stop:
            buf = proc.stdout.read(sw * sh * 3)
            if len(buf) < sw * sh * 3:
                break
            yield i, np.frombuffer(buf, np.uint8).reshape(sh, sw, 3), (W, H)
            i += 1
    finally:
        proc.kill()
        proc.wait()


def clip_motion(video: Path | str, frames: tuple[int, int] | None = None, ref: int | None = None, max_gap: int = 4,
                scale: int = DEFAULT_SCALE, method: str = "lk", boxes: dict[int, Array] | None = None,
                mask_mode: str = "auto", progress=None) -> ClipMotion:
    """Estimate (t-d -> t) for d = 1..max_gap over ``frames = (start, stop)`` (default: the whole clip).

    ``boxes``: ``frame_index -> (N, 4)`` person/ball boxes (see :func:`load_person_boxes`). ``ref`` is only
    validated here (use :meth:`ClipMotion.to_reference`). Pairs spanning a detected cut are marked not ok.
    """
    start, stop = frames if frames else (0, None)
    hist_prev: Array | None = None
    window: list[tuple[int, _Prep]] = []  # last max_gap prepared frames
    rows: list[dict] = []
    size = (0, 0)
    t_dec = time.perf_counter()
    for idx, small, size in iter_video_small(video, scale, start, stop):
        t0 = time.perf_counter()
        t_decode = (t0 - t_dec) * 1000
        pb = prep_frame(small, None, None if boxes is None else boxes.get(idx), scale, mask_mode,
                        size_full=size, already_small=True)
        hist = _hist_thumb(small)
        hd = hist_distance(hist_prev, hist) if hist_prev is not None else 0.0
        hist_prev = hist
        t1 = time.perf_counter()
        row = {"Hn": np.tile(np.eye(3), (max_gap, 1, 1)), "ok": np.zeros(max_gap, bool), "inl": np.zeros(max_gap, int),
               "res": np.full(max_gap, np.nan), "score": np.zeros(max_gap), "hd": hd}
        hard_cut = hd >= CUT_HIST_HARD
        for d in range(1, max_gap + 1):
            if len(window) < d or hard_cut:
                continue
            idx_a, pa = window[-d]
            if idx_a != idx - d:
                continue
            pm = _pair(pa, pb, method)
            row["Hn"][d - 1], row["ok"][d - 1], row["inl"][d - 1] = pm.H_norm, pm.ok, pm.inliers
            row["res"][d - 1], row["score"][d - 1] = pm.residual, pm.score
        # a shot cut: scene colours change abruptly AND the 1-gap pair failed (or the change is unmistakable)
        row["cut"] = bool(rows) and (hard_cut or (hd >= CUT_HIST_SOFT and not row["ok"][0]))
        if row["cut"]:
            window = []  # nothing before the cut is paired with what follows
            row["ok"][:] = False
        t2 = time.perf_counter()
        row["ms"] = ((t_decode), (t1 - t0) * 1000, (t2 - t1) * 1000)
        rows.append(row)
        window.append((idx, pb))
        window = window[-max_gap:]
        if progress and (idx % 50 == 0):
            progress(idx)
        t_dec = time.perf_counter()
    if not rows:
        raise RuntimeError(f"no frames decoded from {video}")
    st = start
    return ClipMotion(
        st, size, scale, max_gap, method,
        np.stack([r["Hn"] for r in rows]), np.stack([r["ok"] for r in rows]), np.stack([r["inl"] for r in rows]),
        np.stack([r["res"] for r in rows]), np.stack([r["score"] for r in rows]),
        np.array([r["hd"] for r in rows]), np.array([r["cut"] for r in rows]), np.array([r["ms"] for r in rows]))


# --------------------------------------------------------------------------
# carrying a calibration
# --------------------------------------------------------------------------


def propagate(H_key: Array, motion: ClipMotion, key: int, t: int) -> Array | None:
    """Calibration at frame ``t`` from the one at ``key`` (both: normalised image -> pitch m).

    ``H_t = H_key @ H(t -> key)``. None if ``t`` is not reachable from ``key`` (cut, failed pairs).
    """
    M = motion.H(t, key)
    return None if M is None else geo.orient_homography(np.asarray(H_key) @ M, np.array([[0.5, 0.5]]))


@dataclass
class LineMasks:
    """Grass / white-line / far-touchline masks of one frame at the refinement working resolution."""

    grass: Array
    lines: Array
    far: Array


def line_masks(frame_bgr: Array, work_width: int = 1280) -> LineMasks:
    work, _ = _to_work(frame_bgr, work_width)
    grass = grass_mask(work)
    far = far_edge_line(work, grass)
    return LineMasks(grass, line_mask(work, grass, with_far=False) | far, far)


def refine_frame(H: Array, frame: Array | LineMasks, pitch, k1: float = 0.0, fit_k1: bool = False,
                 work_width: int = 1280, **kw):
    """Per-frame line refinement (``refine.refine_homography``) of ``H`` on one frame (BGR, or its
    precomputed :class:`LineMasks`); returns the ``RefineResult``."""
    m = frame if isinstance(frame, LineMasks) else line_masks(frame, work_width)
    return refine_homography(H, pitch, m.lines, m.grass, far_mask=m.far, k1=k1, fit_k1=fit_k1, **kw)


# --------------------------------------------------------------------------
# constrained camera model: rotation about a fixed centre + zoom
# --------------------------------------------------------------------------


def _centred(aspect: float) -> Array:
    """norm image -> centred, isotropic 'width units' (principal point at the centre)."""
    return np.array([[1, 0, -0.5], [0, 1 / aspect, -0.5 / aspect], [0, 0, 1.0]])


def focal_from_homography(H_norm: Array, aspect: float = geo.DEFAULT_ASPECT) -> float | None:
    """Focal length (in image widths) implied by a pitch homography under square pixels and a central
    principal point: the two pitch axes must map to orthogonal, equal-length rays. None if ill-conditioned."""
    G = _centred(aspect) @ np.linalg.inv(np.asarray(H_norm, np.float64))
    g1, g2 = G[:, 0], G[:, 1]
    num = -(g1[0] * g2[0] + g1[1] * g2[1])
    den = g1[2] * g2[2]
    if abs(den) < 1e-12 or num / den <= 0:
        return None
    return float(np.sqrt(num / den))


def _rot(w: Array) -> Array:
    R, _ = cv2.Rodrigues(np.asarray(w, np.float64))
    return R


def rot_zoom_homography(w: Array, f_a: float, f_b: float, aspect: float = geo.DEFAULT_ASPECT) -> Array:
    """Normalised a -> b homography of a camera rotated by Rodrigues vector ``w`` and zoomed f_a -> f_b."""
    T = _centred(aspect)
    Ka, Kb = np.diag([f_a, f_a, 1.0]), np.diag([f_b, f_b, 1.0])
    return np.linalg.inv(T) @ Kb @ _rot(w) @ np.linalg.inv(Ka) @ T


def project_rot_zoom(H_norm: Array, f_a: float, aspect: float = geo.DEFAULT_ASPECT, grid: int = 9, iters: int = 15
                     ) -> tuple[Array, float, float]:
    """Nearest rotation + zoom motion to a measured a -> b homography.

    Fits (Rodrigues ``w``, f_b) by Gauss-Newton on the transfer of a ``grid x grid`` point lattice (in px of a
    1000-wide image), so the unconstrained shear / perspective components, which random-walk when chained, are
    dropped. Returns ``(H_constrained_norm, f_b, rms_px)``.
    """
    gx, gy = np.meshgrid(np.linspace(0.05, 0.95, grid), np.linspace(0.05, 0.95, grid))
    P = np.stack([gx.ravel(), gy.ravel()], 1)
    tgt, _ = geo.apply_homography(H_norm, P)
    sc = np.array([1000.0, 1000.0 / aspect])

    def res(p):
        Hm = rot_zoom_homography(p[:3], f_a, f_a * np.exp(p[3]), aspect)
        q, _ = geo.apply_homography(Hm, P)
        return ((q - tgt) * sc).ravel()

    p = np.zeros(4)
    for _ in range(iters):
        r = res(p)
        J = np.empty((len(r), 4))
        for k in range(4):
            dp = np.zeros(4)
            dp[k] = 1e-5
            J[:, k] = (res(p + dp) - r) / 1e-5
        step = np.linalg.solve(J.T @ J + 1e-9 * np.eye(4), -J.T @ r)
        p = p + step
        if np.abs(step).max() < 1e-9:
            break
    f_b = f_a * float(np.exp(p[3]))
    return rot_zoom_homography(p[:3], f_a, f_b, aspect), f_b, float(np.sqrt(np.mean(res(p) ** 2)))


# --------------------------------------------------------------------------
# metrics shared by the experiment script and the tests
# --------------------------------------------------------------------------


def pitch_grid_error(H: Array, H_ref: Array, pts_norm: Array) -> Array:
    """Distance in metres between ``H`` and ``H_ref`` (norm image -> pitch) at image points ``pts_norm`` (N, 2)."""
    a, _ = geo.apply_homography(H, pts_norm)
    b, _ = geo.apply_homography(H_ref, pts_norm)
    return np.linalg.norm(a - b, axis=1)
