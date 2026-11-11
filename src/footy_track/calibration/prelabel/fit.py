"""Fit a homography to identified candidates and verify it against the image.

RANSAC over (candidate, keypoint) labels -> refit on inliers -> score = the
fraction of reprojected model-line samples (inside the image and on grass)
that land within ``k`` px of white-line pixels.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.candidates import Detection

#: RANSAC inlier threshold, as a fraction of the working-image width (~8 px @1568).
INLIER_FRAC = 0.005
#: A model-line sample counts as supported within this fraction of the width (~4 px).
SUPPORT_FRAC = 0.0026
MIN_SCORE_SAMPLES = 30
MAX_RANSAC_SUBSETS = 600


@dataclass
class Label:
    """One identified point: a normalised image location with a pitch keypoint id."""

    keypoint_id: str
    uv: tuple[float, float]
    confidence: float = 1.0
    candidate_id: int | None = None  # None for snapped extra points


@dataclass
class FitResult:
    ok: bool
    failure: str | None = None
    H: np.ndarray | None = None
    labels: list[Label] = field(default_factory=list)  # de-duplicated input labels
    inliers: list[bool] = field(default_factory=list)
    err_px: list[float | None] = field(default_factory=list)  # working px
    mean_err_m: float | None = None
    score: float = 0.0
    n_samples: int = 0


def _dedupe(labels: list[Label], pitch: PitchModel) -> list[Label]:
    """One label per keypoint id and per image location (highest confidence)."""
    kps = pitch.keypoints()
    best: dict[str, Label] = {}
    for lb in labels:
        if lb.keypoint_id not in kps:
            continue
        cur = best.get(lb.keypoint_id)
        if cur is None or lb.confidence > cur.confidence:
            best[lb.keypoint_id] = lb
    out: list[Label] = []
    for lb in sorted(best.values(), key=lambda x: -x.confidence):
        if any(np.hypot(lb.uv[0] - o.uv[0], lb.uv[1] - o.uv[1]) < 1e-3 for o in out):
            continue
        out.append(lb)
    return out


def _px_err(H: np.ndarray, uv: np.ndarray, xy: np.ndarray, size: tuple[float, float]) -> np.ndarray:
    back, w = geo.apply_homography(np.linalg.inv(H), xy)
    err = np.linalg.norm((back - uv) * np.array(size), axis=1)
    err[(w <= geo.W_EPS) | ~np.isfinite(err)] = np.inf
    return err


def ransac_fit(
    labels: list[Label], pitch: PitchModel, size: tuple[float, float], thresh_px: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray] | str:
    """-> (H, inlier mask) or a failure string."""
    kps = pitch.keypoints()
    uv = np.array([lb.uv for lb in labels], dtype=np.float64)
    xy = np.array([kps[lb.keypoint_id] for lb in labels], dtype=np.float64)
    n = len(labels)
    subsets = list(itertools.combinations(range(n), 4))
    rng = np.random.default_rng(seed)
    if len(subsets) > MAX_RANSAC_SUBSETS:
        idx = rng.choice(len(subsets), MAX_RANSAC_SUBSETS, replace=False)
        subsets = [subsets[i] for i in idx]
    best = None
    last_err = "no 4 points in general position"
    for s in subsets:
        s = list(s)
        try:
            H, _ = geo.fit_homography(uv[s], xy[s], method="dlt")
        except ValueError as exc:
            last_err = str(exc)
            continue
        err = _px_err(H, uv, xy, size)
        inl = err < thresh_px
        if inl.sum() < 4:
            continue
        key = (int(inl.sum()), -float(np.mean(err[inl])))
        if best is None or key > best[0]:
            best = (key, inl)
    if best is None:
        return last_err
    inl = best[1]
    for _ in range(3):  # refit on the inliers until stable
        try:
            H, _ = geo.fit_homography(uv[inl], xy[inl], method="dlt")
        except ValueError as exc:
            return str(exc)
        new = _px_err(H, uv, xy, size) < thresh_px
        if new.sum() < 4:
            return "fewer than 4 inliers after refit"
        if np.array_equal(new, inl):
            break
        inl = new
    H, _ = geo.fit_homography(uv[inl], xy[inl], method="dlt")
    return H, inl


def line_support_score(
    H: np.ndarray, pitch: PitchModel, det: Detection, k_px: float | None = None
) -> tuple[float, int]:
    """Fraction of reprojected model-line samples (in the image and on grass)
    within ``k_px`` of a white-line pixel. Returns (score, n_samples)."""
    w, h = det.size
    k = k_px if k_px is not None else max(2.0, SUPPORT_FRAC * w)
    dist = cv2.distanceTransform((~det.lines_mask).astype(np.uint8), cv2.DIST_L2, 3)
    grass = det.grass
    hits = tot = 0
    for _, runs in geo.project_pitch_model(H, pitch, step_m=0.4).items():
        for run in runs:
            pts = run * np.array([w, h])
            # densify between projected samples so spacing is ~3 px
            for a, b in zip(pts[:-1], pts[1:], strict=True):
                m = max(1, int(np.hypot(*(b - a)) / 3))
                seg = a + (b - a) * np.linspace(0, 1, m, endpoint=False)[:, None]
                x = np.round(seg[:, 0]).astype(int)
                y = np.round(seg[:, 1]).astype(int)
                ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                x, y = x[ok], y[ok]
                on = grass[y, x]
                x, y = x[on], y[on]
                tot += len(x)
                hits += int((dist[y, x] <= k).sum())
    if tot < MIN_SCORE_SAMPLES:
        return 0.0, tot
    return hits / tot, tot


def fit_and_score(
    labels: list[Label],
    pitch: PitchModel,
    det: Detection,
    image_size: tuple[float, float],
    seed: int = 0,
) -> FitResult:
    labels = _dedupe(labels, pitch)
    if len(labels) < 4:
        return FitResult(False, f"only {len(labels)} usable labels (need 4)", labels=labels)
    w, _ = det.size
    r = ransac_fit(labels, pitch, image_size, INLIER_FRAC * image_size[0], seed)
    if isinstance(r, str):
        return FitResult(False, r, labels=labels)
    H, inl = r
    kps = pitch.keypoints()
    uv = np.array([lb.uv for lb in labels])
    xy = np.array([kps[lb.keypoint_id] for lb in labels])
    errs = geo.reprojection_errors(H, uv, xy, image_size)
    err_px = [None if not np.isfinite(e) else float(e) for e in errs["px"]]
    score, n = line_support_score(H, pitch, det)
    if n < MIN_SCORE_SAMPLES:
        return FitResult(
            False, f"too little of the pitch model projects into the grass ({n} samples)",
            H=H, labels=labels, inliers=inl.tolist(), err_px=err_px, n_samples=n,
        )
    return FitResult(
        True, None, H, labels, inl.tolist(), err_px,
        float(np.mean(errs["metres"][inl])), float(score), n,
    )
