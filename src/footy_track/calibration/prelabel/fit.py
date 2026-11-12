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
from footy_track.calibration.refine import LineLabel, RefineResult, dt_residual, model_samples, refine_homography

#: RANSAC inlier threshold, as a fraction of the working-image width (~8 px @1568).
INLIER_FRAC = 0.005
#: A model-line sample counts as supported within this fraction of the width (~4 px).
SUPPORT_FRAC = 0.0026
MIN_SCORE_SAMPLES = 30
#: A fit whose model lines cover less of the white-line mask than this is not worth a
#: reviewer's time (iteration 2: failed refinements of garbage starts score 0.05-0.2, good ones 0.5+).
MIN_PROPOSAL_SCORE = 0.25
MAX_RANSAC_SUBSETS = 600
_PITCH_SCALE = 50.0


@dataclass
class Label:
    """One identified point: a normalised image location with a pitch keypoint id."""

    keypoint_id: str
    uv: tuple[float, float]
    confidence: float = 1.0
    candidate_id: int | None = None  # None when Claude localised the point directly
    uv_label: tuple[float, float] | None = None  # as labelled; set when ``uv`` was moved by refinement
    synthetic: bool = False  # added from the refined H to reach 4 points (not a real label)


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
    # --- before line refinement (== the fields above when refinement was not applied) ---
    H_coarse: np.ndarray | None = None
    err_px_coarse: list[float | None] = field(default_factory=list)
    mean_err_m_coarse: float | None = None
    score_coarse: float = 0.0
    refine: RefineResult | None = None
    dt_residual: float | None = None  # mean truncated DT (px, working res) of the final model samples
    line_labels: list[LineLabel] = field(default_factory=list)


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


def _mixed_dlt(
    labels: list[Label], line_labels: list[LineLabel], pitch: PitchModel
) -> np.ndarray | None:
    """Linear fit of ``H`` from points (2 rows each) and points-on-a-named-line
    (1 row each: the pitch line equation is linear in ``H``). Needs >= 8 rows."""
    segs = pitch.segments()
    kps = pitch.keypoints()
    rows = []
    sc = _PITCH_SCALE
    for lb in labels:
        X, Y = np.array(kps[lb.keypoint_id]) / sc
        u, v = lb.uv[0] - 0.5, lb.uv[1] - 0.5
        p = np.array([u, v, 1.0])
        rows.append(np.concatenate([p, np.zeros(3), -X * p]))
        rows.append(np.concatenate([np.zeros(3), p, -Y * p]))
    for ll in line_labels:
        if ll.line_id not in segs:
            continue
        (x0, y0), (x1, y1) = segs[ll.line_id] / sc
        l = np.cross([x0, y0, 1.0], [x1, y1, 1.0])
        for u, v in ll.points:
            p = np.array([u - 0.5, v - 0.5, 1.0])
            rows.append(np.concatenate([l[0] * p, l[1] * p, l[2] * p]))
    if len(rows) < 8:
        return None
    _, sv, vt = np.linalg.svd(np.array(rows))
    if sv[-2] < 1e-9 * sv[0]:  # rank deficient: more than one null vector
        return None
    Hn = vt[-1].reshape(3, 3)
    T_img = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1.0]])
    T_pitch_inv = np.diag([sc, sc, 1.0])
    H = T_pitch_inv @ Hn @ T_img
    if not np.all(np.isfinite(H)):
        return None
    return geo.orient_homography(H, np.array([[0.5, 0.5]]))


def _subset_starts(labels: list[Label], pitch: PitchModel, seed: int, cap: int = 20) -> list[np.ndarray]:
    kps = pitch.keypoints()
    subs = list(itertools.combinations(range(len(labels)), 4))
    rng = np.random.default_rng(seed)
    if len(subs) > cap:
        subs = [subs[i] for i in rng.choice(len(subs), cap, replace=False)]
    out = []
    for sub in subs:
        try:
            Hs, _ = geo.fit_homography(
                np.array([labels[i].uv for i in sub]), np.array([kps[labels[i].keypoint_id] for i in sub]),
                method="dlt",
            )
        except ValueError:
            continue
        out.append(Hs)
    return out


def _pad_visible(H: np.ndarray, labels: list[Label], pitch: PitchModel, det: Detection, need: int) -> list[Label]:
    """Add synthetic correspondences (keypoints visible in the image under ``H``,
    on grass, spread out) until there are ``need`` of them."""
    w, h = det.size
    have = {lb.keypoint_id for lb in labels}
    kps = pitch.keypoints()
    cand = []
    for kid, xy in kps.items():
        if kid in have:
            continue
        uv, wv = geo.apply_homography(np.linalg.inv(H), np.array([xy]))
        if wv[0] <= geo.W_EPS or not np.all(np.isfinite(uv)):
            continue
        u, v = float(uv[0, 0]), float(uv[0, 1])
        if 0.03 < u < 0.97 and 0.03 < v < 0.97 and det.grass[int(v * (h - 1)), int(u * (w - 1))]:
            cand.append((kid, (u, v)))
    out = list(labels)
    while len(out) < need and cand:
        far = max(cand, key=lambda c: min(np.hypot(c[1][0] - o.uv[0], c[1][1] - o.uv[1]) for o in out))
        cand.remove(far)
        out.append(Label(far[0], far[1], 0.3, None, None, True))
    return out


def fit_and_score(
    labels: list[Label],
    pitch: PitchModel,
    det: Detection,
    image_size: tuple[float, float],
    seed: int = 0,
    line_labels: list[LineLabel] | None = None,
    refine: bool = True,
) -> FitResult:
    """Coarse fit (RANSAC DLT over point labels; a points+lines DLT when there are
    too few points), then line refinement against the white-line mask
    (:mod:`footy_track.calibration.refine`). Both the pre- and post-refinement
    homography, errors and scores are kept on the result."""
    labels = _dedupe(labels, pitch)
    lines = [ll for ll in (line_labels or []) if ll.line_id in pitch.segments() and len(ll.points) >= 2]
    w, _ = det.size
    H = inl = None
    failure = f"only {len(labels)} usable labels (need 4)" if len(labels) < 4 else None
    if len(labels) >= 4:
        r = ransac_fit(labels, pitch, image_size, INLIER_FRAC * image_size[0], seed)
        if isinstance(r, str):
            failure = r
        else:
            H, inl = r
    mixed = _mixed_dlt(labels, lines, pitch) if lines and labels else None
    if H is None and mixed is not None:
        H, inl = mixed, np.ones(len(labels), bool)
        failure = None
    if H is None:
        return FitResult(False, failure, labels=labels, line_labels=lines)

    kps = pitch.keypoints()

    def stats(Hx, lbs):
        uv = np.array([lb.uv for lb in lbs])
        xy = np.array([kps[lb.keypoint_id] for lb in lbs])
        e = geo.reprojection_errors(Hx, uv, xy, image_size)
        return [None if not np.isfinite(v) else float(v) for v in e["px"]], e["metres"]

    err_c, m_c = stats(H, labels)
    inl_arr = np.asarray(inl, bool)
    mean_m_c = float(np.mean(m_c[inl_arr])) if inl_arr.any() else None
    score_c, n_c = line_support_score(H, pitch, det)
    rr = None
    H_final, final_labels, final_inl = H, labels, inl_arr
    if refine:
        starts = _subset_starts(labels, pitch, seed)
        if mixed is not None:
            starts.append(mixed)
        rr = refine_homography(
            H, pitch, det.lines_mask, det.grass,
            prior_points=[(lb.keypoint_id, lb.uv, lb.confidence) for lb, k in zip(labels, inl_arr, strict=True) if k],
            line_labels=lines,
            starts=starts,
        )
        if rr.applied:
            H_final = rr.H
            # the labels are re-expressed through the refined H so stored points and H agree
            moved, flags = [], []
            for lb, was_inlier in zip(labels, inl_arr, strict=True):
                if not was_inlier:  # RANSAC outliers stay where they were labelled, flagged
                    moved.append(lb)
                    flags.append(False)
                    continue
                uv, wv = geo.apply_homography(np.linalg.inv(H_final), np.array([kps[lb.keypoint_id]]))
                ok = bool(wv[0] > geo.W_EPS and np.all(np.isfinite(uv))
                          and -0.05 < uv[0, 0] < 1.05 and -0.05 < uv[0, 1] < 1.05)
                moved.append(Label(lb.keypoint_id, (float(uv[0, 0]), float(uv[0, 1])) if ok else lb.uv,
                                   lb.confidence, lb.candidate_id, lb.uv, False))
                flags.append(ok)
            final_labels = moved
            final_inl = np.array(flags)
            if final_inl.sum() < 4:  # too few real labels to re-derive H on accept: pad with visible keypoints
                final_labels = _pad_visible(H_final, [m for m, k in zip(moved, final_inl, strict=True) if k], pitch, det, 4)
                final_inl = np.ones(len(final_labels), bool)
    err_f, m_f = stats(H_final, final_labels)
    mean_m = float(np.mean(m_f[final_inl])) if final_inl.any() else None
    score, n = line_support_score(H_final, pitch, det) if (rr and rr.applied) else (score_c, n_c)
    G_px = np.linalg.inv(H_final @ np.diag([1.0 / det.size[0], 1.0 / det.size[1], 1.0]))
    dtr, _ = dt_residual(G_px / np.linalg.norm(G_px), model_samples(pitch), det.lines_mask, det.grass)
    if n < MIN_SCORE_SAMPLES:
        return FitResult(
            False, f"too little of the pitch model projects into the grass ({n} samples)",
            H=H_final, labels=final_labels, inliers=final_inl.tolist(), err_px=err_f, n_samples=n,
            H_coarse=H, err_px_coarse=err_c, refine=rr, line_labels=lines,
        )
    if len(labels) < 4 and not (rr and rr.applied):
        # A <4-point (points+lines) DLT is barely constrained: it can match a few
        # line samples perfectly (score ~1.0) while the rest of the pitch is
        # garbage. Only trust it when line refinement confirmed it.
        return FitResult(
            False, f"only {len(labels)} point labels and line refinement did not confirm the fit",
            H=H_final, labels=final_labels, inliers=final_inl.tolist(), err_px=err_f, n_samples=n, score=float(score),
            H_coarse=H, err_px_coarse=err_c, refine=rr, line_labels=lines,
        )
    if score < MIN_PROPOSAL_SCORE:
        return FitResult(
            False, f"the fitted pitch lines do not match the white lines (score {score:.2f} < {MIN_PROPOSAL_SCORE})",
            H=H_final, labels=final_labels, inliers=final_inl.tolist(), err_px=err_f, n_samples=n, score=float(score),
            H_coarse=H, err_px_coarse=err_c, refine=rr, line_labels=lines,
        )
    return FitResult(
        True, None, H_final, final_labels, final_inl.tolist(), err_f, mean_m, float(score), n,
        H_coarse=H, err_px_coarse=err_c, mean_err_m_coarse=mean_m_c, score_coarse=float(score_c),
        refine=rr, dt_residual=None if not np.isfinite(dtr) else float(dtr), line_labels=lines,
    )
