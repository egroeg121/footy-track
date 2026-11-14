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
from footy_track.calibration.geometry import analyse_configuration
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

    keypoint_id: str | None
    uv: tuple[float, float]
    confidence: float = 1.0
    candidate_id: int | None = None  # None when Claude localised the point directly
    uv_label: tuple[float, float] | None = None  # as labelled; set when ``uv`` was moved by refinement
    synthetic: bool = False  # added from the refined H to reach 4 points (not a real label)
    xy: tuple[float, float] | None = None  # pitch metres of a free (non-keypoint) synthetic point


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
    distortion: dict | None = None  # lens model of ``H`` (geometry.make_distortion); None = plain homography
    line_labels: list[LineLabel] = field(default_factory=list)


def label_xy(lb: Label, pitch: PitchModel) -> tuple[float, float]:
    """Pitch metres of a label (its keypoint, or its free ``xy`` for a synthetic point)."""
    return lb.xy if lb.keypoint_id is None else tuple(pitch.keypoints()[lb.keypoint_id])


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


def _px_err(H: np.ndarray, uv: np.ndarray, xy: np.ndarray, size: tuple[float, float],
            distortion: dict | None = None) -> np.ndarray:
    back, w = geo.pitch_to_image(H, xy, distortion)
    err = np.linalg.norm((back - uv) * np.array(size), axis=1)
    err[(w <= geo.W_EPS) | ~np.isfinite(err)] = np.inf
    return err


def ransac_fit(
    labels: list[Label], pitch: PitchModel, size: tuple[float, float], thresh_px: float, seed: int = 0,
    distortion: dict | None = None,
) -> tuple[np.ndarray, np.ndarray] | str:
    """-> (H, inlier mask) or a failure string. Label points are observed (distorted) coords."""
    kps = pitch.keypoints()
    uv_obs = np.array([lb.uv for lb in labels], dtype=np.float64)
    uv = geo.undistort_points(uv_obs, distortion)
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
    H: np.ndarray, pitch: PitchModel, det: Detection, k_px: float | None = None, distortion: dict | None = None
) -> tuple[float, int]:
    """Fraction of reprojected model-line samples (in the image and on grass)
    within ``k_px`` of a white-line pixel. Returns (score, n_samples)."""
    w, h = det.size
    k = k_px if k_px is not None else max(2.0, SUPPORT_FRAC * w)
    dist = cv2.distanceTransform((~det.lines_mask).astype(np.uint8), cv2.DIST_L2, 3)
    grass = det.grass
    hits = tot = 0
    for _, runs in geo.project_pitch_model(H, pitch, step_m=0.4, distortion=distortion).items():
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
    labels: list[Label], line_labels: list[LineLabel], pitch: PitchModel, distortion: dict | None = None
) -> np.ndarray | None:
    """Linear fit of ``H`` from points (2 rows each) and points-on-a-named-line
    (1 row each: the pitch line equation is linear in ``H``). Needs >= 8 rows."""
    segs = pitch.segments()
    kps = pitch.keypoints()
    rows = []
    sc = _PITCH_SCALE
    for lb in labels:
        X, Y = np.array(kps[lb.keypoint_id]) / sc
        u, v = geo.undistort_points(lb.uv, distortion)[0] - 0.5
        p = np.array([u, v, 1.0])
        rows.append(np.concatenate([p, np.zeros(3), -X * p]))
        rows.append(np.concatenate([np.zeros(3), p, -Y * p]))
    for ll in line_labels:
        if ll.line_id not in segs:
            continue
        (x0, y0), (x1, y1) = segs[ll.line_id] / sc
        l = np.cross([x0, y0, 1.0], [x1, y1, 1.0])
        for u, v in geo.undistort_points(np.array(ll.points), distortion):
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


def _subset_starts(labels: list[Label], pitch: PitchModel, seed: int, cap: int = 20,
                   dist: dict | None = None) -> list[np.ndarray]:
    kps = pitch.keypoints()
    subs = list(itertools.combinations(range(len(labels)), 4))
    rng = np.random.default_rng(seed)
    if len(subs) > cap:
        subs = [subs[i] for i in rng.choice(len(subs), cap, replace=False)]
    out = []
    for sub in subs:
        try:
            Hs, _ = geo.fit_homography(
                geo.undistort_points(np.array([labels[i].uv for i in sub]), dist), np.array([kps[labels[i].keypoint_id] for i in sub]),
                method="dlt",
            )
        except ValueError:
            continue
        out.append(Hs)
    return out


#: A refit of the stored points may differ from H_final by at most this much (metres,
#: over a grid of image points) before the stored set is padded to pin H_final down.
REFIT_TOL_M = 0.25


def refit_gap_m(H_ref: np.ndarray, uv: np.ndarray, xy: np.ndarray, distortion: dict | None = None) -> float:
    """Max metres between ``H_ref`` and the DLT refit of (uv, xy), over an image grid.
    ``inf`` when the points are degenerate (the refit is refused) or the refit is unusable.
    ``uv`` are observed (distorted) points; the refit undistorts them like accept does."""
    try:
        H2, _ = geo.fit_homography(geo.undistort_points(uv, distortion), xy, method="dlt")
    except ValueError:
        return float("inf")
    g = np.array([(u, v) for u in np.linspace(0.05, 0.95, 5) for v in np.linspace(0.05, 0.95, 5)])
    a, wa = geo.apply_homography(H_ref, g)  # the grid is in undistorted coords for both
    b, wb = geo.apply_homography(H2, g)
    ok = (wa > geo.W_EPS) & (wb > geo.W_EPS)
    if not ok.any():
        return float("inf")
    d = np.linalg.norm(a[ok] - b[ok], axis=1)
    return float(np.max(d)) if np.all(np.isfinite(d)) else float("inf")


def pad_points(
    H: np.ndarray,
    uv: list[tuple[float, float]],
    kids: list[str | None],
    xy: list[tuple[float, float]],
    pitch: PitchModel,
    det: Detection | None = None,
    need: int = 4,
    max_extra: int = 6,
    distortion: dict | None = None,
) -> list[tuple[str | None, tuple[float, float], tuple[float, float]]]:
    """Synthetic (keypoint id or None, uv, pitch metres) additions, projected through ``H``, so that the point set
    has >= ``need`` points, is non-degenerate (some 4 in general position) and refits to ``H``.

    Candidates are keypoints visible in the image under ``H`` (on grass when ``det`` is given),
    not already used, then free points on a 5 m pitch grid (a view that shows only one line,
    e.g. the halfway line, has too few keypoints off it). Each step picks the candidate that leaves the most general-position
    4-subsets (this is what breaks a collinear set), ties broken by spread."""
    kps = pitch.keypoints()
    have = {k for k in kids if k}
    cand = []
    pool = [(k, p) for k, p in kps.items() if k not in have]
    pool += [
        (None, (float(x), float(y)))
        for x in np.arange(-pitch.hl, pitch.hl + 1e-9, 5.0) for y in np.arange(-pitch.hw, pitch.hw + 1e-9, 5.0)
    ]
    for kid, p in pool:
        pt, wv = geo.pitch_to_image(H, np.array([p]), distortion)
        if wv[0] <= geo.W_EPS or not np.all(np.isfinite(pt)):
            continue
        u, v = float(pt[0, 0]), float(pt[0, 1])
        if not (0.03 < u < 0.97 and 0.03 < v < 0.97):
            continue
        if det is not None:
            w, h = det.size
            if not det.grass[int(v * (h - 1)), int(u * (w - 1))]:
                continue
        cand.append((kid, (u, v), p))
    cur_uv, cur_xy = list(uv), list(xy)
    added: list = []

    def n_general(pts_uv, pts_xy) -> int:
        if len(pts_uv) < 4:
            return 0
        return analyse_configuration(np.array(pts_uv), np.array(pts_xy))["n_general_subsets"] or 0

    while cand and len(added) < max_extra and (
        len(cur_uv) < need or n_general(cur_uv, cur_xy) == 0
        or refit_gap_m(H, np.array(cur_uv), np.array(cur_xy), distortion) > REFIT_TOL_M
    ):
        def key(c):
            spread = min((np.hypot(c[1][0] - o[0], c[1][1] - o[1]) for o in cur_uv), default=1.0)
            return (n_general(cur_uv + [c[1]], cur_xy + [c[2]]), c[0] is not None, spread)

        best = max(cand, key=key)
        cand.remove(best)
        cur_uv.append(best[1])
        cur_xy.append(best[2])
        added.append((best[0], best[1], best[2]))
    return added


def _pad_visible(H: np.ndarray, labels: list[Label], pitch: PitchModel, det: Detection, need: int,
                 distortion: dict | None = None) -> list[Label]:
    """Add synthetic correspondences (keypoints visible in the image under ``H``, on grass,
    spread out and chosen off any line the existing points are collinear on) until there are
    ``need`` of them, they can determine ``H`` and a refit reproduces ``H``."""
    add = pad_points(
        H, [lb.uv for lb in labels], [lb.keypoint_id for lb in labels],
        [label_xy(lb, pitch) for lb in labels], pitch, det, need, distortion=distortion,
    )
    return list(labels) + [Label(k, uv, 0.3, None, None, True, None if k else xy) for k, uv, xy in add]



#: refit without the far-edge strip only when that lowers the median label error (px at 1568 wide) by this much
FAR_OFF_MARGIN_PX = 4.0


def _label_px_err(rr, labels, inl, pitch, image_size, det) -> float:
    """Median reprojection error (px, scaled to 1568 wide) of the inlier labels under a refinement result."""
    dist = geo.make_distortion(rr.k1, det.size[0] / det.size[1])
    sel = [lb for lb, k in zip(labels, inl, strict=True) if k]
    if not sel:
        return float("inf")
    uv = np.array([lb.uv for lb in sel])
    xy = np.array([label_xy(lb, pitch) for lb in sel])
    e = geo.reprojection_errors(rr.H, uv, xy, image_size, dist)["px"]
    return float(np.median(e[np.isfinite(e)]) * 1568.0 / image_size[0]) if np.isfinite(e).any() else float("inf")


def fit_and_score(
    labels: list[Label],
    pitch: PitchModel,
    det: Detection,
    image_size: tuple[float, float],
    seed: int = 0,
    line_labels: list[LineLabel] | None = None,
    refine: bool = True,
    k1: float = 0.0,
    fit_k1: bool = False,
    k1_prior: float = 1.0,
    camera=None,
) -> FitResult:
    """Coarse fit (RANSAC DLT over point labels; a points+lines DLT when there are
    too few points), then line refinement against the white-line mask
    (:mod:`footy_track.calibration.refine`). Both the pre- and post-refinement
    homography, errors and scores are kept on the result.

    Lens distortion (see :mod:`footy_track.calibration.geometry`): ``k1`` is a given (e.g. per-clip)
    division-model coefficient, held fixed; ``fit_k1`` estimates it jointly with ``H`` in the
    refinement (``k1`` is then only the starting value, ``k1_prior`` the pull towards 0).
    ``FitResult.distortion`` is the lens model the returned ``H`` goes with. Labels, stored points
    and errors stay in observed (distorted) image coordinates."""
    labels = _dedupe(labels, pitch)
    lines = [ll for ll in (line_labels or []) if ll.line_id in pitch.segments() and len(ll.points) >= 2]
    w, _ = det.size
    dist_in = geo.make_distortion(k1, det.size[0] / det.size[1])  # what the coarse fit is made with
    H = inl = None
    failure = f"only {len(labels)} usable labels (need 4)" if len(labels) < 4 else None
    if len(labels) >= 4:
        r = ransac_fit(labels, pitch, image_size, INLIER_FRAC * image_size[0], seed, dist_in)
        if isinstance(r, str):
            failure = r
        else:
            H, inl = r
    mixed = _mixed_dlt(labels, lines, pitch, dist_in) if lines and labels else None
    if H is None and mixed is not None:
        H, inl = mixed, np.ones(len(labels), bool)
        failure = None
    if H is None:
        return FitResult(False, failure, labels=labels, line_labels=lines)

    kps = pitch.keypoints()

    def stats(Hx, lbs, dist=None):
        uv = np.array([lb.uv for lb in lbs])
        xy = np.array([label_xy(lb, pitch) for lb in lbs])
        e = geo.reprojection_errors(Hx, uv, xy, image_size, dist)
        return [None if not np.isfinite(v) else float(v) for v in e["px"]], e["metres"]

    err_c, m_c = stats(H, labels, dist_in)
    inl_arr = np.asarray(inl, bool)
    mean_m_c = float(np.mean(m_c[inl_arr])) if inl_arr.any() else None
    score_c, n_c = line_support_score(H, pitch, det, distortion=dist_in)
    rr = None
    H_final, final_labels, final_inl = H, labels, inl_arr
    dist_f = dist_in
    if refine:
        starts = _subset_starts(labels, pitch, seed, dist=dist_in)
        if mixed is not None:
            starts.append(mixed)
        rr = refine_homography(
            H, pitch, det.lines_mask, det.grass,
            prior_points=[(lb.keypoint_id, lb.uv, lb.confidence) for lb, k in zip(labels, inl_arr, strict=True) if k],
            line_labels=lines,
            far_mask=det.far_line,
            starts=starts,
            k1=k1, fit_k1=fit_k1, k1_prior=k1_prior, camera=camera,
        )
        if camera and rr.applied and det.far_line is not None and det.far_line.any():
            # The far-edge strip can be the hoarding base a metre or two behind the real touchline (circle-only
            # views): it then shrinks the whole pitch. Refit without it and keep that one if it explains the
            # labels (independent evidence of scale) clearly better.
            rr2 = refine_homography(
                H, pitch, det.lines_mask, det.grass,
                prior_points=[(lb.keypoint_id, lb.uv, lb.confidence) for lb, k in zip(labels, inl_arr, strict=True) if k],
                line_labels=lines, far_mask=None, starts=starts, k1=k1, fit_k1=fit_k1, k1_prior=k1_prior, camera=camera,
            )
            if rr2.applied and _label_px_err(rr2, labels, inl_arr, pitch, image_size, det) + FAR_OFF_MARGIN_PX < \
                    _label_px_err(rr, labels, inl_arr, pitch, image_size, det):
                rr = rr2
        if rr.applied:
            H_final = rr.H
            dist_f = geo.make_distortion(rr.k1, det.size[0] / det.size[1])
            # the labels are re-expressed through the refined H so stored points and H agree
            moved, flags = [], []
            for lb, was_inlier in zip(labels, inl_arr, strict=True):
                if not was_inlier:  # RANSAC outliers stay where they were labelled, flagged
                    moved.append(lb)
                    flags.append(False)
                    continue
                uv, wv = geo.pitch_to_image(H_final, np.array([kps[lb.keypoint_id]]), dist_f)
                ok = bool(wv[0] > geo.W_EPS and np.all(np.isfinite(uv))
                          and -0.05 < uv[0, 0] < 1.05 and -0.05 < uv[0, 1] < 1.05)
                moved.append(Label(lb.keypoint_id, (float(uv[0, 0]), float(uv[0, 1])) if ok else lb.uv,
                                   lb.confidence, lb.candidate_id, lb.uv, False))
                flags.append(ok)
            final_labels = moved
            final_inl = np.array(flags)
    # The stored inlier points must refit to H_final (accept re-fits from them, and the
    # human reviewed the H on screen): too few, collinear (degenerate) or drifting -> pad.
    real = [m for m, k in zip(final_labels, final_inl, strict=True) if k]
    r_uv = np.array([m.uv for m in real]).reshape(-1, 2)
    r_xy = np.array([label_xy(m, pitch) for m in real]).reshape(-1, 2)
    if len(real) < 4 or refit_gap_m(H_final, r_uv, r_xy, dist_f) > REFIT_TOL_M:
        padded = _pad_visible(H_final, real, pitch, det, 4, dist_f)
        if len(padded) > len(real):
            final_labels = list(final_labels) + padded[len(real):]
            final_inl = np.concatenate([final_inl, np.ones(len(padded) - len(real), bool)])
    err_f, m_f = stats(H_final, final_labels, dist_f)
    mean_m = float(np.mean(m_f[final_inl])) if final_inl.any() else None
    score, n = line_support_score(H_final, pitch, det, distortion=dist_f) if (rr and rr.applied) else (score_c, n_c)
    G_px = np.linalg.inv(H_final @ np.diag([1.0 / det.size[0], 1.0 / det.size[1], 1.0]))
    dtr, _ = dt_residual(G_px / np.linalg.norm(G_px), model_samples(pitch), det.lines_mask, det.grass,
                         k1=(dist_f or {}).get("k1", 0.0))
    if n < MIN_SCORE_SAMPLES:
        return FitResult(
            False, f"too little of the pitch model projects into the grass ({n} samples)",
            H=H_final, labels=final_labels, inliers=final_inl.tolist(), err_px=err_f, n_samples=n,
            H_coarse=H, err_px_coarse=err_c, refine=rr, line_labels=lines, distortion=dist_f,
        )
    if len(labels) < 4 and not (rr and rr.applied):
        # A <4-point (points+lines) DLT is barely constrained: it can match a few
        # line samples perfectly (score ~1.0) while the rest of the pitch is
        # garbage. Only trust it when line refinement confirmed it.
        return FitResult(
            False, f"only {len(labels)} point labels and line refinement did not confirm the fit",
            H=H_final, labels=final_labels, inliers=final_inl.tolist(), err_px=err_f, n_samples=n, score=float(score),
            H_coarse=H, err_px_coarse=err_c, refine=rr, line_labels=lines, distortion=dist_f,
        )
    if score < MIN_PROPOSAL_SCORE:
        return FitResult(
            False, f"the fitted pitch lines do not match the white lines (score {score:.2f} < {MIN_PROPOSAL_SCORE})",
            H=H_final, labels=final_labels, inliers=final_inl.tolist(), err_px=err_f, n_samples=n, score=float(score),
            H_coarse=H, err_px_coarse=err_c, refine=rr, line_labels=lines, distortion=dist_f,
        )
    return FitResult(
        True, None, H_final, final_labels, final_inl.tolist(), err_f, mean_m, float(score), n,
        H_coarse=H, err_px_coarse=err_c, mean_err_m_coarse=mean_m_c, score_coarse=float(score_c),
        refine=rr, dt_residual=None if not np.isfinite(dtr) else float(dtr), line_labels=lines,
        distortion=dist_f,
    )
