"""PTZ rig proposals (V2 of ``docs/design/calibration_ptz.md``): fit a ground's rig, solve frames, build records.

Pure-ish building blocks used by ``scripts/ptz_propose.py``:

* :func:`fit_rig_robust`   rig fit that drops inconsistent points / frames (reported, never silent)
* :func:`ransac_points`    consensus over possibly wrong (LLM) point labels -> camera + inlier mask
* :func:`solve_frame`      per-frame decision: LLM points + lines, else lines only, else abstain
* :func:`pick_camera`      the (pure) decision rule between the points candidate and the lines candidate
* :func:`projected_keypoints`, :func:`proposal_record`, :func:`failed_record` build records that the review page and
  editor read (same writer / schema as the pre-labelling pipeline: ``prelabel.pipeline`` / ``proposals``)
* :func:`evenly_spaced`    extra-frame selection

The rig files live in ``calibration/camera_rigs/<ground>.json`` (``Rig.to_json``).
"""

# ruff: noqa: PLR0912, PLR0913, PLR0915, PLC0415
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration import ptz
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel import pipeline as pl

RIG_DIR = Path(__file__).parent / "camera_rigs"

#: ground -> clip-name prefixes (video stems / canonical ids), the rig's prior centre (from the proposal's
#: decomposition table) and the clip alias of the broadcaster
GROUNDS: dict[str, dict] = {
    "vitality": {"prefixes": ("arsenal_bournemouth", "bournemouth"), "C0": (-0.1, -49.3, 7.3)},
    "emirates": {"prefixes": ("arsenal_mancity", "arsenal_astonvilla", "mancity", "astonvilla"),
                 "C0": (0.0, -65.0, 15.0)},
}

#: decision thresholds. ``support`` is :func:`ptz.explained_f1` of a camera on the frame's line evidence (bounded in
#: [0, 1]; V1 used the unbounded 1 - score, which went to -1..-4 on correct Vitality fits). The Emirates GT (leave-shot-out
#: lines-only, 67 frames) separates cleanly: every answer within 1 m has F1 >= 0.61, every wrong best hypothesis <= 0.48,
#: so the threshold sits in that gap; margin is the V1 value. Bournemouth was not used to set either.
MIN_SUPPORT = 0.55
MARGIN = 0.08
CONFIRM_DELTA = 0.15  # two labels with no lines hypothesis agreeing need support >= MIN_SUPPORT + this
POINT_PX = 10.0  # a labelled point is an inlier when it reprojects within this many canonical (1920 px) px
AGREE_M = 3.0  # two cameras "agree" when the pitch-plane displacement over a probe grid is below this
MAIN_P = 0.5  # shot classifier: below this and weakly supported -> not the main camera
MAIN_SUPPORT = 0.60
#: >= STRONG_N consensus points reprojecting within STRONG_RMS px (1920 wide) are accepted without line support:
#: the white-line mask of the high Bournemouth camera is too weak to veto a fit that many independent labels agree on
STRONG_N = 4
STRONG_RMS = 6.0
STRONG_N3_RMS = 4.0  # exactly 3 consensus points must be tighter
MIN_SPREAD_M = 3.0  # and not (nearly) collinear: points on one pitch line leave a camera degree of freedom open


def ground_of(clip: str) -> str | None:
    for g, d in GROUNDS.items():
        if clip.startswith(d["prefixes"]):
            return g
    return None


# --------------------------------------------------------------------------
# rig fit
# --------------------------------------------------------------------------


@dataclass
class RigFit:
    rig: ptz.Rig
    cams: dict[str, ptz.Camera]  # key -> camera of the frames used
    used: list[ptz.RigFrame]
    excluded: list[dict] = field(default_factory=list)  # frames inconsistent with a single C: {key, why, ...}
    dropped_points: list[dict] = field(default_factory=list)  # {key, index, res_px}


def _res_px(fr: ptz.RigFrame, c: ptz.Camera) -> np.ndarray:
    return ptz._frame_resid_px(fr, c.C, c.roll, c.k1, c.pan, c.tilt, c.logf)


def fit_rig_robust(frames: list[ptz.RigFrame], *, C0, sigma_C=(1.0, 3.0, 1.0), point_px: float = 12.0,
                   frame_px: float = 12.0, max_iter: int = 3, min_points: int = 4, **kw) -> RigFit:
    """:func:`ptz.fit_rig` with explicit outlier handling. After each fit a frame whose *median* point residual is
    above ``frame_px`` is inconsistent with a single centre of projection (another camera, or a wrong labelling of
    the whole frame) and is excluded; in the others, points above ``point_px`` are dropped (frames left with fewer
    than ``min_points`` are excluded). Everything removed is returned."""
    use = list(frames)
    excluded: list[dict] = []
    dropped: list[dict] = []
    init = None
    rig, cams = ptz.fit_rig(use, C0=C0, sigma_C=sigma_C, **kw)
    for _ in range(max_iter):
        new, new_init, changed = [], [], False
        for fr, c in zip(use, cams, strict=True):
            r = _res_px(fr, c)
            if float(np.median(r)) > frame_px:
                excluded.append({"key": fr.key, "why": "median residual", "median_px": float(np.median(r)),
                                 "rms_px": float(np.sqrt(np.mean(r**2)))})
                changed = True
                continue
            keep = r <= point_px
            if keep.all():
                new.append(fr)
                new_init.append(c)
                continue
            changed = True
            if keep.sum() < min_points:
                excluded.append({"key": fr.key, "why": "too few consistent points", "n": int(keep.sum())})
                continue
            for i in np.flatnonzero(~keep):
                dropped.append({"key": fr.key, "index": int(i), "res_px": float(r[i])})
            new.append(ptz.RigFrame(fr.img[keep], fr.pitch[keep], fr.key, fr.shot))
            new_init.append(c)
        if not changed:
            break
        use, init = new, new_init
        rig, cams = ptz.fit_rig(use, C0=rig.C, sigma_C=sigma_C, init=init, **kw)
    return RigFit(rig, {c.info["key"]: c for c in cams}, use, excluded, dropped)


def bootstrap_rig(frames: list[ptz.RigFrame], n_boot: int, *, C0, seed: int = 0, **kw) -> dict:
    """Spread of the rig under resampling of whole shots (``RigFrame.shot``), fitted with ``fit_rig_robust``."""
    rng = np.random.default_rng(seed)
    shots = sorted({f.shot for f in frames})
    Cs, rolls, k1s = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(shots, len(shots))
        fr = [f for s in pick for f in frames if f.shot == s]
        try:
            rf = fit_rig_robust(fr, C0=C0, **kw)
        except Exception:  # noqa: BLE001 - degenerate resample
            continue
        Cs.append(rf.rig.C)
        rolls.append(np.degrees(rf.rig.roll))
        k1s.append(rf.rig.k1)
    Cs = np.array(Cs)
    return {"n": len(Cs), "C_mean": Cs.mean(0).tolist(), "C_std": Cs.std(0).tolist(),
            "C_p5": np.percentile(Cs, 5, axis=0).tolist(), "C_p95": np.percentile(Cs, 95, axis=0).tolist(),
            "roll_mean_deg": float(np.mean(rolls)), "roll_std_deg": float(np.std(rolls)),
            "k1_mean": float(np.mean(k1s)), "k1_std": float(np.std(k1s))}


def rig_cost_for_pitch(frames_ids: list[tuple[ptz.RigFrame, list[str | None]]], dims: tuple[float, float], *, C0,
                       **kw) -> float:
    """Robust cost of the rig fit when the pitch is ``dims`` (metres): labelled keypoints are re-read from the model
    of that size (free, non-keypoint points keep their stored metres). Used to check the 105 x 68 assumption."""
    kps = PitchModel(*dims).keypoints()
    frs = []
    for fr, ids in frames_ids:
        pit = np.array([kps[i] if i in kps else fr.pitch[j] for j, i in enumerate(ids)], float)
        frs.append(ptz.RigFrame(fr.img, pit, fr.key, fr.shot))
    rf = fit_rig_robust(frs, C0=C0, pitch=dims, **kw)
    return float(rf.rig.info["cost"])


# --------------------------------------------------------------------------
# points: consensus over possibly wrong labels
# --------------------------------------------------------------------------


def ransac_points(img: np.ndarray, pitch_m: np.ndarray, rig: ptz.Rig, *, px_thr: float = POINT_PX,
                  n_starts: int = 1, max_pairs: int = 21) -> tuple[ptz.Camera, np.ndarray] | None:
    """Consensus camera of labelled points that may contain wrong ones. Every pair of points gives a PTZ solve
    (the rig's C prior makes 2 points enough); the camera with most inliers (reprojection < ``px_thr`` canonical px),
    ties broken by the smaller inlier rms, is refit on its inliers. -> (camera, inlier mask) or ``None`` (< 2
    points). With two labels the answer is trivially "both inliers": callers must demand more evidence then."""
    img = np.asarray(img, float).reshape(-1, 2)
    pitch_m = np.asarray(pitch_m, float).reshape(-1, 2)
    n = len(img)
    if n < 2:
        return None
    obs = ptz.undistort_px(img * ptz.CANON, rig.k1)

    def inliers(cam):
        uv, z = ptz.project_ground(pitch_m, cam.C, cam.R, cam.f)
        e = np.linalg.norm(uv - obs, axis=1)
        return (e < px_thr) & (z > 0), e

    best = None
    pairs = [p for p in itertools.combinations(range(n), 2) if np.linalg.norm(pitch_m[p[0]] - pitch_m[p[1]]) >= 1.0]
    if len(pairs) > max_pairs:  # a random subset keeps the cost bounded; a good pair is likely among them
        pairs = [pairs[k] for k in np.random.default_rng(0).choice(len(pairs), max_pairs, replace=False)]
    for i, j in pairs:
        cam = ptz.solve_points(img[[i, j]], pitch_m[[i, j]], rig, n_starts=n_starts)
        if cam is None:
            continue
        inl, e = inliers(cam)
        key = (int(inl.sum()), -float(np.sqrt(np.mean(e[inl] ** 2))) if inl.any() else -1e9)
        if best is None or key > best[0]:
            best = (key, inl)
    if best is None:
        return None
    inl = best[1]
    for _ in range(2):
        cam = ptz.solve_points(img[inl], pitch_m[inl], rig, n_starts=5)
        if cam is None:
            return None
        new, _ = inliers(cam)
        if new.sum() < 2 or (new == inl).all():
            break
        inl = new
    cam = ptz.solve_points(img[inl], pitch_m[inl], rig, n_starts=5)
    return (cam, inl) if cam is not None else None


# --------------------------------------------------------------------------
# frame evidence / main-camera check
# --------------------------------------------------------------------------


def frame_masks(bgr: np.ndarray, adaptive: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(white-line mask, grass mask) of a frame at the pre-labelling working size. ``adaptive``: the line mask's
    saturation limit is relative to the pitch's own colour (see ``candidates.LINE_SAT_REL``); the fixed limit of the
    pre-labelling pipeline loses most lines on the saturated Vitality turf."""
    from footy_track.calibration.prelabel.candidates import (
        LINE_SAT_REL,
        grass_mask,
        line_mask,
        resize_to_work,
    )

    work, _ = resize_to_work(bgr)
    grass = grass_mask(work)
    return line_mask(work, grass, sat_rel=LINE_SAT_REL if adaptive else None), grass


def main_camera_p(bgr: np.ndarray) -> float:
    """Frame-level probability that this is the main wide camera (``shots`` classifier, no LLM)."""
    import cv2

    from footy_track.calibration import shots

    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (shots.DEFAULT_WORK_WIDTH, round(h * shots.DEFAULT_WORK_WIDTH / w)),
                       interpolation=cv2.INTER_AREA)
    return float(shots.frame_p_main(shots.frame_features(small), shots.default_model()))


# --------------------------------------------------------------------------
# per-frame decision
# --------------------------------------------------------------------------


@dataclass
class PointsCand:
    cam: ptz.Camera
    n_inliers: int
    n_labels: int
    inlier: np.ndarray
    support: float  # explained_f1 of the camera on the line evidence (after the line refinement)
    refined: bool = False
    rms_px: float = float("nan")  # rms reprojection (canonical px) of the consensus points under ``cam``
    spread_m: float = 0.0  # std (m) of the consensus points perpendicular to their best line: 0 = collinear


@dataclass
class Outcome:
    cam: ptz.Camera | None
    mode: str  # "lines-only" | "<k> LLM pts + lines" | "abstain"
    reason: str | None  # abstain reason
    support: float | None
    margin: float | None = None
    n_inliers: int = 0
    n_labels: int = 0
    inlier: np.ndarray | None = None
    p_main: float | None = None
    notes: list[str] = field(default_factory=list)
    secs: float = 0.0
    lines_info: dict = field(default_factory=dict)
    hyp_cams: list[ptz.Camera] = field(default_factory=list)


def pick_camera(pts: PointsCand | None, lines_cam: ptz.Camera | None, lines_support: float | None,
                lines_reason: str | None, hyp_cams: list[ptz.Camera], *, min_support: float = MIN_SUPPORT,
                agree_m: float = AGREE_M) -> tuple[ptz.Camera | None, str, str | None, list[str]]:
    """Decision between the points candidate and the lines-only answer -> (camera, mode, abstain reason, notes).

    * every points answer needs line support >= ``min_support`` (the same gate as lines-only); below it the points
      are dropped (a lines-only answer or abstention follows), however many labels agree.
    * >= ``STRONG_N`` consensus points within ``STRONG_RMS`` px (3 points: ``STRONG_N3_RMS``), not collinear
      (``MIN_SPREAD_M``) and supported: points (+ lines); a lines-only solve overrides them only when it is better
      supported by more than ``CONFIRM_DELTA``.
    * >= 3 consensus points and the refined camera has line support >= ``min_support``: points (+ lines).
    * exactly 2 points: only when some top lines hypothesis (the lines search's distinct hypotheses, answered or
      not) is the same camera (within ``agree_m``) or the support is high (>= ``min_support`` + ``CONFIRM_DELTA``); two
      labels fit any camera.
    * both answered but different cameras: the one with the larger line support (ties: points).
    * otherwise the lines-only answer, else abstain with the reason."""
    notes: list[str] = []
    pts_ok = False
    strong = (pts is not None and pts.spread_m >= MIN_SPREAD_M
              and ((pts.n_inliers >= STRONG_N and pts.rms_px <= STRONG_RMS)
                   or (pts.n_inliers == 3 and pts.rms_px <= STRONG_N3_RMS))
              and ptz.F_RANGE[0] * 1.5 <= pts.cam.f <= ptz.F_RANGE[1])
    if strong and pts.support < min_support:
        # consistent labels are not enough: LLM-point fits face the same line-support gate as lines-only (human review
        # of ptz_bmth1 rejected all three ungated "N consistent points" proposals; support was 0.24-0.46)
        notes.append(f"{pts.n_inliers} consistent LLM pts rejected: line support {pts.support:.2f} < {min_support:.2f}")
    elif strong:
        pts_ok = True
    elif pts is not None and pts.support >= min_support:
        if pts.n_inliers >= 3:
            pts_ok = True
        else:
            agrees = any(ptz._cam_distance(pts.cam, h) < agree_m for h in hyp_cams[:3])
            pts_ok = agrees or pts.support >= min_support + CONFIRM_DELTA
            if not pts_ok:
                notes.append("2 LLM pts not confirmed by lines")
    if pts_ok and lines_cam is not None and ptz._cam_distance(pts.cam, lines_cam) > agree_m:
        # strong labels still lose to a clearly better line fit: labels can be consistent with each other and
        # wrong (seg085:49: six LLM points on a circle drawn 20% too small), the lines are independent evidence
        if (lines_support or 0.0) > pts.support + (CONFIRM_DELTA if strong else 0.1):
            notes.append(f"{pts.n_inliers} LLM pts disagree with a better-supported lines-only solve")
            return lines_cam, "lines-only", None, notes
        notes.append("lines-only search disagreed")
    if pts_ok:
        return pts.cam, f"{pts.n_inliers} LLM pts + lines", None, notes
    if lines_cam is not None:
        return lines_cam, "lines-only", None, notes
    why = lines_reason or "no hypothesis"
    if pts is not None:
        why = f"{why}; LLM points {pts.n_inliers}/{pts.n_labels} consistent, line support {pts.support:.2f}"
    return None, "abstain", why, notes


def _spread(pitch_m: np.ndarray) -> float:
    """Std (m) of 2-D points perpendicular to their principal line (0 for collinear points, < 3 points: 0)."""
    p = np.asarray(pitch_m, float).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    return float(np.sqrt(max(np.linalg.eigvalsh(np.cov(p.T))[0], 0.0)))


def _inlier_rms(cam: ptz.Camera, img: np.ndarray, pitch_m: np.ndarray, inl: np.ndarray) -> float:
    """rms reprojection error (canonical px) of the ``inl`` labelled points under ``cam``."""
    obs = ptz.undistort_px(np.asarray(img, float)[inl] * ptz.CANON, cam.k1)
    uv, _ = ptz.project_ground(np.asarray(pitch_m, float)[inl], cam.C, cam.R, cam.f)
    return float(np.sqrt(np.mean(np.sum((uv - obs) ** 2, axis=1))))


def solve_frame(lines_mask: np.ndarray, grass: np.ndarray, rig: ptz.Rig, llm_img: np.ndarray | None = None,
                llm_pitch: np.ndarray | None = None, *, p_main: float | None = None, pitch: PitchModel | None = None,
                margin: float = MARGIN, min_support: float = MIN_SUPPORT, use_llm_points: bool = True) -> Outcome:
    """Solve one frame from its white-line / grass masks and (optionally) LLM point labels, ignored when ``use_llm_points`` is False (image: observed
    normalised coords, pitch: metres): robust point solve + line refinement, the lines-only search as a second
    opinion / fallback, then :func:`pick_camera`; abstains on low support, near-ties and non-main cameras."""
    t0 = time.time()
    pitch = pitch or PitchModel(*rig.pitch)
    ev = ptz.LineEvidence.from_masks(lines_mask, grass)
    lcam, hyps, linfo = ptz.search_lines(ev, rig, pitch, margin=margin, min_f1=min_support)
    hyp_cams = [h.cam for h in hyps]
    pts = None
    n_lab = 0 if llm_img is None else len(llm_img)
    if not use_llm_points:
        n_lab = 0
    if n_lab >= 2:
        r = ransac_points(llm_img, llm_pitch, rig)
        if r is not None:
            cam_p, inl = r
            s0 = float(ptz.score_cameras([cam_p], ev, pitch)[0])
            r0 = _inlier_rms(cam_p, llm_img, llm_pitch, inl)
            cam_r = ptz._refine_lines(cam_p, ev, rig, pitch)
            s1 = float(ptz.score_cameras([cam_r], ev, pitch)[0])
            r1 = _inlier_rms(cam_r, llm_img, llm_pitch, inl)
            # the lines may sharpen the points' camera, never wander off them
            refined = s1 < s0 and ptz._cam_distance(cam_p, cam_r) < 2.0 and r1 <= max(1.5 * r0, 4.0)
            cam, rms = (cam_r, r1) if refined else (cam_p, r0)
            sup = float(ptz.line_support([cam], ev, pitch)[0])
            pts = PointsCand(cam, int(inl.sum()), n_lab, inl, sup, refined, rms, _spread(llm_pitch[inl]))
    sup_l = linfo.get("f1")
    cam, mode, reason, notes = pick_camera(pts, lcam, sup_l, linfo.get("reason"), hyp_cams, min_support=min_support)
    out = Outcome(cam, mode, reason, None, linfo.get("margin"), 0, n_lab, None, p_main, notes,
                  lines_info=linfo, hyp_cams=hyp_cams)
    if cam is not None:
        out.support = pts.support if mode.endswith("+ lines") and pts is not None else sup_l
        if pts is not None and mode.endswith("+ lines"):
            out.n_inliers, out.inlier = pts.n_inliers, pts.inlier
        if p_main is not None and p_main < MAIN_P and (out.support or 0.0) < MAIN_SUPPORT:
            out.cam, out.mode, out.reason = None, "abstain", f"not the main camera (p_main {p_main:.2f}, support {out.support:.2f})"
    else:
        out.support = pts.support if pts is not None else sup_l
    out.secs = time.time() - t0
    return out


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


def projected_keypoints(cam: ptz.Camera, pitch: PitchModel, *, exclude_ids=(), n_max: int = 10,
                        margin: float = 0.03) -> list[dict]:
    """Visible pitch keypoints projected through ``cam`` as correspondences (spread out: greedy farthest point in
    the image, up to ``n_max``). Marked ``synthetic`` (not a real label) so the editor / accept path know."""
    ids, xy = pitch.keypoint_array()
    uv, ok = cam.pitch_to_image(xy)
    ok = ok & np.all((uv > margin) & (uv < 1 - margin), axis=1)
    cand = [i for i in np.flatnonzero(ok) if ids[i] not in set(exclude_ids)]
    if not cand:
        return []
    pts = uv[cand] * np.array([ptz.ASPECT, 1.0])
    chosen = [int(np.argmax(np.linalg.norm(pts - pts.mean(0), axis=1)))]
    while len(chosen) < min(n_max, len(cand)):
        d = np.min(np.linalg.norm(pts[:, None] - pts[chosen][None], axis=2), axis=1)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    return [{"image": [float(uv[cand[k]][0]), float(uv[cand[k]][1])], "pitch_id": ids[cand[k]],
             "pitch_xy": [float(xy[cand[k]][0]), float(xy[cand[k]][1])], "confidence": 0.3, "synthetic": True}
            for k in chosen]


def _view(notes: str, main: bool | None) -> SimpleNamespace:
    return SimpleNamespace(view={"notes": notes, "main_camera_wide": main, "reverse_angle_suspected": False,
                                 "end": None}, unusable=False, unusable_reason=None)


def cam_params(cam: ptz.Camera) -> dict:
    p = cam.params()
    return {k: (round(v, 4) if isinstance(v, float) else [round(c, 3) for c in v] if isinstance(v, list) else v)
            for k, v in p.items()}


def proposal_record(idx: int, image_size: tuple[int, int], pitch: PitchModel, out: Outcome, llm_corr: list[dict],
                    extra: dict | None = None) -> dict:
    """A ``proposed`` record (the schema of ``prelabel.pipeline.proposal_record``): H, distortion, the consensus LLM
    points as real correspondences plus projected key points the editor can adjust, score = line support."""
    cam = out.cam
    rec = pl._base_record(idx, image_size, pitch)
    real = []
    if out.inlier is not None:
        real = [dict(c) for c, o in zip(llm_corr, out.inlier, strict=True) if o]
    synth = projected_keypoints(cam, pitch, exclude_ids={c.get("pitch_id") for c in real})
    corr = real + synth
    rec["correspondences"] = corr
    rec["H"] = cam.homography().tolist()
    rec["inliers"] = [True] * len(corr)
    rec["method"] = "ptz-rig"
    rec.update(status="proposed", score=float(np.clip(out.support or 0.0, 0.0, 1.0)), failure=None,
               view=pl._view_record(_view(_note(out), True)))
    if cam.k1:
        rec["distortion"] = geo.make_distortion(cam.k1, aspect=ptz.ASPECT)
    rec["ptz"] = {"mode": out.mode, "support": out.support, "margin": out.margin, "n_inliers": out.n_inliers,
                  "n_labels": out.n_labels, "p_main": out.p_main, "secs": round(out.secs, 1), **cam_params(cam),
                  **(extra or {})}
    return rec


def failed_record(idx: int, image_size: tuple[int, int], pitch: PitchModel, out: Outcome, llm_corr: list[dict],
                  extra: dict | None = None) -> dict:
    reason = f"PTZ rig abstained: {out.reason}"
    main = False if (out.reason or "").startswith("not the main") else None
    rec = pl.failed_record(idx, image_size, pitch, reason, _view(_note(out), main))
    rec["correspondences"] = [dict(c) for c in llm_corr]
    rec["inliers"] = [True] * len(llm_corr)
    rec["method"] = "ptz-rig"
    rec["ptz"] = {"mode": out.mode, "support": out.support, "margin": out.margin, "n_labels": out.n_labels,
                  "p_main": out.p_main, "reason": out.reason, "secs": round(out.secs, 1), **(extra or {})}
    return rec


def _note(out: Outcome) -> str:
    if out.cam is None:
        s = f"PTZ rig: abstained ({out.reason})"
    else:
        s = f"PTZ rig: {out.mode}"
        if out.mode != "lines-only" and out.n_labels:
            s += f" (of {out.n_labels} labelled)"
    if out.notes:
        s += "; " + "; ".join(out.notes)
    return s


def evenly_spaced(n_frames: int, k: int, avoid=(), min_gap: int = 8) -> list[int]:
    """Up to ``k`` frames at the centres of ``k`` equal segments of the clip, each moved (within its segment, to the
    nearest free frame) so it is at least ``min_gap`` frames from every frame in ``avoid`` and from each other."""
    if n_frames <= 0 or k <= 0:
        return []
    taken = sorted(set(avoid))
    out: list[int] = []
    for s in range(k):
        lo, hi = s * n_frames // k, (s + 1) * n_frames // k - 1
        mid = (lo + hi) // 2
        for d in range(0, max(1, hi - lo) + 1):
            hit = None
            for c in (mid + d, mid - d):
                if lo <= c <= hi and all(abs(c - t) >= min_gap for t in taken + out):
                    hit = c
                    break
            if hit is not None:
                out.append(hit)
                break
    return sorted(out)


def write_run(gt_dir: Path, run_id: str, source: str, params: dict, per_clip: dict[str, dict[int, dict]]) -> None:
    """Add (never replace others) ``run_id`` to each clip's proposals sidecar; one atomic update per clip."""
    from footy_track.calibration.prelabel.proposals import update_proposals

    created = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for clip, frames in per_clip.items():
        def _store(doc, frames=frames):
            run = doc["runs"].setdefault(run_id, {"source": source, "created": created, "params": params,
                                                  "frames": {}})
            run["params"] = params
            for idx, rec in frames.items():
                run["frames"][str(idx)] = rec

        update_proposals(Path(gt_dir), clip, _store)
