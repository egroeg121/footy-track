"""Line refinement (calibration/refine.py), the noisy mock and the noise sweep."""

from __future__ import annotations

import numpy as np
import pytest

from footy_track.calibration import geometry as geo
from footy_track.calibration.prelabel import identifier as idf
from footy_track.calibration.prelabel import sweep as sw
from footy_track.calibration.prelabel.fit import Label, _mixed_dlt, fit_and_score
from footy_track.calibration.refine import LineLabel, refine_homography

from .test_geometry import H, W, _camera_h
from .test_prelabel import KP, PITCH, _gt_dir, _inp, _true_uv, det, frame  # noqa: F401


def _kp_err_px(Hx):
    errs = []
    for k in KP:
        uv = _true_uv(k)
        if not (0.02 < uv[0] < 0.98 and 0.02 < uv[1] < 0.98):
            continue
        back, w = geo.apply_homography(np.linalg.inv(Hx), np.array([PITCH.keypoints()[k]]))
        errs.append(np.hypot(*((back[0] - uv) * [W, H])))
    return float(np.mean(errs))


def _perturbed(dx=0.025, dy=-0.02, persp=2e-3):
    P = np.array([[1, 0, dx], [0, 1, dy], [persp, 0, 1.0]])
    return geo.orient_homography(_camera_h() @ P, np.array([[0.5, 0.5]]))


def test_refinement_recovers_a_perturbed_homography(det):
    H0 = _perturbed()
    before = _kp_err_px(H0)
    assert before > 15  # a ~30 px miss: well inside the capture range, far outside "good"
    r = refine_homography(H0, PITCH, det.lines_mask, det.grass)
    assert r.applied and r.reason is None
    assert _kp_err_px(r.H) < 3.0 < before
    assert r.cost_after < r.cost_before
    assert r.stage_costs and r.n_samples >= 30


def test_refinement_with_a_label_prior_does_not_undo_it(det):
    H0 = _perturbed()
    prior = []
    for k in KP:
        uv = _true_uv(k)
        if 0.05 < uv[0] < 0.95 and 0.05 < uv[1] < 0.95:
            prior.append((k, tuple(uv), 1.0))
    r = refine_homography(H0, PITCH, det.lines_mask, det.grass, prior_points=prior)
    assert r.applied and _kp_err_px(r.H) < 3.0


def test_refinement_is_rejected_without_lines(det):
    empty = np.zeros_like(det.lines_mask)
    H0 = _perturbed()
    r = refine_homography(H0, PITCH, empty, det.grass)
    assert not r.applied and np.array_equal(r.H, H0)  # input returned untouched
    assert r.reason


def test_refinement_rejects_a_degenerate_start(det):
    junk = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1e-12]])
    r = refine_homography(junk, PITCH, det.lines_mask, det.grass)
    assert not r.applied and np.array_equal(r.H, junk)


def test_good_homography_is_left_essentially_unchanged(det):
    r = refine_homography(_camera_h(), PITCH, det.lines_mask, det.grass)
    assert _kp_err_px(r.H) < 1.5  # refinement of an already-correct H must not move it away


def test_fit_keeps_pre_and_post_refine_results(det):
    sig = 12 / 1920
    rng = np.random.default_rng(1)
    labels = [Label(k, tuple(_true_uv(k) + rng.normal(0, sig, 2)), 1.0) for k in KP
              if 0.02 < _true_uv(k)[0] < 0.98 and 0.02 < _true_uv(k)[1] < 0.98]
    fit = fit_and_score(labels, PITCH, det, (W, H))
    assert fit.ok and fit.H_coarse is not None and fit.refine is not None
    assert fit.refine.applied
    assert _kp_err_px(fit.H) <= _kp_err_px(fit.H_coarse) + 0.5
    assert fit.score >= fit.score_coarse - 1e-9 and fit.dt_residual is not None and fit.dt_residual < 2.0
    # refined labels are re-expressed through the refined H; the as-labelled position is kept
    moved = [lb for lb in fit.labels if lb.uv_label is not None]
    assert moved and all(lb.uv != lb.uv_label for lb in moved)
    off = fit_and_score(labels, PITCH, det, (W, H), refine=False)
    assert off.refine is None and np.allclose(off.H, fit.H_coarse)


def test_lines_and_points_dlt_solves_with_only_three_points():
    names = ["R_PEN_BOX_FRONT", "HALFWAY_LINE", "TOUCHLINE_TOP", "GOAL_LINE_R", "R_PEN_BOX_TOP"]
    segs = PITCH.segments()
    lines = []
    for n in names:
        run = max(geo.project_polyline_to_image(_camera_h(), segs[n], 0.25), key=len, default=None)
        if run is None or len(run) < 6:
            continue
        lines.append(LineLabel(n, [tuple(run[len(run) // 4]), tuple(run[3 * len(run) // 4])]))
    pts = [Label(k, tuple(_true_uv(k)), 1.0) for k in ("CENTRE_SPOT", "HALFWAY_TOP", "CENTRE_CIRCLE_BOTTOM")]
    assert len(lines) >= 3
    Hm = _mixed_dlt(pts, lines, PITCH)
    assert Hm is not None and _kp_err_px(Hm) < 2.0
    assert _mixed_dlt(pts, lines[:1], PITCH) is None  # 6 + 2 rows < 8 unknown-determining rows


def test_fit_with_three_points_and_lines_pads_to_four_correspondences(det):
    segs = PITCH.segments()
    lines = []
    for n in ("R_PEN_BOX_FRONT", "HALFWAY_LINE", "TOUCHLINE_TOP", "GOAL_LINE_R", "R_PEN_BOX_TOP"):
        run = max(geo.project_polyline_to_image(_camera_h(), segs[n], 0.25), key=len, default=None)
        if run is not None and len(run) >= 6:
            lines.append(LineLabel(n, [tuple(run[len(run) // 4]), tuple(run[3 * len(run) // 4])]))
    pts = [Label(k, tuple(_true_uv(k)), 1.0) for k in ("CENTRE_SPOT", "HALFWAY_TOP", "CENTRE_CIRCLE_BOTTOM")]
    fit = fit_and_score(pts, PITCH, det, (W, H), line_labels=lines)
    assert fit.ok and _kp_err_px(fit.H) < 3.0
    assert len(fit.labels) >= 4 and sum(fit.inliers) >= 4  # accepting needs >= 4 points to re-fit H


# ---- the noisy mock ----------------------------------------------------------


def test_mock_noise_is_gaussian_seeded_and_scaled_to_1920(tmp_path, det):
    d = _gt_dir(tmp_path, det)
    clean = {lb.keypoint_id: np.array(lb.uv) for lb in idf.MockIdentifier(d).identify(_inp(det)).labels}
    a = idf.MockIdentifier(d, sigma_px=20, seed=3).identify(_inp(det)).labels
    b = idf.MockIdentifier(d, sigma_px=20, seed=3).identify(_inp(det)).labels
    c = idf.MockIdentifier(d, sigma_px=20, seed=4).identify(_inp(det)).labels
    assert [lb.uv for lb in a] == [lb.uv for lb in b] and [lb.uv for lb in a] != [lb.uv for lb in c]
    # W x H test frames: sigma is in px of a 1920-wide frame, so 20 px -> 20 * W/1920 px here
    errs = np.concatenate([(np.array(lb.uv) - clean[lb.keypoint_id]) * [W, H] for lb in a])
    assert 0.3 * 20 * W / 1920 < errs.std() < 2.5 * 20 * W / 1920


def test_mock_drop_mislabel_and_lines(tmp_path, det):
    d = _gt_dir(tmp_path, det)
    assert idf.MockIdentifier(d, drop_prob=1.0).identify(_inp(det)).labels == []
    clean_ids = [lb.keypoint_id for lb in idf.MockIdentifier(d).identify(_inp(det)).labels]
    ids = [lb.keypoint_id for lb in idf.MockIdentifier(d, mislabel_prob=1.0, seed=1).identify(_inp(det)).labels]
    assert len(ids) == len(clean_ids) and ids != clean_ids
    res = idf.MockIdentifier(d, lines=True, sigma_px=0).identify(_inp(det))
    assert res.line_labels and all(len(ll.points) == 2 and ll.line_id in PITCH.segments() for ll in res.line_labels)
    # noiseless line points lie on their projected pitch line (< 1 px)
    ll = res.line_labels[0]
    seg = PITCH.segments()[ll.line_id]
    l3 = np.cross([*seg[0], 1.0], [*seg[1], 1.0])
    for u, v in ll.points:
        xy, _ = geo.apply_homography(_camera_h(), np.array([[u, v]]))
        assert abs(l3 @ np.array([*xy[0], 1.0])) / np.hypot(l3[0], l3[1]) < 0.05  # metres


# ---- the noise sweep -----------------------------------------------------------


def test_noise_sweep_reports_pre_and_post_per_sigma(tmp_path, det, frame, monkeypatch):  # noqa: F811
    gt = _gt_dir(tmp_path, det)
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "arsenal_mancity_20250925_seg010.mp4").touch()
    monkeypatch.setattr(sw, "read_frame", lambda v, i: frame)
    res = sw.noise_sweep(gt, clips, "arsenal_mancity_seg010", [7], sigmas=[0, 25], n_seeds=3)
    assert [s["sigma"] for s in res["summary"]] == [0.0, 25.0] and len(res["rows"]) == 6
    s0 = res["summary"][0]
    assert s0["n"] == 3 and s0["pre_le5"] == 1.0 and s0["post_le5"] == 1.0 and s0["post_le10"] == 1.0
    s25 = res["summary"][1]
    assert s25["post_le10"] >= s25["pre_le10"]  # refinement does not lose frames on a clean synthetic view
    txt = sw.format_sweep(res)
    assert "sigma" in txt and "<=5px pre" in txt and "<=10px pre" in txt
