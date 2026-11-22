"""PTZ V3: the constrained inter-frame model (rotation + zoom about a fixed centre) on synthetic data."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from footy_track.calibration import ptz
from footy_track.calibration import ptz_temporal as pt
from footy_track.calibration.pitch_model import PitchModel

C_TRUE = np.array([0.5, -63.0, 14.5])


def _cam(pan_deg, tilt_deg, f, roll_deg=0.3, k1=0.0):
    return ptz.Camera(np.deg2rad(pan_deg), np.deg2rad(tilt_deg), float(np.log(f)), np.deg2rad(roll_deg), C_TRUE, k1)


def _ground_points(cam, n=300, seed=0):
    """Random pitch points visible in ``cam`` -> pitch (n,2), canonical px (n,2)."""
    rng = np.random.default_rng(seed)
    xy = np.column_stack([rng.uniform(-45, 45, 4 * n), rng.uniform(-30, 30, 4 * n)])
    uv, z = ptz.project_ground(xy, cam.C, cam.R, cam.f)
    ok = (z > 1) & (uv[:, 0] > 60) & (uv[:, 0] < 1860) & (uv[:, 1] > 60) & (uv[:, 1] < 1020)
    return xy[ok][:n], uv[ok][:n]


def _visible_keypoints(cam, margin=0.03):
    ids, xy = PitchModel().keypoint_array()
    uv, ok = cam.pitch_to_image(xy)
    ok &= np.all((uv > margin) & (uv < 1 - margin), axis=1)
    return [i for i, o in zip(ids, ok, strict=True) if o]


def _relative(c1, c2):
    """Ground truth (rotation vector camera1 -> camera2, ln f2/f1) of two cameras sharing C."""
    Rrel = c2.R @ c1.R.T
    w = cv2.Rodrigues(Rrel)[0].ravel()
    return w, c2.logf - c1.logf


def test_rodrigues_matches_opencv():
    w = np.array([0.03, -0.02, 0.05])
    assert np.allclose(pt.rodrigues(w), cv2.Rodrigues(w)[0], atol=1e-12)
    assert pt.rodrigues(np.zeros(3)).shape == (3, 3)


def test_rotzoom_map_identity_and_pure_zoom():
    u = np.array([[100.0, -50.0], [-300.0, 200.0]])
    assert np.allclose(pt.rotzoom_map(u, 3000.0, np.zeros(3), 0.0), u)
    assert np.allclose(pt.rotzoom_map(u, 3000.0, np.zeros(3), np.log(1.1)), 1.1 * u)


@pytest.mark.parametrize("d_pan,d_tilt,zoom,d_roll", [(0.6, 0.0, 1.0, 0.0), (-0.4, 0.3, 1.03, 0.0), (1.2, -0.8, 0.97, 0.1)])
def test_ransac_recovers_camera_deltas_with_outliers(d_pan, d_tilt, zoom, d_roll):
    """Render ground points under two cameras (same C), add noise and 35% gross outliers (moving players): the
    constrained RANSAC must recover the relative rotation and zoom."""
    c1 = _cam(88.0, 14.0, 3200.0)
    c2 = _cam(88.0 + d_pan, 14.0 + d_tilt, 3200.0 * zoom, roll_deg=0.3 + d_roll)
    xy, u1 = _ground_points(c1, 400)
    u2, z2 = ptz.project_ground(xy, c2.C, c2.R, c2.f)
    keep = z2 > 1
    u1, u2 = u1[keep], u2[keep]
    rng = np.random.default_rng(1)
    u2 = u2 + rng.normal(0, 0.4, u2.shape)  # canonical px
    bad = rng.random(len(u2)) < 0.35
    u2[bad] += rng.normal(0, 40.0, (bad.sum(), 2))
    o1, o2 = u1 - ptz.CENTRE, u2 - ptz.CENTRE
    m = pt.rotzoom_ransac(o1, o2, c1.f, thr=2.0)
    assert m.ok
    assert (m.inl & ~bad).sum() > 0.9 * (~bad).sum()
    assert (m.inl & bad).sum() < 0.05 * bad.sum() + 2
    w_true, s_true = _relative(c1, c2)
    assert np.allclose(m.w, w_true, atol=2e-4)  # 0.01 deg, noise-limited (0.4 px per point)
    assert abs(m.s - s_true) < 1e-3
    assert m.rms < 1.0


def test_wrong_nominal_focal_scales_the_estimated_angle():
    """Why the joint solve re-fits with the solved focal length: the rotation fitted under a nominal f1 that is
    wrong by 30% is off by about that factor (the flow is f * dtheta)."""
    c1, c2 = _cam(88.0, 14.0, 3200.0), _cam(88.8, 14.0, 3200.0)
    xy, u1 = _ground_points(c1, 300)
    u2, _ = ptz.project_ground(xy, c2.C, c2.R, c2.f)
    w_true, _ = _relative(c1, c2)
    w, _ = pt.rotzoom_fit(u1 - ptz.CENTRE, u2 - ptz.CENTRE, 3200.0 * 1.3)
    assert 0.6 < np.linalg.norm(w) / np.linalg.norm(w_true) < 0.95
    w, _ = pt.rotzoom_fit(u1 - ptz.CENTRE, u2 - ptz.CENTRE, 3200.0)
    assert np.allclose(w, w_true, atol=1e-5)


def test_tracked_texture_pair_recovers_motion():
    """Image-level round trip: a textured plane ('stands') warped by the exact H = K2 R K1^-1 of a pan + zoom, features
    tracked by LK, constrained RANSAC with a moving (inconsistent) block of 'players'."""
    rng = np.random.default_rng(3)
    Wc, Hc = 960, 540
    s = Wc / ptz.CANON_W
    tex = cv2.GaussianBlur((rng.random((Hc, Wc)) * 255).astype(np.float32), (0, 0), 1.6)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    f1, f2 = 3000.0, 3000.0 * 1.04
    w_true = np.array([0.004, 0.012, 0.0015])
    K = lambda f: np.array([[f * s, 0, Wc / 2], [0, f * s, Hc / 2], [0, 0, 1.0]])  # noqa: E731
    Hm = K(f2) @ pt.rodrigues(w_true) @ np.linalg.inv(K(f1))
    img2 = cv2.warpPerspective(tex, Hm, (Wc, Hc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img1 = tex.copy()
    img2[200:300, 400:520] = rng.integers(0, 255, (100, 120), dtype=np.uint8)  # a block that does not move rigidly
    pts = pt.detect_features(img1, None, 600)
    a, b = pt.track(img1, img2, pts)
    assert len(a) > 150
    u1, u2 = pt.work_to_canon(a, 0.0), pt.work_to_canon(b, 0.0)
    m = pt.rotzoom_ransac(u1, u2, f1)
    assert m.ok and m.n_in > 100
    assert np.allclose(m.w, w_true, atol=4e-4)
    assert abs(m.s - np.log(1.04)) < 1e-3


def test_segment_shots_histogram_cuts():
    hd = np.zeros(100)
    hd[40] = 0.9  # hard cut
    hd[70] = 0.35  # soft jump with a failed pair -> cut; without one -> not a cut
    pair_ok = np.ones(100, bool)
    assert pt.segment_shots(hd, pair_ok) == [(0, 39), (40, 99)]
    pair_ok[70] = False
    assert pt.segment_shots(hd, pair_ok) == [(0, 39), (40, 69), (70, 99)]


# --------------------------------------------------------------------------
# joint solve on a synthetic pan / zoom sweep
# --------------------------------------------------------------------------


def _sweep(n=9):
    """Cameras panning 3 deg and zooming 25% over n frames."""
    return [_cam(86.0 + 0.4 * k, 14.0 + 0.1 * k, 2800.0 * (1 + 0.03 * k), roll_deg=0.2, k1=0.014) for k in range(n)]


def _evidence(cam):
    lines, grass = ptz.render_synthetic(cam, PitchModel(), width=1568, thickness=3)
    return ptz.LineEvidence.from_masks(lines, grass)


def _pairs(cams, rng, noise=0.3, n=120, gap=1):
    """Synthetic motion pairs: rays of random directions inside the field of view (depth-free), observed (distorted)
    in work px under both cameras, with 25% gross outliers in the second frame."""
    out = []
    for i in range(0, len(cams) - gap):
        c1, c2 = cams[i], cams[i + gap]
        u = np.column_stack([rng.uniform(-800, 800, n), rng.uniform(-420, 420, n)])  # undistorted offsets in frame 1
        d1 = np.column_stack([u / c1.f, np.ones(n)])
        ray = d1 @ c1.R  # world direction (R^T d)
        v = ray @ c2.R.T
        u2 = v[:, :2] / v[:, 2:3] * c2.f
        ok = (v[:, 2] > 0) & (np.abs(u2[:, 0]) < 940) & (np.abs(u2[:, 1]) < 520)
        a_obs = ptz.distort_px(u[ok] + ptz.CENTRE, c1.k1)
        b_obs = ptz.distort_px(u2[ok] + ptz.CENTRE, c2.k1) + rng.normal(0, noise, (ok.sum(), 2))
        out.append(pt.Pair(i, i + gap, a_obs * pt.S, b_obs * pt.S, np.zeros(3), 0.0, 0.3, int(ok.sum()), int(ok.sum())))
    return out


def test_joint_solve_recovers_sweep_from_wrong_initial_states():
    cams = _sweep()
    rng = np.random.default_rng(0)
    rig = ptz.Rig(C_TRUE, cams[0].roll, cams[0].k1, (1.0, 3.0, 1.0))
    sh = pt.Shared.from_rig(rig)
    evs = [_evidence(c) for c in cams]
    truth = np.array([[c.pan, c.tilt, c.logf] for c in cams])
    x0 = truth + rng.normal(0, 1, truth.shape) * [0.01, 0.006, 0.04]  # ~0.6 deg / 0.35 deg / 4% off
    pairs = _pairs(cams, rng)
    shn, poses, info = pt.joint_solve(len(cams), sh, x0, [evs[k] for k in range(0, len(cams), 2)],
                                      np.arange(0, len(cams), 2), pairs, {}, PitchModel())
    cam_est = [pt.cam_from_state(p, shn) for p in poses]
    for ce, ct in zip(cam_est, cams, strict=True):
        gx, gy = np.meshgrid(np.linspace(-25, 25, 5), np.linspace(-15, 15, 5))
        g = np.column_stack([gx.ravel(), gy.ravel()])
        uv, ok = ct.pitch_to_image(g)
        ok &= np.all((uv > 0.03) & (uv < 0.97), axis=1)
        back, _ = ce.image_to_pitch(uv[ok])
        assert np.median(np.linalg.norm(back - g[ok], axis=1)) < 0.6
    assert info["stage2"]["cost"] < info["stage1"]["cost0"]


def test_motion_alone_carries_an_anchor_across_frames_without_lines():
    """Only frame 0 has line evidence + an anchor; the other frames are pinned by the motion pairs alone."""
    cams = _sweep()
    rng = np.random.default_rng(1)
    rig = ptz.Rig(C_TRUE, cams[0].roll, cams[0].k1, (1.0, 3.0, 1.0))
    sh = pt.Shared.from_rig(rig)
    truth = np.array([[c.pan, c.tilt, c.logf] for c in cams])
    pairs = _pairs(cams, rng)
    pb = {(p.i, p.j): p for p in pairs}
    anchor = truth[0] + [0.002, 0.001, 0.01]
    init = np.full(truth.shape, np.nan)
    for t, v in pt.chain(anchor, pb, 0, len(cams) - 1, sh).items():
        init[t] = v
    assert np.isfinite(init).all()
    # chain alone: the anchor offset (0.1 deg) is carried, no drift beyond it
    assert np.degrees(np.abs(init[:, 0] - truth[:, 0]).max()) < 0.35
    shn, poses, _ = pt.joint_solve(len(cams), sh, init, [_evidence(cams[0])], np.array([0]), pairs,
                                   {0: (anchor, np.array([0.004, 0.004, 0.04]))}, PitchModel())
    assert np.degrees(np.abs(poses[:, 0] - truth[:, 0]).max()) < 0.2
    assert np.abs(poses[:, 2] - truth[:, 2]).max() < 0.03


def test_shared_centre_is_recovered_from_many_views():
    """C and k1 are estimated from line evidence of independent frames (no motion): start 2 m / 5 m off."""
    rng = np.random.default_rng(2)
    cams = [_cam(70.0 + 6 * k, 12.0 + (k % 3) * 2.5, 2400.0 + 500 * (k % 4), roll_deg=0.2, k1=0.014) for k in range(10)]
    cams = [c for c in cams if len(_visible_keypoints(c)) >= 4]
    truth = np.array([[c.pan, c.tilt, c.logf] for c in cams])
    units = [pt.Unit(k, [0], (truth[k] + rng.normal(0, 1, 3) * [0.004, 0.003, 0.02])[None], {0: _evidence(c)})
             for k, c in enumerate(cams)]
    rig0 = ptz.Rig(C_TRUE + [1.0, 5.0, -2.0], 0.2 * np.pi / 180, 0.014, (5.0, 10.0, 3.0))
    shn, _, info = pt.fit_shared(units, rig0, PitchModel(), free=("Cx", "Cy", "Cz", "k1"))
    # the y / z ridge is weakly determined: demand improvement on the horizontal-distance-like combination
    err0 = np.linalg.norm(rig0.C - C_TRUE)
    assert np.linalg.norm(shn.C - C_TRUE) < 0.6 * err0


def test_finder_style_point_anchors_pin_a_frame_with_no_line_evidence():
    """Point correspondences (the finders' output) of one frame enter the joint solve as reprojection residuals."""
    cams = _sweep(5)
    rng = np.random.default_rng(4)
    rig = ptz.Rig(C_TRUE, cams[0].roll, cams[0].k1, (1.0, 3.0, 1.0))
    sh = pt.Shared.from_rig(rig)
    truth = np.array([[c.pan, c.tilt, c.logf] for c in cams])
    ids, xy = PitchModel().keypoint_array()
    uv, ok = cams[2].pitch_to_image(xy)
    ok &= np.all((uv > 0.05) & (uv < 0.95), axis=1)
    assert ok.sum() >= 4
    obs = ptz.undistort_px(uv[ok] * ptz.CANON, 0.0) + rng.normal(0, 1.0, (ok.sum(), 2))  # observed px (distorted)
    x0 = truth + [0.008, -0.004, 0.03]
    points = [(2, xy[ok], obs, 2.0)]
    shn, poses, _ = pt.joint_solve(5, sh, x0, [], np.zeros(0, int), _pairs(cams, rng), {}, PitchModel(), points=points)
    est = pt.cam_from_state(poses[2], shn)
    gx, gy = np.meshgrid(np.linspace(-20, 20, 5), np.linspace(-12, 12, 5))
    g = np.column_stack([gx.ravel(), gy.ravel()])
    u, okg = cams[2].pitch_to_image(g)
    okg &= np.all((u > 0.03) & (u < 0.97), axis=1)
    back, _ = est.image_to_pitch(u[okg])
    assert np.median(np.linalg.norm(back - g[okg], axis=1)) < 1.0
    assert np.degrees(abs(poses[0, 0] - truth[0, 0])) < 0.15  # motion carries the pinned pose to its neighbours


def _shared_units(rng, k1=0.014):
    cams = [_cam(70.0 + 6 * k, 12.0 + (k % 3) * 2.5, 2400.0 + 500 * (k % 4), roll_deg=0.2, k1=k1) for k in range(10)]
    cams = [c for c in cams if len(_visible_keypoints(c)) >= 4]
    truth = np.array([[c.pan, c.tilt, c.logf] for c in cams])
    return cams, [pt.Unit(k, [0], (truth[k] + rng.normal(0, 1, 3) * [0.004, 0.003, 0.02])[None], {0: _evidence(c)})
                  for k, c in enumerate(cams)]


def test_reseed_states_keeps_the_image_of_the_pitch_at_another_centre():
    cam = _cam(80.0, 15.0, 3000.0, k1=0.014)
    st = np.array([[cam.pan, cam.tilt, cam.logf]])
    src = pt.Shared(C_TRUE, cam.roll, cam.k1)
    dst = pt.Shared(C_TRUE + [1.0, 4.0, -1.5], cam.roll, cam.k1)
    same = pt.reseed_states(st, src, src)
    assert np.allclose(same, st, atol=1e-6)
    new = pt.reseed_states(st, src, dst)
    gx, gy = np.meshgrid(np.linspace(-30, 30, 7), np.linspace(-20, 20, 5))
    g = np.column_stack([gx.ravel(), gy.ravel()])
    u0, ok0 = cam.pitch_to_image(g)
    u1, ok1 = pt.cam_from_state(new[0], dst).pitch_to_image(g)
    ok = ok0 & ok1 & np.all((u0 > 0.05) & (u0 < 0.95), axis=1)
    assert ok.sum() >= 8
    assert np.abs((u1[ok] - u0[ok]) * ptz.CANON).max() < 40.0  # px at 1920 over the visible pitch
    assert not np.allclose(new, st, atol=1e-3)  # the pose did move


def test_seed_offsets_grid_is_centred_and_ordered_by_distance():
    offs = pt.seed_offsets(step=(3.0, 4.0, 3.0), n=(1, 2, 1))
    assert len(offs) == 3 * 5 * 3
    assert np.allclose(offs[0], 0.0)
    d = [np.linalg.norm(o) for o in offs]
    assert d == sorted(d)
    assert np.allclose(np.mean(offs, axis=0), 0.0, atol=1e-9)


def test_multistart_shared_fit_converges_from_a_start_5m_off():
    """A single start from C + (2, 5, -2) m stalls on the y / z ridge; seeds around the prior keep the lowest cost."""
    rng = np.random.default_rng(2)
    cams, units = _shared_units(rng)
    truth_src = pt.Shared(C_TRUE, cams[0].roll, 0.014)
    rig0 = ptz.Rig(C_TRUE + [2.0, 5.0, -2.0], cams[0].roll, 0.014, (5.0, 10.0, 3.0))
    offs = pt.seed_offsets(step=(3.0, 4.0, 3.0), n=(1, 2, 1))
    shn, _, info = pt.fit_shared_multistart(units, rig0, PitchModel(), states_at=truth_src, offsets=offs, top_k=3,
                                            free=("Cx", "Cy", "Cz", "roll"), n_iter=60)
    assert len(info["starts"]) == len(offs)
    finals = [s for s in info["starts"] if "final_cost" in s]
    assert len(finals) == 3
    assert info["starts"][info["best_start"]]["final_cost"] == min(s["final_cost"] for s in finals)
    assert np.linalg.norm(shn.C - C_TRUE) < 0.6  # from 5.7 m off (a single start ends ~2 m off)
    assert np.linalg.norm(rig0.C - C_TRUE) > 5.0


def _chain(n=60, n_in=850, step_deg=0.1, seed=0):
    """Gap-1 pairs of a slow pan: ~``n_in`` inliers, rotation ~``step_deg`` each (a small jitter on both)."""
    rng = np.random.default_rng(seed)
    return [pt.Pair(i, i + 1, np.zeros((20, 2)), np.zeros((20, 2)), np.array([0.0, np.radians(step_deg) * (1 + 0.1 * rng.normal()), 0.0]),
                    0.0, 0.3, int(n_in * (1 + 0.05 * rng.normal())), n_in) for i in range(n)]


def _set(pairs, i, n_in, ang_deg):
    from dataclasses import replace

    pairs[i] = replace(pairs[i], n_in=n_in, w=np.array([0.0, np.radians(ang_deg), 0.0]))


def test_weak_pair_guard_rejects_few_inliers_with_a_big_rotation_jump():
    pairs = _chain()
    _set(pairs, 30, 131, 6.6)  # the seg060 case: 131 inliers vs ~850, a 6.6 deg step among 0.1 deg ones
    out, weak = pt.reject_weak_pairs(pairs)
    assert weak == [30]
    assert [p.i for p in out if not p.ok] == [30]
    assert all(p.ok for p in pairs)  # the input is not mutated


def test_weak_pair_guard_keeps_either_symptom_alone():
    pairs = _chain()
    _set(pairs, 10, 131, 0.1)  # few inliers, but a normal step (blurred, smooth pan)
    _set(pairs, 40, 850, 6.6)  # a big step with the usual inlier count (a genuine fast whip)
    _, weak = pt.reject_weak_pairs(pairs)
    assert weak == []


def test_weak_pair_guard_ignores_noise_on_a_static_camera():
    pairs = _chain(step_deg=0.0)
    _set(pairs, 20, 200, 0.1)  # tiny absolute step: below the angle floor
    assert pt.reject_weak_pairs(pairs)[1] == []


def test_weak_pair_is_a_chain_break_in_segments():
    from footy_track.scripts import ptz_temporal as pte

    pairs = _chain(n=60)
    _set(pairs, 30, 131, 6.6)
    shot = [{"start": 0, "end": 60, "p_main": 1.0, "n_ev": 12}]
    assert pte.segments(shot, pairs) == [(0, 60)]  # unguarded: one run across the bad pair
    guarded, _ = pt.reject_weak_pairs(pairs)
    assert pte.segments(shot, guarded) == [(0, 30), (31, 60)]


# --------------------------------------------------------------------------
# level-1 rig estimate of one segment (free C, roll; covariance with the poses marginalised)
# --------------------------------------------------------------------------

_EV_CACHE: dict = {}
PRIOR_OFF = np.array([0.3, 1.0, -0.3])  # a loose prior centred 1.1 m off; farther starts stall on the y / z ridge (multi-start's job)


def _seg_cams(n, pan_span, zoom_span):
    pans = np.linspace(0.0, pan_span, n)
    zooms = np.linspace(0.0, zoom_span, n)
    return [_cam(86.0 - pan_span / 2 + pans[k], 14.0 + 0.8 * np.sin(k / 2), 2800.0 * (1 + zooms[k]), roll_deg=0.2, k1=0.014) for k in range(n)]


def _noisy_evidence(cam, rng, shift_px=2):
    """Rendered lines shifted by a random whole-pixel offset (stand-in for line detection noise); the render is cached."""
    key = (round(float(cam.pan), 6), round(float(cam.tilt), 6), round(float(cam.logf), 6))
    if key not in _EV_CACHE:
        _EV_CACHE[key] = ptz.render_synthetic(cam, PitchModel(), width=1568, thickness=3)
    lines, grass = _EV_CACHE[key]
    dx, dy = rng.integers(-shift_px, shift_px + 1, 2)
    return ptz.LineEvidence.from_masks(np.roll(lines, (dy, dx), (0, 1)), grass)


def _rig_problem(seed, n=8, pan_span=3.0, zoom_span=0.25):
    rng = np.random.default_rng(seed)
    cams = _seg_cams(n, pan_span, zoom_span)
    truth = np.array([[c.pan, c.tilt, c.logf] for c in cams])
    x0 = truth + rng.normal(0, 1, truth.shape) * [0.003, 0.002, 0.01]
    unit = pt.Unit(0, list(range(n)), x0, {k: _noisy_evidence(c, rng) for k, c in enumerate(cams)},
                   _pairs(cams, rng, noise=0.5))
    rig0 = ptz.Rig(C_TRUE + PRIOR_OFF, cams[0].roll, 0.014, (5.0, 10.0, 3.0))  # prior centre off the truth
    return cams, unit, rig0


def _estimate(unit, rig0, **kw):
    shn, poses, info = pt.fit_shared([unit], rig0, PitchModel(), free=("Cx", "Cy", "Cz", "roll"), n_iter=30, want_cov=True, **kw)
    return shn, info["cov"]


def test_rig_estimate_within_3_sigma_of_truth_in_most_seeds():
    z, zm, errs = [], [], []
    for seed in range(5):
        cams, unit, rig0 = _rig_problem(seed)
        shn, cv = _estimate(unit, rig0)
        err = np.array([*(shn.C - C_TRUE), shn.roll - cams[0].roll])
        z.append(err / np.sqrt(np.diag(cv["cov"])))
        zm.append(err / np.sqrt(np.diag(cv["cov_model"])))
        errs.append(err[:3])
    z, zm = np.array(z), np.array(zm)
    inside = np.all(np.abs(z) < 3, axis=1)
    print(f"\nrig estimate err/sigma per seed (Cx, Cy, Cz, roll), inflated Sigma:\n{np.round(z, 2)}"
          f"\nsame with the raw Gauss-Newton Sigma (max |z| per seed): {np.round(np.abs(zm).max(1), 1)}"
          f"\nC errors (m): {np.round(errs, 2).tolist()}; coverage {inside.sum()}/5")
    assert inside.sum() >= 4
    # and the loose prior was actually moved towards the truth on C
    assert np.mean([np.linalg.norm(e) for e in errs]) < np.linalg.norm(PRIOR_OFF)


def test_rig_sigma_shrinks_with_focal_and_pan_range():
    _, u_small, rig0 = _rig_problem(0, pan_span=0.6, zoom_span=0.04)
    _, u_big, _ = _rig_problem(0, pan_span=6.0, zoom_span=0.6)
    _, cv_small = _estimate(u_small, rig0)
    _, cv_big = _estimate(u_big, rig0)
    iy = 1
    print(f"\nC_y sigma: narrow pan/zoom {np.sqrt(cv_small['cov'][iy, iy]):.2f} m, wide {np.sqrt(cv_big['cov'][iy, iy]):.2f} m")
    assert cv_big["cov"][iy, iy] < cv_small["cov"][iy, iy]


def test_rig_covariance_is_symmetric_psd_with_cy_cz_ridge():
    _, unit, rig0 = _rig_problem(1)
    _, cv = _estimate(unit, rig0)
    cov = cv["cov"]
    assert cov.shape == (4, 4) and cv["names"] == ["Cx", "Cy", "Cz", "roll"]
    assert np.allclose(cov, cov.T, atol=1e-12 * np.abs(cov).max())
    assert np.linalg.eigvalsh(cov).min() > 0
    assert cv["scale"] >= 1.0 and np.allclose(cv["cov_model"], cv["scale"] * cv["cov_nominal"])
    assert np.allclose(cov, cv["inflation"] * cv["cov_model"])
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    print(f"\nC_y-C_z corr {corr[1, 2]:+.2f}, C_x-C_y {corr[0, 1]:+.2f}, C_y-roll {corr[1, 3]:+.2f}; sd {np.round(sd, 3).tolist()}")
    assert abs(corr[1, 2]) > 0.3  # strong off-diagonal structure on the y / z ridge: a diagonal Sigma would miss it
    assert np.all(np.abs(corr) <= 1 + 1e-9)


def test_estimate_rig_off_path_is_bit_identical():
    """``want_cov`` only appends ``info["cov"]``: the solve (shared block, poses, costs) is the same bit for bit, through
    ``joint_solve`` (what ``solve_segment`` calls)."""
    cams = _sweep()
    rng = np.random.default_rng(0)
    rig = ptz.Rig(C_TRUE, cams[0].roll, cams[0].k1, (1.0, 3.0, 1.0))
    sh = pt.Shared.from_rig(rig)
    evs = [_evidence(c) for c in cams]
    truth = np.array([[c.pan, c.tilt, c.logf] for c in cams])
    x0 = truth + rng.normal(0, 1, truth.shape) * [0.01, 0.006, 0.04]
    pairs = _pairs(cams, rng)
    args = (len(cams), sh, x0, [evs[k] for k in range(0, len(cams), 2)], np.arange(0, len(cams), 2), pairs, {}, PitchModel())
    sh_a, p_a, i_a = pt.joint_solve(*args)
    sh_b, p_b, i_b = pt.joint_solve(*args, want_cov=False)
    sh_c, p_c, i_c = pt.joint_solve(*args, want_cov=True)  # nothing free: no covariance either
    for sh_x, p_x, i_x in ((sh_b, p_b, i_b), (sh_c, p_c, i_c)):
        assert np.array_equal(p_a, p_x) and np.array_equal(sh_a.vec(), sh_x.vec())
        assert i_a == i_x
    kw = {"free_shared": ("Cx", "Cy", "Cz", "roll"), "shared_sigma": [5.0, 10.0, 3.0, 0.035, 0.05]}
    sh_d, p_d, i_d = pt.joint_solve(*args, **kw)
    sh_e, p_e, i_e = pt.joint_solve(*args, want_cov=True, **kw)
    assert np.array_equal(p_d, p_e) and np.array_equal(sh_d.vec(), sh_e.vec())
    assert "cov" in i_e and {k: v for k, v in i_e.items() if k != "cov"} == i_d


def test_segment_runner_rig_estimate_record_and_states_untouched():
    """``solve_rig_estimate`` (the ``--estimate-rig`` step of the segment runner) on a stub cache: a json-able record with the
    required fields, and the normal solve's ``states`` array is not modified."""
    import json

    from footy_track.scripts import ptz_temporal_segment as seg

    cams, unit, rig0 = _rig_problem(0, n=6)

    class _Cache:
        ev_frames = list(range(6))

        def evidence(self, t):
            return unit.evidence[t]

    states = unit.states.copy()
    before = states.copy()
    rec = seg.solve_rig_estimate(_Cache(), states, [(0, 5)], unit.pairs, {}, rig0)
    assert np.array_equal(states, before)
    for k in ("rig", "cov", "cov_model", "prior", "n_anchors", "n_evidence_frames", "n_motion_pairs", "focal_min", "focal_max",
              "pan_min_deg", "pan_max_deg", "final_cost", "converged", "code_sha"):
        assert k in rec, k
    assert rec["n_evidence_frames"] == 6 and rec["n_motion_pairs"] == 5
    assert rec["prior"]["source"] == "rig_json" and rec["focal_min"] < rec["focal_max"]
    assert np.array(rec["cov"]).shape == (4, 4)
    json.dumps(rec)


def test_rig_bootstrap_record_shape_and_json():
    """``--bootstrap-rig``: half-sample resamples return the raw rigs, a 4x4 bootstrap covariance and z^2 statistics."""
    import json

    from footy_track.scripts import ptz_temporal_segment as seg

    cams, unit, rig0 = _rig_problem(1, n=8)

    class _Cache:
        ev_frames = list(range(8))

        def evidence(self, t):
            return unit.evidence[t]

    rec = seg.solve_rig_estimate(_Cache(), unit.states.copy(), [(0, 7)], unit.pairs, {}, rig0, bootstrap=5, seed=3)
    b = rec["bootstrap"]
    assert b["n"] + b["n_failed"] == 5 and b["n"] >= 4
    assert np.array(b["cov_boot"]).shape == (4, 4) and np.array(b["rigs"]).shape == (b["n"], 4)
    assert set(b) >= {"z2_cov_model", "z2_cov", "lambda_vs_cov_model"}
    json.dumps(rec)


def test_rig_estimate_multistart_option_runs_and_records_starts():
    import json

    from footy_track.scripts import ptz_temporal_segment as seg

    cams, unit, rig0 = _rig_problem(2, n=6)

    class _Cache:
        ev_frames = list(range(6))

        def evidence(self, t):
            return unit.evidence[t]

    rec = seg.solve_rig_estimate(_Cache(), unit.states.copy(), [(0, 5)], unit.pairs, {}, rig0, multistart=True)
    assert len(rec["multistart"]) == 9 and "final_C" in max(rec["multistart"], key=lambda s: "final_C" in s)
    json.dumps(rec)


def test_lm_rel_tol_default_is_unchanged_and_tight_rule_runs_at_least_as_long():
    """``rel_tol`` defaults to the production rule (1e-4): omitting it, passing the default, and ``fit_shared`` /
    ``joint_solve`` defaults are bit-identical; a tighter rule never stops earlier and never ends at a higher cost."""
    assert pt.LM_REL_TOL == 1e-4
    rng = np.random.default_rng(2)
    cams = [_cam(70.0 + 6 * k, 12.0 + (k % 3) * 2.5, 2400.0 + 500 * (k % 4), roll_deg=0.2, k1=0.014) for k in range(10)]
    cams = [c for c in cams if len(_visible_keypoints(c)) >= 4]
    truth = np.array([[c.pan, c.tilt, c.logf] for c in cams])
    units = [pt.Unit(k, [0], (truth[k] + rng.normal(0, 1, 3) * [0.004, 0.003, 0.02])[None], {0: _evidence(c)})
             for k, c in enumerate(cams)]
    rig0 = ptz.Rig(C_TRUE + [1.0, 5.0, -2.0], 0.2 * np.pi / 180, 0.014, (5.0, 10.0, 3.0))
    kw = {"free": ("Cx", "Cy", "Cz", "k1")}
    a, _, ia = pt.fit_shared(units, rig0, PitchModel(), **kw)
    b, _, ib = pt.fit_shared(units, rig0, PitchModel(), rel_tol=pt.LM_REL_TOL, **kw)
    assert np.array_equal(a.vec(), b.vec()) and ia == ib
    c, _, ic = pt.fit_shared(units, rig0, PitchModel(), rel_tol=1e-9, **kw)
    assert ic["stage2"]["iters"] >= ia["stage2"]["iters"]
    assert ic["stage2"]["cost"] <= ia["stage2"]["cost"] + 1e-9
