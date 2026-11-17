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
