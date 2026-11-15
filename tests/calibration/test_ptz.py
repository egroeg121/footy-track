"""PTZ rig model: synthetic known-camera round trips (see docs/design/calibration_ptz.md)."""

from __future__ import annotations

import numpy as np
import pytest

from footy_track.calibration import ptz
from footy_track.calibration.pitch_model import PitchModel

C_TRUE = np.array([0.5, -63.0, 14.5])


def _cam(pan_deg=88.0, tilt_deg=14.0, f=3500.0, roll_deg=0.4, C=C_TRUE, k1=0.015):
    return ptz.Camera(np.deg2rad(pan_deg), np.deg2rad(tilt_deg), float(np.log(f)), np.deg2rad(roll_deg),
                      np.asarray(C, float), k1)


def _visible_keypoints(cam, margin=0.03):
    ids, xy = PitchModel().keypoint_array()
    uv, ok = cam.pitch_to_image(xy)
    ok &= np.all((uv > margin) & (uv < 1 - margin), axis=1)
    return [i for i, o in zip(ids, ok, strict=True) if o], xy[ok], uv[ok]


def _pitch_err(est, truth, n=7):
    gx, gy = np.meshgrid(np.linspace(-30, 30, n), np.linspace(-20, 20, n))
    g = np.column_stack([gx.ravel(), gy.ravel()])
    uv, ok = truth.pitch_to_image(g)
    ok &= np.all((uv > 0.02) & (uv < 0.98), axis=1)
    back, ok2 = est.image_to_pitch(uv[ok])
    assert ok2.all()
    return np.linalg.norm(back - g[ok], axis=1)


def test_look_at_centre_projects_to_image_centre():
    C = np.array([0.0, -65.0, 15.0])
    cam = ptz.Camera(np.pi / 2, np.arctan2(15.0, 65.0), np.log(3000.0), 0.0, C, 0.0)
    uv, ok = cam.pitch_to_image(np.array([[0.0, 0.0]]))
    assert ok.all() and np.allclose(uv, 0.5, atol=1e-6)
    # +x on the pitch is to the right of the image for a camera behind the near touchline looking at +y
    right, _ = cam.pitch_to_image(np.array([[20.0, 0.0]]))
    assert right[0, 0] > 0.5


def test_project_backproject_roundtrip_with_lens():
    cam = _cam()
    pts = np.array([[-10.0, -5.0], [12.0, 8.0], [0.0, 0.0], [25.0, -12.0]])
    uv, ok = cam.pitch_to_image(pts)
    assert ok.all()
    back, ok2 = cam.image_to_pitch(uv)
    assert ok2.all() and np.allclose(back, pts, atol=1e-6)
    # homography convention: undistorted normalised image -> pitch
    uvu = ptz.undistort_px(uv * ptz.CANON, cam.k1) / ptz.CANON
    ph = np.column_stack([uvu, np.ones(len(uvu))]) @ cam.homography().T
    assert np.allclose(ph[:, :2] / ph[:, 2:], pts, atol=1e-5)


def test_solve_points_recovers_camera_with_perturbed_prior():
    cam = _cam()
    ids, xy, uv = _visible_keypoints(cam)
    assert len(ids) >= 5
    rng = np.random.default_rng(0)
    noisy = uv + rng.normal(0, 1.0, uv.shape) / ptz.CANON
    # the rig believes C is 2 m off in y: with a tolerance the frame can absorb it
    rig = ptz.Rig(C_TRUE + [0.0, 2.0, 0.0], cam.roll, cam.k1, (1.0, 4.0, 1.0))
    est = ptz.solve_points(noisy, xy, rig)
    assert est is not None
    assert _pitch_err(est, cam).max() < 1.0
    assert est.info["rms_px"] < 3.0


@pytest.mark.parametrize("n", [2, 3])
def test_solve_points_few_points_fits_the_points(n):
    cam = _cam(pan_deg=80.0, tilt_deg=11.0, f=2600.0)
    ids, xy, uv = _visible_keypoints(cam)
    sel = np.arange(n) * (len(ids) // n)
    rig = ptz.Rig(C_TRUE, cam.roll, cam.k1, (0.0, 0.0, 0.0))  # hard-fixed C
    est = ptz.solve_points(uv[sel], xy[sel], rig)
    assert est is not None and est.info["rms_px"] < 5.0
    assert ptz.solve_points(uv[:1], xy[:1], rig) is None


def test_fit_rig_recovers_centre_roll_and_k1():
    rng = np.random.default_rng(1)
    frames, truth = [], []
    for pan, tilt, f in [(70, 12, 2500), (80, 13, 3500), (90, 15, 5000), (100, 13, 3000), (110, 11, 2600),
                         (85, 18, 2200), (95, 10, 4200), (75, 16, 3300), (105, 16, 2800), (88, 12, 3800)]:
        cam = _cam(pan, tilt, f, roll_deg=0.5, k1=0.02)
        ids, xy, uv = _visible_keypoints(cam)
        if len(ids) < 4:
            continue
        frames.append(ptz.RigFrame(uv + rng.normal(0, 1.0, uv.shape) / ptz.CANON, xy, key=f"f{pan}"))
        truth.append(cam)
    assert len(frames) >= 7
    rig, cams = ptz.fit_rig(frames, C0=(0.0, -65.0, 15.0))
    assert np.linalg.norm(rig.C - C_TRUE) < 2.5, rig.C
    assert abs(rig.k1 - 0.02) < 0.012
    assert abs(rig.roll - np.deg2rad(0.5)) < np.deg2rad(1.5)
    assert max(c.info["rms_px"] for c in cams) < 3.0


def test_search_lines_finds_known_camera_and_abstains_on_empty():
    cam = _cam(pan_deg=84.0, tilt_deg=13.0, f=2800.0)
    lines, grass = ptz.render_synthetic(cam, width=960)
    ev = ptz.LineEvidence.from_masks(lines, grass)
    rig = ptz.Rig(C_TRUE + [0.0, 1.5, 0.0], cam.roll, cam.k1, (1.0, 3.0, 1.0))
    est, hyps, info = ptz.search_lines(ev, rig, n_refine=6)
    assert est is not None, info
    assert _pitch_err(est, cam).mean() < 1.0
    assert hyps[0].score < 0.5
    # no evidence at all: nothing to anchor on, must not return a confident camera
    empty = ptz.LineEvidence.from_masks(np.zeros_like(lines), grass)
    est2, _, info2 = ptz.search_lines(empty, rig, n_refine=4)
    assert est2 is None and info2["reason"] in {"low support", "near-tie", "no hypothesis"}
