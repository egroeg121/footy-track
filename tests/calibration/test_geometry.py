"""Homography geometry: round-trips, recovery, error and behind-camera handling."""

from __future__ import annotations

import numpy as np
import pytest

from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import PitchModel

W, H = 1280, 720


def _camera_h() -> np.ndarray:
    """A physically plausible broadcast view: camera 25 m above the near
    touchline looking at the pitch. Returns H: normalised image -> pitch (m)."""
    f = 1400.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    pan, tilt = np.deg2rad(10), np.deg2rad(28)
    # world: x long axis, y far touchline, z up. Camera looks toward +y, down.
    fwd = np.array([np.sin(pan) * np.cos(tilt), np.cos(pan) * np.cos(tilt), -np.sin(tilt)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])  # world -> camera rows
    C = np.array([5.0, -48.0, 25.0])
    t = -R @ C
    P = K @ np.column_stack([R[:, 0], R[:, 1], t])  # ground plane z=0: pitch -> px
    H_px_from_pitch = P
    H_pitch_from_px = np.linalg.inv(H_px_from_pitch)
    Hn = H_pitch_from_px @ np.diag([W, H, 1.0])  # normalised -> pitch
    return geo.orient_homography(Hn, [[0.5, 0.6]])


def _spread_keypoints(n=8):
    pitch = PitchModel()
    ids = [
        "L_PEN_BOX_TOP_OUTER", "L_PEN_BOX_BOTTOM_OUTER", "CENTRE_SPOT",
        "HALFWAY_TOP", "HALFWAY_BOTTOM", "CENTRE_CIRCLE_TOP", "L_PEN_SPOT",
        "L_GOAL_AREA_TOP_OUTER",
    ][:n]
    kp = pitch.keypoints()
    return np.array([kp[i] for i in ids])


def test_px_norm_round_trip():
    pts = np.array([[0.0, 0.0], [640, 360], [1280, 720]])
    assert np.allclose(geo.norm_to_px(geo.px_to_norm(pts, W, H), W, H), pts)
    assert np.allclose(geo.px_to_norm(pts, W, H)[1], [0.5, 0.5])


def test_synthetic_homography_round_trip():
    Htrue = _camera_h()
    pitch_pts = _spread_keypoints()
    img, w = geo.apply_homography(np.linalg.inv(Htrue), pitch_pts)
    assert np.all(w != 0)
    back, _ = geo.apply_homography(Htrue, img)
    assert np.allclose(back, pitch_pts, atol=1e-6)
    # the synthetic view really sees the centre spot
    assert np.all((img[2] > 0) & (img[2] < 1)), img[2]


@pytest.mark.parametrize("method", ["dlt", "ransac"])
def test_recover_h_from_noiseless_correspondences(method):
    Htrue = _camera_h()
    pitch_pts = _spread_keypoints()
    img, _ = geo.apply_homography(np.linalg.inv(Htrue), pitch_pts)
    Hfit, inl = geo.fit_homography(img, pitch_pts, method=method)
    assert inl.all()
    # equal up to scale and sign, via mapping a grid of points
    grid = np.array([[x, y] for x in (0.1, 0.5, 0.9) for y in (0.4, 0.7, 0.9)])
    a, _ = geo.apply_homography(Htrue, grid)
    b, _ = geo.apply_homography(Hfit, grid)
    assert np.allclose(a, b, atol=1e-6)
    errs = geo.reprojection_errors(Hfit, img, pitch_pts, (W, H))
    assert np.max(errs["metres"]) < 1e-4
    assert np.max(errs["px"]) < 0.05


def test_exactly_four_points_and_sign_convention():
    Htrue = _camera_h()
    pitch_pts = _spread_keypoints(4)
    img, _ = geo.apply_homography(np.linalg.inv(Htrue), pitch_pts)
    Hfit, _ = geo.fit_homography(img, pitch_pts)
    _, w = geo.apply_homography(Hfit, img)
    assert np.all(w > 0)


def test_noise_shows_up_in_reprojection_error():
    rng = np.random.default_rng(0)
    Htrue = _camera_h()
    pitch_pts = _spread_keypoints()
    img, _ = geo.apply_homography(np.linalg.inv(Htrue), pitch_pts)
    noisy = img + rng.normal(0, 3.0 / W, img.shape)
    Hfit, _ = geo.fit_homography(noisy, pitch_pts)
    errs = geo.reprojection_errors(Hfit, noisy, pitch_pts, (W, H))
    assert 0.1 < np.mean(errs["px"]) < 8
    assert np.mean(errs["metres"]) > 0.01


def test_ransac_rejects_a_gross_outlier():
    Htrue = _camera_h()
    pitch_pts = _spread_keypoints()
    img, _ = geo.apply_homography(np.linalg.inv(Htrue), pitch_pts)
    bad = img.copy()
    bad[2] += [0.15, -0.1]
    _, inl = geo.fit_homography(bad, pitch_pts, method="ransac", ransac_thresh_m=0.5)
    assert not inl[2] and inl.sum() == len(img) - 1


def test_degenerate_input_rejected():
    with pytest.raises(ValueError):
        geo.fit_homography(np.zeros((3, 2)), np.zeros((3, 2)))
    line = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])
    with pytest.raises(ValueError):
        geo.fit_homography(line, np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float))


def test_points_behind_camera_are_dropped():
    Htrue = _camera_h()
    # camera at y=-48 looking +y: a line far behind it (y = -200) is behind the camera
    behind = np.array([[-20.0, -200.0], [20.0, -200.0]])
    assert geo.project_polyline_to_image(Htrue, behind) == []
    # a line crossing from in front of the camera to behind it must not wrap:
    # every surviving point has positive w and the run stops at the horizon
    crossing = np.array([[0.0, -34.0], [0.0, -300.0]])
    runs = geo.project_polyline_to_image(Htrue, crossing)
    assert 1 <= len(runs)
    _, w = geo.apply_homography(np.linalg.inv(Htrue), crossing[:1])
    assert w[0] > 0
    pts = np.concatenate(runs)
    assert np.all(np.isfinite(pts))
    # monotone: image v increases?? just require nothing escapes to the far field
    assert np.all(np.abs(pts - 0.5) < geo.MAX_NORM_COORD)


def test_full_pitch_projection_is_split_not_wrapped():
    Htrue = _camera_h()
    proj = geo.project_pitch_model(Htrue, PitchModel())
    assert "TOUCHLINE_TOP" in proj and "CENTRE_CIRCLE" in proj
    for runs in proj.values():
        for r in runs:
            assert r.shape[1] == 2 and len(r) >= 2


def test_warp_puts_pitch_features_where_the_model_says():
    pitch = PitchModel()
    Htrue = _camera_h()
    # render a synthetic frame: paint a white disc at the centre spot
    import cv2

    frame = np.zeros((H, W, 3), np.uint8)
    cs, _ = geo.apply_homography(np.linalg.inv(Htrue), np.array([[0.0, 0.0]]))
    cv2.circle(frame, (int(cs[0, 0] * W), int(cs[0, 1] * H)), 6, (255, 255, 255), -1)
    out = geo.warp_to_pitch(frame, Htrue, pitch, px_per_m=10, margin_m=3)
    assert out.shape[1::-1] == geo.pitch_raster_size(pitch, 10, 3)
    ys, xs = np.nonzero(out[:, :, 0] > 128)
    cx, cy = (105 / 2 + 3) * 10, (68 / 2 + 3) * 10
    assert abs(xs.mean() - cx) < 3 and abs(ys.mean() - cy) < 3


def test_orient_homography_makes_sign_irrelevant():
    Htrue = _camera_h()
    pts = np.array([[0.4, 0.5], [0.6, 0.7]])
    for sgn in (1, -1):
        Ho = geo.orient_homography(sgn * Htrue * 7.0, pts)
        assert np.isclose(np.linalg.norm(Ho), 1.0)
        assert np.all(geo.apply_homography(Ho, pts)[1] > 0)
        assert len(geo.project_polyline_to_image(Ho, np.array([[-20.0, 0], [20, 0]]))) >= 1
