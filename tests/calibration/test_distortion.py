"""Lens distortion (division model): geometry, back-compat (k1 = 0), refinement, projection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.candidates import find_candidates
from footy_track.calibration.refine import refine_homography
from footy_track.calibration.prelabel.fit import Label, fit_and_score

from .test_geometry import H, W, _camera_h
from .test_prelabel import KP, PITCH

K1 = 0.03


def _dist(k=K1):
    return geo.make_distortion(k, W / H)


def test_distort_undistort_roundtrip():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, (200, 2))
    for k in (-0.05, 0.02, 0.1):
        q = geo.undistort_norm(geo.distort_norm(p, k), k)
        ok = np.isfinite(q).all(axis=1)
        assert ok.sum() > 150
        assert np.allclose(q[ok], p[ok], atol=1e-9)


def test_zero_or_missing_k1_is_the_plain_homography():
    pts = np.array([[0.1, 0.2], [0.9, 0.8]])
    for d in (None, {}, {"model": "division", "k1": 0.0}, {"model": "division", "k1": None}):
        assert np.array_equal(geo.undistort_points(pts, d), pts)
        assert np.array_equal(geo.distort_points(pts, d), pts)
    assert geo.make_distortion(0.0) is None
    Hc = _camera_h()
    plain = geo.project_pitch_model(Hc, PITCH)
    for name, runs in geo.project_pitch_model(Hc, PITCH, distortion={"k1": 0.0}).items():
        for a, b in zip(runs, plain[name], strict=True):
            assert np.array_equal(a, b)


def test_distortion_bows_a_straight_line_and_the_centre_is_fixed():
    Hc = _camera_h()
    top = PITCH.polylines()["TOUCHLINE_TOP"]
    straight = geo.project_polyline_to_image(Hc, top)[0]
    bowed = geo.project_polyline_to_image(Hc, top, distortion=_dist())[0]
    # straight: collinear (all points within 1e-9 of the chord); bowed: not
    def sag(r):
        a, b = r[0], r[-1]
        n = np.array([b[1] - a[1], a[0] - b[0]]) / np.hypot(*(b - a))
        return np.abs((r - a) @ n).max()
    assert sag(straight) < 1e-9
    assert sag(bowed) > 1e-3
    assert np.allclose(geo.distort_norm([[0.5, 0.5]], K1), [[0.5, 0.5]])


def test_reprojection_errors_with_distortion_are_zero_for_consistent_points():
    Hc = _camera_h()
    d = _dist()
    xy = np.array([[0.0, 0.0], [10.0, 5.0], [-20.0, -10.0], [30.0, 20.0]])
    obs, _ = geo.pitch_to_image(Hc, xy, d)
    e = geo.reprojection_errors(Hc, obs, xy, (W, H), d)
    assert np.allclose(e["metres"], 0, atol=1e-6) and np.allclose(e["px"], 0, atol=1e-6)
    # without the lens the same points do not fit
    assert geo.reprojection_errors(Hc, obs, xy, (W, H))["metres"].max() > 0.05


def test_warp_without_distortion_is_unchanged_and_with_it_is_undistorted():
    frame = np.random.default_rng(1).integers(0, 255, (H, W, 3), dtype=np.uint8)
    Hc = _camera_h()
    plain = geo.warp_to_pitch(frame, Hc, PITCH, 4.0, 3.0)
    assert np.array_equal(plain, geo.warp_to_pitch(frame, Hc, PITCH, 4.0, 3.0, distortion=None))
    assert np.array_equal(plain, geo.warp_to_pitch(frame, Hc, PITCH, 4.0, 3.0, distortion={"k1": 0.0}))
    # a uniform-colour frame stays uniform inside the view but the view shape differs
    d = geo.warp_to_pitch(frame, Hc, PITCH, 4.0, 3.0, distortion=_dist())
    assert d.shape == plain.shape and not np.array_equal(d, plain)


def _distorted_frame(k):
    """Green pitch with white lines drawn through the camera H AND the lens."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (40, 140, 50)
    d = geo.make_distortion(k, W / H)
    for runs in geo.project_pitch_model(_camera_h(), PITCH, distortion=d).values():
        for run in runs:
            cv2.polylines(img, [np.round(run * np.array([W, H])).astype(np.int32)], False, (250, 250, 250), 5,
                          cv2.LINE_AA)
    return img


def _line_err_px(Hx, d, k_true):
    """Mean px distance of the model's keypoints from where the true (distorted) camera puts them."""
    errs = []
    for kid, xy in PITCH.keypoints().items():
        true, wt = geo.pitch_to_image(_camera_h(), np.array([xy]), geo.make_distortion(k_true, W / H))
        if not (0.05 < true[0, 0] < 0.95 and 0.05 < true[0, 1] < 0.95):
            continue
        got, _ = geo.pitch_to_image(Hx, np.array([xy]), d)
        errs.append(np.hypot(*((got[0] - true[0]) * [W, H])))
    return float(np.mean(errs))


@pytest.fixture(scope="module")
def det_k():
    return find_candidates(_distorted_frame(K1))


def test_refinement_estimates_k1_jointly_and_reduces_the_error(det_k):
    H0 = _camera_h()
    plain = refine_homography(H0, PITCH, det_k.lines_mask, det_k.grass)
    free = refine_homography(H0, PITCH, det_k.lines_mask, det_k.grass, fit_k1=True)
    assert free.k1_fitted
    assert 0.25 * K1 < free.k1 < 1.6 * K1  # right sign, same order (k1 trades off against H, so not exact)
    d_free = geo.make_distortion(free.k1, W / H)
    assert _line_err_px(free.H, d_free, K1) < _line_err_px(plain.H, None, K1)


def test_k1_is_pulled_to_zero_on_an_undistorted_frame(frame_plain=None):
    det0 = find_candidates(_distorted_frame(0.0))
    r = refine_homography(_camera_h(), PITCH, det0.lines_mask, det0.grass, fit_k1=True)
    assert abs(r.k1) < 0.005


def test_fixed_k1_is_held_and_reported_unchanged(det_k):
    r = refine_homography(_camera_h(), PITCH, det_k.lines_mask, det_k.grass, k1=0.02)
    assert not r.k1_fitted and r.k1 == 0.02
    assert "k1" in r.to_dict()
    assert "k1" not in refine_homography(_camera_h(), PITCH, det_k.lines_mask, det_k.grass).to_dict()


def test_fit_and_score_with_distortion_stores_it_and_matches_labels(det_k):
    d = geo.make_distortion(K1, W / H)
    labels = []
    for kid in KP:
        uv, w = geo.pitch_to_image(_camera_h(), np.array([PITCH.keypoints()[kid]]), d)
        if w[0] > 0 and 0.03 < uv[0, 0] < 0.97 and 0.03 < uv[0, 1] < 0.97:
            labels.append(Label(kid, (float(uv[0, 0]), float(uv[0, 1])), 1.0))
    fit = fit_and_score(labels, PITCH, det_k, (W, H), k1=K1)
    assert fit.ok and fit.distortion is not None and fit.distortion["k1"] == pytest.approx(K1)
    assert np.nanmax([e for e in fit.err_px if e is not None]) < 6.0
    off = fit_and_score(labels, PITCH, det_k, (W, H))
    assert off.distortion is None  # k1 = 0: no distortion key at all
