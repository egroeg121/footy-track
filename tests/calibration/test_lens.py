"""Default lens per ground and "snap to lines" (calibration.lens) on synthetic curved-line frames."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from footy_track.calibration import geometry as geo
from footy_track.calibration import lens
from footy_track.calibration.centre_fit import load_rig
from footy_track.calibration.pitch_model import PitchModel

from .test_geometry import H, W, _camera_h

K1_TRUE = 0.03
KP = ["CENTRE_SPOT", "HALFWAY_TOP", "CENTRE_CIRCLE_LEFT", "CENTRE_CIRCLE_RIGHT", "CENTRE_CIRCLE_BOTTOM",
      "R_CORNER_TOP", "R_PEN_BOX_TOP_OUTER", "R_PEN_ARC_MID"]   # all inside the synthetic frame, well spread


def _render_mask(Ht, dist, shift_px=(0.0, 0.0)) -> np.ndarray:
    """Thin white-line mask of the whole pitch model seen through ``Ht`` + ``dist`` (lines curve with k1)."""
    mask = np.zeros((H, W), np.uint8)
    pitch = PitchModel()
    for poly in pitch.polylines().values():
        for run in geo.project_polyline_to_image(Ht, poly, 0.25, dist):
            px = run * np.array([W, H]) - 0.5 + np.array(shift_px)
            cv2.polylines(mask, [np.round(px).astype(np.int32)], False, 255, 2)
    return mask > 0


def _clicks(Ht, dist):
    pitch = PitchModel()
    xy = np.array([pitch.keypoints()[k] for k in KP])
    img, _ = geo.pitch_to_image(Ht, xy, dist)
    return img, xy


def _plain_fit(img, xy):
    H0, _ = geo.fit_homography(img, xy, method="dlt")
    return H0


@pytest.fixture(scope="module")
def scene():
    Ht = _camera_h()
    dist = geo.make_distortion(K1_TRUE, W / H)
    ev = lens.Evidence.from_mask(_render_mask(Ht, dist))
    img, xy = _clicks(Ht, dist)
    return {"Ht": Ht, "dist": dist, "ev": ev, "img": img, "xy": xy, "pitch": PitchModel()}


def test_default_distortion_is_the_rigs_k1_per_ground():
    d = lens.default_distortion("arsenal_mancity_seg010")
    assert d["model"] == "division" and d["k1"] == pytest.approx(load_rig("emirates").k1)
    assert d["aspect"] == pytest.approx(16 / 9) and d["centre"] == [0.5, 0.5]
    assert lens.default_distortion("arsenal_astonvilla_seg020")["k1"] == pytest.approx(load_rig("emirates").k1)
    assert lens.default_distortion("arsenal_bournemouth_1st_seg050")["k1"] == pytest.approx(0.0349, abs=1e-3)
    # the aspect follows the frame, unknown grounds / no clip get nothing
    assert lens.default_distortion("arsenal_mancity_seg010", 4 / 3)["aspect"] == pytest.approx(4 / 3)
    assert lens.default_distortion("chelsea_somebody_seg001") is None
    assert lens.default_distortion(None) is None
    # usable straight away by the geometry
    assert geo.distortion_params(d) is not None


def test_snap_recovers_k1_and_lowers_the_edge_residual(scene):
    ev, pitch = scene["ev"], scene["pitch"]
    img, xy = scene["img"], scene["xy"]
    H0 = _plain_fit(img, xy)                      # what the editor gave before: no lens, straight lines
    before = lens.edge_residual(ev, H0, None, pitch)
    r = lens.snap_to_lines(ev, H0, None, pitch, img, xy)
    assert r.accepted, r.reason
    assert r.k1_after == pytest.approx(K1_TRUE, abs=0.006)
    assert r.f1_after > r.f1_before + 0.02
    after = lens.edge_residual(ev, r.H, r.distortion, pitch)
    assert after["median_px"] < before["median_px"] and after["median_px"] < 1.5
    assert after["p90_px"] < before["p90_px"]
    # the clicked points were moved a few px at most, onto the snapped model: refitting them reproduces it
    assert r.max_click_shift_px <= lens.MAX_CLICK_SHIFT_PX
    H2 = _plain_fit(geo.undistort_points(r.points, r.distortion), xy)
    assert np.allclose(geo.pitch_to_image(H2, xy, r.distortion)[0], r.points, atol=2e-4)


def test_snap_k1_only_keeps_h(scene):
    ev, pitch = scene["ev"], scene["pitch"]
    img, xy = scene["img"], scene["xy"]
    und_true = geo.undistort_points(img, scene["dist"])
    H1 = _plain_fit(und_true, xy)                  # right H, but the lens is missing from the fit
    r = lens.snap_to_lines(ev, H1, None, pitch, img, xy, mode="k1")
    assert r.accepted, r.reason
    assert r.k1_after == pytest.approx(K1_TRUE, abs=0.006)
    assert np.allclose(r.H / r.H[2, 2], H1 / H1[2, 2])     # H untouched


def test_snap_refuses_a_fit_it_cannot_improve(scene):
    ev, pitch = scene["ev"], scene["pitch"]
    img, xy = scene["img"], scene["xy"]
    H_true = _plain_fit(geo.undistort_points(img, scene["dist"]), xy)
    r = lens.snap_to_lines(ev, H_true, scene["dist"], pitch, img, xy)      # already on the lines
    assert not r.accepted and "did not improve" in r.reason
    assert r.H is not None and np.array_equal(r.H, H_true)                  # the old fit comes back, unchanged
    assert r.distortion == scene["dist"] and r.points is None
    assert r.f1_after == r.f1_before


def test_snap_refuses_to_drag_the_clicked_points(scene):
    """Lines 9 px off the user's points: snapping would improve the F1 but move the clicks too far."""
    pitch = scene["pitch"]
    ev = lens.Evidence.from_mask(_render_mask(scene["Ht"], scene["dist"], shift_px=(9.0, 6.0)))
    img, xy = scene["img"], scene["xy"]
    H_true = _plain_fit(geo.undistort_points(img, scene["dist"]), xy)
    # soft point weight pulls it back, so ask for a tight limit to exercise the hard check
    r = lens.snap_to_lines(ev, H_true, scene["dist"], pitch, img, xy, max_click_shift_px=0.2)
    assert not r.accepted and r.reason
    assert np.array_equal(r.H, H_true) and r.points is None


def test_snap_needs_line_evidence(scene):
    ev = lens.Evidence.from_mask(np.zeros((H, W), bool))
    H0 = _plain_fit(scene["img"], scene["xy"])
    r = lens.snap_to_lines(ev, H0, None, scene["pitch"], scene["img"], scene["xy"])
    assert not r.accepted and "few line pixels" in r.reason
