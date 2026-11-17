"""Shape-first centre fit (``calibration.centre_fit``): synthetic pinhole renders.

A full centre circle + halfway line and no touchline must now give the circle's LEFT / RIGHT / SPOT (and free
circle points) within a few px; a long lens (near affine) degrades to "ends of the diameter conjugate to the
line"; a symmetric frontal view (focal length unobservable from the shape) abstains instead of guessing.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from footy_track.calibration import centre_circle as cc
from footy_track.calibration import centre_fit as cf
from footy_track.calibration import geometry as geo
from footy_track.calibration import pitch_finders as pf
from footy_track.calibration.pitch_model import PitchModel

from .test_pitch_finders import H, W, camera_h, project, render, seg


def _frame(cam, target, f, pm: PitchModel, touch: bool = False):
    Hp = camera_h(cam, target, f=f)
    polys = [pm.segments()["HALFWAY_LINE"], pm.curves()["CENTRE_CIRCLE"]]
    if touch:
        polys.append(seg((-40, pm.hw), (40, pm.hw), 200))
    return render(Hp, polys), Hp


def _errs(res, Hp):
    truth = project(Hp, [p.pitch_xy for p in res.points]) / [W, H]
    return np.linalg.norm((np.array([p.uv for p in res.points]) - truth) * [W, H], axis=1)


def _editor_error(points, Hp, pm):
    """Pitch-metre error of the editor's fit from the returned points over a grid around the circle."""
    Hfit, _ = geo.fit_homography(np.array([p.uv for p in points]), np.array([p.pitch_xy for p in points]))
    grid = np.array([(x, y) for x in (-20.0, -10.0, 0.0, 10.0, 20.0) for y in (-10.0, 0.0, 10.0, 25.0)])
    back, _ = geo.apply_homography(Hfit, project(Hp, grid) / [W, H])
    return np.abs(back - grid).max()


def _by_id(res):
    return {p.pitch_id: p for p in res.points if p.pitch_id}


@pytest.mark.parametrize("dims", [(105.0, 68.0), (100.0, 64.0)])
def test_full_circle_without_touchline_gives_left_right_and_a_fit(dims):
    """A camera off to one side (circle away from the principal point, the line leaning): the square-pixel prior
    pins the focal length, so LEFT / RIGHT / SPOT come out without any touchline or rig."""
    pm = PitchModel(*dims)
    img, Hp = _frame((-22.0, -45.0, 18.0), (14.0, 8.0, 0.0), 1250.0, pm)
    det = pf.find_points("centre", img, None, pm)
    assert det.result is not None, det.reason
    r = det.result
    assert r.info["shape"]["accepted"] == "intrinsic", r.info["shape"]
    ids = _by_id(r)
    assert {"CENTRE_CIRCLE_TOP", "CENTRE_CIRCLE_BOTTOM", "CENTRE_CIRCLE_LEFT", "CENTRE_CIRCLE_RIGHT", "CENTRE_SPOT"} <= set(ids)
    assert _errs(r, Hp).max() < 4.0
    assert len(r.points) >= 4
    assert _editor_error(r.points, Hp, pm) < 0.4
    assert r.info["shape"]["f_over_w"] == pytest.approx(1250.0 / W, rel=0.05)


def test_circular_point_condition_holds_at_the_true_vanishing_point():
    """F(tau) = 0 and g = 1/f^2 at the true W: the equation the intrinsic solve is built on."""
    f = 1250.0
    Hp = camera_h((-22.0, -45.0, 18.0), (14.0, 8.0, 0.0), f=f)
    # ellipse of the circle through the camera, halfway line = x = 0
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    px = project(Hp, 9.15 * np.stack([np.cos(th), np.sin(th)], 1))
    (cx, cy), (a, b), ang = cv2.fitEllipse(px.astype(np.float32))
    E = cc.Ellipse((cx, cy), (a / 2, b / 2), np.deg2rad(ang))
    ends = project(Hp, [(0.0, 9.15), (0.0, -9.15)])
    d = ends[0] - ends[1]
    d = d / np.linalg.norm(d)
    g = cf.make_geom(E, ends[1], d, (W, H))
    # true W = image of y -> +inf along the halfway line
    far = project(Hp, [(0.0, 1e6)])[0]
    Wn = np.append((far - [W / 2, H / 2]) / W, 1.0)
    Wn /= np.linalg.norm(Wn)
    roots = cf.intrinsic_roots(g)
    assert roots, "no root"
    t, fw = min(roots, key=lambda r: abs(r[1] - f / W))
    assert fw == pytest.approx(f / W, rel=0.03)
    ax = cf.axis_from_W(g, cf._W_of_tau(g, np.array([t]))[0])
    assert ax is not None
    # W from the root is the true vanishing point (same direction)
    W_root = cf._W_of_tau(g, np.array([t]))[0]
    assert abs(abs(W_root @ Wn) - 1.0) < 1e-3


def test_long_lens_near_affine_degrades_to_the_conjugate_diameter():
    """Camera ~1 km away, f = 25 000 px (f / w ~ 20): the view is affine, O ~ the ellipse centre and LEFT / RIGHT are the ends of the
    diameter conjugate to the halfway line."""
    pm = PitchModel()
    img, Hp = _frame((100.0, -950.0, 130.0), (3.0, 6.0, 0.0), 25000.0, pm)
    det = pf.find_points("centre", img, None, pm)
    assert det.result is not None, det.reason
    r = det.result
    assert r.info["shape"]["accepted"] == "intrinsic", r.info["shape"]
    assert _errs(r, Hp).max() < 4.0
    ids = _by_id(r)
    # analytic affine answer: conjugate diameter of the halfway-line direction
    E = det.debug["ellipse"]
    L = det.debug["anchor"]
    M = E.matrix()
    qd = M @ L.d
    conj = np.linalg.solve(M, np.array([-qd[1], qd[0]]))  # direction conjugate to L.d (in the circle frame: perpendicular)
    conj = conj / np.linalg.norm(conj)
    lr = (ids["CENTRE_CIRCLE_RIGHT"].uv - ids["CENTRE_CIRCLE_LEFT"].uv) * [W, H]
    lr = lr / np.linalg.norm(lr)
    assert abs(abs(lr @ conj) - 1.0) < 2e-3
    spot = ids["CENTRE_SPOT"].uv * [W, H]
    assert np.linalg.norm(spot - np.asarray(E.centre)) < 3.0


def test_symmetric_frontal_view_abstains_instead_of_guessing():
    """No roll / yaw and the circle near the principal point: the focal length is unobservable from the shape alone
    (the circular-point equation is flat); with no touchline and no rig the finder keeps only the 2 seen points."""
    pm = PitchModel()
    img, Hp = _frame((0.0, -48.0, 17.0), (0.0, 1.0, 0.0), 1400.0, pm)
    det = pf.find_points("centre", img, None, pm)
    assert det.result is not None, det.reason
    r = det.result
    assert r.info["shape"]["accepted"] is None
    assert [p.pitch_id for p in r.points] == ["CENTRE_CIRCLE_TOP", "CENTRE_CIRCLE_BOTTOM"]
    assert not any(p.derived for p in r.points)


def test_touchline_end_still_wins_a_symmetric_view_and_gives_a_fit():
    pm = PitchModel()
    img, Hp = _frame((0.0, -48.0, 17.0), (0.0, 1.0, 0.0), 1400.0, pm, touch=True)
    det = pf.find_points("centre", img, None, pm)
    r = det.result
    assert r is not None and r.info["shape"]["accepted"] == "touchline", r and r.info
    assert {"HALFWAY_TOP", "CENTRE_CIRCLE_LEFT", "CENTRE_CIRCLE_RIGHT", "CENTRE_SPOT"} <= set(_by_id(r))
    assert _errs(r, Hp).max() < 4.0
    assert _editor_error(r.points, Hp, pm) < 0.4


def test_rig_fit_from_the_circle_alone_in_a_symmetric_view():
    """With the ground's camera centre known, pan / tilt / f come from the 2 points + the circle outline."""
    pm = PitchModel()
    cam = (0.1, -66.6, 15.7)
    img, Hp = _frame(cam, (0.0, 1.0, 0.0), 2900.0, pm)
    rig = cf.Rig("test", cam, 0.0, 0.0)
    det = pf.find_centre(cc.line_mask_for(img), None, pm, rig=rig)
    r = det.result
    assert r is not None and r.info["shape"]["accepted"] == "rig", r and r.info["shape"]
    assert _errs(r, Hp).max() < 4.0
    assert _editor_error(r.points, Hp, pm) < 0.5


def test_returned_points_always_fit_in_the_editor():
    """An accepted fit returns >= 4 points in general position (the editor fits at once), all from one H."""
    pm = PitchModel()
    img, Hp = _frame((-22.0, -45.0, 18.0), (14.0, 8.0, 0.0), 1250.0, pm)
    r = pf.find_points("centre", img, None, pm).result
    uv = geo.undistort_points(np.array([p.uv for p in r.points]), None)
    cfg = geo.analyse_configuration(uv, np.array([p.pitch_xy for p in r.points]))
    assert len(r.points) >= 4 and cfg["n_general_subsets"]
    geo.fit_homography(uv, np.array([p.pitch_xy for p in r.points]))  # must not raise


def test_to_norm_h_maps_pitch_to_normalised_image():
    pm = PitchModel()
    img, Hp = _frame((-22.0, -45.0, 18.0), (14.0, 8.0, 0.0), 1250.0, pm)
    det = pf.find_points("centre", img, None, pm)
    Hn = cf.to_norm_H(det.debug["H_px"], (W, H))
    pts = np.array([(0.0, 0.0), (5.0, 5.0), (-9.15, 0.0)])
    xy, _ = geo.apply_homography(Hn, pts)
    assert np.abs((xy - project(Hp, pts) / [W, H]) * [W, H]).max() < 4.0


def test_rig_for_clip_maps_known_grounds_only():
    assert cf.rig_for_clip("arsenal_nottinghamforest_seg000").ground == "emirates"
    assert cf.rig_for_clip("arsenal_mancity_seg010").ground == "emirates"
    assert cf.rig_for_clip("arsenal_bournemouth_1st_seg009").ground == "vitality"
    assert cf.rig_for_clip("arsenal_chelsea_seg001") is None
    assert cf.rig_for_clip(None) is None
