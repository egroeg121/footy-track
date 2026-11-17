"""Synthetic-frame tests of the two editor finders: centre (circle + halfway line + touchline end) and D (penalty area).

Frames are rendered through a pinhole camera on the -y side of the pitch (so the vanishing points are
orthogonal for a real focal length, which the template search relies on), for pitches other than 105 x 68
to prove the request's dimensions are used and nothing is hard-wired.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from footy_track.calibration import geometry as geo
from footy_track.calibration import pitch_finders as pf
from footy_track.calibration.pitch_model import PitchModel

W, H = 1280, 720


def camera_h(cam, target, f=1500.0):
    """Pitch plane (metres) -> image px for a pinhole camera at ``cam`` looking at ``target`` (z up)."""
    cam, target = np.asarray(cam, float), np.asarray(target, float)
    fwd = target - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ cam])


def project(Hp, pts_m, dist=None):
    p = np.asarray(pts_m, float).reshape(-1, 2)
    ph = np.c_[p, np.ones(len(p))] @ Hp.T
    px = ph[:, :2] / ph[:, 2:3]
    if dist is None:
        return px
    return geo.norm_to_px(geo.distort_points(geo.px_to_norm(px, W, H), dist), W, H)


def render(Hp, polys, dist=None):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (40, 130, 40)
    rng = np.random.default_rng(1)
    img = np.clip(img.astype(int) + rng.integers(-4, 5, img.shape), 0, 255).astype(np.uint8)
    for poly in polys:
        px = project(Hp, poly, dist)
        cv2.polylines(img, [np.round(px).astype(np.int32).reshape(-1, 1, 2)], False, (235, 235, 235), 4, cv2.LINE_AA)
    return img


def seg(a, b, n=120):
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def centre_frame(pm: PitchModel, dist=None, touch=True):
    """Camera on the near side looking at the centre circle; both touchlines present when ``touch``."""
    Hp = camera_h((8.0, -48.0, 17.0), (0.0, 4.0, 0.0), f=1400.0)
    hw = pm.hw
    polys = [pm.segments()["HALFWAY_LINE"], pm.curves()["CENTRE_CIRCLE"]]
    if touch:
        polys += [seg((-40, hw), (40, hw), 200)]
    return render(Hp, polys, dist), Hp


def penalty_frame(pm: PitchModel, end="R", dist=None):
    s = 1.0 if end == "R" else -1.0
    Hp = camera_h((s * (pm.hl - 22.0), -46.0, 16.0), (s * (pm.hl - 9.0), 0.0, 0.0), f=1500.0)
    seg_, cur = pm.segments(), pm.curves()
    polys = [seg_[f"{end}_PEN_BOX_{k}"] for k in ("TOP", "BOTTOM", "FRONT")]
    polys += [seg_[f"{end}_GOAL_AREA_{k}"] for k in ("TOP", "BOTTOM", "FRONT")]
    polys += [seg_[f"GOAL_LINE_{end}"], cur[f"{end}_PEN_ARC"]]
    return render(Hp, polys, dist), Hp


def _errs(res, Hp, dist=None):
    """Pixel error of every returned point against the true projection of its pitch metres."""
    truth = project(Hp, [p.pitch_xy for p in res.points], dist) / [W, H]
    got = np.array([p.uv for p in res.points])
    return np.linalg.norm((got - truth) * [W, H], axis=1)


@pytest.mark.parametrize("dims", [(105.0, 68.0), (100.0, 64.0)])
def test_centre_finder_touchline_end_and_derived_points(dims):
    pm = PitchModel(*dims)
    img, Hp = centre_frame(pm)
    det = pf.find_points("centre", img, None, pm)
    assert det.result is not None, det.reason
    r = det.result
    ids = {p.pitch_id for p in r.points}
    assert {"CENTRE_CIRCLE_TOP", "CENTRE_CIRCLE_BOTTOM", "HALFWAY_TOP"} <= ids
    assert "CENTRE_SPOT" in ids
    assert _errs(r, Hp).max() < 4.0
    by = {p.pitch_id: p for p in r.points}
    assert by["HALFWAY_TOP"].pitch_xy == (0.0, pm.hw)  # the REQUEST's width, not 68
    assert not by["HALFWAY_TOP"].derived and not by["CENTRE_CIRCLE_TOP"].derived
    assert by["CENTRE_SPOT"].derived
    assert any(p.pitch_id is None and p.derived for p in r.points) or "CENTRE_CIRCLE_LEFT" in ids
    # >= 4 non-collinear points: the editor's fit recovers the camera
    img_n = geo.undistort_points(np.array([p.uv for p in r.points]), None)
    Hfit, _ = geo.fit_homography(img_n, np.array([p.pitch_xy for p in r.points]))
    grid = np.array([(x, y) for x in (-10.0, 0.0, 10.0) for y in (-8.0, 8.0, 20.0)])
    back, _ = geo.apply_homography(Hfit, project(Hp, grid) / [W, H])
    assert np.abs(back - grid).max() < 0.35


def test_centre_finder_uses_the_requested_width():
    """A wrong width moves the derived points: the width really enters the geometry (the sensitivity)."""
    pm_true = PitchModel(105.0, 68.0)
    img, Hp = centre_frame(pm_true)
    good = pf.find_points("centre", img, None, pm_true).result
    bad = pf.find_points("centre", img, None, PitchModel(105.0, 72.0)).result  # 2 m wrong half-width
    assert good is not None and bad is not None
    g = {p.pitch_id or tuple(p.pitch_xy): p for p in good.points}
    b = {p.pitch_id or tuple(p.pitch_xy): p for p in bad.points}
    common = [k for k in g if k in b and g[k].derived]
    assert common
    shift = max(np.linalg.norm((g[k].uv - b[k].uv) * [W, H]) for k in common)
    assert shift > 1.0  # a 4 m wider pitch moves the synthesised points (mildly: the circle pins most of the geometry)


def test_centre_finder_without_touchline_returns_only_the_two_circle_points():
    pm = PitchModel()
    img, Hp = centre_frame(pm, touch=False)
    det = pf.find_points("centre", img, None, pm)
    assert det.result is not None
    assert [p.pitch_id for p in det.result.points] == ["CENTRE_CIRCLE_TOP", "CENTRE_CIRCLE_BOTTOM"]
    assert not any(p.derived for p in det.result.points)


def test_centre_finder_with_lens():
    dist = geo.make_distortion(0.03)
    pm = PitchModel(100.0, 64.0)
    img, Hp = centre_frame(pm, dist)
    det = pf.find_points("centre", img, dist, pm)
    assert det.result is not None, det.reason
    assert {"HALFWAY_TOP", "CENTRE_CIRCLE_TOP"} <= {p.pitch_id for p in det.result.points}
    assert _errs(det.result, Hp, dist).max() < 6.0


@pytest.mark.parametrize("dims,end", [((105.0, 68.0), "R"), ((100.0, 64.0), "R"), ((100.0, 64.0), "L")])
def test_d_finder_fits_the_whole_penalty_area(dims, end):
    pm = PitchModel(*dims)
    img, Hp = penalty_frame(pm, end)
    det = pf.find_points("d", img, None, pm)
    assert det.result is not None, det.reason
    r = det.result
    assert r.info.get("source") in ("lines", "arc")
    assert r.info["end"] == end
    kp = pm.keypoints()
    ids = {p.pitch_id for p in r.points}
    assert {f"{end}_PEN_ARC_TOP", f"{end}_PEN_ARC_BOTTOM", f"{end}_PEN_BOX_TOP_OUTER"} <= ids
    for p in r.points:
        assert p.pitch_xy == tuple(kp[p.pitch_id])  # metres from the request's dims
    assert _errs(r, Hp).max() < 6.0
    by = {p.pitch_id: p for p in r.points}
    assert by[f"{end}_PEN_SPOT"].derived
    # >= 4 non-collinear points -> the editor fit works and is accurate
    img_n = np.array([p.uv for p in r.points])
    Hfit, _ = geo.fit_homography(img_n, np.array([p.pitch_xy for p in r.points]))
    s = 1.0 if end == "R" else -1.0
    grid = np.array([(s * (pm.hl - dx), y) for dx in (4.0, 12.0, 20.0, 28.0) for y in (-15.0, 0.0, 15.0)])
    back, _ = geo.apply_homography(Hfit, project(Hp, grid) / [W, H])
    assert np.abs(back - grid).max() < 0.6


def test_d_finder_dims_other_than_105x68_change_the_pitch_metres_only():
    pm = PitchModel(100.0, 64.0)
    img, Hp = penalty_frame(pm, "R")
    a = pf.find_points("d", img, None, pm).result
    b = pf.find_points("d", img, None, PitchModel(104.0, 64.0)).result  # same picture, other length
    assert a is not None and b is not None
    pa = {p.pitch_id: p for p in a.points}
    pb = {p.pitch_id: p for p in b.points}
    assert pa["R_PEN_SPOT"].pitch_xy == (50.0 - 11.0, 0.0) and pb["R_PEN_SPOT"].pitch_xy == (52.0 - 11.0, 0.0)
    assert np.allclose(pa["R_PEN_ARC_TOP"].uv, pb["R_PEN_ARC_TOP"].uv, atol=2.0 / W)  # same picture, same pixels


def test_d_finder_abstains_on_grass_and_on_a_lone_line():
    pm = PitchModel()
    Hp = camera_h((30.0, -46.0, 16.0), (43.0, 0.0, 0.0), f=1500.0)
    for polys in ([], [seg((pm.hl - 16.5, -20.16), (pm.hl - 16.5, 20.16))]):
        det = pf.find_points("d", render(Hp, polys), None, pm)
        assert det.result is None and det.reason
    plain = render(Hp, [])
    assert pf.find_points("centre", plain, None, pm).result is None


def test_correspondences_shape_and_unknown_kind():
    pm = PitchModel()
    img, _ = centre_frame(pm)
    recs = pf.find_points("centre", img, None, pm).result.correspondences()
    assert all({"image", "pitch_id", "pitch_xy", "confidence", "derived"} <= set(c) for c in recs)
    assert all(0.0 <= v <= 1.0 for c in recs for v in c["image"])
    assert all(c["pitch_id"] is not None or c["derived"] for c in recs)  # free points are always derived
    with pytest.raises(ValueError):
        pf.find_points("nope", img, None, pm)
