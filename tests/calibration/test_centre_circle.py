"""Synthetic-frame tests of the line x circle point detector (centre circle and penalty D)."""

from __future__ import annotations

import cv2
import numpy as np

from footy_track.calibration import centre_circle as cc
from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import PitchModel

W, H = 1280, 720
PM = PitchModel()


def _homography(pitch_quad, img_quad):
    """pitch metres -> observed-plane px (before lens distortion)."""
    return cv2.getPerspectiveTransform(np.float32(pitch_quad), np.float32(img_quad))


def _project(Hp, pts_m, dist=None):
    px = (
        cv2.perspectiveTransform(np.float32(pts_m).reshape(-1, 1, 2), Hp)
        .reshape(-1, 2)
        .astype(np.float64)
    )
    if dist is None:
        return px
    norm = geo.px_to_norm(px, W, H)
    return geo.norm_to_px(geo.distort_points(norm, dist), W, H)


def _render(Hp, strokes, dist=None):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (40, 130, 40)  # grass (BGR)
    rng = np.random.default_rng(1)
    img = np.clip(img.astype(int) + rng.integers(-4, 5, img.shape), 0, 255).astype(
        np.uint8
    )
    for pts in strokes:
        px = _project(Hp, pts, dist)
        cv2.polylines(
            img,
            [np.round(px).astype(np.int32).reshape(-1, 1, 2)],
            False,
            (235, 235, 235),
            4,
            cv2.LINE_AA,
        )
    return img


def _line(a, b, n=200):
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(a) * (1 - t) + np.asarray(b) * t


def _arc(cx, cy, r, a0, a1, n=300):
    t = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], 1)


def _check(det, truth_uv, ids, tol_px=3.0):
    assert det.result is not None, det.reason
    r = det.result
    assert r.pitch_ids == ids
    err = np.linalg.norm((r.points_uv - truth_uv) * [W, H], axis=1)
    assert err.max() < tol_px, err
    assert r.points_uv[0, 1] < r.points_uv[1, 1]  # far first
    assert r.confidence > 0.6


def test_centre_circle_with_lens():
    dist = geo.make_distortion(0.03)
    Hp = _homography(
        [(-20, -15), (20, -15), (20, 25), (-20, 25)],
        [(200, 650), (1080, 650), (900, 200), (380, 200)],
    )
    strokes = [_line((0, -25), (0, 28)), _arc(0, 0, 9.15, 0, 2 * np.pi)]
    img = _render(Hp, strokes, dist)
    pts = _project(Hp, [(0, 9.15), (0, -9.15)], dist) / [W, H]
    _check(
        cc.detect_centre_circle(img, dist),
        pts,
        ("CENTRE_CIRCLE_TOP", "CENTRE_CIRCLE_BOTTOM"),
    )
    # the lens is optional: without it the error is a little larger but still small
    det = cc.detect_centre_circle(img, None)
    assert det.result is not None
    assert np.linalg.norm((det.result.points_uv - pts) * [W, H], axis=1).max() < 8


def test_centre_circle_pole_is_length_axis_vanishing_point():
    Hp = _homography(
        [(-20, -15), (20, -15), (20, 25), (-20, 25)],
        [(200, 650), (1080, 650), (900, 200), (380, 200)],
    )
    img = _render(Hp, [_line((0, -25), (0, 28)), _arc(0, 0, 9.15, 0, 2 * np.pi)])
    r = cc.detect_centre_circle(img, None).result
    assert r is not None
    # true vanishing point of +x: H maps pitch -> px, direction (1,0,0) -> first column
    vp = Hp @ np.array([1.0, 0.0, 0.0])
    mid = r.midpoint_u * [W, H]
    want = vp[:2] - mid * vp[2]
    want = want / np.linalg.norm(want)
    got = r.pole_h[:2] * [W, H] - mid * r.pole_h[2]
    got = got / np.linalg.norm(got)
    assert np.degrees(np.arccos(min(1.0, abs(want @ got)))) < 3.0


def test_penalty_arc_left_end():
    hl = PM.hl
    Hp = _homography(
        [(-52, -25), (-20, -25), (-20, 25), (-52, 25)],
        [(150, 660), (1130, 660), (900, 170), (380, 170)],
    )
    xb = -hl + 16.5
    strokes = [
        _line((xb, -20.16), (xb, 20.16)),
        _arc(-hl + 11.0, 0, 9.15, -np.arccos(5.5 / 9.15), np.arccos(5.5 / 9.15)),
    ]
    img = _render(Hp, strokes)
    kp = PM.keypoints()
    pts = _project(Hp, [kp["L_PEN_ARC_TOP"], kp["L_PEN_ARC_BOTTOM"]]) / [W, H]
    _check(cc.detect_pen_arc(img, None), pts, ("L_PEN_ARC_TOP", "L_PEN_ARC_BOTTOM"))


def test_penalty_arc_right_end_mirrors_to_R():
    hl = PM.hl
    Hp = _homography(
        [(20, -25), (52, -25), (52, 25), (20, 25)],
        [(150, 660), (1130, 660), (900, 170), (380, 170)],
    )
    xb = hl - 16.5
    strokes = [
        _line((xb, -20.16), (xb, 20.16)),
        _arc(
            hl - 11.0,
            0,
            9.15,
            np.pi - np.arccos(5.5 / 9.15),
            np.pi + np.arccos(5.5 / 9.15),
        ),
    ]
    img = _render(Hp, strokes)
    kp = PM.keypoints()
    pts = _project(Hp, [kp["R_PEN_ARC_TOP"], kp["R_PEN_ARC_BOTTOM"]]) / [W, H]
    _check(cc.detect_pen_arc(img, None), pts, ("R_PEN_ARC_TOP", "R_PEN_ARC_BOTTOM"))


def test_abstains_on_plain_grass_and_on_a_lone_line():
    Hp = _homography(
        [(-20, -15), (20, -15), (20, 25), (-20, 25)],
        [(200, 650), (1080, 650), (900, 200), (380, 200)],
    )
    plain = _render(Hp, [])
    d = cc.detect_circle_points(plain)
    assert d.result is None and d.reason
    lone = _render(Hp, [_line((0, -25), (0, 28))])
    d = cc.detect_circle_points(lone)
    assert d.result is None and d.reason


def test_correspondences_use_labeller_shape():
    Hp = _homography(
        [(-20, -15), (20, -15), (20, 25), (-20, 25)],
        [(200, 650), (1080, 650), (900, 200), (380, 200)],
    )
    img = _render(Hp, [_line((0, -25), (0, 28)), _arc(0, 0, 9.15, 0, 2 * np.pi)])
    r = cc.detect_centre_circle(img).result
    recs = r.correspondences()
    assert [c["pitch_id"] for c in recs] == [
        "CENTRE_CIRCLE_TOP",
        "CENTRE_CIRCLE_BOTTOM",
    ]
    assert all(0.0 <= v <= 1.0 for c in recs for v in c["image"])
    kp = PM.keypoints()
    assert kp["CENTRE_CIRCLE_TOP"] == (0.0, 9.15) and kp["CENTRE_CIRCLE_BOTTOM"] == (
        0.0,
        -9.15,
    )
