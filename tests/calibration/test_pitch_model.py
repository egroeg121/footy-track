"""Pitch model: dimensions, symmetry and keypoint id stability."""

from __future__ import annotations

import numpy as np
import pytest

from footy_track.calibration.pitch_model import PitchModel


def test_default_dims_and_touchlines():
    p = PitchModel()
    assert (p.length, p.width) == (105.0, 68.0)
    seg = p.segments()
    assert np.allclose(seg["TOUCHLINE_TOP"], [[-52.5, 34], [52.5, 34]])
    assert np.allclose(seg["GOAL_LINE_R"], [[52.5, -34], [52.5, 34]])
    assert np.allclose(seg["HALFWAY_LINE"], [[0, -34], [0, 34]])


@pytest.mark.parametrize("dims", [(105, 68), (100, 64), (110, 75)])
def test_box_and_area_dimensions(dims):
    p = PitchModel(*dims)
    seg, hl = p.segments(), dims[0] / 2
    front = seg["R_PEN_BOX_FRONT"]
    assert np.isclose(front[0, 0], hl - 16.5)
    assert np.isclose(abs(front[1, 1] - front[0, 1]), 40.32)
    ga = seg["L_GOAL_AREA_FRONT"]
    assert np.isclose(ga[0, 0], -hl + 5.5)
    assert np.isclose(abs(ga[1, 1] - ga[0, 1]), 18.32)
    kp = p.keypoints()
    assert np.isclose(kp["R_PEN_SPOT"][0], hl - 11)


def test_circle_and_arcs():
    p = PitchModel()
    c = p.curves()
    assert np.allclose(np.linalg.norm(c["CENTRE_CIRCLE"], axis=1), 9.15)
    spot = np.array([52.5 - 11, 0])
    arc = c["R_PEN_ARC"]
    assert np.allclose(np.linalg.norm(arc - spot, axis=1), 9.15)
    # only the part outside the box (x < box front edge)
    assert arc[:, 0].max() <= 52.5 - 16.5 + 1e-9
    # arc ends are exactly the arc / box-edge keypoints
    kp = p.keypoints()
    ends = {tuple(np.round(arc[0], 6)), tuple(np.round(arc[-1], 6))}
    assert ends == {
        tuple(np.round(kp["R_PEN_ARC_TOP"], 6)),
        tuple(np.round(kp["R_PEN_ARC_BOTTOM"], 6)),
    }


def test_left_right_and_top_bottom_symmetry():
    kp = PitchModel().keypoints()
    for k, (x, y) in kp.items():
        if k.startswith("L_"):
            assert np.allclose(kp["R_" + k[2:]], (-x, y)), k
        if "TOP" in k and "GOAL" not in k.replace("GOALPOST", "") or k.endswith("_TOP"):
            bottom = k.replace("TOP", "BOTTOM")
            assert np.allclose(kp[bottom], (x, -y)), k
    assert kp["CENTRE_SPOT"] == (0.0, 0.0)
    assert kp["L_CORNER_TOP"] == (-52.5, 34.0)  # +y is the far touchline


def test_keypoint_ids_unique_and_stable():
    p = PitchModel()
    ids, arr = p.keypoint_array()
    assert len(ids) == len(set(ids))
    assert arr.shape == (len(ids), 2)
    # dimension changes move points, never rename them
    assert set(PitchModel(100, 64).keypoints()) == set(ids)
    for stable in (
        "L_PEN_BOX_TOP_OUTER",
        "CENTRE_SPOT",
        "HALFWAY_TOP",
        "CENTRE_CIRCLE_TOP",
        "R_PEN_ARC_BOTTOM",
        "R_CORNER_BOTTOM",
    ):
        assert stable in ids


def test_all_keypoints_lie_on_a_drawn_line():
    p = PitchModel()
    polys = list(p.polylines().values()) + [np.array([[0, 0], [0, 0]])]  # spots
    spots = list(p.spots().values())
    for k, xy in p.keypoints().items():
        if "GOALPOST" in k or k.endswith("PEN_ARC_MID"):
            continue  # posts sit on the goal line; arc-mid is on the arc
        pt = np.array(xy)
        if any(np.allclose(pt, s) for s in spots):
            continue
        d = min(_dist_to_polyline(pt, poly) for poly in polys)
        assert d < 0.1, (k, d)


def _dist_to_polyline(pt, poly):
    best = np.inf
    for a, b in zip(poly[:-1], poly[1:], strict=True):
        ab = b - a
        t = 0.0 if not ab.any() else np.clip((pt - a) @ ab / (ab @ ab), 0, 1)
        best = min(best, np.linalg.norm(pt - (a + t * ab)))
    return best


def test_bad_dimensions_rejected():
    with pytest.raises(ValueError):
        PitchModel(30, 68)
    with pytest.raises(ValueError):
        PitchModel(105, 30)
