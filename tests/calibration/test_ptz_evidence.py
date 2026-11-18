"""PTZ line evidence: the saturation gate of the white-line mask is relative to the pitch's own colour."""

from __future__ import annotations

import cv2
import numpy as np

from footy_track.calibration import ptz
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.candidates import LINE_SAT_REL, line_mask


def _frame(grass_s: int, line_s: int, grass_v: int = 140, line_v: int = 200):
    """Flat turf (OpenCV HSV hue 60) with a 5 px tinted-white horizontal line across the middle."""
    h, w = 360, 640
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0], hsv[..., 1], hsv[..., 2] = 60, grass_s, grass_v
    hsv[h // 2 - 2:h // 2 + 3, 40:w - 40] = (60, line_s, line_v)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), np.ones((h, w), bool)


def _hit(mask: np.ndarray) -> float:
    h, w = mask.shape
    return float(mask[h // 2 - 2:h // 2 + 3, 60:w - 60].any(axis=0).mean())


def test_absolute_gate_loses_tinted_lines_on_a_saturated_pitch():
    bgr, grass = _frame(grass_s=230, line_s=135)  # chroma-bled line, saturation above the fixed 100 limit
    assert _hit(line_mask(bgr, grass, with_far=False)) < 0.2
    assert _hit(line_mask(bgr, grass, with_far=False, sat_rel=LINE_SAT_REL)) > 0.9


def test_relative_gate_keeps_the_emirates_behaviour():
    # median S ~155: 0.65 * 155 ~ 100, so the relative limit is the old absolute one on the pitch it was tuned on
    ok, grass = _frame(grass_s=155, line_s=80)
    bad, _ = _frame(grass_s=155, line_s=130)
    for kw in ({}, {"sat_rel": LINE_SAT_REL}):
        assert _hit(line_mask(ok, grass, with_far=False, **kw)) > 0.9
        assert _hit(line_mask(bad, grass, with_far=False, **kw)) < 0.2


def test_relative_gate_still_rejects_saturated_clutter():
    # a coloured object as saturated as the turf is not a white line, whatever the turf's saturation
    bgr, grass = _frame(grass_s=230, line_s=225)
    assert _hit(line_mask(bgr, grass, with_far=False, sat_rel=LINE_SAT_REL)) < 0.2


def test_explained_f1_is_bounded_and_symmetric():
    assert ptz.explained_f1(100, 100, 100) == 1.0  # every expected and every detected length matched
    assert ptz.explained_f1(0, 100, 100) == 0.0
    assert ptz.explained_f1(0, 0, 0) == 0.0  # nothing predicted, nothing detected
    assert ptz.explained_f1(50, 100, 100) == 0.5
    assert ptz.explained_f1(100, 100, 400) == ptz.explained_f1(100, 400, 100)  # recall and precision enter alike
    assert ptz.explained_f1(120, 100, 100) == 1.0  # a thinned mask can be shorter than the model: capped


def test_line_support_does_not_collapse_when_the_mask_misses_faint_lines():
    """Mask = only the halfway line of a known camera: the raw support is far below 0 (the predicted circle and
    touchline are 'unexplained' against a short detected length), the F1 stays a bounded, positive number that is
    higher for the true camera than for a wrong pan."""
    pitch = PitchModel()
    cam = ptz.Camera(np.deg2rad(90.0), np.deg2rad(14.0), float(np.log(3500.0)), 0.0, np.array([0.0, -50.0, 8.0]), 0.0)
    h, w = 540, 960
    uv, ok = cam.pitch_to_image(np.column_stack([np.zeros(80), np.linspace(-34, 34, 80)]))
    pts = (uv[ok] * [w, h]).astype(np.int32)
    lines = np.zeros((h, w), np.uint8)
    cv2.polylines(lines, [pts.reshape(-1, 1, 2)], False, 1, 3)
    ev = ptz.LineEvidence.from_masks(lines.astype(bool), np.ones((h, w), bool))
    wrong = ptz.Camera(np.deg2rad(97.0), np.deg2rad(14.0), float(np.log(3500.0)), 0.0, np.array([0.0, -50.0, 8.0]), 0.0)
    f_true, f_wrong = ptz.line_support([cam, wrong], ev, pitch)
    raw_true = 1.0 - ptz.score_cameras([cam], ev, pitch)[0]
    assert 0.0 < f_wrong < f_true <= 1.0
    assert raw_true < f_true  # the raw support under-rates the correct camera when most model lines are faint
