"""Multi-frame dataset helpers: name parsing and camera-motion alignment (synthetic)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from footy_track.multiframe.align import boxes_to_mask, estimate_pair, hist_distance, warp_to_ref
from footy_track.multiframe.mapping import FULL_MATCH_PREFIX, parse_name


def test_parse_name_clip_and_full_match():
    assert parse_name("astonvilla_seg080_000012_jpg.rf.f5d927ffdf8b6f47bb65b4537c7757ab.jpg") == (
        "astonvilla_seg080", 12)
    assert parse_name("arsenal_mancity_20250925_000472_png.rf.2fc88d470adac256787b21e1b0f38de7.txt") == (
        FULL_MATCH_PREFIX, 472)
    with pytest.raises(ValueError):
        parse_name("not_a_roboflow_name.jpg")


def _scene(seed=0, w=640, h=360):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w), dtype=np.uint8), (0, 0), 2.0)
    return cv2.cvtColor(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("method", ["ecc", "lk"])
def test_estimate_pair_recovers_pan(method):
    a = _scene()
    H_true = np.array([[1, 0, 9.0], [0, 1, -4.0], [0, 0, 1]])  # a -> b
    b = cv2.warpPerspective(a, H_true, (a.shape[1], a.shape[0]))  # point x in a is at H_true x in b
    pm = estimate_pair(a, b, scale=320, method=method)
    assert pm.ok, pm.reason
    # H maps a pixel coords to b pixel coords
    assert np.allclose(pm.H_px[:2, 2], H_true[:2, 2], atol=1.0)
    aligned = warp_to_ref(b, pm.H_px)
    inner = (slice(20, -20), slice(20, -20))
    assert np.abs(aligned.astype(int) - a.astype(int))[inner].mean() < 3.0


def test_masked_box_ignored_by_ecc():
    a = _scene(1)
    b = a.copy()
    b[100:160, 200:260] = 255  # a "player" that moves independently of the camera
    mask = boxes_to_mask(a.shape[:2], [(230 / 640, 130 / 360, 60 / 640, 60 / 360)], grow=0.5)
    pm = estimate_pair(a, b, mask, mask, scale=320, method="ecc")
    assert pm.ok and np.allclose(pm.H_px, np.eye(3), atol=0.5)


def test_hist_distance_flags_cut():
    a = _scene(0)
    same = hist_distance(a, a)
    other = hist_distance(a, cv2.applyColorMap(a[..., 0], cv2.COLORMAP_JET))
    assert same < 0.01 and other > 0.3
