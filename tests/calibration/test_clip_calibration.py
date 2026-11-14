"""calibrate_clip on a synthetic pan/tilt/zoom clip: keyframes in, per-frame calibration out.

Reuses the synthetic camera of ``test_temporal`` (a textured green pitch with white lines rendered through known
homographies). Shot A is 16 frames of a slow pan, shot B (frames 16..21) an unrelated scene after a hard cut.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from footy_track.calibration import clip_calibration as cc
from footy_track.calibration import geometry as geo
from footy_track.calibration import temporal as tp
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.refine import model_samples
from tests.calibration.test_temporal import GRID, H, W, _cam, _render

N_A = 16


def _perr(Hest, Htrue):
    return float(tp.pitch_grid_error(Hest, Htrue, GRID).max())


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    cams = [_cam(6.0 + 0.6 * i, 28.0 + 0.05 * i, 700.0 + 3 * i) for i in range(N_A)]
    path = tmp_path_factory.mktemp("clipcal") / "clip.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (W, H))
    for c in cams:
        vw.write(_render(c))
    rng = np.random.default_rng(7)
    base = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3)).astype(np.uint8), (0, 0), 3)
    base = np.clip(base.astype(np.float32) * 0.5 + 100, 0, 255).astype(np.uint8)
    base[..., 0] = 220
    for i in range(6):
        vw.write(np.roll(base, 2 * i, axis=1))
    vw.release()
    motion = tp.clip_motion(path, max_gap=4, scale=W, mask_mode="all")
    return path, cams, motion


@pytest.fixture(scope="module")
def masks(clip):
    path, _, _ = clip
    s = cc.MaskStore(W)
    s.ensure(path, range(0, 22))
    return s


POL = cc.Policy(check_every=3, work_width=W)


def test_keyframes_propagate_both_ways_and_stop_at_the_cut(clip, masks):
    path, cams, motion = clip
    cal = cc.calibrate_clip(path, {0: (cams[0], None), N_A - 1: (cams[N_A - 1], None)}, policy=POL, motion=motion, masks=masks)
    t = cal.table
    assert len(t) == 22 and t["ok"].iloc[:N_A].all()
    for f in range(N_A):
        assert _perr(cal.H(f), cams[f]) < 0.6, f  # metres
    assert (t["source"].iloc[[0, N_A - 1]] == cc.KEYFRAME).all() and (t["gap"].iloc[[0, N_A - 1]] == 0).all()
    # nearest keyframe wins
    assert t["key_frame"].iloc[2] == 0 and t["key_frame"].iloc[13] == N_A - 1
    # the second shot has no keyframe: no extrapolation across the cut
    after = t.iloc[N_A:]
    assert not after["ok"].any() and (after["source"] == cc.NONE).all() and after["H"].map(lambda h: np.isnan(h).all()).all()
    assert cal.H(N_A + 2) is None and cal.H_any(N_A + 2) is None
    assert t["shot_id"].iloc[0] == 0 and t["shot_id"].iloc[N_A] == 1


def test_single_keyframe_carries_the_whole_shot(clip, masks):
    path, cams, motion = clip
    cal = cc.calibrate_clip(path, {7: cams[7]}, policy=POL, motion=motion, masks=masks)  # a bare H is accepted
    assert cal.table["ok"].iloc[:N_A].all()
    assert max(_perr(cal.H(f), cams[f]) for f in range(N_A)) < 0.8
    assert cal.table["gap"].iloc[0] == 7 and cal.table["key_frame"].iloc[0] == 7
    assert cal.table["confidence"].iloc[7] == 1.0 and cal.table["confidence"].iloc[0] < 1.0


def test_a_wrong_calibration_is_not_ok(clip, masks):
    path, cams, motion = clip
    bad = geo.orient_homography(cams[4] @ np.array([[1, 0, 0.0], [0, 1, 0.0], [0, 0, 1.0]]) @ np.array([[1, 0, 0.08], [0, 1, 0.06], [0, 0, 1.0]]), [[0.5, 0.5]])
    pol = cc.Policy(check_every=3, work_width=W, refine=False)
    cal = cc.calibrate_clip(path, {4: (bad, None)}, policy=pol, motion=motion, masks=masks)
    t = cal.table.iloc[:N_A]
    assert t["ok"].iloc[4]  # the keyframe is trusted by definition
    assert not t["ok"].iloc[[0, 1, 2, 8, 9, 12]].any()  # the lines are nowhere near the model lines
    assert t["confidence"][~t["ok"]].max() <= 0.2
    good = cc.calibrate_clip(path, {4: cams[4]}, policy=pol, motion=motion, masks=masks).table.iloc[:N_A]
    assert np.nanmean(t["line_cost"]) > 3 * np.nanmean(good["line_cost"])


def test_trigger_refinement_is_gated_and_keeps_the_error_small(clip, masks):
    path, cams, motion = clip
    pol = cc.Policy(check_every=3, work_width=W, max_anchor_gap=4, refine_cooldown=3)  # re-anchor often
    cal = cc.calibrate_clip(path, {0: cams[0]}, policy=pol, motion=motion, masks=masks)
    assert cal.stats["n_refine"] > 0
    assert cal.stats["n_accepted"] + cal.stats["n_rejected"] == cal.stats["n_refine"]
    assert max(_perr(cal.H(f), cams[f]) for f in range(N_A) if cal.H(f) is not None) < 1.0
    refined = cal.table[cal.table["source"] == cc.REFINED]
    assert (refined["anchor_frame"] == refined["frame"]).all()


def test_no_refine_policy_is_pure_propagation(clip, masks):
    path, cams, motion = clip
    cal = cc.calibrate_clip(path, {0: cams[0]}, policy=cc.Policy(check_every=3, work_width=W, refine=False), motion=motion, masks=masks)
    assert set(cal.table["source"]) <= {cc.KEYFRAME, cc.PROPAGATED, cc.NONE} and cal.stats["n_refine"] == 0


def test_keyframe_outside_clip_or_none_is_an_error(clip, masks):
    path, cams, motion = clip
    with pytest.raises(ValueError):
        cc.calibrate_clip(path, {99: cams[0]}, policy=POL, motion=motion, masks=masks)
    with pytest.raises(ValueError):
        cc.calibrate_clip(path, {}, policy=POL, motion=motion, masks=masks)


def test_distortion_is_carried_and_records_round_trip(clip, masks, tmp_path):
    path, cams, motion = clip
    dist = geo.make_distortion(0.02, W / H)
    cal = cc.calibrate_clip(path, {0: (cams[0], dist)}, policy=POL, motion=motion, masks=masks)
    assert cal.distortion(5)["k1"] == pytest.approx(0.02)
    rec = cal.records()
    assert len(rec) == 22 and rec[5]["H"].shape == (3, 3) and rec[5]["distortion"]["k1"] == pytest.approx(0.02)
    assert rec[20]["H"] is None and rec[20]["ok"] is False
    assert {"frame", "H", "distortion", "source", "anchor_frame", "gap", "line_cost", "confidence", "shot_id", "ok"} <= set(rec[0])
    cal.save(tmp_path / "cal.parquet")
    back = cc.ClipCalibration.load(tmp_path / "cal.parquet", W / H)
    assert np.allclose(back.H(5), cal.H(5)) and back.H(20) is None and list(back.table["source"]) == list(cal.table["source"])


def test_mask_store_round_trip(masks, tmp_path):
    masks.save(tmp_path / "m.npz")
    back = cc.MaskStore.load(tmp_path / "m.npz")
    assert back.frames == masks.frames and back.work_width == W
    a, b = masks.get(3), back.get(3)
    assert np.array_equal(a.lines, b.lines) and np.array_equal(a.grass, b.grass) and a.lines.shape == (H, W)
    assert masks.get(999) is None


def test_line_cost_is_low_for_the_true_h_and_high_for_a_shifted_one(clip, masks):
    _, cams, _ = clip
    s = model_samples(PitchModel())
    m = masks.get(3)
    good = cc.line_cost(cams[3], m, s)
    bad = cc.line_cost(geo.orient_homography(cams[3] @ np.array([[1, 0, 0.05], [0, 1, 0.04], [0, 0, 1.0]]), [[0.5, 0.5]]), m, s)
    assert good < 3.0 < bad
