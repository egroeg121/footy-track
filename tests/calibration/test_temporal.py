"""Temporal calibration: inter-frame motion recovery on a synthetic pan/tilt/zoom camera.

A textured pitch raster is warped through known camera homographies (fixed camera
centre, so every pair of views is related by a known homography); the estimators must
recover them, chaining must compose them, cuts must be flagged and a keyframe
calibration must propagate.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from footy_track.calibration import geometry as geo
from footy_track.calibration import temporal as tp
from footy_track.calibration.pitch_model import PitchModel

W, H = 640, 360
PITCH = PitchModel()
PX_PER_M, MARGIN = 8.0, 12.0


def _cam(pan_deg: float, tilt_deg: float = 28.0, f: float = 700.0) -> np.ndarray:
    """Normalised image -> pitch metres for a camera fixed at (5, -48, 25) that pans/tilts/zooms."""
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    pan, tilt = np.deg2rad(pan_deg), np.deg2rad(tilt_deg)
    fwd = np.array([np.sin(pan) * np.cos(tilt), np.cos(pan) * np.cos(tilt), -np.sin(tilt)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    t = -R @ np.array([5.0, -48.0, 25.0])
    P = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return geo.orient_homography(np.linalg.inv(P) @ np.diag([W, H, 1.0]), [[0.5, 0.6]])


def _raster() -> np.ndarray:
    rng = np.random.default_rng(3)
    w, h = geo.pitch_raster_size(PITCH, PX_PER_M, MARGIN)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.0)
    noise *= 14.0 / noise.std()
    stripes = 10.0 * np.sign(np.sin(np.arange(w) / PX_PER_M / 5.0 * np.pi))[None, :]
    g = 140 + noise + stripes
    img = np.stack([0.30 * g, g, 0.35 * g], axis=2)  # BGR green
    T = geo.pitch_raster_transform(PITCH, PX_PER_M, MARGIN)
    for poly in PITCH.polylines().values():
        pts, _ = geo.apply_homography(T, poly)
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, (245, 245, 245), 3, cv2.LINE_AA)
    return np.clip(img, 0, 255).astype(np.uint8)


RASTER = _raster()


def _render(Hn: np.ndarray, size=(W, H)) -> np.ndarray:
    T = geo.pitch_raster_transform(PITCH, PX_PER_M, MARGIN)
    M = np.diag([size[0], size[1], 1.0]) @ np.linalg.inv(Hn) @ np.linalg.inv(T)
    out = cv2.warpPerspective(RASTER, M, size, flags=cv2.INTER_LINEAR, borderValue=(35, 35, 35))
    return out


def _true_motion(Ha: np.ndarray, Hb: np.ndarray) -> np.ndarray:
    """Normalised a -> b."""
    return np.linalg.inv(Hb) @ Ha


GRID = np.array([[x, y] for x in np.linspace(0.15, 0.85, 6) for y in np.linspace(0.4, 0.9, 5)])


def _transfer_px(Hest: np.ndarray, Htrue: np.ndarray, size=(W, H)) -> float:
    a, _ = geo.apply_homography(Hest, GRID)
    b, _ = geo.apply_homography(Htrue, GRID)
    return float(np.linalg.norm((a - b) * np.array(size), axis=1).max())


@pytest.mark.parametrize("method", ["lk", "ecc", "lk_ecc"])
def test_pair_recovers_known_motion(method):
    Ha, Hb = _cam(8.0), _cam(9.2, 28.6, 740.0)  # a pan, a tilt and a zoom step
    fa, fb = _render(Ha), _render(Hb)
    pm = tp.estimate_pair(fa, fb, method=method, scale=W)
    assert pm.ok, pm.reason
    assert _transfer_px(pm.H_norm, _true_motion(Ha, Hb)) < 1.5
    # H_px and H_norm describe the same map
    S = np.diag([W, H, 1.0])
    assert np.allclose(tp._unit(S @ pm.H_norm @ np.linalg.inv(S)), pm.H_px, atol=1e-6)


def test_pair_at_a_different_working_scale_is_rescaled_back_to_full_res():
    Ha, Hb = _cam(8.0), _cam(9.0)
    big = (1280, 720)
    fa, fb = _render(Ha, big), _render(Hb, big)
    pm = tp.estimate_pair(fa, fb, scale=320)
    assert pm.ok, pm.reason
    assert _transfer_px(pm.H_norm, _true_motion(Ha, Hb), big) < 3.0
    a, _ = geo.apply_homography(pm.H_px, np.array([[640.0, 360.0]]))
    b, _ = geo.apply_homography(pm.H_norm, np.array([[0.5, 0.5]]))
    assert np.allclose(a[0], b[0] * [1280, 720], atol=1e-6)  # px really are full-res px


def test_unrelated_frames_are_not_ok():
    fa = _render(_cam(8.0))
    other = np.random.default_rng(1).integers(0, 255, fa.shape).astype(np.uint8)
    pm = tp.estimate_pair(fa, other, mask_mode="all")
    assert not pm.ok and pm.reason


def test_masks_exclude_overlay_players_and_boxes():
    m = tp.overlay_mask((90, 160))
    assert not m[2, 2] and m[45, 80]
    bm = tp.boxes_mask((100, 200), np.array([[0.5, 0.5, 0.1, 0.2]]))
    assert not bm[60, 105] and bm[5, 5]
    img = _render(_cam(8.0))
    img[150:210, 300:330] = (200, 40, 40)  # a "player"
    gm = tp.grass_static_mask(img)
    assert not gm[180, 315] and gm[250, 100]


def test_a_moving_player_does_not_bias_the_motion():
    Ha, Hb = _cam(8.0), _cam(9.0)
    fa, fb = _render(Ha), _render(Hb)
    for f, x in ((fa, 200), (fb, 240)):  # big player blob moving with a different velocity
        f[170:260, x:x + 70] = (60, 60, 200)
    pm = tp.estimate_pair(fa, fb, scale=W)
    assert pm.ok and _transfer_px(pm.H_norm, _true_motion(Ha, Hb)) < 2.0


def test_load_person_boxes(tmp_path):
    p = tmp_path / "x.jsonl"
    rows = [
        {"frame_index": 3, "bbox": {"x": 0.1, "y": 0.2, "w": 0.05, "h": 0.1}, "tags": ["player"], "confidence": 0.9},
        {"frame_index": 3, "bbox": {"x": 0.5, "y": 0.5, "w": 0.01, "h": 0.01}, "tags": ["in_play_ball"], "confidence": 0.5},
        {"frame_index": 3, "bbox": {"x": 0.9, "y": 0.9, "w": 0.05, "h": 0.1}, "tags": ["player"], "confidence": 0.05},
        {"frame_index": 4, "bbox": {"x": 0.3, "y": 0.3, "w": 0.05, "h": 0.1}, "tags": ["crowd"], "confidence": 0.9},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    b = tp.load_person_boxes(p)
    assert set(b) == {3} and b[3].shape == (2, 4)


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory):
    """12 pan frames of shot A (0..11), then 6 frames of an unrelated scene (cut at 12)."""
    cams = [_cam(6.0 + 0.5 * i, 28.0 + 0.05 * i, 700.0 + 3 * i) for i in range(12)]
    path = tmp_path_factory.mktemp("vid") / "clip.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (W, H))
    for c in cams:
        vw.write(_render(c))
    rng = np.random.default_rng(7)
    base = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3)).astype(np.uint8), (0, 0), 3)
    base = np.clip(base.astype(np.float32) * 0.5 + 100, 0, 255).astype(np.uint8)
    base[..., 0] = 220  # very different colour distribution
    for i in range(6):
        vw.write(np.roll(base, 2 * i, axis=1))
    vw.release()
    return path, cams


def test_clip_motion_chains_flags_cuts_and_propagates(synthetic_clip, tmp_path):
    path, cams = synthetic_clip
    cm = tp.clip_motion(path, max_gap=4, scale=W, mask_mode="all")
    assert cm.n == 18 and cm.cuts == [12]
    assert cm.ok[1:12, 0].all() and not cm.ok[12].any()
    # direct pair and chain across several hops, both directions
    assert _transfer_px(cm.H(3, 4), _true_motion(cams[3], cams[4])) < 1.0
    assert _transfer_px(cm.H(0, 11), _true_motion(cams[0], cams[11])) < 3.0
    assert _transfer_px(cm.H(9, 2), _true_motion(cams[9], cams[2])) < 3.0
    assert cm.H(5, 14) is None  # across the cut
    # whole-clip trajectory to a reference frame
    traj = cm.to_reference(6)
    assert np.allclose(traj[6], np.eye(3))
    assert _transfer_px(traj[0], _true_motion(cams[0], cams[6])) < 3.0
    assert _transfer_px(traj[11], _true_motion(cams[11], cams[6])) < 3.0
    assert np.isnan(traj[14]).all()
    # a keyframe calibration carried to another frame
    Hp = tp.propagate(cams[2], cm, 2, 10)
    a, _ = geo.apply_homography(Hp, GRID)
    b, _ = geo.apply_homography(cams[10], GRID)
    assert np.linalg.norm(a - b, axis=1).max() < 0.5  # metres
    assert tp.propagate(cams[2], cm, 2, 15) is None
    # sidecar round trip
    cm.save(tmp_path / "m.npz")
    cm2 = tp.ClipMotion.load(tmp_path / "m.npz")
    assert cm2.cuts == cm.cuts and np.allclose(cm2.Hn, cm.Hn) and cm2.method == "lk"
    assert len(cm.records()) == 18 and cm.pair(4, 3).ok


def test_chain_prefers_long_hops_when_available(synthetic_clip):
    path, cams = synthetic_clip
    cm = tp.clip_motion(path, frames=(0, 12), max_gap=4, scale=W, mask_mode="all")
    # 8 frames apart = 2 hops of 4; make the gap-4 pairs unavailable and it must fall back to single steps
    direct = cm.H(0, 8)
    cm.ok[:, 1:] = False
    single = cm.H(0, 8)
    truth = _true_motion(cams[0], cams[8])
    assert _transfer_px(direct, truth) < 2.0 and _transfer_px(single, truth) < 3.0


def test_focal_and_rot_zoom_projection_round_trip():
    Ha, Hb = _cam(8.0, 28.0, 700.0), _cam(9.0, 28.4, 760.0)
    asp = W / H
    fa = tp.focal_from_homography(Ha, asp)
    assert fa == pytest.approx(700.0 / W, rel=0.02)
    M = _true_motion(Ha, Hb)
    Hc, fb, rms = tp.project_rot_zoom(M, fa, asp)
    assert rms < 0.05 and fb == pytest.approx(760.0 / W, rel=0.02)
    assert _transfer_px(Hc, M) < 0.1
    # a noisy, non-rotational perturbation (shear / perspective) is removed by the projection
    noisy = M @ np.array([[1, 0.002, 0], [0, 1, 0], [0.01, 0, 1.0]])
    Hc2, _, _ = tp.project_rot_zoom(noisy, fa, asp)
    assert _transfer_px(Hc2, M) < _transfer_px(noisy, M)


def test_hist_distance_separates_scenes():
    a, b = _render(_cam(8.0)), _render(_cam(8.4))
    other = np.full_like(a, 90)
    other[..., 0] = 230
    assert tp.hist_distance(tp._hist_thumb(a), tp._hist_thumb(b)) < tp.CUT_HIST_SOFT
    assert tp.hist_distance(tp._hist_thumb(a), tp._hist_thumb(other)) > tp.CUT_HIST_HARD


def test_refine_frame_recovers_a_perturbed_propagated_h():
    Ht = _cam(8.0)
    frame = _render(Ht, (1280, 720))
    Hbad = geo.orient_homography(Ht @ np.array([[1, 0, 0.012], [0, 1, -0.01], [0, 0, 1.0]]), [[0.5, 0.6]])
    r = tp.refine_frame(Hbad, frame, PITCH, work_width=960)
    assert r.applied, r.reason
    pts = np.array([[0.3, 0.7], [0.5, 0.6], [0.7, 0.8], [0.5, 0.9]])
    assert tp.pitch_grid_error(r.H, Ht, pts).max() < tp.pitch_grid_error(Hbad, Ht, pts).max()
