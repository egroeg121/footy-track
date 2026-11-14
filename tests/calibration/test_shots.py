"""Per-shot camera-type classifier on synthetic data: a pitch rendered by a panning camera (main) vs. scenes with no pitch."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from footy_track.calibration import shots as sh
from footy_track.calibration import temporal as tp
from footy_track.scripts import calibrate_clip as cc
from footy_track.scripts import shots_eval as se
from tests.calibration.test_temporal import H, W, _cam, _render

N_A, N_B, N_C = 24, 12, 12  # main pan, a close-up-like scene, a second main shot


def _closeup(i: int) -> np.ndarray:
    """No pitch: big blurred skin / shirt coloured blobs on a stand-coloured background (random texture, drifting)."""
    rng = np.random.default_rng(11)
    base = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3)).astype(np.uint8), (0, 0), 3)
    base = np.clip(base.astype(np.float32) * 0.5 + 100, 0, 255).astype(np.uint8)
    base[..., 0] = 220  # blue-ish: not grass
    return np.roll(base, 2 * i, axis=1)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("shots") / "clip.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (W, H))
    for i in range(N_A):
        vw.write(_render(_cam(6.0 + 0.5 * i, 28.0, 700.0 + 2 * i)))
    for i in range(N_B):
        vw.write(_closeup(i))
    for i in range(N_C):
        vw.write(_render(_cam(10.0 + 0.5 * i, 30.0, 760.0)))
    vw.release()
    motion = tp.clip_motion(path, max_gap=1, scale=W, mask_mode="all")
    return path, motion


def test_frame_features_pitch_vs_no_pitch():
    pitch = sh.frame_features(_render(_cam(8.0)), None, W)
    none = sh.frame_features(_closeup(0), None, W)
    assert pitch["grass_frac"] > 0.4 and pitch["grass_low"] > 0.9
    assert pitch["line_len"] > 0.3 and pitch["n_dirs"] >= 2
    assert none["grass_frac"] < 0.05 and none["line_len"] == 0.0
    assert np.isnan(pitch["person_h_med"]) and np.isnan(pitch["n_person"])  # no boxes given


def test_person_features_ignore_the_ball_and_handle_no_boxes():
    boxes = np.array([[0.1, 0.1, 0.02, 0.10], [0.5, 0.5, 0.03, 0.12], [0.9, 0.9, 0.01, 0.01]])  # last: ball-sized
    med, mx, n = sh.person_features(boxes)
    assert n == 2 and mx == pytest.approx(0.12) and med == pytest.approx(0.11)
    med, mx, n = sh.person_features(np.zeros((0, 4)))
    assert np.isnan(med) and n == 0.0
    assert all(np.isnan(v) for v in sh.person_features(None))


def test_sample_frames_cover_every_shot(clip):
    _, motion = clip
    assert motion.cuts == [N_A, N_A + N_B]
    fr = sh.sample_frames(motion, 100)  # coarser than every shot
    shot = motion.shot
    assert {int(shot[f - motion.start]) for f in fr} == {0, 1, 2}
    assert fr == sorted(set(fr))


def test_classify_shots_main_vs_other(clip):
    path, motion = clip
    shots = sh.classify_shots(path, motion, None, sample_every=4, model=None, work_width=W)
    assert [(s["start"], s["end"]) for s in shots] == [(0, N_A - 1), (N_A, N_A + N_B - 1), (N_A + N_B, N_A + N_B + N_C - 1)]
    assert [s["type"] for s in shots] == [sh.MAIN, sh.CLOSEUP, sh.MAIN]
    assert shots[0]["n_frames"] == N_A and shots[0]["p_main"] > 0.5 > shots[1]["p_main"]
    assert shots[0]["features"]["motion"] > 0 and shots[0]["features"]["log_len"] == pytest.approx(np.log2(N_A))
    # a keyframe candidate only for main shots, and inside its shot
    assert shots[1]["keyframe_candidate"] is None and np.isnan(shots[1]["keyframe_line_cost"])
    for s in (shots[0], shots[2]):
        assert s["start"] <= s["keyframe_candidate"] <= s["end"]
    assert sh.shot_of(shots, N_A + 1)["shot_id"] == 1 and sh.shot_of(shots, 10_000) is None


def test_features_cache_is_used(clip):
    path, motion = clip
    feats = sh.extract_frame_features(path, sh.sample_frames(motion, 4), None, W)
    a = sh.classify_shots(path, motion, None, sample_every=4, model=None, features=feats, work_width=W)
    b = sh.classify_shots(path, motion, None, sample_every=4, model=None, work_width=W)
    assert [s["type"] for s in a] == [s["type"] for s in b]
    assert a[0]["features"]["grass_frac"] == pytest.approx(b[0]["features"]["grass_frac"])


def test_keyframe_prefers_support_and_calibrated_line_cost():
    f = {10: {"line_len": 0.5, "n_dirs": 2.0}, 20: {"line_len": 1.5, "n_dirs": 4.0}, 30: {"line_len": 1.2, "n_dirs": 4.0}, 40: {"line_len": 0.0, "n_dirs": 0.0},
         50: {"line_len": 0.2, "n_dirs": 1.0}}
    assert sh.pick_keyframe(f) == (20, pytest.approx(float("nan"), nan_ok=True))
    cal = type("Cal", (), {"table": pd.DataFrame({"frame": [10, 20, 30, 50], "ok": [True, True, True, True], "line_cost": [4.0, 9.0, 5.0, 1.0]})})()
    # the best three by support are 20, 30, 10: the lowest line cost among them wins; 50 (cost 1.0) is not a candidate
    assert sh.pick_keyframe(f, cal) == (10, 4.0)
    cal.table.loc[cal.table["frame"] == 10, "ok"] = False
    assert sh.pick_keyframe(f, cal) == (30, 5.0)  # an uncalibrated candidate is skipped
    assert sh.pick_keyframe({40: {"line_len": 0.0, "n_dirs": 0.0}}) == (None, pytest.approx(float("nan"), nan_ok=True))


def test_decide_rule_and_closeup_split():
    wide = {"grass_frac": 0.7, "grass_low": 0.95, "line_frac": 2.0, "line_len": 1.0, "blob_h": 0.08}
    assert sh.rule_p_main(wide) == 1.0 and sh.decide(wide, sh.rule_p_main(wide))[0] == sh.MAIN
    close = dict(wide, grass_frac=0.2, grass_low=0.3, line_len=0.0, line_frac=0.0, blob_h=0.6)
    assert sh.rule_p_main(close) == 0.0 and sh.decide(close, 0.0) == (sh.CLOSEUP, 1.0)
    oblique = dict(wide, blob_h=0.1)  # plenty of pitch and small people, but the model says no: another camera, not a close-up
    assert sh.decide(oblique, 0.2) == (sh.OTHER, pytest.approx(0.8))


def test_logistic_fit_and_model_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    names = ("grass_frac", "line_len")  # frame features; the rest of the dict is ignored
    rows = []
    for i in range(80):
        y = i % 2
        rows.append({"f": {"grass_frac": 0.7 * y + 0.2 + 0.05 * rng.normal(), "line_len": float(y) + 0.1 * rng.normal()}, "y": y, "w": 1.0})
    rows[3]["f"]["line_len"] = float("nan")  # missing values are imputed
    m = se.fit_model(rows, names, l2=0.1)
    p = np.array([m.p_main(r["f"]) for r in rows])
    assert ((p >= 0.5) == np.array([r["y"] for r in rows], bool)).mean() > 0.95
    m.save(tmp_path / "m.json")
    m2 = sh.ShotModel.load(tmp_path / "m.json")
    assert m2.p_main(rows[1]["f"]) == pytest.approx(m.p_main(rows[1]["f"]))
    assert m2.p_main({"grass_frac": 0.9, "line_len": 1.0}) > 0.5 > m2.p_main({"grass_frac": 0.2, "line_len": 0.0})


def test_confusion_counts():
    c = se.confusion(np.array([1, 1, 0, 0, 1], bool), np.array([1, 0, 0, 1, 1], bool))
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 1)
    assert c["precision"] == pytest.approx(2 / 3) and c["recall"] == pytest.approx(2 / 3)


def test_labels_human_overrides_claude(tmp_path):
    gt = tmp_path
    (gt / "calibration_gt").mkdir()
    (gt / "calibration_gt" / "arsenal_mancity_seg010.json").write_text(json.dumps({"clip": "arsenal_mancity_seg010", "frames": {"5": {}}}))
    (gt / "calibration_proposals").mkdir()
    runs = {"r": {"frames": {"5": {"view": {"main_camera_wide": False}}, "9": {"view": {"main_camera_wide": False}, "status": "accepted"},
                              "12": {"view": {"main_camera_wide": True}}}}}
    (gt / "calibration_proposals" / "arsenal_mancity_seg010.json").write_text(json.dumps({"clip": "arsenal_mancity_seg010", "runs": runs}))
    (gt / "frame_type").mkdir()
    (gt / "frame_type" / "arsenal_mancity_seg010.json").write_text(json.dumps({"frames": {"12": {"type": "not_broadcast"}}}))
    lab = se.load_labels(gt)["arsenal_mancity_seg010"]
    assert lab[5]["main"] and lab[5]["tier"] == "human" and lab[5]["conflict"] == "claude_view"
    assert lab[9]["main"] and lab[9]["src"] == "accepted"
    assert not lab[12]["main"] and lab[12]["src"] == "frame_type" and lab[12]["conflict"] == "claude_view"
    assert se.match_of("mancity_seg010") == "arsenal_mancity"


def test_blob_features_tell_a_player_from_a_close_up():
    img = _render(_cam(8.0))
    small, big = img.copy(), img.copy()
    cv2.rectangle(small, (300, 200), (312, 232), (40, 40, 200), -1)  # a 32 px (~9 %) tall player
    cv2.rectangle(big, (200, 40), (430, 350), (60, 90, 200), -1)  # a face / shirt filling the frame
    fs, fb = sh.frame_features(small, None, W), sh.frame_features(big, None, W)
    assert 0.05 < fs["blob_h"] < 0.2 and fs["n_blob"] >= 1
    assert fb["blob_h"] > 0.25 or fb["grass_frac"] < 0.5  # the big one either shows as a huge blob or eats the pitch
    assert sh.decide(fb, 0.0)[0] == sh.CLOSEUP


def test_roboflow_items_dedupe_and_drop_contradictions(tmp_path):
    for ds, cls, name in (("roboflow_dataset_9", "Yes", "Arsenal-Aston-Villa_000001"), ("roboflow_dataset_10", "Yes", "Arsenal-Aston-Villa_000001"),
                          ("roboflow_dataset_10", "No", "arsenal_mancity_20250925_000005"), ("roboflow_dataset_9", "Yes", "arsenal_mancity_20250925_000005"),
                          ("roboflow_dataset_10", "No", "arsenal_norwich_000009")):
        d = tmp_path / ds / "train" / cls
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}_jpg.rf.abc.jpg").write_bytes(b"x")
    items = se.roboflow_items(tmp_path)
    assert [(i["key"], i["y"], i["match"]) for i in items] == [("Arsenal-Aston-Villa_000001", 1, "arsenal_astonvilla"), ("arsenal_norwich_000009", 0, "arsenal_norwich")]


def _shots(*spans):
    return [{"shot_id": i, "start": a, "end": b, "type": t} for i, (a, b, t) in enumerate(spans)]


def test_auto_keys_and_proposal_loader(tmp_path):
    good = np.eye(3).tolist()
    degenerate = [[0.0, 0.0, 0.0], [0.05, -0.9, 0.3], [0.0006, 0.0136, 0.0033]]
    fr = {
        "10": {"status": "accepted", "H": good},
        "11": {"status": "proposed", "H": good},
        "12": {"status": "rejected", "H": good},
        "13": {"status": "failed", "H": None},
        "14": {"status": "proposed", "H": degenerate},
        "60": {"status": "proposed", "H": good},  # in a close-up
        "30": {"status": "proposed", "H": good},  # also in GT
    }
    (tmp_path / "calibration_proposals").mkdir()
    (tmp_path / "calibration_proposals" / "arsenal_mancity_seg010.json").write_text(
        json.dumps({"clip": "arsenal_mancity_seg010", "runs": {"r1": {"frames": fr}, "r2": {"frames": {"11": {"status": "accepted", "H": good}}}}}))
    props = cc.load_proposal_keyframes("mancity_seg010", tmp_path)
    assert sorted(props) == [10, 11, 30, 60] and props[11]["status"] == "accepted" and props[11]["run"] == "r2"  # best status wins; bad / rejected dropped
    gt = {30: (2 * np.eye(3), None)}
    keys, rep = cc.auto_keys(_shots((0, 49, "main"), (50, 70, "closeup")), gt, props)
    assert sorted(keys) == [10, 11, 30] and rep["skipped_not_main"] == [60]
    assert rep["used"] == {10: "accepted", 11: "accepted", 30: "gt"} and np.allclose(keys[30][0], 2 * np.eye(3))  # GT beats the proposal
    assert cc.auto_keys(_shots((0, 20, "main")), {}, {99: props[10]})[1]["skipped_no_shot"] == [99]
