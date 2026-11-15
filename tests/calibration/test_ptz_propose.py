"""PTZ rig proposals (V2): robust rig fit, point consensus, decision rule, records and the additive run writer."""

from __future__ import annotations

import numpy as np

from footy_track.calibration import geometry as geo
from footy_track.calibration import ptz
from footy_track.calibration import ptz_propose as pp
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.proposals import read_proposals, update_proposals

C_TRUE = np.array([0.0, -50.0, 7.5])
RIG = ptz.Rig(C_TRUE, 0.0, 0.02, (1.0, 3.0, 1.0))


def _cam(pan_deg=88.0, tilt_deg=9.0, f=3300.0, C=C_TRUE, k1=0.02):
    return ptz.Camera(np.deg2rad(pan_deg), np.deg2rad(tilt_deg), float(np.log(f)), 0.0, np.asarray(C, float), k1)


def _labels(cam, margin=0.03):
    ids, xy = PitchModel().keypoint_array()
    uv, ok = cam.pitch_to_image(xy)
    ok &= np.all((uv > margin) & (uv < 1 - margin), axis=1)
    return [i for i, o in zip(ids, ok, strict=True) if o], xy[ok], uv[ok]


def _frame(cam, key, rng, noise_px=1.0, n=6):
    ids, xy, uv = _labels(cam)
    sel = rng.choice(len(ids), min(n, len(ids)), replace=False)
    return ptz.RigFrame(uv[sel] + rng.normal(0, noise_px, (len(sel), 2)) / ptz.CANON, xy[sel], key, key)


# ---- frame selection --------------------------------------------------------------------------------------------


def test_evenly_spaced_segments_and_avoid():
    f = pp.evenly_spaced(300, 3)
    assert f == [49, 149, 249]
    g = pp.evenly_spaced(300, 3, avoid=[150, 52])
    assert len(g) == 3 and all(abs(a - b) >= 8 for a in g for b in (150, 52)) and g == sorted(g)
    assert all(0 <= x < 300 for x in g)
    assert pp.evenly_spaced(0, 3) == [] and pp.evenly_spaced(10, 0) == []


# ---- point consensus --------------------------------------------------------------------------------------------


def test_ransac_points_ignores_wrong_labels():
    rng = np.random.default_rng(1)
    cam = _cam(pan_deg=110.0, f=2200.0)  # wide, left of centre: many keypoints
    ids, xy, uv = _labels(cam)
    assert len(ids) >= 8
    img = uv[:8].copy()
    img[2] = img[5]  # two wrong labels: the image position of another keypoint
    img[6] = [0.9, 0.2]
    got = pp.ransac_points(img, xy[:8], RIG)
    assert got is not None
    est, inl = got
    assert inl.sum() >= 5 and not inl[2] and not inl[6]
    gx, gy = np.meshgrid(np.linspace(-25, 25, 6), np.linspace(-15, 15, 5))
    g = np.column_stack([gx.ravel(), gy.ravel()])
    p, ok = cam.pitch_to_image(g)
    vis = ok & np.all((p > 0.05) & (p < 0.95), axis=1)
    back, ok2 = est.image_to_pitch(p[vis])
    assert ok2.all() and np.median(np.linalg.norm(back - g[vis], axis=1)) < 0.6
    assert pp.ransac_points(img[:1], xy[:1], RIG) is None
    _ = rng


# ---- robust rig fit ---------------------------------------------------------------------------------------------


def test_fit_rig_robust_excludes_other_camera_and_bad_point():
    rng = np.random.default_rng(2)
    frames = [_frame(_cam(pan_deg=p, tilt_deg=t, f=f), f"a{i}", rng)
              for i, (p, t, f) in enumerate([(70, 8, 2500), (80, 9, 3000), (90, 10, 3500), (100, 8, 4000),
                                             (110, 9, 3200), (85, 12, 2200), (95, 7, 3800)])]
    other = _frame(_cam(pan_deg=88, C=[-36.0, -50.0, 7.5]), "other", rng)  # taken from a different position
    bad = _frame(_cam(pan_deg=92, tilt_deg=9, f=3100), "bad", rng, n=6)
    bad.img[0] = [0.5, 0.9]  # one gross label error
    rf = pp.fit_rig_robust([*frames, other, bad], C0=(0.0, -49.0, 7.0))
    assert [e["key"] for e in rf.excluded] == ["other"]
    assert [(d["key"], d["index"]) for d in rf.dropped_points] == [("bad", 0)]
    assert np.linalg.norm(rf.rig.C - C_TRUE) < 1.5
    assert abs(rf.rig.k1 - 0.02) < 0.01


# ---- decision rule ----------------------------------------------------------------------------------------------


def _cand(cam, n_in, support, n_lab=5, rms=3.0, spread=8.0):
    return pp.PointsCand(cam, n_in, n_lab, np.ones(n_lab, bool), support, True, rms, spread)


def test_pick_camera_rules():
    a, far = _cam(), _cam(pan_deg=60.0, f=6000.0)
    # >= 3 consensus points with line support: points win
    cam, mode, why, _ = pp.pick_camera(_cand(a, 4, 0.4), None, None, "low support", [])
    assert cam is a and mode == "4 LLM pts + lines" and why is None
    # two labels not confirmed by the lines and no lines answer: abstain with both reasons
    cam, mode, why, notes = pp.pick_camera(_cand(a, 2, 0.2), None, None, "near-tie", [far])
    assert cam is None and mode == "abstain" and "near-tie" in why and "2 LLM pts not confirmed by lines" in notes
    # ... confirmed by a lines hypothesis: accepted
    cam, mode, _, _ = pp.pick_camera(_cand(a, 2, 0.2), None, None, "near-tie", [_cam(pan_deg=88.2)])
    assert cam is a and mode == "2 LLM pts + lines"
    # lines only
    cam, mode, _, _ = pp.pick_camera(None, far, 0.5, None, [far])
    assert cam is far and mode == "lines-only"
    # disagreement: the better-supported one wins, ties go to the points
    cam, mode, _, notes = pp.pick_camera(_cand(a, 3, 0.3, spread=0.2), far, 0.6, None, [far])
    assert cam is far and mode == "lines-only" and notes
    cam, mode, _, notes = pp.pick_camera(_cand(a, 3, 0.3, spread=0.2), far, 0.35, None, [far])
    assert cam is a and "lines-only search disagreed" in notes
    # low support, few / loose points: not used
    cam, mode, why, _ = pp.pick_camera(_cand(a, 3, 0.05, spread=0.2), None, None, "low support", [])
    assert cam is None and "3/5 consistent" in why
    cam, _, why, _ = pp.pick_camera(_cand(a, 5, 0.05, rms=9.0), None, None, "low support", [])
    assert cam is None
    # many tight consensus points are accepted whatever the line support, and beat a better lines-only solve
    cam, mode, _, notes = pp.pick_camera(_cand(a, 5, -1.0), None, None, "low support", [])
    assert cam is a and mode == "5 LLM pts + lines" and "accepted on 5 consistent points" in notes[0]
    cam, mode, _, notes = pp.pick_camera(_cand(a, 4, -1.0), far, 0.9, None, [far])
    assert cam is a and "lines-only search disagreed" in notes
    # exactly 3 points need a tighter fit; collinear points never count as strong
    assert pp.pick_camera(_cand(a, 3, -1.0, rms=3.0), None, None, "x", [])[0] is a
    assert pp.pick_camera(_cand(a, 3, -1.0, rms=5.0), None, None, "x", [])[0] is None
    assert pp.pick_camera(_cand(a, 5, -1.0, spread=0.2), None, None, "x", [])[0] is None
    assert pp.pick_camera(_cand(a, 5, 0.3, spread=0.2), None, None, "x", [])[0] is a  # but lines can still confirm


def test_spread_collinear_vs_spread_out():
    assert pp._spread(np.array([[0.0, -9.0], [0.0, 9.0], [0.0, 34.0]])) < 1e-6
    assert pp._spread(np.array([[0.0, 9.0], [9.0, 0.0], [-36.0, 7.0], [-36.0, -7.0]])) > 3.0
    assert pp._spread(np.zeros((2, 2))) == 0.0


# ---- records ----------------------------------------------------------------------------------------------------


def test_proposal_record_h_roundtrip_and_failed_record():
    cam = _cam(pan_deg=110.0, f=2200.0)
    pitch = PitchModel()
    ids, xy, uv = _labels(cam)
    llm = [{"image": uv[i].tolist(), "pitch_id": ids[i], "pitch_xy": xy[i].tolist(), "confidence": 0.9}
           for i in range(4)]
    out = pp.Outcome(cam, "4 LLM pts + lines", None, 0.42, 0.2, 4, 5, np.array([True, True, True, True, False]), 0.9)
    llm5 = [*llm, {**llm[0], "pitch_id": "X"}]
    rec = pp.proposal_record(10, (1920, 1080), pitch, out, llm5)
    assert rec["status"] == "proposed" and abs(rec["score"] - 0.42) < 1e-9 and rec["pitch_dims"] == [105.0, 68.0]
    assert rec["H_convention"] == "normalised_image_to_pitch_m" and rec["distortion"]["k1"] == cam.k1
    assert "PTZ rig: 4 LLM pts + lines" in rec["view"]["notes"]
    real = [c for c in rec["correspondences"] if not c.get("synthetic")]
    synth = [c for c in rec["correspondences"] if c.get("synthetic")]
    assert len(real) == 4 and 4 <= len(synth) <= 10 and len(rec["inliers"]) == len(rec["correspondences"])
    assert not ({c["pitch_id"] for c in real} & {c["pitch_id"] for c in synth})
    # H + distortion in the record map the stored image points to their pitch metres
    H = np.array(rec["H"])
    pts = np.array([c["image"] for c in rec["correspondences"]])
    want = np.array([c["pitch_xy"] for c in rec["correspondences"]])
    got, w = geo.image_to_pitch(H, pts, rec["distortion"])
    assert np.abs(got - want).max() < 1e-3
    # a point projected through the stored H and distortion lands on the camera's projection
    back, _ = geo.pitch_to_image(H, np.array([[10.0, 5.0]]), rec["distortion"])
    assert np.allclose(back, cam.pitch_to_image(np.array([[10.0, 5.0]]))[0], atol=1e-6)
    fo = pp.Outcome(None, "abstain", "low support", 0.05, None, 0, 2, None, 0.8)
    fr = pp.failed_record(11, (1920, 1080), pitch, fo, llm[:2])
    assert fr["status"] == "failed" and fr["H"] is None and fr["failure"].startswith("PTZ rig abstained: low support")
    assert len(fr["correspondences"]) == 2 and "abstained" in fr["view"]["notes"]


def test_write_run_is_additive(tmp_path):
    update_proposals(tmp_path, "arsenal_bournemouth_1st_seg010",
                     lambda d: d["runs"].setdefault("old", {"source": "claude", "created": "x", "params": {},
                                                           "frames": {"3": {"status": "accepted"}}}))
    pitch = PitchModel()
    out = pp.Outcome(None, "abstain", "near-tie", 0.1, 0.0, 0, 0, None, 0.9)
    rec = pp.failed_record(7, (1920, 1080), pitch, out, [])
    pp.write_run(tmp_path, "new", "ptz-rig", {"k": 1}, {"bournemouth_1st_seg010": {7: rec}})
    pp.write_run(tmp_path, "new", "ptz-rig", {"k": 2}, {"bournemouth_1st_seg010": {9: rec}})
    doc = read_proposals(tmp_path, "arsenal_bournemouth_1st_seg010")
    assert set(doc["runs"]) == {"old", "new"}
    assert doc["runs"]["old"]["frames"] == {"3": {"status": "accepted"}}
    assert set(doc["runs"]["new"]["frames"]) == {"7", "9"} and doc["runs"]["new"]["params"] == {"k": 2}
    assert doc["runs"]["new"]["source"] == "ptz-rig"
