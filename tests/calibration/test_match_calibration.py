"""Whole-match calibration files: combine (best of main / recovered per segment), atomic + versioned writes,
the loader and the pitch <-> image projection. Synthetic PTZ outputs only (no real data, no ffprobe)."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from footy_track.calibration import geometry as geo
from footy_track.calibration import match_calibration as mcal
from footy_track.calibration.pitch_model import PitchModel
from footy_track.scripts import calibration_match_combine as script

K1 = 0.0147


def _camera_h(pan_deg: float = 90.0, tilt_deg: float = 14.0, f: float = 4000.0) -> np.ndarray:
    """H: undistorted normalised image (x/1920, y/1080) -> pitch metres, camera at the Emirates centre rig
    (the same construction as the ptz ``Camera.homography``: inverse of K [r1 r2 t])."""
    C = np.array([0.1, -66.6, 15.7])
    pan, tilt = np.deg2rad(pan_deg), np.deg2rad(tilt_deg)
    fwd = np.array([np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), -np.sin(tilt)])
    right = np.array([np.sin(pan), -np.cos(pan), 0.0])
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    t = -R @ C
    K = np.array([[f, 0, 960.0], [0, f, 540.0], [0, 0, 1.0]])
    G = K @ np.stack([R[:, 0], R[:, 1], t], axis=1)
    return np.linalg.inv(G) @ np.diag([1920.0, 1080.0, 1.0])


def _write_run(
    d, n, *, accepted_ranges, uncal=(), ev_every=5, ev=0.7, accepted_frac=None, sha="abc1234567890", variant=None
):
    d.mkdir(parents=True)
    H = np.stack([_camera_h(90.0 + 0.05 * i) for i in range(n)])
    logf = np.full(n, np.log(4000.0))
    pan, tilt = np.full(n, np.pi / 2), np.full(n, 0.24)
    for i in uncal:
        H[i], logf[i], pan[i], tilt[i] = np.nan, np.nan, np.nan, np.nan
    np.savez(d / "calib_full.npz", frame=np.arange(n), pan=pan, tilt=tilt, logf=logf, H=H,
             rig=np.array([0.1, -66.6, 15.7, -0.002, K1]))
    n_acc = sum(b - a + 1 for a, b in accepted_ranges)
    (d / "line_f1_full.json").write_text(json.dumps({
        "threshold": 0.5,
        "evidence_f1": {str(i): ev + 0.001 * i for i in range(0, n, ev_every)},
        "accepted_ranges": [list(r) for r in accepted_ranges],
        "accepted_frac_of_frames": n_acc / n,
    }))
    meta = {"code_sha": sha, "accepted_frac": n_acc / n if accepted_frac is None else accepted_frac}
    if variant:
        meta["rig_variant"] = variant
    (d / "meta.json").write_text(json.dumps(meta))


@pytest.fixture
def world(tmp_path):
    """data dir with a 4-segment match: seg0 good, seg1 weak with a better rerun, seg2 weak with a worse rerun,
    seg3 without any result."""
    data, temporal = tmp_path / "data", tmp_path / "data" / "scratch" / "temporal"
    segs = [
        {"index": 0, "file": "m_seg000.mp4", "start_frame": 100, "end_frame": 119, "frame_count": 20},
        {"index": 1, "file": "m_seg001.mp4", "start_frame": 500, "end_frame": 529, "frame_count": 30},
        {"index": 2, "file": "m_seg002.mp4", "start_frame": 900, "end_frame": 914, "frame_count": 15},
        {"index": 3, "file": "m_seg003.mp4", "start_frame": 1000, "end_frame": 1009, "frame_count": 10},
    ]
    sd = data / "m" / "split_video_broadcast_frames"
    sd.mkdir(parents=True)
    (sd / "segments.json").write_text(json.dumps({"fps": 25.0, "total_frames": 2000, "segments": segs}))
    _write_run(temporal / "m_seg000@rigx", 20, accepted_ranges=[(0, 14)], uncal=(19,))
    _write_run(temporal / "m_seg001@rigx", 30, accepted_ranges=[(0, 4)], sha="main000")
    _write_run(temporal / "rerun_weak" / "m_seg001@rigx", 30, accepted_ranges=[(0, 24)], sha="recov00",
               variant="x+32.7")
    _write_run(temporal / "m_seg002@rigx", 15, accepted_ranges=[(0, 9)])
    _write_run(temporal / "rerun_weak" / "m_seg002@rigx", 15, accepted_ranges=[(0, 4)])
    return {"data": data, "temporal": temporal, "out": tmp_path / "cal"}


def _combine(world, **kw):
    return mcal.combine_match("m", rig_stem="rigx", data_dir=world["data"], temporal=world["temporal"],
                              out_root=world["out"], frame_map=mcal._index_frame_map, **kw)


def test_best_result_per_segment_and_totals(world):
    m = _combine(world)
    segs = {s["segment"]: s for s in m["segments"]}
    assert segs[0]["source"] == "main" and segs[0]["accepted_frac"] == pytest.approx(15 / 20, abs=1e-5)
    assert segs[0]["calibrated_frac"] == pytest.approx(19 / 20, abs=1e-5)
    assert segs[1]["source"] == "recovered" and segs[1]["accepted_frac"] == pytest.approx(25 / 30, abs=1e-5)
    assert segs[1]["main_accepted_frac"] == pytest.approx(5 / 30, abs=1e-5) and segs[1]["code_sha"] == "recov00"
    assert segs[1]["rig_variant"] == "x+32.7"
    assert segs[2]["source"] == "main"  # the rerun accepts fewer frames: not used
    assert segs[2]["rerun_accepted_frac"] == pytest.approx(5 / 15, abs=1e-5)
    assert segs[3]["source"] == "none" and segs[3]["n_frames"] == 0
    t = m["totals"]
    assert t["segments"] == 4 and t["segments_recovered"] == 1 and t["segments_missing"] == 1
    assert t["frames"] == 20 + 30 + 15 and t["frames_accepted"] == 15 + 25 + 10
    assert t["frames_calibrated"] == 19 + 30 + 15
    assert t["accepted_frac"] == pytest.approx(50 / 65, abs=1e-5)
    assert m["version"] == "ptz_v3" and m["version_id"].startswith("ptz_v3+")
    assert m["code_shas"]["recovered"] == ["recov00"] and "abc1234567890"[:12] in m["code_shas"]["main"]
    assert m["status"] == "written"


def test_rows_global_frames_flags_and_nearest_evidence(world):
    _combine(world)
    mc = mcal.load_match_calibration("m", root=world["out"])
    assert mc.n == 65
    # global frame = start_frame + local frame; segment 3 (no result) has no rows
    assert mc.locate(100) == (0, 0) and mc.locate(529) == (1, 29) and mc.locate(1000) is None
    assert mc.global_frame(1, 3) == 503
    assert mc.status(100) == "accepted"
    assert mc.status(115) == "calibrated"  # solved but not in accepted_ranges
    assert mc.status(119) == "uncalibrated"  # NaN solution
    assert mc.status(1000) == "missing" and mc.status(50) == "missing"
    # recovered segment: rerun flags and source
    i = mc.index_of(510)
    assert mc.source[i] == "recovered" and mc.accepted[i]
    assert mc.source[mc.index_of(100)] == "main"
    # nearest evidence frame: local 7 -> evidence at 5 (|2|) not 10 (|3|); local 8 -> 10
    assert mc.evidence_f1[mc.index_of(107)] == pytest.approx(0.7 + 0.005, abs=1e-6)
    assert mc.evidence_f1[mc.index_of(108)] == pytest.approx(0.7 + 0.010, abs=1e-6)
    assert mc.k1_for_frame(100) == pytest.approx(K1)
    assert mc.H_for_frame(119) is None and mc.H_for_frame(1000) is None
    assert mc.H_for_frame(115) is not None and mc.H_for_frame(115, accepted_only=True) is None
    assert mc.H_for_frame(100, accepted_only=True) is not None


def test_idempotent_atomic_and_versioned(world):
    m1 = _combine(world)
    out = world["out"] / "ptz_v3"
    assert sorted(p.name for p in out.iterdir()) == ["m.manifest.json", "m.parquet"]  # no tmp files left
    before = (out / "m.parquet").stat().st_mtime_ns
    m2 = _combine(world)
    assert m2["status"] == "unchanged" and m2["content_sha256"] == m1["content_sha256"]
    assert (out / "m.parquet").stat().st_mtime_ns == before  # not rewritten

    # change an input: refused without update, archived and replaced with it
    _write_run_over(world, "m_seg000@rigx", accepted_ranges=[(0, 18)])
    with pytest.raises(mcal.WouldOverwrite):
        _combine(world)
    assert json.loads((out / "m.manifest.json").read_text())["content_sha256"] == m1["content_sha256"]
    m3 = _combine(world, update=True)
    assert m3["status"] == "written" and m3["content_sha256"] != m1["content_sha256"]
    hist = sorted(p.name for p in (out / "_history").iterdir())
    assert hist == [f"m.{m1['content_sha256'][:8]}.manifest.json", f"m.{m1['content_sha256'][:8]}.parquet"]
    assert mcal.load_match_calibration("m", root=world["out"]).status(115) == "accepted"


def _write_run_over(world, name, **kw):
    d = world["temporal"] / name
    shutil.rmtree(d)
    _write_run(d, 20, uncal=(19,), **kw)


def test_dry_run_writes_nothing(world):
    m = _combine(world, dry_run=True)
    assert m["status"] == "dry_run" and not world["out"].exists()


def test_loader_errors(world, tmp_path):
    with pytest.raises(FileNotFoundError):
        mcal.load_match_calibration("nope", root=tmp_path)
    _combine(world)
    mf = world["out"] / "ptz_v3" / "m.manifest.json"
    doc = json.loads(mf.read_text())
    doc["totals"]["frames"] += 1
    mf.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="torn write"):
        mcal.load_match_calibration("m", root=world["out"])


def test_projection_matches_geometry_and_round_trips():
    H = _camera_h()
    pitch = PitchModel()
    pts = np.array([[0.0, 0.0], [10.0, 5.0], [-20.0, -10.0], [pitch.hl, pitch.hw]])
    xy, ok = mcal.project_pitch_to_image(H, pts, K1)
    assert ok.all() and ((xy > 0) & (xy < 1)).all(axis=1)[:2].all()
    # the lens is applied after H^-1: undistort the observed point and apply H
    und = geo.undistort_norm(xy, K1, aspect=1920 / 1080)
    back, _ = geo.apply_homography(H, und)
    assert np.allclose(back, pts, atol=1e-6)
    back2, _ = mcal.project_image_to_pitch(H, xy, K1)
    assert np.allclose(back2, pts, atol=1e-6)
    # the division model with k1 > 0 maps an undistorted radius to a larger observed radius
    plain, _ = mcal.project_pitch_to_image(H, pts, 0.0)
    assert (np.linalg.norm(xy[:3] - 0.5, axis=1) >= np.linalg.norm(plain[:3] - 0.5, axis=1) - 1e-12).all()
    assert not np.allclose(xy, plain)
    # consistent with how the editor / lens code projects (geometry.pitch_to_image with a make_distortion dict)
    ref, _ = geo.pitch_to_image(H, pts, geo.make_distortion(K1, geo.DEFAULT_ASPECT))
    assert np.allclose(xy, ref)


def test_projection_masks_points_behind_the_camera():
    H = _camera_h()
    xy, ok = mcal.project_pitch_to_image(H, np.array([[0.0, -200.0], [0.0, 0.0]]), K1)
    assert list(ok) == [False, True]


def test_match_calibration_pitch_to_image(world):
    _combine(world)
    mc = mcal.load_match_calibration("m", root=world["out"])
    pts = np.array([[0.0, 0.0], [5.0, 5.0]])
    xy, ok = mc.pitch_to_image(100, pts)
    assert ok.all()
    back, _ = mc.image_to_pitch(100, xy)
    assert np.allclose(back, pts, atol=1e-6)
    assert mc.pitch_to_image(119, pts) is None  # uncalibrated frame


def test_pts_frame_map_uses_index_when_counts_agree(tmp_path):
    fmap = mcal.pts_frame_map("m", tmp_path)  # the segment file does not exist: never opened
    g, how = fmap({"start_frame": 10, "frame_count": 4, "file": "x.mp4"}, 4, {"fps": 25.0})
    assert list(g) == [10, 11, 12, 13] and how == "index"
    # different count and no readable file: falls back to the index map
    g, how = fmap({"start_frame": 10, "frame_count": 6, "file": "x.mp4"}, 4, {"fps": 25.0})
    assert list(g) == [10, 11, 12, 13] and how == "index"


def test_cli_dry_run_and_refusal(world, capsys, monkeypatch):
    monkeypatch.setattr(mcal, "pts_frame_map", lambda match, data_dir=None: mcal._index_frame_map)
    args = ["--match", "m", "--rig", "rigx", "--data-dir", str(world["data"]), "--temporal", str(world["temporal"]),
            "--out-root", str(world["out"])]
    assert script.main(args + ["--dry-run"]) == 0
    assert "dry_run" in capsys.readouterr().out
    assert script.main(args) == 0
    assert script.main(args) == 0 and "unchanged" in capsys.readouterr().out
    _write_run_over(world, "m_seg000@rigx", accepted_ranges=[(0, 3)])
    assert script.main(args) == 2
    assert script.main(args + ["--update"]) == 0
