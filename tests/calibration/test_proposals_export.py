"""Proposals exporter: projection round trip through the page's own refit, out-of-image drop, sampling, file format.
Synthetic combined rows only (no real data)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from footy_track.calibration import geometry as geo
from footy_track.calibration import proposals_export as pe
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.proposals import read_proposals, update_proposals

K1 = 0.0147


def _camera_h(pan_deg: float = 90.0, tilt_deg: float = 14.0, f: float = 4000.0) -> np.ndarray:
    """H: undistorted normalised image (x/1920, y/1080) -> pitch metres, camera at the Emirates centre rig."""
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


def _unit(H):
    H = np.asarray(H, float)
    H = H / np.linalg.norm(H)
    return -H if H.flat[np.argmax(np.abs(H))] < 0 else H


def _rows(n_seg=4, per_seg=240, version_id="ptz_v3+deadbeef", accepted=None) -> pe.FrameRows:
    n = n_seg * per_seg
    seg = np.repeat(np.arange(n_seg), per_seg)
    local = np.tile(np.arange(per_seg), n_seg)
    pan = 86.0 + 6.0 * (local / per_seg) + seg
    H = np.stack([_camera_h(p, 14.0, 3500.0 + 20.0 * s) for p, s in zip(pan, seg, strict=True)])
    acc = np.ones(n, bool) if accepted is None else accepted
    src = np.where(seg == 3, "recovered", "main").astype(object)
    return pe.FrameRows(
        match="arsenal_mancity", version="ptz_v3", version_id=version_id, frame=1000 + np.arange(n), segment=seg,
        local_frame=local, calibrated=np.ones(n, bool), accepted=acc, evidence_f1=np.full(n, 0.8, np.float32),
        source=src, pan=pan, logf=np.log(3500.0 + 20.0 * seg), H=H, k1=np.full(n, K1),
        segment_files={s: f"arsenal_mancity_20250925_seg{s:03d}.mp4" for s in range(n_seg)}, total_frames=2000 + n)


def test_round_trip_refit_recovers_H_and_k1():
    """What accept does: undistort the stored (distorted) points with the record's lens, refit H by DLT."""
    rows = _rows(1, 10)
    pick = pe.Pick(5, ["accepted"], 0.1)
    rec = pe.build_record(rows, 5, pick)
    assert rec["distortion"]["k1"] == K1 and rec["H_convention"] == "normalised_image_to_pitch_m"
    xy = np.array([c["pitch_xy"] for c in rec["correspondences"]])
    uv = np.array([c["image"] for c in rec["correspondences"]])
    assert len(uv) >= 6 and (uv >= 0).all() and (uv <= 1).all()
    H, _ = geo.fit_homography(geo.undistort_points(uv, rec["distortion"]), xy, method="dlt")
    g = np.stack(np.meshgrid(np.linspace(0.1, 0.9, 9), np.linspace(0.2, 0.9, 7)), -1).reshape(-1, 2)
    assert np.abs(geo.apply_homography(H, g)[0] - geo.apply_homography(rows.H[5], g)[0]).max() < 1e-5  # metres
    # k1 itself: the lens that makes the refit reproject the points exactly is the stored one
    def cost(k1):
        d = geo.make_distortion(k1)
        Hk, _ = geo.fit_homography(geo.undistort_points(uv, d), xy, method="dlt")
        return float(np.sum((geo.pitch_to_image(Hk, xy, d)[0] - uv) ** 2))
    res = minimize_scalar(cost, bounds=(0.0, 0.05), method="bounded", options={"xatol": 1e-9})
    assert abs(res.x - K1) < 1e-4


def test_accept_path_reproduces_stored_H():
    """The page's own fit (``labeller.calibration._fit``) on the exported points gives the stored H."""
    cal = pytest.importorskip("footy_track.labeller.calibration")
    rows = _rows(1, 10)
    rec = pe.build_record(rows, 3, pe.Pick(3, ["accepted"], 0.1))
    res = cal._fit({"correspondences": rec["correspondences"], "pitch_dims": rec["pitch_dims"], "method": "dlt",
                    "image_size": rec["image_size"], "distortion": rec["distortion"]})
    assert np.abs(_unit(res["H"]) - _unit(rows.H[3])).max() < 1e-6
    assert max(e for e in res["err_px"] if e is not None) < 1e-3


def test_out_of_image_and_behind_camera_points_are_dropped():
    pitch = PitchModel()
    kps = pitch.keypoints()
    H = _camera_h(90.0, 14.0, 4000.0)
    corr = pe.project_keypoints(H, K1, pitch)
    got = {c["pitch_id"] for c in corr}
    assert 0 < len(got) < len(kps)  # a zoomed view sees some keypoints, not all
    uv_all, ok = pe.mc.project_pitch_to_image(H, np.array(list(kps.values())), K1)
    for i, k in enumerate(kps):
        inside = ok[i] and 0 <= uv_all[i, 0] <= 1 and 0 <= uv_all[i, 1] <= 1
        assert (k in got) == bool(inside)
    assert all(0 <= c["image"][0] <= 1 and 0 <= c["image"][1] <= 1 for c in corr)


def test_distortion_is_applied_after_the_undistorted_projection():
    pitch = PitchModel()
    H = _camera_h(90.0, 14.0, 4000.0)
    corr = pe.project_keypoints(H, K1, pitch)
    c = max(corr, key=lambda c: np.hypot(c["image"][0] - 0.5, 0.5625 * (c["image"][1] - 0.5)))  # far from the centre
    und, _ = geo.apply_homography(np.linalg.inv(H), np.array([c["pitch_xy"]]))
    want = geo.distort_norm(und, K1)[0]
    assert np.allclose(c["image"], want, atol=1e-12) and not np.allclose(und[0], want, atol=1e-4)


def test_sampling_is_seeded_stable_and_records_inclusion_probabilities():
    rows = _rows()
    a, da = pe.sample_frames(rows, seed=1, target=60)
    b, _ = pe.sample_frames(rows, seed=1, target=60)
    c, _ = pe.sample_frames(rows, seed=2, target=60)
    ids = lambda ps: sorted(int(rows.frame[p.row]) for p in ps)  # noqa: E731
    assert ids(a) == ids(b) and ids(a) != ids(c)
    assert all(0 < p.pi <= 1 for p in a) and all(p.strata for p in a)
    n_long = int((np.bincount(rows.segment) >= pe.FLOOR_SEGMENT_FRAMES).sum())
    assert da["segment_floor"]["picked"] == n_long  # one per segment of >= FLOOR_SEGMENT_FRAMES frames
    rec_seg = {int(rows.segment[p.row]) for p in a if "recovered_segment" in p.strata}
    assert rec_seg == {3} and sum("recovered_segment" in p.strata for p in a) == pe.RECOVERED_MIN
    # per-segment cap of the budgeted strata
    for name in ("accepted", "pan_t1"):
        per_seg = np.bincount([int(rows.segment[p.row]) for p in a if name in p.strata], minlength=4)
        assert per_seg.max() <= pe.SEGMENT_CAP + 1


def test_same_frames_for_two_versions_where_pools_agree():
    rows = _rows()
    acc2 = rows.accepted.copy()
    acc2[rows.segment == 2] = False  # version B: segment 2 is no longer accepted
    rows2 = _rows(accepted=acc2, version_id="ptz_v3+cafef00d")
    a = {int(rows.frame[p.row]) for p in pe.sample_frames(rows, 0, 60)[0] if "segment_floor" in p.strata}
    b = {int(rows2.frame[p.row]) for p in pe.sample_frames(rows2, 0, 60)[0] if "segment_floor" in p.strata}
    assert a == b


def test_inclusion_probability_is_unbiased_in_expectation():
    """Mean over seeds of the inclusion indicator ~ pi (Horvitz-Thompson needs the stored probability to be right)."""
    rows = _rows(2, 120)
    n_rep = 300
    hits = np.zeros(len(rows.frame))
    pis = np.zeros(len(rows.frame))
    for s in range(n_rep):
        picks, _ = pe.sample_frames(rows, seed=s, target=40)
        for p in picks:
            hits[p.row] += 1
            pis[p.row] = p.pi  # pi does not depend on the seed
    assert pis.max() > 0
    sel = pis > 0
    assert abs(hits[sel].sum() / n_rep - pis[sel].sum()) < 0.1 * pis[sel].sum()  # sum of pi = expected size (union bound)


def test_export_writes_the_page_format_and_keeps_reviewed_frames(tmp_path):
    rows = _rows()
    s = pe.export_proposals(rows, tmp_path, seed=0, target=60)
    assert s["run_id"] == "ptz_v3@deadbeef" and s["n_written"] > 0
    clips = sorted(s["by_clip"])
    assert clips[0] == "arsenal_mancity_seg000"
    doc = read_proposals(tmp_path, clips[0])
    run = doc["runs"]["ptz_v3@deadbeef"]
    assert run["source"] == "ptz_v3" and run["params"]["match"] == "arsenal_mancity"
    k, rec = next(iter(run["frames"].items()))
    assert rec["frame_index"] == int(k) and rec["status"] == "proposed" and rec["sampling"]["inclusion_prob"] > 0
    assert len(rec["inliers"]) == len(rec["correspondences"]) and rec["image_size"] == [1920, 1080]
    # a reviewer acted on one frame; a re-run must not reset it
    update_proposals(tmp_path, clips[0], lambda d: d["runs"]["ptz_v3@deadbeef"]["frames"][k].update(status="rejected"))
    pe.export_proposals(rows, tmp_path, seed=0, target=60)
    assert read_proposals(tmp_path, clips[0])["runs"]["ptz_v3@deadbeef"]["frames"][k]["status"] == "rejected"
