"""Calibration editor lens: default distortion per ground, GT round-trip, proposals rendered through it,
and the /snap route. Video IO is replaced by a synthetic frame; geometry is real."""

from __future__ import annotations

import numpy as np
import pytest

from footy_track.calibration import geometry as geo
from footy_track.calibration import lens as lens_mod
from footy_track.calibration.centre_fit import load_rig
from footy_track.calibration.pitch_model import PitchModel
from footy_track.calibration.prelabel.proposals import read_proposals, update_proposals
from footy_track.labeller import calibration as cal

from ..calibration.test_geometry import H, W, _camera_h
from ..calibration.test_lens import KP, _render_mask

EMIRATES = "arsenal_mancity_seg010"
UNKNOWN = "chelsea_somebody_seg001"


@pytest.fixture
def env(clips_dir, gt_marks_dir, monkeypatch):
    for c in (EMIRATES, UNKNOWN):
        (clips_dir / f"{c}.mp4").touch()
    frame = np.full((H, W, 3), 90, np.uint8)
    monkeypatch.setattr(cal, "_read_frame", lambda v, i: frame if i < 50 else None)
    monkeypatch.setattr(cal, "_video_meta", lambda v: {"frames": 50, "fps": 25.0, "width": W, "height": H})
    return gt_marks_dir


def _body(dist, **kw):
    pitch = PitchModel()
    obs, _ = geo.pitch_to_image(_camera_h(), np.array([pitch.keypoints()[k] for k in KP]), dist)
    corr = [{"image": p.tolist(), "pitch_id": k} for p, k in zip(obs, KP, strict=True)]
    return {"correspondences": corr, "pitch_dims": [105, 68], "image_size": [W, H], **kw}


def test_clip_meta_carries_the_grounds_default_lens(client, env):
    d = client.get(f"/calibration/api/clips/{EMIRATES}/meta").json()["default_distortion"]
    assert d["k1"] == pytest.approx(load_rig("emirates").k1) and d["aspect"] == pytest.approx(W / H)
    assert client.get(f"/calibration/api/clips/{UNKNOWN}/meta").json()["default_distortion"] is None


def test_saved_lens_round_trips_through_gt(client, env):
    dist = geo.make_distortion(0.0213, W / H)
    saved = client.put(f"/calibration/api/gt/{EMIRATES}/3", json=_body(dist, distortion=dist)).json()["record"]
    assert saved["distortion"]["k1"] == 0.0213 and "lens" not in saved
    got = client.get(f"/calibration/api/gt/{EMIRATES}/3").json()
    assert got["distortion"] == saved["distortion"] and "distortion_source" not in got   # as stored, not the default
    # H was fitted under that lens: the points reproduce it
    errs = geo.reprojection_errors(np.array(got["H"]), np.array([c["image"] for c in got["correspondences"]]),
                                   np.array([c["pitch_xy"] for c in got["correspondences"]]), (W, H), got["distortion"])
    assert np.nanmax(errs["px"]) < 0.05


def test_records_without_a_lens_load_exactly_as_stored(client, env):
    """The ground's k1 is only a suggestion: nothing is defaulted on load, on any ground."""
    for clip in (EMIRATES, UNKNOWN):
        saved = client.put(f"/calibration/api/gt/{clip}/5", json=_body(None)).json()["record"]
        assert "distortion" not in saved and "lens" not in saved
        got = client.get(f"/calibration/api/gt/{clip}/5").json()
        assert got == saved and "distortion" not in got and "distortion_source" not in got
    # an explicit lens-off save (the editor sends distortion: null) stores nothing extra either
    off = client.put(f"/calibration/api/gt/{EMIRATES}/6", json=_body(None, distortion=None)).json()["record"]
    assert "distortion" not in off and "lens" not in off


def _proposal(clip, idx, H_plain, corr):
    def _w(doc):
        doc["runs"]["r1"] = {"source": "claude", "created": "t", "params": {}, "frames": {str(idx): {
            "status": "proposed", "score": 0.5, "H": H_plain, "H_convention": "normalised_image_to_pitch_m",
            "pitch_dims": [105, 68], "image_size": [W, H], "correspondences": corr,
            "inliers": [True] * len(corr)}}}
    return _w


def test_proposal_without_lens_renders_and_accepts_with_the_stored_h(client, env):
    """No lens stored -> the card shows the stored H, straight lines, and Accept saves that same H, byte for byte.
    A line-refined H that the points alone would not give is kept as is (padded to reproduce it)."""
    fit = client.post("/calibration/api/fit", json=_body(None)).json()
    refined = (np.array(fit["H"]) * 1.0).tolist()
    refined[0][2] += 1e-4                                   # an H the points do not re-derive (line-refined)
    update_proposals(env, EMIRATES, _proposal(EMIRATES, 6, refined, fit["correspondences"]))
    rec = client.get(f"/calibration/api/proposals/{EMIRATES}/r1/6").json()
    assert rec["H"] == refined and "distortion" not in rec and "distortion_source" not in rec
    want = {n: [r.tolist() for r in runs] for n, runs in geo.project_pitch_model(np.array(refined), PitchModel()).items()}
    assert rec["overlay"] == want                            # drawn straight, from the stored H
    assert rec["default_distortion"]["k1"] == pytest.approx(load_rig("emirates").k1)   # offered, not applied
    stored = read_proposals(env, EMIRATES)["runs"]["r1"]["frames"]["6"]
    assert "distortion" not in stored and stored["H"] == refined
    acc = client.post(f"/calibration/api/proposals/{EMIRATES}/r1/6/accept", json={}).json()
    assert acc["ok"] and "distortion" not in acc["record"] and "lens" not in acc["record"]
    # as on main: the points are padded so the refit reproduces the shown H within REFIT_TOL_M (a perturbation this
    # small is within it, so no padding is even needed); the lens plays no part
    assert np.allclose(acc["record"]["H"], refined, atol=5e-4)


def test_accept_as_proposed_matches_main_when_the_points_rederive_h(client, env):
    fit = client.post("/calibration/api/fit", json=_body(None)).json()
    update_proposals(env, EMIRATES, _proposal(EMIRATES, 7, fit["H"], fit["correspondences"]))
    acc = client.post(f"/calibration/api/proposals/{EMIRATES}/r1/7/accept", json={}).json()
    assert acc["record"]["H"] == fit["H"] and "distortion" not in acc["record"]


def test_proposal_with_a_stored_lens_keeps_it(client, env):
    d = geo.make_distortion(0.02, W / H)
    fit = client.post("/calibration/api/fit", json=_body(d, distortion=d)).json()
    update_proposals(env, EMIRATES, lambda doc: doc["runs"].update({"r1": {"source": "claude", "created": "t", "params": {}, "frames": {"8": {
        "status": "proposed", "score": 0.5, "H": fit["H"], "pitch_dims": [105, 68], "image_size": [W, H],
        "distortion": d, "correspondences": fit["correspondences"], "inliers": [True] * len(fit["correspondences"])}}}}))
    rec = client.get(f"/calibration/api/proposals/{EMIRATES}/r1/8").json()
    assert rec["distortion"] == d and rec["H"] == fit["H"]
    acc = client.post(f"/calibration/api/proposals/{EMIRATES}/r1/8/accept", json={}).json()
    assert acc["record"]["distortion"] == d and acc["record"]["H"] == fit["H"]


def test_warp_and_clicks_go_through_the_same_lens(env):
    """A pitch point projected to the image (H^-1 then distort) and back (undistort then H) is the identity,
    and the warp raster samples the frame at exactly those image points."""
    dist = geo.make_distortion(0.0349, W / H)
    Ht = _camera_h()
    pts = np.array([[0.0, 0.0], [0.0, 34.0], [36.0, 20.16], [32.35, 0.0], [0.0, -9.15]])
    img, w = geo.pitch_to_image(Ht, pts, dist)
    assert (w > 0).all() and ((img > 0) & (img < 1)).all()
    back, _ = geo.image_to_pitch(Ht, img, dist)
    assert np.allclose(back, pts, atol=1e-6)
    # a coordinate-gradient frame: the warp pixel at a pitch point holds that point's image coordinates
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    frame = np.stack([gx % 256, gy % 256, np.zeros_like(gx)], axis=2).astype(np.uint8)
    pitch = PitchModel()
    ras = geo.warp_to_pitch(frame, Ht, pitch, 10.0, 3.0, dist)
    T = geo.pitch_raster_transform(pitch, 10.0, 3.0)
    for p, q in zip(pts, img, strict=True):
        u, v = (T @ np.array([p[0], p[1], 1.0]))[:2]
        if not (0 <= q[0] < 1 and 0 <= q[1] < 1):
            continue
        b, g, _ = ras[int(round(v)), int(round(u))]
        assert abs(int(b) - (q[0] * W) % 256) < 8 and abs(int(g) - (q[1] * H) % 256) < 8


def test_snap_route_bends_the_lines_and_refuses_bad_input(client, env, monkeypatch):
    k1_true = 0.03
    true = geo.make_distortion(k1_true, W / H)
    mask = _render_mask(_camera_h(), true)
    monkeypatch.setattr(lens_mod.Evidence, "from_frame", classmethod(lambda cls, bgr, with_far=True: cls.from_mask(mask)))
    cal._SNAP_CACHE.clear()
    body = {**_body(true), "clip": EMIRATES, "frame_index": 2}      # the clicks sit on the curved lines
    plain = {**body}
    r = client.post("/calibration/api/snap", json=plain)               # ... but no lens in the fit yet
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["accepted"], d["reason"]
    assert d["tried_suggested_k1"] is True and d["k1_before"] == 0   # lens off: also searched from the ground's k1
    assert d["k1_after"] == pytest.approx(k1_true, abs=0.006) and d["f1_after"] > d["f1_before"]
    assert d["distortion"]["k1"] == pytest.approx(d["k1_after"])
    assert d["max_click_shift_px"] <= 6.0 and d["edge_after"]["median_px"] < d["edge_before"]["median_px"]
    assert len(d["correspondences"]) == len(body["correspondences"])
    # the moved points + the returned lens refit to a fit that matches them
    fit = client.post("/calibration/api/fit", json={**body, "correspondences": d["correspondences"], "distortion": d["distortion"]}).json()
    assert max(fit["err_px"]) < 0.5
    # already on the lines: refused, the input comes back
    again = client.post("/calibration/api/snap", json={**body, "correspondences": d["correspondences"], "distortion": d["distortion"]}).json()
    assert not again["accepted"] and again["reason"] and again["correspondences"] == d["correspondences"]
    # k1-only keeps H fixed and starts from the lens in the request (none here); an unknown ground has no suggestion
    ko = client.post("/calibration/api/snap", json={**plain, "mode": "k1"}).json()
    assert ko["tried_suggested_k1"] is False
    un = client.post("/calibration/api/snap", json={**plain, "clip": UNKNOWN}).json()
    assert un["tried_suggested_k1"] is False and un["accepted"]
    # bad requests
    assert client.post("/calibration/api/snap", json={**body, "mode": "x"}).status_code == 400
    assert client.post("/calibration/api/snap", json={**body, "correspondences": body["correspondences"][:3]}).status_code == 400
    assert client.post("/calibration/api/snap", json={**body, "clip": "nope"}).status_code == 404
    assert client.post("/calibration/api/snap", json={**body, "frame_index": 99}).status_code == 404
