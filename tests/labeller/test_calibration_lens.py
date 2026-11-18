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


def test_lens_turned_off_is_remembered_and_old_records_get_the_default(client, env):
    # operator switched the lens off on a ground that has a default: saved with an explicit marker
    off = client.put(f"/calibration/api/gt/{EMIRATES}/4", json=_body(None, distortion=None)).json()["record"]
    assert "distortion" not in off and off["lens"] == "off"
    got = client.get(f"/calibration/api/gt/{EMIRATES}/4").json()
    assert "distortion" not in got and "distortion_source" not in got
    # a record from before the lens existed (no key at all) is rendered with the ground's default; file untouched
    old = client.put(f"/calibration/api/gt/{EMIRATES}/5", json=_body(None)).json()["record"]
    assert "distortion" not in old and "lens" not in old
    got = client.get(f"/calibration/api/gt/{EMIRATES}/5").json()
    assert got["distortion"]["k1"] == pytest.approx(load_rig("emirates").k1) and got["distortion_source"] == "rig_default"
    assert "distortion" not in cal._read_gt(EMIRATES)["frames"]["5"]
    # unknown ground: nothing guessed, nothing marked
    u = client.put(f"/calibration/api/gt/{UNKNOWN}/1", json=_body(None, distortion=None)).json()["record"]
    assert "distortion" not in u and "lens" not in u
    assert "distortion" not in client.get(f"/calibration/api/gt/{UNKNOWN}/1").json()


def _proposal(clip, idx, H_plain, corr):
    def _w(doc):
        doc["runs"]["r1"] = {"source": "claude", "created": "t", "params": {}, "frames": {str(idx): {
            "status": "proposed", "score": 0.5, "H": H_plain.tolist(), "H_convention": "normalised_image_to_pitch_m",
            "pitch_dims": [105, 68], "image_size": [W, H], "correspondences": corr,
            "inliers": [True] * len(corr)}}}
    return _w


def test_proposal_without_lens_is_rendered_and_accepted_with_the_default(client, env):
    plain = _body(None)                                    # stored: fitted without a lens
    fit = client.post("/calibration/api/fit", json=plain).json()
    update_proposals(env, EMIRATES, _proposal(EMIRATES, 6, np.array(fit["H"]), fit["correspondences"]))
    rec = client.get(f"/calibration/api/proposals/{EMIRATES}/r1/6").json()
    k1 = load_rig("emirates").k1
    assert rec["distortion"]["k1"] == pytest.approx(k1) and rec["distortion_source"] == "rig_default"
    # H and overlay were re-derived from the points under that lens (what a fresh fit in the editor gives)
    refit = client.post("/calibration/api/fit", json={**plain, "correspondences": rec["correspondences"],
                                                      "distortion": rec["distortion"]}).json()
    assert np.allclose(rec["H"], refit["H"]) and rec["overlay"] == refit["overlay"]
    assert rec["overlay"]["TOUCHLINE_TOP"] != fit["overlay"]["TOUCHLINE_TOP"]
    # the stored proposal is unchanged
    stored = read_proposals(env, EMIRATES)["runs"]["r1"]["frames"]["6"]
    assert "distortion" not in stored and stored["H"] == fit["H"]
    # accepting as proposed saves exactly what the card showed
    acc = client.post(f"/calibration/api/proposals/{EMIRATES}/r1/6/accept", json={}).json()
    assert acc["ok"] and acc["record"]["distortion"]["k1"] == pytest.approx(k1)


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
    # bad requests
    assert client.post("/calibration/api/snap", json={**body, "mode": "x"}).status_code == 400
    assert client.post("/calibration/api/snap", json={**body, "correspondences": body["correspondences"][:3]}).status_code == 400
    assert client.post("/calibration/api/snap", json={**body, "clip": "nope"}).status_code == 404
    assert client.post("/calibration/api/snap", json={**body, "frame_index": 99}).status_code == 404
