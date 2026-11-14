"""Calibration API tests (src/footy_track/labeller/README.md §16, LAB-12xx).

Video IO is replaced by a synthetic frame; geometry is real.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from footy_track.calibration import geometry as geo
from footy_track.calibration.pitch_model import PitchModel
from footy_track.labeller import calibration as cal

from ..calibration.test_geometry import H, W, _camera_h

KP = [
    "L_PEN_BOX_TOP_OUTER", "L_PEN_BOX_BOTTOM_OUTER", "CENTRE_SPOT",
    "HALFWAY_TOP", "HALFWAY_BOTTOM", "L_PEN_SPOT",
]


@pytest.fixture
def cal_env(clips_dir, gt_marks_dir, monkeypatch):
    (clips_dir / "arsenal_mancity_20250925_seg010.mp4").touch()
    frame = np.full((H, W, 3), 90, np.uint8)
    monkeypatch.setattr(cal, "_read_frame", lambda v, i: frame if i < 50 else None)
    monkeypatch.setattr(
        cal, "_video_meta", lambda v: {"frames": 50, "fps": 25.0, "width": W, "height": H}
    )
    return gt_marks_dir


def _corr(ids=KP, with_free=False):
    pitch = PitchModel()
    img, _ = geo.apply_homography(
        np.linalg.inv(_camera_h()), np.array([pitch.keypoints()[k] for k in ids])
    )
    out = [{"image": p.tolist(), "pitch_id": k} for p, k in zip(img, ids, strict=True)]
    if with_free:
        free, _ = geo.apply_homography(np.linalg.inv(_camera_h()), np.array([[10.0, 5.0]]))
        out.append({"image": free[0].tolist(), "pitch_xy": [10.0, 5.0]})
    return out


def _body(**kw):
    return {"correspondences": _corr(), "pitch_dims": [105, 68], "image_size": [W, H], **kw}


def test_fit_returns_h_errors_and_overlay(client, cal_env):
    r = client.post("/calibration/api/fit", json=_body())
    assert r.status_code == 200
    d = r.json()
    assert np.max(d["err_m"]) < 1e-4 and np.max(d["err_px"]) < 0.05
    assert len(d["H"]) == 3 and len(d["H_flat"]) == 9
    assert {"TOUCHLINE_TOP", "CENTRE_CIRCLE", "L_PEN_BOX_FRONT"} <= set(d["overlay"])
    run = d["overlay"]["HALFWAY_LINE"][0]
    assert len(run[0]) == 2
    # H is normalised image -> pitch metres
    out, _ = geo.apply_homography(np.array(d["H"]), np.array([_corr()[2]["image"]]))
    assert np.allclose(out, [[0, 0]], atol=1e-5)


def test_fit_accepts_free_points_and_ransac(client, cal_env):
    body = _body(method="ransac")
    body["correspondences"] = _corr(with_free=True)
    d = client.post("/calibration/api/fit", json=body).json()
    assert d["ok"] and all(d["inliers"])
    assert d["correspondences"][-1]["pitch_id"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.update(correspondences=b["correspondences"][:3]),
        lambda b: b["correspondences"][0].update(pitch_id="NOPE"),
        lambda b: b["correspondences"][0].pop("image"),
        lambda b: b.update(pitch_dims=[10, 10]),
        lambda b: b.update(method="magic"),
    ],
)
def test_fit_rejects_bad_input_with_4xx(client, cal_env, mutate):
    body = _body()
    mutate(body)
    r = client.post("/calibration/api/fit", json=body)
    assert r.status_code == 400 and r.json()["ok"] is False


def test_pitch_model_route(client, cal_env):
    d = client.get("/calibration/api/pitch_model?length=100&width=64").json()
    assert d["length"] == 100 and "CENTRE_SPOT" in d["keypoints"]
    assert d["warp"]["margin_m"] == cal.WARP_MARGIN_M


def test_clips_meta_and_frame(client, cal_env):
    clips = client.get("/calibration/api/clips").json()["clips"]
    assert clips == [{"name": "arsenal_mancity_20250925_seg010", "n_gt": 0}]
    m = client.get("/calibration/api/clips/arsenal_mancity_20250925_seg010/meta").json()
    assert m["frames"] == 50 and m["gt_frames"] == []
    r = client.get("/calibration/api/frame/arsenal_mancity_20250925_seg010/3.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    assert client.get("/calibration/api/frame/arsenal_mancity_20250925_seg010/99.jpg").status_code == 404
    assert client.get("/calibration/api/frame/nope/0.jpg").status_code == 404


def test_warp_returns_raster_of_expected_size(client, cal_env):
    import cv2

    fit = client.post("/calibration/api/fit", json=_body()).json()
    r = client.post(
        "/calibration/api/warp",
        json={"clip": "arsenal_mancity_20250925_seg010", "frame_index": 0, "H": fit["H"]},
    )
    assert r.status_code == 200
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert (img.shape[1], img.shape[0]) == geo.pitch_raster_size(
        PitchModel(), cal.WARP_PX_PER_M, cal.WARP_MARGIN_M
    )
    bad = client.post("/calibration/api/warp", json={"clip": "x", "frame_index": 0, "H": [1, 2]})
    assert bad.status_code == 400


def test_gt_save_load_round_trip(client, cal_env):
    clip = "arsenal_mancity_20250925_seg010"
    assert client.get(f"/calibration/api/gt/{clip}/7").status_code == 404
    r = client.put(f"/calibration/api/gt/{clip}/7", json=_body())
    assert r.status_code == 200 and r.json()["gt_frames"] == [7]
    rec = client.get(f"/calibration/api/gt/{clip}/7").json()
    assert rec["frame_index"] == 7 and rec["pitch_dims"] == [105.0, 68.0]
    assert rec["H_convention"] == "normalised_image_to_pitch_m"
    assert len(rec["correspondences"]) == len(KP) and rec["ts"]
    assert rec["correspondences"][0]["pitch_id"] == KP[0]
    assert rec["correspondences"][0]["pitch_xy"] == list(PitchModel().keypoints()[KP[0]])
    # second frame, listing, meta indicator, clip listing counts
    client.put(f"/calibration/api/gt/{clip}/2", json=_body())
    lst = client.get(f"/calibration/api/gt/{clip}").json()
    assert [f["frame_index"] for f in lst["frames"]] == [2, 7]
    assert client.get(f"/calibration/api/clips/{clip}/meta").json()["gt_frames"] == [2, 7]
    assert client.get("/calibration/api/clips").json()["clips"][0]["n_gt"] == 2
    # delete one
    assert client.delete(f"/calibration/api/gt/{clip}/2").json()["gt_frames"] == [7]
    assert client.delete(f"/calibration/api/gt/{clip}/2").status_code == 404


def test_gt_file_location_is_canonical_and_alias_shared(client, cal_env):
    client.put("/calibration/api/gt/arsenal_mancity_20250925_seg010/1", json=_body())
    path = cal_env / "calibration_gt" / "arsenal_mancity_seg010.json"
    assert path.exists()
    doc = json.loads(path.read_text())
    assert doc["clip"] == "arsenal_mancity_seg010" and "1" in doc["frames"]
    # an alias of the same clip reads the same GT
    assert client.get("/calibration/api/gt/mancity_seg010/1").status_code == 200
    # nothing else was written under the GT dir (no sidecars touched)
    assert [p.name for p in cal_env.iterdir()] == ["calibration_gt"]
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no tmp left


def test_gt_save_rejects_too_few_points_and_keeps_existing(client, cal_env):
    clip = "arsenal_mancity_20250925_seg010"
    client.put(f"/calibration/api/gt/{clip}/1", json=_body())
    body = _body()
    body["correspondences"] = body["correspondences"][:2]
    assert client.put(f"/calibration/api/gt/{clip}/1", json=body).status_code == 400
    assert len(client.get(f"/calibration/api/gt/{clip}/1").json()["correspondences"]) == len(KP)


def test_pages_served(client):
    r = client.get("/calibration")
    assert r.status_code == 200 and "Pitch calibration" in r.text
    assert 'href="/calibration"' in client.get("/main").text


_COLL = ["R_PEN_BOX_TOP_OUTER", "R_PEN_ARC_TOP", "R_GOAL_AREA_TOP_OUTER", "R_PEN_ARC_BOTTOM"]


def test_fit_route_400_names_collinear_points(client, cal_env):
    body = _body(correspondences=_corr(_COLL))
    r = client.post("/calibration/api/fit", json=body)
    assert r.status_code == 400 and r.json()["ok"] is False
    assert "points 1, 2, 4 lie on one pitch line" in r.json()["error"]
    body["correspondences"] = _corr([*_COLL, "R_GOALPOST_TOP"])
    d = client.post("/calibration/api/fit", json=body).json()
    assert d["ok"] and any("only one set of 4" in w for w in d["warnings"])
    assert client.post("/calibration/api/fit", json=_body()).json()["warnings"] == []


def test_gt_file_mode_new_is_664_and_existing_is_preserved(client, cal_env):
    clip = "arsenal_mancity_20250925_seg010"
    path = cal._gt_path(clip)
    assert client.put(f"/calibration/api/gt/{clip}/1", json=_body()).status_code == 200
    assert path.stat().st_mode & 0o777 == 0o664
    path.chmod(0o640)
    assert client.put(f"/calibration/api/gt/{clip}/2", json=_body()).status_code == 200
    assert path.stat().st_mode & 0o777 == 0o640


def test_fit_warp_and_gt_with_distortion(client, cal_env):
    """Points are observed (distorted) coordinates; H is fitted from the undistorted ones."""
    pitch = PitchModel()
    lens = geo.make_distortion(-0.03, W / H)
    obs, _ = geo.pitch_to_image(_camera_h(), np.array([pitch.keypoints()[k] for k in KP]), lens)
    corr = [{"image": p.tolist(), "pitch_id": k} for p, k in zip(obs, KP, strict=True)]
    body = {"correspondences": corr, "pitch_dims": [105, 68], "image_size": [W, H]}
    plain = client.post("/calibration/api/fit", json=body).json()
    d = client.post("/calibration/api/fit", json={**body, "distortion": lens}).json()
    assert "distortion" not in plain and d["distortion"]["k1"] == -0.03
    assert np.max(d["err_px"]) < 0.05 < np.max(plain["err_px"])  # the lens model explains the points
    assert np.allclose(d["H"], _camera_h(), atol=1e-6) or np.allclose(
        np.array(d["H"]) / np.linalg.norm(d["H"]), _camera_h() / np.linalg.norm(_camera_h()), atol=1e-6)
    assert d["overlay"]["TOUCHLINE_TOP"] != plain["overlay"]["TOUCHLINE_TOP"]
    # the warp honours it, k1 = 0 leaves the bytes untouched
    wb = {"clip": "arsenal_mancity_20250925_seg010", "frame_index": 0, "H": d["H"], "pitch_dims": [105, 68], "px_per_m": 2}
    a = client.post("/calibration/api/warp", json=wb)
    b = client.post("/calibration/api/warp", json={**wb, "distortion": lens})
    z = client.post("/calibration/api/warp", json={**wb, "distortion": {"model": "division", "k1": 0.0}})
    assert a.status_code == b.status_code == z.status_code == 200 and z.content == a.content
    assert client.post("/calibration/api/fit", json={**body, "distortion": {"k1": "x"}}).status_code == 400
    # GT save keeps the lens with H
    gt = client.put("/calibration/api/gt/arsenal_mancity_seg010/3", json={**body, "distortion": lens}).json()
    assert gt["record"]["distortion"]["k1"] == -0.03
    assert "distortion" not in client.put("/calibration/api/gt/arsenal_mancity_seg010/4", json=body).json()["record"]
