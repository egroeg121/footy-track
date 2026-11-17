"""Calibration API tests (src/footy_track/labeller/README.md §16, LAB-12xx).

Video IO is replaced by a synthetic frame; geometry is real.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from footy_track.clip_id import canonical_clip_id
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
    assert [(c["name"], c["n_gt"]) for c in clips] == [("arsenal_mancity_20250925_seg010", 0)]
    assert clips[0]["canonical"] == canonical_clip_id("arsenal_mancity_20250925_seg010")
    assert clips[0]["n_not_broadcast"] == 0
    m = client.get("/calibration/api/clips/arsenal_mancity_20250925_seg010/meta").json()
    assert m["frames"] == 50 and m["gt_frames"] == []
    r = client.get("/calibration/api/frame/arsenal_mancity_20250925_seg010/3.jpg")
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    assert client.get("/calibration/api/frame/arsenal_mancity_20250925_seg010/99.jpg").status_code == 404
    assert client.get("/calibration/api/frame/nope/0.jpg").status_code == 404


def test_clips_list_counts_not_broadcast_frames(client, cal_env, clips_dir):
    clip = "arsenal_mancity_20250925_seg010"
    (clips_dir / "other_clip_seg001.mp4").touch()
    assert client.put(f"/calibration/api/frame_type/{clip}/3", json={}).status_code == 200
    assert client.put(f"/calibration/api/frame_type/{clip}/9", json={}).status_code == 200
    by_name = {c["name"]: c for c in client.get("/calibration/api/clips").json()["clips"]}
    assert by_name[clip]["n_not_broadcast"] == 2 and by_name[clip]["n_gt"] == 0
    assert by_name["other_clip_seg001"]["n_not_broadcast"] == 0
    assert client.delete(f"/calibration/api/frame_type/{clip}/3").status_code == 200
    by_name = {c["name"]: c for c in client.get("/calibration/api/clips").json()["clips"]}
    assert by_name[clip]["n_not_broadcast"] == 1


def test_too_few_points_editor_endpoint_without_proposals(client, cal_env):
    clip = "arsenal_mancity_20250925_seg010"
    r = client.put(f"/calibration/api/too_few_points/{clip}/4", json={"tags": ["goal_area"]})
    assert r.status_code == 200 and r.json()["too_few_points"] == {"4": ["goal_area"]}
    assert client.put(f"/calibration/api/too_few_points/{clip}/-1", json={}).status_code == 400
    assert client.get(f"/calibration/api/clips/{clip}/meta").json()["too_few_points"] == {"4": ["goal_area"]}
    c = client.get("/calibration/api/clips").json()["clips"][0]
    assert c["n_too_few_points"] == 1 and c["n_not_broadcast"] == 0 and c["n_gt"] == 0
    assert client.delete(f"/calibration/api/too_few_points/{clip}/4").status_code == 200
    assert client.get("/calibration/api/clips").json()["clips"][0]["n_too_few_points"] == 0


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


# ---------------------------------------------------------------------------
# POST /calibration/api/auto_points (centre finder / D finder)
# ---------------------------------------------------------------------------


def _circle_frame(dist=None):
    from ..calibration import test_centre_circle as tcc

    Hp = tcc._homography(
        [(-20, -15), (20, -15), (20, 25), (-20, 25)],
        [(200, 650), (1080, 650), (900, 200), (380, 200)],
    )
    img = tcc._render(Hp, [tcc._line((0, -25), (0, 28)), tcc._arc(0, 0, 9.15, 0, 2 * np.pi)], dist)
    truth = tcc._project(Hp, [(0, 9.15), (0, -9.15)], dist) / [tcc.W, tcc.H]
    return img, truth


PM_SMALL = PitchModel(100.0, 64.0)


@pytest.fixture
def auto_env(clips_dir, gt_marks_dir, monkeypatch):
    from ..calibration import test_pitch_finders as tpf

    (clips_dir / "arsenal_mancity_20250925_seg010.mp4").touch()
    cal._AUTO_CACHE.clear()
    reads = []
    img, truth = _circle_frame()
    plain = np.full(img.shape, (40, 130, 40), np.uint8)
    frames = {
        3: img,                                              # centre circle + halfway line (105 x 68 pitch)
        4: plain,
        5: tpf.centre_frame(PM_SMALL)[0],                    # + far touchline, a 100 x 64 pitch
        6: tpf.penalty_frame(PM_SMALL, "R")[0],              # the whole right penalty area, 100 x 64
    }

    def read(v, i):
        reads.append(i)
        return frames.get(i)

    monkeypatch.setattr(cal, "_read_frame", read)
    yield {"reads": reads, "truth": truth}
    cal._AUTO_CACHE.clear()


CLIP = "arsenal_mancity_20250925_seg010"


def _post(client, **b):
    return client.post("/calibration/api/auto_points", json=b)


def test_auto_points_centre_returns_circle_pairs(client, auto_env):
    r = _post(client, clip=CLIP, frame_index=3, kind="centre")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] and d["kind"] == "centre" and d["confidence"] > 0.55
    # the two seen circle x halfway-line points come first; a shape fit (when one is accepted) adds points computed
    # from its homography, which the response carries
    assert [c["pitch_id"] for c in d["correspondences"]][:2] == ["CENTRE_CIRCLE_TOP", "CENTRE_CIRCLE_BOTTOM"]
    kps = PitchModel().keypoints()
    for c in d["correspondences"]:
        if c["pitch_id"]:
            assert c["pitch_xy"] == list(kps[c["pitch_id"]])
    assert not any(c["derived"] for c in d["correspondences"][:2])
    got = np.array([c["image"] for c in d["correspondences"][:2]])
    assert np.linalg.norm((got - auto_env["truth"]) * [1280, 720], axis=1).max() < 8
    assert ("H" in d) == bool(d["fit"].get("accepted")) and (len(d["correspondences"]) >= 4 if "H" in d else True)


def test_auto_points_centre_with_touchline_adds_the_end_and_derived_points(client, auto_env):
    d = _post(client, clip=CLIP, frame_index=5, kind="centre", pitch_dims=[100, 64]).json()
    assert d["ok"]
    by = {c["pitch_id"]: c for c in d["correspondences"] if c["pitch_id"]}
    assert by["HALFWAY_TOP"]["pitch_xy"] == [0.0, 32.0] and by["HALFWAY_TOP"]["derived"] is False  # the request's width
    assert by["CENTRE_SPOT"]["derived"] is True
    free = [c for c in d["correspondences"] if c["pitch_id"] is None]
    assert free and all(c["derived"] and len(c["pitch_xy"]) == 2 for c in free)
    assert len(d["correspondences"]) >= 5
    # the fitted pitch -> undistorted normalised image homography comes with the points (the page can draw the shape)
    H = np.array(d["H"])
    assert H.shape == (3, 3) and d["H_convention"] == "pitch_m_to_undistorted_normalised_image"
    assert d["fit"]["accepted"] in ("touchline", "rig", "rig+touchline")
    for c in d["correspondences"]:
        p = H @ [c["pitch_xy"][0], c["pitch_xy"][1], 1.0]
        assert np.abs(p[:2] / p[2] - c["image"]).max() < 1e-6  # every point is H applied to its pitch coordinates


def test_auto_points_d_returns_the_penalty_area(client, auto_env):
    d = _post(client, clip=CLIP, frame_index=6, kind="d", pitch_dims=[100, 64]).json()
    assert d["ok"] and d["kind"] == "d"
    by = {c["pitch_id"]: c for c in d["correspondences"]}
    assert {"R_PEN_ARC_TOP", "R_PEN_ARC_BOTTOM", "R_PEN_SPOT"} <= set(by)
    assert by["R_PEN_SPOT"]["pitch_xy"] == [39.0, 0.0] and by["R_PEN_SPOT"]["derived"] is True  # 100 / 2 - 11
    assert len(d["correspondences"]) >= 4
    # the pair list the editor would build is fit-able with 4+ non-collinear points
    fit = client.post("/calibration/api/fit", json={"correspondences": [
        {"image": c["image"], "pitch_id": c["pitch_id"], "pitch_xy": c["pitch_xy"]} for c in d["correspondences"]],
        "pitch_dims": [100, 64]}).json()
    assert fit["ok"] and fit["mean_err_m"] < 0.5


def test_auto_points_abstains_with_a_reason_and_http_200(client, auto_env):
    for kind in ("centre", "d"):
        r = _post(client, clip=CLIP, frame_index=4, kind=kind)
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is False and d["reason"] and "correspondences" not in d


def test_auto_points_errors(client, auto_env):
    assert _post(client, clip="nope", frame_index=3, kind="d").status_code == 404
    assert _post(client, clip=CLIP, frame_index=99, kind="d").status_code == 404   # unreadable frame
    assert _post(client, clip=CLIP, kind="d").status_code == 400                    # no frame_index
    assert _post(client, clip=CLIP, frame_index=-1, kind="d").status_code == 400
    assert _post(client, clip=CLIP, frame_index=3).status_code == 400               # kind is required
    assert _post(client, clip=CLIP, frame_index=3, kind="circle").status_code == 400
    assert _post(client, clip=CLIP, frame_index=3, kind="centre", distortion={"k1": "x"}).status_code == 400


def test_auto_points_is_cached_per_frame_kind_dims_and_distortion(client, auto_env):
    body = {"clip": CLIP, "frame_index": 3, "kind": "centre"}
    a = client.post("/calibration/api/auto_points", json=body).json()
    b = client.post("/calibration/api/auto_points", json=body).json()
    assert a == b and auto_env["reads"] == [3]
    _post(client, **body, distortion=geo.make_distortion(0.03))
    assert auto_env["reads"] == [3, 3]          # a different lens is a different answer
    _post(client, **{**body, "kind": "d"})
    assert auto_env["reads"] == [3, 3, 3]       # so is the other finder
    _post(client, **body, pitch_dims=[100, 64])
    assert auto_env["reads"] == [3, 3, 3, 3]    # and other pitch dims (the metres / touchline end depend on them)
    _post(client, clip=CLIP, frame_index=4, kind="centre")
    _post(client, clip=CLIP, frame_index=4, kind="centre")
    assert auto_env["reads"] == [3, 3, 3, 3, 4]   # abstentions are cached too


def test_auto_points_cache_is_bounded(client, auto_env, monkeypatch):
    monkeypatch.setattr(cal, "_AUTO_CACHE_MAX", 1)
    for i in (3, 4):
        _post(client, clip=CLIP, frame_index=i, kind="centre")
    assert [k[2] for k in cal._AUTO_CACHE] == [4]   # least recently used evicted


def test_editor_page_has_the_two_auto_buttons():
    from fastapi.testclient import TestClient

    from footy_track.labeller import server as labeller_server

    html = TestClient(labeller_server.app).get("/calibration").text
    assert 'id="autoCentreBtn"' in html and ">Centre<" in html and 'id="autoDBtn"' in html and ">D<" in html
    assert 'id="autoBtn"' not in html and ">Auto circle/D<" not in html
    assert "/auto_points" in html and 'e.key === "c"' in html and 'e.key === "d"' in html
    assert 'autoPoints("centre")' in html and 'autoPoints("d")' in html and "kind," in html
    assert "autoPairsToAdd" in html and "snapshot();" in html
    assert "add ${4 - total} more (e.g. touchline / halfway line ends) to fit" in html
