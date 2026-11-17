"""One labeller server, many datasets: routing, isolation, and the default.

Two datasets live in tmp dirs behind a registry file. Everything a request
touches (clips, marks, ball-check verdicts, calibration GT, sessions, caches)
must follow ``?dataset=``; labels saved in one dataset must never appear in the
other; and a request with no parameter keeps meaning "the default dataset".
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from footy_track.labeller import ball_check, datasets
from footy_track.labeller import server as labeller_server

from .conftest import FakeVideoCapture, make_box


def _sidecar_line(frame_index: int, tags: list[str]) -> str:
    return json.dumps(
        {
            "frame_index": frame_index,
            "bbox": {"x": 0.1, "y": 0.1, "w": 0.02, "h": 0.02},
            "center": {"x": 0.11, "y": 0.11},
            "tags": tags,
        }
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_lists_datasets_and_default(web):
    j = web.get("/api/datasets").json()
    assert j["default"] == "calib"
    assert [d["name"] for d in j["datasets"]] == ["calib", "nottinghamforest"]
    assert j["current"] == "calib"
    assert web.get("/api/datasets?dataset=nottinghamforest").json()["current"] == "nottinghamforest"


def test_unknown_dataset_is_404_not_silent_default(web):
    r = web.get("/clips?dataset=nope")
    assert r.status_code == 404
    assert "nope" in r.json()["detail"]
    # a write must be refused too, never routed to the default dataset
    assert web.post("/ball_check/verdict?dataset=nope", json={}).status_code == 404


def test_websocket_unknown_dataset_is_refused(web):
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        web.websocket_connect("/ws?dataset=nope"),
    ):
        pass
    assert exc.value.code == 4404


def test_websocket_accepts_a_known_dataset(web):
    with web.websocket_connect("/ws?dataset=nottinghamforest") as ws:
        ws.send_json({"type": "pause"})
        assert ws.receive_json() == {"type": "status", "state": "paused"}


def test_missing_registry_falls_back_to_legacy_single_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("FOOTY_LABELLER_DATASETS", str(tmp_path / "absent.json"))
    monkeypatch.setenv("FOOTY_CLIPS_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("FOOTY_GT_MARKS_DIR", str(tmp_path / "m"))
    assert datasets.names() == ["calib"]
    ds = datasets.get()
    assert (ds.clips_dir, ds.marks_dir) == (tmp_path / "c", tmp_path / "m")


def test_malformed_registry_does_not_take_the_server_down(tmp_path, monkeypatch):
    reg = tmp_path / "bad.json"
    reg.write_text("{not json")
    monkeypatch.setenv("FOOTY_LABELLER_DATASETS", str(reg))
    assert datasets.names() == ["calib"]
    assert datasets.registry_error()
    assert TestClient(labeller_server.app).get("/api/datasets").json()["error"]


def test_env_overrides_apply_to_the_default_dataset_only(two, tmp_path, monkeypatch):
    other = tmp_path / "elsewhere"
    monkeypatch.setenv("FOOTY_GT_MARKS_DIR", str(other))
    assert datasets.get("calib").marks_dir == other
    assert datasets.get("nottinghamforest").marks_dir == two["nottinghamforest"] / "marks"


def test_registry_edits_are_picked_up_without_restart(web, two, tmp_path):
    assert web.get("/clips?dataset=extra").status_code == 404
    entries = {
        n: {"clips_dir": str(two[n] / "clips"), "marks_dir": str(two[n] / "marks")}
        for n in two
    }
    entries["extra"] = {
        "clips_dir": str(tmp_path / "x"),
        "marks_dir": str(tmp_path / "xm"),
    }
    datasets.write_registry(datasets.registry_path(), "calib", entries)
    assert web.get("/clips?dataset=extra").status_code == 200


# ---------------------------------------------------------------------------
# Routing and isolation
# ---------------------------------------------------------------------------


def test_clips_follow_the_dataset(web, two):
    (two["calib"] / "clips" / "a.mp4").touch()
    (two["nottinghamforest"] / "clips" / "n1.mp4").touch()
    (two["nottinghamforest"] / "clips" / "n2.mp4").touch()
    names = lambda url: [c["name"] for c in web.get(url).json()["clips"]]  # noqa: E731
    assert names("/clips") == ["a.mp4"]  # default, no parameter
    assert names("/clips?dataset=calib") == ["a.mp4"]
    assert names("/clips?dataset=nottinghamforest") == ["n1.mp4", "n2.mp4"]
    assert web.get("/clips?dataset=nottinghamforest").json()["dir"] == str(
        two["nottinghamforest"] / "clips"
    )


def test_labels_saved_in_one_dataset_never_appear_in_the_other(web, two, fake_cv2):
    """Same clip name in both datasets; edit through the real session endpoints."""
    FakeVideoCapture.total_frames = 5
    for name in two:
        (two[name] / "clips" / "same.mp4").touch()
    clip = str(two["nottinghamforest"] / "clips" / "same.mp4")
    box = {"label": "ball", "x": 0.3, "y": 0.3, "w": 0.02, "h": 0.02}

    assert (
        web.post("/session/load?dataset=nottinghamforest", json={"video_path": clip}).status_code
        == 200
    )
    assert (
        web.post("/edit?dataset=nottinghamforest", json={"idx": 2, "objects": [box]}).status_code
        == 200
    )
    labeller_server._SESSIONS["nottinghamforest"]._do_flush()

    assert (two["nottinghamforest"] / "marks" / "same.jsonl").exists()
    assert not (two["calib"] / "marks" / "same.jsonl").exists()
    assert list((two["calib"] / "marks").iterdir()) == []

    # The default dataset's session is untouched: no clip loaded, no boxes.
    assert labeller_server.SESSION.video_path is None
    assert web.get("/marks").json() == {
        "no_ball": [],
        "not_broadcast": [],
        "ball": [],
        "player": [],
    }
    assert web.get("/marks?dataset=nottinghamforest").json()["ball"] == [2]
    assert [c["marked"] for c in web.get("/clips").json()["clips"]] == [False]
    assert [c["marked"] for c in web.get("/clips?dataset=nottinghamforest").json()["clips"]] == [
        True
    ]


def test_sessions_do_not_share_state_and_default_keeps_working(web, two, fake_cv2):
    FakeVideoCapture.total_frames = 4
    (two["calib"] / "clips" / "c.mp4").touch()
    (two["nottinghamforest"] / "clips" / "n.mp4").touch()
    assert (
        web.post(
            "/session/load", json={"video_path": str(two["calib"] / "clips" / "c.mp4")}
        ).status_code
        == 200
    )
    assert (
        web.post(
            "/session/load?dataset=nottinghamforest",
            json={"video_path": str(two["nottinghamforest"] / "clips" / "n.mp4")},
        ).status_code
        == 200
    )
    assert labeller_server.SESSION.video_path.name == "c.mp4"
    assert labeller_server._SESSIONS["nottinghamforest"].video_path.name == "n.mp4"
    assert labeller_server._SESSIONS["nottinghamforest"] is not labeller_server.SESSION
    labeller_server.SESSION.set_frame(1, [make_box(label="ball")])
    assert web.get("/timeline/1").json()["boxes"]
    assert web.get("/timeline/1?dataset=nottinghamforest").json()["boxes"] == []


def test_session_flush_thread_writes_to_its_own_dataset(two, fake_cv2):
    """The debounced flush runs on a timer thread with no request context."""
    FakeVideoCapture.total_frames = 3
    (two["nottinghamforest"] / "clips" / "n.mp4").touch()
    sess = labeller_server.Session("nottinghamforest")
    sess.load(str(two["nottinghamforest"] / "clips" / "n.mp4"))
    sess.set_frame(0, [make_box(label="ball")])
    with datasets.using("calib"):  # a different dataset is "current" while flushing
        sess._do_flush()
    assert (two["nottinghamforest"] / "marks" / "n.jsonl").exists()
    assert list((two["calib"] / "marks").iterdir()) == []


def test_ball_check_verdicts_and_stats_are_per_dataset(web, two):
    for name in two:
        (two[name] / "clips" / "clip.mp4").touch()
        (two[name] / "marks" / "clip.jsonl").write_text(
            _sidecar_line(0, ["in_play_ball", "yolo"]) + "\n"
        )
    body = {"clip": "clip", "frame_index": 0, "box_index": 0, "verdict": "ball"}
    assert web.post("/ball_check/verdict?dataset=nottinghamforest", json=body).json()["ok"]

    assert (two["nottinghamforest"] / "marks" / "ball_checks" / "clip.jsonl").exists()
    assert not (two["calib"] / "marks" / "ball_checks").exists()
    assert web.get("/ball_check/stats?dataset=nottinghamforest").json()["judged"] == 1
    assert web.get("/ball_check/stats").json()["judged"] == 0
    # the card is still to do in calib, already done in nottinghamforest
    assert len(web.get("/ball_check/queue").json()["items"]) == 1
    assert web.get("/ball_check/queue?dataset=nottinghamforest").json()["items"] == []


def test_ball_check_list_pool_is_per_dataset(web, two, tmp_path):
    lst = tmp_path / "nottinghamforest_list.json"
    lst.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "clip": "n",
                        "frame_index": 3,
                        "bbox": {"x": 0.1, "y": 0.1, "w": 0.02, "h": 0.02},
                        "label": "ball",
                    }
                ]
            }
        )
    )
    entries = {
        "calib": {
            "clips_dir": str(two["calib"] / "clips"),
            "marks_dir": str(two["calib"] / "marks"),
        },
        "nottinghamforest": {
            "clips_dir": str(two["nottinghamforest"] / "clips"),
            "marks_dir": str(two["nottinghamforest"] / "marks"),
            "ball_check_list": str(lst),
        },
    }
    datasets.write_registry(datasets.registry_path(), "calib", entries)
    assert (
        len(web.get("/ball_check/queue?source=list&dataset=nottinghamforest").json()["items"])
        == 1
    )
    assert web.get("/ball_check/queue?source=list").json()["items"] == []


def test_crop_cache_is_keyed_by_dataset(web, two, fake_cv2):
    """Same clip stem in two datasets must not serve one dataset's crop for the other."""
    for name in two:
        (two[name] / "clips" / "clip.mp4").touch()
        (two[name] / "marks" / "clip.jsonl").write_text(
            _sidecar_line(0, ["in_play_ball", "yolo"]) + "\n"
        )
    assert web.get("/ball_check/crop/clip/0/0.jpg").status_code == 200
    assert web.get("/ball_check/crop/clip/0/0.jpg?dataset=nottinghamforest").status_code == 200
    assert {k[0] for k in ball_check._BC_CROP_CACHE} == {"calib", "nottinghamforest"}
    # a clip that exists only in nottinghamforest is a 404 in calib even once its crop is cached
    (two["nottinghamforest"] / "clips" / "only_n.mp4").touch()
    (two["nottinghamforest"] / "marks" / "only_n.jsonl").write_text(
        _sidecar_line(0, ["in_play_ball", "yolo"]) + "\n"
    )
    assert web.get("/ball_check/crop/only_n/0/0.jpg?dataset=nottinghamforest").status_code == 200
    assert web.get("/ball_check/crop/only_n/0/0.jpg").status_code == 404


def test_calibration_gt_is_per_dataset(web, two):
    (two["calib"] / "clips" / "x.mp4").touch()
    (two["nottinghamforest"] / "clips" / "y.mp4").touch()
    assert [c["name"] for c in web.get("/calibration/api/clips").json()["clips"]] == [
        "x"
    ]
    assert [
        c["name"]
        for c in web.get("/calibration/api/clips?dataset=nottinghamforest").json()["clips"]
    ] == ["y"]


def test_too_few_points_marks_are_per_dataset(web, two):
    (two["calib"] / "clips" / "x.mp4").touch()
    (two["nottinghamforest"] / "clips" / "x.mp4").touch()
    r = web.put("/calibration/api/too_few_points/x/3", json={"tags": ["halfway_line"]})
    assert r.status_code == 200
    assert (two["calib"] / "marks" / "frame_type" / "x.json").exists()
    assert not (two["nottinghamforest"] / "marks" / "frame_type").exists()
    assert web.get("/calibration/api/too_few_points").json()["total"] == 1
    assert web.get("/calibration/api/too_few_points?dataset=nottinghamforest").json() == {"total": 0, "clips": {}}
    assert web.get("/calibration/api/clips").json()["clips"][0]["n_too_few_points"] == 1
    assert web.get("/calibration/api/clips?dataset=nottinghamforest").json()["clips"][0]["n_too_few_points"] == 0
    assert web.delete("/calibration/api/too_few_points/x/3?dataset=nottinghamforest").status_code == 404
    assert web.delete("/calibration/api/too_few_points/x/3").status_code == 200


def test_identity_dirs_do_not_bleed_between_datasets(two):
    calib, nottinghamforest = datasets.get("calib"), datasets.get("nottinghamforest")
    assert calib.identity_labels_dir != nottinghamforest.identity_labels_dir
    assert calib.tracklets_dir != nottinghamforest.tracklets_dir
    assert nottinghamforest.identity_labels_dir.is_relative_to(two["nottinghamforest"])
    assert calib.embeddings_dir != nottinghamforest.embeddings_dir


def test_pages_carry_the_dataset_shim(web):
    for page in (
        "/",
        "/labeller",
        "/ball_check",
        "/calibration",
        "/calibration/review",
        "/object_review",
        "/identity",
    ):
        assert "/static/dataset.js" in web.get(page).text, page
    assert web.get("/static/dataset.js").status_code == 200


def test_auto_points_follow_the_dataset(web, two, monkeypatch):
    """Same clip stem in both datasets: the detector reads each dataset's own video, and the
    cache never serves one dataset's answer for the other."""
    import numpy as np

    from footy_track.labeller import calibration as cal

    from .test_calibration import _circle_frame

    cal._AUTO_CACHE.clear()
    (two["calib"] / "clips" / "x.mp4").touch()
    (two["nottinghamforest"] / "clips" / "x.mp4").touch()
    (two["nottinghamforest"] / "clips" / "only_nf.mp4").touch()
    img, _ = _circle_frame()
    plain = np.full(img.shape, (40, 130, 40), np.uint8)
    # calib's x.mp4 shows the circle, nottinghamforest's x.mp4 is plain grass
    monkeypatch.setattr(cal, "_read_frame", lambda v, i: img if v.parent.parent.name == "calib" else plain)
    body = {"clip": "x", "frame_index": 0, "kind": "centre"}
    assert web.post("/calibration/api/auto_points", json=body).json()["ok"] is True
    r = web.post("/calibration/api/auto_points?dataset=nottinghamforest", json=body).json()
    assert r["ok"] is False and r["reason"]
    assert web.post("/calibration/api/auto_points", json=body).json()["ok"] is True
    assert {k[0] for k in cal._AUTO_CACHE} == {"calib", "nottinghamforest"}
    # a clip only nottinghamforest has is a 404 in calib
    only = {"clip": "only_nf", "frame_index": 0, "kind": "centre"}
    assert web.post("/calibration/api/auto_points?dataset=nottinghamforest", json=only).status_code == 200
    assert web.post("/calibration/api/auto_points", json=only).status_code == 404
    assert web.post("/calibration/api/auto_points?dataset=nope", json=body).status_code == 404
    cal._AUTO_CACHE.clear()
