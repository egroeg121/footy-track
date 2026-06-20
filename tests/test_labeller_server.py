"""Tests for the SAM3 labeller FastAPI server (ft-b68).

Covers:
- Endpoint happy paths: /, /session/load, /frame/{idx}.jpg, /timeline/{idx}, /edit
- Error paths: missing clip, bad frame index
- Session timeline state: boxes persist, provenance merging
- No SAM3 or YOLO models are loaded; BackgroundLabeller is mocked.
"""

from __future__ import annotations

import pathlib
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

# Reset session state between tests so they don't share global SERVER state.
import footy_track.labeller.server as _server_module
from footy_track.labeller.server import PROV_LABELLER, PROV_SAM3, Session, app
from footy_track.labeller.video_utils import LabelledObject
from footy_track.schema import ObjectDetection

_VIDEO = pathlib.Path(__file__).parent / "data" / "video" / "test_tiny.mp4"


@pytest.fixture()
def client():
    """Fresh TestClient with a clean Session for each test."""
    _server_module.SESSION = Session()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_index_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "SAM3 Video Labeller" in r.text


# ---------------------------------------------------------------------------
# POST /session/load
# ---------------------------------------------------------------------------


def test_load_session_happy_path(client):
    r = client.post("/session/load", json={"video_path": str(_VIDEO)})
    assert r.status_code == 200
    data = r.json()
    assert data["total_frames"] == 10
    assert data["fps"] == pytest.approx(10.0)
    assert data["width"] == 64
    assert data["height"] == 64


def test_load_session_missing_file(client):
    with pytest.raises(FileNotFoundError):
        client.post("/session/load", json={"video_path": "/nonexistent/clip.mp4"})


# ---------------------------------------------------------------------------
# GET /frame/{idx}.jpg
# ---------------------------------------------------------------------------


def test_get_frame_returns_jpeg(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    r = client.get("/frame/0.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert len(r.content) > 100


def test_get_frame_last_frame(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    r = client.get("/frame/9.jpg")
    assert r.status_code == 200


def test_get_frame_out_of_bounds(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    r = client.get("/frame/999.jpg")
    assert r.status_code == 404


def test_get_frame_before_load(client):
    r = client.get("/frame/0.jpg")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /timeline/{idx}
# ---------------------------------------------------------------------------


def test_timeline_empty_frame(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    r = client.get("/timeline/0")
    assert r.status_code == 200
    assert r.json() == {"idx": 0, "boxes": []}


def test_timeline_after_edit(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    boxes = [{"label": "ball", "x": 0.4, "y": 0.3, "w": 0.1, "h": 0.1, "conf": 1.0}]
    client.post("/edit", json={"idx": 0, "objects": boxes})
    r = client.get("/timeline/0")
    assert r.status_code == 200
    result = r.json()
    assert len(result["boxes"]) == 1
    assert result["boxes"][0]["label"] == "ball"


# ---------------------------------------------------------------------------
# POST /edit
# ---------------------------------------------------------------------------


def test_edit_frame_round_trip(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    boxes = [
        {"label": "ball", "x": 0.4, "y": 0.3, "w": 0.1, "h": 0.1, "conf": 1.0},
        {"label": "player", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.4, "conf": 0.9},
    ]
    r = client.post("/edit", json={"idx": 2, "objects": boxes})
    assert r.status_code == 200
    data = r.json()
    assert data["idx"] == 2
    assert len(data["boxes"]) == 2
    labels = {b["label"] for b in data["boxes"]}
    assert labels == {"ball", "player"}


def test_edit_frame_provenance_is_labeller(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    boxes = [{"label": "ball", "x": 0.5, "y": 0.5, "w": 0.05, "h": 0.05, "conf": 1.0}]
    r = client.post("/edit", json={"idx": 0, "objects": boxes})
    data = r.json()
    assert data["boxes"][0]["source"] == "labeller"


def test_edit_overwrite_replaces_boxes(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    boxes_v1 = [
        {"label": "ball", "x": 0.5, "y": 0.5, "w": 0.05, "h": 0.05, "conf": 1.0}
    ]
    client.post("/edit", json={"idx": 0, "objects": boxes_v1})
    boxes_v2 = [
        {"label": "player", "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.4, "conf": 0.9},
        {"label": "player", "x": 0.6, "y": 0.1, "w": 0.2, "h": 0.4, "conf": 0.9},
    ]
    r = client.post("/edit", json={"idx": 0, "objects": boxes_v2})
    data = r.json()
    assert len(data["boxes"]) == 2
    assert all(b["label"] == "player" for b in data["boxes"])


def test_edit_empty_clears_frame(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    boxes = [{"label": "ball", "x": 0.5, "y": 0.5, "w": 0.05, "h": 0.05, "conf": 1.0}]
    client.post("/edit", json={"idx": 1, "objects": boxes})
    r = client.post("/edit", json={"idx": 1, "objects": []})
    data = r.json()
    assert data["boxes"] == []


# ---------------------------------------------------------------------------
# Coordinate clamping (POST /edit)
# ---------------------------------------------------------------------------


def test_edit_clips_out_of_range_coords(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    boxes = [{"label": "ball", "x": -0.1, "y": 1.5, "w": 0.05, "h": 0.05, "conf": 1.0}]
    r = client.post("/edit", json={"idx": 0, "objects": boxes})
    b = r.json()["boxes"][0]
    assert b["x"] == pytest.approx(0.0)
    assert b["y"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# POST /autodetect (mock YOLO to avoid loading a real model)
# ---------------------------------------------------------------------------


def test_autodetect_no_video(client):
    r = client.post("/autodetect", json={"frame_idx": 0, "conf": 0.35, "iou": 0.5})
    assert r.status_code == 200
    assert r.json() == {"idx": 0, "boxes": []}


def test_autodetect_mock_yolo(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})

    def _fake_yolo_seed(video_path, model_path, conf, w, h, iou, idx):
        return [LabelledObject(label="ball", bbox_xyxy_abs=(10.0, 10.0, 20.0, 20.0))]

    with patch("footy_track.labeller.server.yolo_seed_objects", _fake_yolo_seed):
        r = client.post("/autodetect", json={"frame_idx": 0, "conf": 0.35, "iou": 0.5})
    assert r.status_code == 200
    data = r.json()
    assert len(data["boxes"]) == 1
    assert data["boxes"][0]["label"] == "ball"
    assert data["boxes"][0]["source"] == "yolo"


def test_autodetect_preserves_labeller_ground_truth(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    gt_box = [
        {"label": "referee", "x": 0.8, "y": 0.8, "w": 0.05, "h": 0.1, "conf": 1.0}
    ]
    client.post("/edit", json={"idx": 0, "objects": gt_box})

    def _fake_yolo_seed(video_path, model_path, conf, w, h, iou, idx):
        return [LabelledObject(label="ball", bbox_xyxy_abs=(10.0, 10.0, 20.0, 20.0))]

    with patch("footy_track.labeller.server.yolo_seed_objects", _fake_yolo_seed):
        r = client.post("/autodetect", json={"frame_idx": 0, "conf": 0.35, "iou": 0.5})
    sources = {b["source"] for b in r.json()["boxes"]}
    labels = {b["label"] for b in r.json()["boxes"]}
    assert "labeller" in sources
    assert "yolo" in sources
    assert "referee" in labels
    assert "ball" in labels


# ---------------------------------------------------------------------------
# Session.merge_propagated — labeller ground truth survives propagation
# ---------------------------------------------------------------------------


def test_merge_propagated_keeps_labeller_boxes(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    gt = [{"label": "referee", "x": 0.8, "y": 0.8, "w": 0.05, "h": 0.1, "conf": 1.0}]
    client.post("/edit", json={"idx": 3, "objects": gt})

    sam3_box = ObjectDetection(
        label="ball", confidence=0.9, x=0.4, y=0.3, w=0.05, h=0.05, model=PROV_SAM3
    )
    _server_module.SESSION.merge_propagated(3, [sam3_box])

    r = client.get("/timeline/3")
    boxes = r.json()["boxes"]
    sources = {b["source"] for b in boxes}
    assert PROV_LABELLER in sources
    assert PROV_SAM3 in sources


# ---------------------------------------------------------------------------
# Frame navigation bounds
# ---------------------------------------------------------------------------


def test_timeline_out_of_bounds_frame(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    r = client.get("/timeline/999")
    assert r.status_code == 200
    assert r.json()["boxes"] == []


def test_timeline_negative_frame(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    r = client.get("/timeline/-1")
    assert r.status_code == 200
    assert r.json()["boxes"] == []


# ---------------------------------------------------------------------------
# Reload preserves nothing (fresh session on load)
# ---------------------------------------------------------------------------


def test_reload_clears_timeline(client):
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    boxes = [{"label": "ball", "x": 0.4, "y": 0.3, "w": 0.1, "h": 0.1, "conf": 1.0}]
    client.post("/edit", json={"idx": 0, "objects": boxes})
    client.post("/session/load", json={"video_path": str(_VIDEO)})
    r = client.get("/timeline/0")
    assert r.json()["boxes"] == []
