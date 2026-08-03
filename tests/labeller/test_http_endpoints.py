"""HTTP endpoint tests for the labeller server (src/footy_track/labeller/README.md §4, LAB-3xx).

All video IO goes through the fake cv2 layer in conftest.py; YOLO seeding is
stubbed. No real videos, no real inference.
"""

from __future__ import annotations

import json

from footy_track.labeller import server as labeller_server
from footy_track.labeller.server import (
    PROV_LABELLER,
    PROV_VITTRACK,
    PROV_YOLO,
)
from footy_track.labeller.video_utils import LabelledObject

from .conftest import FAKE_JPEG, FakeVideoCapture, load_fake_clip, make_box

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def test_root_and_main_serve_hub_page(client):
    for route in ("/", "/main"):
        r = client.get(route)
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]


def test_labeller_and_review_pages(client):
    assert client.get("/labeller").status_code == 200
    assert client.get("/object_review").status_code == 200


# ---------------------------------------------------------------------------
# /clips and /clips/status
# ---------------------------------------------------------------------------


def test_clips_lists_videos_sorted_with_marked_flag(client, clips_dir, gt_marks_dir):
    (clips_dir / "b.mp4").touch()
    (clips_dir / "a.mov").touch()
    (clips_dir / "notes.txt").touch()  # non-video ignored
    (gt_marks_dir / "a.jsonl").write_text("")

    data = client.get("/clips").json()
    assert data["dir"] == str(clips_dir)
    assert [c["name"] for c in data["clips"]] == ["a.mov", "b.mp4"]
    assert [c["marked"] for c in data["clips"]] == [True, False]


def test_clips_missing_dir_returns_empty(client, tmp_path, monkeypatch):
    monkeypatch.setattr(labeller_server, "_CLIPS_DIR", tmp_path / "nope")
    assert client.get("/clips").json() == {"clips": []}


def _sidecar_line(frame_index: int, tags: list[str], bbox=True) -> str:
    return json.dumps(
        {
            "frame_index": frame_index,
            "bbox": {"x": 0.1, "y": 0.1, "w": 0.05, "h": 0.05} if bbox else None,
            "center": {"x": 0.125, "y": 0.125} if bbox else None,
            "tags": tags,
        }
    )


def test_clips_status_complete_requires_end_reached_and_player(
    client, clips_dir, gt_marks_dir, fake_cv2
):
    FakeVideoCapture.total_frames = 100
    for name in ("done.mp4", "ball_only.mp4", "early.mp4", "unmarked.mp4"):
        (clips_dir / name).touch()
    # done: reaches within 15 frames of the end AND has a player tag.
    (gt_marks_dir / "done.jsonl").write_text(
        "\n".join(
            [
                _sidecar_line(0, ["player", "labeller"]),
                _sidecar_line(90, ["player", "labeller"]),
            ]
        )
    )
    # ball_only: reaches the end but has no player labels -> in progress.
    (gt_marks_dir / "ball_only.jsonl").write_text(
        _sidecar_line(95, ["in_play_ball", "labeller"])
    )
    # early: player labels but stops well before the end.
    (gt_marks_dir / "early.jsonl").write_text(_sidecar_line(10, ["player", "labeller"]))

    by_name = {c["name"]: c for c in client.get("/clips/status").json()["clips"]}
    assert by_name["done.mp4"]["complete"] is True
    assert by_name["done.mp4"]["marked"] is True
    assert by_name["done.mp4"]["label_count"] == 2
    assert by_name["ball_only.mp4"]["complete"] is False
    assert by_name["early.mp4"]["complete"] is False
    assert by_name["unmarked.mp4"] == {
        "name": "unmarked.mp4",
        "marked": False,
        "complete": False,
        "label_count": 0,
    }


# ---------------------------------------------------------------------------
# /session/load and /frame
# ---------------------------------------------------------------------------


def test_session_load_returns_metadata_and_restores_sidecar(
    client, fresh_session, clips_dir, gt_marks_dir, fake_cv2
):
    (gt_marks_dir / "clip.jsonl").write_text(
        "\n".join(
            [
                _sidecar_line(1, ["player", "vittrack"]),
                _sidecar_line(2, ["no_ball"], bbox=False),
            ]
        )
    )
    video = clips_dir / "clip.mp4"
    video.touch()
    FakeVideoCapture.total_frames = 5
    r = client.post("/session/load", json={"video_path": str(video)})
    assert r.status_code == 200
    assert r.json() == {"fps": 25.0, "total_frames": 5, "width": 640, "height": 360}
    assert fresh_session.timeline[1][0].model == PROV_VITTRACK
    assert fresh_session.no_ball_frames == {2}


def test_session_load_flushes_previous_clip_before_switch(
    client, fresh_session, clips_dir, gt_marks_dir, fake_cv2
):
    load_fake_clip(fresh_session, clips_dir, "first.mp4", total_frames=4)
    fresh_session.set_frame(0, [make_box(model=PROV_LABELLER)])
    load_fake_clip(fresh_session, clips_dir, "second.mp4", total_frames=4)
    # The pending edit for first.mp4 must have been flushed on switch.
    lines = (gt_marks_dir / "first.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["tags"] == ["player", PROV_LABELLER]


def test_frame_jpeg_serves_bytes_and_404s_without_video(
    client, fresh_session, clips_dir, fake_cv2
):
    assert client.get("/frame/0.jpg").status_code == 404
    load_fake_clip(fresh_session, clips_dir)
    r = client.get("/frame/0.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == FAKE_JPEG
    # Past the end of the fake clip -> read fails -> 404.
    assert client.get("/frame/999.jpg").status_code == 404


# ---------------------------------------------------------------------------
# /marks, /timeline, /next-detection
# ---------------------------------------------------------------------------


def test_marks_reports_ball_player_and_skip_sets(client, fresh_session):
    fresh_session.total_frames = 6
    fresh_session.timeline = [None] * 6
    fresh_session.timeline[0] = [make_box("in_play_ball")]
    fresh_session.timeline[1] = [make_box("player")]
    fresh_session.timeline[2] = [make_box("in_play_ball"), make_box("referee")]
    fresh_session.no_ball_frames = {4}
    fresh_session.not_broadcast_frames = {5}

    data = client.get("/marks").json()
    assert data == {
        "no_ball": [4],
        "not_broadcast": [5],
        "ball": [0, 2],
        "player": [1, 2],
    }


def test_timeline_returns_boxes_with_source(client, fresh_session):
    fresh_session.total_frames = 3
    fresh_session.timeline = [None] * 3
    fresh_session.timeline[1] = [make_box("player", PROV_YOLO, 0.7, x=0.3)]

    data = client.get("/timeline/1").json()
    assert data["idx"] == 1
    assert data["boxes"] == [
        {
            "label": "player",
            "x": 0.3,
            "y": 0.2,
            "w": 0.05,
            "h": 0.08,
            "conf": 0.7,
            "source": PROV_YOLO,
        }
    ]
    # Unpopulated / out-of-range frames -> empty list, not an error.
    assert client.get("/timeline/0").json()["boxes"] == []
    assert client.get("/timeline/99").json()["boxes"] == []


def test_next_detection_finds_next_populated_frame(client, fresh_session):
    fresh_session.total_frames = 6
    fresh_session.timeline = [None] * 6
    fresh_session.timeline[3] = [make_box()]
    fresh_session.timeline[5] = [make_box()]

    assert client.get("/next-detection/0").json() == {"idx": 3}
    assert client.get("/next-detection/3").json() == {"idx": 5}
    assert client.get("/next-detection/5").json() == {"idx": None}


# ---------------------------------------------------------------------------
# /edit and skip markers
# ---------------------------------------------------------------------------


def test_edit_stamps_labeller_and_clears_skip_markers(
    client, fresh_session, monkeypatch
):
    fresh_session.total_frames = 3
    fresh_session.timeline = [None] * 3
    fresh_session.no_ball_frames = {1}
    fresh_session.not_broadcast_frames = {1}
    flushes = []
    monkeypatch.setattr(fresh_session, "schedule_flush", lambda: flushes.append(1))

    r = client.post(
        "/edit",
        json={
            "idx": 1,
            "objects": [{"label": "player", "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}],
        },
    )
    assert r.json()["boxes"][0]["source"] == PROV_LABELLER
    assert fresh_session.timeline[1][0].model == PROV_LABELLER
    assert fresh_session.no_ball_frames == set()
    assert fresh_session.not_broadcast_frames == set()
    assert flushes  # a debounced flush was scheduled


def test_edit_with_empty_objects_keeps_skip_markers(client, fresh_session, monkeypatch):
    fresh_session.total_frames = 3
    fresh_session.timeline = [None] * 3
    fresh_session.no_ball_frames = {1}
    monkeypatch.setattr(fresh_session, "schedule_flush", lambda: None)

    client.post("/edit", json={"idx": 1, "objects": []})
    assert fresh_session.no_ball_frames == {1}  # only cleared when boxes exist


def test_edit_clamps_coordinates(client, fresh_session, monkeypatch):
    fresh_session.total_frames = 2
    fresh_session.timeline = [None] * 2
    monkeypatch.setattr(fresh_session, "schedule_flush", lambda: None)

    r = client.post(
        "/edit",
        json={
            "idx": 0,
            "objects": [{"label": "player", "x": -0.5, "y": 1.7, "w": 2.0, "h": 0.5}],
        },
    )
    b = r.json()["boxes"][0]
    assert b["x"] == 0.0
    assert b["y"] == 1.0
    assert b["w"] == 1.0


def test_no_ball_strips_ball_boxes_but_keeps_players(
    client, fresh_session, monkeypatch
):
    fresh_session.total_frames = 3
    fresh_session.timeline = [None] * 3
    fresh_session.timeline[1] = [make_box("in_play_ball"), make_box("player")]
    monkeypatch.setattr(fresh_session, "schedule_flush", lambda: None)

    r = client.post("/no-ball", json={"idx": 1})
    assert r.json() == {"idx": 1, "no_ball": True}
    assert fresh_session.no_ball_frames == {1}
    assert [b.label for b in fresh_session.timeline[1]] == ["player"]


def test_no_ball_and_not_broadcast_clear_roundtrip(client, fresh_session, monkeypatch):
    fresh_session.total_frames = 3
    fresh_session.timeline = [None] * 3
    monkeypatch.setattr(fresh_session, "schedule_flush", lambda: None)

    client.post("/not-broadcast", json={"idx": 2})
    assert fresh_session.not_broadcast_frames == {2}
    r = client.post("/not-broadcast/clear", json={"idx": 2})
    assert r.json() == {"idx": 2, "not_broadcast": False}
    assert fresh_session.not_broadcast_frames == set()

    client.post("/no-ball", json={"idx": 0})
    client.post("/no-ball/clear", json={"idx": 0})
    assert fresh_session.no_ball_frames == set()


# ---------------------------------------------------------------------------
# /autodetect
# ---------------------------------------------------------------------------


def test_autodetect_without_video_returns_empty(client, fresh_session):
    assert client.post("/autodetect", json={"frame_idx": 0}).json() == {
        "idx": 0,
        "boxes": [],
    }


def test_autodetect_merges_yolo_on_top_of_client_gt(
    client, fresh_session, clips_dir, fake_cv2, monkeypatch
):
    load_fake_clip(fresh_session, clips_dir)

    def fake_yolo(video_path, model_path, conf, w, h, iou, frame_idx):
        # One box overlapping the client's GT (suppressed), one distinct (kept).
        return [
            LabelledObject(label="player", bbox_xyxy_abs=(64.0, 72.0, 96.0, 100.8)),
            LabelledObject(label="referee", bbox_xyxy_abs=(320.0, 180.0, 352.0, 216.0)),
        ]

    monkeypatch.setattr(labeller_server, "yolo_seed_objects", fake_yolo)

    current = [{"label": "player", "x": 0.1, "y": 0.2, "w": 0.05, "h": 0.08}]
    data = client.post(
        "/autodetect", json={"frame_idx": 2, "current_boxes": current}
    ).json()

    assert data["idx"] == 2
    sources = [(b["label"], b["source"]) for b in data["boxes"]]
    # Client GT first (labeller), then the non-overlapping YOLO box only.
    assert sources == [("player", PROV_LABELLER), ("referee", PROV_YOLO)]
    # Written to the authoritative timeline too.
    assert [b.model for b in fresh_session.get_frame(2)] == [PROV_LABELLER, PROV_YOLO]


def test_autodetect_with_no_current_boxes_keeps_all_yolo(
    client, fresh_session, clips_dir, fake_cv2, monkeypatch
):
    load_fake_clip(fresh_session, clips_dir)
    monkeypatch.setattr(
        labeller_server,
        "yolo_seed_objects",
        lambda *a, **k: [
            LabelledObject(label="player", bbox_xyxy_abs=(0.0, 0.0, 64.0, 36.0))
        ],
    )
    data = client.post("/autodetect", json={"frame_idx": 0, "current_boxes": []}).json()
    assert [b["source"] for b in data["boxes"]] == [PROV_YOLO]


# ---------------------------------------------------------------------------
# /propagate
# ---------------------------------------------------------------------------


def _seed_propagation_session(session) -> None:
    session.total_frames = 6
    session.timeline = [None] * 6
    # Frame 0: corrected GT box (the propagation source).
    session.timeline[0] = [make_box("referee", PROV_LABELLER, 1.0, x=0.10)]
    # Frames 1-2: overlapping YOLO boxes with the wrong label.
    session.timeline[1] = [make_box("player", PROV_YOLO, 0.6, x=0.11)]
    session.timeline[2] = [make_box("player", PROV_YOLO, 0.6, x=0.12)]


def test_propagate_relabels_matching_yolo_boxes_forward(
    client, fresh_session, monkeypatch
):
    _seed_propagation_session(fresh_session)
    monkeypatch.setattr(fresh_session, "schedule_flush", lambda: None)

    data = client.post("/propagate", json={"frame_idx": 0, "box_idx": 0}).json()
    assert data == {"propagated_to": 2}
    for idx in (1, 2):
        box = fresh_session.timeline[idx][0]
        assert box.label == "referee"  # label corrected
        assert box.model == PROV_YOLO  # provenance NOT promoted


def test_propagate_stops_at_frame_with_existing_gt(client, fresh_session, monkeypatch):
    _seed_propagation_session(fresh_session)
    # Frame 2 now holds hand-marked GT — the walk must stop before it.
    fresh_session.timeline[2] = [make_box("player", PROV_LABELLER, 1.0, x=0.12)]
    monkeypatch.setattr(fresh_session, "schedule_flush", lambda: None)

    data = client.post("/propagate", json={"frame_idx": 0, "box_idx": 0}).json()
    assert data == {"propagated_to": 1}
    assert fresh_session.timeline[2][0].label == "player"  # untouched


def test_propagate_stops_when_track_lost(client, fresh_session, monkeypatch):
    _seed_propagation_session(fresh_session)
    # Frame 2's box is far away -> IoU below 0.3 -> track lost.
    fresh_session.timeline[2] = [make_box("player", PROV_YOLO, 0.6, x=0.80)]
    monkeypatch.setattr(fresh_session, "schedule_flush", lambda: None)

    data = client.post("/propagate", json={"frame_idx": 0, "box_idx": 0}).json()
    assert data == {"propagated_to": 1}
    assert fresh_session.timeline[2][0].label == "player"


def test_propagate_skips_empty_frames_and_continues(client, fresh_session, monkeypatch):
    _seed_propagation_session(fresh_session)
    fresh_session.timeline[1] = None  # hole in the timeline
    monkeypatch.setattr(fresh_session, "schedule_flush", lambda: None)

    data = client.post("/propagate", json={"frame_idx": 0, "box_idx": 0}).json()
    assert data == {"propagated_to": 1}  # frame 2 still reached
    assert fresh_session.timeline[2][0].label == "referee"


def test_propagate_refuses_non_labeller_source(client, fresh_session):
    fresh_session.total_frames = 3
    fresh_session.timeline = [None] * 3
    fresh_session.timeline[0] = [make_box("player", PROV_YOLO, 0.6)]

    assert client.post("/propagate", json={"frame_idx": 0, "box_idx": 0}).json() == {
        "propagated_to": 0
    }


def test_propagate_out_of_range_box_idx(client, fresh_session):
    fresh_session.total_frames = 3
    fresh_session.timeline = [None] * 3
    fresh_session.timeline[0] = [make_box(model=PROV_LABELLER)]

    assert client.post("/propagate", json={"frame_idx": 0, "box_idx": 5}).json() == {
        "propagated_to": 0
    }
