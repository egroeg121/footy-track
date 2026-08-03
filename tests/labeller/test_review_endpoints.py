"""Review API tests (src/footy_track/labeller/README.md §5, LAB-4xx).

The review endpoints operate directly on the JSONL sidecars; video IO for
crops/frames goes through the fake cv2 layer.
"""

from __future__ import annotations

import json

import pytest

from footy_track.labeller.server import PROV_LABELLER
from footy_track.schema import FrameDetections, ObjectDetection

from .conftest import FAKE_JPEG


def _line(
    frame_index: int, tags: list[str], x=0.1, y=0.1, w=0.05, h=0.05, bbox=True
) -> str:
    return json.dumps(
        {
            "frame_index": frame_index,
            "bbox": {"x": x, "y": y, "w": w, "h": h} if bbox else None,
            "center": {"x": x + w / 2, "y": y + h / 2} if bbox else None,
            "tags": tags,
        }
    )


def _write_sidecar(gt_marks_dir, stem: str, lines: list[str]) -> None:
    (gt_marks_dir / f"{stem}.jsonl").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# /review/queue
# ---------------------------------------------------------------------------


def test_queue_orders_machine_before_gt_and_by_confidence(
    client, clips_dir, gt_marks_dir
):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(
        gt_marks_dir,
        "clip",
        [
            _line(0, ["player", "labeller"], x=0.1),
            _line(1, ["player", "yolo"], x=0.3),
        ],
    )
    items = client.get("/review/queue").json()["items"]
    assert [(i["provenance"], i["confidence"]) for i in items] == [
        ("yolo", 0.5),  # machine output reviewed first
        ("labeller", 1.0),
    ]
    assert items[0]["image_url"] == "/review/crop/clip/1/0.jpg"


def test_queue_rare_classes_weighted_before_players(client, clips_dir, gt_marks_dir):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(
        gt_marks_dir,
        "clip",
        [
            _line(0, ["player", "yolo"], x=0.1),
            _line(1, ["in_play_ball", "yolo"], x=0.3),
        ],
    )
    labels = [i["label"] for i in client.get("/review/queue").json()["items"]]
    assert labels == ["in_play_ball", "player"]  # ball weighted up at equal conf


def test_queue_excludes_skip_markers_and_dedups_by_iou(client, clips_dir, gt_marks_dir):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(
        gt_marks_dir,
        "clip",
        [
            _line(0, ["no_ball"], bbox=False),
            _line(1, ["player", "yolo"], x=0.100),
            _line(1, ["player", "yolo"], x=0.101),  # IoU > 0.85 vs previous -> deduped
            _line(1, ["player", "yolo"], x=0.700),  # distinct -> kept
        ],
    )
    data = client.get("/review/queue").json()
    assert data["total"] == 2
    # box_index numbering counts ALL box lines in the file, dedup only affects
    # what is queued — so the surviving distinct box is index 2, not 1.
    assert sorted(i["box_index"] for i in data["items"]) == [0, 2]


def test_queue_vittrack_tag_reported_as_labeller_current_behavior(
    client, clips_dir, gt_marks_dir
):
    """Pins OPEN-3 in the spec: the review provenance tag set
    omits 'vittrack', so vittrack boxes surface with provenance 'labeller'
    (while still getting machine confidence 0.5). Behavior-preserving pin."""
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["player", "vittrack"])])
    item = client.get("/review/queue").json()["items"][0]
    assert item["provenance"] == "labeller"
    assert item["confidence"] == 0.5


def test_queue_sam3_legacy_tag_round_trips(client, clips_dir, gt_marks_dir):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["player", "sam3"])])
    item = client.get("/review/queue").json()["items"][0]
    assert item["provenance"] == "sam3"
    assert item["confidence"] == 0.5


# ---------------------------------------------------------------------------
# /review/crop and /review/frame
# ---------------------------------------------------------------------------


def test_crop_serves_jpeg_and_caches(
    client, clips_dir, gt_marks_dir, fake_cv2, crop_cache
):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(gt_marks_dir, "clip", [_line(3, ["player", "yolo"])])

    r = client.get("/review/crop/clip/3/0.jpg")
    assert r.status_code == 200
    assert r.content == FAKE_JPEG
    assert ("clip", 3, 0) in crop_cache
    # Second hit is served from cache (no video needed at all).
    (clips_dir / "clip.mp4").unlink()
    assert client.get("/review/crop/clip/3/0.jpg").status_code == 200


def test_crop_404s_for_missing_video_or_box(
    client, clips_dir, gt_marks_dir, fake_cv2, crop_cache
):
    assert client.get("/review/crop/nope/0/0.jpg").status_code == 404
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["player", "yolo"])])
    assert client.get("/review/crop/clip/0/5.jpg").status_code == 404  # box oob


def test_full_frame_endpoint(client, clips_dir, gt_marks_dir, fake_cv2):
    assert client.get("/review/frame/nope/0.jpg").status_code == 404
    (clips_dir / "clip.mp4").touch()
    r = client.get("/review/frame/clip/2.jpg")
    assert r.status_code == 200
    assert r.content == FAKE_JPEG


# ---------------------------------------------------------------------------
# /review/correct — GT stamping
# ---------------------------------------------------------------------------


def test_correct_rewrites_line_stamped_labeller(
    client, clips_dir, gt_marks_dir, crop_cache
):
    _write_sidecar(
        gt_marks_dir,
        "clip",
        [
            _line(0, ["no_ball"], bbox=False),  # skip markers don't shift box_index
            _line(5, ["player", "yolo"], x=0.2),
            _line(5, ["player", "yolo"], x=0.6),
        ],
    )
    r = client.post(
        "/review/correct",
        json={
            "clip": "clip",
            "frame_index": 5,
            "box_index": 1,
            "label": "referee",
            "bbox": {"x": 0.61, "y": 0.11, "w": 0.04, "h": 0.06},
        },
    )
    assert r.json() == {"ok": True}
    lines = [
        json.loads(x) for x in (gt_marks_dir / "clip.jsonl").read_text().splitlines()
    ]
    assert lines[0]["tags"] == ["no_ball"]  # untouched
    assert lines[1]["tags"] == ["player", "yolo"]  # untouched
    corrected = lines[2]
    assert corrected["tags"] == ["referee", PROV_LABELLER]  # GT promotion
    assert corrected["bbox"] == {"x": 0.61, "y": 0.11, "w": 0.04, "h": 0.06}
    assert corrected["center"]["x"] == 0.61 + 0.04 / 2


def test_correct_clamps_bbox_to_unit_square(
    client, clips_dir, gt_marks_dir, crop_cache
):
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["player", "yolo"])])
    client.post(
        "/review/correct",
        json={
            "clip": "clip",
            "frame_index": 0,
            "box_index": 0,
            "label": "player",
            "bbox": {"x": 0.9, "y": -0.2, "w": 0.5, "h": 0.3},
        },
    )
    b = json.loads((gt_marks_dir / "clip.jsonl").read_text().splitlines()[0])["bbox"]
    assert b["x"] == pytest.approx(0.9)
    assert b["y"] == pytest.approx(0.0)
    assert b["w"] == pytest.approx(0.1)  # clamped so x + w <= 1
    assert b["h"] == pytest.approx(0.3)


def test_correct_invalidates_crop_cache(
    client, clips_dir, gt_marks_dir, fake_cv2, crop_cache
):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["player", "yolo"])])
    client.get("/review/crop/clip/0/0.jpg")
    assert ("clip", 0, 0) in crop_cache
    client.post(
        "/review/correct",
        json={
            "clip": "clip",
            "frame_index": 0,
            "box_index": 0,
            "label": "player",
            "bbox": {"x": 0.1, "y": 0.1, "w": 0.05, "h": 0.05},
        },
    )
    assert ("clip", 0, 0) not in crop_cache


def test_correct_error_shapes(client, clips_dir, gt_marks_dir):
    r = client.post(
        "/review/correct",
        json={
            "clip": "missing",
            "frame_index": 0,
            "box_index": 0,
            "label": "x",
            "bbox": {"x": 0, "y": 0, "w": 0.1, "h": 0.1},
        },
    )
    assert r.json() == {"ok": False, "error": "clip not found"}
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["player", "yolo"])])
    r = client.post(
        "/review/correct",
        json={
            "clip": "clip",
            "frame_index": 0,
            "box_index": 9,
            "label": "x",
            "bbox": {"x": 0, "y": 0, "w": 0.1, "h": 0.1},
        },
    )
    assert r.json() == {"ok": False, "error": "box_index out of range"}


# ---------------------------------------------------------------------------
# /review/delete
# ---------------------------------------------------------------------------


def test_delete_removes_only_the_target_line(
    client, clips_dir, gt_marks_dir, crop_cache
):
    _write_sidecar(
        gt_marks_dir,
        "clip",
        [
            _line(5, ["player", "yolo"], x=0.2),
            _line(5, ["referee", "yolo"], x=0.6),
            _line(6, ["player", "labeller"], x=0.4),
        ],
    )
    r = client.post(
        "/review/delete", json={"clip": "clip", "frame_index": 5, "box_index": 0}
    )
    assert r.json() == {"ok": True}
    lines = [
        json.loads(x) for x in (gt_marks_dir / "clip.jsonl").read_text().splitlines()
    ]
    assert len(lines) == 2
    assert lines[0]["tags"] == ["referee", "yolo"]
    assert lines[1]["frame_index"] == 6


def test_delete_error_shapes(client, clips_dir, gt_marks_dir):
    assert client.post(
        "/review/delete", json={"clip": "missing", "frame_index": 0, "box_index": 0}
    ).json() == {"ok": False, "error": "clip not found"}
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["player", "yolo"])])
    assert client.post(
        "/review/delete", json={"clip": "clip", "frame_index": 0, "box_index": 3}
    ).json() == {"ok": False, "error": "box_index out of range"}


# ---------------------------------------------------------------------------
# /review/yolo — stubbed detector
# ---------------------------------------------------------------------------


def test_review_yolo_returns_rounded_boxes(
    client, clips_dir, gt_marks_dir, fake_cv2, monkeypatch
):
    (clips_dir / "clip.mp4").touch()

    class _FakeDetector:
        def predict_from_path(self, _path):
            return FrameDetections(
                uri="fake",
                width=640,
                height=360,
                detections=[
                    ObjectDetection(
                        label="player",
                        confidence=0.87654,
                        x=0.12345,
                        y=0.2,
                        w=0.1,
                        h=0.2,
                        model="yolo",
                    )
                ],
            )

    monkeypatch.setattr(
        "footy_track.detectors.ultralytics.get_current_best_detector",
        lambda min_confidence: _FakeDetector(),
    )
    data = client.post("/review/yolo", json={"clip": "clip", "frame_index": 1}).json()
    assert data["ok"] is True
    (box,) = data["boxes"]
    assert box["label"] == "player"
    assert box["confidence"] == pytest.approx(0.877, abs=1e-3)
    assert box["x"] == pytest.approx(0.1235, abs=1e-4)  # rounded to 4 dp
    assert box["y"] == pytest.approx(0.2)


def test_review_yolo_missing_video(client, clips_dir, gt_marks_dir):
    assert client.post(
        "/review/yolo", json={"clip": "nope", "frame_index": 0}
    ).json() == {
        "ok": False,
        "error": "video not found",
        "boxes": [],
    }
