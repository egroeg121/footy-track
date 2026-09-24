"""Ball Check API tests (src/footy_track/labeller/README.md §11, LAB-10xx).

Ball Check reads the same sidecars as review but writes only to its own
append-only verdict log; video IO for crops goes through the fake cv2 layer.
"""

from __future__ import annotations

import json

import pytest

from footy_track.labeller import ball_check

from .conftest import FAKE_JPEG


def _line(
    frame_index: int, tags: list[str], x=0.1, y=0.1, w=0.02, h=0.02, bbox=True
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


@pytest.fixture
def bc_cache(monkeypatch):
    """Isolate the Ball Check crop LRU between tests."""
    monkeypatch.setattr(ball_check, "_BC_CROP_CACHE", type(ball_check._BC_CROP_CACHE)())


def _verdict(client, clip, frame_index, box_index, verdict):
    return client.post(
        "/ball_check/verdict",
        json={
            "clip": clip,
            "frame_index": frame_index,
            "box_index": box_index,
            "verdict": verdict,
        },
    ).json()


# ---------------------------------------------------------------------------
# /ball_check/queue  (LAB-1001..1003)
# ---------------------------------------------------------------------------


def test_queue_only_machine_ball_boxes(client, clips_dir, gt_marks_dir):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(
        gt_marks_dir,
        "clip",
        [
            _line(0, ["player", "yolo"]),  # not a ball class
            _line(1, ["in_play_ball", "labeller"]),  # already human GT
            _line(2, ["in_play_ball", "yolo"]),
            _line(3, ["out_of_play_ball", "vittrack"]),
        ],
    )
    data = client.get("/ball_check/queue").json()
    assert {(i["frame_index"], i["label"]) for i in data["items"]} == {
        (2, "in_play_ball"),
        (3, "out_of_play_ball"),
    }
    assert data["remaining"] == 2
    assert data["items"][0]["image_url"].startswith("/ball_check/crop/clip/")


def test_queue_include_gt_and_clip_filters(client, clips_dir, gt_marks_dir):
    for stem in ("a", "b"):
        (clips_dir / f"{stem}.mp4").touch()
        _write_sidecar(
            gt_marks_dir,
            stem,
            [
                _line(0, ["in_play_ball", "yolo"]),
                _line(1, ["in_play_ball", "labeller"]),
            ],
        )
    all_items = client.get("/ball_check/queue").json()["items"]
    assert {i["clip"] for i in all_items} == {"a", "b"}

    one_clip = client.get("/ball_check/queue?clip=a").json()["items"]
    assert {i["clip"] for i in one_clip} == {"a"}

    with_gt = client.get("/ball_check/queue?clip=a&include_gt=true").json()["items"]
    assert {i["provenance"] for i in with_gt} == {"yolo", "labeller"}


def test_queue_order_is_stable_and_respects_limit(client, clips_dir, gt_marks_dir):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(
        gt_marks_dir, "clip", [_line(i, ["in_play_ball", "yolo"]) for i in range(8)]
    )
    first = client.get("/ball_check/queue").json()["items"]
    second = client.get("/ball_check/queue").json()["items"]
    assert [i["frame_index"] for i in first] == [i["frame_index"] for i in second]

    limited = client.get("/ball_check/queue?limit=3").json()
    assert len(limited["items"]) == 3
    assert limited["remaining"] == 8  # remaining counts the whole queue, not the page
    assert [i["frame_index"] for i in limited["items"]] == [
        i["frame_index"] for i in first[:3]
    ]


def test_queue_skips_clips_with_no_video(client, clips_dir, gt_marks_dir):
    (clips_dir / "has_video.mp4").touch()  # no file for "orphan"
    for stem in ("has_video", "orphan"):
        _write_sidecar(gt_marks_dir, stem, [_line(0, ["in_play_ball", "yolo"])])
    data = client.get("/ball_check/queue").json()
    assert [i["clip"] for i in data["items"]] == ["has_video"]
    assert data["remaining"] == 1


# ---------------------------------------------------------------------------
# Verdicts (LAB-1004..1006)
# ---------------------------------------------------------------------------


def test_verdict_appends_to_log_and_leaves_sidecar_untouched(
    client, clips_dir, gt_marks_dir
):
    (clips_dir / "clip.mp4").touch()
    sidecar = gt_marks_dir / "clip.jsonl"
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["in_play_ball", "yolo"])])
    before = sidecar.read_text()

    assert _verdict(client, "clip", 0, 0, "not_ball")["ok"] is True

    assert sidecar.read_text() == before  # hand labels are never rewritten here
    log = gt_marks_dir / "ball_checks" / "clip.jsonl"
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["verdict"] == "not_ball"
    assert (rec["clip"], rec["frame_index"], rec["box_index"]) == ("clip", 0, 0)
    assert isinstance(rec["ts"], float)


def test_judged_boxes_leave_the_queue(client, clips_dir, gt_marks_dir):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(
        gt_marks_dir,
        "clip",
        [_line(0, ["in_play_ball", "yolo"]), _line(1, ["in_play_ball", "yolo"])],
    )
    _verdict(client, "clip", 0, 0, "ball")
    data = client.get("/ball_check/queue").json()
    assert [i["frame_index"] for i in data["items"]] == [1]
    assert (data["remaining"], data["judged"]) == (1, 1)


def test_latest_verdict_wins(client, clips_dir, gt_marks_dir):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["in_play_ball", "yolo"])])
    _verdict(client, "clip", 0, 0, "ball")
    _verdict(client, "clip", 0, 0, "not_ball")
    stats = client.get("/ball_check/stats").json()
    assert stats["counts"]["ball"] == 0
    assert stats["counts"]["not_ball"] == 1


def test_verdict_rejects_unknown_value(client, gt_marks_dir):
    out = _verdict(client, "clip", 0, 0, "maybe")
    assert out["ok"] is False
    assert not (gt_marks_dir / "ball_checks").exists()


def test_verdict_rejects_missing_fields(client, gt_marks_dir):
    out = client.post("/ball_check/verdict", json={"verdict": "ball"}).json()
    assert out["ok"] is False


def test_undo_returns_the_card_to_the_queue(client, clips_dir, gt_marks_dir):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["in_play_ball", "yolo"])])
    _verdict(client, "clip", 0, 0, "ball")
    assert client.get("/ball_check/queue").json()["remaining"] == 0

    assert (
        client.post(
            "/ball_check/undo",
            json={"clip": "clip", "frame_index": 0, "box_index": 0},
        ).json()["ok"]
        is True
    )

    data = client.get("/ball_check/queue").json()
    assert (data["remaining"], data["judged"]) == (1, 0)
    # append-only: the original verdict line is still on disk
    log = (gt_marks_dir / "ball_checks" / "clip.jsonl").read_text().splitlines()
    assert len(log) == 2


def test_undo_without_verdicts_is_an_error(client, gt_marks_dir):
    out = client.post(
        "/ball_check/undo", json={"clip": "clip", "frame_index": 0, "box_index": 0}
    ).json()
    assert out["ok"] is False


# ---------------------------------------------------------------------------
# /ball_check/stats  (LAB-1007)
# ---------------------------------------------------------------------------


def test_stats_precision_excludes_unsure(client, clips_dir, gt_marks_dir):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(
        gt_marks_dir, "clip", [_line(i, ["in_play_ball", "yolo"]) for i in range(4)]
    )
    for box_idx, verdict in enumerate(["ball", "ball", "not_ball", "unsure"]):
        _verdict(client, "clip", box_idx, 0, verdict)
    stats = client.get("/ball_check/stats").json()
    assert stats["counts"] == {
        "ball": 2,
        "not_ball": 1,
        "box_off": 0,
        "corrected": 0,
        "unsure": 1,
    }
    assert stats["precision"] == pytest.approx(2 / 3, abs=1e-4)


def test_stats_box_off_counts_as_found_but_not_clean(client, clips_dir, gt_marks_dir):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(
        gt_marks_dir, "clip", [_line(i, ["in_play_ball", "yolo"]) for i in range(4)]
    )
    for box_idx, verdict in enumerate(["ball", "box_off", "corrected", "not_ball"]):
        _verdict(client, "clip", box_idx, 0, verdict)
    stats = client.get("/ball_check/stats").json()
    # found: ball + box_off + corrected = 3 of 4 decided; clean drops box_off.
    assert stats["precision"] == pytest.approx(3 / 4, abs=1e-4)
    assert stats["clean_precision"] == pytest.approx(2 / 4, abs=1e-4)


def test_crop_meta_gives_window_and_box(client, clips_dir, gt_marks_dir, fake_cv2):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["in_play_ball", "yolo"])])
    meta = client.get("/ball_check/crop_meta/clip/0/0").json()
    assert meta["ok"] is True
    assert meta["frame"] == {"w": 640, "h": 360}
    w = meta["window"]
    assert 0 <= w["x1"] < w["x2"] <= 640
    assert 0 <= w["y1"] < w["y2"] <= 360
    assert meta["bbox"]["w"] == pytest.approx(0.02)


def test_crop_meta_404s_softly_for_missing_clip(client, clips_dir, gt_marks_dir):
    assert client.get("/ball_check/crop_meta/nope/0/0").json()["ok"] is False


def test_stats_precision_none_before_any_decision(client, gt_marks_dir):
    assert client.get("/ball_check/stats").json()["precision"] is None


# ---------------------------------------------------------------------------
# /ball_check/crop  (LAB-1008..1009)
# ---------------------------------------------------------------------------


def test_crop_returns_jpeg_and_caches(
    client, clips_dir, gt_marks_dir, fake_cv2, bc_cache
):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["in_play_ball", "yolo"])])
    res = client.get("/ball_check/crop/clip/0/0.jpg")
    assert res.status_code == 200
    assert res.content == FAKE_JPEG
    assert (
        "sidecar",
        "clip",
        0,
        0,
        round(ball_check._PAD_FACTOR, 2),
        True,
    ) in ball_check._BC_CROP_CACHE

    # A different pad is a different cache entry, not a stale hit.
    client.get("/ball_check/crop/clip/0/0.jpg?pad=20")
    assert ("sidecar", "clip", 0, 0, 20.0, True) in ball_check._BC_CROP_CACHE

    # reticle=false is its own entry: the page draws its own draggable box.
    client.get("/ball_check/crop/clip/0/0.jpg?reticle=false")
    assert (
        "sidecar",
        "clip",
        0,
        0,
        round(ball_check._PAD_FACTOR, 2),
        False,
    ) in ball_check._BC_CROP_CACHE


def test_crop_404s_for_missing_clip_or_box(client, clips_dir, gt_marks_dir, fake_cv2):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["in_play_ball", "yolo"])])
    assert client.get("/ball_check/crop/nosuch/0/0.jpg").status_code == 404
    assert client.get("/ball_check/crop/clip/0/7.jpg").status_code == 404


def test_crop_window_floors_tiny_boxes_and_clamps_to_edges():
    # An 11 px ball at 1280 wide: proportional padding alone is unreadable, so
    # the window is floored to _MIN_CONTEXT_PX.
    x1, y1, x2, y2 = ball_check._crop_window(0.5, 0.5, 0.0086, 0.015, 1280, 720, 6.0)
    assert (x2 - x1) >= ball_check._MIN_CONTEXT_PX
    # Hard against the top-left corner: no negative coordinates.
    x1, y1, x2, y2 = ball_check._crop_window(0.0, 0.0, 0.0086, 0.015, 1280, 720, 6.0)
    assert (x1, y1) == (0, 0)
    assert x2 > 0 and y2 > 0


def test_pad_is_clamped_to_max(client, clips_dir, gt_marks_dir, fake_cv2, bc_cache):
    (clips_dir / "clip.mp4").touch()
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["in_play_ball", "yolo"])])
    assert client.get("/ball_check/crop/clip/0/0.jpg?pad=9999").status_code == 200
    assert (
        "sidecar",
        "clip",
        0,
        0,
        ball_check._MAX_PAD_FACTOR,
        True,
    ) in ball_check._BC_CROP_CACHE


# ---------------------------------------------------------------------------
# Page (LAB-1010)
# ---------------------------------------------------------------------------


def test_ball_check_page_served_and_linked_from_hub(client):
    page = client.get("/ball_check")
    assert page.status_code == 200
    assert "Ball Check" in page.text
    assert 'name="viewport"' in page.text  # mobile-first: must scale to a phone
    assert 'href="/ball_check"' in client.get("/").text


def test_queue_candidates_source_and_namespaced_verdicts(
    client, clips_dir, gt_marks_dir, cand_dir
):
    (clips_dir / "clip.mp4").touch()
    # Same (clip, frame, index) in both sources: the namespaces must not collide.
    _write_sidecar(gt_marks_dir, "clip", [_line(0, ["in_play_ball", "yolo"])])
    (cand_dir / "clip.jsonl").write_text(
        json.dumps(
            {
                "frame_index": 0,
                "bbox": {"x": 0.2, "y": 0.2, "w": 0.01, "h": 0.01},
                "tags": ["in_play_ball", "rtdetr"],
                "confidence": 0.22,
            }
        )
        + "\n"
    )
    data = client.get("/ball_check/queue").json()
    assert data["source"] == "candidates"  # auto prefers candidates
    assert data["items"][0]["source"] == "cand"
    assert data["items"][0]["confidence"] == pytest.approx(0.22)
    assert "tau" in data and "bands" in data

    client.post(
        "/ball_check/verdict",
        json={
            "clip": "clip",
            "frame_index": 0,
            "box_index": 0,
            "source": "cand",
            "confidence": 0.22,
            "verdict": "ball",
        },
    )
    # The candidate is judged; the identically-keyed sidecar box is not.
    assert client.get("/ball_check/queue").json()["remaining"] == 0
    sidecar = client.get("/ball_check/queue?source=sidecar").json()
    assert sidecar["remaining"] == 1
