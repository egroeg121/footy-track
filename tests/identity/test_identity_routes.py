"""Tests for the identity review routes and append-only label store."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from footy_track.identity.labels import (
    CheckedInterval,
    PairLabel,
    TrackletRef,
    TrackletReview,
    Verdict,
)
from footy_track.identity.store import (
    append_pair_label,
    append_tracklet_review,
    load_pair_labels,
    load_tracklet_reviews,
)
from footy_track.labeller import server as labeller_server


def _t(track_id: int, clip: str = "seg000") -> TrackletRef:
    return TrackletRef(clip=clip, track_id=track_id)


# --------------------------------------------------------------------------
# store: append-only semantics
# --------------------------------------------------------------------------


def test_correction_appends_and_last_wins(tmp_path):
    """Changing your mind must never rewrite the log — the newest record wins."""
    append_pair_label(tmp_path, PairLabel(_t(1), _t(2), Verdict.SAME))
    append_pair_label(tmp_path, PairLabel(_t(1), _t(2), Verdict.DIFFERENT))

    labels = load_pair_labels(tmp_path)
    assert len(labels) == 1
    assert labels[0].verdict is Verdict.DIFFERENT
    # Both records survive on disk: the log is an audit trail.
    raw = (tmp_path / "identity_pairs.jsonl").read_text().strip().splitlines()
    assert len(raw) == 2


def test_pair_order_does_not_create_duplicates(tmp_path):
    append_pair_label(tmp_path, PairLabel(_t(1), _t(2), Verdict.SAME))
    append_pair_label(tmp_path, PairLabel(_t(2), _t(1), Verdict.SAME))
    assert len(load_pair_labels(tmp_path)) == 1


def test_torn_line_does_not_poison_the_log(tmp_path):
    """A crash mid-write must cost one record, not the whole file."""
    append_pair_label(tmp_path, PairLabel(_t(1), _t(2), Verdict.SAME))
    with (tmp_path / "identity_pairs.jsonl").open("a") as fh:
        fh.write('{"kind": "pair", "a": {"clip"\n')  # truncated JSON
    append_pair_label(tmp_path, PairLabel(_t(3), _t(4), Verdict.DIFFERENT))
    assert len(load_pair_labels(tmp_path)) == 2


def test_missing_log_reads_as_empty(tmp_path):
    assert load_pair_labels(tmp_path) == []
    assert load_tracklet_reviews(tmp_path) == []


def test_review_replaces_rather_than_accumulates_intervals(tmp_path):
    """Re-review must not let coverage drift upward through partial passes."""
    append_tracklet_review(
        tmp_path,
        TrackletReview(tracklet=_t(1), checked_intervals=[CheckedInterval(0, 9)]),
    )
    append_tracklet_review(
        tmp_path,
        TrackletReview(tracklet=_t(1), checked_intervals=[CheckedInterval(50, 54)]),
    )
    reviews = load_tracklet_reviews(tmp_path)
    assert len(reviews) == 1
    assert reviews[0].checked_frame_count() == 5, "intervals must not accumulate"


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    gt = tmp_path / "ball_gt_marks"
    gt.mkdir()
    clips = tmp_path / "clips"
    clips.mkdir()
    (tmp_path / "tracklets").mkdir()
    monkeypatch.setattr(labeller_server, "_GT_MARKS_DIR", gt)
    monkeypatch.setattr(labeller_server, "_CLIPS_DIR", clips)
    return TestClient(labeller_server.app)


def _write_tracklets(tmp_path, clip: str, rows: list[dict]) -> None:
    p = tmp_path / "tracklets" / f"{clip}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_identity_page_is_served(client):
    r = client.get("/identity")
    assert r.status_code == 200
    assert "Identity review" in r.text


def test_tracklets_groups_detections_by_track(client, tmp_path):
    _write_tracklets(
        tmp_path,
        "seg000",
        [
            {"frame_index": 0, "track_id": 1, "tags": ["player"], "confidence": 0.9},
            {"frame_index": 1, "track_id": 1, "tags": ["player"], "confidence": 0.8},
            {"frame_index": 0, "track_id": 2, "tags": ["player"], "confidence": 0.7},
        ],
    )
    # min_frames=0: this test is about grouping, not the length filter.
    body = client.get("/identity/tracklets?clip=seg000&min_frames=0").json()
    assert len(body["tracklets"]) == 2
    t1 = next(t for t in body["tracklets"] if t["track_id"] == 1)
    assert t1["n_detections"] == 2
    assert t1["start_frame"] == 0 and t1["end_frame"] == 1
    assert t1["reviewed"] is False


def test_tracklets_rejects_unsafe_clip_name(client):
    """Clip names come from the URL and must not escape the clips directory."""
    body = client.get("/identity/tracklets?clip=../../etc/passwd").json()
    assert body["tracklets"] == []
    assert "error" in body


def test_crop_rejects_unsafe_clip_name(client):
    assert client.get("/identity/crop/..%2F..%2Fetc/0.jpg").status_code in (400, 404)


def test_risky_frames_prefers_low_margin(client, tmp_path):
    _write_tracklets(
        tmp_path,
        "seg001",
        [
            {"frame_index": 0, "track_id": 1, "tags": ["player"],
             "confidence": 0.99, "association_margin": 0.99},
            {"frame_index": 5, "track_id": 1, "tags": ["player"],
             "confidence": 0.30, "association_margin": 0.01},
        ],
    )
    body = client.get("/identity/risky-frames?clip=seg001&track_id=1&k=1").json()
    assert body["frames"] == [5]


def test_pair_endpoint_records_and_clusters(client):
    r = client.post("/identity/pair", json={
        "a": {"clip": "seg000", "track_id": 1},
        "b": {"clip": "seg001", "track_id": 4},
        "verdict": "same"}).json()
    assert r["ok"] is True
    assert r["n_clusters"] == 1
    assert r["contradictions"] == []


def test_pair_endpoint_surfaces_contradictions(client):
    """A~B, B~C, A!=C must be reported, never silently resolved."""
    for a, b, v in (
        ((("seg000", 1)), ("seg000", 2), "same"),
        ((("seg000", 2)), ("seg000", 3), "same"),
        ((("seg000", 1)), ("seg000", 3), "different"),
    ):
        r = client.post("/identity/pair", json={
            "a": {"clip": a[0], "track_id": a[1]},
            "b": {"clip": b[0], "track_id": b[1]}, "verdict": v}).json()
    assert r["contradictions"], "contradiction must be surfaced to the human"
    assert client.get("/identity/clusters").json()["is_consistent"] is False


def test_pair_endpoint_rejects_bad_payload(client):
    r = client.post("/identity/pair", json={"a": {"clip": "x"}, "verdict": "same"}).json()
    assert r["ok"] is False


def test_review_endpoint_records_only_inspected_frames(client):
    r = client.post("/identity/review", json={
        "clip": "seg000", "track_id": 1,
        "checked": [[10, 10], [11, 11], [40, 40]], "split_at": []}).json()
    assert r["ok"] is True
    assert r["checked_frames"] == 3, "must not claim the whole tracklet span"
    assert r["is_pure"] is True


def test_review_with_cuts_is_not_pure(client):
    r = client.post("/identity/review", json={
        "clip": "seg000", "track_id": 2,
        "checked": [[0, 0]], "split_at": [7]}).json()
    assert r["is_pure"] is False


def test_risky_frames_returns_boxes_for_cropping(client, tmp_path):
    """Without boxes the client cannot crop and silently shows the whole frame.

    That bug rendered 12 identical wide shots of the pitch instead of one
    player, making the review pane useless while looking like it worked.
    """
    _write_tracklets(
        tmp_path,
        "seg002",
        [
            {"frame_index": 4, "track_id": 3, "tags": ["player"], "confidence": 0.9,
             "bbox": {"x": 0.11, "y": 0.22, "w": 0.03, "h": 0.09}},
        ],
    )
    body = client.get("/identity/risky-frames?clip=seg002&track_id=3&k=5").json()
    assert body["frames"] == [4]
    assert body["boxes"] == [{"x": 0.11, "y": 0.22, "w": 0.03, "h": 0.09}]
    assert len(body["boxes"]) == len(body["frames"]), "one box per returned frame"


def test_crop_draws_only_the_reviewed_box(client, tmp_path, monkeypatch):
    """A padded crop often contains several players; mark which one is tracked.

    Without it the reviewer judges continuity on whichever player is most
    visible, which is not the question being asked.
    """
    import numpy as np
    import footy_track.labeller.identity_routes as ir

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    class _Cap:
        def __init__(self, *a, **k): pass
        def set(self, *a): pass
        def read(self): return True, frame.copy()
        def release(self): pass

    monkeypatch.setattr(ir.cv2, "VideoCapture", _Cap)
    (tmp_path / "clips" / "seg009.mp4").write_bytes(b"stub")

    boxed = client.get("/identity/crop/seg009/0.jpg?x=0.4&y=0.4&w=0.05&h=0.1")
    plain = client.get("/identity/crop/seg009/0.jpg")
    assert boxed.status_code == 200 and plain.status_code == 200
    # The full-frame request must NOT be annotated; the boxed one must be.
    assert len(boxed.content) != len(plain.content)


def test_best_frame_falls_back_to_size_when_motion_unmeasurable(client, tmp_path):
    """Too short to measure motion — size alone, and say so."""
    _write_tracklets(tmp_path, "seg010", [
        {"frame_index": 1, "track_id": 5, "tags": ["player"], "confidence": 0.9,
         "bbox": {"x": 0.1, "y": 0.1, "w": 0.02, "h": 0.05}},
        {"frame_index": 9, "track_id": 5, "tags": ["player"], "confidence": 0.9,
         "bbox": {"x": 0.1, "y": 0.1, "w": 0.06, "h": 0.20}},
    ])
    b = client.get("/identity/best-frame?clip=seg010&track_id=5").json()
    assert b["heuristic"] == "size-only"
    assert b["candidates"][0]["frame"] == 9
    assert b["candidates"][0]["height_px"] == 216


def test_best_frame_prefers_player_moving_away_from_camera(client, tmp_path):
    """A big box facing the camera shows no number; orientation is what matters.

    Two equal-sized frames: one where the player moves UP-screen (away, back
    visible) and one moving DOWN-screen (toward camera, no number). The
    away-moving frame must rank first despite identical box size.
    """
    rows = []
    for f in range(40):
        y = 0.60 - f * 0.004 if f < 20 else 0.20 + (f - 20) * 0.004
        rows.append({"frame_index": f, "track_id": 5, "tags": ["player"],
                     "confidence": 0.9,
                     "bbox": {"x": 0.5, "y": round(y, 4), "w": 0.05, "h": 0.15}})
        # a second track holding still, so camera compensation sees no pan
        rows.append({"frame_index": f, "track_id": 6, "tags": ["player"],
                     "confidence": 0.9,
                     "bbox": {"x": 0.1, "y": 0.5, "w": 0.05, "h": 0.15}})
    _write_tracklets(tmp_path, "seg011", rows)
    b = client.get("/identity/best-frame?clip=seg011&track_id=5&n=3").json()
    assert b["heuristic"] == "orientation+size"
    assert b["candidates"][0]["frame"] < 20, "must prefer the away-moving phase"


def test_best_frame_compensates_for_camera_pan(client, tmp_path):
    """A pan moves every box together and must not read as running away."""
    rows = []
    for f in range(40):
        pan = -f * 0.004                      # whole scene drifts up-screen
        rows.append({"frame_index": f, "track_id": 5, "tags": ["player"],
                     "confidence": 0.9,
                     "bbox": {"x": 0.5, "y": round(0.5 + pan, 4), "w": 0.05, "h": 0.15}})
        for t in (6, 7, 8):
            rows.append({"frame_index": f, "track_id": t, "tags": ["player"],
                         "confidence": 0.9,
                         "bbox": {"x": 0.1 * t, "y": round(0.5 + pan, 4),
                                  "w": 0.05, "h": 0.15}})
    _write_tracklets(tmp_path, "seg012", rows)
    b = client.get("/identity/best-frame?clip=seg012&track_id=5&n=1").json()
    # Everything moves identically, so relative motion is zero everywhere.
    assert b["candidates"][0]["facing_away"] == 0.0


def test_review_records_a_human_read_jersey_number(client):
    r = client.post("/identity/review", json={
        "clip": "seg000", "track_id": 8, "checked": [[3, 3]],
        "split_at": [], "jersey_number": " 17 "}).json()
    assert r["jersey_number"] == "17", "must be trimmed and stored"


def test_unsure_review_is_not_pure(client):
    r = client.post("/identity/review", json={
        "clip": "seg000", "track_id": 9, "checked": [[1, 1]], "unsure": True}).json()
    assert r["unsure"] is True
    assert r["is_pure"] is False, "unsure carries no positive claim"


def test_short_tracklets_are_hidden_and_longest_come_first(client, tmp_path):
    """~1400 tracklets per clip, only ~640 usable; the rest is detection noise."""
    rows = [{"frame_index": 0, "track_id": 1, "tags": ["player"], "confidence": 0.9}]
    rows += [{"frame_index": f, "track_id": 2, "tags": ["player"], "confidence": 0.9}
             for f in range(30)]
    rows += [{"frame_index": f, "track_id": 3, "tags": ["player"], "confidence": 0.9}
             for f in range(60)]
    _write_tracklets(tmp_path, "seg020", rows)
    body = client.get("/identity/tracklets?clip=seg020&min_frames=25").json()
    assert [t["track_id"] for t in body["tracklets"]] == [3, 2], "longest first"
    assert body["hidden_short"] == 1
    assert len(client.get(
        "/identity/tracklets?clip=seg020&min_frames=0").json()["tracklets"]) == 3


def test_pages_have_no_dangling_element_or_function_refs(client):
    """The review and merge pages share one script but not their controls.

    A single reference to an element the page does not own threw a TypeError
    and aborted the whole script before loadClips() ran, so the page served
    HTTP 200 with empty dropdowns and a black video. HTTP 200 is not evidence
    that a page works.
    """
    import re

    for path in ("/identity", "/identity/merge"):
        html = client.get(path).text
        ids = set(re.findall(r'id="([A-Za-z0-9_]+)"', html))
        refs = set(re.findall(r'\$\("([A-Za-z0-9_]+)"\)\.', html))
        assert not refs - ids, f"{path} references missing elements: {sorted(refs - ids)}"
        called = set(re.findall(
            r'\b(loadPair|labelPair|refreshClusters|loadStrip|saveReview|advance'
            r'|loadTracklets|loadClips)\s*\(', html))
        defined = set(re.findall(r'function\s+(\w+)\s*\(', html))
        assert not called - defined, f"{path} calls undefined: {sorted(called - defined)}"


def test_best_frame_rejects_edge_clipped_slivers(client, tmp_path):
    """A clipped box cannot show a number, and clipping inflates 'facing away'.

    The box shrinks against the frame edge, which reads as motion, so these
    slivers outrank genuinely readable frames while being useless.
    """
    rows = []
    for f in range(40):
        # track 5: a clipped sliver hard against the right edge
        rows.append({"frame_index": f, "track_id": 5, "tags": ["player"],
                     "confidence": 0.9,
                     "bbox": {"x": 0.985, "y": 0.4, "w": 0.015, "h": 0.18}})
        # track 6: a normal, well-proportioned box moving up-screen
        rows.append({"frame_index": f, "track_id": 6, "tags": ["player"],
                     "confidence": 0.9,
                     "bbox": {"x": 0.5, "y": round(0.6 - f * 0.004, 4),
                              "w": 0.06, "h": 0.16}})
    _write_tracklets(tmp_path, "seg030", rows)
    assert client.get(
        "/identity/best-frame?clip=seg030&track_id=5").json()["candidates"] == [] or \
        client.get("/identity/best-frame?clip=seg030&track_id=5"
                   ).json()["heuristic"] == "size-only"
    good = client.get("/identity/best-frame?clip=seg030&track_id=6&n=1").json()
    assert good["candidates"], "a well-proportioned box must survive the filter"
