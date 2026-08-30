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
    body = client.get("/identity/tracklets?clip=seg000").json()
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
