"""Tests for tracking-by-detection over precomputed detection JSONL."""

from __future__ import annotations

import json

from footy_track.scripts.track_detections import (
    label_of,
    load_detections,
    summarise,
    track_clip,
)


def _write(tmp_path, rows):
    p = tmp_path / "dets.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _det(frame, x, label="player", conf=0.9):
    return {
        "frame_index": frame,
        "tags": [label, "rtdetr"],
        "bbox": {"x": x, "y": 0.5, "w": 0.04, "h": 0.10},
        "confidence": conf,
    }


def test_label_ignores_source_tags():
    assert label_of(["player", "rtdetr"]) == "player"
    assert label_of(["rtdetr", "referee"]) == "referee"


def test_ball_and_low_confidence_are_excluded(tmp_path):
    path = _write(
        tmp_path,
        [
            _det(0, 0.1),
            _det(0, 0.2, label="in_play_ball"),
            _det(0, 0.3, conf=0.1),
        ],
    )
    by_frame = load_detections(path, min_confidence=0.5, keep_labels=("player",))
    assert len(by_frame[0]) == 1, "ball and low-confidence rows must be dropped"


def test_torn_line_does_not_abort_the_clip(tmp_path):
    p = tmp_path / "dets.jsonl"
    p.write_text(json.dumps(_det(0, 0.1)) + "\n{ broken\n" + json.dumps(_det(1, 0.11)) + "\n")
    by_frame = load_detections(p, min_confidence=0.5, keep_labels=("player",))
    assert len(by_frame) == 2


def test_out_of_frame_box_is_clamped_not_dropped(tmp_path):
    """RT-DETR really emits x=-1.9e-05; the schema requires [0,1]."""
    path = _write(tmp_path, [{**_det(0, 0.1), "bbox": {"x": -1.9e-05, "y": 0.5, "w": 0.04, "h": 0.1}}])
    by_frame = load_detections(path, min_confidence=0.5, keep_labels=("player",))
    rows = track_clip(by_frame, iou_threshold=0.15, max_age=90)
    assert len(rows) == 1
    assert rows[0]["bbox"]["x"] == 0.0


def test_a_steadily_moving_object_keeps_one_track_id(tmp_path):
    """The basic contract: smooth motion must not fragment."""
    path = _write(tmp_path, [_det(f, 0.10 + f * 0.002) for f in range(30)])
    by_frame = load_detections(path, min_confidence=0.5, keep_labels=("player",))
    rows = track_clip(by_frame, iou_threshold=0.15, max_age=90)
    assert len({r["track_id"] for r in rows}) == 1
    assert len(rows) == 30


def test_two_separated_objects_get_distinct_ids(tmp_path):
    rows_in = []
    for f in range(20):
        rows_in.append(_det(f, 0.10 + f * 0.002))
        rows_in.append(_det(f, 0.70 + f * 0.002))
    path = _write(tmp_path, rows_in)
    by_frame = load_detections(path, min_confidence=0.5, keep_labels=("player",))
    rows = track_clip(by_frame, iou_threshold=0.15, max_age=90)
    assert len({r["track_id"] for r in rows}) == 2


def test_max_age_bridges_a_detection_dropout(tmp_path):
    """Dropouts, not bad matching, are what fragment real tracks."""
    frames = [f for f in range(20) if f not in (8, 9, 10)]
    path = _write(tmp_path, [_det(f, 0.10 + f * 0.002) for f in frames])
    by_frame = load_detections(path, min_confidence=0.5, keep_labels=("player",))
    bridged = track_clip(by_frame, iou_threshold=0.15, max_age=90)
    assert len({r["track_id"] for r in bridged}) == 1, "max_age should bridge the gap"


def test_known_quirk_tracks_do_not_age_on_empty_frames(tmp_path):
    """LapTracker does not increment track age when a frame has NO detections.

    ``update`` takes an early ``if not dets: self._age_out(...); return []``
    branch that never increments ``trk.age``, so a track survives an arbitrarily
    long fully-empty gap regardless of ``max_age``. Here a 3-frame gap is
    bridged even with ``max_age=1``, which the parameter is supposed to forbid.

    In broadcast football this rarely bites — a frame with ~20 players is never
    empty, so gaps are per-track rather than global. It matters for sparse
    single-object clips, and it means ``max_age`` cannot be trusted as an upper
    bound on how long a stale track can linger. Pinned here so the behaviour is
    recorded rather than rediscovered; fixing it is a change to LapTracker and
    should be measured against purity, not assumed to be an improvement.
    """
    frames = [f for f in range(20) if f not in (8, 9, 10)]
    path = _write(tmp_path, [_det(f, 0.10 + f * 0.002) for f in frames])
    by_frame = load_detections(path, min_confidence=0.5, keep_labels=("player",))
    rows = track_clip(by_frame, iou_threshold=0.15, max_age=1)
    assert len({r["track_id"] for r in rows}) == 1, "documents the quirk, not desired"


def test_empty_input_is_handled(tmp_path):
    path = _write(tmp_path, [])
    assert track_clip(load_detections(path, min_confidence=0.5, keep_labels=("player",)),
                      iou_threshold=0.15, max_age=90) == []
    assert summarise([]) == {"tracklets": 0}


def test_summary_reports_fragmentation(tmp_path):
    path = _write(tmp_path, [_det(f, 0.10 + f * 0.002) for f in range(30)])
    by_frame = load_detections(path, min_confidence=0.5, keep_labels=("player",))
    stats = summarise(track_clip(by_frame, iou_threshold=0.15, max_age=90))
    assert stats["tracklets"] == 1
    assert stats["median_length"] == 30
    assert stats["singleton_pct"] == 0.0
