"""Round-trip fidelity tests for Session's JSONL sidecar (server.py).

Pins down the label-hierarchy requirements from
``src/footy_track/labeller/README.md``:

- Requirement #2: everything the UI shows must survive save -> reload with
  identical `model` tags, labels, and geometry.
- "On clip load, the full JSONL sidecar is restored into the timeline with
  the original model tag preserved (machine boxes must not be promoted to
  labeller GT by a save/reload cycle)."
- Restored labeller boxes get confidence 1.0; restored machine boxes get 0.5
  (``_load_existing_marks`` hardcodes this since raw confidence isn't
  persisted in the sidecar).

These tests bypass ``Session.load`` (which opens a real video via cv2) by
constructing Session state directly and driving ``_do_flush`` /
``_load_existing_marks`` against a monkeypatched ``_GT_MARKS_DIR``.
"""

from __future__ import annotations

import json

import pytest

from footy_track.labeller import server as labeller_server
from footy_track.labeller.server import (
    PROV_LABELLER,
    PROV_SAM3,
    PROV_VITTRACK,
    PROV_YOLO,
    Session,
)
from footy_track.schema import ObjectDetection


@pytest.fixture
def gt_marks_dir(tmp_path, monkeypatch):
    """Redirect the module-level GT-marks dir to a tmp dir for this test."""
    d = tmp_path / "ball_gt_marks"
    monkeypatch.setattr(labeller_server, "_GT_MARKS_DIR", d)
    return d


class _StemPath:
    """Minimal stand-in for a pathlib.Path exposing only `.stem`."""

    def __init__(self, stem: str) -> None:
        self.stem = stem


def _make_session(video_stem: str, total_frames: int) -> Session:
    """Build a Session with bare-bones state, bypassing Session.load (no cv2).

    _do_flush / _load_existing_marks only ever touch `video_path.stem`, so a
    minimal stand-in is enough to avoid opening a real video file.
    """
    session = Session()
    session.video_path = _StemPath(video_stem)
    session.total_frames = total_frames
    session.timeline = [None] * total_frames
    session.no_ball_frames = set()
    session.not_broadcast_frames = set()
    return session


def _box(label: str, model: str, confidence: float, x: float) -> ObjectDetection:
    return ObjectDetection(
        label=label,
        confidence=confidence,
        x=x,
        y=0.2,
        w=0.05,
        h=0.08,
        model=model,
    )


# ---------------------------------------------------------------------------
# 1. Sidecar round-trip preserves model tags
# ---------------------------------------------------------------------------


def test_roundtrip_preserves_model_tags_label_and_geometry(gt_marks_dir):
    session = _make_session("clip_a", total_frames=5)

    original = {
        0: _box("in_play_ball", PROV_LABELLER, 1.0, x=0.10),
        1: _box("player", PROV_VITTRACK, 0.9, x=0.20),
        2: _box("player", PROV_YOLO, 0.8, x=0.30),
        3: _box("out_of_play_ball", PROV_SAM3, 0.7, x=0.40),
    }
    for idx, box in original.items():
        session.timeline[idx] = [box]

    session._do_flush()

    # Sanity: file was written with 4 lines.
    out_path = gt_marks_dir / "clip_a.jsonl"
    assert out_path.exists()
    lines = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
    assert len(lines) == 4

    # Restore into a fresh session against the same GT-marks dir.
    restored = _make_session("clip_a", total_frames=5)
    restored._load_existing_marks()

    for idx, orig_box in original.items():
        frame = restored.timeline[idx]
        assert frame is not None, f"frame {idx} missing after restore"
        assert len(frame) == 1
        got = frame[0]
        assert got.model == orig_box.model, (
            f"frame {idx}: expected model tag {orig_box.model!r}, got {got.model!r} "
            "-- machine boxes must not be promoted to labeller on reload"
        )
        assert got.label == orig_box.label
        assert got.x == pytest.approx(orig_box.x)
        assert got.y == pytest.approx(orig_box.y)
        assert got.w == pytest.approx(orig_box.w)
        assert got.h == pytest.approx(orig_box.h)


def test_roundtrip_confidence_by_provenance(gt_marks_dir):
    """Restored labeller boxes get confidence 1.0; machine boxes get 0.5."""
    session = _make_session("clip_b", total_frames=4)
    session.timeline[0] = [_box("player", PROV_LABELLER, 1.0, x=0.1)]
    session.timeline[1] = [_box("player", PROV_VITTRACK, 0.42, x=0.2)]
    session.timeline[2] = [_box("player", PROV_YOLO, 0.99, x=0.3)]
    session.timeline[3] = [_box("in_play_ball", PROV_SAM3, 0.13, x=0.4)]
    session._do_flush()

    restored = _make_session("clip_b", total_frames=4)
    restored._load_existing_marks()

    assert restored.timeline[0][0].confidence == pytest.approx(1.0)
    assert restored.timeline[1][0].confidence == pytest.approx(0.5)
    assert restored.timeline[2][0].confidence == pytest.approx(0.5)
    assert restored.timeline[3][0].confidence == pytest.approx(0.5)


def test_roundtrip_multiple_boxes_per_frame_preserve_each_tag(gt_marks_dir):
    """A frame with mixed-provenance boxes restores every box's own tag independently."""
    session = _make_session("clip_c", total_frames=2)
    session.timeline[0] = [
        _box("player", PROV_LABELLER, 1.0, x=0.1),
        _box("player", PROV_YOLO, 0.6, x=0.5),
        _box("in_play_ball", PROV_VITTRACK, 0.55, x=0.8),
    ]
    session._do_flush()

    restored = _make_session("clip_c", total_frames=2)
    restored._load_existing_marks()

    frame = restored.timeline[0]
    assert frame is not None
    assert len(frame) == 3
    by_x = {round(b.x, 2): b for b in frame}
    assert by_x[0.1].model == PROV_LABELLER
    assert by_x[0.1].confidence == pytest.approx(1.0)
    assert by_x[0.5].model == PROV_YOLO
    assert by_x[0.5].confidence == pytest.approx(0.5)
    assert by_x[0.8].model == PROV_VITTRACK
    assert by_x[0.8].confidence == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 2. no_ball / not_broadcast round-trip
# ---------------------------------------------------------------------------


def test_roundtrip_no_ball_and_not_broadcast_sets(gt_marks_dir):
    session = _make_session("clip_d", total_frames=6)
    session.timeline[2] = [_box("player", PROV_LABELLER, 1.0, x=0.1)]
    session.no_ball_frames = {0, 1}
    session.not_broadcast_frames = {4, 5}
    session._do_flush()

    restored = _make_session("clip_d", total_frames=6)
    restored._load_existing_marks()

    assert restored.no_ball_frames == {0, 1}
    assert restored.not_broadcast_frames == {4, 5}
    # no_ball / not_broadcast frames produce no boxes.
    for idx in (0, 1, 4, 5):
        assert restored.timeline[idx] is None
    # The labeller-marked frame is untouched by skip-marker handling.
    assert restored.timeline[2] is not None
    assert restored.timeline[2][0].model == PROV_LABELLER


def test_flush_writes_skip_markers_with_null_bbox(gt_marks_dir):
    session = _make_session("clip_e", total_frames=3)
    session.no_ball_frames = {0}
    session.not_broadcast_frames = {1}
    session._do_flush()

    out_path = gt_marks_dir / "clip_e.jsonl"
    lines = [json.loads(x) for x in out_path.read_text().splitlines() if x.strip()]
    by_idx = {rec["frame_index"]: rec for rec in lines}
    assert by_idx[0]["bbox"] is None
    assert by_idx[0]["tags"] == ["no_ball"]
    assert by_idx[1]["bbox"] is None
    assert by_idx[1]["tags"] == ["not_broadcast"]
    # frame 2 has neither boxes nor markers -> not written at all.
    assert 2 not in by_idx


# ---------------------------------------------------------------------------
# 3. merge_propagated: GT-authoritative + return value
# ---------------------------------------------------------------------------


def test_merge_propagated_keeps_gt_and_returns_true():
    session = _make_session("clip_f", total_frames=3)
    gt_box = _box("player", PROV_LABELLER, 1.0, x=0.1)
    session.timeline[0] = [gt_box]

    incoming = [_box("player", PROV_VITTRACK, 0.5, x=0.9)]
    result = session.merge_propagated(0, incoming)

    assert result is True
    assert session.timeline[0] == [gt_box]  # unchanged, not augmented either


def test_merge_propagated_writes_machine_boxes_and_returns_false():
    session = _make_session("clip_g", total_frames=3)
    session.timeline[0] = [_box("player", PROV_YOLO, 0.4, x=0.1)]

    incoming = [_box("player", PROV_VITTRACK, 0.5, x=0.9)]
    result = session.merge_propagated(0, incoming)

    assert result is False
    assert session.timeline[0] == incoming


def test_merge_propagated_out_of_range_returns_false_no_crash():
    session = _make_session("clip_h", total_frames=3)
    incoming = [_box("player", PROV_VITTRACK, 0.5, x=0.1)]

    assert session.merge_propagated(-1, incoming) is False
    assert session.merge_propagated(3, incoming) is False
    assert session.merge_propagated(1000, incoming) is False


def test_merge_propagated_empty_frame_writes_boxes_returns_false():
    session = _make_session("clip_i", total_frames=3)
    assert session.timeline[0] is None

    incoming = [_box("player", PROV_VITTRACK, 0.5, x=0.1)]
    result = session.merge_propagated(0, incoming)

    assert result is False
    assert session.timeline[0] == incoming
