"""Unit tests for video_utils pure logic and BackgroundLabeller semantics
(src/footy_track/labeller/README.md §8, LAB-7xx). No videos, no inference: the worker is
driven with a scripted labeller stub.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from footy_track.labeller.video_utils import (
    BackgroundLabeller,
    LabelledObject,
    VitTrackVideoLabeller,
    _frame_index_from_uri,
    _nms_filter,
    _track_anomaly_reason,
)
from footy_track.schema import FrameDetections, ObjectDetection


def _det(label="player", conf=1.0, x=0.1, y=0.1, w=0.05, h=0.05) -> ObjectDetection:
    return ObjectDetection(
        label=label, confidence=conf, x=x, y=y, w=w, h=h, model="vittrack"
    )


def _fd(idx: int, detections: list[ObjectDetection]) -> FrameDetections:
    return FrameDetections(
        uri=Path(f"clip_frame_{idx:06d}"), width=640, height=360, detections=detections
    )


# ---------------------------------------------------------------------------
# LabelledObject validation
# ---------------------------------------------------------------------------


def test_labelled_object_requires_exactly_one_seed():
    with pytest.raises(ValueError):
        LabelledObject(label="player")
    with pytest.raises(ValueError):
        LabelledObject(label="player", bbox_xyxy_abs=(0, 0, 1, 1), point_xy_abs=(1, 1))
    # Either form alone is fine.
    LabelledObject(label="player", bbox_xyxy_abs=(0, 0, 1, 1))
    LabelledObject(label="player", point_xy_abs=(1, 1))


def test_vittrack_labeller_rejects_empty_objects_and_missing_video(tmp_path):
    with pytest.raises(ValueError):
        VitTrackVideoLabeller(video_path=tmp_path / "clip.mp4", objects=[])
    with pytest.raises(FileNotFoundError):
        VitTrackVideoLabeller(
            video_path=tmp_path / "missing.mp4",
            objects=[LabelledObject(label="player", bbox_xyxy_abs=(0, 0, 1, 1))],
        )


# ---------------------------------------------------------------------------
# _frame_index_from_uri
# ---------------------------------------------------------------------------


def test_frame_index_from_uri_roundtrip():
    fd = _fd(37, [])
    assert _frame_index_from_uri(fd) == 37


def test_frame_index_from_uri_fallback_on_unparseable():
    fd = FrameDetections(uri=Path("no_marker_here"), width=1, height=1, detections=[])
    assert _frame_index_from_uri(fd, default=5) == 5
    fd2 = FrameDetections(
        uri=Path("clip_frame_notanumber"), width=1, height=1, detections=[]
    )
    assert _frame_index_from_uri(fd2, default=9) == 9


# ---------------------------------------------------------------------------
# _track_anomaly_reason
# ---------------------------------------------------------------------------


def test_anomaly_none_for_small_motion():
    prev = _fd(0, [_det(x=0.10)])
    cur = _fd(1, [_det(x=0.12)])
    assert _track_anomaly_reason(prev, cur) is None


def test_anomaly_on_large_centre_jump():
    prev = _fd(0, [_det(x=0.05, y=0.05)])
    cur = _fd(1, [_det(x=0.90, y=0.90)])
    reason = _track_anomaly_reason(prev, cur)
    assert reason is not None
    assert "jumped" in reason


def test_anomaly_on_area_explosion():
    prev = _fd(0, [_det(x=0.50, y=0.50, w=0.01, h=0.01)])
    # Same centre, area x400 — jump check passes, size check trips.
    cur = _fd(1, [_det(x=0.405, y=0.405, w=0.2, h=0.2)])
    reason = _track_anomaly_reason(prev, cur)
    assert reason is not None
    assert "changed size" in reason


def test_new_label_is_not_an_anomaly():
    prev = _fd(0, [_det("player", x=0.1)])
    cur = _fd(1, [_det("player", x=0.1), _det("referee", x=0.9)])
    assert _track_anomaly_reason(prev, cur) is None


# ---------------------------------------------------------------------------
# _nms_filter
# ---------------------------------------------------------------------------


def test_nms_keeps_highest_confidence_and_drops_overlaps():
    a = _det("player", conf=0.9, x=0.10)
    b = _det("player", conf=0.5, x=0.11)  # heavy overlap with a
    c = _det("player", conf=0.7, x=0.80)  # distinct
    kept = _nms_filter([b, a, c], iou_threshold=0.3)
    assert a in kept
    assert c in kept
    assert b not in kept


def test_nms_empty_input():
    assert _nms_filter([]) == []


# ---------------------------------------------------------------------------
# BackgroundLabeller: frame_at vs completed_frames (mid-clip regression pin)
# ---------------------------------------------------------------------------


def test_frame_at_serves_mid_clip_frames_where_completed_frames_cannot():
    bg = BackgroundLabeller()
    fd3 = _fd(3, [_det()])
    bg.frames = [None, None, None, fd3, None]
    # The legacy contiguous-from-0 scan sees nothing for a run seeded at 3...
    assert bg.completed_frames() == []
    # ...but frame_at reaches past the leading holes (the mid-clip run fix).
    assert bg.frame_at(3) is fd3
    assert bg.frame_at(0) is None
    assert bg.frame_at(99) is None
    assert bg.frame_at(-1) is None


class _ScriptedLabeller:
    """Stub matching VitTrackVideoLabeller.iter_frames_from's contract."""

    def __init__(self, frames: list[FrameDetections]):
        self._frames = frames

    def iter_frames_from(self, start_frame=0, stop_event=None, progress_callback=None):
        total = len(self._frames)
        for i, fd in enumerate(self._frames):
            yield fd
            if progress_callback is not None:
                progress_callback(start_frame + i + 1, total)
            if stop_event is not None and stop_event.is_set():
                return


def _run_worker(bg: BackgroundLabeller, frames: list[FrameDetections], start_frame=0):
    bg.frames = [None] * 10
    bg.running = True
    bg._worker(_ScriptedLabeller(frames), start_frame)


def test_worker_slots_frames_by_absolute_index():
    bg = BackgroundLabeller()
    _run_worker(bg, [_fd(4, [_det()]), _fd(5, [_det(x=0.11)])], start_frame=4)
    assert bg.frames[4] is not None
    assert bg.frames[5] is not None
    assert bg.frames[0] is None
    assert bg.last_completed_frame == 5
    assert bg.running is False
    assert bg.anomaly_frame is None
    assert bg.error is None


def test_worker_confidence_handback_stops_run():
    bg = BackgroundLabeller()
    frames = [
        _fd(0, [_det(conf=1.0)]),
        _fd(1, [_det(conf=0.31, x=0.11)]),  # below the 0.5 handback threshold
        _fd(2, [_det(conf=1.0, x=0.12)]),
    ]
    _run_worker(bg, frames)
    assert bg.anomaly_frame == 1
    assert "confidence dropped" in bg.anomaly_reason
    assert "0.31" in bg.anomaly_reason
    # The run stopped: frame 2 was never ingested.
    assert bg.frames[2] is None


def test_worker_motion_anomaly_stops_run():
    bg = BackgroundLabeller()
    frames = [
        _fd(0, [_det(x=0.05, y=0.05)]),
        _fd(1, [_det(x=0.90, y=0.90)]),
    ]
    _run_worker(bg, frames)
    assert bg.anomaly_frame == 1
    assert "jumped" in bg.anomaly_reason


def test_worker_anomaly_detection_can_be_disabled():
    bg = BackgroundLabeller()
    bg.anomaly_detection = False
    frames = [
        _fd(0, [_det(x=0.05, y=0.05)]),
        _fd(1, [_det(x=0.90, y=0.90, conf=0.1)]),
        _fd(2, [_det(x=0.91, y=0.91)]),
    ]
    _run_worker(bg, frames)
    assert bg.anomaly_frame is None
    assert bg.frames[2] is not None  # ran to completion


def test_worker_records_errors():
    bg = BackgroundLabeller()

    class _Boom:
        def iter_frames_from(self, **_kwargs):
            raise RuntimeError("tracker exploded")
            yield  # pragma: no cover

    bg.frames = [None] * 3
    bg.running = True
    bg._worker(_Boom(), 0)
    assert isinstance(bg.error, RuntimeError)
    assert bg.running is False


def test_pause_without_thread_is_safe():
    bg = BackgroundLabeller()
    bg.running = True
    bg.pause()
    assert bg.running is False
    assert isinstance(bg._stop_event, threading.Event)
    assert bg._stop_event.is_set()


def test_is_done_semantics():
    bg = BackgroundLabeller()
    assert bg.is_done() is False  # nothing completed yet
    bg.last_completed_frame = 4
    assert bg.is_done() is True
    bg.running = True
    assert bg.is_done() is False
