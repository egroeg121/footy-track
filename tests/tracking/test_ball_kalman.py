"""Tests for the Kalman-filter ball tracker."""

from __future__ import annotations

import pathlib

from footy_track.schema import FrameDetections, ObjectDetection
from footy_track.trackers.ball_kalman import BallKalmanTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ball(
    x: float, y: float, w: float = 0.04, h: float = 0.04, conf: float = 0.9
) -> ObjectDetection:
    return ObjectDetection(label="ball", confidence=conf, x=x, y=y, w=w, h=h)


def _fd(dets: list[ObjectDetection], frame: int = 0) -> FrameDetections:
    return FrameDetections(
        uri=pathlib.Path(f"frame_{frame:04d}.png"),
        width=1920,
        height=1080,
        detections=dets,
    )


def _frames_with_gap(
    n_before: int = 3,
    gap: int = 4,
    n_after: int = 3,
    start_x: float = 0.3,
    vx: float = 0.01,
) -> list[FrameDetections]:
    """Ball moves linearly, disappears for `gap` frames, then reappears."""
    frames = []
    x = start_x
    for i in range(n_before):
        frames.append(_fd([_ball(x + i * vx, 0.5)], frame=i))
    for i in range(gap):
        frames.append(_fd([], frame=n_before + i))
    for i in range(n_after):
        frames.append(
            _fd([_ball(x + (n_before + gap + i) * vx, 0.5)], frame=n_before + gap + i)
        )
    return frames


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_ball_detection_spawns_track():
    tracker = BallKalmanTracker()
    result = tracker.update(_fd([_ball(0.3, 0.5)]), frame_t=0.0)
    assert len(result) == 1
    assert result[0].track_id == 1
    assert result[0].is_interpolated is False


def test_no_detection_returns_empty_when_no_track():
    tracker = BallKalmanTracker()
    result = tracker.update(_fd([]), frame_t=0.0)
    assert result == []


def test_continuous_tracking_same_id():
    tracker = BallKalmanTracker()
    ids = set()
    for i in range(5):
        res = tracker.update(
            _fd([_ball(0.3 + i * 0.01, 0.5)], frame=i), frame_t=i / 25.0
        )
        assert len(res) == 1
        ids.add(res[0].track_id)
    assert len(ids) == 1, "track ID should be stable across consecutive detections"


def test_gap_filled_with_interpolated_detections():
    tracker = BallKalmanTracker(max_gap=5)
    frames = _frames_with_gap(n_before=3, gap=3, n_after=2)
    all_results = []
    for i, fd in enumerate(frames):
        res = tracker.update(fd, frame_t=i / 25.0)
        all_results.append(res)

    # frames 3,4,5 are gap frames — each should yield one interpolated detection
    gap_results = all_results[3:6]
    for res in gap_results:
        assert len(res) == 1
        assert res[0].is_interpolated is True
        assert res[0].track_id == 1


def test_gap_within_max_preserves_track_id():
    """Ball reappearing within max_gap should keep the same track ID."""
    tracker = BallKalmanTracker(max_gap=5)
    frames = _frames_with_gap(n_before=3, gap=4, n_after=3)
    ids_before = set()
    ids_after = set()
    for i, fd in enumerate(frames):
        res = tracker.update(fd, frame_t=i / 25.0)
        if i < 3:
            for r in res:
                ids_before.add(r.track_id)
        elif i >= 7:
            for r in res:
                ids_after.add(r.track_id)
    assert ids_before == ids_after, "track ID must survive a gap within max_gap"


def test_gap_exceeding_max_drops_track():
    """A gap longer than max_gap should finalise the track; reappearance gets a new ID."""
    tracker = BallKalmanTracker(max_gap=3)
    frames = _frames_with_gap(n_before=2, gap=5, n_after=2)
    ids_before = set()
    ids_after = set()
    for i, fd in enumerate(frames):
        res = tracker.update(fd, frame_t=i / 25.0)
        if i < 2:
            for r in res:
                ids_before.add(r.track_id)
        elif i >= 7:
            for r in res:
                ids_after.add(r.track_id)
    assert ids_before.isdisjoint(ids_after), (
        "new track should get a different ID after gap > max_gap"
    )


def test_interpolated_positions_are_within_bounds():
    tracker = BallKalmanTracker(max_gap=5)
    frames = _frames_with_gap(n_before=3, gap=4, n_after=2)
    for i, fd in enumerate(frames):
        for r in tracker.update(fd, frame_t=i / 25.0):
            assert 0.0 <= r.x <= 1.0
            assert 0.0 <= r.y <= 1.0
            assert 0.0 <= r.x + r.w <= 1.0
            assert 0.0 <= r.y + r.h <= 1.0


def test_non_ball_detections_ignored():
    tracker = BallKalmanTracker()
    fd = _fd(
        [ObjectDetection(label="person", confidence=0.9, x=0.1, y=0.2, w=0.05, h=0.1)],
        frame=0,
    )
    result = tracker.update(fd, frame_t=0.0)
    assert result == []


def test_finalise_returns_track_meta():
    tracker = BallKalmanTracker()
    for i in range(3):
        tracker.update(_fd([_ball(0.3 + i * 0.01, 0.5)], frame=i), frame_t=i / 25.0)
    metas = tracker.finalise()
    assert len(metas) == 1
    assert metas[0].label == "ball"
    assert metas[0].start_frame == 0
    assert metas[0].end_frame == 2


def test_highest_confidence_ball_used():
    """When multiple ball detections exist, the tracker uses the highest-confidence one."""
    tracker = BallKalmanTracker()
    dets = [
        _ball(0.3, 0.5, conf=0.5),
        _ball(0.6, 0.5, conf=0.95),
    ]
    result = tracker.update(_fd(dets), frame_t=0.0)
    assert len(result) == 1
    # The x coord of the higher-confidence detection
    cx = result[0].x + result[0].w / 2
    assert abs(cx - (0.6 + 0.04 / 2)) < 1e-6


def test_configurable_max_gap():
    for max_gap in (1, 3, 10):
        tracker = BallKalmanTracker(max_gap=max_gap)
        # Before gap: 2 frames
        for i in range(2):
            tracker.update(_fd([_ball(0.3, 0.5)], frame=i), frame_t=i / 25.0)
        # Gap frames
        gap_results = []
        for j in range(max_gap + 2):
            res = tracker.update(_fd([]), frame_t=(2 + j) / 25.0)
            gap_results.append(res)
        # First max_gap gap frames should return interpolated; beyond that, empty
        for k in range(max_gap):
            assert len(gap_results[k]) == 1 and gap_results[k][0].is_interpolated
        assert gap_results[max_gap] == [] or gap_results[max_gap + 1] == []
