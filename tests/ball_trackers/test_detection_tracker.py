"""Tests for the detection-based ball tracker (synthetic candidates, no model)."""

from __future__ import annotations

from footy_track.ball_trackers.detection_tracker import (
    DETECTED,
    INTERPOLATED,
    NONE,
    BallCandidate,
    DetectionBallTracker,
)

W, H = 1000, 1000
SIZE = 0.01  # ball is 10 px


def _cand(cx_px: float, cy_px: float = 500, conf: float = 0.9) -> BallCandidate:
    return BallCandidate(cx_px / W - SIZE / 2, cy_px / H - SIZE / 2, SIZE, SIZE, conf)


def _tracker(**kw) -> DetectionBallTracker:
    return DetectionBallTracker(frame_size=(W, H), **kw)


def test_follows_a_moving_ball_with_one_track():
    tr = _tracker()
    for f in range(20):
        tr.update(f, [_cand(100 + 15 * f)])
    pts = tr.points()
    assert [p.source for p in pts] == [DETECTED] * 20
    assert {p.track_id for p in pts} == {1}


def test_bridges_a_gap_by_interpolation():
    tr = _tracker()
    for f in range(4):
        tr.update(f, [_cand(100 + 10 * f)])
    for f in (4, 5, 6):
        tr.update(f, [])
    tr.update(7, [_cand(100 + 10 * 7)])
    pts = tr.points()
    assert [p.source for p in pts[4:7]] == [INTERPOLATED] * 3
    assert all(p.track_id == 1 and p.confidence is None for p in pts[4:7])
    # linear between x=130 (frame 3) and x=170 (frame 7): frame 5 is 150
    mid = pts[5]
    assert abs((mid.x + mid.w / 2) * W - 150) < 1e-6


def test_low_confidence_candidate_near_track_is_accepted():
    tr = _tracker()
    tr.update(0, [_cand(100)])
    tr.update(1, [_cand(110)])
    p = tr.update(2, [_cand(121, conf=0.15)])
    assert p.source == DETECTED and p.track_id == 1 and p.confidence == 0.15


def test_stray_low_confidence_false_positive_is_ignored():
    tr = _tracker()
    tr.update(0, [_cand(100)])
    tr.update(1, [_cand(110)])
    p = tr.update(2, [_cand(120), _cand(700, 200, conf=0.3)])
    assert (p.x + p.w / 2) * W == 120.0
    # alone, far from the track, it is not accepted either
    p = tr.update(3, [_cand(700, 200, conf=0.3)])
    assert p.source == NONE


def test_weak_candidate_cannot_start_a_track():
    tr = _tracker(start_conf=0.6)
    assert tr.update(0, [_cand(100, conf=0.5)]).source == NONE
    assert tr.update(1, [_cand(100, conf=0.6)]).source == DETECTED


def test_track_ends_after_missed_frame_limit():
    tr = _tracker(max_missed=3)
    tr.update(0, [_cand(100)])
    for f in (1, 2, 3):
        tr.update(f, [])
    # 3 misses is still bridgeable ...
    assert tr.update(4, [_cand(102)]).track_id == 1
    for f in (5, 6, 7, 8):
        tr.update(f, [])
    # ... 4 is not: a new track starts and the gap stays "none"
    p = tr.update(9, [_cand(104)])
    assert p.track_id == 2
    assert [q.source for q in tr.points()[5:9]] == [NONE] * 4


def test_strong_detection_far_from_prediction_restarts_track():
    tr = _tracker()
    tr.update(0, [_cand(100)])
    p = tr.update(1, [_cand(800, 100, conf=0.9)])
    assert p.source == DETECTED and p.track_id == 2
