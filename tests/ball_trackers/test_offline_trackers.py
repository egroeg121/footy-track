"""Synthetic tests for the bidirectional and Viterbi ball trackers (no GPU, no video)."""

from __future__ import annotations

import math

import pytest

from footy_track.ball_trackers.bidirectional import (
    BidirectionalBallTracker,
    run_reversed,
)
from footy_track.ball_trackers.detection_tracker import (
    DETECTED,
    INTERPOLATED,
    NONE,
    BallCandidate,
    DetectionBallTracker,
)
from footy_track.ball_trackers.factory import TRACKER_KINDS, make_ball_tracker
from footy_track.ball_trackers.viterbi import ViterbiBallTracker, confidence_logodds

W, H = 1920, 1080
OFFLINE = ("bidirectional", "viterbi")


def cand(cx: float, cy: float, conf: float) -> BallCandidate:
    """Candidate centred at pixel (cx, cy); 10 px box."""
    return BallCandidate((cx - 5) / W, (cy - 5) / H, 10 / W, 10 / H, conf)


def ball_xy(f: int) -> tuple[float, float]:
    return 400 + 10 * f, 600 - 2 * f  # real ball: slow, steady


def run(kind: str, frames: dict[int, list[BallCandidate]], n: int, **params):
    tracker = make_ball_tracker(kind, (W, H), **params)
    for f in range(n):
        tracker.update(f, frames.get(f, []))
    return tracker.points()


def err_px(p, f: int) -> float:
    gx, gy = ball_xy(f)
    return math.hypot((p.x + p.w / 2) * W - gx, (p.y + p.h / 2) * H - gy)


def fp_first_then_ball(early_conf: float = 0.3) -> dict[int, list[BallCandidate]]:
    """A confident crowd FP first, weak followers near it, the real ball only
    weakly visible (``early_conf``) until it becomes confident at frame 20."""
    frames: dict[int, list[BallCandidate]] = {f: [] for f in range(30)}
    for f in range(30):
        x, y = ball_xy(f)
        frames[f].append(cand(x, y, early_conf if f < 20 else 0.85))
    for f in range(5):
        frames[f].append(cand(1500, 80, 0.9))  # steward at the hoarding
    for f in range(5, 20):  # weak candidates that keep following the FP
        frames[f].append(cand(1500 + 3 * (f - 5), 80, 0.25))
    return frames


def test_factory_kinds_and_unknown():
    for kind in TRACKER_KINDS:
        assert make_ball_tracker(kind, (W, H)) is not None
    with pytest.raises(ValueError):
        make_ball_tracker("nope", (W, H))


@pytest.mark.parametrize("kind", ["online", *OFFLINE])
def test_empty_input(kind):
    assert make_ball_tracker(kind, (W, H)).points() == []
    pts = run(kind, {}, 5)
    assert [p.source for p in pts] == [NONE] * 5


@pytest.mark.parametrize("kind", OFFLINE)
def test_recovers_early_real_ball_after_false_positive(kind):
    # Viterbi only believes a candidate that is better than a coin flip or sits on
    # a consistent path, so its real ball is a notch stronger than for the
    # bidirectional tracker (which follows weak candidates from a strong anchor).
    frames = fp_first_then_ball(0.3 if kind == "bidirectional" else 0.55)
    online = run("online", frames, 30)
    # the online tracker starts on the FP and never sees the weak real ball early
    assert all(
        p.source == NONE or err_px(p, f) > 100 for f, p in enumerate(online[:20])
    )
    pts = run(kind, frames, 30)
    good = [f for f in range(30) if pts[f].source != NONE and err_px(pts[f], f) < 10]
    assert set(range(5, 30)) <= set(good)
    # frames 0-4 hold a confident FP and a weak real ball: Viterbi may take either
    assert len(good) >= (28 if kind == "bidirectional" else 25)


@pytest.mark.parametrize("kind", OFFLINE)
def test_follows_one_ball_through_clean_sequence(kind):
    frames = {f: [cand(*ball_xy(f), 0.9)] for f in range(20)}
    pts = run(kind, frames, 20)
    assert all(p.source == DETECTED and err_px(p, f) < 1 for f, p in enumerate(pts))
    assert len({p.track_id for p in pts}) == 1


@pytest.mark.parametrize("kind", OFFLINE)
def test_short_gap_is_interpolated(kind):
    frames = {f: [cand(*ball_xy(f), 0.9)] for f in range(20) if not 8 <= f < 12}
    pts = run(kind, frames, 20)
    for f in range(8, 12):
        assert pts[f].source == INTERPOLATED
        assert err_px(pts[f], f) < 2
        assert pts[f].confidence is None


@pytest.mark.parametrize("kind", ["online", *OFFLINE])
def test_no_interpolation_across_a_very_long_gap(kind):
    frames = {f: [cand(*ball_xy(f), 0.9)] for f in range(60) if not 10 <= f < 50}
    pts = run(kind, frames, 60, **({"max_missed": 12} if kind == "online" else {}))
    assert all(pts[f].source == NONE for f in range(15, 45))


@pytest.mark.parametrize("kind", OFFLINE)
def test_camera_cut_keeps_both_sides_without_bridging(kind):
    frames = {}
    for f in range(30):
        pos = (400 + 8 * f, 500) if f < 15 else (1500 - 8 * (f - 15), 300)
        frames[f] = [cand(*pos, 0.9)]
    pts = run(kind, frames, 30)
    assert all(p.source == DETECTED for p in pts)
    assert pts[0].track_id != pts[29].track_id or kind == "viterbi"
    # no point is dragged between the two camera positions
    assert all(
        abs((p.x + p.w / 2) * W - ((400 + 8 * f) if f < 15 else (1500 - 8 * (f - 15))))
        < 2
        for f, p in enumerate(pts)
    )


@pytest.mark.parametrize("kind", OFFLINE)
def test_lone_weak_candidates_are_not_a_track(kind):
    frames = {f: [cand(300 + 400 * (f % 3), 200 + 50 * f, 0.15)] for f in range(10)}
    pts = run(kind, frames, 10)
    assert all(p.source == NONE for p in pts)


def test_reversed_run_matches_frame_indices_and_order():
    frames = [(f, [cand(*ball_xy(f), 0.9)]) for f in range(10, 20)]
    pts = run_reversed(frames, frame_size=(W, H))
    assert [p.frame_index for p in pts] == list(range(10, 20))
    assert all(p.source == DETECTED for p in pts)


def test_bidirectional_matches_online_when_online_is_right():
    frames = {f: [cand(*ball_xy(f), 0.9)] for f in range(20)}
    a = run("online", frames, 20)
    b = run("bidirectional", frames, 20)
    assert [(p.frame_index, p.source) for p in a] == [
        (p.frame_index, p.source) for p in b
    ]


def test_bidirectional_support_rules():
    frames = fp_first_then_ball()
    by_count = run("bidirectional", frames, 30, support="strong_count")
    assert sum(1 for f in range(5, 20) if err_px(by_count[f], f) < 10) >= 14
    with pytest.raises(ValueError):
        run("bidirectional", frames, 30, support="bogus")


def test_viterbi_and_bidirectional_write_the_same_row_format():
    frames = {f: [cand(*ball_xy(f), 0.9)] for f in range(5)}
    for kind in OFFLINE:
        row = run(kind, frames, 5)[0].to_row()
        assert set(row) == {
            "frame_index",
            "track_id",
            "source",
            "bbox",
            "center",
            "confidence",
        }


def test_confidence_logodds_is_monotone_and_follows_the_bands():
    xs = [0.1, 0.15, 0.3, 0.55, 0.65, 0.75, 0.85, 0.99]
    ys = [float(confidence_logodds(x)) for x in xs]
    assert ys == sorted(ys)
    assert confidence_logodds(0.55) < 0 < confidence_logodds(0.65)


def test_offline_trackers_accept_irregular_frame_indices():
    tracker = ViterbiBallTracker((W, H))
    for f in range(0, 40, 2):  # stride 2
        tracker.update(f, [cand(*ball_xy(f), 0.9)])
    pts = tracker.points()
    assert [p.frame_index for p in pts] == list(range(0, 40, 2))
    assert all(p.source == DETECTED for p in pts)


def test_online_tracker_is_unchanged_by_the_new_modules():
    t = DetectionBallTracker((W, H))
    t.update(0, [cand(100, 100, 0.9)])
    assert t.points()[0].source == DETECTED
    assert isinstance(BidirectionalBallTracker((W, H)), BidirectionalBallTracker)
