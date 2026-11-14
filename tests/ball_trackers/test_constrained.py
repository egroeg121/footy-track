"""Tests for the constrained Viterbi tracker (anchors, forbidden spots, margins)."""

from __future__ import annotations

import math
import random

from footy_track.ball_trackers.constrained import ConstrainedViterbiTracker
from footy_track.ball_trackers.detection_tracker import (
    DETECTED,
    INTERPOLATED,
    NONE,
    BallCandidate,
)
from footy_track.ball_trackers.viterbi import ViterbiBallTracker

W, H = 1920, 1080


def cand(cx: float, cy: float, conf: float) -> BallCandidate:
    return BallCandidate((cx - 5) / W, (cy - 5) / H, 10 / W, 10 / H, conf)


def ball(f: int) -> tuple[float, float]:
    return 400 + 10 * f, 600 - 2 * f


def centre(p) -> tuple[float, float]:
    return (p.x + p.w / 2) * W, (p.y + p.h / 2) * H


def make(frames: dict[int, list[BallCandidate]], n: int, **params):
    t = ConstrainedViterbiTracker((W, H), **params)
    for f in range(n):
        t.update(f, frames.get(f, []))
    return t


def two_balls(n: int = 40) -> dict[int, list[BallCandidate]]:
    """Real ball (conf 0.6) and a parked spare ball (conf 0.8) at a fixed spot."""
    return {f: [cand(*ball(f), 0.6), cand(1500, 900, 0.8)] for f in range(n)}


def test_matches_viterbi_without_constraints():
    rng = random.Random(3)
    frames = {}
    for f in range(80):
        cs = [
            cand(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.1, 0.9))
            for _ in range(rng.randint(0, 4))
        ]
        if f % 3:
            cs.append(cand(*ball(f % 40), rng.uniform(0.3, 0.95)))
        frames[f] = cs
    ref = ViterbiBallTracker((W, H))
    new = ConstrainedViterbiTracker((W, H))
    for f in range(80):
        ref.update(f, frames[f])
        new.update(f, frames[f])
    for a, b in zip(ref.points(), new.points(), strict=True):
        assert a.source == b.source
        if a.x is not None:
            assert math.isclose(a.x, b.x) and math.isclose(a.y, b.y)


def test_anchor_forces_path_through_weak_candidate():
    frames = {f: [cand(1500, 900, 0.9)] for f in range(30)}
    frames[10].append(cand(300, 300, 0.2))
    t = make(frames, 30)
    assert (
        math.hypot(
            *(a - b for a, b in zip(centre(t.points()[10]), (1500, 900), strict=True))
        )
        < 2
    )
    t.anchor(10, 300, 300)
    p = t.points()[10]
    assert p.source == DETECTED
    assert math.hypot(centre(p)[0] - 300, centre(p)[1] - 300) < 2


def test_anchor_with_no_candidate_injects_one():
    t = make({f: [cand(1500, 900, 0.9)] for f in range(20)}, 20)
    t.anchor(7, 100, 100)
    p = t.points()[7]
    assert p.source == DETECTED
    assert abs(centre(p)[0] - 100) < 1 and abs(centre(p)[1] - 100) < 1


def test_anchor_on_frame_without_any_detections():
    t = make({}, 10)
    t.anchor(4, 640, 360)
    pts = t.points()
    assert pts[4].source == DETECTED
    assert all(pts[f].source == NONE for f in range(10) if f != 4)


def test_forbid_removes_candidate_in_that_frame_only():
    frames = {f: [cand(1500, 900, 0.9)] for f in range(30)}
    t = make(frames, 30)
    t.forbid(10, 1500, 900)
    pts = t.points()
    assert pts[10].source == INTERPOLATED  # bridged from the neighbours
    assert pts[9].source == DETECTED and pts[11].source == DETECTED


def test_forbid_everywhere_kills_parked_ball():
    frames = two_balls()
    t = make(frames, 40)
    assert (
        math.hypot(centre(t.points()[20])[0] - 1500, centre(t.points()[20])[1] - 900)
        < 2
    )
    t.forbid_everywhere(1500, 900, tol=5)
    for p in t.points()[5:35]:
        assert (
            math.hypot(
                centre(p)[0] - ball(p.frame_index)[0],
                centre(p)[1] - ball(p.frame_index)[1],
            )
            < 2
        )


def test_forbid_tolerance_respected():
    frames = {f: [cand(1500, 900, 0.9), cand(1530, 900, 0.5)] for f in range(20)}
    t = make(frames, 20)
    t.forbid_everywhere(1500, 900, tol=5)
    assert abs(centre(t.points()[5])[0] - 1530) < 2


def test_anchor_overrides_forbid_in_same_frame():
    frames = {f: [cand(1500, 900, 0.9)] for f in range(20)}
    t = make(frames, 20)
    t.forbid(5, 1500, 900)
    t.anchor(5, 1500, 900)
    assert t.points()[5].source == DETECTED


def test_margin_small_for_competing_candidates_and_inf_for_anchor():
    frames = {f: [cand(*ball(f), 0.65), cand(1500, 900, 0.7)] for f in range(40)}
    frames_clear = {f: [cand(*ball(f), 0.95)] for f in range(40)}
    amb = make(frames, 40).solve()
    clear = make(frames_clear, 40).solve()
    assert amb.margins[20] < clear.margins[20]
    assert clear.margins[20] > 3
    t = make(frames, 40)
    t.anchor(20, *ball(20))
    sol = t.solve()
    assert math.isinf(sol.margins[20]) and 20 in sol.anchored


def test_anchor_pulls_neighbouring_frames_onto_the_ball():
    """Two near-equal tracks; anchoring one frame puts the whole clip on that track."""
    frames = {
        f: [cand(*ball(f), 0.6), cand(900 + 6 * f, 300, 0.605)] for f in range(40)
    }
    t = make(frames, 40)
    t.anchor(20, *ball(20))
    for p in t.points()[10:30]:
        bx, by = ball(p.frame_index)
        assert math.hypot(centre(p)[0] - bx, centre(p)[1] - by) < 2


def test_chained_segments_carry_the_track_across_the_join():
    """Segment B alone prefers the louder decoy; chained after A it continues the ball."""
    a = {f: [cand(*ball(f), 0.9)] for f in range(30)}
    b = {f: [cand(*ball(30 + f), 0.545), cand(1500, 200, 0.55)] for f in range(30)}
    alone = ConstrainedViterbiTracker((W, H))
    chained = ConstrainedViterbiTracker((W, H))
    for f in range(30):
        chained.update(f, a[f])
        alone.update(f, b[f])
    for f in range(30):
        chained.update(30 + f, b[f])
    bx, by = ball(35)
    on_ball = chained.points()[35]
    assert math.hypot(centre(on_ball)[0] - bx, centre(on_ball)[1] - by) < 2
    assert abs(centre(alone.points()[5])[0] - 1500) < 2
