"""Tests for gap bridging and uncertainty triage."""

from __future__ import annotations

from footy_track.identity.linker import (
    DEFAULT_MAX_GAP,
    Candidate,
    Tracklet,
    candidate_links,
    tracklets_from_rows,
    triage,
)


def _t(tid, start, end, sx=0.5, sy=0.5, ex=None, ey=None, team=None, n=None):
    return Tracklet(
        track_id=tid,
        start_frame=start,
        end_frame=end,
        start_xy=(sx, sy),
        end_xy=(sx if ex is None else ex, sy if ey is None else ey),
        n_detections=n if n is not None else end - start + 1,
        team=team,
    )


def test_overlapping_tracklets_are_never_candidates():
    """They co-exist in some frame, so they are provably different players."""
    a = _t(1, 0, 100)
    b = _t(2, 50, 150)
    assert candidate_links([a, b]) == []


def test_a_short_gap_with_continuous_position_is_a_candidate():
    a = _t(1, 0, 100, ex=0.50, ey=0.50)
    b = _t(2, 105, 200, sx=0.52, sy=0.51)
    cands = candidate_links([a, b])
    assert len(cands) == 1
    assert cands[0].gap == 5


def test_gaps_beyond_the_measured_window_are_dropped():
    """AUC falls from 0.912 at 4 frames to 0.761 by 25; 12 is the default."""
    a = _t(1, 0, 100)
    b = _t(2, 100 + DEFAULT_MAX_GAP + 5, 200)
    assert candidate_links([a, b]) == []
    assert candidate_links([a, b], max_gap=100), "widening the window admits it"


def test_teleporting_link_is_rejected():
    """A player cannot cross the pitch during a 3-frame gap."""
    a = _t(1, 0, 100, ex=0.05, ey=0.5)
    b = _t(2, 103, 200, sx=0.95, sy=0.5)
    assert candidate_links([a, b]) == []


def test_different_teams_are_not_linked():
    a = _t(1, 0, 100, team=0)
    b = _t(2, 104, 200, team=1)
    assert candidate_links([a, b]) == []
    assert len(candidate_links([a, b], respect_team=False)) == 1


def test_unknown_team_does_not_block_a_link():
    """Team is often unknown; absence of evidence must not veto."""
    a = _t(1, 0, 100, team=None)
    b = _t(2, 104, 200, team=1)
    assert len(candidate_links([a, b])) == 1


def test_triage_splits_on_the_measured_thresholds():
    cands = [
        Candidate(a=1, b=2, gap=3, distance=0.01, similarity=0.92),  # confident same
        Candidate(a=3, b=4, gap=3, distance=0.01, similarity=0.30),  # confident diff
        Candidate(a=5, b=6, gap=3, distance=0.01, similarity=0.67),  # ambiguous
    ]
    out = triage(cands)
    assert [c.a for c in out["merge"]] == [1]
    assert [c.a for c in out["reject"]] == [3]
    assert [c.a for c in out["ask"]] == [5]


def test_ask_queue_is_ordered_most_uncertain_first():
    """Human attention goes where the model is least useful."""
    cands = [
        Candidate(a=1, b=2, gap=1, distance=0.0, similarity=0.79),  # near boundary-high
        Candidate(a=3, b=4, gap=1, distance=0.0, similarity=0.675),  # dead centre
        Candidate(a=5, b=6, gap=1, distance=0.0, similarity=0.56),  # near boundary-low
    ]
    asked = triage(cands)["ask"]
    assert asked[0].a == 3, "the most uncertain candidate must come first"


def test_unscored_candidates_default_to_ask():
    """No similarity is not evidence of anything — never auto-merge on silence."""
    c = Candidate(a=1, b=2, gap=4, distance=0.01)
    assert c.decision == "ask"
    assert c.uncertainty == 0.0


def test_tracklets_from_rows_summarises_and_filters():
    rows = []
    for f in range(40):
        rows.append({"frame_index": f, "track_id": 1,
                     "bbox": {"x": 0.10 + f * 0.001, "y": 0.5, "w": 0.04, "h": 0.1}})
    rows.append({"frame_index": 0, "track_id": 2,
                 "bbox": {"x": 0.8, "y": 0.5, "w": 0.04, "h": 0.1}})  # too short
    ts = tracklets_from_rows(rows, min_frames=25)
    assert [t.track_id for t in ts] == [1]
    t = ts[0]
    assert (t.start_frame, t.end_frame, t.n_detections) == (0, 39, 40)
    assert t.start_xy[0] < t.end_xy[0], "movement direction preserved"


def test_no_candidates_from_a_single_tracklet():
    assert candidate_links([_t(1, 0, 100)]) == []
