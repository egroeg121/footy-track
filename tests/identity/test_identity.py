"""Tests for the player-identity labelling primitives."""

from __future__ import annotations

import pytest

from footy_track.identity import (
    TIER_HUMAN_CHECKED,
    TIER_MACHINE,
    CheckedInterval,
    FrameRisk,
    PairLabel,
    TrackletRef,
    TrackletReview,
    Verdict,
    build_clusters,
    rank_risky_frames,
    select_eval_clips,
    stable_detection_id,
    unknown_rate,
)
from footy_track.identity.clusters import merge_savings
from footy_track.identity.sampling import review_budget


def _t(track_id: int, clip: str = "seg000") -> TrackletRef:
    return TrackletRef(clip=clip, track_id=track_id)


# --------------------------------------------------------------------------
# stable detection ids
# --------------------------------------------------------------------------


def test_stable_id_is_independent_of_position():
    """The whole point: filtering rows must not change any id."""
    a = stable_detection_id(7, "player", (0.1, 0.2, 0.05, 0.1))
    b = stable_detection_id(7, "player", (0.1, 0.2, 0.05, 0.1))
    assert a == b


def test_stable_id_accepts_dict_and_tuple_bbox():
    assert stable_detection_id(3, "player", (0.1, 0.2, 0.3, 0.4)) == stable_detection_id(
        3, "player", {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    )


def test_stable_id_survives_float_noise():
    """Float formatting differs between producers; ids must not."""
    assert stable_detection_id(1, "player", (0.1, 0.2, 0.3, 0.4)) == (
        stable_detection_id(1, "player", (0.1 + 1e-12, 0.2, 0.3, 0.4))
    )


def test_stable_id_distinguishes_frame_label_and_geometry():
    base = stable_detection_id(1, "player", (0.1, 0.2, 0.3, 0.4))
    assert base != stable_detection_id(2, "player", (0.1, 0.2, 0.3, 0.4))
    assert base != stable_detection_id(1, "referee", (0.1, 0.2, 0.3, 0.4))
    assert base != stable_detection_id(1, "player", (0.9, 0.2, 0.3, 0.4))


# --------------------------------------------------------------------------
# tiering by frame interval
# --------------------------------------------------------------------------


def test_tier_applies_only_inside_checked_intervals():
    """A 'reviewed' tracklet is mixed-tier: only inspected frames are TIER 2."""
    review = TrackletReview(
        tracklet=_t(1),
        checked_intervals=[CheckedInterval(10, 20), CheckedInterval(100, 105)],
    )
    assert review.tier_at(15) == TIER_HUMAN_CHECKED
    assert review.tier_at(100) == TIER_HUMAN_CHECKED
    assert review.tier_at(50) == TIER_MACHINE
    assert review.tier_at(9) == TIER_MACHINE
    assert review.checked_frame_count() == 11 + 6


def test_interval_rejects_reversed_range():
    with pytest.raises(ValueError):
        CheckedInterval(20, 10)


def test_tracklet_with_split_is_not_pure():
    assert TrackletReview(tracklet=_t(1)).is_pure()
    assert not TrackletReview(tracklet=_t(1), split_at=[42]).is_pure()


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------


def test_same_verdicts_are_transitive():
    res = build_clusters(
        [
            PairLabel(_t(1), _t(2), Verdict.SAME),
            PairLabel(_t(2), _t(3), Verdict.SAME),
        ]
    )
    assert res.cluster_of(_t(1)) == res.cluster_of(_t(3))
    assert res.n_clusters == 1
    assert res.is_consistent


def test_different_verdicts_keep_clusters_apart():
    res = build_clusters(
        [
            PairLabel(_t(1), _t(2), Verdict.SAME),
            PairLabel(_t(3), _t(4), Verdict.SAME),
            PairLabel(_t(1), _t(3), Verdict.DIFFERENT),
        ]
    )
    assert res.cluster_of(_t(1)) != res.cluster_of(_t(3))
    assert res.n_clusters == 2
    assert res.is_consistent


def test_contradiction_is_reported_not_silently_resolved():
    """A~B, B~C but A!=C: one verdict is wrong and a human must decide which."""
    res = build_clusters(
        [
            PairLabel(_t(1), _t(2), Verdict.SAME),
            PairLabel(_t(2), _t(3), Verdict.SAME),
            PairLabel(_t(1), _t(3), Verdict.DIFFERENT),
        ]
    )
    assert not res.is_consistent
    assert len(res.contradictions) == 1


def test_unknown_carries_no_constraint():
    res = build_clusters(
        [
            PairLabel(_t(1), _t(2), Verdict.UNKNOWN),
            PairLabel(_t(3), _t(4), Verdict.UNKNOWN),
        ]
    )
    assert res.n_clusters == 4, "UNKNOWN must not merge anything"
    assert res.n_unknown == 2
    assert res.is_consistent


def test_cluster_ids_are_order_independent():
    labels = [
        PairLabel(_t(1), _t(2), Verdict.SAME),
        PairLabel(_t(3), _t(4), Verdict.SAME),
    ]
    a = build_clusters(labels)
    b = build_clusters(list(reversed(labels)))
    assert a.clusters == b.clusters


def test_unknown_rate_is_the_annotator_ceiling():
    labels = [
        PairLabel(_t(1), _t(2), Verdict.SAME),
        PairLabel(_t(3), _t(4), Verdict.UNKNOWN),
    ]
    assert unknown_rate(labels) == 0.5
    assert unknown_rate([]) == 0.0


def test_merge_savings_against_exhaustive_pairs():
    # 100 tracklets = 4950 pairs; labelling 495 of them saves 90%.
    assert merge_savings(100, 495) == pytest.approx(0.9)
    assert merge_savings(0, 0) == 0.0


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def test_eval_clips_are_spread_not_clustered():
    clips = [f"seg{i:03d}" for i in range(100)]
    picked = select_eval_clips(clips, 10)
    assert len(picked) == 10
    idx = sorted(clips.index(c) for c in picked)
    gaps = [b - a for a, b in zip(idx, idx[1:], strict=False)]
    assert min(gaps) >= 5, f"eval clips are clustered together: {idx}"


def test_eval_selection_is_deterministic():
    clips = [f"seg{i:03d}" for i in range(50)]
    assert select_eval_clips(clips, 7) == select_eval_clips(clips, 7)


def test_eval_selection_rejects_impossible_request():
    with pytest.raises(ValueError):
        select_eval_clips(["a", "b"], 5)


def test_risky_frames_prioritise_low_association_margin():
    """A tracker that cannot choose between two candidates is the best signal."""
    risks = [
        FrameRisk(frame_index=1, crowding=0, confidence=0.99, association_margin=0.99),
        FrameRisk(frame_index=2, crowding=8, confidence=0.40, association_margin=0.02),
        FrameRisk(frame_index=3, crowding=1, confidence=0.95, association_margin=0.90),
    ]
    assert rank_risky_frames(risks, 1) == [2]
    assert rank_risky_frames(risks, 0) == []


def test_review_budget_quantifies_coverage():
    assert review_budget(288, sampled=12) == pytest.approx(12 / 288)
    assert review_budget(0, sampled=5) == 0.0
