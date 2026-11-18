"""Stage 3: tracker wrapper (gap frames), node merging within 8 px, neighbour / tracker features, plausibility, pool schema."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from footy_track.ball_mining import candidates as C  # noqa: N812
from footy_track.ball_mining import motion as M  # noqa: N812
from footy_track.ball_mining import pool as P  # noqa: N812
from footy_track.ball_mining import schema
from footy_track.ball_mining import track as T  # noqa: N812
from footy_track.ball_trackers.bidirectional import BidirectionalBallTracker
from footy_track.ball_trackers.detection_tracker import BallCandidate

WW, HH = P.WW, P.HH


def rt(cx, cy, conf, w=11.0, h=11.0):
    return np.array([[cx, cy, w, h, conf]], np.float32)


def mot_row(x, y, score=0.1, **kw):
    r = np.zeros(len(M.COLS), np.float32)
    r[:3] = (x, y, score)
    r[3:] = (kw.get("contrast", 60.0), 100, 10, 10, 1.0, 1.0, 0.0, 0.0)
    return r[None, :]


def nodes_of(fi, stride=5):
    return P.frame_nodes(fi, stride)


def fi_(**kw):
    kw.setdefault("rt", np.zeros((0, 5), np.float32))
    kw.setdefault("persons", np.zeros((0, 4), np.float32))
    return P.FrameInputs(**kw)


# ---------------------------------------------------------------------------------------------- clustering and node boxes
def test_cluster_single_linkage_within_8px():
    xy = np.array([[0, 0], [5, 0], [10, 0], [40, 0]], float)
    assert list(P._cluster(xy)) == [
        0,
        0,
        0,
        1,
    ]  # 0-5-10 chain links at 8 px, 40 does not
    assert len(P._cluster(np.zeros((0, 2)))) == 0


def test_rtdetr_and_motion_within_8px_merge_into_one_node_on_the_detector_box():
    n = nodes_of(fi_(rt=rt(500, 300, 0.7, 12, 14), motion=mot_row(505, 302, 0.2)))
    assert len(n) == 1
    assert n[0]["sources"] == ["rtdetr", "motion"] and np.allclose(
        n[0]["source_scores"], [0.7, 0.2]
    )
    assert (n[0]["w"], n[0]["h"]) == (12.0, 14.0)  # the card shows the detector's box
    assert (
        n[0]["rt_score"] == pytest.approx(0.7)
        and n[0]["n_src"] == 2
        and n[0]["motion_n"] == 1
    )


def test_far_points_stay_separate_and_motion_only_node_has_blob_size():
    row = mot_row(900, 500)
    row[0, 5:7] = (9, 8)
    n = nodes_of(fi_(rt=rt(100, 100, 0.8), motion=row))
    assert len(n) == 2
    mo = next(x for x in n if x["sources"] == ["motion"])
    assert (
        (mo["w"], mo["h"]) == (9.0, 8.0)
        and np.isnan(mo["rt_score"])
        and mo["rt_n"] == 0.0
    )
    assert mo["n_src"] == 1


def test_rt_rank_counts_higher_confidence_boxes_in_the_frame():
    boxes = np.array(
        [[100, 100, 10, 10, 0.9], [300, 300, 10, 10, 0.5], [600, 100, 10, 10, 0.2]],
        np.float32,
    )
    n = sorted(nodes_of(fi_(rt=boxes)), key=lambda x: -x["rt_score"])
    assert [x["rt_rank"] for x in n] == [0.0, 1.0, 2.0]


def test_person_proximity_features():
    persons = np.array(
        [[500, 300, 40, 100]], np.float32
    )  # feet at y=350, head at y=250
    n = nodes_of(fi_(rt=rt(500, 345, 0.6), persons=persons))[0]
    assert (
        n["in_person"] == 1.0
        and n["d_foot"] == pytest.approx(5.0)
        and n["d_head"] == pytest.approx(95.0)
    )
    assert np.isnan(nodes_of(fi_(rt=rt(500, 345, 0.6)))[0]["d_foot"])


# ---------------------------------------------------------------------------------------------- neighbours, plausibility
def test_neighbour_distances_nan_999_and_static():
    prev = np.array([[503, 300, 0.9]], np.float32)  # 3 px away, high conf
    n = nodes_of(
        fi_(rt=rt(500, 300, 0.6), prev=prev, next=np.zeros((0, 3), np.float32))
    )[0]
    assert (
        n["d_prev"] == pytest.approx(3.0)
        and n["d_next"] == 999.0
        and n["d_prev_hc"] == pytest.approx(3.0)
    )
    assert (
        n["nb_plaus"] == 1.0
        and n["d_min_nb"] == pytest.approx(3.0)
        and n["static_nb"] == 0.0
    )
    n2 = nodes_of(
        fi_(rt=rt(500, 300, 0.6), prev=np.array([[501, 300, 0.2]], np.float32))
    )[0]
    assert (
        n2["d_prev"] == 1.0 and n2["static_nb"] == 1.0 and np.isnan(n2["d_next"])
    )  # no next sample: unknown, not far
    assert (
        n2["d_prev_hc"] == 999.0
    )  # the neighbour box is below the high-confidence floor


def test_plausible_true_false_none():
    near = np.array([[520, 300, 0.4]], np.float32)  # within 43.3 * 5 px, conf >= 0.3
    far = np.array([[1200, 650, 0.9]], np.float32)
    assert nodes_of(fi_(rt=rt(500, 300, 0.9), prev=near))[0]["plausible"] is True
    assert (
        nodes_of(fi_(rt=rt(500, 300, 0.9), prev=far, next=far))[0]["plausible"] is False
    )
    assert (
        nodes_of(fi_(rt=rt(500, 300, 0.9)))[0]["plausible"] is None
    )  # no neighbouring sample at all
    weak = np.array([[505, 300, 0.2]], np.float32)  # in range but below PLAUS_CONF
    assert nodes_of(fi_(rt=rt(500, 300, 0.9), prev=weak))[0]["plausible"] is False


def test_tracker_features_and_gap_node():
    on = P.TrackSample(
        2, 502.0, 300.0, 11, 11, agree=True, support=3.2, run_len=7, track_id=4
    )
    n = nodes_of(fi_(rt=rt(500, 300, 0.9), trk=on))[0]
    assert (
        n["trk_member"] == 1.0
        and n["trk_agree"] == 1.0
        and n["trk_support"] == 3.2
        and n["trk_len"] == 7.0
    )
    assert n["track_id"] == 4 and n["plausible"] is True
    off = nodes_of(fi_(rt=rt(900, 300, 0.9), trk=on))[0]
    assert (
        off["trk_member"] == 0.0
        and off["trk_support"] == 0.0
        and off["track_id"] == -1
        and off["trk_dist"] == pytest.approx(398.0)
    )
    gap = P.TrackSample(1, 640.0, 360.0, 11, 11, track_id=4, gap_plaus=0.62)
    nodes = nodes_of(fi_(rt=rt(100, 100, 0.8), trk=gap))
    g = next(x for x in nodes if x["is_gap"])
    assert (
        g["sources"] == ["tracker_gap"]
        and g["source_scores"] == [pytest.approx(0.62)]
        and g["track_id"] == 4
    )
    assert not next(x for x in nodes if not x["is_gap"])["is_gap"]
    assert all(
        np.isnan(nodes_of(fi_(rt=rt(500, 300, 0.9)))[0][c])
        for c in ("trk_state", "trk_member")
    )  # no tracker run: unknown


# ---------------------------------------------------------------------------------------------- tracker wrapper
def synthetic_clip(confs, stride=5, w=1920, h=1080):
    """A ball moving 20 px / sample to the right; ``confs[i]`` None = no detection at sample i."""
    frames, balls = [], []
    for i, c in enumerate(confs):
        frames.append(i * stride)
        balls.append(
            np.zeros((0, 5), np.float32)
            if c is None
            else np.array(
                [[(300 + 20 * i - 5) / w, (500 - 5) / h, 10 / w, 10 / h, c]], np.float32
            )
        )
    return C.build_clip(
        frames,
        balls,
        [np.zeros((0, 5), np.float32)] * len(frames),
        size=(w, h),
        fps=50.0,
        n_src_frames=len(frames) * stride,
        stride=stride,
    )


def test_run_tracker_equals_bidirectional_points_and_finds_gap_frames():
    cand = synthetic_clip([0.9, 0.85, 0.9, None, None, 0.8, 0.9, 0.9])
    tr = T.run_tracker(cand)
    seq = [
        (int(f), [BallCandidate(*map(float, r[:5])) for r in cand.balls(int(f))])
        for f in cand.frames
    ]
    ref = BidirectionalBallTracker((1920, 1080), **T.tracker_kwargs(5))
    for f, c in seq:
        ref.update(f, c)
    pts = ref.points()
    assert [p.source for p in pts] == [
        {0: "none", 1: "interpolated", 2: "detected"}[int(s)] for s in tr.state
    ]  # identical to points()
    assert list(tr.state) == [
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        2,
    ]  # two missed samples bridged: gap frames
    assert (
        np.isnan(tr.gap_plaus[:3]).all()
        and (tr.gap_plaus[3:5] > 0).all()
        and tr.gap_plaus[3] == tr.gap_plaus[4]
    )  # one value per gap run
    one = T.run_tracker(synthetic_clip([0.9, 0.85, 0.9, None, 0.8, 0.9, 0.9, 0.9]))
    assert one.gap_plaus[3] > tr.gap_plaus[3]  # a longer gap is less plausible
    assert tr.run_len.max() == 8 and (tr.support > 0).all() and tr.agree[0]


def test_run_tracker_does_not_bridge_more_than_two_missed_samples():
    tr = T.run_tracker(synthetic_clip([0.9, 0.9, None, None, None, None, 0.9, 0.9]))
    assert 1 not in set(
        tr.state
    )  # beyond max_missed = 3 * stride - 1 the track ends, nothing is interpolated


def test_run_tracker_empty_clip_and_gap_plausibility():
    empty = C.build_clip(
        [], [], [], size=(1920, 1080), fps=50.0, n_src_frames=0, stride=5
    )
    assert len(T.run_tracker(empty).frames) == 0
    assert T.gap_plausibility(0.9, 0.9, 1) == pytest.approx(0.9) and T.gap_plausibility(
        0.9, 0.9, 3
    ) < T.gap_plausibility(0.9, 0.9, 1)
    assert T.gap_plausibility(0.0, 0.9, 1) == 0.0


# ---------------------------------------------------------------------------------------------- clip -> pool
def test_clip_nodes_to_valid_pool_rows():
    cand = synthetic_clip([0.9, 0.85, 0.9, None, 0.8, 0.9])
    tr = T.run_tracker(cand)
    df = P.clip_nodes("m1", "m1_seg000", cand, None, tr, detector_sha="abcd1234")
    assert set(df["frame"]) == set(cand.frames.tolist())
    assert df["gap_frame"].eq(True).sum() == 1 and df.loc[
        df["gap_frame"].eq(True), "sources"
    ].iloc[0] == ["tracker_gap"]
    assert df[["x", "y", "w", "h"]].min().min() > 0 and df[["x", "y"]].max().max() <= 1
    scored = df.assign(
        p_real=np.float32(0.5), scorer_version="t", solver_post=np.float32(np.nan)
    )
    pool = scored[[c for c in schema.COLUMNS if c in scored.columns]]
    out = schema.validate_pool(pool)  # the miner's own validator accepts it
    assert out["static_flag"].notna().all() and out["fps"].iloc[0] == 50.0


def test_static_flag_marks_a_parked_ball():
    cand = synthetic_clip([0.9] * 30)
    balls = [
        np.vstack(
            [cand.balls(int(f)), np.array([[0.1, 0.1, 0.005, 0.01, 0.8]], np.float32)]
        )
        for f in cand.frames
    ]
    cand2 = C.build_clip(
        list(cand.frames),
        balls,
        [np.zeros((0, 5), np.float32)] * 30,
        size=(1920, 1080),
        fps=50.0,
        n_src_frames=150,
        stride=5,
    )
    df = P.clip_nodes("m1", "c", cand2, None, None, detector_sha="x")
    parked = df[(df["x"] < 0.2) & (df["y"] < 0.2)]
    assert len(parked) == 30 and parked["static_flag"].eq(True).all()
    assert df[df["x"] > 0.2]["static_flag"].eq(False).all()


def test_build_match_nodes_end_to_end(tmp_path):
    d = tmp_path / "m1"
    cdir = d / "candidates.abcd1234"
    cdir.mkdir(parents=True)
    cand = synthetic_clip([0.9, 0.85, 0.9, None, 0.8, 0.9])
    C.save_clip(cand, cdir / "m1_seg000.npz")
    (cdir / C.META).write_text(
        json.dumps(
            {
                "complete": True,
                "expected": ["m1_seg000"],
                "stride": 5,
                "detector": {"sha8": "abcd1234"},
            }
        )
    )
    meta = P.build_match_nodes(d, "abcd1234", log=lambda *_: None)
    assert (
        meta["clips"] == 1
        and meta["gap_nodes"] == 1
        and meta["motion"] is False
        and meta["clips_without_motion"] == 1
    )
    df = pd.read_parquet(P.nodes_path(d, "abcd1234"))
    assert len(df) == meta["nodes"] and set(P.FEATURE_COLS) <= set(df.columns)
    with pytest.raises(RuntimeError, match="not complete"):
        P.build_match_nodes(d, "zzzz")
