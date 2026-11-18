"""Miner rules (design §3.2, §4): leakage refusal, quotas / spill / cap, dedupe and tombstones, pi, diversity, gold."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from footy_track.ball_mining import miner
from footy_track.ball_mining.buckets import BucketConfig
from footy_track.ball_mining.clips import (
    load_clip_spans,
    neighbour_clips,
    read_verdict_frames,
)
from footy_track.ball_mining.miner import LeakageError, allocate, mine_round
from footy_track.labeller import jobs

from .conftest import make_pool


def only(res, bucket):
    return [c for c in res.cards if c["sampling"]["bucket"] == bucket]


def u_rows(match="m1", clip="m1_seg001", n=40, p=0.5, step=200, **kw):
    return [
        {"match": match, "clip": clip, "frame": i * step, "p_real": p, **kw}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------------------------- leakage


def test_heldout_matches_and_eval_clips_are_refused_not_mined(splitter):
    splitter.eval_clips = {"m1_seg005"}
    pool = make_pool(
        u_rows("m1", "m1_seg001")
        + u_rows("m1", "m1_seg005")
        + u_rows("held1", "held1_seg001")
        + u_rows("nope1", "nope1_seg001")
    )
    res = mine_round(
        pool, budget=30, quotas={"U": 1.0}, split_of=splitter, per_match_cap=1.0
    )
    assert {c["match"] for c in res.cards} == {"m1"}
    assert {c["clip"] for c in res.cards} == {"m1_seg001"}
    rows = res.report["refused"]["pool"]["rows"]
    assert rows == {"eval_heldout_match": 40, "eval_in_match": 40, "unregistered": 40}
    assert all(
        c["sampling"]["regime"] == "train" and c["sampling"]["split"] == "train"
        for c in res.cards
    )


def test_strict_mode_raises_on_any_non_train_row(splitter):
    pool = make_pool(u_rows("m1") + u_rows("held1", "held1_seg001", n=3))
    with pytest.raises(LeakageError, match="held1"):
        mine_round(pool, budget=10, quotas={"U": 1.0}, split_of=splitter, strict=True)


def test_a_pool_of_only_heldout_footage_mines_nothing(splitter):
    res = mine_round(
        make_pool(u_rows("held1", "held1_seg001")), budget=20, split_of=splitter
    )
    assert res.cards == [] and res.report["refused"]["pool"]["matches"] == ["held1"]


def test_every_card_is_rechecked_against_the_split_authority(splitter, monkeypatch):
    pool = make_pool(u_rows("m1"))
    calls = []

    def sneaky(match, clip):  # train while gating, eval at the final assertion
        calls.append(1)
        return miner.TRAIN if len(calls) < 6 else miner.EVAL_HELDOUT

    with pytest.raises(LeakageError, match="not train"):
        mine_round(pool, budget=5, quotas={"U": 1.0}, split_of=sneaky)


def test_ten_seconds_of_footage_around_eval_clips_are_excluded(splitter):
    splitter.eval_clips = {"m1_seg001"}
    spans = {
        "m1_seg001": (100.0, 110.0),
        "m1_seg002": (115.0, 140.0),
        "m1_seg009": (300.0, 310.0),
    }
    # seg002 node time = 115 + frame / 25: frame 125 -> 120 s (edge of the buffer, excluded), frame 150 -> 121 s (kept)
    rows = [
        {"clip": "m1_seg002", "frame": f, "p_real": 0.5}
        for f in (0, 50, 125, 150, 400, 600)
    ]
    rows += [{"clip": "m1_seg009", "frame": f, "p_real": 0.5} for f in (0, 400)]
    pool = make_pool(rows)
    res = mine_round(
        pool,
        budget=20,
        quotas={"U": 1.0},
        split_of=splitter,
        clip_spans=spans,
        per_match_cap=1.0,
        match_of=lambda c: "m1",
    )
    got = sorted((c["clip"], c["frame_index"]) for c in res.cards)
    assert got == [
        ("m1_seg002", 150),
        ("m1_seg002", 400),
        ("m1_seg002", 600),
        ("m1_seg009", 0),
        ("m1_seg009", 400),
    ]
    assert res.report["excluded"]["eval_buffer_footage_time"] == 3


def test_without_footage_spans_the_neighbouring_segments_of_eval_clips_are_excluded(
    splitter,
):
    splitter.eval_clips = {"m1_seg005"}
    pool = make_pool(
        [
            {"clip": f"m1_seg{n:03d}", "frame": 10, "p_real": 0.5}
            for n in (3, 4, 6, 7, 8)
        ]
    )  # the eval clip is not in the pool
    res = mine_round(
        pool,
        budget=10,
        quotas={"U": 1.0},
        split_of=splitter,
        per_match_cap=1.0,
        eval_clips=[("m1", "m1_seg005")],
    )
    assert sorted(c["clip"] for c in res.cards) == [
        "m1_seg003",
        "m1_seg007",
        "m1_seg008",
    ]
    assert neighbour_clips("arsenal_mancity_20250925_seg010") == [
        "arsenal_mancity_seg009",
        "arsenal_mancity_seg011",
    ]


def test_load_clip_spans_reads_segments_json(tmp_path):
    d = tmp_path / "arsenal_mancity" / "split_video_broadcast_frames"
    d.mkdir(parents=True)
    (d / "segments.json").write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "file": "arsenal_mancity_20250925_seg003.mp4",
                        "start_s": 10.5,
                        "end_s": 20.0,
                    }
                ]
            }
        )
    )
    assert load_clip_spans(tmp_path) == {"arsenal_mancity_seg003": (10.5, 20.0)}


# ---------------------------------------------------------------------------------------- quotas, spill, cap


def test_quotas_fill_the_budget_and_follow_the_split(big_pool, splitter):
    res = mine_round(big_pool, budget=300, split_of=splitter, seed=1)
    assert len(res) == 300
    assert Counter(c["sampling"]["bucket"] for c in res.cards) == Counter(
        {"U": 105, "M": 60, "N": 45, "R": 60, "A": 30}
    )
    # no gold supplied: its 15 cards spill to R
    assert res.report["spilled_to_R"] == 15 and res.report["got"]["R"] == 60


def test_allocate_is_exact_and_respects_capacity():
    assert (
        sum(
            allocate(
                300,
                {"a": 0.35, "b": 0.2, "c": 0.15, "d": 0.15, "e": 0.1, "f": 0.05},
                dict.fromkeys("abcdef", 300),
            ).values()
        )
        == 300
    )
    assert allocate(10, {"a": 1, "b": 1}, {"a": 2, "b": 100}) == {"a": 2, "b": 8}
    assert allocate(10, {"a": 1}, {"a": 3}) == {"a": 3}


def test_a_short_bucket_spills_its_quota_to_random(splitter):
    pool = make_pool(
        u_rows(n=200, p=0.4)
    )  # only U-eligible nodes: no gap, static, or audit-region nodes
    res = mine_round(pool, budget=100, split_of=splitter, per_match_cap=1.0, seed=3)
    got = Counter(c["sampling"]["bucket"] for c in res.cards)
    assert got["M"] == got["N"] == got["A"] == got["G"] == 0
    assert got["U"] == 35 and got["R"] == 100 - 35  # every other quota went to R
    assert res.report["spilled_to_R"] == 50


def test_no_match_exceeds_the_cap_and_a_lone_match_relaxes_it(big_pool, splitter):
    res = mine_round(
        big_pool, budget=300, per_match_cap=0.15, split_of=splitter, seed=2
    )
    per = Counter(c["match"] for c in res.cards)
    assert (
        res.report["per_match_cap"]["relaxed_to_equal_share"] is True
    )  # 3 matches x 45 < 300
    assert max(per.values()) <= 100 and len(per) == 3
    many = pd.concat(
        [
            big_pool.assign(
                match=f"x{i}", clip=big_pool["clip"].str.replace("m", f"x{i}_", n=1)
            )
            for i in range(8)
        ],
        ignore_index=True,
    )
    res = mine_round(many, budget=300, per_match_cap=0.15, split_of=splitter, seed=2)
    assert not res.report["per_match_cap"]["relaxed_to_equal_share"]
    assert max(Counter(c["match"] for c in res.cards).values()) <= 45
    one = big_pool[big_pool["match"] == "m1"]
    assert len(mine_round(one, budget=60, per_match_cap=0.15, split_of=splitter)) == 60


def test_same_seed_same_cards_other_seed_other_cards(big_pool, splitter):
    a = mine_round(big_pool, budget=60, split_of=splitter, seed=7)
    b = mine_round(big_pool, budget=60, split_of=splitter, seed=7)
    c = mine_round(big_pool, budget=60, split_of=splitter, seed=8)

    def key(r):
        return [(x["clip"], x["frame_index"], x["sampling"]["bucket"]) for x in r.cards]

    assert key(a) == key(b) and key(a) != key(c)


# -------------------------------------------------------------------------------------- dedupe, tombstones


def test_at_most_one_card_per_clip_per_two_second_window(big_pool, splitter):
    res = mine_round(big_pool, budget=300, split_of=splitter, seed=4)
    wins = [(c["clip"], int(c["frame_index"] // 25 // 2.0)) for c in res.cards]
    assert len(wins) == len(set(wins))


def test_already_verdicted_windows_are_not_mined_again_but_tombstoned_ones_are(
    tmp_path, splitter
):
    marks = tmp_path / "marks"
    vdir = marks / "ball_checks"
    vdir.mkdir(parents=True)

    def rec(f, b, v):
        return json.dumps(
            {
                "clip": "m1_seg001",
                "frame_index": f,
                "box_index": b,
                "source": "list",
                "verdict": v,
            }
        )

    (vdir / "m1_seg001.jsonl").write_text(
        "\n".join(
            [
                rec(100, 1, "ball"),
                rec(300, 2, "not_ball"),
                rec(300, 2, None),
                rec(500, 3, "unsure"),
            ]
        )
        + "\n"
    )
    # the verdict file of the other alias of the same clip counts too
    (vdir / "arsenal_mancity_20250925_seg010.jsonl").write_text(
        json.dumps(
            {
                "clip": "arsenal_mancity_20250925_seg010",
                "frame_index": 7,
                "box_index": 1,
                "verdict": "ball",
            }
        )
        + "\n"
    )
    assert read_verdict_frames(marks) == {
        ("m1_seg001", 100),
        ("m1_seg001", 500),
        ("arsenal_mancity_seg010", 7),
    }
    pool = make_pool(
        [{"clip": "m1_seg001", "frame": f, "p_real": 0.5} for f in (100, 300, 500, 900)]
    )
    res = mine_round(
        pool,
        budget=10,
        quotas={"U": 1.0},
        split_of=splitter,
        per_match_cap=1.0,
        already_labelled=read_verdict_frames(marks),
    )
    assert sorted(c["frame_index"] for c in res.cards) == [
        300,
        900,
    ]  # 100 and 500 are labelled; 300 was undone
    assert res.report["excluded"]["already_labelled_window"] == 2


def test_the_window_of_a_labelled_frame_blocks_its_neighbours(splitter):
    pool = make_pool(
        [{"frame": f, "p_real": 0.5} for f in (95, 100, 105, 160)]
    )  # 25 fps: frames 50..99 / 100..149 are windows
    res = mine_round(
        pool,
        budget=5,
        quotas={"U": 1.0},
        split_of=splitter,
        per_match_cap=1.0,
        already_labelled=[("m1_seg001", 120)],
    )
    assert sorted(c["frame_index"] for c in res.cards) == [95, 160]


# --------------------------------------------------------------------------- inclusion probabilities, A / R


def test_random_bucket_logs_the_exact_inclusion_probability(splitter):
    pool = make_pool(u_rows(n=100, p=0.1))  # 100 windows, one node each
    res = mine_round(
        pool, budget=20, quotas={"R": 1.0}, split_of=splitter, per_match_cap=1.0, seed=5
    )
    cards = only(res, "R")
    assert len(cards) == 20
    assert {c["sampling"]["pi"] for c in cards} == {0.2} and {
        c["sampling"]["frame_units"] for c in cards
    } == {100}
    assert res.report["inclusion_probabilities"]["R"]["m1"] == {
        "frame_units": 100,
        "n_drawn": 20,
        "pi_node_mean": 0.2,
    }


def test_random_draw_is_uniform_over_the_frame(splitter):
    pool = make_pool(u_rows(n=50, p=0.1))
    hits = Counter()
    for seed in range(300):
        hits.update(
            c["frame_index"]
            for c in mine_round(
                pool,
                budget=10,
                quotas={"R": 1.0},
                split_of=splitter,
                per_match_cap=1.0,
                seed=seed,
            ).cards
        )
    freq = np.array([hits[i * 200] / 300 for i in range(50)])
    assert abs(freq.mean() - 0.2) < 1e-9 and np.abs(freq - 0.2).max() < 0.12


def test_node_probability_inside_a_unit_is_one_over_its_size(splitter):
    # window 0 holds 4 nodes, window 1 holds 1: pi_node = n/N * 1/m
    pool = make_pool(
        [{"frame": f, "p_real": 0.1, "x": 0.1 * (f + 1)} for f in (0, 1, 2, 3)]
        + [{"frame": 60, "p_real": 0.1}]
    )
    res = mine_round(
        pool, budget=1, quotas={"R": 1.0}, split_of=splitter, per_match_cap=1.0, seed=1
    )
    (c,) = res.cards
    m = 4 if c["frame_index"] < 50 else 1
    assert c["sampling"]["pi"] == pytest.approx(1 / 2 / m)


def test_audit_bucket_is_stratified_by_match_and_band_with_pi_per_stratum(splitter):
    rows = (
        u_rows("m1", "m1_seg001", n=60, p=0.55)
        + u_rows("m1", "m1_seg002", n=30, p=0.97)
        + u_rows("m2", "m2_seg001", n=60, p=0.55)
    )
    res = mine_round(
        make_pool(rows),
        budget=20,
        quotas={"A": 1.0},
        split_of=splitter,
        per_match_cap=1.0,
        seed=6,
    )
    cards = only(res, "A")
    assert len(cards) == 20
    assert {c["sampling"]["band"] for c in cards} == {"0.50-0.70", "0.95-0.98"}
    assert Counter(c["match"] for c in cards) == {"m1": 10, "m2": 10}  # equal per match
    inc = res.report["inclusion_probabilities"]["A"]
    for c in cards:
        s = c["sampling"]
        assert s["pi"] == pytest.approx(
            s["n_drawn"] / s["frame_units"]
        )  # one node per window here
    assert inc["m2|0.50-0.70"]["frame_units"] == 60


# ---------------------------------------------------------------------------------------------- diversity


def test_u_cards_come_from_distinct_embedding_clusters_not_just_the_most_uncertain(
    splitter,
):
    rng = np.random.default_rng(0)
    centres = np.eye(3, 8) * 5
    rows = []
    for ci, (n, p) in enumerate(
        [(60, 0.50), (20, 0.55), (20, 0.62)]
    ):  # the biggest, most uncertain cluster would take every card
        for i in range(n):
            rows.append(
                {
                    "frame": (len(rows)) * 200,
                    "p_real": p + 0.0001 * i,
                    "embedding": (centres[ci] + rng.normal(0, 0.05, 8)).tolist(),
                }
            )
    cfg = BucketConfig(cluster_overshoot=1.0, cluster_time_weight=0.0)
    res = mine_round(
        make_pool(rows),
        budget=3,
        quotas={"U": 1.0},
        split_of=splitter,
        per_match_cap=1.0,
        cfg=cfg,
        seed=0,
    )
    assert sorted(round(c["p_real"], 2) for c in only(res, "U")) == [
        0.5,
        0.55,
        0.62,
    ]  # the best node of each cluster
    assert len({c["sampling"]["stratum"] for c in only(res, "U")}) == 3
    # without diversity (k = 1 per card would be the top-3 by uncertainty) all three would be cluster 0
    top3 = sorted(rows, key=lambda r: abs(r["p_real"] - 0.5))[:3]
    assert all(r["p_real"] < 0.51 for r in top3)


def test_u_order_inside_the_round_is_most_uncertain_first(splitter):
    pool = make_pool(
        [
            {"frame": i * 200, "p_real": p}
            for i, p in enumerate([0.9, 0.31, 0.5, 0.7, 0.45, 0.6])
        ]
    )
    res = mine_round(
        pool,
        budget=6,
        quotas={"U": 1.0},
        split_of=splitter,
        per_match_cap=1.0,
        cfg=BucketConfig(cluster_overshoot=1.0),
    )
    ranks = [c["sampling"]["rank"] for c in res.cards]
    assert ranks == sorted(ranks)


def test_missed_bucket_takes_plausible_gaps_longest_first_and_skips_implausible(
    splitter,
):
    rows = [{"frame": i * 200, "p_real": p, "gap_frame": True, "plausible": pl, "sources": ["tracker_gap"]}
            for i, (p, pl) in enumerate([(0.9, True), (0.6, True), (0.95, False), (0.1, True), (0.8, None)])]  # fmt: skip
    res = mine_round(
        make_pool(rows),
        budget=10,
        quotas={"M": 1.0},
        split_of=splitter,
        per_match_cap=1.0,
        cfg=BucketConfig(cluster_overshoot=1.0),
    )
    got = [c["p_real"] for c in only(res, "M")]
    assert got == [
        0.9,
        0.8,
        0.6,
    ]  # 0.95 is implausible, 0.1 is below the plausibility floor; best first


def test_hard_negatives_are_location_capped(splitter):
    same = [
        {"frame": i * 200, "p_real": 0.95, "static_flag": True, "x": 0.31, "y": 0.52}
        for i in range(10)
    ]
    other = [
        {
            "frame": 5000 + i * 200,
            "p_real": 0.7,
            "static_flag": True,
            "x": 0.8,
            "y": 0.1,
        }
        for i in range(10)
    ]
    res = mine_round(
        make_pool(same + other),
        budget=10,
        quotas={"N": 1.0},
        split_of=splitter,
        per_match_cap=1.0,
    )
    cards = only(res, "N")
    assert len(cards) == 4  # two per location cell, however many static boxes there are
    assert Counter((round(c["bbox"]["x"], 1)) for c in cards) == {0.3: 2, 0.8: 2}


# ---------------------------------------------------------------------------------------------- gold


def gold_df(n=30, kind="gt_shifted"):
    return pd.DataFrame(
        [
            {
                "match": "m1",
                "clip": "m1_seg001",
                "frame": 5000 + i * 200,
                "fps": 25.0,
                "x": 0.5,
                "y": 0.5,
                "w": 0.01,
                "h": 0.02,
                "kind": kind,
            }
            for i in range(n)
        ]
    )


def test_gold_cards_are_the_ground_truth_box_shifted_15_to_30_px(big_pool, splitter):
    res = mine_round(big_pool, budget=300, split_of=splitter, gold=gold_df(), seed=9)
    g = only(res, "G")
    assert len(g) == 15
    for c in g:
        b = c["bbox"]
        dx = (b["x"] + b["w"] / 2 - 0.5) * 1920
        dy = (b["y"] + b["h"] / 2 - 0.5) * 1080
        assert 15 - 0.01 <= np.hypot(dx, dy) <= 30 + 0.01
        assert c["sampling"]["gold"]["shift_px"] == pytest.approx(
            np.hypot(dx, dy), abs=0.05
        )
        assert "box_off" in c["sampling"]["gold"]["expected"]


def test_known_not_ball_gold_keeps_its_box(splitter):
    res = mine_round(
        make_pool(u_rows(n=3)),
        budget=5,
        quotas={"G": 1.0},
        split_of=splitter,
        gold=gold_df(5, "known_not_ball"),
        per_match_cap=1.0,
    )
    for c in res.cards:
        assert c["bbox"] == {"x": 0.495, "y": 0.49, "w": 0.01, "h": 0.02} and c[
            "sampling"
        ]["gold"]["expected"] == ["not_ball"]


def test_gold_from_a_heldout_match_or_eval_clip_is_refused(splitter):
    splitter.eval_clips = {"m1_seg001"}
    res = mine_round(
        make_pool(u_rows("m1", "m1_seg009", n=3)),
        budget=5,
        quotas={"G": 1.0},
        split_of=splitter,
        gold=gold_df(5),
        per_match_cap=1.0,
    )
    assert not only(res, "G") and res.report["refused"]["gold"]["rows"] == {
        "eval_in_match": 5
    }  # the quota spilled to R instead


def test_gold_cards_are_indistinguishable_from_the_others_in_what_the_phone_gets(
    big_pool, splitter, ds
):
    res = mine_round(big_pool, budget=300, split_of=splitter, gold=gold_df(), seed=11)
    job, _ = jobs.submit(
        {
            "title": "t",
            "kind": "ball_check",
            "dataset": "train",
            "items": res.cards,
            "key": "ball-round/2026-11-18/1",
        },
        "train",
    )
    seen = {}
    for it in job["items"]:
        b = it["sampling"]["bucket"]
        seen.setdefault(b, []).append(jobs.page_card(it))
    assert set(seen) == {"U", "M", "N", "R", "A", "G"}
    shapes = {b: {tuple(sorted(c)) for c in cards} for b, cards in seen.items()}
    assert len({s for v in shapes.values() for s in v}) == 1, (
        shapes
    )  # exactly one payload shape for every bucket
    (shape,) = shapes["G"]
    assert set(shape) == {
        "clip",
        "frame_index",
        "box_index",
        "source",
        "bbox",
    }  # no label / provenance / confidence / note / role
    for cards in seen.values():
        assert all(
            set(c["bbox"]) == {"x", "y", "w", "h"} and c["source"] == "list"
            for c in cards
        )
    # everything that identifies the bucket is a hidden key (kept in results, never sent to the page)
    assert {
        "sampling",
        "p_real",
        "match",
        "sources",
        "scorer_version",
        "detector_sha",
    } <= set(jobs.hidden_keys(job["items"][0]))


def test_gold_only_matches_do_not_shrink_the_cap_of_the_only_pool_match(splitter):
    gold = pd.concat(
        [
            gold_df(5).assign(match="g1", clip="g1_seg001"),
            gold_df(5).assign(match="g2", clip="g2_seg001"),
        ],
        ignore_index=True,
    )
    res = mine_round(
        make_pool(u_rows(n=200, p=0.4)),
        budget=100,
        split_of=splitter,
        gold=gold,
        seed=1,
    )
    assert len(res) == 100 and res.report["per_match_cap"]["relaxed_to_equal_share"]
    assert Counter(c["match"] for c in res.cards)["m1"] >= 90
