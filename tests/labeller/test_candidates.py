"""Confidence-stratified and adaptive sampling (README §11, LAB-11xx).

These are statistical properties, so they are asserted over a synthetic pool
big enough for the shape to be stable, not over a handful of rows.
"""

from __future__ import annotations

import json

import pytest

from footy_track.labeller import candidates as cand

PRESENT = ("ball", "box_off", "corrected")


def _pool(n: int = 2000) -> list[dict]:
    """Candidates spread evenly over the confidence range."""
    return [
        {
            "clip": "clip",
            "frame_index": i,
            "cand_index": 0,
            "confidence": (i + 0.5) / n,
            "bbox": {"x": 0.5, "y": 0.5, "w": 0.01, "h": 0.01},
            "label": "in_play_ball",
        }
        for i in range(n)
    ]


def _write_cand_file(cand_dir, stem: str, rows: list[dict]) -> None:
    (cand_dir / f"{stem}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )


# ---------------------------------------------------------------------------
# Reading (LAB-1101)
# ---------------------------------------------------------------------------


def test_read_candidates_keeps_ball_rows_and_numbers_them(cand_dir):
    _write_cand_file(
        cand_dir,
        "clip",
        [
            {
                "frame_index": 0,
                "bbox": {"x": 0.1, "y": 0.1, "w": 0.01, "h": 0.01},
                "tags": ["player", "rtdetr"],
                "confidence": 0.9,
            },
            {
                "frame_index": 0,
                "bbox": {"x": 0.2, "y": 0.2, "w": 0.01, "h": 0.01},
                "tags": ["in_play_ball", "rtdetr"],
                "confidence": 0.42,
            },
            {
                "frame_index": 0,
                "bbox": {"x": 0.3, "y": 0.3, "w": 0.01, "h": 0.01},
                "tags": ["in_play_ball", "rtdetr"],
                "confidence": 0.13,
            },
            {
                "frame_index": 1,
                "bbox": {"x": 0.4, "y": 0.4, "w": 0.01, "h": 0.01},
                "tags": ["out_of_play_ball", "rtdetr"],
                "confidence": 0.77,
            },
        ],
    )
    rows = cand.read_candidates()
    assert [r["label"] for r in rows] == [
        "in_play_ball",
        "in_play_ball",
        "out_of_play_ball",
    ]
    # cand_index is per-frame, in file order — two ball rows on frame 0.
    assert [(r["frame_index"], r["cand_index"]) for r in rows] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert rows[0]["confidence"] == pytest.approx(0.42)


def test_read_candidate_box_round_trips(cand_dir):
    _write_cand_file(
        cand_dir,
        "clip",
        [
            {
                "frame_index": 7,
                "bbox": {"x": 0.25, "y": 0.5, "w": 0.01, "h": 0.02},
                "tags": ["in_play_ball", "rtdetr"],
                "confidence": 0.3,
            }
        ],
    )
    assert cand.read_candidate_box("clip", 7, 0) == (0.25, 0.5, 0.01, 0.02)
    assert cand.read_candidate_box("clip", 7, 3) is None


def test_read_candidates_skips_malformed_lines(cand_dir):
    (cand_dir / "clip.jsonl").write_text(
        "\n".join(
            [
                "not json",
                "",
                json.dumps({"frame_index": 0, "tags": ["in_play_ball"], "bbox": None}),
                json.dumps(
                    {
                        "bbox": {"x": 0, "y": 0, "w": 1, "h": 1},
                        "tags": ["in_play_ball"],
                        "confidence": 0.5,
                    }
                ),
                json.dumps(
                    {
                        "frame_index": 2,
                        "bbox": {"x": 0.1, "y": 0.1, "w": 0.01, "h": 0.01},
                        "tags": ["in_play_ball"],
                        "confidence": 0.5,
                    }
                ),
            ]
        )
        + "\n"
    )
    assert [r["frame_index"] for r in cand.read_candidates()] == [2]


# ---------------------------------------------------------------------------
# Static stratification (LAB-1102)
# ---------------------------------------------------------------------------


def test_sampling_weight_peaks_at_tau_and_never_reaches_zero():
    w_at = cand.sampling_weight(0.25, tau=0.25)
    w_near = cand.sampling_weight(0.30, tau=0.25)
    w_far = cand.sampling_weight(0.95, tau=0.25)
    assert w_at > w_near > w_far
    # The uniform floor is what keeps the far tail sampled at all.
    assert w_far >= 1.0 - cand.DEFAULT_FOCUS


def test_stratified_order_concentrates_hard_on_the_boundary():
    order = cand.stratified_order(_pool(), tau=0.25)
    first = order[:200]
    near = sum(1 for r in first if abs(r["confidence"] - 0.25) <= 0.1)
    # sigma is 0.10, so ~2/3 of the weighted stream lands within one sigma of
    # the boundary — the Gaussian shape, not an arbitrary number. Coverage of
    # the rest of the range is the uniform stream's job (see the select tests).
    assert near > 110
    within_two_sigma = sum(1 for r in first if abs(r["confidence"] - 0.25) <= 0.2)
    assert within_two_sigma > 170


def test_stratified_order_is_deterministic():
    a = cand.stratified_order(_pool(300), tau=0.25)
    b = cand.stratified_order(_pool(300), tau=0.25)
    assert [r["frame_index"] for r in a] == [r["frame_index"] for r in b]


def test_stratified_order_is_a_permutation():
    pool = _pool(300)
    order = cand.stratified_order(pool, tau=0.25)
    assert sorted(r["frame_index"] for r in order) == sorted(
        r["frame_index"] for r in pool
    )


# ---------------------------------------------------------------------------
# Adaptive sampling (LAB-1103)
# ---------------------------------------------------------------------------


def _verdicts(pairs: list[tuple[float, str]]) -> list[dict]:
    return [{"confidence": c, "verdict": v} for c, v in pairs]


def test_band_stats_counts_decided_and_ignores_unsure():
    stats = cand.band_stats(
        _verdicts([(0.25, "ball"), (0.25, "not_ball"), (0.25, "unsure")]), PRESENT
    )
    band = stats[cand.band_of(0.25)]
    assert band["decided"] == 2
    assert band["precision"] == pytest.approx(0.5)


def test_box_off_counts_as_found_for_the_boundary_estimate():
    stats = cand.band_stats(_verdicts([(0.35, "box_off"), (0.35, "ball")]), PRESENT)
    assert stats[cand.band_of(0.35)]["precision"] == pytest.approx(1.0)


def test_estimate_tau_falls_back_until_there_is_evidence():
    assert cand.estimate_tau(cand.band_stats([], PRESENT)) == cand.DEFAULT_TAU
    thin = cand.band_stats(_verdicts([(0.25, "ball"), (0.25, "not_ball")]), PRESENT)
    assert cand.estimate_tau(thin) == cand.DEFAULT_TAU


def test_estimate_tau_finds_the_measured_crossing():
    # Everything below 0.5 is wrong, everything above is right: the boundary
    # sits near 0.5, not at the configured default of 0.25.
    pairs = []
    for conf in (0.15, 0.25, 0.35, 0.45):
        pairs += [(conf, "not_ball")] * 4
    for conf in (0.55, 0.65, 0.75, 0.85):
        pairs += [(conf, "ball")] * 4
    tau = cand.estimate_tau(cand.band_stats(_verdicts(pairs), PRESENT))
    assert 0.45 <= tau <= 0.65
    assert tau != cand.DEFAULT_TAU


def test_adaptive_order_follows_the_measured_boundary():
    pairs = []
    for conf in (0.15, 0.25, 0.35, 0.45):
        pairs += [(conf, "not_ball")] * 4
    for conf in (0.55, 0.65, 0.75, 0.85):
        pairs += [(conf, "ball")] * 4
    order, tau, _stats = cand.adaptive_order(_pool(), _verdicts(pairs), PRESENT)
    near_new = sum(1 for r in order[:200] if abs(r["confidence"] - tau) <= 0.1)
    near_default = sum(
        1 for r in order[:200] if abs(r["confidence"] - cand.DEFAULT_TAU) <= 0.1
    )
    assert near_new > near_default


def test_adaptive_weight_boosts_bands_with_little_evidence():
    stats = cand.band_stats(_verdicts([(0.85, "ball")] * 50), PRESENT)
    well_measured = cand.adaptive_weight(0.85, 0.25, stats)
    unmeasured = cand.adaptive_weight(0.95, 0.25, stats)
    # Same distance-from-tau region, but the thin band is preferred.
    assert unmeasured > well_measured


# ---------------------------------------------------------------------------
# Binned pool (LAB-1104)
# ---------------------------------------------------------------------------


def test_binned_pool_matches_a_full_race():
    # The fast path must be an optimisation, not a different sampler.
    pool_records = _pool(600)
    weight = lambda c: cand.sampling_weight(c, tau=0.25)  # noqa: E731
    fast = cand.BinnedPool(pool_records).select(weight, top=50, uniform_share=0)
    slow = cand._race(pool_records, lambda r: weight(cand.bin_centre(r["confidence"])))[
        :50
    ]
    assert [r["frame_index"] for r in fast] == [r["frame_index"] for r in slow]


def test_binned_pool_skips_judged_and_still_fills_the_page():
    pool_records = _pool(600)
    skip_ids = {r["frame_index"] for r in pool_records[:300]}
    out = cand.BinnedPool(pool_records).select(
        lambda c: cand.sampling_weight(c, tau=0.25),
        top=40,
        skip=lambda r: r["frame_index"] in skip_ids,
        uniform_share=0,
    )
    assert len(out) == 40
    assert not any(r["frame_index"] in skip_ids for r in out)


def test_select_serves_about_one_quarter_uniform_cards():
    # A lopsided pool, like the real one (201k of 357k candidates sit in one
    # band): a weight-expressed "25% uniform" would not deliver 25% uniform
    # cards, because a big band floods the weighted stream by sheer count.
    lopsided = [
        {"clip": "clip", "frame_index": i, "cand_index": 0, "confidence": 0.25}
        for i in range(400)
    ] + [
        {"clip": "clip", "frame_index": 10_000 + i, "cand_index": 0, "confidence": 0.95}
        for i in range(5000)
    ]
    page = cand.BinnedPool(lopsided).select(
        lambda c: cand.sampling_weight(c, tau=0.25), top=40
    )
    assert len(page) == 40
    # Distinct cards only: the two streams must not serve the same candidate.
    assert len({(r["frame_index"], r["cand_index"]) for r in page}) == 40
    far = sum(1 for r in page if r["confidence"] > 0.9)
    # The far band can only reach the page through the uniform quarter, which
    # it dominates (5000 of 5400 rows) — so ~10 of 40, never most of the page.
    assert 6 <= far <= 14


def test_select_mix_holds_for_a_short_page():
    page = cand.BinnedPool(_pool(2000)).select(
        lambda c: cand.sampling_weight(c, tau=0.25), top=8
    )
    assert len(page) == 8
