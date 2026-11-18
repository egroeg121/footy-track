"""Learn-then-Test (design §3.3): validity, conservatism at small n, exchangeability."""

from __future__ import annotations

import numpy as np
import pytest

from footy_track.ball_mining.threshold import (
    FALLBACK_SCALE,
    ExchangeabilityError,
    audit_rows,
    binom_sf,
    clopper_pearson_lower,
    learn_then_test,
    learn_then_test_per_match,
)


def q(s):
    """The known precision curve: P(correct | p_real = s)."""
    return 1.0 - 0.5 * (1.0 - s)


def true_precision_above(t):
    """Mean of q over s ~ U[t, 1] (what auto-accepting everything >= t really achieves)."""
    return 1.0 - 0.5 * (1.0 - t) / 2.0


def synth(n, seed, *, lo=0.5, match="m1", curve=q):
    rng = np.random.default_rng(seed)
    s = rng.uniform(lo, 1.0, n)
    ok = rng.uniform(size=n) < curve(s)
    return [
        {
            "bucket": "A",
            "p_real": float(a),
            "verdict": "ball" if o else "not_ball",
            "match": match,
            "scorer_version": "v1",
        }
        for a, o in zip(s, ok, strict=True)
    ]


def test_exact_binomial_tail_matches_closed_forms():
    assert binom_sf(114, 114, 0.98) == pytest.approx(0.98**114)
    assert binom_sf(0, 10, 0.5) == 1.0 and binom_sf(11, 10, 0.5) == 0.0
    assert binom_sf(8, 10, 0.5) == pytest.approx((45 + 10 + 1) / 1024)
    assert clopper_pearson_lower(10, 10, 0.1) == pytest.approx(
        0.1 ** (1 / 10), abs=1e-6
    )


def test_picks_a_threshold_whose_true_precision_meets_the_target():
    r = learn_then_test(synth(10000, 0), target=0.98, delta=0.1)
    assert r.status == "valid" and r.threshold is not None and not r.fallback
    assert (
        true_precision_above(r.threshold) >= 0.98 - 0.003
    )  # the curve crosses 0.98 at t = 0.92
    assert r.precision >= 0.98 and r.p_value <= 0.1 and r.n >= 114
    assert r.effective_threshold == r.threshold and r.effective_scale == "p_real"
    assert r.precision_lower_bound is not None and r.precision_lower_bound < r.precision


def test_error_rate_of_a_wrong_threshold_stays_below_delta():
    """Over many audit samples the chosen t has true precision < target in at most ~delta of them."""
    bad = valid = 0
    for seed in range(100):
        r = learn_then_test(synth(3000, 100 + seed), target=0.98, delta=0.1)
        if r.status == "valid":
            valid += 1
            bad += true_precision_above(r.threshold) < 0.98
    assert valid > 30  # it is not vacuously conservative
    assert bad / 100 <= 0.1 + 0.05


def test_it_is_conservative_at_small_n_and_records_the_fallback():
    rows = [
        {"bucket": "A", "p_real": 0.99, "verdict": "ball"} for _ in range(60)
    ]  # 60 / 60 correct: 0.98**60 = 0.30 > 0.1
    r = learn_then_test(rows)
    assert r.status == "no_valid_threshold" and r.threshold is None
    assert (
        r.fallback
        and r.effective_threshold == 0.95
        and r.effective_scale == FALLBACK_SCALE == r.scale
    )
    ok = learn_then_test(
        [{"bucket": "A", "p_real": 0.99, "verdict": "ball"} for _ in range(120)]
    )  # 0.98**120 = 0.089
    assert ok.status == "valid" and ok.threshold == 0.5 and ok.n == 120


def test_one_wrong_audit_needs_far_more_evidence():
    def one_wrong(n):
        return [
            {"bucket": "A", "p_real": 0.99, "verdict": "ball"} for _ in range(n)
        ] + [{"bucket": "A", "p_real": 0.99, "verdict": "box_off"}]

    assert (
        learn_then_test(one_wrong(150)).status == "no_valid_threshold"
    )  # P(Bin(151, .98) >= 150) = 0.19
    assert (
        learn_then_test(one_wrong(200)).status == "valid"
    )  # P(Bin(201, .98) >= 200) = 0.088


def test_threshold_walks_down_only_while_every_step_is_rejected():
    rows = [{"bucket": "A", "p_real": 0.995, "verdict": "ball"} for _ in range(150)]
    rows += [{"bucket": "A", "p_real": 0.85, "verdict": "not_ball"} for _ in range(40)]
    r = learn_then_test(rows)
    assert (
        r.status == "valid" and r.threshold >= 0.86
    )  # the 0.85 band is wrong: the sequence stops above it
    assert r.n == 150 and r.k == 150


def test_unsure_is_excluded_and_box_off_counts_as_wrong():
    rows = [
        {"bucket": "A", "p_real": 0.99, "verdict": v}
        for v in ["ball"] * 100 + ["corrected"] * 30 + ["unsure"] * 500
    ]
    r = learn_then_test(rows)
    assert (r.n_audits, r.n_unsure, r.n, r.k) == (130, 500, 130, 130)
    wrong = learn_then_test(
        [
            {"bucket": "A", "p_real": 0.99, "verdict": v}
            for v in ["ball"] * 130 + ["box_off"] * 5 + ["not_ball"] * 5
        ]
    )
    assert wrong.status == "no_valid_threshold" and wrong.precision == pytest.approx(
        130 / 140
    )


def test_non_audit_verdicts_are_rejected_by_bucket_tag():
    rows = synth(300, 1)
    for bad in ("U", "M", "N", "R", "G", None):
        with pytest.raises(ExchangeabilityError, match="A-bucket"):
            learn_then_test([*rows, {"bucket": bad, "p_real": 0.99, "verdict": "ball"}])
        with pytest.raises(ExchangeabilityError):
            learn_then_test_per_match(
                [
                    *rows,
                    {"bucket": bad, "p_real": 0.99, "verdict": "ball", "match": "m1"},
                ]
            )
    assert (
        len(audit_rows([*rows, {"bucket": "U", "p_real": 0.99, "verdict": "ball"}]))
        == 300
    )


def test_audits_from_two_scorer_versions_are_not_pooled():
    rows = [
        {"bucket": "A", "p_real": 0.99, "verdict": "ball", "scorer_version": v}
        for v in ["v1"] * 70 + ["v2"] * 70
    ]
    with pytest.raises(ExchangeabilityError, match="scorer"):
        learn_then_test(rows)


def test_per_match_thresholds_differ_when_the_matches_do():
    a = synth(8000, 5, match="easy")
    b = synth(
        8000, 6, match="hard", curve=lambda s: 1.0 - 3.2 * (1.0 - s)
    )  # needs s >= ~0.99 for 98 %
    res = learn_then_test_per_match(a + b)
    assert set(res) == {"easy", "hard"} and res["easy"].status == "valid"
    assert (
        res["hard"].status != "valid" or res["hard"].threshold > res["easy"].threshold
    )


def test_unequal_inclusion_probabilities_use_a_conservative_effective_sample():
    even = [
        {"bucket": "A", "p_real": 0.99, "verdict": "ball", "pi": 0.1}
        for _ in range(130)
    ]
    assert learn_then_test(even).n == 130
    skew = [
        {"bucket": "A", "p_real": 0.99, "verdict": "ball", "pi": 0.5} for _ in range(65)
    ] + [
        {"bucket": "A", "p_real": 0.99, "verdict": "ball", "pi": 0.05}
        for _ in range(65)
    ]
    r = learn_then_test(skew)
    assert r.n < 130  # the Kish effective n is smaller than the raw count
