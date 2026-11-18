"""Panel estimator (footy_track.ball_eval.panel_score): outcomes, Horvitz-Thompson ratios on unequal pi,
stratified cluster bootstrap CIs, QA exclusion, and the prediction / label loaders."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from footy_track.ball_eval import panel as P
from footy_track.ball_eval import panel_score as S
from footy_track.scripts import ball_eval_panel as cli

W = H = 1000  # 1 normalised unit = 1000 px, so px values are easy to read
BALL = S.Human("ball", [(0.5, 0.5)])
NO_BALL = S.Human("no_ball", [])
NOT_BC = S.Human("not_broadcast", [])


def window(match, i, clip="c0", gap=5, base=0, pi=0.5):
    frames = [base + k * gap for k in range(5)]
    return {
        "window_id": f"{match}:w{i}",
        "clip": clip,
        "segment": 0,
        "frame_start": frames[0],
        "frame_end": frames[-1],
        "step": gap,
        "frames": frames,
        "pi_window": pi,
        "pi_frame": pi,
    }


def panel(match, windows, qa=(), version="ball-eval-v1"):
    return {
        "panel_version": version,
        "match": match,
        "width": W,
        "height": H,
        "windows": windows,
        "qa": list(qa),
        "code_sha": "t",
    }


def synth(match, spec, pi, clip="c0"):
    """Windows from ``spec = [(present_mask, hit_mask), ...]`` -> (panel, human, preds). A present frame has the ball
    at the centre; a hit has a prediction 3 px away; a present non-hit frame has NO prediction (miss)."""
    wins, human, preds = [], {}, {}
    for i, (present, hit) in enumerate(spec):
        w = window(match, i, clip, base=i * 100, pi=pi)
        wins.append(w)
        for f, pr, ht in zip(w["frames"], present, hit, strict=True):
            human[(clip, f)] = BALL if pr else NO_BALL
            if pr and ht:
                preds[(clip, f)] = (503.0, 500.0, 0.9)
    return panel(match, wins), human, preds


# ---- outcomes -------------------------------------------------------------------------------------
def test_classify_every_outcome_at_two_radii():
    size = (W, H)
    near, mid, far = (503.0, 500.0, 1.0), (515.0, 500.0, 1.0), (560.0, 500.0, 1.0)
    assert S.classify(BALL, near, 10, size) == "hit"
    assert (
        S.classify(BALL, mid, 10, size) == "off"
        and S.classify(BALL, mid, 25, size) == "hit"
    )
    assert S.classify(BALL, far, 25, size) == "off"
    assert S.classify(BALL, None, 10, size) == "miss"
    assert S.classify(NO_BALL, near, 10, size) == "false_ball"
    assert S.classify(NO_BALL, None, 10, size) == "true_empty"
    assert S.classify(NOT_BC, near, 10, size) is None  # out of the population
    # the scale is pixels of the frame, not normalised units: a 1280x720 frame
    assert S.classify(BALL, (640 + 9.0, 360.0, 1), 10, (1280, 720)) == "hit"
    assert S.classify(BALL, (640 + 11.0, 360.0, 1), 10, (1280, 720)) == "off"
    # several human centres in a frame (ball + spare ball): nearest one counts
    assert (
        S.classify(S.Human("ball", [(0.1, 0.1), (0.5, 0.5)]), near, 10, size) == "hit"
    )


def test_ratios_and_undefined_denominators():
    r = S.ratios(
        np.array([6.0, 2.0, 2.0, 1.0, 9.0])
    )  # hit off miss false_ball true_empty
    assert (
        r["recall"] == pytest.approx(0.6)
        and r["precision"] == pytest.approx(6 / 9)
        and r["false_ball"] == pytest.approx(0.1)
    )
    z = S.ratios(np.zeros(5))
    assert all(math.isnan(float(v)) for v in z.values())


# ---- hand-computed score ----------------------------------------------------------------------------
def test_score_matches_a_hand_computation_and_reports_the_radius_effect():
    wins = [window("m", 0, base=0, pi=0.25), window("m", 1, base=100, pi=0.25)]
    human, preds = {}, {}
    f0, f1 = wins[0]["frames"], wins[1]["frames"]
    for f in f0:  # window 0: hit, off@10(15px), miss, false-ball, true-empty
        human[("c0", f)] = BALL if f != f0[3] and f != f0[4] else NO_BALL
    preds[("c0", f0[0])] = (503.0, 500.0, 0.9)
    preds[("c0", f0[1])] = (515.0, 500.0, 0.9)
    preds[("c0", f0[3])] = (100.0, 100.0, 0.9)
    for f in f1:  # window 1: all five hit
        human[("c0", f)] = BALL
        preds[("c0", f)] = (498.0, 501.0, 0.9)
    rep = S.score([panel("m", wins)], human, preds, draws=50, seed=1)
    m = rep["matches"]["m"]
    assert m["counts"]["r10"] == {
        "hit": 6,
        "off": 1,
        "miss": 1,
        "false_ball": 1,
        "true_empty": 1,
    }
    assert m["counts"]["r25"] == {
        "hit": 7,
        "off": 0,
        "miss": 1,
        "false_ball": 1,
        "true_empty": 1,
    }
    assert m["metrics"]["r10"]["recall"]["est"] == pytest.approx(6 / 8, abs=1e-4)
    assert m["metrics"]["r25"]["recall"]["est"] == pytest.approx(7 / 8, abs=1e-4)
    assert m["metrics"]["r10"]["precision"]["est"] == pytest.approx(
        6 / 8, abs=1e-4
    )  # hit+off+false_ball = 8
    assert m["metrics"]["r10"]["false_ball"]["est"] == pytest.approx(0.5, abs=1e-4)
    assert rep["units"].startswith("pixels") and rep["radii_px"] == [10.0, 25.0]
    # one match: match-averaged == frame-weighted == the match
    for k in ("recall", "precision", "false_ball"):
        assert (
            rep["pooled"]["match_averaged"]["r10"][k]["est"]
            == rep["pooled"]["frame_weighted"]["r10"][k]["est"]
            == m["metrics"]["r10"][k]["est"]
        )


def test_pooled_match_averaged_vs_frame_weighted_with_unequal_pi():
    # match A: recall 1.0, pi 0.5; match B: recall 0.0, pi 0.05 (so each B frame stands for 10x more frames)
    pa, ha, qa_ = synth("A", [([1] * 5, [1] * 5)] * 4, pi=0.5, clip="ca")
    pb, hb, qb = synth("B", [([1] * 5, [0] * 5)] * 4, pi=0.05, clip="cb")
    rep = S.score([pa, pb], {**ha, **hb}, {**qa_, **qb}, draws=20, seed=0)
    assert rep["pooled"]["match_averaged"]["r10"]["recall"]["est"] == pytest.approx(0.5)
    # HT totals: A = 20 frames / 0.5 = 40 hits of 40; B = 20 / 0.05 = 400 present, 0 hits -> 40 / 440
    assert rep["pooled"]["frame_weighted"]["r10"]["recall"]["est"] == pytest.approx(
        40 / 440, abs=1e-4
    )


# ---- Horvitz-Thompson unbiasedness on unequal pi -------------------------------------------------------
def _population(rng):
    """Two strata, very different sampling fractions and recalls. Per window: (present mask, hit mask)."""

    def stratum(n_w, p_present, q_hit):
        spec = []
        for _ in range(n_w):
            present = (rng.random(5) < p_present).astype(int)
            hit = present * (rng.random(5) < q_hit).astype(int)
            spec.append((present, hit))
        return spec

    return {"A": (stratum(300, 0.8, 0.4), 15), "B": (stratum(100, 0.8, 0.9), 50)}


def test_ht_estimator_is_unbiased_with_unequal_inclusion_probabilities():
    rng = np.random.default_rng(7)
    pop = _population(rng)
    hits = {k: np.array([h.sum() for _, h in spec]) for k, (spec, _) in pop.items()}
    pres = {k: np.array([p.sum() for p, _ in spec]) for k, (spec, _) in pop.items()}
    true_recall = sum(h.sum() for h in hits.values()) / sum(
        p.sum() for p in pres.values()
    )
    true_hits, true_pres = (
        sum(h.sum() for h in hits.values()),
        sum(p.sum() for p in pres.values()),
    )

    ratios, naive, tot_hits, tot_pres = [], [], [], []
    for _ in range(4000):
        th = tp = 0.0
        nh = npres = 0
        for k, (spec, n) in pop.items():
            idx = rng.choice(len(spec), size=n, replace=False)
            pi = np.full(n, n / len(spec))
            th += S.ht_total(hits[k][idx], pi)
            tp += S.ht_total(pres[k][idx], pi)
            nh += hits[k][idx].sum()
            npres += pres[k][idx].sum()
        tot_hits.append(th)
        tot_pres.append(tp)
        ratios.append(th / tp)
        naive.append(nh / npres)
    # totals are exactly unbiased; the ratio is consistent (tiny ratio bias at n=15/50)
    assert np.mean(tot_hits) == pytest.approx(true_hits, rel=0.01)
    assert np.mean(tot_pres) == pytest.approx(true_pres, rel=0.01)
    assert np.mean(ratios) == pytest.approx(true_recall, abs=0.01)
    # ... while ignoring pi is badly biased towards the over-sampled, high-recall stratum
    assert abs(np.mean(naive) - true_recall) > 0.1


def test_score_pooled_frame_weighted_is_unbiased_end_to_end():
    rng = np.random.default_rng(11)
    pop = _population(rng)
    hits = sum(h.sum() for spec, _ in pop.values() for _, h in spec)
    pres = sum(p.sum() for spec, _ in pop.values() for p, _ in spec)
    est = []
    for _ in range(500):
        panels, human, preds = [], {}, {}
        for k, (spec, n) in pop.items():
            idx = rng.choice(len(spec), size=n, replace=False)
            p, h, q = synth(k, [spec[i] for i in idx], pi=n / len(spec), clip=f"c{k}")
            panels.append(p)
            human.update(h)
            preds.update(q)
        est.append(
            S.score(panels, human, preds, draws=2)["pooled"]["frame_weighted"]["r10"][
                "recall"
            ]["est"]
        )
    assert np.mean(est) == pytest.approx(hits / pres, abs=0.012)


# ---- bootstrap ----------------------------------------------------------------------------------------
def test_bootstrap_is_seedable_and_brackets_the_estimate():
    rng = np.random.default_rng(3)
    spec = [([1] * 5, [int(rng.random() < 0.6)] * 5) for _ in range(24)]
    p, h, q = synth("m", spec, pi=0.1)
    a = S.score([p], h, q, draws=300, seed=5)
    b = S.score([p], h, q, draws=300, seed=5)
    c = S.score([p], h, q, draws=300, seed=6)
    assert a == b and a != c
    cell = a["matches"]["m"]["metrics"]["r10"]["recall"]
    assert cell["lo"] <= cell["est"] <= cell["hi"] and cell["hi"] - cell["lo"] > 0.05


def test_cluster_bootstrap_ci_has_roughly_nominal_coverage():
    """Windows are perfectly correlated inside (all 5 frames share the hit status), so only a CLUSTER bootstrap
    covers: a frame-level one would be ~sqrt(5) too narrow (coverage ~0.6)."""
    rng = np.random.default_rng(2024)
    n_pop, n = 400, 24
    pop_hit = (rng.random(n_pop) < 0.6).astype(int)
    truth = pop_hit.mean()
    covered, reps = 0, 300
    for r in range(reps):
        idx = rng.choice(n_pop, size=n, replace=False)
        p, h, q = synth(
            "m", [([1] * 5, [int(pop_hit[i])] * 5) for i in idx], pi=n / n_pop
        )
        cell = S.score([p], h, q, draws=400, seed=r)["matches"]["m"]["metrics"]["r10"][
            "recall"
        ]
        covered += cell["lo"] <= truth <= cell["hi"]
    assert 0.88 <= covered / reps <= 0.99


def test_bootstrap_resamples_windows_within_each_match_not_across():
    # B has no variation, A has: B's CI must be degenerate however the draws go
    pa, ha, qa_ = synth(
        "A", [([1] * 5, [1] * 5), ([1] * 5, [0] * 5)] * 5, pi=0.5, clip="ca"
    )
    pb, hb, qb = synth("B", [([1] * 5, [1] * 5)] * 6, pi=0.5, clip="cb")
    rep = S.score([pa, pb], {**ha, **hb}, {**qa_, **qb}, draws=200, seed=0)
    b = rep["matches"]["B"]["metrics"]["r10"]["recall"]
    a = rep["matches"]["A"]["metrics"]["r10"]["recall"]
    assert b["lo"] == b["hi"] == b["est"] == 1.0
    assert a["lo"] < a["est"] < a["hi"]


# ---- exclusions -----------------------------------------------------------------------------------------
def test_qa_cards_never_enter_the_estimate_and_are_reported_separately():
    spec = [([1] * 5, [1, 1, 1, 0, 0]) for _ in range(6)]
    p, human, preds = synth("m", spec, pi=0.2)
    base = S.score([p], human, preds, draws=100, seed=1)
    w0 = p["windows"][0]
    qa = [
        {
            "qa": "repeat",
            "clip": "c0__b",
            "frame": w0["frames"][0],
            "of": {
                "clip": "c0",
                "frame": w0["frames"][0],
                "window_id": w0["window_id"],
            },
        },
        {
            "qa": "repeat",
            "clip": "c0__b",
            "frame": w0["frames"][1],
            "of": {
                "clip": "c0",
                "frame": w0["frames"][1],
                "window_id": w0["window_id"],
            },
        },
        {
            "qa": "gold",
            "clip": "c0__b",
            "frame": 7,
            "of": {"clip": "c0", "frame": 7},
            "gold": {"x": 0.3, "y": 0.3, "unit": "normalised"},
        },
    ]
    p2 = {**p, "qa": qa}
    human2 = {
        **human,
        ("c0__b", w0["frames"][0]): S.Human(
            "ball", [(0.5 + 4 / W, 0.5)]
        ),  # 4 px from the first placement
        ("c0__b", w0["frames"][1]): S.Human("no_ball", []),  # disagrees on presence
        ("c0__b", 7): S.Human("ball", [(0.3 + 12 / W, 0.3)]),
    }  # 12 px from gold
    preds2 = {
        **preds,
        ("c0__b", 7): (0.0, 0.0, 1.0),
    }  # even a prediction on a QA frame is ignored
    with_qa = S.score([p2], human2, preds2, draws=100, seed=1)
    assert (
        with_qa["matches"] == base["matches"] and with_qa["pooled"] == base["pooled"]
    )  # estimates untouched
    rep = with_qa["qa"]["repeat"]
    assert (
        rep["cards"] == 2
        and rep["both_ball"] == 1
        and rep["presence_disagree"] == 1
        and rep["pending"] == 0
    )
    assert (
        rep["spread"]["n"] == 1
        and rep["spread"]["median_px"] == 4.0
        and rep["spread"]["frac_le_5px"] == 1.0
    )
    gold = with_qa["qa"]["gold"]
    assert (
        gold["cards"] == 1
        and gold["placed"] == 1
        and gold["agreement"]["median_px"] == 12.0
    )
    assert gold["agreement"]["frac_le_10px"] == 0.0


def test_not_broadcast_frames_leave_the_population_and_unlabelled_windows_are_dropped():
    p, human, preds = synth("m", [([1] * 5, [1] * 5)] * 3, pi=0.5)
    w0, w1 = p["windows"][0], p["windows"][1]
    human[("c0", w0["frames"][0])] = NOT_BC
    del human[("c0", w1["frames"][2])]  # one frame of window 1 not labelled yet
    rep = S.score([p], human, preds, draws=10)["matches"]["m"]
    assert (
        rep["n_windows"] == 3
        and rep["n_windows_scored"] == 2
        and rep["n_windows_dropped_incomplete"] == 1
    )
    assert rep["n_frames_not_broadcast"] == 1 and rep["n_frames_unlabelled"] == 1
    assert (
        sum(rep["counts"]["r10"].values()) == 5 + 4
    )  # window 2 (5) + window 0 minus the not-broadcast frame
    part = S.score([p], human, preds, draws=10, include_partial=True)["matches"]["m"]
    assert (
        part["n_windows_scored"] == 3
        and sum(part["counts"]["r10"].values()) == 4 + 4 + 5
    )


def test_a_panel_without_frame_size_cannot_be_scored_in_pixels():
    p, human, preds = synth("m", [([1] * 5, [1] * 5)], pi=0.5)
    p["width"] = None
    with pytest.raises(ValueError, match="frame size"):
        S.score([p], human, preds)


# ---- loaders ------------------------------------------------------------------------------------------
def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_load_predictions_top1_none_and_min_score(tmp_path):
    f = _write(
        tmp_path / "p.jsonl",
        [
            {"clip": "c", "frame": 1, "x": 10, "y": 20, "score": 0.4},
            {"clip": "c", "frame": 1, "x": 50, "y": 60, "score": 0.9},  # top-1 wins
            {
                "clip": "c",
                "frame": 2,
                "x": None,
                "y": None,
                "score": None,
            },  # explicit none
            {"clip": "c", "frame": 3, "x": 5, "y": 5, "score": 0.1},
            {"clip": "c", "frame": 4, "x": 0.5, "y": 0.25},  # no score = 1.0
        ],
    )
    got = S.load_predictions(f)
    assert (
        got[("c", 1)] == (50.0, 60.0, 0.9)
        and ("c", 2) not in got
        and got[("c", 4)][2] == 1.0
    )
    assert ("c", 3) not in S.load_predictions(f, min_score=0.3)
    assert S.load_predictions(f, units="norm", size=(1000, 500))[("c", 4)][:2] == (
        500.0,
        125.0,
    )


def test_load_human_from_sidecars_only_counts_tier1_and_markers(tmp_path):
    rows = [
        {
            "frame_index": 1,
            "bbox": {"x": 0.4, "y": 0.4, "w": 0.02, "h": 0.02},
            "center": {"x": 0.41, "y": 0.41},
            "tags": ["in_play_ball", "labeller"],
        },
        {
            "frame_index": 2,
            "bbox": {"x": 0.4, "y": 0.4, "w": 0.02, "h": 0.02},
            "center": {"x": 0.41, "y": 0.41},
            "tags": ["in_play_ball", "vittrack"],
        },
        {
            "frame_index": 3,
            "bbox": {"x": 0.4, "y": 0.4, "w": 0.02, "h": 0.02},
            "center": None,
            "tags": ["in_play_ball", "vittrack", "human_checked"],
        },
        {"frame_index": 4, "bbox": None, "center": None, "tags": ["no_ball"]},
        {"frame_index": 5, "bbox": None, "center": None, "tags": ["not_broadcast"]},
        {
            "frame_index": 6,
            "bbox": {"x": 0.1, "y": 0.1, "w": 0.02, "h": 0.02},
            "center": None,
            "tags": ["player", "labeller"],
        },
    ]
    _write(tmp_path / "c.jsonl", rows)
    _write(tmp_path / "c__b.jsonl", rows[:1])
    h = S.load_human_sidecars(tmp_path, ["c", "c__b", "missing"])
    assert {k: v.state for k, v in h.items() if k[0] == "c"} == {
        ("c", 1): "ball",
        ("c", 4): "no_ball",
        ("c", 5): "not_broadcast",
    }
    assert (
        h[("c", 1)].centres == [(0.41, 0.41)] and ("c__b", 1) in h
    )  # alias clips stay separate


def test_load_human_from_job_results(tmp_path):
    doc = {
        "labels": [
            {
                "clip": "c",
                "frame_index": 1,
                "status": "ball",
                "tier": 1,
                "label": "in_play_ball",
                "center": {"x": 0.2, "y": 0.3},
                "bbox": None,
                "tags": [],
            },
            {
                "clip": "c",
                "frame_index": 2,
                "status": "ball",
                "tier": 2,
                "center": {"x": 0.2, "y": 0.3},
                "tags": ["in_play_ball", "vittrack", "human_checked"],
            },
            {"clip": "c", "frame_index": 3, "status": "no_ball", "tags": ["no_ball"]},
        ]
    }
    f = tmp_path / "r.json"
    f.write_text(json.dumps(doc))
    h = S.load_human_results(f)
    assert sorted((k[1], v.state) for k, v in h.items()) == [
        (1, "ball"),
        (3, "no_ball"),
    ]


def test_machine_labels_adapter_takes_the_top_ball_of_each_frame(tmp_path):
    f = _write(
        tmp_path / "c.jsonl",
        [
            {
                "frame_index": 1,
                "bbox": {"x": 0.4, "y": 0.4, "w": 0.02, "h": 0.02},
                "confidence": 0.3,
                "tags": ["in_play_ball"],
            },
            {
                "frame_index": 1,
                "bbox": {"x": 0.6, "y": 0.6, "w": 0.02, "h": 0.02},
                "confidence": 0.8,
                "tags": ["in_play_ball"],
            },
            {
                "frame_index": 1,
                "bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.2},
                "confidence": 0.99,
                "tags": ["player"],
            },
            {
                "frame_index": 2,
                "bbox": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.2},
                "confidence": 0.99,
                "tags": ["player"],
            },
        ],
    )
    got = S.predictions_from_machine_labels(f, "c", (1000, 1000))
    assert (
        list(got) == [("c", 1)]
        and got[("c", 1)][:2] == pytest.approx((610.0, 610.0))
        and got[("c", 1)][2] == 0.8
    )


def test_find_clip_file_resolves_date_stamped_aliases(tmp_path):
    (tmp_path / "arsenal_mancity_20250925_seg010.jsonl").write_text("")
    assert (
        S.find_clip_file(tmp_path, "arsenal_mancity_seg010").name
        == "arsenal_mancity_20250925_seg010.jsonl"
    )
    assert S.find_clip_file(tmp_path, "arsenal_mancity_seg011") is None


def test_cli_score_end_to_end(tmp_path, monkeypatch, capsys):
    """build a panel file by hand, label it through sidecars, score a jsonl: the report round-trips through the CLI."""
    p, human, preds = synth("hm", [([1] * 5, [1, 1, 1, 1, 0])] * 3, pi=0.25)
    doc = {**p, "schema": P.SCHEMA, "N_windows": 12}
    pf = P.panel_path("hm", root=tmp_path)
    pf.parent.mkdir(parents=True)
    pf.write_text(json.dumps(doc))
    marks = P.marks_dir(root=tmp_path)
    marks.mkdir()
    rows = [
        {
            "frame_index": f,
            "bbox": {"x": 0.49, "y": 0.49, "w": 0.02, "h": 0.02},
            "center": {"x": 0.5, "y": 0.5},
            "tags": ["in_play_ball", "labeller"],
        }
        for w in p["windows"]
        for f in w["frames"]
    ]
    _write(marks / "c0.jsonl", rows)
    pred_rows = [
        {"clip": c, "frame": f, "x": x, "y": y, "score": s}
        for (c, f), (x, y, s) in preds.items()
    ]
    pj = _write(tmp_path / "pred.jsonl", pred_rows)
    out = tmp_path / "report.json"
    assert (
        cli.main(
            [
                "--root",
                str(tmp_path),
                "score",
                "--pred",
                str(pj),
                "--model",
                "m1",
                "--draws",
                "50",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    rep = json.loads(out.read_text())
    assert rep["model"] == "m1" and rep["matches"]["hm"]["metrics"]["r10"]["recall"][
        "est"
    ] == pytest.approx(0.8)
    assert json.loads(capsys.readouterr().out)["schema"] == S.SCHEMA
