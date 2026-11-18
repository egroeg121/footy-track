"""The scorer: production-only features, heldout matches never train, leave-one-match-out, versioned artifact, apply."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from footy_track.ball_mining import pool as P  # noqa: N812
from footy_track.ball_mining import scorer as S  # noqa: N812


def synth_table(seed=0, n_per=400):
    """Three matches (A, B train; H heldout). rt_score separates the classes with noise; tracker membership helps."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in ("A", "B", "H"):
        for i in range(n_per):
            y = int(rng.random() < 0.3)
            r = dict.fromkeys(P.FEATURE_COLS, np.nan)
            r.update(
                rt_score=float(np.clip(rng.normal(0.7 if y else 0.35, 0.15), 0, 1)), rt_rank=float(rng.integers(0, 4)), rt_n=1.0,
                xn=float(rng.random()), yn=float(rng.random()), trk_member=float(y and rng.random() < 0.7), trk_agree=0.0,
                trk_support=float(y * rng.random() * 3), trk_len=float(y * rng.integers(0, 9)), trk_state=2.0 if y else 0.0, n_src=1.0,
                match=m, clip=f"{m}_seg{i % 20:03d}", frame=5 * i, label=y, contam=False, clip_split="train", x=0.0, y=0.0,
            )  # fmt: skip
            rows.append(r)
    return pd.DataFrame(rows)


def split_of(m):
    return {"A": "train", "B": "train", "H": "eval_heldout_match"}.get(
        m, "unregistered"
    )


def test_feature_list_is_production_only():
    S.check_features(S.FEATURES)
    for bad in (
        ["yolo3_score"],
        ["sam2_n"],
        ["motion_cov"],
        ["b1_p"],
        ["n_src_cov"],
        ["sam3_text_score"],
    ):
        with pytest.raises(ValueError):
            S.check_features(bad)
    assert set(S.FEATURES) <= set(
        P.FEATURE_COLS
    )  # every scorer feature is something the pool stage computes


def test_train_uses_train_matches_only_and_lomo_treats_heldout_as_test_fold():
    t = synth_table()
    sc, man = S.train(t, split_of=split_of)
    assert (
        man["training"]["train_matches"] == ["A", "B"]
        and man["training"]["heldout_trained"] == []
    )
    assert man["mode"] == "shadow" and "heldout" in man["leakage_policy"]
    folds = man["lomo"]["folds"]
    assert folds["A"]["trained_on"] == ["B"] and folds["B"]["trained_on"] == [
        "A"
    ]  # a train match never sees itself
    assert folds["H"]["trained_on"] == [
        "A",
        "B",
    ]  # the heldout match is only ever tested on
    assert set(man["lomo"]["per_match"]) == {"A", "B", "H"}
    assert man["lomo"]["pooled"]["E2"]["model"]["auroc"] > 0.9
    assert (
        man["lomo_reference_heldout_allowed_to_train"] is not None
    )  # recorded for comparison, never shipped
    # the final model was fit without H: its rows are not in the training digest
    assert man["training"]["n_nodes"] == int((t["match"].isin(["A", "B"])).sum())


def test_unregistered_and_clip_eval_rows_never_train():
    t = synth_table()
    t.loc[t["match"] == "B", "clip_split"] = (
        "eval_in_match"  # every clip of B is in-match eval
    )
    sc, man = S.train(t, split_of=split_of)
    assert man["training"]["n_nodes"] == int((t["match"] == "A").sum())
    with pytest.raises(ValueError, match="no train-register match"):
        S.train(t, split_of=lambda m: "unregistered")


def test_contaminated_and_unlabelled_rows_are_excluded():
    t = synth_table()
    t.loc[t.index[:100], "contam"] = True
    t.loc[t.index[100:150], "label"] = -1
    _, man = S.train(t, split_of=split_of)
    assert man["training"]["excluded_contaminated_nodes"] == 100
    assert man["training"]["n_nodes"] == int(
        ((t["match"] != "H") & ~t["contam"] & (t["label"] >= 0)).sum()
    )


def test_forced_heldout_is_recorded_as_an_override():
    _, man = S.train(synth_table(), split_of=split_of, extra_train_matches={"H"})
    assert (
        man["training"]["heldout_trained"] == ["H"]
        and "H" in man["training"]["train_matches"]
    )


def test_version_is_deterministic_and_changes_with_data(tmp_path):
    t = synth_table()
    a, _ = S.train(t, split_of=split_of)
    b, _ = S.train(t, split_of=split_of)
    c, _ = S.train(synth_table(seed=1), split_of=split_of)
    assert a.version == b.version != c.version and a.version.startswith("hgb-rtm-")


def test_save_load_latest_roundtrip_and_immutable(tmp_path):
    t = synth_table()
    sc, _ = S.train(t, split_of=split_of)
    assert S.latest_version(tmp_path) is None
    with pytest.raises(FileNotFoundError):
        S.load("latest", tmp_path)
    d = S.save(sc, tmp_path)
    assert {p.name for p in d.iterdir()} == {
        "model.joblib",
        "features.json",
        "manifest.json",
        "lomo.json",
    }
    assert S.latest_version(tmp_path) == sc.version
    loaded = S.load("latest", tmp_path)
    assert loaded.version == sc.version and loaded.features == sc.features
    assert (
        json.loads((d / "manifest.json").read_text())["lomo"]["pooled"]["E2"]["model"][
            "auroc"
        ]
        is not None
    )
    x = t.head(50)
    assert np.allclose(
        sc.hgb.predict_proba(x[list(sc.features)])[:, 1],
        loaded.hgb.predict_proba(x[list(sc.features)])[:, 1],
    )
    with pytest.raises(FileExistsError):
        S.save(sc, tmp_path)
    assert (
        S.load(sc.version, tmp_path).version == sc.version
    )  # a pinned version loads too


def test_apply_scores_detections_and_uses_gap_plausibility_for_gap_nodes():
    t = synth_table()
    sc, _ = S.train(t, split_of=split_of)
    nodes = t.head(6).copy()
    nodes["gap_frame"] = [False] * 5 + [True]
    nodes["source_scores"] = [[0.5]] * 5 + [[0.42]]
    out = sc.apply(nodes)
    assert (out["scorer_version"] == sc.version).all() and out["p_real"].between(
        0, 1
    ).all()
    assert out["p_real"].iloc[5] == pytest.approx(0.42) and np.isnan(
        out["solver_post"].iloc[5]
    )  # gap: plausibility, no solver posterior
    assert out["solver_post"].iloc[:5].between(0, 1).all()
    hi = (
        nodes.iloc[:1]
        .copy()
        .assign(
            rt_score=0.95, trk_member=1.0, trk_state=2.0, trk_support=2.0, trk_len=6.0
        )
    )
    lo = (
        nodes.iloc[:1]
        .copy()
        .assign(
            rt_score=0.15, trk_member=0.0, trk_state=0.0, trk_support=0.0, trk_len=0.0
        )
    )
    assert sc.apply(hi)["p_real"].iloc[0] > sc.apply(lo)["p_real"].iloc[0]
    with pytest.raises(ValueError, match="lack scorer features"):
        sc.apply(nodes.drop(columns=["rt_score"]))


def test_metrics_and_ece():
    y = np.array([0, 0, 1, 1, 1])
    m = S.metrics(y, np.array([0.1, 0.4, 0.35, 0.8, 0.9]))
    assert (
        m["n"] == 5
        and m["pos"] == 3
        and 0.5 < m["auroc"] <= 1
        and m["p_at_r50"] is not None
    )
    assert S.metrics(np.zeros(3, int), np.zeros(3))["auroc"] is None
    assert S.ece(np.array([1, 1, 0, 0]), np.array([1.0, 1.0, 0.0, 0.0])) == 0.0
