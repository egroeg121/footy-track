"""The pool-table schema: validator, Parquet round trip, RT-DETR adapter."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from footy_track.ball_mining import schema
from footy_track.ball_mining.schema import (
    PoolSchemaError,
    pool_from_machine_labels,
    read_pool,
    validate_pool,
    write_pool,
)

from .conftest import make_pool


def test_a_valid_pool_passes_and_gains_every_optional_column():
    df = make_pool(
        [
            {"frame": 1},
            {
                "frame": 2,
                "gap_frame": True,
                "plausible": False,
                "sources": ["tracker_gap", "sam2"],
                "source_scores": [0.4, 0.7],
            },
        ]
    )
    out = validate_pool(df.drop(columns=["embedding", "track_id"]))
    assert set(schema.COLUMNS) <= set(out.columns)


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda d: d.drop(columns=["p_real"]), "missing required column"),
        (lambda d: d.assign(x=1.5), "normalised"),
        (lambda d: d.assign(p_real=-0.1), "p_real"),
        (lambda d: d.assign(n_sources=2), "n_sources"),
        (
            lambda d: d.assign(source_scores=[[0.5, 0.5]] * len(d)),
            "one score per source",
        ),
        (
            lambda d: d.assign(
                sources=[[]] * len(d), source_scores=[[]] * len(d), n_sources=0
            ),
            "at least one source",
        ),
        (lambda d: d.assign(match=""), "match"),
        (lambda d: d.assign(fps=0.0), "fps"),
        (lambda d: d.assign(w=0.0), "w:"),
        (lambda d: d.assign(gap_frame="yes"), "gap_frame"),
        (lambda d: d.assign(source_scores=[[1.5]] * len(d)), "0..1"),
        (lambda d: d.assign(bogus=1), "unknown column"),
    ],
)
def test_validator_names_the_violation(mutate, needle):
    with pytest.raises(PoolSchemaError, match=needle):
        validate_pool(mutate(make_pool([{"frame": 1}, {"frame": 2}])))


def test_inconsistent_embedding_dimensions_are_rejected():
    df = make_pool(
        [{"frame": 1, "embedding": [0.1] * 72}, {"frame": 2, "embedding": [0.1] * 72}]
    )
    df.at[1, "embedding"] = [0.1] * 8
    with pytest.raises(PoolSchemaError, match="embedding"):
        validate_pool(df)


def test_parquet_round_trip_keeps_lists_nulls_and_types(tmp_path):
    df = make_pool([
        {"frame": 1, "embedding": [0.25] * 72, "track_id": 7, "gap_frame": True, "solver_post": 0.8},
        {"frame": 2, "sources": ["rtdetr", "sam2"], "source_scores": [0.5, 0.9], "sam2_agree": False},
    ])  # fmt: skip
    path = write_pool(df, tmp_path / "ball_pool" / "m1" / "pool.raw_conf.parquet")
    back = read_pool(path)
    assert list(back["sources"][1]) == ["rtdetr", "sam2"] and list(
        back["source_scores"][1]
    ) == [0.5, 0.9]
    assert len(back["embedding"][0]) == 72 and back["embedding"][1] is None
    assert back["gap_frame"][0] is True or back["gap_frame"][0] == True  # noqa: E712
    assert pd.isna(back["gap_frame"][1]) and back["sam2_agree"][1] == False  # noqa: E712
    assert back["track_id"][0] == 7 and pd.isna(back["track_id"][1])
    assert back["frame"].dtype == np.int64 and back["x"].dtype == np.float32
    assert (
        read_pool(tmp_path / "ball_pool").shape == back.shape
    )  # a directory reads every parquet below it


def _jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _row(frame, conf, tag="in_play_ball", x=0.4, y=0.3):
    return {"frame_index": frame, "bbox": {"x": x - 0.005, "y": y - 0.01, "w": 0.01, "h": 0.02}, "center": {"x": x, "y": y}, "tags": [tag, "rtdetr"],
            "confidence": conf, "model_id": "rtdetr-l_1920_v11", "run_id": "run-1", "reviewed": False}  # fmt: skip


def test_adapter_turns_rtdetr_ball_rows_into_a_single_source_pool(tmp_path):
    d = tmp_path / "ml"
    d.mkdir()
    _jsonl(
        d / "arsenal_mancity_20250925_seg000.jsonl",
        [
            _row(0, 0.91),
            _row(0, 0.12, x=0.7),
            _row(5, 0.5, "ball"),
            {**_row(5, 0.99), "tags": ["player", "rtdetr"]},
        ],
    )
    _jsonl(d / "arsenal_mancity_20250925_seg001.jsonl", [_row(10, 0.33)])
    pool = pool_from_machine_labels(
        [d], match="arsenal_mancity", fps=25.0, derive_static=False
    )
    assert len(pool) == 4  # the player row is not a ball candidate
    assert set(pool["clip"]) == {
        "arsenal_mancity_20250925_seg000",
        "arsenal_mancity_20250925_seg001",
    }
    assert (pool["n_sources"] == 1).all() and all(
        s == ["rtdetr"] for s in pool["sources"]
    )
    assert pool["p_real"].tolist() == pytest.approx(
        [0.91, 0.12, 0.5, 0.33], abs=1e-6
    )  # raw confidence until a scorer exists
    assert (pool["scorer_version"] == "raw_conf").all() and (
        pool["detector_sha"] == "run-1"
    ).all()
    assert pool["x"].iloc[0] == pytest.approx(0.4) and pool["y"].iloc[
        0
    ] == pytest.approx(0.3)  # centre, normalised
    assert pool["gap_frame"].isna().all() and pool["plausible"].isna().all()
    schema.validate_pool(pool)


def test_adapter_derives_static_flags_for_parked_balls(tmp_path):
    rows = [_row(f, 0.9, x=0.2, y=0.8) for f in range(30)] + [
        _row(f, 0.9, x=0.3 + f * 0.01, y=0.5) for f in range(30)
    ]  # a parked ball, a moving one
    _jsonl(tmp_path / "m1_seg001.jsonl", rows)
    pool = pool_from_machine_labels(
        [tmp_path / "m1_seg001.jsonl"],
        match="m1",
        derive_static=True,
        static_min_frames=25,
    )
    flags = pool["static_flag"].astype(bool)
    assert flags[pool["x"] < 0.21].all() and not flags[pool["x"] > 0.25].any()


def test_adapter_on_nothing_gives_an_empty_valid_pool(tmp_path):
    _jsonl(tmp_path / "m1_seg001.jsonl", [{**_row(0, 0.9), "tags": ["player"]}])
    assert pool_from_machine_labels([tmp_path / "m1_seg001.jsonl"], match="m1").empty
