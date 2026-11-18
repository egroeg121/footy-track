"""Fixtures for the ball-mining tests: synthetic pools, a dict-driven split authority, a one-dataset registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footy_track.ball_mining import miner
from footy_track.ball_mining.schema import validate_pool
from footy_track.labeller import datasets


def make_pool(rows: list[dict]) -> pd.DataFrame:
    """A valid pool from partial row dicts (defaults: one rtdetr source, p_real 0.6, 25 fps, mid-frame position)."""
    out = []
    for i, r in enumerate(rows):
        p = float(r.get("p_real", 0.6))
        out.append({
            "match": r.get("match", "m1"), "clip": r.get("clip", "m1_seg001"), "frame": int(r.get("frame", i * 100)),
            "fps": np.float32(r.get("fps", 25.0)), "x": r.get("x", 0.5), "y": r.get("y", 0.5), "w": r.get("w", 0.01), "h": r.get("h", 0.02),
            "track_id": r.get("track_id"), "sources": r.get("sources", ["rtdetr"]),
            "source_scores": r.get("source_scores", [p] * len(r.get("sources", ["rtdetr"]))),
            "n_sources": len(r.get("sources", ["rtdetr"])), "solver_post": r.get("solver_post", np.nan),
            "sam2_agree": r.get("sam2_agree"), "plausible": r.get("plausible"), "static_flag": r.get("static_flag"),
            "gap_frame": r.get("gap_frame"), "embedding": r.get("embedding"), "p_real": p,
            "scorer_version": r.get("scorer_version", "raw_conf"), "detector_sha": r.get("detector_sha", "sha0"),
        })  # fmt: skip
    return validate_pool(pd.DataFrame(out))


@pytest.fixture
def pool_factory():
    return make_pool


@pytest.fixture
def splitter():
    """``split_of(match, clip)``: matches starting ``held`` are heldout; clips in ``eval_clips`` are in-match eval."""

    class S:
        eval_clips: set[str] = set()

        def __call__(self, match: str, clip: str) -> str:
            if match.startswith("held"):
                return miner.EVAL_HELDOUT
            if match.startswith("nope"):
                return miner.UNREGISTERED
            return miner.EVAL_IN_MATCH if clip in self.eval_clips else miner.TRAIN

    return S()


def synthetic_big_pool(
    matches=("m1", "m2", "m3"), clips_per_match=6, frames_per_clip=60, seed=0, gap=True
) -> pd.DataFrame:
    """Many nodes over several matches with every bucket populated (gap nodes, static nodes, high and low p)."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in matches:
        for c in range(clips_per_match):
            for f in range(frames_per_clip):
                for k in range(3):
                    p = float(rng.uniform(0.05, 1.0))
                    row = {"match": m, "clip": f"{m}_seg{c:03d}", "frame": f * 50 + k, "p_real": p,
                           "x": float(rng.uniform(0.05, 0.95)), "y": float(rng.uniform(0.05, 0.95))}  # fmt: skip
                    if gap and k == 2 and f % 3 == 0:
                        row.update(
                            gap_frame=True, plausible=True, sources=["tracker_gap"]
                        )
                    if k == 1 and f % 7 == 0:
                        row.update(static_flag=True, p_real=0.9, source_scores=[0.9])
                    rows.append(row)
    return make_pool(rows)


@pytest.fixture
def big_pool():
    return synthetic_big_pool()


@pytest.fixture
def ds(tmp_path, monkeypatch):
    """A one-dataset registry ``train`` with tmp clips / marks dirs. Returns the marks dir."""
    root = tmp_path / "train"
    (root / "clips").mkdir(parents=True)
    (root / "marks").mkdir()
    reg = tmp_path / "registry.json"
    datasets.write_registry(reg, "train", {"train": {"label": "Train", "clips_dir": str(root / "clips"), "marks_dir": str(root / "marks"),
                                                       "candidates_dir": str(root / "cands")}})  # fmt: skip
    monkeypatch.setenv("FOOTY_LABELLER_DATASETS", str(reg))
    datasets.invalidate()
    yield root / "marks"
    datasets.invalidate()
