"""Fixtures for the ball-mining tests: synthetic pools, a dict-driven split authority, a one-dataset registry."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from footy_track.ball_mining import miner
from footy_track.ball_mining.schema import validate_pool
from footy_track.labeller import datasets, job_edit, jobs, jobs_client


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


class LocalApi:
    """The labeller API calls the refit / close code makes (``list_jobs``, ``results``, ``reorder``), served in-process from the
    real job and label stores, with the same JSON document shapes and 409 / 422 bodies as ``jobs_routes``. ``before_reorder`` is
    called with the ``(ref, order)`` of every reorder, to simulate a concurrent change."""

    JobsError = jobs_client.JobsError

    def __init__(self):
        self.calls: list[tuple] = []
        self.before_reorder = None

    def list_jobs(self, **f):
        self.calls.append(("list", f))
        return [
            jobs.public(j, None, items=False)
            for j in jobs.filter_jobs(kind=f.get("kind"), dataset=f.get("dataset"))
        ]

    def results(self, ref):
        self.calls.append(("results", ref))
        job, c = jobs.job_with_progress(jobs.resolve(ref))
        return json.loads(json.dumps(jobs.results_doc(job, c), default=str))

    def reorder(self, ref, order, *, by="api", note=None, if_rev=None, url=None):
        self.calls.append(("reorder", ref, list(order), by, note, if_rev))
        if self.before_reorder:
            self.before_reorder(ref, order)
        try:
            return job_edit.reorder_unanswered(
                jobs.resolve(ref)["id"], order, by=by, note=note, if_rev=if_rev
            )
        except jobs.JobError as e:
            raise jobs_client.JobsError(
                e.status, {"detail": e.detail, **e.extra}
            ) from None


@pytest.fixture
def api():
    return LocalApi()


@pytest.fixture(autouse=True)
def _no_label_hooks(monkeypatch):
    """Completing a round job in a test must not start the hook worker (which would try to queue a Prefect run)."""
    monkeypatch.setenv("FOOTY_LABELLING_HOOKS", "0")


@pytest.fixture(autouse=True)
def legacy_spacing(request, monkeypatch):
    """The synthetic pools are a few dense clips, which the per-clip cap and the +-``exclude_s`` spacing would starve. Tests of
    the other rules run with both off by default; tests of the spacing rules pass the values explicitly or use the
    ``spacing_rules`` marker to keep the real defaults."""
    if request.node.get_closest_marker("spacing_rules"):
        return
    import dataclasses

    from footy_track.ball_mining.buckets import BucketConfig

    names = [f.name for f in dataclasses.fields(BucketConfig)]
    defaults = list(BucketConfig.__init__.__defaults__)
    off = {"exclude_s": 0.0, "max_per_clip": 0}
    n_no_default = len(names) - len(defaults)
    for k, v in off.items():
        defaults[names.index(k) - n_no_default] = v
    monkeypatch.setattr(BucketConfig.__init__, "__defaults__", tuple(defaults))
