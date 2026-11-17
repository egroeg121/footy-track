"""/progress page + /api/progress: synthetic manifests and a tmp DuckDB (no real data, no server import).

The router is mounted on a bare FastAPI app (the real labeller app pulls in torch); a source-level
test checks that ``server.py`` mounts it and the hub links to it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from footy_track import match_splits, splits
from footy_track.datasets import results as res
from footy_track.datasets import snapshot as snap
from footy_track.feature_store import ingest_all as ia
from footy_track.feature_store.lock import store_lock
from footy_track.feature_store.store import FeatureStore
from footy_track.labeller import progress, progress_routes

LABELLER = Path(progress.__file__).parent


def manifest(dataset, version, by_split, *, kinds=None, parent=None, diff=None, created="2026-11-17T10:00:00Z", sr=None):
    kinds = kinds or {"train": {"ball": by_split.get("train", 0)}}
    return {
        "dataset": dataset, "version": version, "created_at": created, "content_hash": f"{version:02d}" * 32,
        "item_schema": 1, "parent_version": parent, "note": f"note v{version}",
        "counts": {"n_items": sum(by_split.values()), "by_split": by_split, "by_split_kind": kinds,
                   "clips_by_split": {}},
        "n_items_unregistered_match": 0,
        "split_registers": sr or {"preview": False},
        "git_sha": "abcdef1234567890", "git_dirty": False,
        "diff_vs_parent": diff or {"added": 0, "removed": 0, "changed": 0, "parent_version": parent},
    }


def write_manifest(root: Path, m: dict) -> None:
    d = root / "datasets" / m["dataset"] / f"v{m['version']}"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(m))  # deliberately no items.parquet


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FOOTY_DATA", str(root))
    monkeypatch.setenv("FOOTY_DATASETS_ROOT", str(root))
    monkeypatch.setenv("FOOTY_FEATURE_STORE", str(root / "feature_store" / "fs.duckdb"))
    monkeypatch.setenv("FOOTY_MATCH_SPLITS", str(root / "match_splits.json"))
    monkeypatch.setenv("FOOTY_CLIP_SPLITS", str(root / "clip_splits.json"))
    monkeypatch.delenv("FOOTY_MATCH_SPLITS_ALLOW_SEED", raising=False)
    match_splits.clear_cache()
    splits.clear_cache()
    progress.clear_cache()
    yield root
    progress.clear_cache()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(progress_routes.router)
    return TestClient(app)


def registers(root: Path) -> None:
    (root / "match_splits.json").write_text(json.dumps({
        "h1": {"split": "heldout"}, "h2": {"split": "heldout"}, "t1": {"split": "train"},
    }))
    (root / "clip_splits.json").write_text(json.dumps({
        "version": 1, "rule": {"modulus": 5, "eval_remainder": 0},
        "clips": {"t1_seg001": {"split": "eval"}, "t1_seg002": {"split": "train"}, "t1_seg003": {"split": "train"}},
    }))


def seed_results(db: Path, rows: list[res.ResultRow]) -> None:
    res.record_results(db, rows)


def rr(**kw) -> res.ResultRow:
    base = dict(task="broadcast", dataset="broadcast", dataset_version=1, split="eval_in_match",
                model_name="clf", model_version="rf10", metric="accuracy", value=0.9, n_items=40,
                contaminated=False)
    return res.ResultRow(**(base | kw))


# ---- empty states: never a 500 ----------------------------------------------------------------
def test_everything_missing_is_not_a_500(env, client):
    r = client.get("/api/progress")
    assert r.status_code == 200
    d = r.json()
    assert d["store"]["state"] == "missing" and d["store"]["exists"] is False
    assert "ingest_all" in d["store"]["hint"] and "ingest_all" in d["commands"]["ingest"]
    assert d["registers"]["matches"]["state"] == "missing"
    assert d["registers"]["clips"]["state"] == "missing" and "splits assign" in d["registers"]["clips"]["hint"]
    for ds in ("tracking", "broadcast", "calibration"):
        x = d["datasets"][ds]
        assert x["growth"]["state"] == "empty" and "datasets bump all" in x["growth"]["hint"]
        assert x["models"] == [] and x["health"]["version"] is None and x["health"]["warnings"] == []
    page = client.get("/progress")
    assert page.status_code == 200 and "Progress" in page.text and "no-store" in page.headers["cache-control"]


def test_versions_but_no_store_and_store_without_results(env, client):
    registers(env)
    write_manifest(env, manifest("tracking", 1, {"train": 10, "eval_in_match": 3, "eval_heldout_match": 2}))
    d = client.get("/api/progress?refresh=true").json()
    assert d["datasets"]["tracking"]["growth"]["state"] == "ok"
    assert d["datasets"]["tracking"]["results_state"] == "missing"
    assert d["datasets"]["broadcast"]["growth"]["state"] == "empty"

    db = Path(os.environ["FOOTY_FEATURE_STORE"])
    FeatureStore.open(db).close()  # a real store with an empty eval_result table
    d = client.get("/api/progress?refresh=true").json()
    assert d["store"]["state"] == "empty" and d["store"]["exists"] is True
    assert d["datasets"]["tracking"]["results_state"] == "empty"


def test_garbage_files_do_not_500(env, client):
    (env / "match_splits.json").write_text("{not json")
    (env / "clip_splits.json").write_text("[]")
    write_manifest(env, manifest("tracking", 1, {"train": 1}))
    bad = env / "datasets" / "tracking" / "v2"
    bad.mkdir()
    (bad / "manifest.json").write_text("garbage")
    db = Path(os.environ["FOOTY_FEATURE_STORE"])
    db.parent.mkdir(parents=True)
    db.write_bytes(b"this is not a duckdb file")
    r = client.get("/api/progress?refresh=true")
    assert r.status_code == 200
    d = r.json()
    assert d["store"]["state"] in ("error", "busy")
    assert d["registers"]["matches"]["state"] == "error" and d["registers"]["clips"]["state"] == "error"
    g = d["datasets"]["tracking"]["growth"]
    assert [v["version"] for v in g["versions"]] == [1] and g["errors"] and g["errors"][0].startswith("v2")


# ---- growth ---------------------------------------------------------------------------------------
def test_growth_shape_and_diff(env, client):
    registers(env)
    write_manifest(env, manifest("tracking", 1, {"train": 10, "eval_in_match": 3, "eval_heldout_match": 2},
                                 kinds={"train": {"ball": 8, "no_ball": 2}, "eval_in_match": {"ball": 3},
                                        "eval_heldout_match": {"ball": 2}}))
    write_manifest(env, manifest("tracking", 2, {"train": 12, "eval_in_match": 4, "eval_heldout_match": 2}, parent=1,
                                 created="2026-11-18T10:00:00Z",
                                 diff={"added": 3, "removed": 1, "changed": 2, "added_by_split": {"train": 3}}))
    d = client.get("/api/progress").json()
    g = d["datasets"]["tracking"]["growth"]
    assert g["latest"] == 2 and [v["version"] for v in g["versions"]] == [1, 2]
    v1, v2 = g["versions"]
    assert v1["n_items"] == 15 and v1["by_split"] == {"train": 10, "eval_in_match": 3, "eval_heldout_match": 2}
    assert v1["by_kind"] == {"ball": 13, "no_ball": 2}
    assert v1["created_at"] == "2026-11-17T10:00:00Z" and v1["parent_version"] is None
    assert v2["parent_version"] == 1
    assert (v2["diff"]["added"], v2["diff"]["removed"], v2["diff"]["changed"]) == (3, 1, 2)
    assert v2["diff"]["added_by_split"] == {"train": 3}
    # split health from the registers
    reg = d["registers"]
    assert reg["matches"]["heldout"] == 2 and reg["matches"]["train"] == 1
    assert reg["matches"]["heldout_names"] == ["h1", "h2"]
    assert reg["clips"]["eval"] == 1 and reg["clips"]["train"] == 2
    assert d["datasets"]["tracking"]["health"]["warnings"] == []  # has held-out items


# ---- zero held-out warning ------------------------------------------------------------------------
def test_zero_heldout_warns_with_call_to_action(env, client):
    registers(env)
    write_manifest(env, manifest("broadcast", 1, {"train": 129, "eval_in_match": 72, "eval_heldout_match": 0}))
    h = client.get("/api/progress").json()["datasets"]["broadcast"]["health"]
    assert h["eval_heldout_match"] == 0 and h["eval_in_match"] == 72
    w = {x["code"]: x for x in h["warnings"]}
    assert w["no_heldout"]["level"] == "action"
    assert "held-out match" in w["no_heldout"]["message"] and "/broadcast_review" in w["no_heldout"]["message"]


def test_register_change_since_snapshot_warns(env, client):
    registers(env)
    sr = {"preview": False, "match_splits_sha256": "0" * 64, "clip_splits_sha256": snap._file_sha(env / "clip_splits.json")}
    write_manifest(env, manifest("tracking", 1, {"train": 1, "eval_in_match": 1, "eval_heldout_match": 1}, sr=sr))
    codes = [w["code"] for w in client.get("/api/progress").json()["datasets"]["tracking"]["health"]["warnings"]]
    assert codes == ["registers_changed"]


# ---- results: contamination, cohort vs full -----------------------------------------------------------
def test_results_surface_contamination_and_cohort(env, client):
    registers(env)
    write_manifest(env, manifest("broadcast", 1, {"train": 5, "eval_in_match": 3, "eval_heldout_match": 2}))
    write_manifest(env, manifest("broadcast", 2, {"train": 6, "eval_in_match": 4, "eval_heldout_match": 2}, parent=1))
    db = Path(os.environ["FOOTY_FEATURE_STORE"])
    seed_results(db, [
        rr(dataset_version=1, value=0.80, contaminated=False),                       # clean, full
        rr(dataset_version=2, value=0.85, contaminated=False),
        rr(dataset_version=2, value=0.82, cohort_version=1, contaminated=False),     # clean, cohort v1
        rr(dataset_version=1, split="eval_heldout_match", value=0.99, contaminated=True,
           contamination_note="13% of training images are Nottingham Forest"),
        rr(dataset_version=1, model_name="other", model_version="w2", split="eval_all", value=0.7, contaminated=None,
           contamination_note="training data not declared"),
        rr(dataset_version=1, split="train", value=1.0, contaminated=None),          # train: not counted as unknown
    ])
    d = client.get("/api/progress").json()
    assert d["store"]["state"] == "ok" and d["store"]["n_results"] == 6
    x = d["datasets"]["broadcast"]
    models = {m["model_name"]: m for m in x["models"]}
    clf, other = models["clf"], models["other"]
    assert clf["model_version"] == "rf10" and clf["task"] == "broadcast"

    series = {(s["split"], s["cohort_version"]): s for s in clf["metrics"]["accuracy"]}
    full = series[("eval_in_match", None)]
    cohort = series[("eval_in_match", 1)]
    assert full["kind"] == "full" and cohort["kind"] == "cohort"
    assert "full current" in full["tier"] and "cohort v1" in cohort["tier"]
    assert [(p["dataset_version"], p["value"]) for p in full["points"]] == [(1, 0.80), (2, 0.85)]
    assert [(p["dataset_version"], p["value"]) for p in cohort["points"]] == [(2, 0.82)]
    assert all(p["contaminated"] is False for p in full["points"])

    bad = series[("eval_heldout_match", None)]["points"][0]
    assert bad["contaminated"] is True and "Nottingham" in bad["note"]
    assert clf["n_contaminated"] == 1 and clf["n_unknown"] == 0  # train NULL not counted

    unk = other["metrics"]["accuracy"][0]["points"][0]
    assert unk["contaminated"] is None  # surfaced as NULL, never coerced to clean
    assert other["n_unknown"] == 1 and other["n_contaminated"] == 0
    assert x["results_state"] == "ok"
    assert d["datasets"]["tracking"]["models"] == []


def test_latest_run_wins(env, client):
    registers(env)
    db = Path(os.environ["FOOTY_FEATURE_STORE"])
    from datetime import UTC, datetime
    res.record_results(db, [rr(value=0.1)], now=datetime(2026, 1, 1, tzinfo=UTC))
    res.record_results(db, [rr(value=0.2)], now=datetime(2026, 1, 2, tzinfo=UTC))
    d = client.get("/api/progress").json()
    pts = d["datasets"]["broadcast"]["models"][0]["metrics"]["accuracy"][0]["points"]
    assert [p["value"] for p in pts] == [0.2]


# ---- store handling -------------------------------------------------------------------------------
def test_store_opened_read_only_and_not_held(env, client):
    registers(env)
    db = Path(os.environ["FOOTY_FEATURE_STORE"])
    seed_results(db, [rr()])
    assert client.get("/api/progress").status_code == 200
    # a writer can take the file right after the request (no connection kept between requests)
    with store_lock(db, exclusive=True):
        store = FeatureStore.open(db)
        store.close()
    # ...and while a writer holds the lock the page degrades instead of failing
    progress.clear_cache()
    with store_lock(db, exclusive=True):
        d = client.get("/api/progress").json()
    assert d["store"]["state"] == "busy" and d["datasets"]["broadcast"]["models"] == []
    # the read path never writes: the connection used is read-only
    with pytest.raises(duckdb.Error):
        conn = duckdb.connect(str(db), read_only=True)
        try:
            conn.execute("DELETE FROM eval_result")
        finally:
            conn.close()


def test_cache_is_keyed_on_mtimes(env, client, monkeypatch):
    registers(env)
    write_manifest(env, manifest("tracking", 1, {"train": 1}))
    built = []
    orig = progress._build
    monkeypatch.setattr(progress, "_build", lambda *a, **k: built.append(1) or orig(*a, **k))
    client.get("/api/progress")
    client.get("/api/progress")
    assert len(built) == 1  # second call served from the cache
    write_manifest(env, manifest("tracking", 2, {"train": 2}, parent=1))  # a new manifest invalidates
    d = client.get("/api/progress").json()
    assert len(built) == 2 and d["datasets"]["tracking"]["growth"]["latest"] == 2
    client.get("/api/progress?refresh=true")
    assert len(built) == 3


def test_dataset_query_param_is_ignored(env, client):
    r = client.get("/api/progress?dataset=whatever")
    assert r.status_code == 200 and "datasets" in r.json()
    assert client.get("/progress?dataset=whatever").status_code == 200


# ---- real manifests from ingest + bump (shape compatibility with the writer) --------------------------
def test_reads_real_manifests(world, client):
    progress.clear_cache()
    ia.run(world.db, data_root=world.root, marks_dirs=[world.marks])
    for ds in ("tracking", "broadcast", "calibration"):
        snap.bump(ds, world.db, world.ds_root)
    res.record_results(world.db, [rr(dataset_version=1, contaminated=None)])
    os.environ["FOOTY_FEATURE_STORE"] = str(world.db)
    try:
        d = client.get("/api/progress?refresh=true").json()
    finally:
        os.environ.pop("FOOTY_FEATURE_STORE", None)
    t = d["datasets"]["tracking"]["growth"]["versions"][0]
    m = json.loads((world.ds_root / "datasets/tracking/v1/manifest.json").read_text())
    assert t["n_items"] == m["counts"]["n_items"] > 0
    assert t["by_split"]["eval_heldout_match"] == m["counts"]["by_split"].get("eval_heldout_match", 0)
    assert d["registers"]["matches"]["heldout"] == 1 and d["registers"]["clips"]["eval"] == 2
    assert d["datasets"]["broadcast"]["models"][0]["n_unknown"] == 1
    assert d["datasets"]["tracking"]["health"]["warnings"] == []


# ---- wiring ---------------------------------------------------------------------------------------------
def test_wired_into_server_and_hub():
    assert "progress_routes.router" in (LABELLER / "server.py").read_text()
    assert 'href="/progress"' in (LABELLER / "web" / "main.html").read_text()
    page = (LABELLER / "web" / "progress.html").read_text()
    assert "/api/progress" in page and "contaminated" in page and "<svg" in page.lower() or "svg" in page
