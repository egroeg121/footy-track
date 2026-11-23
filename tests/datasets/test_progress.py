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


def test_broadcast_edge_and_all_biased_metrics_are_flagged_diagnostic(env, client):
    registers(env)
    write_manifest(env, manifest("broadcast", 1, {"train": 5, "eval_in_match": 72}))
    db = Path(os.environ["FOOTY_FEATURE_STORE"])
    seed_results(db, [
        rr(metric="accuracy", value=0.87, n_items=72), rr(metric="f1_yes", value=0.9, n_items=72),
        rr(metric="n_random", value=23, n_items=72), rr(metric="accuracy_random", value=0.87, n_items=72),
        rr(metric="accuracy_edge", value=0.33, n_items=72), rr(metric="n_edge", value=49, n_items=72),
        rr(metric="accuracy_all_biased", value=0.50, n_items=72),
        rr(metric="f1_yes_all_biased", value=0.55, n_items=72),
    ])
    m = client.get("/api/progress").json()["datasets"]["broadcast"]["models"][0]
    meta = m["metric_meta"]
    for metric in ("accuracy_edge", "accuracy_all_biased", "f1_yes_all_biased"):
        assert meta[metric]["role"] == "diagnostic" and "Diagnostic only" in meta[metric]["note"]
    assert meta["accuracy"]["role"] == meta["f1_yes"]["role"] == "headline"
    assert meta["n_random"]["role"] == "count"
    # headline with its n; each metric's n is its own stratum's, not the all-frames n_items
    (h,) = m["headline"]
    assert h["accuracy"] == 0.87 and h["n"] == 23
    assert m["metrics"]["accuracy"][0]["points"][0]["n_items"] == 23
    assert m["metrics"]["accuracy_edge"][0]["points"][0]["n_items"] == 49
    # fewer than 50 random frames: warned
    assert "small eval sample" in h["warning"]


def test_broadcast_headline_no_warning_at_50_and_legacy_all_frames_flagged(env, client):
    registers(env)
    write_manifest(env, manifest("broadcast", 1, {"train": 5, "eval_in_match": 72}))
    db = Path(os.environ["FOOTY_FEATURE_STORE"])
    seed_results(db, [
        rr(metric="accuracy", value=0.8), rr(metric="n_random", value=50),
        rr(model_name="old", metric="accuracy", value=0.5, n_items=72),  # recorded before strata existed
    ])
    models = {m["model_name"]: m for m in client.get("/api/progress").json()["datasets"]["broadcast"]["models"]}
    assert models["clf"]["headline"][0]["warning"] is None and models["clf"]["headline"][0]["n"] == 50
    assert models["old"]["headline"][0]["n"] is None and "edge-weighted" in models["old"]["headline"][0]["warning"]


def test_progress_page_greys_diagnostic_metrics():
    html = (LABELLER / "web" / "progress.html").read_text()
    assert "tr.diag" in html and "(diagnostic)" in html and "small eval sample" in html and "headlineStrip" in html


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


# ---- charts: two small line charts per stage -------------------------------------------------------------
def test_performance_config_names_real_adapter_metrics():
    """The config dict is the one place to change a chart; it must cover the three stages and stay small."""
    cfg = progress.PERFORMANCE_SERIES
    assert set(cfg) == {"tracking", "broadcast", "calibration"}
    assert all(1 <= len(v) <= 3 for v in cfg.values())
    metrics = {(ds, s["metric"]) for ds, v in cfg.items() for s in v}
    assert ("broadcast", "accuracy") in metrics            # random stratum headline, not accuracy_edge / _all_biased
    assert ("calibration", "success_at_5px") in metrics
    assert ("tracking", "ball_recall_10px") in metrics and ("tracking", "player_recall") in metrics
    assert not any(m.endswith(("_edge", "_all_biased")) for _, m in metrics)
    assert [s for s, _ in progress.SIZE_SERIES] == ["train", "eval_in_match", "eval_heldout_match"]


def test_single_version_renders_points_not_a_stretched_axis(env, client):
    registers(env)
    write_manifest(env, manifest("broadcast", 1, {"train": 129, "eval_in_match": 72, "eval_heldout_match": 0}))
    seed_results(env / "feature_store" / "fs.duckdb", [
        rr(model_name="broadcast_classifier", model_version="sha1", value=0.79, n_items=23),
        rr(model_name="broadcast_classifier_at_review", model_version="unknown-stored", value=0.5, n_items=23),
    ])
    c = client.get("/api/progress").json()["datasets"]["broadcast"]["charts"]
    assert c["versions"] == [{"version": 1, "date": "2026-11-17"}]
    size = {s["split"]: s["points"] for s in c["size"]}
    assert size == {"train": [{"version": 1, "value": 129}], "eval_in_match": [{"version": 1, "value": 72}],
                    "eval_heldout_match": [{"version": 1, "value": 0}]}
    lines = c["performance"]
    assert [len(s["points"]) for s in lines] == [1, 1]
    assert [s["points"][0]["model_name"] for s in lines] == ["broadcast_classifier", "broadcast_classifier_at_review"]
    assert lines[0]["points"][0]["value"] == 0.79 and lines[0]["points"][0]["n_items"] == 23


def test_tracking_lines_split_detector_from_ball_tracker_and_skip_other_splits(env, client):
    registers(env)
    for v in (1, 2, 3):
        write_manifest(env, manifest("tracking", v, {"train": 10 * v, "eval_in_match": 5, "eval_heldout_match": 3},
                                     parent=v - 1 or None, created=f"2026-11-{16 + v}T10:00:00Z"))
    t = dict(task="tracking", dataset="tracking")
    seed_results(env / "feature_store" / "fs.duckdb", [
        rr(**t, dataset_version=1, split="eval_heldout_match", model_name="rtdetr", model_version="w1",
           metric="ball_f1_10px", value=0.60, contaminated=True),
        rr(**t, dataset_version=2, split="eval_heldout_match", model_name="rtdetr", model_version="w2",
           metric="ball_f1_10px", value=0.70, contaminated=False),
        rr(**t, dataset_version=2, split="eval_in_match", model_name="rtdetr", model_version="w2",
           metric="ball_f1_10px", value=0.99),                       # wrong split: not on the held-out line
        rr(**t, dataset_version=2, split="eval_heldout_match", model_name="rtdetr", model_version="w2",
           metric="ball_recall_10px", value=0.80),                    # detector recall: not the tracker line
        rr(**t, dataset_version=2, split="eval_heldout_match", model_name="ball_tracker_kalman",
           model_version="w1", metric="ball_recall_10px", value=0.50),
        rr(**t, dataset_version=3, split="eval_heldout_match", model_name="rtdetr", model_version="w2",
           metric="ball_f1_10px", value=None, n_items=0),            # nothing scored: no point
    ])
    c = client.get("/api/progress").json()["datasets"]["tracking"]["charts"]
    assert [v["version"] for v in c["versions"]] == [1, 2, 3]
    assert [v["date"] for v in c["versions"]] == ["2026-11-17", "2026-11-18", "2026-11-19"]
    f1, player, tracker = c["performance"]
    assert [(p["version"], p["value"], p["model_version"], p["contaminated"]) for p in f1["points"]] == [
        (1, 0.60, "w1", True), (2, 0.70, "w2", False)]
    assert player["points"] == []
    assert [(p["version"], p["value"], p["model_name"]) for p in tracker["points"]] == [(2, 0.50, "ball_tracker_kalman")]
    assert [p["value"] for p in c["size"][0]["points"]] == [10, 20, 30]


def test_calibration_picks_the_best_evidenced_run_per_version(env, client):
    registers(env)
    write_manifest(env, manifest("calibration", 1, {"train": 5, "eval_in_match": 20, "eval_heldout_match": 4}))
    write_manifest(env, manifest("calibration", 2, {"train": 6, "eval_in_match": 22, "eval_heldout_match": 4}, parent=1))
    k = dict(task="calibration", dataset="calibration", model_name="calibration_proposals",
             metric="success_at_5px", split="eval_in_match")
    seed_results(env / "feature_store" / "fs.duckdb", [
        rr(**k, dataset_version=1, model_version="batch1", value=0.64, n_items=14),
        rr(**k, dataset_version=1, model_version="tiny", value=0.0, n_items=2),
        rr(**k, dataset_version=2, model_version="batch1", value=0.70, n_items=14),
    ])
    (line,) = client.get("/api/progress").json()["datasets"]["calibration"]["charts"]["performance"]
    assert [(p["version"], p["value"], p["model_version"]) for p in line["points"]] == [
        (1, 0.64, "batch1"), (2, 0.70, "batch1")]
    # the default pick is the most recent run
    rows = [
        {"dataset": "x", "metric": "m", "split": "s", "cohort_version": None, "value": v, "model_name": "a",
         "model_version": mv, "run_at": ra, "n_items": n, "dataset_version": 1, "contaminated": None,
         "contamination_note": None}
        for v, mv, ra, n in [(0.1, "old", "2026-01-01", 99), (0.2, "new", "2026-01-02", 1)]
    ]
    progress.PERFORMANCE_SERIES["x"] = ({"label": "l", "metric": "m", "split": "s", "model": "*"},)
    try:
        assert progress.performance_series("x", rows)[0]["points"][0]["model_version"] == "new"
    finally:
        del progress.PERFORMANCE_SERIES["x"]


def test_no_results_still_gives_size_chart_and_empty_performance_lines(env, client):
    registers(env)
    write_manifest(env, manifest("tracking", 1, {"train": 10, "eval_in_match": 5, "eval_heldout_match": 3}))
    c = client.get("/api/progress").json()["datasets"]["tracking"]["charts"]
    assert [len(s["points"]) for s in c["size"]] == [1, 1, 1]
    assert len(c["performance"]) == 3 and all(s["points"] == [] for s in c["performance"])
    empty = client.get("/api/progress").json()["datasets"]["calibration"]["charts"]
    assert empty["versions"] == [] and all(s["points"] == [] for s in empty["size"])


def test_page_is_charts_first_with_everything_else_behind_details():
    html = (LABELLER / "web" / "progress.html").read_text()
    body = html.split("<body>")[1].split("<script>")[0]
    assert body.index('id="stages"') < body.index("<details") < body.index('id="datasets"')
    assert body.count("<details") == 1 and 'id="top"' in body.split("<details")[1]   # health + tables inside Details
    assert "<table" not in body and "banner" not in body                             # nothing noisy on the main view
    for stage in ("Tracking", "Broadcast view", "Pitch calibration"):
        assert stage in html
    assert 'get("details") === "1"' in html   # ?details=1 opens it
    # contaminated: hollow marker + one footnote; the tables / diagnostics code is still there
    assert "Hollow marker" in html and "fill=\"var(--panel)\"" in html
    assert "tierTable" in html and "growthTable" in html and "headlineStrip" in html and "healthTop" in html
    assert "cdn" not in html and "unpkg" not in html


# ---- wiring ---------------------------------------------------------------------------------------------
def test_wired_into_server_and_hub():
    assert "progress_routes.router" in (LABELLER / "server.py").read_text()
    assert 'href="/progress"' in (LABELLER / "web" / "main.html").read_text()
    page = (LABELLER / "web" / "progress.html").read_text()
    assert "/api/progress" in page and "contaminated" in page and "<svg" in page.lower() or "svg" in page


# ---- tracking split into ball vs player-side groups (never pooled) -----------------------------------------
def _write_items(root: Path, version: int, rows: list[tuple[str, list[str]]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    payloads = [json.dumps({"boxes": [{"label": lb} for lb in labels], "checks": [], "marker": None})
                for _, labels in rows]
    d = root / "datasets" / "tracking" / f"v{version}"
    pq.write_table(pa.table({"split": [s for s, _ in rows], "payload": payloads}), d / "items.parquet")


def test_tracking_groups_split_ball_and_player_side(env, client):
    registers(env)
    kinds = {"train": {"ball": 4, "no_ball": 1, "check_only": 9}, "eval_in_match": {"ball": 2},
             "eval_heldout_match": {"ball": 3, "check_only": 5}}
    write_manifest(env, manifest("tracking", 1, {"train": 14, "eval_in_match": 2, "eval_heldout_match": 8}, kinds=kinds))
    _write_items(env, 1, [("train", ["player", "referee"]), ("train", ["in_play_ball"]), ("train", ["coach"]),
                          ("eval_in_match", ["player"]), ("eval_heldout_match", ["in_play_ball"])])
    db = Path(os.environ["FOOTY_FEATURE_STORE"])
    seed_results(db, [
        rr(task="tracking", dataset="tracking", model_name="detector", metric="ball_recall_10px", value=0.7,
           split="eval_heldout_match", contaminated=False),
        rr(task="tracking", dataset="tracking", model_name="detector", metric="player_recall", value=0.95,
           split="eval_in_match", contaminated=True),
        rr(task="tracking", dataset="tracking", model_name="detector", metric="map50", value=0.9,
           split="eval_in_match", contaminated=True),
        rr(task="tracking", dataset="tracking", model_name="ball_tracker_viterbi", metric="ball_recall_10px",
           value=0.6, split="eval_heldout_match", contaminated=False),
    ])
    d = client.get("/api/progress?refresh=true").json()["datasets"]["tracking"]
    assert d["charts"]["performance"] and "growth" in d          # legacy keys intact
    g = d["groups"]
    assert set(g) == {"ball", "player"}
    size = lambda grp: {s["split"]: s["points"][0]["value"] for s in g[grp]["charts"]["size"]}  # noqa: E731
    assert size("ball") == {"train": 5, "eval_in_match": 2, "eval_heldout_match": 3}   # ball + no_ball, not check_only
    assert size("player") == {"train": 2, "eval_in_match": 1, "eval_heldout_match": 0}  # frames with player-side boxes
    pts = lambda grp: {s["label"]: s["points"] for s in g[grp]["charts"]["performance"]}  # noqa: E731
    b, p = pts("ball"), pts("player")
    assert b["Detector recall @10px (held-out)"][0]["value"] == 0.7
    assert b["Tracker recall @10px (held-out)"][0]["model_name"] == "ball_tracker_viterbi"
    assert p["Player recall (in-match)"][0]["contaminated"] is True      # contaminated flag carried through
    assert not any(s["metric"] == "map50" for grp in g.values() for s in grp["charts"]["performance"])
    assert not any("player" in s["metric"] for s in g["ball"]["charts"]["performance"])
    assert not g["ball"]["performance_empty"] and not g["player"]["performance_empty"]
    assert p["Player recall (held-out)"] == [] and any("held-out" in n for n in g["player"]["notes"])


def test_tracking_groups_empty_state_has_record_command(env, client):
    registers(env)
    write_manifest(env, manifest("tracking", 1, {"train": 3}))   # no items.parquet, no store
    g = client.get("/api/progress?refresh=true").json()["datasets"]["tracking"]["groups"]
    for grp in ("ball", "player"):
        assert g[grp]["performance_empty"] is True
        assert "results record --task tracking" in g[grp]["record_cmd"]
    assert g["player"]["charts"]["size"][0]["points"] == []       # unreadable snapshot: no invented numbers
    assert "player_recall" in g["player"]["record_cmd"] and "ball_recall_10px" in g["ball"]["record_cmd"]


def test_page_renders_tracking_as_two_group_stages():
    html = (LABELLER / "web" / "progress.html").read_text()
    assert "x.groups" in html and "o.group" in html and "record_cmd" in html
