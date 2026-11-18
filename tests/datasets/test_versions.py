"""Dataset snapshots: item tables, content hash, bump-only-on-change, immutability, manifest diff,
growing eval sets and cohorts."""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime

import duckdb
import pyarrow.parquet as pq
import pytest

from footy_track import splits
from footy_track.datasets import snapshot as snap
from footy_track.datasets.__main__ import main as cli
from footy_track.datasets.items import DatasetBuildError, build_items
from footy_track.feature_store import ingest_all as ia
from tests.datasets.fixtures import box, calibration_doc, skip, write_broadcast


@pytest.fixture
def ingested(world):
    ia.run(world.db, data_root=world.root, marks_dirs=[world.marks])
    return world


def bump(w, dataset, **kw):
    return snap.bump(dataset, w.db, w.ds_root, **kw)


def reingest(w):
    ia.run(w.db, data_root=w.root, marks_dirs=[w.marks])


def items(w, dataset):
    conn = duckdb.connect(str(w.db), read_only=True)
    try:
        return {i.item_id: i for i in build_items(dataset, conn)}
    finally:
        conn.close()


# ---- item tables ----------------------------------------------------------------------------
def test_tracking_items_carry_split_kind_tier(ingested):
    it = items(ingested, "tracking")
    assert it["hmatch_seg001:0"].split == splits.EVAL_HELDOUT
    assert it["tmatch_seg001:0"].split == splits.EVAL_IN_MATCH
    assert it["tmatch_seg002:0"].split == splits.TRAIN
    assert it["hmatch_seg001:0"].label_kind == "ball" and it["hmatch_seg001:0"].tier == 1
    assert it["hmatch_seg001:2"].label_kind == "no_ball"
    assert it["tmatch_seg001:1"].label_kind == "not_broadcast"
    assert it["tmatch_seg003:5"].label_kind == "other_boxes"
    # tiers: human_checked machine box = 2, machine only = 3
    assert it["tmatch_seg002:1"].tier == 2 and it["tmatch_seg002:2"].tier == 3
    assert it["tmatch_seg002:0"].match_key == "tmatch" and it["tmatch_seg002:0"].clip == "tmatch_seg002"
    payload = json.loads(it["hmatch_seg001:0"].payload)
    assert {b["label"] for b in payload["boxes"]} == {"in_play_ball", "player"}


def test_calibration_and_broadcast_items(ingested):
    cal = items(ingested, "calibration")
    assert cal["hmatch_seg001:3"].split == splits.EVAL_HELDOUT and cal["hmatch_seg001:3"].label_kind == "homography"
    assert cal["tmatch_seg003:9"].label_kind == "not_broadcast"
    assert len(json.loads(cal["tmatch_seg002:7"].payload)["correspondences"]) == 4
    bc = items(ingested, "broadcast")
    assert len(bc) == 5
    # the review's match is its video stem; "hmatch_full" is an alias of the held-out match
    assert {i.split for i in bc.values()} == {splits.EVAL_HELDOUT}
    assert bc["hmatch_full:10"].label_kind == "yes" and bc["hmatch_full:14"].label_kind == "unsure"
    assert bc["hmatch_full:10"].clip == "hmatch_w0000"


def test_missing_register_fails_closed(ingested, monkeypatch):
    monkeypatch.setenv("FOOTY_CLIP_SPLITS", str(ingested.root / "nope.json"))
    splits.clear_cache()
    with pytest.raises(DatasetBuildError, match="PREVIEW"):
        items(ingested, "tracking")
    with pytest.raises(snap.SnapshotError, match="refusing to bump"):
        bump(ingested, "tracking")


# ---- hashing ---------------------------------------------------------------------------------
def test_content_hash_ignores_time_git_and_paths(ingested, tmp_path):
    a = bump(ingested, "tracking", now=datetime(2026, 1, 1, tzinfo=UTC))
    other_root = tmp_path / "elsewhere"
    b = snap.bump("tracking", ingested.db, other_root, now=datetime(2030, 6, 1, tzinfo=UTC), note="x")
    assert a.content_hash == b.content_hash
    assert a.manifest["created_at"] != b.manifest["created_at"]


def test_hash_is_stable_across_independent_ingests(world, tmp_path):
    ia.run(world.db, data_root=world.root, marks_dirs=[world.marks])
    db2 = tmp_path / "second.duckdb"
    ia.run(db2, data_root=world.root, marks_dirs=[world.marks])
    for ds in ("tracking", "broadcast", "calibration"):
        a = snap.bump(ds, world.db, tmp_path / "r1")
        b = snap.bump(ds, db2, tmp_path / "r2")
        assert a.content_hash == b.content_hash, ds


# ---- bump ------------------------------------------------------------------------------------
def test_bump_writes_only_when_content_changes(ingested):
    r1 = bump(ingested, "tracking", note="first")
    assert r1.changed and r1.version == 1 and r1.manifest["parent_version"] is None
    r2 = bump(ingested, "tracking")
    assert not r2.changed and r2.version == 1
    assert snap.list_versions("tracking", ingested.ds_root) == [1]

    ingested.sidecar("tmatch_seg003", [box(5, "player"), box(6, "in_play_ball")])
    reingest(ingested)
    r3 = bump(ingested, "tracking")
    assert r3.changed and r3.version == 2 and r3.manifest["parent_version"] == 1
    assert r3.manifest["parent_content_hash"] == r1.content_hash
    assert not bump(ingested, "tracking").changed
    assert snap.list_versions("tracking", ingested.ds_root) == [1, 2]


def test_datasets_version_independently(ingested):
    for ds in ("tracking", "broadcast", "calibration"):
        bump(ingested, ds)
    write_broadcast(ingested, "hmatch_full", {10: ("yes", "Yes", "edge")})
    reingest(ingested)
    assert bump(ingested, "broadcast").version == 2
    assert not bump(ingested, "tracking").changed
    assert not bump(ingested, "calibration").changed
    assert [snap.latest_version(d, ingested.ds_root) for d in ("tracking", "broadcast", "calibration")] == [1, 2, 1]


def test_empty_dataset_is_refused_unless_allowed(world):
    ia.run(world.db, data_root=world.root, marks_dirs=[world.marks])
    write_broadcast(world, "hmatch_full", {})  # every verdict cleared in the UI: the file stays, empty
    reingest(world)
    with pytest.raises(snap.SnapshotError, match="empty"):
        bump(world, "broadcast")
    assert bump(world, "broadcast", allow_empty=True).version == 1


def test_version_is_recorded_in_the_store(ingested):
    bump(ingested, "calibration", note="n")
    conn = duckdb.connect(str(ingested.db), read_only=True)
    try:
        row = conn.execute("SELECT dataset, version, content_hash, n_items, note FROM dataset_version").fetchall()
    finally:
        conn.close()
    dv = snap.load_version("calibration", 1, ingested.ds_root)
    assert row == [("calibration", 1, dv.content_hash, len(dv.rows()), "n")]
    assert len(dv.rows()) == 5  # 1 + 1 + 2 homographies + 1 not_broadcast flag


# ---- immutability ----------------------------------------------------------------------------
def test_snapshots_are_read_only_and_never_overwritten(ingested):
    r = bump(ingested, "tracking")
    for f in r.path.iterdir():
        assert not (f.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)), f
    dv = snap.load_version("tracking", 1, ingested.ds_root)
    with pytest.raises(snap.SnapshotExists):
        snap.write_snapshot("tracking", 1, dv.rows(), dv.manifest, ingested.ds_root)
    with pytest.raises(PermissionError):
        (r.path / "manifest.json").write_text("{}")
    assert snap.load_version("tracking", 1, ingested.ds_root).content_hash == r.content_hash


def test_tampered_snapshot_is_detected(ingested):
    r = bump(ingested, "tracking")
    p = r.path / "items.parquet"
    os.chmod(p, 0o644)
    t = pq.read_table(p).to_pylist()
    t[0]["payload"] = t[0]["payload"].replace("0.", "9.", 1)
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(t, schema=snap.ITEM_SCHEMA_ARROW), p)
    with pytest.raises(snap.SnapshotTampered):
        snap.load_version("tracking", 1, ingested.ds_root)
    with pytest.raises(snap.SnapshotTampered):  # and bump refuses to build on a tampered parent
        bump(ingested, "tracking")


# ---- manifest + diff -------------------------------------------------------------------------
def test_manifest_fields_counts_and_diff(ingested):
    r1 = bump(ingested, "tracking")
    m1 = r1.manifest
    assert m1["counts"]["by_split"] == {"eval_heldout_match": 3, "eval_in_match": 2, "train": 4}
    assert m1["counts"]["by_split_kind"]["eval_heldout_match"] == {"ball": 2, "no_ball": 1}
    assert m1["split_registers"]["match_splits_sha256"] and m1["split_registers"]["clip_splits_sha256"]
    assert m1["split_registers"]["preview"] is False
    assert "git_sha" in m1 and m1["diff_vs_parent"]["added"] == 9 and m1["diff_vs_parent"]["removed"] == 0

    # change one box, add a frame in an eval clip, delete a train frame
    ingested.sidecar("tmatch_seg002", [box(0, "in_play_ball", x=0.6), box(2, "in_play_ball", prov="vittrack")])
    ingested.sidecar("tmatch_seg001", [box(0, "in_play_ball"), skip(1, "not_broadcast"), box(8, "in_play_ball")])
    reingest(ingested)
    r2 = bump(ingested, "tracking")
    d = r2.manifest["diff_vs_parent"]
    assert d["added_by_split"] == {"eval_in_match": 1}
    assert d["removed_by_split"] == {"train": 1}  # tmatch_seg002 frame 1
    assert d["changed_by_split"] == {"train": 1}  # tmatch_seg002 frame 0 moved
    assert (d["added"], d["removed"], d["changed"]) == (1, 1, 1)
    full = snap.diff_versions("tracking", 1, 2, ingested.ds_root)
    assert full["added_ids"] == ["tmatch_seg001:8"]
    assert full["removed_ids"] == ["tmatch_seg002:1"]
    assert full["changed_ids"] == ["tmatch_seg002:0"]


# ---- growing eval set, first_version, cohorts ------------------------------------------------
def test_eval_is_monotone_and_cohorts_filter_by_first_version(ingested):
    bump(ingested, "tracking")
    # a new in-match eval clip and a new heldout frame, plus new train data
    ingested.sidecar("tmatch_seg004", [box(0, "in_play_ball"), box(1, "in_play_ball")])
    ingested.sidecar("hmatch_seg001", [box(0, "in_play_ball"), box(0, "player", x=0.2), box(1, "in_play_ball"),
                                       skip(2, "no_ball"), box(7, "in_play_ball")])
    ingested.sidecar("tmatch_seg003", [box(5, "player"), box(6, "in_play_ball")])
    reingest(ingested)
    bump(ingested, "tracking")
    v1 = snap.load_version("tracking", 1, ingested.ds_root)
    v2 = snap.load_version("tracking", 2, ingested.ds_root)
    for split in ("eval_heldout_match", "eval_in_match", "eval_all"):
        ids1 = {r["item_id"] for r in v1.select(split)}
        ids2 = {r["item_id"] for r in v2.select(split)}
        assert ids1 < ids2, split  # eval(v2) is a strict superset of eval(v1)
        # the fixed cohort is exactly the earlier eval set
        assert {r["item_id"] for r in v2.select(split, cohort=1)} == ids1
        assert {r["item_id"] for r in v2.select(split, cohort=2)} == ids2
    # items keep the version they first appeared in
    fv = {r["item_id"]: r["first_version"] for r in v2.rows()}
    assert fv["hmatch_seg001:0"] == 1 and fv["hmatch_seg001:7"] == 2 and fv["tmatch_seg004:0"] == 2
    # nothing moved between splits
    s1 = {r["item_id"]: r["split"] for r in v1.rows()}
    assert all({r["item_id"]: r["split"] for r in v2.rows()}[k] == v for k, v in s1.items())


def test_removed_then_returning_item_gets_a_new_first_version(ingested):
    bump(ingested, "tracking")
    ingested.sidecar("tmatch_seg003", [])
    reingest(ingested)
    assert bump(ingested, "tracking").manifest["diff_vs_parent"]["removed"] == 1
    ingested.sidecar("tmatch_seg003", [box(5, "player")])
    reingest(ingested)
    bump(ingested, "tracking")
    v3 = snap.load_version("tracking", 3, ingested.ds_root)
    assert {r["item_id"]: r["first_version"] for r in v3.rows()}["tmatch_seg003:5"] == 3


# ---- CLI -------------------------------------------------------------------------------------
def test_cli_build_bump_list_show_diff(ingested, capsys):
    argv = ["--db", str(ingested.db), "--root", str(ingested.ds_root)]
    assert cli([*argv, "build", "calibration"]) == 0
    assert "would write v1" in capsys.readouterr().out
    assert cli([*argv, "bump", "all", "--note", "init"]) == 0
    out = capsys.readouterr().out
    assert out.count("wrote v1") == 3
    assert cli([*argv, "bump", "all"]) == 0
    assert capsys.readouterr().out.count("unchanged at v1") == 3
    assert cli([*argv, "build", "calibration"]) == 0
    assert "identical to v1" in capsys.readouterr().out
    assert cli([*argv, "list"]) == 0
    assert "v1" in capsys.readouterr().out
    assert cli([*argv, "show", "broadcast"]) == 0
    assert json.loads(capsys.readouterr().out)["dataset"] == "broadcast"
    ingested.marks.joinpath("calibration_gt", "tmatch_seg002.json").write_text(
        json.dumps(calibration_doc("tmatch_seg002", {6: 0.0, 7: 5.0, 8: 1.0}))
    )
    reingest(ingested)
    assert cli([*argv, "bump", "calibration"]) == 0
    capsys.readouterr()
    assert cli([*argv, "diff", "calibration", "v1", "v2"]) == 0
    assert json.loads(capsys.readouterr().out)["added_ids"] == ["tmatch_seg002:8"]
