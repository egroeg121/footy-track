"""Spike: Lance as storage, DuckDB as query engine (docs/design/lance_feature_store.md).

Skipped when lancedb is missing or cannot even be imported -- the lancedb native
wheels abort with SIGILL on CPUs without AVX2 (the Mac mini), and a SIGILL kills
the interpreter, so the import is probed in a subprocess first.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _lancedb_importable() -> bool:
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import lancedb"],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


if not _lancedb_importable():
    pytest.skip(
        "lancedb not importable here (not installed, or no AVX2)",
        allow_module_level=True,
    )

from footy_track.feature_store import DetectionRow, FrameRow, GameRow, RunRow  # noqa: E402
from footy_track.feature_store.lance_store import LanceFeatureStore  # noqa: E402


def _frames(n: int = 4) -> list[FrameRow]:
    return [
        FrameRow(
            game_id="g1",
            frame_index=i,
            frame_uri=f"f{i}.jpg",
            width=1920,
            height=1080,
            continuous_time_s=i * 0.04,
        )
        for i in range(n)
    ]


def _dets(run_id: str, label: str, n: int = 4, x: float = 0.1) -> list[DetectionRow]:
    return [
        DetectionRow(
            game_id="g1",
            frame_index=i,
            detection_id=0,
            source="hand_label",
            run_id=run_id,
            label=label,
            confidence=1.0,
            bbox_x=x + i * 0.01,
            bbox_y=0.2,
            bbox_w=0.05,
            bbox_h=0.05,
            continuous_time_s=i * 0.04,
        )
        for i in range(n)
    ]


@pytest.fixture
def store(tmp_path):
    s = LanceFeatureStore.open(tmp_path / "lance")
    s.upsert("game", [GameRow(game_id="g1", home_team="A", away_team="B", fps=25.0)])
    s.upsert(
        "run",
        [
            RunRow(
                run_id="gt_import_1",
                stage="detection",
                source="hand_label",
                model_name="human",
            )
        ],
    )
    s.upsert("frame", _frames())
    s.upsert("detection", _dets("gt_import_1", "ball"))
    return s


def test_reingest_is_idempotent_and_creates_no_new_version(store):
    v_before = store.table("detection").version
    res = store.upsert("detection", _dets("gt_import_1", "ball"))
    assert (res.inserted, res.updated, res.skipped) == (0, 0, 4)
    assert store.count("detection") == 4
    assert store.table("detection").version == v_before  # nothing written

    # a changed row updates in place (no duplicate), unchanged rows are skipped
    changed = _dets("gt_import_1", "ball")
    changed[0] = _dets("gt_import_1", "ball", x=0.9)[0]
    res = store.upsert("detection", changed)
    assert (res.inserted, res.updated, res.skipped) == (0, 1, 3)
    assert store.count("detection") == 4


def test_tagged_version_survives_label_replacement(store):
    manifest = store.tag_dataset("tracking", 1, note="spike", split_register_hash="abc")
    assert manifest["tag"] == "tracking-v1"
    assert manifest["counts_json"]
    assert (
        store.query("SELECT split_register_hash h FROM dataset_manifest")["h"][0]
        == "abc"
    )

    # GT re-ingest replaces the run's rows (the adcb818 behaviour): delete + insert
    store.delete_where(
        "detection", "game_id = 'g1' AND starts_with(run_id, 'gt_import_')"
    )
    store.upsert(
        "run",
        [
            RunRow(
                run_id="gt_import_2",
                stage="detection",
                source="hand_label",
                model_name="human",
            )
        ],
    )
    store.upsert("detection", _dets("gt_import_2", "in_play_ball", n=2))
    assert store.count("detection") == 2

    # (i) checkout of the tagged version returns the OLD rows
    assert store.count("detection", tag="tracking-v1") == 4
    old = store.query("SELECT DISTINCT run_id, label FROM detection", tag="tracking-v1")
    assert old[["run_id", "label"]].values.tolist() == [["gt_import_1", "ball"]]
    new = store.query("SELECT DISTINCT run_id, label FROM detection")
    assert new[["run_id", "label"]].values.tolist() == [["gt_import_2", "in_play_ball"]]

    # tagging the same version number twice is refused (versions are immutable)
    with pytest.raises(Exception):  # noqa: B017,PT011 - lance raises a generic error
        store.tag_dataset("tracking", 1)

    # a later tag has a different content hash
    m2 = store.tag_dataset("tracking", 2)
    assert m2["content_hash"] != manifest["content_hash"]
    mf = store.query(
        "SELECT dataset, version, tag FROM dataset_manifest ORDER BY version"
    )
    assert mf.values.tolist() == [
        ["tracking", 1, "tracking-v1"],
        ["tracking", 2, "tracking-v2"],
    ]


def test_duckdb_sql_join_across_lance_tables_and_existing_views(store):
    # (ii) plain SQL join across two Lance tables
    df = store.query(
        "SELECT f.frame_uri, d.label, d.bbox_x FROM detection d "
        "JOIN frame f USING (game_id, frame_index) ORDER BY f.frame_index"
    )
    assert df["frame_uri"].tolist() == ["f0.jpg", "f1.jpg", "f2.jpg", "f3.jpg"]
    assert df["label"].unique().tolist() == ["ball"]

    # the real, unmodified views from schema.VIEWS work over Lance tables
    ff = store.query(
        "SELECT game_id, home_team FROM frame_features WHERE frame_index = 0"
    )
    assert ff.values.tolist() == [["g1", "A"]]
    de = store.query("SELECT count(*) AS n FROM detections_enriched")
    assert int(de["n"][0]) == 4


def test_results_table_records_split_and_cohort(store):
    store.record_result(
        task="tracking",
        dataset="tracking",
        dataset_version=1,
        split="eval_heldout_match",
        cohort_version=1,
        model_name="yolo11n",
        model_version="v3",
        metric="ball_recall",
        value=0.77,
        code_sha="abc1234",
    )
    # same key again is a distinct row only if the key differs; this is append-only
    df = store.query("SELECT split, cohort_version, value FROM eval_result")
    assert df.values.tolist() == [["eval_heldout_match", 1, 0.77]]
