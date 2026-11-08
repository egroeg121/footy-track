"""T-04: re-ingesting a GT sidecar REPLACES the clip's gt_import_* rows."""

from __future__ import annotations

import json

import pytest

from footy_track.feature_store import (
    DetectionRow,
    FeatureStore,
    FrameRow,
    GameRow,
    RunRow,
)
from footy_track.feature_store.ingest_gt import ingest_gt_dir, ingest_gt_jsonl
from footy_track.feature_store.schema import Stage

STEM = "clip_a"


def _rec(fi, prov, x=0.5, label="in_play_ball"):
    return {
        "frame_index": fi,
        "bbox": {"x": x, "y": 0.5, "w": 0.02, "h": 0.03},
        "tags": [label, prov],
    }


def _write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _snapshot(store):
    return store.query(
        "SELECT * FROM detection ORDER BY game_id, source, run_id, frame_index, detection_id"
    ).to_dict("records")


def _frozen(store):
    # NaN != NaN, so compare a textual form.
    return repr(_snapshot(store))


def _ingest(store, tmp_path, **kw):
    return ingest_gt_jsonl(store, tmp_path / f"{STEM}.jsonl", video_dir=tmp_path, **kw)


def _canonical(store):
    return store.query(
        "SELECT frame_index, source, run_id FROM detections_enriched "
        "WHERE game_id = ? AND canonical ORDER BY frame_index, source",
        [STEM],
    )


def _roboflow(store):
    store.upsert_games([GameRow(game_id=STEM)])
    store.upsert_frames(
        [
            FrameRow(
                game_id=STEM,
                frame_index=0,
                frame_uri="rf0",
                width=10,
                height=10,
                continuous_time_s=0.0,
            )
        ]
    )
    store.upsert_runs(
        [
            RunRow(
                run_id="roboflow_v1",
                stage=Stage.DETECTION,
                source="hand_label",
                model_name="roboflow",
            )
        ]
    )
    store.upsert_detections(
        [
            DetectionRow(
                game_id=STEM,
                frame_index=0,
                continuous_time_s=0.0,
                detection_id=0,
                source="hand_label",
                run_id="roboflow_v1",
                label="ball",
                bbox_x=0.1,
                bbox_y=0.1,
                bbox_w=0.1,
                bbox_h=0.1,
                reviewed=True,
            )
        ]
    )


def test_provenance_change_and_delete_leave_no_stale_rows(tmp_path):
    store = FeatureStore.open(":memory:")
    sidecar = tmp_path / f"{STEM}.jsonl"
    _write(
        sidecar,
        [
            _rec(0, "vittrack"),
            _rec(1, "vittrack"),
            _rec(2, "vittrack"),
            _rec(2, "vittrack", x=0.7),
        ],
    )
    _ingest(store, tmp_path)
    assert store.count("detection") == 4

    # frame 0: vittrack -> labeller; frame 1 deleted; frame 2 first box deleted
    _write(sidecar, [_rec(0, "labeller"), _rec(2, "vittrack", x=0.7)])
    r = _ingest(store, tmp_path)
    assert r.detections_deleted == 4

    rows = _snapshot(store)
    assert [
        (d["frame_index"], d["source"], d["run_id"], d["detection_id"]) for d in rows
    ] == [
        (0, "hand_label", "gt_import_labeller", 0),
        (2, "vittrack", "gt_import_vittrack", 0),
    ]
    assert abs(rows[1]["bbox_x"] - 0.7) < 1e-6  # shifted id holds the right content
    canon = _canonical(store)
    assert len(canon) == 2 and set(canon["frame_index"]) == {0, 2}


def test_other_runs_untouched(tmp_path):
    store = FeatureStore.open(":memory:")
    _roboflow(store)
    _write(tmp_path / f"{STEM}.jsonl", [_rec(0, "vittrack")])
    _ingest(store, tmp_path)
    _write(tmp_path / f"{STEM}.jsonl", [_rec(0, "labeller")])
    _ingest(store, tmp_path)
    rf = store.query("SELECT * FROM detection WHERE run_id = 'roboflow_v1'")
    assert len(rf) == 1 and abs(rf.iloc[0]["bbox_x"] - 0.1) < 1e-6
    assert store.count("detection") == 2
    # an emptied sidecar must still not touch roboflow
    _write(tmp_path / f"{STEM}.jsonl", [])
    _ingest(store, tmp_path)
    assert store.count("detection") == 1


def test_idempotent(tmp_path):
    store = FeatureStore.open(":memory:")
    _write(
        tmp_path / f"{STEM}.jsonl",
        [_rec(0, "labeller"), _rec(1, "vittrack"), _rec(1, "vittrack", x=0.8)],
    )
    _ingest(store, tmp_path)
    first = _frozen(store)
    counts = {t: store.count(t) for t in ("game", "frame", "run", "detection")}
    _ingest(store, tmp_path)
    ingest_gt_dir(store, tmp_path, video_dir=tmp_path)
    assert _frozen(store) == first
    assert {t: store.count(t) for t in counts} == counts


def test_emptied_sidecar_clears_rows(tmp_path):
    store = FeatureStore.open(":memory:")
    sidecar = tmp_path / f"{STEM}.jsonl"
    _write(sidecar, [_rec(0, "labeller"), _rec(1, "vittrack")])
    _ingest(store, tmp_path)
    assert store.count("detection") == 2
    _write(sidecar, [])
    _ingest(store, tmp_path)
    assert store.count("detection") == 0
    # sidecar with only skip markers also clears
    _write(sidecar, [_rec(0, "labeller")])
    _ingest(store, tmp_path)
    _write(sidecar, [{"frame_index": 0, "bbox": None, "tags": ["no_ball"]}])
    _ingest(store, tmp_path)
    assert store.count("detection") == 0


def test_no_replace_keeps_old_merge_behaviour(tmp_path):
    store = FeatureStore.open(":memory:")
    sidecar = tmp_path / f"{STEM}.jsonl"
    _write(sidecar, [_rec(0, "vittrack")])
    _ingest(store, tmp_path)
    _write(sidecar, [_rec(0, "labeller")])
    _ingest(store, tmp_path, replace=False)
    assert store.count("detection") == 2


def test_bad_sidecar_keeps_old_rows(tmp_path):
    store = FeatureStore.open(":memory:")
    sidecar = tmp_path / f"{STEM}.jsonl"
    _write(sidecar, [_rec(0, "labeller")])
    _ingest(store, tmp_path)
    sidecar.write_text("{not json\n")
    with pytest.raises(ValueError):
        _ingest(store, tmp_path)
    assert store.count("detection") == 1


def test_only_target_game_replaced(tmp_path):
    store = FeatureStore.open(":memory:")
    for stem in ("clip_a", "clip_b"):
        _write(tmp_path / f"{stem}.jsonl", [_rec(0, "labeller")])
    ingest_gt_dir(store, tmp_path, video_dir=tmp_path)
    _write(tmp_path / "clip_a.jsonl", [])
    ingest_gt_dir(store, tmp_path, video_dir=tmp_path, clip="clip_a")
    left = store.query("SELECT DISTINCT game_id FROM detection")
    assert list(left["game_id"]) == ["clip_b"]
