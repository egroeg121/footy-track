"""Tests for the GT-marks flush driver (ingest_gt) and schema additions."""

from __future__ import annotations

import json

from footy_track.feature_store import FeatureStore
from footy_track.feature_store.ingest_gt import (
    DATASET_TAG,
    _split_tags,
    ingest_gt_dir,
    ingest_gt_jsonl,
)


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _marks(stem, tmp_path):
    records = [
        # ball, labeller -> hand_label
        {"frame_index": 0, "bbox": {"x": 0.5, "y": 0.5, "w": 0.02, "h": 0.03},
         "tags": ["in_play_ball", "labeller"]},
        # player, labeller
        {"frame_index": 0, "bbox": {"x": 0.1, "y": 0.2, "w": 0.05, "h": 0.1},
         "tags": ["player", "labeller"]},
        # yolo provenance -> yolo source
        {"frame_index": 1, "bbox": {"x": 0.3, "y": 0.3, "w": 0.04, "h": 0.09},
         "tags": ["player", "yolo"]},
        # skip markers -> frame recorded, no detection
        {"frame_index": 2, "bbox": None, "tags": ["no_ball"]},
        {"frame_index": 3, "bbox": None, "tags": ["not_broadcast"]},
    ]
    path = tmp_path / f"{stem}.jsonl"
    _write_jsonl(path, records)
    return path


def test_split_tags() -> None:
    assert _split_tags(["in_play_ball", "labeller"]) == ("in_play_ball", "labeller")
    assert _split_tags(["player", "yolo"]) == ("player", "yolo")
    assert _split_tags(["no_ball"]) == (None, None)


def test_ingest_maps_provenance_and_flags(tmp_path) -> None:
    path = _marks("arsenal_demo", tmp_path)
    store = FeatureStore.open(":memory:")
    report = ingest_gt_jsonl(store, path, video_dir=tmp_path)

    assert report.detections_written == 3
    assert report.by_source == {"hand_label": 2, "yolo": 1}
    # all 4 distinct frame indices recorded on the spine (0,1,2,3)
    assert report.frames_written == 4

    df = store.query("SELECT source, label, reviewed, dataset_tag FROM detection ORDER BY source, label")
    assert set(df["source"]) == {"hand_label", "yolo"}
    assert all(bool(v) for v in df["reviewed"])
    assert all(v == DATASET_TAG for v in df["dataset_tag"])


def test_ingest_skips_no_ball_and_not_broadcast(tmp_path) -> None:
    path = _marks("arsenal_demo", tmp_path)
    store = FeatureStore.open(":memory:")
    ingest_gt_jsonl(store, path, video_dir=tmp_path)
    # frames 2 and 3 exist but carry no detection
    assert store.count("frame") == 4
    empty = store.query(
        "SELECT frame_index FROM frame f WHERE NOT EXISTS "
        "(SELECT 1 FROM detection d WHERE d.game_id=f.game_id AND d.frame_index=f.frame_index) "
        "ORDER BY frame_index"
    )
    assert list(empty["frame_index"]) == [2, 3]


def test_ingest_is_idempotent(tmp_path) -> None:
    path = _marks("arsenal_demo", tmp_path)
    store = FeatureStore.open(":memory:")
    ingest_gt_jsonl(store, path, video_dir=tmp_path)
    ingest_gt_jsonl(store, path, video_dir=tmp_path)
    assert store.count("detection") == 3
    assert store.count("frame") == 4


def test_ingest_default_fps_when_no_video(tmp_path) -> None:
    path = _marks("arsenal_demo", tmp_path)
    store = FeatureStore.open(":memory:")
    report = ingest_gt_jsonl(store, path, video_dir=tmp_path)
    assert "arsenal_demo" in report.clips_missing_video
    # continuous_time = frame_index / DEFAULT_FPS(25) -> frame 1 = 0.04
    ct = store.query("SELECT continuous_time_s FROM detection WHERE frame_index=1")
    assert abs(float(ct["continuous_time_s"][0]) - 0.04) < 1e-6


def test_canonical_prefers_hand_label(tmp_path) -> None:
    # same frame has a hand_label and a yolo ball; canonical should pick hand_label
    records = [
        {"frame_index": 0, "bbox": {"x": 0.5, "y": 0.5, "w": 0.02, "h": 0.03},
         "tags": ["in_play_ball", "labeller"]},
        {"frame_index": 0, "bbox": {"x": 0.51, "y": 0.51, "w": 0.02, "h": 0.03},
         "tags": ["in_play_ball", "yolo"]},
    ]
    path = tmp_path / "arsenal_demo.jsonl"
    _write_jsonl(path, records)
    store = FeatureStore.open(":memory:")
    ingest_gt_jsonl(store, path, video_dir=tmp_path)
    df = store.query(
        "SELECT source FROM detections_enriched WHERE canonical AND frame_index=0"
    )
    assert set(df["source"]) == {"hand_label"}


def test_ingest_dir_clip_filter(tmp_path) -> None:
    _marks("clip_a", tmp_path)
    _marks("clip_b", tmp_path)
    store = FeatureStore.open(":memory:")
    report = ingest_gt_dir(store, tmp_path, video_dir=tmp_path, clip="clip_a")
    assert report.games == {"clip_a"}
