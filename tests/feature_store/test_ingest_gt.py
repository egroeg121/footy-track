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


def _marks_with_vittrack(stem, tmp_path):
    records = [
        # ball, labeller -> hand_label, reviewed
        {"frame_index": 0, "bbox": {"x": 0.5, "y": 0.5, "w": 0.02, "h": 0.03},
         "tags": ["in_play_ball", "labeller"]},
        # ball, vittrack -> vittrack source, unreviewed
        {"frame_index": 1, "bbox": {"x": 0.5, "y": 0.5, "w": 0.02, "h": 0.03},
         "tags": ["in_play_ball", "vittrack"]},
        # player, yolo -> yolo source, unreviewed
        {"frame_index": 2, "bbox": {"x": 0.3, "y": 0.3, "w": 0.04, "h": 0.09},
         "tags": ["player", "yolo"]},
    ]
    path = tmp_path / f"{stem}.jsonl"
    _write_jsonl(path, records)
    return path


def test_split_tags() -> None:
    assert _split_tags(["in_play_ball", "labeller"]) == ("in_play_ball", "labeller")
    assert _split_tags(["player", "yolo"]) == ("player", "yolo")
    assert _split_tags(["in_play_ball", "vittrack"]) == ("in_play_ball", "vittrack")
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
    # per-tag tiering (default): only hand_label rows are reviewed
    reviewed_by_source = dict(zip(df["source"], df["reviewed"], strict=True))
    for source, reviewed in reviewed_by_source.items():
        assert bool(reviewed) == (source == "hand_label")
    assert all(v == DATASET_TAG for v in df["dataset_tag"])


def test_ingest_vittrack_lands_as_model_tier_unreviewed(tmp_path) -> None:
    path = _marks_with_vittrack("arsenal_demo", tmp_path)
    store = FeatureStore.open(":memory:")
    report = ingest_gt_jsonl(store, path, video_dir=tmp_path)

    assert report.by_source == {"hand_label": 1, "vittrack": 1, "yolo": 1}

    df = store.query("SELECT source, reviewed FROM detection ORDER BY source")
    reviewed_by_source = dict(zip(df["source"], df["reviewed"], strict=True))
    assert reviewed_by_source == {"hand_label": True, "vittrack": False, "yolo": False}


def test_ingest_labeller_lands_as_hand_label_reviewed(tmp_path) -> None:
    path = _marks_with_vittrack("arsenal_demo", tmp_path)
    store = FeatureStore.open(":memory:")
    ingest_gt_jsonl(store, path, video_dir=tmp_path)

    df = store.query("SELECT reviewed FROM detection WHERE source='hand_label'")
    assert all(bool(v) for v in df["reviewed"])


def test_ingest_mixed_provenance_splits_correctly(tmp_path) -> None:
    """A single sidecar mixing labeller/vittrack/yolo tags should split into
    distinct sources with the correct per-tier reviewed flags."""
    path = _marks_with_vittrack("mixed_clip", tmp_path)
    store = FeatureStore.open(":memory:")
    report = ingest_gt_jsonl(store, path, video_dir=tmp_path)

    assert report.detections_written == 3
    assert set(report.by_source.keys()) == {"hand_label", "vittrack", "yolo"}

    df = store.query("SELECT source, reviewed FROM detection ORDER BY frame_index")
    assert list(df["source"]) == ["hand_label", "vittrack", "yolo"]
    assert [bool(v) for v in df["reviewed"]] == [True, False, False]


def test_ingest_legacy_all_reviewed_flag(tmp_path) -> None:
    """--legacy-all-reviewed opt-in reproduces the original blanket-reviewed
    behavior for callers that still depend on it."""
    path = _marks_with_vittrack("arsenal_demo", tmp_path)
    store = FeatureStore.open(":memory:")
    ingest_gt_jsonl(store, path, video_dir=tmp_path, legacy_all_reviewed=True)

    df = store.query("SELECT reviewed FROM detection")
    assert all(bool(v) for v in df["reviewed"])


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


def test_canonical_keeps_coexisting_machine_rows_from_same_sidecar_flush(
    tmp_path,
) -> None:
    """Regression for the frame-203 bug: a single sidecar flush that hand-
    corrects one object on a frame while leaving other machine-detected
    objects on that same frame is one coherent, already-resolved snapshot.
    The labeller has already decided both boxes belong together, so both
    must be canonical -- canonical must not re-collapse the frame down to
    only the hand_label row.
    """
    records = [
        {"frame_index": 0, "bbox": {"x": 0.5, "y": 0.5, "w": 0.02, "h": 0.03},
         "tags": ["in_play_ball", "labeller"]},
        {"frame_index": 0, "bbox": {"x": 0.51, "y": 0.51, "w": 0.02, "h": 0.03},
         "tags": ["player", "yolo"]},
    ]
    path = tmp_path / "arsenal_demo.jsonl"
    _write_jsonl(path, records)
    store = FeatureStore.open(":memory:")
    ingest_gt_jsonl(store, path, video_dir=tmp_path)
    df = store.query(
        "SELECT source FROM detections_enriched WHERE canonical AND frame_index=0"
    )
    assert set(df["source"]) == {"hand_label", "yolo"}


def test_canonical_sidecar_with_hand_label_beats_roboflow_on_same_frame(
    tmp_path,
) -> None:
    """Competing-sources case: a Roboflow import and a GT-marks sidecar both
    cover the same (game_id, frame_index). The sidecar's run_group contains a
    hand_label row, so it wins outright -- every row in the sidecar's
    run_group is canonical and every Roboflow row on that frame is not, even
    though the Roboflow rows are also source=hand_label.
    """
    from footy_track.feature_store.schema import (  # noqa: PLC0415
        DetectionRow,
        FrameRow,
        GameRow,
        RunRow,
        Stage,
    )

    store = FeatureStore.open(":memory:")

    # Fabricate a roboflow-style import for the same frame the sidecar
    # will also cover, without needing a real dataset dir on disk.
    store.upsert_games([GameRow(game_id="arsenal_demo", fps=25.0, width=1920, height=1080)])
    store.upsert_frames(
        [FrameRow(game_id="arsenal_demo", frame_index=0, frame_uri="x", width=1920,
                   height=1080, continuous_time_s=0.0)]
    )
    store.upsert_runs(
        [RunRow(run_id="roboflow_v1", stage=Stage.DETECTION, source="hand_label", model_name="human")]
    )
    store.upsert_detections(
        [DetectionRow(game_id="arsenal_demo", frame_index=0, continuous_time_s=0.0,
                       detection_id=0, source="hand_label", run_id="roboflow_v1",
                       label="in_play_ball", bbox_x=0.2, bbox_y=0.2, bbox_w=0.02, bbox_h=0.03)]
    )

    # Sidecar flush covering the same frame, with a hand label.
    records = [
        {"frame_index": 0, "bbox": {"x": 0.5, "y": 0.5, "w": 0.02, "h": 0.03},
         "tags": ["in_play_ball", "labeller"]},
        {"frame_index": 0, "bbox": {"x": 0.51, "y": 0.51, "w": 0.02, "h": 0.03},
         "tags": ["player", "yolo"]},
    ]
    path = tmp_path / "arsenal_demo.jsonl"
    _write_jsonl(path, records)
    ingest_gt_jsonl(store, path, video_dir=tmp_path)

    df = store.query(
        "SELECT source, run_id FROM detections_enriched WHERE canonical AND frame_index=0"
    )
    assert set(df["run_id"]) == {"gt_import_labeller", "gt_import_yolo"}
    assert set(df["source"]) == {"hand_label", "yolo"}

    non_canonical = store.query(
        "SELECT run_id FROM detections_enriched WHERE NOT canonical AND frame_index=0"
    )
    assert set(non_canonical["run_id"]) == {"roboflow_v1"}


def test_canonical_yolo_only_frame_stays_canonical(tmp_path) -> None:
    """A frame with no hand label at all (pure machine tier, single batch)
    keeps all of its rows canonical -- there's no competing batch to lose to.
    """
    path = _marks("arsenal_demo", tmp_path)  # frame 1 is yolo-only
    store = FeatureStore.open(":memory:")
    ingest_gt_jsonl(store, path, video_dir=tmp_path)
    df = store.query(
        "SELECT source, canonical FROM detections_enriched WHERE frame_index=1"
    )
    assert list(df["source"]) == ["yolo"]
    assert bool(df["canonical"][0]) is True


def test_ingest_dir_clip_filter(tmp_path) -> None:
    _marks("clip_a", tmp_path)
    _marks("clip_b", tmp_path)
    store = FeatureStore.open(":memory:")
    report = ingest_gt_dir(store, tmp_path, video_dir=tmp_path, clip="clip_a")
    assert report.games == {"clip_a"}
