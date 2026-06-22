"""Tests for the pipeline-output -> feature-store ingestion adapters."""

from __future__ import annotations

import pytest

from footy_track.feature_store import (
    ClipRow,
    FeatureStore,
    GameRow,
    detector_run,
    ingest_frame,
    to_detection_rows,
    to_frame_row,
)
from footy_track.schema import (
    BroadcastClassification,
    EnumBroadcastClassification,
    FrameDetections,
    ObjectDetection,
)


def _frame_detections(uri: str = "f0.jpg", n: int = 2) -> FrameDetections:
    return FrameDetections(
        uri=uri,
        width=1920,
        height=1080,
        detections=[
            ObjectDetection(
                label="player",
                confidence=0.8 + 0.01 * i,
                x=0.4,
                y=0.4,
                w=0.05,
                h=0.1,
                model="yolo11n.pt",
            )
            for i in range(n)
        ],
    )


@pytest.fixture
def store() -> FeatureStore:
    s = FeatureStore.open(":memory:")
    s.upsert_games([GameRow(game_id="g1")])
    s.upsert_clips([ClipRow(clip_id="g1_clip0", game_id="g1")])
    s.upsert_runs(
        [detector_run("yolo_v3", "yolo11n.pt", source="yolo", model_version="v3")]
    )
    return s


def test_to_detection_rows_uses_per_frame_index() -> None:
    rows = to_detection_rows(
        _frame_detections(n=3),
        clip_id="g1_clip0",
        game_id="g1",
        frame_index=5,
        continuous_time_s=0.2,
        source="yolo",
        run_id="yolo_v3",
    )
    assert [r.detection_id for r in rows] == [0, 1, 2]
    assert all(r.frame_index == 5 for r in rows)
    assert rows[0].bbox_x == pytest.approx(0.4)
    assert rows[0].track_id is None


def test_to_detection_rows_with_track_ids() -> None:
    rows = to_detection_rows(
        _frame_detections(n=2),
        clip_id="g1_clip0",
        game_id="g1",
        frame_index=0,
        continuous_time_s=0.0,
        source="bytetrack",
        run_id="bt_1",
        track_ids=[7, 9],
    )
    assert [r.track_id for r in rows] == [7, 9]


def test_to_detection_rows_track_id_length_mismatch() -> None:
    with pytest.raises(ValueError, match="track_ids length"):
        to_detection_rows(
            _frame_detections(n=2),
            clip_id="g1_clip0",
            game_id="g1",
            frame_index=0,
            continuous_time_s=0.0,
            source="bytetrack",
            run_id="bt_1",
            track_ids=[7],
        )


def test_to_frame_row_maps_broadcast_yes_no() -> None:
    yes = to_frame_row(
        clip_id="g1_clip0",
        game_id="g1",
        frame_index=0,
        frame_uri="f0.jpg",
        width=1920,
        height=1080,
        continuous_time_s=0.0,
        classification=BroadcastClassification(
            label=EnumBroadcastClassification.YES, confidence=0.97
        ),
        broadcast_run_id="cls_1",
    )
    assert yes.is_broadcast is True
    assert yes.broadcast_confidence == pytest.approx(0.97)
    assert yes.broadcast_model_version == "cls_1"

    no = to_frame_row(
        clip_id="g1_clip0",
        game_id="g1",
        frame_index=1,
        frame_uri="f1.jpg",
        width=1920,
        height=1080,
        continuous_time_s=0.04,
        classification=BroadcastClassification(
            label=EnumBroadcastClassification.NO, confidence=0.6
        ),
    )
    assert no.is_broadcast is False


def test_to_frame_row_without_classification_leaves_broadcast_null() -> None:
    row = to_frame_row(
        clip_id="g1_clip0",
        game_id="g1",
        frame_index=0,
        frame_uri="f0.jpg",
        width=1920,
        height=1080,
        continuous_time_s=0.0,
    )
    assert row.is_broadcast is None
    assert row.broadcast_confidence is None


def test_ingest_frame_writes_frame_and_detections(store: FeatureStore) -> None:
    n = ingest_frame(
        store,
        clip_id="g1_clip0",
        game_id="g1",
        frame_index=0,
        frame_uri="f0.jpg",
        width=1920,
        height=1080,
        continuous_time_s=0.0,
        half=1,
        classification=BroadcastClassification(
            label=EnumBroadcastClassification.YES, confidence=0.9
        ),
        broadcast_run_id="cls_1",
        detections=_frame_detections(n=3),
        detection_source="yolo",
        detection_run_id="yolo_v3",
    )
    assert n == 3
    assert store.count("frame") == 1
    assert store.count("detection") == 3
    assert (
        bool(store.query("SELECT is_broadcast FROM frame")["is_broadcast"][0]) is True
    )


def test_ingest_frame_is_idempotent(store: FeatureStore) -> None:
    for _ in range(2):
        ingest_frame(
            store,
            clip_id="g1_clip0",
            game_id="g1",
            frame_index=0,
            frame_uri="f0.jpg",
            width=1920,
            height=1080,
            continuous_time_s=0.0,
            detections=_frame_detections(n=2),
            detection_source="yolo",
            detection_run_id="yolo_v3",
        )
    assert store.count("frame") == 1
    assert store.count("detection") == 2


def test_ingest_frame_without_detections(store: FeatureStore) -> None:
    n = ingest_frame(
        store,
        clip_id="g1_clip0",
        game_id="g1",
        frame_index=2,
        frame_uri="f2.jpg",
        width=1920,
        height=1080,
        continuous_time_s=0.08,
    )
    assert n == 0
    assert store.count("frame") == 1
    assert store.count("detection") == 0


def test_ingest_frame_requires_source_with_detections(store: FeatureStore) -> None:
    with pytest.raises(ValueError, match="detection_source and detection_run_id"):
        ingest_frame(
            store,
            clip_id="g1_clip0",
            game_id="g1",
            frame_index=0,
            frame_uri="f0.jpg",
            width=1920,
            height=1080,
            continuous_time_s=0.0,
            detections=_frame_detections(n=1),
        )
