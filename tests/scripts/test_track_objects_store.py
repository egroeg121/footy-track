"""Tests for feature-store integration in track_objects pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from footy_track.feature_store import FeatureStore, GameRow
from footy_track.feature_store.ingest import detector_run
from footy_track.scripts.track_objects import _push_tracked_frame_to_store
from footy_track.trackers.base import TrackedDetection


def _tracked_det(
    track_id: int,
    frame_index: int = 0,
    continuous_time_s: float = 0.0,
    label: str = "player",
    confidence: float = 0.9,
) -> TrackedDetection:
    return TrackedDetection(
        track_id=track_id,
        frame_index=frame_index,
        continuous_time_s=continuous_time_s,
        label=label,
        confidence=confidence,
        x=0.4,
        y=0.4,
        w=0.05,
        h=0.1,
        model="yolo",
    )


@pytest.fixture
def store() -> FeatureStore:
    s = FeatureStore.open(":memory:")
    s.upsert_games([GameRow(game_id="g1", fps=25.0)])
    s.upsert_runs([detector_run("bt_run1", "best.pt", source="bytetrack")])
    return s


def test_push_frame_writes_frame_and_detections(store: FeatureStore) -> None:
    tracked = [_tracked_det(1), _tracked_det(2)]
    _push_tracked_frame_to_store(
        store, "bt_run1", "g1", Path("frame_000000.jpg"), tracked, 0, 25.0, 1920, 1080
    )
    assert store.count("frame") == 1
    assert store.count("detection") == 2


def test_push_frame_stores_track_ids(store: FeatureStore) -> None:
    tracked = [_tracked_det(7), _tracked_det(9)]
    _push_tracked_frame_to_store(
        store, "bt_run1", "g1", Path("frame_000000.jpg"), tracked, 0, 25.0, 1920, 1080
    )
    track_ids = sorted(
        store.query("SELECT track_id FROM detection ORDER BY track_id")[
            "track_id"
        ].tolist()
    )
    assert track_ids == [7, 9]


def test_push_empty_frame_writes_frame_only(store: FeatureStore) -> None:
    _push_tracked_frame_to_store(
        store, "bt_run1", "g1", Path("frame_000000.jpg"), [], 0, 25.0, 1920, 1080
    )
    assert store.count("frame") == 1
    assert store.count("detection") == 0


def test_push_is_idempotent(store: FeatureStore) -> None:
    tracked = [_tracked_det(1), _tracked_det(2)]
    for _ in range(2):
        _push_tracked_frame_to_store(
            store,
            "bt_run1",
            "g1",
            Path("frame_000000.jpg"),
            tracked,
            0,
            25.0,
            1920,
            1080,
        )
    assert store.count("frame") == 1
    assert store.count("detection") == 2


def test_push_multiple_frames_accumulates(store: FeatureStore) -> None:
    for frame_idx in range(3):
        tracked = [
            _tracked_det(
                frame_idx + 1, frame_index=frame_idx, continuous_time_s=frame_idx / 25.0
            )
        ]
        _push_tracked_frame_to_store(
            store,
            "bt_run1",
            "g1",
            Path(f"frame_{frame_idx:06d}.jpg"),
            tracked,
            frame_idx,
            25.0,
            1920,
            1080,
        )
    assert store.count("frame") == 3
    assert store.count("detection") == 3
