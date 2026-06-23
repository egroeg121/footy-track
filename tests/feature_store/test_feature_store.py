"""Tests for the footy-track feature store."""

from __future__ import annotations

import pytest

from footy_track.feature_store import (
    DetectionRow,
    FeatureStore,
    FrameRow,
    GameRow,
    Point,
    RunRow,
    TrackMetaRow,
)


@pytest.fixture
def store() -> FeatureStore:
    s = FeatureStore.open(":memory:")
    s.upsert_games(
        [
            GameRow(
                game_id="g1",
                home_team="Arsenal",
                away_team="ManCity",
                half2_start_continuous_s=2910.0,
            )
        ]
    )
    s.upsert_runs(
        [
            RunRow(
                run_id="yolo_v3",
                stage="detection",
                source="yolo",
                model_name="yolo11n.pt",
                model_version="v3",
            ),
            RunRow(
                run_id="human_1",
                stage="detection",
                source="hand_label",
                model_name="human",
            ),
            RunRow(
                run_id="bt_1",
                stage="tracking",
                source="bytetrack",
                model_name="bytetrack",
            ),
        ]
    )
    s.upsert_frames(
        [
            FrameRow(
                game_id="g1",
                frame_index=0,
                frame_uri="f0.jpg",
                width=1920,
                height=1080,
                continuous_time_s=0.0,
                half=1,
                is_broadcast=True,
                broadcast_confidence=0.97,
                pitch_polygon=[
                    Point(x=0.1, y=0.1),
                    Point(x=0.9, y=0.1),
                    Point(x=0.5, y=0.9),
                ],
                pitch_seg_threshold=0.3,
                homography=[1, 0, 0, 0, 1, 0, 0, 0, 1],
            ),
            FrameRow(
                game_id="g1",
                frame_index=1,
                frame_uri="f1.jpg",
                width=1920,
                height=1080,
                continuous_time_s=0.04,
                half=1,
                is_broadcast=False,
            ),
        ]
    )
    return s


def _det(
    frame: int,
    did: int,
    source: str,
    run: str,
    *,
    track: int | None = None,
    conf: float = 0.9,
) -> DetectionRow:
    return DetectionRow(
        game_id="g1",
        frame_index=frame,
        continuous_time_s=frame * 0.04,
        detection_id=did,
        source=source,
        run_id=run,
        label="player",
        confidence=conf,
        bbox_x=0.4,
        bbox_y=0.4,
        bbox_w=0.05,
        bbox_h=0.1,
        track_id=track,
    )


def test_schema_and_views_created(store: FeatureStore) -> None:
    # views resolve
    assert "game_id" in store.query("SELECT * FROM frame_features LIMIT 0").columns
    assert (
        "model_name" in store.query("SELECT * FROM detections_enriched LIMIT 0").columns
    )
    assert "team_id" in store.query("SELECT * FROM tracks_enriched LIMIT 0").columns


def test_non_broadcast_frame_is_recorded(store: FeatureStore) -> None:
    # invariant #5: non-broadcast frames recorded, not dropped
    assert store.count("frame") == 2
    row = store.query("SELECT is_broadcast FROM frame WHERE frame_index = 1")
    assert bool(row["is_broadcast"][0]) is False


def test_polygon_and_homography_roundtrip(store: FeatureStore) -> None:
    poly = store.query("SELECT pitch_polygon FROM frame WHERE frame_index = 0")[
        "pitch_polygon"
    ][0]
    assert [dict(p) for p in poly] == [
        {"x": 0.1, "y": 0.1},
        {"x": 0.9, "y": 0.1},
        {"x": 0.5, "y": 0.9},
    ]
    h = store.query("SELECT homography FROM frame WHERE frame_index = 0")["homography"][
        0
    ]
    assert list(h) == [1, 0, 0, 0, 1, 0, 0, 0, 1]


def test_multiple_sources_coexist_on_same_frame(store: FeatureStore) -> None:
    # hand_label and yolo over the same frame+object index -> distinct rows
    store.upsert_detections(
        [
            _det(0, 0, "yolo", "yolo_v3"),
            _det(0, 0, "hand_label", "human_1"),
        ]
    )
    assert store.count("detection") == 2
    sources = set(
        store.query("SELECT DISTINCT source FROM detection WHERE frame_index = 0")[
            "source"
        ]
    )
    assert sources == {"yolo", "hand_label"}


def test_detection_id_is_per_frame(store: FeatureStore) -> None:
    # same detection_id on different frames within one run must NOT collide
    store.upsert_detections(
        [_det(0, 0, "yolo", "yolo_v3"), _det(1, 0, "yolo", "yolo_v3")]
    )
    assert store.count("detection") == 2


def test_upsert_is_idempotent_and_updates(store: FeatureStore) -> None:
    store.upsert_detections([_det(0, 0, "yolo", "yolo_v3", conf=0.5)])
    assert store.count("detection") == 1
    # re-ingest same key with new confidence: no new row, value updated
    store.upsert_detections([_det(0, 0, "yolo", "yolo_v3", conf=0.99)])
    assert store.count("detection") == 1
    conf = store.query("SELECT confidence FROM detection WHERE frame_index = 0")[
        "confidence"
    ][0]
    assert conf == pytest.approx(0.99, abs=1e-5)


def test_re_run_does_not_disturb_other_sources(store: FeatureStore) -> None:
    store.upsert_detections(
        [_det(0, 0, "yolo", "yolo_v3"), _det(0, 0, "hand_label", "human_1")]
    )
    # re-run yolo only
    store.upsert_detections([_det(0, 0, "yolo", "yolo_v3", conf=0.1)])
    assert store.count("detection") == 2  # hand_label untouched
    assert store.count("detection") == store.count("detection")


def test_player_trajectory_across_game(store: FeatureStore) -> None:
    store.upsert_detections(
        [
            _det(0, 0, "yolo", "yolo_v3", track=7),
            _det(1, 0, "yolo", "yolo_v3", track=7),
        ]
    )
    store.upsert_track_meta(
        [
            TrackMetaRow(
                game_id="g1",
                source="yolo",
                run_id="yolo_v3",
                track_id=7,
                label="player",
                start_frame=0,
                end_frame=1,
                start_continuous_time_s=0.0,
                end_continuous_time_s=0.04,
                team_id="home",
                jersey_number=10,
            )
        ]
    )
    traj = store.player_trajectory("g1", track_id=7, source="yolo")
    assert len(traj) == 2
    # ordered by time, identity resolved from track_meta
    assert list(traj["continuous_time_s"]) == [0.0, 0.04]
    assert set(traj["jersey_number"]) == {10}


def test_parquet_export_and_rebuild(store: FeatureStore, tmp_path) -> None:
    store.upsert_detections(
        [_det(0, 0, "yolo", "yolo_v3", track=7), _det(0, 1, "yolo", "yolo_v3")]
    )
    out = tmp_path / "export"
    written = store.export_parquet(out)
    assert set(written) == {"game", "run", "frame", "detection"}

    rebuilt = FeatureStore.from_parquet(out)
    assert rebuilt.count("detection") == store.count("detection")
    assert rebuilt.count("frame") == store.count("frame")
    # polygon survives the parquet roundtrip
    poly = rebuilt.query("SELECT pitch_polygon FROM frame WHERE frame_index = 0")[
        "pitch_polygon"
    ][0]
    assert len(poly) == 3


def test_frame_features_joins_game_metadata(store: FeatureStore) -> None:
    df = store.query(
        "SELECT home_team, half2_start_continuous_s FROM frame_features WHERE frame_index = 0"
    )
    assert df["home_team"][0] == "Arsenal"
    assert df["half2_start_continuous_s"][0] == pytest.approx(2910.0)


def _ball_det(
    frame: int,
    source: str,
    run: str,
    *,
    label: str = "ball",
    conf: float = 0.8,
) -> DetectionRow:
    return DetectionRow(
        game_id="g1",
        frame_index=frame,
        continuous_time_s=frame * 0.04,
        detection_id=0,
        source=source,
        run_id=run,
        label=label,
        confidence=conf,
        bbox_x=0.4,
        bbox_y=0.4,
        bbox_w=0.02,
        bbox_h=0.02,
    )


def test_ball_trajectory_returns_ball_labels(store: FeatureStore) -> None:
    store.upsert_runs(
        [
            RunRow(
                run_id="roi_r1",
                stage="detection",
                source="roi_yolo",
                model_name="yolo11s",
            )
        ]
    )
    store.upsert_detections(
        [
            _ball_det(0, "roi_yolo", "roi_r1", label="ball"),
            _ball_det(1, "roi_yolo", "roi_r1", label="in_play_ball"),
            # player on same run — must not appear in ball_trajectory
            _det(0, 1, "roi_yolo", "roi_r1"),
        ]
    )
    traj = store.ball_trajectory("g1", source="roi_yolo")
    assert len(traj) == 2
    assert set(traj["label"]) <= {"ball", "in_play_ball", "out_of_play_ball"}
    assert list(traj["continuous_time_s"]) == [0.0, 0.04]


def test_ball_trajectory_filters_by_run_id(store: FeatureStore) -> None:
    store.upsert_runs(
        [
            RunRow(
                run_id="roi_r1", stage="detection", source="roi_yolo", model_name="m"
            ),
            RunRow(
                run_id="roi_r2", stage="detection", source="roi_yolo", model_name="m"
            ),
        ]
    )
    store.upsert_detections(
        [
            _ball_det(0, "roi_yolo", "roi_r1"),
            _ball_det(1, "roi_yolo", "roi_r2"),
        ]
    )
    traj = store.ball_trajectory("g1", source="roi_yolo", run_id="roi_r1")
    assert len(traj) == 1
    assert traj["frame_index"][0] == 0


def test_ball_trajectory_custom_labels(store: FeatureStore) -> None:
    store.upsert_runs(
        [RunRow(run_id="roi_r1", stage="detection", source="roi_yolo", model_name="m")]
    )
    store.upsert_detections(
        [
            DetectionRow(
                game_id="g1",
                frame_index=0,
                continuous_time_s=0.0,
                detection_id=0,
                source="roi_yolo",
                run_id="roi_r1",
                label="sports_ball",
                confidence=0.7,
                bbox_x=0.3,
                bbox_y=0.3,
                bbox_w=0.01,
                bbox_h=0.01,
            )
        ]
    )
    traj = store.ball_trajectory("g1", source="roi_yolo", labels=("sports_ball",))
    assert len(traj) == 1
    assert traj["label"][0] == "sports_ball"
