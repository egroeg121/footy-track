"""Shared fixtures for output/ tests.

Generates a minimal in-memory parquet + sidecar JSON that mirrors the
player_tracking_format.md §4.1 schema, without requiring real match data.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

MATCH_ID = "arsenal_bournemouth_1st_half"

# Two frames, three detections total (two players + one ball in frame 0,
# one player in frame 1).
_TRACKS_ROWS = [
    {
        "match_id": MATCH_ID,
        "frame_index": 0,
        "continuous_time_s": 0.0,
        "track_id": 1,
        "label": "player",
        "confidence": 0.92,
        "bbox_x": 0.10,
        "bbox_y": 0.20,
        "bbox_w": 0.05,
        "bbox_h": 0.10,
        "detector_model": "yolo11n.pt@v3",
        "tracker": "bytetrack",
        "is_interpolated": False,
    },
    {
        "match_id": MATCH_ID,
        "frame_index": 0,
        "continuous_time_s": 0.0,
        "track_id": 2,
        "label": "ball",
        "confidence": 0.88,
        "bbox_x": 0.50,
        "bbox_y": 0.50,
        "bbox_w": 0.02,
        "bbox_h": 0.02,
        "detector_model": "yolo11n.pt@v3",
        "tracker": "bytetrack",
        "is_interpolated": False,
    },
    {
        "match_id": MATCH_ID,
        "frame_index": 1,
        "continuous_time_s": 0.04,
        "track_id": 1,
        "label": "player",
        "confidence": 0.90,
        "bbox_x": 0.11,
        "bbox_y": 0.21,
        "bbox_w": 0.05,
        "bbox_h": 0.10,
        "detector_model": "yolo11n.pt@v3",
        "tracker": "bytetrack",
        "is_interpolated": False,
    },
]

_TRACKS_META = {
    "schema_version": "1.0.0",
    "match_id": MATCH_ID,
    "produced_by": {
        "detector": "yolo11n.pt@v3",
        "tracker": "bytetrack",
    },
    "video": {"width": 1920, "height": 1080, "fps": 25.0},
    "tracks": {
        "1": {
            "label": "player",
            "start_frame": 0,
            "end_frame": 1,
            "start_continuous_time_s": 0.0,
            "end_continuous_time_s": 0.04,
            "team_id": None,
            "jersey_number": None,
            "player_id": None,
            "reid_parent_track_id": None,
        },
        "2": {
            "label": "ball",
            "start_frame": 0,
            "end_frame": 0,
            "start_continuous_time_s": 0.0,
            "end_continuous_time_s": 0.0,
            "team_id": None,
            "jersey_number": None,
            "player_id": None,
            "reid_parent_track_id": None,
        },
    },
}


@pytest.fixture()
def match_dir(tmp_path: Path) -> Path:
    """Populate a temporary match directory with tracks.parquet + tracks_meta.json."""
    tracks_dir = tmp_path / "tracks"
    tracks_dir.mkdir()

    df = pd.DataFrame(_TRACKS_ROWS)
    df.to_parquet(tracks_dir / "tracks.parquet", index=False)

    with (tracks_dir / "tracks_meta.json").open("w") as fh:
        json.dump(_TRACKS_META, fh)

    return tmp_path
