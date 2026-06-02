"""Tests for MatchExporter (to_json, to_csv, to_fiftyone).

Covers:
- Correct JSON schema (top-level keys, frame ordering, schema_version)
- Idempotent JSON export (second write overwrites, not appends)
- CSV columns match spec (docs/design/output.md §4)
- Idempotent CSV export
- FiftyOne samples created correctly (one per frame, detections present)
- Idempotent FiftyOne export (overwrite=True)
"""

import json
import uuid
from pathlib import Path

import fiftyone as fo
import pandas as pd

from footy_track.output.exporters import SCHEMA_VERSION, MatchExporter
from tests.output.conftest import MATCH_ID

# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------


class TestToJson:
    def test_top_level_keys(self, match_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        MatchExporter(match_dir).to_json(out)
        data = json.loads(out.read_text())
        assert set(data.keys()) == {
            "match_id",
            "schema_version",
            "frames",
            "tracks_meta",
        }

    def test_match_id(self, match_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        MatchExporter(match_dir).to_json(out)
        data = json.loads(out.read_text())
        assert data["match_id"] == MATCH_ID

    def test_schema_version(self, match_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        MatchExporter(match_dir).to_json(out)
        data = json.loads(out.read_text())
        assert data["schema_version"] == SCHEMA_VERSION

    def test_frame_count(self, match_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        MatchExporter(match_dir).to_json(out)
        data = json.loads(out.read_text())
        # Fixture has 2 unique continuous_time_s values → 2 frames
        assert len(data["frames"]) == 2

    def test_frames_ordered_by_continuous_time(
        self, match_dir: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.json"
        MatchExporter(match_dir).to_json(out)
        times = [f["continuous_time"] for f in json.loads(out.read_text())["frames"]]
        assert times == sorted(times)

    def test_frame_has_required_keys(self, match_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        MatchExporter(match_dir).to_json(out)
        frame = json.loads(out.read_text())["frames"][0]
        assert "continuous_time" in frame
        assert "detections" in frame
        assert "tracks" in frame

    def test_detection_bbox_keys(self, match_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        MatchExporter(match_dir).to_json(out)
        det = json.loads(out.read_text())["frames"][0]["detections"][0]
        assert set(det["bbox"].keys()) == {"x", "y", "w", "h"}

    def test_tracks_meta_populated(self, match_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        MatchExporter(match_dir).to_json(out)
        data = json.loads(out.read_text())
        assert len(data["tracks_meta"]) == 2

    def test_idempotent(self, match_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        exporter = MatchExporter(match_dir)
        exporter.to_json(out)
        first = out.read_text()
        exporter.to_json(out)
        second = out.read_text()
        assert first == second

    def test_creates_parent_dirs(self, match_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "deep" / "out.json"
        MatchExporter(match_dir).to_json(out)
        assert out.exists()


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------

_EXPECTED_DETECTIONS_COLS = {
    "match_id",
    "frame_index",
    "continuous_time_s",
    "track_id",
    "label",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "detector_model",
    "tracker",
    "is_interpolated",
}

_EXPECTED_TRACKS_COLS = {
    "track_id",
    "frame_index",
    "continuous_time_s",
    "label",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
}

_EXPECTED_TRACKS_META_COLS = {
    "track_id",
    "label",
    "start_frame",
    "end_frame",
    "start_continuous_time_s",
    "end_continuous_time_s",
    "team_id",
    "jersey_number",
    "player_id",
    "reid_parent_track_id",
}


class TestToCsv:
    def test_detections_csv_exists(self, match_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "csv"
        MatchExporter(match_dir).to_csv(out_dir)
        assert (out_dir / "detections.csv").exists()

    def test_tracks_csv_exists(self, match_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "csv"
        MatchExporter(match_dir).to_csv(out_dir)
        assert (out_dir / "tracks.csv").exists()

    def test_tracks_meta_csv_exists(self, match_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "csv"
        MatchExporter(match_dir).to_csv(out_dir)
        assert (out_dir / "tracks_meta.csv").exists()

    def test_detections_columns(self, match_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "csv"
        MatchExporter(match_dir).to_csv(out_dir)
        cols = set(pd.read_csv(out_dir / "detections.csv").columns)
        assert _EXPECTED_DETECTIONS_COLS.issubset(cols)

    def test_tracks_columns(self, match_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "csv"
        MatchExporter(match_dir).to_csv(out_dir)
        cols = set(pd.read_csv(out_dir / "tracks.csv").columns)
        assert _EXPECTED_TRACKS_COLS.issubset(cols)

    def test_tracks_meta_columns(self, match_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "csv"
        MatchExporter(match_dir).to_csv(out_dir)
        cols = set(pd.read_csv(out_dir / "tracks_meta.csv").columns)
        assert _EXPECTED_TRACKS_META_COLS.issubset(cols)

    def test_detections_row_count(self, match_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "csv"
        MatchExporter(match_dir).to_csv(out_dir)
        df = pd.read_csv(out_dir / "detections.csv")
        assert len(df) == 3  # fixture has 3 detection rows

    def test_tracks_meta_row_count(self, match_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "csv"
        MatchExporter(match_dir).to_csv(out_dir)
        df = pd.read_csv(out_dir / "tracks_meta.csv")
        assert len(df) == 2  # two tracks in fixture

    def test_pitch_positions_absent_without_columns(
        self, match_dir: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "csv"
        MatchExporter(match_dir).to_csv(out_dir)
        assert not (out_dir / "pitch_positions.csv").exists()

    def test_idempotent(self, match_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "csv"
        exporter = MatchExporter(match_dir)
        exporter.to_csv(out_dir)
        first = (out_dir / "detections.csv").read_text()
        exporter.to_csv(out_dir)
        second = (out_dir / "detections.csv").read_text()
        assert first == second

    def test_creates_out_dir(self, match_dir: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "nested" / "csv"
        MatchExporter(match_dir).to_csv(out_dir)
        assert out_dir.is_dir()


# ---------------------------------------------------------------------------
# to_fiftyone
# ---------------------------------------------------------------------------


class TestToFiftyOne:
    def _unique_name(self, suffix: str = "") -> str:
        return f"test_footy_track_{uuid.uuid4().hex[:8]}{suffix}"

    def test_sample_count_equals_frame_count(self, match_dir: Path) -> None:
        name = self._unique_name()
        try:
            ds = MatchExporter(match_dir).to_fiftyone(name)
            assert len(ds) == 2  # two unique continuous_time_s values
        finally:
            if fo.dataset_exists(name):
                fo.delete_dataset(name)

    def test_samples_have_tracks_field(self, match_dir: Path) -> None:
        name = self._unique_name()
        try:
            ds = MatchExporter(match_dir).to_fiftyone(name)
            sample = ds.first()
            assert "tracks" in sample
            assert isinstance(sample["tracks"], fo.Detections)
        finally:
            if fo.dataset_exists(name):
                fo.delete_dataset(name)

    def test_samples_have_continuous_time_field(self, match_dir: Path) -> None:
        name = self._unique_name()
        try:
            ds = MatchExporter(match_dir).to_fiftyone(name)
            sample = ds.first()
            assert "continuous_time" in sample
            assert isinstance(sample["continuous_time"], float)
        finally:
            if fo.dataset_exists(name):
                fo.delete_dataset(name)

    def test_detections_per_frame(self, match_dir: Path) -> None:
        name = self._unique_name()
        try:
            ds = MatchExporter(match_dir).to_fiftyone(name)
            # Frame at t=0 has 2 detections (player + ball)
            samples_by_time = {s["continuous_time"]: s for s in ds}
            frame0 = samples_by_time[0.0]
            assert len(frame0["tracks"].detections) == 2
        finally:
            if fo.dataset_exists(name):
                fo.delete_dataset(name)

    def test_detection_has_track_id(self, match_dir: Path) -> None:
        name = self._unique_name()
        try:
            ds = MatchExporter(match_dir).to_fiftyone(name)
            det = ds.first()["tracks"].detections[0]
            assert hasattr(det, "track_id")
            assert isinstance(det.track_id, int)
        finally:
            if fo.dataset_exists(name):
                fo.delete_dataset(name)

    def test_detection_bounding_box_normalised(self, match_dir: Path) -> None:
        name = self._unique_name()
        try:
            ds = MatchExporter(match_dir).to_fiftyone(name)
            for sample in ds:
                for det in sample["tracks"].detections:
                    x, y, w, h = det.bounding_box
                    assert 0.0 <= x <= 1.0
                    assert 0.0 <= y <= 1.0
                    assert 0.0 <= w <= 1.0
                    assert 0.0 <= h <= 1.0
        finally:
            if fo.dataset_exists(name):
                fo.delete_dataset(name)

    def test_idempotent_overwrites_dataset(self, match_dir: Path) -> None:
        name = self._unique_name()
        try:
            exporter = MatchExporter(match_dir)
            ds1 = exporter.to_fiftyone(name)
            count1 = len(ds1)
            ds2 = exporter.to_fiftyone(name)
            count2 = len(ds2)
            assert count1 == count2
            assert fo.dataset_exists(name)
        finally:
            if fo.dataset_exists(name):
                fo.delete_dataset(name)
