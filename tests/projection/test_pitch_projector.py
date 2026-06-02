"""Tests for PitchProjector and PitchPosition.

Uses mock homography matrices — no real camera data required.
Covers:
  - project_frame() maps bbox anchor to pitch coordinates
  - NaN propagation when H_quality is below threshold
  - bottom-centre anchor for players
  - centroid anchor for ball
  - project_match() writes pitch_positions.parquet
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from footy_track.constants import BALL_TAG, PLAYER_TAG
from footy_track.projection.projector import (
    H_QUALITY_THRESHOLD,
    PitchPosition,
    PitchProjector,
    TrackedDetection,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


@pytest.fixture
def projector() -> PitchProjector:
    return PitchProjector(pitch_dims=(PITCH_LENGTH, PITCH_WIDTH))


@pytest.fixture
def identity_H() -> np.ndarray:
    """Identity homography — image coords equal pitch coords (in metres)."""
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def scale_H() -> np.ndarray:
    """Homography that scales x by 2 and y by 3 (no translation)."""
    H = np.eye(3, dtype=np.float64)
    H[0, 0] = 2.0
    H[1, 1] = 3.0
    return H


@pytest.fixture
def player_detection() -> TrackedDetection:
    """A player detection with a known bbox."""
    return TrackedDetection(
        track_id=1,
        continuous_time=0.5,
        label=PLAYER_TAG,
        x=0.4,
        y=0.3,
        w=0.05,
        h=0.10,
    )


@pytest.fixture
def ball_detection() -> TrackedDetection:
    """A ball detection with a known bbox."""
    return TrackedDetection(
        track_id=2,
        continuous_time=0.5,
        label=BALL_TAG,
        x=0.48,
        y=0.47,
        w=0.04,
        h=0.04,
    )


# ---------------------------------------------------------------------------
# Anchor point selection
# ---------------------------------------------------------------------------


class TestAnchorPoint:
    def test_player_anchor_is_bottom_centre(self, projector: PitchProjector) -> None:
        det = TrackedDetection(
            track_id=1,
            continuous_time=0.0,
            label=PLAYER_TAG,
            x=0.2,
            y=0.3,
            w=0.06,
            h=0.12,
        )
        ax, ay = projector._anchor(det)
        assert ax == pytest.approx(det.x + det.w / 2.0)
        assert ay == pytest.approx(det.y + det.h)

    def test_ball_anchor_is_centroid(self, projector: PitchProjector) -> None:
        det = TrackedDetection(
            track_id=2,
            continuous_time=0.0,
            label=BALL_TAG,
            x=0.5,
            y=0.5,
            w=0.04,
            h=0.04,
        )
        ax, ay = projector._anchor(det)
        assert ax == pytest.approx(det.x + det.w / 2.0)
        assert ay == pytest.approx(det.y + det.h / 2.0)

    def test_non_ball_class_uses_bottom_centre(self, projector: PitchProjector) -> None:
        """Referee and coach detections should use the bottom-centre anchor."""
        for label in ("referee", "coach", "person"):
            det = TrackedDetection(
                track_id=3,
                continuous_time=0.0,
                label=label,
                x=0.1,
                y=0.1,
                w=0.05,
                h=0.15,
            )
            ax, ay = projector._anchor(det)
            assert ax == pytest.approx(det.x + det.w / 2.0)
            assert ay == pytest.approx(det.y + det.h)

    def test_player_and_ball_anchors_differ_for_same_bbox(
        self, projector: PitchProjector
    ) -> None:
        x, y, w, h = 0.4, 0.3, 0.05, 0.10
        player = TrackedDetection(1, 0.0, PLAYER_TAG, x, y, w, h)
        ball = TrackedDetection(2, 0.0, BALL_TAG, x, y, w, h)
        ax_p, ay_p = projector._anchor(player)
        ax_b, ay_b = projector._anchor(ball)
        assert ax_p == pytest.approx(ax_b)  # same x
        assert ay_p != pytest.approx(ay_b)  # different y


# ---------------------------------------------------------------------------
# project_frame — basic mapping
# ---------------------------------------------------------------------------


class TestProjectFrame:
    def test_returns_one_position_per_detection(
        self,
        projector: PitchProjector,
        identity_H: np.ndarray,
        player_detection: TrackedDetection,
        ball_detection: TrackedDetection,
    ) -> None:
        positions = projector.project_frame(
            [player_detection, ball_detection],
            H=identity_H,
            H_quality=1.0,
        )
        assert len(positions) == 2

    def test_empty_detections_returns_empty(
        self, projector: PitchProjector, identity_H: np.ndarray
    ) -> None:
        positions = projector.project_frame([], H=identity_H, H_quality=1.0)
        assert positions == []

    def test_returns_pitch_position_objects(
        self,
        projector: PitchProjector,
        identity_H: np.ndarray,
        player_detection: TrackedDetection,
    ) -> None:
        positions = projector.project_frame(
            [player_detection], H=identity_H, H_quality=1.0
        )
        assert isinstance(positions[0], PitchPosition)

    def test_track_id_and_time_preserved(
        self,
        projector: PitchProjector,
        identity_H: np.ndarray,
        player_detection: TrackedDetection,
    ) -> None:
        positions = projector.project_frame(
            [player_detection], H=identity_H, H_quality=1.0
        )
        pos = positions[0]
        assert pos.track_id == player_detection.track_id
        assert pos.continuous_time == pytest.approx(player_detection.continuous_time)

    def test_class_label_preserved(
        self,
        projector: PitchProjector,
        identity_H: np.ndarray,
        player_detection: TrackedDetection,
        ball_detection: TrackedDetection,
    ) -> None:
        positions = projector.project_frame(
            [player_detection, ball_detection], H=identity_H, H_quality=1.0
        )
        assert positions[0].class_label == PLAYER_TAG
        assert positions[1].class_label == BALL_TAG


# ---------------------------------------------------------------------------
# project_frame — coordinate mapping via known homographies
# ---------------------------------------------------------------------------


class TestProjectFrameCoordinates:
    def test_player_bottom_centre_anchor_projected(
        self, projector: PitchProjector, identity_H: np.ndarray
    ) -> None:
        """With identity H, pitch coords == anchor image coords."""
        det = TrackedDetection(1, 0.0, PLAYER_TAG, x=0.4, y=0.3, w=0.06, h=0.10)
        expected_ax = det.x + det.w / 2.0  # 0.43
        expected_ay = det.y + det.h  # 0.40

        positions = projector.project_frame([det], H=identity_H, H_quality=1.0)
        pos = positions[0]
        assert pos.x_pitch == pytest.approx(expected_ax, abs=1e-9)
        assert pos.y_pitch == pytest.approx(expected_ay, abs=1e-9)

    def test_ball_centroid_anchor_projected(
        self, projector: PitchProjector, identity_H: np.ndarray
    ) -> None:
        det = TrackedDetection(2, 0.0, BALL_TAG, x=0.48, y=0.47, w=0.04, h=0.04)
        expected_ax = det.x + det.w / 2.0  # 0.50
        expected_ay = det.y + det.h / 2.0  # 0.49

        positions = projector.project_frame([det], H=identity_H, H_quality=1.0)
        pos = positions[0]
        assert pos.x_pitch == pytest.approx(expected_ax, abs=1e-9)
        assert pos.y_pitch == pytest.approx(expected_ay, abs=1e-9)

    def test_scaling_homography_applied_correctly(
        self, projector: PitchProjector, scale_H: np.ndarray
    ) -> None:
        """H that doubles x and triples y."""
        det = TrackedDetection(1, 0.0, PLAYER_TAG, x=0.2, y=0.1, w=0.04, h=0.08)
        ax = det.x + det.w / 2.0  # 0.22
        ay = det.y + det.h  # 0.18

        positions = projector.project_frame([det], H=scale_H, H_quality=1.0)
        pos = positions[0]
        assert pos.x_pitch == pytest.approx(ax * 2.0, abs=1e-9)
        assert pos.y_pitch == pytest.approx(ay * 3.0, abs=1e-9)

    def test_translation_homography(self, projector: PitchProjector) -> None:
        """H with a translation of (+5, -3) metres."""
        H = np.eye(3, dtype=np.float64)
        H[0, 2] = 5.0
        H[1, 2] = -3.0

        det = TrackedDetection(1, 0.0, PLAYER_TAG, x=0.0, y=0.0, w=0.04, h=0.08)
        ax = det.x + det.w / 2.0
        ay = det.y + det.h

        positions = projector.project_frame([det], H=H, H_quality=0.9)
        pos = positions[0]
        assert pos.x_pitch == pytest.approx(ax + 5.0, abs=1e-9)
        assert pos.y_pitch == pytest.approx(ay - 3.0, abs=1e-9)

    def test_normalised_coordinates_consistency(
        self, projector: PitchProjector, identity_H: np.ndarray
    ) -> None:
        """x_pitch_norm and y_pitch_norm are derived from x_pitch / y_pitch."""
        det = TrackedDetection(1, 0.0, PLAYER_TAG, x=0.3, y=0.2, w=0.04, h=0.08)
        positions = projector.project_frame([det], H=identity_H, H_quality=1.0)
        pos = positions[0]

        expected_xn = (pos.x_pitch + PITCH_LENGTH / 2.0) / PITCH_LENGTH
        expected_yn = (pos.y_pitch + PITCH_WIDTH / 2.0) / PITCH_WIDTH
        assert pos.x_pitch_norm == pytest.approx(expected_xn, abs=1e-9)
        assert pos.y_pitch_norm == pytest.approx(expected_yn, abs=1e-9)


# ---------------------------------------------------------------------------
# NaN propagation when H_quality below threshold
# ---------------------------------------------------------------------------


class TestNaNPropagation:
    def test_low_quality_yields_nan_positions(
        self,
        projector: PitchProjector,
        identity_H: np.ndarray,
        player_detection: TrackedDetection,
    ) -> None:
        quality = H_QUALITY_THRESHOLD - 0.01  # just below threshold
        positions = projector.project_frame(
            [player_detection], H=identity_H, H_quality=quality
        )
        pos = positions[0]
        assert math.isnan(pos.x_pitch)
        assert math.isnan(pos.y_pitch)
        assert math.isnan(pos.x_pitch_norm)
        assert math.isnan(pos.y_pitch_norm)

    def test_low_quality_yields_inf_uncertainty(
        self,
        projector: PitchProjector,
        identity_H: np.ndarray,
        player_detection: TrackedDetection,
    ) -> None:
        quality = H_QUALITY_THRESHOLD - 0.01
        positions = projector.project_frame(
            [player_detection], H=identity_H, H_quality=quality
        )
        pos = positions[0]
        assert math.isinf(pos.uncertainty_m)
        assert pos.uncertainty_m > 0

    def test_zero_quality_is_low(
        self,
        projector: PitchProjector,
        identity_H: np.ndarray,
        player_detection: TrackedDetection,
    ) -> None:
        positions = projector.project_frame(
            [player_detection], H=identity_H, H_quality=0.0
        )
        assert math.isnan(positions[0].x_pitch)

    def test_none_quality_is_low(
        self,
        projector: PitchProjector,
        identity_H: np.ndarray,
        player_detection: TrackedDetection,
    ) -> None:
        positions = projector.project_frame(
            [player_detection], H=identity_H, H_quality=None
        )
        assert math.isnan(positions[0].x_pitch)
        assert math.isinf(positions[0].uncertainty_m)

    def test_threshold_boundary_exact(
        self,
        projector: PitchProjector,
        identity_H: np.ndarray,
        player_detection: TrackedDetection,
    ) -> None:
        """Quality exactly at the threshold should produce valid positions."""
        positions = projector.project_frame(
            [player_detection], H=identity_H, H_quality=H_QUALITY_THRESHOLD
        )
        pos = positions[0]
        assert not math.isnan(pos.x_pitch)
        assert not math.isnan(pos.y_pitch)

    def test_quality_just_above_threshold_valid(
        self,
        projector: PitchProjector,
        identity_H: np.ndarray,
        player_detection: TrackedDetection,
    ) -> None:
        positions = projector.project_frame(
            [player_detection], H=identity_H, H_quality=H_QUALITY_THRESHOLD + 0.01
        )
        pos = positions[0]
        assert not math.isnan(pos.x_pitch)
        assert not math.isnan(pos.y_pitch)

    def test_low_quality_preserves_track_id_and_label(
        self,
        projector: PitchProjector,
        identity_H: np.ndarray,
        player_detection: TrackedDetection,
    ) -> None:
        """NaN rows still carry the track identity."""
        quality = H_QUALITY_THRESHOLD - 0.01
        positions = projector.project_frame(
            [player_detection], H=identity_H, H_quality=quality
        )
        pos = positions[0]
        assert pos.track_id == player_detection.track_id
        assert pos.class_label == player_detection.label
        assert pos.continuous_time == pytest.approx(player_detection.continuous_time)

    def test_mixed_quality_frames_independent(
        self,
        projector: PitchProjector,
        identity_H: np.ndarray,
        player_detection: TrackedDetection,
        ball_detection: TrackedDetection,
    ) -> None:
        """All detections in a low-quality frame get NaN, not just some."""
        quality = 0.1  # below threshold
        positions = projector.project_frame(
            [player_detection, ball_detection], H=identity_H, H_quality=quality
        )
        for pos in positions:
            assert math.isnan(pos.x_pitch)
            assert math.isinf(pos.uncertainty_m)

    def test_uncertainty_set_for_valid_frame(
        self,
        projector: PitchProjector,
        identity_H: np.ndarray,
        player_detection: TrackedDetection,
    ) -> None:
        """uncertainty_m is set (not inf) for frames with good calibration."""
        positions = projector.project_frame(
            [player_detection], H=identity_H, H_quality=0.9
        )
        pos = positions[0]
        assert pos.uncertainty_m is not None
        assert not math.isinf(pos.uncertainty_m)


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------


class TestCustomThreshold:
    def test_custom_threshold_respected(self, identity_H: np.ndarray) -> None:
        strict = PitchProjector(h_quality_threshold=0.8)
        det = TrackedDetection(1, 0.0, PLAYER_TAG, x=0.3, y=0.3, w=0.04, h=0.08)

        # quality = 0.5 is fine for default threshold but fails strict
        positions = strict.project_frame([det], H=identity_H, H_quality=0.5)
        assert math.isnan(positions[0].x_pitch)

        # quality = 0.9 passes even strict
        positions = strict.project_frame([det], H=identity_H, H_quality=0.9)
        assert not math.isnan(positions[0].x_pitch)


# ---------------------------------------------------------------------------
# project_match — writes pitch_positions.parquet
# ---------------------------------------------------------------------------


class TestProjectMatch:
    def _make_tracks_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "frame_index": 0,
                    "continuous_time_s": 0.0,
                    "track_id": 1,
                    "label": PLAYER_TAG,
                    "confidence": 0.9,
                    "bbox_x": 0.4,
                    "bbox_y": 0.3,
                    "bbox_w": 0.05,
                    "bbox_h": 0.10,
                },
                {
                    "frame_index": 1,
                    "continuous_time_s": 0.04,
                    "track_id": 2,
                    "label": BALL_TAG,
                    "confidence": 0.8,
                    "bbox_x": 0.48,
                    "bbox_y": 0.47,
                    "bbox_w": 0.04,
                    "bbox_h": 0.04,
                },
            ]
        )

    def _make_geometry_df(self) -> pd.DataFrame:
        H_flat = list(np.eye(3).flatten())
        return pd.DataFrame(
            [
                {"frame_index": 0, "H_flat": H_flat, "H_quality": 0.9},
                {"frame_index": 1, "H_flat": H_flat, "H_quality": 0.9},
            ]
        )

    def test_writes_parquet_file(self, tmp_path: Path) -> None:
        projector = PitchProjector()
        tracks = tmp_path / "tracks.parquet"
        geometry = tmp_path / "geometry.parquet"
        out = tmp_path / "pitch_positions.parquet"

        self._make_tracks_df().to_parquet(tracks, index=False)
        self._make_geometry_df().to_parquet(geometry, index=False)

        projector.project_match(tracks, geometry, out)
        assert out.exists()

    def test_output_row_count_matches_input(self, tmp_path: Path) -> None:
        projector = PitchProjector()
        tracks = tmp_path / "tracks.parquet"
        geometry = tmp_path / "geometry.parquet"
        out = tmp_path / "pitch_positions.parquet"

        tracks_df = self._make_tracks_df()
        self._make_tracks_df().to_parquet(tracks, index=False)
        self._make_geometry_df().to_parquet(geometry, index=False)

        projector.project_match(tracks, geometry, out)
        result = pd.read_parquet(out)
        assert len(result) == len(tracks_df)

    def test_output_has_required_columns(self, tmp_path: Path) -> None:
        projector = PitchProjector()
        tracks = tmp_path / "tracks.parquet"
        geometry = tmp_path / "geometry.parquet"
        out = tmp_path / "pitch_positions.parquet"

        self._make_tracks_df().to_parquet(tracks, index=False)
        self._make_geometry_df().to_parquet(geometry, index=False)

        projector.project_match(tracks, geometry, out)
        result = pd.read_parquet(out)

        expected_cols = {
            "track_id",
            "continuous_time",
            "x_pitch",
            "y_pitch",
            "x_pitch_norm",
            "y_pitch_norm",
            "uncertainty_m",
            "class_label",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_low_quality_rows_have_nan_pitch_coords(self, tmp_path: Path) -> None:
        projector = PitchProjector()
        tracks = tmp_path / "tracks.parquet"
        geometry = tmp_path / "geometry.parquet"
        out = tmp_path / "pitch_positions.parquet"

        H_flat = list(np.eye(3).flatten())
        # Both frames get low quality
        geo_df = pd.DataFrame(
            [
                {"frame_index": 0, "H_flat": H_flat, "H_quality": 0.1},
                {"frame_index": 1, "H_flat": H_flat, "H_quality": 0.1},
            ]
        )

        self._make_tracks_df().to_parquet(tracks, index=False)
        geo_df.to_parquet(geometry, index=False)

        projector.project_match(tracks, geometry, out)
        result = pd.read_parquet(out)
        assert result["x_pitch"].isna().all()
        assert result["y_pitch"].isna().all()

    def test_player_and_ball_positions_differ(self, tmp_path: Path) -> None:
        """Player uses bottom-centre, ball uses centroid — coords differ for same bbox."""
        projector = PitchProjector()

        shared_bbox = {"bbox_x": 0.4, "bbox_y": 0.3, "bbox_w": 0.05, "bbox_h": 0.10}
        tracks_df = pd.DataFrame(
            [
                {
                    "frame_index": 0,
                    "continuous_time_s": 0.0,
                    "track_id": 1,
                    "label": PLAYER_TAG,
                    "confidence": 0.9,
                    **shared_bbox,
                },
                {
                    "frame_index": 0,
                    "continuous_time_s": 0.0,
                    "track_id": 2,
                    "label": BALL_TAG,
                    "confidence": 0.8,
                    **shared_bbox,
                },
            ]
        )
        H_flat = list(np.eye(3).flatten())
        geo_df = pd.DataFrame([{"frame_index": 0, "H_flat": H_flat, "H_quality": 1.0}])

        tracks = tmp_path / "tracks.parquet"
        geometry = tmp_path / "geometry.parquet"
        out = tmp_path / "pitch_positions.parquet"
        tracks_df.to_parquet(tracks, index=False)
        geo_df.to_parquet(geometry, index=False)

        projector.project_match(tracks, geometry, out)
        result = pd.read_parquet(out)

        player_row = result[result["class_label"] == PLAYER_TAG].iloc[0]
        ball_row = result[result["class_label"] == BALL_TAG].iloc[0]

        # x_pitch (bottom-centre x == centroid x for same bbox) should match
        assert player_row["x_pitch"] == pytest.approx(ball_row["x_pitch"], abs=1e-9)
        # y_pitch differs: player uses bottom (y+h), ball uses centroid (y+h/2)
        assert player_row["y_pitch"] != pytest.approx(ball_row["y_pitch"])

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        projector = PitchProjector()
        tracks = tmp_path / "tracks.parquet"
        geometry = tmp_path / "geometry.parquet"
        out = tmp_path / "nested" / "deep" / "pitch_positions.parquet"

        self._make_tracks_df().to_parquet(tracks, index=False)
        self._make_geometry_df().to_parquet(geometry, index=False)

        projector.project_match(tracks, geometry, out)
        assert out.exists()
