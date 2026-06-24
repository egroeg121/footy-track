"""Tests for the Calibrator (manual-keypoint image->pitch homography).

Strategy: build a *known* pitch->image homography, invert it to synthesise
image points for a set of pitch landmarks, then assert the Calibrator
recovers an H that round-trips those landmarks back to pitch metres with
near-zero error and high H_quality. Degenerate inputs are checked too.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from footy_track.constants import PLAYER_TAG
from footy_track.projection.calibrator import (
    MIN_CORRESPONDENCES,
    CalibrationResult,
    Calibrator,
    KeypointLabel,
)
from footy_track.projection.pitch_model import PitchModel
from footy_track.projection.projector import PitchProjector

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


@pytest.fixture
def pitch() -> PitchModel:
    return PitchModel(length_m=PITCH_LENGTH, width_m=PITCH_WIDTH)


@pytest.fixture
def calibrator() -> Calibrator:
    return Calibrator(pitch_dims=(PITCH_LENGTH, PITCH_WIDTH))


@pytest.fixture
def pitch_to_image_H(pitch: PitchModel) -> np.ndarray:
    """A plausible pitch(m)->image(normalised) homography.

    Maps the pitch extents (~[-52.5, 52.5] x [-34, 34]) into the unit image
    square with a mild perspective term so it isn't a pure affine map.
    """
    H = np.array(
        [
            [1.0 / PITCH_LENGTH, 0.0, 0.5],
            [0.0, 1.0 / PITCH_WIDTH, 0.5],
            [0.0008, 0.0006, 1.0],
        ],
        dtype=np.float64,
    )
    return H


def _image_point_for(
    pitch_to_image_H: np.ndarray, pitch_xy: tuple[float, float]
) -> tuple[float, float]:
    """Project a pitch (x, y) into image coordinates via the known H."""
    src = np.array([[pitch_xy]], dtype=np.float64)  # shape (1, 1, 2)
    dst = cv2.perspectiveTransform(src, pitch_to_image_H)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


def _labels_from_landmarks(
    pitch: PitchModel,
    pitch_to_image_H: np.ndarray,
    names: list[str],
    frame_index: int = 0,
) -> list[KeypointLabel]:
    labels = []
    for name in names:
        px, py = pitch.landmark(name)
        ix, iy = _image_point_for(pitch_to_image_H, (px, py))
        labels.append(
            KeypointLabel(
                frame_index=frame_index,
                landmark=name,
                image_x=ix,
                image_y=iy,
            )
        )
    return labels


# Well-spread landmarks (avoid collinear sets that degrade homography fits).
GOOD_LANDMARKS = [
    "corner_left_top",
    "corner_right_top",
    "corner_right_bottom",
    "corner_left_bottom",
    "centre_spot",
    "penalty_spot_left",
]


class TestCalibrateFrameRecovery:
    def test_recovers_known_homography(
        self,
        calibrator: Calibrator,
        pitch: PitchModel,
        pitch_to_image_H: np.ndarray,
    ) -> None:
        labels = _labels_from_landmarks(pitch, pitch_to_image_H, GOOD_LANDMARKS)
        result = calibrator.calibrate_frame(0, labels)

        assert result.H is not None
        # Round-trip every landmark's image point back to pitch metres.
        for lbl in labels:
            img = np.array([[[lbl.image_x, lbl.image_y]]], dtype=np.float64)
            recovered = cv2.perspectiveTransform(img, result.H)[0, 0]
            expected = pitch.landmark(lbl.landmark)
            assert recovered[0] == pytest.approx(expected[0], abs=1e-3)
            assert recovered[1] == pytest.approx(expected[1], abs=1e-3)

    def test_perfect_fit_high_quality(
        self,
        calibrator: Calibrator,
        pitch: PitchModel,
        pitch_to_image_H: np.ndarray,
    ) -> None:
        labels = _labels_from_landmarks(pitch, pitch_to_image_H, GOOD_LANDMARKS)
        result = calibrator.calibrate_frame(0, labels)
        assert result.H_quality > 0.99
        assert result.mean_reproj_error_m < 0.1

    def test_inliers_counted(
        self,
        calibrator: Calibrator,
        pitch: PitchModel,
        pitch_to_image_H: np.ndarray,
    ) -> None:
        labels = _labels_from_landmarks(pitch, pitch_to_image_H, GOOD_LANDMARKS)
        result = calibrator.calibrate_frame(0, labels)
        assert result.n_correspondences == len(GOOD_LANDMARKS)
        assert result.n_inliers == len(GOOD_LANDMARKS)


class TestQualityDegradesWithNoise:
    def test_noisy_labels_lower_quality(
        self,
        calibrator: Calibrator,
        pitch: PitchModel,
        pitch_to_image_H: np.ndarray,
    ) -> None:
        clean = _labels_from_landmarks(pitch, pitch_to_image_H, GOOD_LANDMARKS)
        clean_q = calibrator.calibrate_frame(0, clean).H_quality

        # Perturb one label by a fixed, deterministic offset.
        noisy = list(clean)
        noisy[0] = KeypointLabel(
            frame_index=0,
            landmark=noisy[0].landmark,
            image_x=noisy[0].image_x + 0.05,
            image_y=noisy[0].image_y - 0.05,
        )
        noisy_q = calibrator.calibrate_frame(0, noisy).H_quality
        assert noisy_q < clean_q

    def test_quality_in_unit_range(
        self,
        calibrator: Calibrator,
        pitch: PitchModel,
        pitch_to_image_H: np.ndarray,
    ) -> None:
        labels = _labels_from_landmarks(pitch, pitch_to_image_H, GOOD_LANDMARKS)
        q = calibrator.calibrate_frame(0, labels).H_quality
        assert 0.0 <= q <= 1.0


class TestDegenerateInputs:
    def test_too_few_correspondences(self, calibrator: Calibrator) -> None:
        labels = [
            KeypointLabel(0, "centre_spot", 0.5, 0.5),
            KeypointLabel(0, "corner_left_top", 0.1, 0.1),
            KeypointLabel(0, "corner_right_top", 0.9, 0.1),
        ]
        assert len(labels) < MIN_CORRESPONDENCES
        result = calibrator.calibrate_frame(0, labels)
        assert result.H is None
        assert result.H_quality == 0.0
        assert math.isinf(result.mean_reproj_error_m)

    def test_empty_labels(self, calibrator: Calibrator) -> None:
        result = calibrator.calibrate_frame(0, [])
        assert result.H is None
        assert result.H_quality == 0.0

    def test_collinear_points_low_quality_or_none(self, calibrator: Calibrator) -> None:
        """Degenerate (collinear-ish) image points must not score confidently."""
        labels = [
            KeypointLabel(0, "corner_left_top", 0.1, 0.5),
            KeypointLabel(0, "centre_spot", 0.4, 0.5),
            KeypointLabel(0, "corner_right_top", 0.7, 0.5),
            KeypointLabel(0, "penalty_spot_left", 0.2, 0.5),
        ]
        result = calibrator.calibrate_frame(0, labels)
        # Either unsolved, or solved but with poor quality — never confident.
        assert result.H is None or result.H_quality < 0.99


class TestCalibrateMatch:
    def _write_labels(
        self,
        path: Path,
        pitch: PitchModel,
        pitch_to_image_H: np.ndarray,
        frame_indices: list[int],
    ) -> None:
        rows = []
        for fi in frame_indices:
            for lbl in _labels_from_landmarks(
                pitch, pitch_to_image_H, GOOD_LANDMARKS, frame_index=fi
            ):
                rows.append(
                    {
                        "frame_index": lbl.frame_index,
                        "landmark": lbl.landmark,
                        "image_x": lbl.image_x,
                        "image_y": lbl.image_y,
                    }
                )
        pd.DataFrame(rows).to_parquet(path, index=False)

    def test_writes_geometry_parquet(
        self,
        calibrator: Calibrator,
        pitch: PitchModel,
        pitch_to_image_H: np.ndarray,
        tmp_path: Path,
    ) -> None:
        labels = tmp_path / "labels.parquet"
        out = tmp_path / "geometry.parquet"
        self._write_labels(labels, pitch, pitch_to_image_H, [0, 1, 2])

        calibrator.calibrate_match(labels, out)
        assert out.exists()

    def test_geometry_schema_matches_projector(
        self,
        calibrator: Calibrator,
        pitch: PitchModel,
        pitch_to_image_H: np.ndarray,
        tmp_path: Path,
    ) -> None:
        labels = tmp_path / "labels.parquet"
        out = tmp_path / "geometry.parquet"
        self._write_labels(labels, pitch, pitch_to_image_H, [0, 1])

        calibrator.calibrate_match(labels, out)
        geo = pd.read_parquet(out)

        # Exactly the columns PitchProjector.project_match reads.
        assert {"frame_index", "H_flat", "H_quality"}.issubset(geo.columns)
        # H_flat must be 9 floats per row.
        for h_flat in geo["H_flat"]:
            assert len(h_flat) == 9

    def test_one_row_per_frame(
        self,
        calibrator: Calibrator,
        pitch: PitchModel,
        pitch_to_image_H: np.ndarray,
        tmp_path: Path,
    ) -> None:
        labels = tmp_path / "labels.parquet"
        out = tmp_path / "geometry.parquet"
        self._write_labels(labels, pitch, pitch_to_image_H, [0, 1, 2, 5])

        calibrator.calibrate_match(labels, out)
        geo = pd.read_parquet(out)
        assert sorted(geo["frame_index"].tolist()) == [0, 1, 2, 5]

    def test_unsolvable_frame_emitted_with_identity_and_zero_quality(
        self,
        calibrator: Calibrator,
        pitch: PitchModel,
        pitch_to_image_H: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """A frame with too few labels is still recorded (not dropped)."""
        rows = []
        # Frame 0: good. Frame 1: only 2 labels (unsolvable).
        for lbl in _labels_from_landmarks(
            pitch, pitch_to_image_H, GOOD_LANDMARKS, frame_index=0
        ):
            rows.append(
                {
                    "frame_index": 0,
                    "landmark": lbl.landmark,
                    "image_x": lbl.image_x,
                    "image_y": lbl.image_y,
                }
            )
        rows.append(
            {
                "frame_index": 1,
                "landmark": "centre_spot",
                "image_x": 0.5,
                "image_y": 0.5,
            }
        )
        rows.append(
            {
                "frame_index": 1,
                "landmark": "halfway_top",
                "image_x": 0.5,
                "image_y": 0.1,
            }
        )
        labels = tmp_path / "labels.parquet"
        out = tmp_path / "geometry.parquet"
        pd.DataFrame(rows).to_parquet(labels, index=False)

        calibrator.calibrate_match(labels, out)
        geo = pd.read_parquet(out).set_index("frame_index")

        assert geo.loc[1, "H_quality"] == 0.0
        # Identity fallback so the projector sees a valid 3x3.
        assert np.allclose(np.array(geo.loc[1, "H_flat"]).reshape(3, 3), np.eye(3))
        assert geo.loc[0, "H_quality"] > 0.99

    def test_creates_parent_directories(
        self,
        calibrator: Calibrator,
        pitch: PitchModel,
        pitch_to_image_H: np.ndarray,
        tmp_path: Path,
    ) -> None:
        labels = tmp_path / "labels.parquet"
        out = tmp_path / "nested" / "deep" / "geometry.parquet"
        self._write_labels(labels, pitch, pitch_to_image_H, [0])

        calibrator.calibrate_match(labels, out)
        assert out.exists()


class TestEndToEndWithProjector:
    def test_calibrator_output_feeds_projector(
        self,
        calibrator: Calibrator,
        pitch: PitchModel,
        pitch_to_image_H: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """Calibrator geometry.parquet drives PitchProjector end-to-end."""
        # Build geometry from labels.
        rows = []
        for lbl in _labels_from_landmarks(
            pitch, pitch_to_image_H, GOOD_LANDMARKS, frame_index=0
        ):
            rows.append(
                {
                    "frame_index": 0,
                    "landmark": lbl.landmark,
                    "image_x": lbl.image_x,
                    "image_y": lbl.image_y,
                }
            )
        labels = tmp_path / "labels.parquet"
        geometry = tmp_path / "geometry.parquet"
        pd.DataFrame(rows).to_parquet(labels, index=False)
        calibrator.calibrate_match(labels, geometry)

        # A player whose bottom-centre anchor is the image of the centre spot
        # should project back to ~(0, 0) on the pitch.
        cx, cy = _image_point_for(pitch_to_image_H, (0.0, 0.0))
        w, h = 0.04, 0.08
        tracks = pd.DataFrame(
            [
                {
                    "frame_index": 0,
                    "continuous_time_s": 0.0,
                    "track_id": 1,
                    "label": PLAYER_TAG,
                    "confidence": 0.9,
                    "bbox_x": cx - w / 2.0,
                    "bbox_y": cy - h,
                    "bbox_w": w,
                    "bbox_h": h,
                }
            ]
        )
        tracks_path = tmp_path / "tracks.parquet"
        out_path = tmp_path / "pitch_positions.parquet"
        tracks.to_parquet(tracks_path, index=False)

        PitchProjector(pitch_dims=(PITCH_LENGTH, PITCH_WIDTH)).project_match(
            tracks_path, geometry, out_path
        )
        result = pd.read_parquet(out_path)
        row = result.iloc[0]
        assert row["x_pitch"] == pytest.approx(0.0, abs=0.5)
        assert row["y_pitch"] == pytest.approx(0.0, abs=0.5)


class TestCustomQualityScale:
    def test_stricter_scale_lowers_quality(
        self,
        pitch: PitchModel,
        pitch_to_image_H: np.ndarray,
    ) -> None:
        labels = _labels_from_landmarks(pitch, pitch_to_image_H, GOOD_LANDMARKS)
        # Introduce a known reprojection error via a perturbed label.
        labels[0] = KeypointLabel(
            0, labels[0].landmark, labels[0].image_x + 0.02, labels[0].image_y
        )
        lenient = Calibrator(
            pitch_dims=(PITCH_LENGTH, PITCH_WIDTH), quality_error_scale_m=10.0
        )
        strict = Calibrator(
            pitch_dims=(PITCH_LENGTH, PITCH_WIDTH), quality_error_scale_m=1.0
        )
        assert (
            strict.calibrate_frame(0, labels).H_quality
            < lenient.calibrate_frame(0, labels).H_quality
        )


def test_calibration_result_dataclass_fields() -> None:
    res = CalibrationResult(
        frame_index=3,
        H=np.eye(3),
        H_quality=0.8,
        n_correspondences=6,
        n_inliers=6,
        mean_reproj_error_m=0.2,
    )
    assert res.frame_index == 3
    assert res.n_inliers == 6
