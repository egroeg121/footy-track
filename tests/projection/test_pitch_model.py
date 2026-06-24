"""Tests for the canonical PitchModel landmark coordinates.

Verifies that landmark positions follow the projector's coordinate
convention (origin at centre spot, +x toward the attacking goal, metres)
and respect IFAB fixed feature dimensions.
"""

import math

import pytest

from footy_track.projection.pitch_model import (
    CENTRE_CIRCLE_RADIUS,
    PENALTY_AREA_LENGTH,
    PENALTY_SPOT_DISTANCE,
    PitchModel,
)

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


@pytest.fixture
def pitch() -> PitchModel:
    return PitchModel(length_m=PITCH_LENGTH, width_m=PITCH_WIDTH)


class TestOriginAndExtents:
    def test_centre_spot_is_origin(self, pitch: PitchModel) -> None:
        assert pitch.landmark("centre_spot") == (0.0, 0.0)

    def test_half_length_and_width(self, pitch: PitchModel) -> None:
        assert pitch.half_length == pytest.approx(PITCH_LENGTH / 2.0)
        assert pitch.half_width == pytest.approx(PITCH_WIDTH / 2.0)

    def test_corners_at_pitch_extents(self, pitch: PitchModel) -> None:
        hl, hw = PITCH_LENGTH / 2.0, PITCH_WIDTH / 2.0
        assert pitch.landmark("corner_left_top") == pytest.approx((-hl, hw))
        assert pitch.landmark("corner_left_bottom") == pytest.approx((-hl, -hw))
        assert pitch.landmark("corner_right_top") == pytest.approx((hl, hw))
        assert pitch.landmark("corner_right_bottom") == pytest.approx((hl, -hw))

    def test_halfway_line_at_x_zero(self, pitch: PitchModel) -> None:
        assert pitch.landmark("halfway_top")[0] == pytest.approx(0.0)
        assert pitch.landmark("halfway_bottom")[0] == pytest.approx(0.0)


class TestFixedFeatures:
    def test_centre_circle_radius(self, pitch: PitchModel) -> None:
        top = pitch.landmark("centre_circle_top")
        assert top == pytest.approx((0.0, CENTRE_CIRCLE_RADIUS))

    def test_penalty_spots_distance_from_goal(self, pitch: PitchModel) -> None:
        hl = PITCH_LENGTH / 2.0
        left = pitch.landmark("penalty_spot_left")
        right = pitch.landmark("penalty_spot_right")
        assert left == pytest.approx((-hl + PENALTY_SPOT_DISTANCE, 0.0))
        assert right == pytest.approx((hl - PENALTY_SPOT_DISTANCE, 0.0))

    def test_penalty_area_depth(self, pitch: PitchModel) -> None:
        hl = PITCH_LENGTH / 2.0
        outer = pitch.landmark("pen_area_left_top_outer")
        inner = pitch.landmark("pen_area_left_top_inner")
        assert outer[0] == pytest.approx(-hl)
        assert inner[0] == pytest.approx(-hl + PENALTY_AREA_LENGTH)

    def test_feature_dimensions_independent_of_pitch_size(self) -> None:
        """IFAB features are absolute metres, not scaled with pitch size."""
        small = PitchModel(length_m=90.0, width_m=45.0)
        big = PitchModel(length_m=120.0, width_m=90.0)
        # Penalty spot is always 11 m from its goal line, regardless of size.
        assert small.landmark("penalty_spot_left")[0] == pytest.approx(
            -45.0 + PENALTY_SPOT_DISTANCE
        )
        assert big.landmark("penalty_spot_left")[0] == pytest.approx(
            -60.0 + PENALTY_SPOT_DISTANCE
        )


class TestSymmetry:
    def test_left_right_mirror_in_x(self, pitch: PitchModel) -> None:
        lt = pitch.landmark("corner_left_top")
        rt = pitch.landmark("corner_right_top")
        assert lt[0] == pytest.approx(-rt[0])
        assert lt[1] == pytest.approx(rt[1])

    def test_top_bottom_mirror_in_y(self, pitch: PitchModel) -> None:
        top = pitch.landmark("pen_area_left_top_outer")
        bottom = pitch.landmark("pen_area_left_bottom_outer")
        assert top[0] == pytest.approx(bottom[0])
        assert top[1] == pytest.approx(-bottom[1])


class TestLandmarkAccess:
    def test_unknown_landmark_raises(self, pitch: PitchModel) -> None:
        with pytest.raises(KeyError):
            pitch.landmark("not_a_real_landmark")

    def test_landmark_names_sorted_and_nonempty(self, pitch: PitchModel) -> None:
        names = pitch.landmark_names()
        assert names == sorted(names)
        assert len(names) >= 4  # need at least 4 for a homography

    def test_landmarks_returns_all_named(self, pitch: PitchModel) -> None:
        marks = pitch.landmarks()
        assert "centre_spot" in marks
        assert set(marks) == set(pitch.landmark_names())

    def test_all_landmarks_within_pitch_bounds(self, pitch: PitchModel) -> None:
        hl, hw = PITCH_LENGTH / 2.0, PITCH_WIDTH / 2.0
        for name, (x, y) in pitch.landmarks().items():
            assert -hl - 1e-6 <= x <= hl + 1e-6, name
            assert -hw - 1e-6 <= y <= hw + 1e-6, name
            assert math.isfinite(x) and math.isfinite(y), name


class TestDefaults:
    def test_default_dimensions(self) -> None:
        pitch = PitchModel()
        assert pitch.length_m == pytest.approx(105.0)
        assert pitch.width_m == pytest.approx(68.0)
