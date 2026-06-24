"""Canonical pitch model — named landmark coordinates in pitch metres.

These landmarks are the *target* points for calibration: a labeller marks
where a known pitch feature (e.g. the centre spot, a penalty-area corner)
appears in a broadcast frame, and the :class:`~footy_track.projection.calibrator.Calibrator`
solves for the homography that maps those image points onto the landmark
positions defined here.

Coordinate convention (must match
:mod:`footy_track.projection.projector` and ``docs/design/projection.md``):

- Origin at the **centre spot**.
- ``+x`` along the long axis, from the defending goal to the attacking goal
  of the home team. Range ``[-L/2, +L/2]``.
- ``+y`` perpendicular, toward the technical-area touchline.
  Range ``[-W/2, +W/2]``.
- Units: **metres**.

Landmark positions are parameterised by the pitch dimensions so that a
match with non-standard dimensions still produces a self-consistent model.
The fixed feature sizes (penalty area, goal area, centre circle, penalty
spot distance) follow the IFAB Laws of the Game, which specify them in
absolute metres independent of overall pitch size.
"""

from __future__ import annotations

from dataclasses import dataclass

# IFAB Laws of the Game — fixed feature dimensions in metres.
PENALTY_AREA_LENGTH = 16.5  # depth from goal line
PENALTY_AREA_WIDTH = 40.32  # total width (16.5 + 7.32 + 16.5)
GOAL_AREA_LENGTH = 5.5  # depth from goal line
GOAL_AREA_WIDTH = 18.32  # total width (5.5 + 7.32 + 5.5)
CENTRE_CIRCLE_RADIUS = 9.15
PENALTY_SPOT_DISTANCE = 11.0  # from goal line
GOAL_WIDTH = 7.32

DEFAULT_PITCH_DIMS = (105.0, 68.0)


@dataclass(frozen=True)
class PitchModel:
    """Named pitch landmarks in metres, origin at the centre spot.

    Args:
        length_m: pitch length (goal line to goal line).
        width_m: pitch width (touchline to touchline).
    """

    length_m: float = DEFAULT_PITCH_DIMS[0]
    width_m: float = DEFAULT_PITCH_DIMS[1]

    @property
    def half_length(self) -> float:
        return self.length_m / 2.0

    @property
    def half_width(self) -> float:
        return self.width_m / 2.0

    def landmarks(self) -> dict[str, tuple[float, float]]:
        """Return the full set of named landmarks ``{name: (x, y)}``.

        ``+x`` points toward the *attacking* (right, ``+x``) goal; the
        *defending* goal is at ``-x``. Names use ``left`` / ``right`` to
        mean the defending (``-x``) and attacking (``+x``) ends so that the
        mapping is unambiguous regardless of which half a feature is in.
        """
        hl = self.half_length
        hw = self.half_width
        pa_half_w = PENALTY_AREA_WIDTH / 2.0
        ga_half_w = GOAL_AREA_WIDTH / 2.0
        goal_half_w = GOAL_WIDTH / 2.0

        marks: dict[str, tuple[float, float]] = {
            # Centre
            "centre_spot": (0.0, 0.0),
            "centre_circle_top": (0.0, CENTRE_CIRCLE_RADIUS),
            "centre_circle_bottom": (0.0, -CENTRE_CIRCLE_RADIUS),
            # Pitch corners
            "corner_left_top": (-hl, hw),
            "corner_left_bottom": (-hl, -hw),
            "corner_right_top": (hl, hw),
            "corner_right_bottom": (hl, -hw),
            # Halfway line ends
            "halfway_top": (0.0, hw),
            "halfway_bottom": (0.0, -hw),
            # Penalty spots
            "penalty_spot_left": (-hl + PENALTY_SPOT_DISTANCE, 0.0),
            "penalty_spot_right": (hl - PENALTY_SPOT_DISTANCE, 0.0),
            # Left penalty area (defending end, -x)
            "pen_area_left_top_outer": (-hl, pa_half_w),
            "pen_area_left_bottom_outer": (-hl, -pa_half_w),
            "pen_area_left_top_inner": (-hl + PENALTY_AREA_LENGTH, pa_half_w),
            "pen_area_left_bottom_inner": (-hl + PENALTY_AREA_LENGTH, -pa_half_w),
            # Right penalty area (attacking end, +x)
            "pen_area_right_top_outer": (hl, pa_half_w),
            "pen_area_right_bottom_outer": (hl, -pa_half_w),
            "pen_area_right_top_inner": (hl - PENALTY_AREA_LENGTH, pa_half_w),
            "pen_area_right_bottom_inner": (hl - PENALTY_AREA_LENGTH, -pa_half_w),
            # Left goal area
            "goal_area_left_top_outer": (-hl, ga_half_w),
            "goal_area_left_bottom_outer": (-hl, -ga_half_w),
            "goal_area_left_top_inner": (-hl + GOAL_AREA_LENGTH, ga_half_w),
            "goal_area_left_bottom_inner": (-hl + GOAL_AREA_LENGTH, -ga_half_w),
            # Right goal area
            "goal_area_right_top_outer": (hl, ga_half_w),
            "goal_area_right_bottom_outer": (hl, -ga_half_w),
            "goal_area_right_top_inner": (hl - GOAL_AREA_LENGTH, ga_half_w),
            "goal_area_right_bottom_inner": (hl - GOAL_AREA_LENGTH, -ga_half_w),
            # Goal posts
            "goal_left_top": (-hl, goal_half_w),
            "goal_left_bottom": (-hl, -goal_half_w),
            "goal_right_top": (hl, goal_half_w),
            "goal_right_bottom": (hl, -goal_half_w),
        }
        return marks

    def landmark(self, name: str) -> tuple[float, float]:
        """Return a single landmark's ``(x, y)`` in metres.

        Raises:
            KeyError: if ``name`` is not a known landmark.
        """
        marks = self.landmarks()
        if name not in marks:
            raise KeyError(
                f"unknown pitch landmark {name!r}; valid names: {sorted(marks)}"
            )
        return marks[name]

    def landmark_names(self) -> list[str]:
        """Return the sorted list of valid landmark names."""
        return sorted(self.landmarks())
