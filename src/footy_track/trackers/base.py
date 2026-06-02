"""Tracker protocol and shared data schemas."""

from typing import Protocol

from pydantic import Field

from footy_track.schema import FrameDetections, ObjectDetection


class TrackedDetection(ObjectDetection):
    """ObjectDetection extended with tracker-assigned identity."""

    track_id: int = Field(..., description="Monotone track ID, unique within a match")
    frame_index: int = Field(..., description="Zero-based frame number")
    continuous_time_s: float = Field(..., description="ContinuousTime in seconds")
    is_interpolated: bool = Field(
        default=False, description="True when filled from Kalman, no raw detection"
    )


class TrackMeta:
    """Per-track summary for tracks_meta.json sidecar."""

    __slots__ = (
        "track_id",
        "label",
        "start_frame",
        "end_frame",
        "start_continuous_time_s",
        "end_continuous_time_s",
        "reid_parent_track_id",
        "team_id",
        "jersey_number",
        "player_id",
    )

    def __init__(
        self,
        track_id: int,
        label: str,
        start_frame: int,
        end_frame: int,
        start_continuous_time_s: float,
        end_continuous_time_s: float,
        reid_parent_track_id: int | None = None,
        team_id: str | None = None,
        jersey_number: int | None = None,
        player_id: str | None = None,
    ) -> None:
        self.track_id = track_id
        self.label = label
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_continuous_time_s = start_continuous_time_s
        self.end_continuous_time_s = end_continuous_time_s
        self.reid_parent_track_id = reid_parent_track_id
        self.team_id = team_id
        self.jersey_number = jersey_number
        self.player_id = player_id

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_continuous_time_s": self.start_continuous_time_s,
            "end_continuous_time_s": self.end_continuous_time_s,
            "reid_parent_track_id": self.reid_parent_track_id,
            "team_id": self.team_id,
            "jersey_number": self.jersey_number,
            "player_id": self.player_id,
        }


class ObjectTracker(Protocol):
    """Protocol every tracker implementation must satisfy."""

    def update(
        self, frame_detections: FrameDetections, frame_t: float
    ) -> list[TrackedDetection]:
        """Assign track IDs to a single frame's detections.

        Stateful — call once per broadcast frame in time order.
        """
        ...

    def finalise(self) -> list[TrackMeta]:
        """Return per-track summaries after the last update()."""
        ...
