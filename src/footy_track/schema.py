"""Schema's, typing and Ontologies for Footy Track

A lot of this is originally planned with schema but is not used yet, so leave
comment out for now. Uncomment if they are needed
"""

import pathlib
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    import fiftyone as fo

from footy_track.constants import (
    BALL_TAG,
    COACH_TAG,
    IN_PLAY_BALL_TAG,
    OUT_OF_PLAY_BALL_TAG,
    PERSON_TAG,
    PLAYER_SUB_TAG,
    PLAYER_TAG,
    REFEREE_TAG,
)

DETECTION_CLASSES = [
    PERSON_TAG,
    BALL_TAG,
    OUT_OF_PLAY_BALL_TAG,
    IN_PLAY_BALL_TAG,
    PLAYER_TAG,
    PLAYER_SUB_TAG,
    REFEREE_TAG,
    COACH_TAG,
]


class BaseSchema(BaseModel):
    """Base Schema for Footy Track"""

    model_config = ConfigDict(frozen=True)


class EnumBroadcastClassification(StrEnum):
    """Enumeration for broadcast classification."""

    YES = "Yes"
    NO = "No"


class BroadcastClassification(BaseModel):
    """Result of a Broadcast frame Classification."""

    label: EnumBroadcastClassification
    confidence: float | None = None


class ObjectDetection(BaseModel):
    label: str = Field(..., description="Class name")
    confidence: float = Field(..., ge=0.0, le=1.0)
    x: float = Field(..., ge=0.0, le=1.0, description="Top-left x (normalized)")
    y: float = Field(..., ge=0.0, le=1.0, description="Top-left y (normalized)")
    w: float = Field(..., ge=0.0, le=1.0, description="Width (normalized)")
    h: float = Field(..., ge=0.0, le=1.0, description="Height (normalized)")
    model: str | None = Field(None, description="Model name or identifier")
    model_config = {"frozen": True}


class Person(ObjectDetection):
    label: Literal["person"] = "person"


class Ball(ObjectDetection):
    """Schema for a ball"""

    label: Literal["ball"] = "ball"


class FrameDetections(BaseModel):
    uri: pathlib.Path = Field(..., description="Path to the image file or identifier")
    width: int
    height: int
    detections: list[ObjectDetection]


class FrameClassifications(BaseModel):
    uri: pathlib.Path = Field(..., description="Path to the image file or identifier")
    classification: BroadcastClassification

    def to_fiftyone_sample(self) -> "fo.Sample":
        """Convert to a FiftyOne Sample.

        ``fiftyone`` is imported lazily: it is a heavy dependency (and pulls in
        a bundled MongoDB) that the labeller does not need at all. Importing it
        at module scope made ``import footy_track.labeller.server`` cost ~12 s
        and 245 MB, and made the labeller unrunnable on a machine without it.
        """
        import fiftyone as fo  # noqa: PLC0415

        sample = fo.Sample(filepath=str(self.uri))
        sample["broadcast_classification"] = fo.Classification(
            label=self.classification.label.value,
            confidence=self.classification.confidence,
        )
        return sample


class FrameDetectionsWithMeta(FrameDetections):
    clock: str | None = None


# class OutOfBoundsBall(Ball):
#     """Schema for an out of bounds ball"""

#     label: Literal["out_of_bounds_ball"] = "out_of_bounds_ball"
#     reason: str  # e.g., "out of play", "goal", etc.


# class Player(Person):
#     label: Literal["player"] = "player"
#     id: str
#     name: str


# class Ref(Person):
#     label: Literal["ref"] = "ref"
#     pass


# class Coach(Person):
#     label: Literal["coach"] = "coach"
#     pass


# class OtherPerson(Person):
#     label: Literal["other_person"] = "other_person"
#     pass


# class Team(BaseSchema):
#     name: str
#     players: list[Player]


# class Frame(BaseSchema):
#     frame_number: int
#     game_timestamp: float
#     embedding: Embedding


# class BroadcastFrame(Frame):
#     """Schema for a broadcast frame which has detections"""

#     players: list[Player | Ref | Coach | OtherPerson]
#     ball: list[Ball | OutOfBoundsBall]


# class Video(BaseSchema):
#     """Schema for Video"""

#     id: str
#     name: str
#     teams: tuple[str, str]
#     frames: OrderedDict[int, BroadcastFrame | Frame]  # frame number to Frame mapping
