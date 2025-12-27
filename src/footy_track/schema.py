"""Schema's, typing and Ontologies for Footy Track

A lot of this is originally planned with schema but is not used yet, so leave
comment out for now. Uncomment if they are needed
"""

import pathlib
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


class FrameDetections(BaseModel):
    uri: pathlib.Path = Field(..., description="Path to the image file or identifier")
    width: int
    height: int
    detections: list[ObjectDetection]


class FrameClassifications(BaseModel):
    uri: pathlib.Path = Field(..., description="Path to the image file or identifier")
    classification: BroadcastClassification


class FrameDetectionsWithMeta(FrameDetections):
    clock: str | None = None


class Ball(ObjectDetection):
    """Schema for a ball"""

    label: Literal["ball"] = "ball"
    # id: str


# class OutOfBoundsBall(Ball):
#     """Schema for an out of bounds ball"""

#     label: Literal["out_of_bounds_ball"] = "out_of_bounds_ball"
#     reason: str  # e.g., "out of play", "goal", etc.


class Person(ObjectDetection):
    label: Literal["person"] = "person"


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
