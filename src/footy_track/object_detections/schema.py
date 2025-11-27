"""Schemas for object detection results."""

import pathlib

from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str = Field(..., description="Class name")
    confidence: float = Field(..., ge=0.0, le=1.0)
    x: float = Field(..., ge=0.0, le=1.0, description="Top-left x (normalized)")
    y: float = Field(..., ge=0.0, le=1.0, description="Top-left y (normalized)")
    w: float = Field(..., ge=0.0, le=1.0, description="Width (normalized)")
    h: float = Field(..., ge=0.0, le=1.0, description="Height (normalized)")

    model_config = {"frozen": True}


class FrameDetections(BaseModel):
    uri: pathlib.Path = Field(..., description="Path to the image file or identifier")
    width: int
    height: int
    detections: list[Detection]


class FrameDetectionsWithMeta(FrameDetections):
    clock: str | None = None
