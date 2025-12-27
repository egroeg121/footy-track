"""Tests for the schema module."""

import pathlib

import pytest
from pydantic import ValidationError

from footy_track.schema import (
    Ball,
    BroadcastClassification,
    EnumBroadcastClassification,
    FrameClassifications,
    FrameDetections,
    ObjectDetection,
    Person,
)

# Constants for test values to avoid magic numbers
CONFIDENCE = 0.9
HIGH_CONFIDENCE = 0.85
X_COORD = 0.1
Y_COORD = 0.2
WIDTH = 0.3
HEIGHT = 0.4
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080


def test_broadcast_classification():
    """Test the BroadcastClassification model."""
    bc = BroadcastClassification(label=EnumBroadcastClassification.YES, confidence=CONFIDENCE)
    assert bc.label == "Yes"
    assert bc.confidence == CONFIDENCE


def test_object_detection():
    """Test the ObjectDetection model."""
    od = ObjectDetection(
        label="player",
        confidence=HIGH_CONFIDENCE,
        x=X_COORD,
        y=Y_COORD,
        w=WIDTH,
        h=HEIGHT,
        model="yolo",
    )
    assert od.label == "player"
    assert od.confidence == HIGH_CONFIDENCE
    assert od.x == X_COORD
    assert od.y == Y_COORD
    assert od.w == WIDTH
    assert od.h == HEIGHT
    assert od.model == "yolo"


def test_object_detection_invalid_confidence():
    """Test that ObjectDetection raises an error for invalid confidence."""
    with pytest.raises(ValidationError):
        ObjectDetection(label="player", confidence=1.1, x=0, y=0, w=0, h=0)

    with pytest.raises(ValidationError):
        ObjectDetection(label="player", confidence=-0.1, x=0, y=0, w=0, h=0)


def test_frame_detections():
    """Test the FrameDetections model."""
    od = ObjectDetection(label="ball", confidence=0.99, x=0.5, y=0.5, w=0.05, h=0.05)
    fd = FrameDetections(
        uri=pathlib.Path("/path/to/frame.jpg"),
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        detections=[od],
    )
    assert fd.uri == pathlib.Path("/path/to/frame.jpg")
    assert fd.width == FRAME_WIDTH
    assert fd.height == FRAME_HEIGHT
    assert len(fd.detections) == 1
    assert fd.detections[0] == od


def test_frame_classifications():
    """Test the FrameClassifications model."""
    bc = BroadcastClassification(label=EnumBroadcastClassification.NO, confidence=0.95)
    fc = FrameClassifications(
        uri=pathlib.Path("/path/to/another/frame.png"),
        classification=bc,
    )
    assert fc.uri == pathlib.Path("/path/to/another/frame.png")
    assert fc.classification == bc


def test_ball_schema():
    """Test the Ball schema to ensure the label is always 'ball'."""
    ball = Ball(confidence=CONFIDENCE, x=X_COORD, y=X_COORD, w=X_COORD, h=X_COORD)
    assert ball.label == "ball"

    # Test that providing a different label raises a validation error
    with pytest.raises(ValidationError):
        Ball(label="not a ball", confidence=CONFIDENCE, x=X_COORD, y=X_COORD, w=X_COORD, h=X_COORD)


def test_person_schema():
    """Test the Person schema to ensure the label is always 'person'."""
    person = Person(confidence=0.8, x=Y_COORD, y=Y_COORD, w=Y_COORD, h=Y_COORD)
    assert person.label == "person"

    # Test that providing a different label raises a validation error
    with pytest.raises(ValidationError):
        Person(label="not a person", confidence=0.8, x=Y_COORD, y=Y_COORD, w=Y_COORD, h=Y_COORD)
