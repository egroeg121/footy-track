"""Schema's, typing and Ontologies for Footy Track"""

from collections import OrderedDict

from pydantic import BaseModel

from footy_track.object_detections.schema import Detection


class BaseSchema(BaseModel):
    """Base Schema for Footy Track"""


class Ball(Detection):
    """Schema for a ball"""

    id: str


class OutOfBoundsBall(Ball):
    """Schema for an out of bounds ball"""

    reason: str  # e.g., "out of play", "goal", etc.


class Person(Detection):
    pass


class Player(Person):
    id: str
    name: str


class Ref(Person):
    pass


class Coach(Person):
    pass


class OtherPerson(Person):
    pass


class Team(BaseSchema):
    name: str
    players: list[Player]


class Embedding(BaseSchema):
    model: str
    vector: list[float]


class Frame(BaseSchema):
    frame_number: int
    game_timestamp: float
    embedding: Embedding


class BroadcastFrame(Frame):
    """Schema for a broadcast frame which has detections"""

    players: list[Player | Ref | Coach | OtherPerson]
    ball: list[Ball | OutOfBoundsBall]


class Video(BaseSchema):
    """Schema for Video"""

    id: str
    name: str
    teams: tuple[str, str]
    frames: OrderedDict[int, BroadcastFrame | Frame]  # frame number to Frame mapping
