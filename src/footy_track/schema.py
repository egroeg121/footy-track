"""Schema's, typing and Ontologies for Footy Track"""

from collections import OrderedDict
from pydantic import BaseModel

class Person()

class BaseSchema(BaseModel):
    """Base Schema for Footy Track"""


class Ball(BaseSchema):
    """Schema for a ball"""
    
    id: str

class OutOfBoundsBall(BaseSchema):
    """Schema for an out of bounds ball"""
    reason: str  # e.g., "out of play", "goal", etc.

class Player:
    id: str
    name: str

class Ref:
    type: 

class Coach
    pass

class OtherPerson:
    pass



class Team(BaseSchema):
    name: str
    players: list[Player]

class Embedding(BaseSchema):
    model: str
    vector: list[float]

class BroadcastFrame(BaseSchema):
    """Schema for a broadcast frame"""

    frame_number: int
    game_timestamp: float
    players: list[Player | Ref | Coach | OtherPerson]
    ball: list[Ball | OutOfBoundsBall]
    embedding: Embedding

class OtherFrame(BaseSchema):
    pass

class Video(BaseSchema):
    """Schema for Video"""

    id: str
    name: str
    teams: tuple[str, str]
    frames: OrderedDict[int, BroadcastFrame | OtherFrame]  # frame number to Frame mapping