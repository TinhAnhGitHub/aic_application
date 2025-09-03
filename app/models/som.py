from __future__ import annotations
from beanie import Document, Indexed
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal


class SomFeedbackEvent(Document):
    question_filename: str
    identification: int
    action: Literal['up', 'down']
    weight: float = 1.0
    timestamp: Indexed(datetime) = Field(default_factory=datetime.now)

    class Settings:
        name = 'som_feedback_events'
        indexes = [
            "question_filename",
            "identification",
            ('ts', -1)
        ]



class SomOverlay(Document):
    question_filename: str
    grid_h: int
    grid_w: int 
    pos: str
    neg: str
    updated_at: Indexed(datetime) = Field(default_factory=datetime.now)

    class Settings:
        name = "som_overlays"
        indexes = [
            'question_filename',
            ('updated_at', -1)
        ]
