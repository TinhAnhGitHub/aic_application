from beanie import Document
from pydantic import BaseModel, Field


class KeyframeModel(Document):
    identification: str = Field(..., description="A unique identifier for the keyframe.")
    group_id: str = Field(..., description="The group ID of the keyframe.")
    video_id: str = Field(..., description="The video ID associated with the keyframe.")
    keyframe_id: str = Field(..., description="The unique ID of the keyframe.")

    class Config:
        collection = "keyframes"
        indexes = [
            "group_id",
            "video_id",
            "keyframe_id",
            {"fields": ["identification"], "unique": True}
        ]

    
    

