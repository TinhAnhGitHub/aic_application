from beanie import Document 
from pydantic import Field
from pymongo import IndexModel

class CaptionIndexMap(Document):
    identification: int = Field(..., description="Row id in caption index (dense/sparse)")
    group_id: str
    video_id: str
    keyframe_id: str
    caption_idx: int

    class Settings:
        name = "caption_index_map"
        indexes = [
            IndexModel([("identification", 1)], unique=True),
            "group_id", "video_id", "keyframe_id", "caption_idx",
        ]
    


    