from pydantic import BaseModel, Field
from pydantic import ConfigDict
from typing import Literal
from typing_extensions import Annotated, Union  


class KeyframeInstance(BaseModel):
    group_id: str
    video_id: str
    keyframe_id: str
    identification: int = Field(..., description="The identification of the keyframe, corresponding to the index of the embeddings in the Milvus Collection")
    tags: list[str] | None = Field(None, description="List of tags associated with the keyframe")
    ocr: list[str] | None = Field(None, description="List of OCR texts associated with the keyframe")


class KeyframeRef(BaseModel):
    group_id: str
    video_id: str
    keyframe_id: str
 

