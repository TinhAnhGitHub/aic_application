from pydantic import BaseModel, Field
from typing import Literal


class KeyframeInstance(BaseModel):
    group_id: str
    video_id: str
    keyframe_id: str
    identification: int = Field(..., description="The identification of the keyframe, corresponding to the index of the embeddings in the Milvus Collection")
    tags: list[str] | None = Field(None, description="List of tags associated with the keyframe")
    ocr: list[str] | None = Field(None, description="List of OCR texts associated with the keyframe")


class MilvusSearchRequestInput(BaseModel):
    """
    Input schema for Milvus vector search requests.
    """

    embedding: list[float] = Field(..., description="The embedding vector to search for.")
    top_k: int = Field(..., description="The number of top similar items to retrieve.")


class MilvusSearchResponseItem(BaseModel):
    """
    Response item schema for Milvus vector search results.
    """

    id_: str = Field(..., description="The unique identifier of the retrieved item.")
    score: float = Field(..., description="The similarity score of the retrieved item.")
    
    

class KeyframeSearchMilvusResponseItem(MilvusSearchResponseItem):
    group_id: str
    video_id: str
    keyframe_id: str
    identification: int = Field(..., description="The identification of the keyframe, corresponding to the index of the embeddings in the Milvus Collection")
    tags: list[str] | None = Field(None, description="List of tags associated with the keyframe")



class TagInstance(BaseModel):
    tag_name: str
    tag_score: float


class ElasticSortedKeyframe(BaseModel):
    keyframe_instance: KeyframeInstance
    score: float




class EventOrder(BaseModel):
    """
    Event Text, chunked from the original text, with order to indicate the sequence.
    """
    order: int 
    event_text: str



class EventHit(BaseModel):
    event: EventOrder
    video_id: str
    keyframe_id: str
    group_id: str
    score: float


