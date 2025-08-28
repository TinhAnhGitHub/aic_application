# app/schemas/search_results.py
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from app.schemas.search_queries import SearchModality, FusionMethod




class MilvusSearchResponseItem(BaseModel):
    """
    Response item schema for Milvus vector search results.
    """
    identification: int = Field(..., description="The identification of the keyframe, corresponding to the index of the embeddings in the Milvus Collection")
    score: float = Field(..., description="The similarity score of the retrieved item.")


class RRFDetail(BaseModel):
    k: int = 60

class WeightedDetail(BaseModel):
    weights: List[float]  
    norm_score: bool = True

class KeyframeScore(BaseModel):
    identification: int
    group_id: str
    video_id: str
    keyframe_id: str
    tags: Optional[list[str]] = None
    ocr: Optional[list[str]] = None
    score: float

class ModalityResult(BaseModel):
    modality: SearchModality
    items: List[KeyframeScore]

class FusionSummary(BaseModel):
    method: FusionMethod
    detail: Union[RRFDetail, WeightedDetail, None] = None

class SingleSearchResponse(BaseModel):
    fused: list[KeyframeScore]
    per_modality: list[ModalityResult]
    fusion: FusionSummary
    meta: dict = Field(default_factory=dict)

class TrakePath(BaseModel):
    items: List[KeyframeScore]  # One keyframe for each event. 
    score: float

class TrakePathResponse(BaseModel):
    paths: List[TrakePath]   
    meta: dict = Field(default_factory=dict)