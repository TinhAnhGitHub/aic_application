from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Union

SearchModality = Literal['keyframe', 'caption', 'ocr']
FusionMethod = Literal['rrf', 'weighted']

class BaseModalityQuery(BaseModel):
    modality: SearchModality
    tag_boost_alpha: float = Field(0.0, ge=0.0, le=1.0)

class KeyframeQuery(BaseModalityQuery):
    modality: Literal['keyframe'] = 'keyframe'
    text: str  

class CaptionQuery(BaseModalityQuery):
    modality: Literal['caption'] = 'caption'
    text: str
    fusion: FusionMethod = 'rrf'
    weighted: float | None = Field(None, description="If 'weighted': weight for dense; (1-weight) for sparse")


class OCRQuery(BaseModalityQuery):
    modality: Literal['ocr'] = 'ocr'
    text: str


class SingleSearchRequest(BaseModel):
    keyframe: Optional[KeyframeQuery] = None
    caption: Optional[CaptionQuery] = None
    ocr: Optional[OCRQuery] = None

class EventQuery(BaseModel):
    event_order: int
    query: SingleSearchRequest

class TrakeSearchRequest(BaseModel):
    events: list[EventQuery]


