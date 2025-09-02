from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Union

from app.schemas.search_settings import TopKReturn, ControllerParams

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
    question_filename: str = Field(..., description="Logical name used to group related searches")
    keyframe: Optional[KeyframeQuery] = None
    caption: Optional[CaptionQuery] = None
    ocr: Optional[OCRQuery] = None



class SingleSearchPayload(BaseModel):
    req: SingleSearchRequest
    ctrl: ControllerParams = ControllerParams()



class EventQuery(BaseModel):
    event_order: int
    query: SingleSearchPayload

class TrakeSearchRequest(BaseModel):
    events: list[EventQuery]


