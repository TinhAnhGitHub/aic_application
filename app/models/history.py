from beanie import Document, Indexed
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime

from app.schemas.search_queries import SingleSearchRequest, TrakeSearchRequest, SingleSearchPayload
from app.schemas.search_results import SingleSearchResponse, TrakePathResponse

from app.schemas.application import KeyframeRef

HistoryType = Literal["single", "trake"]


class HistoryEvent(BaseModel):
    event_order: int
    query: SingleSearchRequest

class HistoryResult(BaseModel):
    count: int
    top_idents: list[int]

class SearchHistory(Document):
    timestamp: Indexed(datetime) = Field(default_factory=datetime.now)
    question_filename: str 
    kind: HistoryType

    single_request : SingleSearchPayload | SingleSearchRequest | None = None
    trake_request: TrakeSearchRequest | None = None

    single_response: SingleSearchResponse | None = None
    trake_response: TrakePathResponse | None = None

    tags_used: list[str] | None = None
    class Settings:
        indexes = [
            [("question_filename", 1), ("timestamp", -1)],
        ]
    

class IntermediateResult(Document):
    question_filename: str
    items: list[KeyframeRef] = Field(default_factory=list)
    updated_at: Indexed(datetime) = Field(default_factory=datetime.now)

    class Settings:
        name = "intermediate_results"
        indexes = [
            "question_filename",
            [("updated_at", -1)],  
        ]
