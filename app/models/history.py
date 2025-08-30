from beanie import Document, Indexed
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime

from app.schemas.search_queries import SingleSearchRequest, TrakeSearchRequest
from app.schemas.search_results import SingleSearchResponse, TrakePathResponse

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

    # Input
    single_request : SingleSearchRequest | None = None
    trake_request: TrakeSearchRequest | None = None

    #output
    single_response: SingleSearchResponse | None = None
    trake_response: TrakePathResponse | None = None

    # metadta
    tags_used: list[str] | None = None
    class Settings:
        indexes = [
            [("question_filename", 1), ("created_at", -1)],
        ]
    
