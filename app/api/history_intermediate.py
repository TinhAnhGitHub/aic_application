from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from pydantic import ConfigDict
from typing import List
from datetime import datetime
from app.core.dependencies import get_intermediate_repo
from app.repository.intermediate_repo import IntermediateRepo
from app.schemas.application import KeyframeRef
from app.core.dependencies import  get_chat_repo
from app.repository.chat_repo import ChatRepo
from app.models.history import SearchHistory
from typing_extensions import DefaultDict, Dict, List
from collections import defaultdict

router = APIRouter(prefix="/history", tags=["history"]) 


class IntermediateSetPayload(BaseModel):
    question_filename: str = Field(..., description="Logical name used to group related searches")
    items: List[KeyframeRef] = Field(default_factory=list)
 


class IntermediateResponse(BaseModel):
    question_filename: str
    items: List[KeyframeRef]
    updated_at: str
  


@router.get("/question-filename", summary="Get all the available question filename")
async def get_all_question_filename(
    chat_repo: ChatRepo = Depends(get_chat_repo),
) -> list[str]:
    questions = await chat_repo.get_all_question_filename()
    return questions



@router.get(
    "/intermediate",
    response_model=IntermediateResponse,
    summary="Get intermediate selections",
    description="Fetch current UI-driven intermediate keyframe selections for a question group.",
)
async def get_intermediate(
    question_filename: str = Query(
        ...,
        description="Question group identifier",
        example="demo-1",
    ),
    repo: IntermediateRepo = Depends(get_intermediate_repo),
):
    doc = await repo.get_by_question(question_filename)
    if not doc:
        return IntermediateResponse(question_filename=question_filename, items=[], updated_at="")
    return IntermediateResponse(
        question_filename=doc.question_filename,
        items=doc.items,
        updated_at=doc.updated_at.isoformat(),
    )


@router.put(
    "/intermediate",
    response_model=IntermediateResponse,
    summary="Set intermediate selections",
    description="Replace the intermediate items for a question group.",
)
async def set_intermediate(
    payload: IntermediateSetPayload,
    repo: IntermediateRepo = Depends(get_intermediate_repo),
):
    try:
        doc = await repo.set_items(payload.question_filename, payload.items)
        return IntermediateResponse(
            question_filename=doc.question_filename,
            items=doc.items,
            updated_at=doc.updated_at.isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set intermediate results: {str(e)}", 
        )


@router.delete(
    "/intermediate",
    response_model=dict,
    summary="Clear intermediate selections",
    description="Clear and delete intermediate selections for a question group.",
)
async def clear_intermediate(
    question_filename: str = Query(..., example="demo-1"),
    repo: IntermediateRepo = Depends(get_intermediate_repo),
):
    deleted = await repo.clear(question_filename)
    return {"deleted": deleted}

@router.get(
    '',
    response_model=dict[str ,list[SearchHistory]],
    summary="All history grouped by question",
    description="Return all search history grouped by question (latest first per group).",
)
async def get_all_history(chat_repo: ChatRepo = Depends(get_chat_repo)):
    try:
        docs =  await chat_repo.get_all()
        buckets: DefaultDict[str, list[SearchHistory]] = defaultdict(list)
        for d in docs:
            buckets[d.question_filename].append(d)
        for qf in buckets:
            buckets[qf].sort(key=lambda x: x.timestamp, reverse=True)
        
        groups_sorted = sorted(
            buckets.items(),
            key=lambda kv: (kv[1][0].timestamp if kv[1] else datetime.min),
            reverse=False,
        )

        ordered_groups: Dict[str, List[SearchHistory]] = {k: v for k, v in groups_sorted}
        return ordered_groups

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch history: {str(e)}"
        )
    


@router.get(
    '/by-question',
    response_model=list[SearchHistory],
    summary="History by question",
    description="List search history entries for a given question filename.",
)
async def history_by_question(
    question_filename: str = Query(..., description="Group identifier", example="demo-1"),
    limit: int = Query(100, ge=1, le=1000, example=50),
    chat_repo: ChatRepo = Depends(get_chat_repo),
):
    try:
        docs = await chat_repo.get_by_question(question_filename, limit=limit)
        return [d.model_dump(by_alias=True) for d in docs]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch history by question: {str(e)}"
        )
    


@router.get(
    "/by-timestamp",
    response_model=SearchHistory,
    summary="History by timestamp",
    description="Fetch a single history entry by its timestamp.",
)
async def list_history_by_question(
    timestamp: str = Query(..., description="timestamp", example="2024-06-01T12:34:56"),
    limit: int = Query(100, ge=1, le=1000, example=100),
    chat_repo: ChatRepo = Depends(get_chat_repo),
):

    try:
        doc = await chat_repo.get_by_timestamp(timestamp=timestamp, limit=limit)
        if not doc:
            raise HTTPException(status_code=404, detail="No history at that timestamp.")
        return doc.model_dump(by_alias=True)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch history: {str(e)}"
        )
    