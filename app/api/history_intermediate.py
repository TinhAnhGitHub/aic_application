from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import List

from app.core.dependencies import get_intermediate_repo
from app.repository.intermediate_repo import IntermediateRepo
from app.schemas.application import KeyframeRef


router = APIRouter(prefix="/history/intermediate", tags=["history", "intermediate"]) 


class IntermediateSetPayload(BaseModel):
    question_filename: str = Field(..., description="Logical name used to group related searches")
    items: List[KeyframeRef] = Field(default_factory=list)


class IntermediateResponse(BaseModel):
    question_filename: str
    items: List[KeyframeRef]
    updated_at: str


@router.get("", response_model=IntermediateResponse)
async def get_intermediate(
    question_filename: str = Query(..., description="Question group identifier"),
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


@router.put("", response_model=IntermediateResponse)
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


@router.delete("", response_model=dict)
async def clear_intermediate(
    question_filename: str = Query(...),
    repo: IntermediateRepo = Depends(get_intermediate_repo),
):
    deleted = await repo.clear(question_filename)
    return {"deleted": deleted}

