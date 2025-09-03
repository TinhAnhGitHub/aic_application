from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from app.services.som_service import SomFeedbackService
from app.core.dependencies import get_som_service


router = APIRouter(prefix="/feedback", tags=["feedback"]) 


class FeedbackPayload(BaseModel):
    question_filename: str = Field(...)
    identification: int = Field(...)
    action: str = Field(pattern="^(up|down)$")
    weight: float = 1.0


@router.post("", response_model=dict)
async def submit_feedback(payload: FeedbackPayload, svc: SomFeedbackService = Depends(get_som_service)):
    try:
        await svc.apply_feedback(payload.question_filename, payload.identification, payload.action, payload.weight)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("", response_model=dict)
async def get_overlay_info(question_filename: str = Query(...), svc: SomFeedbackService = Depends(get_som_service)):
    try:
        pos, neg = await svc._get_overlay(question_filename)
        return {
            "question_filename": question_filename,
            "sum_pos": float(pos.sum()),
            "sum_neg": float(neg.sum()),
            "shape": [int(pos.shape[0]), int(pos.shape[1])],
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("", response_model=dict)
async def clear_overlay(question_filename: str = Query(...), svc: SomFeedbackService = Depends(get_som_service)):
    try:
        d_overlay, d_events = await svc.repo.clear(question_filename)
        if question_filename in svc._cache:
            del svc._cache[question_filename]
        return {"deleted_overlay": int(d_overlay), "deleted_events": int(d_events)}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
