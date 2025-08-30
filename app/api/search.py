from fastapi import APIRouter, Depends, Query
from app.controller.search_controller import SearchController
from app.schemas.search_queries import SingleSearchRequest, TrakeSearchRequest
from app.schemas.search_results import SingleSearchResponse, KeyframeScore, TrakeResponse
from app.schemas.search_settings import TopKReturn, ControllerParams
from app.core.dependencies import get_controller, get_chat_repo
from app.repository.chat_repo import ChatRepo
from app.models.history import SearchHistory, HistoryEvent, HistoryType
from datetime import datetime

router = APIRouter(prefix="/search", tags=["search"])

from pydantic import BaseModel

class SingleSearchPayload(BaseModel):
    req: SingleSearchRequest
    topk: TopKReturn = TopKReturn()
    ctrl: ControllerParams = ControllerParams()




@router.get("", response_model=list[SearchHistory])
async def list_history_by_question(
    question_filename: str = Query(..., description="Group key used when saving searches"),
    limit: int = Query(100, ge=1, le=1000),
    chat_repo: ChatRepo = Depends(get_chat_repo),
):
    """
    Return history documents for the provided question_filename, newest first.
    """
    return await chat_repo.get_by_question(question_filename=question_filename, limit=limit)



@router.post("/single", response_model=SingleSearchResponse)
async def search_single(
    payload: SingleSearchPayload,
    controller: SearchController = Depends(get_controller),
    chat_repo: ChatRepo = Depends(get_chat_repo),
):
    resp = await controller.single_search(payload.req, payload.topk, payload.ctrl)
    hist = SearchHistory(
        question_filename=payload.req.question_filename,
        kind='single',
        single_request=payload.req,
        single_response=resp
        )

    await chat_repo.create_one(hist)
    return resp


@router.post("/trake", response_model=TrakeResponse)
async def search_trake(
    req: TrakeSearchRequest,
    topk: TopKReturn = Depends(TopKReturn),
    ctrl: ControllerParams = Depends(ControllerParams),
    controller: SearchController = Depends(get_controller),
):
    resp, _raw = await controller.trake_search(
        req,
        topk=topk,
        ctrl=ctrl,
        window=6,
        beam_size=50,
        per_bucket_top_k=None,
        global_top_k=20,
        norm_method="zscore",
        norm_temperature=1.0,
    )
    response = TrakeResponse(
        trake_paths=resp,
        raw=_raw,
    )
    return response

