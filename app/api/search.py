from fastapi import APIRouter, Depends, Query,  HTTPException, status
from app.controller.search_controller import SearchController
from app.schemas.search_queries import SingleSearchPayload, TrakeSearchRequest
from app.schemas.search_results import SingleSearchResponse, KeyframeScore, TrakeResponse
from app.schemas.search_settings import TopKReturn, ControllerParams
from app.core.dependencies import get_controller, get_chat_repo
from app.repository.chat_repo import ChatRepo
from app.models.history import SearchHistory, HistoryEvent, HistoryType
from datetime import datetime
from typing_extensions import DefaultDict, Dict, List
from collections import defaultdict
router = APIRouter(prefix="/search", tags=["search"])

from pydantic import BaseModel




@router.get('/history', response_model=dict[str ,list[SearchHistory]])
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


@router.get("", response_model=SearchHistory)
async def list_history_by_question(
    timestamp: str = Query(..., description="timestamp"),
    limit: int = Query(100, ge=1, le=1000),
    chat_repo: ChatRepo = Depends(get_chat_repo),
):
    """
    Return history documents for the provided question_filename, newest first.
    """
    try:
        ts = datetime.fromisoformat(timestamp)
        doc = await chat_repo.get_by_timestamp(timestamp=ts, limit=limit)
        return doc.model_dump(by_alias=True) 
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch history: {str(e)}"
        )



@router.post("/single", response_model=SingleSearchResponse)
async def search_single(
    payload: SingleSearchPayload,
    controller: SearchController = Depends(get_controller),
    chat_repo: ChatRepo = Depends(get_chat_repo),
):
    try:
        resp = await controller.single_search(payload.req, payload.ctrl.topk_settings, payload.ctrl)
        hist = SearchHistory(
            question_filename=payload.req.question_filename,
            kind='single',
            single_request=payload.req,
            single_response=resp
        )
        await chat_repo.create_one(hist)
        return resp
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Single search failed: {str(e)}"
        )



@router.post("/trake", response_model=TrakeResponse)
async def search_trake(
    req: TrakeSearchRequest,
    controller: SearchController = Depends(get_controller),
):
    
    resp, _raw = await controller.trake_search(
        req,
        window=10,
        beam_size=50,
        per_bucket_top_k=None,
        global_top_k=100,
        norm_method="minmax",
        norm_temperature=1.0,
    )
    response = TrakeResponse(
        trake_paths=resp,
        raw=_raw,
    )
    return response


