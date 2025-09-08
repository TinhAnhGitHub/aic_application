from fastapi import APIRouter, Depends, Query,  HTTPException, status
from app.controller.search_controller import SearchController
from app.schemas.search_queries import SingleSearchPayload, TrakeSearchRequest
from app.schemas.search_results import SingleSearchResponse,  TrakeResponse
from app.core.dependencies import get_controller, get_chat_repo
from app.repository.chat_repo import ChatRepo
from app.models.history import SearchHistory
from datetime import datetime

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "/single",
    response_model=SingleSearchResponse,
    summary="Single search (multi-modality fusion)",
    description=(
        "Search across keyframe, caption, and OCR modalities and fuse the results. "
        "Optionally control fusion method, top-k per modality, and user tag boosts via `ctrl`. "
        "Set `req.question_filename` to group related searches."
    ),
    responses={
        500: {"description": "Single search failed"},
    },
)
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
            single_request=payload,
            single_response=resp
        )
        await chat_repo.create_one(hist)
        return resp
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Single search failed: {str(e)}"
        )

@router.post(
    "/trake",
    response_model=TrakeResponse,
    summary="Trake search (sequence of events)",
    description=(
        "Sequence‑aware search across multiple events. "
        "Provide `events` as ordered list of subqueries (each a SingleSearchPayload)."
    ),
    responses={
        500: {"description": "Trake search failed"},
    },
)
async def search_trake(
    req: TrakeSearchRequest,
    controller: SearchController = Depends(get_controller),
    chat_repo: ChatRepo = Depends(get_chat_repo),
):
    
    resp, _raw = await controller.trake_search(
        req,
        window=10,
        beam_size=50,
        per_bucket_top_k=None,
        global_top_k=100,
    )
    hist = SearchHistory(
        question_filename=req.events[0].query.req.question_filename,
        kind='trake',
        trake_request=req,
        trake_response= resp
    )
    await chat_repo.create_one(hist)
    response = TrakeResponse(
        trake_paths=resp,
        raw=_raw,
    )
    return response