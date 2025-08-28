from fastapi import APIRouter, Depends
from app.controller.search_controller import SearchController
from app.schemas.search_queries import SingleSearchRequest, TrakeSearchRequest
from app.schemas.search_results import SingleSearchResponse, KeyframeScore, TrakeResponse
from app.schemas.search_settings import TopKReturn, ControllerParams
from app.core.dependencies import get_controller

router = APIRouter(prefix="/search", tags=["search"])

@router.post("/single", response_model=SingleSearchResponse)
async def search_single(
    req: SingleSearchRequest,
    topk: TopKReturn = Depends(TopKReturn),
    ctrl: ControllerParams = Depends(ControllerParams),
    controller: SearchController = Depends(get_controller),
):
    return await controller.single_search(req, topk, ctrl)


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

