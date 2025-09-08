from contextlib import asynccontextmanager
from typing  import Any, Dict, Optional
import json

from fastapi import Request, FastAPI

from app.core.config import settings
from app.models.index import CaptionIndexMap
from app.repository.keyframe_repo import init_mongo, KeyframeRepo
from app.repository.elastic_repo import ElasticsearchKeyframeRepo
from app.repository.vector_repo import KeyframeSearchRepo, CaptionSearchRepo
from app.services.search_services import SearchService
from app.services.model_services import ModelService
from app.services.tag_services import TagService
from app.controller.search_controller import SearchController
from app.repository.chat_repo import ChatRepo
from motor.motor_asyncio import AsyncIOMotorClient
from app.models.history import SearchHistory, IntermediateResult
from app.models.som import SomFeedbackEvent, SomOverlay

from beanie import init_beanie
async def init_mongo2(db_uri: str, db_name: str):
    """
    Call this once at startup.
    """
    client = AsyncIOMotorClient(db_uri)
    db = client[db_name]
    await init_beanie(database=db, document_models=[SearchHistory, IntermediateResult, CaptionIndexMap])
    return client


class AppState:
    mongo_client: Any
    keyframe_repo: KeyframeRepo
    es_repo: ElasticsearchKeyframeRepo
    kf_search: KeyframeSearchRepo
    cap_search: CaptionSearchRepo
    search_service: SearchService
    model_service: ModelService
    tag_service: TagService
    controller: SearchController
    chat_repo: ChatRepo
    intermediate_repo: Any
    som_service: Any

async def build_app_state() -> AppState:
    state = AppState()
    
    state.mongo_client = await init_mongo(settings.mongo_uri, settings.mongo_db)
    state.keyframe_repo = KeyframeRepo()

    state.es_repo = ElasticsearchKeyframeRepo(
        hosts=settings.es_hosts,
        index=settings.es_index,
        api_key=settings.es_api_key,
        basic_auth=(settings.es_basic_user, settings.es_basic_pass),
        verify_certs=settings.es_verify_certs,
    )

    try:
        await state.es_repo.ensure_index()
    except Exception as e:
        print(f"[startup] ES ensure_index warning: {e}")
    
    state.kf_search = KeyframeSearchRepo(
        uri=settings.milvus_uri,
        collection=settings.milvus_collection_keyframe,
    )

    state.cap_search = CaptionSearchRepo(
        uri=settings.milvus_uri,
        collection=settings.milvus_collection_caption,
    )

    state.search_service = SearchService(
        keyframe_search=state.kf_search, caption_search=state.cap_search
    )

    if settings.id2tags:
        id2tags = json.load(open(settings.id2tags,'r', encoding='utf-8'))
        state.tag_service = TagService(id_to_tags=id2tags)
    
    state.model_service = ModelService(
        beit3_ckpt=settings.beit3_ckpt,
        beit3_tokenizer_path=settings.beit3_tokenizer_path,
        text_model_name=settings.st_model,
    )

    state.controller = SearchController(
        ocr_repo=state.es_repo,
        keyframe_repo=state.keyframe_repo,
        search_service=state.search_service,
        tag_service=state.tag_service,
        model_service=state.model_service,
        # som_service=state.som_service,
    )
    await init_mongo2(settings.mongo_uri, settings.mongo_db)
    state.chat_repo = ChatRepo()
    from app.repository.intermediate_repo import IntermediateRepo
    state.intermediate_repo = IntermediateRepo()

    from app.repository.som_repo import SomRepo
    from app.services.som_service import SomFeedbackService, SomConfig
    bmu_map = SomFeedbackService.load_bmu_map(settings.som_bmu_map_path)
    som_cfg = SomConfig(
        grid_h=settings.som_grid_h,
        grid_w=settings.som_grid_w,
        r=settings.som_kernel_radius,
        sigma=settings.som_kernel_sigma,
        w_pos=settings.som_w_pos,
        w_neg=settings.som_w_neg,
        alpha=settings.som_alpha,
        beta=settings.som_beta,
        gamma=settings.som_gamma,
        kappa=settings.som_kappa,
    )
    state.som_service = SomFeedbackService(bmu_map=bmu_map, repo=SomRepo(), cfg=som_cfg)


    return state

def get_controller(request: Request) -> SearchController:
    return request.app.state.controller


def get_chat_repo(request: Request) -> ChatRepo:
    return request.app.state.chat_repo

def get_intermediate_repo(request: Request):
    return request.app.state.intermediate_repo

def get_som_service(request: Request):
    return request.app.state.som_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = await build_app_state()
    app.state.mongo_client = state.mongo_client
    app.state.keyframe_repo = state.keyframe_repo
    app.state.es_repo = state.es_repo
    app.state.kf_search = state.kf_search
    app.state.cap_search = state.cap_search
    app.state.search_service = state.search_service
    app.state.model_service = state.model_service
    app.state.tag_service = state.tag_service
    app.state.controller = state.controller
    app.state.chat_repo = state.chat_repo
    app.state.som_service = state.som_service


    yield 

    try:
        await app.state.es_repo.es.close()
    except Exception:
        pass

    try:
        await app.state.kf_search.close()
    except Exception:
        pass

    try:
        await app.state.cap_search.close()
    except Exception:
        pass

    try:
        app.state.mongo_client.close()
    except Exception:
        pass

    
