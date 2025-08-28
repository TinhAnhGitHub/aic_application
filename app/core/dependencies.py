from contextlib import asynccontextmanager
from typing  import Any, Dict, Optional
import json

from fastapi import Request, FastAPI

from app.core.config import settings
from app.repository.keyframe_repo import init_mongo, KeyframeRepo
from app.repository.elastic_repo import ElasticsearchKeyframeRepo
from app.repository.vector_repo import KeyframeSearchRepo, CaptionSearchRepo
from app.services.search_services import SearchService
from app.services.model_services import ModelService
from app.services.sparse_encoder import MilvusSparseEncoder
from app.services.tag_services import TagService
from app.controller.search_controller import SearchController


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

    sparse_encoder: MilvusSparseEncoder | None = None   
    if settings.bm25_model_path:
        sparse_encoder = MilvusSparseEncoder(
            language=settings.bm25_language,
            model_state_path=settings.bm25_model_path,
        )

    tags = []
    if settings.tags_path:
        with open(settings.tags_path, "r", encoding="utf-8") as f:
            tags = [line.strip() for line in f if line.strip()]
        
    state.tag_service = TagService(tag_list=tags)
    
    state.model_service = ModelService(
        beit3_ckpt=settings.beit3_ckpt,
        beit3_tokenizer_path=settings.beit3_tokenizer_path,
        text_model_name=settings.st_model,
        sparse_encoder=sparse_encoder,
    )

    state.controller = SearchController(
        ocr_repo=state.es_repo,
        keyframe_repo=state.keyframe_repo,
        search_service=state.search_service,
        tag_service=state.tag_service,
        model_service=state.model_service,
    )


    return state

def get_controller(request: Request) -> SearchController:
    return request.app.state.controller



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
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

    