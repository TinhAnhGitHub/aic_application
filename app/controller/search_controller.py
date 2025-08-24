from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
from statistics import mean, pstdev
import asyncio
from typing import Literal

from app.schemas.application import (
    EventOrder,
    EventHit,
    KeyframeSearchMilvusResponseItem,
)

from app.models.common import KeyframeModel
from app.repository.vector_repo import KeyframeSearch, CaptionSearch
from app.repository.elastic_repo import ElasticsearchKeyframeRepo  
from app.repository.keyframe_repo import KeyframeRepo
from app.services.tag_services import TagService
from app.services.search_services import SearchService
from app.schemas.search_settings import FusionWeights, TopKReturn, ControllerParams

from app.models.common import CaptionSearch, KeyframeSearch, OCRSearch



def _log1p_if_positive(x: float) -> float:
    return math.log1p(max(x, 0.0))


class SearchController:
    def __init__(
        self,
        ocr_repo: ElasticsearchKeyframeRepo,
        keyframe_repo: KeyframeRepo,
        search_embed: SearchService,
        tag_service: TagService,
        
    ):
        self.ocr_repo = ocr_repo
        self.keyframe_repo = keyframe_repo
        self.search_embed = search_embed
        self.tag_service = tag_service
    

    async def single_search(
        self,
        keyframe_search: KeyframeSearch | None,
        caption_search: CaptionSearch | None,
        ocr_search: OCRSearch | None,
        rerank: Literal['rrf', 'weigted'],
        weights: list[float] | None
    ):
        tasks = []

        if keyframe_search:
            
            
        



