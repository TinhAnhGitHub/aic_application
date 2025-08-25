from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast
import math
from statistics import mean, pstdev
import asyncio
from typing import Literal

from app.schemas.application import (
    EventSearch,
    EventHit,
    KeyframeScore,
    MilvusSearchResponseItem,
        
)


from app.repository.elastic_repo import ElasticsearchKeyframeRepo  
from app.repository.keyframe_repo import KeyframeRepo
from app.services.tag_services import TagService
from app.services.search_services import SearchService
from app.schemas.search_settings import FusionWeights, TopKReturn, ControllerParams
from app.services.model_services import ModelService
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
        model_service: ModelService
        
    ):
        self.ocr_repo = ocr_repo
        self.keyframe_repo = keyframe_repo
        self.search_embed = search_embed
        self.tag_service = tag_service
        self.model_service = model_service

    
    async def _milvus_to_keyframe_score(
        self,
        result: list[MilvusSearchResponseItem]
    ) -> list[KeyframeScore]:
        identifications = [str(r.identification) for r in result]
        keyframes = await self.keyframe_repo.get_many_by_identifications(identifications)
        id_to_kf = {kf.identification: kf for kf in keyframes}
        scores = []
        for r in result:
            kf = id_to_kf.get(r.identification)
            if kf:
                scores.append(
                    KeyframeScore(
                        identification=kf.identification,
                        group_id=kf.group_id,
                        video_id=kf.video_id,
                        keyframe_id=kf.keyframe_id,
                        tags=kf.tags,
                        ocr=kf.ocr,
                        score=r.score
                    )
                )
        return scores    

    async def keyframe_search(self, keyframe_search: KeyframeSearch, topk_visual: int, kf_search_param: dict) -> list[KeyframeScore]:
        
        keyframe_search_text =  cast(str, keyframe_search.keyframe_search_text)
        embedding_text = self.model_service.embed_text(
            keyframe_search_text
        )

        tags = None
        if keyframe_search.tag_boost_alpha > 0.0:
            ## Using tags as a boost
            tags = self.tag_service.scan_tags(
                user_query=keyframe_search_text,
            )

        # keyframe search 
        results = await self.search_embed.search_keyframe_dense(
            query_embedding=embedding_text,
            top_k=topk_visual,
            param=kf_search_param,
        )

        results = await self._milvus_to_keyframe_score(results)

        if tags:
            results = self.tag_service.rerank_keyframe_search_with_tags(
                tags=tags,
                results_search=results,
                alpha=keyframe_search.tag_boost_alpha
            )

        return results


    async def caption_search(self, caption_search: CaptionSearch, topk_caption: int, cap_search_param: dict) -> list[KeyframeScore]:
        caption_search_text = cast(str, caption_search.caption_search_text)

        embedding_text = self.model_service.embed_text(
            caption_search_text
        )
        tags = None
        if caption_search.tag_boost_alpha > 0.0:
            tags = self.tag_service.scan_tags(
                user_query=caption_search_text,
            )

        # caption search
        results = await self.search_embed.search_caption_dense(
            query_embedding=embedding_text,
            top_k=topk_caption,
            param=cap_search_param,
        )
        results = await self._milvus_to_keyframe_score(results)

        if tags:
            results = self.tag_service.rerank_keyframe_search_with_tags(
                tags=tags,
                results_search=results,
                alpha=caption_search.tag_boost_alpha
            )
        return results

    async def ocr_search(self, ocr_search: OCRSearch, topk_ocr: int) -> list[KeyframeScore]:
        results = await self.ocr_repo.search(
            query_text=ocr_search.list_ocr,
            top_k=topk_ocr,
        )
        return results

    def reciprocal_ranking(
        self,
        keyframe_results: list[KeyframeScore],
        caption_results: list[KeyframeScore],
        ocr_results: list[KeyframeScore],
    ):
        K = 60
        def ranks(items: list[KeyframeScore]) -> dict[int,int]:
            if not items:
                return {}

            ordered = sorted(items, key=lambda x: x.score, reverse=True)
            return{
                it.identification: idx + 1 for idx, it in enumerate(ordered)
            }
        
        kf_ranks = ranks(keyframe_results)
        cap_ranks = ranks(caption_results)
        ocr_ranks = ranks(ocr_results)

        rep: dict[int, KeyframeScore] = {}

        def get_rep(rep: dict[int, KeyframeScore], it: KeyframeScore):
            if it.identification not in rep:
                rep[it.identification] = it
            

        for it in keyframe_results:
            get_rep(rep, it)
        for it in caption_results:
            get_rep(rep, it)
        for it in ocr_results: 
            get_rep(rep, it)
        
        all_ids: set[int] = set(kf_ranks) | set(cap_ranks) | set(ocr_ranks)
        fused: list[KeyframeScore] = []

        for ident in all_ids:
            s = 0.0
            if ident in kf_ranks:
                s += 1.0 / (K + kf_ranks[ident])
            
            if ident in cap_ranks:
                s += 1.0 / (K + cap_ranks[ident])
            
            if ident in ocr_ranks:
                s += 1.0 / (K + ocr_ranks[ident])
            
            r = rep[ident]
            fused.append(
                KeyframeScore(
                    identification=r.identification,
                    group_id=r.group_id,
                    video_id=r.video_id,
                    keyframe_id=r.keyframe_id,
                    tags=r.tags,
                    ocr=r.ocr,
                    score=s,
                )
            )
        
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused



    def weighted_ranking(
        self,
        keyframe_results: list[KeyframeScore],
        caption_results: list[KeyframeScore],
        ocr_results: list[KeyframeScore],
        weights: FusionWeights
    ):
        def ranks(items: list[KeyframeScore]) -> dict[int, float]:
            if not items:
                return {}

            scores = [it.score for it in items]
            mu = mean(scores)
            sigma = pstdev(scores) if pstdev(scores) > 1e-6 else 1.0

            return {
                it.identification: (it.score - mu) / sigma for it in items
            }
        
        kf_ranks = ranks(keyframe_results)
        cap_ranks = ranks(caption_results)
        ocr_ranks = ranks(ocr_results)

        fused: dict[int, KeyframeScore] = {}

        def get_fused(fused: dict[int, KeyframeScore], it: KeyframeScore):
            if it.identification not in fused:
                fused[it.identification] = it
            

        for it in keyframe_results:
            get_fused(fused, it)
        for it in caption_results:
            get_fused(fused, it)
        for it in ocr_results: 
            get_fused(fused, it)
        
        all_ids: set[int] = set(kf_ranks) | set(cap_ranks) | set(ocr_ranks)
        out: list[KeyframeScore] = []

        for ident in all_ids:
            s = 0.0
            if ident in kf_ranks:
                s += weights.w_visual * kf_ranks[ident]
            
            if ident in cap_ranks:
                s += weights.w_caption * cap_ranks[ident]
            
            if ident in ocr_ranks:
                s += weights.w_ocr * ocr_ranks[ident]
            
            r = fused[ident]
            out.append(
                KeyframeScore(
                    identification=r.identification,
                    group_id=r.group_id,
                    video_id=r.video_id,
                    keyframe_id=r.keyframe_id,
                    tags=r.tags,
                    ocr=r.ocr,
                    score=s,
                )
            )
        
        out.sort(key=lambda x: x.score, reverse=True)
        return out  
    



    async def single_search(
        self,
        keyframe_search: Optional[KeyframeSearch],
        caption_search: Optional[CaptionSearch],
        ocr_search: Optional[OCRSearch],
        top_k_return: TopKReturn,
        fusion_weights: FusionWeights | None,
        controller_params: ControllerParams,
    ) -> List[KeyframeScore]:
        tasks = []
        if keyframe_search:
            tasks.append(
                self.keyframe_search(
                    keyframe_search=keyframe_search,
                    topk_visual=top_k_return.topk_visual,
                    kf_search_param=controller_params.kf_search_param or {}
                )
            )
        else:
            tasks.append(asyncio.sleep(0, result=[]))

        if caption_search:
            tasks.append(
                self.caption_search(
                    caption_search=caption_search,
                    topk_caption=top_k_return.topk_caption,
                    cap_search_param=controller_params.cap_search_param or {}
                )
            )
        else:
            tasks.append(asyncio.sleep(0, result=[]))

        if ocr_search:
            tasks.append(
                self.ocr_search(
                    ocr_search=ocr_search,
                    topk_ocr=top_k_return.topk_ocr
                )
            )
        else:
            tasks.append(asyncio.sleep(0, result=[]))

        keyframe_results, caption_results, ocr_results = await asyncio.gather(*tasks)

        if fusion_weights is None:
            fused = self.reciprocal_ranking(
                keyframe_results=keyframe_results,
                caption_results=caption_results,
                ocr_results=ocr_results
            )
        else:
            fused = self.weighted_ranking(
                keyframe_results=keyframe_results,
                caption_results=caption_results,
                ocr_results=ocr_results,
                weights=fusion_weights
            )
        
        
            fused = fused[:top_k_return.final_topk]
        
        return fused
    

    async def trake_search(
        self, 
        events: list[EventSearch],
    ):
        all_results: list[list[EventHit]] = []
        for event in events:
            


            
            
        



