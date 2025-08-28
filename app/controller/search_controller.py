from typing import Dict, List, Optional, Tuple


from app.schemas.search_queries import SingleSearchRequest, TrakeSearchRequest, FusionMethod
from app.schemas.search_results import (
    KeyframeScore,
    ModalityResult,
    SingleSearchResponse,
    TrakePath,
    TrakePathResponse,
    FusionSummary,
    RRFDetail,
    WeightedDetail, 
    MilvusSearchResponseItem
)

from app.services.fusion_services import (
    rrf_fuse,
    weighted_fuse,
    organize_and_dedup_group_video_kf,
    normalize_event_scores_kf,
    beam_sequences_single_bucket_kf,
    rerank_across_videos_kf
)


from app.repository.elastic_repo import ElasticsearchKeyframeRepo  
from app.repository.keyframe_repo import KeyframeRepo
from app.services.tag_services import TagService
from app.services.search_services import SearchService
from app.schemas.search_settings import TopKReturn, ControllerParams
from app.services.model_services import ModelService



class SearchController:
    def __init__(
        self,
        ocr_repo: ElasticsearchKeyframeRepo,
        keyframe_repo: KeyframeRepo,
        search_service: SearchService,
        tag_service: TagService,
        model_service: ModelService
        
    ):
        self.ocr_repo = ocr_repo
        self.keyframe_repo = keyframe_repo
        self.search_service = search_service
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

    async def _search_keyframe(self, text: str, topk: int, param: dict, tag_boost_alpha: float = 0.0) -> List[KeyframeScore]:
        assert self.model_service is not None and self.tag_service is not None
        emb = self.model_service.embed_text(text)
        milvus = await self.search_service.search_caption_dense(emb, topk, param)
        scored = await self._milvus_to_keyframe_score(milvus)
        if tag_boost_alpha > 0.0:
            tags = self.tag_service.scan_tags(text)
            scored = self.tag_service.rerank_keyframe_search_with_tags(tags, scored, tag_boost_alpha)
        return scored

    

    # async def _search_caption_dense(self, text: str, topk: int, param: dict, tag_boost_alpha: float = 0.0) -> List[KeyframeScore]:
    #     assert self.model_service is not None and self.tag_service is not None
    #     emb = self.model_service.embed_text(text)
    #     milvus = await self.search_service.search_caption_dense(emb, topk, param)
    #     scored = await self._milvus_to_keyframe_score(milvus)
    #     if tag_boost_alpha > 0.0:
    #         tags = self.tag_service.scan_tags(text)
    #         scored = self.tag_service.rerank_keyframe_search_with_tags(tags, scored, tag_boost_alpha)
    #     return scored

    async def _search_caption(
        self,
        text: str,
        topk: int,
        param:dict,
        tag_boost_alpha: float,
        fusion: FusionMethod,
        weighted: float | None 
    ):
        dense_emb = self.model_service.embed_text(text)
        dense_req = await self.search_service.caption_search.construct_dense_request(dense_emb, topk, param)

        milvus_hits = None
        try:
            sparse_vec = self.model_service.embed_sparse_text(text)
            sparse_req = await self.search_service.caption_search.construct_sparse_request(sparse_vec, topk, param)
            sparse_req = await self.search_service.caption_search.construct_sparse_request(sparse_vec, topk, param)
            if fusion == "weighted":
                w_dense = weighted if (weighted is not None) else 0.5
                w_sparse = 1.0 - w_dense
                milvus_hits = await self.search_service.caption_search.search_caption_hybrid(
                    dense_req=dense_req,
                    sparse_req=sparse_req,
                    rerank="weighted",
                    weights=[w_dense, w_sparse],
                )
            else:
                milvus_hits = await self.search_service.caption_search.search_caption_hybrid(
                    dense_req=dense_req,
                    sparse_req=sparse_req,
                    rerank="rrf",
                )
        except NotImplementedError:
            milvus_hits = await self.search_service.search_caption_dense(dense_emb, topk, param)

        scored = await self._milvus_to_keyframe_score(milvus_hits)
        if tag_boost_alpha > 0.0:
            tags = self.tag_service.scan_tags(text)
            scored = self.tag_service.rerank_keyframe_search_with_tags(tags, scored, tag_boost_alpha)
        return scored

    async def _search_ocr(self, text: str, topk: int) -> List[KeyframeScore]:
        assert self.ocr_repo is not None, "ocr_repo not set"
        return await self.ocr_repo.search(query_text=text, top_k=topk)





    async def single_search(self, req: SingleSearchRequest, topk: TopKReturn, ctrl: ControllerParams) -> SingleSearchResponse:
        per_modality: list[ModalityResult] = []
        lists_in_order: List[List[KeyframeScore]] = []

        if req.keyframe:
            kf = await self._search_keyframe(req.keyframe.text, topk.topk_visual, ctrl.kf_search_param, req.keyframe.tag_boost_alpha)
            per_modality.append(ModalityResult(modality="keyframe", items=kf))
            lists_in_order.append(kf)
        
        fusion_method = 'rrf'
        if req.caption:
            cap = await self._search_caption(
                text=req.caption.text,
                topk=topk.topk_caption,
                param=ctrl.cap_search_param,
                tag_boost_alpha=req.caption.tag_boost_alpha,
                fusion=req.caption.fusion,
                weighted=req.caption.weighted
            )
            lists_in_order.append(cap)
            fusion_method = req.caption.fusion
        
        if req.ocr:
            ocr = await self._search_ocr(req.ocr.text, topk.topk_ocr)
            per_modality.append(ModalityResult(modality="ocr", items=ocr))
            lists_in_order.append(ocr)

        if not any(lists_in_order):
            return SingleSearchResponse(fused=[], per_modality=[], fusion=FusionSummary(method="rrf", detail=RRFDetail(k=60)), meta={})

        if fusion_method == "weighted":
            wv, wc, wo = ctrl.fusion.ensure_scale()
            used_weights: List[float] = []
            for m in [pm.modality for pm in per_modality]:
                used_weights.append({"keyframe": wv, "caption": wc, "ocr": wo}[m])
            fused = weighted_fuse(lists_in_order, used_weights)
            fusion_summary = FusionSummary(method="weighted", detail=WeightedDetail(weights=used_weights, norm_score=True))
        else:
            fused = rrf_fuse(lists_in_order, k=60)
            fusion_summary = FusionSummary(method="rrf", detail=RRFDetail(k=60))

        fused = fused[:topk.final_topk]
        return SingleSearchResponse(fused=fused, per_modality=per_modality, fusion=fusion_summary, meta={})
    

    
    async def trake_search(
        self,
        req: TrakeSearchRequest,
        *,
        topk: TopKReturn,
        ctrl: ControllerParams,
        window: int = 6,
        beam_size: int = 50,
        per_bucket_top_k: Optional[int] = None,
        global_top_k: Optional[int] = 20,
        norm_method: str = "zscore",
        norm_temperature: float = 1.0,
        per_event_cap: Optional[int] = None
    ) -> tuple[TrakePathResponse, list[list[KeyframeScore]]]:
        """
        For each EventQuery (sorted by event_order):
          1) Run single_search for the event query.
          2) Build the event's candidate pool by UNION of per-modality lists (no cross-event fusion).
          3) Group by (group_id, video_id), de-dup per event (window).
          4) Normalize per event, beam search per bucket (temporal prior).
          5) Global rerank across buckets and return top paths.

        Return
            - The TrakePathResponse
            - The list[list[keyframe_score]] with raw score
        """

        events_sorted = sorted(req.events, key=lambda e: e.event_order)
        raw_hits_per_event: list[list[KeyframeScore]] = []

        for ev in events_sorted:
            single = await self.single_search(ev.query, topk, ctrl)
            fused_list: List[KeyframeScore] = single.fused
            if per_event_cap is not None and per_event_cap > 0:
                fused_list = fused_list[:per_event_cap]

            raw_hits_per_event.append(fused_list)
        
        by_group_video = organize_and_dedup_group_video_kf(raw_hits_per_event, window=window)

        by_bucket_paths: Dict[Tuple[str, str], List[Tuple[List[KeyframeScore], float]]] = {}
        for bucket, event_lists in by_group_video.items():
            norm_lists = normalize_event_scores_kf(
                event_lists,
                method=norm_method,
                temperature=norm_temperature,
            )
            paths = beam_sequences_single_bucket_kf(
                event_lists=norm_lists,
                beam_size=beam_size,
                K=per_bucket_top_k,
            )
            if paths:
                by_bucket_paths[bucket] = paths
        
        reranked = rerank_across_videos_kf(by_bucket_paths, top_k=global_top_k)
        trake_paths: list[TrakePath] = [
            TrakePath(items=path, score=score) for path, score in reranked
        ]
        return TrakePathResponse(paths=trake_paths, meta={}), raw_hits_per_event
