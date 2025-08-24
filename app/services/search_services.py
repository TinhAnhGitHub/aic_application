from app.repository.vector_repo import KeyframeSearch, CaptionSearch
from app.schemas.application import EventOrder, EventHit
from typing import Literal, Optional, Sequence, Callable
import heapq
from statistics import mean, pstdev



def _dedup_hits(
    hits: list[EventHit],
    window: int = 6,
):
    if not hits:
        return []
    def _pos(h: EventHit)-> int:
        return int(h.keyframe_id)

    hits = sorted(hits, key=lambda h: int(h.keyframe_id))


    kept: list[EventHit]= []
    i = 0
    n = len(hits)
    while i < n:
        start_pos = _pos(hits[i])
        j = i
        segment = []
        while j < n and _pos(hits[j]) < start_pos + window:
            segment.append(hits[j])
            j += 1
        
        best = max(segment, key=lambda x: x.score)
        kept.append(best)

        while j < n and (_pos(hits[j]) - start_pos) < window:
            j += 1
        i = j
    return kept


def organize_and_dedup_group_video(
    hits: list[list[EventHit]], # [hits for event 0, hits for event 1, ...]
    window: int = 6
):
    
    """
    Input shape: 
        raw_hits_per_event: list[list[EventHit]], # [hits for E0, hits for E1, ...]
    
    Output shape:
        by_group_video: dict[tuple, list[list[EventHit]]], # (group_id, video_id) -> [ [hits for E0, hits for E1, ...] ]
    we keep  (groupid, videoid) that can have less than the amount of events. For example, there might be a pair that has only 3 events matched, while the query has 5 events.
    """
    
    if not hits:
        return {}
    for i, hit in enumerate(hits):
        if any(h.event.order != i for h in hit):
            raise ValueError("Event order must be in increment order")
    


    T = len(hits)

    # assume we have 4 events
    # tmp[(group_id, video_id)][event_index] = [EventHit0, EventHit1, EventHit2, EventHit3]
    # tmp[(group_id, video_id)][event_index] = [EventHit1, EventHit2, EventHit3]
    # tmp[(group_id, video_id)][event_index] = [EventHit0, EventHit2, EventHit3]
    # tmp[(group_id, video_id)][event_index] = [EventHit0, EventHit1]
    tmp: dict[tuple[str, str], dict[int, list[EventHit]]] = {}

    for event_index, event_hits in enumerate(hits):
        for h in event_hits:
            key = (h.group_id, h.video_id)
            if key not in tmp:
                tmp[key] = {}
            tmp[key].setdefault(event_index, []).append(h)
    
    by_group_video: dict[tuple[str, str], list[list[EventHit]]] = {}
    for gr_vid_key, per_event in tmp.items():
        dedup_lists: list[list[EventHit]] = []
        completed=True
        for e_idx in range(T):
            ev_list = per_event.get(e_idx, [])
            if len(ev_list) == 0:
                print(f"Warning: {gr_vid_key} event list is empty")
            ev_list = _dedup_hits(ev_list, window)
            if len(ev_list) == 0:
                completed=False
                break
            ev_list.sort(key=lambda x: x.score, reverse=True)
            dedup_lists.append(ev_list)
        if completed:
            by_group_video[gr_vid_key] = dedup_lists
    return by_group_video






def beam_sequences_single_bucket(
    event_lists: list[list[EventHit]], # for one video: [hits for E0, hits for E1, ...], all non-empty. It should be same video id, group id
    K: int | None = 5,
    beam_size: int = 50,
    trans_sigma: float = 1.5 * 6,
    trans_weight: float = 0.6,
):
    # ensure the same video_id, group_id
    

    def temporal_prior(prev: EventHit, curr: EventHit) -> float:
        gap = int(curr.keyframe_id) - int(prev.keyframe_id)
        return - (gap * gap) / (2 * trans_sigma * trans_sigma) * trans_weight

    first = event_lists[0] # event hit A

    beam = [(- (h.score) , [h]) for h in first] # (neg log prob, [sequence])
    
    heapq.heapify(beam)
    beam = heapq.nsmallest(beam_size, beam)

    for event_index in range(1, len(event_lists)):
        nxt = []
        for negative_score, path in beam:
            prev = path[-1]
            base = -negative_score

            for cur in event_lists[event_index]:
                if int(cur.keyframe_id) <= int(prev.keyframe_id):
                    continue
                
                new_score = base + cur.score + temporal_prior(prev, cur)
                heapq.heappush(nxt, (-new_score, path + [cur]))
        if not nxt:
            return []
    
        beam = heapq.nsmallest(beam_size, nxt)
    

    
    if K is None:
        return [(path, -negative_score) for (negative_score, path) in beam]
    else:
        topK = heapq.nsmallest(min(K, len(beam)), beam)
        return [(path, -negative_score) for (negative_score, path) in topK]



def _clone_with_score(h: EventHit, new_score: float) -> EventHit:
    obj = h.model_copy(deep=True)
    obj.score = float(new_score)
    return obj

def _normalize_event_scores(
    event_lists: list[list[EventHit]],
    method: str = "zscore",   # "zscore" | "minmax"
    eps: float = 1e-6,
    temperature: float = 1.0,
):
    """
    Norm for each event in the same group,video
    """
    norm_lists: list[list[EventHit]] = []
    for ev_hits in event_lists:
        scores = [h.score for h in ev_hits]
        if method=='zscore':
            mu = mean(scores)
            sd = pstdev(scores) if len(scores) > 1 else 0.0
            normed = [ (s - mu) / (sd + eps) for s in scores ]
        else:  
            lo, hi = min(scores), max(scores)
            normed = [ (s - lo) / (max(hi - lo, eps)) for s in scores ]
        if temperature and temperature != 1.0:
            normed = [ s / float(temperature) for s in normed ]
        norm_hits = [_clone_with_score(h, s) for h, s in zip(ev_hits, normed)]
        norm_hits.sort(key=lambda x: x.score, reverse=True)
        norm_lists.append(norm_hits)
    return norm_lists


def rerank_across_videos(
    by_bucket_paths: dict[tuple[str, str], list[tuple[list[EventHit], float]]],
    top_k: int | None = None
):
    flat: list[tuple[list[EventHit], float]] = []
    for _, paths in by_bucket_paths.items():
        flat.extend(paths)
    
    flat.sort(key=lambda x: x[1], reverse=True)
    return flat if top_k is None else flat[:top_k]



class SearchService:
    def __init__(
        self,
        keyframe_search: KeyframeSearch,
        caption_search: CaptionSearch,
    ):
        self.keyframe_search = keyframe_search
        self.caption_search = caption_search

    
    async def search_keyframe_dense(self, query_embedding: list[float], top_k: int, param: dict, **kwargs):
        return await self.keyframe_search.search_dense(query_embedding, top_k, param, **kwargs)
    
    async def search_caption_dense(self, query_embedding: list[float], top_k: int, param: dict, **kwargs):
        return await self.caption_search.search_dense(query_embedding, top_k, param, **kwargs)

    async def search_caption_hybrid(
        self,
        dense_req,
        sparse_req,
        rerank: Literal["rrf", "weighted"] = "rrf",
        weights: Optional[Sequence[float]] = None,
    ):
        return await self.caption_search.search_combination(
            requests=[dense_req, sparse_req],
            rerank=rerank,
            weights=weights,
        )


    async def seach_keyframe_caption_hybrid(
        self,
        kf_req,
        caption_dense_req,
        caption_sparse_req,
        rerank: Literal["rrf", "weighted"] = "rrf",
        weights: Optional[Sequence[float]] = None,
    ):
        return await self.keyframe_search.search_combination(
            requests=[kf_req, caption_dense_req, caption_sparse_req],
            rerank=rerank,
            weights=weights,
        )
    

    def trake_search(
        self,
        raw_hits_per_event: list[list[EventHit]],
        window: int, 
        beam_size: int = 50,
        per_bucket_top_k: int | None = None,
        global_top_k: int | None = None,
        norm_method: str = "zscore",
        norm_temperature: float = 1.0,
    ) -> list[tuple[list[EventHit], float]]:
        
        by_group_video = organize_and_dedup_group_video(raw_hits_per_event, window)

        by_bucket_paths: dict[tuple[str, str], list[tuple[list[EventHit], float]]] = {}
        for bucket, event_lists in by_group_video.items():
            norm_lists = _normalize_event_scores(
                event_lists,
                method=norm_method,
                temperature=norm_temperature
            )
            paths = beam_sequences_single_bucket(
                event_lists=norm_lists,
                beam_size=beam_size,
                K=per_bucket_top_k,
            )
            if paths:
                by_bucket_paths[bucket] = paths
        
        reranked = rerank_across_videos(by_bucket_paths, top_k=global_top_k)
        return reranked




    