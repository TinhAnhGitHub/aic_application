from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from statistics import pstdev, mean

from app.schemas.search_results import KeyframeScore

def rrf_fuse(
    per_lists:list[list[KeyframeScore]],
    k: int = 60
):
    rank_maps: list[dict[int,int]] = []
    rep: dict[int, KeyframeScore] = {}

    for items in per_lists:
        ordered = sorted(items, key=lambda x: x.score, reverse=True)
        rank_maps.append(
            {it.identification: idx + 1 for idx, it in enumerate(ordered)}
        )
        for it in items:
            if it.identification not in rep:
                rep[it.identification] = it
    
    all_ids = set().union(*(set(m) for m in rank_maps)) if rank_maps else set()
    fused: List[KeyframeScore] = []

    for ident in all_ids:
        s = 0.0
        for m in rank_maps:
            if ident in m:
                s += 1.0 / (k + m[ident])
        r = rep[ident]
        fused.append(r.__class__(**{**r.model_dump(), "score": float(s)}))
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


def weighted_fuse(
    per_lists: list[list[KeyframeScore]],
    weights: Sequence[float],
) -> list[KeyframeScore]:
    
    assert len(per_lists) == len(weights), "weights length must match number of lists"
    z_maps: List[Dict[int, float]] = []
    rep: Dict[int, KeyframeScore] = {}

    for items in per_lists:
        for it in items:
            rep.setdefault(it.identification, it)
    
    for items in per_lists:
        if not items:
            z_maps.append({})
            continue
        scores = [it.score for it in items]
        mu = mean(scores)
        sd_raw = pstdev(scores) if len(scores) > 1 else 0.0
        sd = sd_raw if sd_raw > 1e-6 else 1.0
        z_maps.append({it.identification: (it.score - mu) / sd for it in items})
    
    all_ids = set().union(*(set(m) for m in z_maps)) if z_maps else set()
    out: List[KeyframeScore] = []
    for ident in all_ids:
        s = 0.0
        for w, m in zip(weights, z_maps):
            if ident in m:
                s += float(w) * m[ident]
        r = rep[ident]
        out.append(r.__class__(**{**r.model_dump(), "score": float(s)}))

    out.sort(key=lambda x: x.score, reverse=True)
    return out




def _kf_pos(h: KeyframeScore)->int:
    return int(h.keyframe_id)


def _dedup_hits_kf(
    hits: list[KeyframeScore], window: int = 6
)-> list[KeyframeScore]:
    if not hits: 
        return []

    hits = sorted(hits, key=lambda h: _kf_pos(h))
    kept: list[KeyframeScore] = []
    i, n = 0, len(hits)

    while i < n:
        start = _kf_pos(hits[i])
        j = i
        segment: list[KeyframeScore] = []
        while j < n and _kf_pos(hits[j]) < start + window:
            segment.append(hits[j])
            j += 1

        best = max(segment, key=lambda x: x.score)
        kept.append(best)
        while j < n and (_kf_pos(hits[j]) - start) < window:
            j += 1
        i = j
    return kept


def organize_and_dedup_group_video_kf(
    hits_per_event: List[List[KeyframeScore]],
    window: int = 6,
) -> Dict[Tuple[str, str], List[List[KeyframeScore]]]:
    if not hits_per_event:
        return {}
    T = len(hits_per_event)

    tmp: Dict[Tuple[str, str], Dict[int, List[KeyframeScore]]] = {}
    for event_index, event_hits in enumerate(hits_per_event):
        for h in event_hits:
            key = (h.group_id, h.video_id)
            tmp.setdefault(key, {}).setdefault(event_index, []).append(h)

    by_group_video: Dict[Tuple[str, str], List[List[KeyframeScore]]] = {}
    for bucket, per_event in tmp.items():
        dedup_lists: List[List[KeyframeScore]] = []
        completed = True
        for e_idx in range(T):
            ev_list = per_event.get(e_idx, [])
            ev_list = _dedup_hits_kf(ev_list, window)
            if not ev_list:
                completed = False
                break
            ev_list.sort(key=lambda x: x.score, reverse=True)
            dedup_lists.append(ev_list)
        if completed:
            by_group_video[bucket] = dedup_lists
    return by_group_video

def _clone_with_score_kf(h: KeyframeScore, new_score: float) -> KeyframeScore:
    return h.__class__(**{**h.model_dump(), "score": float(new_score)})



def normalize_event_scores_kf(
    event_lists: List[List[KeyframeScore]],
    method: str = "zscore",  # "zscore" | "minmax"
    eps: float = 1e-6,
    temperature: float = 1.0,
) -> List[List[KeyframeScore]]:
    """
    Normalize scores within each event list (per bucket).
    """
    norm_lists: List[List[KeyframeScore]] = []
    for ev_hits in event_lists:
        scores = [h.score for h in ev_hits]
        if not scores:
            norm_lists.append([])
            continue
        if method == "zscore":
            mu = mean(scores)
            sd = pstdev(scores) if len(scores) > 1 else 0.0
            sd = sd if sd > eps else 1.0
            normed = [(s - mu) / sd for s in scores]
        else:
            lo, hi = min(scores), max(scores)
            rng = hi - lo
            rng = rng if rng > eps else 1.0
            normed = [(s - lo) / rng for s in scores]
        if temperature and temperature != 1.0:
            t = float(temperature)
            normed = [s / t for s in normed]
        norm_hits = [_clone_with_score_kf(h, s) for h, s in zip(ev_hits, normed)]
        norm_hits.sort(key=lambda x: x.score, reverse=True)
        norm_lists.append(norm_hits)
    return norm_lists



def beam_sequences_single_bucket_kf(
    event_lists: List[List[KeyframeScore]],   # one bucket: [E0 list, E1 list, ...], all non-empty
    K: Optional[int] = 5,
    beam_size: int = 50,
    trans_sigma: float = 1.5 * 6,
    trans_weight: float = 0.6,
) -> List[Tuple[List[KeyframeScore], float]]:
    """
    Beam search over ordered events for a single (group_id, video_id) bucket.
    Enforces strictly increasing keyframe_id and adds Gaussian temporal prior.
    """
    import heapq

    def temporal_prior(prev: KeyframeScore, curr: KeyframeScore) -> float:
        gap = _kf_pos(curr) - _kf_pos(prev)
        return - (gap * gap) / (2 * trans_sigma * trans_sigma) * trans_weight

    first = event_lists[0]
    beam: List[Tuple[float, List[KeyframeScore]]] = [(-h.score, [h]) for h in first]
    heapq.heapify(beam)
    beam = heapq.nsmallest(beam_size, beam)

    for idx in range(1, len(event_lists)):
        nxt: List[Tuple[float, List[KeyframeScore]]] = []
        for neg, path in beam:
            prev = path[-1]
            base = -neg
            for cur in event_lists[idx]:
                if _kf_pos(cur) <= _kf_pos(prev):
                    continue
                new_score = base + cur.score + temporal_prior(prev, cur)
                heapq.heappush(nxt, (-new_score, path + [cur]))
        if not nxt:
            return []
        beam = heapq.nsmallest(beam_size, nxt)

    if K is None:
        return [(path, -neg) for (neg, path) in beam]
    topK = min(K, len(beam))
    best = heapq.nsmallest(topK, beam)
    return [(path, -neg) for (neg, path) in best]



def rerank_across_videos_kf(
    by_bucket_paths: Dict[Tuple[str, str], List[Tuple[List[KeyframeScore], float]]],
    top_k: Optional[int] = None
) -> List[Tuple[List[KeyframeScore], float]]:
    flat: List[Tuple[List[KeyframeScore], float]] = []
    for _, paths in by_bucket_paths.items():
        flat.extend(paths)
    flat.sort(key=lambda x: x[1], reverse=True)
    return flat if top_k is None else flat[:top_k]