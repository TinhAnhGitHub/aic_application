from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from statistics import pstdev, mean
import math
from app.schemas.search_results import KeyframeScore
import heapq

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
    rep: Dict[int, KeyframeScore] = {}
    z_maps: List[Dict[int, float]] = []

    for items in per_lists:
        for it in items:
            rep.setdefault(it.identification, it)

    for items in per_lists:
        if not items:
            z_maps.append({})
            continue
        scores = [it.score for it in items]
        mu = mean(scores)
        sd = pstdev(scores) if len(scores) > 1 else 0.0
        sd = sd if sd > 1e-6 else 1.0
        normed = [(s - mu) / sd for s in scores]
        z_maps.append({it.identification: n for it, n in zip(items, normed)})

    all_ids = set().union(*(set(m) for m in z_maps)) if z_maps else set()
    out: list[KeyframeScore] = []
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
    if 'frame' in h.keyframe_id:
        return int(h.keyframe_id.split('_')[1])
    return int(h.keyframe_id)


# def _dedup_hits_kf(
#     hits: list[KeyframeScore], window: int = 6
# )-> list[KeyframeScore]:
#     if not hits: 
#         return []

#     hits = sorted(hits, key=lambda h: _kf_pos(h))
#     kept: list[KeyframeScore] = []
#     i, n = 0, len(hits)

#     while i < n:
#         start = _kf_pos(hits[i])
#         j = i
#         segment: list[KeyframeScore] = []
#         while j < n and _kf_pos(hits[j]) < start + window:
#             segment.append(hits[j])
#             j += 1

#         best = max(segment, key=lambda x: x.score)
#         kept.append(best)
#         while j < n and (_kf_pos(hits[j]) - start) < window:
#             j += 1
#         i = j
#     return kept


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
            # ev_list = _dedup_hits_kf(ev_list, window)
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
    bucket: tuple[str,str],
    fps_map: dict[str, float],
    K: Optional[int] = 5,
    beam_size: int = 50,
    mu_s: float = 0.0,
    sigma_s: float = 3.0,
    W: float = 0.08,
    gap_cap_s: float = 10.0

) -> List[Tuple[List[KeyframeScore], float]]:
    """
    Beam search over ordered events for a single (group_id, video_id) bucket.
    Enforces strictly increasing keyframe_id and adds Gaussian temporal prior.
    """
    import heapq
    def kf_pos(h: KeyframeScore) -> int:
        return int(h.keyframe_id.split('_')[1])

    
    key_fps = f"{bucket[0]}_{bucket[1]}.mp4"
    fps = fps_map[key_fps]
    def temporal_bonus_seconds(prev: KeyframeScore, curr: KeyframeScore) -> float:
        gap_frames = kf_pos(curr) - kf_pos(prev)
        gap_s = gap_frames / fps
        return float(W * math.exp(- ((gap_s - mu_s) ** 2) / (2.0 * sigma_s * sigma_s)))
    


    first = event_lists[0]
    seq = 0
    def _push(heap, neg_score: float, path: List[KeyframeScore]):
        nonlocal seq
        heapq.heappush(heap, (neg_score, seq, path))
        seq += 1
    

    beam: List[Tuple[float, int, List[KeyframeScore]]] = []
    for h in first:
        _push(beam, -h.score, [h])
    beam = heapq.nsmallest(beam_size, beam)


    for idx in range(1, len(event_lists)):
        nxt: List[Tuple[float, int, List[KeyframeScore]]] = []
        for neg, _, paths in beam:
            prev = paths[-1]
            base = -neg
            for cur in event_lists[idx]:
                if _kf_pos(cur) <= _kf_pos(prev):
                    continue
                new_score = base + cur.score + temporal_bonus_seconds(prev, cur)
                _push(nxt, -new_score, paths + [cur])
        if not nxt:
            return []
        beam = heapq.nsmallest(beam_size, nxt)

    if K is None:
        return [(path, -neg) for (neg,_,path) in beam]
    topK = min(K, len(beam))
    best = heapq.nsmallest(topK, beam)
    return [(path, -neg) for (neg, _,path) in best]



def kbest_viterbi_paths_single_bucket_kf(
    event_lists: List[List[KeyframeScore]],   # one bucket: [E0 list, E1 list, ...], all non-empty
    bucket: tuple[str, str],
    fps_map: dict[str, float],
    K: Optional[int] = 5,                     # number of paths to return (global top-K)
    mu_s: float = 0.0,                        # Gaussian prior mean (seconds)
    sigma_s: float = 3.0,                     # Gaussian prior std (seconds)
    W: float = 0.08,                          # weight for temporal bonus
    gap_cap_s: float = 10.0,                  # clamp overly large gaps
    per_state_k: Optional[int] = None,        # keep top-k partial paths per node (defaults to K)
) -> List[Tuple[List[KeyframeScore], float]]:
    
    if not event_lists or any(not l for l in event_lists):
        return []

    key_with_ext = f"{bucket[0]}_{bucket[1]}.mp4"
    fps = fps_map.get(key_with_ext) or fps_map.get(f"{bucket[0]}_{bucket[1]}") or 30.0

    def temporal_bonus(prev: KeyframeScore, curr: KeyframeScore) -> float:
        gap_frames = _kf_pos(curr) - _kf_pos(prev)
        if gap_frames <= 0:
            return -1e9  
        gap_s = gap_frames / fps
        if gap_cap_s is not None:
            gap_s = max(0.0, min(gap_s, gap_cap_s))
        return float(W * math.exp(- ((gap_s - mu_s) ** 2) / (2.0 * sigma_s * sigma_s)))

    P = int(per_state_k or (K if K is not None else 10))

    T = len(event_lists)

    # dp scores [i][j] list of top p cummulative score ending at node i,j
    # dp bp: list of backpointers (prevj, prev rank index) aligned with dp_scores[i][j]
    dp_scores: List[List[List[float]]] = []
    dp_bp:     List[List[List[Tuple[int, int]]]] = []

    layer0_scores: List[List[float]] = []
    layer0_bp:     List[List[Tuple[int,int]]] = []

    for h in event_lists[0]:
        layer0_scores.append([float(h.score)])
        layer0_bp.append([(-1,-1)])
    dp_scores.append(layer0_scores)
    dp_bp.append(layer0_bp)


    for i in range(1, T):
        prev_layer = event_lists[i-1]
        curr_layer = event_lists[i]

        prev_scores = dp_scores[i-1]
        prev_bp = dp_bp[i-1]

        cur_score_layer: list[list[float]] = []
        cur_bp_layer: list[list[tuple[int, int]]] = []\
        
        for j, cur in enumerate(curr_layer):
            candidates: list[tuple[float,int,int]] = [] # score, prev J, prev rank score
            pos_cur = _kf_pos(cur)
            for pj, prev in enumerate(prev_layer):

                if _kf_pos(prev) >= pos_cur:
                    continue
                    
                time_bonus = temporal_bonus(prev, cur)
                if time_bonus < -1e8:  # prohibited
                    continue
                    
                prev_list = prev_scores[pj]
                
                for r, base in enumerate(prev_list[:P]):
                    s = base + float(cur.score) + time_bonus
                    candidates.append((s,pj, r))
            
            if not candidates:
                cur_score_layer.append([])
                cur_bp_layer.append([])
                continue

            best = heapq.nlargest(P, candidates, key=lambda t: t[0])
            cur_score_layer.append(
                [s for (s, _, _) in best]
            )
            cur_bp_layer.append(
                [(pj,r) for (_, pj, r) in best]
            )
        dp_scores.append(cur_score_layer)
        dp_bp.append(cur_bp_layer)
    
    last_scores = dp_scores[-1]
    end_heap: List[Tuple[float, int, int]] = []  # (score, end_j, rank_idx)
    for j, scores in enumerate(last_scores):
        for r, s in enumerate(scores[:P]):
            end_heap.append((s, j, r))

    if not end_heap:
        return []

    if K is None:
        K = len(end_heap)

    top_end = heapq.nlargest(K, end_heap, key=lambda t: t[0])

    
    def backtrack(
        end_j: int, r: int
    ) -> list[KeyframeScore]:
        path_rev: list[KeyframeScore] = []
        j = end_j
        i = T - 1
        rank = r
        while i >= 0 and j >= 0:
            path_rev.append(event_lists[i][j])
            if i == 0:
                break
                
            prev_j, prev_rank = dp_bp[i][j][rank]
            j, rank = prev_j, prev_rank
            i -= 1
        path_rev.reverse()
        return path_rev

    out: list[tuple[list[KeyframeScore], float]] = []
    seen = set()
    for s, end_j, r in top_end:
        path = backtrack(end_j, r)
        sig = tuple((h.group_id, h.video_id, h.keyframe_id) for h in path)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((path, float(s)))
    
    out.sort(key=lambda x: x[1], reverse=True)
    return out


        
                    

def rerank_across_videos_kf(
    by_bucket_paths: Dict[Tuple[str, str], List[Tuple[List[KeyframeScore], float]]],
    top_k: Optional[int] = None
) -> List[Tuple[List[KeyframeScore], float]]:
    flat: List[Tuple[List[KeyframeScore], float]] = []
    for _, paths in by_bucket_paths.items():
        flat.extend(paths)
    flat.sort(key=lambda x: x[1], reverse=True)
    return flat if top_k is None else flat[:top_k]  