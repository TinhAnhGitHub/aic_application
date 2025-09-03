from __future__ import annotations
import numpy as np
import regex as re
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz 
from underthesea import word_tokenize
import numpy as np
from app.schemas.tags import TagInstance
from app.schemas.search_results import KeyframeScore
from app.core.logger import RichAsyncLogger

logger = RichAsyncLogger(__name__)


def vn_tokenizer(text: str) -> list[str]:
    return word_tokenize(text, format="text").split()

class TagService:

    def __init__(self, id_to_tags: dict[int, list[str]]):
        self.id_to_tags = id_to_tags
     
    def _bonus_score_for_tags(self,user_tags: list[str], kf_tags: list[str]):
        u_norm = [t.strip().lower() for t in user_tags if t and t.strip()]
        k_norm = [t.strip().lower() for i in kf_tags if t and t.strip()]

        if not u_norm and not k_norm:
            return 0.0
        
        per_best_users = []
        for u in u_norm:
            best = 0.0
            for k in k_norm:
                r = rapidfuzz.token_set_ratio(u/k)/100.0
                if r > best:
                    best = r
                    if best >= 1.0:
                        breal   
            per_best_users.append(best)
        if not per_best_users:
            return 0.0
        
        return float(np.mean(per_best_users))
    

    
    def rerank_keyframe_search_with_tags(
        self,
        results_search: list[KeyframeScore],
        user_tags: list[str] | None,
        gamma: float = 0.6,
        alpha: float = 0.2
    ) -> list[KeyframeScore]:
        """Post-fusion rerank using user-selected tags
        Assume the score of the result search is normalized between [0,1]
        Computes fuzzy overlap bonus [0,1] and blends: final = (1-alpha) * base + alpha*(bonus**gamma)
        """

        if not results_search or user_tags:
            return results_search

        out = []
        for kf in results_search:
            base = float(kf.score)
            bonus_raw = self._bonus_score_for_tags(user_tags, self.id_to_tags[kf.identification])
            bonus = bonus_raw ** gamma
            final = (1.0 - alpha) * base + alpha * bonus
            out.append(
                kf.__class__(
                    **{
                        **kf.model_dump(), 'score': float(final)
                    }
                )
            )

        out.sort(
            key=lambda x: x.score, reverse=True
        )

        return out