from __future__ import annotations
import numpy as np
import regex as re
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz 
from underthesea import word_tokenize

from app.schemas.application import TagInstance
from app.schemas.application import KeyframeScore



def vn_tokenizer(text: str) -> list[str]:
    return word_tokenize(text, format="text").split()


class TagService:
    def __init__(
        self, 
        tag_list: list[str],
        preselect_k: int = 100,
        bm25_weight: float = 0.65,
        tfidf_weight: float = 0.30,
        rf_weight: float = 0.05
    ):
        if not tag_list:
            raise ValueError("tag_list must not be empty")
        
        self.bm25_weight = bm25_weight / (bm25_weight + tfidf_weight + rf_weight)
        self.tfidf_weight = tfidf_weight / (bm25_weight + tfidf_weight + rf_weight)
        self.rf_weight = rf_weight / (bm25_weight + tfidf_weight + rf_weight)


        self.tag_list = tag_list
        self._tags_np = np.array(tag_list)

        self._tfidf_word = TfidfVectorizer(
            tokenizer=vn_tokenizer,
            ngram_range=(1, 2),   # unigrams + bigrams of words
            lowercase=True,
            norm="l2"
        )
        self._X_word = self._tfidf_word.fit_transform(tag_list)


        self._tokenized_tags = [vn_tokenizer(tag) for tag in tag_list]
        self._bm25 = BM25Okapi(self._tokenized_tags)
        self._preselect_k = max(
            10, 
            min(preselect_k, len(tag_list))
        )

    def scan_tags(
        self,
        user_query: str,
        top_tags: int = 6
    ) -> list[TagInstance]:
        """
        From the user query, scan each tag against the query, and return the top 6 tags
        Using TfIDF, BM25 and rapid fuzz
        """
        if not user_query.strip():
            return []
        
        q_word = self._tfidf_word.transform([user_query])

        scored_words = (self._X_word @ q_word.T).toarray().ravel()

        M = min(
            self._preselect_k, len(self.tag_list)
        )

        # tdidf pre selection
        if M < len(self.tag_list):
            candidate_index = np.argpartition(-scored_words, M)[:M]
        else:
            candidate_index = np.arange(len(self.tag_list))
        
        # BM25 RANK
        q_tokens = vn_tokenizer(user_query)
        bm25_all = self._bm25.get_scores(q_tokens)
        bm25_scores = bm25_all[candidate_index]

        rf = np.array([
            fuzz.token_set_ratio(user_query, self.tag_list[i]) / 100.0 for i in candidate_index
        ])

        def norm(x: np.ndarray) -> np.ndarray:
            m = x.max() if x.size else 0.0
            return (x / m) if m > 0 else np.zeros_like(x)

        s_word = norm(scored_words[candidate_index])
        s_bm25 = norm(bm25_scores)
        s_rf = norm(rf) 

        final = self.bm25_weight * s_bm25 + self.tfidf_weight * s_word + self.rf_weight * s_rf

        k = min(top_tags, len(candidate_index))
        top_local = np.argpartition(-final, k - 1)[:k]
        top_sorted = top_local[np.argsort(-final[top_local])]

        results = []
        for j in top_sorted:
            idx = candidate_index[j]
            score = float(final[j])
            results.append(TagInstance(
                tag_name=self.tag_list[idx],
                tag_score=score
            ))
        return results
    

    def rerank_keyframe_search_with_tags(
        self,
        tags: list[TagInstance],
        results_search: list[KeyframeScore],
        alpha: float = 0.2
    ) -> list[KeyframeScore]:
        """Rerank the keyframe search results with the tags

        Args:
            tags (list[TagInstance]): List of tags with name and score
            results_search (list[MilvusSearchResponseItem]): list of returned keyframes
        """

        if not results_search:
            return []

        top_tag_names = {t.tag_name for t in tags}

        clip_scores = np.array([
            kf.score for kf in results_search
        ], dtype=float)

        mu = clip_scores.mean()
        sigma = clip_scores.std() if clip_scores.std() > 1e-6 else 1.0
        norm_scores = (clip_scores - mu) / sigma

        final_scores = []

        for i, kf in enumerate(results_search):
            if hasattr(kf, "tags") and kf.tags:
                kf_tags = set(kf.tags)
            else:
                kf_tags = set()
        

            m = len(kf_tags & top_tag_names)
            boost = alpha * np.log1p(m) * sigma
            final_score = norm_scores[i] + boost
            final_scores.append(final_score)
        
        for kf, fs in zip(results_search, final_scores):
            kf.score = float(fs)
        
        reranked = sorted(
            results_search,
            key=lambda x: x.score,
            reverse=True
        )

        return reranked

