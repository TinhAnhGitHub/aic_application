from typing import  Dict, List, Optional
import math
from underthesea import word_tokenize
from scipy.sparse import csr_matrix
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer


class MilvusSparseEncoder:
    """
    Thin wrapper around pymilvus BM25EmbeddingFunction so that:
      - During MIGRATION: use encode_documents() on captions (docs)
      - During RETRIEVAL:  use encode_queries()   on user text (queries)

    Notes:
      * BM25 needs an analyzer (tokenizer+filters) and corpus stats (via .fit()).
      * After fitting, call .save(path) during migration and .load(path) at runtime so both
        sides share identical statistics (idf, etc).
    """

    def __init__(
        self,
        language: str = "icu",
        model_state_path: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        num_workers: Optional[int] = None,
    ):
        analyzer = build_default_analyzer(language=language)
        self._bm25 = BM25EmbeddingFunction(
            analyzer=analyzer, k1=k1, b=b, epsilon=epsilon, num_workers=num_workers
        )
        if model_state_path:
            self._bm25.load(model_state_path)  
    
    def fit(self, corpus: List[str]) -> None:
        """Fit BM25 statistics on your caption corpus."""
        self._bm25.fit(corpus) 
    
    def save(self, path: str) -> None:
        """Persist BM25 parameters/statistics to JSON so runtime can .load(path)."""
        self._bm25.save(path)
    
    def encode_documents(self, docs: List[str]):
        """
        Use ONLY for documents at ingestion/migration time.
        Returns a 2D scipy.sparse.csr_array compatible with Milvus.
        """
        return self._bm25.encode_documents(docs)

    def encode_queries(self, queries: List[str]):
        """
        Use ONLY for queries at retrieval time.
        Returns a 2D scipy.sparse.csr_array compatible with Milvus.
        """
        return self._bm25.encode_queries(queries)
    