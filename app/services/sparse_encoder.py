from typing import  Dict, List, Optional
import math
from underthesea import word_tokenize
from scipy.sparse import csr_matrix



class MilvusSparseEncoder:
    def __init__(
        self,
        vocab: Dict[str, int],
        idf_by_tid: Optional[Dict[int, float]] = None,
        vocab_size: Optional[int] = None,
        lowercase: bool = True,
        l2_normalize: bool = True,
    ):
        self.vocab = vocab
        self.idf_by_tid = idf_by_tid or {}
        self.lowercase = lowercase
        self.l2_normalize = l2_normalize
        self.vocab_size = vocab_size
    
    def _tokenize(self, text: str) -> List[str]:
        return word_tokenize(text, format="text").split()

    def encode(self, text: str) ->csr_matrix:
        tokens = self._tokenize(text)
        if not tokens:
            return csr_matrix((1, self.vocab_size))

        tf: dict[int,int] = {}
        for tok in tokens:
            tid = self.vocab.get(tok)
            if tid is None:
                continue
            tf[tid] = tf.get(tid, 0) + 1

        if not tf: 
            return csr_matrix((1, self.vocab_size))

        indices: List[int] = []
        values: List[float] = []
        for tid, cnt in tf.items():
            idf = self.idf_by_tid.get(tid, 1.0)
            w = (1.0 + math.log(cnt)) * idf
            indices.append(tid)
            values.append(float(w))

        if self.l2_normalize and values:
            norm = math.sqrt(sum(v * v for v in values)) or 1.0
            values = [v / norm for v in values]

        return csr_matrix(
            (values, ([0] * len(indices), indices)),
            shape=(1, self.vocab_size),
        )