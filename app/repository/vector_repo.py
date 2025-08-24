
from pymilvus import AnnSearchRequest
from typing import Literal, Optional, Sequence

from app.common.repository import MilvusVectorSearch






class KeyframeSearch(MilvusVectorSearch):
    KF_DENSE_FIELD = "kf_embedding"

    @property
    def dense_field(self) -> str:
        return self.KF_DENSE_FIELD
    


class CaptionSearch(MilvusVectorSearch):
    CAPTION_DENSE_FIELD = "caption_embedding"
    CAPTION_SPARSE_FIELD = "caption_sparse"

    @property
    def dense_field(self) -> str:
        return self.CAPTION_DENSE_FIELD

    @property
    def sparse_field(self) -> str:
        return self.CAPTION_SPARSE_FIELD

    async def search_caption_dense(self, query_embedding: list[float], top_k: int, param: dict, **kwargs):
        return await self.search_dense(query_embedding, top_k, param, **kwargs)

    async def search_caption_hybrid(
        self,
        dense_req: AnnSearchRequest,
        sparse_req: AnnSearchRequest,
        rerank: Literal["rrf", "weighted"] = "rrf",
        weights: Optional[Sequence[float]] = None,
    ):
        return await self.search_combination(
            requests=[dense_req, sparse_req],
            rerank=rerank,
            weights=weights,
        )




