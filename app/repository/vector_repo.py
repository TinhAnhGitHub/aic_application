
from pymilvus import AnnSearchRequest
from typing import Literal, Optional, Sequence
from app.common.repository import MilvusVectorSearch
from scipy.sparse import csr_matrix

from app.schemas.search_results  import MilvusSearchResponseItem


class KeyframeSearchRepo(MilvusVectorSearch):
    KF_DENSE_FIELD = "kf_embedding"

    @property
    def dense_field(self) -> str:
        return self.KF_DENSE_FIELD
    


class CaptionSearchRepo(MilvusVectorSearch):
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

    def construct_dense_request(self, embedding: list[float], top_k: int, param: dict) -> AnnSearchRequest:
        return  self.construct_request_for(
            data=embedding,
            anns_field=self.dense_field,
            top_k=top_k,
            param=param,
        )

    def construct_sparse_request(self, sparse_vec: csr_matrix, top_k: int, param: dict) -> AnnSearchRequest:
        return self.construct_request_for(
            data=sparse_vec,            
            anns_field=self.sparse_field,
            top_k=top_k,
            param=param,
        )
    
    async def search_caption_hybrid(
        self,
        dense_req: AnnSearchRequest,
        sparse_req: AnnSearchRequest,
        rerank: Literal["rrf", "weighted"] = "rrf",
        weights: Optional[Sequence[float]] = None,
    ) -> list[MilvusSearchResponseItem]:
        return await self.search_combination(
            requests=[dense_req, sparse_req],
            rerank=rerank,
            weights=weights,
            output_fields=["id"],  
        )




