from __future__ import annotations
from typing import List, Optional, Sequence, Iterable, cast, Literal
from abc import abstractmethod

from pymilvus import AsyncMilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker
from pymilvus import Function, FunctionType
from scipy.sparse import csr_matrix

from app.schemas.search_results  import MilvusSearchResponseItem



class MilvusVectorSearch:
    def __init__(
        self,
        uri: str,
        token: Optional[str],
        collection: str,
    ):
        self.client = AsyncMilvusClient(uri=uri, token=token)
        self.collection = collection

    async def close(self):
        await self.client.close()

    @staticmethod
    def _flatten_hits(search_result) -> Iterable:
        if isinstance(search_result, list) and search_result and isinstance(search_result[0], list):
            for hits in search_result:
                for hit in hits:
                    yield hit
        else:
            for hit in search_result:
                yield hit
    @staticmethod
    def _hit_to_item(hit) -> MilvusSearchResponseItem:
        return MilvusSearchResponseItem(
            identification=hit.entity.get("id"),
            score=hit.score,
        )

    def _to_items(self, search_result) -> List[MilvusSearchResponseItem]:
        return [self._hit_to_item(h) for h in self._flatten_hits(search_result)]

    @property
    @abstractmethod
    def dense_field(self) -> str: ...

    @property
    def sparse_field(self) -> Optional[str]:
        return None

    # ---- single dense search ----
    async def search_dense(
        self,
        query_embedding: list[float],
        top_k: int,
        param: dict,
        expr: Optional[str] = None,
        with_embedding: bool = False,
    ):
        ofs = ["id"]
        if with_embedding:
            ofs.append("embedding")

        res = await self.client.search(
            collection_name=self.collection,
            data=[query_embedding],
            anns_field=self.dense_field,
            param=param,
            limit=top_k,
            expr=expr,
            output_fields=ofs,
        )
        return self._to_items(res)

    async def construct_request_for(
        self,
        *,
        data: list[float] | csr_matrix,                 
        anns_field: str,      
        top_k: int,
        param: dict,
        expr: Optional[str] = None,
    ) -> AnnSearchRequest:

        
        return AnnSearchRequest(
            data=[data],
            anns_field=anns_field,
            param=param,
            limit=top_k,
            expr=expr,
        )
    async def search_combination(
        self,
        requests: list[AnnSearchRequest],
        output_fields: list[str],   
        rerank: Literal["rrf", "weighted"] = "rrf",
        weights: Optional[Sequence[float]] = None,
    ):
        if rerank == "weighted":
            weights = cast(Sequence[float], weights)
            assert len(requests) == len(weights), "Weights length must match requests"

        if rerank == "rrf":
            ranker = RRFRanker(k=60)
        else:
            weights = cast(Sequence[float], weights)
            ranker = WeightedRanker(*weights)

        ofs = output_fields or ["id"]

        res = await self.client.hybrid_search(
            collection_name=self.collection,
            reqs=requests,
            output_fields=ofs,            #
            ranker=ranker,
        )   
        return self._to_items(res)
    
