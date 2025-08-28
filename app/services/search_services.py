from app.repository.vector_repo import KeyframeSearchRepo, CaptionSearchRepo
from typing import Literal, Optional, Sequence
class SearchService:
    def __init__(
        self,
        keyframe_search: KeyframeSearchRepo,
        caption_search: CaptionSearchRepo,
    ):
        self.keyframe_search = keyframe_search
        self.caption_search = caption_search

    
    async def search_keyframe_dense(self, query_embedding: list[float], top_k: int, param: dict, **kwargs):
        return await self.keyframe_search.search_dense(query_embedding, top_k, param, **kwargs)
    
    async def search_caption_dense(self, query_embedding: list[float], top_k: int, param: dict, **kwargs):
        return await self.caption_search.search_caption_dense(query_embedding, top_k, param, **kwargs)

    
    




    