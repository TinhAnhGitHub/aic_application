from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union, Any

from beanie import init_beanie, PydanticObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from tqdm import tqdm
from pymongo import ReplaceOne
from pymongo.results import InsertManyResult

from beanie.operators import In 
from app.models.common import KeyframeModel



def _chunks(seq: Sequence[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]


async def init_mongo(db_uri: str, db_name: str):
    """
    Call this once at startup.
    """
    client = AsyncIOMotorClient(db_uri)
    db = client[db_name]
    await init_beanie(database=db, document_models=[KeyframeModel])
    return client

class KeyframeRepo:
    def __init__(self, model=KeyframeModel):
        self.model = model

    async def create_one(
        self, item: Union[KeyframeModel, dict]
    ) -> KeyframeModel:
        if isinstance(item, dict):
            item = self.model(**item)
        await item.insert()  # sets item.id
        return item

    async def create_many(
        self, items: Sequence[Union[KeyframeModel, dict]], ordered: bool = False
    ):
        docs: List[KeyframeModel] = [
            i if isinstance(i, KeyframeModel) else self.model(**i) for i in items
        ]
        size = 2000
        for i in tqdm(range(0, len(docs), size)):
            batch = docs[i:i+size]
            batch_ids = await self.model.insert_many(batch, ordered=ordered)
        

    
    
    async def get_by_triplet(
        self, group_id: str, video_id: str, keyframe_id: str
    ) -> Optional[KeyframeModel]:
        return await self.model.find_one(
            (self.model.group_id == group_id)
            & (self.model.video_id == video_id)
            & (self.model.keyframe_id == keyframe_id)
        )

    async def get_many_by_identifications(
        self, identifications: Iterable[int]
    ) -> List[KeyframeModel]:
        id_list = list(identifications)

        docs = await self.model.find(
            In(self.model.identification, id_list)
        ).to_list()

        lookup = {
            d.identification:d for d in docs
        }    
        ordered_docs = [lookup[i] for i in id_list if i in lookup]
        return ordered_docs
