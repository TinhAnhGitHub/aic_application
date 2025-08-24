from typing import List, Optional, Union
from beanie import PydanticObjectId
from pymongo.results import DeleteResult, InsertManyResult
from app.models.common import ChatHistory


class ChatRepo:
    def __init__(self, model=ChatHistory):
        self.model = model

    # ---------- CREATE ----------
    async def create_one(self, item: Union[dict, ChatHistory]) -> ChatHistory:
        if isinstance(item, dict):
            item = self.model(**item)
        await item.insert()
        return item

    async def create_many(
        self, items: List[Union[dict, ChatHistory]]
    ) -> InsertManyResult:
        docs = [i if isinstance(i, self.model) else self.model(**i) for i in items]
        return await self.model.insert_many(docs)

    async def get_by_id(self, id_: PydanticObjectId) -> Optional[ChatHistory]:
        return await self.model.get(id_)

    async def get_by_question(
        self, question_filename: str, limit: int = 50
    ) -> List[ChatHistory]:
        return (
            await self.model.find(self.model.question_filename == question_filename)
            .sort(-self.model.timestamp)
            .limit(limit)
            .to_list()
        )

    async def list_all(self, limit: int = 100, skip: int = 0) -> List[ChatHistory]:
        return (
            await self.model.find_all()
            .sort(-self.model.timestamp)
            .skip(skip)
            .limit(limit)
            .to_list()
        )
    
    async def delete_by_id(self, id_: PydanticObjectId) -> Optional[DeleteResult]:
        doc = await self.model.get(id_)
        if doc:
            return await doc.delete()
        return None

    async def delete_by_question(self, question_filename: str) -> int:
        res = await self.model.find(self.model.question_filename == question_filename).delete()
        return res.deleted_count
