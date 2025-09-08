from typing import List, Optional, Union
from beanie import PydanticObjectId
from pymongo.results import DeleteResult, InsertManyResult
from app.models.history import SearchHistory
from datetime import datetime, timedelta
class ChatRepo:
    def __init__(self, model=SearchHistory):
        self.model = model

    async def create_one(self, item: Union[dict, SearchHistory]) -> SearchHistory:
        if isinstance(item, dict):
            item = self.model(**item)
        await item.insert()
        return item

    async def create_many(
        self, items: List[Union[dict, SearchHistory]]
    ) -> InsertManyResult:
        docs = [i if isinstance(i, self.model) else self.model(**i) for i in items]
        return await self.model.insert_many(docs)

    async def get_by_id(self, id_: PydanticObjectId) -> Optional[SearchHistory]:
        return await self.model.get(id_)

    async def get_all(self) -> List[SearchHistory]:
        return await self.model.find_all().to_list()

    async def get_by_timestamp(self, timestamp: str, limit: int = 50) -> Optional[SearchHistory]:
        try:
            dt = datetime.fromisoformat(timestamp)
        except ValueError:
            return None
        lo, hi = dt - timedelta(milliseconds=1), dt + timedelta(milliseconds=1)
        docs = await self.model.find(
            self.model.timestamp >= lo,
            self.model.timestamp <= hi,
        ).limit(1).to_list()
        return docs[0] if docs else None
    
    async def get_all_question_filename(self):
        all_history = await self.get_all()
        return list(
            set(
                his.question_filename for his in all_history
            )
        )
    

    async def get_by_question(
        self, question_filename: str, limit: int = 50
    ) -> List[SearchHistory]:
        return (
            await self.model.find(self.model.question_filename == question_filename)
            .sort(-self.model.timestamp)
            .limit(limit)
            .to_list()
        )

    async def list_all(self, limit: int = 100, skip: int = 0) -> List[SearchHistory]:
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
        if res is not None:
            return res.deleted_count
        return 0
