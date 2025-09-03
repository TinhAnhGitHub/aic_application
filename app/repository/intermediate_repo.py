from __future__ import annotations
from typing import Optional
from datetime import datetime

from beanie import PydanticObjectId

from app.models.history import IntermediateResult
from app.schemas.application import KeyframeRef


class IntermediateRepo:
    def __init__(self, model=IntermediateResult):
        self.model = model

    async def get_by_question(self, question_filename: str) -> Optional[IntermediateResult]:
        return await self.model.find_one(self.model.question_filename == question_filename)

    async def set_items(self, question_filename: str, items: list[KeyframeRef]) -> IntermediateResult:
        doc = await self.get_by_question(question_filename)
        if doc is None:
            doc = self.model(question_filename=question_filename, items=items, updated_at=datetime.now())
            await doc.insert()
            return doc
        doc.items = items
        doc.updated_at = datetime.now()
        await doc.save()
        return doc

    async def clear(self, question_filename: str) -> int:
        doc = await self.get_by_question(question_filename)
        if doc:
            await doc.delete()
            return 1
        return 0

