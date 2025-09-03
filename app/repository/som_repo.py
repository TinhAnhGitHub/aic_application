import base64
import io
import zlib 
from datetime import datetime
import numpy as np
from typing import Literal
from app.models.som import  SomFeedbackEvent, SomOverlay


def arr_to_b64(a: np.ndarray) -> str:
    a = np.asarray(a, dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf,a)
    raw = buf.getvalue()

    return base64.b64encode(zlib.compress(raw)).decode('ascii')


def b64_to_arr(s: str) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(s.encode('ascii')))
    return np.load(
        io.BytesIO(raw)
    )




class SomRepo:
    async def get_overlay(
        self,
        question_filename: str,
        shape: tuple[int,int]
    ) -> tuple[np.ndarray, np.ndarray, SomOverlay | None]:
        doc = await SomOverlay.find_one(SomOverlay.question_filename==question_filename)
        if not doc:
            h,w = shape
            pos = np.zeros((h,w), dtype = np.float32)
            neg = np.zeros((h,w), dtype = np.float32)

            return pos,neg,None
        return b64_to_arr(doc.pos), b64_to_arr(doc.neg), doc
    

    async def save_overlay(
        self,
        question_filename: str,
        pos: np.ndarray,
        neg: np.ndarray,
        doc: SomOverlay | None
    ) -> SomOverlay:
        if doc is None:
            doc = SomOverlay(
                question_filename = question_filename,
                grid_h = pos.shape[0],
                grid_w = pos.shape[1],
                pos = arr_to_b64(pos),
                neg = arr_to_b64(neg),
                updated_at=datetime.now()
            )
            await doc.insert()
            return doc
        
        doc.pos = arr_to_b64(pos)
        doc.neg = arr_to_b64(neg)
        doc.updated_at = datetime.now()
        await doc.save()
        return doc
    

    async def add_events(self, question: str, identification: int, action: Literal['up', 'down'], weight: float=1.0):
        ev = SomFeedbackEvent(question_filename=question, identification=identification, action=action, weight=weight)
        await ev.insert()
    
    async def clear(self, question: str):
        d1 = await SomOverlay.find(SomOverlay.question_filename==question).delete()
        d2 = await SomFeedbackEvent.find(SomFeedbackEvent.question_filename==question).delete()
        return (d1.deleted_count if d1 else 0), (d2.deleted_count if d2 else 0)
    


    

