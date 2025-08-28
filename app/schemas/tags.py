from pydantic import BaseModel

class TagInstance(BaseModel):
    tag_name: str
    tag_score: float

