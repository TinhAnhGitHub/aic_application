# from beanie import Document
# from pydantic import BaseModel, Field
# from datetime import datetime
# from typing import Literal

# from app.schemas.search_results import KeyframeScore
# from app.schemas.search_queries import SingleSearchRequest, KeyframeQuery, CaptionQuery, OCRQuery


# class KeyframeModel(Document):
#     identification: int = Field(..., description="A unique identifier for the keyframe.")
#     group_id: str = Field(..., description="The group ID of the keyframe.")
#     video_id: str = Field(..., description="The video ID associated with the keyframe.")
#     keyframe_id: str = Field(..., description="The unique ID of the keyframe.")
#     tags: list[str] | None = None
#     ocr: list[str] | None = None

#     class Settings:
#         collection = "keyframes"
#         indexes = [
#             "group_id",
#             "video_id",
#             "keyframe_id",
#             {"fields": ["identification"], "unique": True},
#         ]



# class ChatHistory(Document):
#     """
#     Represents a chat or question history item.
#     """
#     question_filename: str = Field(..., description="The name/identifier of the question or search.")
#     timestamp: datetime = Field(default_factory=datetime.now, description="When the history item was created.")
    
#     return_images: list[KeyframeScore] = Field(..., description="Search images associated with this history.")

#     keyframe_search_text: KeyframeQuery | None = Field(None, description="The keyframe search text")
#     caption_search_text: CaptionQuery | None = Field(None, description="The caption search text")
#     ocr: OCRQuery | None = Field(None, description="List of OCR for matching")

    
#     rerank: Literal['rrf', 'weigted'] | None = Field(None, description="Enable if both caption, keyframe search and OCR")
#     weights: list[float] | None = Field(None, description="Weighted of visual embedding. Caption search will be (1-keyframe embedding)")
    
