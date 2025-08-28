from pydantic import BaseModel, Field
from typing import Literal
from typing_extensions import Annotated, Union  


class KeyframeInstance(BaseModel):
    group_id: str
    video_id: str
    keyframe_id: str
    identification: int = Field(..., description="The identification of the keyframe, corresponding to the index of the embeddings in the Milvus Collection")
    tags: list[str] | None = Field(None, description="List of tags associated with the keyframe")
    ocr: list[str] | None = Field(None, description="List of OCR texts associated with the keyframe")


# class KeyframeScore(KeyframeInstance):
#     score: float

# class MilvusSearchRequestInput(BaseModel):
#     """
#     Input schema for Milvus vector search requests.
#     """

#     embedding: list[float] = Field(..., description="The embedding vector to search for.")
#     top_k: int = Field(..., description="The number of top similar items to retrieve.")


# class MilvusSearchResponseItem(BaseModel):
#     """
#     Response item schema for Milvus vector search results.
#     """
#     identification: int = Field(..., description="The identification of the keyframe, corresponding to the index of the embeddings in the Milvus Collection")
#     score: float = Field(..., description="The similarity score of the retrieved item.")
    


# class TagInstance(BaseModel):
#     tag_name: str
#     tag_score: float
# class CaptionSearch(BaseModel):
#     type_search: str = 'caption_search'
#     caption_search_text: str = Field(..., description="The keyframe search text")
#     mode: Literal['rrf', 'weighted']
#     weighted: float | None = Field(None, description="The weighted if using weighted, of the embedding. The bm25 will be (1 - embedding_weight)")
#     tag_boost_alpha: float = Field(...,ge=0, le=1.0, description="Tag boost alpha, if 0.0 then it will not be used")

# class KeyframeSearch(BaseModel):
#     type_search: str = 'keyframe_search'
#     keyframe_search_text: str = Field(..., description="The keyframe search text")
#     tag_boost_alpha: float = Field(...,ge=0, le=1.0, description="Tag boost alpha, if 0.0 then it will not be used")

# class OCRSearch(BaseModel):
#     type_search: str = 'ocr_search'
#     list_ocr: str = Field(..., description="List of OCR")




# # class EventOrder(BaseModel):
# #     """
# #     Event Text, chunked from the original text, with order to indicate the sequence.
# #     """
# #     order: int 
# #     event_text: str

# class EventSearch(BaseModel):
#     keyframe_search: KeyframeSearch | None = Field(None , description="Keyframe search")
#     caption_search: CaptionSearch | None = Field(None, description="Caption search")
#     ocr_search: OCRSearch | None = Field(None, description="OCR search")
#     event_order: int = Field(..., description="The event order")
    



# class EventHit(BaseModel):
#     search_setting: EventSearch = Field(..., description="Search setting")
#     video_id: str
#     keyframe_id: str
#     group_id: str
#     score: float


