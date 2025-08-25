from __future__ import annotations
from typing import Iterable, List, Optional, Sequence
from elasticsearch import AsyncElasticsearch, NotFoundError


from app.schemas.application import KeyframeInstance, KeyframeScore



class ElasticsearchKeyframeRepo:
    """
    Stores and searches KeyframeInstance by OCR using the Vietnamese analyzer.
    - ocr: vi_analyzer (with underthesea stopwords)
    - ocr.fold: ascii-folded analyzer for tone-less queries
    Search strategy ("mostly exact with a bit of fuzzy"):
      1) match_phrase on ocr (boost high)
      2) match_phrase on ocr.fold (boost high but a bit lower)
      3) match (AND) with fuzziness on both fields (low boost)
    """

    def __init__(
        self,
        hosts: Sequence[str] = ("http://localhost:9200",),
        index: str = "keyframes",
        api_key: Optional[str] = None,
        basic_auth: Optional[tuple[str, str]] = ("elastic", "changeme"),
        verify_certs: bool = False,
        request_timeout: int = 30,
    ):
        self.es = AsyncElasticsearch(
            hosts=hosts,
            api_key=api_key,
            basic_auth=basic_auth,
            verify_certs=verify_certs,
            request_timeout=request_timeout,
        )
        self.index = index
    
    def ensure_index(
        self,
        recreate: bool = False,
        stop_words: list[str] | None = None,
        keep_punctuation: Optional[bool] = None,
        dict_path: Optional[str] = None,
        split_url: Optional[bool] = None,
    ):
        if recreate:
            try:
                self.es.indices.delete(index=self.index)
            except NotFoundError:
                pass
                
        if self.es.indices.exists(index=self.index):
            return

        vi_params = {
            "stopwords": stop_words or [],
        }
        if keep_punctuation is not None:
            vi_params["keep_punctuation"] = keep_punctuation
        if dict_path is not None:
            vi_params["dict_path"] = dict_path
        if split_url is not None:
            vi_params["split_url"] = split_url
        
        settings = {
            "analysis": {
                "analyzer": {
                    "my_vi_custom": {"type": "vi_analyzer", **{k: v for k, v in vi_params.items() if v is not None}},
                    "my_vi_fold": {
                        "tokenizer": "vi_tokenizer",
                        "filter": ["lowercase", "my_ascii_folding"],
                    },
                },
                "filter": {
                    "my_ascii_folding": {"type": "asciifolding", "preserve_original": False}
                },
            }
        }

        body = {
            "settings": settings,
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "group_id": {"type": "keyword"},
                    "video_id": {"type": "keyword"},
                    "keyframe_id": {"type": "keyword"},
                    "identification": {"type": "integer"},
                    "tags": {"type": "keyword"},
                    # Store OCR list in _source, but index joined text for analysis
                    "ocr": {
                        "type": "text",
                        "analyzer": "my_vi_custom",
                        "search_analyzer": "my_vi_custom",
                        "fields": {
                            "fold": {"type": "text", "analyzer": "my_vi_fold"},
                        },
                    },
                    "_ocr_joined": {  # internal convenience (not queried directly here)
                        "type": "text",
                        "index": False,
                    },
                }
            },
        }

        self.es.indices.create(index=self.index, body=body)

    @staticmethod
    def _make_id(d: KeyframeInstance)->str:
        return f"{d.group_id}::{d.video_id}::{d.keyframe_id}"
        
    def upsert(self, item: KeyframeInstance):
        _id = self._make_id(item)
        body = item.model_dump(mode="json")
        if item.ocr:
            body["_ocr_joined"] = " ".join(item.ocr)
            body['ocr'] = [" ".join(item.ocr)]
        self.es.index(index=self.index, id=_id, document=body)

    def bulk_upsert(self, docs: Iterable[KeyframeInstance], refresh: bool = True):
        from elasticsearch.helpers import streaming_bulk

        def gen_actions():
            for d in docs:
                body = d.model_dump()
                if d.ocr:
                    body["_ocr_joined"] = " ".join(d.ocr)
                    body["ocr"] = [" ".join(d.ocr)]
                yield {
                    "_op_type": "index",
                    "_index": self.index,
                    "_id": self._make_id(d),
                    "_source": body,
                }

        for ok, _ in streaming_bulk(self.es, gen_actions()):
            if not ok:
                pass
        if refresh:
            self.es.indices.refresh(index=self.index)
    
    def get(self, group_id: str, video_id: str, keyframe_id: str) -> Optional[KeyframeInstance]:
        _id = f"{group_id}::{video_id}::{keyframe_id}"
        try:
            resp = self.es.get(index=self.index, id=_id)
        except NotFoundError:
            return None
        return KeyframeInstance(**resp["_source"])

    async def search(
        self,
        query_text: str,
        top_k: int = 10,
        group_id: Optional[str] = None,
        video_id: Optional[str] = None,
        min_score: Optional[float] = None,
        from_: int = 0,
        explain: bool = False,
    ) -> List[KeyframeScore]:
        """
        “Mostly exact” OCR search with a little fuzziness.

        Scoring order (highest -> lowest):
          A) Exact phrase match on ocr            (boost 6.0)
          B) Exact phrase match on ocr.fold       (boost 5.0)
          C) ANDed match with fuzziness on ocr    (boost 1.7)
          D) ANDed match with fuzziness on ocr.fold (boost 1.4)
        """

        filters: list[dict] = []
        if group_id is not None:
            filters.append({"term": {"group_id": group_id}})
        if video_id is not None:
            filters.append({"term": {"video_id": video_id}})
        
        shoulds = [
            {"match_phrase": {"ocr": {"query": query_text, "slop": 0, "boost": 6.0}}},
             {"match_phrase": {"ocr.fold": {"query": query_text, "slop": 0, "boost": 5.0}}},
             {
                "match": {
                    "ocr": {
                        "query": query_text,
                        "operator": "and",
                        "fuzziness": "AUTO:4,6",
                        "prefix_length": 1,
                        "max_expansions": 20,
                        "boost": 1.7,
                    }
                }
            },
            {
                "match": {
                    "ocr.fold": {
                        "query": query_text,
                        "operator": "and",
                        "fuzziness": "AUTO:4,6",
                        "prefix_length": 1,
                        "max_expansions": 20,
                        "boost": 1.4,
                    }
                }
            },
        ]

        body = {
            "query": {
                "bool": {
                    "should": shoulds,
                    "minimum_should_match": 1,
                    "filter": filters or None,
                }
            },
            "_source": True,
            "size": top_k,
            "from": from_,
        }

        if min_score is not None:
            body["min_score"] = min_score
        
        resp = await self.es.search(index=self.index, body=body, explain=explain)
        hits = resp.get('hits', {}).get('hits', [])
        out = []
        for hit in hits:
            src = hit.get("_source", {})
            out.append(
                KeyframeScore(
                    score=hit.get("_score", 0.0),
                    group_id=src['group_id'],
                    video_id=src['video_id'],
                    keyframe_id=src['keyframe_id'],
                    identification=src['identification'],
                    tags=src.get("tags"),
                    ocr=src.get("ocr")
                )
            )
        return out



