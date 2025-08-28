from typing import Optional, Sequence
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "aic_database"

    es_hosts: Sequence[str] = ("http://localhost:9200",)
    es_index: str = "keyframes"
    es_api_key: str | None = None
    es_basic_user: str = "elastic"
    es_basic_pass: str = "changeme"
    es_verify_certs: bool = False

    milvus_uri: str = "http://localhost:19530"
    milvus_collection_keyframe: str = "keyframes"
    milvus_collection_caption: str = "caption"

    sparse_vocab_path: str | None = None
    sparse_idf_path: str | None = None

    tags_path: str | None = None    

    st_model: str = "AITeamVN/Vietnamese_Embedding"
    beit3_ckpt: str = "checkpoints/beit3.pth"
    beit3_tokenizer_path: str = "xlm-roberta-base"

    bm25_language: str = "icu"
    bm25_model_path: str | None = "bm25.json"   

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__"
    )

settings = Settings()