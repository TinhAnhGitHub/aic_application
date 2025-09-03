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


    tags_path: str | None = "/media/tinhanhnguyen/Projects/aic_application/data/tags.txt"   

    st_model: str = "AITeamVN/Vietnamese_Embedding"
    beit3_ckpt: str = "/media/tinhanhnguyen/Projects/HCMAI/local/beit3/beit3_large_patch16_384_f30k_retrieval.pth"
    beit3_tokenizer_path: str = "/media/tinhanhnguyen/Projects/HCMAI/local/beit3/beit3.spm"

    som_bmu_map_path: str | None = None  # Path to a .npz/.npy or JSON mapping identification->(u,v)
    som_grid_h: int = 64
    som_grid_w: int = 64
    som_kernel_radius: int = 2
    som_kernel_sigma: float = 1.2
    som_w_pos: float = 1.0
    som_w_neg: float = 0.4
    som_alpha: float = 1.0
    som_beta: float = 0.75
    som_gamma: float = 0.15
    som_kappa: float = 3.0
    

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__"
    )

settings = Settings()
