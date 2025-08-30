import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import  List, Optional, Iterator
import numpy as np
import typer
from dataclasses import dataclass
from tqdm import tqdm
from pymilvus import(
    AsyncMilvusClient,
)

from scipy.sparse import csr_matrix
from app.core.config import settings
from app.repository.keyframe_repo import KeyframeRepo
from app.core.logger import RichAsyncLogger

from pymilvus import connections, Collection
import numpy as np
from tqdm import tqdm



def _batch_iter(n: int, batch: int):
    s = 0
    while s < n:
        e = min(s + batch, n)
        yield s, e
        s = e


@dataclass
class KeyframeInsertRequest:
    ids: list[int]
    kf_emb: np.ndarray


@dataclass
class CaptionInsertRequest:
    ids:list[int]
    caption_text: list[str]
    caption_embedding: np.ndarray



def insert_keyframes_milvus_sync(
    collection_name: str,
    ids: list[int],
    kf_emb: np.ndarray,
    host: str = "localhost",
    port: str = "19530",
    batch_size: int = 10000,
):
    kf_emb = np.asarray(kf_emb, dtype=np.float32)
    assert len(ids) == kf_emb.shape[0]

    
    connections.connect("default", host=host, port=port)
    collection = Collection(collection_name)

    print(f"[keyframe insert] inserting {len(ids)} rows into {collection_name} in batches of {batch_size}...")

    for s in tqdm(range(0, len(ids), batch_size), desc="Insert batches"):
        e = min(s + batch_size, len(ids))

        batch_ids = [int(i) for i in ids[s:e]]
        batch_embs = kf_emb[s:e].tolist()

        entities = [batch_ids, batch_embs]
        collection.insert(entities)

    collection.flush()
    collection.load()
    print(f"[keyframe insert] Done. Total entities = {collection.num_entities}")

    return collection




def insert_captions_milvus_sync(
    collection_name: str,
    ids: list[int],
    caption_texts: list[str],
    caption_emb: np.ndarray,
    host: str = "localhost",
    port: str = "19530",
    batch_size: int = 1,
):
    # Ensure numpy float32
    caption_emb = np.asarray(caption_emb, dtype=np.float32)
    assert len(ids) == caption_emb.shape[0] == len(caption_texts)

    # Connect
    connections.connect("default", host=host, port=port)
    collection = Collection(collection_name)

    print(f"[caption insert] inserting {len(ids)} rows into {collection_name} in batches of {batch_size}...")

    for s in tqdm(range(0, len(ids), batch_size), desc="Insert caption batches"):
        e = min(s + batch_size, len(ids))

        batch_ids = [int(i) for i in ids[s:e]]
        batch_embs = caption_emb[s:e].tolist()
        batch_texts = caption_texts[s:e]

        entities = [batch_ids, batch_texts, batch_embs]
        collection.insert(entities)

    collection.flush()
    collection.load()
    print(f"[caption insert] Done. Total entities = {collection.num_entities}")

    return collection
    