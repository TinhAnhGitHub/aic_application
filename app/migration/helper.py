import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import  List, Optional, Iterator

import numpy as np
import typer

from pymilvus import(
    AsyncMilvusClient,
    DataType,
    FieldSchema,
    CollectionSchema,
    FunctionType,
    Function
)

from scipy.sparse import csr_matrix
from app.core.config import settings
from app.repository.keyframe_repo import KeyframeRepo
from app.core.logger import RichAsyncLogger

logger = RichAsyncLogger(__name__)

app_cli = typer.Typer(
    add_completion=False, help="Hotspot Search - Data Migration CLI"
)

IMG_EXTS = {'.webp', '.png'}

@dataclass(frozen=True)
class KfRow:
    identification: int
    group_id: str
    video_id: str
    keyframe_id: str
    caption: Optional[str]
    tags: Optional[List[str]]
    ocr: Optional[List[str]]


def _parse_triplet_from_path(
    p: Path
) -> tuple[str,str,str]:
    """
    Expect keyframe image path like: {base_folder}/<group_id>/<video_id>/keyframes/<keyframe_id>.webp
    """
    group_id: str = p.parent.parent.parent.name
    video_id: str = p.parent.parent.name
    keyframe_id: str = p.stem
    return group_id, video_id, keyframe_id

def _scan_keyframes(
    keyframes_root_path: Path
)-> list[Path]:
    files = []
    for p in keyframes_root_path.rglob('*'):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            files.append(p)
    files.sort() # -> important, sort based on groupid, video id and keyframe number
    return files


def _load_caption_json(
    caption_root: Path,
    group: str,
    video: str,
    kf_id: str
) -> dict | None:
    path = caption_root/group/video/f"{kf_id}.json"
    if not path.exists():
        return None

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _batch_iter(n: int, batch: int):
    s = 0
    while s < n:
        e = min(s + batch, n)
        yield s, e
        s = e


async def _ensure_keyframe_collection(client: AsyncMilvusClient, dim: int):
    name = settings.milvus_collection_keyframe
    exists = await client.has_collection(collection_name=name)

    logger.info(f"Collection name: {name}. Exist: {exists}")

    if not exists:
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
                description="primary id"
            ),
            FieldSchema(
                name="kf_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim,
                description="keyframe dense embedding"
            ),
        ]
        schema = CollectionSchema(fields, description="Keyframe dense vectors")
        await client.create_collection(collection_name=name, schema=schema, description="Keyframe dense vectors")
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name='kf_embedding',
            index_type='IVF_FLAT',
            metric_type='IP',
            params={"nlist": 1024}
        )
        await client.create_index(
            collection_name=name,
            index_params=index_params
        )
        

    await client.load_collection(collection_name=name)




def _collect_rows_from_fs(keyframes_root_path: Path, captions_root: Path) -> List[KfRow]:
    """
    Use keyframe file order to define identification (0..N-1)
    and pull caption/tags/ocr per frame.
    """
    rows: List[KfRow] = []
    kf_files = _scan_keyframes(keyframes_root_path)
    for ident, img_path in enumerate(kf_files):
        g, v, kf_id = _parse_triplet_from_path(img_path)
        meta = _load_caption_json(captions_root, g, v, kf_id) or {}
        caption = meta["caption"]
        tags = meta["tags_list"] 
        ocr = meta.get('ocr_blocks')
        rows.append(KfRow(
            identification=ident,
            group_id=g,
            video_id=v,
            keyframe_id=kf_id,
            caption=caption,
            tags=tags,
            ocr=ocr,
        ))
    return rows


def _build_bm25_corpus(rows: List[KfRow]) -> List[str]:
    """
    Build per-row BM25 text: caption + tags + ocr blocks (space-joined).
    """
    docs: List[str] = []
    for r in rows:
        parts = []
        if r.caption: parts.append(r.caption)
        if r.tags: parts.append(" ".join(r.tags))
        if r.ocr: parts.append(" ".join(r.ocr))
        docs.append(" ".join(parts).strip())
    return docs


async def _ensure_caption_collection(
    client,
    dense_dim: int,
    name: str = "caption",
    has_sparse: bool = True,
):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False, description="Caption dense/sparse vectors")
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("caption_text", DataType.VARCHAR, max_length=32768, enable_analyzer=True)
    schema.add_field("caption_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
    if has_sparse:
        schema.add_field("caption_sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(
            name="bm25_caption",
            input_field_names=["caption_text"],
            output_field_names=["caption_sparse"],
            function_type=FunctionType.BM25,
        ))

    if not await client.has_collection(name):
        await client.create_collection(collection_name=name, schema=schema)
    
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="caption_embedding",
        index_type="AUTOINDEX",          
        metric_type="IP",
    )   

    if has_sparse:
        index_params.add_index(
            field_name="caption_sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",  # or "DAAT_WAND"
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            },
        )
    await client.create_index(collection_name=name, index_params=index_params)
    await client.load_collection(collection_name=name)
    

def iter_caption_json(root:Path) -> Iterator[tuple[int,str]]:
    for i, p in enumerate(sorted(root.glob("*.json"))):
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = data['caption']
        yield i, data

async def _insert_keyframes_milvus(
    client: AsyncMilvusClient,
    ids: List[int],
    kf_emb: np.ndarray,
    batch_size: int = 1000,
):
    name =  settings.milvus_collection_keyframe
    assert len(ids) == kf_emb.shape[0]

    for s, e in _batch_iter(len(ids), batch_size):
        payload = {
            'id': ids[s:e],
            'kf_embedding': kf_emb[s:e].astype(np.float32).tolist(),
        }
        await client.insert(collection_name=name, data=payload)
    await client.flush(collection_name=name)
    
async def _insert_caption_milvus(
    client: AsyncMilvusClient,
    ids: list[int],
    caption_emb: np.ndarray,
    caption_sparse_rows: List[csr_matrix],
    batch_size: int = 1000
):
    name = settings.milvus_collection_caption
    assert len(ids) == caption_emb.shape[0] == len(caption_sparse_rows)

    for s, e in _batch_iter(len(ids), batch_size):
        batch = []
        for i in range(s, e):
            batch.append({
                'id': int(ids[i]),
                'caption_embedding': caption_emb[i].astype(np.float32).tolist(),
                'caption_sparse': caption_sparse_rows[i],
            })
        await client.insert(collection_name=name, data=batch)
    await client.flush(collection_name=name)

    
async def _insert_mongo(repo: KeyframeRepo, rows: list[KfRow]):
    items = []
    for r in rows:
        doc = {
            "identification": r.identification,
            "group_id": r.group_id,
            "video_id": r.video_id,
            "keyframe_id": r.keyframe_id,
            "tags": r.tags or None,
        }
        items.append(doc)
    await repo.create_many(items)

@app_cli.command("init")
def init_collection(
    keyframe_embedding_path: Path = typer.Option(..., help="Path to keyframe embeddings .npy"),
    caption_embedding_path: Path = typer.Option(..., help="Path to the caption embeddings .npy"),
):
    async def _run():
        kf = np.load(keyframe_embedding_path, mmap_mode="r")
        cap = np.load(caption_embedding_path, mmap_mode="r")
        client = AsyncMilvusClient(uri=settings.milvus_uri)
        await _ensure_keyframe_collection(client, dim=int(kf.shape[1]))
        await _ensure_caption_collection(client, dense_dim=int(cap.shape[1]), has_sparse=True)
        await client.close()
        print("[init] collections and indexes are ready.")
    asyncio.run(_run())


