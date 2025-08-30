import typer
import numpy as np
from pathlib import Path
import asyncio
import os
import sys
import glob

ROOT_DIR = os.path.abspath(
    os.path.join(__name__, '..')
)
print(f"{ROOT_DIR=}")
sys.path.insert(0, ROOT_DIR)

from pymilvus import(
    AsyncMilvusClient,
)

from app.migration.helper import _ensure_caption_collection, _ensure_keyframe_collection
from app.core.config import settings
from app.repository.keyframe_repo import init_mongo, KeyframeRepo
from app.repository.elastic_repo import ElasticsearchKeyframeRepo
from app.schemas.application import KeyframeInstance



from app.migration.helper import (
    _collect_rows_from_fs,
    _insert_mongo,
)

from app.migration.ingestion import insert_captions_milvus_sync, insert_keyframes_milvus_sync

from app.core.logger import RichAsyncLogger

logger = RichAsyncLogger(__name__)


app_cli = typer.Typer(
    add_completion=False, help="Hotspot Search - Data Migration CLI"
)


@app_cli.command("init")
def init_collection(
    keyframe_embedding_path: Path = typer.Option(..., help="Path to keyframe embeddings .npy"),
    caption_embedding_path: Path = typer.Option(..., help="Path to the caption embeddings .npy"),
):
    print("HI")
    async def _run():
        kf = np.load(keyframe_embedding_path, mmap_mode="r")
        cap = np.load(caption_embedding_path, mmap_mode="r")
        client = AsyncMilvusClient(uri=settings.milvus_uri)
        await _ensure_keyframe_collection(client, dim=int(kf.shape[1]))
        await _ensure_caption_collection(client, dense_dim=int(cap.shape[1]), has_sparse=True)
        await client.close()
        print("[init] collections and indexes are ready.")
    asyncio.run(_run())

import json
import re
def _load_caption(path: str):
    json_dict = json.load(open(path, 'r', encoding='utf-8'))
    return json_dict['caption']


def _parse_caption_parts(caption_path: str, base_folder: str) -> tuple[int, int, int]:
    """
    Parse <base>/<group_id>/<video_id>/captions/<keyframe_number>.json
    Returns ints for proper numeric sorting.
    """
    rel_path = os.path.relpath(caption_path, base_folder)
    parts = rel_path.split(os.sep)
    group_id, video_id, _, _, filename = parts
    keyframe_number = os.path.splitext(filename)[0]

    gnum = int(re.sub(r"\D", "", group_id)) if re.search(r"\d", group_id) else 0
    vnum = int(re.sub(r"\D", "", video_id)) if re.search(r"\d", video_id) else 0
    knum = int(keyframe_number) if keyframe_number.isdigit() else 0
    return gnum, vnum, knum

from tqdm import tqdm

@app_cli.command('ingest_embedding')
def ingest_embedding(
    keyframes_dir: Path = typer.Option(..., help="Root folder of keyframes"),
    captions_dir: Path = typer.Option(..., help="Root folder of caption JSONs"),
    keyframe_embedding_path: Path = typer.Option(..., help="Path to keyframe embeddings .npy"),
    caption_embedding_path: Path = typer.Option(..., help="Path to the caption embeddings .npy"),
):
    
    keyframe_paths = glob.glob(f"{keyframes_dir}/**/*.webp", recursive=True)
    keyframe_paths = sorted(keyframe_paths)

    caption_paths = glob.glob(f"{captions_dir}/**/*.json", recursive=True)
    caption_paths = sorted(
        caption_paths,
        key=lambda p: _parse_caption_parts(p, str(captions_dir))
    )

    logger.info(f"{keyframe_paths[:5]}")
    logger.info(f"{caption_paths[:5]}")

    keyframe_embedding = np.load(keyframe_embedding_path)
    caption_embedding = np.load(caption_embedding_path)

    N = len(keyframe_paths)
    assert keyframe_embedding.shape[0] == N, "Keyframe embedding count must match number of keyframe images"
    assert caption_embedding.shape[0] == N, "Caption embedding count must match number of keyframe images"
    assert N == len(caption_paths), "hihi"
    logger.info(f"{N=}")

    ids = list(range(N))

    caption_texts = []
    lengths = []
    for i, caption_path in tqdm(enumerate(caption_paths)):
        text = _load_caption(caption_path)
        lengths.append(len(text))
        if len(text) > 65535:
            print(f"⚠️ Caption {i} too long: {len(text)} chars, file={caption_paths[i]}")
            text = text[:65535]
        caption_texts.append(text)
    print(f"Total captions: {len(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Mean length: {sum(lengths)/len(lengths):.2f}")

    logger.info("Begin kf ingestion")
    insert_keyframes_milvus_sync(
        collection_name=settings.milvus_collection_keyframe,
        ids=ids,
        kf_emb=keyframe_embedding,
        batch_size=2000,
    )
    logger.info("Begin caption ingestion")
    insert_captions_milvus_sync(
        collection_name=settings.milvus_collection_caption,
        ids=ids,
        caption_texts=caption_texts,
        caption_emb=caption_embedding,
        batch_size=2000,
    )

    logger.info("Ingestion embeddings successful")
        

    
def _parse_triplet_from_path(p: Path) -> tuple[str, str, str]:
    """
    Expect keyframe image path like:
    <base>/<group_id>/<video_id>/keyframes/<keyframe_id>.webp
    """
    group_id: str = p.parent.parent.parent.name
    video_id: str = p.parent.parent.name
    keyframe_id: str = p.stem
    return group_id, video_id, keyframe_id

@app_cli.command("ingest_meta")
def ingest_metadata(
    keyframes_dir: Path = typer.Option(..., help="Root folder of keyframes"),
    captions_dir: Path = typer.Option(..., help="Root folder of caption JSONs"),
):
    async def _run():
        keyframe_paths = glob.glob(f"{keyframes_dir}/**/*.webp", recursive=True)
        keyframe_paths = sorted(keyframe_paths)

        caption_paths = glob.glob(f"{captions_dir}/**/*.json", recursive=True)
        caption_paths = sorted(
            caption_paths,
            key=lambda p: _parse_caption_parts(p, str(captions_dir))
        )

        keyframes: list[KeyframeInstance] = []
        for ident, (kf, cap_path) in tqdm(enumerate(zip(keyframe_paths, caption_paths))):
            g, v, kf_id = _parse_triplet_from_path(Path(kf))
            json_dict = json.load(open(cap_path, 'r', encoding='utf-8'))

            ocr = None
            if json_dict.get("ocr_blocks",None) is not None:
                try:

                    ocr_blocks = json_dict['ocr_blocks']
                    if isinstance(ocr_blocks, dict )and ocr_blocks.get('ocr_blocks'):
                        ocr_blocks = ocr_blocks['ocr_blocks']
                    ocr = [block['content'] for block in ocr_blocks]

                except Exception as e:
                    print(cap_path)
                    raise e

            tags = json_dict['tag_list']
            keyframes.append(
                    KeyframeInstance(
                        group_id=g,
                        video_id=v,
                        keyframe_id=kf_id,
                        identification=ident,
                        tags=tags,
                        ocr=ocr,
                    )
                )
            
        mongo_client = await init_mongo(settings.mongo_uri, settings.mongo_db)
        repo = KeyframeRepo()
        await repo.create_many([k.model_dump() for k in keyframes], ordered=False)
        mongo_client.close()
        logger.info(f"[MongoDB] Inserted {len(keyframes)} documents")

        # es_repo = ElasticsearchKeyframeRepo(
        #     hosts=settings.es_hosts,
        #     index=settings.es_index,
        #     api_key=settings.es_api_key,
        #     basic_auth=(settings.es_basic_user, settings.es_basic_pass),
        #     verify_certs=settings.es_verify_certs,
        # )
        # await es_repo.ensure_index(recreate=False)
        # await es_repo.bulk_upsert(keyframes, refresh=True)
        # await es_repo.es.close()
        logger.info(f"[Elasticsearch] Indexed {len(keyframes)} documents")
    asyncio.run(_run())






    







if __name__ == "__main__":
    app_cli()