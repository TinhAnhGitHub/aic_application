import typer
import numpy as np
from pathlib import Path
import asyncio
import os
import sys

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
# from app.services.sparse_encoder import MilvusSparseEncoder



from app.migration.helper import (
    _collect_rows_from_fs,
    _build_bm25_corpus,
    _insert_keyframes_milvus,
    _insert_caption_milvus,
    _insert_mongo,
)
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


@app_cli.command('ingest')
def ingest_all(
    keyframes_dir: Path = typer.Option(..., help="Root folder of keyframes"),
    captions_dir: Path = typer.Option(..., help="Root folder of caption JSONs"),
    keyframe_embedding_path: Path = typer.Option(..., help="Path to keyframe embeddings .npy"),
    caption_embedding_path: Path = typer.Option(..., help="Path to the caption embeddings .npy"),
    bm25_model_out: Path = typer.Option(Path(settings.bm25_model_path or "bm25.json"), help="Where to save BM25 model"),
):
    async def _run():
        rows = _collect_rows_from_fs(keyframes_dir, captions_dir)
        ids = [r.identification for r in rows]
        kf_embed = np.load(keyframe_embedding_path)
        cap_embed = np.load(caption_embedding_path)

        if len(rows) != kf_embed.shape[0] or len(rows) != cap_embed.shape[0]:
            raise RuntimeError(f"Row mismatch: rows={len(rows)}, kf={kf_embed.shape[0]}, cap={cap_embed.shape[0]}")
        
        bm25_docs = _build_bm25_corpus(rows)

        bm25 = MilvusSparseEncoder(language=settings.bm25_language)
        bm25.fit(bm25_docs)
        bm25.save(str(bm25_model_out))

        docs_sparse_2d = bm25.encode_documents(bm25_docs)


        # mongo
        mongo_client = await init_mongo(settings.mongo_uri, settings.mongo_db)
        kf_repo = KeyframeRepo()
        es_repo = ElasticsearchKeyframeRepo(
            hosts=settings.es_hosts,
            api_key=settings.es_api_key,
            basic_auth=(settings.es_basic_user, settings.es_basic_pass),
            verify_certs=settings.es_verify_certs
        )
        await es_repo.ensure_index()

        milvus_client = AsyncMilvusClient(uri=settings.milvus_uri)
        await _ensure_keyframe_collection(milvus_client, dim=int(kf_embed.shape[1]))
        await _ensure_keyframe_collection(milvus_client,dim=int(kf_embed.shape[1]))

        # insert
        await _insert_keyframes_milvus(client=milvus_client, ids=ids, kf_emb=kf_embed)
        await _insert_caption_milvus(client=milvus_client, ids=ids, caption_emb=cap_embed, caption_sparse_rows=docs_sparse_2d)

        await _insert_mongo(kf_repo, rows)

        async def _doc_iter():
            for r in rows:
                yield KeyframeInstance(
                    group_id=r.group_id,
                    video_id=r.video_id,
                    keyframe_id=r.keyframe_id,
                    identification=r.identification,
                    tags=r.tags or None,
                    ocr=r.ocr or None,
                )
        await es_repo.bulk_upsert(_doc_iter(), refresh=True)

        try: await es_repo.es.close()
        except: pass
        await milvus_client.close()
        await mongo_client.close()
        print(f"[ingest] inserted {len(rows)} rows. BM25 saved to {bm25_model_out}")


    asyncio.run(_run())

if __name__ == "__main__":
    app_cli()