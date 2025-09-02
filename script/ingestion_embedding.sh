#!/usr/bin/env bash
set -e



KEYFRAME_DIR="/media/tinhanhnguyen/Projects/HCMAI/data/asr_chunking"
CAPTION_DIR="/media/tinhanhnguyen/Projects/HCMAI/data/keyframe_output"
KEYFRAME_EMBEDDING="/media/tinhanhnguyen/Projects/aic_application/data/beit3_large_itc_patch16_224_features_array.npy"
CAPTION_EMBEDDING="/media/tinhanhnguyen/Projects/aic_application/data/L21_L22_L23_L24_L25_L26_L27_L28_L29_L30_text_embedding.npy"

echo "[ingestion] Ingesting embeddings..."
python //media/tinhanhnguyen/Projects/aic_application/app/migration/cli.py ingest_embedding \
    --keyframes-dir "$KEYFRAME_DIR" \
    --captions-dir "$CAPTION_DIR" \
    --keyframe-embedding-path "$KEYFRAME_EMBEDDING" \
    --caption-embedding-path "$CAPTION_EMBEDDING"