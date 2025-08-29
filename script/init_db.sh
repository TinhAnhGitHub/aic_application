#!/usr/bin/env bash
set -e


KEYFRAME_EMB="/media/tinhanhnguyen/Projects/aic_application/data/beit3_large_itc_patch16_224_features_array.npy"
CAPTION_EMB="/media/tinhanhnguyen/Projects/aic_application/data/L21_L22_L23_L24_L25_L26_L27_L28_L29_L30_text_embedding.npy"

echo "[migration] Initializing Milvus collections..."
python /media/tinhanhnguyen/Projects/aic_application/app/migration/cli.py init \
  --keyframe-embedding-path "$KEYFRAME_EMB" \
  --caption-embedding-path "$CAPTION_EMB"
