#!/usr/bin/env bash
set -e


KEYFRAME_EMB="/media/tinhanhnguyen/Projects/HCMAI/data/embedding/total_embedding.npy"
CAPTION_EMB="/media/tinhanhnguyen/Projects/HCMAI/data/embedding/total_caption_embedding.npy"

echo "[migration] Initializing Milvus collections..."
python /media/tinhanhnguyen/Projects/aic_application/app/migration/cli.py init \
  --keyframe-embedding-path "$KEYFRAME_EMB" \
  --caption-embedding-path "$CAPTION_EMB"
