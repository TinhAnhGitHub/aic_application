#!/usr/bin/env bash
set -e



KEYFRAME_DIR="/media/tinhanhnguyen/Projects/HCMAI/data/keyframe"
CAPTION_DIR="/media/tinhanhnguyen/Projects/HCMAI/data/kf_metadata"



echo "========== [3/3] Ingest metadata to MongoDB + Elasticsearch =========="

python /media/tinhanhnguyen/Projects/aic_application/app/migration/cli.py ingest_meta \
    --keyframes-dir "$KEYFRAME_DIR" \
    --captions-dir "$CAPTION_DIR" \