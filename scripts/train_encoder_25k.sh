#!/usr/bin/env bash
set -euo pipefail

# Train the LeJEPA encoder with 25k steps.
# Typical default dataset is a mixed corpus prepared with:
#   python scripts/prepare_encoder_corpus.py --output data/encoder_mix.txt <ultrachat_pairs> <wikipedia_cache> <multilang_pairs>
# Override defaults through environment variables if needed.
DATASET="${DATASET:-data/encoder_mix.txt}"
STEPS="${STEPS:-25000}"
BATCH="${BATCH:-32}"
DIM="${DIM:-768}"
MAX_SEQ="${MAX_SEQ:-128}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"
MAX_VOCAB="${MAX_VOCAB:-8000}"
WIKI_MAX_FILES="${WIKI_MAX_FILES:-1}"

if [[ ! -f "${DATASET}" && "${DATASET}" == "data/encoder_mix.txt" ]]; then
  echo "ERROR: default encoder corpus '${DATASET}' not found."
  echo "  Build it with scripts/prepare_encoder_corpus.py or set DATASET=<path>."
  exit 1
fi

echo "Training LeJEPA encoder: dataset=${DATASET} steps=${STEPS} batch=${BATCH} dim=${DIM}"
if [[ "${DATASET}" == hub:*wikipedia* ]]; then
  echo "Applying Wikipedia cap: JEPA_WIKI_MAX_FILES=${WIKI_MAX_FILES}"
  JEPA_WIKI_MAX_FILES="${WIKI_MAX_FILES}" cargo run --release -- --latent "${DATASET}" "${STEPS}" "${BATCH}" "${DIM}" "${MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}"
else
  cargo run --release -- --latent "${DATASET}" "${STEPS}" "${BATCH}" "${DIM}" "${MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}"
fi
