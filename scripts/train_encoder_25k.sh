#!/usr/bin/env bash
set -euo pipefail

# Train JEPA encoder with 25k steps (default recommendation).
# Override defaults through environment variables if needed.
DATASET="${DATASET:-hub:wikimedia/wikipedia}"
STEPS="${STEPS:-25000}"
BATCH="${BATCH:-32}"
DIM="${DIM:-368}"
MAX_SEQ="${MAX_SEQ:-128}"
LAYERS="${LAYERS:-4}"
HEADS="${HEADS:-8}"
MAX_VOCAB="${MAX_VOCAB:-8000}"
WIKI_MAX_FILES="${WIKI_MAX_FILES:-1}"

echo "Training encoder: dataset=${DATASET} steps=${STEPS} batch=${BATCH} dim=${DIM}"
if [[ "${DATASET}" == hub:*wikipedia* ]]; then
  echo "Applying Wikipedia cap: JEPA_WIKI_MAX_FILES=${WIKI_MAX_FILES}"
  JEPA_WIKI_MAX_FILES="${WIKI_MAX_FILES}" cargo run --release -- --latent "${DATASET}" "${STEPS}" "${BATCH}" "${DIM}" "${MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}"
else
  cargo run --release -- --latent "${DATASET}" "${STEPS}" "${BATCH}" "${DIM}" "${MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}"
fi
