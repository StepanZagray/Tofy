#!/usr/bin/env bash
set -euo pipefail

# Full training order:
# 1) Encoder pretraining (JEPA latent)
# 2) World model warm start from encoder
# 3) End-to-end world fine-tuning (encoder + transition + bridge together)

LATENT_DATA="${LATENT_DATA:-hub:wikimedia/wikipedia}"
WORLD_DATA="${WORLD_DATA:-data/ultrachat_pairs.txt}"
WIKI_MAX_FILES="${WIKI_MAX_FILES:-1}"

LATENT_STEPS="${LATENT_STEPS:-25000}"
WORLD_STAGE1_STEPS="${WORLD_STAGE1_STEPS:-30000}"
WORLD_STAGE2_STEPS="${WORLD_STAGE2_STEPS:-10000}"

BATCH="${BATCH:-32}"
DIM="${DIM:-368}"
LATENT_MAX_SEQ="${LATENT_MAX_SEQ:-128}"
WORLD_MAX_SEQ="${WORLD_MAX_SEQ:-128}"
LAYERS="${LAYERS:-4}"
HEADS="${HEADS:-8}"
MAX_VOCAB="${MAX_VOCAB:-8000}"
BRIDGE_DIM="${BRIDGE_DIM:-256}"

WORLD_LR1="${WORLD_LR1:-2e-4}"
WORLD_LR2="${WORLD_LR2:-1e-4}"

if [[ ! -f "${WORLD_DATA}" ]]; then
  echo "ERROR: WORLD_DATA not found at '${WORLD_DATA}'"
  echo "Prepare UltraChat pairs first:"
  echo "  cargo run --release -- --prepare-ultrachat data/ultrachat_pairs.txt 6 2"
  exit 1
fi

echo "== Stage 1/3: Encoder pretraining (JEPA latent) =="
if [[ "${LATENT_DATA}" == hub:*wikipedia* ]]; then
  echo "Applying Wikipedia cap for encoder stage: JEPA_WIKI_MAX_FILES=${WIKI_MAX_FILES}"
  JEPA_WIKI_MAX_FILES="${WIKI_MAX_FILES}" cargo run --release -- --latent "${LATENT_DATA}" "${LATENT_STEPS}" "${BATCH}" "${DIM}" "${LATENT_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}"
else
  cargo run --release -- --latent "${LATENT_DATA}" "${LATENT_STEPS}" "${BATCH}" "${DIM}" "${LATENT_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}"
fi

LATENT_MODEL="${LATENT_MODEL:-$(ls -1t model_latent_*.safetensors 2>/dev/null | awk '!/_teacher/ {print; exit}')}"
if [[ -z "${LATENT_MODEL}" ]]; then
  echo "ERROR: could not find model_latent_*.safetensors. Set LATENT_MODEL explicitly."
  exit 1
fi
echo "Using latent checkpoint: ${LATENT_MODEL}"

echo "== Stage 2/3: World model warm start from encoder =="
cargo run --release -- --train-world "${WORLD_DATA}" "${WORLD_STAGE1_STEPS}" "${BATCH}" "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}" "${BRIDGE_DIM}" --init-encoder "${LATENT_MODEL}" --lr "${WORLD_LR1}"

WORLD_MODEL="${WORLD_MODEL:-$(ls -1t model_world_*.safetensors 2>/dev/null | awk '!/_teacher/ {print; exit}')}"
if [[ -z "${WORLD_MODEL}" ]]; then
  echo "ERROR: could not find model_world_*.safetensors after stage 2."
  exit 1
fi
echo "Using world checkpoint for end-to-end tuning: ${WORLD_MODEL}"

echo "== Stage 3/3: End-to-end world tuning (encoder + transition + bridge) =="
cargo run --release -- --train-world "${WORLD_DATA}" "${WORLD_STAGE2_STEPS}" "${BATCH}" "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}" "${BRIDGE_DIM}" --init-encoder "${WORLD_MODEL}" --lr "${WORLD_LR2}"

echo "Pipeline complete."
