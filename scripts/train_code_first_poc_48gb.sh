#!/usr/bin/env bash
set -euo pipefail

# 48 GB A40 cloud profile for the code-first POC.
#
# This is a smaller test scale than the 80 GB profile: shared width 1536 instead
# of 2048. Relative to the 640 baseline, that is about a 5.8x parameter increase.

print_usage() {
  echo "Usage:"
  echo "  ./scripts/train_code_first_poc_48gb.sh"
  echo "  TOFY_48GB_OOM_PROBE=1 ./scripts/train_code_first_poc_48gb.sh"
  echo "  ./scripts/train_code_first_poc_48gb.sh --resume latest"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  print_usage
  exit 0
fi

export TOFY_GPU_PROFILE="${TOFY_GPU_PROFILE:-48gb}"
export TOFY_TRAIN_DTYPE="${TOFY_TRAIN_DTYPE:-bf16}"

# Model-size profile. Override any of these in the environment before launch.
export DIM="${DIM:-1536}"
export BRIDGE_DIM="${BRIDGE_DIM:-1536}"
export LAYERS="${LAYERS:-7}"
export HEADS="${HEADS:-12}"
export MAX_VOCAB="${MAX_VOCAB:-12000}"
export CODE_DECODER_MAX_VOCAB="${CODE_DECODER_MAX_VOCAB:-24000}"
export NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-96}"

# Keep context at the proven POC shape initially; raise these only after an OOM probe.
export LATENT_MAX_SEQ="${LATENT_MAX_SEQ:-256}"
export WORLD_MAX_SEQ="${WORLD_MAX_SEQ:-256}"
export CODE_DECODER_MAX_SEQ="${CODE_DECODER_MAX_SEQ:-128}"

# Test-scale cloud budget: 3x the current code-first step counts.
export LATENT_STEPS="${LATENT_STEPS:-75000}"
export WORLD_STEPS="${WORLD_STEPS:-180000}"
export ROUTER_STEPS="${ROUTER_STEPS:-0}"
export CODE_DECODER_STEPS="${CODE_DECODER_STEPS:-120000}"
export CODE_POLISH_STEPS="${CODE_POLISH_STEPS:-24000}"

# Conservative A40 microbatches. Increase after a sustained probe passes.
export LATENT_BATCH="${LATENT_BATCH:-6}"
export LATENT_GRAD_ACCUM="${LATENT_GRAD_ACCUM:-4}"
export TOFY_LATENT_WARMUP_BATCH="${TOFY_LATENT_WARMUP_BATCH:-3}"
export TOFY_LATENT_WARMUP_GRAD_ACCUM="${TOFY_LATENT_WARMUP_GRAD_ACCUM:-2}"
export WORLD_BATCH="${WORLD_BATCH:-24}"
export WORLD_GRAD_ACCUM="${WORLD_GRAD_ACCUM:-3}"
export TOFY_WORLD_WARMUP_BATCH="${TOFY_WORLD_WARMUP_BATCH:-12}"
export TOFY_WORLD_WARMUP_GRAD_ACCUM="${TOFY_WORLD_WARMUP_GRAD_ACCUM:-1}"
export CODE_DECODER_BATCH="${CODE_DECODER_BATCH:-2}"
export CODE_DECODER_GRAD_ACCUM="${CODE_DECODER_GRAD_ACCUM:-6}"

export TOFY_CACHE_PREFETCH_BATCHES="${TOFY_CACHE_PREFETCH_BATCHES:-1}"
export TOFY_PLANNER_SEGMENT_BATCH="${TOFY_PLANNER_SEGMENT_BATCH:-16}"

detect_total_vram_mb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]'
}

TOTAL_VRAM_MB="$(detect_total_vram_mb || true)"
if [[ -n "${TOTAL_VRAM_MB}" && "${TOTAL_VRAM_MB}" -lt 44000 && "${TOFY_ALLOW_SMALL_VRAM:-0}" != "1" ]]; then
  echo "ERROR: detected ${TOTAL_VRAM_MB} MB VRAM, expected a 48 GB GPU."
  echo "Set TOFY_ALLOW_SMALL_VRAM=1 to bypass this guard for dry runs."
  exit 1
fi

if [[ "${TOFY_48GB_OOM_PROBE:-0}" == "1" ]]; then
  cargo run --release -- --sustained-oom-probe \
    --stage all \
    --dtype "${TOFY_TRAIN_DTYPE}" \
    --dim "${DIM}" \
    --max-seq "${LATENT_MAX_SEQ}" \
    --layers "${LAYERS}" \
    --heads "${HEADS}" \
    --bridge-dim "${BRIDGE_DIM}" \
    --planner-slots "${NUM_LATENT_TOKENS}" \
    --vocab "${MAX_VOCAB}" \
    --latent-batch "${LATENT_BATCH}" \
    --latent-accum "${LATENT_GRAD_ACCUM}" \
    --world-batch "${WORLD_BATCH}" \
    --world-accum "${WORLD_GRAD_ACCUM}" \
    --world-warmup-batch "${TOFY_WORLD_WARMUP_BATCH}" \
    --world-warmup-accum "${TOFY_WORLD_WARMUP_GRAD_ACCUM}" \
    --decoder-batch "${CODE_DECODER_BATCH}" \
    --decoder-accum "${CODE_DECODER_GRAD_ACCUM}" \
    --decoder-max-seq "${CODE_DECODER_MAX_SEQ}" \
    --decoder-max-vocab "${CODE_DECODER_MAX_VOCAB}" \
    --min-headroom-mb "${TOFY_48GB_MIN_HEADROOM_MB:-3072}" \
    --max-late-growth-mb "${TOFY_48GB_MAX_LATE_GROWTH_MB:-1536}"
fi

exec ./scripts/train_code_first_poc.sh "$@"
