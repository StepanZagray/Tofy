#!/usr/bin/env bash
set -euo pipefail

# Train the LeJEPA encoder with 25k steps.
# Typical default dataset is a mixed corpus prepared with:
#   cargo run --release -- --prepare-encoder-corpus --output data/encoder_mix.txt <ultrachat_pairs> <wikipedia_cache> <multilang_pairs>
# Override defaults through environment variables if needed.

print_usage() {
  echo "Usage:"
  echo "  ./scripts/train_encoder_25k.sh"
  echo "  TOFY_GPU_PROFILE=48gb ./scripts/train_encoder_25k.sh"
  echo "  TOFY_GPU_PROFILE=80gb ./scripts/train_encoder_25k.sh"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  print_usage
  exit 0
fi

DATASET="${DATASET:-data/encoder_mix.txt}"
STEPS="${STEPS:-25000}"
TOFY_GPU_PROFILE="${TOFY_GPU_PROFILE:-8gb}"
TOFY_TRAIN_DTYPE="${TOFY_TRAIN_DTYPE:-bf16}"
TOFY_LATENT_CONTEXT_SEGMENTS="${TOFY_LATENT_CONTEXT_SEGMENTS:-4}"
TOFY_LATENT_RECENT_FULL_SEGMENTS="${TOFY_LATENT_RECENT_FULL_SEGMENTS:-1}"
TOFY_LATENT_HISTORY_RATIO="${TOFY_LATENT_HISTORY_RATIO:-0.35}"
export TOFY_TRAIN_DTYPE TOFY_LATENT_CONTEXT_SEGMENTS TOFY_LATENT_RECENT_FULL_SEGMENTS TOFY_LATENT_HISTORY_RATIO
case "${TOFY_GPU_PROFILE}" in
  8gb)
    PROFILE_DIM=640
    PROFILE_MAX_SEQ=256
    PROFILE_LAYERS=7
    PROFILE_HEADS=8
    PROFILE_MAX_VOCAB=8000
    PROFILE_BATCH=12
    PROFILE_WARMUP_BATCH=12
    PROFILE_GRAD_ACCUM=2
    ;;
  48gb)
    PROFILE_DIM=1536
    PROFILE_MAX_SEQ=256
    PROFILE_LAYERS=7
    PROFILE_HEADS=12
    PROFILE_MAX_VOCAB=12000
    PROFILE_BATCH=6
    PROFILE_WARMUP_BATCH=3
    PROFILE_GRAD_ACCUM=4
    ;;
  80gb)
    PROFILE_DIM=2048
    PROFILE_MAX_SEQ=256
    PROFILE_LAYERS=7
    PROFILE_HEADS=16
    PROFILE_MAX_VOCAB=16000
    PROFILE_BATCH=4
    PROFILE_WARMUP_BATCH=2
    PROFILE_GRAD_ACCUM=8
    ;;
  *)
    echo "ERROR: unsupported TOFY_GPU_PROFILE='${TOFY_GPU_PROFILE}' (expected 8gb, 48gb, or 80gb)"
    exit 1
    ;;
esac

DIM="${DIM:-${PROFILE_DIM}}"
MAX_SEQ="${MAX_SEQ:-${PROFILE_MAX_SEQ}}"
LAYERS="${LAYERS:-${PROFILE_LAYERS}}"
HEADS="${HEADS:-${PROFILE_HEADS}}"
MAX_VOCAB="${MAX_VOCAB:-${PROFILE_MAX_VOCAB}}"
WIKI_MAX_FILES="${WIKI_MAX_FILES:-1}"

detect_total_vram_mb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]'
}

maybe_export_cuda_compat() {
  if [[ -z "${CUDA_COMPUTE_CAP:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    local compute_cap
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '.[:space:]')"
    if [[ -n "${compute_cap}" ]]; then
      export CUDA_COMPUTE_CAP="${compute_cap}"
    fi
  fi
}

TOTAL_VRAM_MB="$(detect_total_vram_mb || true)"

DEFAULT_BATCH="${PROFILE_BATCH}"
DEFAULT_WARMUP_BATCH="${PROFILE_WARMUP_BATCH}"
DEFAULT_GRAD_ACCUM="${PROFILE_GRAD_ACCUM}"

BATCH="${BATCH:-${DEFAULT_BATCH}}"
GRAD_ACCUM="${GRAD_ACCUM:-${DEFAULT_GRAD_ACCUM}}"
TOFY_LATENT_WARMUP_BATCH="${TOFY_LATENT_WARMUP_BATCH:-${DEFAULT_WARMUP_BATCH:-${BATCH}}}"
TOFY_LATENT_WARMUP_GRAD_ACCUM="${TOFY_LATENT_WARMUP_GRAD_ACCUM:-1}"
export TOFY_LATENT_WARMUP_BATCH TOFY_LATENT_WARMUP_GRAD_ACCUM

if [[ ! -f "${DATASET}" && "${DATASET}" == "data/encoder_mix.txt" ]]; then
  echo "ERROR: default encoder corpus '${DATASET}' not found."
  echo "  Build it with cargo run --release -- --prepare-encoder-corpus --output data/encoder_mix.txt ... or set DATASET=<path>."
  exit 1
fi

echo "Training LeJEPA encoder: dataset=${DATASET} steps=${STEPS} batch=${BATCH} grad_accum=${GRAD_ACCUM} effective_batch=$((BATCH * GRAD_ACCUM)) dim=${DIM} gpu_profile=${TOFY_GPU_PROFILE} vram_mb=${TOTAL_VRAM_MB:-unknown} dtype=${TOFY_TRAIN_DTYPE} latent_segments=${TOFY_LATENT_CONTEXT_SEGMENTS} recent_full=${TOFY_LATENT_RECENT_FULL_SEGMENTS} history_ratio=${TOFY_LATENT_HISTORY_RATIO} warmup_batch=${TOFY_LATENT_WARMUP_BATCH} warmup_accum=${TOFY_LATENT_WARMUP_GRAD_ACCUM} warmup_effective=$((TOFY_LATENT_WARMUP_BATCH * TOFY_LATENT_WARMUP_GRAD_ACCUM)) warmup_steps=${TOFY_LATENT_WARMUP_STEPS:-20%}"
maybe_export_cuda_compat
if [[ -n "${CUDA_COMPUTE_CAP:-}" ]]; then
  echo "CUDA build env: CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-unset}"
fi
if [[ "${DATASET}" == hub:*wikipedia* ]]; then
  echo "Applying Wikipedia cap: JEPA_WIKI_MAX_FILES=${WIKI_MAX_FILES}"
  JEPA_WIKI_MAX_FILES="${WIKI_MAX_FILES}" cargo run --release -- --latent "${DATASET}" "${STEPS}" "${BATCH}" "${DIM}" "${MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}" --grad-accum "${GRAD_ACCUM}"
else
  cargo run --release -- --latent "${DATASET}" "${STEPS}" "${BATCH}" "${DIM}" "${MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}" --grad-accum "${GRAD_ACCUM}"
fi
