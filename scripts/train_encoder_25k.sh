#!/usr/bin/env bash
set -euo pipefail

# Train the LeJEPA encoder with 25k steps.
# Typical default dataset is a mixed corpus prepared with:
#   python scripts/prepare_encoder_corpus.py --output data/encoder_mix.txt <ultrachat_pairs> <wikipedia_cache> <multilang_pairs>
# Override defaults through environment variables if needed.
DATASET="${DATASET:-data/encoder_mix.txt}"
STEPS="${STEPS:-25000}"
TOFY_GPU_PROFILE="${TOFY_GPU_PROFILE:-auto}"
TOFY_TRAIN_DTYPE="${TOFY_TRAIN_DTYPE:-bf16}"
TOFY_LATENT_CONTEXT_SEGMENTS="${TOFY_LATENT_CONTEXT_SEGMENTS:-4}"
TOFY_LATENT_RECENT_FULL_SEGMENTS="${TOFY_LATENT_RECENT_FULL_SEGMENTS:-1}"
TOFY_LATENT_HISTORY_RATIO="${TOFY_LATENT_HISTORY_RATIO:-0.35}"
export TOFY_TRAIN_DTYPE TOFY_LATENT_CONTEXT_SEGMENTS TOFY_LATENT_RECENT_FULL_SEGMENTS TOFY_LATENT_HISTORY_RATIO
DIM="${DIM:-640}"
MAX_SEQ="${MAX_SEQ:-256}"
LAYERS="${LAYERS:-7}"
HEADS="${HEADS:-8}"
MAX_VOCAB="${MAX_VOCAB:-8000}"
WIKI_MAX_FILES="${WIKI_MAX_FILES:-1}"

detect_total_vram_mb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '[:space:]'
}

maybe_export_cuda_compat() {
  if ! command -v nvcc >/dev/null 2>&1; then
    return 0
  fi
  local nvcc_release
  nvcc_release="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)"
  if [[ -n "${nvcc_release}" ]] && [[ "${nvcc_release}" == "13.2" || "${nvcc_release}" == 13.[3-9]* || "${nvcc_release}" == 1[4-9].* ]]; then
    export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-13010}"
  fi
  if [[ -z "${CUDA_COMPUTE_CAP:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    local compute_cap
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d '.[:space:]')"
    if [[ -n "${compute_cap}" ]]; then
      export CUDA_COMPUTE_CAP="${compute_cap}"
    fi
  fi
}

TOTAL_VRAM_MB="$(detect_total_vram_mb || true)"
if [[ "${TOFY_GPU_PROFILE}" == "auto" ]]; then
  if [[ -n "${TOTAL_VRAM_MB}" && "${TOTAL_VRAM_MB}" -le 9000 ]]; then
    TOFY_GPU_PROFILE="8gb"
  else
    TOFY_GPU_PROFILE="balanced"
  fi
fi

case "${TOFY_GPU_PROFILE}" in
  8gb)
    DEFAULT_BATCH=32
    DEFAULT_WARMUP_BATCH=32
    DEFAULT_GRAD_ACCUM=1
    ;;
  balanced)
    DEFAULT_BATCH=8
    DEFAULT_WARMUP_BATCH=8
    DEFAULT_GRAD_ACCUM=1
    ;;
  *)
    echo "ERROR: unsupported TOFY_GPU_PROFILE='${TOFY_GPU_PROFILE}' (expected auto, 8gb, or balanced)"
    exit 1
    ;;
esac

BATCH="${BATCH:-${DEFAULT_BATCH}}"
GRAD_ACCUM="${GRAD_ACCUM:-${DEFAULT_GRAD_ACCUM}}"
TOFY_LATENT_WARMUP_BATCH="${TOFY_LATENT_WARMUP_BATCH:-${DEFAULT_WARMUP_BATCH:-${BATCH}}}"
TOFY_LATENT_WARMUP_GRAD_ACCUM="${TOFY_LATENT_WARMUP_GRAD_ACCUM:-1}"
export TOFY_LATENT_WARMUP_BATCH TOFY_LATENT_WARMUP_GRAD_ACCUM

if [[ ! -f "${DATASET}" && "${DATASET}" == "data/encoder_mix.txt" ]]; then
  echo "ERROR: default encoder corpus '${DATASET}' not found."
  echo "  Build it with scripts/prepare_encoder_corpus.py or set DATASET=<path>."
  exit 1
fi

echo "Training LeJEPA encoder: dataset=${DATASET} steps=${STEPS} batch=${BATCH} grad_accum=${GRAD_ACCUM} effective_batch=$((BATCH * GRAD_ACCUM)) dim=${DIM} gpu_profile=${TOFY_GPU_PROFILE} vram_mb=${TOTAL_VRAM_MB:-unknown} dtype=${TOFY_TRAIN_DTYPE} latent_segments=${TOFY_LATENT_CONTEXT_SEGMENTS} recent_full=${TOFY_LATENT_RECENT_FULL_SEGMENTS} history_ratio=${TOFY_LATENT_HISTORY_RATIO} warmup_batch=${TOFY_LATENT_WARMUP_BATCH} warmup_accum=${TOFY_LATENT_WARMUP_GRAD_ACCUM} warmup_effective=$((TOFY_LATENT_WARMUP_BATCH * TOFY_LATENT_WARMUP_GRAD_ACCUM)) warmup_steps=${TOFY_LATENT_WARMUP_STEPS:-20%}"
maybe_export_cuda_compat
if [[ -n "${CUDARC_CUDA_VERSION:-}" || -n "${CUDA_COMPUTE_CAP:-}" ]]; then
  echo "CUDA build env: CUDARC_CUDA_VERSION=${CUDARC_CUDA_VERSION:-unset} CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-unset}"
fi
if [[ "${DATASET}" == hub:*wikipedia* ]]; then
  echo "Applying Wikipedia cap: JEPA_WIKI_MAX_FILES=${WIKI_MAX_FILES}"
  JEPA_WIKI_MAX_FILES="${WIKI_MAX_FILES}" cargo run --release -- --latent "${DATASET}" "${STEPS}" "${BATCH}" "${DIM}" "${MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}" --grad-accum "${GRAD_ACCUM}"
else
  cargo run --release -- --latent "${DATASET}" "${STEPS}" "${BATCH}" "${DIM}" "${MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}" --grad-accum "${GRAD_ACCUM}"
fi
