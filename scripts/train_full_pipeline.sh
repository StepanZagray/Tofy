#!/usr/bin/env bash
set -euo pipefail

# Full training pipeline (encoder -> world transition -> planner/orchestrator tune -> code decoder -> text decoder).
# Override via env: ENCODER_DATA, WORLD_DATA, CODE_DATA, TEXT_DATA, WIKI_DATA, *_STEPS, BATCH, DIM, etc.
#
# 1) LeJEPA encoder (--latent)
# 2) Pure latent world model (--train-world) using frozen encoder artifacts
# 3) Planner/orchestrator action tune (--train-orchestrator)
# 4) Code decoder (--train-decoder --decoder-kind code) on CODE_DATA
# 5) Text decoder (--train-decoder --decoder-kind text) on TEXT_DATA

WORLD_TEXT_DATA="${WORLD_TEXT_DATA:-data/ultrachat_pairs.txt}"
WORLD_DATA="${WORLD_DATA:-data/world_mix_pairs.txt}"
CODE_DATA="${CODE_DATA:-data/multilang_pairs.txt}"
TEXT_DATA="${TEXT_DATA:-data/ultrachat_pairs.txt}"
WIKI_DATA="${WIKI_DATA:-data/cached_wikimedia_wikipedia_1.txt}"
ENCODER_DATA="${ENCODER_DATA:-data/encoder_mix.txt}"
TOFY_GPU_PROFILE="${TOFY_GPU_PROFILE:-auto}"
TOFY_TRAIN_DTYPE="${TOFY_TRAIN_DTYPE:-bf16}"
TOFY_LATENT_CONTEXT_SEGMENTS="${TOFY_LATENT_CONTEXT_SEGMENTS:-4}"
TOFY_LATENT_RECENT_FULL_SEGMENTS="${TOFY_LATENT_RECENT_FULL_SEGMENTS:-1}"
TOFY_LATENT_HISTORY_RATIO="${TOFY_LATENT_HISTORY_RATIO:-0.35}"
TOFY_DECODER_SYNTAX_LOSS_WEIGHT="${TOFY_DECODER_SYNTAX_LOSS_WEIGHT:-0.35}"
export TOFY_TRAIN_DTYPE TOFY_LATENT_CONTEXT_SEGMENTS TOFY_LATENT_RECENT_FULL_SEGMENTS TOFY_LATENT_HISTORY_RATIO TOFY_DECODER_SYNTAX_LOSS_WEIGHT

LATENT_STEPS="${LATENT_STEPS:-25000}"
WORLD_STEPS="${WORLD_STEPS:-60000}"
ROUTER_STEPS="${ROUTER_STEPS:-15000}"
CODE_DECODER_STEPS="${CODE_DECODER_STEPS:-40000}"
TEXT_DECODER_STEPS="${TEXT_DECODER_STEPS:-40000}"

DIM="${DIM:-768}"
LATENT_MAX_SEQ="${LATENT_MAX_SEQ:-256}"
WORLD_MAX_SEQ="${WORLD_MAX_SEQ:-256}"
DECODER_MAX_SEQ="${DECODER_MAX_SEQ:-192}"
CODE_DECODER_MAX_SEQ="${CODE_DECODER_MAX_SEQ:-192}"
TEXT_DECODER_MAX_SEQ="${TEXT_DECODER_MAX_SEQ:-128}"
LAYERS="${LAYERS:-9}"
HEADS="${HEADS:-8}"
MAX_VOCAB="${MAX_VOCAB:-8000}"
CODE_DECODER_MAX_VOCAB="${CODE_DECODER_MAX_VOCAB:-16000}"
TEXT_DECODER_MAX_VOCAB="${TEXT_DECODER_MAX_VOCAB:-16000}"
BRIDGE_DIM="${BRIDGE_DIM:-256}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
WIKI_MAX_FILES="${WIKI_MAX_FILES:-1}"
WORLD_LR="${WORLD_LR:-2e-4}"
WORLD_LAMBDA="${WORLD_LAMBDA:-0.2}"
WORLD_ACTION_LOSS_WEIGHT="${WORLD_ACTION_LOSS_WEIGHT:-1.0}"
WORLD_ROUTER_WARMUP="${WORLD_ROUTER_WARMUP:-5000}"
WORLD_CODE_RATIO="${WORLD_CODE_RATIO:-0.35}"
WORLD_DONE_RATIO="${WORLD_DONE_RATIO:-0.18}"
WORLD_MAX_ROWS="${WORLD_MAX_ROWS:-0}"
ENCODER_VOCAB="${ENCODER_VOCAB:-local_models/vocabs/vocab_encoder.txt}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

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
    DEFAULT_BATCH=2
    DEFAULT_DECODER_BATCH=4
    DEFAULT_LATENT_GRAD_ACCUM=3
    DEFAULT_WORLD_GRAD_ACCUM=1
    DEFAULT_DECODER_GRAD_ACCUM=2
    ;;
  balanced)
    DEFAULT_BATCH=12
    DEFAULT_DECODER_BATCH=8
    DEFAULT_LATENT_GRAD_ACCUM=1
    DEFAULT_WORLD_GRAD_ACCUM=1
    DEFAULT_DECODER_GRAD_ACCUM=1
    ;;
  *)
    echo "ERROR: unsupported TOFY_GPU_PROFILE='${TOFY_GPU_PROFILE}' (expected auto, 8gb, or balanced)"
    exit 1
    ;;
esac

BATCH="${BATCH:-${DEFAULT_BATCH}}"
DECODER_BATCH="${DECODER_BATCH:-${DEFAULT_DECODER_BATCH}}"
LATENT_BATCH="${LATENT_BATCH:-${BATCH}}"
WORLD_BATCH="${WORLD_BATCH:-${BATCH}}"
CODE_DECODER_BATCH="${CODE_DECODER_BATCH:-${DECODER_BATCH}}"
TEXT_DECODER_BATCH="${TEXT_DECODER_BATCH:-${DECODER_BATCH}}"
DECODER_GRAD_ACCUM="${DECODER_GRAD_ACCUM:-${DEFAULT_DECODER_GRAD_ACCUM}}"
LATENT_GRAD_ACCUM="${LATENT_GRAD_ACCUM:-${DEFAULT_LATENT_GRAD_ACCUM}}"
WORLD_GRAD_ACCUM="${WORLD_GRAD_ACCUM:-${DEFAULT_WORLD_GRAD_ACCUM}}"
ROUTER_BATCH="${ROUTER_BATCH:-${WORLD_BATCH}}"
ROUTER_GRAD_ACCUM="${ROUTER_GRAD_ACCUM:-${WORLD_GRAD_ACCUM}}"
CODE_DECODER_GRAD_ACCUM="${CODE_DECODER_GRAD_ACCUM:-${DECODER_GRAD_ACCUM}}"
TEXT_DECODER_GRAD_ACCUM="${TEXT_DECODER_GRAD_ACCUM:-${DECODER_GRAD_ACCUM}}"

PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-pipeline_$(date +%Y-%m-%d_%H-%M-%S)}"
PIPELINE_RUN_ROOT="runs/${PIPELINE_RUN_ID}"
mkdir -p "${PIPELINE_RUN_ROOT}"
cat > "${PIPELINE_RUN_ROOT}/meta.json" <<EOF
{
  "pipeline_run_id": "${PIPELINE_RUN_ID}",
  "encoder_data": "${ENCODER_DATA}",
  "world_data": "${WORLD_DATA}",
  "code_data": "${CODE_DATA}",
  "text_data": "${TEXT_DATA}",
  "latent_steps": "${LATENT_STEPS}",
  "world_steps": "${WORLD_STEPS}",
  "router_steps": "${ROUTER_STEPS}",
  "code_decoder_steps": "${CODE_DECODER_STEPS}",
  "text_decoder_steps": "${TEXT_DECODER_STEPS}",
  "gpu_profile": "${TOFY_GPU_PROFILE}",
  "total_vram_mb": "${TOTAL_VRAM_MB}",
  "latent_batch": "${LATENT_BATCH}",
  "world_batch": "${WORLD_BATCH}",
  "code_decoder_batch": "${CODE_DECODER_BATCH}",
  "text_decoder_batch": "${TEXT_DECODER_BATCH}",
  "latent_grad_accum": "${LATENT_GRAD_ACCUM}",
  "world_grad_accum": "${WORLD_GRAD_ACCUM}",
  "code_decoder_grad_accum": "${CODE_DECODER_GRAD_ACCUM}",
  "text_decoder_grad_accum": "${TEXT_DECODER_GRAD_ACCUM}",
  "dim": "${DIM}",
  "layers": "${LAYERS}",
  "heads": "${HEADS}",
  "bridge_dim": "${BRIDGE_DIM}",
  "num_latent_tokens": "${NUM_LATENT_TOKENS}"
}
EOF
echo "Pipeline run directory: ${PIPELINE_RUN_ROOT}"
echo "GPU profile: ${TOFY_GPU_PROFILE} (vram_mb=${TOTAL_VRAM_MB:-unknown})"
echo "Microbatches: latent=${LATENT_BATCH} world=${WORLD_BATCH} code_decoder=${CODE_DECODER_BATCH} text_decoder=${TEXT_DECODER_BATCH}"
echo "Grad accum: latent=${LATENT_GRAD_ACCUM} world=${WORLD_GRAD_ACCUM} code_decoder=${CODE_DECODER_GRAD_ACCUM} text_decoder=${TEXT_DECODER_GRAD_ACCUM}"
echo "Training dtype: ${TOFY_TRAIN_DTYPE} | latent_segments=${TOFY_LATENT_CONTEXT_SEGMENTS} recent_full=${TOFY_LATENT_RECENT_FULL_SEGMENTS} history_ratio=${TOFY_LATENT_HISTORY_RATIO}"

maybe_export_cuda_compat
if [[ -n "${CUDARC_CUDA_VERSION:-}" || -n "${CUDA_COMPUTE_CAP:-}" ]]; then
  echo "CUDA build env: CUDARC_CUDA_VERSION=${CUDARC_CUDA_VERSION:-unset} CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-unset}"
fi

# --- Checks ---
if [[ ! -f "${WORLD_TEXT_DATA}" ]]; then
  echo "ERROR: WORLD_TEXT_DATA not found at '${WORLD_TEXT_DATA}'"
  echo "  Prepare chat pairs: cargo run --release -- --prepare-ultrachat data/ultrachat_pairs.txt 6 2"
  echo "  Or code pairs: ${PYTHON_BIN} scripts/prepare_github_top_code.py --default-languages --max-files 100000 --output data/multilang_pairs.txt"
  exit 1
fi

if [[ ! -f "${CODE_DATA}" ]]; then
  echo "CODE_DATA not found at '${CODE_DATA}', generating multilingual code pairs..."
  "${PYTHON_BIN}" -c "import datasets" >/dev/null 2>&1 || {
    echo "ERROR: Python package 'datasets' is required to generate ${CODE_DATA}."
    echo "  Create a repo venv and install it with:"
    echo "    python -m venv .venv && .venv/bin/pip install datasets"
    echo "  Then rerun this script, or set PYTHON_BIN=/path/to/python."
    exit 1
  }
  "${PYTHON_BIN}" scripts/prepare_github_top_code.py --output "${CODE_DATA}" --default-languages --max-files 200000
fi

if [[ ! -f "${WIKI_DATA}" ]]; then
  echo "ERROR: WIKI_DATA not found at '${WIKI_DATA}'"
  echo "  Expected a downloaded Wikipedia cache, e.g. data/cached_wikimedia_wikipedia_1.txt"
  exit 1
fi

if [[ -n "${TEXT_DATA}" && ! -f "${TEXT_DATA}" ]]; then
  echo "ERROR: TEXT_DATA set but not found at '${TEXT_DATA}'"
  exit 1
fi

echo "Preparing encoder corpus at ${ENCODER_DATA}"
"${PYTHON_BIN}" scripts/prepare_encoder_corpus.py --output "${ENCODER_DATA}" "${WORLD_TEXT_DATA}" "${WIKI_DATA}" "${CODE_DATA}"

echo "Preparing world-model mix at ${WORLD_DATA}"
"${PYTHON_BIN}" scripts/prepare_world_mix.py \
  --output "${WORLD_DATA}" \
  --text-pairs "${WORLD_TEXT_DATA}" \
  --code-pairs "${CODE_DATA}" \
  --code-ratio "${WORLD_CODE_RATIO}" \
  --done-ratio "${WORLD_DONE_RATIO}" \
  --max-rows "${WORLD_MAX_ROWS}"

# --- Stage 1: LeJEPA encoder ---
echo "== Stage 1/5: LeJEPA encoder (--latent) =="
TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="latent" cargo run --release -- --latent "${ENCODER_DATA}" "${LATENT_STEPS}" "${LATENT_BATCH}" "${DIM}" "${LATENT_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}" --grad-accum "${LATENT_GRAD_ACCUM}"

LATENT_MODEL="${LATENT_MODEL:-$(ls -1t local_models/model_latent_*.safetensors 2>/dev/null | awk '{print; exit}')}"
if [[ -z "${LATENT_MODEL}" ]]; then
  echo "ERROR: No local_models/model_latent_*.safetensors found. Set LATENT_MODEL explicitly."
  exit 1
fi
if [[ ! -f "${ENCODER_VOCAB}" ]]; then
  echo "ERROR: ${ENCODER_VOCAB} not found after encoder training."
  exit 1
fi
echo "  Using: ${LATENT_MODEL}"

# --- Stage 2: Planner/world model ---
echo "== Stage 2/5: Planner/world model (--train-world) =="
TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="world" cargo run --release -- --train-world "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_DATA}" "${WORLD_STEPS}" "${WORLD_BATCH}" "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" --lambda "${WORLD_LAMBDA}" --lr "${WORLD_LR}" --grad-accum "${WORLD_GRAD_ACCUM}" --action-loss-weight "${WORLD_ACTION_LOSS_WEIGHT}" --router-warmup "${WORLD_ROUTER_WARMUP}"

WORLD_MODEL="${WORLD_MODEL:-$(ls -1t local_models/model_world_*.safetensors 2>/dev/null | awk '{print; exit}')}"
if [[ -z "${WORLD_MODEL}" ]]; then
  echo "ERROR: No local_models/model_world_*.safetensors found after world training."
  exit 1
fi
echo "  Using: ${WORLD_MODEL}"

# --- Stage 3: Orchestrator/planner fine-tune ---
echo "== Stage 3/5: Orchestrator/planner (--train-orchestrator) =="
TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="orchestrator" cargo run --release -- --train-orchestrator "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${WORLD_DATA}" "${ROUTER_STEPS}" "${ROUTER_BATCH}" "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" --lr "${WORLD_LR}" --grad-accum "${ROUTER_GRAD_ACCUM}" --output "${WORLD_MODEL}"

# --- Stage 4: Code decoder ---
echo "== Stage 4/5: Code decoder (--train-decoder --decoder-kind code) =="
if [[ ! -f "${CODE_DATA}" ]]; then
  echo "  Skipping (CODE_DATA not found: ${CODE_DATA})"
else
  TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="decoder_code" cargo run --release -- --train-decoder "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${CODE_DATA}" "${CODE_DECODER_STEPS}" "${CODE_DECODER_BATCH}" "${CODE_DECODER_MAX_SEQ}" "${DIM}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" --decoder-kind code --decoder-max-vocab "${CODE_DECODER_MAX_VOCAB}" --grad-accum "${CODE_DECODER_GRAD_ACCUM}"
  echo "  Code decoder: local_models/code_decoder_*.safetensors"
fi

# --- Stage 5: Text decoder (chat) ---
echo "== Stage 5/5: Text decoder (--train-decoder --decoder-kind text) =="
if [[ ! -f "${TEXT_DATA}" ]]; then
  echo "  Skipping (TEXT_DATA not found: ${TEXT_DATA}). Set TEXT_DATA= to skip, or provide a path."
else
  TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="decoder_text" cargo run --release -- --train-decoder "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${TEXT_DATA}" "${TEXT_DECODER_STEPS}" "${TEXT_DECODER_BATCH}" "${TEXT_DECODER_MAX_SEQ}" --decoder-kind text --decoder-max-vocab "${TEXT_DECODER_MAX_VOCAB}" --decoder-output local_models/text_decoder_90M.safetensors --grad-accum "${TEXT_DECODER_GRAD_ACCUM}"
  echo "  Text decoder: local_models/text_decoder_90M.safetensors"
fi

echo "Pipeline complete. Serve with: cargo run --release -- --serve ${LATENT_MODEL} ${ENCODER_VOCAB} ${WORLD_MODEL} 0.0.0.0:8080 ${DIM} ${WORLD_MAX_SEQ} ${LAYERS} ${HEADS} ${BRIDGE_DIM} ${NUM_LATENT_TOKENS}"
