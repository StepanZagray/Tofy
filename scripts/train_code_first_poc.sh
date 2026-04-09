#!/usr/bin/env bash
set -euo pipefail

# Code-first proof-of-concept pipeline:
# encoder -> world transition -> planner/orchestrator tune -> code decoder -> code eval suite

WORLD_TEXT_DATA="${WORLD_TEXT_DATA:-data/ultrachat_pairs.txt}"
WORLD_DATA="${WORLD_DATA:-data/world_mix_pairs.txt}"
CODE_DATA="${CODE_DATA:-data/rust_code_pairs.txt}"
WIKI_DATA="${WIKI_DATA:-data/cached_wikimedia_wikipedia_1.txt}"
ENCODER_DATA="${ENCODER_DATA:-data/encoder_mix.txt}"
EVAL_SUITE="${EVAL_SUITE:-eval/code_assistant_rust_hard.jsonl}"
RUST_TASK_DATA="${RUST_TASK_DATA:-data/rust_instruction_pairs.txt}"
RUST_DOCS_ROOT="${RUST_DOCS_ROOT:-data/sunface_rust-by-practice_en}"
RUST_DOCS_JEPA_DATA="${RUST_DOCS_JEPA_DATA:-data/rust_docs_jepa.txt}"
RUST_DOCS_PAIR_DATA="${RUST_DOCS_PAIR_DATA:-data/rust_docs_pairs.txt}"
CODE_TRAIN_DATA="${CODE_TRAIN_DATA:-data/code_poc_mix.txt}"
CODE_TASK_REPEAT="${CODE_TASK_REPEAT:-4}"
CODE_EXTRA_REPEAT="${CODE_EXTRA_REPEAT:-1}"
CODE_TRAIN_MAX_ROWS="${CODE_TRAIN_MAX_ROWS:-0}"
TOFY_GPU_PROFILE="${TOFY_GPU_PROFILE:-auto}"
TOFY_TRAIN_DTYPE="${TOFY_TRAIN_DTYPE:-bf16}"
TOFY_LATENT_CONTEXT_SEGMENTS="${TOFY_LATENT_CONTEXT_SEGMENTS:-4}"
TOFY_LATENT_RECENT_FULL_SEGMENTS="${TOFY_LATENT_RECENT_FULL_SEGMENTS:-1}"
TOFY_LATENT_HISTORY_RATIO="${TOFY_LATENT_HISTORY_RATIO:-0.35}"
TOFY_WORLD_CONTEXT_SEGMENTS="${TOFY_WORLD_CONTEXT_SEGMENTS:-2}"
TOFY_WORLD_RECENT_FULL_SEGMENTS="${TOFY_WORLD_RECENT_FULL_SEGMENTS:-1}"
TOFY_RECURSIVE_PLANNER_MEMORY="${TOFY_RECURSIVE_PLANNER_MEMORY:-1}"
TOFY_WORLD_TRAIN_ROLLOUT_STEPS="${TOFY_WORLD_TRAIN_ROLLOUT_STEPS:-2}"
TOFY_WORLD_ROLLOUT_STEPS="${TOFY_WORLD_ROLLOUT_STEPS:-2}"
TOFY_DECODER_SYNTAX_LOSS_WEIGHT="${TOFY_DECODER_SYNTAX_LOSS_WEIGHT:-0.35}"
TOFY_DECODER_SIGNATURE_LOSS_WEIGHT="${TOFY_DECODER_SIGNATURE_LOSS_WEIGHT:-0.45}"
export TOFY_TRAIN_DTYPE TOFY_LATENT_CONTEXT_SEGMENTS TOFY_LATENT_RECENT_FULL_SEGMENTS TOFY_LATENT_HISTORY_RATIO TOFY_WORLD_CONTEXT_SEGMENTS TOFY_WORLD_RECENT_FULL_SEGMENTS TOFY_RECURSIVE_PLANNER_MEMORY TOFY_WORLD_TRAIN_ROLLOUT_STEPS TOFY_WORLD_ROLLOUT_STEPS TOFY_DECODER_SYNTAX_LOSS_WEIGHT TOFY_DECODER_SIGNATURE_LOSS_WEIGHT

LATENT_STEPS="${LATENT_STEPS:-25000}"
WORLD_STEPS="${WORLD_STEPS:-60000}"
ROUTER_STEPS="${ROUTER_STEPS:-15000}"
CODE_DECODER_STEPS="${CODE_DECODER_STEPS:-40000}"
CODE_POLISH_STEPS="${CODE_POLISH_STEPS:-8000}"

DIM="${DIM:-768}"
LATENT_MAX_SEQ="${LATENT_MAX_SEQ:-256}"
WORLD_MAX_SEQ="${WORLD_MAX_SEQ:-256}"
CODE_DECODER_MAX_SEQ="${CODE_DECODER_MAX_SEQ:-192}"
LAYERS="${LAYERS:-9}"
HEADS="${HEADS:-8}"
MAX_VOCAB="${MAX_VOCAB:-8000}"
CODE_DECODER_MAX_VOCAB="${CODE_DECODER_MAX_VOCAB:-16000}"
BRIDGE_DIM="${BRIDGE_DIM:-256}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
WORLD_LR="${WORLD_LR:-2e-4}"
CODE_DECODER_LR="${CODE_DECODER_LR:-3e-4}"
CODE_POLISH_LR="${CODE_POLISH_LR:-1e-4}"
WORLD_LAMBDA="${WORLD_LAMBDA:-0.2}"
WORLD_ACTION_LOSS_WEIGHT="${WORLD_ACTION_LOSS_WEIGHT:-1.0}"
WORLD_ROUTER_WARMUP="${WORLD_ROUTER_WARMUP:-5000}"
WORLD_CODE_RATIO="${WORLD_CODE_RATIO:-0.45}"
WORLD_DONE_RATIO="${WORLD_DONE_RATIO:-0.18}"
WORLD_MAX_ROWS="${WORLD_MAX_ROWS:-0}"
ENCODER_VOCAB="${ENCODER_VOCAB:-local_models/vocabs/vocab_encoder.txt}"
CODE_DECODER_OUTPUT="${CODE_DECODER_OUTPUT:-local_models/code_decoder_poc.safetensors}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-code_poc_$(date +%Y-%m-%d_%H-%M-%S)}"

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
DECODER_GRAD_ACCUM="${DECODER_GRAD_ACCUM:-${DEFAULT_DECODER_GRAD_ACCUM}}"
LATENT_GRAD_ACCUM="${LATENT_GRAD_ACCUM:-${DEFAULT_LATENT_GRAD_ACCUM}}"
WORLD_GRAD_ACCUM="${WORLD_GRAD_ACCUM:-${DEFAULT_WORLD_GRAD_ACCUM}}"
ROUTER_BATCH="${ROUTER_BATCH:-${WORLD_BATCH}}"
ROUTER_GRAD_ACCUM="${ROUTER_GRAD_ACCUM:-${WORLD_GRAD_ACCUM}}"
CODE_DECODER_GRAD_ACCUM="${CODE_DECODER_GRAD_ACCUM:-${DECODER_GRAD_ACCUM}}"

echo "GPU profile: ${TOFY_GPU_PROFILE} (vram_mb=${TOTAL_VRAM_MB:-unknown})"
echo "Microbatches: latent=${LATENT_BATCH} world=${WORLD_BATCH} code_decoder=${CODE_DECODER_BATCH}"
echo "Grad accum: latent=${LATENT_GRAD_ACCUM} world=${WORLD_GRAD_ACCUM} code_decoder=${CODE_DECODER_GRAD_ACCUM}"
echo "Training dtype: ${TOFY_TRAIN_DTYPE} | latent_segments=${TOFY_LATENT_CONTEXT_SEGMENTS} recent_full=${TOFY_LATENT_RECENT_FULL_SEGMENTS} history_ratio=${TOFY_LATENT_HISTORY_RATIO}"
echo "World memory: segments=${TOFY_WORLD_CONTEXT_SEGMENTS} recent_full=${TOFY_WORLD_RECENT_FULL_SEGMENTS} recursive=${TOFY_RECURSIVE_PLANNER_MEMORY} rollout_train=${TOFY_WORLD_TRAIN_ROLLOUT_STEPS} rollout_serve=${TOFY_WORLD_ROLLOUT_STEPS}"

maybe_export_cuda_compat
if [[ -n "${CUDARC_CUDA_VERSION:-}" || -n "${CUDA_COMPUTE_CAP:-}" ]]; then
  echo "CUDA build env: CUDARC_CUDA_VERSION=${CUDARC_CUDA_VERSION:-unset} CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-unset}"
fi

if [[ ! -f "${CODE_DATA}" ]]; then
  echo "CODE_DATA not found at '${CODE_DATA}', generating Rust-only code pairs..."
  "${PYTHON_BIN}" -c "import datasets" >/dev/null 2>&1 || {
    echo "ERROR: Python package 'datasets' is required to generate ${CODE_DATA}."
    echo "  Create a repo venv and install it with:"
    echo "    python -m venv .venv && .venv/bin/pip install datasets"
    exit 1
  }
  "${PYTHON_BIN}" scripts/prepare_github_top_code.py --output "${CODE_DATA}" --languages Rust --max-files 120000
fi

if [[ ! -f "${WORLD_TEXT_DATA}" ]]; then
  echo "ERROR: WORLD_TEXT_DATA not found at '${WORLD_TEXT_DATA}'"
  exit 1
fi

EXTRA_CODE_MIX_ARGS=()
EXTRA_ENCODER_INPUTS=()
if [[ -d "${RUST_DOCS_ROOT}" ]]; then
  echo "Preparing Rust docs JEPA corpus at ${RUST_DOCS_JEPA_DATA}"
  "${PYTHON_BIN}" scripts/prepare_rust_by_practice_md.py --input "${RUST_DOCS_ROOT}" --mode jepa --output "${RUST_DOCS_JEPA_DATA}"
  if [[ -s "${RUST_DOCS_JEPA_DATA}" ]]; then
    EXTRA_ENCODER_INPUTS+=("${RUST_DOCS_JEPA_DATA}")
  fi
  echo "Preparing Rust docs pairs at ${RUST_DOCS_PAIR_DATA}"
  "${PYTHON_BIN}" scripts/prepare_rust_by_practice_md.py --input "${RUST_DOCS_ROOT}" --mode pairs --output "${RUST_DOCS_PAIR_DATA}"
  if [[ -s "${RUST_DOCS_PAIR_DATA}" ]]; then
    EXTRA_CODE_MIX_ARGS+=(--extra-pairs "${RUST_DOCS_PAIR_DATA}" --extra-repeat "${CODE_EXTRA_REPEAT}")
  fi
fi

if [[ ! -f "${WIKI_DATA}" ]]; then
  echo "ERROR: WIKI_DATA not found at '${WIKI_DATA}'"
  exit 1
fi

echo "Preparing encoder corpus at ${ENCODER_DATA}"
"${PYTHON_BIN}" scripts/prepare_encoder_corpus.py --output "${ENCODER_DATA}" "${WORLD_TEXT_DATA}" "${WIKI_DATA}" "${CODE_DATA}" "${EXTRA_ENCODER_INPUTS[@]}"

echo "Preparing world-model mix at ${WORLD_DATA}"
"${PYTHON_BIN}" scripts/prepare_world_mix.py \
  --output "${WORLD_DATA}" \
  --text-pairs "${WORLD_TEXT_DATA}" \
  --code-pairs "${CODE_DATA}" \
  --code-ratio "${WORLD_CODE_RATIO}" \
  --done-ratio "${WORLD_DONE_RATIO}" \
  --max-rows "${WORLD_MAX_ROWS}"

echo "Preparing Rust instruction pairs at ${RUST_TASK_DATA}"
"${PYTHON_BIN}" scripts/prepare_rust_function_tasks.py --input "${CODE_DATA}" --output "${RUST_TASK_DATA}"
if [[ ! -s "${RUST_TASK_DATA}" ]]; then
  echo "No Rust instruction pairs extracted from ${CODE_DATA}; falling back to github-top-code Rust files..."
  "${PYTHON_BIN}" -c "import datasets" >/dev/null 2>&1 || {
    echo "ERROR: Python package 'datasets' is required for Rust instruction-task fallback generation."
    exit 1
  }
  "${PYTHON_BIN}" scripts/prepare_rust_function_tasks.py --github-top-code --max-files 120000 --output "${RUST_TASK_DATA}"
fi

echo "Preparing code-first decoder mix at ${CODE_TRAIN_DATA}"
"${PYTHON_BIN}" scripts/prepare_code_poc_mix.py \
  --output "${CODE_TRAIN_DATA}" \
  --base-pairs "${CODE_DATA}" \
  --instruction-pairs "${RUST_TASK_DATA}" \
  --instruction-repeat "${CODE_TASK_REPEAT}" \
  "${EXTRA_CODE_MIX_ARGS[@]}" \
  --max-rows "${CODE_TRAIN_MAX_ROWS}"

echo "Generating code eval suite at ${EVAL_SUITE}"
"${PYTHON_BIN}" scripts/generate_code_eval_suite.py --output "${EVAL_SUITE}"

echo "== Stage 1/5: encoder =="
TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="latent" cargo run --release -- \
  --latent "${ENCODER_DATA}" "${LATENT_STEPS}" "${LATENT_BATCH}" "${DIM}" "${LATENT_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}" \
  --grad-accum "${LATENT_GRAD_ACCUM}"

LATENT_MODEL="${LATENT_MODEL:-$(ls -1t local_models/model_latent_*.safetensors 2>/dev/null | awk '{print; exit}')}"
if [[ -z "${LATENT_MODEL}" || ! -f "${LATENT_MODEL}" ]]; then
  echo "ERROR: latent checkpoint not found"
  exit 1
fi

echo "== Stage 2/5: world transition =="
TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="world" cargo run --release -- \
  --train-world "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_DATA}" "${WORLD_STEPS}" "${WORLD_BATCH}" "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" \
  --lambda "${WORLD_LAMBDA}" --lr "${WORLD_LR}" --grad-accum "${WORLD_GRAD_ACCUM}" \
  --action-loss-weight "${WORLD_ACTION_LOSS_WEIGHT}" --router-warmup "${WORLD_ROUTER_WARMUP}"

WORLD_MODEL="${WORLD_MODEL:-$(ls -1t local_models/model_world_*.safetensors 2>/dev/null | awk '{print; exit}')}"
if [[ -z "${WORLD_MODEL}" || ! -f "${WORLD_MODEL}" ]]; then
  echo "ERROR: world checkpoint not found"
  exit 1
fi

echo "== Stage 3/5: orchestrator/planner tune =="
TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="orchestrator" cargo run --release -- \
  --train-orchestrator "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${WORLD_DATA}" "${ROUTER_STEPS}" "${ROUTER_BATCH}" "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" \
  --lr "${WORLD_LR}" --grad-accum "${ROUTER_GRAD_ACCUM}" --output "${WORLD_MODEL}"

echo "== Stage 4/5: code decoder =="
TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="decoder_code" cargo run --release -- \
  --train-decoder "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${CODE_TRAIN_DATA}" "${CODE_DECODER_STEPS}" "${CODE_DECODER_BATCH}" "${CODE_DECODER_MAX_SEQ}" "${DIM}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" \
  --decoder-kind code --decoder-max-vocab "${CODE_DECODER_MAX_VOCAB}" --decoder-output "${CODE_DECODER_OUTPUT}" --grad-accum "${CODE_DECODER_GRAD_ACCUM}" --lr "${CODE_DECODER_LR}"

if [[ "${CODE_POLISH_STEPS}" -gt 0 ]]; then
  echo "== Stage 4b/5: code decoder instruction polish =="
  TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="decoder_code_polish" cargo run --release -- \
    --train-decoder "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${RUST_TASK_DATA}" "${CODE_POLISH_STEPS}" "${CODE_DECODER_BATCH}" "${CODE_DECODER_MAX_SEQ}" "${DIM}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" \
    --decoder-kind code --decoder-max-vocab "${CODE_DECODER_MAX_VOCAB}" --decoder-output "${CODE_DECODER_OUTPUT}" --init-decoder "${CODE_DECODER_OUTPUT}" --grad-accum "${CODE_DECODER_GRAD_ACCUM}" --lr "${CODE_POLISH_LR}"
fi

echo "== Stage 5/5: code eval suite =="
cargo run --release -- \
  --eval-code-assistant "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${EVAL_SUITE}" 384 "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" \
  --code-decoder "${CODE_DECODER_OUTPUT}"

echo "Code-first POC pipeline complete."
