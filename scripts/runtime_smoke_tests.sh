#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SMOKE_DIR="${TOFY_RUNTIME_SMOKE_DIR:-$(mktemp -d -t tofy-runtime-smoke-XXXXXX)}"
KEEP_SMOKE_DIR="${TOFY_RUNTIME_SMOKE_KEEP:-0}"
RUN_ID="${TOFY_RUNTIME_SMOKE_RUN_ID:-runtime_smoke_$(date +%Y-%m-%d_%H-%M-%S)}"

cleanup() {
  if [[ "${KEEP_SMOKE_DIR}" == "1" ]]; then
    echo "Keeping smoke artifacts at ${SMOKE_DIR}"
  else
    rm -rf "${SMOKE_DIR}"
  fi
}
trap cleanup EXIT

have_cuda() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
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

latest_artifact() {
  local pattern="$1"
  find local_models -maxdepth 1 -type f -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null \
    | awk '$0 !~ /\.safetensors\..*\.safetensors$/' \
    | sort -nr \
    | head -n1 \
    | cut -d' ' -f2-
}

run_stage() {
  local stage_name="$1"
  shift
  echo "== Smoke stage: ${stage_name} =="
  TOFY_RUN_GROUP="${RUN_ID}" TOFY_RUN_STAGE_NAME="${stage_name}" "$@"
}

LATENT_DATA="${TOFY_RUNTIME_SMOKE_LATENT_DATA:-data/rust_docs_jepa.txt}"
WORLD_DATA="${TOFY_RUNTIME_SMOKE_WORLD_DATA:-data/world_mix_pairs.txt}"
EVAL_SUITE="${TOFY_RUNTIME_SMOKE_EVAL_SUITE:-eval/code_assistant_rust_hard.jsonl}"

LATENT_ROWS="${TOFY_RUNTIME_SMOKE_LATENT_ROWS:-0}"
WORLD_ROWS="${TOFY_RUNTIME_SMOKE_WORLD_ROWS:-32}"
EVAL_ROWS="${TOFY_RUNTIME_SMOKE_EVAL_ROWS:-1}"

SMOKE_DIM="${TOFY_RUNTIME_SMOKE_DIM:-128}"
SMOKE_MAX_SEQ="${TOFY_RUNTIME_SMOKE_MAX_SEQ:-64}"
SMOKE_LAYERS="${TOFY_RUNTIME_SMOKE_LAYERS:-2}"
SMOKE_HEADS="${TOFY_RUNTIME_SMOKE_HEADS:-4}"
SMOKE_VOCAB="${TOFY_RUNTIME_SMOKE_VOCAB:-512}"
SMOKE_BRIDGE_DIM="${TOFY_RUNTIME_SMOKE_BRIDGE_DIM:-32}"
SMOKE_PLANNER_SLOTS="${TOFY_RUNTIME_SMOKE_PLANNER_SLOTS:-8}"
SMOKE_MAX_NEW_TOKENS="${TOFY_RUNTIME_SMOKE_MAX_NEW_TOKENS:-96}"

if [[ ! -f "${LATENT_DATA}" ]]; then
  echo "ERROR: latent smoke data not found at ${LATENT_DATA}"
  exit 1
fi
if [[ ! -f "${WORLD_DATA}" ]]; then
  echo "ERROR: world smoke data not found at ${WORLD_DATA}"
  exit 1
fi
if [[ ! -f "${EVAL_SUITE}" ]]; then
  echo "ERROR: eval smoke suite not found at ${EVAL_SUITE}"
  exit 1
fi

if have_cuda; then
  TOFY_TRAIN_DTYPE="${TOFY_TRAIN_DTYPE:-bf16}"
  maybe_export_cuda_compat
else
  TOFY_TRAIN_DTYPE="${TOFY_TRAIN_DTYPE:-f32}"
fi
export TOFY_TRAIN_DTYPE

LATENT_SMOKE_DATA="${SMOKE_DIR}/latent_smoke.txt"
WORLD_SMOKE_DATA="${SMOKE_DIR}/world_smoke.txt"
EVAL_SMOKE_SUITE="${SMOKE_DIR}/eval_smoke.jsonl"

if [[ "${LATENT_ROWS}" == "0" ]]; then
  cp "${LATENT_DATA}" "${LATENT_SMOKE_DATA}"
else
  head -n "${LATENT_ROWS}" "${LATENT_DATA}" > "${LATENT_SMOKE_DATA}"
fi
head -n "${WORLD_ROWS}" "${WORLD_DATA}" > "${WORLD_SMOKE_DATA}"
head -n "${EVAL_ROWS}" "${EVAL_SUITE}" > "${EVAL_SMOKE_SUITE}"

echo "Runtime smoke dir: ${SMOKE_DIR}"
echo "Run id: ${RUN_ID}"
echo "Train dtype: ${TOFY_TRAIN_DTYPE}"
if [[ -n "${CUDA_COMPUTE_CAP:-}" ]]; then
  echo "CUDA build env: CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-unset}"
fi

run_stage latent \
  cargo run --release -- --latent \
  "${LATENT_SMOKE_DATA}" 1 1 "${SMOKE_DIM}" "${SMOKE_MAX_SEQ}" "${SMOKE_LAYERS}" "${SMOKE_HEADS}" "${SMOKE_VOCAB}" \
  --grad-accum 1

LATENT_MODEL="$(latest_artifact 'model_latent_*.safetensors')"
ENCODER_VOCAB="local_models/vocabs/vocab_encoder.txt"
if [[ -z "${LATENT_MODEL}" || ! -f "${LATENT_MODEL}" ]]; then
  echo "ERROR: failed to locate latent smoke checkpoint"
  exit 1
fi

run_stage world \
  cargo run --release -- --train-world \
  "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_SMOKE_DATA}" \
  1 1 "${SMOKE_DIM}" "${SMOKE_MAX_SEQ}" "${SMOKE_LAYERS}" "${SMOKE_HEADS}" "${SMOKE_BRIDGE_DIM}" "${SMOKE_PLANNER_SLOTS}" \
  --grad-accum 1

WORLD_MODEL="$(latest_artifact 'model_world_*.safetensors')"
if [[ -z "${WORLD_MODEL}" || ! -f "${WORLD_MODEL}" ]]; then
  echo "ERROR: failed to locate world smoke checkpoint"
  exit 1
fi

ORCH_MODEL="${SMOKE_DIR}/world_orchestrator_smoke.safetensors"
run_stage orchestrator \
  cargo run --release -- --train-orchestrator \
  "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${WORLD_SMOKE_DATA}" \
  1 1 "${SMOKE_DIM}" "${SMOKE_MAX_SEQ}" "${SMOKE_LAYERS}" "${SMOKE_HEADS}" "${SMOKE_BRIDGE_DIM}" "${SMOKE_PLANNER_SLOTS}" \
  --grad-accum 1 --output "${ORCH_MODEL}"

run_stage eval_world \
  cargo run --release -- --eval-world \
  "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${ORCH_MODEL}" "${WORLD_SMOKE_DATA}" \
  1 1 "${SMOKE_DIM}" "${SMOKE_MAX_SEQ}" "${SMOKE_LAYERS}" "${SMOKE_HEADS}" "${SMOKE_BRIDGE_DIM}" "${SMOKE_PLANNER_SLOTS}"

DECODER_MODEL="${SMOKE_DIR}/code_decoder_smoke.safetensors"
run_stage decoder \
  cargo run --release -- --train-decoder \
  "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${ORCH_MODEL}" "${WORLD_SMOKE_DATA}" \
  1 1 "${SMOKE_MAX_SEQ}" "${SMOKE_DIM}" "${SMOKE_LAYERS}" "${SMOKE_HEADS}" "${SMOKE_BRIDGE_DIM}" "${SMOKE_PLANNER_SLOTS}" \
  --decoder-kind code --decoder-max-vocab "${SMOKE_VOCAB}" --decoder-output "${DECODER_MODEL}" --grad-accum 1

DECODER_VOCAB="${DECODER_MODEL%.safetensors}.vocab.txt"
if [[ "${TOFY_RUNTIME_SMOKE_SKIP_CODE_EVAL:-0}" != "1" ]]; then
  run_stage eval_code \
    cargo run --release -- --eval-code-assistant \
    "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${ORCH_MODEL}" "${EVAL_SMOKE_SUITE}" \
    "${SMOKE_MAX_NEW_TOKENS}" "${SMOKE_DIM}" "${SMOKE_MAX_SEQ}" "${SMOKE_LAYERS}" "${SMOKE_HEADS}" "${SMOKE_BRIDGE_DIM}" "${SMOKE_PLANNER_SLOTS}" \
    --code-decoder "${DECODER_MODEL}" --code-decoder-vocab "${DECODER_VOCAB}"
fi

echo "Runtime smoke tests completed successfully."
