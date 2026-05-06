#!/usr/bin/env bash
set -euo pipefail

# Code-first proof-of-concept pipeline:
# vocab/token cache -> encoder -> world transition -> planner/orchestrator tune -> code decoder -> code eval suite

WORLD_TEXT_DATA="${WORLD_TEXT_DATA:-data/ultrachat_pairs.txt}"
WORLD_DATA="${WORLD_DATA:-data/world_mix_pairs.txt}"
CODE_DATA="${CODE_DATA:-data/rust_code_pairs.txt}"
WIKI_DATA="${WIKI_DATA:-data/cached_wikimedia_wikipedia_1.txt}"
ENCODER_DATA="${ENCODER_DATA:-data/encoder_mix.txt}"
EVAL_SUITE="${EVAL_SUITE:-eval/code_assistant_rust_hard.jsonl}"
RUST_TASK_DATA="${RUST_TASK_DATA:-data/rust_instruction_pairs.txt}"
RUST_REPAIR_DATA="${RUST_REPAIR_DATA:-data/rust_repair_pairs.txt}"
RUST_DOCS_ROOT="${RUST_DOCS_ROOT:-data/sunface_rust-by-practice_en}"
RUST_DOCS_JEPA_DATA="${RUST_DOCS_JEPA_DATA:-data/rust_docs_jepa.txt}"
RUST_DOCS_PAIR_DATA="${RUST_DOCS_PAIR_DATA:-data/rust_docs_pairs.txt}"
CODE_TRAIN_DATA="${CODE_TRAIN_DATA:-data/code_poc_mix.txt}"
CODE_TASK_REPEAT="${CODE_TASK_REPEAT:-6}"
CODE_REPAIR_REPEAT="${CODE_REPAIR_REPEAT:-2}"
CODE_EXTRA_REPEAT="${CODE_EXTRA_REPEAT:-1}"
CODE_TRAIN_MAX_ROWS="${CODE_TRAIN_MAX_ROWS:-0}"
TOFY_PREPARE_REPAIR_TASKS="${TOFY_PREPARE_REPAIR_TASKS:-auto}"
RUST_REPAIR_VARIANTS_PER_SAMPLE="${RUST_REPAIR_VARIANTS_PER_SAMPLE:-2}"
RUST_REPAIR_MAX_ROWS="${RUST_REPAIR_MAX_ROWS:-2000}"
RUST_REPAIR_TIMEOUT_SEC="${RUST_REPAIR_TIMEOUT_SEC:-4.0}"
RUSTC_BIN="${RUSTC_BIN:-rustc}"
TOFY_GPU_PROFILE="${TOFY_GPU_PROFILE:-8gb}"
TOFY_TRAIN_DTYPE="${TOFY_TRAIN_DTYPE:-bf16}"
TOFY_SIGREG_SLICES="${TOFY_SIGREG_SLICES:-1024}"
TOFY_SIGREG_POINTS="${TOFY_SIGREG_POINTS:-17}"
TOFY_LATENT_CONTEXT_SEGMENTS="${TOFY_LATENT_CONTEXT_SEGMENTS:-4}"
TOFY_LATENT_RECENT_FULL_SEGMENTS="${TOFY_LATENT_RECENT_FULL_SEGMENTS:-1}"
TOFY_LATENT_HISTORY_RATIO="${TOFY_LATENT_HISTORY_RATIO:-0.35}"
TOFY_WORLD_CONTEXT_SEGMENTS="${TOFY_WORLD_CONTEXT_SEGMENTS:-2}"
TOFY_WORLD_RECENT_FULL_SEGMENTS="${TOFY_WORLD_RECENT_FULL_SEGMENTS:-1}"
TOFY_RECURSIVE_PLANNER_MEMORY="${TOFY_RECURSIVE_PLANNER_MEMORY:-1}"
TOFY_WORLD_TRAIN_ROLLOUT_STEPS="${TOFY_WORLD_TRAIN_ROLLOUT_STEPS:-2}"
TOFY_WORLD_ROLLOUT_STEPS="${TOFY_WORLD_ROLLOUT_STEPS:-2}"
TOFY_WORLD_INVERSE_LOSS_WEIGHT="${TOFY_WORLD_INVERSE_LOSS_WEIGHT:-0.0}"
TOFY_DECODER_SYNTAX_LOSS_WEIGHT="${TOFY_DECODER_SYNTAX_LOSS_WEIGHT:-0.0}"
TOFY_DECODER_SIGNATURE_LOSS_WEIGHT="${TOFY_DECODER_SIGNATURE_LOSS_WEIGHT:-0.0}"
TOFY_DECODER_STRUCTURE_LOSS_WEIGHT="${TOFY_DECODER_STRUCTURE_LOSS_WEIGHT:-0.0}"
TOFY_DECODER_CONDITIONING_LOSS_WEIGHT="${TOFY_DECODER_CONDITIONING_LOSS_WEIGHT:-auto}"
TOFY_DECODER_CONDITIONING_MARGIN="${TOFY_DECODER_CONDITIONING_MARGIN:-0.10}"
TOFY_CODE_VOCAB_SAMPLE_ROWS="${TOFY_CODE_VOCAB_SAMPLE_ROWS:-25000}"
TOFY_CODE_VOCAB_SAMPLE_BYTES="${TOFY_CODE_VOCAB_SAMPLE_BYTES:-16777216}"
export TOFY_TRAIN_DTYPE TOFY_SIGREG_SLICES TOFY_SIGREG_POINTS
export TOFY_LATENT_CONTEXT_SEGMENTS TOFY_LATENT_RECENT_FULL_SEGMENTS TOFY_LATENT_HISTORY_RATIO TOFY_WORLD_CONTEXT_SEGMENTS TOFY_WORLD_RECENT_FULL_SEGMENTS TOFY_RECURSIVE_PLANNER_MEMORY TOFY_WORLD_TRAIN_ROLLOUT_STEPS TOFY_WORLD_ROLLOUT_STEPS TOFY_WORLD_INVERSE_LOSS_WEIGHT
export TOFY_DECODER_SYNTAX_LOSS_WEIGHT TOFY_DECODER_SIGNATURE_LOSS_WEIGHT TOFY_DECODER_STRUCTURE_LOSS_WEIGHT TOFY_DECODER_CONDITIONING_LOSS_WEIGHT TOFY_DECODER_CONDITIONING_MARGIN
export TOFY_CODE_VOCAB_SAMPLE_ROWS TOFY_CODE_VOCAB_SAMPLE_BYTES

case "${TOFY_GPU_PROFILE}" in
  8gb)
    PROFILE_LATENT_STEPS=25000
    PROFILE_WORLD_STEPS=60000
    PROFILE_HIGH_WORLD_STEPS=0
    PROFILE_ROUTER_STEPS=0
    PROFILE_CODE_DECODER_STEPS=40000
    PROFILE_CODE_POLISH_STEPS=8000
    PROFILE_DIM=640
    PROFILE_LATENT_MAX_SEQ=256
    PROFILE_WORLD_MAX_SEQ=256
    PROFILE_CODE_DECODER_MAX_SEQ=128
    PROFILE_LAYERS=7
    PROFILE_HEADS=8
    PROFILE_MAX_VOCAB=8000
    PROFILE_CODE_DECODER_MAX_VOCAB=16000
    PROFILE_BRIDGE_DIM=640
    PROFILE_NUM_LATENT_TOKENS=64
    PROFILE_BATCH=8
    PROFILE_DECODER_BATCH=6
    PROFILE_LATENT_BATCH=12
    PROFILE_LATENT_WARMUP_BATCH=12
    PROFILE_WORLD_BATCH=64
    PROFILE_WORLD_WARMUP_BATCH=64
    PROFILE_LATENT_GRAD_ACCUM=2
    PROFILE_WORLD_GRAD_ACCUM=2
    PROFILE_DECODER_GRAD_ACCUM=4
    ;;
  48gb)
    # 1536 / 640 squared is about 5.8x parameters, a smaller A40-friendly test scale.
    PROFILE_LATENT_STEPS=75000
    PROFILE_WORLD_STEPS=180000
    PROFILE_HIGH_WORLD_STEPS=0
    PROFILE_ROUTER_STEPS=0
    PROFILE_CODE_DECODER_STEPS=120000
    PROFILE_CODE_POLISH_STEPS=24000
    PROFILE_DIM=1536
    PROFILE_LATENT_MAX_SEQ=256
    PROFILE_WORLD_MAX_SEQ=256
    PROFILE_CODE_DECODER_MAX_SEQ=128
    PROFILE_LAYERS=7
    PROFILE_HEADS=12
    PROFILE_MAX_VOCAB=12000
    PROFILE_CODE_DECODER_MAX_VOCAB=24000
    PROFILE_BRIDGE_DIM=1536
    PROFILE_NUM_LATENT_TOKENS=96
    PROFILE_BATCH=4
    PROFILE_DECODER_BATCH=2
    PROFILE_LATENT_BATCH=6
    PROFILE_LATENT_WARMUP_BATCH=3
    PROFILE_WORLD_BATCH=24
    PROFILE_WORLD_WARMUP_BATCH=12
    PROFILE_LATENT_GRAD_ACCUM=4
    PROFILE_WORLD_GRAD_ACCUM=3
    PROFILE_DECODER_GRAD_ACCUM=6
    ;;
  80gb)
    # 640 * sqrt(10) ~= 2024, so 2048 gives an even-head 10x-ish parameter scale.
    PROFILE_LATENT_STEPS=250000
    PROFILE_WORLD_STEPS=600000
    PROFILE_HIGH_WORLD_STEPS=0
    PROFILE_ROUTER_STEPS=0
    PROFILE_CODE_DECODER_STEPS=400000
    PROFILE_CODE_POLISH_STEPS=80000
    PROFILE_DIM=2048
    PROFILE_LATENT_MAX_SEQ=256
    PROFILE_WORLD_MAX_SEQ=256
    PROFILE_CODE_DECODER_MAX_SEQ=128
    PROFILE_LAYERS=7
    PROFILE_HEADS=16
    PROFILE_MAX_VOCAB=16000
    PROFILE_CODE_DECODER_MAX_VOCAB=32000
    PROFILE_BRIDGE_DIM=2048
    PROFILE_NUM_LATENT_TOKENS=128
    PROFILE_BATCH=4
    PROFILE_DECODER_BATCH=2
    PROFILE_LATENT_BATCH=4
    PROFILE_LATENT_WARMUP_BATCH=2
    PROFILE_WORLD_BATCH=16
    PROFILE_WORLD_WARMUP_BATCH=8
    PROFILE_LATENT_GRAD_ACCUM=8
    PROFILE_WORLD_GRAD_ACCUM=4
    PROFILE_DECODER_GRAD_ACCUM=8
    ;;
  *)
    echo "ERROR: unsupported TOFY_GPU_PROFILE='${TOFY_GPU_PROFILE}' (expected 8gb, 48gb, or 80gb)"
    exit 1
    ;;
esac

LATENT_STEPS="${LATENT_STEPS:-${PROFILE_LATENT_STEPS}}"
WORLD_STEPS="${WORLD_STEPS:-${PROFILE_WORLD_STEPS}}"
HIGH_WORLD_STEPS="${HIGH_WORLD_STEPS:-${PROFILE_HIGH_WORLD_STEPS}}"
ROUTER_STEPS="${ROUTER_STEPS:-${PROFILE_ROUTER_STEPS}}"
CODE_DECODER_STEPS="${CODE_DECODER_STEPS:-${PROFILE_CODE_DECODER_STEPS}}"
CODE_POLISH_STEPS="${CODE_POLISH_STEPS:-${PROFILE_CODE_POLISH_STEPS}}"

DIM="${DIM:-${PROFILE_DIM}}"
LATENT_MAX_SEQ="${LATENT_MAX_SEQ:-${PROFILE_LATENT_MAX_SEQ}}"
WORLD_MAX_SEQ="${WORLD_MAX_SEQ:-${PROFILE_WORLD_MAX_SEQ}}"
CODE_DECODER_MAX_SEQ="${CODE_DECODER_MAX_SEQ:-auto}"
LAYERS="${LAYERS:-${PROFILE_LAYERS}}"
HEADS="${HEADS:-${PROFILE_HEADS}}"
MAX_VOCAB="${MAX_VOCAB:-${PROFILE_MAX_VOCAB}}"
CODE_DECODER_MAX_VOCAB="${CODE_DECODER_MAX_VOCAB:-${PROFILE_CODE_DECODER_MAX_VOCAB}}"
BRIDGE_DIM="${BRIDGE_DIM:-${PROFILE_BRIDGE_DIM}}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-${PROFILE_NUM_LATENT_TOKENS}}"
WORLD_LR="${WORLD_LR:-2e-4}"
CODE_DECODER_LR="${CODE_DECODER_LR:-3e-4}"
CODE_POLISH_LR="${CODE_POLISH_LR:-1e-4}"
WORLD_LAMBDA="${WORLD_LAMBDA:-0.2}"
WORLD_ACTION_LOSS_WEIGHT="${WORLD_ACTION_LOSS_WEIGHT:-0.0}"
HWM_MACRO_MIN_LEN="${HWM_MACRO_MIN_LEN:-2}"
HWM_MACRO_MAX_LEN="${HWM_MACRO_MAX_LEN:-4}"
WORLD_CODE_RATIO="${WORLD_CODE_RATIO:-0.45}"
WORLD_DONE_RATIO="${WORLD_DONE_RATIO:-0.18}"
WORLD_MAX_ROWS="${WORLD_MAX_ROWS:-0}"
ENCODER_VOCAB="${ENCODER_VOCAB:-}"
CODE_DECODER_OUTPUT="${CODE_DECODER_OUTPUT:-}"
CODE_DECODER_BASE_OUTPUT="${CODE_DECODER_BASE_OUTPUT:-}"
CODE_DECODER_POLISH_OUTPUT="${CODE_DECODER_POLISH_OUTPUT:-}"
TOFY_PRETOKENIZE="${TOFY_PRETOKENIZE:-1}"
TOFY_CACHE_DIR="${TOFY_CACHE_DIR:-data/cache}"
ENCODER_CACHE_MAX_SEQ="${ENCODER_CACHE_MAX_SEQ:-$((LATENT_MAX_SEQ * TOFY_LATENT_CONTEXT_SEGMENTS))}"
ENCODER_CACHE_VOCAB="${ENCODER_CACHE_VOCAB:-local_models/vocabs/vocab_encoder_${MAX_VOCAB}_default.txt}"
CODE_DECODER_CACHE_VOCAB="${CODE_DECODER_CACHE_VOCAB:-local_models/vocabs/vocab_code_${CODE_DECODER_MAX_VOCAB}_codeaware.txt}"
CODE_DECODER_VOCAB="${CODE_DECODER_VOCAB:-}"
TOFY_RESUME="${TOFY_RESUME:-0}"
TOFY_RESUME_RUN="${TOFY_RESUME_RUN:-}"
PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-}"
ORIGINAL_CMD=("$0" "$@")

is_true() {
  local value="${1:-0}"
  [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "True" ]]
}

print_usage() {
  echo "Usage:"
  echo "  ./scripts/train_code_first_poc.sh"
  echo "  ./scripts/train_code_first_poc.sh --resume latest"
  echo "  ./scripts/train_code_first_poc.sh --resume <run_id|runs/path>"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      TOFY_RESUME=1
      if [[ $# -ge 2 && "${2}" != --* ]]; then
        TOFY_RESUME_RUN="$2"
        shift 2
      else
        TOFY_RESUME_RUN="latest"
        shift 1
      fi
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      echo "ERROR: unsupported argument '$1'"
      print_usage
      exit 1
      ;;
  esac
done

resolve_latest_run_root() {
  local prefix="$1"
  find runs -maxdepth 1 -mindepth 1 -type d -name "${prefix}*" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n1 \
    | cut -d' ' -f2-
}

resolve_run_root() {
  local selector="$1"
  local prefix="$2"
  if [[ -z "${selector}" || "${selector}" == "latest" ]]; then
    resolve_latest_run_root "${prefix}"
    return 0
  fi
  if [[ -d "${selector}" ]]; then
    printf '%s\n' "${selector}"
    return 0
  fi
  if [[ -d "runs/${selector}" ]]; then
    printf '%s\n' "runs/${selector}"
    return 0
  fi
  return 1
}

stage_resume_state_path() {
  local model_path="$1"
  local stage="$2"
  printf '%s.%s.resume.json\n' "${model_path}" "${stage}"
}

RESUME_ARGS=()
if is_true "${TOFY_RESUME}"; then
  RESUME_ARGS=(--resume)
  if [[ -z "${TOFY_RESUME_RUN}" ]]; then
    TOFY_RESUME_RUN="${PIPELINE_RUN_ID:-latest}"
  fi
  PIPELINE_RUN_ROOT="$(resolve_run_root "${TOFY_RESUME_RUN}" "code_poc_")"
  if [[ -z "${PIPELINE_RUN_ROOT}" || ! -d "${PIPELINE_RUN_ROOT}" ]]; then
    echo "ERROR: could not resolve resume run '${TOFY_RESUME_RUN}'"
    exit 1
  fi
  PIPELINE_RUN_ID="$(basename "${PIPELINE_RUN_ROOT}")"
else
  PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-code_poc_$(date +%Y-%m-%d_%H-%M-%S)}"
  PIPELINE_RUN_ROOT="runs/${PIPELINE_RUN_ID}"
  if [[ -e "${PIPELINE_RUN_ROOT}" ]]; then
    echo "ERROR: run directory already exists at '${PIPELINE_RUN_ROOT}'"
    exit 1
  fi
fi

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
DEFAULT_DECODER_BATCH="${PROFILE_DECODER_BATCH}"
DEFAULT_LATENT_BATCH="${PROFILE_LATENT_BATCH}"
DEFAULT_LATENT_WARMUP_BATCH="${PROFILE_LATENT_WARMUP_BATCH}"
DEFAULT_WORLD_BATCH="${PROFILE_WORLD_BATCH}"
DEFAULT_WORLD_WARMUP_BATCH="${PROFILE_WORLD_WARMUP_BATCH}"
DEFAULT_LATENT_GRAD_ACCUM="${PROFILE_LATENT_GRAD_ACCUM}"
DEFAULT_WORLD_GRAD_ACCUM="${PROFILE_WORLD_GRAD_ACCUM}"
DEFAULT_DECODER_GRAD_ACCUM="${PROFILE_DECODER_GRAD_ACCUM}"
DEFAULT_CODE_DECODER_MAX_SEQ="${PROFILE_CODE_DECODER_MAX_SEQ}"
DEFAULT_DECODER_CONDITIONING_LOSS_WEIGHT=0.0

if [[ "${CODE_DECODER_MAX_SEQ}" == "auto" ]]; then
  CODE_DECODER_MAX_SEQ="${DEFAULT_CODE_DECODER_MAX_SEQ}"
fi
if [[ "${TOFY_DECODER_CONDITIONING_LOSS_WEIGHT}" == "auto" ]]; then
  TOFY_DECODER_CONDITIONING_LOSS_WEIGHT="${DEFAULT_DECODER_CONDITIONING_LOSS_WEIGHT}"
fi

BATCH="${BATCH:-${DEFAULT_BATCH}}"
DECODER_BATCH="${DECODER_BATCH:-${DEFAULT_DECODER_BATCH}}"
LATENT_BATCH="${LATENT_BATCH:-${DEFAULT_LATENT_BATCH:-${BATCH}}}"
TOFY_LATENT_WARMUP_BATCH="${TOFY_LATENT_WARMUP_BATCH:-${DEFAULT_LATENT_WARMUP_BATCH:-${LATENT_BATCH}}}"
TOFY_LATENT_WARMUP_GRAD_ACCUM="${TOFY_LATENT_WARMUP_GRAD_ACCUM:-1}"
WORLD_BATCH="${WORLD_BATCH:-${DEFAULT_WORLD_BATCH:-${BATCH}}}"
TOFY_WORLD_WARMUP_BATCH="${TOFY_WORLD_WARMUP_BATCH:-${DEFAULT_WORLD_WARMUP_BATCH:-${WORLD_BATCH}}}"
TOFY_WORLD_WARMUP_GRAD_ACCUM="${TOFY_WORLD_WARMUP_GRAD_ACCUM:-1}"
TOFY_WORLD_WARMUP_STEPS="${TOFY_WORLD_WARMUP_STEPS:-1200}"
TOFY_WORLD_LOG_EVERY="${TOFY_WORLD_LOG_EVERY:-1000}"
TOFY_ORCHESTRATOR_LOG_EVERY="${TOFY_ORCHESTRATOR_LOG_EVERY:-500}"
TOFY_DECODER_LOG_EVERY="${TOFY_DECODER_LOG_EVERY:-500}"
CODE_DECODER_BATCH="${CODE_DECODER_BATCH:-${DECODER_BATCH}}"
DECODER_GRAD_ACCUM="${DECODER_GRAD_ACCUM:-${DEFAULT_DECODER_GRAD_ACCUM}}"
LATENT_GRAD_ACCUM="${LATENT_GRAD_ACCUM:-${DEFAULT_LATENT_GRAD_ACCUM}}"
WORLD_GRAD_ACCUM="${WORLD_GRAD_ACCUM:-${DEFAULT_WORLD_GRAD_ACCUM}}"
ROUTER_BATCH="${ROUTER_BATCH:-${WORLD_BATCH}}"
ROUTER_GRAD_ACCUM="${ROUTER_GRAD_ACCUM:-${WORLD_GRAD_ACCUM}}"
CODE_DECODER_GRAD_ACCUM="${CODE_DECODER_GRAD_ACCUM:-${DECODER_GRAD_ACCUM}}"
export TOFY_LATENT_WARMUP_BATCH TOFY_LATENT_WARMUP_GRAD_ACCUM
export TOFY_WORLD_WARMUP_BATCH TOFY_WORLD_WARMUP_GRAD_ACCUM TOFY_WORLD_WARMUP_STEPS
export TOFY_WORLD_LOG_EVERY TOFY_ORCHESTRATOR_LOG_EVERY TOFY_DECODER_LOG_EVERY

resume_stage_complete() {
  local model_path="$1"
  local stage="$2"
  local target_steps="$3"
  local state_path
  state_path="$(stage_resume_state_path "${model_path}" "${stage}")"
  [[ -f "${state_path}" ]] || return 1
  local step
  step="$(grep -o '"step"[[:space:]]*:[[:space:]]*[0-9]\+' "${state_path}" | head -n1 | grep -o '[0-9]\+' || true)"
  [[ "${step:-0}" -ge "${target_steps}" ]]
}

LATENT_STAGE_DIR="${PIPELINE_RUN_ROOT}/latent"
WORLD_STAGE_DIR="${PIPELINE_RUN_ROOT}/world"
HIGH_WORLD_STAGE_DIR="${PIPELINE_RUN_ROOT}/high_world"
ORCHESTRATOR_STAGE_DIR="${PIPELINE_RUN_ROOT}/orchestrator"
DECODER_STAGE_DIR="${PIPELINE_RUN_ROOT}/decoder_code"
DECODER_POLISH_STAGE_DIR="${PIPELINE_RUN_ROOT}/decoder_code_polish"
CODE_EVAL_STAGE_DIR="${PIPELINE_RUN_ROOT}/code_eval"
LATENT_MODEL="${LATENT_MODEL:-${LATENT_STAGE_DIR}/model.safetensors}"
WORLD_MODEL="${WORLD_MODEL:-${WORLD_STAGE_DIR}/model.safetensors}"
WORLD_ENCODER_MODEL="${WORLD_ENCODER_MODEL:-${WORLD_STAGE_DIR}/model.encoder.safetensors}"
HIGH_WORLD_MODEL="${HIGH_WORLD_MODEL:-${HIGH_WORLD_STAGE_DIR}/model.safetensors}"
CODE_DECODER_BASE_OUTPUT="${CODE_DECODER_BASE_OUTPUT:-${DECODER_STAGE_DIR}/model.safetensors}"
CODE_DECODER_POLISH_OUTPUT="${CODE_DECODER_POLISH_OUTPUT:-${DECODER_POLISH_STAGE_DIR}/model.safetensors}"
CODE_DECODER_OUTPUT="${CODE_DECODER_OUTPUT:-${CODE_DECODER_BASE_OUTPUT}}"
if [[ -z "${CODE_DECODER_VOCAB}" ]]; then
  CODE_DECODER_VOCAB="${CODE_DECODER_BASE_OUTPUT%.safetensors}.vocab.txt"
fi
mkdir -p \
  "${PIPELINE_RUN_ROOT}" \
  "${LATENT_STAGE_DIR}" \
  "${WORLD_STAGE_DIR}" \
  "${HIGH_WORLD_STAGE_DIR}" \
  "${ORCHESTRATOR_STAGE_DIR}" \
  "${DECODER_STAGE_DIR}" \
  "${DECODER_POLISH_STAGE_DIR}" \
  "${CODE_EVAL_STAGE_DIR}"

if [[ -z "${TOFY_PIPELINE_LOG_INITIALIZED:-}" ]]; then
  export TOFY_PIPELINE_LOG_INITIALIZED=1
  exec > >(tee -a "${PIPELINE_RUN_ROOT}/pipeline.log") 2>&1
fi

{
  printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'command='
  printf '%q ' "${ORIGINAL_CMD[@]}"
  printf '\n'
} > "${PIPELINE_RUN_ROOT}/launch.txt"

write_pipeline_meta() {
  cat > "${PIPELINE_RUN_ROOT}/meta.json" <<EOF
{
  "pipeline_run_id": "${PIPELINE_RUN_ID}",
  "pipeline_kind": "code_first_poc",
  "resume_enabled": $(if is_true "${TOFY_RESUME}"; then printf 'true'; else printf 'false'; fi),
  "resume_selector": "${TOFY_RESUME_RUN}",
  "run_root": "${PIPELINE_RUN_ROOT}",
  "latent_model": "${LATENT_MODEL}",
  "world_model": "${WORLD_MODEL}",
  "world_encoder_model": "${WORLD_ENCODER_MODEL}",
  "high_world_model": "${HIGH_WORLD_MODEL}",
  "code_decoder_model": "${CODE_DECODER_OUTPUT}",
  "code_decoder_base_model": "${CODE_DECODER_BASE_OUTPUT}",
  "code_decoder_polish_model": "${CODE_DECODER_POLISH_OUTPUT}",
  "code_decoder_vocab": "${CODE_DECODER_VOCAB}",
  "encoder_data": "${ENCODER_DATA}",
  "world_data": "${WORLD_DATA}",
  "code_train_data": "${CODE_TRAIN_DATA}",
  "eval_suite": "${EVAL_SUITE}",
  "latent_steps": "${LATENT_STEPS}",
  "world_steps": "${WORLD_STEPS}",
  "high_world_steps": "${HIGH_WORLD_STEPS}",
  "router_steps": "${ROUTER_STEPS}",
  "code_decoder_steps": "${CODE_DECODER_STEPS}",
  "code_polish_steps": "${CODE_POLISH_STEPS}",
  "gpu_profile": "${TOFY_GPU_PROFILE}",
  "total_vram_mb": "${TOTAL_VRAM_MB}",
  "latent_batch": "${LATENT_BATCH}",
  "latent_warmup_batch": "${TOFY_LATENT_WARMUP_BATCH}",
  "world_batch": "${WORLD_BATCH}",
  "world_warmup_batch": "${TOFY_WORLD_WARMUP_BATCH}",
  "code_decoder_batch": "${CODE_DECODER_BATCH}",
  "latent_grad_accum": "${LATENT_GRAD_ACCUM}",
  "latent_warmup_grad_accum": "${TOFY_LATENT_WARMUP_GRAD_ACCUM}",
  "world_grad_accum": "${WORLD_GRAD_ACCUM}",
  "world_warmup_grad_accum": "${TOFY_WORLD_WARMUP_GRAD_ACCUM}",
  "code_decoder_grad_accum": "${CODE_DECODER_GRAD_ACCUM}",
  "dim": "${DIM}",
  "layers": "${LAYERS}",
  "heads": "${HEADS}",
  "bridge_dim": "${BRIDGE_DIM}",
  "num_latent_tokens": "${NUM_LATENT_TOKENS}"
}
EOF
}

echo "GPU profile: ${TOFY_GPU_PROFILE} (vram_mb=${TOTAL_VRAM_MB:-unknown})"
echo "Pipeline run directory: ${PIPELINE_RUN_ROOT}"
if is_true "${TOFY_RESUME}"; then
  echo "Resuming pipeline run: ${PIPELINE_RUN_ID} (selector=${TOFY_RESUME_RUN})"
fi
echo "Microbatches: latent=${LATENT_BATCH} world=${WORLD_BATCH} code_decoder=${CODE_DECODER_BATCH}"
echo "Grad accum: latent=${LATENT_GRAD_ACCUM} world=${WORLD_GRAD_ACCUM} code_decoder=${CODE_DECODER_GRAD_ACCUM}"
echo "Effective batch: latent=$((LATENT_BATCH * LATENT_GRAD_ACCUM)) world=$((WORLD_BATCH * WORLD_GRAD_ACCUM)) code_decoder=$((CODE_DECODER_BATCH * CODE_DECODER_GRAD_ACCUM))"
echo "Latent warmup: batch=${TOFY_LATENT_WARMUP_BATCH} grad_accum=${TOFY_LATENT_WARMUP_GRAD_ACCUM} effective=$((TOFY_LATENT_WARMUP_BATCH * TOFY_LATENT_WARMUP_GRAD_ACCUM)) for ${TOFY_LATENT_WARMUP_STEPS:-20%} of latent steps"
echo "World warmup: batch=${TOFY_WORLD_WARMUP_BATCH} grad_accum=${TOFY_WORLD_WARMUP_GRAD_ACCUM} effective=$((TOFY_WORLD_WARMUP_BATCH * TOFY_WORLD_WARMUP_GRAD_ACCUM)) for ${TOFY_WORLD_WARMUP_STEPS:-20%} of world steps"
echo "Training dtype: ${TOFY_TRAIN_DTYPE} | latent_segments=${TOFY_LATENT_CONTEXT_SEGMENTS} recent_full=${TOFY_LATENT_RECENT_FULL_SEGMENTS} history_ratio=${TOFY_LATENT_HISTORY_RATIO}"
echo "LeJEPA objective: online prediction + SIGReg | slices=${TOFY_SIGREG_SLICES} points=${TOFY_SIGREG_POINTS}"
echo "World memory: segments=${TOFY_WORLD_CONTEXT_SEGMENTS} recent_full=${TOFY_WORLD_RECENT_FULL_SEGMENTS} recursive=${TOFY_RECURSIVE_PLANNER_MEMORY} rollout_train=${TOFY_WORLD_TRAIN_ROLLOUT_STEPS} rollout_serve=${TOFY_WORLD_ROLLOUT_STEPS}"
echo "High-world: steps=${HIGH_WORLD_STEPS} macro_len=${HWM_MACRO_MIN_LEN}..${HWM_MACRO_MAX_LEN}"
write_pipeline_meta

maybe_export_cuda_compat
if [[ -n "${CUDA_COMPUTE_CAP:-}" ]]; then
  echo "CUDA build env: CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-unset}"
fi

if [[ ! -f "${CODE_DATA}" ]]; then
  echo "CODE_DATA not found at '${CODE_DATA}', generating Rust-only code pairs..."
  cargo run --release -- --prepare-github-top-code --output "${CODE_DATA}" --languages Rust --max-files 120000
fi

if [[ ! -f "${WORLD_TEXT_DATA}" ]]; then
  echo "ERROR: WORLD_TEXT_DATA not found at '${WORLD_TEXT_DATA}'"
  exit 1
fi

EXTRA_CODE_MIX_ARGS=()
EXTRA_ENCODER_INPUTS=()
if [[ -d "${RUST_DOCS_ROOT}" ]]; then
  echo "Preparing Rust docs JEPA corpus at ${RUST_DOCS_JEPA_DATA}"
  cargo run --release -- --prepare-rust-by-practice --input "${RUST_DOCS_ROOT}" --mode jepa --output "${RUST_DOCS_JEPA_DATA}"
  if [[ -s "${RUST_DOCS_JEPA_DATA}" ]]; then
    EXTRA_ENCODER_INPUTS+=("${RUST_DOCS_JEPA_DATA}")
  fi
  echo "Preparing Rust docs pairs at ${RUST_DOCS_PAIR_DATA}"
  cargo run --release -- --prepare-rust-by-practice --input "${RUST_DOCS_ROOT}" --mode pairs --output "${RUST_DOCS_PAIR_DATA}"
  if [[ -s "${RUST_DOCS_PAIR_DATA}" ]]; then
    EXTRA_CODE_MIX_ARGS+=(--extra-pairs "${RUST_DOCS_PAIR_DATA}" --extra-repeat "${CODE_EXTRA_REPEAT}")
  fi
fi

if [[ ! -f "${WIKI_DATA}" ]]; then
  echo "ERROR: WIKI_DATA not found at '${WIKI_DATA}'"
  exit 1
fi

echo "== Stage 1/6: data prep + vocab/token cache =="

echo "Preparing encoder corpus at ${ENCODER_DATA}"
cargo run --release -- --prepare-encoder-corpus --output "${ENCODER_DATA}" "${WORLD_TEXT_DATA}" "${WIKI_DATA}" "${CODE_DATA}" "${EXTRA_ENCODER_INPUTS[@]}"

echo "Preparing Rust instruction pairs at ${RUST_TASK_DATA}"
cargo run --release -- --prepare-rust-function-tasks --input "${CODE_DATA}" --output "${RUST_TASK_DATA}"
if [[ ! -s "${RUST_TASK_DATA}" ]]; then
  echo "No Rust instruction pairs extracted from ${CODE_DATA}; falling back to github-top-code Rust files..."
  cargo run --release -- --prepare-rust-function-tasks --github-top-code --max-files 120000 --output "${RUST_TASK_DATA}"
fi

case "${TOFY_PREPARE_REPAIR_TASKS}" in
  0|false|False|no|No)
    echo "Rust compiler-feedback repair pairs skipped (TOFY_PREPARE_REPAIR_TASKS=${TOFY_PREPARE_REPAIR_TASKS})."
    ;;
  auto|1|true|True|yes|Yes)
    if command -v "${RUSTC_BIN}" >/dev/null 2>&1; then
      echo "Preparing Rust compiler-feedback repair pairs at ${RUST_REPAIR_DATA}"
      cargo run --release -- --prepare-rust-repair-tasks \
        --input "${RUST_TASK_DATA}" \
        --output "${RUST_REPAIR_DATA}" \
        --rustc "${RUSTC_BIN}" \
        --variants-per-sample "${RUST_REPAIR_VARIANTS_PER_SAMPLE}" \
        --timeout-sec "${RUST_REPAIR_TIMEOUT_SEC}" \
        --max-rows "${RUST_REPAIR_MAX_ROWS}"
    elif [[ "${TOFY_PREPARE_REPAIR_TASKS}" == "auto" ]]; then
      echo "Rust compiler-feedback repair pairs skipped: '${RUSTC_BIN}' not found."
    else
      echo "ERROR: TOFY_PREPARE_REPAIR_TASKS=${TOFY_PREPARE_REPAIR_TASKS} but '${RUSTC_BIN}' was not found."
      exit 1
    fi
    ;;
  *)
    echo "ERROR: unsupported TOFY_PREPARE_REPAIR_TASKS='${TOFY_PREPARE_REPAIR_TASKS}' (expected auto, 0, or 1)"
    exit 1
    ;;
esac

WORLD_CODE_PAIR_ARGS=(--code-pairs "${CODE_DATA}" --code-pairs "${RUST_TASK_DATA}")
if [[ -s "${RUST_REPAIR_DATA}" ]]; then
  WORLD_CODE_PAIR_ARGS+=(--code-pairs "${RUST_REPAIR_DATA}")
  EXTRA_CODE_MIX_ARGS+=(--extra-pairs "${RUST_REPAIR_DATA}" --extra-repeat "${CODE_REPAIR_REPEAT}")
fi

echo "Preparing world-model mix at ${WORLD_DATA}"
cargo run --release -- --prepare-world-mix \
  --output "${WORLD_DATA}" \
  --text-pairs "${WORLD_TEXT_DATA}" \
  "${WORLD_CODE_PAIR_ARGS[@]}" \
  --code-ratio "${WORLD_CODE_RATIO}" \
  --done-ratio "${WORLD_DONE_RATIO}" \
  --max-rows "${WORLD_MAX_ROWS}"

echo "Preparing code-first decoder mix at ${CODE_TRAIN_DATA}"
cargo run --release -- --prepare-code-poc-mix \
  --output "${CODE_TRAIN_DATA}" \
  --base-pairs "${CODE_DATA}" \
  --instruction-pairs "${RUST_TASK_DATA}" \
  --instruction-repeat "${CODE_TASK_REPEAT}" \
  "${EXTRA_CODE_MIX_ARGS[@]}" \
  --max-rows "${CODE_TRAIN_MAX_ROWS}"

echo "Generating code eval suite at ${EVAL_SUITE}"
cargo run --release -- --generate-code-eval-suite --output "${EVAL_SUITE}"

if [[ "${TOFY_PRETOKENIZE}" == "1" || "${TOFY_PRETOKENIZE}" == "true" ]]; then
  echo "Stage 1 final: vocab + token cache"
  echo "Code decoder vocab sample budget: rows=${TOFY_CODE_VOCAB_SAMPLE_ROWS}, bytes=${TOFY_CODE_VOCAB_SAMPLE_BYTES}"
  cargo run --release -- \
    --prepare-pipeline-cache "${ENCODER_DATA}" "${WORLD_DATA}" "${CODE_TRAIN_DATA}" "${ENCODER_CACHE_VOCAB}" "${CODE_DECODER_CACHE_VOCAB}" "${TOFY_CACHE_DIR}" \
    --encoder-max-vocab "${MAX_VOCAB}" --code-max-vocab "${CODE_DECODER_MAX_VOCAB}" \
    --encoder-max-seq "${ENCODER_CACHE_MAX_SEQ}" --world-max-seq "${WORLD_MAX_SEQ}" --code-max-seq "${CODE_DECODER_MAX_SEQ}"
  if [[ -f "${CODE_DECODER_CACHE_VOCAB}" && ! -f "${CODE_DECODER_VOCAB}" ]]; then
    mkdir -p "$(dirname "${CODE_DECODER_VOCAB}")"
    cp "${CODE_DECODER_CACHE_VOCAB}" "${CODE_DECODER_VOCAB}"
  fi
  if ! is_true "${TOFY_RESUME}"; then
    export TOFY_ENCODER_VOCAB="${ENCODER_CACHE_VOCAB}"
  fi
else
  echo "Stage 1 final: vocab + token cache skipped (TOFY_PRETOKENIZE=0)"
fi

DECODER_VOCAB_ARGS=()
if [[ -n "${CODE_DECODER_VOCAB}" ]]; then
  DECODER_VOCAB_ARGS=(--decoder-vocab "${CODE_DECODER_VOCAB}")
fi

extract_summary_metric() {
  local summary_path="$1"
  local key="$2"
  awk -F= -v target="${key}" '$1 == target { print $2; exit }' "${summary_path}" 2>/dev/null
}

compare_float_gt() {
  python3 - "$1" "$2" <<'PY'
import sys
a = float(sys.argv[1])
b = float(sys.argv[2])
sys.exit(0 if a > b else 1)
PY
}

run_code_eval_with_label() {
  local decoder_path="$1"
  local label="$2"
  local summary_dest="${CODE_EVAL_STAGE_DIR}/${label}_summary.txt"
  echo "Evaluating ${label} decoder: ${decoder_path}"
  TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="code_eval_${label}" cargo run --release -- \
    --eval-code-assistant "${WORLD_ENCODER_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${EVAL_SUITE}" 384 "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" \
    --code-decoder "${decoder_path}" --code-decoder-vocab "${CODE_DECODER_VOCAB}" \
    --candidates "${TOFY_EVAL_CANDIDATES:-4}" --repair-attempts "${TOFY_EVAL_REPAIR_ATTEMPTS:-2}"
  local latest_summary
  latest_summary="$(find "${CODE_EVAL_STAGE_DIR}" -maxdepth 2 -name summary.txt -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
  if [[ -z "${latest_summary}" || ! -f "${latest_summary}" ]]; then
    echo "ERROR: eval summary not found for ${label}"
    exit 1
  fi
  cp "${latest_summary}" "${summary_dest}"
  echo "Saved ${label} eval summary to ${summary_dest}"
}

select_best_decoder_checkpoint() {
  local selected_path="${CODE_DECODER_BASE_OUTPUT}"
  local selected_label="base"
  local selected_suite
  local selected_compile
  local selected_tests
  local selected_constraints
  selected_suite="$(extract_summary_metric "${CODE_EVAL_STAGE_DIR}/base_summary.txt" suite_pass_rate)"
  selected_compile="$(extract_summary_metric "${CODE_EVAL_STAGE_DIR}/base_summary.txt" compile_rate)"
  selected_tests="$(extract_summary_metric "${CODE_EVAL_STAGE_DIR}/base_summary.txt" test_pass_rate)"
  selected_constraints="$(extract_summary_metric "${CODE_EVAL_STAGE_DIR}/base_summary.txt" constraint_pass_rate)"
  if [[ -f "${CODE_EVAL_STAGE_DIR}/polish_summary.txt" ]]; then
    local polish_suite polish_compile polish_tests polish_constraints
    polish_suite="$(extract_summary_metric "${CODE_EVAL_STAGE_DIR}/polish_summary.txt" suite_pass_rate)"
    polish_compile="$(extract_summary_metric "${CODE_EVAL_STAGE_DIR}/polish_summary.txt" compile_rate)"
    polish_tests="$(extract_summary_metric "${CODE_EVAL_STAGE_DIR}/polish_summary.txt" test_pass_rate)"
    polish_constraints="$(extract_summary_metric "${CODE_EVAL_STAGE_DIR}/polish_summary.txt" constraint_pass_rate)"
    if compare_float_gt "${polish_suite:-0}" "${selected_suite:-0}" \
      || { [[ "${polish_suite:-0}" == "${selected_suite:-0}" ]] && compare_float_gt "${polish_tests:-0}" "${selected_tests:-0}"; } \
      || { [[ "${polish_suite:-0}" == "${selected_suite:-0}" && "${polish_tests:-0}" == "${selected_tests:-0}" ]] && compare_float_gt "${polish_compile:-0}" "${selected_compile:-0}"; } \
      || { [[ "${polish_suite:-0}" == "${selected_suite:-0}" && "${polish_tests:-0}" == "${selected_tests:-0}" && "${polish_compile:-0}" == "${selected_compile:-0}" ]] && compare_float_gt "${polish_constraints:-0}" "${selected_constraints:-0}"; }; then
      selected_path="${CODE_DECODER_POLISH_OUTPUT}"
      selected_label="polish"
      selected_suite="${polish_suite}"
      selected_compile="${polish_compile}"
      selected_tests="${polish_tests}"
      selected_constraints="${polish_constraints}"
    fi
  fi
  CODE_DECODER_OUTPUT="${selected_path}"
  echo "Selected ${selected_label} decoder for final eval/promotion: ${CODE_DECODER_OUTPUT}"
  echo "Selected metrics: suite=${selected_suite:-0} tests=${selected_tests:-0} compile=${selected_compile:-0} constraints=${selected_constraints:-0}"
}

echo "== Stage 2/6: encoder =="
if is_true "${TOFY_RESUME}" && [[ -f "${LATENT_MODEL}" ]] && resume_stage_complete "${LATENT_MODEL}" "latent" "${LATENT_STEPS}"; then
  echo "Skipping encoder; resume state already reached ${LATENT_STEPS} steps."
else
  TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="latent" cargo run --release -- \
    --latent "${ENCODER_DATA}" "${LATENT_STEPS}" "${LATENT_BATCH}" "${DIM}" "${LATENT_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}" \
    --grad-accum "${LATENT_GRAD_ACCUM}" --output "${LATENT_MODEL}" "${RESUME_ARGS[@]}"
fi
if [[ ! -f "${LATENT_MODEL}" ]]; then
  echo "ERROR: latent checkpoint not found at '${LATENT_MODEL}'"
  exit 1
fi
if [[ -z "${ENCODER_VOCAB}" ]]; then
  MATCHED_ENCODER_VOCAB="${LATENT_MODEL%.safetensors}.vocab.txt"
  if [[ -f "${MATCHED_ENCODER_VOCAB}" ]]; then
    ENCODER_VOCAB="${MATCHED_ENCODER_VOCAB}"
  else
    echo "ERROR: encoder vocab not found at '${MATCHED_ENCODER_VOCAB}'"
    exit 1
  fi
fi
echo "Encoder vocab: ${ENCODER_VOCAB}"

echo "== Stage 3/6: world transition =="
if is_true "${TOFY_RESUME}" && [[ -f "${WORLD_MODEL}" ]] && resume_stage_complete "${WORLD_MODEL}" "world" "${WORLD_STEPS}"; then
  echo "Skipping world transition; resume state already reached ${WORLD_STEPS} steps."
else
  TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="world" cargo run --release -- \
    --train-world "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_DATA}" "${WORLD_STEPS}" "${WORLD_BATCH}" "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" \
    --lambda "${WORLD_LAMBDA}" --lr "${WORLD_LR}" --grad-accum "${WORLD_GRAD_ACCUM}" \
    --output "${WORLD_MODEL}" --encoder-output "${WORLD_ENCODER_MODEL}" --action-loss-weight "${WORLD_ACTION_LOSS_WEIGHT}" "${RESUME_ARGS[@]}"
fi
if [[ ! -f "${WORLD_MODEL}" ]]; then
  echo "ERROR: world checkpoint not found at '${WORLD_MODEL}'"
  exit 1
fi
if [[ ! -f "${WORLD_ENCODER_MODEL}" ]]; then
  echo "ERROR: LeWM encoder checkpoint not found at '${WORLD_ENCODER_MODEL}'"
  exit 1
fi

if [[ "${HIGH_WORLD_STEPS}" -gt 0 ]]; then
  echo "== Stage 3b/6: high-level world transition =="
  if is_true "${TOFY_RESUME}" && [[ -f "${HIGH_WORLD_MODEL}" ]] && resume_stage_complete "${HIGH_WORLD_MODEL}" "high_world" "${HIGH_WORLD_STEPS}"; then
    echo "Skipping high-level world transition; resume state already reached ${HIGH_WORLD_STEPS} steps."
  else
    TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="high_world" cargo run --release -- \
      --train-high-world "${WORLD_ENCODER_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${WORLD_DATA}" "${HIGH_WORLD_STEPS}" "${WORLD_BATCH}" "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" \
      --macro-min-len "${HWM_MACRO_MIN_LEN}" --macro-max-len "${HWM_MACRO_MAX_LEN}" \
      --lambda "${WORLD_LAMBDA}" --lr "${WORLD_LR}" --grad-accum "${WORLD_GRAD_ACCUM}" \
      --output "${HIGH_WORLD_MODEL}" "${RESUME_ARGS[@]}"
  fi
  if [[ ! -f "${HIGH_WORLD_MODEL}" ]]; then
    echo "ERROR: high-level world checkpoint not found at '${HIGH_WORLD_MODEL}'"
    exit 1
  fi
  export TOFY_HIGH_WORLD_MODEL="${HIGH_WORLD_MODEL}"
fi

if [[ "${ROUTER_STEPS}" -gt 0 ]]; then
  echo "== Stage 4/6: downstream router/planner tune =="
  if is_true "${TOFY_RESUME}" && resume_stage_complete "${WORLD_MODEL}" "orchestrator" "${ROUTER_STEPS}"; then
    echo "Skipping downstream router/planner tune; resume state already reached ${ROUTER_STEPS} steps."
  else
    TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="orchestrator" cargo run --release -- \
      --train-orchestrator "${WORLD_ENCODER_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${WORLD_DATA}" "${ROUTER_STEPS}" "${ROUTER_BATCH}" "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" \
      --lr "${WORLD_LR}" --grad-accum "${ROUTER_GRAD_ACCUM}" --output "${WORLD_MODEL}" "${RESUME_ARGS[@]}"
  fi
else
  echo "== Stage 4/6: downstream router/planner tune skipped (ROUTER_STEPS=0; strict LeJEPA world model remains auxiliary-free) =="
fi

echo "== Stage 5/6: code decoder =="
if is_true "${TOFY_RESUME}" && [[ -f "${CODE_DECODER_BASE_OUTPUT}" ]] && resume_stage_complete "${CODE_DECODER_BASE_OUTPUT}" "decoder_code" "${CODE_DECODER_STEPS}"; then
  echo "Skipping code decoder; resume state already reached ${CODE_DECODER_STEPS} steps."
else
  TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="decoder_code" cargo run --release -- \
    --train-decoder "${WORLD_ENCODER_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${CODE_TRAIN_DATA}" "${CODE_DECODER_STEPS}" "${CODE_DECODER_BATCH}" "${CODE_DECODER_MAX_SEQ}" "${DIM}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" \
    --decoder-kind code --decoder-max-vocab "${CODE_DECODER_MAX_VOCAB}" "${DECODER_VOCAB_ARGS[@]}" --decoder-output "${CODE_DECODER_BASE_OUTPUT}" --grad-accum "${CODE_DECODER_GRAD_ACCUM}" --lr "${CODE_DECODER_LR}" "${RESUME_ARGS[@]}"
fi
if [[ ! -f "${CODE_DECODER_BASE_OUTPUT}" ]]; then
  echo "ERROR: code decoder checkpoint not found at '${CODE_DECODER_BASE_OUTPUT}'"
  exit 1
fi

if [[ "${CODE_POLISH_STEPS}" -gt 0 ]]; then
  echo "== Stage 5b/6: code decoder instruction polish =="
  if is_true "${TOFY_RESUME}" && resume_stage_complete "${CODE_DECODER_POLISH_OUTPUT}" "decoder_code_polish" "${CODE_POLISH_STEPS}"; then
    echo "Skipping code decoder polish; resume state already reached ${CODE_POLISH_STEPS} steps."
  else
    TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="decoder_code_polish" cargo run --release -- \
      --train-decoder "${WORLD_ENCODER_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${RUST_TASK_DATA}" "${CODE_POLISH_STEPS}" "${CODE_DECODER_BATCH}" "${CODE_DECODER_MAX_SEQ}" "${DIM}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" \
      --decoder-kind code --decoder-max-vocab "${CODE_DECODER_MAX_VOCAB}" "${DECODER_VOCAB_ARGS[@]}" --decoder-output "${CODE_DECODER_POLISH_OUTPUT}" --init-decoder "${CODE_DECODER_BASE_OUTPUT}" --grad-accum "${CODE_DECODER_GRAD_ACCUM}" --lr "${CODE_POLISH_LR}" "${RESUME_ARGS[@]}"
  fi
fi

echo "== Stage 5c/6: verifier-guided checkpoint selection =="
run_code_eval_with_label "${CODE_DECODER_BASE_OUTPUT}" "base"
if [[ "${CODE_POLISH_STEPS}" -gt 0 && -f "${CODE_DECODER_POLISH_OUTPUT}" ]]; then
  run_code_eval_with_label "${CODE_DECODER_POLISH_OUTPUT}" "polish"
fi
select_best_decoder_checkpoint
write_pipeline_meta

echo "== Stage 6/6: code eval suite =="
TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="code_eval" cargo run --release -- \
  --eval-code-assistant "${WORLD_ENCODER_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${EVAL_SUITE}" 384 "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" \
  --code-decoder "${CODE_DECODER_OUTPUT}" --code-decoder-vocab "${CODE_DECODER_VOCAB}" \
  --candidates "${TOFY_EVAL_CANDIDATES:-4}" --repair-attempts "${TOFY_EVAL_REPAIR_ATTEMPTS:-2}"

echo "Code-first POC pipeline complete."
