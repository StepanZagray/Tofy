#!/usr/bin/env bash
set -euo pipefail

# Full training pipeline (vocab/token cache -> encoder -> world transition -> planner/orchestrator tune -> code decoder -> text decoder).
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
RUST_TASK_DATA="${RUST_TASK_DATA:-data/rust_instruction_pairs.txt}"
TOFY_GPU_PROFILE="${TOFY_GPU_PROFILE:-8gb}"
TOFY_TRAIN_DTYPE="${TOFY_TRAIN_DTYPE:-bf16}"
TOFY_SIGREG_SLICES="${TOFY_SIGREG_SLICES:-1024}"
TOFY_SIGREG_POINTS="${TOFY_SIGREG_POINTS:-17}"
TOFY_LATENT_CONTEXT_SEGMENTS="${TOFY_LATENT_CONTEXT_SEGMENTS:-4}"
TOFY_LATENT_RECENT_FULL_SEGMENTS="${TOFY_LATENT_RECENT_FULL_SEGMENTS:-1}"
TOFY_LATENT_HISTORY_RATIO="${TOFY_LATENT_HISTORY_RATIO:-0.35}"
TOFY_DECODER_SYNTAX_LOSS_WEIGHT="${TOFY_DECODER_SYNTAX_LOSS_WEIGHT:-0.0}"
TOFY_DECODER_SIGNATURE_LOSS_WEIGHT="${TOFY_DECODER_SIGNATURE_LOSS_WEIGHT:-0.0}"
TOFY_DECODER_STRUCTURE_LOSS_WEIGHT="${TOFY_DECODER_STRUCTURE_LOSS_WEIGHT:-0.0}"
TOFY_DECODER_CONDITIONING_LOSS_WEIGHT="${TOFY_DECODER_CONDITIONING_LOSS_WEIGHT:-auto}"
TOFY_DECODER_CONDITIONING_MARGIN="${TOFY_DECODER_CONDITIONING_MARGIN:-0.08}"
TOFY_WORLD_INVERSE_LOSS_WEIGHT="${TOFY_WORLD_INVERSE_LOSS_WEIGHT:-0.0}"
export TOFY_TRAIN_DTYPE TOFY_SIGREG_SLICES TOFY_SIGREG_POINTS
export TOFY_LATENT_CONTEXT_SEGMENTS TOFY_LATENT_RECENT_FULL_SEGMENTS TOFY_LATENT_HISTORY_RATIO
export TOFY_DECODER_SYNTAX_LOSS_WEIGHT TOFY_DECODER_SIGNATURE_LOSS_WEIGHT TOFY_DECODER_STRUCTURE_LOSS_WEIGHT TOFY_DECODER_CONDITIONING_LOSS_WEIGHT TOFY_DECODER_CONDITIONING_MARGIN TOFY_WORLD_INVERSE_LOSS_WEIGHT

case "${TOFY_GPU_PROFILE}" in
  8gb)
    PROFILE_LATENT_STEPS=25000
    PROFILE_WORLD_STEPS=60000
    PROFILE_HIGH_WORLD_STEPS=0
    PROFILE_ROUTER_STEPS=0
    PROFILE_CODE_DECODER_STEPS=40000
    PROFILE_CODE_POLISH_STEPS=4000
    PROFILE_TEXT_DECODER_STEPS=40000
    PROFILE_DIM=640
    PROFILE_LATENT_MAX_SEQ=256
    PROFILE_WORLD_MAX_SEQ=256
    PROFILE_DECODER_MAX_SEQ=192
    PROFILE_CODE_DECODER_MAX_SEQ=128
    PROFILE_TEXT_DECODER_MAX_SEQ=128
    PROFILE_LAYERS=7
    PROFILE_HEADS=8
    PROFILE_MAX_VOCAB=8000
    PROFILE_CODE_DECODER_MAX_VOCAB=16000
    PROFILE_TEXT_DECODER_MAX_VOCAB=16000
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
    PROFILE_CODE_POLISH_STEPS=12000
    PROFILE_TEXT_DECODER_STEPS=120000
    PROFILE_DIM=1536
    PROFILE_LATENT_MAX_SEQ=256
    PROFILE_WORLD_MAX_SEQ=256
    PROFILE_DECODER_MAX_SEQ=192
    PROFILE_CODE_DECODER_MAX_SEQ=128
    PROFILE_TEXT_DECODER_MAX_SEQ=128
    PROFILE_LAYERS=7
    PROFILE_HEADS=12
    PROFILE_MAX_VOCAB=12000
    PROFILE_CODE_DECODER_MAX_VOCAB=24000
    PROFILE_TEXT_DECODER_MAX_VOCAB=24000
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
    PROFILE_CODE_POLISH_STEPS=40000
    PROFILE_TEXT_DECODER_STEPS=400000
    PROFILE_DIM=2048
    PROFILE_LATENT_MAX_SEQ=256
    PROFILE_WORLD_MAX_SEQ=256
    PROFILE_DECODER_MAX_SEQ=192
    PROFILE_CODE_DECODER_MAX_SEQ=128
    PROFILE_TEXT_DECODER_MAX_SEQ=128
    PROFILE_LAYERS=7
    PROFILE_HEADS=16
    PROFILE_MAX_VOCAB=16000
    PROFILE_CODE_DECODER_MAX_VOCAB=32000
    PROFILE_TEXT_DECODER_MAX_VOCAB=32000
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
TEXT_DECODER_STEPS="${TEXT_DECODER_STEPS:-${PROFILE_TEXT_DECODER_STEPS}}"

DIM="${DIM:-${PROFILE_DIM}}"
LATENT_MAX_SEQ="${LATENT_MAX_SEQ:-${PROFILE_LATENT_MAX_SEQ}}"
WORLD_MAX_SEQ="${WORLD_MAX_SEQ:-${PROFILE_WORLD_MAX_SEQ}}"
DECODER_MAX_SEQ="${DECODER_MAX_SEQ:-${PROFILE_DECODER_MAX_SEQ}}"
CODE_DECODER_MAX_SEQ="${CODE_DECODER_MAX_SEQ:-auto}"
TEXT_DECODER_MAX_SEQ="${TEXT_DECODER_MAX_SEQ:-${PROFILE_TEXT_DECODER_MAX_SEQ}}"
LAYERS="${LAYERS:-${PROFILE_LAYERS}}"
HEADS="${HEADS:-${PROFILE_HEADS}}"
MAX_VOCAB="${MAX_VOCAB:-${PROFILE_MAX_VOCAB}}"
CODE_DECODER_MAX_VOCAB="${CODE_DECODER_MAX_VOCAB:-${PROFILE_CODE_DECODER_MAX_VOCAB}}"
TEXT_DECODER_MAX_VOCAB="${TEXT_DECODER_MAX_VOCAB:-${PROFILE_TEXT_DECODER_MAX_VOCAB}}"
BRIDGE_DIM="${BRIDGE_DIM:-${PROFILE_BRIDGE_DIM}}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-${PROFILE_NUM_LATENT_TOKENS}}"
WIKI_MAX_FILES="${WIKI_MAX_FILES:-1}"
WORLD_LR="${WORLD_LR:-2e-4}"
CODE_POLISH_LR="${CODE_POLISH_LR:-1e-4}"
WORLD_LAMBDA="${WORLD_LAMBDA:-0.2}"
WORLD_ACTION_LOSS_WEIGHT="${WORLD_ACTION_LOSS_WEIGHT:-0.0}"
HWM_MACRO_MIN_LEN="${HWM_MACRO_MIN_LEN:-2}"
HWM_MACRO_MAX_LEN="${HWM_MACRO_MAX_LEN:-4}"
WORLD_CODE_RATIO="${WORLD_CODE_RATIO:-0.35}"
WORLD_DONE_RATIO="${WORLD_DONE_RATIO:-0.18}"
WORLD_MAX_ROWS="${WORLD_MAX_ROWS:-0}"
ENCODER_VOCAB="${ENCODER_VOCAB:-}"
CODE_DECODER_OUTPUT="${CODE_DECODER_OUTPUT:-}"
CODE_DECODER_BASE_OUTPUT="${CODE_DECODER_BASE_OUTPUT:-}"
CODE_DECODER_POLISH_OUTPUT="${CODE_DECODER_POLISH_OUTPUT:-}"
TEXT_DECODER_OUTPUT="${TEXT_DECODER_OUTPUT:-}"
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
  echo "  ./scripts/train_full_pipeline.sh"
  echo "  ./scripts/train_full_pipeline.sh --resume latest"
  echo "  ./scripts/train_full_pipeline.sh --resume <run_id|runs/path>"
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
TEXT_DECODER_BATCH="${TEXT_DECODER_BATCH:-${DECODER_BATCH}}"
DECODER_GRAD_ACCUM="${DECODER_GRAD_ACCUM:-${DEFAULT_DECODER_GRAD_ACCUM}}"
LATENT_GRAD_ACCUM="${LATENT_GRAD_ACCUM:-${DEFAULT_LATENT_GRAD_ACCUM}}"
WORLD_GRAD_ACCUM="${WORLD_GRAD_ACCUM:-${DEFAULT_WORLD_GRAD_ACCUM}}"
ROUTER_BATCH="${ROUTER_BATCH:-${WORLD_BATCH}}"
ROUTER_GRAD_ACCUM="${ROUTER_GRAD_ACCUM:-${WORLD_GRAD_ACCUM}}"
CODE_DECODER_GRAD_ACCUM="${CODE_DECODER_GRAD_ACCUM:-${DECODER_GRAD_ACCUM}}"
TEXT_DECODER_GRAD_ACCUM="${TEXT_DECODER_GRAD_ACCUM:-${DECODER_GRAD_ACCUM}}"
export TOFY_LATENT_WARMUP_BATCH TOFY_LATENT_WARMUP_GRAD_ACCUM
export TOFY_WORLD_WARMUP_BATCH TOFY_WORLD_WARMUP_GRAD_ACCUM TOFY_WORLD_WARMUP_STEPS
export TOFY_WORLD_LOG_EVERY TOFY_ORCHESTRATOR_LOG_EVERY TOFY_DECODER_LOG_EVERY

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

RESUME_ARGS=()
if is_true "${TOFY_RESUME}"; then
  RESUME_ARGS=(--resume)
fi
if is_true "${TOFY_RESUME}"; then
  if [[ -z "${TOFY_RESUME_RUN}" ]]; then
    TOFY_RESUME_RUN="${PIPELINE_RUN_ID:-latest}"
  fi
  PIPELINE_RUN_ROOT="$(resolve_run_root "${TOFY_RESUME_RUN}" "pipeline_")"
  if [[ -z "${PIPELINE_RUN_ROOT}" || ! -d "${PIPELINE_RUN_ROOT}" ]]; then
    echo "ERROR: could not resolve resume run '${TOFY_RESUME_RUN}'"
    exit 1
  fi
  PIPELINE_RUN_ID="$(basename "${PIPELINE_RUN_ROOT}")"
else
  PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-pipeline_$(date +%Y-%m-%d_%H-%M-%S)}"
  PIPELINE_RUN_ROOT="runs/${PIPELINE_RUN_ID}"
  if [[ -e "${PIPELINE_RUN_ROOT}" ]]; then
    echo "ERROR: run directory already exists at '${PIPELINE_RUN_ROOT}'"
    exit 1
  fi
fi

LATENT_STAGE_DIR="${PIPELINE_RUN_ROOT}/latent"
WORLD_STAGE_DIR="${PIPELINE_RUN_ROOT}/world"
HIGH_WORLD_STAGE_DIR="${PIPELINE_RUN_ROOT}/high_world"
ORCHESTRATOR_STAGE_DIR="${PIPELINE_RUN_ROOT}/orchestrator"
CODE_DECODER_STAGE_DIR="${PIPELINE_RUN_ROOT}/decoder_code"
CODE_POLISH_STAGE_DIR="${PIPELINE_RUN_ROOT}/decoder_code_polish"
TEXT_DECODER_STAGE_DIR="${PIPELINE_RUN_ROOT}/decoder_text"
LATENT_MODEL="${LATENT_MODEL:-${LATENT_STAGE_DIR}/model.safetensors}"
WORLD_MODEL="${WORLD_MODEL:-${WORLD_STAGE_DIR}/model.safetensors}"
WORLD_ENCODER_MODEL="${WORLD_ENCODER_MODEL:-${WORLD_STAGE_DIR}/model.encoder.safetensors}"
HIGH_WORLD_MODEL="${HIGH_WORLD_MODEL:-${HIGH_WORLD_STAGE_DIR}/model.safetensors}"
CODE_DECODER_BASE_OUTPUT="${CODE_DECODER_BASE_OUTPUT:-${CODE_DECODER_STAGE_DIR}/model.safetensors}"
CODE_DECODER_POLISH_OUTPUT="${CODE_DECODER_POLISH_OUTPUT:-${CODE_POLISH_STAGE_DIR}/model.safetensors}"
CODE_DECODER_OUTPUT="${CODE_DECODER_OUTPUT:-${CODE_DECODER_BASE_OUTPUT}}"
TEXT_DECODER_OUTPUT="${TEXT_DECODER_OUTPUT:-${TEXT_DECODER_STAGE_DIR}/model.safetensors}"
if [[ -z "${CODE_DECODER_VOCAB}" ]]; then
  CODE_DECODER_VOCAB="${CODE_DECODER_BASE_OUTPUT%.safetensors}.vocab.txt"
fi
mkdir -p \
  "${PIPELINE_RUN_ROOT}" \
  "${LATENT_STAGE_DIR}" \
  "${WORLD_STAGE_DIR}" \
  "${HIGH_WORLD_STAGE_DIR}" \
  "${ORCHESTRATOR_STAGE_DIR}" \
  "${CODE_DECODER_STAGE_DIR}" \
  "${CODE_POLISH_STAGE_DIR}" \
  "${TEXT_DECODER_STAGE_DIR}"

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
  "pipeline_kind": "full_pipeline",
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
  "text_decoder_model": "${TEXT_DECODER_OUTPUT}",
  "code_decoder_vocab": "${CODE_DECODER_VOCAB}",
  "encoder_data": "${ENCODER_DATA}",
  "world_data": "${WORLD_DATA}",
  "code_data": "${CODE_DATA}",
  "text_data": "${TEXT_DATA}",
  "latent_steps": "${LATENT_STEPS}",
  "world_steps": "${WORLD_STEPS}",
  "high_world_steps": "${HIGH_WORLD_STEPS}",
  "router_steps": "${ROUTER_STEPS}",
  "code_decoder_steps": "${CODE_DECODER_STEPS}",
  "text_decoder_steps": "${TEXT_DECODER_STEPS}",
  "gpu_profile": "${TOFY_GPU_PROFILE}",
  "total_vram_mb": "${TOTAL_VRAM_MB}",
  "latent_batch": "${LATENT_BATCH}",
  "latent_warmup_batch": "${TOFY_LATENT_WARMUP_BATCH}",
  "world_batch": "${WORLD_BATCH}",
  "world_warmup_batch": "${TOFY_WORLD_WARMUP_BATCH}",
  "code_decoder_batch": "${CODE_DECODER_BATCH}",
  "text_decoder_batch": "${TEXT_DECODER_BATCH}",
  "latent_grad_accum": "${LATENT_GRAD_ACCUM}",
  "latent_warmup_grad_accum": "${TOFY_LATENT_WARMUP_GRAD_ACCUM}",
  "world_grad_accum": "${WORLD_GRAD_ACCUM}",
  "world_warmup_grad_accum": "${TOFY_WORLD_WARMUP_GRAD_ACCUM}",
  "code_decoder_grad_accum": "${CODE_DECODER_GRAD_ACCUM}",
  "text_decoder_grad_accum": "${TEXT_DECODER_GRAD_ACCUM}",
  "dim": "${DIM}",
  "layers": "${LAYERS}",
  "heads": "${HEADS}",
  "bridge_dim": "${BRIDGE_DIM}",
  "num_latent_tokens": "${NUM_LATENT_TOKENS}"
}
EOF
}
echo "Pipeline run directory: ${PIPELINE_RUN_ROOT}"
if is_true "${TOFY_RESUME}"; then
  echo "Resuming pipeline run: ${PIPELINE_RUN_ID} (selector=${TOFY_RESUME_RUN})"
fi
echo "GPU profile: ${TOFY_GPU_PROFILE} (vram_mb=${TOTAL_VRAM_MB:-unknown})"
echo "Microbatches: latent=${LATENT_BATCH} world=${WORLD_BATCH} code_decoder=${CODE_DECODER_BATCH} text_decoder=${TEXT_DECODER_BATCH}"
echo "Grad accum: latent=${LATENT_GRAD_ACCUM} world=${WORLD_GRAD_ACCUM} code_decoder=${CODE_DECODER_GRAD_ACCUM} text_decoder=${TEXT_DECODER_GRAD_ACCUM}"
echo "Effective batch: latent=$((LATENT_BATCH * LATENT_GRAD_ACCUM)) world=$((WORLD_BATCH * WORLD_GRAD_ACCUM)) code_decoder=$((CODE_DECODER_BATCH * CODE_DECODER_GRAD_ACCUM)) text_decoder=$((TEXT_DECODER_BATCH * TEXT_DECODER_GRAD_ACCUM))"
echo "Latent warmup: batch=${TOFY_LATENT_WARMUP_BATCH} grad_accum=${TOFY_LATENT_WARMUP_GRAD_ACCUM} effective=$((TOFY_LATENT_WARMUP_BATCH * TOFY_LATENT_WARMUP_GRAD_ACCUM)) for ${TOFY_LATENT_WARMUP_STEPS:-20%} of latent steps"
echo "World warmup: batch=${TOFY_WORLD_WARMUP_BATCH} grad_accum=${TOFY_WORLD_WARMUP_GRAD_ACCUM} effective=$((TOFY_WORLD_WARMUP_BATCH * TOFY_WORLD_WARMUP_GRAD_ACCUM)) for ${TOFY_WORLD_WARMUP_STEPS:-20%} of world steps"
echo "Training dtype: ${TOFY_TRAIN_DTYPE} | latent_segments=${TOFY_LATENT_CONTEXT_SEGMENTS} recent_full=${TOFY_LATENT_RECENT_FULL_SEGMENTS} history_ratio=${TOFY_LATENT_HISTORY_RATIO}"
echo "LeJEPA objective: online prediction + SIGReg | slices=${TOFY_SIGREG_SLICES} points=${TOFY_SIGREG_POINTS}"
echo "High-world: steps=${HIGH_WORLD_STEPS} macro_len=${HWM_MACRO_MIN_LEN}..${HWM_MACRO_MAX_LEN}"
write_pipeline_meta

maybe_export_cuda_compat
if [[ -n "${CUDA_COMPUTE_CAP:-}" ]]; then
  echo "CUDA build env: CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-unset}"
fi

# --- Checks ---
if [[ ! -f "${WORLD_TEXT_DATA}" ]]; then
  echo "ERROR: WORLD_TEXT_DATA not found at '${WORLD_TEXT_DATA}'"
  echo "  Prepare chat pairs: cargo run --release -- --prepare-ultrachat data/ultrachat_pairs.txt 6 2"
  echo "  Or code pairs: cargo run --release -- --prepare-github-top-code --default-languages --max-files 100000 --output data/multilang_pairs.txt"
  exit 1
fi

if [[ ! -f "${CODE_DATA}" ]]; then
  echo "CODE_DATA not found at '${CODE_DATA}', generating multilingual code pairs..."
  cargo run --release -- --prepare-github-top-code --output "${CODE_DATA}" --default-languages --max-files 200000
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

echo "== Stage 1/6: data prep + vocab/token cache =="

echo "Preparing encoder corpus at ${ENCODER_DATA}"
cargo run --release -- --prepare-encoder-corpus --output "${ENCODER_DATA}" "${WORLD_TEXT_DATA}" "${WIKI_DATA}" "${CODE_DATA}"

echo "Preparing Rust instruction pairs at ${RUST_TASK_DATA}"
cargo run --release -- --prepare-rust-function-tasks --input "${CODE_DATA}" --output "${RUST_TASK_DATA}" || true

echo "Preparing world-model mix at ${WORLD_DATA}"
cargo run --release -- --prepare-world-mix \
  --output "${WORLD_DATA}" \
  --text-pairs "${WORLD_TEXT_DATA}" \
  --code-pairs "${CODE_DATA}" \
  $( [[ -s "${RUST_TASK_DATA}" ]] && printf -- '--code-pairs %q ' "${RUST_TASK_DATA}" ) \
  --code-ratio "${WORLD_CODE_RATIO}" \
  --done-ratio "${WORLD_DONE_RATIO}" \
  --max-rows "${WORLD_MAX_ROWS}"

# --- Stage 1: Vocab/token cache ---
if [[ "${TOFY_PRETOKENIZE}" == "1" || "${TOFY_PRETOKENIZE}" == "true" ]]; then
  echo "Stage 1 final: vocab + token cache"
  cargo run --release -- \
    --prepare-pipeline-cache "${ENCODER_DATA}" "${WORLD_DATA}" "${CODE_DATA}" "${ENCODER_CACHE_VOCAB}" "${CODE_DECODER_CACHE_VOCAB}" "${TOFY_CACHE_DIR}" \
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

# --- Stage 2: LeJEPA encoder ---
echo "== Stage 2/6: LeJEPA encoder (--latent) =="
if is_true "${TOFY_RESUME}" && [[ -f "${LATENT_MODEL}" ]] && resume_stage_complete "${LATENT_MODEL}" "latent" "${LATENT_STEPS}"; then
  echo "Skipping encoder; resume state already reached ${LATENT_STEPS} steps."
else
  TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="latent" cargo run --release -- --latent "${ENCODER_DATA}" "${LATENT_STEPS}" "${LATENT_BATCH}" "${DIM}" "${LATENT_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}" --grad-accum "${LATENT_GRAD_ACCUM}" --output "${LATENT_MODEL}" "${RESUME_ARGS[@]}"
fi
if [[ ! -f "${LATENT_MODEL}" ]]; then
  echo "ERROR: latent checkpoint not found at '${LATENT_MODEL}'."
  exit 1
fi
if [[ -z "${ENCODER_VOCAB}" ]]; then
  MATCHED_ENCODER_VOCAB="${LATENT_MODEL%.safetensors}.vocab.txt"
  if [[ -f "${MATCHED_ENCODER_VOCAB}" ]]; then
    ENCODER_VOCAB="${MATCHED_ENCODER_VOCAB}"
  else
    echo "ERROR: encoder vocab not found at '${MATCHED_ENCODER_VOCAB}'."
    exit 1
  fi
fi
if [[ ! -f "${ENCODER_VOCAB}" ]]; then
  echo "ERROR: ${ENCODER_VOCAB} not found after encoder training."
  exit 1
fi
echo "  Using: ${LATENT_MODEL}"
echo "  Encoder vocab: ${ENCODER_VOCAB}"

# --- Stage 3: Planner/world model ---
echo "== Stage 3/6: Planner/world model (--train-world) =="
if is_true "${TOFY_RESUME}" && [[ -f "${WORLD_MODEL}" ]] && resume_stage_complete "${WORLD_MODEL}" "world" "${WORLD_STEPS}"; then
  echo "Skipping planner/world model; resume state already reached ${WORLD_STEPS} steps."
else
    TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="world" cargo run --release -- --train-world "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_DATA}" "${WORLD_STEPS}" "${WORLD_BATCH}" "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" --lambda "${WORLD_LAMBDA}" --lr "${WORLD_LR}" --grad-accum "${WORLD_GRAD_ACCUM}" --output "${WORLD_MODEL}" --encoder-output "${WORLD_ENCODER_MODEL}" --action-loss-weight "${WORLD_ACTION_LOSS_WEIGHT}" "${RESUME_ARGS[@]}"
fi
if [[ ! -f "${WORLD_MODEL}" ]]; then
  echo "ERROR: world checkpoint not found at '${WORLD_MODEL}'."
  exit 1
fi
if [[ ! -f "${WORLD_ENCODER_MODEL}" ]]; then
  echo "ERROR: LeWM encoder checkpoint not found at '${WORLD_ENCODER_MODEL}'."
  exit 1
fi
echo "  Using: ${WORLD_MODEL}"
echo "  LeWM encoder: ${WORLD_ENCODER_MODEL}"

if [[ "${HIGH_WORLD_STEPS}" -gt 0 ]]; then
  echo "== Stage 3b/6: High-level world model (--train-high-world) =="
  if is_true "${TOFY_RESUME}" && [[ -f "${HIGH_WORLD_MODEL}" ]] && resume_stage_complete "${HIGH_WORLD_MODEL}" "high_world" "${HIGH_WORLD_STEPS}"; then
    echo "Skipping high-level world model; resume state already reached ${HIGH_WORLD_STEPS} steps."
  else
    TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="high_world" cargo run --release -- --train-high-world "${WORLD_ENCODER_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${WORLD_DATA}" "${HIGH_WORLD_STEPS}" "${WORLD_BATCH}" "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" --macro-min-len "${HWM_MACRO_MIN_LEN}" --macro-max-len "${HWM_MACRO_MAX_LEN}" --lambda "${WORLD_LAMBDA}" --lr "${WORLD_LR}" --grad-accum "${WORLD_GRAD_ACCUM}" --output "${HIGH_WORLD_MODEL}" "${RESUME_ARGS[@]}"
  fi
  if [[ ! -f "${HIGH_WORLD_MODEL}" ]]; then
    echo "ERROR: high-level world checkpoint not found at '${HIGH_WORLD_MODEL}'."
    exit 1
  fi
  export TOFY_HIGH_WORLD_MODEL="${HIGH_WORLD_MODEL}"
fi

# --- Stage 4: Orchestrator/planner fine-tune ---
if [[ "${ROUTER_STEPS}" -gt 0 ]]; then
  echo "== Stage 4/6: downstream orchestrator/planner (--train-orchestrator) =="
  if is_true "${TOFY_RESUME}" && resume_stage_complete "${WORLD_MODEL}" "orchestrator" "${ROUTER_STEPS}"; then
    echo "Skipping downstream orchestrator/planner; resume state already reached ${ROUTER_STEPS} steps."
  else
    TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="orchestrator" cargo run --release -- --train-orchestrator "${WORLD_ENCODER_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${WORLD_DATA}" "${ROUTER_STEPS}" "${ROUTER_BATCH}" "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" --lr "${WORLD_LR}" --grad-accum "${ROUTER_GRAD_ACCUM}" --output "${WORLD_MODEL}" "${RESUME_ARGS[@]}"
  fi
else
  echo "== Stage 4/6: downstream orchestrator/planner skipped (ROUTER_STEPS=0; strict LeJEPA world model remains auxiliary-free) =="
fi

# --- Stage 5: Code decoder ---
echo "== Stage 5/6: Code decoder (--train-decoder --decoder-kind code) =="
if [[ ! -f "${CODE_DATA}" ]]; then
  echo "  Skipping (CODE_DATA not found: ${CODE_DATA})"
else
  if is_true "${TOFY_RESUME}" && [[ -f "${CODE_DECODER_BASE_OUTPUT}" ]] && resume_stage_complete "${CODE_DECODER_BASE_OUTPUT}" "decoder_code" "${CODE_DECODER_STEPS}"; then
    echo "  Skipping code decoder; resume state already reached ${CODE_DECODER_STEPS} steps."
  else
    TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="decoder_code" cargo run --release -- --train-decoder "${WORLD_ENCODER_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${CODE_DATA}" "${CODE_DECODER_STEPS}" "${CODE_DECODER_BATCH}" "${CODE_DECODER_MAX_SEQ}" "${DIM}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" --decoder-kind code --decoder-max-vocab "${CODE_DECODER_MAX_VOCAB}" "${DECODER_VOCAB_ARGS[@]}" --decoder-output "${CODE_DECODER_BASE_OUTPUT}" --grad-accum "${CODE_DECODER_GRAD_ACCUM}" "${RESUME_ARGS[@]}"
  fi
  if [[ -s "${RUST_TASK_DATA}" && "${CODE_POLISH_STEPS}" -gt 0 ]]; then
    if is_true "${TOFY_RESUME}" && resume_stage_complete "${CODE_DECODER_POLISH_OUTPUT}" "decoder_code_polish" "${CODE_POLISH_STEPS}"; then
      echo "  Skipping code decoder polish; resume state already reached ${CODE_POLISH_STEPS} steps."
    else
      TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="decoder_code_polish" cargo run --release -- --train-decoder "${WORLD_ENCODER_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${RUST_TASK_DATA}" "${CODE_POLISH_STEPS}" "${CODE_DECODER_BATCH}" "${CODE_DECODER_MAX_SEQ}" "${DIM}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" --decoder-kind code --decoder-max-vocab "${CODE_DECODER_MAX_VOCAB}" "${DECODER_VOCAB_ARGS[@]}" --decoder-output "${CODE_DECODER_POLISH_OUTPUT}" --grad-accum "${CODE_DECODER_GRAD_ACCUM}" --lr "${CODE_POLISH_LR}" --init-decoder "${CODE_DECODER_BASE_OUTPUT}" "${RESUME_ARGS[@]}"
    fi
  fi
  CODE_DECODER_OUTPUT="${CODE_DECODER_BASE_OUTPUT}"
  if [[ ! -f "${CODE_DECODER_OUTPUT}" ]]; then
    echo "ERROR: code decoder checkpoint not found at '${CODE_DECODER_OUTPUT}'."
    exit 1
  fi
  echo "  Code decoder: ${CODE_DECODER_OUTPUT}"
fi

# --- Stage 6: Text decoder (chat) ---
echo "== Stage 6/6: Text decoder (--train-decoder --decoder-kind text) =="
if [[ ! -f "${TEXT_DATA}" ]]; then
  echo "  Skipping (TEXT_DATA not found: ${TEXT_DATA}). Set TEXT_DATA= to skip, or provide a path."
else
  if is_true "${TOFY_RESUME}" && [[ -f "${TEXT_DECODER_OUTPUT}" ]] && resume_stage_complete "${TEXT_DECODER_OUTPUT}" "decoder_text" "${TEXT_DECODER_STEPS}"; then
    echo "  Skipping text decoder; resume state already reached ${TEXT_DECODER_STEPS} steps."
  else
    TOFY_RUN_GROUP="${PIPELINE_RUN_ID}" TOFY_RUN_STAGE_NAME="decoder_text" cargo run --release -- --train-decoder "${WORLD_ENCODER_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${TEXT_DATA}" "${TEXT_DECODER_STEPS}" "${TEXT_DECODER_BATCH}" "${TEXT_DECODER_MAX_SEQ}" "${DIM}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" --decoder-kind text --decoder-max-vocab "${TEXT_DECODER_MAX_VOCAB}" --decoder-output "${TEXT_DECODER_OUTPUT}" --grad-accum "${TEXT_DECODER_GRAD_ACCUM}" "${RESUME_ARGS[@]}"
  fi
  if [[ ! -f "${TEXT_DECODER_OUTPUT}" ]]; then
    echo "ERROR: text decoder checkpoint not found at '${TEXT_DECODER_OUTPUT}'."
    exit 1
  fi
  echo "  Text decoder: ${TEXT_DECODER_OUTPUT}"
fi

echo "Pipeline complete. Serve with: cargo run --release -- --serve ${WORLD_ENCODER_MODEL} ${ENCODER_VOCAB} ${WORLD_MODEL} 0.0.0.0:8080 ${DIM} ${WORLD_MAX_SEQ} ${LAYERS} ${HEADS} ${BRIDGE_DIM} ${NUM_LATENT_TOKENS}"
