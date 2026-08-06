#!/usr/bin/env bash
# Source this file before serving or evaluating a trained run.
# Usage: source scripts/tofy_pi_runtime_env.sh [runs/code_poc_<id>|latest] [8gb|48gb|80gb]

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source this script: source scripts/tofy_pi_runtime_env.sh [run_dir|latest] [profile]" >&2
  exit 2
fi

_tofy_run_selector="${1:-latest}"
_tofy_profile_arg="${2:-}"

if [[ "${_tofy_run_selector}" == "latest" ]]; then
  _tofy_run_dir="$(ls -td runs/code_poc_* 2>/dev/null | head -n1)"
else
  _tofy_run_dir="${_tofy_run_selector}"
fi

if [[ -z "${_tofy_run_dir}" || ! -d "${_tofy_run_dir}" ]]; then
  echo "Tofy run directory not found: ${_tofy_run_selector}" >&2
  return 2
fi

_tofy_meta="${_tofy_run_dir}/meta.json"
_tofy_meta_value() {
  local key="$1"
  if [[ -f "${_tofy_meta}" ]]; then
    sed -n "s/.*\"${key}\": \"\\([^\"]*\\)\".*/\\1/p" "${_tofy_meta}" | head -n1
  fi
}

_tofy_profile="${_tofy_profile_arg}"
if [[ -z "${_tofy_profile}" ]]; then
  _tofy_profile="$(_tofy_meta_value profile)"
fi
_tofy_profile="${_tofy_profile:-80gb}"

case "${_tofy_profile}" in
  8gb)
    export TOFY_PROFILE_DIM="${TOFY_PROFILE_DIM:-640}"
    export TOFY_PROFILE_MAX_SEQ="${TOFY_PROFILE_MAX_SEQ:-192}"
    export TOFY_PROFILE_LAYERS="${TOFY_PROFILE_LAYERS:-7}"
    export TOFY_PROFILE_HEADS="${TOFY_PROFILE_HEADS:-8}"
    export TOFY_PROFILE_BRIDGE_DIM="${TOFY_PROFILE_BRIDGE_DIM:-640}"
    export TOFY_PROFILE_CONTEXT_SLOTS="${TOFY_PROFILE_CONTEXT_SLOTS:-64}"
    export TOFY_ENCODER_CONTEXT_SEGMENTS="${TOFY_ENCODER_CONTEXT_SEGMENTS:-4}"
    export TOFY_WORLD_CONTEXT_SEGMENTS="${TOFY_WORLD_CONTEXT_SEGMENTS:-4}"
    export TOFY_CONTEXT_HYBRID_EXACT_TAIL="${TOFY_CONTEXT_HYBRID_EXACT_TAIL:-256}"
    export TOFY_CONTEXT_RETRIEVAL_SLOTS="${TOFY_CONTEXT_RETRIEVAL_SLOTS:-8}"
    export TOFY_CONTEXT_EXACT_OLD_TOKENS="${TOFY_CONTEXT_EXACT_OLD_TOKENS:-16}"
    export JEPA_CANDLE_DECODER_CTX="${JEPA_CANDLE_DECODER_CTX:-768}"
    export TOFY_DECODER_LOCAL_WINDOW="${TOFY_DECODER_LOCAL_WINDOW:-192}"
    ;;
  48gb)
    export TOFY_PROFILE_DIM="${TOFY_PROFILE_DIM:-768}"
    export TOFY_PROFILE_MAX_SEQ="${TOFY_PROFILE_MAX_SEQ:-256}"
    export TOFY_PROFILE_LAYERS="${TOFY_PROFILE_LAYERS:-12}"
    export TOFY_PROFILE_HEADS="${TOFY_PROFILE_HEADS:-16}"
    export TOFY_PROFILE_BRIDGE_DIM="${TOFY_PROFILE_BRIDGE_DIM:-768}"
    export TOFY_PROFILE_CONTEXT_SLOTS="${TOFY_PROFILE_CONTEXT_SLOTS:-96}"
    export TOFY_ENCODER_CONTEXT_SEGMENTS="${TOFY_ENCODER_CONTEXT_SEGMENTS:-6}"
    export TOFY_WORLD_CONTEXT_SEGMENTS="${TOFY_WORLD_CONTEXT_SEGMENTS:-6}"
    export TOFY_CONTEXT_HYBRID_EXACT_TAIL="${TOFY_CONTEXT_HYBRID_EXACT_TAIL:-320}"
    export TOFY_CONTEXT_RETRIEVAL_SLOTS="${TOFY_CONTEXT_RETRIEVAL_SLOTS:-12}"
    export TOFY_CONTEXT_EXACT_OLD_TOKENS="${TOFY_CONTEXT_EXACT_OLD_TOKENS:-24}"
    export JEPA_CANDLE_DECODER_CTX="${JEPA_CANDLE_DECODER_CTX:-1024}"
    export TOFY_DECODER_LOCAL_WINDOW="${TOFY_DECODER_LOCAL_WINDOW:-256}"
    ;;
  80gb)
    export TOFY_PROFILE_DIM="${TOFY_PROFILE_DIM:-1024}"
    export TOFY_PROFILE_MAX_SEQ="${TOFY_PROFILE_MAX_SEQ:-320}"
    export TOFY_PROFILE_LAYERS="${TOFY_PROFILE_LAYERS:-16}"
    export TOFY_PROFILE_HEADS="${TOFY_PROFILE_HEADS:-16}"
    export TOFY_PROFILE_BRIDGE_DIM="${TOFY_PROFILE_BRIDGE_DIM:-1024}"
    export TOFY_PROFILE_CONTEXT_SLOTS="${TOFY_PROFILE_CONTEXT_SLOTS:-128}"
    export TOFY_ENCODER_CONTEXT_SEGMENTS="${TOFY_ENCODER_CONTEXT_SEGMENTS:-8}"
    export TOFY_WORLD_CONTEXT_SEGMENTS="${TOFY_WORLD_CONTEXT_SEGMENTS:-8}"
    export TOFY_CONTEXT_HYBRID_EXACT_TAIL="${TOFY_CONTEXT_HYBRID_EXACT_TAIL:-384}"
    export TOFY_CONTEXT_RETRIEVAL_SLOTS="${TOFY_CONTEXT_RETRIEVAL_SLOTS:-16}"
    export TOFY_CONTEXT_EXACT_OLD_TOKENS="${TOFY_CONTEXT_EXACT_OLD_TOKENS:-32}"
    export JEPA_CANDLE_DECODER_CTX="${JEPA_CANDLE_DECODER_CTX:-1280}"
    export TOFY_DECODER_LOCAL_WINDOW="${TOFY_DECODER_LOCAL_WINDOW:-256}"
    ;;
  *)
    echo "Unsupported Tofy profile: ${_tofy_profile}" >&2
    return 2
    ;;
esac

_tofy_decoder="$(_tofy_meta_value code_decoder_model)"
if [[ -z "${_tofy_decoder}" || ! -f "${_tofy_decoder}" ]]; then
  if [[ -f "${_tofy_run_dir}/decoder_code_go_feedback/model.safetensors" ]]; then
    _tofy_decoder="${_tofy_run_dir}/decoder_code_go_feedback/model.safetensors"
  else
    _tofy_decoder="${_tofy_run_dir}/decoder_code/model.safetensors"
  fi
fi
_tofy_decoder_vocab="${_tofy_decoder%.safetensors}.vocab.txt"
if [[ ! -f "${_tofy_decoder_vocab}" ]]; then
  _tofy_decoder_vocab="${_tofy_run_dir}/decoder_code/model.vocab.txt"
fi

export TOFY_RUN_DIR="${_tofy_run_dir}"
export TOFY_RUNTIME_DTYPE="${TOFY_RUNTIME_DTYPE:-bf16}"
export TOFY_TRAIN_DTYPE="${TOFY_TRAIN_DTYPE:-bf16}"
export TOFY_WORLD_ENCODER_MODEL="${TOFY_WORLD_ENCODER_MODEL:-${_tofy_run_dir}/world/model.encoder.safetensors}"
export TOFY_ENCODER_VOCAB="${TOFY_ENCODER_VOCAB:-${_tofy_run_dir}/latent/model.vocab.txt}"
export TOFY_WORLD_MODEL="${TOFY_WORLD_MODEL:-${_tofy_run_dir}/world/model.safetensors}"
export TOFY_HIGH_WORLD_MODEL="${TOFY_HIGH_WORLD_MODEL:-${_tofy_run_dir}/high_world/model.safetensors}"

export JEPA_USE_CANDLE_DECODER="${JEPA_USE_CANDLE_DECODER:-1}"
export JEPA_CANDLE_DECODER="${JEPA_CANDLE_DECODER:-${_tofy_decoder}}"
export JEPA_CANDLE_DECODER_VOCAB="${JEPA_CANDLE_DECODER_VOCAB:-${_tofy_decoder_vocab}}"
export JEPA_DECODER_TEMP="${JEPA_DECODER_TEMP:-0}"
export TOFY_DECODER_TREECODER="${TOFY_DECODER_TREECODER:-0}"

export TOFY_DECODER_RLM="${TOFY_DECODER_RLM:-1}"
export TOFY_LATENT_REASONING="${TOFY_LATENT_REASONING:-1}"
export TOFY_WORLD_ROLLOUT_STEPS="${TOFY_WORLD_ROLLOUT_STEPS:-2}"
export TOFY_RECURSIVE_CONTEXT_COMPRESSION="${TOFY_RECURSIVE_CONTEXT_COMPRESSION:-0}"
export TOFY_CONTEXT_HYBRID_MEMORY="${TOFY_CONTEXT_HYBRID_MEMORY:-1}"
export TOFY_ENCODER_RECENT_FULL_SEGMENTS="${TOFY_ENCODER_RECENT_FULL_SEGMENTS:-1}"
export TOFY_WORLD_RECENT_FULL_SEGMENTS="${TOFY_WORLD_RECENT_FULL_SEGMENTS:-1}"
export TOFY_CONTEXT_HYBRID_BLOCK_SIZE="${TOFY_CONTEXT_HYBRID_BLOCK_SIZE:-32}"
export TOFY_DECODER_CSA_COMPRESS_RATE="${TOFY_DECODER_CSA_COMPRESS_RATE:-8}"
export TOFY_DECODER_HCA_COMPRESS_RATE="${TOFY_DECODER_HCA_COMPRESS_RATE:-128}"
export TOFY_DECODER_ANCHOR_PERIOD="${TOFY_DECODER_ANCHOR_PERIOD:-3}"
export TOFY_DECODER_CSA_TOPK="${TOFY_DECODER_CSA_TOPK:-16}"
export TOFY_HWM_MACRO_MIN_LEN="${TOFY_HWM_MACRO_MIN_LEN:-2}"
export TOFY_HWM_MACRO_MAX_LEN="${TOFY_HWM_MACRO_MAX_LEN:-4}"

echo "Tofy Pi runtime env loaded:"
echo "  run=${TOFY_RUN_DIR}"
echo "  profile=${_tofy_profile}"
echo "  decoder=${JEPA_CANDLE_DECODER}"
echo "  vocab=${JEPA_CANDLE_DECODER_VOCAB}"
