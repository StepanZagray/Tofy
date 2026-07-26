#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${TOFY_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BINARY="${TOFY_BINARY:-./target/release/tofy}"
RUN_DIR="${1:-}"
REPORT_PATH="${2:-}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "usage: $0 <runs/code_poc_<id>> [report.json]" >&2
  exit 2
fi

cd "${REPO_DIR}"
[[ -d "${RUN_DIR}" ]] || { echo "run directory not found: ${RUN_DIR}" >&2; exit 2; }
: "${TOFY_QWEN_DIR:?set TOFY_QWEN_DIR to the frozen Qwen3 base-model directory}"

META="${RUN_DIR}/meta.json"
[[ -f "${META}" ]] || { echo "pipeline metadata not found: ${META}" >&2; exit 2; }

meta_number() {
  sed -n "s/.*\"$1\": \([0-9][0-9]*\).*/\1/p" "${META}" | head -n1
}

meta_string() {
  sed -n "s/.*\"$1\": \"\([^\"]*\)\".*/\1/p" "${META}" | head -n1
}

BRIDGE_MODEL="${TOFY_FLOOR_BRIDGE_MODEL:-${RUN_DIR}/bridge/weights.best.safetensors}"
BRIDGE_REGIME="weights"
if [[ ! -f "${BRIDGE_MODEL}" ]]; then
  BRIDGE_MODEL="${RUN_DIR}/bridge/context.best.safetensors"
  BRIDGE_REGIME="context"
fi

ENCODER_MODEL="${RUN_DIR}/world/model.encoder.safetensors"
ENCODER_VOCAB="${RUN_DIR}/latent/model.vocab.txt"
WORLD_MODEL="${BRIDGE_MODEL%.safetensors}.world.safetensors"
[[ -f "${WORLD_MODEL}" ]] || WORLD_MODEL="${RUN_DIR}/world/model.safetensors"
EVAL_SUITE="${TOFY_FLOOR_EVAL_SUITE:-eval/veclab_eval.jsonl}"
REPORT_PATH="${REPORT_PATH:-${RUN_DIR}/eval/decoder_floor_manual.json}"

for required in "${BRIDGE_MODEL}" "${ENCODER_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${EVAL_SUITE}"; do
  [[ -f "${required}" ]] || { echo "required artifact not found: ${required}" >&2; exit 2; }
done

PROFILE="$(meta_string profile)"
case "${PROFILE}" in
  minimal) DEFAULT_BRIDGE_MAX_SEQ=256 ;;
  *) echo "unsupported or missing profile in ${META}: ${PROFILE}" >&2; exit 2 ;;
esac

export TOFY_ENCODER_DIM="${TOFY_ENCODER_DIM:-$(meta_number dim)}"
export TOFY_ENCODER_LAYERS="${TOFY_ENCODER_LAYERS:-$(meta_number layers)}"
export TOFY_ENCODER_HEADS="${TOFY_ENCODER_HEADS:-$(meta_number heads)}"
export TOFY_BRIDGE_DIM="${TOFY_BRIDGE_DIM:-$(meta_number bridge_dim)}"
export TOFY_NUM_LATENT_TOKENS="${TOFY_NUM_LATENT_TOKENS:-$(meta_number num_latent_tokens)}"
export TOFY_ADAPTER_OUTPUT_SLOTS="${TOFY_ADAPTER_OUTPUT_SLOTS:-${TOFY_NUM_LATENT_TOKENS}}"
export TOFY_BRIDGE_MAX_SEQ="${TOFY_BRIDGE_MAX_SEQ:-${DEFAULT_BRIDGE_MAX_SEQ}}"
export TOFY_EVAL_MODE=floor
export TOFY_BRIDGE_REGIME="${BRIDGE_REGIME}"
unset TOFY_STATIC_SOFT_PREFIX TOFY_QWEN_LORA_RANK

[[ -x "$BINARY" ]] || {
  echo "release binary not found; build it with: cargo build --release" >&2
  exit 2
}

echo "Running manual decoder-only floor for ${RUN_DIR}; training never invokes this control."
"$BINARY" --eval-bridge \
  "${TOFY_QWEN_DIR}" \
  "${BRIDGE_MODEL}" \
  "${ENCODER_MODEL}" \
  "${ENCODER_VOCAB}" \
  "${WORLD_MODEL}" \
  "${EVAL_SUITE}" \
  "${REPORT_PATH}"
