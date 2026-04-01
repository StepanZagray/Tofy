#!/usr/bin/env bash
set -euo pipefail

# Full training pipeline (encoder -> pure world model -> code decoder -> text decoder).
# Override via env: ENCODER_DATA, WORLD_DATA, CODE_DATA, TEXT_DATA, WIKI_DATA, *_STEPS, BATCH, DIM, etc.
#
# 1) LeJEPA encoder (--latent)
# 2) Pure latent world model (--train-world) using frozen encoder artifacts
# 3) Code decoder (--train-decoder --decoder-kind code) on CODE_DATA
# 4) Text decoder (--train-decoder --decoder-kind text) on TEXT_DATA

WORLD_DATA="${WORLD_DATA:-data/ultrachat_pairs.txt}"
CODE_DATA="${CODE_DATA:-data/multilang_pairs.txt}"
TEXT_DATA="${TEXT_DATA:-data/ultrachat_pairs.txt}"
WIKI_DATA="${WIKI_DATA:-data/cached_wikimedia_wikipedia_1.txt}"
ENCODER_DATA="${ENCODER_DATA:-data/encoder_mix.txt}"

LATENT_STEPS="${LATENT_STEPS:-25000}"
WORLD_STEPS="${WORLD_STEPS:-60000}"
CODE_DECODER_STEPS="${CODE_DECODER_STEPS:-40000}"
TEXT_DECODER_STEPS="${TEXT_DECODER_STEPS:-40000}"

BATCH="${BATCH:-32}"
DECODER_BATCH="${DECODER_BATCH:-8}"
DIM="${DIM:-768}"
LATENT_MAX_SEQ="${LATENT_MAX_SEQ:-128}"
WORLD_MAX_SEQ="${WORLD_MAX_SEQ:-128}"
DECODER_MAX_SEQ="${DECODER_MAX_SEQ:-128}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-8}"
MAX_VOCAB="${MAX_VOCAB:-8000}"
BRIDGE_DIM="${BRIDGE_DIM:-256}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
WIKI_MAX_FILES="${WIKI_MAX_FILES:-1}"
WORLD_LR="${WORLD_LR:-2e-4}"
WORLD_LAMBDA="${WORLD_LAMBDA:-0.2}"
ENCODER_VOCAB="${ENCODER_VOCAB:-local_models/vocabs/vocab_encoder.txt}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

# --- Checks ---
if [[ ! -f "${WORLD_DATA}" ]]; then
  echo "ERROR: WORLD_DATA not found at '${WORLD_DATA}'"
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
"${PYTHON_BIN}" scripts/prepare_encoder_corpus.py --output "${ENCODER_DATA}" "${WORLD_DATA}" "${WIKI_DATA}" "${CODE_DATA}"

# --- Stage 1: LeJEPA encoder ---
echo "== Stage 1/4: LeJEPA encoder (--latent) =="
cargo run --release -- --latent "${ENCODER_DATA}" "${LATENT_STEPS}" "${BATCH}" "${DIM}" "${LATENT_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${MAX_VOCAB}"

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
echo "== Stage 2/4: Planner/world model (--train-world) =="
cargo run --release -- --train-world "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_DATA}" "${WORLD_STEPS}" "${BATCH}" "${DIM}" "${WORLD_MAX_SEQ}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" --lambda "${WORLD_LAMBDA}" --lr "${WORLD_LR}"

WORLD_MODEL="${WORLD_MODEL:-$(ls -1t local_models/model_world_*.safetensors 2>/dev/null | awk '{print; exit}')}"
if [[ -z "${WORLD_MODEL}" ]]; then
  echo "ERROR: No local_models/model_world_*.safetensors found after world training."
  exit 1
fi
echo "  Using: ${WORLD_MODEL}"

# --- Stage 3: Code decoder ---
echo "== Stage 3/4: Code decoder (--train-decoder --decoder-kind code) =="
if [[ ! -f "${CODE_DATA}" ]]; then
  echo "  Skipping (CODE_DATA not found: ${CODE_DATA})"
else
  cargo run --release -- --train-decoder "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${CODE_DATA}" "${CODE_DECODER_STEPS}" "${DECODER_BATCH}" "${DECODER_MAX_SEQ}" "${DIM}" "${LAYERS}" "${HEADS}" "${BRIDGE_DIM}" "${NUM_LATENT_TOKENS}" --decoder-kind code
  echo "  Code decoder: local_models/code_decoder_*.safetensors"
fi

# --- Stage 4: Text decoder (chat) ---
echo "== Stage 4/4: Text decoder (--train-decoder --decoder-kind text) =="
if [[ ! -f "${TEXT_DATA}" ]]; then
  echo "  Skipping (TEXT_DATA not found: ${TEXT_DATA}). Set TEXT_DATA= to skip, or provide a path."
else
  cargo run --release -- --train-decoder "${LATENT_MODEL}" "${ENCODER_VOCAB}" "${WORLD_MODEL}" "${TEXT_DATA}" "${TEXT_DECODER_STEPS}" "${DECODER_BATCH}" "${DECODER_MAX_SEQ}" --decoder-kind text --decoder-output local_models/text_decoder_90M.safetensors
  echo "  Text decoder: local_models/text_decoder_90M.safetensors"
fi

echo "Pipeline complete. Serve with: cargo run --release -- --serve ${LATENT_MODEL} ${ENCODER_VOCAB} ${WORLD_MODEL} 0.0.0.0:8080 ${DIM} ${WORLD_MAX_SEQ} ${LAYERS} ${HEADS} ${BRIDGE_DIM} ${NUM_LATENT_TOKENS}"
