#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${TOFY_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE="${WORKSPACE:-/workspace}"
ENV_FILE="${TOFY_RUNPOD_ENV_FILE:-${WORKSPACE}/tofy-runpod.env}"
HF_DATASET="${TOFY_CACHE_HF_DATASET:-Grayza/80gb-profile-go-cache}"
HF_MAX_WORKERS="${TOFY_CACHE_HF_MAX_WORKERS:-16}"
ZSTD_THREADS="${TOFY_ZSTD_THREADS:-$(nproc 2>/dev/null || echo 1)}"
RUNPOD_CACHE_DIR="${TOFY_RUNPOD_CACHE_DIR:-${WORKSPACE}/tofy-cache}"
ZSTD_WRITE_BLOCK_SIZE="${TOFY_ZSTD_WRITE_BLOCK_SIZE:-64M}"

source "$HOME/.cargo/env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

cd "$REPO_DIR"

if ! command -v hf >/dev/null 2>&1; then
  python3 -m pip install -U "huggingface_hub[cli]" --break-system-packages
fi
if ! command -v pzstd >/dev/null 2>&1 && ! command -v zstd >/dev/null 2>&1; then
  apt-get update
  apt-get install -y zstd
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "warning: HF_TOKEN is not set; private or rate-limited cache downloads may fail" >&2
else
  export HF_TOKEN
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ -d "${REPO_DIR}/.cache/huggingface/download" ]]; then
  echo "Clearing stale Hugging Face local-dir download locks..."
  find "${REPO_DIR}/.cache/huggingface/download" -type f -name '*.lock' -delete
fi

decompress_zstd_file() {
  local compressed="$1"
  local output="$2"

  if command -v pzstd >/dev/null 2>&1; then
    pzstd -dc -p "$ZSTD_THREADS" -f "$compressed" \
      | dd of="$output" bs="$ZSTD_WRITE_BLOCK_SIZE" iflag=fullblock status=progress
  else
    zstd -dc -T0 -f "$compressed" \
      | dd of="$output" bs="$ZSTD_WRITE_BLOCK_SIZE" iflag=fullblock status=progress
  fi
}

if ! hf download "$HF_DATASET" \
  --repo-type dataset \
  --include "data/**" \
  --include "eval/**" \
  --include "local_models/vocabs/**" \
  --local-dir "$REPO_DIR" \
  --max-workers "$HF_MAX_WORKERS"; then
  echo "error: failed to download prepared cache tree from Hugging Face dataset ${HF_DATASET}" >&2
  echo "Check that HF_TOKEN is exported and the dataset was uploaded with local prepare cache." >&2
  exit 1
fi

token_cache_paths=(
  "data/cache/encoder.tokens.bin"
  "data/cache/world.tokens.bin"
  "data/cache/code_decoder.tokens.bin"
  "data/cache/code_decoder_dual.tokens.bin"
  "data/cache/go_feedback/encoder.tokens.bin"
  "data/cache/go_feedback/world.tokens.bin"
  "data/cache/go_feedback/code_decoder.tokens.bin"
  "data/cache/go_feedback/code_decoder_dual.tokens.bin"
)

required_paths=(
  "local_models/vocabs"
  "data/encoder_mix.txt"
  "data/world_mix_pairs.txt"
  "data/code_poc_mix.txt"
  "data/code_poc_go_mix.txt"
  "data/go_repair_pairs.txt"
  "data/cache/encoder_vocab.manifest.json"
  "data/cache/code_decoder_vocab.manifest.json"
  "data/cache/encoder_tokens.manifest.json"
  "data/cache/world_tokens.manifest.json"
  "data/cache/code_decoder_tokens.manifest.json"
  "data/cache/code_decoder_dual_tokens.manifest.json"
  "data/cache/go_feedback/encoder_vocab.manifest.json"
  "data/cache/go_feedback/code_decoder_vocab.manifest.json"
  "data/cache/go_feedback/encoder_tokens.manifest.json"
  "data/cache/go_feedback/world_tokens.manifest.json"
  "data/cache/go_feedback/code_decoder_tokens.manifest.json"
  "data/cache/go_feedback/code_decoder_dual_tokens.manifest.json"
  "${token_cache_paths[@]}"
)

token_cache_files=()
repo_cache_files=()
for path in "${token_cache_paths[@]}"; do
  compressed="${path}.zst"
  if [[ ! -e "$path" && -f "$compressed" ]]; then
    token_cache_files+=("$compressed")
  fi
done
mapfile -d '' repo_cache_files < <(
  find data eval local_models/vocabs -type f -name '*.zst' ! -path 'data/cache/*.tokens.bin.zst' ! -path 'data/cache/go_feedback/*.tokens.bin.zst' -print0 2>/dev/null | sort -z
)

if (( ${#token_cache_files[@]} > 0 )); then
  echo "Decompressing ${#token_cache_files[@]} required token cache files..."
  echo "Writing decompressed cache files to local scratch: ${RUNPOD_CACHE_DIR}"
  mkdir -p "$RUNPOD_CACHE_DIR"
  for compressed in "${token_cache_files[@]}"; do
    output="${compressed%.zst}"
    scratch_output="${RUNPOD_CACHE_DIR}/${output}"
    echo "Decompressing ${compressed} -> ${scratch_output}"
    mkdir -p "$(dirname "$scratch_output")"
    rm -f "$scratch_output" "$output"
    decompress_zstd_file "$compressed" "$scratch_output"
    ln -s "$scratch_output" "$output"
    rm -f "$compressed"
  done
fi
if (( ${#repo_cache_files[@]} > 0 )); then
  echo "Decompressing ${#repo_cache_files[@]} prepared repository files..."
  for compressed in "${repo_cache_files[@]}"; do
    output="${compressed%.zst}"
    if [[ ! -f "$compressed" ]]; then
      continue
    fi
    if [[ -e "$output" ]]; then
      rm -f "$compressed"
      continue
    fi
    echo "Decompressing ${compressed} -> ${output}"
    rm -f "$output"
    decompress_zstd_file "$compressed" "$output"
    rm -f "$compressed"
  done
fi

echo "Prepared cache:"
du -sh data/cache eval local_models 2>/dev/null || true
ls -lh data/cache eval local_models/vocabs 2>/dev/null || true

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "error: prepared cache tree is missing required path: $path" >&2
    echo "Rebuild and upload locally with: cargo run --release -- prepare cache 80gb --auto-hf-upload --hf-dataset <org/dataset>" >&2
    exit 1
  fi
done

cargo build --release
