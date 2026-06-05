#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${TOFY_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE="${WORKSPACE:-/workspace}"
ENV_FILE="${TOFY_RUNPOD_ENV_FILE:-${WORKSPACE}/tofy-runpod.env}"
HF_DATASET="${TOFY_CACHE_HF_DATASET:-Grayza/80gb-profile-go-cache}"
HF_MAX_WORKERS="${TOFY_CACHE_HF_MAX_WORKERS:-16}"
ZSTD_THREADS="${TOFY_ZSTD_THREADS:-$(nproc 2>/dev/null || echo 1)}"

source "$HOME/.cargo/env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

cd "$REPO_DIR"
if [[ "${SKIP_GIT_PULL:-0}" != "1" && -d .git ]]; then
  git fetch origin
  git pull --ff-only
fi

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

mapfile -d '' compressed_files < <(
  find data eval local_models/vocabs -type f -name '*.zst' -print0 2>/dev/null | sort -z
)
if (( ${#compressed_files[@]} > 0 )); then
  echo "Decompressing ${#compressed_files[@]} prepared cache files..."
  for compressed in "${compressed_files[@]}"; do
    output="${compressed%.zst}"
    echo "Decompressing ${compressed} -> ${output}"
    if command -v pzstd >/dev/null 2>&1; then
      pzstd -d -p "$ZSTD_THREADS" -f "$compressed" -o "$output"
    else
      zstd -d -T0 -f "$compressed" -o "$output"
    fi
    rm -f "$compressed"
  done
fi

echo "Prepared cache:"
du -sh data/cache eval local_models 2>/dev/null || true
ls -lh data/cache eval local_models/vocabs 2>/dev/null || true

required_paths=(
  "local_models/vocabs"
  "data/cache/encoder.tokens.bin"
  "data/cache/encoder_tokens.manifest.json"
  "data/cache/world.tokens.bin"
  "data/cache/world_tokens.manifest.json"
  "data/cache/code_decoder.tokens.bin"
  "data/cache/code_decoder_tokens.manifest.json"
  "data/cache/code_decoder_dual.tokens.bin"
  "data/cache/code_decoder_dual_tokens.manifest.json"
  "data/cache/go_feedback/code_decoder.tokens.bin"
  "data/cache/go_feedback/code_decoder_tokens.manifest.json"
  "data/cache/go_feedback/code_decoder_dual.tokens.bin"
  "data/cache/go_feedback/code_decoder_dual_tokens.manifest.json"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "error: prepared cache tree is missing required path: $path" >&2
    echo "Rebuild and upload locally with: cargo run --release -- prepare cache 80gb --auto-hf-upload --hf-dataset <org/dataset>" >&2
    exit 1
  fi
done

cargo build --release
