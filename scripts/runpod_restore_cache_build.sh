#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${TOFY_REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORKSPACE="${WORKSPACE:-/workspace}"
HF_DATASET="${TOFY_CACHE_HF_DATASET:-Grayza/80gb-profile-go-cache}"
CACHE_ARCHIVE="${TOFY_CACHE_ARCHIVE:-tofy-cache-80gb-a8e7916-1780391272.tar.zst}"
ZSTD_THREADS="${TOFY_ZSTD_THREADS:-$(nproc 2>/dev/null || echo 1)}"

source "$HOME/.cargo/env"

cd "$REPO_DIR"
if [[ "${SKIP_GIT_PULL:-0}" != "1" && -d .git ]]; then
  git fetch origin
  git pull --ff-only
fi

if ! command -v hf >/dev/null 2>&1; then
  python3 -m pip install -U "huggingface_hub[cli]" --break-system-packages
fi

hf download "$HF_DATASET" "$CACHE_ARCHIVE" \
  --repo-type dataset \
  --local-dir "$WORKSPACE"

if command -v pzstd >/dev/null 2>&1; then
  tar -I "pzstd -d -p ${ZSTD_THREADS}" --no-same-owner --no-same-permissions \
    -xf "${WORKSPACE}/${CACHE_ARCHIVE}" \
    -C "$REPO_DIR"
else
  tar --zstd --no-same-owner --no-same-permissions \
    -xf "${WORKSPACE}/${CACHE_ARCHIVE}" \
    -C "$REPO_DIR"
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
    echo "error: prepared cache archive is missing required path: $path" >&2
    echo "Rebuild and upload locally with: cargo run --release -- prepare cache 80gb --auto-hf-upload --hf-dataset <org/dataset>" >&2
    exit 1
  fi
done

cargo build --release
