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

if [[ ! -d local_models/vocabs ]]; then
  echo "error: local_models/vocabs is missing after cache extraction" >&2
  exit 1
fi

cargo build --release
