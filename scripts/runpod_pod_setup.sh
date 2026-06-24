#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ENV_FILE="${TOFY_RUNPOD_ENV_FILE:-${WORKSPACE}/tofy-runpod.env}"

apt-get update
apt-get install -y \
  git curl ca-certificates build-essential pkg-config libssl-dev \
  openssh-client tmux htop nvtop pciutils jq rsync \
  python3-pip golang-go zstd

if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
source "$HOME/.cargo/env"

if ! command -v hf >/dev/null 2>&1; then
  python3 -m pip install -U "huggingface_hub[cli]" --break-system-packages
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  read -rsp "Hugging Face token for cache download (blank to skip): " HF_TOKEN
  echo
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  printf 'export HF_TOKEN=%q\n' "$HF_TOKEN" > "$ENV_FILE"
fi
if [[ -s "$ENV_FILE" ]]; then
  chmod 600 "$ENV_FILE"
fi

nvidia-smi
nvcc --version || true
go version
hf --help >/dev/null
