#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ENV_FILE="${TOFY_RUNPOD_ENV_FILE:-${WORKSPACE}/tofy-runpod.env}"
STOP_WRAPPER="${TOFY_RUNPOD_STOP_WRAPPER:-${WORKSPACE}/run-tofy-and-stop.sh}"

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

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  read -rsp "RunPod API key for auto-stop: " RUNPOD_API_KEY
  echo
fi
if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
  read -rp "RunPod pod id for auto-stop: " RUNPOD_POD_ID
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  read -rsp "Hugging Face token for cache download (blank to skip): " HF_TOKEN
  echo
fi

{
  if [[ -n "${RUNPOD_API_KEY:-}" && -n "${RUNPOD_POD_ID:-}" ]]; then
    printf 'export RUNPOD_API_KEY=%q\n' "$RUNPOD_API_KEY"
    printf 'export RUNPOD_POD_ID=%q\n' "$RUNPOD_POD_ID"
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf 'export HF_TOKEN=%q\n' "$HF_TOKEN"
  fi
} > "$ENV_FILE"
if [[ -s "$ENV_FILE" ]]; then
  chmod 600 "$ENV_FILE"
fi

cat > "$STOP_WRAPPER" <<'EOF'
#!/usr/bin/env bash
set -o pipefail

if [ -f /workspace/tofy-runpod.env ]; then
  . /workspace/tofy-runpod.env
fi

stop_pod() {
  echo "Stopping RunPod pod..."
  echo "RUNPOD_POD_ID=${RUNPOD_POD_ID:-}"
  echo "RUNPOD_API_KEY length=${#RUNPOD_API_KEY}"
  if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    curl -fsS --request POST \
      --url "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
      --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
      -o /workspace/runpod-stop-response.json \
      -w "HTTP %{http_code}\n" || true
  else
    echo "RUNPOD_POD_ID or RUNPOD_API_KEY missing; cannot auto-stop pod"
  fi
}

trap stop_pod EXIT
"$@"
EOF
chmod +x "$STOP_WRAPPER"

nvidia-smi
nvcc --version || true
go version
hf --help >/dev/null
