#!/usr/bin/env bash
# One-time RunPod bootstrap for P2 readiness resume (A40/L40S).
# Run on the pod as root: bash scripts/runpod_readiness_setup.sh
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
PERSONAL="${WORKSPACE}/Personal"
TOFY="${PERSONAL}/Tofy"
CANDLE_GRAPH="${PERSONAL}/candle_graph"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  git curl ca-certificates build-essential pkg-config libssl-dev \
  openssh-client tmux htop rsync jq

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# shellcheck disable=SC1091
source "${HOME}/.cargo/env"
rustup default stable

mkdir -p "${HOME}/.ssh" "${PERSONAL}"
chmod 700 "${HOME}/.ssh"
if [[ ! -f "${HOME}/.ssh/id_ed25519" ]]; then
  ssh-keygen -t ed25519 -N '' -f "${HOME}/.ssh/id_ed25519" -C "runpod-tofy-readiness"
fi
cat >"${HOME}/.ssh/config" <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
  StrictHostKeyChecking accept-new
EOF
chmod 600 "${HOME}/.ssh/config"

echo "=== GPU ==="
nvidia-smi

echo "=== Deploy key (add to StepanZagray/Tofy + candle_graph repos) ==="
cat "${HOME}/.ssh/id_ed25519.pub"

if [[ ! -d "${TOFY}/.git" ]]; then
  echo "Clone Tofy + candle_graph after activating the deploy key above:"
  echo "  git clone git@github.com:StepanZagray/candle_graph.git ${CANDLE_GRAPH}"
  echo "  git clone git@github.com:StepanZagray/Tofy.git ${TOFY}"
fi

cat <<EOF

=== After clone or rsync ===
cd ${TOFY}
export P2_PHYSICAL_BATCH=512 P2_GRAD_ACCUM=1 P2_DEVICE=cuda
cargo build --release --features cudnn
bash scripts/p2_readiness_train.sh run

=== tmux ===
tmux new -s p2-readiness 'cd ${TOFY} && export P2_PHYSICAL_BATCH=512 P2_GRAD_ACCUM=1 && bash scripts/p2_readiness_train.sh run'
EOF
