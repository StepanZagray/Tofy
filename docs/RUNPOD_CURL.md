# RunPod Curl Setup

This runbook creates a RunPod pod through the REST API with:

- template `obgryfbuad`
- one `NVIDIA L40S`
- CUDA host filter `13.0`
- a network volume mounted at `/workspace`
- SSH enabled by the template

Do not paste API keys or full RunPod API responses into public logs. RunPod pod
responses can include environment variables.

## Local Prerequisites

Install local helpers:

```bash
sudo pacman -S --needed curl jq openssh
```

Load the RunPod API key only into the current shell:

```bash
read -rsp "RunPod API key: " RUNPOD_API_KEY
export RUNPOD_API_KEY
echo
echo "key length: ${#RUNPOD_API_KEY}"
```

If the printed length is `0`, the key is not set.

## Pick A Datacenter

Network volumes are tied to one datacenter. The pod must be created in the same
datacenter as the network volume.

If `runpodctl` is installed:

```bash
runpodctl datacenter list
```

Set the datacenter ID. Replace `SE` with the exact ID you want to use:

```bash
export RUNPOD_DC_ID="SE"
```

## Create Network Volume

Create a persistent network volume:

```bash
curl -sS -X POST "https://rest.runpod.io/v1/networkvolumes" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  --data-binary @- <<EOF | tee /tmp/runpod-network-volume.json | jq '{id, name, dataCenterId, size, volumeType}'
{
  "dataCenterId": "${RUNPOD_DC_ID}",
  "name": "tofy-netvol",
  "size": 200
}
EOF
```

Save the volume ID:

```bash
export RUNPOD_NETWORK_VOLUME_ID="$(jq -r '.id' /tmp/runpod-network-volume.json)"
echo "$RUNPOD_NETWORK_VOLUME_ID"
```

## Create The Pod

Create the pod with the repo template, L40S only, and CUDA 13.0 host filter:

```bash
cat > /tmp/runpod-tofy-l40s-cuda13-netvol.json <<EOF
{
  "name": "tofy-l40s-cuda13-full-run",
  "cloudType": "SECURE",
  "computeType": "GPU",
  "templateId": "obgryfbuad",
  "gpuTypeIds": ["NVIDIA L40S"],
  "gpuCount": 1,
  "allowedCudaVersions": ["13.0"],
  "dataCenterIds": ["${RUNPOD_DC_ID}"],
  "containerDiskInGb": 50,
  "networkVolumeId": "${RUNPOD_NETWORK_VOLUME_ID}",
  "volumeMountPath": "/workspace"
}
EOF

curl -sS -X POST "https://rest.runpod.io/v1/pods" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/runpod-tofy-l40s-cuda13-netvol.json \
  | tee /tmp/runpod-pod.json \
  | jq 'if type=="array" then . else {id, name, desiredStatus, imageName, costPerHr, machine, networkVolumeId, volumeMountPath} end'
```

If the response is an array, it is an API validation or availability error. Read
the error text and adjust the datacenter or GPU choice.

Save the pod ID:

```bash
export RUNPOD_POD_ID="$(jq -r '.id' /tmp/runpod-pod.json)"
echo "$RUNPOD_POD_ID"
```

Get the exact SSH command from the RunPod Connect tab. The suffix after the pod
ID can differ between pods.

## Pod Setup

SSH into the pod and install build tools:

```bash
apt-get update
apt-get install -y \
  git curl ca-certificates build-essential pkg-config libssl-dev \
  openssh-client tmux htop nvtop pciutils jq

curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"

nvidia-smi
nvcc --version || true
```

## GitHub Deploy Key

Create a deploy key on the network volume so it persists across pod recreations:

```bash
mkdir -p /workspace/.ssh ~/.ssh
chmod 700 /workspace/.ssh ~/.ssh

ssh-keygen -t ed25519 -C "runpod-tofy-netvol" -f /workspace/.ssh/runpod_tofy -N ""

cat > ~/.ssh/config <<'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile /workspace/.ssh/runpod_tofy
    IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/config
ssh-keyscan github.com >> ~/.ssh/known_hosts

cat /workspace/.ssh/runpod_tofy.pub
```

Add the printed key to:

```text
GitHub repo -> Settings -> Deploy keys -> Add deploy key
```

Keep write access disabled unless you explicitly need to push from the pod.

Verify access:

```bash
ssh -T git@github.com
```

Expected result:

```text
Hi StepanZagray/Tofy! You've successfully authenticated, but GitHub does not provide shell access.
```

## Clone Or Update

```bash
cd /workspace

if [ ! -d Tofy/.git ]; then
  git clone git@github.com:StepanZagray/Tofy.git Tofy
fi

cd /workspace/Tofy
git fetch origin
git pull --ff-only
```

## Auto-Stop Wrapper

The wrapper stops the pod when the wrapped command exits. It prints only the API
key length, not the key.

```bash
cat > /workspace/run-tofy-and-stop.sh <<'EOF'
#!/usr/bin/env bash
set -o pipefail

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

chmod +x /workspace/run-tofy-and-stop.sh
```

## Full VRAM Test

Build first:

```bash
cd /workspace/Tofy
source "$HOME/.cargo/env"
cargo build --release
```

Run the full short VRAM probe in `tmux`:

```bash
tmux new -s tofy-full-vram
```

Inside tmux:

```bash
export RUST_BACKTRACE=1

/workspace/run-tofy-and-stop.sh \
  ./target/release/jepa_ai --max-vram-probe --profile 48gb --stage all --probe-dir /workspace/tofy-vram-probe-full \
  2>&1 | tee /workspace/tofy-vram-probe-full.log
```

Detach without stopping:

```text
Ctrl+b, then d
```

Check status:

```bash
tmux attach -t tofy-full-vram
tail -f /workspace/tofy-vram-probe-full.log
nvidia-smi
```

## Full Training

After the VRAM probe passes, start a full run:

```bash
cd /workspace/Tofy
source "$HOME/.cargo/env"
git fetch origin
git pull --ff-only
cargo build --release

tmux new -s tofy-train
```

Inside tmux:

```bash
export RUST_BACKTRACE=1
export TOFY_TRAIN_DTYPE=bf16

/workspace/run-tofy-and-stop.sh \
  ./target/release/jepa_ai train 48gb \
  2>&1 | tee /workspace/tofy-train-48gb.log
```

Detach:

```text
Ctrl+b, then d
```

Check status:

```bash
tmux attach -t tofy-train
tail -f /workspace/tofy-train-48gb.log
nvidia-smi
```

## Reuse The Network Volume

For a later pod, skip network volume creation and reuse:

```bash
export RUNPOD_NETWORK_VOLUME_ID="<existing-network-volume-id>"
```

Then create a new pod with the same `networkVolumeId` and `volumeMountPath`.
