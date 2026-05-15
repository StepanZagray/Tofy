# RunPod Curl Setup

This runbook creates a RunPod pod through the REST API with:

- template `obgryfbuad`
- one `NVIDIA L40S`
- CUDA host filter `13.0`
- no fixed datacenter
- regular pod volume storage mounted at `/workspace`

This is the recommended flow for the 48 GB profile because availability matters
more than storage portability. A network volume is tied to one datacenter, which
can make it hard to find an L40S with the needed CUDA host version in the same
place.

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

## Create The Pod

Create the pod with availability-based datacenter selection:

```bash
cat > /tmp/runpod-tofy-l40s-cuda13.json <<'EOF'
{
  "name": "tofy-l40s-cuda13-full-run",
  "cloudType": "SECURE",
  "computeType": "GPU",
  "templateId": "obgryfbuad",
  "gpuTypeIds": ["NVIDIA L40S"],
  "gpuTypePriority": "availability",
  "gpuCount": 1,
  "allowedCudaVersions": ["13.0"],
  "dataCenterPriority": "availability",
  "containerDiskInGb": 50,
  "volumeInGb": 200,
  "volumeMountPath": "/workspace"
}
EOF

curl -sS -X POST "https://rest.runpod.io/v1/pods" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/runpod-tofy-l40s-cuda13.json \
  | tee /tmp/runpod-pod.json \
  | jq 'if type=="array" then . else {id, name, desiredStatus, imageName, costPerHr, machine, volumeInGb, volumeMountPath} end'
```

Template for A100 SXM:

```bash
cat > /tmp/runpod-tofy-a100sxm-cuda13.json <<'EOF'
{
  "name": "tofy-a100sxm-cuda13-full-run",
  "cloudType": "SECURE",
  "computeType": "GPU",
  "templateId": "obgryfbuad",
  "gpuTypeIds": ["NVIDIA A100-SXM4-80GB"],
  "gpuTypePriority": "availability",
  "gpuCount": 1,
  "allowedCudaVersions": ["13.0"],
  "dataCenterPriority": "availability",
  "containerDiskInGb": 50,
  "volumeInGb": 200,
  "volumeMountPath": "/workspace"
}
EOF
```

If the response is an array, it is an API validation or availability error. Read
the error text and try again later, loosen CUDA to `["12.9", "12.8"]`, or switch
GPU type.

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

Create a deploy key, print the public key, then add it to GitHub. Keep the
private key under `/root/.ssh`, not `/workspace/.ssh`: some RunPod `/workspace`
mounts report permissive `0666/0777` permissions even after `chmod`, and OpenSSH
will refuse to use a private key with those permissions.

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh

ssh-keygen -t ed25519 -C "runpod-tofy-pod" -f ~/.ssh/runpod_tofy -N ""

cat > ~/.ssh/config <<'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/runpod_tofy
    IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/runpod_tofy
chmod 644 ~/.ssh/runpod_tofy.pub
chmod 600 ~/.ssh/config
ssh-keyscan github.com >> ~/.ssh/known_hosts
chmod 644 ~/.ssh/known_hosts

cat ~/.ssh/runpod_tofy.pub
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

If SSH fails with this warning:

```text
WARNING: UNPROTECTED PRIVATE KEY FILE!
Permissions 0666 for '/workspace/.ssh/runpod_tofy' are too open.
Load key "/workspace/.ssh/runpod_tofy": bad permissions
```

copy the key into `/root/.ssh` and point SSH there:

```bash
cp /workspace/.ssh/runpod_tofy ~/.ssh/runpod_tofy
cp /workspace/.ssh/runpod_tofy.pub ~/.ssh/runpod_tofy.pub

chmod 700 ~/.ssh
chmod 600 ~/.ssh/runpod_tofy
chmod 644 ~/.ssh/runpod_tofy.pub

cat > ~/.ssh/config <<'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/runpod_tofy
    IdentitiesOnly yes
EOF

chmod 600 ~/.ssh/config
ssh -T git@github.com
```

If GitHub says the key is already in use, the public key is already registered
somewhere in GitHub. Either reuse that key after fixing local permissions, or
generate a new key with a different filename and add the new public key as the
repo deploy key.

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
  ./target/release/jepa_ai train 48gb --stream \
  2>&1 | tee /workspace/tofy-train-48gb.log
```

`--stream` skips the expensive Stage 1 vocab/token-cache prebuild and trains
from raw streaming tokenization. This is the preferred rented-GPU launch mode:
the GPU starts training sooner, instead of waiting for hours of CPU-only cache
construction. On a fresh regular pod volume, Stage 1 still bootstraps the
required source data in the pod workspace before training.

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

## Storage Notes

Regular pod volume storage:

- mounted at `/workspace`
- persists across stop/start of the same pod
- is deleted when the pod is terminated
- lets RunPod place the pod wherever an eligible GPU is available

Network volumes:

- can survive pod termination
- can be reused by later pods
- are tied to one datacenter
- are not the default here because they reduce the chance of finding an L40S
  with CUDA `13.0`
