# Training Pod Runbook

This is the low-back-and-forth path for launching a RunPod training pod,
restoring the prepared cache, probing VRAM, running the full pipeline, and
recovering artifacts.

Tofy training pods should use CUDA 13.0. Keep `allowedCudaVersions: ["13.0"]`
in raw RunPod payloads and keep `CUDA_VERSION=13.0` when using the helper
script. Lower CUDA hosts have caused build/runtime failures.

The normal 80 GB target is A100/H100/RTX PRO 6000-class hardware with
`./target/release/jepa_ai train 80gb`. Use `train 48gb` only for L40S/A40-class
pods.

Do not use `--build-conditioned-cache`, `--until decoder-cache`, or
`--from-conditioned-cache` for a normal full run. The default `train 80gb` path
streams world conditioning and avoids the hundreds-of-GiB conditioned-slot
cache.

## 1. Local Setup

Paste this locally once:

```bash
sudo pacman -S --needed curl jq openssh rsync

read -rsp "RunPod API key: " RUNPOD_API_KEY
export RUNPOD_API_KEY
echo
echo "RUNPOD_API_KEY length: ${#RUNPOD_API_KEY}"
```

If the key length is `0`, stop and set the key again. Do not paste API keys or
full RunPod responses into public logs.

## 2. Create Pod

Pick exactly one option.

### Option A: Regular Pod Volume

Best when GPU availability matters. This creates a regular pod volume mounted
at `/workspace`.

```bash
export RUNPOD_POD_NAME="tofy-train-80gb"
export RUNPOD_GPU_TYPE="NVIDIA A100-SXM4-80GB"
export RUNPOD_TEMPLATE_ID="obgryfbuad"
export RUNPOD_CONTAINER_DISK_GB=80
export RUNPOD_VOLUME_GB=200
export RUNPOD_MOUNT_PATH="/workspace"

jq -n \
  --arg name "$RUNPOD_POD_NAME" \
  --arg templateId "$RUNPOD_TEMPLATE_ID" \
  --arg gpuType "$RUNPOD_GPU_TYPE" \
  --arg mountPath "$RUNPOD_MOUNT_PATH" \
  --argjson containerDisk "$RUNPOD_CONTAINER_DISK_GB" \
  --argjson volumeGb "$RUNPOD_VOLUME_GB" \
  '{
    name: $name,
    cloudType: "SECURE",
    computeType: "GPU",
    templateId: $templateId,
    gpuTypeIds: [$gpuType],
    gpuTypePriority: "availability",
    gpuCount: 1,
    dataCenterPriority: "availability",
    allowedCudaVersions: ["13.0"],
    containerDiskInGb: $containerDisk,
    volumeInGb: $volumeGb,
    volumeMountPath: $mountPath
  }' > /tmp/runpod-tofy-train.json

curl -sS -X POST "https://rest.runpod.io/v1/pods" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/runpod-tofy-train.json \
  | tee /tmp/runpod-pod.json \
  | jq 'if type=="array" then . else {id, name, desiredStatus, imageName, costPerHr, machine, volumeInGb, volumeMountPath} end'

export RUNPOD_POD_ID="$(jq -r '.id // empty' /tmp/runpod-pod.json)"
echo "RUNPOD_POD_ID=${RUNPOD_POD_ID}"
```

GPU substitutions:

- H100 80 GB: `export RUNPOD_GPU_TYPE="NVIDIA H100 80GB HBM3"`
- A100 PCIe: `export RUNPOD_GPU_TYPE="NVIDIA A100 80GB PCIe"`
- RTX PRO 6000: `export RUNPOD_GPU_TYPE="NVIDIA RTX PRO 6000 Blackwell Server Edition"`
- L40S: `export RUNPOD_GPU_TYPE="NVIDIA L40S"` and train with `48gb`

### Option B: Existing Network Volume

Use this when you already created a network volume. Pod and network volume must
be in the same datacenter. This intentionally omits `volumeInGb`, so it does not
create a regular pod volume.

```bash
export RUNPOD_POD_NAME="tofy-train-80gb"
export RUNPOD_GPU_TYPE="NVIDIA A100 80GB PCIe"
export RUNPOD_TEMPLATE_ID="obgryfbuad"
export RUNPOD_DATA_CENTER_ID="CA-MTL-3"
export RUNPOD_NETWORK_VOLUME_ID="7r75d9slhq"
export RUNPOD_CONTAINER_DISK_GB=80
export RUNPOD_MOUNT_PATH="/workspace"

jq -n \
  --arg name "$RUNPOD_POD_NAME" \
  --arg templateId "$RUNPOD_TEMPLATE_ID" \
  --arg gpuType "$RUNPOD_GPU_TYPE" \
  --arg dataCenter "$RUNPOD_DATA_CENTER_ID" \
  --arg networkVolumeId "$RUNPOD_NETWORK_VOLUME_ID" \
  --arg mountPath "$RUNPOD_MOUNT_PATH" \
  --argjson containerDisk "$RUNPOD_CONTAINER_DISK_GB" \
  '{
    name: $name,
    cloudType: "SECURE",
    computeType: "GPU",
    templateId: $templateId,
    gpuTypeIds: [$gpuType],
    gpuTypePriority: "availability",
    gpuCount: 1,
    dataCenterIds: [$dataCenter],
    allowedCudaVersions: ["13.0"],
    containerDiskInGb: $containerDisk,
    networkVolumeId: $networkVolumeId,
    volumeMountPath: $mountPath
  }' > /tmp/runpod-tofy-train.json

curl -sS -X POST "https://rest.runpod.io/v1/pods" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/runpod-tofy-train.json \
  | tee /tmp/runpod-pod.json \
  | jq 'if type=="array" then . else {id, name, desiredStatus, imageName, costPerHr, machine, networkVolume, volumeMountPath} end'

export RUNPOD_POD_ID="$(jq -r '.id // empty' /tmp/runpod-pod.json)"
echo "RUNPOD_POD_ID=${RUNPOD_POD_ID}"
```

### Option C: New Network Volume Script

Use this when you want the helper to create a probe pod, create a network volume
in the discovered datacenter, then launch the final pod attached to that volume.

```bash
RUNPOD_API_KEY="${RUNPOD_API_KEY}" \
GPU_TYPE_ID="NVIDIA A100-SXM4-80GB" \
CUDA_VERSION=13.0 \
TEMPLATE_ID="obgryfbuad" \
POD_NAME="tofy-train-80gb" \
NETWORK_VOLUME_GB=250 \
VOLUME_MOUNT_PATH="/workspace" \
./scripts/runpod_cuda_volume_pod.sh | tee /tmp/runpod-volume-pod.log

export RUNPOD_POD_ID="$(awk -F'"' '/finalPodId/ {print $4}' /tmp/runpod-volume-pod.log | tail -n1)"
echo "RUNPOD_POD_ID=${RUNPOD_POD_ID}"
```

If the API returns `There are no instances currently available`, retry later or
switch GPU type. Keep the CUDA 13.0 filter unless you are intentionally
debugging a lower-CUDA machine.

## 3. SSH

Get the exact SSH command from the RunPod Connect tab. The host suffix can vary.
You will paste the next sections inside that SSH shell.

```bash
ssh <runpod-user>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

## 4. Pod Bootstrap

Paste this inside the pod. It installs system tools, Rust, Go, HF CLI, saves
auto-stop credentials locally, writes the auto-stop wrapper, checks CUDA, and
prepares SSH for GitHub.

```bash
set -euo pipefail

apt-get update
apt-get install -y \
  git curl ca-certificates build-essential pkg-config libssl-dev \
  openssh-client tmux htop nvtop pciutils jq rsync zstd \
  python3-pip golang-go

if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
source "$HOME/.cargo/env"

if ! command -v hf >/dev/null 2>&1; then
  python3 -m pip install -U "huggingface_hub[cli]" --break-system-packages
fi

read -rsp "RunPod API key for auto-stop: " RUNPOD_API_KEY
echo
read -rp "RunPod pod id for auto-stop: " RUNPOD_POD_ID
cat > /workspace/tofy-runpod.env <<EOF
export RUNPOD_API_KEY='${RUNPOD_API_KEY}'
export RUNPOD_POD_ID='${RUNPOD_POD_ID}'
EOF
chmod 600 /workspace/tofy-runpod.env

cat > /workspace/run-tofy-and-stop.sh <<'EOF'
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
chmod +x /workspace/run-tofy-and-stop.sh

mkdir -p ~/.ssh
chmod 700 ~/.ssh
if [ ! -f ~/.ssh/runpod_tofy ]; then
  ssh-keygen -t ed25519 -C "runpod-tofy-pod" -f ~/.ssh/runpod_tofy -N ""
fi
cat > ~/.ssh/config <<'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/runpod_tofy
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/runpod_tofy ~/.ssh/config
chmod 644 ~/.ssh/runpod_tofy.pub
ssh-keyscan github.com >> ~/.ssh/known_hosts
chmod 644 ~/.ssh/known_hosts

nvidia-smi
nvcc --version || true
go version
hf --help >/dev/null

echo
echo "Add this deploy key to GitHub repo -> Settings -> Deploy keys:"
cat ~/.ssh/runpod_tofy.pub
```

Add the printed public key to GitHub. Keep write access disabled unless you need
to push from the pod.

## 5. Repo And Cache

After adding the deploy key, paste this inside the pod. It clones or updates the
repo, downloads the prepared cache, extracts it into `/workspace/Tofy`, verifies
the expected files, and builds release.

```bash
set -euo pipefail
source "$HOME/.cargo/env"

cd /workspace
if [ ! -d Tofy/.git ]; then
  git clone git@github.com:StepanZagray/Tofy.git Tofy
fi

cd /workspace/Tofy
git fetch origin
git pull --ff-only

hf download Grayza/80gb-profile-go-cache \
  tofy-cache-80gb-a8e7916-1780391272.tar.zst \
  --repo-type dataset \
  --local-dir /workspace

tar --zstd --no-same-owner --no-same-permissions \
  -xf /workspace/tofy-cache-80gb-a8e7916-1780391272.tar.zst \
  -C /workspace/Tofy

echo "Prepared cache:"
du -sh data/cache eval local_models 2>/dev/null || true
ls -lh data/cache eval local_models/vocabs 2>/dev/null || true

cargo build --release
```

If `local_models/vocabs` is missing after extraction, do not start the long run;
rebuild or copy the prepared cache with vocabs included.

For unpushed local changes, run this from your workstation instead of `git pull`
on the pod:

```bash
rsync -az --delete \
  --exclude target --exclude runs --exclude .git \
  /home/stepan/Coding/Personal/Tofy/ \
  root@<pod-host>:/workspace/Tofy/
```

## 6. Probe Or Train

Probe first on a fresh GPU shape. Paste once; it creates a probe script, starts
tmux, and attaches to the session:

```bash
cd /workspace/Tofy
source "$HOME/.cargo/env"

cat > /workspace/tofy-probe.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Tofy
source "$HOME/.cargo/env"
export RUST_BACKTRACE=1

/workspace/run-tofy-and-stop.sh \
  ./target/release/jepa_ai --max-vram-probe --profile 80gb --stage all --probe-dir /workspace/tofy-vram-probe-80gb \
  2>&1 | tee /workspace/tofy-vram-probe-80gb.log
EOF
chmod +x /workspace/tofy-probe.sh

tmux new -s tofy-probe /workspace/tofy-probe.sh
```

Detach with `Ctrl+b`, then `d`. Reattach or monitor with:

```bash
tmux attach -t tofy-probe
tail -f /workspace/tofy-vram-probe-80gb.log
nvidia-smi
```

Start the full 80 GB run. Paste once; it updates the repo, builds, creates a
train script, starts tmux, and attaches:

```bash
cd /workspace/Tofy
source "$HOME/.cargo/env"
git fetch origin
git pull --ff-only
cargo build --release

cat > /workspace/tofy-train.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Tofy
source "$HOME/.cargo/env"
export RUST_BACKTRACE=1
export TOFY_TRAIN_DTYPE=bf16

/workspace/run-tofy-and-stop.sh \
  ./target/release/jepa_ai train 80gb \
  2>&1 | tee /workspace/tofy-train-80gb.log
EOF
chmod +x /workspace/tofy-train.sh

tmux new -s tofy-train /workspace/tofy-train.sh
```

For 48 GB pods, use `--profile 48gb`, `train 48gb`, and log names ending in
`48gb`.

## 7. Resume

Paste once inside the pod:

```bash
cd /workspace/Tofy
source "$HOME/.cargo/env"

cat > /workspace/tofy-resume.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Tofy
source "$HOME/.cargo/env"
export RUST_BACKTRACE=1
export TOFY_TRAIN_DTYPE=bf16

/workspace/run-tofy-and-stop.sh \
  ./target/release/jepa_ai train 80gb --resume latest \
  2>&1 | tee /workspace/tofy-train-80gb-resume.log
EOF
chmod +x /workspace/tofy-resume.sh

tmux new -s tofy-resume /workspace/tofy-resume.sh
```

Resume a specific run with:

```bash
./target/release/jepa_ai train 80gb --resume code_poc_<timestamp>
```

Resume only with the same architecture/profile that created the run.

## 8. Manual Go Eval

The full pipeline already runs Go eval and uses it for decoder promotion. Use
this only to rescore a checkpoint:

```bash
cd /workspace/Tofy
source "$HOME/.cargo/env"
RUN_ID=$(ls -td runs/code_poc_* | head -n1 | xargs -r basename)
test -n "${RUN_ID}"
source scripts/tofy_pi_runtime_env.sh runs/${RUN_ID} 80gb

./target/release/jepa_ai --generate-go-code-eval-suite --output eval/code_assistant_go_hard.jsonl

./target/release/jepa_ai --eval-code-assistant \
  "$TOFY_WORLD_ENCODER_MODEL" \
  "$TOFY_ENCODER_VOCAB" \
  "$TOFY_WORLD_MODEL" \
  eval/code_assistant_go_hard.jsonl \
  384 "$TOFY_PROFILE_DIM" "$TOFY_PROFILE_MAX_SEQ" "$TOFY_PROFILE_LAYERS" "$TOFY_PROFILE_HEADS" "$TOFY_PROFILE_BRIDGE_DIM" "$TOFY_PROFILE_CONTEXT_SLOTS" \
  --high-world-model "$TOFY_HIGH_WORLD_MODEL" \
  --code-decoder "$JEPA_CANDLE_DECODER" \
  --code-decoder-vocab "$JEPA_CANDLE_DECODER_VOCAB" \
  --pi-agent-env \
  --go-timeout-sec 6 \
  2>&1 | tee /workspace/tofy-go-eval-${RUN_ID}.log
```

## 9. Artifact Recovery

Run from your workstation before terminating the pod:

```bash
export TOFY_POD_HOST="<pod-host-from-runpod-connect>"

rsync -az --info=progress2 \
  root@${TOFY_POD_HOST}:/workspace/Tofy/runs/ \
  /home/stepan/Coding/Personal/Tofy/runs/

rsync -az --info=progress2 \
  root@${TOFY_POD_HOST}:/workspace/Tofy/data/cache/ \
  /home/stepan/Coding/Personal/Tofy/data/cache/

rsync -az --info=progress2 \
  root@${TOFY_POD_HOST}:/workspace/tofy-train-80gb*.log \
  /home/stepan/Coding/Personal/Tofy/runs/ || true

rsync -az --info=progress2 \
  root@${TOFY_POD_HOST}:/workspace/tofy-go-eval-*.log \
  /home/stepan/Coding/Personal/Tofy/runs/ || true
```

At minimum, preserve `runs/code_poc_<timestamp>/`. It contains checkpoints,
optimizer sidecars, TensorBoard event files, metadata, vocab files, and pipeline
metadata.

After a completed run reports a better metric, update `docs/RESULTS.md` with
the exact command and metric. Do not update it for failed probes or unchanged
metrics.

## 10. Stop Or Terminate

Stop keeps a regular pod volume attached to the same pod:

```bash
curl -fsS --request POST \
  --url "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -w "HTTP %{http_code}\n"
```

Terminate deletes a regular pod volume. Network volumes survive pod termination
but are tied to one datacenter. Delete a network volume manually only after you
have recovered or intentionally discarded its contents.
