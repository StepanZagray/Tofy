# Training Pod Runbook

This is the low-back-and-forth RunPod path. Pod-side commands live in
`scripts/`, so the console flow is mostly “git pull, run script.”

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

Run locally:

```bash
sudo pacman -S --needed curl jq openssh rsync

read -rsp "RunPod API key: " RUNPOD_API_KEY
export RUNPOD_API_KEY
echo
echo "RUNPOD_API_KEY length: ${#RUNPOD_API_KEY}"
```

If the key length is `0`, stop and set the key again. Do not paste API keys or
full RunPod responses into public logs.

Build and upload the complete cache locally before starting the pod. This is
the only place the large token caches should be generated:

```bash
cargo run --release -- prepare cache 80gb \
  --auto-hf-upload \
  --hf-dataset Grayza/80gb-profile-go-cache
```

This uploads `data/`, `eval/`, and `local_models/vocabs/` to the dataset root.
Large files are stored as individual `*.zst` objects; small manifests and vocab
files stay plain. There is no `.tar.zst` archive name to pass to the pod.

Create a Hugging Face token with read access to the cache dataset. This is
separate from the GitHub deploy key; GitHub keys only allow `git pull`.

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

Use this when you want the helper to probe network-volume-capable datacenters
one at a time, create a network volume in the first datacenter that can place
the GPU, and launch the final pod attached to that volume.

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

## 3. Clone Repo To Pod

Get the exact SSH command from the RunPod Connect tab, then set these locally.
For a command like `ssh 2pb...@ssh.runpod.io -i ~/.ssh/id_ed25519`, use:

```bash
export TOFY_POD_SSH="2pb...@ssh.runpod.io"
export TOFY_POD_KEY="$HOME/.ssh/id_ed25519"
```

Commit and push your local changes before using the pod. The pod should pull
from Git, not receive local working-tree edits.

Connect to the pod:

```bash
ssh -tt -i "${TOFY_POD_KEY}" "${TOFY_POD_SSH}"
```

Inside the pod, create a GitHub SSH key if the pod does not already have one:

```bash
apt-get update
apt-get install -y git openssh-client ca-certificates

mkdir -p ~/.ssh
chmod 700 ~/.ssh
test -f ~/.ssh/runpod_tofy || ssh-keygen -t ed25519 -C "runpod-tofy-pod" -f ~/.ssh/runpod_tofy -N ""
cat > ~/.ssh/config <<'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/runpod_tofy
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/runpod_tofy ~/.ssh/config
ssh-keyscan github.com >> ~/.ssh/known_hosts
cat ~/.ssh/runpod_tofy.pub
```

Add the printed public key to GitHub repo -> Settings -> Deploy keys. Then
clone or update the repo:

```bash
cd /workspace
git clone git@github.com:StepanZagray/Tofy.git Tofy || true
cd /workspace/Tofy
git pull --ff-only
```

Use `ssh -tt` for interactive pod access. Some RunPod endpoints reject
one-shot, non-PTY commands such as `ssh ... 'ps aux'` with
`Error: Your SSH client doesn't support PTY`; connect with `ssh -tt` first and
then run status commands inside the shell.

## 4. Pod Setup

Run inside the pod from the cloned repo after Step 3. This installs system
tools, Rust, Go, HF CLI, creates `/workspace/run-tofy-and-stop.sh`, saves
auto-stop credentials, saves the HF token for cache download, and checks CUDA.

Export these inside the pod before setup. `RUNPOD_API_KEY` is your RunPod
account REST API key (RunPod console → Settings → API Keys). Training uses it
to stop the pod when the run exits; without it, auto-stop fails and the pod
keeps billing.

```bash
export RUNPOD_API_KEY="<runpod-rest-api-key>"
export RUNPOD_POD_ID="<pod-id-from-create-step>"
export HF_TOKEN="<hf-read-token>"
scripts/runpod_pod_setup.sh
```

Setup writes `/workspace/tofy-runpod.env` with those values. The auto-stop
wrapper sources that file on train exit.

If you skip the exports above, the script prompts for each value instead.

`export` matters: `HF_TOKEN=...` without `export` is only a shell variable and
child processes such as `scripts/runpod_restore_cache_build.sh` and `hf
download` will not see it.

To fix a bad or expired key later, edit `/workspace/tofy-runpod.env` on the pod
or re-run setup with fresh exports.

## 5. Restore Cache And Build

Run inside the pod:

```bash
cd /workspace/Tofy
scripts/runpod_restore_cache_build.sh
```

The restore script downloads the prepared cache tree directly into
`/workspace/Tofy`, decompresses downloaded `*.zst` files in place, and removes
the compressed copies. It does not download or extract a tarball.

Defaults and required inputs:

- dataset: `Grayza/80gb-profile-go-cache`
- restore target: `/workspace/Tofy`

Override if needed:

```bash
TOFY_CACHE_HF_DATASET="Grayza/80gb-profile-go-cache" \
scripts/runpod_restore_cache_build.sh
```

If `local_models/vocabs` is missing after restore, do not start the long run;
rebuild and upload the prepared cache locally.

The restore script also checks for the base and Go-feedback token cache
manifests. If it reports a missing `data/cache/go_feedback/...` path, rebuild
and upload the cache locally with the command from Step 1 before training.

## 6. Probe Or Train

Both `runpod_probe.sh` and `runpod_train.sh` start inside **tmux** automatically.
That keeps long runs alive after you disconnect SSH. Detach without stopping the
job with **`Ctrl+b`**, then **`d`**. Reattach later with `tmux attach -t
<session>`. List sessions with `tmux ls`.

Probe first on a fresh GPU shape. The probe intentionally does not use the
auto-stop wrapper, so the pod stays running for the training launch if the probe
passes.

```bash
cd /workspace/Tofy
PROFILE=80gb scripts/runpod_probe.sh
```

Default probe session: `tofy-probe`.

```bash
tmux attach -t tofy-probe
tail -f /workspace/tofy-vram-probe-80gb.log
nvidia-smi
```

Start the full 80 GB run. This uses the auto-stop wrapper by default.

```bash
cd /workspace/Tofy
PROFILE=80gb scripts/runpod_train.sh train
```

Default train session: `tofy-train`. Log file:
`/workspace/tofy-train-80gb.log`.

```bash
tmux attach -t tofy-train
tail -f /workspace/tofy-train-80gb.log
nvidia-smi
```

`scripts/runpod_train.sh` defaults to `TOFY_REQUIRE_PREPARED_CACHE=1`, so a
missing cache exits immediately instead of writing large token caches to the
RunPod volume. It also defaults `TOFY_GO_MODEL_FEEDBACK_ROWS=0`, because
model-failure feedback mining depends on the just-trained decoder and would
force pod-side data/cache regeneration. Override those variables only when you
intentionally want the pod to generate new feedback/cache files.

For 48 GB pods, use `PROFILE=48gb`.

## 7. Resume

Resume latest:

```bash
cd /workspace/Tofy
PROFILE=80gb scripts/runpod_train.sh resume
```

Default resume session: `tofy-resume`. Reattach with `tmux attach -t
tofy-resume`.

Resume a specific run:

```bash
cd /workspace/Tofy
PROFILE=80gb RESUME_TARGET=code_poc_<timestamp> scripts/runpod_train.sh resume
```

Resume a run but treat the existing encoder as complete and start world training:

```bash
cd /workspace/Tofy
PROFILE=80gb RESUME_TARGET=code_poc_<timestamp> SKIP_TRAINED_STAGES=latent scripts/runpod_train.sh resume
```

Resume only with the same architecture/profile that created the run.

## 8. Manual Go Eval

The full pipeline already runs Go eval and uses it for decoder promotion. Use
this only to rescore a checkpoint:

```bash
cd /workspace/Tofy
PROFILE=80gb scripts/runpod_go_eval.sh
```

Evaluate a specific run:

```bash
cd /workspace/Tofy
PROFILE=80gb RUN_ID=code_poc_<timestamp> scripts/runpod_go_eval.sh
```

## 9. Artifact Recovery

Run locally before terminating the pod:

```bash
export TOFY_POD_SSH="2pb...@ssh.runpod.io"
export TOFY_POD_KEY="$HOME/.ssh/id_ed25519"

rsync -az --info=progress2 -e "ssh -i ${TOFY_POD_KEY}" \
  "${TOFY_POD_SSH}:/workspace/Tofy/runs/" \
  /home/stepan/Coding/Personal/Tofy/runs/

rsync -az --info=progress2 -e "ssh -i ${TOFY_POD_KEY}" \
  "${TOFY_POD_SSH}:/workspace/Tofy/data/cache/" \
  /home/stepan/Coding/Personal/Tofy/data/cache/

rsync -az --info=progress2 -e "ssh -i ${TOFY_POD_KEY}" \
  "${TOFY_POD_SSH}:/workspace/tofy-train-80gb*.log" \
  /home/stepan/Coding/Personal/Tofy/runs/ || true

rsync -az --info=progress2 -e "ssh -i ${TOFY_POD_KEY}" \
  "${TOFY_POD_SSH}:/workspace/tofy-go-eval-*.log" \
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
