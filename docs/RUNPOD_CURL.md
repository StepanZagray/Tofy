# Training Pod Runbook

This document is the copy-paste checklist for launching a cloud training pod,
running the current Tofy pipeline, resuming it, and pulling artifacts back.

Current recommended pod target:

- one high-VRAM NVIDIA GPU: L40S 48 GB minimum, A100/H100/RTX PRO 6000 preferred when available
- Ubuntu-like CUDA image with Rust build tools installed manually
- Pi coding agent installed with `curl -fsSL https://pi.dev/install.sh | sh`
- regular pod volume mounted at `/workspace`
- repo checkout at `/workspace/Tofy`
- canonical training command: `./target/release/jepa_ai train 48gb`

The `48gb` profile is defined in `config/model_profiles.json`: `DIM=1024`,
`LAYERS=12`, `HEADS=16`, `BRIDGE_DIM=1024`, `NUM_LATENT_TOKENS=96`, code
decoder `dim=1024`, decoder layers `12`, decoder FF `4096`, code decoder
`max_seq=192`, and test-scale budgets of latent `75000`, world `180000`,
high-world `36000`, code decoder `120000`, and Go feedback `24000`.
Current 48 GB training batches are encoder `256x2` (`512` effective), world
`256x2` (`512` effective), code decoder `128x2` (`256` effective), and Go
feedback `256x1` (`256` effective). Decoder and Go-feedback pipeline stages
pass `--conditioning-loss-weight 0.0` explicitly for throughput; direct manual
decoder runs still default to the conditioning-margin loss unless that flag is
provided.

Do not paste API keys or full RunPod API responses into public logs. Pod
responses can include environment variables.

## 1. Local Prerequisites

Install helpers on your local machine:

```bash
sudo pacman -S --needed curl jq openssh rsync
```

Load the RunPod API key into the current shell only:

```bash
read -rsp "RunPod API key: " RUNPOD_API_KEY
export RUNPOD_API_KEY
echo
echo "key length: ${#RUNPOD_API_KEY}"
```

If the printed length is `0`, the key is not set.

## 2. Create A Pod

Use availability-based placement first. Regular pod volume storage is preferred
for this project because it lets RunPod place the job wherever a suitable GPU is
available. Network volumes are useful for long-lived datasets, but they are tied
to one datacenter and can reduce GPU availability.

L40S 48 GB:

```bash
cat > /tmp/runpod-tofy-train.json <<'EOF'
{
  "name": "tofy-train-48gb",
  "cloudType": "SECURE",
  "computeType": "GPU",
  "templateId": "obgryfbuad",
  "gpuTypeIds": ["NVIDIA L40S"],
  "gpuTypePriority": "availability",
  "gpuCount": 1,
  "dataCenterPriority": "availability",
  "containerDiskInGb": 80,
  "volumeInGb": 250,
  "volumeMountPath": "/workspace"
}
EOF

curl -sS -X POST "https://rest.runpod.io/v1/pods" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/runpod-tofy-train.json \
  | tee /tmp/runpod-pod.json \
  | jq 'if type=="array" then . else {id, name, desiredStatus, imageName, costPerHr, machine, volumeInGb, volumeMountPath} end'
```

Alternatives for the same command:

- A100 80 GB: set `"gpuTypeIds": ["NVIDIA A100-SXM4-80GB"]`
- H100 80 GB: set `"gpuTypeIds": ["NVIDIA H100 80GB HBM3"]`
- RTX PRO 6000: set `"gpuTypeIds": ["NVIDIA RTX PRO 6000 Blackwell Server Edition"]`

If the response is an array, it is an API validation or availability error.
Read the error text, retry later, or switch GPU type. Only add an
`allowedCudaVersions` filter when you have a specific image/driver mismatch to
avoid; CUDA availability changes often and over-filtering can make placement
fail.

Save the pod ID:

```bash
export RUNPOD_POD_ID="$(jq -r '.id' /tmp/runpod-pod.json)"
echo "$RUNPOD_POD_ID"
```

Get the exact SSH command from the RunPod Connect tab. The host suffix can
differ between pods.

## 3. Pod Bootstrap

SSH into the pod, then install tools:

```bash
apt-get update
apt-get install -y \
  git curl ca-certificates build-essential pkg-config libssl-dev \
  openssh-client tmux htop nvtop pciutils jq rsync

curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"

nvidia-smi
nvcc --version || true
```

Install Go for autonomous Go-feedback repair data generation and hard eval:

```bash
apt-get install -y golang-go
go version
```

Install Pi on the pod. The installer is the official Linux/macOS path from
Pi's quickstart docs.

```bash
curl -fsSL https://pi.dev/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$HOME/.bun/bin:$PATH"
command -v pi
pi --version || true
```

## 4. GitHub Deploy Key

Create a deploy key on the pod and add the public key to the repo in GitHub:

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

Verify:

```bash
ssh -T git@github.com
```

Expected:

```text
Hi StepanZagray/Tofy! You've successfully authenticated, but GitHub does not provide shell access.
```

Keep private keys under `/root/.ssh`, not `/workspace/.ssh`; some RunPod volume
mounts report permissive permissions that OpenSSH rejects.

## 5. Clone Or Update

```bash
cd /workspace

if [ ! -d Tofy/.git ]; then
  git clone git@github.com:StepanZagray/Tofy.git Tofy
fi

cd /workspace/Tofy
git fetch origin
git pull --ff-only
```

If you are testing unpushed local changes, use `rsync` from your workstation
instead of `git pull`:

```bash
rsync -az --delete \
  --exclude target --exclude runs --exclude .git \
  /home/stepan/Coding/Personal/Tofy/ \
  root@<pod-host>:/workspace/Tofy/
```

## 6. Auto-Stop Wrapper

The wrapper stops the pod after the wrapped command exits. It prints only the
API key length, not the key.

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

Inside the pod, export the same `RUNPOD_API_KEY` and `RUNPOD_POD_ID` before
using the wrapper.

## 7. Build And Probe

Build the release binary:

```bash
cd /workspace/Tofy
source "$HOME/.cargo/env"
cargo build --release
```

Run the current profile-aware probe before a long 48 GB launch:

```bash
tmux new -s tofy-probe
```

Inside tmux:

```bash
export RUST_BACKTRACE=1

/workspace/run-tofy-and-stop.sh \
  ./target/release/jepa_ai --max-vram-probe --profile 48gb --stage all --probe-dir /workspace/tofy-vram-probe-48gb \
  2>&1 | tee /workspace/tofy-vram-probe-48gb.log
```

Detach without stopping:

```text
Ctrl+b, then d
```

Monitor:

```bash
tmux attach -t tofy-probe
tail -f /workspace/tofy-vram-probe-48gb.log
nvidia-smi
```

Use the sustained probe when changing profile shapes or batch sizes:

```bash
./target/release/jepa_ai --sustained-oom-probe --profile 48gb --stage all
```

## 8. Full Training

Start a full 48 GB run. This command is intended to run unattended from data
prep through latent, world, high-world, base code decoder, and Go-feedback
decoder training. Do not add `--with-code-eval` to the main training launch;
eval is a separate post-run step.

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
# Pipeline defaults set TOFY_CACHE_PREFETCH_BATCHES=4; keep this enabled unless
# debugging input-order or memory issues. It now covers both raw and cached streams.

/workspace/run-tofy-and-stop.sh \
  ./target/release/jepa_ai train 48gb \
  2>&1 | tee /workspace/tofy-train-48gb.log
```

Stage 1 bootstraps source data, materializes `data/encoder_mix.txt`, builds
vocabs, and writes token caches before model training starts. Reruns reuse
compatible manifests and token caches. Hub-backed files are written atomically,
so an interrupted pod should leave temporary files rather than corrupting
canonical datasets. The same pipeline also prepares Go instruction/repair data,
builds `data/code_poc_go_mix.txt`, initializes the Go-feedback decoder from the
base code decoder, and trains `runs/<run_id>/decoder_code_go_feedback/model.safetensors`
without a separate manual command.

Monitor:

```bash
tmux attach -t tofy-train
tail -f /workspace/tofy-train-48gb.log
nvidia-smi
```

## 9. Post-Training Go Eval

Run eval only after the full training command has completed. This is the
Golang eval path: `--generate-go-code-eval-suite` writes `language: "go"` tasks,
and `--eval-code-assistant` runs those tasks through a temporary `go test`
harness. The command below uses the newest `runs/code_poc_*` directory; set
`RUN_ID` manually if you want to evaluate a specific run.

```bash
cd /workspace/Tofy
source "$HOME/.cargo/env"
RUN_ID=$(ls -td runs/code_poc_* | head -n1 | xargs -r basename)
test -n "${RUN_ID}"

./target/release/jepa_ai --generate-go-code-eval-suite --output eval/code_assistant_go_hard.jsonl

./target/release/jepa_ai --eval-code-assistant \
  runs/${RUN_ID}/world/model.encoder.safetensors \
  runs/${RUN_ID}/latent/model.vocab.txt \
  runs/${RUN_ID}/world/model.safetensors \
  eval/code_assistant_go_hard.jsonl \
  384 1024 256 12 16 1024 96 \
  --high-world-model runs/${RUN_ID}/high_world/model.safetensors \
  --code-decoder runs/${RUN_ID}/decoder_code_go_feedback/model.safetensors \
  --go-timeout-sec 6 \
  2>&1 | tee /workspace/tofy-go-eval-${RUN_ID}.log
```

Use `--with-code-eval` only for deliberate debug runs where eval should be
attached to a resumed pipeline invocation. It is not the default pod training
path.

## 10. Manual Go Training With Pi Harness

Use this manual path only when you want to regenerate Go-feedback data or train
a separate Go decoder outside the autonomous `train 48gb` pipeline. Go is the
fast language curriculum; Pi is the coding-agent harness used to exercise the
trained checkpoint through the same OpenAI-compatible server a user would call.

Important boundary: the current Rust binary trains with supervised decoder CE
over generated pair files. Pi does not run inside the optimizer. The full
pipeline is already Go-feedback-driven after base decoder training by creating Go
instruction and repair pairs, training the `decoder_code_go_feedback` stage on
that data, then using Pi as the harness for end-to-end coding-agent checks
against the served checkpoint.

Generate Go execution-feedback data:

```bash
cd /workspace/Tofy
source "$HOME/.cargo/env"

./target/release/jepa_ai --prepare-github-top-code --output data/go_code_pairs.txt --languages Go --max-files 120000
./target/release/jepa_ai --prepare-go-function-tasks --input data/go_code_pairs.txt --output data/go_instruction_pairs.txt
./target/release/jepa_ai --prepare-go-repair-tasks --input data/go_instruction_pairs.txt --output data/go_repair_pairs.txt
./target/release/jepa_ai --prepare-code-poc-mix --output data/code_poc_go_mix.txt --base-pairs data/go_code_pairs.txt --instruction-pairs data/go_instruction_pairs.txt --instruction-repeat 4 --extra-pairs data/go_repair_pairs.txt --extra-repeat 2
```

Train a Go-feedback code decoder against the current encoder/world checkpoint.
Replace `<run_id>` with the run you want to continue from:

```bash
./target/release/jepa_ai --train-decoder \
  runs/<run_id>/world/model.encoder.safetensors \
  runs/<run_id>/latent/model.vocab.txt \
  runs/<run_id>/world/model.safetensors \
  data/code_poc_go_mix.txt \
  24000 256 192 1024 12 16 1024 96 \
  --decoder-kind code \
  --decoder-output runs/<run_id>/decoder_code_go/model.safetensors \
  --decoder-max-vocab 32000 \
  --grad-accum 1 \
  --conditioning-loss-weight 0.0 \
  --init-decoder runs/<run_id>/decoder_code/model.safetensors
```

Run the Go hard eval directly through Tofy's verifier:

Generate and run the Go hard eval suite:

```bash
./target/release/jepa_ai --generate-go-code-eval-suite --output eval/code_assistant_go_hard.jsonl

./target/release/jepa_ai --eval-code-assistant \
  runs/<run_id>/world/model.encoder.safetensors \
  runs/<run_id>/latent/model.vocab.txt \
  runs/<run_id>/world/model.safetensors \
  eval/code_assistant_go_hard.jsonl \
  384 1024 256 12 16 1024 96 \
  --high-world-model runs/<run_id>/high_world/model.safetensors \
  --code-decoder runs/<run_id>/decoder_code_go/model.safetensors \
  --go-timeout-sec 6
```

Serve the Go-trained checkpoint for Pi:

```bash
export JEPA_USE_CANDLE_DECODER=1
export JEPA_CANDLE_DECODER=/workspace/Tofy/runs/<run_id>/decoder_code_go/model.safetensors
export JEPA_CANDLE_DECODER_VOCAB=/workspace/Tofy/runs/<run_id>/decoder_code_go/model.vocab.txt
export TOFY_SERVE_BIND=127.0.0.1:8080

./target/release/jepa_ai --serve \
  runs/<run_id>/world/model.encoder.safetensors \
  runs/<run_id>/latent/model.vocab.txt \
  runs/<run_id>/world/model.safetensors \
  127.0.0.1:8080 \
  1024 256 12 16 1024 96 \
  --high-world-model runs/<run_id>/high_world/model.safetensors \
  2>&1 | tee /workspace/tofy-serve-pi.log
```

In another tmux pane, configure Pi to use the local Tofy server. Pi custom
models live at `~/.pi/agent/models.json`; use OpenAI Chat Completions
compatibility and disable unsupported reasoning/developer-role fields:

```bash
mkdir -p ~/.pi/agent
cat > ~/.pi/agent/models.json <<'EOF'
{
  "providers": {
    "tofy": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "sk-local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "tofy",
          "name": "Tofy Go Harness",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 8192,
          "maxTokens": 2048,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
EOF
```

Run Pi in JSON mode against the repo as the harness. Keep prompts Go-specific
so the model exercises the Go-trained decoder path:

```bash
cd /workspace/Tofy
mkdir -p runs/<run_id>/pi_go_harness

pi --model tofy/tofy --mode json \
  "Implement a small Go function and tests in /tmp/tofy-pi-go-smoke. Use go test and fix failures until it passes. Return only a concise summary." \
  2>/workspace/pi-go-smoke.stderr \
  | tee runs/<run_id>/pi_go_harness/smoke.jsonl
```

The Pi harness output is not automatically fed back into decoder training yet.
For now, use it as an end-to-end agent check beside `--eval-code-assistant`.
If a Pi run produces useful repair traces, convert them into TSV pairs before
the next Go feedback decoder pass.

## 11. Resume

Resume the newest run:

```bash
cd /workspace/Tofy
source "$HOME/.cargo/env"

tmux new -s tofy-resume
```

Inside tmux:

```bash
export RUST_BACKTRACE=1
export TOFY_TRAIN_DTYPE=bf16

/workspace/run-tofy-and-stop.sh \
  ./target/release/jepa_ai train 48gb --resume latest \
  2>&1 | tee /workspace/tofy-train-48gb-resume.log
```

Resume a specific run:

```bash
./target/release/jepa_ai train 48gb --resume code_poc_<timestamp>
```

Resume requires matching architecture/profile arguments. Do not resume a run
created with a different `config/model_profiles.json` shape.

## 12. Artifact Recovery

Before terminating a pod, copy the run artifacts back:

```bash
rsync -az --info=progress2 \
  root@<pod-host>:/workspace/Tofy/runs/ \
  /home/stepan/Coding/Personal/Tofy/runs/

rsync -az --info=progress2 \
  root@<pod-host>:/workspace/Tofy/data/cache/ \
  /home/stepan/Coding/Personal/Tofy/data/cache/

rsync -az --info=progress2 \
  root@<pod-host>:/workspace/tofy-train-48gb*.log \
  /home/stepan/Coding/Personal/Tofy/runs/

rsync -az --info=progress2 \
  root@<pod-host>:/workspace/tofy-go-eval-*.log \
  /home/stepan/Coding/Personal/Tofy/runs/
```

At minimum, preserve the run directory:

```text
runs/code_poc_<timestamp>/
```

It contains model checkpoints, optimizer sidecars, TensorBoard event files,
metadata, vocab files, and pipeline metadata. Standalone post-training eval logs
are written to `/workspace/tofy-go-eval-*.log` by the command above; recover
those logs separately if you need the verifier output.

After a completed run reports a better metric, update `docs/RESULTS.md` with
the exact command and metric. Do not update it for failed probes or unchanged
metrics.

## 13. Stop Or Terminate

Stop keeps the regular pod volume attached to the same pod:

```bash
curl -fsS --request POST \
  --url "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -w "HTTP %{http_code}\n"
```

Terminate deletes the regular pod volume. Only terminate after artifact
recovery is complete.

Storage behavior:

- regular pod volume: persists across stop/start of the same pod, deleted on terminate
- network volume: survives pod termination but is tied to one datacenter
- container disk: useful for build dependencies, not for run artifacts
