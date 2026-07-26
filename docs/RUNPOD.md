# RunPod training (veclab + Qwen bridge)

This is the current cloud workflow for the **knowledge-injection experiment**:
encoder → world knowledge → frozen **Qwen3-1.7B-Base** bridge → veclab eval.

> **Do not schedule decoder-only evaluation.** The unchanged frozen/base Qwen
> control has already been evaluated and failed with zero seen and held-out
> VecLab suite passes. It is a closed control for this experiment, not a stage
> to repeat after resume or training. Run `--eval-bridge` only for a qualifying
> bridge checkpoint and retain matched, wrong-conditioning, and zeroed-state
> comparisons. Reopen the decoder-only control only if the base decoder,
> tokenizer, prompt contract, or evaluation suite is deliberately changed and
> the resulting run is treated as a new experiment. Training never invokes
> this control; the only supported entry point is
> `scripts/run_veclab_decoder_floor.sh <runs/code_poc_<id>>` (loads that run's
> bridge/world/latent artifacts with `TOFY_EVAL_MODE=floor`).

## 1. Create the pod (website)

On [RunPod](https://www.runpod.io/), create a GPU pod:

| Goal | Suggested GPU | Profile |
|------|---------------|---------|
| Current experiment | RTX PRO 6000 Blackwell (96 GiB) | `minimal` |

Template: **RunPod PyTorch** or any CUDA image with Ubuntu. Attach a **network
volume** only if you want checkpoints to survive pod termination; otherwise
`/workspace` on the pod volume is fine — **rsync artifacts off before
terminating**.

From the Connect tab, copy the SSH command, e.g.
`ssh abc123@ssh.runpod.io -i ~/.ssh/id_ed25519`.

Set these on your **local** machine:

```bash
export TOFY_POD_SSH="abc123@ssh.runpod.io"
export TOFY_POD_KEY="$HOME/.ssh/id_ed25519"
```

**Push your branch to GitHub before training.** The pod pulls from Git; do not
rely on copying uncommitted local edits.

## 2. Connect

RunPod may reject one-shot non-PTY SSH. Always allocate a TTY:

```bash
ssh -tt -i "${TOFY_POD_KEY}" "${TOFY_POD_SSH}"
```

Run status commands **inside** that shell (`nvidia-smi`, `tmux ls`, etc.).

## 3. Clone repo on the pod

If the pod has no GitHub deploy key yet:

```bash
apt-get update
apt-get install -y git openssh-client ca-certificates

mkdir -p ~/.ssh && chmod 700 ~/.ssh
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

Add the printed key as a **Deploy key** on the GitHub repo, then:

```bash
cd /workspace
git clone git@github.com:StepanZagray/Tofy.git Tofy || true
cd /workspace/Tofy
git pull --ff-only
```

## 4. Pod setup (once per pod)

Installs Rust, Go, HF CLI, tmux; checks CUDA.

```bash
cd /workspace/Tofy
export HF_TOKEN="<hf-read-token>"   # needed for Qwen weight download
scripts/runpod_pod_setup.sh
```

`export HF_TOKEN` matters — without it, child processes (`hf download`) will not
see the token. The token is saved to `/workspace/tofy-runpod.env` when provided.

## 5. Download Qwen weights (required)

Bridge and eval need **Qwen3-1.7B-Base** on disk:

```bash
cd /workspace/Tofy
source /workspace/tofy-runpod.env 2>/dev/null || true
mkdir -p models
hf download Qwen/Qwen3-1.7B-Base --local-dir models/qwen3-1.7b-base
export TOFY_QWEN_DIR=/workspace/Tofy/models/qwen3-1.7b-base
```

Add to `/workspace/tofy-runpod.env` so tmux sessions inherit it:

```bash
echo 'export TOFY_QWEN_DIR=/workspace/Tofy/models/qwen3-1.7b-base' >> /workspace/tofy-runpod.env
```

## 6. Data (veclab)

**Option A — let the pipeline generate (default):** Stage 1 of `train minimal`
runs `--prepare-veclab` and builds the encoder vocab cache (~minutes). No HF
dataset restore needed.

**Option B — copy from local (faster, skips regen):**

```bash
# on local machine
rsync -az --info=progress2 -e "ssh -i ${TOFY_POD_KEY}" \
  /home/stepan/Coding/Personal/Tofy/data/fictional/ \
  "${TOFY_POD_SSH}:/workspace/Tofy/data/fictional/"
rsync -az -e "ssh -i ${TOFY_POD_KEY}" \
  /home/stepan/Coding/Personal/Tofy/eval/veclab_eval.jsonl \
  "${TOFY_POD_SSH}:/workspace/Tofy/eval/"
```

Verify on the pod:

```bash
cd /workspace/Tofy
go test ./data/fictional/veclab/...
cargo run --release -- --prepare-veclab --seed 20260705 --out data/fictional --print-split-stats
```

You do **not** need a large HF token-cache restore for this experiment. Stage 1
builds the small encoder vocab cache locally. Keep
`TOFY_REQUIRE_PREPARED_CACHE=0` (the train script default) unless you are
deliberately consuming a prepared HF cache tree.

`veclab_bridge_transfer.txt` is **not** part of the rsync'd corpus; the
pipeline rebuilds it from knowledge + seen task rows before bridge training.

Optional local handoff (encoder vocab only, small):

```bash
# local
cargo run --release -- prepare cache minimal
# upload local_models/vocabs + data/cache encoder manifests only if you want
```

## 7. Probe (optional)

Quick VRAM sanity check before a long run (the script uses `minimal`):

```bash
cd /workspace/Tofy
source /workspace/tofy-runpod.env
export TOFY_QWEN_DIR=/workspace/Tofy/models/qwen3-1.7b-base
scripts/runpod_probe.sh
```

Attach: `tmux attach -t tofy-probe` · Log: `/workspace/tofy-vram-probe-minimal.log`

## 8. Train

Both train scripts start **tmux** automatically. Detach: `Ctrl+b` then `d`.
Reattach: `tmux attach -t tofy-train`.

```bash
cd /workspace/Tofy
source /workspace/tofy-runpod.env
export TOFY_QWEN_DIR=/workspace/Tofy/models/qwen3-1.7b-base

# VecLab experiment — `minimal` targets RTX PRO 6000 96 GiB; see RESULTS for fit-check status
# TOFY_REQUIRE_PREPARED_CACHE must be 0 when the pipeline builds its own vocab
TOFY_REQUIRE_PREPARED_CACHE=0 SKIP_GIT_PULL=1 scripts/runpod_train.sh train
```

Log: `/workspace/tofy-train-minimal.log`

Pipeline stages (automatic):

1. veclab data + encoder vocab cache  
2. encoder (LeJEPA on `data/fictional/veclab_encoder_mix.txt`)  
3. LeWorldModel knowledge (`--train-world-knowledge` on
   `veclab_knowledge_train.txt`, validating on `veclab_knowledge_val.txt`)
4. Qwen bridge (`--train-bridge` on pipeline-built
   `data/fictional/veclab_bridge_transfer.txt`: world docs for fns 1–200 +
   code tasks for fns 1–80)
5. veclab eval (`--eval-bridge` on `eval/veclab_eval.jsonl`)

Run root: `runs/code_poc_<timestamp>/{latent,world,bridge,eval}`.

Bridge checkpoints:

- `bridge/context.safetensors` (+ `.best` / `.latest` sidecars)
- `bridge/weights.safetensors` (+ `.best` / `.latest` / `.world` sidecars)

Encoder vocab for the run: `latent/model.vocab.txt`.

The world stage follows LeWorldModel's two-term objective: raw next-embedding
MSE plus exact Epps–Pulley SIGReg (`λ=0.09`, 1,024 resampled Gaussian
projections, 17 knots). It trains the encoder, compressor, post-normalization
projectors, and six-block AdaLN-Zero action-conditioned predictor jointly.
Association top-1 is telemetry only.

World/bridge checkpoint structure changed with the LeWorldModel rewrite. Start a
new run root; do not resume a pre-LeWorldModel world or bridge checkpoint.
Tokenizer specification v9 also invalidates earlier encoder vocabularies and
token caches: it prevents BPE merges across lexical boundaries and rejects
one-token masked-prediction rows. Start a new encoder run after this change;
the cache builder prints token-fertility statistics that should be retained in
the run log.

`minimal` is the only supported profile.

## 9. Resume

```bash
cd /workspace/Tofy
source /workspace/tofy-runpod.env
export TOFY_QWEN_DIR=/workspace/Tofy/models/qwen3-1.7b-base
TOFY_REQUIRE_PREPARED_CACHE=0 scripts/runpod_train.sh resume
```

Resume a specific run:

```bash
RESUME_TARGET=code_poc_<timestamp> scripts/runpod_train.sh resume
```

Skip completed stages (example: encoder already done):

```bash
RESUME_TARGET=code_poc_<timestamp> \
  SKIP_TRAINED_STAGES=latent scripts/runpod_train.sh resume
```

### Automatic CUDA OOM recovery

Every training arm in the `minimal` pipeline (encoder, world, both bridges,
static prefix, and both LoRA controls) records its actual batch schedule in:

```text
runs/code_poc_<timestamp>/adaptive_batches.json
```

When an isolated child reports a CUDA-specific allocation failure, the parent
halves physical batch and doubles gradient accumulation. Both values remain
powers of two, and their product must equal the stage's original effective
batch. Encoder/world warmup pairs retain their own initial effective batch and
advance by the same number of halving/doubling reductions. The child resumes
only when a resume sidecar exists (or the pipeline was
explicitly launched with `--resume`); otherwise the new attempt starts that
stage from step zero. Checkpoint metadata rejects a different effective batch.

The controller stops and reports failure if batch one also OOMs. It does not
retry generic host-memory errors, SIGKILL/exit 137, panics, data errors, or a
bridge that completes without qualifying. Disable it with:

```bash
export TOFY_AUTO_BATCH_OOM_RECOVERY=false
```

Preserving effective batch preserves the optimizer schedule, but it does not
make retries bit-identical: SIGReg and other microbatch-local statistics see a
different physical grouping. Bridge sampling resumes at the same effective
sample offset; latent and world streaming readers currently restart their data
streams on resume, so those two stages are checkpoint-safe but not exact
data-order continuations.

The run metadata must identify the supported `minimal` profile.
Resume tuples created before checkpoint generation IDs were introduced are
rejected: their weights cannot be proven to match their optimizer/state
sidecars, so start a fresh run instead of overriding that check.

## 10. Manual eval only

The pipeline runs eval in stage 5. To rescore a checkpoint:

```bash
cd /workspace/Tofy
source /workspace/tofy-runpod.env
export TOFY_QWEN_DIR=/workspace/Tofy/models/qwen3-1.7b-base

RUN=code_poc_<timestamp>
# Use context.best.safetensors or weights.best.safetensors as appropriate.
BRIDGE="runs/${RUN}/bridge/weights.best.safetensors"
WORLD="${BRIDGE%.safetensors}.world.safetensors"
[[ -f "${WORLD}" ]] || WORLD="runs/${RUN}/world/model.safetensors"

cargo run --release -- --eval-bridge \
  "$TOFY_QWEN_DIR" \
  "${BRIDGE}" \
  "runs/${RUN}/world/model.encoder.safetensors" \
  "runs/${RUN}/latent/model.vocab.txt" \
  "${WORLD}" \
  eval/veclab_eval.jsonl \
  "runs/${RUN}/eval/report.json"
```

For the closed decoder-only floor control (not part of training), use
`scripts/run_veclab_decoder_floor.sh runs/${RUN}` with `TOFY_QWEN_DIR` set.

## 11. Recover artifacts (before terminate)

**Local machine:**

```bash
export TOFY_POD_SSH="..."
export TOFY_POD_KEY="$HOME/.ssh/id_ed25519"

rsync -az --info=progress2 -e "ssh -i ${TOFY_POD_KEY}" \
  "${TOFY_POD_SSH}:/workspace/Tofy/runs/" \
  /home/stepan/Coding/Personal/Tofy/runs/

rsync -az -e "ssh -i ${TOFY_POD_KEY}" \
  "${TOFY_POD_SSH}:/workspace/tofy-train-*.log" \
  /home/stepan/Coding/Personal/Tofy/runs/ || true
```

Minimum to keep: `runs/code_poc_<timestamp>/` (checkpoints, TensorBoard,
`eval/report.json`, vocabs).

Update `docs/RESULTS.md` when a run beats prior metrics.

## 12. Environment reference

| Variable | Required | Purpose |
|----------|----------|---------|
| `TOFY_QWEN_DIR` | **yes** | Path to Qwen3-1.7B-Base (HF download dir) |
| `HF_TOKEN` | yes (download) | Hugging Face auth for model weights |
| `TOFY_AUTO_BATCH_OOM_RECOVERY` | no (default `true`) | Retry a confirmed CUDA allocation OOM with half physical batch and double accumulation |
| `TOFY_REQUIRE_PREPARED_CACHE` | no | Default `0` in `scripts/runpod_train.sh` (pipeline builds vocab). Set `1` only when restoring a prepared HF cache that must be authoritative. |
| `TOFY_TRAIN_DTYPE` | no | Default `bf16` on GPU |
| `TOFY_BRIDGE_REGIME` | no | `context` (Step 2) or `weights` (Step 3); see bridge spec |
| `TOFY_DECODER_CONDITIONING_NEGATIVES` | no | `hard`; uses a true different-function row even at batch one |
| `TOFY_BRIDGE_MIN_SEMANTIC_GAP` | no | Minimum `wrong_ce - matched_ce` for world-bridge checkpoint eligibility; default `0.02` |
| `TOFY_BRIDGE_LR` | no | Default `1e-4`; lower than the former `2e-4` to preserve matched CE after causal alignment begins |
| `TOFY_BRIDGE_COUNTERFACTUAL_PROMPTS` | no | Default `true`; training/validation reveal only the shared `func Solve` signature, forcing behavior to come from the state |
| `TOFY_BRIDGE_TRAIN_FUNCTION_MAX` | no | Default `80`; functions `1..80` are bridge training groups |
| `TOFY_BRIDGE_VALIDATION_FUNCTION_MAX` | no | Default `100`; functions `81..100` are function-disjoint bridge validation groups |
| `TOFY_DECODER_CONDITIONING_UNLIKELIHOOD_WEIGHT` | no | Default `0.25`; penalizes target tokens under a wrong state without CE-max saturation |
| `TOFY_DECODER_CONDITIONING_SEPARATION_WEIGHT` | no | Default `0.05`; enforces state-dependent adapter outputs for hard negatives |
| `TOFY_DECODER_CONDITIONING_MIN_DISTANCE` | no | Default `0.1`; squared-distance hinge used by the separation loss |
| `TOFY_CONDITIONING_DROPOUT` | no | Default `0`; exact-zero conditioning is an identity path through frozen Qwen and supplies no trainable bridge gradient; use hard negatives for causal supervision |
| `TOFY_BRIDGE_SEMANTIC_WARMUP` / `TOFY_BRIDGE_SEMANTIC_PATIENCE` | no | Defaults `400` / `1200`; a qualified plateau early-stops successfully, while an unqualified plateau fails the stage |
| `TOFY_KNOWLEDGE_UNFREEZE_WORLD` | no | Pipeline default: false for context, true for practical weights-mode joint alignment |
| `TOFY_WEIGHTS_BRIDGE_BATCH` / `TOFY_WEIGHTS_BRIDGE_GRAD_ACCUM` | no | Weights-mode-only physical batch and accumulation overrides; use `1` / `128` on a 48 GB L40S to retain effective batch 128 after an unfrozen-world OOM |

Full-sequence CE trains Go syntax and the wrapper. Semantic margin,
unlikelihood, validation-gap, and checkpoint eligibility are computed over the
fictional API identifiers in the completion's `return` expression (or the full
expression when no such identifier exists), so shared boilerplate cannot
satisfy the causal objective. In bridge eval mode Qwen sees the same
signature-only counterfactual prompt used during training; the full task remains
available to the world-state pathway.

The LeWorldModel bridge was capacity-tested on RunPod pod
`0ynxc2a65zgog1-64410ad3` (46,068 MiB L40S). Batch `8` OOMs during the frozen
Qwen backward pass even after encoder features are detached and only prompt
states are encoded. Batch `4` completes at about 35,943 MiB with gradients on
all 73 optimizer tensors. The minimal profile therefore uses batch `4` with
accumulation `32`, retaining effective batch `128`. Do not reuse the larger
legacy larger batches without a fresh capacity probe; weights mode also needs
its own probe because unfreezing the world path increases activation memory.

See [VECLAB_DATA_SPEC.md](VECLAB_DATA_SPEC.md) and the
[Qwen knowledge-injection specification](QWEN_KNOWLEDGE_INJECTION_SPEC.md#61-july-2026-causal-control-repair)
for experiment design and causal controls.
