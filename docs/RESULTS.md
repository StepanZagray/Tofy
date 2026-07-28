# Results tracking

Update the **Best So Far** section when a run improves on a reported metric.
Record the exact command used. (Referenced by [AGENTS.md](../AGENTS.md).)

Historical entries that cite `--train-world`, `data/world_mix_pairs.txt`, or
in-repo Candle emitter checkpoints are **legacy-only** and are not runnable on
the current CLI. Current stages are `--train-world-knowledge`, `--train-bridge`,
and `--eval-bridge`.

## Best So Far

### 2026-07-23 rewrite baseline and failed-run preservation

The old experiment is not a valid knowledge-transfer result. Its context
channel passed `193/300` seen tasks (`64.33%`) and `0/300` held-out tasks; all
shuffled, swapped, and zero controls passed `0/300`. The selected weights
checkpoint reported CE `0.2584163` and semantic gap `0.70437014`, but its
generation validation covered only `4/300` seen tasks and no held-out tasks.
The selected world checkpoint reached validation prediction MSE
`0.0006566337` and association top-1 `1.0` at step 6,000, then the live
gradient norm rose from `0.0459` to `20.17` and `1564.02`. These observations
motivate the incompatible clean-slate architecture recorded in
[`MODEL_REWRITE_2026-07.md`](MODEL_REWRITE_2026-07.md).

The repaired prompt and compile-and-harness evaluator establish a usable
frozen-Qwen RAG ceiling on the full suite. Exact Go AST validation requires an
actual `veclab.<required>(...)` call; comments, strings, longer identifiers,
Unicode identifier continuations, and nested selectors do not qualify.

| Split | Tasks | Compiled | Passed | Suite pass |
|---|---:|---:|---:|---:|
| Seen | 300 | 149 | 105 | `0.3500` |
| Held-out | 300 | 129 | 125 | `0.4167` |

Seen command:

```bash
TOFY_EVAL_MODE=rag TOFY_EVAL_TASK_OFFSET=0 TOFY_EVAL_MAX_TASKS=300 TOFY_EVAL_FAILURE_CODE_LIMIT=20 TOFY_EVAL_MIN_PASS_RATE=0.05 TOFY_QWEN_CROSS_DIM=512 target/release/jepa_ai --eval-bridge models/qwen3-1.7b-base runs/nonexistent-control.safetensors runs/nonexistent-encoder.safetensors runs/nonexistent-vocab.txt runs/nonexistent-world.safetensors eval/veclab_eval.jsonl runs/rewrite_rag_ast_seen300.json
```

Held-out command:

```bash
TOFY_EVAL_MODE=rag TOFY_EVAL_TASK_OFFSET=300 TOFY_EVAL_MAX_TASKS=300 TOFY_EVAL_FAILURE_CODE_LIMIT=20 TOFY_EVAL_MIN_PASS_RATE=0.05 TOFY_QWEN_CROSS_DIM=512 target/release/jepa_ai --eval-bridge models/qwen3-1.7b-base runs/nonexistent-control.safetensors runs/nonexistent-encoder.safetensors runs/nonexistent-vocab.txt runs/nonexistent-world.safetensors eval/veclab_eval.jsonl runs/rewrite_rag_ast_heldout300.json
```

Reports and all failure-forensics logs were copied from pod
`aq9dzbs2741eyw-644122a3` to
`artifacts/runpod-aq9dzbs2741eyw-20260723/`. The report hashes are
`1dd513f25a647322bbccd34c0147f257fc2d73f2fd50430d7708e660a3b44b64`
(seen) and
`cb40f6512b9a3b4ebc051645e6107b26f6498e4cb179c01b8a40f51c72588e2a`
(held-out).

### Rewritten-model RTX PRO 6000 qualification and launch (2026-07-23)

Pod `aq9dzbs2741eyw-644122a3` exposes 97,887 MiB on an RTX PRO 6000
Blackwell Server Edition. Every probe used BF16, the production Muon/AdamW
optimizer, 1,024 SIGReg projections, actual cached VecLab sequences, and a full
backward update. The selected physical batch/accumulation pairs preserve the
minimal profile's effective batches:

| Stage | Failed larger batch | Selected batch/accum | Effective batch | Observed live peak/free |
|---|---:|---:|---:|---:|
| Encoder | none possible without changing effective batch | `64/1` | 64 | `94,465 / 2,786 MiB` |
| World | `128/2` OOM | `64/4` | 256 | `94,049 / 3,202 MiB` |
| Weights bridge | `32/4` OOM | `16/8` | 128 | `84,385 / 12,866 MiB` |

Selected encoder probe:

```bash
TOFY_TRAIN_DTYPE=bf16 TOFY_OPTIMIZER=muon TOFY_SIGREG_SLICES=1024 TOFY_SIGREG_POINTS=17 TOFY_SIGREG_POSITION_CHUNK=8 TOFY_USE_TOKEN_CACHE=1 TOFY_CACHE_DIR=data/cache TOFY_ENCODER_VOCAB=local_models/vocabs/vocab_encoder_8000_default.txt TOFY_REQUIRE_PREPARED_CACHE=1 TOFY_LATENT_WARMUP_BATCH=64 TOFY_LATENT_WARMUP_GRAD_ACCUM=1 TOFY_LATENT_WARMUP_STEPS=0 target/release/tofy --latent data/fictional/veclab_encoder_mix.txt 3 64 640 256 7 8 8000 --grad-accum 1 --output /workspace/tofy-rewrite-probe/latent-b64/model.safetensors
```

Selected world probe:

```bash
TOFY_TRAIN_DTYPE=bf16 TOFY_OPTIMIZER=muon TOFY_SIGREG_SLICES=1024 TOFY_SIGREG_POINTS=17 TOFY_SIGREG_POSITION_CHUNK=8 TOFY_USE_TOKEN_CACHE=1 TOFY_CACHE_DIR=data/cache TOFY_ENCODER_VOCAB=/workspace/tofy-rewrite-probe/latent-b64/model.vocab.txt TOFY_REQUIRE_PREPARED_CACHE=1 TOFY_WORLD_WARMUP_BATCH=64 TOFY_WORLD_WARMUP_GRAD_ACCUM=4 TOFY_WORLD_WARMUP_STEPS=0 TOFY_WORLD_LOG_EVERY=1 TOFY_WORLD_VAL_BATCHES=1 target/release/tofy --train-world-knowledge /workspace/tofy-rewrite-probe/latent-b64/model.safetensors /workspace/tofy-rewrite-probe/latent-b64/model.vocab.txt data/fictional/veclab_knowledge_train.txt 2 64 640 384 7 8 640 64 --lambda 0.09 --lr 2e-4 --grad-accum 4 --output /workspace/tofy-rewrite-probe/world-b64-g4/model.safetensors --encoder-output /workspace/tofy-rewrite-probe/world-b64-g4/model.encoder.safetensors
```

Selected worst-case weights-bridge probe:

```bash
TOFY_TRAIN_DTYPE=bf16 TOFY_OPTIMIZER=muon TOFY_ENCODER_DIM=640 TOFY_ENCODER_LAYERS=7 TOFY_ENCODER_HEADS=8 TOFY_BRIDGE_DIM=640 TOFY_NUM_LATENT_TOKENS=64 TOFY_ADAPTER_OUTPUT_SLOTS=64 TOFY_BRIDGE_MAX_SEQ=256 TOFY_QWEN_CROSS_DIM=512 TOFY_BRIDGE_REGIME=weights TOFY_KNOWLEDGE_UNFREEZE_WORLD=false TOFY_BRIDGE_ALIGNMENT_STEPS=0 TOFY_BRIDGE_GRAD_ACCUM=8 target/release/tofy --train-bridge "$TOFY_QWEN_DIR" /workspace/tofy-rewrite-probe/world-b64-g4/model.encoder.safetensors /workspace/tofy-rewrite-probe/latent-b64/model.vocab.txt /workspace/tofy-rewrite-probe/world-b64-g4/model.safetensors data/fictional/veclab_bridge_transfer.txt 1 16 /workspace/tofy-rewrite-probe/bridge-b16-g8/model.safetensors
```

The clean full experiment is `code_poc_1784808958`, tmux session
`tofy-rewrite-minimal`, with log
`/workspace/tofy-rewrite-minimal.log`. Launch command:

```bash
PROFILE=minimal SKIP_GIT_PULL=1 TMUX_SESSION=tofy-rewrite-minimal LOG_PATH=/workspace/tofy-rewrite-minimal.log scripts/runpod_train.sh train
```

The direct-documentation RAG gate passed on both splits and the encoder reached
step `200/20000`, including two validation and asynchronous best-checkpoint
saves. It held `87,873 / 97,887 MiB` VRAM at 97% GPU utilization and 301.55 W
during the first checkpoint; `val_pred` improved from `3.9503` at step 100 to
`2.3509` at step 200. This confirms that the selected `64/1` batch survives the
production validation path, not only the capacity probe. A local log snapshot
through step 200 is
`artifacts/runpod-aq9dzbs2741eyw-20260723/tofy-rewrite-minimal-step200.log`
(SHA-256
`9176a4098350cc84c547478252f0393e98ca9f827f5f02cc1877206bb729c645`).
Training later OOMed at encoder step `6600/20000` when the curriculum advanced
from `seq=128` to `seq=192` while still using physical batch `64`
(`CUDA_ERROR_OUT_OF_MEMORY` during latent backward). Resume state was saved at
step `6600` with best `val_pred=0.024110552`.

### Minimal profile VRAM retune and resume (2026-07-24)

The early qualification probes understated late-curriculum encoder memory
(short runs stay near `max_seq/2`). After the production OOM, `minimal` was
retuned to keep the same effective batches while fitting `active_seq=max_seq`
with context segments `4`:

| Stage | Previous batch/accum | Selected batch/accum | Effective batch |
|---|---:|---:|---:|
| Encoder | `64/1` | `16/4` | 64 |
| World | `64/4` | `32/8` | 256 |
| Weights bridge | `16/8` | `16/8` | 128 |

Pipeline warmup env defaults now mirror the profile's stage `grad_accum` so
warmup cannot silently shrink effective batch. Resume command on
`aq9dzbs2741eyw-644122a3`:

```bash
tmux new -d -s tofy-rewrite-minimal-resume "cd /workspace/Tofy && source \$HOME/.cargo/env && PROFILE=minimal RESUME_TARGET=code_poc_1784808958 SKIP_GIT_PULL=1 TOFY_REQUIRE_PREPARED_CACHE=0 TOFY_RUNPOD_TMUX_CHILD=1 LOG_PATH=/workspace/tofy-rewrite-minimal-resume.log bash scripts/runpod_train.sh resume"
```

Log: `/workspace/tofy-rewrite-minimal-resume.log`. Attach:
`tmux attach -t tofy-rewrite-minimal-resume`.

Confirmed live after resume: encoder continued at step `6700+` with
`seq=192`, peak VRAM about `50,081 / 97,887 MiB`, and best `val_pred`
improved to `0.009814121` by step `6800`.

On 2026-07-24 the encoder was still only using about half the card
(`~50 GiB`, ~67% util) at physical batch `16`, so `minimal` was bumped again
while keeping effective batches unchanged: encoder `32/2`, world `64/4`,
bridge `16/8`. Resume continued from step `11800`, then OOMed at step
`12400` with peak VRAM `94,817 / 97,887 MiB` still at `seq=192`. The profile
was restored to the proven-fit pair encoder `16/4`, world `32/8`, bridge
`16/8` and resumed from step `12400`.

Encoder finished `20000/20000` under that pair (`best val_pred≈0.00155`,
late `seq=256`, ~57 GiB). World then OOMed immediately at physical batch
`32/8` in-process after the encoder (standalone `32/8` later fit on a clean
GPU). On 2026-07-25 `minimal` was retuned again so every remaining stage fits
on a clean ~98 GiB card while preserving effective batches:

| Stage | Previous batch/accum | Selected batch/accum | Effective batch | Fit check |
|---|---:|---:|---:|---|
| Encoder | `16/4` | `16/4` | 64 | completed production run |
| World | `32/8` → `16/16` (live peaked `96,801 / 97,887 MiB` before step log) | `8/32` | 256 | live steps `1+` at ~`28 GiB` with `TOFY_CACHE_PREFETCH_CHUNK=16` |
| Context / weights bridge | `16/8` | `8/16` | 128 | 1-step OK (~50 GiB live) |

Resume from completed latent (`code_poc_1784808958`), skipping latent:

```bash
tmux new -d -s tofy-rewrite-minimal-resume "cd /workspace/Tofy && source \$HOME/.cargo/env && PROFILE=minimal RESUME_TARGET=code_poc_1784808958 SKIP_GIT_PULL=1 TOFY_REQUIRE_PREPARED_CACHE=0 SKIP_TRAINED_STAGES=latent TOFY_CACHE_PREFETCH_CHUNK=16 TOFY_RUNPOD_TMUX_CHILD=1 LOG_PATH=/workspace/tofy-rewrite-minimal-resume.log bash scripts/runpod_train.sh resume"
```

Confirmed live after the `8/32` retune: world steps `1`–`5` logged, peak observed about `28,385 / 97,887 MiB`.

### Minimal pre-launch reliability hardening (2026-07-26)

No new model-quality or long-run capacity result is claimed here. The
orchestrator was hardened before the next `minimal` run:

- minimal-profile training arms now recover only from attempt-bound,
  CUDA-specific allocation OOM reports by halving physical batch and doubling
  gradient accumulation; the actual pair and attempt history are saved
  atomically in `adaptive_batches.json`, and resume metadata enforces the
  unchanged effective batch. This is implementation validation, not a new
  sustained capacity result;
- every CUDA-owning RAG, encoder, world, bridge, evaluation, LoRA, and probe
  stage re-executes in a child process, so allocator state cannot leak across
  stage boundaries;
- both context-bridge nonqualification exits emit an attempt-scoped structured
  outcome and continue to the independent weights stage;
- full evaluation forces the complete suite, validates all 300 unique held-out
  task IDs and report provenance, recomputes paired causal tests from task
  outcomes with a six-comparison Bonferroni bound, and writes one aggregate
  success verdict after all independent arms;
- Step 2 requires at least 50% of the measured held-out RAG ceiling, paired
  causal significance, and a significant advantage over static prefix; Step 3
  requires a decisive nonzero held-out rate, paired causal significance, and
  parity with the rank-16 LoRA comparator (rank 512 remains diagnostic);
- resume now regenerates/restores inputs before provenance validation, rejects
  checkpoint tuples without matching generation IDs, holds run and shared-input
  locks, terminates orphaned Linux stage children with their parent, fails
  closed on missing dependent artifacts, and treats successful early stops as
  terminal;
- production training/evaluation fails immediately when CUDA device 0 is not
  available, and the RunPod tmux wrapper preserves nonsecret `TOFY_*`
  overrides.

Local validation covered formatting, compilation, Clippy with warnings denied,
the Rust test suite, shell syntax, and tmux argument propagation. Hardware
status remains unchanged: encoder `16/4` completed; world `8/32` has only the
five-step observation above and bridge `8/16` only a one-step fit check. Those
two pairs must be sustained-qualified on the target pod before being called
fully qualified.

### Effective-batch SIGReg correction and H100 restart (2026-07-26)

The first clean H100 80 GiB launch at commit `d872574` created
`code_poc_1785098309`. Its direct-documentation RAG preflight passed both
splits, and encoder training reached step 400 before being stopped
deliberately. At step 400 it reported total loss `0.6118`, prediction loss
`0.1535`, held-out prediction loss `0.1495`, and SIGReg `2.2919`, with a peak
of `44,725 / 81,559 MiB`. Exact launch:

```bash
TOFY_RUNPOD_TMUX_CHILD=1 SKIP_GIT_PULL=1 TOFY_REQUIRE_PREPARED_CACHE=0 LOG_PATH=/workspace/tofy-train-minimal.log TOFY_REPO_DIR=/workspace/Tofy scripts/runpod_train.sh train
```

This run is not a model-quality result and must not be resumed. Inspection
showed that gradient accumulation averaged independent physical-microbatch
SIGReg losses; it did not expose the Epps-Pulley statistic to the optimizer's
effective batch. The encoder therefore used only 16 independent sequences
(48 correlated three-view rows) per SIGReg estimate despite reporting an
effective optimizer batch of 64. The world stage had the same defect.

The replacement training path computes one seeded SIGReg objective over the
entire effective batch. It first linearizes the pooled detached objective in
position-bounded chunks, then replays one live physical microbatch at a time
and backpropagates its exact chain-rule contribution. Prediction gradients
remain averaged over accumulation; pooled SIGReg is applied once and is not
divided by accumulation. World validation now pools its validation batches as
well. CPU property tests compare pooled values and gradients against direct
full-batch Epps-Pulley evaluation, including unequal valid lengths.

The revised minimal pairs are encoder `16/8` (128 independent sequences, 384
three-view rows), world `8/32` (256 independent examples), and bridge `8/16`
(128 examples). Commit `0ba8e75` implemented the correction and was deployed
through Git as pod HEAD `09e3b5a` (the latter adds only preflight records).
Fresh run `code_poc_1785100782` passed the direct-documentation RAG preflight:
seen `104/300` (`0.3467`) and held-out `123/300` (`0.4100`). It then launched
the encoder at physical batch `16`, accumulation `8`, effective batch `128`;
`adaptive_batches.json` records the first attempt as running. An early health
check after roughly 85 seconds of encoder work showed `29,621 / 81,559 MiB`,
100% GPU utilization, and no OOM/error signature in
`/workspace/tofy-train-minimal-pooled.log`. This is startup evidence, not a
sustained qualification or a new best model metric.

The encoder subsequently completed all 20,000 steps. Its final preflight on
resume remained viable: seen evaluation `1785145737` passed `104/300`
(`0.3467`), and held-out evaluation `1785145835` passed `123/300` (`0.4100`).

World training exposed a separate peak in the pooled SIGReg linearization.
The `8/32` attempt ran from Unix `1785145835` to `1785149978`, then OOMed
inside `sigreg_epps_pulley_linearization_chunked_seeded` during
`Tensor::backward`. Automatic recovery preserved effective batch 256 and
retried `4/64`; it failed in the same call after running from `1785149978` to
`1785157448`. The surviving `2/128` attempt reached a complete step-2,000
checkpoint (`world-2000-676135-1785228114437820007`) with best validation
selection `0.3346264`; a live sample showed `69,429 / 81,559 MiB`, 67% GPU
utilization, and 130 W. This is checkpoint/fit evidence, not a qualified
model-quality result.

The old fixed-length linearizer built an autograd graph over
`[batch, position_chunk, 1024 projections, 17 knots]` and differentiated the
detached statistic. The replacement evaluates the same Epps-Pulley value and
its analytical input derivative while streaming projection chunks (default
128) and one knot at a time, then retains the existing live-microbatch replay
for the exact encoder/world chain rule. With the defaults, the dominant
temporary shrinks from 17,408 projection-knot elements per batch-position to
128, while the effective batch, sampled projections, knots, objective, and
optimizer schedule stay unchanged. The CPU parity test compares the value,
analytical input gradient, and replay gradient against the original full
autograd computation at `1e-5` tolerance. Full Rust tests, Clippy with
warnings denied, release build, formatting, and `git diff --check` pass
locally. An H100 `8/32` resume from step 2,000 remains required before this
optimization is hardware-qualified.

The first deployment also exposed a pre-training resume-loader bug: after the
step-2,000 checkpoint loaded, a blanket BF16 cast tried to copy BF16 storage
into the LeWorldModel projectors' intentionally F32 BatchNorm variables and
failed with `dtype mismatch in copy_strided op`. World checkpoints already
load into a model constructed with the exact mixed-dtype schema, so the
redundant post-load cast was removed. The failed attempt performed no optimizer
step and did not modify the checkpoint tuple.

After that loader fix, the first analytical SIGReg CUDA call rejected the
strided column view returned by slicing the projection matrix
(`matmul is only supported for contiguous tensors`). The implementation now
materializes only the active `640 x 128` projection chunk and its transpose as
contiguous tensors. This does not change the calculation or its bounded
workspace; the failed attempt occurred before the optimizer step.

### `code_poc_1784364765` decoder-transfer result (2026-07-21)

The completed pod evaluations exposed a decoder-grounding bottleneck. The
context bridge scored `193/300` seen tasks (`0.6433`, compile `0.7133`) with
`0/300` for shuffled, swapped, and zeroed controls, but it scored `0/300` on
every held-out condition. The weights bridge and RAG ceiling also scored
`0/300` on both splits. The latter generated repetitive instruction text, so it
is not a valid capability ceiling.

The next run changes bridge training from seen-code-only supervision to an
explicit world-to-decoder transfer curriculum. It trains Qwen's adapter and
cross-attention to decode query-conditioned world states into documentation for
all 200 functions, while code completions remain restricted to functions 1--80
and code validation remains functions 81--100. Held-out functions 101--200
therefore contribute world knowledge and decoder grounding, but never code
solutions. The generated two-field curriculum is
`data/fictional/veclab_bridge_transfer.txt`; the pipeline rebuilds it from the
action-labelled world corpus plus the seen task corpus before bridge training.

This supersedes resuming either existing bridge checkpoint: their optimizer
state and sample cursor describe the old task-only distribution. The retained
encoder and qualified step-6,000 world checkpoint remain usable.

The replacement weights bridge started from step zero in tmux session
`tofy-world-qwen-transfer-v2`, logging to
`/workspace/tofy-world-qwen-transfer-v2.log`. Preflight reports 10,800 training
rows, 800 function-disjoint code-validation rows, 180,338,699 trainable
parameters, gradients on 218/230 tensors, global norm `0.249756`, and 30,046
MiB live VRAM. Exact command:

```bash
TOFY_TRAIN_DTYPE=bf16 TOFY_ENCODER_DIM=640 TOFY_ENCODER_LAYERS=7 TOFY_ENCODER_HEADS=8 TOFY_BRIDGE_DIM=640 TOFY_NUM_LATENT_TOKENS=64 TOFY_ADAPTER_OUTPUT_SLOTS=64 TOFY_BRIDGE_MAX_SEQ=256 TOFY_BRIDGE_REGIME=weights TOFY_KNOWLEDGE_UNFREEZE_WORLD=true TOFY_BRIDGE_GRAD_ACCUM=128 TOFY_BRIDGE_LOG_EVERY=10 TOFY_BRIDGE_VAL_EVERY=100 ./target/release/jepa_ai --train-bridge /workspace/Tofy/models/qwen3-1.7b-base runs/code_poc_1784364765/world/model.encoder.safetensors runs/code_poc_1784364765/latent/model.vocab.txt runs/code_poc_1784364765/world/model.safetensors data/fictional/veclab_bridge_transfer.txt 20000 1 runs/code_poc_1784364765/bridge_transfer_v2/weights.safetensors --seed 42
```

### RTX PRO 6000 hardware qualification (2026-07-22)

The resumed bridge run uses the RTX PRO 6000 Blackwell Server Edition (96 GiB).
To preserve its original effective batch of 128, batch `16` with accumulation
`8` was tested but failed in `scaled_gradients` during the first full backward
pass after its preflight, despite allocating successfully. The largest stable
configuration is batch `8`, accumulation `16`, which uses 54,582 MiB during
the preflight and keeps the same optimizer schedule and data cursor. Exact
command:

```bash
TOFY_TRAIN_DTYPE=bf16 TOFY_ENCODER_DIM=640 TOFY_ENCODER_LAYERS=7 TOFY_ENCODER_HEADS=8 TOFY_BRIDGE_DIM=640 TOFY_NUM_LATENT_TOKENS=64 TOFY_ADAPTER_OUTPUT_SLOTS=64 TOFY_BRIDGE_MAX_SEQ=256 TOFY_BRIDGE_REGIME=weights TOFY_KNOWLEDGE_UNFREEZE_WORLD=true TOFY_BRIDGE_GRAD_ACCUM=16 TOFY_BRIDGE_LOG_EVERY=10 TOFY_BRIDGE_VAL_EVERY=100 ./target/release/jepa_ai --train-bridge /workspace/Tofy/models/qwen3-1.7b-base runs/code_poc_1784364765/world/model.encoder.safetensors runs/code_poc_1784364765/latent/model.vocab.txt runs/code_poc_1784364765/world/model.safetensors data/fictional/veclab_bridge_transfer.txt 20000 8 runs/code_poc_1784364765/bridge_transfer_v2/weights.safetensors --seed 42 --resume
```

The run ended normally through early stopping at step 7,500 (of 20,000), with
no accelerator fault. The best observed held-out semantic gap was `1.6219`;
the selected atomic checkpoint at step 4,100 retains selection CE `0.2584163`
and qualifying semantic gap `0.70437014`. The final validation at step 7,500
was CE `0.3804` and semantic gap `1.1096`. Training used about 70 GiB during
the full workload. No post-training evaluation has been started.

The first Qwen knowledge-injection ladder completed on `code_poc_1783547471`,
but its causal controls invalidate the world-knowledge claim. It remains the
baseline that the repaired semantic-negative run must beat.
World training uses `--train-world-knowledge` with explicit per-function
paraphrase train/validation splits; bridge training uses `--train-bridge` and
produces separate context and weights checkpoints.

**Closed control — do not rerun decoder-only evaluation.** The frozen/base
decoder was already evaluated on all 600 VecLab tasks for
`code_poc_1783547471` and scored `0.0000` on both seen and held-out suite pass
rates. With the base decoder, tokenizer, prompt contract, and suite unchanged,
another decoder-only run is redundant and must not be scheduled as part of a
resume or replacement run. Subsequent evaluation should start only after a
bridge checkpoint qualifies and must compare matched, wrong, and zeroed
conditioning. Reopening this control requires a deliberately changed component
and a new experiment record.

Set `TOFY_STATIC_SOFT_PREFIX=true` for the equal-slot static-prefix control.
Set `TOFY_QWEN_LORA_RANK=16` for the Q/V LoRA control. The pipeline gives this
control the same fictional documentation plus seen gold tasks. `train ... --until full`
runs all controls, the all-function held-out-paraphrase channel probe, and
RAG/latent/weights evaluations. It never runs the decoder-only floor; that
closed control is available only through the manual
`scripts/run_veclab_decoder_floor.sh` entry point.

| Run | Regime | Trainable parameters | Seen pass | Held-out pass |
|---|---|---:|---:|---:|
| `code_poc_1783547471` | context | 124,159,368 | 0.7033 | 0.2767 |
| `code_poc_1783547471` | weights | 124,159,368 | 0.2833 | 0.0600 |
| Static prefix | prefix | emitted as `trainable_params` with `TOFY_STATIC_SOFT_PREFIX=true` | pending | pending |
| LoRA r=16 | Q/V LoRA | emitted as `trainable_params` with `TOFY_QWEN_LORA_RANK=16` | pending | pending |
| LoRA r=512 | Q/V LoRA capacity control | emitted as `trainable_params` with `TOFY_QWEN_LORA_RANK=512` | pending | pending |

The contamination-floor/decoder-only control is already complete and closed;
do not rerun it for an unchanged experiment. The generator itself is
deterministic: `cargo run --release -- --prepare-veclab --seed 20260705 --out data/fictional`.

**VecLab corpus v1.2.0 (seed 20260705):** `data/fictional/MANIFEST.json` SHA-256
`veclab_encoder_mix.txt` = `d93f0817ec918e0cba5da42a8a3c6fbb76219ce204a1b61a28f449bdebbf15b9`;
`veclab_knowledge_train.txt` = `face1f84d1ef7dc5497a0c6fb523365ca8116ca33e9c69ec7e16de98dad597ce`;
`veclab_knowledge_val.txt` = `036342a2842222e36954442d2e34ee5b86579f044f686178b4fb3210fa93b11d`;
200 functions, 600 eval cases, 0 held-out gold rows upstream (`--print-split-stats`).
**Step-0 zero-shot floor (RTX 5060 smoke, Qwen3-1.7B-Base BF16):** held-out
suite pass `0/5` (`0.0000`), compile rate `0/5` (`0.0000`); failures were four
`must_call_violation` and one `compile_error`. Report:
`runs/bridge_eval/5060_floor_heldout5/report.json`. Exact command:

```bash
TOFY_EVAL_MODE=floor TOFY_EVAL_TASK_OFFSET=300 TOFY_EVAL_MAX_TASKS=5 TOFY_ENCODER_DIM=640 TOFY_ENCODER_LAYERS=7 TOFY_ENCODER_HEADS=8 TOFY_BRIDGE_DIM=640 TOFY_NUM_LATENT_TOKENS=64 TOFY_ADAPTER_OUTPUT_SLOTS=64 TOFY_BRIDGE_MAX_SEQ=64 cargo run --release -- --eval-bridge local_models/Qwen3-1.7B-Base local_models/smoke/bridge.safetensors local_models/smoke/encoder.safetensors local_models/smoke/encoder.vocab.txt local_models/smoke/world.safetensors eval/veclab_eval.jsonl runs/bridge_eval/5060_floor_heldout5/report.json
```

This is a five-task contamination smoke check, not a statistically powered full-suite result.
The required near-zero floor was confirmed.

### `code_poc_1784356720` encoder run (invalidated by tokenizer collapse)

The RunPod run was manually stopped near encoder step `10200/20000`; preserve
its logs for diagnosis, but do not resume or compare its checkpoints. Although
all 20,000 source rows contain 27--55 whitespace-delimited words, the generated
boundaryless-BPE cache contained 14,449 one-token records (`72.245%`) and
16,931 records of at most four tokens (`84.655%`). The learned vocabulary had
memorized complete repetitive VecLab prompt templates as individual tokens.
Masking a one-token record hides its only token, making both JEPA views the
same all-mask input. This explains the repeating `targets=16`, zero
chunk/global losses, cosine `1.0`, and unstable negative prediction cosine.

Tokenizer specification v9 prevents merges across lexical boundaries, cache
construction rejects records shorter than two tokens, and preparation reports
minimum/mean/maximum token fertility. The latent trainer now restores held-out
validation when using a cached vocabulary and averages logging across every
gradient-accumulation microbatch. These changes alter vocabulary semantics, so
a clean encoder and world run is required.

Exact command for the invalidated run:

```bash
TOFY_QWEN_DIR=/workspace/Tofy/models/qwen3-1.7b-base TOFY_REQUIRE_PREPARED_CACHE=0 TOFY_TRAIN_DTYPE=bf16 ./target/release/jepa_ai train minimal --until full
```

Replacement run `code_poc_1784362609` was stopped at encoder step
`2800/20000` and invalidated by the full data-shape audit. Token shapes were
healthy (20,000 cache rows, fertility min `83`, mean `123.98`, max `311`, zero
cross-boundary tokens), but the raw corpus contained 16,090 duplicate encoder
rows, 6,600 duplicate world-train rows, 200 duplicate world-validation rows,
and 2,281 duplicate bridge rows. Thus nominal row counts substantially
overstated effective dataset size, and validation weighted identical queries
twice. Generator v1.2 produces unique row variants and refuses duplicates.
The same audit also found that Stage-5 LoRA construction concatenated
three-field world rows into a two-field bridge loader; it now strips and
validates the action column explicitly. Do not resume this run. Launch command:

```bash
TOFY_QWEN_DIR=/workspace/Tofy/models/qwen3-1.7b-base TOFY_REQUIRE_PREPARED_CACHE=0 TOFY_TRAIN_DTYPE=bf16 ./target/release/jepa_ai train minimal --until full
```

Replacement run `code_poc_1784364765` uses generator v1.2.0 and started from
scratch in tmux session `tofy-leworld-shaped-20260718`, logging to
`/workspace/tofy-leworld-shaped-20260718.log`. Preflight verified 20,000
unique encoder rows, 7,600 unique world-train queries, 400 unique and
train-disjoint world-validation queries, and 4,000 unique rows in each bridge
task partition. The rebuilt 1,591-token cache round-trips every record exactly:
encoder lengths `83..311` (mean `135.45`), world-state `20..87` (mean `36.91`),
and world-target `75..138` (mean `106.39`). This is data qualification, not a
quality metric. Encoder training is live and continued through step `500` with
`569..646` real masked targets per logged optimizer step, nonzero train and
held-out prediction losses, and 13,031 MiB peak VRAM. At step `500`, train
prediction loss was `1.2904` and held-out prediction loss was `1.3719`.
It continued to step `4100/20000` and then stopped with
`CUDA_ERROR_OUT_OF_MEMORY` immediately after the scheduled transition from
warmup accumulation to batch `16`, gradient accumulation `4`. The L40S had
used 31,879 MiB at the first full-accumulation step; the next backward pass
could not allocate its additional gradient storage. The best recorded
held-out prediction loss before the stop was `0.5246` at step `4100`. This is
a capacity/configuration failure, not a data-shape failure; do not report it
as a completed encoder result.

The encoder resumed at batch `8`, gradient accumulation `8`, completed all
20,000 steps, and retained its best held-out prediction checkpoint from step
17,200 (`val_pred=0.040172406`). World training then produced a qualified best
checkpoint at step 6,000: held-out total `0.1450909`, prediction MSE
`0.0006566337`, SIGReg `1.6048250`, and association top-1 `1.0000`. Its train
snapshot was total `0.0606249`, prediction `0.0014344`, SIGReg `0.6576718`,
association top-1 `1.0000`, with global gradient norm `0.0459`.

The live world weights became unstable after that checkpoint: global gradient
norm rose to `20.17` at step 7,000 and `1564.02` at step 8,000, while held-out
total regressed to `1.4965` and association top-1 fell to `0.0352`. The run did
not recover through the step-16,500 resume sidecar. It was manually stopped;
the atomic step-6,000 world and matched encoder exports were preserved, and the
pipeline advanced to bridge training with:

```bash
TOFY_QWEN_DIR=/workspace/Tofy/models/qwen3-1.7b-base TOFY_REQUIRE_PREPARED_CACHE=0 TOFY_TRAIN_DTYPE=bf16 TOFY_EVAL_LADDER=rag_ceiling,latent_channel,knowledge_in_weights,static_prefix,lora_r16,lora_r512,channel_probe ./target/release/jepa_ai train minimal --until full --resume code_poc_1784364765 --skip-trained world
```

The context bridge subsequently qualified and early-stopped at step `2,100`
(selection score `1.3864`, semantic gap `0.7441`). The first unfrozen
weights-mode backward OOMed at physical batch `4`, accumulation `32`; its
preflight had passed with `218/230` gradients and global norm `0.595932`.
On 2026-07-19 it was resumed on the same 46,068 MiB L40S with physical batch
`1`, accumulation `128` (the same effective batch `128`), explicitly skipping
the retained world step-6,000 and context checkpoints. The new preflight passed
with `218/230` gradients and global norm `0.595932`, using 30,663 MiB. Live
training reached step `2,220` without an OOM. Its best live weights validation
semantic gap so far is `1.3347` at step `2,100` (matched `4.5651`, wrong
`5.8998`, zero gap `1.5970`), and it published the qualifying weights and
joint-world checkpoints. This is a training metric, not yet a completed
evaluation. Live command:

```bash
TOFY_QWEN_DIR=/workspace/Tofy/models/qwen3-1.7b-base TOFY_REQUIRE_PREPARED_CACHE=0 TOFY_TRAIN_DTYPE=bf16 TOFY_EVAL_LADDER=rag_ceiling,latent_channel,knowledge_in_weights,static_prefix,lora_r16,lora_r512,channel_probe TOFY_WEIGHTS_BRIDGE_BATCH=1 TOFY_WEIGHTS_BRIDGE_GRAD_ACCUM=128 ./target/release/jepa_ai train minimal --until full --resume code_poc_1784364765 --skip-trained world,bridge_context
```

### LeWorldModel RunPod qualification (smoke, not a quality result)

Pod `0ynxc2a65zgog1-64410ad3` is healthy: its L40S exposes 46,068 MiB and the
successful world run sustained high utilization. The failures were in the
experiment implementation: three-field action-labelled rows were rejected by
an obsolete two-field preflight, BatchNorm mixed BF16 activations with F32
statistics incorrectly, gradient clipping summed mixed-dtype tensors, and the
bridge retained encoder/gold-completion work that could not affect its output.
Those paths are repaired. The world checkpoint format is intentionally
incompatible with pre-LeWorldModel checkpoints.

A production-shaped world step completed at batch `32`, accumulation `8`
(effective batch `256`), using 30,055 MiB at the sample and reporting
`total=3.1621`, `prediction=1.6230`, `sigreg=17.1007`, diagnostic
`association_top1=0.031`, and one-batch `val_selection=4.0202`. These values
qualify execution and gradients only; one step is not evidence of model
quality. Exact command:

```bash
TOFY_USE_TOKEN_CACHE=0 TOFY_TRAIN_DTYPE=bf16 TOFY_WORLD_LOG_EVERY=1 TOFY_WORLD_VAL_BATCHES=1 TOFY_CHECKPOINT_EVERY=10 ./target/release/jepa_ai --train-world-knowledge runs/code_poc_1784148037/latent/model.safetensors runs/code_poc_1784148037/latent/model.vocab.txt data/fictional/veclab_knowledge_train.txt 1 32 640 384 7 8 640 64 --lambda 0.09 --lr 2e-4 --grad-accum 8 --output runs/lewm_smoke_b32_g8/model.safetensors --encoder-output runs/lewm_smoke_b32_g8/model.encoder.safetensors
```

Bridge batch `8` still OOMed during frozen-Qwen backward. Batch `4` with
accumulation `32` preserved effective batch `128`, used about 35,943 MiB, and
completed the optimizer step with gradients on `73/73` tensors and global norm
`0.509470`. Its one-step validation semantic gap was negative, so the stage
correctly refused to publish a best checkpoint and retained only
`context.latest.safetensors`. This is capacity qualification, not a quality
result. Exact command:

```bash
TOFY_TRAIN_DTYPE=bf16 TOFY_ENCODER_DIM=640 TOFY_ENCODER_LAYERS=7 TOFY_ENCODER_HEADS=8 TOFY_BRIDGE_DIM=640 TOFY_NUM_LATENT_TOKENS=64 TOFY_ADAPTER_OUTPUT_SLOTS=64 TOFY_BRIDGE_MAX_SEQ=256 TOFY_BRIDGE_REGIME=context TOFY_KNOWLEDGE_UNFREEZE_WORLD=false TOFY_BRIDGE_GRAD_ACCUM=32 TOFY_BRIDGE_LOG_EVERY=1 TOFY_BRIDGE_VAL_EVERY=100 ./target/release/jepa_ai --train-bridge /workspace/Tofy/models/qwen3-1.7b-base runs/lewm_smoke_b32_g8/model.encoder.safetensors runs/code_poc_1784148037/latent/model.vocab.txt runs/lewm_smoke_b32_g8/model.safetensors data/fictional/veclab_tasks_train.txt 1 4 runs/lewm_bridge_b4_g32/context.safetensors --seed 42
```

### `code_poc_1784148037` repaired-causal run (failed quality result)

This run produced qualifying teacher-forced checkpoints but failed the actual
generation criterion. Context bridge training early-stopped at step `2600` with
selected validation CE `0.5253479` and full-completion semantic gap `0.026195109`.
The jointly aligned weights bridge stopped at step `4300`; its selected
validation CE was `0.36094847` and selected full-completion semantic gap was
`0.03429568` (best observed gap `0.1243`). The old plateau path incorrectly
returned a fatal error despite exceeding the required `0.02` gap.

The full weights evaluation scored seen pass `0/300` and held-out pass `0/300`.
Matched compile rate was `0.0567` on both subsets; zeroed compile rate was zero.
All paired matched advantages were `0.0000`. Generated programs often had a
valid wrapper but hallucinated or combined the wrong VecLab identifiers. This
showed that full-completion negative CE was dominated by shared Go boilerplate:
it measured a conditioning-present effect without directly supervising the
task-defining return expression. The repaired objective now keeps full-sequence
CE for syntax but applies matched-vs-wrong margin and unlikelihood losses to the
fictional API identifier tokens in that expression (falling back to the whole
expression for other targets). Bridge-mode evaluation also now uses the same
counterfactual signature-only prompt as bridge training, removing the former
prompt distribution shift and explicit-name causal leak.

Training command recorded by the run:

```bash
./target/release/jepa_ai train minimal --until full --resume code_poc_1784148037
```

Exact weights evaluation command:

```bash
TOFY_TRAIN_DTYPE=bf16 TOFY_ENCODER_DIM=640 TOFY_ENCODER_LAYERS=7 TOFY_ENCODER_HEADS=8 TOFY_BRIDGE_DIM=640 TOFY_NUM_LATENT_TOKENS=64 TOFY_ADAPTER_OUTPUT_SLOTS=64 TOFY_BRIDGE_MAX_SEQ=256 TOFY_EVAL_MODE=bridge TOFY_BRIDGE_REGIME=weights ./target/release/jepa_ai --eval-bridge "$TOFY_QWEN_DIR" runs/code_poc_1784148037/bridge/weights.best.safetensors runs/code_poc_1784148037/world/model.encoder.safetensors runs/code_poc_1784148037/latent/model.vocab.txt runs/code_poc_1784148037/bridge/weights.best.world.safetensors eval/veclab_eval.jsonl runs/code_poc_1784148037/eval/world_conditioned_weights.json
```

### `code_poc_1783869616` repaired causal attempt (superseded, not a result)

The world stage was manually stopped at `17,000/20,000` after its selection
metric plateaued; its best observed selection score was `0.123667315` near step
14k. The first context bridge reached roughly step 3,700 before being stopped
because matched and wrong-state validation CE remained effectively equal while
zeroed CE differed--another generic nonzero-prefix collapse. It produced no
checkpoint meeting the required `wrong_ce - matched_ce >= 0.02` semantic gap,
so it must not be evaluated or added to **Best So Far**. The corrective
counterfactual/centred-adapter objective is intentionally started from a new
run root, not resumed from these incompatible bridge weights.

Command used for the discarded context attempt:

```bash
./target/release/jepa_ai train minimal --resume runs/code_poc_1783869616 --skip-trained world
```

### `code_poc_1783547471` full causal evaluation (invalidated baseline)

The 600-task base and RAG controls both scored `0.0000` seen/held-out pass.
Context matched scored `0.7033` seen and `0.2767` held-out, but shuffled scored
`0.7267` and `0.2833`. On held-out paired outcomes, matched-only was `0/300`
and shuffled-only `2/300`; all 83 matched passes were explicit-name tasks and
implicit held-out pass was `0/200`. Weights matched scored `0.2833` seen and
`0.0600` held-out; held-out matched-only and shuffled-only were both `1/300`.
Zeroed conditioning scored `0.0000`, proving that the bridge turns behavior on
but not that it transmits the correct task knowledge. These checkpoints used
`TOFY_DECODER_CONDITIONING_NEGATIVES=none`; their world stage also saw function
metadata. Start the repaired experiment from a new run root rather than resuming.

Exact context evaluation command:

```bash
TOFY_TRAIN_DTYPE=bf16 TOFY_ENCODER_DIM=640 TOFY_ENCODER_LAYERS=7 TOFY_ENCODER_HEADS=8 TOFY_BRIDGE_DIM=640 TOFY_NUM_LATENT_TOKENS=64 TOFY_BRIDGE_MAX_SEQ=256 TOFY_EVAL_MODE=bridge TOFY_BRIDGE_REGIME=context ./target/release/jepa_ai --eval-bridge /workspace/Tofy/models/qwen3-1.7b-base runs/code_poc_1783547471/bridge/context.best.safetensors runs/code_poc_1783547471/world/model.encoder.safetensors runs/code_poc_1783547471/latent/model.vocab.txt runs/code_poc_1783547471/world/model.safetensors eval/veclab_eval.jsonl runs/code_poc_1783547471/eval/latent_channel.json
```

Exact weights evaluation command:

```bash
TOFY_TRAIN_DTYPE=bf16 TOFY_ENCODER_DIM=640 TOFY_ENCODER_LAYERS=7 TOFY_ENCODER_HEADS=8 TOFY_BRIDGE_DIM=640 TOFY_NUM_LATENT_TOKENS=64 TOFY_BRIDGE_MAX_SEQ=256 TOFY_EVAL_MODE=bridge TOFY_BRIDGE_REGIME=weights ./target/release/jepa_ai --eval-bridge /workspace/Tofy/models/qwen3-1.7b-base runs/code_poc_1783547471/bridge/weights.best.safetensors runs/code_poc_1783547471/world/model.encoder.safetensors runs/code_poc_1783547471/latent/model.vocab.txt runs/code_poc_1783547471/world/model.safetensors eval/veclab_eval.jsonl runs/code_poc_1783547471/eval/knowledge_in_weights.json
```

The legacy metrics below were produced before the strict LeJEPA default rewrite.
Treat them as legacy baselines until a strict run reports better metrics with
its exact command.

- **World transition selection score:** `0.0041821287` at step `17000/60000` on `data/world_mix_pairs.txt` validation stream, logged peak VRAM `7280/8151 MB` — legacy pre-CLI run with `DIM=640`, `LAYERS=7`, `HEADS=8`, `WORLD_BATCH=128`, `WORLD_GRAD_ACCUM=1`, `TOFY_TRAIN_DTYPE=bf16`. This is not the current default 8 GB schedule.
- **World training throughput:** `1.2209 steps/s` (`0.8191 s/step`) over logged steps `18000 -> 21000` on `data/world_mix_pairs.txt`, batch `128x1`, from `runs/code_poc_2026-04-21_21-50-54/world/events.out.tfevents.1776804849.zephyrus.25746.0`. This is about `1.98x` faster than `0.6170 steps/s` (`1.6208 s/step`) from `runs/code_poc_2026-04-20_22-32-35/world/events.out.tfevents.1776732255.zephyrus.86212.0` with the same world shape. Command: legacy pre-CLI resume run whose world stage invoked `target/release/jepa_ai --train-world local_models/model_latent_39.53M.safetensors local_models/model_latent_39.53M.vocab.txt data/world_mix_pairs.txt 60000 128 640 256 7 8 640 64 --lambda 0.2 --lr 2e-4 --grad-accum 1 --action-loss-weight 1.0 --router-warmup 5000 --resume`, with default token-cache prefetch enabled (`TOFY_CACHE_PREFETCH_BATCHES` unset, default `2`; `TOFY_TOKEN_CACHE_READER_MB` unset, default `8`).

## Observed Baseline Runs

These were existing TensorBoard runs in `runs/` before the current training-code fixes. They are useful as a baseline, but they do not have the exact launch commands recorded, so they are not listed under **Best So Far**.

- `runs/code_poc_1783369888` (`train minimal --until full`) is invalid. Candle
  0.10's fused LayerNorm/RMSNorm kernels are forward-only; normalization at the
  adapter output, Qwen output, compressor output, and reconstruction head severed
  autograd. Consequently all bridge optimizer moments and gates remained at zero,
  the encoder collapsed under normalized-MSE/stop-gradient training, and world
  association stayed at batch-32 chance (`3.125%`). Do not compare its metrics to
  corrected runs or resume its checkpoints; corrected encoder checkpoints add
  post-normalization projection parameters.

- `runs/world/1775175560`: raw `metrics/action_acc` stayed near 1.0 while `metrics/code_rate` stayed near 0 and `metrics/pred_code_rate` stayed at 0. This indicates majority-class router collapse, not healthy routing.
- `runs/latent/1775164725`: `loss/pred_token` worsened while `metrics/chunk_cosine` and `metrics/global_cosine` stayed near 1.0, which suggests the masking/task balance was too easy and the multiscale metrics were flattering.

## Bridge eval notes

- Bridge eval `1784322552`: regime `weights`; report `runs/code_poc_1784148037/eval/world_conditioned_weights.json`.

- Bridge eval `1784803983`: regime `rag`; report `runs/local_rag_scaffold_seen10.json`.

- Bridge eval `1784804077`: regime `rag`; report `runs/local_rag_scaffold_heldout10.json`.

- Bridge eval `1785098478`: regime `rag`; report `runs/code_poc_1785098309/eval/rag_preflight_seen.json`.

- Bridge eval `1785098580`: regime `rag`; report `runs/code_poc_1785098309/eval/rag_preflight_heldout.json`.

- Bridge eval `1785100929`: regime `rag`; report `runs/code_poc_1785100782/eval/rag_preflight_seen.json`.

- Bridge eval `1785101027`: regime `rag`; report `runs/code_poc_1785100782/eval/rag_preflight_heldout.json`.
