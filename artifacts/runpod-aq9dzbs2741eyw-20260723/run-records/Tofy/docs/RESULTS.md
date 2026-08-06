# Results tracking

Update the **Best So Far** section when a run improves on a reported metric.
Record the exact command used. (Referenced by [AGENTS.md](../AGENTS.md).)

Historical entries that cite `--train-world`, `data/world_mix_pairs.txt`, or
in-repo Candle emitter checkpoints are **legacy-only** and are not runnable on
the current CLI. Current stages are `--train-world-knowledge`, `--train-bridge`,
and `--eval-bridge`.

## Best So Far

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

- Bridge eval `1784797740`: regime `weights`; report `runs/code_poc_1784364765/eval/setup_smoke.json`.

- Bridge eval `1784801231`: regime `weights`; report `runs/code_poc_1784364765/eval/world_conditioned_weights.json`.
