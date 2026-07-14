# Results tracking

Update the **Best So Far** section when a run improves on a reported metric. Record the exact command used. (Referenced by [AGENTS.md](../AGENTS.md).)

## Best So Far

The first Qwen knowledge-injection ladder completed on `code_poc_1783547471`,
but its causal controls invalidate the world-knowledge claim. It remains the
baseline that the repaired semantic-negative run must beat.
World training uses `--train-world-knowledge` with explicit per-function
paraphrase train/validation splits; bridge training uses `--train-bridge` and
produces separate context and weights checkpoints.
Set `TOFY_STATIC_SOFT_PREFIX=true` for the equal-slot static-prefix control.
Set `TOFY_QWEN_LORA_RANK=16` for the Q/V LoRA control. The pipeline gives this
control the same fictional documentation plus seen gold tasks. `train ... --until full`
runs all controls, the all-function held-out-paraphrase channel probe, and
floor/RAG/latent/weights evaluations.

| Run | Regime | Trainable parameters | Seen pass | Held-out pass |
|---|---|---:|---:|---:|
| `code_poc_1783547471` | context | 124,159,368 | 0.7033 | 0.2767 |
| `code_poc_1783547471` | weights | 124,159,368 | 0.2833 | 0.0600 |
| Static prefix | prefix | emitted as `trainable_params` with `TOFY_STATIC_SOFT_PREFIX=true` | pending | pending |
| LoRA r=16 | Q/V LoRA | emitted as `trainable_params` with `TOFY_QWEN_LORA_RANK=16` | pending | pending |
| LoRA r=512 | Q/V LoRA capacity control | emitted as `trainable_params` with `TOFY_QWEN_LORA_RANK=512` | pending | pending |

Before training, run the contamination floor with
`TOFY_EVAL_MODE=floor ... --eval-bridge ... eval/veclab_eval.jsonl`; record the
exact command and verify near-zero held-out pass rate here. The generator itself is
deterministic: `cargo run --release -- --prepare-veclab --seed 20260705 --out data/fictional`.

**VecLab corpus (seed 20260705):** `data/fictional/MANIFEST.json` SHA-256
`veclab_encoder_mix.txt` = `450c4a7518f1f9f926d25e9a794eaeb1bc1f9002754009e46ab060dbe03ab9eb`;
`veclab_knowledge_train.txt` = `4b14e7c862180350939b0bb46870b54500cae10c644969f108e3a8f98b66f502`;
`veclab_knowledge_val.txt` = `7983e7eb977cf94ccf33aa03b1ec29bd16c7768bba728895bc1a7860cfefa7a9`;
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
- **Decoder GPU utilization:** sustained `100%` on five `nvidia-smi` samples spaced 5 seconds apart, with `41961/46068 MiB` used on RunPod `m1m5vxqjjsr7ju` after removing decoder CPU synchronization points. Run root: `runs/code_poc_1779626851`, resumed from decoder step `900`. Training command: `./target/release/jepa_ai train 48gb --resume latest 2>&1 | tee /workspace/tofy-train-48gb-resume-gpu-fix.log`. Measurement command: `for i in 1 2 3 4 5; do nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; sleep 5; done`.
- **Zero-conditioned code decoder constraint pass rate:** `0.3000` (`3/10`) with `suite_pass_rate=0.0000`, `compile_rate=0.0000`, and `route_code_acc=1.0000` on `eval/code_assistant_rust_hard.jsonl`, from `runs/code_eval/1776932354`. This used the current 68.50M code decoder with the context/state conditioning vector filled with zeros; passing constraints were `merge_intervals`, `top_k_words`, and `compact_sorted_numbers`, but none compiled. Command: `cargo run --release -- --eval-code-assistant local_models/model_latent_39.53M.safetensors local_models/model_latent_39.53M.vocab.txt local_models/model_world_13.58M.safetensors eval/code_assistant_rust_hard.jsonl 384 640 256 7 8 640 64 --code-decoder local_models/code_decoder_poc_68.50M.safetensors --ablate-conditioning`.

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
- `runs/decoder/1775190816`: text decoder improved but still ended with triple-digit perplexity, so text-decoder capacity/data shaping needed work.
- `runs/decoder/1775183480`: code decoder learned meaningfully and was the strongest stage of the pipeline, but still not at a strong code-model quality bar.

## RunPod Results

### RunPod `slpg14xbgt0g4x` 48 GB pipeline, May 16-18 2026

Source log: `/workspace/tofy-train-48gb.log` on RunPod `slpg14xbgt0g4x`. Run root: `runs/code_poc_1778927222`. The run `launch.txt` records `command=train 48gb`; equivalent repo command:

```bash
cargo run --release -- train 48gb
```

Pipeline metadata: `profile=48gb`, `dim=768`, `layers=12`, `heads=12`, `bridge_dim=768`, `num_latent_tokens=96`, decoder `dim=768`, decoder `layers=12`, decoder `heads=12`, decoder `ff=3072`, `with_code_eval=false`.

Completed stages:

- Latent: `75000` steps, batch `32x1`, checkpoint `runs/code_poc_1778927222/latent/model.safetensors`, logged peak VRAM `20046/97887 MB`.
- World: `180000` steps, batch `32x1`, checkpoint `runs/code_poc_1778927222/world/model.safetensors`, encoder checkpoint `runs/code_poc_1778927222/world/model.encoder.safetensors`, logged peak VRAM `20319/97887 MB`.
- High-world: `36000` steps, checkpoint `runs/code_poc_1778927222/high_world/model.safetensors`, logged peak VRAM `10128/97887 MB`.
- Code decoder base: `120000` steps on `928039` train rows / `48845` val rows, vocab `8720`, `max_seq=192`, selected checkpoint `runs/code_poc_1778927222/decoder_code/model.safetensors` with best selection loss `1.9219`; final logged step `120000/120000` had `token_ce=2.5312`, `ppl=12.57`, `tok_acc=29.30%`, peak VRAM `5633/97887 MB`.
- Code decoder polish: `24000` steps initialized from the base decoder, selected checkpoint `runs/code_poc_1778927222/decoder_code_polish/model.safetensors` with best selection loss `10.3125`; final logged step `24000/24000` had `token_ce=11.5000`, `ppl=98715.77`, `tok_acc=0.00%`, peak VRAM `4001/97887 MB`.

The pipeline log ended with `Skipping code eval suite; pass --with-code-eval to run model code tests.` and `Pipeline complete.` Serve command printed by the run:

```bash
cargo run --release -- --serve runs/code_poc_1778927222/world/model.encoder.safetensors runs/code_poc_1778927222/latent/model.vocab.txt runs/code_poc_1778927222/world/model.safetensors 0.0.0.0:8080 768 256 12 12 768 96
```

Follow-up code eval smoke artifacts were written under `runs/code_poc_1778927222/code_eval_*.log` and `runs/code_eval/`. The latest completed eval was `runs/code_eval/1779092867` (`code_eval_base_decoder_greedy_smoke.log`): `suite_pass_rate=0.0000`, `route_code_acc=1.0000`, `rlm_used_rate=1.0000`, `docs_used_rate=0.0000`, `constraint_pass_rate=0.0000`, `compile_rate=0.0000`, `test_pass_rate=0.0000`, `tasks=10`, with all tasks failing constraints because the generated output missed the required `pub fn ...` signatures. Earlier smoke evals `1779090235` and `1779090331` reported `constraint_pass_rate=1.0000` but still had `suite_pass_rate=0.0000`, `compile_rate=0.0000`, and `test_pass_rate=0.0000`; those outputs included decoder-unavailable / prompt-text artifacts, so they are not treated as quality improvements.
