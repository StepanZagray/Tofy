# Results tracking

Update the **Best So Far** section when a run improves on a reported metric. Record the exact command used. (Referenced by [AGENTS.md](../AGENTS.md).)

## Best So Far

These best metrics were produced before the strict LeJEPA default rewrite. Treat them as legacy baselines until a strict run reports better metrics with its exact command.

- **World transition selection score:** `0.0041821287` at step `17000/60000` on `data/world_mix_pairs.txt` validation stream, logged peak VRAM `7280/8151 MB` — legacy pre-CLI run with `DIM=640`, `LAYERS=7`, `HEADS=8`, `WORLD_BATCH=128`, `WORLD_GRAD_ACCUM=1`, `TOFY_TRAIN_DTYPE=bf16`. This is not the current default 8 GB schedule.
- **World training throughput:** `1.2209 steps/s` (`0.8191 s/step`) over logged steps `18000 -> 21000` on `data/world_mix_pairs.txt`, batch `128x1`, from `runs/code_poc_2026-04-21_21-50-54/world/events.out.tfevents.1776804849.zephyrus.25746.0`. This is about `1.98x` faster than `0.6170 steps/s` (`1.6208 s/step`) from `runs/code_poc_2026-04-20_22-32-35/world/events.out.tfevents.1776732255.zephyrus.86212.0` with the same world shape. Command: legacy pre-CLI resume run whose world stage invoked `target/release/jepa_ai --train-world local_models/model_latent_39.53M.safetensors local_models/model_latent_39.53M.vocab.txt data/world_mix_pairs.txt 60000 128 640 256 7 8 640 64 --lambda 0.2 --lr 2e-4 --grad-accum 1 --action-loss-weight 1.0 --router-warmup 5000 --resume`, with default token-cache prefetch enabled (`TOFY_CACHE_PREFETCH_BATCHES` unset, default `2`; `TOFY_TOKEN_CACHE_READER_MB` unset, default `8`).
- **Zero-conditioned code decoder constraint pass rate:** `0.3000` (`3/10`) with `suite_pass_rate=0.0000`, `compile_rate=0.0000`, and `route_code_acc=1.0000` on `eval/code_assistant_rust_hard.jsonl`, from `runs/code_eval/1776932354`. This used the current 68.50M code decoder with the planner/world conditioning vector filled with zeros; passing constraints were `merge_intervals`, `top_k_words`, and `compact_sorted_numbers`, but none compiled. Command: `cargo run --release -- --eval-code-assistant local_models/model_latent_39.53M.safetensors local_models/model_latent_39.53M.vocab.txt local_models/model_world_13.58M.safetensors eval/code_assistant_rust_hard.jsonl 384 640 256 7 8 640 64 --code-decoder local_models/code_decoder_poc_68.50M.safetensors --ablate-conditioning`.

## Observed Baseline Runs

These were existing TensorBoard runs in `runs/` before the current training-code fixes. They are useful as a baseline, but they do not have the exact launch commands recorded, so they are not listed under **Best So Far**.

- `runs/world/1775175560`: raw `metrics/action_acc` stayed near 1.0 while `metrics/code_rate` stayed near 0 and `metrics/pred_code_rate` stayed at 0. This indicates majority-class router collapse, not healthy routing.
- `runs/latent/1775164725`: `loss/pred_token` worsened while `metrics/chunk_cosine` and `metrics/global_cosine` stayed near 1.0, which suggests the masking/task balance was too easy and the multiscale metrics were flattering.
- `runs/decoder/1775190816`: text decoder improved but still ended with triple-digit perplexity, so text-decoder capacity/data shaping needed work.
- `runs/decoder/1775183480`: code decoder learned meaningfully and was the strongest stage of the pipeline, but still not at a strong code-model quality bar.

## Current Fixes In Repo

- The default training objective is now LeJEPA/LeWorldModel-style: online masked-view prediction plus SIGReg for the encoder, action-conditioned next-latent prediction plus SIGReg for the world model, `TOFY_SIGREG_SLICES=1024`, no EMA target update, no detached teacher, no contrastive auxiliary, no predictor heads, no world action/inverse loss by default, and no decoder syntax/signature/structure auxiliaries unless explicitly re-enabled.
- Encoder masking now enforces a non-trivial minimum target fraction and uses paired augmented views for chunk/global targets.
- World-model training still logs router/action diagnostics when the heads are present, but strict checkpoint selection is based on transition/SIGReg rather than router or inverse-action losses.
- The Rust `train <8gb|48gb|80gb>` pipeline now prepares the mixed world/code datasets by default and is the canonical full training path.
- The repo now has a hard code-first eval suite at `eval/code_assistant_rust_hard.jsonl` plus `--eval-code-assistant` for end-to-end proof-of-concept scoring.
- Latent, world, and decoder training now support `--grad-accum <int>` so effective batch size can grow on 8 GB GPUs without increasing microbatch memory.
- `cargo run --release -- train 8gb` now defines the canonical path: strict encoder -> strict world transition -> integrated high-world transition -> downstream code decoder -> hard Rust eval suite.
- The code-first pipeline now generates compiler-feedback Rust repair rows when `rustc` is available, mixes them into both world/code data as code-route examples, and uses tool/context tags while preserving three-action checkpoint compatibility.
- The Rust pipeline saves stage checkpoints inside run-owned directories under `runs/` and resumes by explicit run id or `latest`, which prevents stale checkpoints from unrelated runs being reused.
- Tokenization and token caching now use an explicit tokenizer-spec fingerprint plus UTF-8 byte fallback, so tokenizer/cache invalidation is tied to tokenizer behavior rather than only source hashes and mode names.

## OOM Testing Notes

CUDA OOM checks should use the real release binary, not `cargo check`, because the memory failure happens at runtime after Candle builds CUDA tensors. The repo does not ship bash wrappers; call `./target/release/jepa_ai` or `cargo run --release -- …` directly. Use short training runs with the same dtype, sequence length, segmentation settings, and dimensions as the intended pipeline. Preserve existing `local_models/` artifacts before probes, because smoke runs save checkpoints and vocab files.

Current 8 GB RTX 5060 measurements:

- Latent encoder `DIM=640`, `LAYERS=7`, `HEADS=8`, `MAX_VOCAB=8000`, `seq_len=256`, `bf16`: `32x1` passed at about `4942 MB` peak; `32x2` OOM; `64x1` OOM.
- Previous latent encoder `DIM=768`, `LAYERS=9`, `HEADS=8`, `MAX_VOCAB=8000`, `seq_len=256`, `bf16`: `32x1` passed at about `7545 MB` peak; `32x2` OOM.
- Shared-width world model `DIM=640`, `BRIDGE_DIM=640`, `NUM_LATENT_TOKENS=64`: `128x1` resumed from step `6100` and passed to step `7000`, but live `nvidia-smi` later reached about `7259/8151 MB`, so it is too close for unattended full-pipeline use; `96x1` passed in a shallow probe but later OOMed during a long run when logging/checkpointing was more frequent; `64x2` passed at about `4379 MB` peak and is the safer fallback.
- New 68.50M code decoder (`dim=640`, `layers=6`, `heads=8`, `ff=2560`): `6x4` with `seq_len=192` OOMed during Stage 5 startup after token-cache prefetch in the resumed code-first pipeline. The current 8 GB default is `6x4`, `CODE_DECODER_MAX_SEQ=128`, with the conditioning-margin ablation forward disabled by default.

Current default batch schedules for the 8 GB profile:

- Latent: `12x2`.
- World: warmup `64x1`, then `64x2` after `TOFY_WORLD_WARMUP_STEPS=1200`.
- Decoder: old 82.85M `12x2` and `8x3` OOMed in the full pipeline decoder startup path; old 82.85M `6x4` passed at about `7545 MB` peak but was too tight for unattended training; new 68.50M decoder (`dim=640`, `layers=6`, `heads=8`, `ff=2560`, `seq_len=192`) OOMed at `6x4` during resumed Stage 5 startup. The current 8 GB default is `6x4` with `seq_len=128` and `TOFY_DECODER_CONDITIONING_LOSS_WEIGHT=0`.

Short full-pipeline smoke pass:

```bash
cargo run --release -- train 8gb --resume latest
```

This completed all stages through `--eval-code-assistant` with the temporary world/decoder outputs and the previous 82.85M `6x4` decoder schedule. It is a pipeline/runtime smoke pass, not a quality result.

Current runtime generation uses the RLM code path by default (`TOFY_RLM_CODE=1`): Rust prompts are held as external RLM environment state, an RLM command program decomposes them into local work units, `SUB_RLM` recursively calls the same decoder on local prompts, and each sub-call is re-encoded through the world/planner. Disable with `TOFY_RLM_CODE=0` for the old one-shot code decoder path.

Recommended OOM probe pattern:

Use the sustained probe for batch defaults:

```bash
cargo run --release -- --sustained-oom-probe --stage all
```

One-step probes are not trusted for batch decisions. They only prove that the code path starts. The world `96x1` case passed shallow testing but OOMed in a real run after VRAM climbed across thousands of steps.

Legacy one-step fit pattern:

```bash
TOFY_TRAIN_DTYPE=bf16 \
TOFY_LATENT_CONTEXT_SEGMENTS=4 \
TOFY_LATENT_RECENT_FULL_SEGMENTS=1 \
TOFY_LATENT_HISTORY_RATIO=0.35 \
TOFY_LATENT_WARMUP_STEPS=0 \
./target/release/jepa_ai --latent <encoder_pairs.txt> 100 32 640 256 7 8 8000 --grad-accum 1
```

For world schedule transition testing, force a two-step run:

```bash
TOFY_TRAIN_DTYPE=bf16 \
TOFY_WORLD_WARMUP_BATCH=64 \
TOFY_WORLD_WARMUP_GRAD_ACCUM=1 \
TOFY_WORLD_WARMUP_STEPS=1 \
./target/release/jepa_ai --train-world <latent.safetensors> <vocab.txt> <world_pairs.txt> 2 64 640 256 7 8 640 64 --grad-accum 2
```

Treat a passing one-step OOM probe as a startup check only. It does not prove long-run memory safety or training quality. Record real quality improvements in **Best So Far** with the exact full command.

Example:

- **JEPA eval (retrieval_top1 / pred_cosine):** 0.42 / 0.61 — `cargo run --release -- --eval-jepa local_models/model_latent_48.00M.safetensors local_models/vocabs/vocab_encoder.txt hub:wikimedia/wikipedia 500 32 768 256 9 8`
- **World eval (transition_cos):** 0.38 — `cargo run --release -- --eval-world ...`
