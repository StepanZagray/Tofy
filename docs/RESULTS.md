# Results tracking

Update the **Best So Far** section when a run improves on a reported metric. Record the exact command used. (Referenced by [AGENTS.md](../AGENTS.md).)

## Best So Far

*(No entries yet. Add metric, dataset/split, and full command when you have a run to record.)*

## Observed Baseline Runs

These were existing TensorBoard runs in `runs/` before the current training-code fixes. They are useful as a baseline, but they do not have the exact launch commands recorded, so they are not listed under **Best So Far**.

- `runs/world/1775175560`: raw `metrics/action_acc` stayed near 1.0 while `metrics/code_rate` stayed near 0 and `metrics/pred_code_rate` stayed at 0. This indicates majority-class router collapse, not healthy routing.
- `runs/latent/1775164725`: `loss/pred_token` worsened while `metrics/chunk_cosine` and `metrics/global_cosine` stayed near 1.0, which suggests the masking/task balance was too easy and the multiscale metrics were flattering.
- `runs/decoder/1775190816`: text decoder improved but still ended with triple-digit perplexity, so text-decoder capacity/data shaping needed work.
- `runs/decoder/1775183480`: code decoder learned meaningfully and was the strongest stage of the pipeline, but still not at a strong code-model quality bar.

## Current Fixes In Repo

- Encoder masking now enforces a non-trivial minimum target fraction and uses paired augmented views for chunk/global targets.
- World-model training now uses class-balanced action loss, balanced router batches, and logs balanced accuracy plus code precision/recall/F1.
- The full pipeline now prepares a mixed world dataset by default and gives the text decoder a larger default vocab budget.
- The repo now has a hard code-first eval suite at `eval/code_assistant_rust_hard.jsonl` plus `--eval-code-assistant` for end-to-end proof-of-concept scoring.
- Latent, world, and decoder training now support `--grad-accum <int>` so effective batch size can grow on 8 GB GPUs without increasing microbatch memory.
- `scripts/train_code_first_poc.sh` now defines the narrow proof-of-concept path: encoder -> world -> code decoder -> hard Rust eval suite.

## OOM Testing Notes

CUDA OOM checks should use the real release binary, not `cargo check`, because the memory failure happens at runtime after Candle builds CUDA tensors. Use short training runs with the same dtype, sequence length, segmentation settings, and dimensions as the intended pipeline. Preserve existing `local_models/` artifacts before probes, because smoke runs save checkpoints and vocab files.

Current 8 GB RTX 5060 measurements:

- Latent encoder `DIM=640`, `LAYERS=7`, `HEADS=8`, `MAX_VOCAB=8000`, `seq_len=256`, `bf16`: `32x1` passed at about `4942 MB` peak; `32x2` OOM; `64x1` OOM.
- Previous latent encoder `DIM=768`, `LAYERS=9`, `HEADS=8`, `MAX_VOCAB=8000`, `seq_len=256`, `bf16`: `32x1` passed at about `7545 MB` peak; `32x2` OOM.
- Shared-width world model `DIM=640`, `BRIDGE_DIM=640`, `NUM_LATENT_TOKENS=64`: `128x1` OOM; `96x1` passed; `64x2` passed.

Current default batch schedules for the 8 GB profile:

- Latent: `32x1`.
- World: warmup `96x1`, then `64x2` after `TOFY_WORLD_WARMUP_STEPS` (defaults to 20% of world steps when warmup differs from the main schedule).
- Decoder: `12x2` passed at effective batch 24, peaking around `6233 MB` total sampled VRAM in the one-step decoder fit probe; `24x1` OOMed around `7641 MB`.

Recommended OOM probe pattern:

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
TOFY_WORLD_WARMUP_BATCH=96 \
TOFY_WORLD_WARMUP_GRAD_ACCUM=1 \
TOFY_WORLD_WARMUP_STEPS=1 \
./target/release/jepa_ai --train-world <latent.safetensors> <vocab.txt> <world_pairs.txt> 2 64 640 256 7 8 640 64 --grad-accum 2
```

Treat a passing one-step OOM probe as a memory fit check only. It does not prove training quality. Record real quality improvements in **Best So Far** with the exact full command.

Example:

- **JEPA eval (retrieval_top1 / pred_cosine):** 0.42 / 0.61 — `cargo run --release -- --eval-jepa local_models/model_latent_48.00M.safetensors local_models/vocabs/vocab_encoder.txt hub:wikimedia/wikipedia 500 32 768 256 9 8`
- **World eval (transition_cos):** 0.38 — `cargo run --release -- --eval-world ...`
