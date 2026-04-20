# Results tracking

Update the **Best So Far** section when a run improves on a reported metric. Record the exact command used. (Referenced by [AGENTS.md](../AGENTS.md).)

## Best So Far

- **World transition selection score:** `0.0096` at step `7000/60000` on `data/world_mix_pairs.txt` validation stream, logged peak VRAM `6139/8151 MB` — `TOFY_RESUME=1 ./scripts/train_code_first_poc.sh` with `DIM=640`, `LAYERS=7`, `HEADS=8`, `WORLD_BATCH=128`, `WORLD_GRAD_ACCUM=1`, `TOFY_TRAIN_DTYPE=bf16`.

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
- Shared-width world model `DIM=640`, `BRIDGE_DIM=640`, `NUM_LATENT_TOKENS=64`: `128x1` resumed from step `6100` and passed to step `7000`, but live `nvidia-smi` later reached about `7259/8151 MB`, so it is too close for unattended full-pipeline use; `96x1` passed in a shallow probe but later OOMed during a long run when logging/checkpointing was more frequent; `64x2` passed at about `4379 MB` peak and is the safer default.

Current default batch schedules for the 8 GB profile:

- Latent: `32x1`.
- World: warmup `64x1`, then `64x2` after `TOFY_WORLD_WARMUP_STEPS=1200`.
- Decoder: old 82.85M `12x2` and `8x3` OOMed in the full pipeline decoder startup path; old 82.85M `6x4` passed at about `7545 MB` peak but was too tight for unattended training; new 68.50M decoder (`dim=640`, `layers=6`, `heads=8`, `ff=2560`, `seq_len=192`) passed at `6x4` effective batch 24 with about `6489 MB` peak and is the current default.

Short full-pipeline smoke pass:

```bash
TOFY_RESUME=1 WORLD_STEPS=8000 ROUTER_STEPS=1 CODE_DECODER_STEPS=1 CODE_POLISH_STEPS=1 \
WORLD_MODEL=local_models/tmp_pipeline_world.safetensors \
CODE_DECODER_OUTPUT=local_models/tmp_code_decoder_poc.safetensors \
PIPELINE_RUN_ID=code_poc_smoke_4x6_2026-04-18_06-22-04 \
./scripts/train_code_first_poc.sh
```

This completed all stages through `--eval-code-assistant` with the temporary world/decoder outputs and the previous 82.85M `4x6` decoder schedule. It is a pipeline/runtime smoke pass, not a quality result.

Current runtime generation uses an RLM-style code path by default (`TOFY_RLM_CODE=1`): Rust prompts are decomposed into local function work units, each work unit is re-encoded through the world/planner, and one shared decoder is reused sequentially with short local prompts. Disable with `TOFY_RLM_CODE=0` for the old one-shot code decoder path.

Recommended OOM probe pattern:

Use the sustained probe for batch defaults:

```bash
./scripts/sustained_oom_probe.py --stage all
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
