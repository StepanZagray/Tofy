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

Example:

- **JEPA eval (retrieval_top1 / pred_cosine):** 0.42 / 0.61 — `cargo run --release -- --eval-jepa local_models/model_latent_48.00M.safetensors local_models/vocabs/vocab_encoder.txt hub:wikimedia/wikipedia 500 32 768 256 9 8`
- **World eval (transition_cos):** 0.38 — `cargo run --release -- --eval-world ...`
