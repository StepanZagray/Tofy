# Results tracking

Update the **Best So Far** section when a run improves on a reported metric. Record the exact command used. (Referenced by [AGENTS.md](../AGENTS.md).)

## Best So Far

These best metrics were produced before the strict LeJEPA default rewrite. Treat them as legacy baselines until a strict run reports better metrics with its exact command.

- **World transition selection score:** `0.0041821287` at step `17000/60000` on `data/world_mix_pairs.txt` validation stream, logged peak VRAM `7280/8151 MB` — legacy pre-CLI run with `DIM=640`, `LAYERS=7`, `HEADS=8`, `WORLD_BATCH=128`, `WORLD_GRAD_ACCUM=1`, `TOFY_TRAIN_DTYPE=bf16`. This is not the current default 8 GB schedule.
- **World training throughput:** `1.2209 steps/s` (`0.8191 s/step`) over logged steps `18000 -> 21000` on `data/world_mix_pairs.txt`, batch `128x1`, from `runs/code_poc_2026-04-21_21-50-54/world/events.out.tfevents.1776804849.zephyrus.25746.0`. This is about `1.98x` faster than `0.6170 steps/s` (`1.6208 s/step`) from `runs/code_poc_2026-04-20_22-32-35/world/events.out.tfevents.1776732255.zephyrus.86212.0` with the same world shape. Command: legacy pre-CLI resume run whose world stage invoked `target/release/jepa_ai --train-world local_models/model_latent_39.53M.safetensors local_models/model_latent_39.53M.vocab.txt data/world_mix_pairs.txt 60000 128 640 256 7 8 640 64 --lambda 0.2 --lr 2e-4 --grad-accum 1 --action-loss-weight 1.0 --router-warmup 5000 --resume`, with default token-cache prefetch enabled (`TOFY_CACHE_PREFETCH_BATCHES` unset, default `2`; `TOFY_TOKEN_CACHE_READER_MB` unset, default `8`).
- **Zero-conditioned code decoder constraint pass rate:** `0.3000` (`3/10`) with `suite_pass_rate=0.0000`, `compile_rate=0.0000`, and `route_code_acc=1.0000` on `eval/code_assistant_rust_hard.jsonl`, from `runs/code_eval/1776932354`. This used the current 68.50M code decoder with the context/state conditioning vector filled with zeros; passing constraints were `merge_intervals`, `top_k_words`, and `compact_sorted_numbers`, but none compiled. Command: `cargo run --release -- --eval-code-assistant local_models/model_latent_39.53M.safetensors local_models/model_latent_39.53M.vocab.txt local_models/model_world_13.58M.safetensors eval/code_assistant_rust_hard.jsonl 384 640 256 7 8 640 64 --code-decoder local_models/code_decoder_poc_68.50M.safetensors --ablate-conditioning`.

## Observed Baseline Runs

These were existing TensorBoard runs in `runs/` before the current training-code fixes. They are useful as a baseline, but they do not have the exact launch commands recorded, so they are not listed under **Best So Far**.

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

