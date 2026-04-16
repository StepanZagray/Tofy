# Runbook

Current split architecture:

`encoder -> planner memory -> router/orchestrator -> decoder-specific adapter -> decoder`

## Recommended proof of concept

For the current repo, the cleanest proof of concept is:

1. train encoder
2. train world router/transition on mixed text+code pairs
3. train only the code decoder, but bias it toward Rust instruction-following rather than generic multilingual continuation
4. score the result on the hard Rust eval suite

That path is narrower and more defensible than trying to prove a general chat assistant first.

The code-first script now also does a short instruction-only decoder polish pass after the mixed decoder stage:

- base decoder run on `data/code_poc_mix.txt`
- optional polish run on `data/rust_instruction_pairs.txt`

For a local GPU around 8 GB VRAM, the training scripts now auto-pick a safer profile:

- encoder/world keep `256` context
- encoder microbatch defaults to `2` with `LATENT_GRAD_ACCUM=3`
- decoder microbatch defaults to `4` with `*_DECODER_GRAD_ACCUM=2`
- world stays at `WORLD_GRAD_ACCUM=1` because it is much cheaper than the encoder/decoders
- training defaults to `TOFY_TRAIN_DTYPE=bf16` on GPU, with CPU forced back to `f32`
- code decoder defaults now use `max_seq=192` rather than `160`

Override that behavior with:

- `TOFY_GPU_PROFILE=balanced`
- or explicit stage overrides such as `LATENT_BATCH=4`, `WORLD_BATCH=12`, `CODE_DECODER_BATCH=3`

Training-side latent context knobs:

- `TOFY_LATENT_CONTEXT_SEGMENTS=<int>` widens the source window sampled during latent training
- `TOFY_LATENT_RECENT_FULL_SEGMENTS=<int>` keeps the newest latent-training segments at full resolution
- `TOFY_LATENT_HISTORY_RATIO=<float>` reserves part of the latent window for sampled older-history tokens

World/planner memory knobs:

- `TOFY_WORLD_CONTEXT_SEGMENTS=<int>` widens state-context folding for world/orchestrator/decoder training
- `TOFY_WORLD_RECENT_FULL_SEGMENTS=<int>` keeps the newest world-context segments at full resolution
- `TOFY_RECURSIVE_PLANNER_MEMORY=1` turns on recurrent planner-slot folding across segments
- `TOFY_WORLD_TRAIN_ROLLOUT_STEPS=<int>` rolls the transition model forward multiple times before decoder conditioning during training
- `TOFY_WORLD_ROLLOUT_STEPS=<int>` does the same for serve/eval generation

Decoder training knobs:

- `TOFY_DECODER_SYNTAX_LOSS_WEIGHT=<float>` mixes syntax-weighted CE into decoder training
- `TOFY_DECODER_SIGNATURE_LOSS_WEIGHT=<float>` upweights the predicted Rust function-signature span during decoder training
- `CODE_POLISH_STEPS=<int>` controls the instruction-only polish phase in the code-first pipeline, default `8000`
- `CODE_POLISH_LR=<float>` sets the polish-phase learning rate, default `1e-4`

Inference-side context hierarchy knobs:

- `TOFY_ENCODER_CONTEXT_SEGMENTS=<int>` controls how many encoder segments are retained at serve/eval time, default `4`
- `TOFY_ENCODER_RECENT_FULL_SEGMENTS=<int>` controls how many newest segments keep full token-level memory, default `1`

The main pipeline scripts also auto-export:

- `CUDARC_CUDA_VERSION=13010` when the local CUDA toolkit reports `13.2+`
- `CUDA_COMPUTE_CAP` from `nvidia-smi` when it is available

That avoids the current `cudarc` / `candle-kernels` build failures on newer local CUDA installs without requiring a toolkit downgrade.

Artifact ownership:

- encoder checkpoint + encoder vocab
- world checkpoint only
- text decoder checkpoint + text decoder vocab
- code decoder checkpoint + code decoder vocab

Important: the encoder now also saves a checkpoint-matched vocab next to the latent model, for example `local_models/model_latent_69.84M.vocab.txt`. Use that sibling vocab when reusing an older latent checkpoint; `local_models/vocabs/vocab_encoder.txt` is only the latest shared encoder vocab.

## Resuming training

Training stages support resumable sidecar checkpoints. The easiest path is to rerun the same pipeline with:

```bash
TOFY_RESUME=1 scripts/train_code_first_poc.sh
```

or:

```bash
TOFY_RESUME=1 scripts/train_full_pipeline.sh
```

The scripts pass `--resume` into the supported stages automatically. Direct commands can also use `--resume`:

```bash
cargo run --release -- --latent data/encoder_mix.txt 25000 32 640 256 7 8 8000 --grad-accum 1 --resume
```

```bash
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/world_mix_pairs.txt 60000 64 640 256 7 8 640 64 --grad-accum 2 --resume
```

```bash
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/code_poc_mix.txt 40000 12 224 640 7 8 640 64 --decoder-kind code --decoder-output local_models/code_decoder_poc.safetensors --grad-accum 2 --resume
```

Resume files are saved next to the target model path:

- `<model>.STAGE.train.safetensors`
- `<model>.STAGE.optimizer.safetensors`
- `<model>.STAGE.resume.json`

For latent training there is also:

- `<model>.STAGE.target.safetensors`

`STAGE` comes from `TOFY_RUN_STAGE_NAME` when it is set by the scripts, otherwise it is the default stage name such as `latent`, `world`, `orchestrator`, or `decoder`.

Resume rules:

- Keep the same architecture arguments: `DIM`, `LAYERS`, `HEADS`, `BRIDGE_DIM`, `NUM_LATENT_TOKENS`, vocab size, and decoder architecture must match the saved sidecars.
- Keep the same model output path. Changing output paths creates a new resume namespace.
- If optimizer sidecars do not exist, `--resume` can still load the exported best/final model weights when available, but optimizer momentum and exact step continuation are not restored.
- If `resume.json` already reached the requested step count, the stage exits without doing more training. Increase `LATENT_STEPS`, `WORLD_STEPS`, `CODE_DECODER_STEPS`, etc. to continue further.
- Do not use old checkpoints from a different architecture, for example the previous `DIM=768` encoder/world with the current shared-width `DIM=640` setup.

## Context Guide

In this project, "context" has three layers:

1. **Runtime conversation context**
2. **Pair-file context**
3. **Decoder autoregressive context**

### Runtime conversation context

At serve time, the full `messages` array is formatted into one prompt string with role prefixes such as `System:`, `User:`, and `Assistant:`. The encoder sees that full prompt text.

The encoder is still bounded per forward pass, but runtime context is no longer limited to a pure hard truncation. The agent now supports **segmented hierarchical prompt memory**:

- the newest segment keeps full token-level encoder memory
- older segments are re-encoded one segment at a time and compressed into chunk/global/planner summaries
- the planner attends over the concatenation of compressed older memory and full recent memory

So runtime context is now better described as:

- full conversation
- clipped to the last `TOFY_ENCODER_CONTEXT_SEGMENTS * max_seq` encoder tokens
- with only the newest `TOFY_ENCODER_RECENT_FULL_SEGMENTS` segments kept at full token resolution

This is a better fit for recent hierarchical-memory long-context work than the old "keep only the last `max_seq` tokens" path, while still staying close to the existing Tofy architecture.

### Pair-file context

Most training files use:

`left<TAB>right`

For decoder/world training:

- `left` = current state / context
- `right` = next response / continuation

So when the docs say "context pair", they usually mean the **left side** of the dataset row.

### Decoder autoregressive context

The selected decoder gets two conditioning sources:

- the prompt text tokenized with that decoder's own tokenizer/vocab
- planner slots from the latent world path through cross-attention

So the decoder is not conditioned only on prompt tokens and not conditioned only on planner memory. It uses both.

Important: each module counts context in its **own tokens**:

- encoder uses encoder vocab/tokenizer
- text decoder uses text-decoder vocab/tokenizer
- code decoder uses code-aware vocab/tokenizer

That means equal `max_seq` values across modules do not correspond to exactly equal raw text length.

### Training-length rule for decoders

Decoder training uses both sides of the pair in teacher forcing:

- `input = left + shifted(right)`
- `target = shifted(left) + right`

So if decoder `max_seq = 160`, that means:

- up to `160` tokens from the left side
- up to `160` tokens from the right side
- effectively up to `320` autoregressive positions in the decoder loss

### Practical rule of thumb

- Increase **encoder `max_seq`** if the model forgets earlier conversation instructions.
- Increase **code decoder `max_seq`** if code continuation quality drops because prefixes are too short.
- Remember that planner slots are compressed memory: they preserve high-level state, not a perfect copy of every past token.

## 1. Prepare chat data

```bash
cargo run --release -- --prepare-ultrachat data/ultrachat_pairs.txt 6 2
```

## 2. Prepare code data

```bash
python scripts/prepare_github_top_code.py --output data/multilang_pairs.txt --default-languages --max-files 200000
```

## 3. Build the mixed encoder corpus

Assumes you already have the downloaded Wikipedia cache such as `data/cached_wikimedia_wikipedia_1.txt`.

```bash
python scripts/prepare_encoder_corpus.py --output data/encoder_mix.txt data/ultrachat_pairs.txt data/cached_wikimedia_wikipedia_1.txt data/multilang_pairs.txt
```

## 4. Train the encoder

```bash
cargo run --release -- --latent data/encoder_mix.txt 25000 32 768 256 9 8 8000
```

What the encoder now trains:

- local token attention for short-range syntax/patterns
- chunk-level global attention for longer structure
- local attention now uses a real sliding window instead of building a dense `seq x seq` mask
- chunk size grows with sequence length so the global latent path stays compact as context grows
- learned global latent tokens that summarize the whole sequence
- predictor heads with multiscale JEPA losses on token, chunk, and global representations
- chunk/global targets come from a second augmented view instead of the raw unmasked sequence
- structured masking now enforces a minimum masked fraction with stronger code-block/comment focus and text-boundary spans

Outputs:

- `local_models/model_latent_<size>.safetensors`
- `local_models/vocabs/vocab_encoder.txt`

## 5. Evaluate the encoder

```bash
cargo run --release -- --eval-jepa local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/encoder_mix.txt 200 32 768 256 9 8
```

## 6. Train the pure dialog transition model

The transition model is latent-only. It loads the frozen encoder checkpoint and encoder vocab, but it saves only planner/world/orchestrator weights.

Build a mixed world dataset first so the router sees text, code, and terminal done actions:

```bash
python scripts/prepare_world_mix.py --output data/world_mix_pairs.txt --text-pairs data/ultrachat_pairs.txt --code-pairs data/multilang_pairs.txt --code-ratio 0.35 --done-ratio 0.18
```

```bash
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/world_mix_pairs.txt 40000 32 768 256 9 8 256 64 --lambda 0.2 --action-loss-weight 1.0 --router-warmup 5000
```

Output:

- `local_models/model_world_<size>.safetensors`
- logs now include `metrics/action_balanced_acc`, `metrics/action_macro_f1`, `metrics/code_precision`, `metrics/code_recall`, `metrics/code_f1`, `metrics/done_f1`, plus held-out `val/*` metrics

## 6b. Fine-tune planner/orchestrator on explicit action labels

This stage reuses the saved world checkpoint and focuses the planner/action path on explicit `text_reply`, `code`, and `done` labels.

```bash
cargo run --release -- --train-orchestrator local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/world_mix_pairs.txt 15000 32 768 256 9 8 256 64
```

## 7. Evaluate the world model

```bash
cargo run --release -- --eval-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/ultrachat_pairs.txt 200 32 768 256 9 8 256 64
```

## 8. Train the text decoder

UltraChat only by default.

```bash
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/ultrachat_pairs.txt 20000 8 128 --decoder-kind text --decoder-max-vocab 16000 --decoder-output local_models/text_decoder_90M.safetensors
```

Artifacts:

- `local_models/text_decoder_90M.safetensors`
- `local_models/text_decoder_90M.vocab.txt`

## 9. Train the code decoder

Default code dataset is the multilingual preset from `prepare_github_top_code.py`.
The code decoder uses a stronger code path: pair files are now single-line escaped rows, code formatting is restored before tokenization, identifiers are split on `_`, camelCase, and digit boundaries, literals are normalized, language/context tags are added, and rare identifiers fall back to learned code subtokens plus characters instead of becoming pure `<unk>`.

```bash
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/multilang_pairs.txt 20000 8 160 --decoder-kind code --decoder-max-vocab 16000 --decoder-output local_models/code_decoder_90M.safetensors
```

Artifacts:

- `local_models/code_decoder_90M.safetensors`
- `local_models/code_decoder_90M.vocab.txt`

For the narrow code-first POC, prefer Rust-only code data plus instruction-shaped Rust tasks:

```bash
python scripts/prepare_github_top_code.py --output data/rust_code_pairs.txt --languages Rust --max-files 120000
python scripts/prepare_rust_function_tasks.py --input data/rust_code_pairs.txt --output data/rust_instruction_pairs.txt
python scripts/prepare_code_poc_mix.py --output data/code_poc_mix.txt --base-pairs data/rust_code_pairs.txt --instruction-pairs data/rust_instruction_pairs.txt --instruction-repeat 4
```

Then train the code decoder on `data/code_poc_mix.txt`. This matches the hard Rust eval much better than a decoder trained only on multilingual code continuation.

## 10. Serve

With GGUF fallback:

```bash
export JEPA_DECODER_MODEL=./models/your_model.gguf
cargo run --release -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 256 9 8 256 64
```

With Candle text + code decoders:

```bash
export JEPA_USE_CANDLE_DECODER=1
export JEPA_CANDLE_DECODER=./local_models/code_decoder_90M.safetensors
export JEPA_USE_TEXT_DECODER=1
export JEPA_TEXT_DECODER=./local_models/text_decoder_90M.safetensors
cargo run --release -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 256 9 8 256 64
```

If the decoder vocab files are not next to the decoder checkpoints, set:

- `JEPA_CANDLE_DECODER_VOCAB`
- `JEPA_TEXT_DECODER_VOCAB`

Optional Candle decoder inference tuning:

- `JEPA_CANDLE_DECODER_CTX=<tokens>` limits the prompt tokens kept by the Candle decoder runtime before generation

## 10b. Code-first eval suite

Generate the suite:

```bash
python scripts/generate_code_eval_suite.py --output eval/code_assistant_rust_hard.jsonl
```

Run the end-to-end eval:

```bash
cargo run --release -- --eval-code-assistant local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors eval/code_assistant_rust_hard.jsonl 384 768 256 9 8 256 64 --code-decoder local_models/code_decoder_90M.safetensors
```

The eval writes:

- `runs/code_eval/<timestamp>/results.jsonl`
- `runs/code_eval/<timestamp>/summary.txt`

The main KPI is `suite_pass_rate`. Support metrics are `route_code_acc`, `compile_rate`, and `test_pass_rate`.

## 11. TensorBoard

This project uses the Rust crate `tensorboard-rs` via `tensorboard_rs::summary_writer::SummaryWriter`.

Training commands write event files under `runs/`.

- standalone commands still write stage-based runs such as `runs/latent/<timestamp>`
- `./scripts/train_full_pipeline.sh` now groups one full pipeline under:
  - `runs/pipeline_<timestamp>/latent`
  - `runs/pipeline_<timestamp>/world`
  - `runs/pipeline_<timestamp>/decoder_code`
  - `runs/pipeline_<timestamp>/decoder_text`
- each grouped pipeline run also writes `runs/pipeline_<timestamp>/meta.json`
- decoder training now logs `val/token_accuracy`, `val/identifier_accuracy`, and `val/delimiter_balance_rate` in addition to CE / perplexity / OOV

Start TensorBoard from the repository root:

```bash
tensorboard --logdir runs/
```

Then open the local URL printed by TensorBoard in your browser, usually `http://localhost:6006`.

Typical flow:

1. Start a training command such as `--latent`, `--train-world`, or `--train-decoder`.
2. In another terminal, run `tensorboard --logdir runs/`.
3. Open the Scalars tab to watch useful tags such as `loss/total`, `loss/pred`, `loss/contrastive`, `loss/sigreg`, `loss/trans`, `loss/action`, `loss/token_ce`, `metrics/pred_cosine`, `metrics/contrastive_cosine`, `metrics/trans_cosine`, `metrics/action_acc`, `metrics/perplexity`, and `memory/used_mb`.
4. For encoder runs, pay special attention to `loss/pred_token`, `loss/pred_chunk`, `loss/pred_global`, `metrics/chunk_cosine`, and `metrics/global_cosine`. Those tell you whether the encoder is learning local detail, mid-level structure, and whole-sequence semantics instead of only improving one pooled number.
5. For transition-model runs, compare `metrics/action_acc` with `metrics/action_balanced_acc`, `metrics/code_rate`, `metrics/pred_code_rate`, `metrics/code_f1`, and `val/total`. High raw accuracy with near-zero predicted code rate still means collapse.
6. For proof-of-concept tracking, prefer the code eval suite over any single proxy metric. A lower world loss with no improvement in `suite_pass_rate` is not a meaningful win.

Each run also writes `memory_summary.txt` in its run directory with peak VRAM usage when NVIDIA telemetry is available.

If you want to clear old charts before a fresh run, remove the relevant stage directory under `runs/` first.

## 12. CPU-only

```bash
cargo run --release --no-default-features -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 256 9 8 256 64
```

## 13. Scripts

```bash
./scripts/prepare_encoder_corpus.py --help
./scripts/prepare_world_mix.py --help
./scripts/generate_code_eval_suite.py --help
./scripts/train_encoder_25k.sh
./scripts/train_full_pipeline.sh
./scripts/train_code_first_poc.sh

## Runtime Smoke Tests

Use this before long GPU runs when you change dtypes, attention, planner/world logic, or decoder runtime:

```bash
./scripts/runtime_smoke_tests.sh
```

What it covers:
- tiny `--latent` BF16/F32 training run
- tiny `--train-world`
- tiny `--train-orchestrator`
- tiny `--eval-world`
- tiny `--train-decoder`
- tiny `--eval-code-assistant`

Useful overrides:
- `TOFY_RUNTIME_SMOKE_SKIP_CODE_EVAL=1` to skip the final code-assistant eval
- `TOFY_RUNTIME_SMOKE_KEEP=1` to keep temp artifacts
- `TOFY_TRAIN_DTYPE=bf16|f16|f32`

Static dtype discipline check:

```bash
python scripts/check_dtype_discipline.py
```
```
