# Tofy

Tofy is a local Rust/Candle agent with a split architecture:

- **Encoder**: reads text and produces latent features using its own vocab
- **Dialog transition model**: pure latent planner stack (`planner_memory -> world_transition -> orchestrator`)
- **Text decoder**: text-generalist decoder with its own vocab
- **Code decoder**: code-specialist decoder with its own vocab

Current inference path:

`encoder -> planner memory -> router/orchestrator -> decoder adapter -> decoder`

## Current proof-of-concept target

The best narrow target for this repo right now is a **code-first assistant**, not a general assistant.

The strongest parts of the stack are:

- shared encoder + planner/world memory
- code routing
- code decoder

The weakest part is still the text decoder, so the main KPI should be a **tiny hard code eval suite**, not generic chat quality.

## Important clarification

The **encoder and world model are not the same artifact anymore**.

- The **encoder** turns token ids into hidden states and uses `local_models/vocabs/vocab_encoder.txt`
- Each encoder checkpoint also saves a matched sibling vocab, for example `local_models/model_latent_69.84M.vocab.txt`
- The **dialog transition model** works only in latent space and does **not** own a text vocab
- Each Candle decoder has its **own** vocab saved next to its checkpoint, for example:
  - `local_models/text_decoder_90M.vocab.txt`
  - `local_models/code_decoder_90M.vocab.txt`

## Training stages

1. **Prepare data**
- chat pairs: `data/ultrachat_pairs.txt`
- code pairs: `data/multilang_pairs.txt`
- wikipedia cache: `data/cached_wikimedia_wikipedia_1.txt`
- encoder mix: generated from all of the above

2. **Train encoder**
- LeJEPA pretraining on the mixed encoder corpus
- hierarchical encoder with true sliding-window local attention, adaptive chunk/global latent states, learned global tokens, multiscale predictor heads, contrastive loss, and structured masking
- output: `local_models/model_latent_<size>.safetensors`
- vocab: `local_models/vocabs/vocab_encoder.txt`
- matched vocab: `local_models/model_latent_<size>.vocab.txt`

3. **Train dialog transition model**
- uses the frozen encoder + encoder vocab
- default pipeline now trains on a mixed text+code world dataset instead of chat-only pairs
- trains planner/world/orchestrator weights with stronger router pressure and an early router warmup
- output: `local_models/model_world_<size>.safetensors`

4. **Tune planner/orchestrator**
- reuses the saved world checkpoint
- trains the action path separately on explicit `text_reply` / `code` / `done` labels
- optionally tunes planner memory along with the action head
- output: updated `local_models/model_world_<size>.safetensors`

5. **Train decoders**
- text decoder uses UltraChat data and its own vocab
- code decoder uses multilingual code data and its own vocab
- each decoder checkpoint gets a sibling vocab file

## One-command pipeline

```bash
/scripts/train_full_pipeline.sh
```

Code-first proof of concept:

```bash
./scripts/train_code_first_poc.sh

Runtime smoke tests for CUDA/BF16 dtype and inference/training-path regressions:

```bash
./scripts/runtime_smoke_tests.sh
```

This runs tiny `--latent`, `--train-world`, `--train-orchestrator`, `--eval-world`, `--train-decoder`, and `--eval-code-assistant` stages with temp data so mixed-precision/runtime issues fail fast.

Also run the static dtype-discipline check before long BF16 runs:

```bash
python scripts/check_dtype_discipline.py
```
```

The code-first POC now biases the decoder toward the hard Rust eval format:

- base code data defaults to Rust-only code pairs
- `scripts/prepare_rust_function_tasks.py` derives synthetic instruction -> function pairs from the Rust code corpus
- `scripts/prepare_code_poc_mix.py` oversamples those instruction-shaped rows before code-decoder training

Default behavior:

- encoder corpus = UltraChat + downloaded Wikipedia + multilingual code pairs
- world model data = balanced chat+code mix from `scripts/prepare_world_mix.py`
- world/orchestrator rows now carry explicit action labels and synthetic terminal `done` rows
- text decoder data = UltraChat
- code decoder data = multilingual code pairs from `scripts/prepare_github_top_code.py`
- training now streams batches from disk with a small shuffle buffer instead of loading whole corpora into RAM first
- the training scripts now auto-select a safer `8gb` profile on cards with about 8 GB VRAM:
  - encoder/world keep `256` context
  - encoder microbatch defaults to `2` with `LATENT_GRAD_ACCUM=3`
  - decoder microbatch defaults to `4` with `*_DECODER_GRAD_ACCUM=2`
  - override with `TOFY_GPU_PROFILE=balanced` or explicit `LATENT_BATCH=...`, `WORLD_BATCH=...`, `CODE_DECODER_BATCH=...`, `TEXT_DECODER_BATCH=...`
- on CUDA toolkit `13.2+`, the scripts also auto-export `CUDARC_CUDA_VERSION=13010` and `CUDA_COMPUTE_CAP` if you did not already set them

```bash
python scripts/prepare_github_top_code.py --output data/multilang_pairs.txt --default-languages --max-files 200000
```

## Manual quick start

Prepare UltraChat:

```bash
cargo run --release -- --prepare-ultrachat data/ultrachat_pairs.txt 6 2
```

Build multilingual code pairs:

```bash
python scripts/prepare_github_top_code.py --output data/multilang_pairs.txt --default-languages --max-files 200000
```

Build mixed encoder corpus:

```bash
python scripts/prepare_encoder_corpus.py --output data/encoder_mix.txt data/ultrachat_pairs.txt data/cached_wikimedia_wikipedia_1.txt data/multilang_pairs.txt
```

Build world-model mix:

```bash
python scripts/prepare_world_mix.py --output data/world_mix_pairs.txt --text-pairs data/ultrachat_pairs.txt --code-pairs data/multilang_pairs.txt --code-ratio 0.35 --done-ratio 0.18
```

Train encoder:

```bash
cargo run --release -- --latent data/encoder_mix.txt 25000 32 768 256 9 8 8000
```

Train pure world model:

```bash
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/world_mix_pairs.txt 40000 32 768 256 9 8 256 64 --lambda 0.2 --action-loss-weight 1.0 --router-warmup 5000
```

Tune planner/orchestrator:

```bash
cargo run --release -- --train-orchestrator local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/world_mix_pairs.txt 15000 32 768 256 9 8 256 64
```

Train text decoder:

```bash
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/ultrachat_pairs.txt 20000 8 128 --decoder-kind text --decoder-max-vocab 16000 --decoder-output local_models/text_decoder_90M.safetensors
```

Train code decoder:

```bash
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/multilang_pairs.txt 20000 8 160 --decoder-kind code --decoder-max-vocab 16000 --decoder-output local_models/code_decoder_90M.safetensors
```

Run the code-assistant eval suite:

```bash
python scripts/generate_code_eval_suite.py --output eval/code_assistant_rust_hard.jsonl
cargo run --release -- --eval-code-assistant local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors eval/code_assistant_rust_hard.jsonl 384 768 256 9 8 256 64 --code-decoder local_models/code_decoder_90M.safetensors
```

Main KPI from this eval:

- `suite_pass_rate`
- supporting metrics: `route_code_acc`, `compile_rate`, `test_pass_rate`

The code decoder now uses a stronger code path during vocab building, training, and inference. It preserves escaped multiline formatting, emits structural tokens like `<nl>` / indentation buckets, normalizes literals such as `<str_lit>` and `<num_lit>`, adds language/task tags like `<lang:rust>` and `<ctx>`, and falls back to learned code subtokens plus character pieces instead of collapsing rare identifiers to `<unk>`.

Serve:

```bash
cargo run --release -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 256 9 8 256 64
```

## Main modes

| Mode | Output |
|------|--------|
| `--latent` | encoder checkpoint + encoder vocab |
| `--eval-jepa` | encoder metrics |
| `--train-world` | pure latent world-model checkpoint |
| `--train-orchestrator` | planner/orchestrator action fine-tune |
| `--eval-world` | world-model latent/action metrics |
| `--eval-code-assistant` | end-to-end code-first assistant KPI suite |
| `--train-decoder` | decoder checkpoint + decoder vocab |
| `--serve` | OpenAI-compatible HTTP server |

## Docs

- `docs/RUNBOOK.md`
- `docs/CODE_DATA.md`
- `docs/OPENCODE.md`
- `docs/DECODER_RUNTIME.md`
- `docs/ARCHITECTURE_AND_CAPACITY.md`
- `docs/RESULTS.md`
- `docs/RUNBOOK.md` also explains `tensorboard-rs` logging and TensorBoard usage

## Notes

- CUDA is enabled by default
- CPU-only: `cargo run --release --no-default-features -- ...`
- training logs go to per-run directories under `runs/`
- full pipeline runs are grouped as `runs/pipeline_<timestamp>/{latent,world,decoder_code,decoder_text}`
- each training run also records GPU memory telemetry under `memory/*` in TensorBoard and `memory_summary.txt` in the run directory
- latent, world, and decoder training now support `--grad-accum <int>` so you can trade wall-clock time for larger effective batch / context on small GPUs
- training also supports `TOFY_TRAIN_DTYPE=bf16|f16|f32`; the main scripts now default to `bf16` on GPU and fall back to `f32` on CPU
- the main training scripts now expose stage-specific microbatch overrides:
  - `LATENT_BATCH`, `WORLD_BATCH`, `CODE_DECODER_BATCH`, `TEXT_DECODER_BATCH`
  - `LATENT_GRAD_ACCUM`, `WORLD_GRAD_ACCUM`, `CODE_DECODER_GRAD_ACCUM`, `TEXT_DECODER_GRAD_ACCUM`
- encoder masking now enforces a real minimum target fraction, masks code rows more aggressively, and uses paired augmented views plus an EMA target encoder so chunk/global cosine is harder to game
- latent pretraining now also supports segmented training context with:
  - `TOFY_LATENT_CONTEXT_SEGMENTS=<int>`
  - `TOFY_LATENT_RECENT_FULL_SEGMENTS=<int>`
  - `TOFY_LATENT_HISTORY_RATIO=<float>`
- encoder local attention is now truly sliding-window instead of dense masked attention, and chunk size grows with sequence length so longer encoder contexts stay practical
- transition-model logging now includes balanced accuracy, macro F1, code precision/recall/F1, and done precision/recall/F1, and world checkpoint selection is router-first rather than total-loss-only
- decoder validation logging now also includes token accuracy, identifier accuracy, syntax-token accuracy, signature-token accuracy, signature-exact rate, function-skeleton rate, delimiter-balance rate, syntax-weighted CE, and signature-weighted CE for code generation quality
- the new code eval suite writes per-task results plus `summary.txt` under `runs/code_eval/<timestamp>/`
- Candle decoder inference now uses batched prompt prefill, per-layer self-attention KV cache, and precomputed cross-attention K/V for the world latent
- set `JEPA_CANDLE_DECODER_CTX=<tokens>` to cap Candle decoder prompt context at inference if you want a predictable latency / memory ceiling
- encoder/planner inference now supports segmented hierarchical prompt memory:
  - `TOFY_ENCODER_CONTEXT_SEGMENTS=<int>` sets how many encoder segments are retained, default `4`
  - `TOFY_ENCODER_RECENT_FULL_SEGMENTS=<int>` sets how many newest segments keep full token-level memory, default `1`
  - older segments are compressed into chunk/global/planner summaries instead of being dropped outright
- world/planner training and serving now also support recurrent latent folding across segments:
  - `TOFY_WORLD_CONTEXT_SEGMENTS=<int>` sets how many state segments are folded for world/decoder training, default `1` unless the code-first script overrides it
  - `TOFY_WORLD_RECENT_FULL_SEGMENTS=<int>` sets how many newest world segments keep full token-level memory, default `1`
  - `TOFY_RECURSIVE_PLANNER_MEMORY=1` enables recursive planner-slot folding across segments
  - `TOFY_WORLD_TRAIN_ROLLOUT_STEPS=<int>` and `TOFY_WORLD_ROLLOUT_STEPS=<int>` control how many transition steps are rolled out before decoder conditioning
- the code-first POC decoder path now uses Rust-only code pairs plus oversampled synthetic Rust instruction/function tasks, optional Rust-by-Practice pairs, shuffled mixed-code training data, and a short instruction-only decoder polish phase because the hard eval suite measures instruction-following Rust generation rather than generic multilingual code continuation
- encoder TensorBoard now includes `loss/pred_token`, `loss/pred_chunk`, `loss/pred_global`, `metrics/chunk_cosine`, and `metrics/global_cosine`
- view metrics with `tensorboard --logdir runs/`
