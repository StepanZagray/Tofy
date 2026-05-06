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

## Current code layout

The repository now mirrors that split more directly in Rust:

- `src/main.rs`: thin entrypoint and mode dispatch
- `src/cli.rs`: shared CLI/path helpers
- `src/config/latent.rs`: latent train/eval config parsing
- `src/config/world.rs`: world/orchestrator/decoder/eval/serve config parsing
- `src/tasks/latent.rs`: latent training and JEPA evaluation
- `src/tasks/world.rs`: world/orchestrator/decoder training and runtime engine
- `src/tasks/world_support.rs`: shared world/decoder metrics, masking, and evaluation helpers

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
- hierarchical encoder with true sliding-window local attention, adaptive chunk/global latent states, learned global tokens, online prediction, SIGReg, and structured masking
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
./scripts/train_full_pipeline.sh
```

Code-first proof of concept:

```bash
./scripts/train_code_first_poc.sh
./scripts/train_code_first_poc.sh --resume latest
./scripts/train_full_pipeline.sh --resume latest
```

Runtime smoke tests for CUDA/BF16 dtype and inference/training-path regressions:

```bash
./scripts/runtime_smoke_tests.sh
```

This runs tiny `--latent`, `--train-world`, `--train-orchestrator`, `--eval-world`, `--train-decoder`, and `--eval-code-assistant` stages with temp data so mixed-precision/runtime issues fail fast.

Also run the static dtype-discipline check before long BF16 runs:

```bash
cargo run --release -- --check-dtype-discipline
```

The code-first POC now biases the decoder toward the hard Rust eval format:

- base code data defaults to Rust-only code pairs
- `--prepare-rust-function-tasks` derives synthetic instruction -> function pairs from the Rust code corpus
- `--prepare-code-poc-mix` oversamples those instruction-shaped rows before code-decoder training

Default behavior:

- encoder corpus = UltraChat + downloaded Wikipedia + multilingual code pairs
- world model data = balanced chat+code mix from `--prepare-world-mix`
- world/orchestrator rows now carry explicit action labels and synthetic terminal `done` rows
- text decoder data = UltraChat
- code decoder data = multilingual code pairs from `--prepare-github-top-code`
- Stage 1 now covers both prepared-data artifacts and vocab/token caches; the encoder corpus, Rust instruction pairs, Rust repair pairs, world mix, and code POC mix all use sidecar manifest caches so unchanged reruns avoid rebuilding them
- pipeline scripts now save stage checkpoints, logs, launch metadata, and grouped TensorBoard outputs under run-owned directories such as `runs/code_poc_<timestamp>/...` and `runs/pipeline_<timestamp>/...`
- training now streams batches from disk with a small shuffle buffer instead of loading whole corpora into RAM first
- cached token streams prefetch two ordered chunks by default; set `TOFY_CACHE_PREFETCH_BATCHES=0` to disable, `TOFY_CACHE_PREFETCH_BATCHES=N` to tune queue depth, `TOFY_CACHE_PREFETCH_CHUNK=N` to force chunk size, or `TOFY_TOKEN_CACHE_READER_MB=N` to tune the cache reader buffer
- world/orchestrator/decoder planner encoding batches token segments on GPU; set `TOFY_PLANNER_SEGMENT_BATCH=N` to tune the segment micro-batch size, default `64`
- the training scripts now default to the `8gb` profile:
  - encoder/world keep `256` context
  - encoder defaults to `12x2`
  - world defaults to `64x2` after a `64x1` warmup
  - code decoder defaults to `6x4` with `CODE_DECODER_MAX_SEQ=128`
  - the 8 GB profile disables the decoder conditioning-margin ablation pass during training; set `TOFY_DECODER_CONDITIONING_LOSS_WEIGHT>0` or `TOFY_DECODER_ABLATION_METRICS=1` to restore it
  - override with explicit `LATENT_BATCH=...`, `WORLD_BATCH=...`, `CODE_DECODER_BATCH=...`, `TEXT_DECODER_BATCH=...`
- the 80 GB cloud profile is available through `./scripts/train_code_first_poc_80gb.sh`:
  - sets `TOFY_GPU_PROFILE=80gb`
  - uses shared `DIM=2048`, `BRIDGE_DIM=2048`, `LAYERS=7`, `HEADS=16`
  - uses `MAX_VOCAB=16000`, `CODE_DECODER_MAX_VOCAB=32000`, `NUM_LATENT_TOKENS=128`
  - keeps context at the proven POC shape initially: encoder/world `256`, code decoder `128`
  - runs 10x step budgets by default: latent `250000`, world `600000`, code decoder `400000`, polish `80000`
  - set `TOFY_80GB_OOM_PROBE=1` to run the sustained OOM probe before the long pipeline
- the 48 GB A40 profile is available through `./scripts/train_code_first_poc_48gb.sh`:
  - sets `TOFY_GPU_PROFILE=48gb`
  - uses shared `DIM=1536`, `BRIDGE_DIM=1536`, `LAYERS=7`, `HEADS=12`
  - uses `MAX_VOCAB=12000`, `CODE_DECODER_MAX_VOCAB=24000`, `NUM_LATENT_TOKENS=96`
  - keeps context at the proven POC shape initially: encoder/world `256`, code decoder `128`
  - runs test-scale budgets by default: latent `75000`, world `180000`, code decoder `120000`, polish `24000`
  - set `TOFY_48GB_OOM_PROBE=1` to run the sustained OOM probe before the long pipeline
- on CUDA, the scripts auto-export `CUDA_COMPUTE_CAP` if you did not already set it

```bash
cargo run --release -- --prepare-github-top-code --output data/multilang_pairs.txt --default-languages --max-files 200000
```

## Manual quick start

Prepare UltraChat:

```bash
cargo run --release -- --prepare-ultrachat data/ultrachat_pairs.txt 6 2
```

Build multilingual code pairs:

```bash
cargo run --release -- --prepare-github-top-code --output data/multilang_pairs.txt --default-languages --max-files 200000
```

Build mixed encoder corpus:

```bash
cargo run --release -- --prepare-encoder-corpus --output data/encoder_mix.txt data/ultrachat_pairs.txt data/cached_wikimedia_wikipedia_1.txt data/multilang_pairs.txt
```

Build world-model mix:

```bash
cargo run --release -- --prepare-world-mix --output data/world_mix_pairs.txt --text-pairs data/ultrachat_pairs.txt --code-pairs data/multilang_pairs.txt --code-ratio 0.35 --done-ratio 0.18
```

Train encoder:

```bash
cargo run --release -- --latent data/encoder_mix.txt 25000 32 768 256 9 8 8000
```

Train pure world model:

```bash
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/world_mix_pairs.txt 40000 32 768 256 9 8 256 64 --lambda 0.2 --action-loss-weight 0
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
cargo run --release -- --generate-code-eval-suite --output eval/code_assistant_rust_hard.jsonl
cargo run --release -- --eval-code-assistant local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors eval/code_assistant_rust_hard.jsonl 384 768 256 9 8 256 64 --code-decoder local_models/code_decoder_90M.safetensors
```

Main KPI from this eval:

- `suite_pass_rate`
- supporting metrics: `route_code_acc`, `compile_rate`, `test_pass_rate`

The code decoder now uses a stronger code path during vocab building, training, and inference. It preserves escaped multiline formatting, emits structural tokens like `<nl>` / indentation buckets, normalizes literals such as `<str_lit>` and `<num_lit>`, adds language/task tags like `<lang:rust>` and `<ctx>`, and falls back to UTF-8 byte tokens for uncovered pieces instead of collapsing rare identifiers to `<unk>`.

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
- pipeline scripts group code-first runs as `runs/code_poc_<timestamp>/{latent,world,decoder_code,code_eval}` and full runs as `runs/pipeline_<timestamp>/{latent,world,decoder_code,decoder_text}`
- each training run also records GPU memory telemetry under `memory/*` in TensorBoard and `memory_summary.txt` in the run directory
- latent, world, and decoder training now support `--grad-accum <int>` so you can trade wall-clock time for larger effective batch / context on small GPUs
- training also supports `TOFY_TRAIN_DTYPE=bf16|f16|f32`; the main scripts now default to `bf16` on GPU and fall back to `f32` on CPU
- the main training scripts now expose stage-specific microbatch overrides:
  - `LATENT_BATCH`, `WORLD_BATCH`, `CODE_DECODER_BATCH`, `TEXT_DECODER_BATCH`
  - `LATENT_GRAD_ACCUM`, `WORLD_GRAD_ACCUM`, `CODE_DECODER_GRAD_ACCUM`, `TEXT_DECODER_GRAD_ACCUM`
- encoder masking now enforces a real minimum target fraction, masks code rows more aggressively, and uses paired augmented views so chunk/global cosine is harder to game
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
- the code-first pipeline now adds compiler-feedback Rust repair pairs when `rustc` is available; repair prompts use tool/context tags like `<action:repair_patch>`, `<tool:read_error>`, and `<ctx:compiler_feedback>` while remaining compatible with the existing three-action router
- generated Rust repair pairs now use a manifest-validated cache keyed by the instruction-pair input hash, `rustc` version, and generation settings; reruns print `Repair pair cache hit: ...` when the artifact can be reused
- encoder TensorBoard now includes `loss/pred_token`, `loss/pred_chunk`, `loss/pred_global`, `metrics/chunk_cosine`, and `metrics/global_cosine`
- view metrics with `tensorboard --logdir runs/`
