# Tofy

Tofy is a local Rust/Candle agent with a split architecture:

- **Encoder**: reads text and produces latent features using its own vocab
- **Dialog transition model**: pure latent state stack (`context_compressor -> action_state_transition -> action classifier`)
- **Text decoder**: text-generalist decoder with its own vocab
- **Code decoder**: code-specialist decoder with its own vocab

Current inference path:

`encoder -> context compressor -> router/action classifier -> decoder conditioning adapter -> decoder`

## Current proof-of-concept target

The best narrow target for this repo right now is a **code-first assistant**, not a general assistant.

The strongest parts of the stack are:

- shared encoder + context/state memory
- code routing
- code decoder

The weakest part is still the text decoder, so the main KPI should be a **tiny hard code eval suite**, not generic chat quality.

## Current code layout

The repository now mirrors that split more directly in Rust:

- `src/main.rs`: thin binary entrypoint
- `src/lib.rs`: crate module tree and mode dispatch
- `src/cli.rs`: shared CLI/path helpers
- `src/config/latent.rs`: latent train/eval config parsing
- `src/config/world.rs`: world/action classifier/decoder/eval/serve config parsing
- `src/tasks/latent.rs`: latent training and JEPA evaluation
- `src/tasks/pipeline.rs`: canonical `train 8gb|48gb|80gb` full pipeline (prep -> encoder -> world -> high-world -> code decoder -> Go eval selection)
- `src/tasks/world.rs`: world/high-world/action classifier/decoder training and runtime engine
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
- Go code pairs: `data/go_code_pairs.txt`
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
- trains context/state/action classifier weights with stronger router pressure and an early router warmup
- output: `local_models/model_world_<size>.safetensors`

4. **Tune context compressor/action classifier**
- reuses the saved world checkpoint
- trains the action path separately on explicit `text_reply` / `code` / `done` labels
- optionally tunes context compressor along with the action head
- output: updated `local_models/model_world_<size>.safetensors`

5. **Train decoders**
- text decoder uses UltraChat data and its own vocab
- code decoder uses multilingual code data and its own vocab
- each decoder checkpoint gets a sibling vocab file

## One-command pipeline

```bash
cargo run --release -- train 8gb
cargo run --release -- train 48gb
```

By default, `train` runs verifier-guided decoder selection and the hard Go
code-test eval suite after decoder training:

```bash
cargo run --release -- train 8gb
```

Resume an existing run:

```bash
cargo run --release -- train 8gb --resume latest
cargo run --release -- train 48gb --resume code_poc_1234567890
```

Before long BF16 runs, run the static dtype discipline check:

```bash
cargo run --release -- --check-dtype-discipline
```

For VRAM headroom and batch-shape decisions on GPU, use the sustained probe in `docs/OOM_TESTING.md`.

The code-first POC is now Go-focused rather than programming-language
generalist:

- base code data defaults to Go code pairs
- `--prepare-go-function-tasks` derives synthetic instruction -> function pairs from the Go code corpus
- `--prepare-code-poc-mix` builds EOS-terminated rows from compiler-verified documented functions, curated algorithm tasks, and verified repair trajectories
- the second decoder stage trains on `data/code_poc_go_mix.txt`, built from verified Go instruction pairs, cold compiler-feedback repair rows, model-failure repair rows, and pass-only self-training rows when available
- the canonical eval suite is `eval/code_assistant_go_hard.jsonl`

Default behavior:

- fresh runs generate missing or empty Stage 1 source files on the current machine: Go code pairs, UltraChat pairs, and a one-parquet Wikipedia cache
- encoder corpus = UltraChat + downloaded Wikipedia + Go code pairs
- world model data = balanced chat+code mix from `--prepare-world-mix`
- world/action classifier rows now carry explicit action labels and synthetic terminal `done` rows
- text decoder data = UltraChat
- code decoder data = Go-only code POC mix built from documented, standalone-compiling GitHub functions, curated Go tasks, and compiler-verified repair rows
- after the base code decoder trains, the pipeline can mine its failed Go attempts into `data/go_model_failure_repair_pairs.txt`, chosen/rejected preference records in `data/go_model_preference_pairs.jsonl`, and pass-only rows in `data/go_pass_self_train_pairs.txt`; by default it keeps failed candidates alive for two compiler-feedback repair rounds, producing transcript-shaped repair rows inspired by MiniMax-M3's interactive coding curriculum; tune with `TOFY_GO_MODEL_FEEDBACK_ROWS`, `TOFY_GO_MODEL_FEEDBACK_CANDIDATES`, `TOFY_GO_MODEL_FEEDBACK_REPAIR_ROUNDS`, and `TOFY_GO_MODEL_FEEDBACK_PASS_MIN_COMPILE_RATE`
- Stage 1 now covers source bootstrapping, prepared-data artifacts, and vocab/token caches; unchanged reruns avoid rebuilding them through sidecar manifests and non-empty file checks
- hub-backed dataset files are published atomically, so an interrupted pod leaves a temporary file instead of replacing the canonical training input
- the pipeline CLI now saves stage checkpoints, launch metadata, and grouped TensorBoard outputs under run-owned directories such as `runs/code_poc_<timestamp>/...`
- training now streams cached token batches from disk instead of retokenizing raw text in the hot loop
- pipeline locations are configurable with `TOFY_RUNS_DIR`, `TOFY_CACHE_DIR`, `TOFY_VOCAB_DIR`, `TOFY_HUB_CACHE_DIR`, and `TOFY_SERVE_BIND`; `TOFY_DATA_DIR` also redirects default hub cache files to `$TOFY_DATA_DIR/hub`
- cache preparation overlaps independent CPU jobs and encodes cache misses with Rayon; set `RAYON_NUM_THREADS=N` to cap CPU workers, `TOFY_PREPARE_CHUNK_LINES=N` to tune Stage 1 text chunks, `TOFY_TOKEN_CACHE_ENCODE_CHUNK_LINES=N` to tune token-cache build chunks, or `TOFY_VOCAB_SCAN_CHUNK_LINES=N` to tune vocab sampling chunks
- `cargo run --release -- prepare cache <8gb|48gb|80gb>` runs Stage 1 only, so you can prepare source data, mixes, eval data, vocabs, and token caches locally before copying `data/`, `eval/`, and `local_models/vocabs/` to a pod; add `--auto-hf-upload --hf-dataset <org/dataset-name>` to archive those handoff directories and upload them to your Hugging Face dataset with the `hf` CLI (see [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md#prepared-cache-hf-upload))
- raw and cached training streams prefetch ordered chunks by default; set `TOFY_CACHE_PREFETCH_BATCHES=0` to disable, `TOFY_CACHE_PREFETCH_BATCHES=N` to tune queue depth, `TOFY_CACHE_PREFETCH_CHUNK=N` to force chunk size, or `TOFY_TOKEN_CACHE_READER_MB=N` to tune the cache reader buffer
- world/action classifier/decoder context encoding batches token segments on GPU; set `TOFY_CONTEXT_SEGMENT_BATCH=N` to tune the segment micro-batch size, default `64`
- token caches keep overlong prompt/state tails so late instructions and compiler feedback survive, while decoder targets keep their heads for imports and exact declarations
- decoder target caches carry an explicit `completion_head_v1` policy and keep completion heads so imports and exact declarations cannot be replaced by stale tail-only cache rows
- generated Go instruction and repair targets are compiler-verified before entering the decoder mix; project-context functions that do not compile standalone are discarded
- the canonical training entrypoint is `cargo run --release -- train <8gb|48gb|80gb>`:
  - encoder/world keep `256` per-segment context
  - encoder defaults to `16x4` (`64` rows with segmented context) after a `16x1` warmup
  - world defaults to `32x8` (`256` effective) after a `32x1` warmup
  - high-world planning is trained by default from the selected profile (`12000` steps for `8gb`/`48gb`, `18000` for `80gb`) and loaded automatically from the run directory
  - code decoder defaults to `8x16` (`128` effective) with `CODE_DECODER_MAX_SEQ=192`, `CODE_DECODER_MAX_VOCAB=24000`, and decoder FF width `3072`
  - model-failure Go feedback mining defaults to `1024` rows on `8gb`, `2048` on `48gb`, and `4096` on `80gb`, with `TOFY_GO_MODEL_FEEDBACK_REPAIR_ROUNDS=2`; set `TOFY_GO_MODEL_FEEDBACK_ROWS=0` to disable
  - Go compile/test eval runs by default for verifier-guided base vs Go-feedback decoder promotion
  - the canonical pipeline selects checkpoints with `--direct-greedy`: temperature, TreeCoder, recursive decoding, latent reasoning, candidates, and repair attempts are disabled so compile rate measures the decoder itself; `--pi-agent-env` remains available for a separate served-agent evaluation
  - decoder conditioning-margin ablation is fixed off in the canonical pipeline
- the 80 GB decoder uses width `1024`, `12` layers, FF width `4096`, and `32x8` (`256` effective) for `240000` base plus `60000` Go-feedback steps; FP32 optimizer master weights preserve BF16 updates
- the 48 GB A40 profile is available through `cargo run --release -- train 48gb`:
  - uses shared `DIM=768`, `BRIDGE_DIM=768`, `LAYERS=12`, `HEADS=16`
  - uses `MAX_VOCAB=16000`, `CODE_DECODER_MAX_VOCAB=32000`, `NUM_LATENT_TOKENS=96`
  - encoder uses `48x11` (`528` effective) after a `32x1` warmup
  - world uses a frozen encoder with `128x4` (`512` effective) after a `64x1` warmup
  - code decoder and Go feedback use `32x8` (`256` effective) with `CODE_DECODER_MAX_SEQ=256`, decoder width `768`, and FF width `3072`
  - runs long code-quality budgets by default: latent `16000`, world `60000`, high-world `12000`, code decoder `80000`, Go feedback `20000`
- on CUDA, the pipeline auto-exports `CUDA_COMPUTE_CAP` if you did not already set it

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

Tune context compressor/action classifier:

```bash
cargo run --release -- --train-orchestrator local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/world_mix_pairs.txt 15000 32 768 256 9 8 256 64
```

Train text decoder:

```bash
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/ultrachat_pairs.txt 20000 8 128 --decoder-kind text --decoder-max-vocab 16000 --decoder-output local_models/text_decoder_90M.safetensors
```

Train code decoder:

```bash
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/multilang_pairs.txt 20000 8 192 --decoder-kind code --decoder-max-vocab 24000 --decoder-output local_models/code_decoder_90M.safetensors
```

Run the code-assistant eval suite:

```bash
cargo run --release -- --generate-go-code-eval-suite --output eval/code_assistant_go_hard.jsonl
source scripts/tofy_pi_runtime_env.sh latest 80gb
cargo run --release -- --eval-code-assistant "$TOFY_WORLD_ENCODER_MODEL" "$TOFY_ENCODER_VOCAB" "$TOFY_WORLD_MODEL" eval/code_assistant_go_hard.jsonl 384 "$TOFY_PROFILE_DIM" "$TOFY_PROFILE_MAX_SEQ" "$TOFY_PROFILE_LAYERS" "$TOFY_PROFILE_HEADS" "$TOFY_PROFILE_BRIDGE_DIM" "$TOFY_PROFILE_CONTEXT_SLOTS" --high-world-model "$TOFY_HIGH_WORLD_MODEL" --code-decoder "$JEPA_CANDLE_DECODER" --code-decoder-vocab "$JEPA_CANDLE_DECODER_VOCAB" --direct-greedy --go-timeout-sec 6
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
| `train` | full pipeline for memory profile `8gb`, `48gb`, or `80gb`: prep -> encoder -> world -> high-world -> code decoder -> Go feedback decoder -> hard Go eval selection (`src/tasks/pipeline.rs`) |
| `--latent` | encoder checkpoint + encoder vocab |
| `--eval-jepa` | encoder metrics |
| `--train-world` | pure latent world-model checkpoint |
| `--train-high-world` | high-level world (macro-action) checkpoint |
| `--train-orchestrator` | context compressor/action classifier action fine-tune |
| `--eval-world` | world-model latent/action metrics |
| `--eval-code-assistant` | end-to-end code-first assistant KPI suite |
| `--train-decoder` | decoder checkpoint + decoder vocab |
| `--serve` | OpenAI-compatible HTTP server |

## Docs

- Index: [`docs/README.md`](docs/README.md)
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md): setup through training, eval, serve, TensorBoard
- [`docs/OOM_TESTING.md`](docs/OOM_TESTING.md): sustained VRAM probes
- [`docs/DATA_FORMATS.md`](docs/DATA_FORMATS.md), [`docs/CODE_DATA.md`](docs/CODE_DATA.md)
- [`docs/ARCHITECTURE_AND_CAPACITY.md`](docs/ARCHITECTURE_AND_CAPACITY.md), [`docs/DECODER_RUNTIME.md`](docs/DECODER_RUNTIME.md)
- [`docs/OPENCODE.md`](docs/OPENCODE.md), [`docs/RESULTS.md`](docs/RESULTS.md)

## Notes

- CUDA is enabled by default
- CPU-only: `cargo run --release --no-default-features -- ...`
- training logs go to per-run directories under `runs/`
- the canonical pipeline groups runs as `runs/code_poc_<timestamp>/{latent,world,high_world,decoder_code,decoder_code_go_feedback}` and writes Go eval artifacts under `code_eval*`
- each training run also records GPU memory telemetry under `memory/*` in TensorBoard and `memory_summary.txt` in the run directory
- latent, world, and decoder training now support `--grad-accum <int>` so you can trade wall-clock time for larger effective batch / context on small GPUs
- the canonical pipeline fixes training dtype and microbatch schedules through the selected memory profile
- encoder masking now enforces a real minimum target fraction, masks code rows more aggressively, and uses paired augmented views so chunk/global cosine is harder to game
- latent pretraining now also supports segmented training context with:
  - `TOFY_LATENT_CONTEXT_SEGMENTS=<int>`
  - `TOFY_LATENT_RECENT_FULL_SEGMENTS=<int>`
  - `TOFY_LATENT_HISTORY_RATIO=<float>`
- encoder local attention is now truly sliding-window instead of dense masked attention, and chunk size grows with sequence length so longer encoder contexts stay practical
- world checkpoint selection for `--train-world` minimizes `transition_loss + 0.2 * sigreg_loss` on validation (`world_selection_score`); logged router metrics are diagnostic unless you run `--train-orchestrator`
- decoder validation logging now also includes token accuracy, identifier accuracy, syntax-token accuracy, signature-token accuracy, signature-exact rate, function-skeleton rate, delimiter-balance rate, syntax-weighted CE, and signature-weighted CE for code generation quality
- the new code eval suite writes per-task results plus `summary.txt` under `runs/code_eval/<timestamp>/`; summaries include `pass_after_0_rate`, `pass_after_1_rate`, `pass_after_2_rate`, and `pass_after_4_rate` so repair-depth gains are visible
- Candle decoder inference now uses batched prompt prefill, prefix-LM-style latent memory tokens, DeepSeek-V4-inspired hybrid local/compressed self-attention, query-selected CSA blocks, separate HCA blocks, per-layer compressed self-attention KV cache, and precomputed cross-attention K/V for the world latent
- set `JEPA_CANDLE_DECODER_CTX=<tokens>` to cap Candle decoder prompt context at inference if you want a predictable latency / memory ceiling
- set `TOFY_DECODER_CSA_TOPK=<blocks>` to tune how many compressed long-range blocks each CSA query keeps via the on-device DSA-style top-k mask; standalone default is `8`, pipeline default is `16`
- set `TOFY_DECODER_LATENT_PREFIX=0|1` to disable/enable planner/context slots as self-attention prefix tokens, default `1`
- set `TOFY_DECODER_CONDITION_BUDGET=<slots>` and `TOFY_DECODER_CROSS_ATTN_SCHEDULE=all|every-2nd|every-3rd|last-only` to sweep conditioning efficiency at eval/runtime; `--eval-code-assistant --conditioning-pareto` writes `conditioning_pareto.csv`
- Candle latent conditioning is vector-native only: GGUF fallback no longer receives textual latent summaries, and the Candle adapter uses exact recent context slots plus compressed older latent blocks
- decoder training can use zero, near-shuffled, and farther mismatched-conditioning negatives; logs include `zero_gain`, `shuffle_gain`, and `hard_negative_gain`
- encoder/context-compressor inference now supports segmented hierarchical prompt memory:
  - `TOFY_ENCODER_CONTEXT_SEGMENTS=<int>` sets how many encoder segments are retained, default `4`
  - `TOFY_ENCODER_RECENT_FULL_SEGMENTS=<int>` sets how many newest segments keep full token-level memory, default `1`
  - `TOFY_CONTEXT_HYBRID_MEMORY=0|1` enables hybrid context-compressor memory for multi-segment context, default `1`
	  - `TOFY_CONTEXT_HYBRID_EXACT_TAIL=<int>` keeps newest memory slots exact before old-memory compression, default `max_seq * recent_full_segments`
	  - `TOFY_CONTEXT_HYBRID_BLOCK_SIZE=<int>` sets old-memory block compression size, default `16`
	  - `TOFY_CONTEXT_RETRIEVAL_SLOTS=<int>` sets query-adaptive old-memory retrieval slots, default `8`
	  - `TOFY_CONTEXT_EXACT_OLD_TOKENS=<int>` keeps learned high-salience old tokens exact alongside learned block summaries
	  - older segments are compressed into chunk/global/context summaries instead of being dropped outright
	- context/state training and serving now also support recurrent latent folding across segments:
	  - `TOFY_WORLD_CONTEXT_SEGMENTS=<int>` sets how many state segments are folded for world/decoder training
	  - `TOFY_WORLD_RECENT_FULL_SEGMENTS=<int>` sets how many newest world segments keep full token-level memory, default `1`
	  - the canonical pipeline uses hybrid context memory across segments by default; recursive context-slot folding is available with `TOFY_RECURSIVE_CONTEXT_COMPRESSION=1`
	  - `TOFY_WORLD_POST_STATE_LOSS_WEIGHT=<float>` and `TOFY_WORLD_ROLLOUT_LOSS_WEIGHT=<float>` add post-turn and chained rollout targets to world training
	  - `TOFY_WORLD_TRAIN_ROLLOUT_STEPS=<int>` controls world-training rollout depth and decoder-training transition rollouts
	  - `TOFY_WORLD_ROLLOUT_STEPS=<int>` controls serve/eval transition rollouts before decoder conditioning
	- runtime can spend adaptive test-time compute in latent space before decoding:
	  - `TOFY_LATENT_REASONING=0|1` disables/enables recurrent latent refinement, default `1`
	  - `TOFY_LATENT_REASONING_STEPS=<int>` caps refinement depth, default `8` for code-like requests and `3` for text
	  - `TOFY_LATENT_REASONING_ALPHA=<float>` blends recurrent proposals with the selected next-action latent anchor, default `0.35`
	- the code-first POC decoder path now uses Go-only code pairs plus oversampled synthetic Go instruction/function tasks and a Go execution-feedback decoder stage because the default hard eval suite measures fast compile/test repair behavior in Go
- the code-first pipeline now adds compiler-feedback Go repair pairs when `go` is available; repair prompts are plain code-fix instructions with the original request, previous attempt, compiler feedback, and code-only constraints
- encoder TensorBoard now includes `loss/pred_token`, `loss/pred_chunk`, `loss/pred_global`, `metrics/chunk_cosine`, and `metrics/global_cosine`
- view metrics with `tensorboard --logdir runs/`
