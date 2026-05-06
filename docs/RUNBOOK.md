# Runbook

Current split architecture:

`strict LeJEPA encoder -> action-conditioned LeJEPA world transition -> planner memory -> downstream decoder adapter -> decoder`

Code organization now follows that split:

- `src/main.rs` is only the entrypoint/dispatcher
- `src/cli.rs` owns shared CLI helpers
- `src/config/{latent,world}.rs` own typed command configs
- `src/tasks/latent.rs` owns latent train/eval
- `src/tasks/world.rs` owns world/orchestrator/decoder training and runtime
- `src/tasks/world_support.rs` holds shared world/decoder helper logic

## Recommended proof of concept

For the current repo, the cleanest proof of concept is now paper-strict by default:

1. train the encoder with online masked-view prediction plus SIGReg only
2. train the action-conditioned world transition with next-latent prediction plus SIGReg only
3. train only the code decoder as a downstream emitter
4. score the result on the hard Rust eval suite

The default scripts export `TOFY_SIGREG_SLICES=1024`, zero world action/inverse auxiliary weights, and zero decoder syntax/signature/structure auxiliary weights. A router/orchestrator tune is treated as a downstream compatibility stage and is skipped unless `ROUTER_STEPS>0`.

The code-first script now also does a short instruction-only decoder polish pass after the mixed decoder stage:

- base decoder run on `data/code_poc_mix.txt`
- optional polish run on `data/rust_instruction_pairs.txt`
- compiler-feedback repair rows from `data/rust_repair_pairs.txt` are added when `rustc` is available
  reruns reuse `data/rust_repair_pairs.txt` when its manifest still matches the instruction-pair input hash, `rustc` version, and generation settings

The training scripts now default to the 8 GB-safe profile:

- encoder/world keep `256` context
- encoder defaults to `12x2`
- world defaults to `64x2` after a `64x1` warmup
- code decoder defaults to `6x4`
- training defaults to `TOFY_TRAIN_DTYPE=bf16` on GPU, with CPU forced back to `f32`
- code decoder defaults now use `max_seq=128`
- decoder conditioning-margin ablation is disabled by default to keep the decoder downstream rather than part of the LeJEPA objective; set `TOFY_DECODER_CONDITIONING_LOSS_WEIGHT>0` or `TOFY_DECODER_ABLATION_METRICS=1` to restore it

Override that behavior with explicit stage overrides such as `LATENT_BATCH=24`, `WORLD_BATCH=48`, `CODE_DECODER_BATCH=8`.

For batch/VRAM decisions, use the sustained OOM probe rather than one-step smoke tests:

```bash
cargo run --release -- --sustained-oom-probe --stage all
```

See [OOM_TESTING.md](OOM_TESTING.md).

Training-side latent context knobs:

- `TOFY_SIGREG_SLICES=<int>` controls SIGReg random projections, default `1024`
- `TOFY_SIGREG_POINTS=<int>` controls Epps-Pulley evaluation points, default `17`
- `TOFY_LATENT_CONTEXT_SEGMENTS=<int>` widens the source window sampled during latent training
- `TOFY_LATENT_RECENT_FULL_SEGMENTS=<int>` keeps the newest latent-training segments at full resolution
- `TOFY_LATENT_HISTORY_RATIO=<float>` reserves part of the latent window for sampled older-history tokens

World/planner memory knobs:

- `TOFY_WORLD_CONTEXT_SEGMENTS=<int>` widens state-context folding for world/orchestrator/decoder training
- `TOFY_WORLD_RECENT_FULL_SEGMENTS=<int>` keeps the newest world-context segments at full resolution
- `TOFY_RECURSIVE_PLANNER_MEMORY=1` turns on recurrent planner-slot folding across segments
- `TOFY_WORLD_TRAIN_ROLLOUT_STEPS=<int>` rolls the transition model forward multiple times before decoder conditioning during training
- `TOFY_WORLD_ROLLOUT_STEPS=<int>` does the same for serve/eval generation
- `HIGH_WORLD_STEPS=<int>` enables the optional high-level world training stage in pipeline scripts, default `0`
- `HWM_MACRO_MIN_LEN=<int>` and `HWM_MACRO_MAX_LEN=<int>` set the primitive-action span encoded into each macro-action, defaults `2..4`
- `TOFY_HWM_PLANNING=1` enables hierarchical inference when a high-world checkpoint is loaded
- `TOFY_HIGH_WORLD_MODEL=<path>` or `--high-world-model <path>` loads the high-level world checkpoint for serve/eval
- `TOFY_HWM_HIGH_HORIZON`, `TOFY_HWM_LOW_HORIZON`, `TOFY_HWM_MACRO_CANDIDATES`, and `TOFY_HWM_SUBGOAL_WEIGHT` tune high-level subgoal search and low-level action search

Decoder training knobs:

- `TOFY_DECODER_SYNTAX_LOSS_WEIGHT=<float>` mixes syntax-weighted CE into decoder training
- `TOFY_DECODER_SIGNATURE_LOSS_WEIGHT=<float>` upweights the predicted Rust function-signature span during decoder training
- `TOFY_PREPARE_REPAIR_TASKS=auto|0|1` controls compiler-feedback repair data generation, default `auto`
- `RUST_REPAIR_VARIANTS_PER_SAMPLE=<int>` controls synthetic corruptions per Rust task, default `2`
- `CODE_REPAIR_REPEAT=<int>` controls repair-row oversampling in the code decoder mix, default `2`
- `CODE_POLISH_STEPS=<int>` controls the instruction-only polish phase in the code-first pipeline, default `8000`
- `CODE_POLISH_LR=<float>` sets the polish-phase learning rate, default `1e-4`

Inference-side context hierarchy knobs:

- `TOFY_ENCODER_CONTEXT_SEGMENTS=<int>` controls how many encoder segments are retained at serve/eval time, default `4`
- `TOFY_ENCODER_RECENT_FULL_SEGMENTS=<int>` controls how many newest segments keep full token-level memory, default `1`

Token-cache throughput knobs:

- `TOFY_CACHE_PREFETCH_BATCHES=<int>` controls the bounded cached-stream prefetch queue, default `2`; set `0` to disable
- `TOFY_CACHE_PREFETCH_CHUNK=<int>` overrides the number of cached examples decoded per prefetch chunk, default current training batch size
- `TOFY_TOKEN_CACHE_READER_MB=<int>` controls the per-stream token-cache read buffer, default `8`
- `TOFY_PLANNER_SEGMENT_BATCH=<int>` controls the encoder/planner segment micro-batch used by world, orchestrator, decoder conditioning, and eval paths, default `64`

The main pipeline scripts also auto-export `CUDA_COMPUTE_CAP` from `nvidia-smi` when it is available. CUDA toolkit version detection is left to `cudarc`.

Artifact ownership:

- encoder checkpoint + encoder vocab
- world checkpoint only
- text decoder checkpoint + text decoder vocab
- code decoder checkpoint + code decoder vocab

Important: the encoder now also saves a checkpoint-matched vocab next to the latent model, for example `local_models/model_latent_69.84M.vocab.txt`. Use that sibling vocab when reusing an older latent checkpoint; `local_models/vocabs/vocab_encoder.txt` is only the latest shared encoder vocab.

## Vocab and Token Cache

The pipeline scripts run a Rust cache stage before training by default:

```bash
./scripts/train_code_first_poc.sh
```

80 GB cloud run, 10x shared-width profile:

```bash
./scripts/train_code_first_poc_80gb.sh
```

This keeps the code-first architecture shape but scales all trainable modules
through the shared width: `DIM=2048`, `BRIDGE_DIM=2048`, `LAYERS=7`, `HEADS=16`,
and `NUM_LATENT_TOKENS=128`. Since most transformer weights scale with `dim^2`,
moving from `640` to `2048` is roughly a 10x parameter increase for the encoder,
world/planner path, and code decoder. It also defaults to 10x stage budgets:
latent `250000`, world `600000`, code decoder `400000`, and code polish `80000`.

Before a multi-day cloud launch, run the same profile with a sustained VRAM
probe:

```bash
TOFY_80GB_OOM_PROBE=1 ./scripts/train_code_first_poc_80gb.sh
```

48 GB A40 test run:

```bash
./scripts/train_code_first_poc_48gb.sh
```

This is a smaller profile for checking whether scaling helps the coding
assistant before committing to the full 80 GB shape. It uses `DIM=1536`,
`BRIDGE_DIM=1536`, `LAYERS=7`, `HEADS=12`, and `NUM_LATENT_TOKENS=96`, which is
about a 5.8x parameter increase from the 640-wide baseline. It defaults to
test-scale budgets: latent `75000`, world `180000`, code decoder `120000`, and
code polish `24000`.

Before a long A40 launch, run:

```bash
TOFY_48GB_OOM_PROBE=1 ./scripts/train_code_first_poc_48gb.sh
```

Stage 1 builds or validates:

- prepared encoder corpus: `data/encoder_mix.txt`
- prepared Rust instruction pairs: `data/rust_instruction_pairs.txt`
- prepared Rust repair pairs: `data/rust_repair_pairs.txt`
- prepared world mix: `data/world_mix_pairs.txt`
- prepared code-decoder mix: `data/code_poc_mix.txt`
- encoder vocab, default `local_models/vocabs/vocab_encoder_8000_default.txt`
- code-decoder vocab, default `local_models/vocabs/vocab_code_16000_codeaware.txt`
- latent encoder token cache: `data/cache/encoder.tokens.bin`
- world token cache: `data/cache/world.tokens.bin`
- code-decoder token cache: `data/cache/code_decoder.tokens.bin`
- dual-vocab code-decoder token cache: `data/cache/code_decoder_dual.tokens.bin`
- JSON manifests under `data/cache/`

The standalone command is:

```bash
cargo run --release -- --prepare-pipeline-cache data/encoder_mix.txt data/world_mix_pairs.txt data/code_poc_mix.txt local_models/vocabs/vocab_encoder_8000_default.txt local_models/vocabs/vocab_code_16000_codeaware.txt data/cache --encoder-max-vocab 8000 --code-max-vocab 16000 --encoder-max-seq 1024 --world-max-seq 256 --code-max-seq 128
```

The prepared-data commands also use sidecar manifests keyed by their input files and relevant generation settings, so Stage 1 skips rebuilding unchanged text artifacts before it reaches the binary vocab/token caches.

The vocab/token manifests include source path, byte length, content hash, tokenizer mode, tokenizer-spec fingerprint, vocab signature, max sequence length, and row count. If source/tokenizer-spec/vocab match and the cached max sequence is at least the requested max sequence, the stage skips rebuilding even if the source file mtime changed. Set `TOFY_PRETOKENIZE=0` to skip the binary token-cache portion, or pass `--force` to the standalone cache command to rebuild those artifacts.

Tokenizer behavior is now versioned explicitly:

- both tokenizer modes keep their deterministic rule-based pretokenizers
- both modes reserve UTF-8 byte tokens and use byte fallback instead of raw `<unk>` collapse
- code-aware mode still does identifier-aware splitting first, then falls back to UTF-8 bytes only for uncovered pieces
- cache invalidation no longer depends only on the source file and the loose mode label; changing the tokenizer spec bumps the cache/vocab manifests automatically

Fresh non-resume pipeline runs export the cached encoder vocab to latent training and pass the cached code vocab to code-decoder training. Resume runs keep using the checkpoint-matched vocabs to avoid accidentally pairing old weights with a new vocab.

Latent training consumes `data/cache/encoder.tokens.bin` only when the manifest vocab signature matches the active encoder vocab and the cache max sequence is large enough for segmented context. The scripts set `--encoder-max-seq` to `LATENT_MAX_SEQ * TOFY_LATENT_CONTEXT_SEGMENTS`, which is `1024` with the current defaults.

World and orchestrator training consume `data/cache/world.tokens.bin` directly when it exists, so both stages skip raw-text tokenization in per-step training and validation batches.

Code-decoder training consumes `data/cache/code_decoder_dual.tokens.bin` when it exists. This cache stores each row twice, once with the encoder vocab for world conditioning and once with the code-decoder vocab for teacher forcing, so the two views stay row-aligned.

Disable token-cache reads with `TOFY_USE_TOKEN_CACHE=0`.

## Resuming training

Training stages support resumable sidecar checkpoints. Pipeline scripts now keep model files, optimizer state, TensorBoard logs, and pipeline logs inside a single run directory under `runs/`, so resume must target a specific run instead of scanning `local_models/`.

```bash
./scripts/train_code_first_poc.sh --resume latest
```

or:

```bash
./scripts/train_code_first_poc.sh --resume code_poc_2026-04-25_12-34-56
```

```bash
./scripts/train_full_pipeline.sh --resume latest
```

The scripts pass `--resume` into the supported stages automatically and reuse the exact stage output paths from the selected run. Direct commands can also use `--resume`:

```bash
cargo run --release -- --latent data/encoder_mix.txt 25000 32 640 256 7 8 8000 --grad-accum 1 --output runs/latent/manual_run/model.safetensors --resume
```

```bash
cargo run --release -- --train-world runs/latent/manual_run/model.safetensors runs/latent/manual_run/model.vocab.txt data/world_mix_pairs.txt 60000 64 640 256 7 8 640 64 --grad-accum 2 --output runs/world/manual_run/model.safetensors --resume
```

Train the optional high-level world model:

```bash
cargo run --release -- --train-high-world runs/world/manual_run/model.encoder.safetensors runs/latent/manual_run/model.vocab.txt runs/world/manual_run/model.safetensors data/world_mix_pairs.txt 20000 64 640 256 7 8 640 64 --macro-min-len 2 --macro-max-len 4 --output runs/high_world/manual_run/model.safetensors --resume
```

```bash
cargo run --release -- --train-decoder runs/latent/manual_run/model.safetensors runs/latent/manual_run/model.vocab.txt runs/world/manual_run/model.safetensors data/code_poc_mix.txt 40000 4 128 640 7 8 640 64 --decoder-kind code --decoder-output runs/decoder/manual_run/model.safetensors --grad-accum 6 --resume
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
- For pipeline scripts, use the same run directory. `--resume latest` picks the newest matching run directory by timestamp; `--resume <run_id>` resumes that exact run.
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
cargo run --release -- --prepare-github-top-code --output data/multilang_pairs.txt --default-languages --max-files 200000
```

## 3. Build the mixed encoder corpus

Assumes you already have the downloaded Wikipedia cache such as `data/cached_wikimedia_wikipedia_1.txt`.

```bash
cargo run --release -- --prepare-encoder-corpus --output data/encoder_mix.txt data/ultrachat_pairs.txt data/cached_wikimedia_wikipedia_1.txt data/multilang_pairs.txt
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

## 6. Train the strict LeJEPA world transition

The transition model is latent-only and action-conditioned. It loads the frozen encoder checkpoint and encoder vocab, then trains planner/world weights with only next-latent prediction plus SIGReg.

Build a mixed world dataset first so the router sees text, code, and terminal done actions:

```bash
cargo run --release -- --prepare-world-mix --output data/world_mix_pairs.txt --text-pairs data/ultrachat_pairs.txt --code-pairs data/multilang_pairs.txt --code-ratio 0.35 --done-ratio 0.18
```

```bash
TOFY_SIGREG_SLICES=1024 \
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/world_mix_pairs.txt 40000 32 768 256 9 8 256 64 --lambda 0.2 --action-loss-weight 0
```

Output:

- `local_models/model_world_<size>.safetensors`
- strict runs select checkpoints by transition/SIGReg score; action/router metrics may still be logged for diagnosis, but they are not part of the strict world loss

## 6b. Fine-tune planner/orchestrator on explicit action labels

This is a downstream compatibility stage, not part of strict LeJEPA world training. The default scripts skip it with `ROUTER_STEPS=0`; enable it only when runtime routing quality is the experiment.

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

Default code dataset is the multilingual preset from `--prepare-github-top-code`.
The code decoder uses a stronger code path: pair files are now single-line escaped rows, code formatting is restored before tokenization, identifiers are split on `_`, camelCase, and digit boundaries, literals are normalized, language/context tags are added, and uncovered pieces fall back to reserved UTF-8 byte tokens instead of becoming pure `<unk>`.

```bash
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/multilang_pairs.txt 20000 8 160 --decoder-kind code --decoder-max-vocab 16000 --decoder-output local_models/code_decoder_90M.safetensors
```

Artifacts:

- `local_models/code_decoder_90M.safetensors`
- `local_models/code_decoder_90M.vocab.txt`

For the narrow code-first POC, prefer Rust-only code data plus instruction-shaped Rust tasks:

```bash
cargo run --release -- --prepare-github-top-code --output data/rust_code_pairs.txt --languages Rust --max-files 120000
cargo run --release -- --prepare-rust-function-tasks --input data/rust_code_pairs.txt --output data/rust_instruction_pairs.txt
cargo run --release -- --prepare-rust-repair-tasks --input data/rust_instruction_pairs.txt --output data/rust_repair_pairs.txt
cargo run --release -- --prepare-code-poc-mix --output data/code_poc_mix.txt --base-pairs data/rust_code_pairs.txt --instruction-pairs data/rust_instruction_pairs.txt --instruction-repeat 4 --extra-pairs data/rust_repair_pairs.txt --extra-repeat 2
```

Then train the code decoder on `data/code_poc_mix.txt`. This matches the hard Rust eval much better than a decoder trained only on multilingual code continuation. Repair rows include compiler feedback and tool-like tags such as `<action:repair_patch>`, `<tool:read_error>`, and `<ctx:compiler_feedback>`; these tags still collapse to the existing `code` router label so old three-action checkpoints remain compatible.

## 10. Serve

With GGUF fallback:

```bash
export JEPA_DECODER_MODEL=./models/your_model.gguf
cargo run --release -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 256 9 8 256 64
```

With Candle text + code decoders:

```bash
export JEPA_USE_CANDLE_DECODER=1
export JEPA_CANDLE_DECODER=./local_models/code_decoder_68.50M.safetensors
export JEPA_USE_TEXT_DECODER=1
export JEPA_TEXT_DECODER=./local_models/text_decoder_68.50M.safetensors
cargo run --release -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 256 9 8 256 64
```

If the decoder vocab files are not next to the decoder checkpoints, set:

- `JEPA_CANDLE_DECODER_VOCAB`
- `JEPA_TEXT_DECODER_VOCAB`

Optional Candle decoder inference tuning:

- `JEPA_CANDLE_DECODER_CTX=<tokens>` limits the prompt tokens kept by the Candle decoder runtime before generation
- `TOFY_RLM_CODE=1` enables the default RLM code path: keep the full prompt as external RLM environment state, execute an RLM program over prompt slices/work units, call `SUB_RLM` recursively for local Rust units, re-encode each sub-call through the world/planner, and reuse one decoder with short prompts
- `TOFY_RLM_CODE=0` disables the recursive code path and uses the old one-shot decoder call
- `TOFY_RLM_UNIT_TOKENS=<tokens>` sets the per-work-unit generation budget, default `192`
- `TOFY_RLM_MAX_UNITS=<n>` caps generated work units, default `4`
- `TOFY_RLM_MAX_DEPTH=<n>` caps recursive `SUB_RLM` depth, default `2`
- `TOFY_RLM_MAX_OPS=<n>` caps root RLM command execution, default derived from `TOFY_RLM_MAX_UNITS`
- `TOFY_RLM_ROOT_CONTEXT_CHARS=<n>` controls the short root prefix shown as metadata when model-program drafting is enabled, default `1200`
- `TOFY_RLM_MODEL_PROGRAM=1` lets the decoder draft the root RLM command program; default `0` uses the deterministic program generated from discovered Rust work units
- `TOFY_RLM_PROGRAM_TOKENS=<tokens>` sets the model-generated RLM program budget, default `160`

## 10b. Code-first eval suite

Generate the suite:

```bash
cargo run --release -- --generate-code-eval-suite --output eval/code_assistant_rust_hard.jsonl
```

Run the end-to-end eval:

```bash
cargo run --release -- --eval-code-assistant local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors eval/code_assistant_rust_hard.jsonl 384 768 256 9 8 256 64 --code-decoder local_models/code_decoder_68.50M.safetensors
```

The eval writes:

- `runs/code_eval/<timestamp>/results.jsonl`
- `runs/code_eval/<timestamp>/summary.txt`

The main KPI is `suite_pass_rate`. Support metrics are `route_code_acc`, `compile_rate`, and `test_pass_rate`.

## 11. TensorBoard

This project uses the Rust crate `tensorboard-rs` via `tensorboard_rs::summary_writer::SummaryWriter`.

Training commands write event files under `runs/`.

- standalone commands still write stage-based runs such as `runs/latent/<timestamp>`
- `./scripts/train_code_first_poc.sh` groups one proof-of-concept pipeline under:
  - `runs/code_poc_<timestamp>/latent`
  - `runs/code_poc_<timestamp>/world`
  - `runs/code_poc_<timestamp>/decoder_code`
  - `runs/code_poc_<timestamp>/code_eval`
- `./scripts/train_full_pipeline.sh` groups one full pipeline under:
  - `runs/pipeline_<timestamp>/latent`
  - `runs/pipeline_<timestamp>/world`
  - `runs/pipeline_<timestamp>/decoder_code`
  - `runs/pipeline_<timestamp>/decoder_text`
- grouped pipeline runs also write `meta.json`, `launch.txt`, and `pipeline.log` at the run root
- decoder training now logs `val/token_accuracy`, `val/identifier_accuracy`, and `val/delimiter_balance_rate` in addition to CE / perplexity / OOV

Start TensorBoard from the repository root:

```bash
tensorboard --logdir runs/
```

Then open the local URL printed by TensorBoard in your browser, usually `http://localhost:6006`.

Typical flow:

1. Start a training command such as `--latent`, `--train-world`, or `--train-decoder`.
2. In another terminal, run `tensorboard --logdir runs/`.
3. Open the Scalars tab to watch useful tags such as `loss/total`, `loss/pred`, `loss/sigreg`, `loss/trans`, `loss/token_ce`, `metrics/pred_cosine`, `metrics/trans_cosine`, `metrics/perplexity`, and `memory/used_mb`.
4. For encoder runs, pay special attention to `loss/pred_token`, `loss/pred_chunk`, `loss/pred_global`, `metrics/chunk_cosine`, and `metrics/global_cosine`. Those tell you whether the encoder is learning local detail, mid-level structure, and whole-sequence semantics instead of only improving one pooled number.
5. For strict transition-model runs, prioritize `loss/trans`, `loss/sigreg`, `metrics/trans_cosine`, and `val/selection_score`. If you enable downstream router training, compare `metrics/action_acc` with `metrics/action_balanced_acc`, `metrics/code_rate`, `metrics/pred_code_rate`, and `metrics/code_f1`; high raw accuracy with near-zero predicted code rate still means router collapse.
6. For proof-of-concept tracking, prefer the code eval suite over any single proxy metric. A lower world loss with no improvement in `suite_pass_rate` is not a meaningful win.

Each run also writes `memory_summary.txt` in its run directory with peak VRAM usage when NVIDIA telemetry is available.

If you want to clear old charts before a fresh run, remove the relevant stage directory under `runs/` first.

## 12. CPU-only

```bash
cargo run --release --no-default-features -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 256 9 8 256 64
```

## 13. Scripts

```bash
cargo run --release -- --prepare-encoder-corpus --help
cargo run --release -- --prepare-world-mix --help
cargo run --release -- --generate-code-eval-suite --help
./scripts/train_encoder_25k.sh
./scripts/train_full_pipeline.sh
./scripts/train_code_first_poc.sh
```

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
cargo run --release -- --check-dtype-discipline
```
