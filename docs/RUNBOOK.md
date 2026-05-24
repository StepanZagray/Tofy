# Runbook

Current split architecture:

`strict LeJEPA encoder -> LeJEPA action-state transition -> context compressor -> downstream decoder conditioning adapter -> decoder`

Code organization now follows that split:

- `src/main.rs` is only the entrypoint/dispatcher
- `src/cli.rs` owns shared CLI helpers
- `src/config/{latent,world}.rs` own typed command configs
- `src/tasks/latent.rs` owns latent train/eval
- `src/tasks/pipeline.rs` owns the canonical `train` multi-stage pipeline
- `src/tasks/world.rs` owns world/high-world/orchestrator/decoder training and runtime
- `src/tasks/world_support.rs` holds shared world/decoder helper logic

## Recommended proof of concept

For the current repo, the cleanest proof of concept is now paper-strict by default:

1. train the encoder with online masked-view prediction plus SIGReg only
2. train the action-state transition with next-latent prediction plus SIGReg only
3. train the integrated high-level action-conditioned state transition in the same context-slot latent space
4. train only the code decoder as a downstream emitter
5. train a Go execution-feedback decoder stage initialized from the base code decoder
6. optionally score the result on the hard Go eval suite with `--with-code-eval`

The canonical pipeline fixes `TOFY_SIGREG_SLICES=1024`, zero world action/inverse auxiliary weights, and zero decoder syntax/signature/structure auxiliary weights.

The pipeline now does Go execution-feedback decoder training after the mixed decoder stage:

- base decoder run on `data/code_poc_mix.txt`
- Go feedback decoder run on `data/code_poc_go_mix.txt`
- compiler-feedback Go repair rows from `data/go_repair_pairs.txt` are added when `go` is available
  reruns reuse `data/go_repair_pairs.txt` when its manifest still matches the instruction-pair input hash, Go version, and generation settings

The canonical training command is:

```bash
cargo run --release -- train 8gb
```

This trains the modules only. To also run verifier-guided decoder selection and
the hard Go eval suite, pass:

```bash
cargo run --release -- train 8gb --with-code-eval
```

The `8gb` and `48gb` model/profile sizes are defined in
`config/model_profiles.json`. Override that file path with
`TOFY_MODEL_PROFILES=<path>` when testing a different shape.

Use `train 48gb` for the A40 profile. Resume uses the run directory layout:

```bash
cargo run --release -- train 48gb --resume latest
cargo run --release -- train 48gb --resume code_poc_1234567890
```

Training builds and uses the pipeline vocab/token cache by default:

```bash
cargo run --release -- train 48gb
```

Stage 1 prepares the text datasets, materializes `data/encoder_mix.txt`, builds
profile-specific vocabs, and writes binary token caches before Stage 2 starts.
Later stages stream those token caches instead of retokenizing raw text in the
training loop.

The 8 GB profile uses:

- encoder/world keep `256` context
- encoder defaults to `16x16` (`256` effective) after a `16x1` warmup
- world defaults to `32x8` (`256` effective) after a `32x1` warmup
- code decoder defaults to `8x16` (`128` effective)
- training defaults to `TOFY_TRAIN_DTYPE=bf16` on GPU, with CPU forced back to `f32`
- code decoder defaults now use `max_seq=160`, `CODE_DECODER_MAX_VOCAB=24000`, and decoder FF width `3072`
- decoder conditioning-margin ablation is disabled by default to keep the decoder downstream rather than part of the LeJEPA objective

For batch/VRAM decisions, use the sustained OOM probe rather than one-step smoke tests:

```bash
cargo run --release -- --sustained-oom-probe --stage all
```

See [OOM_TESTING.md](OOM_TESTING.md).

For cloud training pod launch, bootstrap, resume, and artifact recovery, see
[RUNPOD_CURL.md](RUNPOD_CURL.md).

Training-side latent context knobs:

- `TOFY_SIGREG_SLICES=<int>` controls SIGReg random projections, default `1024`
- `TOFY_SIGREG_POINTS=<int>` controls Epps-Pulley evaluation points, default `17`
- `TOFY_LATENT_CONTEXT_SEGMENTS=<int>` widens the source window sampled during latent training
- `TOFY_LATENT_RECENT_FULL_SEGMENTS=<int>` keeps the newest latent-training segments at full resolution
- `TOFY_LATENT_HISTORY_RATIO=<float>` reserves part of the latent window for sampled older-history tokens

World/context compressor knobs:

- `TOFY_WORLD_CONTEXT_SEGMENTS=<int>` widens state-context folding for world/orchestrator/decoder training
- `TOFY_WORLD_RECENT_FULL_SEGMENTS=<int>` keeps the newest world-context segments at full resolution
- `TOFY_RECURSIVE_CONTEXT_COMPRESSION=1` turns on recurrent context-slot folding across segments
- `TOFY_CONTEXT_HYBRID_MEMORY=0|1` enables hybrid context-compressor memory for multi-segment context, default `1`
- `TOFY_CONTEXT_HYBRID_EXACT_TAIL=<int>` keeps this many newest memory slots exact before old-memory compression, default `max_seq * recent_full_segments`
- `TOFY_CONTEXT_HYBRID_BLOCK_SIZE=<int>` sets the old-memory compression block size, default `16`
- `TOFY_CONTEXT_RETRIEVAL_SLOTS=<int>` sets how many compressor slot queries retrieve old-memory summaries, default `8`
- `TOFY_CONTEXT_EXACT_OLD_TOKENS=<int>` keeps this many learned high-salience old tokens exact alongside learned old-block summaries, default `min(16, 2 * TOFY_CONTEXT_RETRIEVAL_SLOTS)`
- `TOFY_WORLD_POST_STATE_LOSS_WEIGHT=<float>` trains the transition prediction against the encoded post-turn state (`state + next`) as an auxiliary state-update target, default `0.35`
- `TOFY_WORLD_ROLLOUT_LOSS_WEIGHT=<float>` trains open-loop transition rollouts on real continuation chains found inside the world batch, default `0.25`
- `TOFY_WORLD_TRAIN_ROLLOUT_STEPS=<int>` controls both world-model chained rollout-loss depth and decoder-training transition rollouts, default pipeline value `2`
- `TOFY_WORLD_ROLLOUT_MIN_OVERLAP=<int>` minimum token overlap required to treat two rows as a real continuation chain for rollout loss, default `24`
- `TOFY_WORLD_ROLLOUT_STEPS=<int>` does the same for serve/eval generation
- `TOFY_LATENT_REASONING=0|1` enables adaptive recurrent latent test-time compute before decoder conditioning, default `1`
- `TOFY_LATENT_REASONING_STEPS=<int>` sets the max latent refinement depth, default `8` for code-like requests and `3` for text
- `TOFY_LATENT_REASONING_MIN_STEPS=<int>` sets the minimum refinement depth before early stopping, default `2` for code-like requests and `1` for text
- `TOFY_LATENT_REASONING_PATIENCE=<int>` stops latent refinement after this many non-improving steps beyond the minimum, default `2`
- `TOFY_LATENT_REASONING_ALPHA=<float>` blends each recurrent proposal with the selected next-action latent anchor, default `0.35`
- `TOFY_LATENT_REASONING_GOAL_WEIGHT`, `TOFY_LATENT_REASONING_ROUTE_WEIGHT`, and `TOFY_LATENT_REASONING_STABILITY_WEIGHT` tune the latent selection score
- the integrated high-world training stage is fixed by profile: `12000` for `8gb` and `36000` for `48gb`
- `HWM_MACRO_MIN_LEN=<int>` and `HWM_MACRO_MAX_LEN=<int>` set the primitive-action span encoded into each macro-action, defaults `2..4`
- serve/eval auto-load `runs/.../high_world/model.safetensors` next to the world checkpoint; `TOFY_HIGH_WORLD_MODEL=<path>` or `--high-world-model <path>` overrides that path
- `TOFY_HWM_HIGH_HORIZON`, `TOFY_HWM_LOW_HORIZON`, `TOFY_HWM_MACRO_CANDIDATES`, and `TOFY_HWM_SUBGOAL_WEIGHT` tune high-level subgoal search and low-level action search

Decoder training knobs:

- `TOFY_DECODER_CONDITIONING_LOSS_WEIGHT=<float>` or `--conditioning-loss-weight <float>` mixes a conditioning-margin loss into decoder training, default `0.30` for direct `--train-decoder` runs and explicitly `0.0` in the canonical pipeline decoder and Go-feedback stages
- `TOFY_DECODER_CONDITIONING_MARGIN=<float>` sets the conditioning-loss margin, default `0.10`
- `TOFY_DECODER_CONTEXT_CACHE_ROWS=<int>` bounds the in-memory cache of frozen world/context slots during decoder training, default `1024`; set `0` to disable
- Decoder training batches all gradient-accumulation rows into one frozen encoder/world prefill before slicing latents back into decoder microbatches; the logged `config/decoder_prefill_batch_rows` is `batch * grad_accum`
- `TOFY_DECODER_SYNTAX_LOSS_WEIGHT=<float>` mixes syntax-weighted CE into decoder training
- `TOFY_DECODER_SIGNATURE_LOSS_WEIGHT=<float>` upweights the predicted function-signature span during decoder training
- `TOFY_PREPARE_REPAIR_TASKS=auto|0|1` controls compiler-feedback repair data generation, default `auto`
- `CODE_REPAIR_REPEAT=<int>` controls repair-row oversampling in the code decoder mix, default `2`
- `go_feedback_steps`, `go_feedback_batch`, and `go_feedback_grad_accum` in `config/model_profiles.json` control the Go execution-feedback decoder stage

Inference-side context hierarchy knobs:

- `TOFY_ENCODER_CONTEXT_SEGMENTS=<int>` controls how many encoder segments are retained at serve/eval time, default `4`
- `TOFY_ENCODER_RECENT_FULL_SEGMENTS=<int>` controls how many newest segments keep full token-level memory, default `1`

Input prefetch throughput knobs:

- `TOFY_CACHE_PREFETCH_BATCHES=<int>` controls the bounded raw and cached stream prefetch queue, pipeline default `4`; set `0` to disable
- `TOFY_CACHE_PREFETCH_CHUNK=<int>` overrides the number of raw/cached examples decoded per prefetch chunk, default current training batch size
- `TOFY_TOKEN_CACHE_READER_MB=<int>` controls the per-stream token-cache read buffer, default `8`
- cache preparation overlaps independent source fingerprinting, vocab builds, and token-cache builds; token-cache misses are encoded in parallel with Rayon
- set `RAYON_NUM_THREADS=<int>` to cap CPU workers, `TOFY_PREPARE_CHUNK_LINES=<int>` to tune Stage 1 text-artifact chunks, `TOFY_TOKEN_CACHE_ENCODE_CHUNK_LINES=<int>` to tune token-cache build chunk size, and `TOFY_VOCAB_SCAN_CHUNK_LINES=<int>` to tune vocab sampling chunk size; chunk defaults are `16384`
- `TOFY_CONTEXT_SEGMENT_BATCH=<int>` controls the encoder/context segment micro-batch used by world, action classifier, decoder conditioning, and eval paths, default `64`
- `TOFY_ENCODER_VOCAB_SAMPLE_ROWS=<int>` and `TOFY_ENCODER_VOCAB_SAMPLE_BYTES=<int>` cap the encoder vocab scan before Stage 2 training starts; the pipeline defaults to `500000` usable sequences or `67108864` text bytes
- `TOFY_BPE_MAX_MERGES=<int>` caps tokenizer merge training; the pipeline defaults to `8192` to bound CPU-only startup time

The training pipeline also auto-exports `CUDA_COMPUTE_CAP` from `nvidia-smi` when it is available. CUDA toolkit version detection is left to `cudarc`.

Artifact ownership:

- encoder checkpoint + encoder vocab
- world checkpoint only
- text decoder checkpoint + text decoder vocab
- code decoder checkpoint + code decoder vocab

Important: the encoder now also saves a checkpoint-matched vocab next to the latent model, for example `local_models/model_latent_69.84M.vocab.txt`. Use that sibling vocab when reusing an older latent checkpoint; `local_models/vocabs/vocab_encoder.txt` is only the latest shared encoder vocab.

## Vocab and Token Cache

The training pipeline streams inputs by default:

```bash
cargo run --release -- train 8gb
```

48 GB A40 test run:

```bash
cargo run --release -- train 48gb
```

This is the larger local/cloud profile for checking whether scaling helps the
coding assistant. It uses `DIM=768`, `BRIDGE_DIM=768`, `LAYERS=12`,
`HEADS=16`, decoder width `768`, decoder FF width `3072`, and
`NUM_LATENT_TOKENS=96`.
Current 48 GB batches are encoder `32x16` (`512` effective), world `256x2`
(`512` effective) with the encoder frozen, decoder `128x2` (`256` effective),
and Go feedback `256x1` (`256` effective), replacing the old decoder `4x1`
microbatch that left most VRAM idle in the recorded RunPod training run. It
defaults to test-scale budgets: latent `75000`, world
`180000`, high-world `36000`, code decoder `120000`, and Go feedback `24000`.

Before a long A40 launch, run:

```bash
cargo run --release -- --max-vram-probe --profile 48gb --stage all
cargo run --release -- --sustained-oom-probe --profile 48gb --stage all
```

Stage 1 builds or validates:

- source data on fresh pods:
  - Go GitHub code pairs: `data/go_code_pairs.txt`
  - UltraChat pairs: `data/ultrachat_pairs.txt`
  - one-parquet Wikipedia cache: `data/cached_wikimedia_wikipedia_1.txt`
- prepared encoder corpus: `data/encoder_mix.txt`
- prepared Go instruction pairs: `data/go_instruction_pairs.txt`
- prepared Go repair pairs: `data/go_repair_pairs.txt`
- prepared world mix: `data/world_mix_pairs.txt`
- prepared code-decoder mix: `data/code_poc_mix.txt`
- prepared Go feedback decoder mix: `data/code_poc_go_mix.txt`
- encoder vocab, default `local_models/vocabs/vocab_encoder_8000_default.txt`
- code-decoder vocab, default `local_models/vocabs/vocab_code_16000_codeaware.txt`
- latent encoder token cache: `data/cache/encoder.tokens.bin`
- world token cache: `data/cache/world.tokens.bin`
- code-decoder token cache: `data/cache/code_decoder.tokens.bin`
- dual-vocab code-decoder token cache: `data/cache/code_decoder_dual.tokens.bin`
- JSON manifests under `data/cache/`

Stage 1 runs independent preparation work in parallel. Go GitHub code pairs,
UltraChat/Wikipedia source data, and the eval suite can
prepare at the same time; after instruction pairs exist, repair trajectories,
world mix, and code-decoder mix are also overlapped where dependencies allow.
This means Stage 1 log lines can interleave.

The prepared-data commands also use sidecar manifests keyed by their input files and relevant generation settings, so Stage 1 skips rebuilding unchanged text artifacts before it reaches the binary vocab/token caches. Hub-backed source files are written to temporary files and atomically renamed into place, so a stopped pod should not leave a partially written canonical dataset. Empty source files are treated as missing and regenerated where Stage 1 owns the source.

The vocab/token manifests include source path, byte length, content hash, tokenizer mode, tokenizer-spec fingerprint, vocab signature, max sequence length, and row count. If source/tokenizer-spec/vocab match and the cached max sequence is at least the requested max sequence, the pipeline skips rebuilding even if the source file mtime changed. Set `TOFY_USE_TOKEN_CACHE=0` only for manual debugging when you explicitly want to bypass compatible token caches.

Tokenizer behavior is now versioned explicitly:

- both tokenizer modes keep their deterministic rule-based pretokenizers
- both modes reserve UTF-8 byte tokens and use byte fallback instead of raw `<unk>` collapse
- code-aware mode still does identifier-aware splitting first, then falls back to UTF-8 bytes only for uncovered pieces
- cache invalidation no longer depends only on the source file and the loose mode label; changing the tokenizer spec bumps the cache/vocab manifests automatically

Fresh streaming pipeline runs save the encoder vocab immediately after BPE finishes and then keep training. Resume runs keep using checkpoint-matched vocabs to avoid accidentally pairing old weights with a new vocab.

Latent training consumes `data/cache/encoder.tokens.bin` only when the manifest vocab signature matches the active encoder vocab and the cache max sequence is large enough for segmented context. The full pipeline sets `--encoder-max-seq` to `LATENT_MAX_SEQ * 4`, which is `1024` with the current defaults.

World and orchestrator training consume `data/cache/world.tokens.bin` directly when it exists, so both stages skip raw-text tokenization in per-step training and validation batches.

Code-decoder training consumes `data/cache/code_decoder_dual.tokens.bin` when it exists. This cache stores each row twice, once with the encoder vocab for world conditioning and once with the code-decoder vocab for teacher forcing, so the two views stay row-aligned.

Disable token-cache reads with `TOFY_USE_TOKEN_CACHE=0`.

## Resuming training

Training stages support resumable sidecar checkpoints. The full pipeline keeps model files, optimizer state, TensorBoard logs, and metadata inside a single run directory under `runs/`, so resume targets a specific run instead of scanning `local_models/`.

```bash
cargo run --release -- train 8gb --resume latest
```

or:

```bash
cargo run --release -- train 8gb --resume code_poc_1234567890
```

The pipeline passes `--resume` into the supported stages automatically and reuses the exact stage output paths from the selected run. Direct commands can also use `--resume`:

```bash
cargo run --release -- --latent data/encoder_mix.txt 25000 32 640 256 7 8 8000 --grad-accum 1 --output runs/latent/manual_run/model.safetensors --resume
```

```bash
cargo run --release -- --train-world runs/latent/manual_run/model.safetensors runs/latent/manual_run/model.vocab.txt data/world_mix_pairs.txt 60000 64 640 256 7 8 640 64 --grad-accum 2 --output runs/world/manual_run/model.safetensors --resume
```

Train the integrated macro-action state transition:

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

`STAGE` comes from `TOFY_RUN_STAGE_NAME` when it is set by the pipeline, otherwise it is the default stage name such as `latent`, `world`, `orchestrator`, or `decoder`.

Resume rules:

- Keep the same architecture arguments: `DIM`, `LAYERS`, `HEADS`, `BRIDGE_DIM`, `NUM_LATENT_TOKENS`, vocab size, and decoder architecture must match the saved sidecars.
- Keep the same model output path. Changing output paths creates a new resume namespace.
- For the full pipeline, use the same run directory. `--resume latest` picks the newest matching run directory by timestamp; `--resume <run_id>` resumes that exact run.
- If optimizer sidecars do not exist, `--resume` can still load the exported best/final model weights when available, but optimizer momentum and exact step continuation are not restored.
- If `resume.json` already reached the profile step count, the stage exits without doing more training.
- Do not use old checkpoints from a different architecture, for example previous `DIM=768` 48 GB encoder/world checkpoints with the current `DIM=1024` 48 GB setup.

## Context Guide

In this project, "context" has three layers:

1. **Runtime conversation context**
2. **Pair-file context**
3. **Decoder autoregressive context**

### Runtime conversation context

At serve time, the full `messages` array is formatted into one prompt string with role prefixes such as `System:`, `User:`, and `Assistant:`. The encoder sees that full prompt text.

The encoder is still bounded per forward pass, but runtime context is no longer limited to a pure hard truncation. The agent now supports **segmented hierarchical prompt memory**:

- the newest segment keeps full token-level encoder memory
- older segments are re-encoded one segment at a time and compressed into chunk/global/context summaries
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
- context slots from the latent world path through cross-attention

So the decoder is not conditioned only on prompt tokens and not conditioned only on context compressor. It uses both.

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
- Remember that context slots are compressed memory: they preserve high-level state, not a perfect copy of every past token.

## 1. Prepare chat data

```bash
cargo run --release -- --prepare-ultrachat data/ultrachat_pairs.txt 6 2
```

## 2. Prepare code data

```bash
cargo run --release -- --prepare-github-top-code --output data/multilang_pairs.txt --default-languages --max-files 200000
```

## 3. Build the mixed encoder corpus

The one-command `train` pipeline creates `data/cached_wikimedia_wikipedia_1.txt` automatically on a fresh pod. For this manual flow, make sure that cache exists first; set `JEPA_WIKI_MAX_FILES=1` when creating it so the filename matches the command below.

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

## 6. Train the strict LeJEPA action-conditioned state transition

The transition model is latent-only and action-conditioned. It loads the encoder checkpoint and encoder vocab, then trains context/state weights with next-latent prediction plus SIGReg. The low-level predictor follows the LeWorldModel shape more closely: 6 action-conditioned transformer blocks, 16 heads, and per-block action-conditioned normalization.

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

## 6b. Fine-tune context compressor/action classifier on explicit action labels

This is a downstream action-label stage, not part of strict LeJEPA world training. The canonical `cargo run --release -- train …` pipeline does not run action-classifier fine-tuning; use this command when explicit action-label tuning is the experiment.

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

For the current Go-focused code-first POC, use Go-only code data plus instruction-shaped Go tasks:

```bash
cargo run --release -- --prepare-github-top-code --output data/go_code_pairs.txt --languages Go --max-files 120000
cargo run --release -- --prepare-go-function-tasks --input data/go_code_pairs.txt --output data/go_instruction_pairs.txt
cargo run --release -- --prepare-go-repair-tasks --input data/go_instruction_pairs.txt --output data/go_repair_pairs.txt
cargo run --release -- --prepare-code-poc-mix --output data/code_poc_mix.txt --base-pairs data/go_code_pairs.txt --instruction-pairs data/go_instruction_pairs.txt --instruction-repeat 6 --extra-pairs data/go_repair_pairs.txt --extra-repeat 2
```

Then train the base code decoder on `data/code_poc_mix.txt`. The canonical pipeline follows that with an additional Go execution-feedback pass on `data/code_poc_go_mix.txt`. Repair rows include compiler feedback and tool-like tags such as `<action:repair_patch>`, `<tool:read_error>`, and `<ctx:compiler_feedback>`; these tags still collapse to the existing `code` router label so old three-action checkpoints remain compatible.

For the second Go execution-feedback pass, build `data/code_poc_go_mix.txt`:

```bash
cargo run --release -- --prepare-github-top-code --output data/go_code_pairs.txt --languages Go --max-files 120000
cargo run --release -- --prepare-go-function-tasks --input data/go_code_pairs.txt --output data/go_instruction_pairs.txt
cargo run --release -- --prepare-go-repair-tasks --input data/go_instruction_pairs.txt --output data/go_repair_pairs.txt
cargo run --release -- --prepare-code-poc-mix --output data/code_poc_go_mix.txt --base-pairs data/go_code_pairs.txt --instruction-pairs data/go_instruction_pairs.txt --instruction-repeat 4 --extra-pairs data/go_repair_pairs.txt --extra-repeat 2
```

Go repair generation uses `go test -c` on corrupted known-good answers, keeping short compiler diagnostics as the repair signal while still giving static type errors and executable unit-test feedback.

### Manual Go-feedback decoder and Pi harness

Use this only when you want to regenerate Go-feedback data, train a separate Go
decoder outside the autonomous `train <profile>` pipeline, or exercise a trained
checkpoint through Pi. The normal `train 48gb` pipeline already prepares
`data/code_poc_go_mix.txt` and trains
`runs/<run_id>/decoder_code_go_feedback/model.safetensors`.

Important boundary: the Rust binary trains with supervised decoder CE over
generated pair files. Pi does not run inside the optimizer; it is an end-to-end
agent harness for a served checkpoint. If a Pi run produces useful repair
traces, convert them into TSV pairs before the next decoder pass.

Generate Go execution-feedback data manually:

```bash
cargo run --release -- --prepare-github-top-code --output data/go_code_pairs.txt --languages Go --max-files 120000
cargo run --release -- --prepare-go-function-tasks --input data/go_code_pairs.txt --output data/go_instruction_pairs.txt
cargo run --release -- --prepare-go-repair-tasks --input data/go_instruction_pairs.txt --output data/go_repair_pairs.txt
cargo run --release -- --prepare-code-poc-mix --output data/code_poc_go_mix.txt --base-pairs data/go_code_pairs.txt --instruction-pairs data/go_instruction_pairs.txt --instruction-repeat 4 --extra-pairs data/go_repair_pairs.txt --extra-repeat 2
```

Train a separate Go-feedback decoder from an existing run:

```bash
RUN_ID=code_poc_1234567890

cargo run --release -- --train-decoder \
  runs/${RUN_ID}/world/model.encoder.safetensors \
  runs/${RUN_ID}/latent/model.vocab.txt \
  runs/${RUN_ID}/world/model.safetensors \
  data/code_poc_go_mix.txt \
  24000 256 192 1024 12 16 1024 96 \
  --decoder-kind code \
  --decoder-output runs/${RUN_ID}/decoder_code_go/model.safetensors \
  --decoder-max-vocab 32000 \
  --grad-accum 1 \
  --conditioning-loss-weight 0.0 \
  --init-decoder runs/${RUN_ID}/decoder_code/model.safetensors
```

Serve that decoder for Pi:

```bash
export JEPA_USE_CANDLE_DECODER=1
export JEPA_CANDLE_DECODER=$PWD/runs/${RUN_ID}/decoder_code_go/model.safetensors
export JEPA_CANDLE_DECODER_VOCAB=$PWD/runs/${RUN_ID}/decoder_code_go/model.vocab.txt

cargo run --release -- --serve \
  runs/${RUN_ID}/world/model.encoder.safetensors \
  runs/${RUN_ID}/latent/model.vocab.txt \
  runs/${RUN_ID}/world/model.safetensors \
  127.0.0.1:8080 \
  1024 256 12 16 1024 96 \
  --high-world-model runs/${RUN_ID}/high_world/model.safetensors
```

Configure Pi to call the local OpenAI-compatible server:

```bash
curl -fsSL https://pi.dev/install.sh | sh
mkdir -p ~/.pi/agent
cat > ~/.pi/agent/models.json <<'EOF'
{
  "providers": {
    "tofy": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "sk-local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "tofy",
          "name": "Tofy Go Harness",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 8192,
          "maxTokens": 2048,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
EOF
```

Run a Go-specific Pi smoke:

```bash
mkdir -p runs/${RUN_ID}/pi_go_harness

pi --model tofy/tofy --mode json \
  "Implement a small Go function and tests in /tmp/tofy-pi-go-smoke. Use go test and fix failures until it passes. Return only a concise summary." \
  2>runs/${RUN_ID}/pi_go_harness/smoke.stderr \
  | tee runs/${RUN_ID}/pi_go_harness/smoke.jsonl
```

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
- `TOFY_DECODER_CONDITION_BUDGET=<slots>` or `JEPA_DECODER_CONDITION_BUDGET=<slots>` caps context-conditioning slots before the decoder conditioning adapter; `0` zeros conditioning for ablation
- `TOFY_DECODER_CROSS_ATTN_SCHEDULE=all|every-2nd|every-3rd|last-only` controls which decoder layers use context/state cross-attention
- `TOFY_DECODER_RLM=1` enables the decoder-level recursive scaffold: keep the full prompt as external RLM state, execute a command program over semantic work units, call `SUB_RLM` recursively for bounded snippets, and reuse the selected decoder backend for leaf calls
- `TOFY_DECODER_RLM=0` disables the recursive wrapper and uses one-shot decoder calls
- `TOFY_DECODER_RLM_ACTIONS=<csv>` selects wrapped actions, default `code,text,text_reply`
- `TOFY_DECODER_RLM_LEAF_TOKENS=<tokens>` sets the per-work-unit generation budget, default `256`
- `TOFY_DECODER_RLM_CHUNK_CHARS=<chars>` sets semantic work-unit size, default `2400`
- `TOFY_DECODER_RLM_MAX_UNITS=<n>` caps generated work units, default `8`
- `TOFY_DECODER_RLM_MAX_DEPTH=<n>` caps recursive `SUB_RLM` depth, default `3`
- `TOFY_DECODER_RLM_MAX_OPS=<n>` caps root RLM command execution
- `TOFY_DECODER_RLM_MODEL_PROGRAM=1` lets the decoder draft the root RLM command program; default `0` uses the deterministic semantic chunk program
- `TOFY_DECODER_RLM_PROGRAM_TOKENS=<tokens>` sets the model-generated RLM program budget, default `192`

## 10b. Code-first eval suite

Generate the suite:

```bash
cargo run --release -- --generate-go-code-eval-suite --output eval/code_assistant_go_hard.jsonl
```

Run the end-to-end eval:

```bash
cargo run --release -- --eval-code-assistant local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors eval/code_assistant_go_hard.jsonl 384 768 256 9 8 256 64 --code-decoder local_models/code_decoder_68.50M.safetensors --go-timeout-sec 6
```

The eval writes:

- `runs/code_eval/<timestamp>/results.jsonl`
- `runs/code_eval/<timestamp>/summary.txt`

The main KPI is `suite_pass_rate`. Support metrics are `route_code_acc`, `compile_rate`, and `test_pass_rate`.

Run the conditioning-efficiency Pareto sweep:

```bash
cargo run --release -- --eval-code-assistant local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors eval/code_assistant_go_hard.jsonl 384 768 256 9 8 256 64 --code-decoder local_models/code_decoder_68.50M.safetensors --go-timeout-sec 6 --conditioning-pareto --condition-budgets 0,4,8,16,32,64 --cross-schedules last-only,every-3rd,every-2nd,all
```

The Pareto sweep writes per-budget/per-schedule `results_*.jsonl` and `summary_*.txt` files plus `runs/code_eval/<timestamp>/conditioning_pareto.csv`. Use `suite_pass_rate`, `compile_rate`, `test_pass_rate`, and required-signature/constraint pass rates as the quality side of the efficiency tradeoff.

## 11. TensorBoard

This project uses the Rust crate `tensorboard-rs` via `tensorboard_rs::summary_writer::SummaryWriter`.

Training commands write event files under `runs/`.

- standalone commands still write stage-based runs such as `runs/latent/<timestamp>`
- `cargo run --release -- train <profile>` groups one full code pipeline under:
  - `runs/code_poc_<timestamp>/latent`
  - `runs/code_poc_<timestamp>/world`
  - `runs/code_poc_<timestamp>/high_world`
  - `runs/code_poc_<timestamp>/decoder_code`
  - `runs/code_poc_<timestamp>/decoder_code_go_feedback`
  - `runs/code_poc_<timestamp>/code_eval`
- grouped pipeline runs also write `meta.json` and `launch.txt` at the run root
- decoder training now logs `zero_gain`, `shuffle_gain`, and `hard_negative_gain` from zero and mismatched-conditioning ablations, plus `val/token_accuracy`, `val/identifier_accuracy`, and `val/delimiter_balance_rate` in addition to CE / perplexity / OOV

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

## 13. Full pipeline (Rust CLI only)

This repository does not ship shell training entrypoints. Use the release binary (package `jepa_ai` in `Cargo.toml`):

```bash
cargo run --release -- train 8gb
cargo run --release -- train 48gb
```

After changing dtypes, attention, context/state logic, or decoder runtime, run `--check-dtype-discipline` and use the sustained probe in `docs/OOM_TESTING.md` before long GPU runs.

```bash
cargo run --release -- --check-dtype-discipline
```
