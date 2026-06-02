# Data formats and hub caching

Formats and paths for JEPA, world model, and decoder training. See [RUNBOOK.md](RUNBOOK.md) for commands.

## JEPA / latent encoder

- **Input:** One context per line (e.g. one paragraph or one phrase per line).
- **Pairs:** If you have `context<TAB>response` or `context|||response`, JEPA tokenization includes both sides separated by a newline. This keeps the latent encoder exposed to completions as well as prompts when pair files are used directly.
- **Hub (Wikipedia):** `hub:wikimedia/wikipedia` yields one paragraph per line. First run downloads to `data/` (see [Hub caching](#hub-caching)).

## World model and decoder

- **Format:** `context<TAB>next_turn` (tab-separated). Also supported: `context|||next_turn`. Multiline fields should be escaped onto one physical line (`\n`, `\r`, tabs as spaces), which is what the repo's prepare commands write.
- **Explicit action labels:** `context<TAB>next_turn<TAB>action` is also supported. Core labels are `text_reply`, `code`, `done`, and `fetch_docs`. Legacy code aliases such as `inspect_file`, `edit_file`, `run_tests`, `read_error`, and `repair_patch` are accepted and collapse to the existing `code` route for checkpoint compatibility.
- **Implicit action labels:** if the third field is absent, the repo falls back to the heuristic classifier on `next_turn`.
- **World interpretation:** `context` is the current latent state, `action` conditions the transition, and `next_turn` is encoded into the next latent state. Action labels condition the action-conditioned state transition; router training is a separate downstream stage.
- **Long rows:** token caches and training batches keep the **tail** of both `context` and `next_turn` when a side exceeds `max_seq`, so late instructions, compiler feedback, return statements, and closing braces survive truncation.
- **Prepare from UltraChat:** `--prepare-ultrachat` produces this format from Hugging Face UltraChat (see [RUNBOOK.md](RUNBOOK.md)).
- **Code:** Use `context<TAB>completion` pairs (e.g. from [CODE_DATA.md](CODE_DATA.md)). Compiler-feedback repair rows should be plain code-fix prompts with the original request, previous attempt, compiler feedback, and code-only constraints.

## Hub caching

- **Path:** Use `hub:<dataset_id>` as the data path (e.g. `hub:wikimedia/wikipedia`, `hub:stingning/ultrachat`). The first run downloads the dataset under `data/` (or the path you pass to the prepare/train command).
- **Atomic publishes:** Hub conversion writes to a temporary file first and renames it only after rows were produced. If a pod is stopped during conversion, reruns remove stale temp files instead of trusting them as canonical data.
- **Pipeline bootstrap:** `cargo run --release -- train <8gb|48gb|80gb>` creates missing or empty Stage 1 source files on the current machine, including `data/ultrachat_pairs.txt`, `data/go_code_pairs.txt`, and `data/cached_wikimedia_wikipedia_1.txt`. The pipeline temporarily sets `JEPA_WIKI_MAX_FILES=1` for the Wikipedia source so fresh pods use the expected one-parquet cache file.
- **HF CLI:** `hf auth login` (or a read-only token) is required for gated or private datasets. See [RUNBOOK.md](RUNBOOK.md) for installing the `hf` CLI.

## Prepared cache HF upload

After Stage 1 completes locally or on a pod, you can publish the handoff tree to
a Hugging Face **dataset** you control:

```bash
cargo run --release -- prepare cache 80gb --auto-hf-upload --hf-dataset <org/dataset-name>
```

Requirements:

- `--auto-hf-upload` enables the archive/upload step and must be paired with
  `--hf-dataset <org/dataset-name>` (for example `--hf-dataset my-org/tofy-cache`).
  The command errors if either flag is missing its partner. There is no built-in
  default dataset.
- `tar`, `hf`, and either `pzstd` or `zstd` must be on `PATH`; run
  `hf auth login` with write access to the target dataset before uploading.
  Cache archives prefer `pzstd -p <available-cpus> -1` when available so future
  `.tar.zst` restores can parallelize the zstd decode step. Set
  `TOFY_CACHE_ARCHIVE_COMPRESS_PROGRAM` to override the compressor.

The command archives:

- `data/` (including `data/cache/` token caches and manifests)
- `eval/`
- `local_models/vocabs/`

Archives land in `runs/prepare_cache/` as
`tofy-cache-<profile>-<git-sha>-<timestamp>.tar.zst` with a matching
`.info.txt` sidecar, then both files are uploaded to the dataset you named.

Restore on another machine (same profile as the archive):

```bash
hf download --repo-type dataset <org/dataset-name> tofy-cache-80gb-<sha>-<timestamp>.tar.zst
tar -I "pzstd -d -p $(nproc)" -xf tofy-cache-80gb-<sha>-<timestamp>.tar.zst
cargo run --release -- train 80gb
```

If `pzstd` is not installed, use `tar --zstd -xf ...`. Low CPU usage during
restore usually means the bottleneck is volume write throughput rather than
zstd decoding.

Use the same memory profile (`8gb`, `48gb`, or `80gb`) when training as when the
cache was built.

## Technical / wiki-like / expert world model data

For Q&A or expert-style pairs (e.g. SciQ, SQuAD), use `cargo run --release -- --prepare-expert-pairs ...` to produce `context<TAB>next_turn` (e.g. question TAB answer), then train the world model on that file.

For directories of JSON/JSONL records with string fields named `context` and `response`, use `--convert-jsonl-context-response-to-tsv`. It accepts both `.json` and `.jsonl` files, filters by code-aware token counts, and writes escaped single-line TSV fields so multiline code does not corrupt row boundaries.

**Example (SciQ):**

```bash
cargo run --release -- --prepare-expert-pairs --dataset sciq --output data/sciq_pairs.txt
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/sciq_pairs.txt 40000 32 768 256 9 8 256 64 --lambda 0.2
```

You can also build pairs manually: one line per pair, `context\tnext_turn`, and use that file with `--train-world` and `--train-decoder` as in the [RUNBOOK](RUNBOOK.md).
