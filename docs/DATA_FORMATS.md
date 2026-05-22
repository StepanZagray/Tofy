# Data formats and hub caching

Formats and paths for JEPA, world model, and decoder training. See [RUNBOOK.md](RUNBOOK.md) for commands.

## JEPA / latent encoder

- **Input:** One context per line (e.g. one paragraph or one phrase per line).
- **Pairs:** If you have `context<TAB>response` or `context|||response`, only the **left** side is used for JEPA (encoder sees context only).
- **Hub (Wikipedia):** `hub:wikimedia/wikipedia` yields one paragraph per line. First run downloads to `data/` (see [Hub caching](#hub-caching)).

## World model and decoder

- **Format:** `context<TAB>next_turn` (tab-separated). Also supported: `context|||next_turn`.
- **Explicit action labels:** `context<TAB>next_turn<TAB>action` is also supported. Core labels are `text_reply`, `code`, and `done`. Code-tool aliases such as `inspect_file`, `edit_file`, `run_tests`, `read_error`, and `repair_patch` are accepted and collapse to the existing `code` route for checkpoint compatibility.
- **Implicit action labels:** if the third field is absent, the repo falls back to the heuristic classifier on `next_turn`.
- **World interpretation:** `context` is the current latent state, `action` conditions the transition, and `next_turn` is encoded into the next latent state. Action labels condition the action-conditioned state transition; router training is a separate downstream stage.
- **Prepare from UltraChat:** `--prepare-ultrachat` produces this format from Hugging Face UltraChat (see [RUNBOOK.md](RUNBOOK.md)).
- **Code:** Use `context<TAB>completion` pairs (e.g. from [CODE_DATA.md](CODE_DATA.md)). Compiler-feedback repair rows can include tags such as `<action:repair_patch>` and `<ctx:compiler_feedback>` on the left side; they still train the same code decoder route.

## Hub caching

- **Path:** Use `hub:<dataset_id>` as the data path (e.g. `hub:wikimedia/wikipedia`, `hub:stingning/ultrachat`). The first run downloads the dataset under `data/` (or the path you pass to the prepare/train command).
- **Atomic publishes:** Hub conversion writes to a temporary file first and renames it only after rows were produced. If a pod is stopped during conversion, reruns remove stale temp files instead of trusting them as canonical data.
- **Pipeline bootstrap:** `cargo run --release -- train <8gb|48gb>` creates missing or empty Stage 1 source files on the current machine, including `data/ultrachat_pairs.txt`, `data/rust_code_pairs.txt`, and `data/cached_wikimedia_wikipedia_1.txt`. The pipeline temporarily sets `JEPA_WIKI_MAX_FILES=1` for the Wikipedia source so fresh pods use the expected one-parquet cache file.
- **HF CLI:** `hf auth login` (or a read-only token) is required for gated or private datasets. See [RUNBOOK.md](RUNBOOK.md) for installing the `hf` CLI.

## Technical / wiki-like / expert world model data

For Q&A or expert-style pairs (e.g. SciQ, SQuAD), use `cargo run --release -- --prepare-expert-pairs ...` to produce `context<TAB>next_turn` (e.g. question TAB answer), then train the world model on that file.

**Example (SciQ):**

```bash
cargo run --release -- --prepare-expert-pairs --dataset sciq --output data/sciq_pairs.txt
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/sciq_pairs.txt 40000 32 768 256 9 8 256 64 --lambda 0.2
```

You can also build pairs manually: one line per pair, `context\tnext_turn`, and use that file with `--train-world` and `--train-decoder` as in the [RUNBOOK](RUNBOOK.md).
