# Code training data

Scripts and formats for code-focused training. Output is `context<TAB>completion`; fields are escaped onto one physical line so multiline code survives without breaking the TSV reader. Use with `--train-world` and `--train-decoder --decoder-kind code` (see [RUNBOOK.md](RUNBOOK.md)).

## GitHub Top Code (ronantakizawa/github-top-code)

Use **scripts/prepare_github_top_code.py** to build `context<TAB>completion` pairs for world model and code decoder training.

**Install:** `pip install datasets`

**Default language set** (use `--default-languages`): Rust, TypeScript, Go, JavaScript, C/C++ Header, C, C++, TSX, CSS, HTML.

**Examples:**

```bash
# All languages, first 100k files (safe for memory)
python scripts/prepare_github_top_code.py --output data/github_top_code_pairs.txt --max-files 100000

# Rust only, 50k files
python scripts/prepare_github_top_code.py --output data/rust_code_pairs.txt --languages Rust --max-files 50000

# Multilingual preset: Rust, TypeScript, Go, JavaScript, C/C++ Header, C, C++, TSX, CSS, HTML
python scripts/prepare_github_top_code.py --output data/multilang_pairs.txt --default-languages --max-files 200000

# Or list languages explicitly (handy if dataset uses different names)
python scripts/prepare_github_top_code.py --output data/multilang_pairs.txt --languages Rust TypeScript Go JavaScript "C/C++ Header" C C++ TSX CSS HTML --max-files 200000
```

The generator now also:

- balances languages more evenly when `--default-languages` or `--languages` is used with `--max-files`
- skips obviously generated/minified files
- deduplicates identical pairs
- prefixes code rows with tags such as `<lang:rust>`, `<ctx>`, and `<reply>`

Then train the pure world model + code decoder on the output file, e.g.:

```bash
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/multilang_pairs.txt 40000 24 768 256 9 8 256 64 --lambda 0.2
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/multilang_pairs.txt 20000 8 160 --decoder-kind code --decoder-max-vocab 16000 --decoder-output local_models/code_decoder_90M.safetensors
```

For the router/orchestrator, prefer a chat+code mix instead of code-only or chat-only data. The mix script now writes explicit action labels and synthetic terminal `done` rows:

```bash
python scripts/prepare_world_mix.py --output data/world_mix_pairs.txt --text-pairs data/ultrachat_pairs.txt --code-pairs data/multilang_pairs.txt --code-ratio 0.35 --done-ratio 0.18
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/world_mix_pairs.txt 40000 24 768 256 9 8 256 64 --lambda 0.2 --action-loss-weight 1.0 --router-warmup 5000
cargo run --release -- --train-orchestrator local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/world_mix_pairs.txt 15000 24 768 256 9 8 256 64
```

**Options:** `--split` (train/test/validation), `--min-lines`, `--split-ratio` (prefix/completion split). See `python scripts/prepare_github_top_code.py --help`.

### Rust instruction/function tasks for the code-first POC

The hard Rust eval suite is not a generic code-continuation benchmark. It asks for:

- natural-language instruction
- exact Rust function signature
- compilable function implementation

So for the code-first proof of concept, build a second decoder dataset that matches that shape:

```bash
python scripts/prepare_rust_function_tasks.py --input data/rust_code_pairs.txt --output data/rust_instruction_pairs.txt
python scripts/prepare_code_poc_mix.py --output data/code_poc_mix.txt --base-pairs data/rust_code_pairs.txt --instruction-pairs data/rust_instruction_pairs.txt --instruction-repeat 4 --extra-pairs data/rust_docs_pairs.txt --extra-repeat 1
```

What this does:

- scans Rust code pairs
- extracts `pub fn ... { ... }` functions
- turns them into multiple prompt variants around the same exact function signature
- optionally mixes in Rust-by-Practice section pairs
- shuffles and oversamples those instruction-shaped rows before code-decoder training

This is a decoder-training improvement, not a router/world-model change. It specifically targets the failure mode where `route_code_acc` is high but the decoder still emits generic multilingual code-token soup instead of a Rust function body.

The code-first pipeline now uses these rows twice:

- first as part of the mixed decoder dataset
- then again in a short instruction-only polish phase to improve exact signature retention

---

## Rust-by-Practice markdown (sunface_rust-by-practice_en)

Use **scripts/prepare_rust_by_practice_md.py** to turn the md docs under `data/sunface_rust-by-practice_en` into training data.

**JEPA encoder (--latent):** one chunk per line (split by `##` headings):

```bash
python scripts/prepare_rust_by_practice_md.py --mode jepa --output data/rust_docs_jepa.txt
cargo run --release -- --latent data/rust_docs_jepa.txt 15000 16 768 256 9 8 8000
```

**World / decoder pairs:** consecutive sections as context TAB next:

```bash
python scripts/prepare_rust_by_practice_md.py --mode pairs --output data/rust_docs_pairs.txt
```

You can **concatenate** with code pairs for a mix of docs + code, e.g.:

```bash
cat data/rust_docs_pairs.txt data/rust_code_pairs.txt > data/rust_mixed_pairs.txt
# Then train world/decoder on data/rust_mixed_pairs.txt
```

**Options:** `--input` (default `data/sunface_rust-by-practice_en`), `--no-split-headings` (JEPA: one file per line). See `python scripts/prepare_rust_by_practice_md.py --help`.
