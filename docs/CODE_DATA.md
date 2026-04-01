# Code training data

Scripts and formats for code-focused training. Output is `context<TAB>completion`; use with `--train-world` and `--train-decoder --decoder-kind code` (see [RUNBOOK.md](RUNBOOK.md)).

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

Then train the pure world model + code decoder on the output file, e.g.:

```bash
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/multilang_pairs.txt 40000 24 768 128 6 8 256 64 --lambda 0.2
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/multilang_pairs.txt 20000 8 128 --decoder-kind code --decoder-output local_models/code_decoder_90M.safetensors
```

**Options:** `--split` (train/test/validation), `--min-lines`, `--split-ratio` (prefix/completion split). See `python scripts/prepare_github_top_code.py --help`.

---

## Rust-by-Practice markdown (sunface_rust-by-practice_en)

Use **scripts/prepare_rust_by_practice_md.py** to turn the md docs under `data/sunface_rust-by-practice_en` into training data.

**JEPA encoder (--latent):** one chunk per line (split by `##` headings):

```bash
python scripts/prepare_rust_by_practice_md.py --mode jepa --output data/rust_docs_jepa.txt
cargo run --release -- --latent data/rust_docs_jepa.txt 15000 16 768 128 6 8 8000
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
