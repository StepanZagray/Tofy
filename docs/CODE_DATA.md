# Code training data

CLI preparation commands and formats for code-focused training. Output is `context<TAB>completion`; fields are escaped onto one physical line so multiline code survives without breaking the TSV reader. Use with `--train-world` and `--train-decoder --decoder-kind code` (see [RUNBOOK.md](RUNBOOK.md)).

## GitHub Top Code (ronantakizawa/github-top-code)

Use `cargo run --release -- --prepare-github-top-code` to build `context<TAB>completion` pairs for world model and code decoder training.

**Default language set** (use `--default-languages`): Rust, TypeScript, Go, JavaScript, C/C++ Header, C, C++, TSX, CSS, HTML.

**Examples:**

```bash
# All languages, first 100k files (safe for memory)
cargo run --release -- --prepare-github-top-code --output data/github_top_code_pairs.txt --max-files 100000

# Rust only, 50k files
cargo run --release -- --prepare-github-top-code --output data/rust_code_pairs.txt --languages Rust --max-files 50000

# Multilingual preset: Rust, TypeScript, Go, JavaScript, C/C++ Header, C, C++, TSX, CSS, HTML
cargo run --release -- --prepare-github-top-code --output data/multilang_pairs.txt --default-languages --max-files 200000

# Or list languages explicitly (handy if dataset uses different names)
cargo run --release -- --prepare-github-top-code --output data/multilang_pairs.txt --languages Rust TypeScript Go JavaScript "C/C++ Header" C C++ TSX CSS HTML --max-files 200000
```

The generator now also:

- balances languages more evenly when `--default-languages` or `--languages` is used with `--max-files`
- skips obviously generated/minified files
- deduplicates identical pairs
- prefixes code rows with tags such as `<lang:rust>`, `<ctx>`, and `<reply>`
- relies on the code-aware tokenizer path, which still does identifier-aware splitting first but now falls back to reserved UTF-8 byte tokens for uncovered pieces instead of raw `<unk>` collapse

Then train the pure world model + code decoder on the output file, e.g.:

```bash
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/multilang_pairs.txt 40000 24 768 256 9 8 256 64 --lambda 0.2
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/multilang_pairs.txt 20000 8 160 --decoder-kind code --decoder-max-vocab 16000 --decoder-output local_models/code_decoder_90M.safetensors
```

For the router/orchestrator, prefer a chat+code mix instead of code-only or chat-only data. `--prepare-world-mix` writes explicit action labels and synthetic terminal `done` rows:

```bash
cargo run --release -- --prepare-world-mix --output data/world_mix_pairs.txt --text-pairs data/ultrachat_pairs.txt --code-pairs data/multilang_pairs.txt --code-ratio 0.35 --done-ratio 0.18
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/world_mix_pairs.txt 40000 24 768 256 9 8 256 64 --lambda 0.2 --action-loss-weight 0
cargo run --release -- --train-orchestrator local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/world_mix_pairs.txt 15000 24 768 256 9 8 256 64
```

**Options:** `--split` (train/test/validation), `--min-lines`, `--split-ratio` (prefix/completion split). See `cargo run --release -- --prepare-github-top-code --help`.

### Go instruction/function tasks for the code-first POC

The canonical code-first pipeline is Go-focused. The Go function-task generator is not building generic code-continuation rows. It asks for:

- natural-language instruction
- exact Go function signature
- compilable function implementation

So for the code-first proof of concept, build a second decoder dataset that matches that shape:

```bash
cargo run --release -- --prepare-github-top-code --output data/go_code_pairs.txt --languages Go --max-files 120000
cargo run --release -- --prepare-go-function-tasks --input data/go_code_pairs.txt --output data/go_instruction_pairs.txt
cargo run --release -- --prepare-go-repair-tasks --input data/go_instruction_pairs.txt --output data/go_repair_pairs.txt
cargo run --release -- --prepare-code-poc-mix --output data/code_poc_mix.txt --base-pairs data/go_code_pairs.txt --instruction-pairs data/go_instruction_pairs.txt --instruction-repeat 6 --extra-pairs data/go_repair_pairs.txt --extra-repeat 2
```

What this does:

- scans Go code pairs
- extracts `func ... { ... }` functions
- turns them into multiple prompt variants around the same exact function signature
- optionally mixes in Go compiler-feedback repair pairs
- shuffles and oversamples those instruction-shaped rows before code-decoder training

This is a decoder-training improvement, not a router/world-model change. It specifically targets the failure mode where `route_code_acc` is high but the decoder still emits generic multilingual code-token soup instead of a Go function body.

The code-first pipeline uses these Go rows in the base decoder mix:

- base code decoder data: `data/code_poc_mix.txt`
- world/action data: `data/world_mix_pairs.txt`

The second decoder stage is also Go execution-feedback training.

### Manual Rust compiler-feedback repair tasks

These commands are retained for manual Rust experiments. The canonical
`train <8gb|48gb>` pipeline is Go-focused and does not include these Rust rows.

Use `cargo run --release -- --prepare-rust-repair-tasks` to turn instruction/function rows into repair trajectories. The generator corrupts a known-good Rust answer, runs `rustc --crate-type lib`, keeps the compiler diagnostics, and writes a new pair whose left side asks the decoder to repair the failed attempt.

The generated `data/rust_repair_pairs.txt` artifact is cached with a sidecar manifest. If the instruction-pair input hash, `rustc` version, and generation settings still match, reruns print `Repair pair cache hit: ...` and skip regeneration.

The same manifest-cache pattern is used for prepared-data artifacts so manual reruns can skip unchanged generated files.

```bash
cargo run --release -- --prepare-rust-repair-tasks --input data/rust_instruction_pairs.txt --output data/rust_repair_pairs.txt
cargo run --release -- --prepare-code-poc-mix --output data/code_poc_mix.txt --base-pairs data/rust_code_pairs.txt --instruction-pairs data/rust_instruction_pairs.txt --instruction-repeat 4 --extra-pairs data/rust_repair_pairs.txt --extra-repeat 2
```

Repair prompts include tool/context tags:

- `<action:repair_patch>`
- `<tool:read_error>`
- `<tool:repair_patch>`
- `<ctx:compiler_feedback>`

These tags are text conditioning for the decoder and training signal for the context/state path. They intentionally still map to the existing `code` action label so current three-way router checkpoints remain usable.

### Go execution-feedback curriculum

Go is now supported as a fast execution-feedback language for decoder training. Use it when you want dense compile/test rewards without paying Rust compile latency on every candidate. The eval JSONL format is shared with Rust, but `language: "go"` runs a temporary `go test` harness and accepts Go compiler/test output as repair feedback.

```bash
cargo run --release -- --generate-go-code-eval-suite --output eval/code_assistant_go_hard.jsonl
cargo run --release -- --prepare-go-function-tasks --input data/go_code_pairs.txt --output data/go_instruction_pairs.txt
cargo run --release -- --prepare-go-repair-tasks --input data/go_instruction_pairs.txt --output data/go_repair_pairs.txt
```

Go repair rows use the same tags as Rust repair rows: `<action:repair_patch>`, `<tool:read_error>`, `<tool:repair_patch>`, and `<ctx:compiler_feedback>`. The canonical pipeline now builds `data/code_poc_go_mix.txt` from Go code, Go instruction pairs, and Go repair rows, then trains `runs/.../decoder_code_go_feedback/model.safetensors` initialized from the base code decoder.

---

## Rust-by-Practice markdown (sunface_rust-by-practice_en)

Use `cargo run --release -- --prepare-rust-by-practice` to turn the md docs under `data/sunface_rust-by-practice_en` into training data.

**JEPA encoder (--latent):** one chunk per line (split by `##` headings):

```bash
cargo run --release -- --prepare-rust-by-practice --mode jepa --output data/rust_docs_jepa.txt
cargo run --release -- --latent data/rust_docs_jepa.txt 15000 16 768 256 9 8 8000
```

**World / decoder pairs:** consecutive sections as context TAB next:

```bash
cargo run --release -- --prepare-rust-by-practice --mode pairs --output data/rust_docs_pairs.txt
```

You can **concatenate** with code pairs for a mix of docs + code, e.g.:

```bash
cat data/rust_docs_pairs.txt data/rust_code_pairs.txt > data/rust_mixed_pairs.txt
# Then train world/decoder on data/rust_mixed_pairs.txt
```

**Options:** `--input` (default `data/sunface_rust-by-practice_en`), `--no-split-headings` (JEPA: one file per line). See `cargo run --release -- --prepare-rust-by-practice --help`.
