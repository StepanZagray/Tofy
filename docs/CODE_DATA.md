# Code training data

CLI preparation commands and formats for code-focused training. Output is `context<TAB>completion`; fields are escaped onto one physical line so multiline code survives without breaking the TSV reader. Use with `--train-world` and `--train-decoder --decoder-kind code` (see [RUNBOOK.md](RUNBOOK.md)).

## GitHub Top Code (ronantakizawa/github-top-code)

Use `cargo run --release -- --prepare-github-top-code` to build `context<TAB>completion` pairs for world model and code decoder training.

For the full training pipeline, use Go-only source rows. The `--default-languages`
option remains available for manual experiments, but it is not used by
`train <8gb|48gb|80gb>`.

**Examples:**

```bash
# All languages, first 100k files (safe for memory)
cargo run --release -- --prepare-github-top-code --output data/github_top_code_pairs.txt --max-files 100000

# Go only, current full-pipeline source
cargo run --release -- --prepare-github-top-code --output data/go_code_pairs.txt --languages Go --max-files 120000
```

The generator now also:

- balances languages more evenly when `--default-languages` or `--languages` is used with `--max-files`
- skips obviously generated/minified files
- deduplicates identical pairs
- prefixes code rows with tags such as `<lang:go>`, `<ctx>`, and `<reply>`
- relies on the code-aware tokenizer path, which still does identifier-aware splitting first but now falls back to reserved UTF-8 byte tokens for uncovered pieces instead of raw `<unk>` collapse

Then train the pure world model + code decoder on the output file, e.g.:

```bash
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/go_code_pairs.txt 40000 24 768 256 9 8 256 64 --lambda 0.2
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/go_code_pairs.txt 20000 8 192 --decoder-kind code --decoder-max-vocab 24000 --decoder-output local_models/code_decoder_90M.safetensors
```

For the router/orchestrator, prefer a chat+code mix instead of code-only or chat-only data. `--prepare-world-mix` writes explicit action labels, preserves existing third-column labels by embedding them into the target text before mixing, and adds synthetic terminal `done` rows:

```bash
cargo run --release -- --prepare-world-mix --output data/world_mix_pairs.txt --text-pairs data/ultrachat_pairs.txt --code-pairs data/go_code_pairs.txt --code-ratio 0.35 --done-ratio 0.18
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
cargo run --release -- --prepare-go-algorithm-tasks --output data/go_algorithm_pairs.txt
cargo run --release -- --prepare-go-semantics-tasks --output data/go_semantic_pairs.txt
cargo run --release -- --prepare-go-repair-tasks --input data/go_instruction_pairs.txt --output data/go_repair_pairs.txt --max-rows 20000
cargo run --release -- --prepare-code-poc-mix --output data/code_poc_mix.txt --base-pairs data/go_code_pairs.txt --instruction-pairs data/go_instruction_pairs.txt --instruction-repeat 6 --extra-pairs data/go_algorithm_pairs.txt --extra-pairs data/go_repair_pairs.txt --extra-repeat 8
```

What this does:

- scans Go code pairs
- extracts `func ... { ... }` functions
- turns them into multiple prompt variants around the same exact function signature
- adds curated algorithm/parser tasks and execution-semantics trace prompts
- optionally mixes in Go compiler-feedback repair pairs
- shuffles and oversamples those instruction-shaped rows before code-decoder training

This is a decoder-training improvement, not a router/world-model change. It specifically targets the failure mode where `route_code_acc` is high but the decoder still emits generic multilingual code-token soup instead of a Go function body.

The code-first pipeline uses these Go rows in the base decoder mix:

- base code decoder data: `data/code_poc_mix.txt`
- world/action data: `data/world_mix_pairs.txt`

The second decoder stage is also Go execution-feedback training.

### Go execution-feedback curriculum

Go is the full-pipeline execution-feedback language. The eval JSONL format uses
`language: "go"`, which runs a temporary `go test` harness and accepts Go
compiler/test output as repair feedback.

```bash
cargo run --release -- --generate-go-code-eval-suite --output eval/code_assistant_go_hard.jsonl
cargo run --release -- --prepare-go-function-tasks --input data/go_code_pairs.txt --output data/go_instruction_pairs.txt
cargo run --release -- --prepare-go-algorithm-tasks --output data/go_algorithm_pairs.txt
cargo run --release -- --prepare-go-semantics-tasks --output data/go_semantic_pairs.txt
cargo run --release -- --prepare-go-repair-tasks --input data/go_instruction_pairs.txt --output data/go_repair_pairs.txt --max-rows 20000
```

Go repair rows use plain code-fix prompts containing the original request,
previous attempt, compiler feedback, and code-only constraints. The full
pipeline builds `data/code_poc_go_mix.txt` from Go code, Go instruction pairs,
and Go repair rows, then trains
`runs/.../decoder_code_go_feedback/model.safetensors` initialized from the base
code decoder.

The assistant eval uses the same plain repair prompt shape. Direct manual eval defaults to deterministic direct decoding (`JEPA_DECODER_TEMP=0`, `TOFY_DECODER_RLM=0`, `TOFY_LATENT_REASONING=0`) unless those variables are explicitly set before the eval command; the canonical pipeline and Pi-style eval pass `--pi-agent-env` after loading `scripts/tofy_pi_runtime_env.sh`.
