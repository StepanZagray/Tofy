# Tofy

JEPA-style world-model agent in Rust (Candle): encoder pretraining, world-model transition + decoder bridge, and local text generation via `llama.cpp`. Run `cargo run --release --` with no arguments to print usage.

## What this is

- **Encoder:** Transformer encoder trained with JEPA-style latent alignment (predictor + EMA teacher, VICReg).
- **World model:** Fixed `reply` action; transition head predicts next latent; bridge projects to decoder conditioning.
- **Agent inference:** Encode context → transition → bridge → local decoder (e.g. Llama 1B or Qwen GGUF) produces the reply.

No classification; the model is an agent only.

## Modes

| Mode | Command | Output |
|------|---------|--------|
| Prepare UltraChat | `--prepare-ultrachat` | `data/ultrachat_pairs.txt` (context↔next_turn) |
| JEPA train | `--latent` | `model_latent_<size>.safetensors`, teacher, `vocab_latent.txt` |
| JEPA eval | `--eval-jepa` | Cosine/L2 alignment, retrieval Top-1/Top-5/MRR |
| World train | `--train-world` | `model_world_<size>.safetensors`, teacher, `vocab_world.txt` |
| World eval | `--eval-world` | `transition_cos`, `transition_l2` |
| Agent infer | `--infer-agent` | Text reply via `llama-cli` + GGUF (or stub) |

**GPU:** Default build uses CUDA. CPU-only: `cargo run --release --no-default-features --`.

**Workflow:** (1) JEPA pretrain encoder `--latent`; (2) prepare world data `--prepare-ultrachat`; (3) train world model `--train-world ... --init-encoder <model_latent_*.safetensors>`; (4) run agent `--infer-agent`.

---

## How to run

### 1. Hugging Face CLI (global, permanent)

Install the `hf` CLI so it works in any terminal (no venv needed):

```bash
python -m pip install --user pipx
python -m pipx ensurepath
# Reload shell: source ~/.bashrc or open new terminal
pipx install huggingface_hub
hf --version
```

If `pipx` is available from your distro (e.g. `sudo pacman -S python-pipx` or AUR), you can use that instead of `pip install --user pipx`.

### 2. World-model data (UltraChat, Rust-native)

No Python required. Prep runs inside the project:

```bash
cargo run --release -- --prepare-ultrachat data/ultrachat_pairs.txt 6 2
```

For Hub datasets use `hub:<dataset_id>` as data path; first run downloads to `data/` (see [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md)).

### 3. JEPA encoder training (~10M params: dim 368, 4 layers, 8 heads)

Wikipedia (paragraph mode, English; limit files to avoid huge download):

```bash
JEPA_WIKI_MAX_FILES=1 cargo run --release -- --latent hub:wikimedia/wikipedia 25000 32 368 128 4 8 8000
```

Or from a local text file (one paragraph or one phrase per line):

```bash
cargo run --release -- --latent data/cached_ag_news.txt 25000 32 368 128 4 8 8000
```

### 4. JEPA evaluation

Uses the teacher encoder for the target view when `*_teacher.safetensors` exists next to the model.

```bash
cargo run --release -- --eval-jepa model_latent_9.51M.safetensors vocab_latent.txt hub:wikimedia/wikipedia 200 32 368 128 4 8
```

### 5. World-model training

Uses conversation pairs (`context<TAB>next_turn`). Action is fixed to `reply`.

```bash
cargo run --release -- --train-world data/ultrachat_pairs.txt 40000 32 368 128 4 8 8000 256 --init-encoder model_latent_9.51M.safetensors
```

### 6. World-model evaluation

```bash
cargo run --release -- --eval-world model_world_*.safetensors vocab_world.txt data/ultrachat_pairs.txt 200 32 368 128 4 8 256
```

### 7. Agent inference

Requires `llama-cli` on PATH and a GGUF model (e.g. Llama 3.2 1B or Qwen 1.5B). Defaults: first `.gguf` under `./models`, context 2048, 99 GPU layers.

```bash
# Set decoder model (optional if you have a single .gguf in ./models)
export JEPA_DECODER_MODEL=./models/llama-3.2-1b-instruct-q5_k_m.gguf

cargo run --release -- --infer-agent model_world_*.safetensors vocab_world.txt "can you help me choose a gpu?" 368 128 4 8 256 64
```

Ablation (zero conditioning) to test JEPA influence:

```bash
cargo run --release -- --infer-agent model_world_*.safetensors vocab_world.txt "your prompt" 368 128 4 8 256 64 --ablate-conditioning
```

Optional env: `JEPA_DECODER_BIN`, `JEPA_DECODER_CTX`, `JEPA_DECODER_NGL`, `JEPA_DECODER_TEMP`, `JEPA_DECODER_REPEAT_PENALTY`. See [docs/DECODER_RUNTIME.md](docs/DECODER_RUNTIME.md).

---

## Arguments (order)

- **`--prepare-ultrachat`:** `[output_path]` `[context_window]` `[min_tokens]` `[max_rows]` → default `data/ultrachat_pairs.txt`, 6, 2, no limit.
- **`--latent`:** `data_path|hub:id` `[steps]` `[batch]` `[dim]` `[max_seq]` `[num_layers]` `[num_heads]` `[max_vocab]` `[max_spans]` `[max_span_len]` `[max_masked_ratio]`.
- **`--eval-jepa`:** `model_path` `vocab_path` `data_path|hub:id` `[eval_steps]` `[batch]` `[dim]` `[max_seq]` `[num_layers]` `[num_heads]`.
- **`--train-world`:** `data_path|hub:id` `[steps]` `[batch]` `[dim]` `[max_seq]` `[num_layers]` `[num_heads]` `[max_vocab]` `[bridge_dim]` `[--lr <float>]` `[--init-encoder <path>]`.
- **`--eval-world`:** `model_path` `vocab_path` `data_path|hub:id` `[eval_steps]` `[batch]` `[dim]` `[max_seq]` `[num_layers]` `[num_heads]` `[bridge_dim]`.
- **`--infer-agent`:** `model_path` `vocab_path` `prompt` `[dim]` `[max_seq]` `[num_layers]` `[num_heads]` `[bridge_dim]` `[max_new_tokens]` `[--ablate-conditioning]`.

---

## Data format

- **Latent / JEPA:** One context per line; `context<TAB>response` or `context|||response` uses only the left side. Wikipedia hub: one paragraph per line.
- **World model:** `context<TAB>next_turn` (or `|||`). Both sides used; action fixed to `reply`.

See [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md).

---

## Project structure

```
src/
├── main.rs           # CLI: latent, eval-jepa, prepare-ultrachat, train-world, eval-world, infer-agent
├── model/
│   ├── attention.rs
│   ├── encoders/     # Online + teacher
│   ├── vocab.rs
│   ├── predictor.rs  # JEPA predictor
│   ├── world_transition.rs
│   ├── decoder_bridge.rs
│   └── decoder_runtime.rs  # llama-cli + GGUF
├── data/             # Tokenization, batching, hub cache, UltraChat prep
├── tasks/            # world.rs (train/eval/infer)
└── config/
```

---

## Docs

- [docs/README.md](docs/README.md) — Doc index.
- [docs/RUNBOOK.md](docs/RUNBOOK.md) — Copy-paste commands from setup to agent inference (with global `hf`).
- [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) — Purpose and scope.
- [docs/SETUP_AND_RUN.md](docs/SETUP_AND_RUN.md) — Prerequisites and workflows.
- [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) — Data formats and hub caching.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Modules and data flow.
- [docs/DECODER_RUNTIME.md](docs/DECODER_RUNTIME.md) — Decoder env vars and troubleshooting.
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — Conventions and validation.

---

## Dependencies

`candle-core`, `candle-nn` (0.9), `anyhow`, `rand`. CUDA by default. See `Cargo.toml`.
