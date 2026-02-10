# Tofy

A transformer encoder–only proof-of-concept in Rust using the Candle ML framework. **Latent-only (JEPA-style)** prediction in embedding space: train with `--latent`, run with `--diff` (L2 and cosine between predicted embedding and user-provided answer). No decoder; designed to run locally (e.g. on an 8GB GPU).

## Modes

| Training | Inference | Output |
|----------|-----------|--------|
| `--latent` | `--diff` | L2 distance and cosine similarity between predicted embedding and user-provided answer |

---

## How to run

**Training** = learn from a data file (writes `model_*.safetensors` and `vocab_*.txt`).  
**Inference** = run a trained model (no learning; you type prompts and see results).

Run `cargo run --release --` with no arguments to print usage (Training vs Inference).

**GPU (CUDA):** The default build enables CUDA so the binary uses an NVIDIA GPU when available. For CPU-only (no NVIDIA GPU/driver), use `cargo run --release --no-default-features --`.

### 1. Prepare data (once)

```bash
pip install datasets
python3 scripts/prepare_casual_conversation.py --output data/casual_pairs.txt --min-tokens 2 --lower
```

### 2. Training

Train to predict in latent space. Saves `model_latent.safetensors` and `vocab_latent.txt`.
Example below is a ~5M model.

```bash
cargo run --release -- --latent hub:ag_news 120000 32 272 72 3 8 8000
```

### 3. Inference (run the model)

Compare phrase vs answer in latent space (type `phrase _ more => answer`, get L2 and cosine).

```bash
cargo run --release -- --diff model_latent.safetensors vocab_latent.txt 272 72 3 8
```

Example (copy-paste into the program):
```
so how have you _ lately? => been
```

---

## Architecture

- **Encoder**: Transformer with self-attention layers
- **Predictor**: Linear dim→dim — predicts embedding at mask; no vocab-sized decoder

## Project Structure

```
src/
├── main.rs                    # Entry point: --latent, --latent-from-checkpoint, --diff
├── model/
│   ├── attention.rs           # Multi-head attention, transformer blocks
│   ├── encoders/
│   │   ├── online_encoder.rs  # Online encoder (optimizer-updated)
│   │   └── teacher_encoder.rs # EMA teacher encoder utilities
│   ├── vocab.rs               # Vocabulary and tokenization
│   └── predictor.rs           # (legacy, unused)
├── data/
│   └── data.rs                # Data loading, batching, encoding
└── config/
    └── config.rs              # CLI argument parsing
```

## Data Format

Plain text file, one dialogue pair per line:

```
context<TAB>response
```

Example: `hello how are you\tim fine thanks`

Also accepts `context|||response`. For latent training we use **only the left side** (`context`) as a single phrase and ignore the response to avoid duplication. If no separator, the full line is treated as one phrase.

## Casual conversation (Hugging Face)

To train on [SohamGhadge/casual-conversation](https://huggingface.co/datasets/SohamGhadge/casual-conversation) (question/answer pairs, ~3.7k rows):

```bash
pip install datasets
python3 scripts/prepare_casual_conversation.py --output data/casual_pairs.txt --min-tokens 2 --lower
```

Then use `data/casual_pairs.txt` as `<data_path>` in the training commands below. The dataset is ~3.7k pairs, so use fewer steps (e.g. 20k–50k).

## Hugging Face Hub datasets (download once)

You can train on a Hugging Face dataset **without Python**: the Rust binary downloads it once via [candle-datasets](https://docs.rs/candle-datasets) / [hf-hub](https://docs.rs/hf-hub), converts parquet to a local text file, and reuses that file on every future run (no re-download).

Use `hub:<dataset_id>` as the first argument to `--latent`. The cache is written under `data/cached_<id>.txt` (e.g. `data/cached_li2017dailydialog_daily_dialog.txt`). Delete that file to force a re-download.

**Example — AG News, ~5M model:**

```bash
cargo run --release -- --latent hub:ag_news 120000 32 272 72 3 8 8000
```

Then run inference: `cargo run --release -- --diff model_latent.safetensors vocab_latent.txt 272 72 3 8`

**Example — AG News, ~10M model:**

```bash
cargo run --release -- --latent hub:ag_news 160000 64 368 72 5 8 5000
```

Then run inference: `cargo run --release -- --diff model_latent.safetensors vocab_latent.txt 368 72 5 8`

Supported parquet layouts: a row with a **string** column (e.g. `text`) → one line per row; or a **list of strings** column (e.g. `dialog`) → one line per list element. Other schemas are ignored for that row.

 Think of it as a table: each **row** is one record (e.g. one review, one dialogue). Columns might be `text`, `label`, `dialog`, etc. “Parquet rows” = those table rows; we read them one by one and pull out the text column(s).

- **We do save the dataset locally**: The dataset is written to **`data/cached_<id>.txt`** (one phrase per line). That’s your local copy. Every later training run reads from this file; we don’t re-download or re-parse Parquet.

- **How HF caches**: The **hf-hub** crate (same idea as Python’s `huggingface_hub`) keeps a **download cache** on disk, usually `~/.cache/huggingface/hub/`. When we ask for a dataset, hf-hub downloads the Parquet files into that cache (and reuses them on later runs). So:
  1. **First run with `hub:...`**: hf-hub downloads Parquet into `~/.cache/...` (if not already there) → we read those Parquet files once, convert to text → write **`data/cached_<id>.txt`** → training uses that file.
  2. **Next runs**: We see **`data/cached_<id>.txt`** exists → we use it directly (no hf-hub call, no Parquet, no network).

So there are two levels: **hf-hub’s cache** (raw Parquet in `~/.cache/...`) and **our cache** (`data/cached_<id>.txt`). We use our cache for training so we never touch the network or Parquet after the first run.

## Training

**Device:** Commands include `--features cuda`; CUDA(0) is used when available. Omit `--features cuda` for CPU-only if you don't have an NVIDIA GPU.

### Training arguments (same order for `--latent`)

See [TRAINING_ARGS_AND_BPE.md](TRAINING_ARGS_AND_BPE.md) for how each argument affects the model, parameter count, and VRAM.

- `data_path`: Path to training data (tab-separated pairs), or `hub:<dataset_id>` to download once and cache under `data/cached_<id>.txt`
- `steps`: Number of training steps (default: 10000)
- `batch`: Batch size (default: 16)
- `dim`: Embedding dimension (default: 256)
- `max_seq`: Maximum sequence length (default: 72)
- `num_layers`: Number of transformer layers (default: 4)
- `num_heads`: Number of attention heads (default: 4)
- `max_vocab`: Maximum vocabulary size (default: 32000)

---

### Latent-only training (no decoder, JEPA-style)

Train the model to predict **in embedding space only**: encoder + small predictor (dim→dim). Loss = MSE (pred vs target direction) + sampled softmax over random negatives so the correct token ranks higher at inference. Saves `model_latent.safetensors` and `vocab_latent.txt`.

```bash
cargo run --release -- --latent <data_path> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [max_vocab]
```

**Example (~5M params):** `dim=272`, `num_layers=3`, `num_heads=8`, `max_vocab=8000`.

```bash
cargo run --release -- --latent hub:ag_news 120000 32 272 72 3 8 8000
cargo run --release -- --diff model_latent.safetensors vocab_latent.txt 272 72 3 8
```

**Example (~10M params):** `dim=368`, `num_layers=5`, `num_heads=8`, `max_vocab=5000`.

```bash
cargo run --release -- --latent hub:ag_news 160000 64 368 72 5 8 5000
cargo run --release -- --diff model_latent.safetensors vocab_latent.txt 368 72 5 8
```

**Run latent diff** (compare prediction vs your answer in latent space):

```bash
cargo run --release -- --diff model_latent.safetensors vocab_latent.txt <dim> <max_seq> <num_layers> <num_heads>
```

Then type lines in the form: **`phrase with _ or [MASK] => answer`**, e.g.:

```
hello _ world => beautiful
what is your [MASK] => name
```

The program prints **L2 distance** and **cosine similarity** between the predicted embedding (at the mask) and the embedding of the word you gave as answer. Use **only** a checkpoint from `--latent` training.

**Train vs inference:** (1) The answer token is **stripped from the phrase** before encoding so the model doesn’t see the answer in the input (e.g. `good luck _ with school. => with` runs on `good luck _ school.`). (2) Latent training uses **context-only** when the mask is in the context: the target (response) part of the sequence is replaced with pads, so the model learns to predict from phrase + pads only and matches inference (no response). **You need to retrain** with the current code to get this behaviour; older checkpoints were trained with context||target, so inference (phrase only) didn’t match.

---

**Recommended for RTX 5060 Max-Q 8GB:** Default config above (`dim=512`, `num_layers=6`, `num_heads=8`, `batch=64`, `max_seq=96`, `max_vocab=50k`). If OOM, lower `batch` to 32 or 16, or reduce `dim`/`num_layers`.

## Model details

- **Multi-head attention**: Candle `nn::Linear`, `matmul`, `softmax`
- **Positional encoding**: Sinusoidal
- **Layer norm**: Pre-norm
- **Feed-forward**: 4× embedding dimension, GELU
- **Optimizer**: AdamW, lr 3e-4

## Dependencies

- `candle-core` (0.9), `candle-nn` (0.9); optional `--features cuda` for GPU (default build is CPU-only)
- `anyhow`, `rand`

## Notes

- Encoder-only (no seq2seq); local proof-of-concept. Use `--diff` with `model_latent.safetensors` and `vocab_latent.txt`.
- For best results, train for 100k–500k steps depending on data size.
