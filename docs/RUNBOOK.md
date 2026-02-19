# Runbook (End-to-End)

Copy-paste path from a fresh clone to local agent inference. This project is an **agent** (world model + decoder); there is no classification.

## 0) Open project directory

```bash
cd <repo_root>
```

## 1) Install Hugging Face CLI globally (permanent)

So `hf` works in any terminal without activating a venv:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
```

Reload your shell (e.g. `source ~/.bashrc` or new terminal), then:

```bash
pipx install huggingface_hub
hf --version
```

If your distro provides `pipx` (e.g. `python-pipx` on Arch, or AUR), you can install that instead of `pip install --user pipx`.

## 2) Authenticate Hugging Face (for model/downloads)

```bash
hf auth login
```

Use a fine-grained read-only token if you prefer.

## 3) Download a GGUF decoder

Example: Llama 3.2 1B (good for 8GB VRAM):

```bash
mkdir -p ./models
hf download bartowski/Llama-3.2-1B-Instruct-GGUF --include "*Q5_K_M*.gguf" --local-dir ./models
ls -lh ./models
```

If the include pattern matches nothing, list files and use the exact name:

```bash
hf repo-files list bartowski/Llama-3.2-1B-Instruct-GGUF
hf download bartowski/Llama-3.2-1B-Instruct-GGUF <exact_filename.gguf> --local-dir ./models
```

Alternative (Qwen 1.5B):

```bash
mkdir -p ./models
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF --include "*q5*k*m*.gguf" --local-dir ./models
```

## 4) Prepare world-model data (UltraChat, no Python)

```bash
cargo run --release -- --prepare-ultrachat data/ultrachat_pairs.txt 6 2
```

## 5) Train JEPA latent encoder

```bash
JEPA_WIKI_MAX_FILES=1 cargo run --release -- --latent hub:wikimedia/wikipedia 25000 32 368 128 4 8 8000
```

For less RAM/disk, keep `JEPA_WIKI_MAX_FILES=1`. Increase (e.g. 2) if you want more data.

## 6) Evaluate JEPA encoder (optional)

```bash
cargo run --release -- --eval-jepa model_latent_9.51M.safetensors vocab_latent.txt hub:wikimedia/wikipedia 200 32 368 128 4 8
```

## 7) Train world model (fixed action: reply)

Default (UltraChat, chat-style):

```bash
cargo run --release -- --train-world data/ultrachat_pairs.txt 40000 32 368 128 4 8 8000 256 --init-encoder model_latent_9.51M.safetensors
```

**Technical / wiki-like / expert style:** prepare expert pairs (e.g. SciQ, SQuAD) then train on that file (or mix with UltraChat). See [Data formats – Technical / expert](docs/DATA_FORMATS.md#technical--wiki-like--expert-world-model-data) and:

```bash
pip install datasets
python scripts/prepare_expert_pairs.py --dataset sciq --output data/sciq_pairs.txt
cargo run --release -- --train-world data/sciq_pairs.txt 40000 32 368 128 4 8 8000 256 --init-encoder model_latent_9.51M.safetensors
```

## 8) Evaluate world model (optional)

```bash
cargo run --release -- --eval-world model_world_*.safetensors vocab_world.txt data/ultrachat_pairs.txt 200 32 368 128 4 8 256
```

## 9) Install/verify local decoder runtime

You need `llama-cli` (from llama.cpp) on PATH.

```bash
llama-cli --help
```

If the binary is elsewhere, set:

```bash
export JEPA_DECODER_BIN=/full/path/to/llama-cli
```

## 10) Run agent inference

Point to your GGUF (if you have more than one in `./models`, set the path explicitly). Use the **world model** safetensors path only (not the teacher); do not use a glob like `model_world_*.safetensors` or the shell may pass two paths and break argument order.

```bash
export JEPA_DECODER_MODEL=./models/<your_model>.gguf
cargo run --release -- --infer-agent model_world_10.00M.safetensors vocab_world.txt "can you help me choose a gpu?" 368 128 4 8 256 4096
```

Optional env vars (default context size is 4096):

```bash
export JEPA_DECODER_CTX=4096
export JEPA_DECODER_NGL=99
export JEPA_DECODER_TEMP=0.7
```

To test with **nullified conditioning** (zero vector), add `--ablate-conditioning` at the end:

```bash
cargo run --release -- --infer-agent model_world_10.00M.safetensors vocab_world.txt "can you help me choose a gpu?" 368 128 4 8 256 4096 --ablate-conditioning
```

## 11) Run OpenAI-compatible server

Start the server for local inference (CLI clients, IDE integrations, etc.):

```bash
export JEPA_DECODER_MODEL=./models/<your_model>.gguf
cargo run --release -- --serve model_world_10.00M.safetensors vocab_world.txt 0.0.0.0:8080 368 128 4 8 256
```

Append **`--debug`** to log Prompt/Generation t/s on server stderr (response unchanged). The model can be run from **OpenCode**; see **[OPENCODE.md](OPENCODE.md)** for setup and usage.

## 12) CPU-only fallback

```bash
cargo run --release --no-default-features -- --infer-agent model_world_10.00M.safetensors vocab_world.txt "hello" 368 128 4 8 256 4096
```

## 13) Troubleshooting

- **`hf` not found:** Install globally with pipx (step 1) and ensure `~/.local/bin` is in PATH (`pipx ensurepath`).
- **GGUF "File not found":** Use `hf repo-files list <repo>` and pass the exact filename.
- **Stub backend / decoder missing:** Install llama.cpp so `llama-cli` is on PATH, or set `JEPA_DECODER_BIN` and `JEPA_DECODER_MODEL`.
- **VRAM tight:** Lower `JEPA_DECODER_CTX` and/or `JEPA_DECODER_NGL`, or use a smaller GGUF.
- **"stream did not contain valid UTF-8":** You likely passed a glob that expanded to two model files (e.g. `model_world_*.safetensors` → world + teacher). Use a single world-model path, e.g. `model_world_10.00M.safetensors`.

## 14) Scripted training shortcuts

Encoder only (25k steps, Wiki capped):

```bash
./scripts/train_encoder_25k.sh
```

Full pipeline (encoder → world warm start → end-to-end world):

```bash
./scripts/train_full_pipeline.sh
```

Override env as needed (e.g. `WIKI_MAX_FILES=2`).
