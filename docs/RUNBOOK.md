# Runbook

Current split architecture:

`encoder -> planner memory -> router/orchestrator -> decoder-specific adapter -> decoder`

Artifact ownership:

- encoder checkpoint + encoder vocab
- world checkpoint only
- text decoder checkpoint + text decoder vocab
- code decoder checkpoint + code decoder vocab

## 1. Prepare chat data

```bash
cargo run --release -- --prepare-ultrachat data/ultrachat_pairs.txt 6 2
```

## 2. Prepare code data

```bash
python scripts/prepare_github_top_code.py --output data/multilang_pairs.txt --default-languages --max-files 200000
```

## 3. Build the mixed encoder corpus

Assumes you already have the downloaded Wikipedia cache such as `data/cached_wikimedia_wikipedia_1.txt`.

```bash
python scripts/prepare_encoder_corpus.py --output data/encoder_mix.txt data/ultrachat_pairs.txt data/cached_wikimedia_wikipedia_1.txt data/multilang_pairs.txt
```

## 4. Train the encoder

```bash
cargo run --release -- --latent data/encoder_mix.txt 25000 32 768 128 6 8 8000
```

Outputs:

- `local_models/model_latent_<size>.safetensors`
- `local_models/vocabs/vocab_encoder.txt`

## 5. Evaluate the encoder

```bash
cargo run --release -- --eval-jepa local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/encoder_mix.txt 200 32 768 128 6 8
```

## 6. Train the pure world model

The world model is latent-only. It loads the frozen encoder checkpoint and encoder vocab, but it saves only planner/world/orchestrator weights.

```bash
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/ultrachat_pairs.txt 40000 32 768 128 6 8 256 64 --lambda 0.2
```

Output:

- `local_models/model_world_<size>.safetensors`

## 7. Evaluate the world model

```bash
cargo run --release -- --eval-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/ultrachat_pairs.txt 200 32 768 128 6 8 256 64
```

## 8. Train the text decoder

UltraChat only by default.

```bash
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/ultrachat_pairs.txt 20000 8 128 --decoder-kind text --decoder-output local_models/text_decoder_90M.safetensors
```

Artifacts:

- `local_models/text_decoder_90M.safetensors`
- `local_models/text_decoder_90M.vocab.txt`

## 9. Train the code decoder

Default code dataset is the multilingual preset from `prepare_github_top_code.py`.

```bash
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/multilang_pairs.txt 20000 8 128 --decoder-kind code --decoder-output local_models/code_decoder_90M.safetensors
```

Artifacts:

- `local_models/code_decoder_90M.safetensors`
- `local_models/code_decoder_90M.vocab.txt`

## 10. Serve

With GGUF fallback:

```bash
export JEPA_DECODER_MODEL=./models/your_model.gguf
cargo run --release -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 128 6 8 256 64
```

With Candle text + code decoders:

```bash
export JEPA_USE_CANDLE_DECODER=1
export JEPA_CANDLE_DECODER=./local_models/code_decoder_90M.safetensors
export JEPA_USE_TEXT_DECODER=1
export JEPA_TEXT_DECODER=./local_models/text_decoder_90M.safetensors
cargo run --release -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 128 6 8 256 64
```

If the decoder vocab files are not next to the decoder checkpoints, set:

- `JEPA_CANDLE_DECODER_VOCAB`
- `JEPA_TEXT_DECODER_VOCAB`

## 11. TensorBoard

This project uses the Rust crate `tensorboard-rs` via `tensorboard_rs::summary_writer::SummaryWriter`.

Training commands write event files under per-run subdirectories in `runs/`:

- encoder pretraining: `runs/latent/<timestamp>`
- world model training: `runs/world/<timestamp>`
- decoder training: `runs/decoder/<timestamp>`

Start TensorBoard from the repository root:

```bash
tensorboard --logdir runs/
```

Then open the local URL printed by TensorBoard in your browser, usually `http://localhost:6006`.

Typical flow:

1. Start a training command such as `--latent`, `--train-world`, or `--train-decoder`.
2. In another terminal, run `tensorboard --logdir runs/`.
3. Open the Scalars tab to watch useful tags such as `loss/total`, `loss/pred`, `loss/sigreg`, `loss/trans`, `loss/action`, `loss/token_ce`, `metrics/pred_cosine`, `metrics/trans_cosine`, `metrics/action_acc`, and `metrics/perplexity`.

If you want to clear old charts before a fresh run, remove the relevant stage directory under `runs/` first.

## 12. CPU-only

```bash
cargo run --release --no-default-features -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 128 6 8 256 64
```

## 13. Scripts

```bash
./scripts/prepare_encoder_corpus.py --help
./scripts/train_encoder_25k.sh
./scripts/train_full_pipeline.sh
```
