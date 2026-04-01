# Tofy

Tofy is a local Rust/Candle agent with a split architecture:

- **Encoder**: reads text and produces latent features using its own vocab
- **World model**: pure latent planner stack (`planner_memory -> world_transition -> orchestrator`)
- **Text decoder**: text-generalist decoder with its own vocab
- **Code decoder**: code-specialist decoder with its own vocab

Current inference path:

`encoder -> planner memory -> router/orchestrator -> decoder adapter -> decoder`

## Important clarification

The **encoder and world model are not the same artifact anymore**.

- The **encoder** turns token ids into hidden states and uses `local_models/vocabs/vocab_encoder.txt`
- The **world model** works only in latent space and does **not** own a text vocab
- Each Candle decoder has its **own** vocab saved next to its checkpoint, for example:
  - `local_models/text_decoder_90M.vocab.txt`
  - `local_models/code_decoder_90M.vocab.txt`

## Training stages

1. **Prepare data**
- chat pairs: `data/ultrachat_pairs.txt`
- code pairs: `data/multilang_pairs.txt`
- wikipedia cache: `data/cached_wikimedia_wikipedia_1.txt`
- encoder mix: generated from all of the above

2. **Train encoder**
- LeJEPA pretraining on the mixed encoder corpus
- output: `local_models/model_latent_<size>.safetensors`
- vocab: `local_models/vocabs/vocab_encoder.txt`

3. **Train world model**
- uses the frozen encoder + encoder vocab
- trains only planner/world/orchestrator weights
- output: `local_models/model_world_<size>.safetensors`

4. **Train decoders**
- text decoder uses UltraChat data and its own vocab
- code decoder uses multilingual code data and its own vocab
- each decoder checkpoint gets a sibling vocab file

## One-command pipeline

```bash
/scripts/train_full_pipeline.sh
```

Default behavior:

- encoder corpus = UltraChat + downloaded Wikipedia + multilingual code pairs
- world model data = UltraChat
- text decoder data = UltraChat
- code decoder data = multilingual code pairs from:
- training now streams batches from disk with a small shuffle buffer instead of loading whole corpora into RAM first

```bash
python scripts/prepare_github_top_code.py --output data/multilang_pairs.txt --default-languages --max-files 200000
```

## Manual quick start

Prepare UltraChat:

```bash
cargo run --release -- --prepare-ultrachat data/ultrachat_pairs.txt 6 2
```

Build multilingual code pairs:

```bash
python scripts/prepare_github_top_code.py --output data/multilang_pairs.txt --default-languages --max-files 200000
```

Build mixed encoder corpus:

```bash
python scripts/prepare_encoder_corpus.py --output data/encoder_mix.txt data/ultrachat_pairs.txt data/cached_wikimedia_wikipedia_1.txt data/multilang_pairs.txt
```

Train encoder:

```bash
cargo run --release -- --latent data/encoder_mix.txt 25000 32 768 128 6 8 8000
```

Train pure world model:

```bash
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/ultrachat_pairs.txt 40000 32 768 128 6 8 256 64 --lambda 0.2
```

Train text decoder:

```bash
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/ultrachat_pairs.txt 20000 8 128 --decoder-kind text --decoder-output local_models/text_decoder_90M.safetensors
```

Train code decoder:

```bash
cargo run --release -- --train-decoder local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors data/multilang_pairs.txt 20000 8 128 --decoder-kind code --decoder-output local_models/code_decoder_90M.safetensors
```

Serve:

```bash
cargo run --release -- --serve local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt local_models/model_world_<size>.safetensors 0.0.0.0:8080 768 128 6 8 256 64
```

## Main modes

| Mode | Output |
|------|--------|
| `--latent` | encoder checkpoint + encoder vocab |
| `--eval-jepa` | encoder metrics |
| `--train-world` | pure latent world-model checkpoint |
| `--eval-world` | world-model latent/action metrics |
| `--train-decoder` | decoder checkpoint + decoder vocab |
| `--serve` | OpenAI-compatible HTTP server |

## Docs

- `docs/RUNBOOK.md`
- `docs/CODE_DATA.md`
- `docs/OPENCODE.md`
- `docs/DECODER_RUNTIME.md`
- `docs/ARCHITECTURE_AND_CAPACITY.md`
- `docs/RESULTS.md`
- `docs/RUNBOOK.md` also explains `tensorboard-rs` logging and TensorBoard usage

## Notes

- CUDA is enabled by default
- CPU-only: `cargo run --release --no-default-features -- ...`
- training logs go to per-run directories under `runs/`
- view metrics with `tensorboard --logdir runs/`
