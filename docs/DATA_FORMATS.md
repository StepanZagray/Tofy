# Data formats and hub caching

Formats and paths for JEPA, world model, and decoder training. See [RUNBOOK.md](RUNBOOK.md) for commands.

## JEPA / latent encoder

- **Input:** One context per line (e.g. one paragraph or one phrase per line).
- **Pairs:** If you have `context<TAB>response` or `context|||response`, only the **left** side is used for JEPA (encoder sees context only).
- **Hub (Wikipedia):** `hub:wikimedia/wikipedia` yields one paragraph per line. First run downloads to `data/` (see [Hub caching](#hub-caching)).

## World model and decoder

- **Format:** `context<TAB>next_turn` (tab-separated). Also supported: `context|||next_turn`.
- **Explicit action labels:** `context<TAB>next_turn<TAB>action` is also supported, where `action` is one of `text_reply`, `code`, or `done`.
- **Implicit action labels:** if the third field is absent, the repo falls back to the heuristic classifier on `next_turn`.
- **Prepare from UltraChat:** `--prepare-ultrachat` produces this format from Hugging Face UltraChat (see [RUNBOOK.md](RUNBOOK.md)).
- **Code:** Use `context<TAB>completion` pairs (e.g. from [CODE_DATA.md](CODE_DATA.md)).

## Hub caching

- **Path:** Use `hub:<dataset_id>` as the data path (e.g. `hub:wikimedia/wikipedia`, `hub:stingning/ultrachat`). The first run downloads the dataset under `data/` (or the path you pass to the prepare/train command).
- **HF CLI:** `hf auth login` (or a read-only token) is required for gated or private datasets. See [RUNBOOK.md](RUNBOOK.md) for installing the `hf` CLI.

## Technical / wiki-like / expert world model data

For Q&A or expert-style pairs (e.g. SciQ, SQuAD), use a script to produce `context<TAB>next_turn` (e.g. question TAB answer), then train the world model on that file.

**Example (SciQ):**

```bash
pip install datasets
python scripts/prepare_expert_pairs.py --dataset sciq --output data/sciq_pairs.txt
cargo run --release -- --train-world local_models/model_latent_<size>.safetensors local_models/vocabs/vocab_encoder.txt data/sciq_pairs.txt 40000 32 768 256 9 8 256 64 --lambda 0.2
```

If `prepare_expert_pairs.py` is not in the repo, you can build pairs manually: one line per pair, `context\tnext_turn`, and use that file with `--train-world` and `--train-decoder` as in the [RUNBOOK](RUNBOOK.md).
