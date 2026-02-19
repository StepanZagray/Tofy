# Data Formats

## Latent / JEPA training (`--latent`)

Accepted line formats:

- `context<TAB>response`
- `context|||response`
- Plain text line (entire line used as context)

Behavior:

- Only the left side is used when separators are present.
- Wikipedia hub mode writes one paragraph per line.

## World model (`--train-world`, `--eval-world`)

Accepted line formats:

- `context<TAB>next_turn`
- `context|||next_turn`

Behavior:

- Both sides are used (context → state latent, next_turn → target latent).
- Action is fixed to `reply`; no action labels in the data.

## Hub caching

- Hub datasets are downloaded once and cached under `data/`.
- Re-running training reuses caches unless you delete them.
- Wikipedia: English subset; `JEPA_WIKI_MAX_FILES` limits how many parquet files are used.

## Technical / wiki-like / expert world-model data

To steer the world model (and thus the conditioning vector) toward **technical, encyclopedic, or expert-style** replies, train on **context → response** pairs where the response is factual, concise, and expert-like. The format is the same as for UltraChat: one file, each line `context<TAB>next_turn` (or `context|||next_turn`).

### Recommended datasets

| Dataset | Description | How to get pairs |
|--------|-------------|-------------------|
| **SciQ** | Science Q&A (question → short answer, with support text). | `scripts/prepare_expert_pairs.py --dataset sciq --output data/sciq_pairs.txt` |
| **Stack Exchange** (e.g. HuggingFaceH4/stack-exchange-preferences, or pmp-stack-exchange) | Question → accepted/best answer; very technical. | Prepare script (question → answer) or use JSONL with `scripts/convert_jsonl_context_response_to_tsv.py` if you export context/response. |
| **Natural Questions** (Google) | Question → short Wikipedia-based answer. | HF `natural_questions`; map (question, long_answer/short_answer) to context\\tresponse. |
| **SQuAD-style** | Context (paragraph) + question → answer span. | Format as context = "User: question\\nContext: paragraph", next_turn = answer. |
| **Wikipedia + Q&A** | Custom: (question, wiki paragraph) → wiki-style summary or answer. | Build your own TSV from any (context, response) source. |

### Preparation

1. **Same format as UltraChat**: `context<TAB>next_turn`. Context is the “state” (e.g. "User: …" or "User: …\\nContext: …"); next_turn is the target reply.
2. **Prepare from Hugging Face**: Use `scripts/prepare_expert_pairs.py` (see script help) to write a TSV from a supported dataset (e.g. SciQ).
3. **Prepare from JSONL**: If you have files with `context` and `response` keys, use `scripts/convert_jsonl_context_response_to_tsv.py --input_dir <dir> --output data/expert_pairs.txt`.
4. **Train**: `cargo run --release -- --train-world data/<expert_pairs>.txt ... --init-encoder model_latent_*.safetensors`

Mixing or fine-tuning: you can train only on expert data, or concatenate `ultrachat_pairs.txt` and `sciq_pairs.txt` (or similar) and train on the combined file so the conditioning reflects both chat and expert style.
