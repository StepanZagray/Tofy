# Data formats

Formats used by the VecLab + Qwen bridge pipeline. For training commands see
[README.md](README.md) and [RUNPOD.md](RUNPOD.md). Corpus layout and row counts
are defined in [VECLAB_DATA_SPEC.md](VECLAB_DATA_SPEC.md).

## Pair / world rows

- **Format:** `context<TAB>next_turn` (also `context|||next_turn`). Multiline
  fields are escaped onto one physical line (`\n`, `\r`, tabs as spaces).
- **Explicit action labels:** `context<TAB>next_turn<TAB>action`. Knowledge
  rows use `fetch_docs`. Other labels (`text_reply`, `code`, `done`) remain
  recognized for compatibility.
- **`[fn:NNN]` tag:** every generated VecLab row prefixes the state field with
  a function id for sampling/split filters. The bridge strips it before
  encoder and Qwen input (`model_visible_task`).
- **Long rows:** token caches keep the **tail** of both sides when a side
  exceeds `max_seq`.

## Pipeline artifacts under `data/fictional/`

| File | Role |
|------|------|
| `veclab/` | Hidden Go reference module (eval only; never trained on) |
| `veclab_docs.txt` | Per-function documentation sections |
| `veclab_knowledge{,_train,_val}.txt` | Query/`FetchDocs`/doc transitions for LeWorldModel |
| `veclab_tasks_{train,heldout}.txt` | Instruction → gold Go pairs |
| `veclab_encoder_mix.txt` | LeJEPA encoder continue-pretrain mix |
| `veclab_bridge_transfer.txt` | **Pipeline-built** bridge curriculum (world docs + seen code rows) |
| `MANIFEST.json` | Seed, generator version, per-file SHA-256 |

Eval suite: `eval/veclab_eval.jsonl` (600 tasks; schema in
[VECLAB_DATA_SPEC.md](VECLAB_DATA_SPEC.md#4-artifacts-and-formats)).

## Bridge / eval tokenization

Bridge training and evaluation use Qwen's tokenizer (`tokenizer.json` under
`TOFY_QWEN_DIR`). Encoder/world stages use the in-repo lexical BPE vocab under
`local_models/vocabs/` / run `latent/model.vocab.txt`.

## Prepared cache HF upload

Optional local handoff of encoder/world caches:

```bash
cargo run --release -- prepare cache minimal --auto-hf-upload --hf-dataset <org/dataset-name>
```

Uploads include `data/` (with `data/cache/` manifests), `eval/`, and
`local_models/vocabs/`. For the default veclab pod workflow the pipeline builds
the small vocab cache in stage 1 (`TOFY_REQUIRE_PREPARED_CACHE` defaults to `0`
in `scripts/runpod_train.sh`). Set it to `1` only when a prepared HF cache must
be treated as authoritative.
