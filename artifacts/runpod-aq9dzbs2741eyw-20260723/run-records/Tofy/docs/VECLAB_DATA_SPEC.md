# Spec: `veclab` Data Generation for the World-Model Knowledge Experiment

Status: implemented corpus-generator specification. The current causal-control
requirements are in [QWEN_KNOWLEDGE_INJECTION_SPEC.md](QWEN_KNOWLEDGE_INJECTION_SPEC.md#61-july-2026-causal-control-repair).
Consumers: encoder continue-pretrain, `--train-world-knowledge`
(`src/tasks/knowledge.rs`), `--train-bridge` (`src/tasks/bridge.rs`), eval.

## 1. Principles

1. **Verifiable**: every task is checked by `go build` + `go test` against a
   hidden reference implementation.
2. **Decontaminated by construction**: function names and signatures must not
   reveal semantics; Qwen zero-shot must score ~0.
3. **Split-safe**: function-ID split (seen 1–100 / held-out 101–200) is
   embedded in every artifact; leak guards are checkable mechanically.
4. **Deterministic**: one seed → byte-identical corpus; corpus hash recorded
   in `docs/RESULTS.md`.
5. **No nominal inflation**: every generated training and validation row is
   unique. Repeating a small template set to reach a target row count is
   forbidden because it changes the effective dataset size and validation
   weighting.

## 2. Library design

- Go module `veclab.dev/veclab`, single package `veclab`, exactly 200
  exported functions, IDs 1–200.
- **Naming**: pronounceable nonsense, 2–3 syllables from a fixed syllable
  inventory (e.g. `Vorbel`, `Skenith`, `Dramp`), unique, not an English word,
  not a known stdlib/popular-library identifier. The name carries zero
  semantic information — this is the core decontamination device.
- **Types**: only `[]float64`, `float64`, `int`, `string`, `bool` and error
  returns. Pure functions, no IO, no goroutines, deterministic.
- **Semantics**: each function is a composition of 2–3 primitives drawn from
  a generation grammar, with baked-in constants, e.g. "sort descending by
  absolute value, take first k, return alternating-sign sum scaled by 1/k".
  Families (20 functions each, 10 families): slice transforms, reductions,
  pairwise ops, windowed ops, predicate/filters, index selectors, string
  encodings, numeric parsing/formatting, accumulators, hybrid slice+scalar.
  Composition makes semantics non-guessable from the signature; the same
  AST generates implementation, doc text, and test vectors, so the three can
  never disagree.
- **Hidden reference**: implementation + generated table-driven tests live in
  `data/fictional/veclab/` (the Go module). This directory is excluded from
  every training corpus. `go test ./...` must pass at generation time.

## 3. Documentation corpus

Per function, one doc section (target 150–300 encoder tokens, must fit
`world_max_seq` 384):

```
func Vorbel(xs []float64, k int) float64
Vorbel returns the alternating-sign sum of the k largest values of xs by
absolute magnitude, scaled by 1/k. Returns 0 when xs is empty or k < 1.
Example: Vorbel([3, -7, 2], 2) = (7 - 3) / 2 = 2.0
Example: Vorbel([1], 3) = 1.0
```

Signature line, prose semantics (edge cases included), and exactly 2 worked
examples with concrete values. The worked examples are what make the Step-1
RAG ceiling reachable and give the recon objective token-level content worth
storing.

## 4. Artifacts and formats

All TSV files follow `docs/DATA_FORMATS.md` conventions
(`state<TAB>next`, newlines escaped). Every row carries the function ID as a
leading inline tag in the state field: `[fn:NNN]`. It is sampling metadata:
the bridge strips it before encoder and Qwen input, while split filters and the
channel probe can still parse it from the source row.

| File | Rows | Format | Used by |
|---|---|---|---|
| `data/fictional/veclab/` | — | Go module (impl + tests) | eval only, never trained on |
| `data/fictional/veclab_docs.txt` | 200 | `[fn:NNN] <signature line>` TAB `<full doc section>` | knowledge stage (recon reads `next`) |
| `data/fictional/veclab_knowledge.txt` | 8,000 | `[fn:NNN] <unique query>` TAB `<doc section>` TAB `fetch_docs` | LeWorldModel transitions |
| `data/fictional/veclab_knowledge_train.txt` | 7,600 | same | world train split: 38 unique paraphrases for every function |
| `data/fictional/veclab_knowledge_val.txt` | 400 | same | world validation split: 2 unique, unseen paraphrases for every function |
| `data/fictional/veclab_tasks_train.txt` | 4,000 | `[fn:NNN] <unique task instruction>` TAB `<gold Go solution>` | bridge training, fns 1–100 ONLY |
| `data/fictional/veclab_tasks_heldout.txt` | 4,000 | same | never trained on; source for eval + probe eval |
| `data/fictional/veclab_encoder_mix.txt` | 20,000 | unique doc/task/reference pairs (no gold solutions) | encoder continue-pretrain |
| `eval/veclab_eval.jsonl` | 600 | JSONL, schema below | eval harness |

Eval JSONL schema:

```json
{"id": "veclab-137-2", "fn_ids": [137], "subset": "heldout",
 "task": "Write a Go function Solve(xs []float64) float64 that ...",
 "must_call": ["Vorbel"], "harness_dir": "eval/veclab/137-2/"}
```

`harness_dir` contains a `main_test.go` with input/output table checks
generated from the hidden reference.

## 5. Row-type details (matching current training code)

- **Knowledge rows** (`veclab_knowledge_{train,val}.txt`): `state` = a task-like query
  ("how do I get the alternating top-k sum...", or an actual task
  instruction), `next` = the relevant doc section. `knowledge.rs` computes
  raw next-embedding MSE plus SIGReg. Training also assembles batches with unique
  function IDs, so paraphrases of one document are never treated as negatives.
  Validation holds out paraphrases, not function identities: every one of the
  200 functions occurs in both splits.
- **Task rows** (`veclab_tasks_*.txt`): `state` = task instruction only
  (docs are injected at bridge time in the `context` regime, not baked into
  the file — this keeps one file serving both regimes). `next` = gold
  solution: a complete, compiling Go function that calls the required
  veclab function(s). Gold solutions for fns 101–200 exist only in
  `veclab_tasks_heldout.txt` and eval harness dirs.
- **Encoder mix**: doc sections for all 200 fns + task instruction texts;
  NO gold solutions for held-out fns (leak guard §7).

## 6. Task generation

Per function: 40 unique tasks in its partition's file, from two template
families:

1. **Explicit-call** (50%): "Using the veclab package, write
   `Solve(...)` that returns `Vorbel(xs, 3)` for the input" — tests whether
   the channel transmits the signature and calling convention.
2. **Implicit** (50%): describes the *behavior* ("return the alternating-sign
   sum of the 3 largest-magnitude values, scaled by 1/3") without naming the
   function; solvable only by knowing which veclab function implements it.
   Tests retrieval + usage. Gold solution still calls the veclab function.

Surface variety per task: 40 distinct wrapper signatures, varied argument names,
varied wrapper signatures, 30% of tasks compose two veclab calls (both fns
from the same split partition).

Eval: 3 tasks per function (1 explicit, 2 implicit), fresh paraphrases not
present in training files → 300 seen + 300 held-out cases.

## 7. Leak guards (mechanically checked by the generator)

1. No gold solution for fns 101–200 outside `veclab_tasks_heldout.txt` and
   `eval/veclab/`.
2. No eval paraphrase string appears in any training file (exact and
   normalized substring check).
3. `data/fictional/veclab/` (module with implementations) appears in no
   training file.
4. Batch-window uniqueness of fn IDs in `veclab_knowledge.txt` (§5).
5. Generator exits nonzero if any guard fails; guards re-run by
   `--print-split-stats` (P0.2).
6. Every encoder, knowledge-train, knowledge-validation, bridge-train, and
   held-out task row is unique; knowledge query strings are disjoint between
   train and validation.

## 8. Verification at generation time

1. `go vet` + `go test ./...` green on the hidden module.
2. Every gold solution compiles and passes its harness tests.
3. Every eval harness fails against an empty/stub solution (tests actually
   test something).
4. **Zero-shot floor check** (manual, documented): run Step-0 eval on Qwen;
   pass rate must be ≈0. If >2%, regenerate colliding names.
5. Name checks: no dictionary words, no collisions with identifiers in the
   Qwen tokenizer's top frequent merges (cheap heuristic: reject names that
   tokenize to a single Qwen token).

## 9. Determinism and versioning

- Single `--seed` (default 20260705); syllable inventory, grammar weights,
  and paraphrase templates versioned in the generator source.
- Generator writes `data/fictional/MANIFEST.json`: seed, generator version,
  per-file row counts and SHA-256; hash recorded in `docs/RESULTS.md` with
  every experiment result.
- Regeneration with the same seed must be byte-identical (no map iteration
  order, no unseeded RNG — use a single seeded PRNG stream).

## 10. Implementation

- Task: `cargo run --release -- --prepare-veclab --seed N --out data/fictional`,
  implemented in `src/tasks/veclab.rs`.
- Function semantics as a small AST enum; three renderers: Go impl, doc
  text, test vectors (evaluate the AST on generated inputs in Rust to get
  expected outputs — avoids needing Go at doc-gen time; `go test` then
  cross-validates the Go rendering against the same vectors).
- Requires `go` in PATH only for the verification step (§8), consistent with
  the existing Go-toolchain usage in the repo.
- After generation, rebuild encoder vocab including
  `veclab_encoder_mix.txt` so fictional identifiers get whole-ish subwords
  (check: each function name ≤ 4 encoder tokens).

## 11. Acceptance criteria

- [ ] `--prepare-veclab` produces all §4 artifacts + MANIFEST, deterministic
      under fixed seed (run twice, diff empty).
- [ ] `go test ./...` passes in `data/fictional/veclab/`.
- [ ] All §7 leak guards pass; `--print-split-stats` reports 0 held-out gold
      rows upstream.
- [ ] All generated training rows are unique and every masked encoder sample
      retains at least one visible token.
- [ ] 600 eval cases load; every `must_call` function exists; every harness
      fails on a stub and passes on gold.
- [ ] Documented Step-0 zero-shot result ≈0 in `docs/RESULTS.md`.
