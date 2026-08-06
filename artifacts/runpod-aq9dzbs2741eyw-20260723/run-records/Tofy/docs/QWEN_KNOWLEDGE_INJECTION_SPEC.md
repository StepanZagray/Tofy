# Spec: Knowledge Injection from World Model into Frozen Qwen3-1.7B-Base

Status: implemented experiment design; the July 2026 causal-control repair is
described below.
Scope: world-model side + conditioning bridge. Nothing in the current model is
considered load-bearing: any code that does not serve the experiment may be
changed or deleted (see section 7). The custom `CodeDecoder` is replaced by
the Qwen bridge and scheduled for deletion, not preservation.

## 1. Hypothesis and claim

A world model trained on a knowledge corpus can inject that knowledge into a
frozen, pretrained decoder (Qwen3-1.7B-Base) through a latent conditioning
channel, replacing task-specific fine-tuning of the decoder.

The claim is proven if the conditioned decoder correctly uses knowledge that:

1. the decoder verifiably does not have (fictional, invented for the experiment),
2. is not present in the decoder's text prompt,
3. was never seen by the trainable conditioning pathway (adapter + cross-attn)
   during its training — only by the world model.

## 2. Why the current world model is not sufficient

| Problem | Where | Consequence |
|---|---|---|
| All objectives are latent prediction (normalized MSE on L2-normalized slots) | `src/tasks/world.rs` transition/rollout losses, `src/model/lejepa.rs::prediction_loss` | Slots optimize for *predictability*, not *information content*. Knowledge retrievability is never trained. |
| Compressor is a single cross-attn block | `src/model/context_compressor.rs` | Too shallow to extract structured API knowledge (signatures, types, semantics) from encoder states. |
| Action machinery dominates the stage | action CE (weight 1.0), focal loss, routing metrics in selection score | Optimizes for the agent loop, irrelevant to knowledge transmission; pollutes checkpoint selection. |
| Transition model is action-dynamics oriented | `src/model/action_state_transition.rs` | The "world state" for this experiment is static documentation. Rolling a dynamics model adds noise to the channel. |
| Adapter bypassed at serve time | `src/tasks/world.rs::get_decoder_and_cond_from_context_compressor` flattens raw slots | Train/serve mismatch; must be unified. |
| Slot scale unconstrained | L2-normalized losses everywhere | Feeding un-normalized slots into a pretrained model's cross-attn is unstable. |
| Adapter output dim = bridge_dim (640) | `decoder_conditioning_adapter.rs` | Qwen3-1.7B hidden is 2048. |

## 3. The experiment

### 3.1 Synthetic knowledge corpus (new)

Generate a fictional Go library, e.g. package `veclab`:

- ~200 exported functions with invented names, signatures, doc comments, and
  deterministic semantics (implementations exist, are hidden from all models).
- Names decontaminated: verify zero-shot Qwen3-1.7B-Base cannot call any of
  them correctly (Step 0 must measure ~0%).
- Artifacts:
  - `data/fictional/veclab_docs.txt` — documentation corpus (world-model food).
  - `data/fictional/veclab_tasks_train.txt` — instruction→code pairs, functions 1–100 only.
  - `data/fictional/veclab_tasks_heldout.txt` — pairs requiring functions 101–200.
  - `eval/veclab_eval.jsonl` — compile+test tasks (reuse the Go eval harness),
    split into `seen` (1–100) and `heldout` (101–200) subsets.

The function split is the core control: the conditioning pathway trains only on
functions 1–100; success on 101–200 can only come through the world model.

### 3.2 Ladder (run in order; each failure is diagnosable)

| Step | Setup | Question answered |
|---|---|---|
| 0. Floor | Qwen zero-shot on eval | Is the knowledge really absent? (expect ~0%) |
| 1. Ceiling | Qwen with relevant docs pasted in prompt (RAG) | Does Qwen have the *capability* when given the knowledge as text? |
| 2. Latent channel | Docs go into world model **at inference**; Qwen prompt has task only | Can the latent channel transmit knowledge present in context? Metric: fraction of Step-1 ceiling recovered. |
| 3. Knowledge in weights | World model *trained* on docs; at inference nobody sees docs | Can the world model store and emit knowledge from weights? This is the fine-tune-replacement claim. |

### 3.3 Baselines and controls (all mandatory)

- **Static soft prefix**: learned constant conditioning slots `[A, 2048]`, same
  slot count and same cross-attn interface, trained on the same task data.
  If the world model doesn't beat this on the held-out split, it transmits
  nothing beyond task identity.
- **LoRA**: rank-16 LoRA on Qwen trained on the same corpus+tasks — the
  fine-tune being "replaced". Step 3 is compared against this.
- **Ablations at eval**: zeroed latent, shuffled-in-batch latent, cross-instance
  swap (latent from task A, prompt from task B — accuracy must *drop*).
- **Channel probe**: small attention probe trained from adapter output slots to
  the set of API symbols required by the task (multi-label). Measures whether
  knowledge is *present* in the channel independently of whether Qwen decodes
  it. Separates "slots don't contain it" from "Qwen can't read it".

### 3.4 Metrics

- `suite_pass_rate` and `compile_rate` per ladder step, per split (seen/heldout).
- Fraction of RAG ceiling: `pass(step2 or 3) / pass(step1)`.
- API-symbol accuracy: token accuracy restricted to fictional identifier
  positions (moves earlier than pass rates; cheap signal).
- Probe accuracy (channel content).
- Track all of it in `docs/RESULTS.md`.

## 4. Module changes

### 4.1 Encoder (`src/model/encoders/`, `src/tasks/latent.rs`) — minor

- Continue-pretrain the existing encoder checkpoint on `veclab_docs.txt` +
  task pairs (LeJEPA objective unchanged) so fictional identifiers have
  non-degenerate representations. A few thousand steps suffice.
- Build default BPE inside lexical boundaries only. A boundaryless tokenizer
  can memorize an entire repetitive prompt template as one token, leaving no
  visible context after masking. Cache rows must contain at least two tokens,
  preparation must report min/mean/max token fertility, and a cache that has no
  usable rows is a hard error. Fictional identifiers may remain whole tokens;
  whitespace and punctuation boundaries must never be merged into them.

### 4.2 ContextCompressor (`src/model/context_compressor.rs`) — moderate

- **Depth**: 1 → 2 cross-attn blocks (stacked, pre-norm, same head count).
  Env: `TOFY_CONTEXT_COMPRESSOR_DEPTH` (default 1 to preserve old behavior).
- **Output norm**: RMSNorm on output slots. The conditioning interface must
  have stable scale for the frozen consumer.
- **LeWM projector**: the normalized slots pass through a two-layer MLP with
  BatchNorm before they enter the world latent. A second identical projector
  follows the predictor. This is the direct adaptation of LeWorldModel's
  post-final-normalization projectors; SIGReg cannot control the scale of a
  latent constrained by the compressor's final RMSNorm.
- **Slots**: keep 64 initially; make sweepable (`num_latent_tokens` already in
  profiles). Sweep {32, 64, 128} in Step 2 to find channel-capacity knee.
- Do not attach a reconstruction head. Exact downstream generation and
  task-to-document top-1 are probes; they must not introduce competing world
  losses.

### 4.3 LeWorldModel stage (`src/tasks/knowledge.rs`)

`--train-world-knowledge` is based on
[LeWorldModel](https://arxiv.org/abs/2603.19312), not the older multi-loss
knowledge autoencoder:

- **Objective**: raw next-embedding MSE plus `0.09 * SIGReg`; these are the only
  two optimized terms and the same sum selects checkpoints.
- **End-to-end**: gradients update the encoder, compressor, encoder projector,
  predictor, and predictor projector. There is no target stop-gradient, EMA,
  frozen pretrained representation, reconstruction CE, or InfoNCE loss.
- **SIGReg**: 1,024 freshly sampled Gaussian unit projections, 17 trapezoidal
  knots over `[0, 3]`, the Epps–Pulley statistic, and batch-size scaling exactly
  following the official implementation. It is evaluated on compact pooled
  state and next-state embeddings rather than treating all 64 slots as
  independent observations.
- **Action-conditioned dynamics**: knowledge rows explicitly label the
  `FetchDocs` intervention. A six-block predictor receives the discrete action
  through zero-initialized AdaLN modulation at every block.
- **Probe**: in-batch task-to-document top-1 is logged but does not contribute
  a gradient.

The VecLab corpus currently contains one-step retrieval transitions, so the
action-prefix parallel rollout proposed by
[Fast LeWorldModel](https://arxiv.org/abs/2606.26217) is not fabricated from
nonexistent trajectories. Add it only with real multi-horizon state/action
sequences and dense prefix targets. The dimension warning in
[When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379) also
means slot count and latent width remain empirical sweeps: Gaussianity alone
does not identify the right representation when model and intrinsic dimensions
do not match. All current rows use `FetchDocs`, so the experiment trains the
retrieval intervention but cannot identify contrasts among action embeddings;
multi-action claims require an excited dataset containing each action.

### 4.4 DecoderConditioningAdapter (`src/model/decoders/decoder_conditioning_adapter.rs`) — moderate

- `model_dim` = 2048 (Qwen hidden), set from bridge config, not profile decoder_dim.
- Output slots: fixed `TOFY_ADAPTER_OUTPUT_SLOTS` (default 64), remove the
  kind-based clamp for this path.
- Decoder-side action embedding: deleted. The world predictor still receives
  the explicit `FetchDocs` action; the decoder consumes its resulting state.
- Final RMSNorm on output.
- Optional second cross-attn block (`TOFY_ADAPTER_DEPTH`, default 1).

### 4.5 New: Qwen bridge (`src/model/decoders/qwen3_bridge.rs`) — new code

Not a change to the custom decoder; a parallel consumer of the same
conditioning interface.

- Vendor `candle-transformers` `qwen3.rs` (v0.10.2 to match Cargo pins).
- Insert gated cross-attention after self-attention in every 4th layer
  (7 sites in 28 layers; sweepable `TOFY_QWEN_CROSS_EVERY`, default 4):
  `x = x + sigmoid(gate(x_norm)) * cross_attn(x_norm, cond_slots)`,
  gate bias initialized to −4 (near-zero gate at start; Flamingo-style).
- Base weights: bf16, loaded as plain tensors via
  `VarBuilder::from_mmaped_safetensors`, never in the VarMap → frozen, no
  optimizer state. Trainable: cross-attn + gates + adapter (~15–25M params).
- Tokenizer: `tokenizers` crate, Qwen `tokenizer.json`. New deps:
  `candle-transformers`, `tokenizers`.
- Loss: full-completion token CE for syntax plus conditioning margin and
  wrong-state unlikelihood restricted to the fictional API identifiers in the
  task-defining `return` expression (`TOFY_DECODER_CONDITIONING_NEGATIVES=hard`).
  The full expression is the fallback for non-VecLab targets. Shared wrapper
  boilerplate cannot satisfy the semantic objective.
- Serve/eval path: always through the adapter (fixes the flatten-slots bypass
  for this pipeline).

### 4.6 Step-3 variant: end-to-end unfreeze — small but critical

For the knowledge-in-weights regime, add `TOFY_KNOWLEDGE_UNFREEZE_WORLD=true`:
compressor (+ optionally encoder at 0.1× LR) receives gradients from the Qwen
CE loss during conditioning training. Rationale: knowledge stored under
the predictive world objective may not be organized for *transmission*; the decoder
loss is the only signal that shapes it for the consumer. Run both:

- frozen world (pure claim, cleaner attribution),
- unfrozen world (practical system, expected stronger).

Held-out function split applies to both — unfreezing uses only functions 1–100
tasks, so held-out success still requires knowledge stored during world
training.

## 5. Training plan on 1×A100 80GB

| Phase | Trains | Data | Est. |
|---|---|---|---|
| 0. Corpus gen | — | generator + `go` toolchain | hours |
| 1. Encoder continue-pretrain | encoder | docs + tasks | <1 day |
| 2. LeWorldModel stage | encoder + compressor + projectors + predictor | query/`FetchDocs`/document transitions | <1 day |
| 3. Bridge training | adapter + cross-attn (+world if unfrozen) | tasks (fns 1–100) | <1 day |
| 4. Baselines | soft prefix; LoRA | same | <1 day |
| 5. Eval + ablations | — | eval suites | hours |

Everything fits on the target accelerators: Qwen 1.7B bf16 ≈ 3.4 GB and
the measured bridge has 124.2M trainable parameters. Optimizer and activation
memory, rather than frozen Qwen weights, sets the L40S microbatch limit.

## 6. Additional improvements (ordered by expected impact)

1. Exact LeWorldModel MSE + SIGReg objective and end-to-end encoder training.
2. Held-out function split — without it results are unattributable.
3. Static soft-prefix baseline — cheapest way to falsify "world model adds
   nothing".
4. RMSNorm at both interface ends (compressor out, adapter out).
5. Channel probe diagnostics.
6. Compressor depth 2.
7. Docs-side sequence budget: raise `world_max_seq` for doc encoding (docs are
   longer than chat turns); hybrid memory already helps, keep
   `TOFY_CONTEXT_HYBRID_MEMORY=true`.
8. Sweep slot count and cross-attn site density once the pipeline runs.
9. Curriculum for Step 3: first train with docs in context (Step-2 regime),
   then progressively drop docs from context while keeping the task loss —
   distills context knowledge into world weights.
10. Keep conditioning dropout at zero. Exact-zero conditioning is an identity
    path through frozen Qwen and supplies no bridge gradient; explicit hard and
    zero-condition controls provide the useful causal comparison.

## 6.1 July 2026 causal-control repair

Run `code_poc_1783547471` proved that a nonzero learned intervention can turn
on code-following behavior in frozen Qwen, but matched conditioning did not beat
shuffled conditioning. The repaired implementation makes that shortcut
ineligible for checkpoint selection:

- `[fn:NNN]` remains sampling metadata but is stripped before encoder/Qwen input.
- `minimal` uses a real wrong-function negative. Batch-one negatives are fetched
  from another function instead of degenerating to zeros.
- Positive and wrong-condition gradients are accumulated sequentially, keeping
  only one Qwen activation graph resident on the L40S.
- Validation reports matched, zeroed, and two wrong-function full and semantic
  CEs. A world bridge is checkpoint-eligible only when API-identifier
  `wrong_ce - matched_ce >=
  TOFY_BRIDGE_MIN_SEMANTIC_GAP` (default `0.02`).
- Eval reports paired matched-only/control-only counts and matched advantage.
- World training now uses only next-embedding MSE and exact SIGReg (`λ=0.09`);
  retrieval top-1 is diagnostic. The L40S-safe schedule remains 20k steps.
- The context bridge keeps the world frozen. The practical weights bridge trains
  world+bridge jointly at `1e-4` and saves an atomic matching world sidecar.

Old world checkpoints trained with the reconstruction/InfoNCE objective,
model-visible function tags, or an unconditioned transition, and old bridge
checkpoints trained with `TOFY_DECODER_CONDITIONING_NEGATIVES=none` must not be
resumed into this objective; start a new run root.

### 6.1.1 Causal-channel hardening

The first repaired context run still reached almost identical matched and wrong
validation CE: zero conditioning changed generic code-following behavior, but
a wrong nonzero state did not change the answer. The bridge now applies the
following stricter intervention, which is required before interpreting a pass
rate as knowledge injection:

- The adapter is centred at its all-zero world state and does not emit its
  learned query bank as a prefix. Consequently `adapter(0) = 0`; queries can
  select state information but cannot turn on a task-independent coding mode.
- Training, validation, and bridge-mode evaluation expose only the
  `func Solve(...)` signature by default
  (`TOFY_BRIDGE_COUNTERFACTUAL_PROMPTS=true`). Matched and wrong states with
  the same signature therefore receive precisely the same visible request.
- Hard negatives preferentially use a different function with that same public
  signature. Full CE preserves syntax, while semantic CE margin and target-token
  unlikelihood operate on the fictional API identifiers under a wrong state; a
  minimum adapter-output separation is retained. This focuses gradients on API
  selection and composition after generic wrapper CE has saturated.
- Bridge selection is now function-disjoint: functions 1--80 train and 81--100
  validate (`TOFY_BRIDGE_TRAIN_FUNCTION_MAX=80`,
  `TOFY_BRIDGE_VALIDATION_FUNCTION_MAX=100`). This replaces the old row-wise
  split, which allowed paraphrases of each function in both partitions.
- If no semantic-gap progress is observed after the warm-up, training stops
  after `TOFY_BRIDGE_SEMANTIC_PATIENCE` (default 1200 steps). A qualified
  checkpoint makes this successful early stopping so the evaluation ladder can
  run; an unqualified plateau fails the stage (the context pipeline then moves
  to the separately evaluated weights regime). Progress means a new qualified
  historical best, not a later noisy value merely above the threshold.

These changes test the actual claim--the state, rather than a generic nonzero
prefix or visible task text, controls Qwen's output. They intentionally make
the context bridge harder to optimize; a failed causal gate is an experiment
result, not a reason to report its natural-prompt compile score.

## 6.2 2026 research disposition

- [ByteFlow](https://arxiv.org/abs/2603.03583) and
  [Equity with Efficiency](https://arxiv.org/abs/2606.15044) provide current
  evidence that fixed coarse BPE granularity is brittle and that byte-level or
  morphology-aware boundaries improve semantic learning. Tofy retains its
  compact byte-fallback BPE but constrains merges to lexical units, which fixes
  the measured template-memorization failure without replacing the experiment
  with a different architecture.
- [Tokenizer Fertility in the Legal Domain](https://arxiv.org/abs/2605.14890)
  supports auditing tokenizer fertility before model selection. Here the audit
  is a correctness gate because a one-token sample makes masked prediction
  mathematically degenerate.
- [LeWorldModel](https://arxiv.org/abs/2603.19312) supports end-to-end encoder /
  predictor alignment with prediction plus Gaussian regularization. This motivates
  joint weights-mode alignment and preserving LeJEPA prediction + SIGReg.
- [When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379)
  motivates Gaussian latents and explicit identifiability checks; its guarantees
  do not replace task-specific causal controls for text knowledge.
- [UR-JEPA](https://arxiv.org/abs/2606.01443) reports lower seed variance from
  an alternative regularizer in vision. It is not adopted in this run: no
  representation regularizer can recover information from identical all-mask
  views, so tokenizer validity and held-out selection are the causal fixes to
  test first.
- [Fast LeWorldModel](https://arxiv.org/abs/2606.26217) uses action-prefix
  prediction to reduce rollout error. It is relevant when this experiment gains
  multi-step latent rollouts, not to the present one-step knowledge lookup.
- [Qwen3.6](https://github.com/QwenLM/Qwen3.6) emphasizes hybrid gated-delta/MoE
  efficiency, scaled agentic RL, and thinking-context preservation. Its public
  repository still lists the detailed user guide/paper as forthcoming; these
  large-model mechanisms are not transplanted into the Qwen3-1.7B control.
- [DeepSeek-V4](https://www.deepseek.com/en/transparency/) and
  [GLM-5.2](https://z.ai/blog/glm-5.2) target trillion-scale MoE, sparse million-
  token attention, MTP/speculative decoding, and long-horizon agentic RL. Those
  are serving/post-training scale techniques, not remedies for the measured
  wrong-latent invariance. The applicable shared lesson is to optimize and score
  the actual end behavior with hard controls rather than proxy loss alone.

## 7. Cleanup: delete everything that does not serve the experiment

The current model produces no useful output, so no existing code has intrinsic
value. Anything not listed under "keep" below can be deleted without ceremony.
This shrinks the surface area, removes noise from training/selection, and makes
`src/tasks/world.rs` (8k+ lines) maintainable.

### 7.1 Keep (experiment-critical)

| Component | Location | Why |
|---|---|---|
| Encoder + LeJEPA pretraining | `src/model/encoders/`, `src/model/lejepa.rs`, `src/tasks/latent.rs` | Produces representations the compressor reads |
| ContextCompressor (incl. hybrid memory) | `src/model/context_compressor.rs` | The knowledge bottleneck under test |
| DecoderConditioningAdapter (modified per 4.4) | `src/model/decoders/decoder_conditioning_adapter.rs` | The bridge into Qwen |
| Conditioning-margin loss machinery | `src/tasks/world.rs` (extract) | Reused in bridge training |
| Go eval harness (compile + test) | `src/tasks/` eval paths, `eval/` | Verifiable metric |
| Data prep / vocab / token cache infra | `src/tasks/prepare.rs`, `src/data/` (trimmed) | Feeds encoder + world stages |
| Optimizer (Muon/AdamW), checkpointing, TensorBoard, profiles | `src/` misc, `config/model_profiles.json` | Training infra |
| LeWorldModel + ActionStateTransition | `src/model/leworld.rs`, `src/model/action_state_transition.rs` | End-to-end projected latent and AdaLN-Zero `FetchDocs` dynamics |

### 7.2 Delete

| Component | Location | Reason |
|---|---|---|
| Custom `CodeDecoder` + text decoder + its training/eval paths | `src/model/decoders/decoder_cross.rs`, decoder stages in `src/tasks/world.rs`, `src/tasks/pipeline.rs` stages 5/5b | Replaced by the Qwen bridge |
| Custom code-aware tokenizer for the decoder side | `src/data/` code-aware paths, `code_decoder_max_vocab` plumbing | Qwen tokenizer replaces it |
| Routing system: `NextActionClassifier`, inverse classifier, focal/balanced action CE | `src/model/action_classifier_head.rs`, legacy `src/tasks/world.rs`, `src/tasks/world_support.rs` | Classification is not latent dynamics; retain only explicit transition actions in world rows |
| High world: `MacroActionStateTransition`, `ActionSequenceEncoder`, macro chains, HWM planning | `src/model/macro_action_state_transition.rs`, high-world stage in `src/tasks/world.rs` | Multi-step action planning is out of scope |
| LeWM planning + latent test-time reasoning | `lewm_*`, `refine_latent_for_decoder`, `hwm_plan_from_state` in `src/tasks/world.rs` | Serve-time agent machinery |
| `world_selection_score` and routing metrics (code_rate, macro_f1) | `src/tasks/world_support.rs` | Replaced by next-embedding MSE + SIGReg (4.3) |
| Rollout loss, continuation-edge mining, inverse loss | `src/tasks/world.rs` | Dynamics supervision, not knowledge |
| OpenAI-compat HTTP server + GGUF/llama.cpp fallback | `--serve` paths, axum wiring | Not needed to run the ladder; re-add later for demos |
| Go feedback mining, repair-task generation, model-failure curriculum | `src/tasks/` stage 4, repair prep | Replaced by the fictional corpus generator |
| Old data mixes: `world_mix_pairs.txt`, `code_poc_mix.txt`, `code_poc_go_mix.txt`, multilang pairs (and their prep code) | `data/`, `src/tasks/prepare.rs` | Experiment uses the fictional corpus; keep `encoder_mix` prep only if the encoder is retrained from scratch |
| Recursive context compression mode (`fold_slots`, retain schedule) | `src/model/context_compressor.rs`, `TOFY_RECURSIVE_CONTEXT_COMPRESSION` | Hybrid memory is the kept path; one code path only |
| Dead env-var plumbing for all of the above | throughout | Every deleted feature takes its flags with it |

### 7.3 Restructure

- Break `src/tasks/world.rs` apart: extract the kept pieces into
  `src/tasks/knowledge.rs` (world knowledge stage) and
  `src/tasks/bridge.rs` (Qwen bridge training + eval); delete the remainder.
- Pipeline becomes 5 stages: corpus gen → encoder → world-knowledge →
  bridge → eval. Delete stages that no longer exist rather than gating them.
- Do this cleanup **first**, before implementing sections 4.x: every change
  lands in a codebase half the size, and nothing deleted needs migration.

## 8. Success criteria

- Step 2: conditioned pass rate on held-out functions recovers ≥50% of the
  RAG ceiling and beats the static prefix by a clear margin.
- Step 3: beats zero-shot floor decisively on held-out functions; the gap to
  LoRA is the headline number (parity or better = fine-tune replacement
  demonstrated).
- Ablations: zeroed/shuffled/swapped latents collapse performance toward the
  floor (proves causal use of the channel).
