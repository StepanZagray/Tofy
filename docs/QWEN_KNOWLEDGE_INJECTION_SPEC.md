# Spec: Knowledge Injection from World Model into Frozen Qwen3-1.7B-Base

Status: clean-slate architecture implemented on 2026-07-23; a new end-to-end
run is required. See
[MODEL_REWRITE_2026-07.md](MODEL_REWRITE_2026-07.md) for the module audit and
checkpoint incompatibilities. Emission is the frozen Qwen3 bridge only
(`src/tasks/bridge.rs`, `src/model/decoders/qwen3_bridge.rs`).

## 1. Hypothesis and claim

A world model trained on a knowledge corpus can inject that knowledge into a
frozen, pretrained decoder (Qwen3-1.7B-Base) through a latent conditioning
channel, replacing task-specific fine-tuning of the decoder.

The claim is proven if the conditioned decoder correctly uses knowledge that:

1. the decoder verifiably does not have (fictional, invented for the experiment),
2. is not present in the decoder's text prompt,
3. was never seen by the trainable conditioning pathway (adapter + cross-attn)
   during its training — only by the world model.

## 2. Historical motivation (pre–LeWorldModel rewrite)

Before the July 2026 rewrite, the in-repo world stack optimized for latent
predictability and agent routing rather than knowledge transmission. The
problems below motivated the current design; most are resolved by
`src/tasks/knowledge.rs` + `src/model/leworld.rs` + the Qwen bridge.

| Problem (then) | Where (then) | Status now |
|---|---|---|
| Latent prediction without knowledge retrieval | legacy multi-loss world stage | Resolved: LeWorldModel MSE + SIGReg only |
| Shallow compressor | `context_compressor.rs` single block | Partially open: depth still sweepable (`TOFY_CONTEXT_COMPRESSOR_DEPTH`) |
| Action CE / routing dominated selection | action classifier + focal loss | Resolved: no router in knowledge stage |
| Dynamics-oriented transition on static docs | old transition losses | Resolved: explicit `FetchDocs` AdaLN-Zero predictor |
| Adapter bypass at serve time | flatten-slots path | Resolved: bridge always goes through the adapter |
| Slot scale unconstrained for pretrained consumer | L2-normalized losses only | Resolved: compressor/adapter RMSNorm + LeWM projectors |
| Adapter output dim = `bridge_dim` (≠ Qwen 2048) | old adapter | Resolved: adapter `model_dim` = Qwen hidden |

## 3. The experiment

### 3.1 Synthetic knowledge corpus

Generate the fictional Go library package `veclab` (see
[VECLAB_DATA_SPEC.md](VECLAB_DATA_SPEC.md) for full row counts and leak guards):

- Exactly 200 exported functions with invented names, signatures, docs, and
  deterministic semantics (implementations exist, hidden from all models).
- Names decontaminated: Step-0 floor must measure ~0% suite pass.
- Generator artifacts under `data/fictional/`:
  - `veclab/` — hidden Go reference module (eval only)
  - `veclab_docs.txt` — documentation sections
  - `veclab_knowledge{,_train,_val}.txt` — query/`FetchDocs`/doc transitions
  - `veclab_tasks_{train,heldout}.txt` — instruction→code pairs
  - `veclab_encoder_mix.txt` — encoder continue-pretrain mix
  - `MANIFEST.json` — seed + per-file SHA-256
- Eval: `eval/veclab_eval.jsonl` (seen 1–100 / heldout 101–200)
- Pipeline-only: `veclab_bridge_transfer.txt` — world docs (fns 1–200) + code
  tasks (fns 1–80) for bridge grounding

The function split is the core control: the conditioning pathway trains code
on functions 1–100; success on 101–200 can only come through the world model.

### 3.2 Ladder (run in order; each failure is diagnosable)

| Step | Setup | Question answered |
|---|---|---|
| 0. Floor | Qwen zero-shot on eval | Is the knowledge really absent? (expect ~0%) |
| 1. Ceiling | Qwen with relevant docs pasted in prompt (RAG) | Does Qwen have the *capability* when given the knowledge as text? |
| 2. Latent channel | Docs go into world model **at inference**; Qwen prompt has task only | Can the latent channel transmit knowledge present in context? Metric: fraction of Step-1 ceiling recovered. |
| 3. Knowledge in weights | World model *trained* on docs; at inference nobody sees docs | Can the world model store and emit knowledge from weights? This is the fine-tune-replacement claim. |

Step 0 is a distinct manual experiment, not a training stage. The pipeline
never invokes it; use `scripts/run_veclab_decoder_floor.sh` only when the base
decoder, tokenizer, prompt contract, or evaluation suite has deliberately
changed.

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

### 4.1 Encoder (`src/model/encoders/`, `src/tasks/latent.rs`)

- Train a new checkpoint: the encoder schema and masked-prediction objective
  changed and old checkpoints are incompatible.
- Every attention block uses RoPE, pre-RMSNorm, and parameter-matched SwiGLU.
- Depth is exact and hierarchical. A seven-layer model is
  `3 local → 2 chunk/global → feedback → 2 local refinement`, rather than
  placing all token computation before one terminal global feedback.
- Chunk states use learned masked attentive pooling and are recomputed after
  token refinement. Partial chunks and padded tokens are explicitly masked.
- The encoder owns a prediction-space head. Token, chunk, and global masked
  predictions target full unmasked features with weights 70/20/10.
- Build default BPE inside lexical boundaries only. A boundaryless tokenizer
  can memorize an entire repetitive prompt template as one token, leaving no
  visible context after masking. Cache rows must contain at least two tokens,
  preparation must report min/mean/max token fertility, and a cache that has no
  usable rows is a hard error. Fictional identifiers may remain whole tokens;
  whitespace and punctuation boundaries must never be merged into them.

### 4.2 ContextCompressor (`src/model/context_compressor.rs`)

- **Depth**: a Perceiver/Q-Former latent resampler via
  `TOFY_CONTEXT_COMPRESSOR_DEPTH` (default 3). Every layer has source
  cross-attention, latent self-attention, and SwiGLU.
- **Output norm**: RMSNorm on output slots so the frozen Qwen consumer sees
  stable scale.
- **Slots**: profile `num_latent_tokens` (64 on `minimal`); sweep {32, 64, 128}
  in Step 2 for channel-capacity knee.
- Do not attach a reconstruction head. Exact downstream generation and
  task-to-document top-1 are probes only.

LeWorldModel post-normalization projectors live in `src/model/leworld.rs`
(not in the compressor); see §4.3.

### 4.3 LeWorldModel stage (`src/tasks/knowledge.rs`, `src/model/leworld.rs`)

`--train-world-knowledge` is based on
[LeWorldModel](https://arxiv.org/abs/2603.19312):

- **Projectors**: after compressor RMSNorm, slots pass through a two-layer MLP
  with BatchNorm (`encoder_projector`); an identical `predictor_projector`
  follows the predictor. SIGReg cannot control scale if the latent is pinned
  only by the compressor's final RMSNorm.
- **Objective**: raw next-embedding MSE plus `0.09 * SIGReg`; these are the only
  two optimized terms and the same sum selects checkpoints.
- **End-to-end**: gradients update the encoder, compressor, encoder projector,
  predictor, and predictor projector. There is no target stop-gradient, EMA,
  frozen pretrained representation, reconstruction CE, or InfoNCE loss.
- **SIGReg**: 1,024 freshly sampled Gaussian unit projections, 17 trapezoidal
  knots over `[0, 3]`, the Epps–Pulley statistic, and batch-size scaling
  following the official implementation. Every state and next-state slot is
  regularized. Position-chunked evaluation bounds peak memory while reusing
  the same sampled projections; a unit test checks equality with the full
  computation.
- **Action-conditioned dynamics**: knowledge rows explicitly label the
  `FetchDocs` intervention. A six-block predictor receives the discrete action
  through zero-initialized AdaLN modulation at every block and uses attention
  and feed-forward dropout while training.
- **Stability**: reject optimizer updates whose gradient norm exceeds both an
  EMA-relative threshold and an absolute floor; stop after 3,000 unimproved
  steps. The best checkpoint remains atomic and intact after a rejected spike.
- **Probe**: in-batch task-to-document top-1 is logged and constrains model
  selection, but does not contribute a gradient.

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
- Three-layer Perceiver/Q-Former resampling by default
  (`TOFY_ADAPTER_DEPTH=3`), including latent self-attention and SwiGLU.
- Subtract the adapter's all-zero-world baseline exactly. Learned queries may
  select world information but cannot become a task-independent soft prompt.

### 4.5 Qwen bridge (`src/model/decoders/qwen3_bridge.rs`)

Sole decoder path. Consumes `DecoderConditioningAdapter` slots.

- Vendor `candle-transformers` `qwen3.rs` (v0.10.2 to match Cargo pins).
- Insert gated cross-attention after self-attention in every 4th layer
  (7 sites in 28 layers; sweepable `TOFY_QWEN_CROSS_EVERY`, default 4):
  `x = x + sigmoid(gate(x_norm)) * cross_attn(x_norm, cond_slots)`,
  gate bias initialized to −4 (near-zero gate at start; Flamingo-style).
- Each site's internal attention width defaults to `min(512, hidden_size)` and
  is sweepable with `TOFY_QWEN_CROSS_DIM`. Q/K/V/O therefore cost about 4.2M
  parameters per 2048-wide site instead of 16.8M.
- Base weights: bf16, loaded as plain tensors via
  `VarBuilder::from_mmaped_safetensors`, never in the VarMap → frozen, no
  optimizer state. The exact trainable count is emitted from the VarMap.
- Tokenizer: `tokenizers` crate, Qwen `tokenizer.json`. New deps:
  `candle-transformers`, `tokenizers`.
- Generation contract: provide a normal Go task comment and the exact function
  scaffold, supervise only its body and closing brace, and stop with a
  string/comment-aware outer-brace scanner. The complete target plus EOS is
  reserved before prompt truncation.
- Loss: body-completion token CE for syntax plus conditioning margin and
  wrong-state unlikelihood restricted to the fictional API identifiers in the
  task-defining `return` expression (`TOFY_DECODER_CONDITIONING_NEGATIVES=hard`).
  The full expression is the fallback for non-VecLab targets. Shared wrapper
  boilerplate cannot satisfy the semantic objective.
- Serve/eval path: always through the adapter (fixes the flatten-slots bypass
  for this pipeline).

#### World-to-decoder grounding curriculum

The bridge must learn the coordinate transform from LeWorldModel states into
Qwen's token space before task-only tuning can test transfer. The first bridge
phase therefore aligns adapter slots to frozen Qwen documentation-token
embeddings with token-to-best-slot cosine and symmetric pooled InfoNCE. The
world is detached throughout this BLIP-2-style latent-language alignment phase.
Bridge input is
therefore the union of:

- every action-labelled world query/documentation pair (functions 1--200), and
- code task/completion pairs for functions 1--80 only.

Documentation rows train the same adapter, gated cross-attention sites, hard
negative margin, and unlikelihood objective as code rows. Their visible Qwen
prompt is deliberately generic; the matching world state must supply the
reference content. Code validation remains function-disjoint on 81--100, and
held-out code for 101--200 is never used. This separates decoder grounding
from code-solution leakage while ensuring that unseen fictional identifiers
are not out-of-support at the world/Qwen interface.

Autoregressive validation runs during bridge training over the real Go build
and hidden harness tasks for the complete function-disjoint validation set. A
checkpoint is ineligible unless matched conditioning reaches the required
suite-pass rate, beats wrong conditioning, and passes the teacher-forced
semantic gap. This prevents a low-CE, malformed, comment-only, or
non-generating checkpoint from being selected. Required fictional API use is
verified from the Go syntax tree as an exact `veclab.<required>(...)` selector
call, rather than by substring matching.

### 4.6 Step-3 variant: end-to-end unfreeze

The attribution-preserving default keeps the world frozen. As a separate
practical ablation, add `TOFY_KNOWLEDGE_UNFREEZE_WORLD=true`:
compressor (+ optionally encoder at 0.1× LR) receives gradients from the Qwen
CE loss during conditioning training. Rationale: knowledge stored under
the predictive world objective may not be organized for *transmission*; the decoder
loss is the only signal that shapes it for the consumer. Run both:

- frozen world (pure claim, cleaner attribution),
- unfrozen world (practical system; empirical result required).

Held-out function split applies to both — unfreezing uses only functions 1–100
tasks, so held-out success still requires knowledge stored during world
training.

## 5. Training plan on 1×A100 80GB

| Phase | Trains | Data | Est. |
|---|---|---|---|
| 0. Corpus gen | — | generator + `go` toolchain | hours |
| 1. Encoder continue-pretrain | encoder | docs + tasks | <1 day |
| 2. LeWorldModel stage | encoder + compressor + projectors + predictor | query/`FetchDocs`/document transitions | <1 day |
| 3. Bridge training | adapter + cross-attn (+world if unfrozen) | `veclab_bridge_transfer.txt` (docs 1–200 + code 1–80) | <1 day |
| 4. Baselines | soft prefix; LoRA | same | <1 day |
| 5. Eval + ablations | — | eval suites | hours |

Qwen 1.7B remains frozen in bf16. The rewritten bridge uses seven default
512-wide cross-attention bottlenecks (about 29.4M projection parameters total)
plus gates and the three-layer adapter; the runtime emits the exact count.
Physical batch and accumulation must be re-qualified because the checkpoint
schema and activation graph changed.

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

## 7. Cleanup (completed July 2026)

Non-experiment surface area was removed so only the knowledge and bridge stages
remain. Section 7.1 is the live module map; 7.2 records what was deleted.

### 7.1 Keep (experiment-critical)

| Component | Location | Why |
|---|---|---|
| Encoder + LeJEPA pretraining | `src/model/encoders/`, `src/model/lejepa.rs`, `src/tasks/latent.rs` | Produces representations the compressor reads |
| ContextCompressor (incl. hybrid memory) | `src/model/context_compressor.rs` | The knowledge bottleneck under test |
| DecoderConditioningAdapter | `src/model/decoders/decoder_conditioning_adapter.rs` | The bridge into Qwen |
| LeWorldModel + ActionStateTransition | `src/model/leworld.rs`, `src/model/action_state_transition.rs` | End-to-end projected latent and AdaLN-Zero `FetchDocs` dynamics |
| World context assembly | `src/tasks/world_context.rs`, `src/tasks/world_support.rs` | Batching and conditioning helpers |
| Knowledge / bridge / eval stages | `src/tasks/knowledge.rs`, `bridge.rs`, `eval.rs` | Train + ladder |
| VecLab corpus generator | `src/tasks/prepare_veclab.rs`, `src/tasks/veclab.rs` | Deterministic data + split checks |
| Go eval harness | `src/tasks/eval.rs`, `eval/` | Verifiable metric |
| Data / vocab / token cache infra | `src/tasks/cache.rs`, `src/data/` | Feeds encoder + world stages |
| Optimizer, checkpointing, TensorBoard, profiles | `src/util.rs`, `config/model_profiles.json` | Training infra |

### 7.2 Deleted (complete)

Former in-repo autoregressive emitter paths, dual-token caches, high-world
planner, `--serve` stack, Go-feedback mining, and the legacy `world.rs`
monolith are gone. Emission is only the Qwen bridge. Pipeline stages:
corpus gen → encoder → world-knowledge → bridge → eval.

## 8. Success criteria

- Step 2: conditioned pass rate on held-out functions recovers ≥50% of the
  RAG ceiling and beats the static prefix by a clear margin.
- Step 3: beats the closed zero-shot floor decisively on held-out functions
  and reaches the rank-16 LoRA comparator; parity or better demonstrates a
  fine-tune replacement. Rank 512 is a diagnostic capacity control, not a
  mandatory success gate.
- Ablations: zeroed/shuffled/swapped latents collapse performance toward the
  floor (proves causal use of the channel).
