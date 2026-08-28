# ADR 0003: World-core v5 / foundation-v2 training contract

- Status: accepted for implementation (2026-08-25)
- Supersedes: ADR 0001 full-v4 for new training (retained for provenance)
- Basis: `docs/research/2026-08-24-foundation-improvements.md` (diagnosis +
  cross-model plan; theorems in `formal/`). Every deviation from full-v4 below
  cites the root cause it fixes (RC numbers from
  `raw-2026-08-24/diag-final-ranked.md`).

## Non-negotiable invariants (unchanged)

- Synthetic-curriculum training only; public ARC-AGI-3 levels are held-out
  eval, enforced by the arc3_live source-inclusion test.
- Live policy stays a searchless forward pass.
- Deterministic seeds, persisted config, checkpoint bundles with manifests.

## Implementation corrections (2026-08-27, before another run)

- The historical changed/unchanged CE remains the default exactly. Alternative
  weighting constructions redistribute the same per-batch coefficient mass as
  that default. This preserves nominal coefficient mass, not gradient norm,
  direction, or clipping pressure; those remain measured experiment outputs.
  A configured changed budget is a loss-coefficient share, not a claimed
  gradient share. The unrelated current-frame foreground CE is fixed.
- Objective and promotion fields are part of the resume contract. Matched-arm
  launch orchestration remains external/deferred because a bare JSON loader
  cannot establish reviewed-source, binary, smoke-test, and population
  provenance. The selected best checkpoint now exports its EMA weights;
  latest-final EMA is only the fallback when no best exists.
- The v5 event census persists labeled/positive counts for all four slots, but
  `exhausted` is deliberately unlabelled: its required action-budget premise is
  absent from the event head input. Goal-dependent slots are also masked on
  goal-dropout rows. A completeness marker prevents a pre-census checkpoint
  resume from presenting partial post-resume counts as the full population.
  The gate seed is reserved from training, and meta-episode level IDs occupy a
  checked namespace.
- Spatial latent Huber, its canonical companion, rollout Huber, EP inputs, and
  displacement-based branch/inverse-action objectives use the exact content
  mask. The transition/observer canonical path remains unmasked because live
  frames do not expose a content rectangle.
- Exact decoders, copy gates, and action decoders stay on AdamW with the other
  output heads. The WSD schedule reaches its final learning rate even in short
  smoke schedules.
- No model-quality claim follows from these corrections. They have compile,
  deterministic-generator, and objective tests only until a fresh registered
  multi-seed experiment is run.
- The evidence order is the proved local library first (`CrossEntropy`,
  `Identifiability`, `MarginalBlindness`, `Separation`, `Symmetrization`), then
  current primary literature. Those proofs establish only their local stated
  properties. Current 2026 Muon convergence counterexamples make optimizer
  routing conservative; large-model Muon usage is not transferred as a Tofy
  guarantee.

## Implementation corrections (2026-08-27, second set — audit wave)

- **§5 gate 1 demotion recorded.** The latent-MSE `improvement_fraction` gate
  is diagnostic-only: measured and logged, never enforced, never aborting
  (commits 9b9baca5/ffb494ee). It measures proximity in a latent space the v5
  objective does not optimize; pixel-space gates supersede it. §5.1's
  "enforced after step 4096" is void.
- **§3.5 separation distance corrected.** The implementation measured an RMS
  distance over 128 dims of L2-normalized displacements, which caps at
  `2/sqrt(128) ≈ 0.177` and can never satisfy the `m=0.3` hinge; equality
  also produced a non-finite `sqrt(0)` gradient exactly at collapse. The
  distance is now the epsilon-stabilized L2 distance in the normalized
  vectors' own units; the margin is reachable and gradients stay finite.
- **§3.6 EP floor removed.** When the `<= 0.3x` budget requires a weight
  below the old `1e-4` floor (including a zero prediction gradient), the
  controller now disables EP with weight zero instead of violating the bound.
  Each 128-step sample records the achieved weighted ratio, budget
  compliance, and rail state.
- **§1.1 endpoint schedule normalization made explicit.** The written end
  proportions (25/30/20/15/5) sum to 0.95 and are normalized exactly at
  allocation; realized per-stream row counts are a reported output, and the
  goal-dropout intervention reports total/eligible/changed counts.
- **§4/§6 selection integrity.** Trainer state is schema v10 with a
  non-defaulted objective-implementation revision: a checkpoint trained under
  an older objective cannot silently resume under a newer one. The fixed gate
  population and gate policy carry a frozen identity; a resume that
  regenerates a different population fails closed before any update. The
  exported best EMA is verified against the run's own gate-history selection
  and fails closed on mismatch. A `composed_exact_guarded` checkpoint-election
  rule (composed all-row exactness with a false-edit non-regression guard) is
  available and preregistered for the next evidence run; `changed_exact`
  remains the historical default for replay. `PromotionMetric` names in-run
  best-checkpoint election — a `selection_only` mechanism under §6, never the
  promotion-evidence class.
- **Meta-episodes ship generator-first.** The cross-level meta-episode
  subsystem (generators, censuses, reserved ID namespace) has no trainer or
  evaluator consumer yet by design; wiring it into a training mixture is a
  separate preregistered change.
- No model-quality claim follows from these corrections either; they carry
  compile, deterministic-generator, and objective tests only.

## Implementation amendment (2026-08-28, observer target split)

- The Q head retains the graded composed-pixel-accuracy target. The reliability
  head is retargeted to latent self-confidence: whether full-spatial predicted
  vs factual encoded-next latent MSE is at or below `q_mse_threshold`. This is
  the same seam and boundary used by evaluation calibration, and the binary
  target is detached from the world model.
- Previously both detached canonical heads received the same graded pixel
  target. Their 0.25 Q and 0.30 reliability live-policy terms therefore counted
  one measurement twice. The unchanged score weights now combine distinct
  pixel-accuracy and latent-confidence signals; no reward/value claim follows.

## Preregistered model-treatment flags (2026-08-27, all default off)

Five config-gated model treatments exist for the next matched runs. Every
flag defaults to the exact legacy behavior, adds parameters only when
enabled, is recorded in the training contract (a resume across arms fails
closed), and is caller-owned under foundation-v2 validation. At most one
treatment may be enabled per arm; the strongest allowed claim for any of
them is "worth a matched test".

1. `copy_bypass_gate` — copy-bypass gated outer update
   `l = clamp(rms_norm(y + ny)); y' = y + a*(l - y)` with scalar `a`
   zero-initialized. `a = 0` is exact latent copy for any finite state (the
   diagnosed F1 failure: the trained run's one-step latent MSE was ~18.5x
   latent copy); `a = 1` reproduces the legacy update algebraically exactly
   (verified to 1e-6 in f32; the flag-off baseline arm is the bit-identical
   comparator), so the baseline is an interior point of the treatment. The
   interpolation is re-clamped to the legacy activation envelope, and the
   uniform AdamW weight decay drifts the unconstrained gate toward its own
   copy-null; both are recorded interpretation caveats. Evidence: local algebra
   (machine-checkable identity properties) plus the zero-gate residual
   surgery result (arXiv:2607.16568); post/pre-norm Transformer papers
   motivate measurement only. First treatment to test.
2. `copy_gate_bias_prior` — copy-gate bias initialized to `logit(p)` for an
   expected changed-pixel rate `p` (exact on a zero latent; approximate on
   real latents, whose kaiming-normal gate weights perturb the logit).
   Fresh-init training restores the prior after the generic reinitializer,
   which zeroes every bias. Evidence: preregistered-engineering-choice, and
   a recorded counterargument only: the papers review argued a neutral zero
   bias is correct under the class-balanced gate BCE (uninformative optimum
   0.5) and that a negative bias risks re-introducing saturation. No source
   argued for this knob; it stays optional and unscheduled.
3. `grid_scaled_action_impulse` — the ACTION6 Gaussian impulse exponent
   becomes `-(grid-1)^2/2 * d^2` (sigma = one latent cell). The legacy
   fixed `-16` was calibrated for the 8x8 grid; at patch 4 its neighbor-cell
   contrast blurred from 0.72 to 0.93, a silent coordinate-conditioning
   regression consistent with the 0.70 shuffled-action plateau. Evidence:
   proved-local (the lattice algebra was derived and independently
   re-verified in review; provenance in the research run's
   model-improvements-fable-core-math finding) plus
   preregistered-engineering-choice for sigma = 1 cell.
4. `decode_composition = joint_copy_mixture` — MAP of the mixture
   `(1-g)*copy + g*softmax(colors)`. One-directional by construction: a
   sub-0.5 gate can never be overridden (the copy component holds mass
   `>= 1-g`); above-0.5 gates with unconfident or current-favoring colors
   fall back to copy, so false edits are non-increasing by construction
   (the symmetric risk is suppressing true edits; net benefit is empirical).
   Deployable for frozen rescoring
   without retraining; the corresponding mixture NLL training objective is
   deferred and would need an objective-revision bump.
5. `positional_value_readout` — native-grid canonical readout with
   positional values (removes the proved 2x2 pooling alias; +57,344
   parameters). New runs only: loading a checkpoint without the embeddings
   fails closed. Not recommended for the same arm as `copy_bypass_gate`.

Prerequisite control for every arm: the repaired H2 rollout path must
demonstrably fire (>=16 fragments -> finite nonzero loss reaching the
recurrent core; the sealed run trained with a rollout loss of exactly zero).
`foundation_v2_rollout_floor_and_activation_premise` checks this at test
time; it is not a runtime launch gate, so the launch preflight must verify
a nonzero realized rollout loss on the actual batch size before a campaign.
The one-treatment-per-arm rule is enforced at validation: enabling more
than one treatment fails closed unless `allow_multi_treatment_arm`
explicitly waives single-factor attribution.

Bundle-run observability: an exploratory multi-treatment run publishes full
candle-graph bundles, including labeled tensor statistics, at every
preregistered `profile_updates` target; records EMA-weight copy-bypass,
outer-step-cosine, and copy-gate scalars at the 1024-update gate cadence; and
runs a same-weights eval ablation battery for decode composition, action
impulse, and copy-bypass alpha. These observations can confirm that a mechanism
was active and attribute eval-time ablation effects, but they cannot assign
training-dynamics credit among bundled treatments. Both the training evidence
and ablations remain exploratory/`selection_only`.

## 1. Data contract (fixes RC5, geometry pathology, transfer defects)

1. **Mixed stream, no lessons.** One stationary-schedule mixture per batch,
   annealed linearly over training (start → end proportions):
   `random_one_step` 35→25%, `factual_branches` 20→30%, `exploration` 20%,
   `sequential_fragments` (len ≤ 4, for the small open-loop term and event
   labels) 15%, `hazard_one_step` 10→5%. No stage boundaries, no observer-only
   phases; observer heads train concurrently on detached latents from step 0.
2. **Branch groups** (RC5): same-state groups covering all applicable simple
   actions plus ≥3 stratified ACTION6 coordinates (object cells, boundaries,
   empty region, one symmetric counterpart). Groups preserved intact through
   batching (existing `FactualBatch` machinery). Each group labeled with
   pairwise board-effect equivalence (status-row-free).
3. **Geometry randomization** (RC3/RC4): content size sampled from
   {7,8,10,12,16,24,32} (log-skewed toward small), placed uniformly at random
   in the 64×64 canvas. Provenance content rect kept exact; pixel and auxiliary
   reconstruction losses receive the content mask. Held-out splits: unseen-seed 7×7, 8×8 composition
   (existing), plus translated-7×7 and 16×16 splits.
4. **Symmetry augmentation** (Lean: `Symmetrization.lean`): per-sample color
   permutation over colors 1–15 (0 fixed), consistent across
   current/next/branches; D4 transforms with action conjugation (directional
   actions relabeled, ACTION6 coordinates transformed). Status row painted
   after augmentation.
5. **Operator diversity** (transfer): ACTION5/ACTION6 semantics sampled per
   episode from an operator family (teleport, toggle, paint, push-line,
   swap-region…), not one fixed meaning. Entire operator families held out for
   eval. Keep at least the two v4 operators in-distribution for comparability.
6. **Goal dropout**: goal features zeroed for 30% of samples so the live
   zero-goal query is in-distribution (RC7/live mismatch).

## 2. Model (fixes RC1 decode path, action conditioning; keeps LeWM lineage)

1. Keep: conv palette encoder, recurrent core, canonical RMS-normed readout,
   spatial ACTION6 field, JEPA latent prediction.
2. **Patch size 4** (16×16 latent grid) as the default; patch 8 retained as
   config. Motivation: a 7×7 board currently occupies one latent token; v5
   boards span ≥4 tokens even before geometry randomization.
3. **FiLM action injection at every recurrent step** (inner and outer):
   action embedding → per-channel (γ, β) applied inside each residual block,
   in addition to the spatial action field at input.
4. **Predicted-state decoding path**: the exact decoder head applies to
   `out.y` with gradients flowing into predictor AND encoder-shared trunk
   (not detached). Same head still grounds encoded states.
5. **Copy-gate head**: per-pixel sigmoid gate (changed vs copy) from the
   predicted latent; eval composition: gate·argmax(color logits) +
   (1−gate)·current pixel.
6. Tiny projections (<8 rows or cols) move off Muon to Adam (RC8).

## 3. Objective (fixes RC1–RC4, RC6; theorems: CrossEntropy, CopyAttractor,
   MarginalBlindness, Separation, ModeOptimality)

Weights are initial; the gradient-budget controller (below) adapts EP only.

1. **L_pred_ce (weight 1.0, the primary loss)**: per-pixel 16-way CE of
   decoded `out.y` vs true next frame; content-masked; status row excluded;
   split into changed/unchanged means combined as
   `w_c·CE_changed + w_u·CE_unchanged` with `w_c/w_u = clamp((1−p̂)/p̂, 1, 64)`,
   p̂ = changed-pixel fraction of content pixels in the batch. Unimix 1%.
2. **L_gate (0.5)**: BCE on the copy gate, `pos_weight = (1−p̂)/p̂` (clamped 64).
3. **L_latent (0.25)**: spatial + canonical Huber (kept as stabilizer only).
4. **L_enc_ce (0.1)**: exact CE grounding of encoded current/next
   (as v4, but content-masked and changed/foreground-weighted — RC4).
5. **L_sep (0.2) + L_pull (0.1)** on branch pairs: hinge margin m=0.3 on
   normalized displacement distance for distinct board effects; MSE pull for
   equivalent effects. **L_invact (0.1)**: action-type CE (+ coordinate loss
   for ACTION6) from (z_t, ẑ_{t+1}); no-effect pairs masked.
6. **L_ep (init 0.01)** with a **gradient-budget controller** (RC2): every 128
   steps measure encoder-gradient L2 of L_ep vs L_pred_ce; rescale EP weight so
   its gradient norm ≤ 0.3× the prediction gradient norm. Log both. EP stays
   marginal (LeJEPA), population = current+next canonical states.
7. **L_rollout (0.02)**: open-loop horizon 2 ONLY, computed on ≥16 batched
   traces (RC6). No horizon progression. Auto-disabled if the one-step
   changed-exact gate regresses.
8. **Observer heads** (concurrent, detached inputs): Q trains on **graded**
   changed-pixel accuracy (soft BCE), not the ≥99%∧≥90% threshold (RC7);
   reliability trains on thresholded factual latent MSE as amended 2026-08-28;
   event/noop heads remain as v4 with goal dropout.

## 4. Optimization (RC2, RC6, RC8, T0.3)

- Muon+Adam hybrid retained; verify 0.2·√max(fan_in,fan_out) RMS matching;
  tiny/degenerate matrices → Adam.
- **WSD schedule**: 500-step warmup → stable 1e-3 → cosine decay to 1e-4 over
  the final 15%. EMA (decay 0.999) maintained; eval uses EMA weights.
- Batch 2048 is the nominal target; launch preflight selects the largest stable
  physical batch. The EP controller first limits its encoder pressure, then one
  combined global L2 clip at 1.0 is applied to the complete objective.
- 24,576 steps total. Checkpoint every 256; permanent eval bundle every 2048;
  **best-checkpoint tracking** by the configured held-out exactness metric
  (historical default: changed-exact); the exported evaluation model is the
  selected checkpoint's EMA.

## 5. Run gates (automated, in-trainer; RC2/RC6 would have been caught)

Evaluated on a fixed 512-transition held-out set every 1024 steps; each gate
logs PASS/FAIL to the report and a FAIL twice in a row aborts the run with a
sealed diagnostic bundle:

1. `improvement_fraction > 0` (predictor beats latent copy on changed
   transitions) — enforced after step 4096; measured and logged from the
   first evaluation.
2. Shuffled-action changed-pixel ratio ≤ 0.95 (action sensitivity) —
   enforced after step 4096; measured and logged from the first evaluation.
   Amendment 2026-08-28: the denominator now contains only genuinely changed
   ACTION5/ACTION6 tuples whose generation-time counterfactual, replayed with
   the target row's recorded operator and the exact conjugated coordinate shown
   to the model, changes the status-excluded gameplay outcome. Outcome-equivalent
   tuples no longer penalize a correct simulator. At least 32 outcome-changing
   rows are required; a smaller or unavailable causal population is a fail-closed
   evaluation configuration error. The ≤0.95 threshold is unchanged because an
   action-blind model still approaches ratio 1.0 on outcome-changing rows.
3. Foreground reconstruction pixel accuracy (encoded next) ≥ 0.60 after step
   8192 — a decoder-collapse floor. Amendment 2026-08-26: originally ≥ 0.85
   after step 4096; the first launch measured a stable asymptote near 0.67
   (0.639 → 0.675 over steps 4096–9216) while changed-exact climbed 0.42 →
   0.51. That historical instrumented run was not independent healthy evidence;
   the threshold was relaxed only as an internal collapse detector, not as a
   promotion result.
4. One-step changed-exact within 20% of its running best (collapse detector)
   — active from the first evaluation, since it is relative to the run's own
   best. Amendment 2026-08-25: gates 1–2 originally enforced from step 1024
   and aborted the first launch at step 2048 with mid-training thresholds
   applied to a warmup-phase model (changed-exact was already 0.071 at step
   1024, above full-v4's final 0.034); absolute-quality gates now share
   gate 3's warmup grace.

## 6. Evaluation fixes (all §2.6 bugs from the research doc)

Changed-transition stratification via `noop == Some(false)`; shuffled control
partitioned by `provenance.source_kind`; action-masked control uses a trained
NULL action embedding (id 0 added to training range with no-op semantics);
content-mask metrics reported per-source only; full-frame exactness is separate
from changed exactness; content false edits and padding hallucinations are
separate; shuffled controls record genuinely changed tuple counts;
rollout/one-step populations unified or labeled non-comparable; h16 rows
populated or removed; live/train row-63 handling documented; add
action-controllability probe (Δ latent across actions, Genie-style) and
ambiguity-ceiling measurement to the eval report. Foundation-v2 Q calibration
uses the same exact composed-transition labels as its training contract.
The fixed in-trainer gate is fingerprinted and explicitly `selection_only`:
it may choose a checkpoint but cannot satisfy a promotion claim. Promotion
requires a fresh untouched, preregistered, multi-seed evaluation population.
For shuffled actions the report distinguishes total rows, eligible rows, and
genuinely changed input tuples. V5 gate populations additionally replay each
shuffled operator tuple from the target current board and report the exact
outcome-changing count; legacy populations without that sidecar retain `null`
rather than guessing from model output.

## 7. Explicitly deferred (next iteration, not this run)

Goal/belief-conditioned Q-ranking policy head, history/belief state,
prequential test-time adaptation, adaptive-depth recurrence (pending T0.2
premise check), model scaling beyond width 128 (pending this recipe passing
gates at 560K–1M), MaskGIT refinement, ITC/OT decoding.

## 8. Success criteria

Held-out synthetic changed-exact ≥ 0.20 by step 8192 and ≥ 0.35 at best
checkpoint (research-doc estimate band); action-sensitivity gate green
throughout; foreground reconstruction ≥ 0.90 final. Live ARC-AGI-3 score is
reported but NOT a gate (policy work is deferred).
