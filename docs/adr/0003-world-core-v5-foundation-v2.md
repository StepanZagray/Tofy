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
   in the 64×64 canvas. Provenance content rect kept exact; all losses receive
   the content mask. Held-out splits: unseen-seed 7×7, 8×8 composition
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
7. **L_rollout (0.02)**: open-loop horizon 2 ONLY, computed on ≥64 batched
   traces (RC6). No horizon progression. Auto-disabled if the one-step
   changed-exact gate regresses.
8. **Observer heads** (concurrent, detached inputs): Q/reliability trained on
   **graded** changed-pixel accuracy targets (soft BCE), not the ≥99%∧≥90%
   threshold (RC7); event/noop heads as v4 with goal dropout.

## 4. Optimization (RC2, RC6, RC8, T0.3)

- Muon+Adam hybrid retained; verify 0.2·√max(fan_in,fan_out) RMS matching;
  tiny/degenerate matrices → Adam.
- **WSD schedule**: 500-step warmup → stable 1e-3 → cosine decay to 1e-4 over
  the final 15%. EMA (decay 0.999) maintained; eval uses EMA weights.
- Batch 2048 physical; grad clip: **two clip groups** — (L_pred_ce + L_gate +
  L_latent + branch terms) and (L_ep) clipped separately at 1.0 and 0.25.
- 24,576 steps total. Checkpoint every 256; permanent eval bundle every 2048;
  **best-checkpoint tracking** by held-out changed-exact (RC6).

## 5. Run gates (automated, in-trainer; RC2/RC6 would have been caught)

Evaluated on a fixed 512-transition held-out set every 1024 steps; each gate
logs PASS/FAIL to the report and a FAIL twice in a row aborts the run with a
sealed diagnostic bundle:

1. `improvement_fraction > 0` (predictor beats latent copy on changed
   transitions).
2. Shuffled-action changed-pixel ratio ≤ 0.95 (action sensitivity).
3. Foreground reconstruction pixel accuracy (encoded next) ≥ 0.85 after step
   4096.
4. One-step changed-exact within 20% of its running best (collapse detector).

## 6. Evaluation fixes (all §2.6 bugs from the research doc)

Changed-transition stratification via `noop == Some(false)`; shuffled control
partitioned by `provenance.source_kind`; action-masked control uses a trained
NULL action embedding (id 0 added to training range with no-op semantics);
content-mask metrics reported per-source only; rollout/one-step populations
unified or labeled non-comparable; h16 rows populated or removed; live/train
row-63 handling documented; add action-controllability probe (Δ latent across
actions, Genie-style) and ambiguity-ceiling measurement to the eval report.

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
