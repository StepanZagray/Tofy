# Foundation improvement research — 2026-08-24

Multi-model research pass over the full-v4 / foundation-v1 recipe: three Claude
Fable research agents (code diagnosis, architecture frontier, training-process
theory, each fanning out literature sub-agents), three GPT-5.6 Sol agents via
codex (repo-grounded deep theory, independent recipe review, web-frontier
survey), and a Lean 4 prover agent. Raw agent reports with full citations live
in [`raw-2026-08-24/`](raw-2026-08-24/). Machine-checked proofs live in
[`../../formal/`](../../formal/).

Scope note: everything here is analysis of the archived
`full-v4-speed-opt-a40-b2048` campaign and the code at `c2ed65dc`. The
in-flight `foundation-v1` rerun was not touched; see §6 for what to do with it.

## 1. Executive summary

The reported "2–5% changed-pixel exact accuracy plateau" is a misreading of
three stacked phenomena, each now evidenced:

1. **It was a collapse, not a plateau.** Changed-exact peaked at **16.6% at
   step 8192** (end of the dynamics lesson), fell to 13.6% during exploration,
   collapsed to 3.4% during the `sequential` lesson — the only lesson that
   enables the open-loop rollout loss, which is computed on a **single
   trajectory per optimizer step** at batch 2048 (`train.rs:3814-3816`) — and
   then the world core was frozen by design for the final two observer lessons
   (`one_step_latent_mse` is bit-identical from step 20480 on). All world
   learning happens before step 20480; the final number describes a frozen
   post-collapse model, and no checkpoint selection protected the 16.6% one.
1b. **Epps–Pulley gradients dominated and collapsed the encoder.** From the
   run's own telemetry: dynamics-lesson `sigreg_raw` ≈ 80 vs `next_latent`
   0.116 (~70× after weighting); encoder EP-gradient L2 = 778 vs prediction
   11.1 (**70:1**, cosine −0.027 — orthogonal noise pressure); **100% of world
   updates hit the global clip** (`MAX_GRAD_NORM = 1.0`) at scale 0.18–0.41,
   throttling the prediction gradient 2.5–5×. Measured consequence: the
   encoder's latent distance between consecutive states on *changed*
   transitions shrank 0.038 → 0.0066 over training — it learned to erase the
   transition signal to look Gaussian — and `improvement_fraction` was
   negative at every checkpoint (−10.5 final) without tripping any gate.
2. **The metric measures a composition that is never trained.** The loss is
   latent Huber on `out.y` plus pixel CE on *encoded* states only; nothing ever
   asks whether the *predicted* latent decodes to the right pixels. On the
   archived run the predicted-state decode is statistically indistinguishable
   from an all-background control (changed-pixel accuracy 0.459 vs 0.439), and
   the decoder itself caps the metric at 0.60 even on the true encoding.
3. **Action effects are provably unidentifiable from the data.** Full-v4
   trains on one expert action per state. On finite grids the consistent
   hypothesis class has size `|S|^(|R|·(|A|−1))`; an action-blind predictor is
   loss-equivalent to the truth. This is now a machine-checked theorem
   (`formal/TofyFormal/Identifiability.lean`), and the 2026 literature has the
   matching published impossibility results (arXiv:2607.22430 Cor. 1,
   arXiv:2603.17577 Prop. 4.1) and the matching empirical failure mode
   ("context collapse", ActSWM arXiv:2607.26712) — in SIGReg-regularized JEPA
   world models specifically.

Separately, and independent of prediction quality: **the live greedy policy has
no objective to pursue.** Its score is 55% prediction-confidence heads (trained
on ~97%-negative correctness labels), 30% anti-no-op evaluated at an
out-of-distribution all-zeros goal vector, 15% latent displacement magnitude.
A perfect world model attached to this scorer is still not a hidden-goal
solver; a memoryless frame-only policy provably cannot be
(`formal/TofyFormal/Policy.lean`).

The cross-model consensus plan (§4): fix the objective (predicted-state
categorical CE with changed-pixel weighting ≥ (1−p)/p), fix the data
(same-state branch groups; content masking; translation/color randomization),
fix the policy target (goal/belief-conditioned action ranking), and only then
spend on capacity, depth, or optimizers. Sol's combined estimate for the first
four interventions: changed-exact moves from 2–5% into the 20–45% range.

## 2. Diagnosis of the archived run

### 2.1 Timeline: collapse, then freeze

| step | lesson boundary | changed-exact |
|---|---|---|
| 8192 | end `dynamics` (2×4096) | **0.166** |
| 16384 | end `exploration` (2×4096) | 0.136 |
| 20480 | end `sequential` (rollout loss on) | 0.034 |
| 24576 / 28672 | observer lessons, world weight 0 | 0.034 (frozen) |

The `sequential` lesson is the only one enabling `rollout_weight * ramp` (0.1)
and `ptrm_rank` (`train.rs:2044-2054`). The 2025 theory predicts exactly this:
for a well-specified model class, multi-step open-loop training raises one-step
error (arXiv:2504.01766); scheduled-sampling-style mixing optimizes an improper
objective (arXiv:1511.05101). The rollout loss also attenuates signal exactly
on hard transitions via its per-step smooth cap and error-triggered teacher
reset (`train.rs:3802-3850`).

### 2.2 The objective/metric seam

- Metric: all-or-nothing exact match over the changed-pixel mask per
  transition, no-change transitions dropped (`semantic_eval.rs:141-181`).
  Most synthetic transitions change exactly 2 pixels.
- Loss: latent Huber (`train.rs:3067-3073`) + pixel CE on
  `encoded.current`/`encoded.next` only (`train.rs:3081-3088`). `out.y`
  reaches the decoder only detached, to manufacture observer labels
  (`train.rs:3172-3176`).
- Archived-run consequence (`step-000000028672.json`):
  `target_reconstruction` changed-exact 0.600 (decoder ceiling),
  `one_step_prediction` 0.034, changed-pixel accuracy 0.459 vs `zero_control`
  0.439 — the predictor decodes to near-background. Prediction foreground
  accuracy is 8.5% vs 31.9% for decoding the *encoded current* state: the
  predicted latents are off the decoder's training manifold.
- The decoder ceiling itself (0.600 on true encodings, 26.4% foreground) comes
  from the unweighted exact CE over pixels that are 99.4% background: the ADR's
  99%-overall reconstruction gate passes trivially on near-empty boards while
  foreground is lost. Even a flawless transition model cannot exceed 60% under
  this contract.
- Class imbalance, quantified: ≤2 changed pixels per transition out of 4,032
  gameplay pixels (0.05%); in the latent Huber the copy attractor sits within
  ~0.003 of optimal against a total objective of 0.7–8.1 — a 10³–10⁴:1
  dilution. The only balanced changed/unchanged weighting in the codebase
  lives in the disabled legacy histogram head (`grounding.rs:301-321`).
- The encoder is nearly transition-invariant: copy-in-latent-space MSE is
  0.0066 while the predictor's is 0.0760
  (`changed_transitions.improvement_fraction = −10.5`) — the predictor is 11×
  worse than doing nothing, in its own latent space.

### 2.3 Identifiability and the dead action channel

Full-v4 deliberately excludes `factual_branches` (`train.rs:44`, recipe at
`:631`) even though the repo already generates complete four-action same-state
branch groups (`data.rs:995`, `:1117`) and the world-core-v2 ADR records that
single expert actions do not identify counterfactuals. Measured action
blindness on the archived run: shuffled-action latent-MSE ratio 0.975 (CI
spans 1.0), `action_conditioning_pass: false`; changed-pixel accuracy 45.9%
(true action) vs 44.3% (shuffled). With a=π(s) a function
of the state, I(a; s′ | s) = 0: an action-blind transition is a global
minimizer of every prediction loss. Epps–Pulley cannot see this — any
marginal-law regularizer is exactly invariant to replacing z(s,a) with g(s)
(blindness theorem, §3). Published confirmations: identifiability iff
conditional action variance ρ_tr(π) > 0 (arXiv:2607.22430 Thm 1); counterfactual
error amplification 1/ρ_tr diverges for deterministic experts; ActSWM's
"context collapse" observed under SIGReg.

### 2.4 Data geometry

- Boards are 7×7 (train) / 8×8 (held-out), always in the top-left of the
  64×64 canvas (`generator.rs:255`, `data.rs:83`); after status-row masking,
  98.78% of supervised decoder pixels are padding; `PAD` and semantic `EMPTY`
  share palette value 0 (`data.rs:31`); the grounding loss receives no content
  mask (`model.rs:622`).
- The whole board fits inside a single 8×8 encoder patch → one latent cell.
  The "spatial" recurrence trains almost entirely within one latent position;
  cross-patch dynamics are untrained. This also voids the depth-shortage
  hypothesis as stated (§3.4) while creating a worse transfer problem.
- Synthetic action semantics are one operator per action type (ACTION6 =
  teleport agent, ACTION5 = toggle switch; `data.rs:842`, `:899`). Public
  games reassign mechanics freely. There are no iterative-propagation
  mechanics (gravity, flood fill) anywhere in the curriculum.

### 2.5 Policy misalignment

`ModelPolicy::score_candidates` (`arc3_live.rs:620-693`):
`0.25·σ(q) + 0.30·σ(reliability) + 0.30·(1−σ(noop)) + 0.15·effect/(1+effect)`,
goals passed as zeros (training used one-hot goal families —
`data.rs:192-206`), tried-action −1 penalty keyed on exact frame hash.
Q and reliability are both trained on exact-decoder transition-correctness
labels (`train.rs:3167-3178`, `:3210-3224`) — at 2–5% changed-exact these
labels are ~97% negative, explaining the heads' below-chance balanced
accuracy. The score prefers actions the model can predict, not actions that
advance any goal. `levels_completed`, `win_levels`, and full episode history
are available in the API observation (`arc3_live.rs:94`) and unused.

### 2.6 Bugs and audit findings (fix regardless of strategy)

1. Changed-transition stratification compares full frames including the
   status row, so the advancing budget bar makes 1396/1408 transitions
   "changed" (`eval.rs:2591-2598`); should use `noop == Some(false)` or a
   status-excluded comparison.
2. Gradient-pressure diagnostics only fire when `patch_grounding_weight > 0`;
   full-v4's exact-grounding term (0.1) shipped with zero gradient-scale
   evidence (`train.rs:5029`).
3. `shuffled_samples` rotates actions by caller-supplied spans, not
   `provenance.source_kind`; for `arc3_transfer` one span covers everything,
   so the control can donate actions across game families
   (`semantic_eval.rs:420-437`).
4. `action_masked_prediction` uses action id 0, outside the trained 1..=7
   range — it measures OOD-embedding response, not action ablation
   (`semantic_eval.rs:470`).
5. Live frames keep real row-63 content while training always paints a
   synthetic budget bar there and masks it from every loss; the encoder
   consumes all 64 rows (`model.rs:659-662`). Hard-masking row 63 also
   assumes every public game uses it as status — not established.
6. `content`-mask pixel accuracies are dominated by all-background padding for
   full-frame sources; don't quote them (`semantic_eval.rs:242-248`).
7. One-step and rollout semantic tables come from different populations
   (different samplers, ≥4-step filter); they are not a degradation curve.
   h16 rows are never populated (`n16 = 0`).
8. Eval rollout uses `forward_from_latent` while the training rollout uses
   `predict_latent_with_depth` — currently harmless for `y`, but the paths can
   drift.
9. Rollout loss uses one trajectory per optimizer step (`train.rs:3814-3816`)
   — high-variance long-horizon gradients on top of an already clip-saturated
   update; the proximate destabilizer of the sequential lesson.
10. The ±32 latent clamp after per-sample RMS norm can never bind
    (`model.rs:1195`), and `DEFAULT_MAX_ROLLOUT_HORIZON = 8` means the
    documented "2→4→8→max(16)" final stage never occurs.
11. Optimizer details: constant lr 1e-3 with no warmup/decay/EMA anywhere;
    Muon's normalized-buffer updates inject fixed-magnitude (~1.8e-4 RMS)
    whitened steps even at tiny gradients — a refinement noise floor for
    pixel-exact decoding — and Muon orthogonalizes degenerate 128×2/128×4
    projections that belong on Adam.

## 3. Theory: what is proven vs supported

### 3.1 Machine-checked (Lean 4 + mathlib, `formal/`, zero sorries)

All ten theorem groups below build (`lake build`: 8718 jobs, zero warnings);
`#print axioms` on every declaration shows only
`propext, Classical.choice, Quot.sound`.

| file | theorems | content |
|---|---|---|
| `Identifiability.lean` | `exists_agree_on_data_ne_off_support`, `not_identifiable_of_missing_branch`, `eq_of_agree_on_full_coverage` | off-support transitions are unlearnable from data alone; full branch coverage identifies |
| `CrossEntropy.lean` | `indicator_ne_le_ce`, `indicator_exists_ne_le_sum_ce` | changed-pixel CE (in bits) upper-bounds exact-transition error; CE < log 2 per transition is a sound surrogate |
| `Policy.lean` | `no_reactive_policy_optimal_for_both` | a frame-only policy cannot be optimal for two hidden goals with distinct unique optima |
| `CopyAttractor.lean` | `copy_weighted_loss_lower`, `copy_average_loss_le_of_rare_changes` | copy map is ε-optimal under uniform weighting as changes become rare; changed-pixel reweighting removes this |
| `Symmetrization.lean` | `symmetrized_risk_le` | group-averaged predictor has ≤ risk for any convex loss under invariant data (justifies color/translation/D4 augmentation with action conjugation) |
| `Locality.lean` | `LocalAt.comp`, `LocalAt.iterate`, `LocalAt.not_computes` | light-cone bound: depth-d r-local iteration is (d·r)-local and cannot compute longer-range dependencies |
| `MarginalBlindness.lean` | `marginal_regularizer_blind` (+ finite-weight versions) | any regularizer of the marginal latent law (incl. Epps–Pulley) is invariant under z(s,a) ↦ g(s) — cannot penalize action-independence |
| `Separation.lean` | `ne_of_sepLoss_eq_zero`, `sepLoss_ne_zero_of_action_independent` | positive-margin displacement separation provably rules out action-independent maps |
| `Greedy.lean` | `every_maximizer_is_correct`, `ranking_survives_uniform_error`, `greedy_on_approx_is_correct` | ordinal sufficiency for greedy play; ranking survives uniform error 2ε < γ |
| `ModeOptimality.lean` | `mode_maximizes_exact_match`, `huberRisk_argmin`, `huber_decode_mode_mismatch` | argmax/mode decoding is optimal for exact match; concrete distribution where the *global* Huber minimizer (z = 4/5, proved over all of ℝ) decodes to a non-mode |

### 3.2 Key quantitative facts

- **Reweighting threshold**: copy beats a competitor iff
  `p·w_c·I_C ≤ (1−p)·w_u·U_U`; equal-signal parity needs `w_c/w_u > (1−p)/p` —
  **49×** at p = 2% changed pixels, 19× at 5%. (Sol deep theory §2; Lean
  round 1.)
- **Non-identifiability count**: with one action per state over reachable set
  R, `|S|^(|R|·(|A|−1))` transition tables are observationally equivalent.
- **EP blindness**: any regularizer of the marginal latent law is invariant
  under z(s,a) ↦ g(s). Minimal fix: positive-margin displacement separation on
  distinct-effect branch pairs + equivalence pull (the Board Effect design
  from the v2 ADR).
- **Greedy sufficiency**: searchless greedy play needs only ordinal
  correctness of Q̂ with margin — `sup|Q̂−Q| < γ/2` per state suffices. Exact
  pixels are stronger than necessary; but nothing in the current objective
  optimizes any ranking.
- **Underparameterized regime**: at 560K params the model does not
  interpolate, so changed-pixel loss reweighting provably shifts the learned
  solution (the Byrd–Lipton vanishing-effect result applies only to
  interpolating models; arXiv:1812.03372 + 2112.12986).

### 3.3 Regression vs classification

Per-pixel argmax CE is MAP decoding — literally what exact-match scores.
Huber-on-latents decodes a robust conditional location, which under
multimodality/aliasing lands between modes (worked example: accuracies 0.35 vs
0.40). Field consensus for exact discrete prediction is categorical CE
(Stop Regressing ICML 2024; DreamerV3; Genie; Δ-IRIS; CompressARC uses exactly
per-pixel palette CE on ARC). Qualification (Sol): the synthetic simulator is
deterministic given (s,a), so CE's advantage here comes from discrete semantic
grounding and calibrated modes under encoder aliasing, not stochasticity.

### 3.4 The depth question, corrected

The "2 outer steps can't propagate across the grid" claim is **false for this
architecture**: nominal receptive field is 232 input pixels (8×8-stride patch
encoder + 14 conv-3×3 applications through 2 inner/2 outer steps) — the whole
board is visible even at one outer step. The honest statement: receptive-field
coverage is necessary, not sufficient, for algorithmic computation
(flood-fill/collision chains may still need iteration), and the curriculum
contains no such mechanics anyway. Adaptive/variable-depth recurrence is a
principled but lower-confidence bet, to be premise-checked first with the
existing matched-extra-outer-steps diagnostic (`model.rs:1489`) on frozen
checkpoints over constructed long-range tasks. Supporting literature if the
premise holds: TRM/HRM analyses (outer refinement does the work), RSM
stochastic depth, NCA-for-ARC with per-step supervision + a global channel;
warning literature: looped-transformer fixed-point restrictions without input
recall (arXiv:2604.15259), NCA periodic attractors (arXiv:2604.12720).

## 4. Prioritized intervention plan (cross-model consensus)

Both model families converged on the same top tier independently. Ranked by
expected impact ÷ cost; "falsify" = the cheap gate before committing a long
run.

### Tier 0 — measure first (hours)

- **T0.1 Ambiguity ceiling**: group identical visible-history/action inputs;
  count groups with multiple factual successors, history lengths 1–16. If
  one-frame collisions are common, no one-frame model can score high
  exact-match and history conditioning (T3.2) becomes mandatory.
- **T0.2 Depth premise check**: frozen-checkpoint eval with extra outer steps
  on constructed propagation tasks (radius-parameterized). Gates T4.2.
- **T0.3 LR-decay branch before eval**: constant-lr 1e-3 checkpoints
  systematically understate the model (WSD river-valley result,
  arXiv:2410.05192). Decay for ~500 steps before each eval checkpoint, or
  evaluate a schedule-free average. May move the headline number for free.
- **T0.4 Fix §2.6 bugs** so subsequent experiments are read correctly.

### Tier 1 — objective and data (the decisive experiments)

- **T1.0 Rebalance the gradient budget.** Cut `sigreg_weight` ~10× or
  normalize the EP gradient to a fixed fraction of the prediction gradient;
  clip EP separately instead of sharing the single global `MAX_GRAD_NORM=1.0`.
  Add hard run gates: `improvement_fraction > 0` (was negative at every
  checkpoint of the archived run without tripping anything) and an
  action-sensitivity floor. Also: batch the rollout traces (not 1 per step),
  gate each horizon increase on non-regression of one-step changed-exact, and
  select `model.best` by changed-exact before any freeze.
- **T1.1 Predicted-state categorical CE.** Decode `out.y` with the (now
  trainable-through) exact decoder; per-pixel 16-way CE against the true next
  frame, masked to content, split changed/unchanged with
  `w_c/w_u ≥ (1−p)/p` computed on content pixels (p measured per lesson).
  Optional copy-gate factorization: per-pixel binary changed head + color
  head, output = gate·color + (1−gate)·previous (ITC/Δ-IRIS lineage). Add
  unimix (1% uniform) for logit stability. Justified by the CE bound and copy
  theorems; cost ~10–35% step time. Expected: +20–50 pp changed-exact (Sol
  review), +8–25 pp (Sol frontier).
- **T1.2 Same-state branch groups in every world lesson.** Reinstate
  `factual_branches` (≥25% of batches), ACTION6 coordinates stratified over
  objects/boundaries/empty/symmetric sites with outcome-equivalence-class
  sampling rather than exhaustive enumeration. Justified by the
  identifiability theorem — nothing else can substitute. Expected: +10–30 pp.
- **T1.3 Board-effect displacement separation + pull.** On branch pairs:
  hinge-margin push for distinct effects, pull for equivalent effects
  (both defined status-row-free), optional inverse-action head
  (Δ-JEPA/SMWM-style; mask no-op pairs where actions are unrecoverable).
  Kills the EP-blind action-independent solution class. Depends on T1.2.
- **T1.4 Data geometry repair.** Random board placement and scale (7→16→32→64
  content), provenance-based padding mask in every loss, PAD ≠ EMPTY, color
  permutation augmentation (consistent across the pair; keep pinned semantic
  colors fixed), D4 augmentation with action conjugation. Justified by the
  symmetrization theorem; near-zero cost. Expected: +15–40 pp and the main
  transfer fix.
- **T1.5 Demote the horizon curriculum.** Dominant one-step loss; horizons as
  a small-weight auxiliary (or pushforward/noise-injection for stability);
  ablate the curriculum against pure one-step given §2.1. Keep at most
  2→4 with weight ≤0.02 until one-step exactness is high.

The first long-run-worthy experiment is Sol's **2×2 causal screen**: T1.1
on/off × T1.2 on/off, 1–2K updates each from the same checkpoint, identical
seeds/budget, gated on changed-exact, unchanged-exact, copy-forward ratio,
same-state outcome retrieval, action sensitivity, predicted-logit margins.

### Tier 2 — action conditioning and stability (cheap adds to Tier 1)

- **T2.1 FiLM/AdaLN action injection at every recurrent step** (keep the
  spatial ACTION6 field). AdaLN beats concat in the published ablation
  (DisCo). Add a Genie-style action-controllability probe (Δ latent across
  actions on the same state) to the eval report.
- **T2.2 Zero/shuffled-action rollout hinge + frozen action readout**
  (ActSWM): needs no new data; complements T1.3.
- **T2.3 Curriculum mixing**: replace hard 5-lesson staging with annealed
  mixture or ≥25% replay of earlier lessons per batch (strong-shift continual
  learning result); keep observer detachment but let world lessons continue at
  low weight during calibration stages. Run a random-mixture control — the
  curriculum-ordering literature predicts it may match.

### Tier 3 — make the greedy policy an actual policy

- **T3.1 Goal/belief-conditioned Q ranking.** Train listwise/pairwise action
  ranking on synthetic episode returns (branch-relative advantages where
  branches exist). Greedy ordinal sufficiency is the guarantee; the current
  confidence-blend score is the single biggest live-score blocker even under a
  perfect world model. Feed `levels_completed`/terminal signals live instead
  of zero goals.
- **T3.2 History/belief state.** Recurrent episode state over
  (frame, action, outcome) history — required if T0.1 finds collisions;
  justified by the reactive-policy impossibility theorem and the
  version-space identification bound (informative transitions shrink |V_t|
  geometrically).
- **T3.3 Prequential test-time adaptation** (searchless): per-game resettable
  adapter (action embeddings / low-rank / per-game FiLM), 1–5 SGD steps per
  observed transition, replay episode transitions, revert-on-regression,
  meta-trained across synthetic games. Encoder frozen. TTT was decisive on
  static ARC; this is its correct interactive mapping.

### Tier 4 — capacity, depth, optimizer (only after Tiers 0–2 pass)

- **T4.1 Scale 4–30×** (width 256–384 and/or depth; 2–20M params): scaling-law
  evidence says 560K is 1.5–2 orders below compute-optimal for this budget;
  trade batch size down if needed. Do not scale while the small model still
  learns copy-only.
- **T4.2 Variable-depth weight-tied recurrence** with input/action recall at
  every step, per-step supervision, randomized train depth, extended eval
  depth — gated on T0.2.
- **T4.3 Optimizer hygiene**: matched AdamW control (same wd, matched update
  RMS); batch-size warmup 128→2048 per lesson (CBS ≈ 0 early); verify Muon
  0.2·√max(fan_in,fan_out) RMS matching and parameter routing. Rank last —
  no evidence Muon causes or cures the plateau.

## 5. Beyond-frontier positioning

- No published neural next-state baseline exists for ARC-AGI-3 (all public
  leaders are LLM coding agents with executable world models; public set is
  harness-saturated: Tycho 100%, Retrodict 99.9%; verified no-harness model
  results: Claude Opus 5 30.16%, GPT-5.6 Sol 7.78% semi-private; humans 100%).
  A working tiny-neural searchless agent is a novel result at any decent
  score.
- Genuinely novel combinations identified: pixel-CE grounding head inside a
  LeJEPA/EP world model (unpublished); copy-gate palette decoding at native
  pixel resolution (no VQ tokenizer needed — grids are already tokens);
  action-conjugated D4 augmentation for interactive grids; meta-trained
  prequential adapter under a strict no-search policy; branch-group
  interventional curriculum as an explicit identifiability repair.
- The executable-world-model winners' principles translate into trainable
  objectives: factual transition memory (→ T3.2), untested-action tracking
  (→ exploration ledger), discriminating probes (→ information-value head),
  retrodiction verification (→ revert-on-regression in T3.3), and separating
  "predictable" from "advances the goal" (→ T3.1).

## 6. The in-flight foundation-v1 run

Treat strictly as replication (Sol's phrase). Expectations: peak world-model
quality near step 8192; collapse during 16384→20480; frozen after. Do not
extend or modify it. When it finishes: evaluate *all* interim checkpoints (esp.
8192) with a decay branch (T0.3), run the T0.1 ambiguity measurement on its
data, then start the 2×2 screen from the step-8192 checkpoint rather than the
final one.

## 7. Reading map

| report | file |
|---|---|
| Final ranked diagnosis (8 root causes, telemetry-backed) | `raw-2026-08-24/diag-final-ranked.md` |
| Eval metric + live policy code diagnosis | `raw-2026-08-24/diag-eval-metric-live-policy.md` |
| Sol independent recipe review (12 findings, 12-item plan, theory hooks) | `raw-2026-08-24/sol-recipe-review.md` |
| Sol deep theory (6 theorem sections + Lean skeletons + priorities) | `raw-2026-08-24/sol-deep-theory.md` |
| Sol web frontier (ARC-AGI-3 2026 state, TTT, Muon 2026, exploration) | `raw-2026-08-24/sol-frontier-websearch.md` |
| Discrete decoding / token world models | `raw-2026-08-24/arch-discrete-decoding.md` |
| Recurrent depth / NCA / TRM-HRM | `raw-2026-08-24/arch-recurrent-depth.md` |
| Symmetry, action conditioning, scaling laws | `raw-2026-08-24/arch-symmetry-action-scaling.md` |
| Regression vs classification, imbalance, copy attractor | `raw-2026-08-24/loss-regression-vs-classification.md` |
| JEPA collapse, SIGReg guarantees, identifiability | `raw-2026-08-24/loss-jepa-collapse-identifiability.md` |
| Horizons, Muon, batch size, curriculum/replay | `raw-2026-08-24/train-horizons-muon-curriculum.md` |
