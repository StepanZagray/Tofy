# Preregistration: ADR 0005 local falsifiers (2026-09-03)

Scope: the three cheap experiments ADR 0005 §5.1–5.3 that must run on the
local machine (RTX 5060 Laptop, 8 GB) before any pod GPU-hours are spent on a
v6 run. Everything here is `selection_only`/exploratory evidence: it can block
or unblock the v6 pod run, it cannot satisfy a promotion gate.

## E1 — Residual-vs-reliability AUROC (§5.3)

- Claim under test: on the latest local v5 checkpoint, the one-step composed-
  decode residual separates real ARC-AGI-3 frames from in-distribution
  synthetic frames better than the trained reliability head does.
- Checkpoint: `runs/p2/_pod_handoffs/6zp5oip7tvokfl-20260827-foundation-v2/foundation-v2/model.safetensors`
  (sha256 117f786b…, 2026-08-27). Caveat: NOT the s8 step-20480 best model;
  that checkpoint lives on the stopped pod. Conclusions transfer only as far
  as both models share the v5 recipe and the same live-query tuple.
- Real population: 25 public games, seeded uniform-random policy, 150 actions
  each, toolkit local recordings; evaluation only. Synthetic population: same
  size, fixed seed, the trainer's held-out generator incl. 30% goal dropout.
- Query tuple: all-zero goal + UNKNOWN operator (what the live policy sends).
- Metrics: AUROC(real vs synthetic) for residual and reliability; AUROC for
  "composed decode wrong" within each population; bootstrap 95% CI (1,000).
- Controls: synthetic reliability high (positive); shuffled labels ≈ 0.5
  (negative).
- Decision: residual AUROC ≥ 0.80 AND reliability AUROC ≤ 0.60 ⇒ Phase A trust
  switches to residual-derived calibration until a v6 checkpoint exists.
  Otherwise no change to Phase A trust.

## E2 — Memorization diagnostic (§5.1)

- Claim: with the v6 generator (twins, UNKNOWN operator, whole-frame content)
  the model uses the Context Window: changed-exact on the row after the
  context is higher with K = 16 than with K = 0.
- Arms: one v6 model; evaluation with the context masked (K = 0) versus the
  full context (K = 16) on 512 held-out twin-pair meta-episodes, fixed
  evaluation seed 1000002.
- Budget: local run, 4096 optimizer steps, seed 2, evaluation at 1024/2048/3072/4096.
  The old qualification found physical 256 OOM and selected physical 128 ×
  accumulation 2, but recipe/CLI precedence silently overwrote accumulation to
  1. The completed implementation-smoke run therefore used effective batch 128
  and cannot satisfy this registration. Before the fresh 2x2 launch, measure
  physical 256 × accumulation 1 first and fall back to 128 × 2 only if needed,
  preserving effective batch 256 and recording the selected pair.
- Historical invalid launch (2026-09-03, revision 975832f9, binary sha256 in the run root):
  `tofy p2-train --recipe foundation-v2 --world-core-v6 --data-contract-v6 --device cuda
  --seed 2 --init-seed 2 --physical-batch 128 --grad-accum 2 --steps 4096
  --checkpoint-every-steps 512 --output-dir runs/p2/v6-memorization-e2-20260903`.
- Fresh 2x2 threshold: Δ ≥ 0.05 absolute at step 4096. The preflight census separately
  verifies that the population is mutually exclusive. Below threshold after a
  valid preflight means this 2x2 v6 training system did not learn to use that
  history within the budget; the pod run is blocked, but the result does not
  by itself identify the generator as the failed component.
- Also recorded (no threshold): the same delta on the legacy streams
  (expected ≈ 0, since legacy rows carry no context), and the model-free
  `single_frame_rule_identifiable` census (must be 0).

## E3 — Adaptation falsifier (§5.2)

- Claim: Channel B (fast-weight updates per §6.2) improves prequential
  changed-exact over Channel A (context only) on held-out synthetic
  meta-episodes, without making the adapted-then-frozen model worse than the
  prior.
- Population: 256 held-out meta-episodes (levels 2–4), adapt on transitions
  1..t, score t+1..t+4, t ∈ {8, 16, 32}. Seeds 1000002 (population), 7 (adapter).
- Arms: A (context only), A+B default (reset at level), A+B carry.
- Thresholds: promote B if prequential changed-exact improves by ≥ 0.02
  absolute at every t AND frozen-after-adaptation changed-exact ≥ prior on
  the same rows. Else Channel B ships disabled (`--adapt` off by default).
- Multiplicity: three arms, three t values; thresholds are pre-set, no
  post-hoc metric changes. A single seed is a screen only.

## Stop rules and runtime

- E1 ≤ 2 h wall clock; E2 ≤ 6 h (measured: 12 steps at batch 256 ≈ 33 s
  including start-up on the local GPU; per-step timing recorded in the run);
  E3 ≤ 2 h. Any integrity failure (hash, seed, evaluator) fails closed.
- Data-access boundary: public ARC-AGI-3 frames appear only in E1 and only
  as evaluation inputs; no public frame enters any training path.

## E1 outcome (2026-09-03, selection-only)

Run: `runs/p2/residual-probe-20260903/REPORT.md`, commits 95013dc7 / adcd918c.
Checkpoint actually probed: foundation-v2 2026-08-27 EMA-best (sha256 117f786b…),
plus the final export (e5457dd4…); conclusions identical. NOT the s8 model.

| task | residual (pixel-CE) AUROC | 1 − reliability AUROC |
|---|---|---|
| real vs synthetic, all rows (n=7,500) | 0.905 [0.896, 0.914] | 0.634 [0.620, 0.647] |
| real vs synthetic, changed rows only | 0.866 | 0.494 (chance) |
| wrong-prediction within synthetic | 0.488 (chance) | 0.719 |
| shuffled-label control | 0.502 | 0.498 |

Against the preregistered rule (residual ≥ 0.80 AND reliability ≤ 0.60): the
residual condition is met; the reliability condition is **narrowly missed**
(0.634, CI excludes 0.60). The switch of Phase A trust to residual-derived
calibration is therefore NOT triggered by the rule; it is recorded as a
recommended exploratory follow-up with a fresh recording seed, because the
head is at chance on exactly the rows that matter (changed transitions).

Additional facts that change earlier statements: the "~1e-8 reliability on
real frames" observation from the s8 run is NOT reproduced on foundation-v2
(real-frame median reliability 0.511); it is s8-specific and remains
untested. Residual is a distribution-shift detector, not a per-transition
correctness signal (chance within synthetic). Only 1.2% of real changed
transitions were predicted exactly (28/2,410); 0/3,750 full-frame exact.

## Superseded amendment (2026-09-03): recursion depth 3x3

The owner initially selected 3x3 for E2 and E3. The later audit established
that the rationale miscounted execution: 3x3 runs 12 residual blocks versus 6
at 2x2, while both receptive fields already cover the 16x16 latent grid. The
old timing also came from the invalid accumulation/provenance sequence, so it
cannot estimate the fresh run. The 2026-09-04 2x2 amendment below supersedes
this choice while preserving 3x3 as a separate treatment.

## Amendment (2026-09-04): which statistic the §5.1 threshold applies to

Code review (2026-09-04) found that `p2-eval --context-ablation` scores the
generic held-out mixed population (legacy streams plus LearningHistories rows
with K ~ Uniform{0..16}) with and without context, not the preregistered
"512 twin-pair meta-episodes, K = 0 vs K = 16" population. Legacy rows and
K = 0 rows contribute exactly zero delta, so `overall.delta` is diluted by
roughly 2x or more. Before any result is read, the rule is fixed as follows:

- The `>= 0.05` threshold applies to the **`5-16` context-length stratum**
  delta (rows whose generated window has K >= 5, scored with their window vs
  with K = 0), which is the closest implemented proxy for the preregistered
  statistic. `overall.delta` is reported but is not the gate.
- This is a weaker test than K = 16 exactly (windows of 5..16, mean ~10);
  an exact K = 0 vs K = 16 twin-pair evaluation remains to be implemented and
  will supersede this rule when it exists.
- The invalid E2 attempt intended effective batch 256 and depth 3x3; the later
  audit found that the checkpoint actually used effective batch 128, so it is
  not valid E2 evidence. The fresh registered run uses the 2x2 baseline. The
  screen remains a data-contract decision only.

## E2 execution record (2026-09-03/04, local RTX 5060 8 GB)

| attempt | launched | died at | cause (established) |
|---|---|---|---|
| 1 | 2026-09-03 ~14:40 | step 1024 | gate evaluation OOM; concurrent test suite on the GPU |
| 2 | 2026-09-03 22:07 | step 1024 | gate evaluation OOM with 128-row chunking (no concurrent load) |
| 3 | 2026-09-04 09:03 (resume 512) | step ~575 | concurrent CPU evaluation competing for system RAM |
| 4 | 2026-09-04 09:16 (resume 512) | step 1024 | gate evaluation OOM with 32-row chunking |
| repro | 2026-09-04 10:32 (resume 512→1032) | passed | outputs detached; GPU peak 5.7 GB at the gate |
| 5 | 2026-09-04 10:58 (resume 1032→4096) | completed 12:49 CDT | effective batch was 128, not the registered 256; provenance/root integrity also failed |

Root cause of 1/2/4: candle retains every op's inputs while its output lives;
the evaluator held chunk outputs, so chunking bounded nothing until the
outputs were detached. Attempt 5's checkpoints from step 1024 onward were
written by a resume launched with an absolute `output_dir`; evaluations must
use each checkpoint bundle's own `config.json`, which the evaluator enforces
by hash. E2's loss log contains the appended rows of all attempts; the
authoritative trajectory is the checkpoint chain, not the row count. The
completed root is **implementation smoke only**, not E2 evidence: five
attempts reused one root, 512 step values are duplicated, build provenance is
absent, the v6 run persisted v5 identity, and the implemented evaluator was
the mixed K=5..16 proxy rather than the registered twin K=16 versus K=0
statistic. Its inherited post-training evaluation was stopped on 2026-09-04
because it could not answer the registered claim.

## Amendment (2026-09-04): E3 as implemented, before any result is read

`p2-eval --adaptation-falsifier` (commit eaf4b619) implements §5.2. Two facts
found during implementation bound what it can show:

1. **The as-registered arm cannot update.** §6.2 starts Channel B only once a
   level holds >= 8 unique transitions; a synthetic Learning History level has
   exactly 7 (`LEARNING_HISTORY_STEPS_PER_LEVEL = 6` movement rows plus one
   operator row). Under the registered rule the reset and carry arms therefore
   reproduce the context-only arm bit for bit with zero updates, and the
   verdict fails by construction. This is a generator/limits mismatch, not a
   model result. The registered arm is still run and reported as such. A
   **labelled deviation arm** with `--adaptation-falsifier-min-level-transitions 4`
   is the first run that actually exercises Channel B; its result is
   exploratory and cannot satisfy the §5.2 promotion rule on its own.
2. **t = 32 is unreachable** (levels in {2,3,4} give <= 28 chronological
   transitions); the rule is applied over t in {8, 16} only, and the report
   lists the skipped prefix under `verdict.skipped_prefix_lengths`.

Consequence for the generator (deferred, trajectory-changing): synthetic
levels are far shorter than live levels (tens to hundreds of actions), so the
live warm-up rule is untestable on them; a v6.1 generator should draw level
lengths from a wider range.

## Amendment (2026-09-04): the registered E2 statistic and the redesigned E3 are implemented

Independent audit of the two amendments above found (A) that the `5-16`
stratum proxy is not the registered §5.1 statistic and (B) that the training
histories (7 transitions per level, <= 28 per episode) make the registered E3
arm vacuous by construction. Both are fixed in code before any E2 result is
read; the rules below supersede the proxy rule of the first 2026-09-04
amendment.

- **E2, registered statistic: `p2-eval --twin-memorization`.** 256 held-out
  Twin Episode pairs (= 512 twin-pair meta-episodes) on `UnseenSeed7x7`, seed
  1000002, rendered with the stream's v6 unit augmentation (one augmentation
  per pair, so the twins stay byte-identical until the rule acts), TRAINING
  level shape (6 movement rows + 1 operator row per level, levels {2,3,4})
  because E2 screens the data contract the checkpoint was trained on. Every
  chronological row `i >= 16` of every episode is scored with exactly the 16
  preceding transitions as context and again with `K = 0`; changed-exact and
  composed changed-exact are reported for both arms with the delta over (1)
  all scorable rows, (2) rows after the pair's first divergence and (3) rows
  whose 16-window contains an outcome-changing row (the same current frame
  and action mapping to different next frames across the twins). The
  `>= 0.05` rule applies to (3), `verdict.delta_filtered`. A model-free
  preflight census (pairs, single-frame-identifiable pairs, divergent pairs,
  pairs diverging before index 16, outcome-changing rows, scorable rows with
  evidence) fails closed before any forward. The fixed registered population
  is pinned at 256 divergent pairs, 0 single-frame-identifiable pairs, 0
  state-differing rows, 724 outcome-changing rows and 2,858 evidence-bearing
  scorable rows. Forwards run in slices of <= 32 rows with every kept output
  detached.
- **E3, redesigned histories.** The falsifier now renders EXTENDED Learning
  Histories by default: 24 movement rows + 1 operator row per level, levels
  {3,4,5} (>= 75 chronological transitions), so the production warm-up
  (8 unique transitions per level) can be met inside every level and all of
  `t in {8, 16, 32}` are reachable; the registered arm is no longer vacuous
  (`t = 16` is confirmed to update under the fixed seed). `t = 8` can still
  be warm-up-only when its first eight rows contain fewer than eight unique
  `(observation, action)` facts; that is reported telemetry, not hidden.
  `--adaptation-falsifier-min-level-transitions` is not needed for a
  non-vacuous run (it remains a labelled deviation knob). The shape is
  recorded in the report (`history`) and can be overridden with
  `--learning-history-steps-per-level` / `--learning-history-levels`. The
  training stream is unchanged (pinned content digest).
- The `--context-ablation` proxy is kept as a reported number only; it is not
  the gate.
- Exact E2 invocations set `--identifiability=false`. The oracle-latent ridge
  bridge is outside the K=16 versus K=0 claim and, at v6's 32,768-dimensional
  representation seam, its current primal implementation requires a 4 GiB
  Gram matrix and O(d^3) solve. The flag is recorded in the evaluation command
  identity and changes no twin population, model forward, reducer, or verdict.

## Amendment (2026-09-04): 2x2 is the v6 baseline

Before a fresh E2 result is observed, the owner selected
`inner_steps = outer_steps = 2` as the canonical v6 baseline. It executes 6
two-convolution residual blocks (receptive field 25) rather than 12 blocks at
3x3 (receptive field 49); both receptive fields already cover the 16x16 latent
grid. E2 changes no other causal factor and both K=16/K=0 evaluator arms share
the same 2x2 checkpoint. An explicit `--v6-recursion-steps 3` remains a
trajectory-distinct treatment that may be tested only after E2 validates the
history premise. This is a compute- and evidence-conservative baseline choice,
not a claim that 2x2 is globally optimal.

## Deferred treatment: depth ablation 2x2 vs 3x3 (2026-09-04)

- Claim: on the v6 data contract, the explicit 3x3 treatment yields higher
  held-out changed-exact than the 2x2 baseline at equal data, seed, physical
  and effective batch, and steps. Static argument only: the implementation
  executes 12 versus 6 two-convolution residual blocks, giving receptive
  fields 49 versus 25 cells on a 16-cell grid; both already cover the board,
  while 3x3 costs about twice the recurrent dynamics compute. There is no
  measured Tofy depth evidence.
- Arms: two fresh matched runs, differing only in `--v6-recursion-steps 3`
  versus `--v6-recursion-steps 2` (recorded through `inner_steps`/`outer_steps`).
  The old E2 root cannot be reused as the 3x3 arm because it actually ran
  physical 128 x accumulation 1 and lacks valid provenance. One matched seed
  may screen the effect size; any decision to replace the baseline requires a
  fresh multi-seed confirmation.
- Metrics: the held-out gate metrics (`one_step_changed_exact`,
  `one_step_composed_changed_exact`, `one_step_all_rows_exact`) per split and
  the §5.1 ablation delta, both read from the same evaluator at each run's best
  checkpoint. No threshold is preregistered; the result informs whether 3x3
  deserves a fresh confirmation against the 2x2 baseline and is reported as
  an effect size.
- Runs only if E2's §5.1 verdict is PASS; a failing data contract makes depth
  uninterpretable. Runtime must be registered from a fresh 2x2/3x3 batch and
  timing smoke; the old estimate is not valid evidence.

## Post-E2 diagnostic E2R: frozen continuous context response (2026-09-05)

The fresh registered 2x2 E2 run completed and failed: the exact K=16 minus
K=0 changed-exact delta was `0.0` at all four measured checkpoints, including
the fixed step-4096 EMA checkpoint, versus the registered `0.05` threshold.
E3 and the pod campaign remain blocked. Before any retraining, E2R localizes
the failure with a frozen-checkpoint rescore; it is exploratory diagnostic
evidence and cannot promote the model.

- **Claim.** On the exact E2 population and step-4096 EMA checkpoint, K=16
  changes the model's predictive distribution relative to K=0, and any change
  can be classified as target-helpful or target-harmful before choosing a
  training intervention. This is a checkpoint-local empirical hypothesis,
  not a claim about v6 generally.
- **Invariants and comparator.** Use Tofy
  `dadc3e5f4c18751f7f205100738f0653d243580c`'s sealed E2 checkpoint
  `c53bf0c42dc6c8f7945ff4d17bd6bd63a6db23e8b6e377b6b0b92903e66d694a`
  as the source model; evaluator code may add read-only reductions but must not
  change model parameters, population generation, seed 1000002, 256 pairs,
  TRAINING history shape, K=16/K=0 arms, row filters, or exact metrics. The
  population fingerprint must remain
  `sha256:484e8615e41102895997ddb9bec19665604fb7f62d21db9cc5ecea1470e58f42`
  and its registered census must match exactly. No public ARC data is read.
- **Comparisons and metrics.** For all three existing row sets, compare (1)
  own K=16 versus K=0/`ContextBatch=None`, (2) own K=16 versus the paired
  twin's K=16 on the same current-frame/action/target row, and (3) a zero-valid
  K=0 row carried inside a context-bearing mixed batch versus
  a same-size interleaved all-K=0 batch with `ContextBatch=None`. Comparison
  (2) isolates history content from the
  learned global context-FiLM bias; comparison (3) measures that bias seam
  directly. Each comparison reports context-summary and latent RMS distance;
  mean L1 distance
  between the two 16-color distributions over all gameplay pixels and over
  factually changed gameplay pixels; mean target NLL in each arm and
  `NLL(baseline) - NLL(treatment)` over factually changed pixels; mean absolute
  copy-gate difference; and raw/composed argmax disagreement row and pixel
  counts/fractions. Exact changed-exact fields and the registered verdict
  retain their meanings and reduction populations.
- **Fixed interpretation rule.** On evidence-bearing rows, call the predictor
  distribution-sensitive if changed-pixel mean probability L1 is `> 1e-6` or
  any raw argmax pixel differs. Call the continuous change target-helpful only
  if changed-pixel NLL improvement is `> 1e-4`; values in
  `[-1e-4, 1e-4]` are inconclusive at this precision, and values `< -1e-4`
  are target-harmful. Context is operationally inert only if probability L1,
  latent RMS, and gate absolute difference are all `<= 1e-6` and both raw and
  composed argmax disagreement counts are zero. These thresholds select a
  diagnosis, not a promotion. A nonzero mixed-K0 comparison confirms the
  batch-mask defect is behaviorally active in this checkpoint; own-versus-twin
  response is the stricter content-use test. The matched 2N K=0 baseline keeps
  batch geometry constant so kernel selection or batch-size rounding is not
  attributed to context-FiLM bias.
- **Branch.** Inert output leads to an instrumented context-summary/FiLM probe
  and a tiny one-twin overfit wiring test. Sensitive but non-helpful output
  leads to the same overfit test plus a context-contrastive or rule-prediction
  objective proposal. Helpful sub-argmax output leads to a changed-pixel
  context-amplification treatment, which still requires a fresh registered
  train/eval confirmation. No branch authorizes E3, a 3x3 ablation, or remote
  training.
- **Budget and stop rule.** One local CUDA rescore of the frozen checkpoint,
  bounded by the existing 32-row evaluator slices and 30 minutes wall time.
  Stop and fail closed on source/checkpoint/population/census drift, non-finite
  continuous values, CUDA failure, or changed legacy exact fields. Preserve
  the old report and write E2R into a never-reused root.

## Post-E2 diagnostic E2W: two-row context-wiring overfit (2026-09-05)

E2R found that the frozen 2x2 checkpoint reacts strongly to the presence of
the context pathway but barely to the context's factual content: on 4,998
factually changed pixels, own K=16 versus paired K=16 changed probability by
only `0.0000838796685474991` mean L1 and changed target NLL by
`-0.00000022291517032257957`, while context-present versus context-absent
changed probability by `0.02139037169509837`. E2W is the cheapest decisive
test of whether the corrected pathway is locally trainable. It is an
`implementation_smoke` and cannot promote a model, unblock E3, justify 3x3,
or provide ARC-AGI-3 planning evidence.

- **Bounded claim.** Starting from the sealed E2 step-4096 EMA, the production
  2x2 exact decoder can learn, within 256 full-model AdamW updates, to assign
  each of two otherwise-identical queries to the target associated with its
  own K=16 history; swapping those histories during training reverses that
  association. This tests local context-to-decoder wiring on one selected
  synthetic twin pair. It does not test generalization, policy quality,
  rollouts, planning, or global method optimality.
- **Fixed source and data boundary.** The initial checkpoint is the sealed E2
  EMA with SHA-256
  `c53bf0c42dc6c8f7945ff4d17bd6bd63a6db23e8b6e377b6b0b92903e66d694a`.
  Generate only the registered 256-pair E2 TRAINING-shape synthetic twin
  population on `UnseenSeed7x7`, seed `1000002`, meta-episode IDs `0..255`,
  K=16, UNKNOWN operator conditioning, and the v6 data contract. No public ARC
  data may be read. The diagnostic implementation, CUDA build, and exact
  binary hash must be reviewed, committed, pushed, and recorded before the
  registered execution.
- **Model-free row selection.** Scan those 256 twin pairs in ascending
  meta-episode ID and select the first pair that is not single-frame rule
  identifiable and the earliest position `p >= 16` where (a) the two targets
  differ, (b) the transition at `p` is outcome-changing, and (c) at least one
  earlier outcome-changing transition lies in `p-16..p-1`. Fail closed if no
  pair qualifies. Freeze and report the selected pair/row identities and
  cryptographic hashes. Before loading a model, require
  `state_differing_positions == 0`; corresponding current frames and actions
  to be bit-identical throughout both selected K=16 windows; and the two score
  rows to be bit-identical in every model-visible query input except context:
  current frame, action, coordinates, goal, content mask, and v6 UNKNOWN
  operator conditioning. Require their targets to differ and the
  target-disagreement mask to be nonempty. Thus the histories may differ only
  in rule-dependent outcomes. Any failed invariant is an integrity failure,
  not a negative model result.
- **Two arms and one causal difference.** Initialize both arms bit-identically
  from the fixed checkpoint, including identical fresh zero optimizer state.
  In the `correct` arm,
  each target row receives its own preceding K=16 history. In the `swapped`
  arm, the same two target rows in the same order receive one another's
  histories. Train all model parameters, including the context projector,
  with direct raw exact-decoder Unimix cross-entropy (`0.99` model
  probability plus `0.01/16`) reduced only over target-disagreement pixels.
  Use constant, unscheduled AdamW learning rate `1e-3`, beta1 `0.9`, beta2
  `0.999`, epsilon `1e-8`, weight decay `0`, global gradient clip `1.0`, F32
  computation, physical batch `2`, accumulation `1`, fixed row order, and the
  production 2x2 training recurrence with exactly zero latent/training noise.
  This deliberately removes every competing production objective; it is a
  wiring test, not a training-recipe candidate.
- **Budget, checkpoints, and stop rule.** Evaluate before training and after
  updates `8, 16, 32, 64, 128, 256`; the hard cap is 256 updates per arm.
  Record direct loss, update-1 context-parameter gradient norm, global
  pre-clip norm, and clip scale, and fail closed on zero/non-finite context
  gradient, any non-finite value, source/data/hash drift, CUDA failure, or a
  mixed-batch K0 leak. Early success requires both the wiring and promotion
  gates below at the earliest two consecutive evaluation checkpoints; a
  continuous-only wiring pass continues through update 256 and returns
  `wiring_only_no_promotion`. Otherwise run both arms through update 256. No
  fine-tuned checkpoint may be reused as held-out, E2, E3, or ARC evidence.
- **Frozen evaluation comparisons.** At every evaluation checkpoint, score
  both query directions with own K=16, paired K=16, and K0 context. Over only
  target-disagreement pixels report raw-softmax target NLL, the Unimix
  objective NLL, raw-softmax probability L1, latent/context-summary/copy-gate
  differences, and raw/composed argmax pixel counts. Define verdict statistic
  `D = raw_softmax_NLL(paired) - raw_softmax_NLL(own)`, first averaged within
  each query direction and then equally across the two directions. Probability
  L1 is analogously pooled across both directions within each arm. Also compare
  a K0 row inside `[K0, own-context carrier]` with the same row inside a
  shape-matched all-K0 batch; latent, raw probabilities, gate, both NLLs, and
  argmax must be bit-identical for the K0 row.
- **Preregistered verdict.** `wiring_pass` requires, at the same two
  consecutive evaluation checkpoints, `D_correct > 1e-4`,
  `D_swapped < -1e-4`, interaction
  `D_correct - D_swapped > 2e-4`, and, in **each** arm, either pooled
  own-versus-paired raw-softmax probability L1 `> 1e-6` or at least one raw
  argmax disagreement, with the exact mixed-K0 invariant passing. Failure by
  update 256 rejects local trainability under this intervention and budget. A
  continuous-only pass is not enough to scale training: the `promotion_pass`
  needed to authorize a fresh, small, multi-pair objective experiment
  additionally requires that the correct arm's own context have strictly more
  correct raw-logit argmax predictions than paired context and K0 over
  target-disagreement pixels in **each** query direction at those same two
  checkpoints. Composed exactness is reported but not gated because the copy
  gate is not a training target in this diagnostic.
- **Uncertainty and multiplicity.** This is one deterministic selected pair,
  so no confidence interval or population-level inference is reported. The
  complete checkpoint family is fixed to `0, 8, 16, 32, 64, 128, 256`; only
  the first two consecutive registered checkpoints satisfying the stated
  gates may produce early success, and no post-hoc checkpoint, direction,
  metric, seed, or threshold selection is allowed.
- **Execution evidence.** Use a never-reused run root with lifecycle state,
  exact Tofy and sibling revisions, locked build command/features, binary and
  checkpoint SHA-256, device identity, selected physical batch/accumulation,
  row/population hashes, configuration, per-checkpoint metrics, command log,
  and an external finalized-file manifest digest. A bounded CUDA launch
  preflight on the exact binary must pass first; that preflight cannot satisfy
  E2W. The registered E2W result itself remains `implementation_smoke`
  evidence and can authorize only the explicitly stated small multi-pair
  follow-up, never a model promotion or E3.

## Corrections from the 2026-09-04 independent audit (thread 16c2f6f6)

1. **E2 ran at effective batch 128, not 256.** `apply_foundation_v2_recipe`
   overwrote `--grad-accum 2` with 1 (fixed in the source after the run; the
   run's `config.json` records `grad_accum: 1`). Every "128 x 2" timing in this
   file was really 128 x 1. E2 (all attempts) is reclassified as
   **implementation smoke / exploratory**; it cannot satisfy §5.1 as registered.
   A registered E2 must be rerun in a never-reused root from a reviewed commit
   with the exact twin-pair K = 0 vs K = 16 evaluator (in progress).
2. **Depth arithmetic was wrong** (see ADR 0005 §3.5 corrected table):
   2x2 = 6 blocks / receptive field 25; 3x3 = 12 blocks / 49; compute roughly
   doubles. The depth-ablation registration below is re-stated with these
   numbers and with matched, actually-applied accumulation.
3. **E1's decision rule violated the project's held-out policy**: public games
   may not tune thresholds or select policy
   (`docs/specs/P2_ARC_AGI_3_WORLD_MODEL_CORE_REDESIGN.md`). The E1 rule
   "switch Phase A trust on residual AUROC" is withdrawn; E1 stands as an
   evaluation-only observation. Phase A calibration is to be fitted from
   synthetic held-out data (`p2-eval --emit-phase-a-calibration`, in progress).
4. The post-E2 chain launched E3 unconditionally; replaced by a gated chain
   that runs E3 only on a PASS verdict, and E3 is deferred until its redesign
   (longer evaluation histories so the production warm-up is reachable) lands.
5. E3 t = 32 and the warm-up vacuity are being fixed by giving evaluation
   populations longer levels (training rows unchanged).
