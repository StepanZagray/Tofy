# Tofy P2: VRAM, rollout stability, recursion, and ARC-AGI-3 readiness

**Date:** 2026-08-05
**Scope:** current dirty `main` worktree, the v17 checkpoint at optimizer step 16,300,
and primary-source literature available on this date.
**Decision:** do **not** start Stage 3 yet.

## Executive decision

The current model is not blocked by a conventional per-step VRAM leak. A 256-sample,
fixed maximum-depth CUDA run warmed to 3,620 MiB, stayed flat for roughly 40 seconds
and more than 200 optimizer updates, and released memory at shutdown. The growth visible in
`nvidia-smi` is consistent with CUDA's stream-ordered allocation pool reaching a
high-water mark. Activations from explicit backpropagation through the recursively
unrolled F32 model are the real cost.

`256x2` is not currently possible: `TrainConfig::validate` rejects every
`grad_accum != 1`. It is nevertheless the right first engineering target for the
8 GiB RTX 5060 because the measured maximum-depth 256 batch has ample headroom. The
implementation must define accumulation semantics for batch-dependent SIGReg and
quantile Q targets; merely summing two gradients would silently change the objective.

The more serious blocker is model quality. On the step-16,300 v17 checkpoint,
open-loop dynamics MSE averages 0.448 at horizon 4 and 1.219 at horizon 8, while
closed-loop horizon-8 MSE averages 0.076 across four seeds. Open-loop @8 has a 2.4%
sample coefficient of variation; the within-seed open/closed ratio is 13.5--17.2x.
The evidence therefore supports **bad compounding**, not a claim that this particular
mean is wildly seed-flaky. The evaluator is still inadequate because it
reports only a mean and count: no per-family result, median, tail, confidence interval,
finite fraction, or normalized error.

Recursion is already implemented as within-transition refinement: one action remains
fixed during the outer loop, and environment time advances only on the next model
call. What is not justified is making that recursive operator the default transition.
The partial v17
matched-depth and PTRM ablations are not valid tests of the trained model: the
checkpoint has warm-start enabled, but those two evaluation paths cold-start `y`.
Their mixed depth and chance-ranking results are out-of-distribution diagnostics, not
evidence for or against trained recursion. PTRM therefore remains unproven.
Shared-weight refinement should remain an optional computation budget, gated by a
matched-compute ablation and stable multi-step behavior.

The recommended architecture is a hybrid:

1. a stable residual, goal-free, one-action transition model;
2. a direct action-prefix predictor for horizons 1/2/4/8/16 to avoid feeding predicted
   latents back into themselves during planning;
3. a calibrated ensemble for epistemic uncertainty, rather than noise in one model;
4. optional recurrent refinement with learned or residual-norm halting only if it
   beats a non-recursive matched-compute baseline;
5. an online symbolic/program hypothesis layer and episodic memory for goal discovery.

No technical plan can guarantee 100% on ARC-AGI-3. Official scoring requires every
level of every hidden environment to be completed at approximately human action
efficiency, and current leading systems use large local VLM/code-agent harnesses.
The plan below is the shortest defensible route to a competitive system and uses
frozen promotion gates so failed ideas are killed early.

## Evidence gathered

### Repository state and checkpoint

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB.
- Model: 400,293 F32 parameters, hidden 128, latent `128x8x8`, inner depth 2,
  configured outer depth 8, randomized depth, residual-y, warm start, spatial SIGReg.
- The step-16,300 checkpoint completed 8,192 dynamics updates and 8,108 exploration
  updates at physical batch 512, accumulation 1. It is 84 updates before the
  sequential lesson. Under phased training, dynamics and exploration give zero weight
  to rollout, event, Q, and PTRM-ranking objectives. Its poor rollout/Q/PTRM metrics
  prove current non-readiness, but are not a final test after the scheduled training.
- `docs/RESULTS_P2.md` still presents v11 as best and v12 as in progress; it does not
  record the v17 partial run or frozen Stage-1 promotion thresholds. This reporting
  debt must be resolved before another expensive run so checkpoint selection cannot
  move after results are seen.
- Parameters and Adam moments occupy only about 1.6 MB and 3.2 MB on disk. Model state
  is not the VRAM problem; frames, recursive activations, their autograd graph, and CUDA
  workspaces dominate.
- The trainer intentionally stores every outer-step output for deep supervision and
  backpropagates through the explicit recursion. Everything is F32.
- Consecutive training batches are sliding episode windows. The existing profiling
  test notes that almost all episodes in a large batch are reused on the next update.
  Large physical batch therefore does not imply comparably large new-data diversity.
- The CLI/documentation says randomized outer depth covers `1..=configured`, but the
  implementation samples `2..=configured`. Depth `D=1` is therefore absent from randomized
  training and must be tested explicitly rather than assumed to be in-distribution.
- `src/p2/rhae.rs::game_score` omits the official per-game completion cap. Because a
  completed level may score up to 1.15, an agent that misses later levels can currently
  receive a recomputed game score above the maximum allowed by its completion count.
  The existing incomplete-game test uses only 1.0 level scores and does not catch this.
- The warm-start contract is inconsistent across inference paths. Normal `forward`
  initializes `y` from the encoded state when configured; `forward_with_outer_steps`
  and `forward_ptrm_with_depth` rebuild `x` but pass `y_init=None`. Consequently v17's
  matched-compute and PTRM reports do not evaluate the same transition initialization
  that was trained and used by the normal forward path.
- The current PTRM ranking objective cannot train the Q ranker. It converts Q logits
  into host-side best indices, then minimizes latent MSE for those selected paths; no
  differentiable Q-logit term remains. A good oracle-rank score cannot emerge from
  that loss without an explicit detached-label ranking/classification objective.
- Current stochastic evaluation is not sample-stable: noise seeds depend on physical
  batch index and tensors are filled in batch order. Changing evaluation batch size
  changes a sample's noise despite an unchanged evaluation seed.
- The identifiability probe fits an in-sample 8,192-to-16 regression with fewer than
  1,800 examples and weak ridge regularization. Its increment-cosine metric also zips
  the 8,192-vector latent delta with a 16-vector oracle delta, silently using only the
  first 16 latent entries. Neither result is currently suitable as a promotion gate.

### VRAM feedback loop

Red-capable command (current code, isolated temporary output):

```bash
target/release/tofy p2-train --device cuda --lessons dynamics \
  --physical-batch 1024 --grad-accum 1 --steps-per-lesson 16 \
  --max-steps-this-run 16 --checkpoint-every-steps 0 \
  --hidden-dim 128 --action-dim 32 --outer-steps 8 --inner-steps 2 \
  --randomize-depth --residual-y-update --warm-start-y \
  --sigreg-spatial --sigreg-projections 32 --output-dir /tmp/<fresh-dir>
```

It reproduced `CUDA_ERROR_OUT_OF_MEMORY` in 2.45 seconds. Short randomized-depth
probes at 512/640/768 passed, demonstrating why one-step or low-depth capacity tests
are invalid. Fixed-depth sustained probes found batch 512 viable for 64 updates;
larger probes were inconsistent near the memory ceiling, and must not be called stable.
The real 16,300-update checkpoint is the strongest evidence for 512x1.

Leak-discriminating command: run 256 updates at fixed `outer_steps=8`, sample the
process with `nvidia-smi` every 250 ms. Five-second max windows were:

| elapsed | process VRAM max |
|---:|---:|
| 0--5 s | 2,980 MiB |
| 5--10 s | 3,300 MiB |
| 10--15 s | 3,588 MiB |
| 15--20 s | 3,620 MiB |
| 20--55 s | 3,620 MiB |

This falsifies monotonic live-tensor retention in the dynamics update. Cudarc 0.19.8
uses `malloc_async`/`free_async` when supported, so freed blocks can remain reserved by
the CUDA pool; `nvidia-smi` cannot distinguish reserved from live tensor bytes.

A fixed-depth throughput proxy also supports 256x2. Excluding the first warm-up
window, batch 512 averaged 472 ms per update; batch 256 averaged 219 ms per update,
or 438 ms for two current updates over 512 samples. True accumulation will perform
only one optimizer step, so it should be near this cost or slightly lower. This is a
capacity/throughput result, not an equivalence result: the current two 256 updates have
different optimizer and loss semantics from the proposed accumulated update.

### Rollout feedback loop

Checkpoint: `p2-output-v17-stable/checkpoints/step-000000016300/model.safetensors`.

| split / seed | episodes | one-step | open @4 | open @8 | closed @8 |
|---|---:|---:|---:|---:|---:|
| dynamics / 2 | 64 | 0.2533 | 0.4268 | 1.1874 | 0.0696 |
| dynamics / 3 | 16 | 0.2531 | 0.4334 | 1.2016 | 0.0697 |
| dynamics / 4 | 16 | 0.2324 | 0.4779 | 1.2515 | 0.0735 |
| dynamics / 5 | 16 | 0.2411 | 0.4526 | 1.2345 | 0.0917 |
| planner / 2 | 64 | 0.0525 | 0.3970 | 1.0712 | 0.0610 |
| planner / 3 | 16 | not separated | not recorded here | 1.0531 | not recorded here |

The absolute latent MSE is not safely comparable across old checkpoint families
because their encoders and normalization changed. Within one frozen checkpoint, the
open/closed ratio and horizon growth are meaningful. The current report needs to add
normalized error relative to copy-forward and target-pair variance.

Historical completed/late checkpoints show that temporal compounding is not unique to
the partial v17 curriculum. The numbers are not comparable *between* rows because the
latent representation changed, but the horizon growth *within* each row is diagnostic:

| checkpoint | one-step | open @4 | open @8 | closed @8 |
|---|---:|---:|---:|---:|
| v11 control | 0.0258 | 0.0843 | 0.1725 | unavailable |
| v14 | 0.0679 | 2.19e3 | 4.03e11 | unavailable |
| v15 final | 0.8035 | 1.16e3 | 1.50e11 | 2.7420 |
| v15 step 28,700 | 0.0530 | 3.49e3 | 3.74e11 | 0.0465 |
| v17 step 16,300, seed 2 | 0.2533 | 0.4268 | 1.1874 | 0.0696 |

v17's normalization/residual changes eliminated the catastrophic `1e11` scale seen
in v14/v15, but did not close the large open/closed gap. V11 is the strongest old
one-step/open-loop reference; its cold-start matched-depth result also rose from
0.0258 at depth 2 to 1.596 at depth 4, 3.140 at depth 8, and 1.68e6 at depth 16.

Other step-16,300 warnings (all provisional because Q/PTRM/rollout training has not
started and the PTRM path incorrectly cold-starts `y`):

- dynamics Q balanced accuracy is 0.265; planner Q balanced accuracy is 0.491;
- dynamics Q confident-error rate is 0.470;
- `T=2` PTRM pass rate is lower than deterministic `T=1`;
- `T=2` oracle-rank accuracy is 0.509 on dynamics and 0.530 on planner, close to chance;
- the latent-to-oracle linear probe reports a strongly negative R2, but the probe is
  underdetermined and its separate increment-cosine implementation is dimensionally
  invalid, so neither should influence an architecture decision yet.

These are sufficient to block Stage 3 at this checkpoint. They are not sufficient to
reject the scheduled sequential/Q lessons; evaluate again after fixing warm-start and
finishing the curriculum.

## Root causes

### VRAM

1. **Explicit recursive autograd is activation-bound.** Each outer step performs
   several convolutional block applications; all intermediates needed by backward
   survive until `total.backward()`.
2. **F32 everywhere.** Frames are expanded into two `B x 16 x 64 x 64` F32 one-hot
   tensors, and model activations/parameters are F32.
3. **The public forward result is too broad.** `run_recursion` builds event and Q heads
   at every outer step and retains `StepOutput` objects, although the training loss
   uses only every step's `y` and the final heads.
4. **CUDA pool high-water behavior obscures liveness.** Process VRAM can remain high
   after tensors are freed.
5. **No gradient accumulation exists.** The CLI records the value but validation
   explicitly rejects it.

### Temporal rollout

1. **The within-transition recursive operator is rollout-unstable.** The code correctly
   holds one action fixed during refinement and advances time only on the next call,
   but open-loop planning repeatedly composes the resulting one-action operator.
2. **Training distribution mismatch remains.** The dynamics/exploration phases are
   dominated by one-step real latents. Multi-step rollout loss exists only in selected
   lessons and uses scheduled resets when error is already large. Evaluation is fully
   open-loop.
3. **The state is absolute and high-dimensional.** RMS normalization controls global
   scale but does not make the action-conditioned operator contractive or preserve a
   meaningful small delta.
4. **One shared deterministic model does not express epistemic uncertainty.** Adding
   Gaussian noise to its internal `z` explores off-manifold states; it is not a
   posterior over plausible environment rules.
5. **PTRM trajectory sets are not nested.** `T=1` is deterministic, whereas `T>1`
   makes every trajectory noisy. `pass@T` therefore need not be monotone and is
   misleadingly named when compared with `T=1`.
6. **Q has incompatible jobs and lacks differentiable ranking supervision.** It is
   used as an absolute correctness classifier, a trajectory ranker, and an early-stop
   proxy. The evaluated checkpoint uses absolute targets (`q_quantile_targets=false`),
   so quantile semantics did not cause its result. If retained, batch-quantile targets
   can produce a ranker but not an absolute calibrated probability of reliability.
   The current PTRM ranking loss itself supplies no gradient to Q.
7. **The evaluation statistic hides the failure shape.** A mean over a small number of
   episode endpoints cannot reveal heavy tails, family failures, or non-finite growth.
8. **The official-score recomputation is optimistic in one edge case.** The weighted
   score must be capped by the weighted fraction of completed levels; the local RHAE
   helper currently returns only the weighted score.
9. **Inference adapters violate the model's initialization contract.** Cold-start
   matched-depth/PTRM calls are being compared with warm-start normal inference.
10. **Evaluation randomness depends on batch layout.** A fixed seed does not identify
    a fixed trajectory unless physical batch and ordering are also fixed.

### Training-data efficiency

The cache faithfully memoizes a sliding sampling schedule in which optimizer update
N starts at episode N. That schedule—not memoization itself—makes adjacent updates
nearly identical. At batch 512 or 1,024 only a small fraction of the episode window
changes each step. This reduces effective data diversity, makes an apparently huge
batch misleading, and can amplify curriculum/local-minimum effects. Fixing this is
higher priority than forcing the physical batch back to 1,024.

## What the literature does and does not justify

- [TRM](https://arxiv.org/abs/2510.04871) is strong evidence that weight-tied latent
  refinement can be parameter-efficient on static, supervised puzzles. It is not
  evidence that repeatedly applying the same learned transition is stable over an
  interactive time horizon.
- [PTRM](https://arxiv.org/abs/2605.19943) reports large gains from noisy parallel
  trajectories on Sudoku/Pencil Puzzle Bench. Tofy's Q is uncalibrated, its noise is
  not trained in the evaluated stable run, and its target is a next-state latent, so
  transfer of that result is a hypothesis, not a justification.
- [ACT](https://arxiv.org/abs/1603.08983),
  [Universal Transformers](https://arxiv.org/abs/1807.03819), and
  [PonderNet](https://arxiv.org/abs/2107.05407) show that input-dependent computation
  can help algorithmic tasks. None makes a non-contractive transition safe. Halting
  must be trained and calibrated, with a compute penalty and a hard maximum.
- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) and
  [Jacobian-regularized DEQs](https://arxiv.org/abs/2106.14342) show a route to
  constant-memory implicit depth and explicitly warn that fixed-point models are
  brittle without stability control. A DEQ rewrite is a later option, not the first
  fix: Tofy has not yet shown that equilibrium depth improves its task metric.
- [LeWorldModel](https://arxiv.org/abs/2603.19312) supports a compact goal-free JEPA
  with next-embedding prediction and Gaussian regularization. Tofy has added recursion,
  auxiliary heads, spatial SIGReg, depth randomization, rollout resets, and PTRM; the
  paper does not validate that combined stack.
- [Fast LeWorldModel](https://arxiv.org/abs/2606.26217) directly addresses Tofy's
  failure: repeated local latent rollout accumulates errors; action-prefix prediction
  supervises multiple horizons directly and reports slower open-loop error growth.
- [Plan2Explore](https://arxiv.org/abs/2005.05960) supports using ensemble disagreement
  to seek informative states. This is a better epistemic signal for falsification than
  noise in one model.
- Official [ARC-AGI-3 methodology](https://docs.arcprize.org/methodology) squares
  action inefficiency and requires all levels for a 100% cap. The official
  [Milestone 1 report](https://arcprize.org/blog/arc-prize-2026-milestone-1) shows that
  the leading approaches used a local code-writing VLM or vision-LLM policies with
  reflection/memory. This is evidence against expecting the current 400k-parameter
  synthetic world model alone to acquire arbitrary hidden game semantics.

## Recommended target design

### 1. Transition module: one action, one transition

Make `TransitionModel` a deep module with one inference seam:

```text
predict(belief_state, action_or_prefix, compute_budget) -> TransitionBelief
```

The interface returns predicted latent(s), event probabilities, uncertainty, and
actual compute used. It hides the residual transition, optional refinement, prefix
head, and ensemble adapters. Callers must not choose raw inner/outer loops.

The single-action implementation should predict a gated delta:

```text
z_next = normalize(z_now + gate(z_now, action) * delta(z_now, action))
```

No-op/action-effect contrastive losses must prevent both copy-forward collapse and
unnecessary global movement. Dynamics remain goal-free.

### 2. Prefix predictor for planning

Add an action-prefix encoder that predicts `z[t+h]` from the latest real belief state
for `h in {1,2,4,8,16}`. Train every prefix against an encoded real future. Add a
consistency loss between direct prefix predictions and composed one-step predictions,
but do not require the planner to use the unstable composed path.

This separates:

- **temporal depth:** number of environment actions in a candidate sequence;
- **computation depth:** optional refinement spent on one prediction.

### 3. Uncertainty and Q

Use 3--5 bootstrap transition heads (shared encoder/trunk, independent residual/output
heads and bootstrap masks) to estimate epistemic disagreement. Predict an error scale
or ordinal error bins for aleatoric/reliability calibration. Keep two named outputs:

- `reliability`: calibrated probability/error interval used for risk and halting;
- `rank_score`: relative trajectory ordering used within a candidate set.

Calibrate on a frozen held-out split and report ECE, Brier, AUROC, risk-coverage, and
error quantiles. Do not use batch-median Q labels as an absolute probability.

PTRM, if retained, must include the deterministic trajectory in every K set; K adds
`K-1` stochastic candidates. Report paired gain over that same deterministic member.

### 4. Recursion policy

Keep shared-weight recursion only behind `compute_budget` and test three adapters:

1. fixed `D=1` residual baseline;
2. fixed `D in {2,4,8}` at matched parameter/compute budgets;
3. adaptive `D` with hard max 8 and a ponder cost.

First try deterministic halting from a normalized residual criterion plus calibrated
reliability. Only add a learned PonderNet-style halt head if the deterministic rule
leaves material accuracy/compute on the table. Add a Jacobian-vector contraction probe
as a diagnostic; promotion depends on bounded multi-step amplification and downstream
rollout/planning, not on requiring every local Jacobian norm to be below one.

### 5. ARC-AGI-3 agent layer

A path aimed at 100% needs more than a latent predictor:

- deterministic palette/object/connected-component/diff features alongside neural
  grid features;
- episodic memory over **real** `(observation, action, effect, event)` tuples;
- candidate mechanics and goal programs generated by a capable offline local VLM/code
  model or a program synthesizer;
- Bayesian/score-based belief over those programs;
- safe information-gain actions while beliefs are ambiguous;
- prefix-model MPC/beam search after one hypothesis is clearly ahead;
- immediate re-scoring and retargeting after every real observation;
- reusable within-game skills and level-to-level memory, with no hidden-goal or oracle
  state leakage.

The neural world model is the fast predictive/risk module in this system, not the sole
source of semantic hypotheses.

## Code plan

### Phase A -- make evidence trustworthy (must land first)

1. `src/p2/eval.rs`
   - Replace `RolloutMetrics` mean-only fields with per-horizon distributions: n,
     finite n, mean, median, trimmed mean, p90/p95, bootstrap 95% CI, max, and
     normalized error.
   - Add an explicit transition index to generated samples (the current
     `TransitionSample` has only episode identity), then group by curriculum family
     and seed and sort each episode by that index; preserve macro and micro aggregates.
   - Add copy-forward and closed-loop baselines and report open/closed ratios.
   - Add an `--eval-mode rollout-only` path so multi-seed diagnosis takes seconds/minutes
     rather than recomputing PTRM and matched depth.
   - Seed each stochastic trajectory from stable sample identity, trajectory index,
     and evaluation seed. Add a batch-size invariance test.
   - Replace the identifiability probe with a train/validation split, dimensionally
     valid ridge/CCA bridge selected on training data, and held-out R2. Compare deltas
     only after projecting both spaces into the same learned dimension.
2. `src/p2/model.rs` / `src/p2/eval.rs`
   - Add per-recursion-step residual norm, latent norm, and local amplification probes.
   - Centralize transition preparation (`x`, `goal_h`, and optional warm-start `y`) and
     make normal, matched-depth, latent, and PTRM paths use it. Add warm-start tests
     proving configured-depth matched compute equals normal deterministic inference and
     PTRM `T=1,sigma=0` equals that same output.
3. `src/p2/train.rs`
   - Replace host-index-only PTRM selection with detached oracle-best trajectory labels
     and an explicit differentiable cross-entropy, pairwise, or listwise loss on Q
     logits. Unit-test a non-zero Q-head gradient and zero unintended label gradient.
   - Encode the current frame once per update and reuse it for transition and LeWorld
     losses. Choose and document one target policy: gradient-carrying online current
     encoding plus detached/EMA next target is the first stability experiment; compare
     with the current live online target as a one-change ablation.
4. `tests/p2_vram_probe.rs` (or a dedicated benchmark binary)
   - Record CUDA process/pool memory after stage, forward, backward, optimizer, and
     post-drop; warm up before asserting a slope.
5. Run a pilot to estimate between-seed variance, then pre-register the minimum effect
   and seed count needed for 80% power at two-sided alpha 0.05 (five is a floor, not an
   automatic claim of adequate power). Store raw episode-level metrics as JSONL.
6. `src/p2/rhae.rs`
   - Apply `min(completed_level_weight / all_level_weight, weighted_score)` and add a
     regression test with four completed 1.15-scored levels out of five.

### Phase B -- memory and data pipeline

1. `src/p2/train.rs`, `src/p2/cli.rs`
   - Implement real gradient accumulation. Define `physical_batch` as microbatch and
     `effective_batch = physical_batch * grad_accum` in the resume contract/report.
   - Average microbatch gradients, clip once after accumulation, and call AdamW once.
   - Make optimizer step, curriculum cursor, checkpoint cursor, depth/noise seeds, and
     loss schedules advance once per effective batch.
   - Define batch-local SIGReg and quantile-target semantics explicitly. For spatial
     SIGReg, use each 256 microbatch's `B*H*W` population; do not claim bit-equivalence
     with batch 512. Replace quantile Q for calibrated reliability in Phase D.
2. In a separate change and experiment, replace the sliding-window sampler with a
   deterministic counter-based shuffled
   episode schedule. Two microbatches in one update must be disjoint; adjacent updates
   must not be 99% identical. Resume must reproduce exact episode IDs.
3. Split training and inference recursion results. Retain per-step `y` only when deep
   supervision is requested; compute event/Q heads only for the final step.
4. Add BF16 autocast for convolutional activations/weights only after accumulation is
   correct. Keep normalization, SIGReg, losses, gradient norm, and optimizer master
   state in F32. Compare numerics and throughput; do not assume BF16 is free.

Phase-B gate on this 8 GiB GPU: 256x2 must complete a 1,000-update maximum-depth stress
run with at least 15% free VRAM after warm-up, no more than 32 MiB growth between the
first and last quartiles of the post-warm-up trace, finite losses/gradients, exact
resume equivalence, and the intended one-Adam-step-per-512-samples schedule. Compare
512x1 and 256x2 first, changing accumulation only; then compare the old and shuffled
samplers from the same initialization. For the final configuration, test physical
batch sizes that exactly preserve the chosen effective batch and select the largest
stable physical batch with minimum accumulation, as required by `AGENTS.md`. Treat
larger effective batches such as 320x2 or 384x2 as separate optimizer-schedule studies,
not memory substitutes for 512.

### Phase C -- stop temporal compounding

1. Establish a `D=1` residual-delta baseline; remove PTRM, randomized depth, and learned Q
   from the causal experiment.
2. Add multi-horizon prefix prediction with geometrically balanced horizon weights.
3. Train on model-visited latents as well as real encoder latents, but keep direct
   prefix targets anchored to real future encodings.
4. Add no-op identity, action-effect, and inverse-action contrastive probes.
5. Run one-difference ablations across at least five seeds.

Phase-C promotion gate: one-step normalized error may regress no more than 5%;
open-loop @8 must improve at least 30% and be no more than 2x closed-loop @8; every
@16 sample must be finite; p95/median normalized error must be <=5; and the powered,
seed-stratified 95% bootstrap CI of @8 improvement must exclude zero. Define the
action-heavy retrieval metric before training and allow no more than a 2 percentage
point absolute success regression.

### Phase D -- uncertainty, then recursion

1. Add bootstrap ensemble and calibrated reliability/error prediction.
2. Verify uncertainty on held-out mechanics and deliberate corruptions.
3. Run the fixed-depth/matched-compute/adaptive-depth matrix from the **same** checkpoint.
   Denote recursion depth by `D` and PTRM trajectory count by `T`; include `D=1`
   explicitly because current randomized training never samples outer depth 1.
   Do not run this credit assignment until all adapters share warm-start semantics and
   the relevant scheduled lessons have completed.
4. Retain recursion only if the powered seed-stratified interval beats `D=1` by at
   least 5% relative normalized rollout error or 3 percentage points of downstream
   planning success, and the lower 95% bound remains positive after accounting for
   latency and memory. Do not reject solely because a one-step local Jacobian norm is
   above one; no-op and reversible dynamics can legitimately reach that. Reject on
   unbounded multi-step amplification or failed rollout/planning gates.
5. Test PTRM only after reliability ranking beats random and `T` sets are nested.

Suggested gates: ECE <= 0.05, reliability AUROC >= 0.85, risk at each successive 20%
coverage bucket no lower than the previous bucket (with isotonic confidence bands),
Q-rank accuracy lower 95% bound above `1/T + 0.05`, paired pass@T non-decreasing with
the deterministic candidate nested in every set, and adaptive depth using at least
15% less mean compute than the best fixed-depth setting at no worse than a 1% relative
accuracy margin.

### Phase E -- revised Stage 3

Implement the hybrid hypothesis/planning layer only after Phases A--D pass. Compare:

- random/legal and state-graph baselines;
- always-sequential pursuit;
- symbolic/program hypotheses without neural planning;
- falsification-first plus prefix-model planning;
- oracle-transition and oracle-goal upper bounds.

Report completion, RHAE-compatible action counts, irreversible failures, retargets,
model calls, latency, memory, hypothesis entropy, and calibration across at least five
seeds and structurally held-out game generators. Require falsification-first to improve
the lower confidence bound without reducing ordinary-task completion.

## Experiment order and stop rules

| Order | Experiment | Credit if | Stop if |
|---:|---|---|---|
| 0 | evaluator + memory instrumentation | repeated runs produce CIs and post-warm-up slope | raw episode data cannot be reproduced |
| 1 | 256x2 accumulation, old sampler | stable, resumable, same optimizer schedule | accumulation semantics are not documented/tested |
| 2 | shuffled sampler, fixed 256x2 | stable and less adjacent overlap | resume/sample identities differ unexpectedly |
| 3 | D=1 residual baseline | establishes strong one/open-loop reference | worse than current closed-loop/action-effect gates |
| 4 | prefix horizons | @8/@16 and planning success improve | only latent MSE improves, downstream planning does not |
| 5 | ensemble reliability | calibrated risk and useful information gain | disagreement is uncorrelated with error |
| 6 | fixed/matched/adaptive recursion | powered latency-adjusted gain passes the numerical gate | multi-step amplification or downstream gate fails |
| 7 | nested PTRM | paired pass@T and rank improve | T=1 remains best or Q remains chance |
| 8 | hybrid Stage 3 | hard-retarget efficiency improves | world-model errors dominate oracle gap |

## Final answer to the recursion question

Recursion is scientifically plausible and worth one controlled ablation, but it is not
currently justified in Tofy. TRM/PTRM solve iterative answer-refinement problems; Tofy
needs stable action-conditioned temporal prediction under partial knowledge. Keep at
most a bounded, optional refinement loop inside one transition. Use direct prefix
prediction for horizon, an ensemble for uncertainty, and remove recursion from the
default if the frozen matched-compute gate does not show a reproducible downstream
benefit.

The highest-priority changes are therefore: trustworthy rollout distributions,
real 256x2 accumulation, shuffled episode sampling, a `D=1` residual baseline, and a
multi-horizon action-prefix predictor. Dynamic depth comes later; Stage 3 comes after
all of them.
