# Looped control: first screen registration

Post-run correction (September 7, 2026): the committed model actually used RMS
normalization with a clamped second moment and no learned gain. The LayerNorm
wording below, including the final paragraph's denial of RMS, was an integration
review/documentation error. Preserve that original registration text and the
negative result; this is a named recipe deviation, not a retroactive clean match.
See `2026-09-07-looped-agent-frozen-diagnostic.md` for the next bounded check.

Registered before CUDA execution or training outcomes. This is a bounded empirical
prerequisite screen, not an ARC performance claim or a proof of optimal design.

## Claim and comparisons

Can a from-scratch shared-depth transformer infer an action permutation from
three observed transitions and select the right action on a new synthetic maze?
The decisive first-use test is a paired one-step goal, isolating rule use from
multi-step navigation. All balanced rules occur with identical query pixels.
Without support information, a deterministic action has exactly 25% accuracy.
Three distinct, unblocked calibration movements identify a four-way bijection;
this information result does not imply successful neural learning.

The generator is `permuted-controls-maze-v1`, with 8x8 logical cells rendered as
64x64 palette pixels, 0.18 interior wall probability, visible goal, four legal
actions, and discount 0.95. Hold out rule IDs [0,5,7,9,14,16,18,23]; train on the
other sixteen. Each action/direction pair occurs twice/four times respectively.
Calibration starts, probe order, and omitted action are randomized with a separate
RNG stream from query layout and difficulty. It is a chronological prefix, then a
new maze under the same rule. Every three-step calibration is informative and
scripted; this does not test autonomous exploration or incomplete information.

No public ARC source, levels, trajectories, pretrained weights, or LLM teacher
labels enter training. Oracle policy labels derive only from support, visible
query, and the synthetic specification. Counterfactual transition labels derive
from cloned simulator states; branches never become factual support.

## Fixed recipe

- Model: width 128, 4 attention heads, 2 pre-LayerNorm transformer blocks shared
  over reasoning loops; 8-dimensional per-pixel palette embeddings concatenated
  within each 8x8 patch. Seven frames give 448 spatial/context tokens, plus a
  learned readout token. Distinct policy, value, reward, and next-pixel heads.
- Initialization seed 0; generator seed 9173; named deterministic initialization.
  Separate generator tags for training, prediction evaluation, one-step rule
  probes, and closed-loop evaluation. Store initial weights and input-stream hash.
- AdamW, learning rate 0.0003, weight decay 0.01, global gradient norm clip 1.0;
  default Adam beta/epsilon values from pinned Candle 0.11.0. No dropout or schedule.
- Effective batch 64. Physical batch is selected below; partial last microbatches
  preserve exactly 64 examples and sample-weighted gradients per optimizer update.
- Train 256 updates (16,384 fresh queries), alternating query distances 1 and
  2..10. Loop schedule 1,2,4 repeated by update. All losses act on the final loop.
- Objectives: soft optimal-action CE 1.0, discounted-success sigmoid MSE 0.1,
  immediate-success BCE 0.1, categorical dynamics 0.5. Dynamics loss assigns equal
  mass to changed and unchanged pixels per example over all four action targets.
  Reward and task value have different labels. Value weighting remains an untested
  choice; a failed search comparison cannot establish search's general merit.
- Final checkpoint only, no best-checkpoint selection. Training screen wall-clock
  cap 30 minutes including evaluation; launch only if measured stage timings leave
  at least 5 minutes for evaluation/integrity inside that cap. Otherwise shorten
  the separately labeled implementation smoke and revise this registration before
  evidence training. No automatic partial-run reuse or optimizer-less resume.

## Capacity and implementation gates

Use reviewed, pushed clean source and clean pinned sibling, `--features cudnn`,
recorded build command and binary SHA-256. Copy the binary to a unique launch path.
CPU tests must establish exact simulator targets, balanced splits, informative
randomized calibration, pixel preservation, shared parameter identity, nonzero
gradients, reproducible initialization/frozen inference, and a loss that rejects
copying changed outcomes. Clippy and formatting must pass.

Capacity uses the exact binary: two worst-training-depth optimizer updates per
fresh smoke root, starting physical batch 8 and doubling toward 64 until failure,
then integer bisection of the feasible interval. Each probe uses effective batch
equal to physical batch and at most five minutes. Select the largest successful
batch with at least 512 MiB observed device memory headroom; repeat it once.
If batch 8 fails, bisect 1..7. No more than twelve capacity probes. Record peak
memory, stage durations, device identity and errors. These are implementation
smokes, never promotion evidence. The selected physical batch/accumulation pair
must be recorded externally before the 256-update screen begins.

## Frozen evaluation and decisions

- Prediction population: 32 fresh rows per split, half one-step and half distances
  2..10. Report exact next frames, copy exact frames, changed-pixel accuracy,
  optimal-action accuracy, reward positives/TP/FP and value MSE against the fixed
  constant 0.9 predictor. A low aggregate pixel loss is not a dynamics success.
- Paired rule probe: 32 layouts times all rules in each split. Compare true support,
  cleared support, and valid support from the direction-shifted rule. Every wrong
  support changes the unique correct action while query pixels remain identical;
  log actual eligible/changed/outcome-changing counts and whether choices follow
  the presented rule. Cleared support is an OOD diagnostic, not sole proof of use.
- Closed loop: 32 seed-paired fresh episodes per split, distances 2..10, cap 32 real
  query actions. Report direct policy at 1/2/4/8 loops (8 is untrained extrapolation),
  cleared context, random, observation-limited oracle and optional learned search.
  Calibration costs three additional real actions. Report wins, action counts and
  success-weighted optimal-action efficiency. These are synthetic metrics, not the
  official ARC score. Oracle must solve all full-length episodes.
- Search: four actions at each of two imagined depths, 20 extra model nodes per
  action, predicted categorical states only, learned immediate success and value.
  It receives more compute than direct inference. This screen cannot establish
  superiority under equal compute; record its actual cost and all negative results.
  Terminal states are targets but not training inputs, limiting imagined leaves.
- Uncertainty: Wilson intervals describe absolute episode win rates. Compare arms
  with seed-paired episode bootstrap; cluster rule-probe bootstrap by layout, not
  individual rule row. Use 10,000 resamples, analysis seed 1907. No promotion from
  single-seed screen, no significance claim from choosing a favorable depth.
- Continue to fresh confirmation only if held-out true-support one-step accuracy
  is at least 75%, exceeds wrong-support accuracy under the real rule by at least
  25 points, and matches the rule implied by wrong support at least 75% of the time.
  Also require held-out direct-four-loop win rate at least 25% and changed-pixel
  accuracy at least 80%, with integrity/oracle controls passing. These are screening
  thresholds, not evidence that the full ARC task is ready.
- If any gate fails, analyze grounding, rule use, value and control separately.
  Test the cheapest suspected failure before scaling data, model size or runtime.
  Do not auto-sequence a new intervention from a scalar score alone.
- Confirmation, if justified, requires a new registration: at least seeds 1,2,3,
  200 held-out episodes per arm, paired uncertainty and a declared multiplicity
  policy. Untied/parameter-matched training controls and search distillation are
  deferred. Broader mechanics and uncertain calibration precede hidden objectives
  and the later frozen public ARC evaluation.

## Independent advisor disposition

Claude Opus 5 accepted xHigh review on September 7, 2026. Its detected parity
confound and limited calibration diversity were corrected before any CUDA run.
Valid mismatched-support probes, a constant-value comparator and separate maximum
inference depth were added. Its suggestions to change discount/value weighting
are hypotheses, not measured necessities; keep the initial recipe and diagnose
value quality first. Larger confirmatory samples belong after the cheap screen.
Do not follow its suggestion to omit a failed search result: preserve negatives.
Neither its agreement nor Sol's implementation report substitutes for validation.
Sol's final prose called the normalization RMS; the actual reviewed code uses
LayerNorm. No claim of RMS normalization is made here.
