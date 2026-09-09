# Frozen successor counterfactual diagnostic

Registered September 9, 2026 IST, before the new panel is evaluated. This is a
zero-update diagnostic following the failed matched-coverage screen, not another
training attempt or an ARC performance claim.

## Claim and weakest premise

The earlier fresh checkpoint predicts 32/256 complete successors correctly per
split, all on reward-positive rows. Both checkpoints already predict the changed
pixels of those rows correctly. Complete raw predicted states were not retained,
so the observation does not prove a terminal-template or rule-invariant shortcut.

Test that hypothesis directly: for an unchanged visible query, change the fully
observed control mapping and compare actual predicted states with the correct
counterfactual simulator states. A four-direction bijection is identifiable from
three distinct visible calibration moves and elimination. For fixed query/action,
different mappings can still have the same successor (e.g. two blocked directions).
Only pairs with different simulator outcomes are causal-positive examples.

There is no theorem that a trained network must use this identifying information.
Output sensitivity also cannot identify an internal transformer failure. These
are bounded empirical questions about the two named frozen checkpoints.

## Frozen design

- Parents: `looped-coverage-b33-20260909T0019-IST/{fixed,fresh}-seed0`, both update
  1150 at model source `713e672dfc8e7eb316b8cb622c4b2569f839d761`.
- Weight SHA-256: fixed
  `83a88e881f5262a8bb47e7e742b9ef483b8a928b92666e4034f0c46d47f71804`;
  fresh `81365192586b628177b05cfb600f485489703ad3fbd96047f5057b11e8aa77c0`.
- No changes to task generation, model, initialization, losses or weights. A new
  frozen evaluator only; no optimizer updates or learned probe. Verify model/task
  source bytes against the parent and parent/final checkpoint identities.
- Full panel: 32 new layout IDs, data seed 20260910, ID `0x43464c4f4f50 + index`.
  Even indices use distance 1; odd indices use distance 2–10. Every layout has all
  24 mappings and all four actions: 768 model inputs / 3072 action tuples per arm.
  Keep query, distance and calibration action order identical across mappings.
- Four inference loops, batch/effective batch 1. Fixed then fresh. Preserve raw
  softmax successor probabilities, categorical predicted/target/current pixels,
  input/query hashes, observed actions, policy/reward/value readouts and metrics.
  Binary states retain patch order; record byte offsets, shapes and checksums.
- Familiar and held-out mappings are the same 16/8 partition as the parent. No
  public ARC inputs, weights, controllers or teachers. Before evidence evaluation,
  verify full-panel query hashes overlap neither training stream. No query may be
  removed after looking at predictions. A collision invalidates that novelty claim
  and requires a separately registered fresh confirmation population.

## Controls, endpoints and decisions

Validate categories and region counts with CPU fixtures and the public-observation
oracle. Record blocked, moving-nonterminal and terminal outcomes, demonstrated versus
bijection-inferred actions, total tuples, distinct action/rule tuples and genuinely
different outcome pairs. Each split must contain all three outcome categories and
both demonstration strata before interpreting its category comparison.

The primary counterfactual endpoint is the fraction of within-split, same-query,
same-action, different-mapping pairs with genuinely different targets for which
BOTH complete predicted states are correct. Report all such unordered pairs,
eligible counts and the absolute endpoint for both models. Also report the fraction
with different categorical predictions, and maximum-per-pixel total variation
between probability distributions. A changed prediction is not a correct prediction.
Report different actions under the same mapping separately.

Alongside exact successor and region scores, evaluate copy-current and an explicit
negative control: remove the visible agent and mark the visible goal completed,
ignoring action and history. This goal-completion template is an offline diagnostic,
never an input, target modification or agent. Report full-frame agreement with
that template by outcome and split. Its ability to fit terminal rows is a finite
property, not evidence that the model learned controls.

Strict categorical template agreement means every predicted pixel equals the
template on every row of a stated stratum. Reject that strict hypothesis on any
counterexample; preserve mismatch counts. Separately call probabilities nearly
rule-invariant only if the maximum per-pixel TV across EVERY within-split rule pair
at fixed query/action is <=0.001. This is a preregistered descriptive tolerance,
not a statistical proof of disconnection. No rule-learning promotion follows from
small/nonzero sensitivity alone.

Report absolute scores and paired whole-layout uncertainty using 10,000 PCG64
seed 1909 bootstrap draws, percentile 95% linear quantiles. Resample 32 complete
layouts, retaining all their rules/actions/pairs. No seed, checkpoint, metric or
stratum is selected after inspection; all strata are descriptive. These intervals
do not estimate initialization uncertainty. No multiple-seed or ARC promotion gate
is applied to this frozen diagnostic.

If the terminal-template pattern replicates and correct alternative-state pairs
remain absent, the next decision is a registered representation/optimization
prerequisite, not more training of the unchanged recipe. If correct alternatives
exist materially, examine policy versus dynamics mismatch. Output-only evidence
does not distinguish encoding, attention, readout or competing-gradient causes;
inspect those independently before attributing a mechanism or changing losses.

## Execution and evidence budget

Review and push the evaluator before a clean release build with
`cudnn,profiling,serde_json/float_roundtrip`. Keep all supported existing profiler
planes and first-evaluation capture enabled; collect Nsight CUDA/NVTX/cuDNN/cuBLAS/
OS-runtime and process-tree CPU evidence. Retain the existing documented producer
gaps and incomplete automatic NVTX correlation; verify exact raw application labels
and publish a separate Nsight-bound bundle without modifying finalized captures.

Run a two-layout implementation smoke on the exact binary with seed 20260911.
Verify its raw file layouts, finite/normalized probabilities, all tuples, output
hashes, zero updates and parent identity before full evaluation. Separately rescore
the parent's old inspect panel using the same checkpoint/inputs and require exact
legacy row/report-metric parity (ignore durations and added provenance only).
Tests must catch corrupted offsets, arrays, parent bindings and outcome categories.

Maximum 10 minutes per model invocation including model-side evaluation, plus
bounded external profiler finalization; maximum 20 minutes total accelerator
evaluation for the full pair. Capture actual duration and reserve integrity/analysis
time. Retain at least 512 MiB sampled GPU headroom; abort on integrity/capacity failure.
Never reuse roots or resume partially failed outputs. Stop telemetry and children
before sealing, verify all hashes, and store the final manifest digest externally.

Completing this diagnostic is a research milestone. It does not end the user's
broader instruction to continue improving Tofy autonomously.


# Pre-readout descriptive clarification

Recorded 2026-09-09T09:12:38.056781+01:00, before any model output from this diagnostic.

Retain the panel, all primary endpoints, strict-template definition and TV tolerance.
Add descriptive per-pixel Hamming error to oracle/copy/goal-completion template,
nearest-template rankings (retain ties), and true destination-category probability
and rank. Report same-input different-action-head TV as a diagnostic, not a guaranteed
positive control. Break outcome counts out by distance regime as well as split and
demonstration. In particular, distance-1 wrong-action nonterminal/blocked cases have
the same query as the terminal case and distinguish oracle from goal template.
No causal horizon comparison is made across the different layout populations.
These fields use the same retained raw arrays and add no forward passes or promotion
opportunities. Qualify aggregate strict-template rejection with pixel distances.
