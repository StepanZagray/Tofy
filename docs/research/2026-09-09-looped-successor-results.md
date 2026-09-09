# Frozen successor outputs confirm a goal-completion shortcut

September 9, 2026 IST. The fresh-coverage checkpoint produces the exact same
goal-completion template for **every action and observed mapping at each of 32
new queries**. All 3,072 categorical outputs match that template. Its 384 correct
frames are precisely the terminal cases; copy-current is correct on 1,008 frames.
Neither checkpoint produces a pair of correct, genuinely different successors
under alternative observed mappings.

The probabilities are not identical: fresh maximum rule-pair TV exceeds the
registered 0.001 tolerance. The descriptive near-invariance criterion therefore
fails. Categorical invariance does not establish internal disconnection, an
unlearnable architecture, or a globally optimal training recipe.

## Design and independent checks

[Registration](2026-09-09-looped-successor-counterfactuals.md), including the
pre-readout descriptive clarification. Both parent checkpoints are the final
1,150-update runs at training source
`713e672dfc8e7eb316b8cb622c4b2569f839d761`. The frozen evaluator is
`2dc5c10e973c791b05100c56926e490d865a3918`, binary SHA-256
`c932fa99a1b4ea405fb787469ade36ef2f85e2145dbe5242c86b70002e33d8c3`.
Model, task and module-definition source bytes match the parents. Candle Graph
remains `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a`.

Use seed 20260910 and layout IDs `0x43464c4f4f50 + index`, 32 layouts, all 24
mappings, four actions, four inference loops and physical/effective batch one.
Even layouts have distance one; odd layouts distance 2–10. These are coverage
strata, not a causal comparison of distance. Within each query, all mappings
preserve query pixels, distance and calibration action order. Three factual
calibration moves identify the action-direction bijection.

Both arms consume identical inputs and simulator targets. All 32 query hashes
are absent from both training streams, and initial/final weights equal their
registered parent hashes. Raw outputs retain all 16 palette probabilities for
every pixel, along with categorical current, target and predicted states.
An independent Python oracle reconstructs targets, distances and labels from
visible query cells and the observed controls. It verifies all logged regions,
categories, offsets, hashes, normalization and argmax outputs.

All 11 Rust runner tests and 21 Python control/integrity fixtures pass. A
two-layout device smoke and its profiler evidence pass. The rebuilt old inspect
mode reproduces all 1,536 policy rows and 512 prediction rows byte for byte;
all old report fields agree except elapsed/training durations. A separate Astra
reader, without importing the analyzer, verifies the full raw-state counts and
every fresh within-split rule/action probability comparison.

## Results

| Endpoint | Fixed coverage | Fresh coverage |
|---|---:|---:|
| Exact successor frames | 0/3072 | 384/3072 (12.5%) |
| Copy-current exact baseline | 1008/3072 (32.8125%) | 1008/3072 (32.8125%) |
| Exact terminal frames | 0/384 | 384/384 |
| Exact nonterminal or blocked frames | 0/2688 | 0/2688 |
| Predictions equal to goal-completion template | 15/3072 | 3072/3072 |
| Both correct, different-target familiar-rule pairs | 0/11264 | 0/11264 |
| Both correct, different-target held-out-rule pairs | 0/2816 | 0/2816 |

The offline template removes the visible agent and marks the visible goal as
completed. It is neither an agent nor a modified target. On distance-one queries,
fresh also emits it for all 600 moving-but-nonterminal and 552 blocked action
cases. All 1,536 longer-distance cases receive it too. These wrong-action and
longer-distance cases distinguish the template from the correct simulator state.

| Maximum per-pixel probability TV | Fixed | Fresh |
|---|---:|---:|
| Familiar-rule pairs | 0.375743 | 0.001499679 |
| Held-out-rule pairs | 0.377726 | 0.001473145 |
| Familiar action-head pairs | 0.446835 | 0.026985823 |
| Held-out action-head pairs | 0.393801 | 0.026974154 |

Fresh categorical predictions change on zero eligible rule or action pairs.
Fixed predictions change on 11,256/11,264 familiar and 2,810/2,816 held-out
different-outcome rule pairs, but none has both outputs correct. Sensitivity
alone is therefore an inadequate competence test in either direction.

There are 15,360 familiar-rule and 3,584 held-out-rule comparisons in total.
Respectively 1,024 and 256 compare different directions with identical blocked
outcomes; those are excluded from causal-positive denominators. Same-mapping
different-action comparisons separately contain 2,816 familiar and 1,408 held-out
different-outcome pairs, with zero both-correct pairs in both models.

Fresh gets every vacated pixel correct and every nonterminal destination pixel
wrong. The correct destination color still has mean probability about 0.4064 on
distance-one nonterminal rows and rank two everywhere in that stratum. This
graded signal does not by itself establish correct spatial or rule binding.
All exact, pixel-distance, template-rank, destination-probability and
demonstrated/inferred-action strata are retained in the analysis JSON.

Intervals use 10,000 shared paired whole-layout PCG64 draws, seed 1909, linear
percentile 95% quantiles. Zero-denominator draws are counted explicitly. The
all-zero both-correct endpoint has a degenerate [0,0] empirical bootstrap
interval; it is not a proof of zero performance on the underlying task population.
Only one initialization/checkpoint per arm is represented.

## What the loss can and cannot explain

A separately verified exact-real reference helps interpret these pixels. For a
fixed distance-one query with m nonblocked directions, the source balances
`128m` changed against `16384 - 128m` unchanged pixels across all four actions.
An unconstrained pixel classifier that knows the query but lacks its mapping
minimizes the weighted CE at goal-completed probability `(128-m)/(128+2m)`:
0.912–0.977 for m=1…4. Its probability of vacating the old agent is also high.
Thus those two local choices can arise without correct control inference.

This does **not** derive the whole observed template: under the same weighted
pixel objective, a geometry-aware rule-blind optimum also places agents at other
reachable neighboring cells. Nor does it prove that weights, encoding or shared
gradients caused the model's failure. The optimum depends on the loss: a terminal
frame can be a tied rule-blind optimum under exact-frame 0–1 loss when m=3 or 4.
Probability-simplex boundary optima and F32 rounding are qualified in the
[finite derivation](/home/stepan/Research/_runs/2026-09-09T075728Z-tofy-looped-successor-counterfactuals/findings/weighted-blind-optimum.md).

## Execution and decision

Campaign:
`/home/stepan/Projects/code/.tofy-runs/looped-counterfactual-20260909T091012-IST`.
Both full model invocations completed in about 11 seconds; maximum sampled GPU
usage was 381/8151 MiB. These are invocation timings and whole-GPU samples,
not production or kernel-timing comparisons. All four qualification/full
Nsight-bound bundles pass structural, semantic, hash and raw application-label
checks. Automatic NVTX correlation remains incomplete; operation/activation
linkage, allocator lifetimes/checkpoints and device-event intervals remain
unwired. No optimizer updates or ARC evaluation occurred.

Opus 5 xHigh review succeeded on the shorter retry after the first attempt timed
out without a response. Its useful suggestions added graded output diagnostics
before readout. The primary review corrected its misreading of the copy baseline
and qualified causal interpretations; advisor agreement is not evidence.

The selected next experiment is a **fixed known-mapping spatial prerequisite**:
keep the looped model, losses, optimizer and fresh query order, but use a constant
control mapping with ordinary factual support. This removes the need to infer a
different mapping per episode. It also reduces complete-input diversity and
changes gradient variation, so it cannot be treated as an improvement over the
harder variable-rule task or as unique localization of an internal failure.
Its purpose is to decide whether spatial learning succeeds before testing a
control curriculum or temporal representation change.

No ARC improvement follows. The new core still lacks learned persistent episode
memory, autonomous probing and an ARC adapter; no pretrained LLM or public-level
training was used. The broader autonomous model work continues after this diagnostic.

Exact supervised commands, source/parent bindings, artifacts and all statistical
results are retained in the [research run](/home/stepan/Research/_runs/2026-09-09T075728Z-tofy-looped-successor-counterfactuals/manifest.json),
[paired analysis](/home/stepan/Research/_runs/2026-09-09T075728Z-tofy-looped-successor-counterfactuals/paired-analysis.json)
and campaign `launch-qualification.json`. Never reuse the sealed run roots.

## Final integrity seal

The complete campaign contains261 files,1,803,789,614 bytes. All43 recorded
C4/Opus process IDs are gone. The externally retained [manifest](/home/stepan/Research/_runs/2026-09-09T075728Z-tofy-looped-successor-counterfactuals/completed-campaign.manifest.json)
was rehashed against the complete tree, SHA-256 `e224ee855cd03132a0bdaa39585259ceed0a093a3f2de55341e33974033efc3e`. This is point-in-time
integrity, not immutable storage. The next training campaign is separate.
