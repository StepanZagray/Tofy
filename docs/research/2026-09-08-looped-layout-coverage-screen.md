# Matched procedural-episode coverage screen — registered, not launched

Decision recorded September 8, 2026 IST after the completed fixed fit and frozen
depth sweep. This is the next bounded experiment, not a queued automatic run.
It requires a small reviewed sampler/runner change and exact-binary preflight.

## Claim and controls

Test whether resampling complete procedural episodes, with the same balanced
control-rule schedule, improves fresh-layout factual policy accuracy over repeated exposure
to eight fixed layouts at the same 1150-update budget. This is a single-seed
empirical screen; no theorem forces the optimizer to learn a transferable rule.
A lookup solution fits the finite training set without generalization, while a
correct algorithm can fit and transfer. Existing measurements do not identify
which representation or normalization mechanism is limiting.

One intervention: **complete episode coverage**, including query layout and
calibration action order, omitted action and starting position. The episode ID
seeds both query and calibration generation, so this is not an isolated query-
geometry treatment. A result cannot distinguish those coverage mechanisms.
Both arms use one-step visible-goal queries, the same 16 training rules, three scripted support transitions, identical
loss coefficients, optimizer, clipping, physical 33/tail31/effective64, parameter
initialization and order, 1/2/4 loop schedule, hardware and profiler cadence.
For global sample index i, select rule `train_rules[i % 16]`; fixed episode index
is `(i / 16) % 8`, fresh episode index is `i / 16`, using integer division and
the same training seed/tag. Seeds identify draws, not guaranteed unique geometry;
record unique complete input hashes, query hashes and actual layout collisions.
Use data seed 9173, initialization seed 0 with the already verified original hash.

The existing `train` mode is not this treatment: it randomly samples rules and
mixes one-step with distance-2–10 queries. Merely changing CLI mode would confound
coverage, label order and horizon. Implement a shared explicit sampler, use the
same evaluation/checkpoint cadence, and disable success-based stopping in both
comparison arms. Preserve the old fixed-fit diagnostic semantics in its own mode.

Train both arms fresh for exactly 1150 optimizer updates, 73,600 examples each;
do not resume weight-only checkpoints. Retain checkpoints every 25 updates and
initial/final weights. Report intermediate training diagnostics descriptively;
the primary comparison always uses update 1150, never the best observed point.
Both arms perform identical fixed-reference fitting checks to match overhead.
Capture first evaluation and updates 2/100/1000 under all currently wired evidence
planes, with the same Nsight settings and declared limitations. Capture per-loss
shared-core gradient norms/cosines on matched representative batches if proposing
any later objective change; current combined gradients alone cannot justify it.

## Evaluation and decision

Freeze the evaluator and controls before training. In separate zero-update
`inspect` invocations, evaluate both final checkpoints
at four loops, batch one, **64 layouts**, evaluation data seed **20260909**, with
every mapping in each split. This seed has not been inspected in this campaign.
Keep training metadata at seed 9173 and record the evaluator seed separately; do not change the training seed.
Log all policy input hashes/labels and prediction input hashes; add a generation
check that evaluation query identities are absent from both training streams.
No ARC inputs are allowed. Training uses only the 16 predefined mappings.

Primary treatment comparison: held-out-mapping factual accuracy of fresh minus
fixed, paired on whole layouts. Use 10,000 whole-layout bootstrap draws with
PCG64 seed 1908, percentile 95% with linear quantiles; identical draws for arms,
splits and controls. Report all counts and intervals. A positive screen requires
all of: held-out treatment-minus-control lower bound >10 percentage points;
fresh-arm factual accuracy >=75% on both splits; factual-minus-cleared lower
bound >25 percentage points on both splits. These are conjunctive gates, not
multiple opportunities to select a positive result. No alternate metric,
checkpoint, seed subset or depth can replace them after outcomes.

Report factual-minus-wrong-history accuracy against the original real target,
absolute follows-presented-rule accuracy (redundant under this permutation),
paired action/probability changes, destination/vacated pixel accuracy, full-frame
exactness versus copy, reward confusion matrix and value versus constant baseline.
Prediction/planner validity remains a separate gate regardless of policy success.
Do not use factual-minus-follows-presented-rule as an endpoint: it is identically
zero for deterministic predictions on this evaluator's permuted populations.

Before launch, prove the cleared 25% information bound and verify a perfect oracle
scores factual=1, wrong-under-real=0 and follows-presented=1. For cleared
inputs use a fixed action without hidden support, which scores 0.25 by balance;
the identifiable-support oracle itself cannot operate on cleared support. Validate
same input/label populations across arms and the changed episode distribution.
Any failed integrity or incomplete arm yields no treatment verdict.

One seed only. A pass licenses a separately registered confirmation on at least
two new initialization seeds, with individually recorded initial hashes; it does
not promote an architecture or an ARC agent. A failure rejects only coverage at
this seed/budget/recipe, not the architecture's expressiveness. Next inspect
paired before/after spatial representation and policy-specific credit before
registering one representation intervention. Outer-loop normalization remains a
separate candidate requiring an actual stability premise; no automatic change.

## Budget and launch boundary

Two sequential runs, maximum 35 minutes each, plus a combined 15-minute budget
for exact-binary smoke, frozen scoring, export and integrity. Estimated training
duration is about 30 minutes per arm from today's measured run, not guaranteed.
Never start the pair without enough supervised time to finish both and analyze.
Stop on timeout, numerical failure, data/hash mismatch or profiler failure;
preserve the failed never-reused root. Do not extend the budget or automatically
start follow-up seeds. Source/dependency/binary/CLI hashes and source-fetch checks
must be finalized at launch from a reviewed pushed clean checkout.
