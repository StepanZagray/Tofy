# Matched procedural-episode coverage screen

Decision recorded September 8, 2026 IST after the completed fixed fit and frozen
depth sweep. This is the next bounded experiment, not a queued automatic run.
It requires a reviewed sampler/runner change and exact-binary preflight.

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

## Execution addendum — September 8, 23:24 IST

The user has authorized autonomous continuation. The implementation uses
`--mode coverage --coverage fixed|fresh`; both arms run the requested update
count even if a reference fitting check passes. The existing fixed-fit mode
keeps its separate first-pass diagnostic contract. Model/task source, objective,
optimizer and loop schedule remain unchanged.

Before training, run `coverage-audit` for each arm at updates1150/effective64,
data seed9173 and a300-second cap. This invokes no model or optimizer. It records
every planned example/input/query/target hash, unique input/query/episode counts,
rule balance, all64-layout oracle/constant controls, and query-disjointness for
both policy and prediction evaluators at seed20260909. The actual training loop
records the same per-example schema; analysis must verify exact agreement with
the corresponding sealed planned stream before interpreting outcomes.

Frozen policy rows add query hashes; prediction rows add complete input and query
hashes. Existing numerical fields must match the previous evaluator on the same
checkpoint and original32-layout inputs in one zero-update CUDA parity check.
Then use the registered64-layout readouts for the matched experiment. Evaluator
parity and data audits are integrity checks, not additional model-selection gates.

Requalify physical33/tail31 against physical34 with the exact launch binary and
the prior512MiB sampled memory reserve. Both comparison arms use the same selected
pair; if33 is not qualified, stop and amend the capacity choice before either arm
starts. Keep the explicit `serde_json/float_roundtrip` feature on both producer
and Candle Graph CLI. All wired profiles remain enabled; automatic NVTX
correlation and the previously disclosed producer-plane gaps remain limitations.

## Reproducibility repair addendum — before fresh-arm outcomes

The first fixed arm at273f2c6c was stopped after645 updates, retaining checkpoint625
and its partial stream as exploratory. Initial weights, fixed data and initial
readouts exactly matched the previous fitting run; update1 losses also matched,
but clipping norms differed by one F32 ULP and the later trajectories diverged.
This comparison does not prove that the reduction order caused all divergence.

A direct CPU regression on identical named gradients in independently allocated
parameter maps reproduced changing global clipping scales. The global norm loop
used unordered VarMap iteration for non-associative F32 addition. Sort floating
parameters by name before norm accumulation and scale application, retaining the
same mathematical optimizer, clipping threshold and all other experiment choices.
The regression must pass across64 independent maps, alongside optimizer tests.
This fixes one local reproducibility defect; it is not an accuracy intervention
or a guarantee of deterministic CUDA training.

Restart BOTH full arms from initialization at the reviewed, pushed repair commit;
do not resume or compare the interrupted arm against a repaired treatment. Keep
the original seeds, data,1150 updates, evaluator and promotion gates. Before either
full arm, qualify the new binary/device/profilers and require two independent
six-update fresh-stream CUDA replicas to have identical initial/final checkpoint
hashes and exact losses, global norms and clipping scales on every update. Ignore
timing/root fields for this numerical comparison. Require identical initial reference
predictions and all384 training rows, at least one clipped update, successful
seals and profiler verification. Use reachable capture update2 in both independent
processes. The six steps cover two1/2/4
cycles and the first changed episodes; this is a local preflight, not a global
determinism proof. If it fails, investigate the next numerical source before
spending the full pair's compute budget. No threshold relaxation or favorable
replica selection is permitted.

The previous35-minute-per-arm bound applies independently to the restarted pair.
All failed/interrupted roots remain preserved and excluded from quality evidence.

## Exact-binary capacity amendment — before full-pair outcomes

Qualify physical34/tail30 in independent six-update fresh-stream replicas. Select34 only if both preserve512MiB reserve and pass exact replay, seals and profiler checks. A capacity failure selects33/tail31 and requires fresh independent six-update replicas at33. A numerical mismatch in completed memory-qualified replicas stops the experiment for diagnosis; it is never a reason to select another batch. No full arm has started at this source. Use identical selected pair for both full arms; effective64, seeds, initialization, objective and all quality gates unchanged. Old33-vs-new34 quality comparisons forbidden.

The repaired binary passed two physical34 capacity smokes with547/515MiB reserves; physical35 failed CUDA allocation. The prior fixed33 choice is updated only through the same512MiB safety rule. Source and binary stay713e672d/68ddaf33; record the final pair in selection.json and freeze the analyzer batch expectation before launching either full arm.

## Long-run capacity rejection — before any fresh full arm

Physical34 failed the existing512MiB whole-GPU reserve during the full fixed arm (7732/8151MiB;419MiB reserve). Stop and exclude this interrupted arm. The increase is not localized to the model allocator; short-run capacity did not ensure the reserved headroom under observed long-run load. Select physical33/tail31 under the already registered capacity-fallback branch. Require exact-binary depth4 capacity smoke and independent six-update replicas at33 with the same exact replay/512MiB/profiler gates. Restart BOTH full arms from initialization at source713e672d with physical33+31, all other choices unchanged. Do not compare old34 to new33. No extra seeds, metrics, checkpoints or relaxed thresholds. Preserve previous qualification and interrupted artifacts.

Execution order for the final batch33 pair: fresh then fixed, both completed before paired analysis. This ordering change occurs before either final-arm outcome and changes no training/evaluator budget or gate. A monitor now terminates immediately on the existing512MiB sampled reserve failure rather than waiting for the end-of-run acceptance check.
