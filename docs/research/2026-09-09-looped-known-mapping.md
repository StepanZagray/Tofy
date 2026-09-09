# Known-control spatial prerequisite

Registered September 9, 2026 IST, before launching this training or reading its
frozen panel. This is one empirical prerequisite screen, not an ARC score,
method promotion, or a matched comparison against the harder variable-rule task.

## Claim and finite premise

Can the unchanged looped model learn one-step policy and all four successor
states on new procedural layouts when action meanings are constant across
training? Mapping zero is `[up, down, left, right]`. The visible grid contains
the agent, goal and walls; the mapping and a local simulator rule determine
every target. Thus targets are identifiable without episode-specific rule
inference. This does not prove the particular network/optimizer can learn them.

The completed frozen counterfactual at source
`2dc5c10e973c791b05100c56926e490d865a3918` found the fresh variable-rule model emits
the goal-completion picture for all 3,072 action outputs, with only terminal
frames correct. This prerequisite tests spatial learning before choosing a
control-learning curriculum or representation change. Failure would not uniquely
locate an internal geometry, attention, normalization or loss defect. The finite
rule-blind weighted-loss reference also does not prove that template optimal.

## Intervention, invariants and data boundary

Only the mapping distribution changes: all sixteen original rule slots now use
permutation zero. Keep factual calibration support, original query generation,
`episode_id = global_row / 16`, one-step queries, and original sample order.
There are 73,600 rows and 4,600 query IDs. Repeating each query sixteen times
reduces complete-input diversity; this change is part of the easier task and
precludes attributing a difference from the variable-rule parent to one hidden
mechanism. Report actual unique input/query counts and action-label counts.

Mapping zero belonged to the old held-out partition; it is **trained here**.
Use `KnownMapping` and new-layout terminology. No public ARC data, task-specific
solutions, language weights, architecture, task simulator, loss, optimizer or
normalization changes. Three ordinary support moves remain visible. Cleared
support is a secondary frozen control and has no 25% information ceiling under
a constant known mapping.

Start from seed 0 and data seed 9173, with original initial-weight SHA-256
`4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802`.
Model remains width128, four heads, two shared blocks, 992,393 parameters.
Train at the unchanged deterministic 1/2/4-loop schedule, maximum depth8,
AdamW learning rate0.0003, weight decay0.01, sorted global clipping threshold1.
Loss coefficients remain policy1, value0.1, reward0.1 and balanced categorical
successor0.5. No objective-weight intervention or component-gradient claim.

## Compute, stopping and checkpoint selection

Exactly 1,150 optimizer updates, effective batch64, physical33 plus a tail31
(two accumulation steps), final update1,150 only. Save every25 updates for
recovery evidence; do not select or stop on intermediate quality. Coverage mode
must honor the entire update budget. FitSmoke remains exactly **two** updates.
The eight-query reference keeps sixteen repeats per query (128 forwards); it
is eight independent queries, not128. Its policy gate never stops Coverage.

Batch33 was the largest observed long-run stable physical batch with this
unchanged model/shape:34 breached the512MiB reserve,35 exhausted memory. Verify
the new exact binary at batch33 and maximum training depth4 in a two-update
FitSmoke before launch; if it fails, preserve the failed root and lower the
batch in a recorded pretraining amendment. Never silently change batch mid-run.
Use immediate512MiB sampled device reserve, maximum35 minutes for training and
ten additional minutes for external profiler finalization. Prior identical-shape
1,150-update runs took about27 minutes. Stop on integrity/device/nonfinite or
deadline failures; affected roots are infrastructure evidence only.

Build a reviewed pushed clean commit with
`--release --locked --offline --features cudnn,profiling,serde_json/float_roundtrip`.
Record source and Candle Graph revisions, binary SHA, build command, exact
arguments, hardware/software and initial/final hashes. All wired profiling is
enabled: first evaluation and updates2,100,1000, unique host trace, NVTX,
Candle Graph tensor/scalar/gradient evidence, and Nsight CUDA/NVTX/OS-runtime/
cuDNN/cuBLAS/CPU-sampling reports. Device and capture preflight must pass first.
Automatic NVTX-to-graph correlation, operation/activation linkage, allocator
lifetimes/checkpoints and device-event planes remain incomplete/unwired; retain
raw application-label verification and report gaps. Profiled host spans are not
kernel timings and tensor sizes are not peak VRAM. No performance claim.

## Frozen evaluator and controls

Freeze the implementation and analyzer before the final readout. Evaluate only
the final checkpoint, four loops, physical/effective batch1, zero optimizer
updates. Use64 fresh one-step layouts, data seed20260912 and episode IDs
`0x4b4e4f574e + index`. Evaluate factual then cleared support for each query:
128 model inputs and512 action tuples; primary factual population is64 policy
rows and256 action tuples. CPU audit checks query disjointness from all4,600
training IDs, actual label counts, all outcome classes, and exact target parity
between support conditions. The same generator owns audit and evaluation.

Before training, run the frozen evaluator on the unchanged initialization as
an implementation/negative control; it supplies no checkpoint selection.
Independently reconstruct visible-grid labels and all-action simulator targets.
Check probability normalization, first-index argmax, raw-array dimensions,
offsets/hashes, region counts, oracle=100% and copy behavior. Archive complete
per-pixel probabilities and states, policy logits/probabilities, reward and value.
Treat blocked tuples separately from genuinely changed outcomes.

Report factual and cleared policy accuracy/CE; uniform-random expected accuracy
0.25 and CEln4; actual best-constant panel accuracy `max(label_counts)/64`.
There is no information-theoretic blind bound. Report full successor exactness,
copy-current and goal-completion-template baselines, changed/unchanged/vacated/
destination correctness and blocked/nonterminal/terminal counts. Report reward
precision/recall/Brier at threshold0.5 independently. Every value target is1;
constant-one MSE0 is the baseline, with no long-horizon value interpretation.

## Registered screen decision and uncertainty

Use10,000 paired whole-query bootstrap draws with NumPy PCG64 seed1910,
percentile95% linear interpolation. Resample all four actions and both support
conditions together. Recompute the best of four constant actions within every
draw; the policy advantage statistic is factual accuracy minus that maximum.
Return undefined intervals and count undefined draws when a stratum has no rows;
never turn an empty stratum into success. Report absolute counts and intervals.

The **policy prerequisite passes** only if factual accuracy is at least90% and
the95% lower bound for its advantage over the best constant action exceeds0.25.
The **successor prerequisite passes** only if factual full-state exactness is at
least75% overall and independently at least75% in each of blocked, nonterminal
and terminal strata; all three must be nonempty. These are a conjunction of
screen criteria, with no selecting a metric, subset or checkpoint afterward.
Reward/value are separate descriptive gates, not substitutes for policy/state.
Cleared-support results and template/probability diagnostics cannot rescue a
failed primary gate. A single seed supports a bounded prerequisite only;
promotion of a method would require fresh seeds and a suitable comparator.

If both prerequisites pass, the next decision is a separately registered
variable-control curriculum screen. If either fails, inspect that failed
prerequisite and select one representation/optimization intervention. Do not
auto-launch a dependent treatment from an informal reading of intermediate
metrics. Complete analysis, preserve positive and negative evidence, seal the
artifact tree after all capture children exit, update the local research library,
then continue the broader autonomous model work.
