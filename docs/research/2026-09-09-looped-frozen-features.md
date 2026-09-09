# Frozen spatial feature readout — preregistration

Registered September 9, 2026 IST, before feature extraction or probe fitting.
This is a synthetic diagnostic after C7 replay failed its selection gates. It
does not train the recurrent core or establish ARC competence.

## Claim and mathematical premise

Can a fixed linear readout recover known-control, distance-one policy from the
frozen replay model's CLS state or from its current agent/goal patch states?
Compare the final C7 checkpoint with the identical shared initialization. A
successful probe establishes recoverability under the specified representation,
data and fitting procedure. A failed probe cannot prove information is absent.

The observation is sufficient: for goal-minus-agent displacement `(dx, dy)` on
this distance-one distribution, action scores `(-dy, dy, -dx, dx)` select the
unique correct action under `[up, down, left, right]`. This finite statement is
an oracle control, not a claim that CLS already encodes the displacement.
For standardized features X and centered one-hot labels Y, the registered ridge
objective `mean(||XB-Y||²) + 0.01 ||B||²` is strictly convex in B. Its unique
solution is `(XᵀX/n + 0.01 I)^-1 XᵀY/n`. Solve the system in F64, verify the
normal-equation residual, and retain coefficients. This proves the optimizer's
local objective solution under exact arithmetic, not optimal classification or
the absence of useful nonlinear information.

## Frozen model, panels and extraction

Use exactly two checkpoints: initial SHA256
`4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802`,
and C7 final SHA256
`a983626a888279cbb31651c6abcd1ebf4f808c8927a29a8349d03ffc35bd4e5a`.
C7 source is `8cec1006b49e3c118ec9979f4f5e393e5cd7dc29`.
Preserve all 992,393 parameters and ordinary model operations. Export the exact
post-final-RMS CLS state `[128]` and current-patch states `[64,128]` already used
by ordinary heads. No extra forward, hidden-rule feature or optimizer update.
Extract ordinary policy logits from the same forward. Test output, parameter and
gradient parity for the extraction API; verify exact legacy frozen raw outputs
on the launch binary before new panel extraction.

Generate 768 unique queries with seed 20260915, episode IDs
`0x46454154555245 + i`, mapping 0, minimum/maximum distance 1, and normal factual
calibration. Rows 0..511 fit probes; rows 512..767 evaluate them. Both models use
identical examples and four loops, physical/effective batch 1. No cleared arm,
depth sweep, checkpoint selection or trainable recurrent weights. These are new
synthetic panels; they are not ARC levels. Before neural extraction, audit every
query/input/target/label hash, assert four labels occur in each partition, and
assert no query overlap with the retained C5/C7 training and observed C5/C6
evaluation panels. A collision fails the registered panel; do not silently
replace rows or change the seed.

Export F32 states and policy logits with explicit little-endian offsets/shapes,
complete row identities, current visible 8x8 cell colors, correct action,
checkpoint/source/binary hashes and zero-update count. An independent analyzer
must reconstruct the unique agent/goal positions, geometry label and panel
identity. Reject nonfinite arrays, wrong byte lengths, reused roots, different
input streams, checkpoint/source mismatch or incomplete captures.

## Prespecified probes and controls

For each checkpoint fit both probe families, reporting every result:

1. CLS: 128 features, the existing policy readout seam.
2. Visible-role patch pooling: concatenate the 128 features at the unique current
   agent cell (color 2) and goal cell (color 3), giving 256 features. This uses
   task-specific routing from visible input and changes the readout. Success
   cannot uniquely attribute native policy failure to CLS or establish a
   deployable general-purpose controller.

Also fit the same ridge procedure to four raw coordinate features
`(agent_x, agent_y, goal_x, goal_y)`, normalized by 7. Retain the exact analytic
geometry rule as a positive control. For every learned-feature probe and the
coordinate probe, fit a null counterpart using one fixed permutation of the 512
training labels from NumPy PCG64 seed 1914; score against real evaluation labels.
The permutation is shared across all five probe matrices and preserves counts.

For every feature dimension, subtract the training mean and divide by the
population training standard deviation clamped below at 1e-6. Apply those same
statistics to evaluation features. Center one-hot training labels, fit ridge
with lambda 0.01 in F64, and add the training label mean as an unregularized
intercept at prediction. Argmax ties choose the lowest action index. Do not tune
lambda, normalization, features or labels after observing results. Save means,
scales, clamped-dimension counts, coefficients, residuals and all row predictions.
Use one BLAS thread; this is a small CPU fit, not another GPU training campaign.

## Metrics, uncertainty and decisions

Report absolute train/evaluation accuracy for all probes, native policy accuracy
and CE, best constant action, and all null controls. CE applies to native model
probabilities; ridge outputs are uncalibrated scores and receive no probability
or CE claim. Report feature scale distributions so tiny amplified features are
visible. Require relative normal-equation residual <=1e-9.

Use 10,000 paired whole-evaluation-query PCG64 resamples, seed 1913, with 95%
linear percentile intervals. Recompute the best constant within each resample.
Report all four learned probe comparisons to native policy and initial-to-final
gains within each representation family, without choosing a winner post hoc.
Intervals are pointwise descriptive; no simultaneous or method-promotion claim.

Control validity requires the analytic rule 256/256, coordinate ridge >=95%,
all five permuted-label probes <=50%, and every integrity/numerical check passing.
A failed control makes model interpretation inconclusive and calls for diagnosis,
not a different fitting hyperparameter. For each learned-feature probe separately,
call recoverability supported only when evaluation accuracy >=90% and the lower
95% interval of advantage over resampled best constant exceeds 25 percentage
points. Report train fit alongside evaluation but do not use it to rescue a fail.
Single checkpoint, seed and task family cannot support architectural promotion.

If CLS passes, investigate a frozen-head/end-to-end optimization intervention.
If only visible-role pooling passes, test a separately registered learned spatial
readout with native routing controls. If neither passes with valid controls, test
a separately registered representation/attention intervention; do not conclude
the information is mathematically absent. If initial also passes, disclose that
training has not established the recoverability being claimed. No automatic
dependent training; primary analysis selects and registers the next experiment.

## Compute, profiling and provenance

Budget one CPU population audit, an exact-binary legacy parity evaluation, and
two frozen GPU extractions, each <=300 seconds model runtime plus <=300 seconds
profile finalization. No gradient accumulation or model training. Use the clean,
reviewed, pushed extractor commit with dependency
`1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a` and CUDA/cuDNN/profiling build; separately
hash the binary and all launch arguments. First qualify an actual legacy GPU
forward and an implementation smoke of the new export before full extraction.
The smoke may use only the first fitting row; it cannot inspect evaluation rows.
It is excluded from evidence and cannot choose probe settings.

Keep host trace, NVTX, Nsight CUDA/cuDNN/cuBLAS/OS-runtime/CPU sampling and the
first-forward Candle Graph capture enabled, with new separately bound bundles.
Existing operation/activation linkage, allocator/physical-memory, device-event
and complete automatic-correlation gaps remain disclosed. Host spans are not
kernel timing; profiled-work captures cannot establish production speed. Stop
on nonfinite/integrity failures or whole-GPU sampled reserve below 512 MiB.
Track every process, stop children before sealing, verify final artifact hashes
and preserve the manifest digest outside each never-reused campaign root.

This continues the user's autonomous task. Completing a negative diagnostic
does not complete the broad model objective.
