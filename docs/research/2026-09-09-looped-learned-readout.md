# Learned spatial readout — preregistration

Registered September 9, 2026 IST, before implementation or head fitting. This is
a single-initialization synthetic prerequisite screen, not architecture promotion.

## Claim and mathematical premise

Can a small action-supervised attention readout learn known-control, distance-one
decisions from frozen spatial tokens without the privileged agent/goal routing
used in C8? Compare an approximately parameter-matched CLS head. Test separately
on the shared initialization and the C7 final recurrent checkpoint; report all
arms. C8 role probes were perfect even at initialization, whereas native C7 policy
was constant. That motivates this hypothesis but does not establish its outcome.

For frozen H with shape [64,128], each trainable query q produces
`a = softmax(Hq / sqrt(128))`, `z = sum_i a_i H_i`. Concatenate two pooled vectors
and apply a learned affine map to four logits. Permuting H's rows leaves z
unchanged because it permutes scores and values together. H already contains
spatial metadata, so this does not remove position from the representation.
If every H row is identical, z is independent of q and query gradients vanish:
learned pooling is not guaranteed to recover missing information. Independent
query initialization avoids imposing exact query symmetry. C8 establishes feature
variation and privileged linear recoverability, not trainability of this head.
The nonconvex fit has no claimed global optimum or mathematical ARC implication.

## Frozen inputs and data boundary

Use the sealed C8 initial and final feature artifacts from
`looped-frozen-features-20260909T115455-IST`. Core checkpoint SHA256 identities:

- Initial: `4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802`.
- C7 final: `a983626a888279cbb31651c6abcd1ebf4f808c8927a29a8349d03ffc35bd4e5a`.

Fit only C8 rows 0..511, seed 20260915 and episode IDs
`0x46454154555245+i`. Read only their feature byte ranges and action labels during
fitting; do not load C8 evaluation rows or arrays. Features are the exact F32
post-final-RMS tensors without added normalization, trainable projections, token
selection or positional inputs. No raw cells, simulator state, true-role indices,
coordinates, attention labels, auxiliary objective, LLM or public ARC data enters
the learned head. Source artifact manifests may be verified in full as integrity
checks, without using excluded content for fitting. Core forwards and core
optimizer updates during head fitting are both zero; cached extraction used four
shared-depth loops. Frozen core parameter files must remain unchanged.

After all six final heads are frozen and their external manifest is recorded,
generate one fresh factual known-mapping-zero panel: seed 20260916, episode IDs
`0x524541444f5554+i`, 256 queries, minimum/maximum distance one, four loops. All
rows have partition `fresh_eval`. Do not generate this panel, inspect its labels,
or extract its features until every final head is sealed. Audit uniqueness and
zero query overlap with the four C8 exclusion populations plus all 768 C8 queries.
Require all four labels, matching visible-geometry oracle, exact input/target/label
identities and unchanged checkpoint hashes. A collision or missing class fails
the registered panel; never replace samples. Extract both checkpoints on exactly
the same panel with batch/effective batch one. Fresh-panel labels are available
only to frozen evaluation and independent analysis. C8's already inspected 256
evaluation queries cannot satisfy any C9 gate and need not be rescored.

## Six arms and exact fitting recipe

At each frozen checkpoint, fit:

1. Spatial attention: two independent learned 128-dimensional queries, softmax
   over all 64 current tokens, concatenation of weighted sums, affine 256-to-4
   logits. Exactly 1,284 trainable parameters.
2. CLS comparator: affine 128-to-10, SiLU, affine 10-to-4. Exactly 1,334 parameters;
   slightly larger parameter count is not identical statistical capacity.
3. Spatial null: the same spatial head and initialization, trained on one fixed
   permutation of the 512 fit labels. Use one shared NumPy PCG64 seed-1915 index
   permutation, written and hashed before fitting; require a complete bijection.

All six use seed zero and identical initial bytes within each head family across
checkpoints and true/null labels. Initialize each parameter deterministically
as in the existing probe: SHA256(seed as little-endian u64 followed by parameter
name), first eight digest bytes interpreted little-endian seed a ChaCha8 RNG;
non-bias weights use independent uniform draws in +/-sqrt(6/(first_dim+last_dim)),
biases are zero. Query shape is [2,128]. Save all initial heads and parameter names.

Use F32, mean action cross-entropy only, full physical/effective batch 512,
accumulation one, exactly 1,000 AdamW updates, learning rate 0.003, betas 0.9/0.999,
epsilon 1e-8, weight decay zero, global gradient-norm clipping at one. Sort named
parameters for initialization, optimizer construction and gradient reduction.
Keep data order identical; full batch means every update contains every fit row.
No scheduler, early stopping, intermediate-checkpoint selection, feature scaling
fit, role loss, additional seeds, hyperparameter sweep or head redesign in this
run. Save every arm's update-1000 head; no choice between initial/final features.
A failed numerical/integrity gate is a failed run, not a reason to change settings.

Qualification may use deterministic artificial features for unit controls, plus
two updates on the registered 512 fitting examples for each head type. These
implementation smokes cannot choose hyperparameters and are excluded from model
evidence. Every evidence arm starts from freshly reconstructed initial weights.
Full batch 512 is the largest meaningful batch for this 512-example objective;
measure its actual stable device memory in the smoke. If it cannot leave 512 MiB
sampled whole-GPU reserve, do not silently introduce accumulation or change batch.

## Metrics, controls and decisions

Retain per-query fit and fresh logits, labels, predictions and identities; report
accuracy and stable softmax CE for every arm, including null fits scored against
their fitted labels and separately against true labels. Report native frozen-core
fresh policy, best constant and analytic geometry oracle, plus fit losses every
100 updates, initial/final parameter hashes, query separation, attention entropy,
and parameter gradient norms in selected captures. Attention is a diagnostic, not
an explanation that the queries have learned semantically named roles.

Frozen independent analysis must reconstruct predictions and labels, verify full
provenance, compare native policy reconstruction when feasible, and reject missing,
nonfinite or malformed data. Positive control: geometry 256/256; negative controls:
both permuted-label spatial heads <=50% against real fresh labels. These coarse
controls test obvious failure/leakage, not a permutation-test significance claim.
Require all numerical, gradient, profile and integrity checks before interpretation.

Use 10,000 paired whole-query bootstrap draws from NumPy PCG64 seed 1916, 95%
linear-percentile intervals. Recompute best constant within each draw. Report all
arms' accuracy and CE intervals, both spatial-minus-CLS comparisons, both spatial
advantages over best constant, and initial-to-final differences by true-label
head family. Intervals are pointwise descriptive, not multiplicity-adjusted.

For each checkpoint separately, the learned-routing screen passes only if true
spatial accuracy is >=90%, lower interval of advantage over resampled best constant
is >25 percentage points, and lower interval of advantage over its CLS comparator
is >10 points. Report every pass/fail; never select a winning checkpoint or null
realization after observing results. A passing initial arm establishes a bounded
learned-readout result from untrained recurrent features, not a C7 training gain.
If spatial is competent but CLS also succeeds, report competence without claiming
the registered spatial advantage. Failure leaves head expressiveness, optimization,
feature scaling and representation adequacy unresolved; it does not prove absence
of information. A single seed cannot promote a method or claim general reliability.

Primary analysis will choose and separately register the next experiment. A
positive routing result would justify fresh multi-seed confirmation and then
testing end-to-end integration; a valid negative calls for a cheaper optimization
or pooling-premise diagnosis. No automatic new core training follows this run.

## Budget, profiling, provenance and limits

Use one reviewed, pushed clean C9 source revision with dependency
`1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a`; record exact revision, source C8 identity,
build command/features, separate binary hashes, CUDA device/software, every CLI,
batch/accumulation and never-reused run roots. Build with cudnn,profiling and
serde_json/float_roundtrip. Qualify exact-binary CUDA head gradients/captures and
legacy feature extraction parity before any full head fit. Freeze and test the
independent evaluator before treatment outputs. Check remote exact-commit access.

Budget <=600 seconds summed head-fit model runtime across six arms, plus <=600
seconds summed profile finalization. Estimate from measured smokes before launch;
if budget cannot finish all arms, stop incomplete, without reducing updates.
Fresh audit <=60 seconds, each of two frozen extractions <=300 seconds plus <=300
seconds finalization, frozen head evaluation <=60 seconds each. Wall-clock
supervisors bound stages, retain process groups/start identities, clean children,
and fail closed on incomplete exits. Training requires AC online and stops on
nonfinite values, profile/integrity failure or sampled reserve below 512 MiB.

Enable every wired evidence plane. Add a truthful head-only Candle Graph gradient
contract with no active core/reward/value/dynamics families; tag zero executed
core forwards and cached extraction depth four. Capture update 2 and final head
evaluation; the head has no recurrent depth. Record attention/pooled/logit/loss
statistics, head gradients before clipping, clipping and optimizer phases, with
device synchronization at measurement boundaries. Retain host tracing, NVTX,
Nsight CUDA/cuDNN/cuBLAS/OS-runtime/CPU sampling and separately bound bundles.
Profile the first forward of each fresh extraction. No profiler is silently
disabled. Existing operation/activation linkage, allocator/physical-memory,
device-event and automatic-correlation gaps remain disclosed. Host spans are not
CUDA kernel times; profiled-work captures cannot establish production speed.

Stop all owned processes before point-in-time sealing, rehash finalized trees and
retain manifest digests outside run roots. Preserve failed launches as infrastructure
evidence only. Cached-head supervision, focused policy loss and frozen features
differ from C7 joint training: this experiment cannot uniquely explain that failure,
validate successors, show useful recurrence or planning, or establish ARC competence.
This simplified learned pooling is an adaptation, not a paper-faithful Set
Transformer, Seeker or FocusPool replication, and inherits none of their theorems
or empirical performance claims.
