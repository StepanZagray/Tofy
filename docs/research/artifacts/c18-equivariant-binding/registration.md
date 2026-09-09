# C18 — action-equivariant finite selection, matched larger batches

Status: preregistration before any C18 accelerator invocation or learned outcome.
Prepared September9,2026, machine-local Europe/Dublin IST. User authorizes autonomous
work, a recurrent transformer without pretrained LLMs, and larger fitting
power-of-two batches. Source and every launch artifact will be frozen before use.

## Claim, baseline and weakest prerequisite

C17 completed1150updates at512 and fits256/256 semantic cases, including64/64
omitted actions, but fails held-out maps at71/128 overall and2/32 omitted.
Its frozen result is historical context, not the matched comparator here.
The empirical hypothesis is that shared action-token processing can learn all
16 missing-effect/query types with a strict correct margin after explicit
public-action routing. The optimizer outcome is unknown.

The finite proof in findings/symmetry.md establishes action-ID equivariance
in exact arithmetic and16 action-renaming orbits, not one. Given strict correct
fit representatives, renamed held cases follow structurally. Consequently
held-map accuracy is symmetry/implementation coverage, not independent semantic
generalization, algorithm discovery, native visual control or ARC evidence.
A finite lookup can solve these16 types. There is no convergence theorem for
this initialization, model, data and AdamW recipe. The cheapest prerequisite
is numerical equivariance at initialization before any main training.

## Intervention and comparator

Legacy comparator: unchanged C17 ControlBinder, width256, four heads, two shared
transformer blocks, learned CLS and four-output affine head,1,580,804parameters.
Treatment: four action tokens with five features each: observed effect A^T E,
observed mask A^T1, desired displacement broadcast to each token. Projection,
all shared blocks and scalar token head share weights across action slots;
no positional/action embeddings or slot-specific readout. Two blocks repeat
four times with zero initial state and reinjected input. Treatment has
1,579,265parameters. Shared block initial parameters must match exactly between
arms; projection/head shapes and parameter counts differ by design. This tests
the representation and output-sharing package, not an isolated single weight tie.

A only routes already observed public action IDs; this observed association is
an explicit representation prior. The forward graph neither supplies the
missing effect, matches directions analytically, uses correct labels/map IDs,
nor calls an external solver. Displacements remain differentiable tensors.
There is no pretrained LLM or learned vision execution in this experiment.
The cached visual diagnostic inherits privileged C15 initial selectors and
familiar/reused cases. No reward, value, dynamics, planner or episode memory is
implemented by this component screen.

## Fixed population, schedule, optimizer and seeds

Both arms use exactly C17's fit16/held8 lexicographic cardinal bijections,
1536/768 ordered rows and256/128 canonical support-order cases. Cached1024rows
are unchanged. Source dataset SHA256:
`bc1413652b6bf38cadde28c8f1fd3d88d9cf781ab55413300bc155e341c47295`.
All2,304abstract rows and cached coordinates are reconstructed before scoring.

Seed0 deterministic name-hashed ChaCha8 initialization is unchanged. Legacy
full parameter digest must equal
`55d6d89a7e8a22049d074ae828cabb2f114093ec88886df67744a453c0cd364e`.
The training stream is exactly C17's588,800 flat U32LE indices, SHA256
`a46eac4347bfe16d55d220240452cb67643ccd007a97d2e56da16be45cb6f51e`,
from PCG64 seed20260922:383 full1536-row epochs plus512 rows. Regroup this
unchanged prefix into the selected power-of-two effective batch B, retaining
one explicit last tail if needed. Never pad, drop, duplicate beyond the fixed
stream, or claim repeated examples are independent. Each arm therefore executes
ceil(588800/B) optimizer updates. Both have identical actual batch/tail/order
and update counts; compared with historical C17 the optimizer history changes.

AdamW learning rate0.0003, betas0.9/0.999, epsilon1e-8, weight decay0.01,
mean CE over actual rows, parameter-only weighted gradient accumulation,
global L2 clip1, no warm-up or scheduler. No automatic learning-rate scaling.
A larger batch is a resource setting, not evidence of improved learning.
One fixed seed is a screen only; no method promotion or seed-uncertainty claim.

## Batch qualification and resource budget

Use a clean reviewed pushed source, separately hashed CUDA/cuDNN binary,
sibling candle_graph1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a, current locked
software and local RTX5060Laptop8151MiB. Require AC power, temperature below85°C
and at least512MiB sampled reserve through the existing supervisor. Sampled
VRAM is not allocator peak. Preserve all failed capacity trials and exact PIDs.

Qualify each architecture on4frozen rows. Then try physical/effective powers
512,1024,2048,4096,8192,16384,32768 in ascending order. At each candidate run
legacy then equivariant, each with two disposable updates then five-update
confirmation. These must consume actual full candidate batches, from dedicated
2B/5B prefixes of the fixed stream. Restore exact initial weights afterward;
smoke outcomes are selection-only, never learned-model evidence. Check finite
positive input/core/head gradient families, parameter updates, restore equality
and every wired profiler. A diagnosed capacity/reserve failure stops enlargement;
select the preceding candidate that both architectures completed. Other failures
stop the campaign. If all candidates pass, report a tested ceiling/lower bound,
not a measured hardware maximum. Same largest joint stable batch is used for
both comparison arms; accumulation1, final partial update executes its real size.

After selecting B, the number of training updates N is fixed. Capture training
updates[2,floor(N/2),N] in each arm. Forecast each arm from its own2/5-update
trials: componentwise maxima u of positive whole ordinary-update increments,
c of whole loop minus those increments, s of model lifetime minus loop,
h of supervisor model time minus reported lifetime, and k checkpoint time.
Admission is1.25*((N-3)*u+3*c+s+h)+k <=600seconds per arm. This empirical forecast
is not an upper-bound theorem; c includes cold work/finalization. Invalid timing
boundaries or failed admission stop. The600-second model watchdog stays active.

Total execution <=1500seconds, plus four <=60-second scorer processes and
<=60seconds for sealing, all within one1800-second campaign clock. Reserve
training/evaluation time before each further capacity candidate. Per frozen
inference/smoke model cap120seconds, finalization120seconds. No thermal,
profiling, hash or evaluator failure is reclassified as a scientific failure.
Build/tests/research preparation are separately recorded, outside campaign cap.

All supported profilers remain enabled: profiling,cudnn and candle-graph all,
unique host/NVTX traces, selected tensor/scalar/gradient records and Nsight CUDA,
cuDNN,cuBLAS,OS-runtime/CPU evidence. Bind raw application labels and Nsight
provenance; check structural validity and capture completeness. Operation/activation
linkage, allocation lifetimes, instrumented physical-memory checkpoints and
device-event intervals remain unwired. Automatic GPU correlation is incomplete.
These profiled_work captures do not support production timing comparisons.

## Frozen checkpoints, evaluation and gates

Run both initial fit/held cohorts at4loops before training. Reconstruct data and
actual input hashes, align canonical logits by physical effect for each of16
orbit types, using audit mappings only outside the model. Compare every map
against fixed lexicographic map0 with elementwise absolute tolerance1e-4 plus
relative1e-5 times the reference magnitude. Test winner agreement only when every member has a positive winning margin
and the minimum exceeds twice the measured maximum absolute logit discrepancy;
tied or numerically ambiguous argmax is not equivariant by itself.
Treatment must pass this numerical premise before either main fit. Legacy's
initial symmetry result is descriptive. No repair or selected seed follows a
failure in this campaign.

Train legacy then treatment to their terminal N update, with no checkpoint or
seed selection. Score the same15streams per arm: initialfit/held4, finalfit/held
at1/2/4/8, finaleffect-zero andquery-zero fit/held at4, cachedvisual final4.
The terminal4loop treatment gate requires initial and final numerical action
equivariance, all16 orbit types correct with minimum true-action margin>=0.001
across all24 map members, final support-order winner agreement, and ablation
integrity. Under the exact symmetry assumption, held-map success is redundant
coverage. State this beside any100% result.

Legacy retains the original descriptive screen thresholds: fit>=254/256,
held>=116/128, demonstrated>=87/96, omitted>=29/32, support-order and controls.
Report both full decisions and all positive/negative outcomes. Paired differences
use identical finite populations and fixed checkpoints, with no seed CI.
No alternative-depth result promotes or selects the model. No training-depth
causal claim follows from evaluation depth changes.

Effect-zero clears only three support displacements; query-zero only desired
displacement. Identical-input groups have balanced labels and hence25% overall
for any deterministic model. The action-token model also has structural ties
in some clamps; do not demand a unique renamed winner there. Report demonstrated
and omitted subsets, CE, margins, histograms, per-map counts and actual input
hashes. External constant, always-missing and exact analytic controls stay outside
learned inference. Perfect missing-action subset alone can reflect a shortcut.

## Integrity, uncertainty and next decision

Freeze producer, both scorers, tests, registration, numerical files, build and
parent seal. Primary and independent implementations reconstruct data/schedule,
F32 features and scalar scores separately; the latter imports neither primary
nor producer. Source/device/training/profile/process facts share the verified
runtime receipt and are not an independent execution. Verify exact588800 consumed
rows, every update/tail/gradient, initializer and checkpoint identity. Reject
unsealed or changed artifacts. Stop telemetry/children before final hashing,
record external manifest digest, and verify every tracked PID is gone.

Exact finite counts have no IID/binomial/bootstrap CI. The fixed seed, repeated
finite rows, action-renaming symmetry, changed optimizer count relative toC17,
slightly different module sizes and privileged reused visual cache are explicit
limits. Report severe wrong-confidence/negative outcomes as well as accuracy.

A valid treatment pass supports only this finite learned-selection component
and permits registering a fresh native image-to-action adapter check. It does
not automatically launch that next experiment. A valid failure calls for frozen
logit/gradient/fit diagnosis before another recipe. A numerical or infrastructure
failure is retained separately and requires a new reviewed correction. The
operator never selects an unregistered dependent training recipe from results.
