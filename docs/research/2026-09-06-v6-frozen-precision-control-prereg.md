# Frozen-checkpoint reconstruction precision control

Status: paired characterization completed; all six checks restored with override; independently scrutinized with stated contract wording caveat
Date: 2026-09-06 CDT
Class: frozen-checkpoint precision characterization only

## Claim, baseline and limits before compute

The 274ad4b7 model diagnostic failed both full/prediction reconstructions near
2e-4. The fixed synthetic convolution at dc299bb7 failed default CUDA input/
filter additivity at 2.694e-4/2.721e-4, while CPU passed and disabling TF32 passed
at 4.181e-7/3.047e-7. Every synthetic repeat residual was zero. Its report hash is
f516f9e495e216f18372ad4f7cd1845c260049e3b0688af7dd267c771c1649cb and manifest hash
c7c3431a7db35b60de80202f2f3018967a2e0e4f485c6869ea053f4283567dc2. This motivates
but does not prove the whole-model explanation.

Test the bounded hypothesis that NVIDIA_TF32_OVERRIDE=0 is sufficient to restore
the existing reconstruction criterion on G's exact step-0 checkpoint, train
batch position 0, physical 128, current objective and dedicated rollout gradients.
Real-arithmetic VJP linearity supplies the local identity but no arbitrary F32
error guarantee. No architecture or training claim, new paper claim, optimizer
update, held-out selection or public ARC use follows from this control.

## Single intervention and fixed pair

Two sequential fresh CUDA processes: override absent, then override exactly 0.
Same clean reviewed/pushed source and dependency, same hashed release cudnn test
binary, GPU UUID, config, G seal, snapshot hash, full frozen population, batch,
EP weight, projection seed, masks, coefficients, backward ordering, norms and
thresholds. All parameters must hash identically before and after each arm.

Use only the ignored `frozen_reconstruction_precision_characterization` test.
It binds the original G report and checkpoint, verifies the full population,
loads every parameter exactly and calls the original `gradient_cell` at 0/0.
The warmup is exactly one legacy raw train-batch forward per arm. This differs
from the original full D2/held-out history and is explicitly a new paired
characterization, not an exact replay of the failed preflight.

A cfg(test)-only stop, enabled only by TOFY_FROZEN_RECONSTRUCTION_PROBE=1, captures
both reconstruction objects, actual logit fingerprint, mask binding and false
edit count immediately before the existing guard. It stops regardless of pass
bits so both outcomes are measurable. The production guard, CLI, model,
optimizer and diagnostic admission are unchanged. Original G loss/route controls,
D2/held-out scores, later checkpoints, classifier and runtime admission are
intentionally not executed. Their absence cannot be treated as a pass.

Build: cargo test --release --locked --features cudnn --lib --no-run.
Invocation: exact ignored test with --nocapture --test-threads=1. Record all
probe/environment args, source/dependency/build/binary identities, GPU/software,
G/report/snapshot hashes, never-reused root and per-arm PID/exit/elapsed time.
Cap each process at 60 seconds, pair 120 seconds; stop remaining arms on timeout,
missing/nonfinite telemetry, changed parameters or any infrastructure mismatch.
No automatic retry, altered cap or fixture selection.

## Frozen interpretation and next decision

Report full and prediction checks across global/AdamW/Muon routes, including
all reference norms, absolute residuals, relative residuals and pass bits.
All six references must be at least 1e-6; near-zero cases are inconclusive. The
unchanged 1e-5 relative / 1e-6 near-zero criterion owns pass/fail. Verify count and
fingerprint consistency of the captured mask binding. Forward differences across
precision arms are descriptive, never an integrity failure by themselves.

If the default arm fails and the disabled arm passes all six route checks, the
precision override is sufficient for this fixed checkpoint/batch/execution
history. This supports a backend precision cause for this paired reconstruction
failure, while not uniquely identifying TF32 kernels or excluding algorithm
changes induced by the override. Any mixed or nonreproducing pattern remains
inconclusive as appropriate. A single pair is a bounded mechanism test, not
multiseed promotion evidence. No post hoc tolerance adjustment or CI claim.

After sealing and independent result scrutiny, decide whether a fresh diagnostic
contract should use an explicit precision policy and separately characterized
G comparators. Do not silently run the original full diagnostic with changed
precision: its legacy controls were measured under a different environment.
No training or model promotion is authorized by this result alone.

Required source validation: focused independent review, strict CUDA Clippy,
locked release test compilation, formatting and diff hygiene. The test-only stop
must be absent from production builds and must be explicitly requested by the
ignored test. Capture scripts enforce the external timeouts and sealing.

## Host validation and review provenance

Locked all-target checking, strict CUDA Clippy and all 30 ordinary diagnostic
tests passed. The two CLI review attempts incorrectly tried further delegation
and yielded no final review; they are rejected. A direct bounded reviewer using
the same Sol XHigh model reviews the immutable packet. Its actual verdict and
artifact digest must be recorded in the launch evidence before either arm starts.
The reviewed source packet must match the committed source; a missing GO blocks
launch. No prior failed review attempt is approval.

## Source admission and observed outcome

Direct Sol XHigh source-review SHA-256: `80915b323ce93413bc0bf0dd63aa1c78488a23775dc279901fffc3b15e8fa5ad`. The exact reviewed source diff matched the launch commit. Both rejected CLI attempts supplied no approval.

# Disabling TF32 restores frozen-model reconstruction at step 0

## Bounded finding

For the exact G step-0 checkpoint and train batch position 0 under the frozen
paired warmup, NVIDIA_TF32_OVERRIDE=0 was sufficient to make both full and
prediction reconstruction pass all six route checks. Default CUDA failed all
six. The relative tolerance stayed at 1e-5; every reference norm was nonzero
and well above the 1e-6 near-zero branch threshold. Every parameter fingerprint
was unchanged before/after each arm, and both arms rebound the same sealed G.

| Reconstruction | Route | Default relative residual | TF32 disabled relative residual |
|---|---|---:|---:|
| full | global | 0.0002365730125 | 1.111078057e-06 |
| full | adamw | 0.0002228097424 | 1.035291711e-06 |
| full | muon | 0.0002538104713 | 1.204993784e-06 |
| prediction | global | 0.0002095010641 | 1.047552722e-06 |
| prediction | adamw | 0.0002052264268 | 9.523671357e-07 |
| prediction | muon | 0.0002150186259 | 1.161431628e-06 |

All six default residuals are near 2e-4; all disabled-arm residuals are below
1.3e-6. This is a numerical characterization, not a model-quality improvement.
The original step-0 default residuals from the 274ad4b7 capture were reproduced
to small F32 reduction differences despite the explicitly different warmup.

Precision also changed the forward: the captured false-edit mask count was
496886 with default math and 496914 with the override (+28). Each arm's count
and fingerprint matched its own captured logits/mask binding. Logit and mask
hashes differ across arms. These are initialization outputs, not performance
promotion metrics; no claim of prediction improvement follows. The experiment
changes precision for the whole process, not only backward execution, so it
does not separate forward perturbations from backward kernel behavior.

## Interpretation and limits

The paired intervention supports a backend precision explanation for this
fixed reconstruction failure. Together with the prior fixed-convolution control,
it shows that exact real-arithmetic additivity cannot be assumed to meet the
frozen tolerance under the default backend. It does not identify exact TF32
kernels, exclude algorithm changes induced by the override, or guarantee the
same result at later checkpoints/batches. No repeated model pair or multiseed
promotion claim was registered, so none is inferred.

The test-only stop intentionally occurred before original reconstruction
admission, G loss/route checks, held-out/D2 scores, later checkpoints, anatomy,
classifier and runtime admission. Those remain unmeasured. Its warmup was one
legacy raw train-batch forward, not the original D2/held-out allocator history.
The production CLI and guard are unchanged. No optimizer, EMA, public ARC or
training update occurred. The prior G model-quality failure remains unchanged.

## Exact provenance

- Source: `1ffa75917c5748e19605336c7fe5b792746ca7b3` (clean and pushed).
- Dependency: `8e012f25e38f0c597c14268f0c705e504a5b5c28` (clean and pushed).
- Binary SHA-256: `2a87c29313fc814d1cbad9f81ac03277252edd814bcb9124f7c8a8f72448120e`.
- Build: `cargo test --release --locked --features cudnn --lib --no-run`; CUDA/cuDNN and release profile verified in build identity.
- Root: `/home/stepan/Coding/Personal/.tofy-build/v6-frozen-reconstruction-precision-20260906T120014-CDT`.
- Report SHA-256: `4dacaab638a5f8caa40689e444ae1afcea04a67a9c2fe9295da48b8bae8c6b03`.
- External manifest SHA-256: `d58ddfd624edcb00d77659fbaa3104a25ba73bd95d73f5ca9ebb68afb980136f`.
- G report SHA-256: `03f645a5cccfbd4dcf72bf9927ac15589dc8fa579c6330f254101ed70c18789f`.
- G external manifest SHA-256: `900488b44e0f1623513234839f0f777d2c1d052ec3c46458fb714383868748d4`.
- Step-0 checkpoint SHA-256: `0446ba05f4af1cc0603086bd10e2c38c23b9931473bb5ec3cf4536ca026ffa79`.
- Census SHA-256: `062cc3ebcd1f0b2cf9dfe39ae255c58f930400eb4a28107d559add4fbdab06c3`.
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, GPU-216be468-8184-1801-0563-7c67555dbc45, 610.57.04, 8151 MiB, 2 MiB; cuDNN 9.25.0.15, CUDA 13.3.1.
- Physical batch 128; no optimizer/EMA step or training accumulation.
- Default/disabled elapsed: 21.130977426/22.097731387 seconds, each below 60-second cap.
- PIDs 78024 and 78137 exited 0; both /proc entries verified absent.

Every recursive manifest entry, external digest/sidecar and binary hash verified.
The sealed report stays complete_pending_analysis; this external analysis owns
the decision and does not rewrite the point-in-time seal.

Exact binary invocation for each arm:

```bash
/home/stepan/Coding/Personal/.tofy-build/binaries/tofy-frozen-precision-test-1ffa7591-2a87c29313fc-cudnn --ignored --exact p2::multibatch_frozen_diagnostic::tests::frozen_reconstruction_precision_characterization --nocapture --test-threads=1
```

Both set TOFY_FROZEN_RECONSTRUCTION_PROBE=1, TOFY_PRECISION_G_REPORT to the G
report above, and TOFY_PRECISION_PROBE_ROOT to their unique arm directory.
NVIDIA_TF32_OVERRIDE is absent in cuda_default and exactly0 in cuda_tf32_off.
Other captured CUDA/CUBLAS environment settings are identical and unset.

## Complete reference and residual values

| Arm | Reconstruction | Route | Reference L2 | Absolute residual L2 |
|---|---|---|---:|---:|
| cuda_default | full | global | 450.4061279 | 0.1065539345 |
| cuda_default | full | adamw | 340.6719971 | 0.07590503991 |
| cuda_default | full | muon | 294.6323853 | 0.07478078455 |
| cuda_default | prediction | global | 444.903595 | 0.09320777655 |
| cuda_default | prediction | adamw | 335.6588745 | 0.06888607144 |
| cuda_default | prediction | muon | 292.0143127 | 0.06278851628 |
| cuda_tf32_off | full | global | 450.3165283 | 0.0005003368133 |
| cuda_tf32_off | full | adamw | 340.618042 | 0.0003526390356 |
| cuda_tf32_off | full | muon | 294.5578613 | 0.0003549403918 |
| cuda_tf32_off | prediction | global | 444.8038635 | 0.0004659554979 |
| cuda_tf32_off | prediction | adamw | 335.5940247 | 0.00031960872 |
| cuda_tf32_off | prediction | muon | 291.9367981 | 0.0003390646307 |

## Decision and next falsifiable step

Retain the fixed reconstruction tolerance and adopt an explicit precision
policy for subsequent gradient measurement. Do not silently apply the override
to the old full diagnostic: its legacy G controls and interpretation were frozen
under default math, and this pair demonstrates changed predictions/masks.

The next implementation must bind precision into diagnostic provenance/identity,
keep masks attached to their exact forward, and characterize comparisons with
G under their original precision separately. A new bounded admission preflight
must pass its own same-precision reconstruction and baseline controls before
any full diagnostic or treatment result is claimed. This precision-policy
integration is not implemented by the present test-only control.

[Current NVIDIA math-type documentation](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html#cudnnmathtype-t)
documents TF32 permission under default math and the disabling override; retrieved
2026-09-06 CDT. Actual empirical support comes from the sealed synthetic and
frozen-checkpoint outputs, not documentation alone. The local Tofy literature
map through September 4 was retrieved; no new architecture paper is required
for this numerical prerequisite.

## Contract wording caveat from independent scrutiny

The fixed-pair paragraph listed “masks” among invariants, while its interpretation
section explicitly anticipated precision-induced forward differences. The actual
false-edit masks differed. Read the supported result as a whole-process precision
intervention with the same mask-construction rule, not as an experiment holding
numerical activations and masks fixed. The archived preregistration is retained
unchanged; this is an explicit outcome qualification, not a revised success gate.

Independent result scrutiny artifact: `/home/stepan/Coding/Personal/.tofy-build/reviews/frozen-precision-20260906T114709-CDT/result-scrutiny.md`; SHA-256 `2dccd4c5919b18b4d6ffb2416bfa1740bdda29166f24e7adfd00631227b9aa8f`. All sealed entries and exact PID cleanup verified. No further experiment launched.
