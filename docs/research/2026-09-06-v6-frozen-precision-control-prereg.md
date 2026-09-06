# Frozen-checkpoint reconstruction precision control

Status: design frozen; host validation passed; independent source review pending
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
