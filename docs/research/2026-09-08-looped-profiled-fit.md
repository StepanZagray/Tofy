# Profiled restart of the looped fixed-set fit

Registered September 8 before training outcomes. User has authorized training
again after two interrupted runs. Preserve both partial runs as exploratory;
neither reached the registered decision. This is a fresh initialization, not an
optimizer resume from weight-only checkpoints.

## Claim and learning invariants

Use the [balanced fixed-set contract](2026-09-07-looped-agent-fixed-fit.md): 128
one-step queries (eight layouts times 16 control permutations), seed 0 and data
seed 9173, original initialization hash
`4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802`, identical
model/task bytes, objectives, AdamW, clipping and 1/2/4 loop schedule. Effective
batch 64. The old selected physical batch 33 with a 31-row tail is the starting
capacity candidate, subject to the exact profiled-binary smoke below.

At every existing 25-update check retain the weight checkpoint, accuracy and CE.
Stop at the first >=90% accuracy AND CE<=.35 check, or 1,200 updates/35 minutes.
The cleared-support analytic bound remains exactly 25% accuracy and CE>=ln(4).
Keep the complete log. No automatic extension or public data is allowed. A pass
shows local fitting capacity only; use the separately registered frozen readout
before any generalization or architecture claim.

## Execution differences and profiler preflight

Port the exact looped model/task onto current main ee9546e1, with current clean
candle_graph 1ea5cc5 (0.10.1). Restore the same embedded-provenance build script.
No model, generator, loss or optimizer algorithm change is intended. Record all
source/binary/dependency hashes; the profiler/version/platform differences make
this a new execution, not a bitwise replay claim or a timing comparison.

Build with cudnn,profiling and the dependency's all feature. Set TOFY_PERF_TRACE
to a unique run-owned host trace. Capture the first evaluation forward and
optimizer updates 2,100,1000; on a two-update smoke select only update 2. Capture
all four output statistics, every realized loss, exact pre-clip parameter
gradients in core/policy/value/reward/dynamics families, clipping scalars and
synchronized semantic phases. Legacy TOFY_P2_STEP_PROFILE is not consumed by
this runner; its own phase spans are the applicable timing evidence.

Wrap selected invocations with NVTX range tofy.looped/capture and collect Nsight
Systems CUDA, NVTX, OS-runtime, cuDNN/cuBLAS tracing and process-tree CPU samples.
Use repeat capture ranges with deferred export. Nsight 2026.4.1 is extracted into
a local tools directory; the current perf paranoid level 2 supports process-tree
sampling but not system-wide sampling. No system permissions are changed.

First smoke the exact binary under all these profilers. Recheck physical 34
against the prior 512 MiB sampled reserve, then 33 if 34 is not eligible; increase
one at a time if 34 passes until the reserve fails, capped at 40. Decrease if
needed. Repeat the largest eligible size with effective64/tail accumulation.
This is implementation/capacity evidence only. Verify initialized weights,
dataset identity, finite updates, checkpoint hashes, actual capture structural
validity/completeness, complete gradient manifest, host trace events, Nsight
reports and exact owned-process cleanup before the full run. Infer the run
duration from smoke stage timings and reserve time for export/integrity checks.

## Evidence limits and publication

Candle Graph coverage is profiled_work: labelled output tensors/statistics and
complete declared gradients. Per-operation/activation linkage, logical allocation
lifetimes, physical-memory checkpoints and device-event intervals remain unwired
in this producer. External GPU telemetry supplies sampled whole-device usage,
not allocator peak. Nsight adds observed kernel/transfer/API timing; CPU samples
exclude inaccessible kernel/system-wide work. Do not call this full operation or
memory profiling. Export supported official CSV reports, retain nsys-rep files,
create the matching capture manifest and publish a separate Nsight-bound bundle.
Never modify a finalized original Candle Graph bundle.

The 35-minute model-run cap includes in-process profiling/publication. External
Nsight finalization, CSV exports and integrity checks have a separate ten-minute
cap. Record observed profiler update overhead descriptively; selected and normal
updates are not production-equivalent timing cohorts. Any future treatment uses
the same profiler cadence and settings. This single-seed diagnostic does not
promote a model for ARC or settle the cause of the earlier failed screen.

## Frozen-readout execution addendum

Use this same new profiled CUDA binary for the registered initial/final frozen
readout, replacing the older b1e81e3a executable reference only. Preserve the
registered checkpoint selection, inputs, loops, seeds, metrics and thresholds.
Capture the first evaluation forward under the same supported evidence planes;
there are zero optimizer updates. Keep 300 seconds per checkpoint for model
execution and a separate ten-minute export/integrity budget.
