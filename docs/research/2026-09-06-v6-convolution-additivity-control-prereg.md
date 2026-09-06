# Synthetic convolution backward additivity control

Status: source reviewed; strict CUDA Clippy passed; release build pending
Date: 2026-09-06 CDT
Class: synthetic backend characterization; no model/diagnostic admission authority

## Claim and derivation

The 274ad4b7 frozen-model capture found relative reconstruction residuals of
0.000205–0.000254 on both routes. This does not identify their cause. Test the
weaker prerequisite: for one fixed F32 convolution and loss graph, does backward
of the sum agree with the sum of two backwards within the diagnostic's existing
1e-5 relative / 1e-6 near-zero absolute criterion? Differentiate with respect to
input and filter separately. In exact arithmetic the VJP is linear in the
upstream gradient. F32 evaluation and TF32 operand quantization need not preserve
that identity within an arbitrary tolerance.

NVIDIA's current cuDNN API math-type documentation permits TF32 under default
math and documents NVIDIA_TF32_OVERRIDE=0 to disable it. Vendored Candle leaves
F32 convolution descriptors at default math. These facts motivate an intervention;
they do not establish which kernel the failed model run used.

Source: https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html#cudnnmathtype-t
Retrieved: 2026-09-06 CDT. Installed cuDNN 9.25.0.15, CUDA 13.3.1.

## Frozen fixture and comparisons

Use the ignored test `convolution_backward_additivity_characterization` only.
Fixture: host LCG dyadic F32 values, seeds 17/29/41/53 for input/filter/two
upstream arrays; input [8,32,16,16], filter [32,32,3,3], stride/padding/dilation/
groups = 1/1/1/1. No dataset, model checkpoint, optimizer, EMA or ARC access.
Physical synthetic batch 8; no gradient accumulation training schedule applies.
For the same single attached convolution forward, scalar losses are sum(y*a)
and sum(y*b). Backward their sum twice, then backward each term separately and
add parameter gradients using the existing helper. Report all reference norms,
absolute/relative residuals and fixed pass bits per input/filter. Check both
fixture tensors remain unchanged. Do not assert that additivity must pass: test
success means valid telemetry, and the JSON owns the numerical outcome.

Three fixed processes, sequential GPU work: CPU, CUDA with TF32 override unset,
CUDA with NVIDIA_TF32_OVERRIDE=0. Same clean reviewed/pushed source, exact locked
release test executable compiled with cudnn, dependency revision, fixture and
GPU. No repetition or adaptive fixture selection. Capture the environment,
commands, exact source/dependency/build/binary hashes, GPU, process and exit
status outside a never-reused root. Seal outputs after every process exits.
Build command: cargo test --release --locked --features cudnn --lib --no-run.
Bound each process to 60 seconds, total device budget at most 120 seconds; a
process failure/timeout is failed infrastructure and stops remaining arms.

## Interpretation and stopping

Apply the unchanged reconstruction criterion independently to input/filter and
report all results without choosing a subset. Both same-graph repeat controls
must pass for an additivity attribution. If default CUDA fails either additivity
check while CPU and TF32-disabled CUDA pass both, the override is sufficient to
restore the fixed tolerance for this synthetic graph. It supports a backend
precision explanation as a next hypothesis for the model; it does not identify
exact kernels, exclude all algorithm changes induced by the override, uniquely
explain the full model, or admit new model diagnostics/training. Any other pattern
is mixed, unsupported, or inconclusive as appropriate. Do not change tolerance.
One fixture is a counterexample/control, not a distribution-level accuracy claim;
no confidence interval or promotion claim is justified. No checkpoint selection.

Freeze a subsequent whole-model control only after analyzing this result. The
original G controls and full diagnostic remain unchanged. Required validation:
focused source review, formatting, locked CUDA test compilation and strict CUDA
Clippy. This test is ignored in routine CPU suites and has no production path.

## Admission review

GPT-5.6 Sol XHigh returned GO on the exact source excerpts and contract.
Review: `/home/stepan/Coding/Personal/.tofy-build/reviews/convolution-control-20260906T113807-CDT/final.md`. SHA-256: `9279da7d5fa2affd5834a035d171e4f5067ff1400f617e8fe6133867a86ca770`.
Source-only review did not rerun tests. Strict CUDA Clippy passed locally.
As clarified before any arm launch, missing or reference norms below 1e-6
for either input or filter make the characterization inconclusive regardless
of near-zero pass bits. The launcher must enforce all three timeouts, exact
environment absence/value, fixture geometry, finite numbers and nonzero norms.
