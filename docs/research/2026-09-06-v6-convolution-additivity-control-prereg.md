# Synthetic convolution backward additivity control

Status: completed; default CUDA failed, CPU and TF32-disabled CUDA passed
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

## Outcome

# Default CUDA violates the fixed convolution additivity tolerance

## Bounded finding

On the one preregistered synthetic convolution, the default CUDA backward
failed the fixed 1e-5 additivity tolerance for both input and filter. Disabling
TF32 restored it; CPU also passed. Every same-graph repeated backward had zero
residual, every reference norm was nonzero (about 1960–1970), and both fixture
tensors remained unchanged. All three processes exited successfully.

| Arm | Input relative residual | Filter relative residual | Repeat residuals |
|---|---:|---:|---:|
| cpu | 1.73907786e-07 | 1.12640845e-06 | 0, 0 |
| cuda_default | 0.000269405847 | 0.000272102774 | 0, 0 |
| cuda_tf32_off | 4.1812145e-07 | 3.04660142e-07 | 0, 0 |

The registered favorable pattern is satisfied. NVIDIA_TF32_OVERRIDE=0 is
sufficient to restore additivity within the fixed tolerance for this fixture.
This falsifies a universal assumption that default F32 cuDNN backward additivity
must meet 1e-5. It supports the precision hypothesis for Tofy's observed 2e-4
reconstruction error, but does not identify exact kernels, exclude algorithm
changes induced by the override, uniquely explain the full model, or authorize
training/model promotion. A single fixture is a numerical control, not a
statistical distribution-level result. The unused optimizer route for each
single tensor has a structural zero norm; the nonempty route and global norm
own the stated test result.

## Provenance and verification

Source: `dc299bb77430e4b67e621f7f1a79977d23951661` (clean and pushed); dependency `8e012f25e38f0c597c14268f0c705e504a5b5c28`.
Binary SHA-256: `e2385b6f1b6e949189fa23bd96b5a86171b5faf62c958f4fd235816aedd61d67`.
Build: `cargo test --release --locked --features cudnn --lib --no-run`. The pinned test executable records clean/pushed
source, dependency, CUDA/cuDNN features and release profile in build-identity.txt.
Root: `/home/stepan/Coding/Personal/.tofy-build/v6-convolution-additivity-20260906T114328-CDT`.
Report SHA-256: `f516f9e495e216f18372ad4f7cd1845c260049e3b0688af7dd267c771c1649cb`.
External manifest SHA-256: `c7c3431a7db35b60de80202f2f3018967a2e0e4f485c6869ea053f4283567dc2`. Every recursive entry and sidecar verified.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU, GPU-216be468-8184-1801-0563-7c67555dbc45, 610.57.04, 8151 MiB, 2 MiB. cuDNN 9.25.0.15; CUDA 13.3.1.
CPU/default-CUDA/TF32-off elapsed: 0.064417,
0.466984, 0.417433 seconds.
PIDs 68067, 68085 and 68094 exited and /proc entries are absent.
Physical synthetic batch 8; no optimizer, EMA, training accumulation or ARC data.

All arms used this exact command with TOFY_PRECISION_PROBE_DEVICE=cpu/cuda/cuda
and NVIDIA_TF32_OVERRIDE absent/absent/0, respectively:

```bash
/home/stepan/Coding/Personal/.tofy-build/binaries/tofy-conv-test-dc299bb7-e2385b6f1b6e-cudnn --ignored --exact p2::multibatch_frozen_diagnostic::tests::convolution_backward_additivity_characterization --nocapture --test-threads=1
```

## Decision and next falsifier

Retain the existing diagnostic thresholds. Next test the single G step-0
checkpoint and original gradient_cell under a fixed paired warmup with override
absent versus 0, capturing all route residuals before the integrity guard. This
requires a new test-only characterization contract because the original G
controls were measured under another precision environment and must not be
silently bypassed as if they passed.

The source-only Sol XHigh review approved the control, but did not rerun tests.
Primary validation passed strict CUDA Clippy and locked release test compilation;
actual numerical claims above come from sealed outputs, not reviewer opinion.
The local library's latest architecture survey was retrieved; no new paper is
needed for this narrow backend control. [NVIDIA's current math-type reference](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html#cudnnmathtype-t)
documents default TF32 permission and the disabling override; retrieved
2026-09-06 CDT. Exact kernel selection remains unmeasured.
