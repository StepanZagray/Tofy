# Frozen gradient reconstruction failure capture

Status: completed characterization; failure reproduced on both reconstructions and routes
Date: 2026-09-06 CDT
Evidence class: infrastructure characterization only
Training / model promotion / public ARC authority: none

## Question and known limit

At source `b8599aed5a50ef9cb868fd8d8db893c6f18f50ae`, the single repaired
mask-contract preflight failed with `gradient component reconstruction failed`
before a completed D1 cell. The two ReconstructionCheck objects were computed
but discarded by the generic guard. Which check/route failed, the residual
magnitude, and cause are unknown. The inference that some fixed reconstruction
criterion failed is supported; no CUDA algorithm or training defect is yet
identified.

Parent root: `/home/stepan/Coding/Personal/.tofy-build/v6-mask-contract-preflight-20260906T105609-CDT`.
Report SHA-256: `3b5f0e5e79a89c29e2b0a39e9a546c95843908cc049cf350e8785adf677b403a`.
External manifest SHA-256: `1cb732c54756a0b1cca3428c05a9dc901c2727be83f7e5d3bb9ff4cdad2d0eec`.
Parent binary: `9a422b4adceb65e58a88b7efd7eb22bb01ff01444926d2a103271429102794de`.

## Algebra before compute

For the same forward and trainable-parameter population, differentiation is
linear over the weighted scalar loss in exact arithmetic. Inspection of
`foundation_v2_training_loss_with_event_weights`, `gradient_cell`, and the
accumulation helpers finds matching prediction, gate, latent, encoder CE,
separation, pull, inverse-action, EP, event, Q, and reliability coefficients.
The in-forward rollout is disabled/zero; the dedicated rollout gradient is
added to both full-gradient constructions. Changed and unchanged prediction
terms have the same real-arithmetic coefficients and denominators.

This local algebra does not prove that differently grouped F32 CUDA backward
computations agree within `1e-5`. In particular the diagnostic combines
coefficient/count into one affine while the production positive term has two
affines. That is a candidate numerical confound, not a demonstrated cause.

## Fixed intervention and measurement

Add only observability to the existing module. Return the already-computed
reference L2 and absolute residual L2 for global, AdamW, and Muon routes from
`reconstruction_check`, alongside the existing relative-or-absolute values
and pass bit. Additive fields use serde defaults. On a failed reconstruction
guard, serialize BOTH full and prediction checks into the error that the
existing runner seals in `report.json`. Preserve the generic error prefix.
No additional forward, backward, norm reduction, or model update is permitted;
no objective coefficient, affine grouping, tolerance, ordering, population,
seed, checkpoint, evaluator, backend, batch, or projection parameter changes.

Keep the original preflight CLI and full `[0,1024,2048]` rescore / `[0,1024]`
D1 grid. This avoids a new driver or contradictory partial-spec identity.
If the first guard fails again, existing fail-closed execution stops there.
If it no longer fails, the original bounded preflight may finish; that is
`NOT_REPRODUCED` characterization and does not admit the full diagnostic.

## Exact launch and budget

One newly reviewed, committed, pushed, clean source revision; no dependency
change; exact `cargo build --release --locked --features cudnn` build command
embedded through TOFY_BUILD_COMMAND. Preserve/hash a separate binary. Before
launch, verify parent report/manifest digests above, G's sealed tree, source
and dependency identity, and exact GPU. Use a never-reused root and the same
registered G report, seed, physical 128, and no accumulation/update.

The parent failed in 18.69 seconds. The expected repeated failure is similarly
bounded, but the unchanged preflight retains its 300-second wall limit plus a
330-second external safety stop for sealing. This is intentionally a replay
of the existing grid with additional failure telemetry, not a new 60-second
partial-grid spec. No automatic retry or downstream launch.

## Frozen interpretation and next decision

- A repeated error with finite serialized checks identifies the failed
  reconstruction(s), optimizer route(s), absolute/relative residuals, and
  reference norms. Report every route without selecting the worst only.
- Missing/nonfinite fields or a different prior error are failed infrastructure
  evidence; no model decision follows.
- No repeated failure is NOT_REPRODUCED, not permission to loosen the guard.
- A large discrepancy does not by itself prove an omitted objective; a small
  discrepancy does not by itself prove harmless roundoff. Follow the captured
  values with a separately designed minimum algebra/backward control.
- The existing `1e-5` relative and `1e-6` near-zero absolute criteria, including
  the reference-norm branch threshold, remain unchanged. These numbers are
  integrity thresholds, not a theorem about F32 backward linearity.

The source change and this complete contract require an independent bounded coding review before compute. Opus 5 XHigh
timed out on both the broad and focused attempts without a verdict; Fable 5.1
High remains quota-blocked. GPT-5.6 Sol XHigh is the explicitly identified
available reviewer for this telemetry-only patch under the project delegation
policy. None of those failures is treated as approval. Host tests must demonstrate that deliberate opposing
gradients retain both reference/residual values and serialize both checks on
failure, while the pass/fail boundary is unchanged. Existing diagnostic tests,
compile, strict Clippy, and diff hygiene must pass.

Seal/recheck evidence, verify process cleanup, preserve all positive and
negative findings in ml/tofy, and stop for analysis.

## Source admission

GPT-5.6 Sol XHigh returned GO for exactly this bounded telemetry capture.
Review: `/home/stepan/Coding/Personal/.tofy-build/reviews/gradient-reconstruction-advice-20260906T110051/sol-final.md`.
Review SHA-256: `f13484da6205ef17b8616769f077092aad0a5651b7debdef5e943a7917cb49e7`. The source-excerpt review did not rerun tests;
primary validation passed `cargo check --all-targets --locked`,
`cargo clippy --all-targets --locked -- -D warnings`, all 30 focused diagnostic
tests, formatting and diff hygiene. No threshold or arithmetic change followed
the review. The earlier Opus timeouts and Fable quota rejection supplied no verdict.

## Outcome

The telemetry-only replay reproduced the failure in 18.098100267 seconds.
Both full and prediction checks fail globally and on AdamW and Muon; every
reference is above the 1e-6 branch boundary. All relative residuals are about
2e-4, exceeding the unchanged 1e-5 tolerance. This is failed infrastructure
with usable characterization telemetry, not a valid D1 cell or model result.
No optimizer/EMA update, later anatomy, runtime admission, classifier or public
ARC evaluation occurred. The guard still stops execution before G loss/route
and completed same-forward binding checks.

| Reconstruction | Route | Reference L2 | Residual L2 | Relative residual |
|---|---|---:|---:|---:|
| full | global | 450.406097 | 0.106553927 | 0.000236573012 |
| full | adamw | 340.671967 | 0.0759050325 | 0.00022280974 |
| full | muon | 294.632385 | 0.0747807845 | 0.000253810471 |
| prediction | global | 444.903625 | 0.0932077691 | 0.000209501033 |
| prediction | adamw | 335.658875 | 0.0688860714 | 0.000205226427 |
| prediction | muon | 292.014313 | 0.0627885163 | 0.000215018626 |

Source: `274ad4b7793051ccbfc5d9ac39f9f3ac541d39de` (clean, pushed). Binary: `d7ff974d1e8de8d1d508e6ba890c31447d87f208c18e7690f89a0bab938a642e`.
Dependency: `8e012f25e38f0c597c14268f0c705e504a5b5c28`. Build: `cargo build --release --locked --features cudnn`.
Root: `/home/stepan/Coding/Personal/.tofy-build/v6-gradient-reconstruction-capture-20260906T113302-CDT`. Report SHA-256: `7acf3077244b4aff51a79c6448ec77fa7c4a29d3ff1e0651e6798dd25b7ad044`.
External manifest SHA-256: `a905630ba8e780db0bd1ba3b62e0abf023bf6200b8293fca554f8e85c616217e`. All recursive entries and sidecar verified.
GPU: NVIDIA GeForce RTX 5060 Laptop GPU, GPU-216be468-8184-1801-0563-7c67555dbc45, 610.57.04, 8151 MiB, 2 MiB. Physical batch 128, no accumulation/update.
cuDNN 9.25.0.15, CUDA 13.3.1. NVIDIA_TF32_OVERRIDE and CUBLAS_WORKSPACE_CONFIG unset.
PID 60722 exited 1 and its /proc entry is absent.

Exact command:

```bash
/home/stepan/Coding/Personal/.tofy-build/binaries/tofy-274ad4b7-d7ff974d1e8d-cudnn p2-v6-multibatch-frozen-diagnostic --g-report /home/stepan/Coding/Personal/.tofy-build/v6-multibatch-g-registered-20260906T002436-CDT/report.json --device cuda --output-root /home/stepan/Coding/Personal/.tofy-build/v6-gradient-reconstruction-capture-20260906T113302-CDT
```

## Decision and next falsifier

Do not relax the reconstruction threshold or launch the full diagnostic.
The source algebra matches in real arithmetic; residual magnitude alone does
not identify an omitted loss, harmless roundoff, or backend defect. Candidate
causes include affine grouping, separate backward accumulation, and reduced
precision. Current NVIDIA documentation allows TF32 under default cuDNN math;
this build leaves F32 convolution math at the default. That is an applicability
hypothesis, not evidence that a particular kernel used TF32.

The next cheapest control is a fixed synthetic convolution VJP additivity
characterization, using the same pinned executable with NVIDIA_TF32_OVERRIDE
unset versus 0 and a CPU reference. It can falsify the assumption that default
CUDA backward additivity always meets 1e-5. It cannot uniquely explain this
whole-model failure or authorize different G controls. Freeze this separate
control before execution.

Source: [NVIDIA cuDNN math type reference](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html#cudnnmathtype-t), retrieved 2026-09-06 CDT; `vendor/candle-core/src/cuda_backend/cudnn.rs` at 274ad4b7 sets tensor-op math explicitly only for BF16, leaving F32 default. No current architecture paper is needed to answer this numerical prerequisite; local Tofy literature map through September 4 was consulted.
