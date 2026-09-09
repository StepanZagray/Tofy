# Looped fixed-set fitting passes; generalization fails

Completed September 8, 2026 IST. The from-scratch 992,393-parameter shared-depth
transformer fitted 120/128 fixed queries (93.75%, CE 0.181683) at update 1150,
the first registered passing checkpoint. Fresh-layout accuracy remained 28.9%
with familiar mappings and 28.5% with held-out mappings, versus exactly 25% with
cleared history. Neither advantage has a positive 95% lower confidence bound.
This establishes local fitting reachability, not useful rule inference or ARC
performance. No public levels, pretrained LLM weights, LLM controller or
pretrained teacher model were used. Targets come from the procedural simulator
and oracle. The current core still lacks learned persistent episode memory,
active probe selection, hidden-goal learning and an ARC action adapter.

## Evidence identity and execution

- Source: `b22ce0684c3d8b49ce1c46e9e5404eb746d51153`, clean and pushed at launch.
- Candle Graph: `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a` (0.10.1).
- Binary SHA-256: `209bad33e770b26a20be4cb43eadfeeddd36f80b15f9b889f64531a5f5c6b793`.
- Campaign: `/home/stepan/Projects/code/.tofy-runs/looped-profiled-fit-20260908T2156-IST`.
- Training run: `fit-seed0`; initial/final readouts: `frozen-initial`, `frozen-final`;
  additional final-checkpoint readouts: `depth-1`, `depth-2`, `depth-8`.
- Initial weights: `4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802`.
- Final weights: `0e51cc316ded72b6b07ca8b4f34bdb9fc5d7bf8cac73ab683250481346924cf9`.
- Fixed dataset: `3fcaf3796ee2b900ac647c3fd44743d1b34da1bca5c64ecaee0c7f794b3849d0`.
- Model/task source bytes are identical to the original looped implementation;
  profiling and evaluator diagnostics were added. This is not a bitwise replay
  or a timing comparison with the old environment.

The [registration](2026-09-08-looped-profiled-fit.md) fixes eight layouts × sixteen
training mappings, initialization seed 0, data seed 9173, AdamW learning rate
0.0003, weight decay 0.01, clip 1, and policy/value/reward/dynamics coefficients
1/0.1/0.1/0.5. Training cycles 1/2/4 loops; each loop applies the same two
pre-RMS transformer blocks. Width 128, four heads, maximum inference depth 8.
The carried state has input recall and final-only RMS normalization.

The exact profiled CUDA preflight selected physical batch **33**, tail **31**,
accumulation **2**, effective batch **64**. Batch 34 completed with only 440 MiB
sampled reserve, below the registered 512 MiB; two batch-33 preflights passed.
The fit consumed 73,600 examples (575 passes over 128), taking 1762.35 seconds
including in-process checks/profiling. Sampled GPU peak was 7519/8151 MiB,
maximum temperature 68°C. This is whole-device sampled usage, not allocator peak.

Exact build command, from the clean source checkout:

```bash
CARGO_TARGET_DIR=/home/stepan/Coding/Personal/.tofy-build/v6-recipe-repair/target \
TOFY_BUILD_COMMAND='CARGO_TARGET_DIR=/home/stepan/Coding/Personal/.tofy-build/v6-recipe-repair/target cargo build --release --locked --offline --features cudnn,profiling,serde_json/float_roundtrip --example looped_agent_probe' \
cargo build --release --locked --offline --features cudnn,profiling,serde_json/float_roundtrip --example looped_agent_probe
```

Exact supervisor invocation (the root is intentionally never reusable):

```bash
python3 /home/stepan/Projects/code/.tofy-runs/looped-profiled-fit-20260908T2156-IST/supervise.py \
  --binary /home/stepan/Projects/code/.tofy-runs/looped-profiled-fit-20260908T2156-IST/looped_agent_probe_roundtrip \
  --name fit-seed0 --mode fit --batch 33 --seconds 2100
```

The saved `fit-seed0.process.json` and `fit-seed0/metadata.json` record the full
Nsight wrapper and model argv, including 1200-update cap, captures 2/100/1000,
evaluation capture, unique `TOFY_PERF_TRACE`, and repeat/deferred Nsight export.
The registered first-pass stop ended at 1150. Weight files contain no optimizer
state; do not describe a later load as an optimizer resume.

## Frozen readout and independent prediction gates

Both checkpoints used zero updates, four loops, batch one, identical 32 fresh
layouts, all 24 mappings and the same input hashes/labels. Every cleared input
is identical within a layout; balanced targets give exactly 25% for any
deterministic context-free policy and cross entropy at least ln(4).

| Measurement | Familiar mappings | Held-out mappings |
|---|---:|---:|
| Initial factual accuracy | 128/512 = 25% | 64/256 = 25% |
| Final factual accuracy | 148/512 = 28.90625% | 73/256 = 28.515625% |
| Cleared-history accuracy | 128/512 = 25% | 64/256 = 25% |
| Wrong history, scored against real rule | 115/512 = 22.46094% | 61/256 = 23.82813% |
| Factual minus cleared, 95% CI (percentage points) | +3.90625 [-2.92969, 11.52344] | +3.515625 [-1.953125, 8.984375] |
| Factual minus wrong-real, 95% CI (percentage points) | +6.44531 [-5.66406, 18.55469] | +4.6875 [-4.6875, 14.0625] |
| Exact successor frames | 2/128 | 1/128 |
| Copy-current-frame exact baseline | 42/128 | 42/128 |
| Vacated pixels correct | 5504/5504 | 5504/5504 |
| Destination pixels correct | 1567/5504 = 28.47% | 1648/5504 = 29.94% |
| Reward recall (cutoff 0.5) | 0/16 | 3/16 |
| Reward false positives | 25/112 | 26/112 |
| Value MSE | 0.014863 | 0.015314 |
| Constant-0.9 value MSE | 0.010077 | 0.010077 |

Value scope correction (September 9): every training value target is 1, whereas this
evaluator mixes distances 1 and 2–10. The reported MSEs are correct but do not test
in-distribution constant-target fitting. See the [matched coverage report](2026-09-09-looped-coverage-results.md) for a separate one-step value check.

Intervals use the preregistered 10,000 paired whole-layout bootstrap draws,
PCG64 seed 1907, percentile 95% with linear quantiles. Both generalization gates
fail. The anatomy covers all 32 prediction queries and all four actions per
split; older one-row anatomy must not be substituted. Prediction-input hashes
are not saved, so prediction parity uses seeds and saved labels, unlike the
hash-verified policy input parity.

Support changes policy argmax on 385/512 and 192/256 final rows. Median maximum
probability changes are 0.841/0.778, so the policy is context-sensitive but
usually wrong. Low generalization is not an unwired policy placeholder; all
five parameter families have captured gradients. Combined gradients do not
establish that policy-specific credit reaches the shared representation well.

## Depth diagnostic and advisor correction

| Final inference loops | Familiar accuracy | Held-out accuracy | Model elapsed seconds |
|---:|---:|---:|---:|
| 1 | 136/512 = 26.5625% | 73/256 = 28.515625% | 7.85 |
| 2 | 134/512 = 26.171875% | 72/256 = 28.125% | 10.93 |
| 4 | 148/512 = 28.90625% | 73/256 = 28.515625% | 17.27 |
| 8 | 140/512 = 27.34375% | 71/256 = 27.734375% | 29.58 |

All input hashes, labels and weights match; cleared accuracy is 25% throughout.
All depths are reported without selecting a winner or making a significance
claim. More loops do not rescue this checkpoint. These profiled invocation
durations are descriptive, not production-equivalent speed measurements.

Claude **Opus 5 xHigh** actually returned a next-experiment review. Its proposal
to compare matched fixed versus resampled episodes is useful, but its proposed
`true - follows_presented_wrong_rule` endpoint is invalid. Wrong-rule construction
rotates every direction modulo four, a bijection within each split. Thus the
wrong-input/target population is the factual population reordered: both aggregate
accuracies are equal for any deterministic policy, including a perfect one.
An independent code audit and saved row hashes verified this fact. The correct
contrasts are factual minus cleared and factual minus wrong-history predictions
scored against the original real rule, alongside absolute correctness.

## Profiling integrity and limitations

All current-binary fit/frozen/depth bundles passed hashes, semantic verification,
structural validity and declared capture completeness. The fit captures include
all 44 exact parameter entries at updates 2/100/1000. Captured update durations
were 4.44/3.71/4.84 seconds; ordinary medians at loops 1/2/4 were
1.16/1.39/1.85 seconds. This is descriptive overhead, not a causal timing study.

Nsight 2026.4.1 retained CUDA kernels, copies, NVTX, cuBLAS, OS-runtime and
process-tree CPU samples, plus supported official CSV exports. cuDNN was enabled;
this model has no convolutions. Newly published bundles bind raw Nsight reports
and manifests without changing original captures.

Two explicit qualifications remain. First, `serde_json/float_roundtrip` is
required on both producer and Candle Graph CLI: the default parser changed tiny
f64 gradient values by a ULP and failed strict semantic verification. Old failed
preflights are preserved and excluded, not repaired by weakening checks. Second,
Candle Graph 0.10.1's literal NVTX-name matcher rejects the official CSV's leading
default-domain colon and foreign cuBLAS ranges. Automatic correlation remains
incomplete. A separate check verified every declared application label exactly
once with GPU work, removing only that known colon; raw reports remain unchanged.
No automatic per-phase attribution or clock-alignment claim is made.

Per-operation/activation linkage, logical allocation lifetimes, allocator memory
checkpoints and device-event instrumentation remain unwired. System-wide CPU
sampling is unavailable; process-tree sampling works. Host spans are not CUDA
kernel times. A tiny attention key-bias gradient is expected from softmax's
invariance to adding the same scalar to every score, not evidence of a dead core.

## Decision

Do not scale this checkpoint or deploy its learned planner. Local fitting is
reachable; generalizable rule inference is unresolved. The subsequent
[matched episode-coverage screen](2026-09-09-looped-coverage-results.md)
completed on September 9 and failed every policy promotion gate.
The old 256-update varied-data result is budget/distribution-confounded and is
not its control. Current loop-stability papers supply hypotheses, not evidence
that norm placement caused this failure. No ARC score improvement is established.

## Final artifact seal

The campaign is complete; no training, evaluation, profiler or advisor process
is intentionally active. All 42 recorded process IDs were absent at cleanup,
and the GPU compute-process query was empty. The finalized tree contains 1058
files (516,547,499 bytes); every file hash was reverified. Campaign manifest SHA-256:
`d8427c136e582654853cf1a19fb4ac70e75295b1e315a942deba8f315f478e45`.
The digest is also retained outside the campaign in the research run. This is
point-in-time integrity, not immutable storage. Model reports remain sealed;
the outer campaign status records completed analysis and evidence classes.
