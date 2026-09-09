# Frozen features support role-selected readouts, not the native policy

C8 completes the registered frozen-feature diagnostic. Selecting the current
agent and goal patches by their **true visible colors** and fitting a linear
readout yields **256/256** correct evaluation actions at both the shared initial
checkpoint and the C7 final checkpoint. This is a task-specific routing oracle,
not learned routing or a general-purpose controller. Initial success establishes
that this recoverability predates C7 training.

Neither CLS probe passes the registered gate: initial **53/256 (20.703125%)**,
final **95/256 (37.109375%)**. The final native policy is still constant action 0,
**55/256 (21.484375%)**, below the panel's best constant **81/256 (31.640625%)**.
These results justify testing a learned spatial readout; they establish no ARC
score, successor-learning gain or architecture promotion.

## Registered population and fixed fitting procedure

The two frozen models see exactly the same 768 new factual known-control queries,
seed 20260915, episode IDs `0x46454154555245 + i`, permutation 0 and distance one.
The first 512 rows fit each probe; the last 256 evaluate it. All four labels occur
in both partitions: fitting `[119,136,133,124]`, evaluation `[55,71,49,81]`.
All queries and full input hashes are unique. The query panel has zero overlap
with the 4,664 unique queries retained across the four registered C5/C6/C7
exclusion artifacts. A collision would have failed the panel rather than replaced
an example. Factual calibration remains normally encoded.

All 992,393 model parameters and ordinary numerical operations are retained.
Extraction uses four loops, physical/effective batch 1, accumulation 1 and **zero
model optimizer updates**. CLS `[128]` and current patches `[64,128]` are the exact
post-final-RMS tensors consumed by the existing heads; native policy logits come
from the same forward. The core is frozen; these CPU probes do receive new
supervised fitting labels and therefore are not the unmodified policy.

For each checkpoint, the CLS probe uses 128 features and the visible-role probe
concatenates 128 features at each true agent/goal cell. The coordinate control uses
`(agent_x,agent_y,goal_x,goal_y)/7`. Each of these five matrices has a real-label
ridge fit and a null fit using the same PCG64 seed-1914 permutation of 512 fitting
labels. Counts are preserved. Means and population standard deviations come only
from fitting rows; scales clamp below at 1e-6. Centered one-hot labels are fitted
in F64 with lambda 0.01, then their fitting mean is restored as the intercept.
Argmax ties choose the lowest action. No normalization, feature, lambda or seed
search occurred. One BLAS thread was verified.

## Complete readout results

| Readout | Fit correct / 512 | Evaluation correct / 256 | Evaluation accuracy 95% interval | Null evaluation correct / 256 |
|---|---:|---:|---:|---:|
| Initial CLS ridge | 174 | 53 | [16.015625%,25.78125%] | 65 |
| Initial visible-role ridge | 512 | 256 | [100%,100%] | 81 |
| Final CLS ridge | 212 | 95 | [31.25%,42.96875%] | 66 |
| Final visible-role ridge | 512 | 256 | [100%,100%] | 74 |
| Coordinate ridge | 512 | 256 | [100%,100%] | 64 |
| Initial native policy | 136 | 71 | [22.265625%,33.203125%] | — |
| Final native policy | 119 | 55 | [16.796875%,26.5625%] | — |

The analytic geometry rule is also 512/512 on fitting and 256/256 on evaluation
rows. All five nulls are <=50%; coordinate ridge is >=95%. Thus the registered
controls pass. All ten ridge residuals are below 7.16e-15 versus the 1e-9 maximum,
and no feature dimension hits the scale clamp. Coefficients, fitting means,
scales, scale distributions and every prediction remain in the
[complete analysis](/home/stepan/Research/_runs/2026-09-09T104547Z-tofy-looped-frozen-features/completed-analysis.json).

Both visible-role probes satisfy evaluation accuracy >=90% and lower 95% advantage
over the resampled best constant >25 percentage points. Their measured advantage
is 68.359375 points, interval [62.5,71.875]. Initial CLS advantage is -10.9375 points,
interval [-19.140625,-4.6875]; final CLS advantage is 5.46875 points, interval
[-2.34375,12.109375]. Neither CLS probe satisfies the conjunction.

Final-minus-initial CLS accuracy is +16.40625 points, interval
[10.15625,22.65625], but the final absolute gate still fails. Role-readout gain is
0 points, interval [0,0]. Initial/final role-minus-native gains are 72.265625 and
78.515625 points, intervals [66.796875,77.734375] and [73.4375,83.203125]. Initial/final
CLS-minus-native gains are -7.03125 and +15.625 points, intervals
[-14.453125,0.390625] and [7.421875,23.4375]. All comparisons are reported; no winner
was selected after viewing them.

Intervals use 10,000 paired whole-evaluation-query PCG64 resamples, seed 1913,
linear 95% percentiles and best-constant reselection within every draw. All draws
have nonzero denominators. The empirical **[1,1]** interval follows from resampling
256 successful rows; it is **not a guarantee of 100% population accuracy**.
Intervals are pointwise descriptive, with no simultaneous guarantee.

The initial native policy chooses action 1 on all 768 queries; final chooses
action 0 on all 768. Native fit/evaluation CE is 1.474369/1.415602 initially and
1.400358/1.395964 finally. These are measured on C8 rather than inferred from C7.
Ridge scores are uncalibrated and receive no CE/probability claim.

## Provenance, execution and verification

- Campaign: `/home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST`.
- Extractor source: `2f2aaa711eb8f39eb823cc4e354288c7b4fbcf42`.
- Dependency: `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a`.
- Binary SHA256: `83150acaf7275bab63204d897972194422b2eca01137c9b7752ade456105ea8e`.
- Shared initial weights: `4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802`.
- C7 final weights: `a983626a888279cbb31651c6abcd1ebf4f808c8927a29a8349d03ffc35bd4e5a`.
- Frozen analyzer SHA256: `0cbe6f14902e692e558c9c56c55d06f23e70f13d962933153d9b6b0eaf42e869`.
- Analyzer fixture SHA256: `4cdeb314df67c0004f7e79fc66832b83599e40ded7897c101708e2b6abb10677`.
- Complete analysis SHA256: `482717449868960d0caf0bfac8d6ac1c69054e7c9feb78edcd92839ecd0ecd07`.
- Launch qualification SHA256: `03e472701ca55d58bdee8bc40b3752487bc2d868c4eb5fafc53eaecfd2584c23`.

The reviewed clean commit was pushed and fetch-checked before launch. Build:

```bash
cargo build --release --locked --offline \
  --features cudnn,profiling,serde_json/float_roundtrip \
  --example looped_agent_probe
```

Primary reports 28 example and six model CPU tests, including exact
old-body/output/head/parameter/gradient parity at depths 1 and 4. All 23 independent
analyzer fixtures and 11 fake-child supervisor lifecycle fixtures pass. Exact
legacy frozen raw files and the full report except elapsed time match C5. The
new one-row implementation smoke is excluded from fitted/model evidence.

The two full extractions each execute 768 forwards; together they execute 1,536.
Their reported model durations are 8.793875 and 8.576938 seconds; supervised totals
are 9.741021 and 9.553472 seconds. This is operational context, not a speed comparison.
Sampled whole-device peak is 382/8,151 MiB, reserve 7,769 MiB; this is not an
allocator peak measurement. Both finish within the independent 300-second model
and 300-second finalization budgets. The RTX 5060 Laptop GPU is the recorded device.

All four GPU invocations retain host traces, first-forward Candle Graph captures,
NVTX and separately Nsight-bound CUDA/cuDNN/cuBLAS/OS-runtime/CPU-sampling evidence.
The feature captures include six tensor/stat seams. Operation/activation linkage,
logical allocation lifetimes, physical-memory checkpoints, device-event intervals
and complete automatic GPU correlation remain unavailable. Gradient evidence is
absent by design in these zero-update evaluations. Host spans are not CUDA kernel
times; profiled-work captures establish no production timing comparison.

An independent reader reconstructs all 9,216 stored probe/native predictions,
coefficients and scales, verifies the 29 manifest-listed files across the audit
and full pair, and finds maximum native-head logit reconstruction error 4.06e-7.
This re-review excludes external exclusions/source/profile siblings; the frozen
analyzer verifies those separately. Full support tensors are not exported, so
support-byte reconstruction remains unavailable; their hashes agree across the
sealed identities. Visible query/target/label bytes are reconstructed independently.
The unchanged analyzer was re-executed for publication and reproduced the complete
analysis exactly.

All **65** recorded campaign/review/verification PIDs, including supervisor-owned
child lists and both Opus attempts, were absent before sealing. The finalized
campaign contains **259 files / 273,376,186 bytes**, all rehashed against the
external [manifest](/home/stepan/Research/_runs/2026-09-09T104547Z-tofy-looped-frozen-features/completed-campaign.manifest.json), SHA256
`ebae6f94562879ed343895dc09e9243ab616900f5f8f5e64b54f7ac408cf9e60`. This is point-in-time integrity, not immutable storage. No further
writes to C8 are authorized by publication.

## Recorded commands

These are historical invocations of now-sealed roots; do not reuse those output
paths. Metadata also retains exact profiler-wrapper invocation and provenance.

Initial extraction:

```bash
/home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST/looped_agent_probe \
  --known-mapping \
  --seed 0 \
  --mode known-features \
  --loops 4 \
  --output-dir /home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST/features-initial \
  --device cuda:0 \
  --batch 1 \
  --effective-batch 1 \
  --max-seconds 300 \
  --updates 1 \
  --eval-episodes 768 \
  --data-seed 20260915 \
  --profile-eval true \
  --checkpoint /home/stepan/Projects/code/.tofy-runs/looped-known-replay-20260909T110206-IST/replay-seed0/initial.safetensors \
  --known-features-exclude /home/stepan/Projects/code/.tofy-runs/looped-known-mapping-20260909T095023-IST/known-seed0/training-stream.jsonl \
  --known-features-exclude /home/stepan/Projects/code/.tofy-runs/looped-known-replay-20260909T110206-IST/replay-seed0/training-stream.jsonl \
  --known-features-exclude /home/stepan/Projects/code/.tofy-runs/looped-known-mapping-20260909T095023-IST/final-frozen/successor-rows.jsonl \
  --known-features-exclude /home/stepan/Projects/code/.tofy-runs/looped-known-seen-20260909T104446-IST/final-seen/successor-rows.jsonl
```

Final extraction:

```bash
/home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST/looped_agent_probe \
  --known-mapping \
  --seed 0 \
  --mode known-features \
  --loops 4 \
  --output-dir /home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST/features-final \
  --device cuda:0 \
  --batch 1 \
  --effective-batch 1 \
  --max-seconds 300 \
  --updates 1 \
  --eval-episodes 768 \
  --data-seed 20260915 \
  --profile-eval true \
  --checkpoint /home/stepan/Projects/code/.tofy-runs/looped-known-replay-20260909T110206-IST/replay-seed0/final.safetensors \
  --known-features-exclude /home/stepan/Projects/code/.tofy-runs/looped-known-mapping-20260909T095023-IST/known-seed0/training-stream.jsonl \
  --known-features-exclude /home/stepan/Projects/code/.tofy-runs/looped-known-replay-20260909T110206-IST/replay-seed0/training-stream.jsonl \
  --known-features-exclude /home/stepan/Projects/code/.tofy-runs/looped-known-mapping-20260909T095023-IST/final-frozen/successor-rows.jsonl \
  --known-features-exclude /home/stepan/Projects/code/.tofy-runs/looped-known-seen-20260909T104446-IST/final-seen/successor-rows.jsonl
```

CPU analysis (choose a new output path):

```bash
python -B /home/stepan/Research/_runs/2026-09-09T104547Z-tofy-looped-frozen-features/frozen_feature_analysis.py \
  --audit /home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST/audit-features \
  --smoke /home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST/smoke-initial \
  --initial /home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST/features-initial \
  --final /home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST/features-final \
  --qualification /home/stepan/Projects/code/.tofy-runs/looped-frozen-features-20260909T115455-IST/launch-qualification.json \
  --source 2f2aaa711eb8f39eb823cc4e354288c7b4fbcf42 \
  --binary 83150acaf7275bab63204d897972194422b2eca01137c9b7752ade456105ea8e \
  --output /absolute/new/c8-analysis.json
```

The generic CLI requires positive `--updates 1`; dispatch performs zero model
updates and empty update logs plus unchanged initial/final/source weight hashes
are required.

## Decision and limits

Test a separately registered learned spatial readout without true-role indices
or a color parser at inference, retaining shared-depth recurrence. The
[next registration](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/registration.md) owns its exact fitting choices, comparator,
new supervision, held-out policy and budget; this result grants no automatic
training or promotion claim. Do not credit C7 for the initial-role success.

Fixed lambda and standardization do not match probe power across different
feature spectra. A lower score would not prove information destruction, and a
failed linear probe does not prove that useful nonlinear information is absent.
One shared label permutation is a coarse leakage control, not a null-distribution
estimate. Privileged spatial selection changes the readout and cannot uniquely
identify the original failure's cause. The new core still lacks learned episode
memory, active probing and ARC integration. The native policy, successor and
planner prerequisites remain separate; the broader autonomous task remains active.

[Registration](2026-09-09-looped-frozen-features.md) ·
[Research synthesis](/home/stepan/Research/_runs/2026-09-09T104547Z-tofy-looped-frozen-features/synthesis.md) ·
[Independent review](/home/stepan/Research/_runs/2026-09-09T104547Z-tofy-looped-frozen-features/findings/final-independent-review.md)
