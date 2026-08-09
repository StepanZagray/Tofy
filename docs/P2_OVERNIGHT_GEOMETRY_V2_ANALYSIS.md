# P2 geometry-isolation overnight run analysis

Date analyzed: 2026-08-09
Captured pod: `ikkry5s9gidt85`
Run: `runs/p2/ab-sigreg-geometry-v2`
Status: **incomplete experiment; no A/B decision and no promoted arm**

## Executive conclusion

The overnight queue did not complete a valid SIGReg geometry A/B pilot. Both seed-1
arms trained successfully through update 1,000, but the `pre-rms-spatial` treatment
aborted during its update-1,000 CUDA evaluation. The control continued alone through
update 2,000 and completed both evaluations. Because the treatment has no held-out
report at either decision point, the preregistered gate cannot be evaluated and the
control-only numbers cannot answer the geometry hypothesis.

The most useful scientific result is negative but narrower: in the control, the
evaluator reported stronger aggregate/random-one-step action dependence, higher raw
and normalized held-out latent rollout error, and a representation still below the
effective-rank noncollapse floor. From update 1,000 to 2,000, reported aggregate
true-action MSE increased 4.51x, raw horizon-8 MSE increased 7.85x, and normalized
horizon-8 MSE increased from 1.441 to 4.452 even though both preregistered action
gates passed at 2,000. Passing those two action gates was therefore insufficient
evidence of rollout stability or noncollapse.

The most useful operational result is that pairing two full-batch training arms on one
L40S was not worthwhile. It improved aggregate update throughput by only 2.36% over
the observed single-arm phase, saved about 156 seconds versus two estimated sequential
1,000-update phases, and left shared-device state as an uncontrolled variable when
the treatment evaluation failed. Future arms should be run sequentially, at minimum
for evaluation, to improve attribution and recovery.

## Provenance and preservation

| Field | Verified value |
|---|---|
| Tofy commit | `0a3f8205fbd6ac61b9c79ded2c04247104df21a8` |
| `candle_graph` commit | `c9fa15ee917f9dab96bb070cbbcd5ce8dfdb5f48` |
| Exact release binary SHA-256 | `682dc1e9e783bc3170d677db4079b1681e853cc18ada87167565b5a0d9afbb81` |
| Control config SHA-256 | `a089d993ea1ab17c08e83b964f7e4945f2d0e923971e55f2ba7667b1a3b8ca4c` |
| Treatment config SHA-256 | `0757964361b53d96523d707c92d6e92b9ef975db58ed6e7bc82bd8f9e121c715` |
| Shared config-contract SHA-256 | `b9c7f2e5cd509977dbfdd6d29445539cab9050cca03239591cc546e3ece62c92` |
| GPU | NVIDIA L40S, 46,068 MiB |
| Driver / CUDA / cuDNN | 580.159.03 / 13.0 / 9.14.0.64 |
| Rust / Cargo | 1.97.1 / 1.97.1 |
| Build status | 0; release build completed in 3m16s |

The two arm configs have the same comparison-contract hash after removing only the
preregistered SIGReg geometry fields and output path. The source repositories were
clean at capture. The preserved archive is
`runs/p2/_pod_handoffs/ikkry5s9gidt85-20260809/tofy-analysis-ikkry5s9gidt85-20260809.tar.gz`
with SHA-256
`b5b778d7d2600bbd7fffe7cc9a8ff7dd9240e83dfa0951c8b930bfee562ec1c4`.
All 113 entries in its internal `FILES.sha256` manifest verify locally, including
checkpoints, optimizer states, trainer states, logs, telemetry, profiles, the exact
binary, and environment metadata.

Read-only audits requested from Composer 2.5 and GPT-5.6 Luna High were rejected
before execution because that external-agent account was out of usage. A separate
GPT-5.6 Sol read-only review did execute successfully. It independently reproduced
the arithmetic, checksum remapping, gate error, and no-decision conclusion, and it
identified overconfident CUDA and cross-checkpoint interpretations that were corrected
in this report. All retained claims were then checked against the local raw artifacts.

The control's update-1,000 and update-2,000 evaluation manifests also verify after
mapping their original absolute pod paths to the local extracted tree. The treatment
correctly has no evaluation manifest or arm manifest because its first evaluation did
not finish. No core dump was present in the captured pod filesystem.

## Frozen experiment contract

Both arms used fresh seed-1 initialization, dynamics-only training, physical batch
1,024, accumulation 1, 2,000 scheduled updates, shuffled episodes, randomized depth
up to 8 outer by 2 inner recursion steps, final-outer-only supervision, and a 32,768
SIGReg row cap. Held-out evaluation used seed 424242, 64 synthetic episodes,
deterministic PTRM K=1, zero PTRM noise, and one ensemble member.

- `control`: post-RMS, pooled 2x2 spatial cells.
- `pre-rms-spatial`: pre-RMS, unpooled 8x8 spatial cells deterministically
  subsampled to the same 32,768-row cap.

The treatment adds no parameters; both reports show 413,798 parameters. Event, Q,
rollout, prefix, and reliability losses were disabled. These are synthetic diagnostic
runs with `research_claim=false`, not public ARC results.

## Exact timeline

All timestamps are UTC on 2026-08-08.

| Arm | Stage | Start | Finish | Wall time | Status |
|---|---|---|---|---:|---:|
| control | train 0->1,000 | 20:21:19 | 22:11:29 | 6,610s (110m10s) | 0 |
| treatment | train 0->1,000 | 20:21:19 | 22:11:35 | 6,616s (110m16s) | 0 |
| control | eval 1,000 | 22:11:29 | 22:23:44 | 735s (12m15s) | 0 |
| treatment | eval 1,000 | 22:11:35 | 22:23:10 | 695s (11m35s) | 134 |
| control | train 1,000->2,000 | 22:23:44 | 23:20:10 | 3,386s (56m26s) | 0 |
| control | eval 2,000 | 23:20:10 | 23:31:58 | 708s (11m48s) | 0 |

The captured work ended after 3h10m39s. The supervisor exited 1 immediately after
the paired seed-1 stage returned treatment status 134. Consequently, it did not run
the pilot gate, seeds 2/3, or any 4,000-update extension.

The capture at 08:03:32 on 2026-08-09 found no trainer or evaluator. The pod had
therefore been idle for about 8h31m after the supervisor returned, or 72.8% of the
11h42m interval from experiment start to capture. This directly violated the goal of
keeping the rented accelerator useful overnight; fail-stop supervision without a
recovery queue was the larger utilization loss than the 2.36% pairing gain.

## GPU utilization and throughput

The global GPU telemetry was sampled every 15 seconds. Because both arm-specific
telemetry files observed the same physical device, one control stream was used to
avoid double-counting.

| Window | Mean util. | Zero-util. samples | Mean / peak memory | Mean / peak power |
|---|---:|---:|---:|---:|
| paired train 0->1,000 | 99.2% | 0.5% | 29,495 / 29,798 MiB | 120.2 / 175.34 W |
| paired eval 1,000 | 12.2% | 60.4% | 5,651 / 34,754 MiB | 113.5 / 126.54 W |
| single control train 1,000->2,000 | 92.3% | 4.0% | 14,779 / 15,348 MiB | 119.1 / 151.29 W |
| single control eval 2,000 | 3.3% | 59.6% | 2,880 / 17,315 MiB | 111.2 / 119.97 W |
| telemetry-covered workload | 85.7% | n/a | n/a / 34,754 MiB | 118.9 / 175.34 W |

The evaluation's low sampled GPU utilization is consistent with CPU-side generation,
bootstrap summaries, and many small synchronized GPU operations. It is not evidence
that the process was idle.

Peak observed memory left 11,314 MiB free, and the error was
`CUDA_ERROR_ASSERT`, not an allocation failure. Ordinary VRAM exhaustion is therefore
not supported. Telemetry cannot exclude a library or concurrency defect involving
workspace allocation, but it does exclude the simple claim that the two processes
filled the L40S. The post-run device report also recorded zero volatile and aggregate
correctable/uncorrectable SRAM or DRAM ECC errors, no row remapping, and no requested
GPU recovery action, so captured hardware health does not support an ECC fault.

Paired training completed 2,000 aggregate arm-updates in 6,616 seconds, or 18.138
updates/minute. The observed single-arm control phase completed 1,000 in 3,386
seconds, or 17.720 updates/minute. Pairing therefore increased aggregate throughput
only 2.36%; two sequential phases at the observed single-arm rate would take about
6,772 seconds versus 6,616 seconds paired. This is an operational estimate rather
than a controlled benchmark: the paired interval included fresh-start and update-2
profile costs, while the single interval resumed an existing checkpoint.

## Training-objective evidence

The checkpointed accumulators are diagnostic training values, not held-out metrics.

| Arm / interval | Mean total | Mean next-latent | Mean raw SIGReg | Mean bounded SIGReg |
|---|---:|---:|---:|---:|
| control, updates 1-1,000 | 5.71339 | 0.055189 | 2,612.223 | 1,886.066 |
| treatment, updates 1-1,000 | 17.02015 | 0.013166 | 13,089.336 | 5,668.996 |
| control, updates 1,001-2,000 | 3.81091 | 0.013159 | 1,466.722 | 1,265.917 |

At 1,000 updates, the treatment's raw SIGReg statistic was 5.01x control and its
bounded statistic was 3.01x control. With weight 0.003, bounded SIGReg contributed
17.007 of the treatment's 17.020 mean total loss. This is expected to change the
optimization trajectory substantially even though the model topology and row cap are
unchanged. The treatment's lower training next-latent loss cannot be interpreted as
better generalization without its held-out report.

The control's second-interval mean training next-latent loss fell to 0.01316, while
reported held-out one-step error rose to 0.10886 and normalized H8 error rose from
1.441 to 4.452. Cross-checkpoint latent MSE is partly confounded because each
checkpoint's changing encoder defines its own target space, so this is evidence of an
unfavourable evaluation trajectory, not a pure measurement of dynamics degradation
in one fixed representation.

## Completed held-out results

Only the control has valid held-out reports. “Hard pass” requires valid artifacts,
non-near-bound SIGReg, both action gates, changed-transition improvement, and
noncollapse.

| Update | Aggregate shuffle ratio [95% CI] | Random one-step [95% CI] | Changed improvement [95% CI] | Variance | Rank fraction | SIGReg raw / bounded | H8 MSE | Hard pass |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1,000 | 1.0364 [1.0173, 1.0587] | 1.1388 [1.0637, 1.2236] | 0.6666 [0.6188, 0.7052] | 0.002780 | 0.01390 | 1,218.318 / 1,079.515 | 0.10303 | no |
| 2,000 | 1.1817 [1.1308, 1.2405] | 1.6625 [1.4453, 1.9146] | 0.4977 [0.4489, 0.5456] | 0.007120 | 0.02341 | 1,010.860 / 910.396 | 0.80915 | no |

At update 1,000, control passed random-one-step action conditioning and the
changed-transition gate, but failed the aggregate action threshold and noncollapse.
At 2,000 it passed both action gates and the changed-transition gate, but still failed
noncollapse because effective-rank fraction was 0.02341 versus the 0.10 floor.
SIGReg was valid and not near its 10,000 bound at both checkpoints.

The control changed from 1,000 to 2,000 as follows:

- aggregate true-action MSE: `0.02411 -> 0.10886` (4.51x worse);
- horizon-8 rollout MSE: `0.10303 -> 0.80915` (7.85x worse);
- aggregate shuffle ratio: `1.0364 -> 1.1817` (action dependence improved);
- random-one-step shuffle ratio: `1.1388 -> 1.6625` (action dependence improved);
- encoder variance: `0.002780 -> 0.007120` (2.56x higher);
- effective-rank fraction: `0.01390 -> 0.02341` (1.68x higher, but still collapsed).

This is the same qualitative warning seen in geometry-v1 control: the two
preregistered action gates improve while normalized rollout stability and noncollapse
remain poor. Exploration and hazard action-shuffle ratios still remain near 1.0 at
update 2,000, so stronger action dependence is not general across sources. The
update-1,000 H8 value is not promoted to “Best So Far” because this dynamics-only A/B
uses a different training/evaluation contract from the full-curriculum rollout
baseline and fails its own preregistered representation gate.

## Treatment evaluation failure

The first reported error is:

```text
DriverError(CUDA_ERROR_ASSERT, "device-side assert triggered")
```

The captured stack reaches Candle/cuDNN `launch_conv2d<f32,f32>`, then
`GridResidualBlock::forward`, latent recursion, and `eval_rollout_group`. The rollout
path feeds one transition at a time through autoregressive latent recursion. After
the Rust error unwound, the process printed `free(): corrupted unsorted chunks` and
aborted with exit 134. The order makes a shutdown interaction possible, but a poisoned
CUDA context does not by itself explain corrupted host-allocator metadata. The glibc
message is therefore retained as co-equal evidence for an FFI, driver, allocator, or
memory-safety failure rather than dismissed as merely secondary.

Important discriminating evidence:

- both arms passed a paired four-update CUDA training smoke, but that smoke did not
  run evaluation and therefore did not exercise `eval_rollout_group`;
- both full arms trained 1,000 updates successfully;
- the treatment's entire 695-second evaluation was concurrent with control; control
  began six seconds earlier and finished 34 seconds later;
- control's concurrent batch-1,024 evaluation succeeded 34 seconds after treatment
  aborted;
- the exact captured binary and exact treatment checkpoint later completed an
  isolated local CUDA evaluation with one synthetic episode and physical batch 16;
  this rules out a universally corrupt checkpoint or universally invalid rollout;
- the same binary/checkpoint then completed the full 64-episode sample set locally
  at physical batch 256; this strongly disfavors a sample- or rollout-specific fault,
  but is not a preregistered substitute for the L40S batch-1,024 report.

### Ranked root-cause hypotheses

| Rank | Hypothesis | Current evidence | Smallest decisive test |
|---:|---|---|---|
| 1 | Unlocalized CUDA/FFI/host-memory corruption during the treatment's batch-one rollout path | CUDA assertion plus glibc heap-corruption message; the log does not identify rollout family, episode, group, step, or exact originating operation | Repeat with `CUDA_LAUNCH_BLOCKING=1`; if it fails, bisect rollout family/group/step and use compute-sanitizer |
| 2 | Pod-specific driver/cuDNN/allocator defect | Exact full-sample binary/checkpoint succeeds on a different local CUDA stack; captured L40S health is clean | Re-run alone on the same L40S/software stack, then compare on an updated stack if reproducible |
| 3 | Shared-device concurrency or resource pressure triggered an underlying runtime defect | Treatment failed while fully concurrent with control, but separate processes normally have isolated contexts/heaps and control survived | Run the exact treatment evaluation alone twice at frozen settings before attributing concurrency |
| 4 | Residual allocator/workspace state from preceding physical-batch-1,024 evaluation | The failing rollout itself is batch one and earlier batched passes synchronize; only residual runtime state remains plausible | Record allocator/device memory between sections and compare isolated 256/512/1,024 runs |
| 5 | Ordinary OOM | Peak left 11.3 GiB free and CUDA reported an assertion, not allocation failure | Already strongly disfavored by telemetry; retain only as a library-workspace edge case |

Asynchrony is a localization mechanism, not an independent root cause. The artifacts
do not prove a Candle bug, a cuDNN bug, a Tofy indexing bug, or concurrency. The
successful isolated replay used a different GPU/driver and smaller batch; those
attributions require the bounded reruns above.

## Local full-sample diagnostic

The exact captured pod binary and exact treatment step-1,000 checkpoint completed the
full 64-episode evaluation locally in 604 seconds with exit 0. The replay used an RTX
5060 Laptop GPU, driver 610.43.03, cuDNN 9.24.0.43, and physical batch 256. The copied
training config changed only `output_dir`. Hardware, CUDA/cuDNN version, and physical
batch therefore differ from the preregistered pod evaluation.

| Diagnostic metric | Value | Gate implication |
|---|---:|---|
| aggregate shuffle ratio [95% CI] | 0.99995 [0.99971, 1.00016] | fail |
| random-one-step ratio [95% CI] | 0.99978 [0.99926, 1.00028] | fail |
| changed learned-vs-copy improvement [95% CI] | -39.4038 [-41.7653, -36.9101] | fail |
| mean encoder variance | 1.933e-6 | fail (`<1e-4`) |
| effective-rank fraction | 0.01204 | fail (`<0.10`) |
| raw / bounded SIGReg | 12,493.534 / 5,513.085 | valid; not near bound |
| horizon-8 MSE | 0.003579 | degenerate, not positive |

Learned changed-transition MSE was `0.0010030` versus copy-forward
`0.00002482`, so the learned predictor was 40.4x copy-forward. Combined with
action-shuffle ratios indistinguishable from 1.0 and collapsed variance/rank, the tiny
absolute H8 MSE is a collapsed latent-space result, not useful dynamics. This is
directionally a strong negative treatment result at update 1,000, but it cannot enter
the preregistered gate: physical batch and hardware differ, it has no frozen arm
manifest, and the pilot decision also requires update 2,000.

The durable diagnostic files are under
`runs/p2/_pod_handoffs/ikkry5s9gidt85-20260809/local-diagnostics/treatment-update-1000-batch-256/`.
The evaluation-report SHA-256 is
`f67483e7ddd8438064c61c68cfe83d8871643adea831c1c05bea6e8adeacc5ee`.

## `candle_graph` evidence

Both arm profiles are marked `health.trusted=true`. Capture update 2 reported 7.328s
for control and 7.370s for treatment. The treatment spent more captured time in
backward (314.14ms versus 97.38ms) and forward (63.01ms versus 30.27ms), consistent
with the more expensive pre-RMS spatial statistic.

Attribution remains limited: both packets explicitly lack timed operation evidence,
tensor-memory events, device-memory checkpoints, and Nsight capture. They contain
nine spans, two tensor facts, and 28 gradient facts, but zero operations and zero
memory events. The integration worked and produced trusted structural evidence; it
did not capture enough detail to explain the CUDA assertion or optimize individual
kernels.

## Automation and supervision findings

1. `scripts/p2_sigreg_geometry_overnight.sh` waits for both members of a pair and then
   fails the whole queue if either arm fails. This preserved the successful control
   result, but it left the recoverable treatment checkpoint idle for the remainder of
   the night—about 8h31m before capture.
2. The smoke stage covers paired training only. It must include at least one held-out
   rollout evaluation for each arm to exercise the actual overnight phase boundary.
   A tiny smoke proves path reachability only; it cannot exclude a checkpoint-,
   episode-, or long-runtime-specific fault.
3. Running `scripts/p2_ab_gate.py` on this partial result reveals a separate error-path
   defect. Missing treatment manifests produce provenance tuples containing `None`;
   line 548 attempts to sort those together with string tuples and raises `TypeError`
   before the intended artifact-validation report is written. This did not cause the
   overnight failure because the supervisor stopped before invoking the gate.
4. Evaluation checksum manifests store absolute pod paths. Their contents are intact,
   but direct verification is not relocation-safe. Future manifests should prefer
   paths relative to the manifest or experiment root.
5. RunPod agent provisioning ultimately installed Git, tmux, Python, Node, Rust, and
   Codex CLI persistently. Codex remained unauthenticated, so no autonomous agent was
   available to diagnose and recover the stopped queue overnight.

## Local remediation implemented

The recoverable software and supervision defects were fixed locally after this
analysis. These changes still require a reviewed commit, release build, and deployment
before another pod run:

- `p2_ab_gate.py` now writes the intended structured validation failure for partial
  runs before comparing provenance. Artifact lookup is root-authoritative: both new
  root-relative manifests and legacy absolute pod paths resolve beneath the selected
  experiment root, and traversal or unrelated absolute paths are rejected.
- New phase records, arm output paths, and checksum manifests are root-relative.
  The arm runner uses the same relocation-aware verifier, so a moved valid evaluation
  can be resumed without being mistaken for a corrupt one.
- The overnight queue serializes complete arms. It inspects only phase records added
  by the failed invocation, except that a latest historical failed evaluation is
  deliberately recovered before normal resume. Malformed or pre-phase failures fail
  closed rather than being mislabeled as CUDA errors.
- Every failed evaluation update gets at most one isolated recovery sequence. The
  first recovery and one exact repeat run with `CUDA_LAUNCH_BLOCKING=1` and a full
  Rust backtrace. Their report SHA-256 values must match before the repeat marker is
  cleared and training can continue; a mismatch or second failure stops the queue
  while retaining the baseline report and verification record.
- The new geometry preflight runs four CUDA training updates and then the full frozen
  evaluation boundary—64 synthetic episodes at physical batch 1,024—for both arms,
  sequentially. Its summary binds the binary, settings, and both report hashes, so an
  unchanged successful preflight is safely reused. This is still path/runtime
  validation, not model-quality evidence.
- Rollout errors now carry the source (`synthetic_dynamics` or
  `synthetic_planner`), open/closed-loop mode, seed, episode, transition index, and
  exact operation. Combined with synchronous CUDA recovery, a recurrence should be
  substantially more localizable.

The public regression tests cover partial-gate output, legacy relocation with a
conflicting still-present old tree, root-authoritative corruption detection,
sequential arm ordering, bounded recovery/repeat/resume, stale-phase rejection,
exact preflight settings and reuse, and the arm runner's checksum-equality barrier.

## Decision and next run

No preregistered terminal branch is available. Do not label this branch A, do not
promote control, and do not start seeds 2/3 or a 4,000-update extension.

After deploying the reviewed local fixes, the recovery sequence should be:

1. Run the default exact-setting preflight for both arms. Do not set
   `P2_GEOMETRY_SKIP_SMOKE=1` for the first deployment of a new binary or CUDA stack.
2. Resume the overnight wrapper against the preserved experiment root. It detects the
   treatment's latest failed update-1,000 evaluation, runs it alone with synchronous
   CUDA reporting, repeats it, and requires identical report checksums before resume.
   If either attempt fails, use the new source/episode/transition/operation context to
   bisect and then use compute-sanitizer if necessary.
3. Allow the wrapper to resume only treatment from its exact step-1,000 checkpoint
   through 2,000 and evaluate it alone. Do not retrain the completed control. This is
   worth doing only to close the preregistered pilot; the local update-1,000 diagnostic
   already makes treatment collapse the expected outcome. Resume depth, episode order,
   and SIGReg seeds are derived from frozen seed/global-step state, and the repository
   has a pause-vs-uninterrupted regression test. That test passed during this analysis;
   it establishes the CPU resume contract within tolerance, not bitwise CUDA equality.
4. Generate the missing treatment manifest/checksums, verify all four seed-1 reports,
   and run the unchanged pilot gate.
5. Follow the gate result. Only then consider seeds 2/3.

For future paired experiments, serialize complete arms. The estimated throughput cost
is approximately 2.3%, while isolation simplifies attribution and limits shared-device
trigger risk. If training concurrency is retained for some reason, evaluations should
still be serialized and failure of one arm must enqueue an isolated recovery attempt
instead of ending the entire overnight queue.

## Evidence locations

- Full local handoff: `runs/p2/_pod_handoffs/ikkry5s9gidt85-20260809/`
- Control phases: `runs/p2/ab-sigreg-geometry-v2/seed-1/control/phases.jsonl`
- Treatment phases: `runs/p2/ab-sigreg-geometry-v2/seed-1/pre-rms-spatial/phases.jsonl`
- CUDA failure log: `runs/p2/ab-sigreg-geometry-v2/seed-1/pre-rms-spatial/eval-update-1000/eval.log`
- Control reports: `runs/p2/ab-sigreg-geometry-v2/seed-1/control/eval-update-{1000,2000}/eval_report.json`
- GPU telemetry: `runs/p2/ab-sigreg-geometry-v2/seed-1/control/telemetry/gpu.csv`
- Profiler evidence: `runs/p2/ab-sigreg-geometry-v2/seed-1/{control,pre-rms-spatial}/profile/update-000000000002/EVIDENCE.md`
- Preserved supervisor log: `runs/p2/_pod_handoffs/ikkry5s9gidt85-20260809/tofy-analysis-ikkry5s9gidt85-20260809/logs/geometry-overnight-0a3f8205.log`
- Local treatment diagnostic: `runs/p2/_pod_handoffs/ikkry5s9gidt85-20260809/local-diagnostics/treatment-update-1000-batch-256/`

Validation command:

```bash
python3 -m unittest tests.test_p2_ab_gate tests.test_p2_geometry_scripts -v
bash -n scripts/p2_sigreg_action_ab.sh scripts/p2_sigreg_geometry_ab.sh \
  scripts/p2_sigreg_geometry_overnight.sh scripts/p2_sigreg_geometry_smoke.sh
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```
