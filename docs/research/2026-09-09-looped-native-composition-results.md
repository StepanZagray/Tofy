# Frozen native image-to-action composition — C19–C21

Completed September 9, 2026, Europe/Dublin IST. **The registered finite native
composition gate passes:** factual inference answers **768/768**, including
**576/576 demonstrated** and **192/192 omitted-action** cases. Uniform attention
answers **192/768**, always choosing the omitted action: **0/576 demonstrated,
192/192 omitted and 25% overall by construction**. This is a frozen component
result using an initial visual core and privileged role selectors. Public ARC
performance remains **0/23**; no new training or recurrence benefit is claimed.

The owning [analysis](/home/stepan/Projects/code/.tofy-runs/looped-native-f32-rescore-20260909T224747-IST/analysis.json)
records `supported_frozen_native_composition`. C20 performed the CUDA inference;
C21 repaired a declared-F32 identity check and rescored those same outputs.
C19's failed qualification and both recovery boundaries remain part of the evidence.

## Fixed components and inference boundary

Public seven-frame images and metadata enter the frozen visual transformer.
The same learned two-slot selector reads each frame. Normalized attention gives
expected public-grid coordinates; support agent-after minus agent-before gives
three movement records, and current goal minus agent gives the desired movement.
Public demonstrated action IDs accompany these values into the frozen C18
action-equivariant binder. All these connections execute as device tensors.

The visual core and binder each execute **four shared loops**. The initial visual
core was not learned by C7; the selectors inherit privileged role fitting and fixed
agent/goal slot identities. The binder learned the abstract C18 action task.
C19–C21 perform **zero optimizer updates** and leave all three components unchanged.
The producer reports **2,572,942 stored /2,043,397 executed parameters**; unused
legacy visual heads remain loaded. This accounting is not a necessity estimate.

No hidden role coordinates, map ID, correct label, CPU role argmax, palette parser,
analytic direction matcher or analytical missing-effect completion enters factual
inference. Independent audit/scoring uses visible-cell truth. The coordinate adapter
and action-routing symmetry are explicit representation priors; this experiment
does not learn those interfaces or discover the meaning of its two role slots.

## Panel and registered decision

The [registration](/home/stepan/Research/_runs/2026-09-09T213625Z-tofy-looped-native-f32-rescore/registration.md)
fixes seed **20260923**, tag **0x4e415449564542**, 32 query groups and all 24
lexicographic cardinal action mappings, ordered query-first. Every query has
distance one, three distinct unblocked demonstrations, 18 demonstrated-label
and six omitted-label cases. Each action is the correct answer 192 times.
The four physical query-direction group counts are **5, 8, 8 and 11**.

The audit verifies 32 distinct queries and zero overlap with the **6,200-query
union** of six named historical files: the full C8 panel, three C11 panels, C12
training and C12 familiar evaluation. Their 76,160 rows are not independent
queries. Novelty is relative to this named history, not universal prior exposure.
The earlier C19 qualification used this same preregistered panel.

The action gate requires every factual case correct, minimum true-action margin
at least **0.001**, uniform same-input/balance checks, replay parity and accepted
runtime integrity. Role mass and coordinate accuracy are separate descriptive
checks. Sampling units are the 32 queries; no IID 768-row confidence interval or
population-perfection claim is made. Full per-query/map/direction results remain
in the machine-readable analysis rather than being repeated here.

## Action results and the negative control

| Stream / subset | Correct | Accuracy | Mean CE | Minimum true margin |
|---|---:|---:|---:|---:|
| Factual / all | 768/768 | 100% | 1.49450225849e-8 | 16.7352153063 |
| Factual / demonstrated | 576/576 | 100% | 9.16300977224e-9 | 16.7987508774 |
| Factual / omitted | 192/192 | 100% | 3.22910610230e-8 | 16.7352153063 |
| Uniform / all | 192/768 | 25% | 25.6894612312 | −34.2526149750 |
| Uniform / demonstrated | 0/576 | 0% | 34.2526149750 | −34.2526149750 |
| Uniform / omitted | 192/192 | 100% | 3.55271367880e-15 | 34.2526149750 |

Factual inference is correct on **32/32 complete mapping groups**, on the C18
familiar-map split **512/512**, and on its held-map split **256/256**. Predictions
are balanced `[192,192,192,192]`. Uniform scores exactly six of 24 per group,
128/512 familiar and 64/256 held; no group has all mappings correct.

Uniform executes the same real vision and seven selector passes, then replaces
all role attention with 1/64 before the adapter. Every expected position agrees,
so support and query displacements are zero. Records, logits and predictions
are identical across mappings within each query. Balanced labels force 25%
overall for any deterministic winner. The observed winner is always the missing
public action ID; its perfect omitted score is a shortcut, not evidence that it
recovered the omitted physical effect. Its large demonstrated CE preserves the
confident failure that the aggregate 25% obscures. This combined intervention
tests visual information as a whole, not separate support/query necessity.

Scorer-only analytic geometry answers 768/768; constant action 0 and always-missing
controls each answer 192/768. The analytic solver is absent from learned inference.

## Grounding and numerical verification

All **14 frame/role argmax locations** are correct on every row, including unused
support-goal slots: joint **768/768**. All three support displacements and the
current desired displacement are also correct by argmax, with correct endpoints.
Minimum true-role mass across all slots is **0.999998807907**. Worst position
L-infinity error is **4.40086038e-6 cells**; worst support soft-displacement error
is **1.44088652e-6**, and current-query error is **1.39488866e-6**.

Uniform retains exactly the same exported learned attention, but its effective
mass is 1/64. Effective role and displacement argmax counts are zero here: ties
select cell 0, which contains neither role. These are control consequences,
not deterioration of the frozen learned selector.

Independent NumPy F32 replay rebuilds adapter records from attention and public
metadata, then evaluates the frozen binder; stored adapter records are not its
inputs. The fixed tolerance is **1e-4 + 1e-5 × abs(reference)**. Maximum record
error is **1.83593508e-6**; maximum logit error is **4.74452972412e-5**, with worst
error/tolerance ratio **0.377042343 < 1**. All **1,536** factual/control winners
are eligible and agree: each reference winner margin exceeds zero and twice
its measured maximum logit discrepancy. There are no ineligible rows.

The [independent scalar review](/home/stepan/Projects/code/.tofy-runs/looped-native-f32-rescore-20260909T224747-IST/independent-review.json)
agrees on **716 floating fields** within **4.48e-16** and **606 discrete fields**
exactly. It independently reconstructs input hashes, geometry, records and scalar
endpoints, but shares the pinned NumPy neural replay. Both scorers rely on the
external source/device/checkpoint/profiler/cleanup receipt; neither supplies a
second physical execution. Current frame 6 is bitwise identical to the ordinary
current-frame selector input; this does not claim an eighth selector execution.

## Failure history, capacity and cost

C19 stopped after four-row CUDA qualification because its profiler declared one
generic `/forward` region while execution used four component phases. It never
reached the capacity ladder or scientific streams. Its outer execution cost was
**12.356667 s**; the failed inventory retains 70 files and 35 recorded PIDs gone.
C20 corrected this producer contract, preserving numerical sources, weights and
the exact panel SHA. Its outer execution completed at **22:32:41 IST** in
**111.419627 s**; the release build cost **22.730139 s** separately.
The [build record](/home/stepan/Research/_runs/2026-09-09T212336Z-tofy-looped-native-profile-recovery/build.json)
pins `cudnn,profiling,serde_json/float_roundtrip` and the offline release command.

Native capacity trials jointly confirmed **32, 64, 128 and 256** twice each;
**512 failed with CUDA OOM**, so 1024 was not attempted. Both scientific streams
use physical batch **256**, three batches each, with no repeated rows and no
gradient accumulation because there is no training. This differs from C18's
abstract binder training at **4096, accumulation 1**: the native workload also
executes vision and seven selectors. It is not evidence of a causal batch effect.

Maximum sampled memory at 256 is **5,118 MiB** on an 8,151 MiB RTX 5060 Laptop,
leaving **3,033 MiB sampled reserve**; factual/uniform temperatures peak at
61/62°C. The failed 512 trial's 3,582 MiB sample precedes the allocation failure
and is not its required or peak memory. Model-reported factual/uniform lifetimes
are **8.383031347 /8.635717866 s**, including instrumentation.

C20's original scorer then failed before action summaries: audit JSON serialized
F32 directly, whereas runtime JSON expanded it through F64 (`0.14285715` versus
`0.1428571492433548`). An input-only check found **712,704 decimal differences
per stream**, but identical F32 bits/hashes on all 768 rows and identical other
fields. C21 changes only this identity comparison to declared F32 bytes, with
equivalent-decimal and one-ULP/type/nonfinite rejection fixtures. Gates, replay,
weights and data are unchanged. C21 is a same-panel rescore, not fresh replication.

C21 analysis/review completed at **22:47:59 IST**, costing **11.190735 s** externally
(primary **4.858056 s**, independent **6.182893 s**). Sealing cost **1.073478 s**;
verification completed at **22:48:34 IST**. The C21 seal covers 14 files,
406,948 bytes and 781 external bindings, including the C20 runtime inventory.
Recorded processes are gone, including the subsequently closed sealing wrapper.

All **11 successful CUDA captures** are healthy and Nsight-bound: qualification,
eight successful capacity trials and two scientific captures. Each captures the
first batch, with semantic/tensor/scalar evidence and CUDA/cuDNN/cuBLAS/OS-runtime/
CPU traces. Operation/activation linkage, allocation lifetimes, instrumented
physical-memory checkpoints and device-event intervals remain unwired; automatic
GPU correlation is incomplete. Sampled memory is not allocator peak, and profiled
host durations do not establish production kernel throughput.

## Identities and historical commands

| Identity | Exact revision / SHA-256 |
|---|---|
| Runtime source | `e68703cf4d42119407e721edf42170beec515037` |
| C21 analyzer source | `6ccd87fee115ae64e9056a0f91676e3052592143` |
| Runtime binary | `e5bfc4c8b7da35ad2031b3bbe28af6bce891c34824436de3052d1dd067d521dc` |
| Initial visual checkpoint | `4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802` |
| Imported selector payload | `a43ad78fe7a6b1594d23c205fb191c8ac81610deaaff25198cb421d423edc678` |
| C18 terminal binder | `d2deeba7b0fafc2386c9a19d39bc99b7534b91a0155d66e77792a5c4037bd717` |
| Primary analysis | `d177c1df57460256e66ad90b51e93fb4912b46eec04f05a47b5cccf2565c08d6` |
| Independent review | `ee69ce0379edce62396b4baaeb9ad86e3fa808b3cebb7bc4b3e1a0776c538ada` |

Full bindings, including candle_graph `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a`,
are in the [analysis configuration](/home/stepan/Projects/code/.tofy-runs/looped-native-f32-rescore-20260909T224747-IST/analysis-config.json).
Point-in-time outer seals preserve the [C19 failure](/home/stepan/Research/_runs/2026-09-09T204022Z-tofy-looped-native-binding/failed-campaign.manifest.json)
`cd938e9aa40a2010b43bf57cca3b1add47dea8968f6e62edf145ee719478a8a6`,
[C20 runtime](/home/stepan/Research/_runs/2026-09-09T212336Z-tofy-looped-native-profile-recovery/runtime-campaign.manifest.json)
`a8bd0584802dc3f3652f9f83ba0551cd64634a4f523df4844a65e9e5250e3e3f`, and
[C21 completed rescore](/home/stepan/Research/_runs/2026-09-09T213625Z-tofy-looped-native-f32-rescore/completed-campaign.manifest.json)
`cbbcb0878bec40c4603c581c0201fa47fbc0bfbe98745707a3ac2c2428b04043`.

These exact commands come from the [CUDA process record](/home/stepan/Research/_runs/2026-09-09T212336Z-tofy-looped-native-profile-recovery/operations/outer-execute.process.json)
and [rescore process record](/home/stepan/Research/_runs/2026-09-09T213625Z-tofy-looped-native-f32-rescore/operations/outer-analysis.process.json).
Sealed roots must never be reused; reproduction needs newly registered roots.

```bash
cd /home/stepan/Projects/code/Tofy-native-binding-profile-fix
/home/stepan/venvs/tensorboard/bin/python3 /home/stepan/Research/_runs/2026-09-09T212336Z-tofy-looped-native-profile-recovery/campaign_operator.py execute \
  --campaign /home/stepan/Projects/code/.tofy-runs/looped-native-profile-recovery-20260909T223049-IST \
  --spec-sha256 932857b1da431330c843fd6c8f0b844d8a7e2b77222da55e3f6c5622bdb5d65f

cd /home/stepan/Projects/code/Tofy-native-binding-results
/home/stepan/venvs/tensorboard/bin/python3 /home/stepan/Research/_runs/2026-09-09T213625Z-tofy-looped-native-f32-rescore/rescore_operator.py analyze \
  --campaign /home/stepan/Projects/code/.tofy-runs/looped-native-f32-rescore-20260909T224747-IST
```

## What this supports

The frozen components compose on this finite new-query distribution. Uniform
8×8 color patches, wall-free support frames with goal fixed at cell 63, three
distinct unit demonstrations, cardinal bijections and distance-one queries leave
substantial template/geometry restrictions. There is no pretrained LLM controller,
ARC evaluation, planning, episode memory, learned world dynamics or integrated
training result. Initial success earns no C7 learning credit. C18 already solved
its factual task at one binder loop; this four-loop test supplies no recurrence
ablation. The ordinary legacy policy head is unchanged. The next task requires
a separate claim and registration; this finite pass does not promote a general
controller or change the public **0/23** record.

A prospective [paired-detour counterexample](/home/stepan/Research/_runs/2026-09-09T213625Z-tofy-looped-native-f32-rescore/navigation-boundary.json)
shows why explicit map context matters: identical ideal agent/goal coordinates
can require opposite optimal first moves. This assumes exact semantic coordinates;
actual soft attention may encode walls in coordinate offsets. A scene-context
connection is a proposed next interface test, with no implementation result yet.
