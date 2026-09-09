# C17 action binding: perfect finite fit, failed held-out complement screen

**Completed September 9, 2026, machine-local IST. Accepted integrity and
independent scoring; registered decision: `registered_binding_screen_not_supported`.**

The 1,580,804-parameter ControlBinder learns every registered fit case:
**256/256 semantic cases**, including all 64 cases whose correct action was
absent from the demonstrations. It transfers incompletely to the eight held-out
control mappings: **71/128 overall, 69/96 demonstrated and 2/32 omitted**.
All three held-out accuracy gates fail. This is a valid negative result for
the single-seed learning screen, with successful fitting as a separate positive.
It does not reject all learned binders, prove a size/depth benefit, or improve
native policy or ARC performance.

The terminal binder also scores **1024/1024 on retained C15 visual features**.
Those rows use familiar control mappings and reused, privileged initial role
attention. This positive cache result does not establish fresh visual learning,
held-out control transfer or native end-to-end inference.

The owning [numerical analysis](/home/stepan/Projects/code/.tofy-runs/looped-binding-profile-amortized-20260909T203340-IST/analysis.json)
and [independent review](/home/stepan/Projects/code/.tofy-runs/looped-binding-profile-amortized-20260909T203340-IST/independent-review.json)
are both accepted. All figures below use their fixed terminal checkpoint;
no best checkpoint or depth was selected.

## The task, model and finite population

The task presents three distinct action IDs and their observed cardinal
displacements, then asks which action produces a desired direction. A
demonstrated case asks about one of the three observed actions. An omitted
case asks about the fourth action, so the model must infer its missing effect.
An exact external control can do this because the four cardinal displacement
vectors sum to zero. That analytic solver is absent from the learned model.

The forward input is F32[4,7]: three records containing displacement, a four-way
action indicator and query flag, followed by the desired-direction record.
Labels, map IDs and grouping metadata do not enter forward computation. The
model has width 256, a learned readout token, two shared RMS/attention/SiLU
residual blocks, four attention heads, and a four-action readout. It has no
positional encoding and uses four shared loops during training. The visual
core is not run; reward, value, dynamics, planner, episode memory and ARC
evaluation are absent. No pretrained LLM or LLM controller is used.

Fit map IDs are `[0,2,4,5,7,8,9,10,13,14,15,16,18,19,21,23]`; held-out IDs
are `[1,3,6,11,12,17,20,22]`, from the 24 lexicographic cardinal bijections.
Enumerating 24 ordered triples of observed actions and four desired directions
gives 1536 fit and 768 held-out rows. Six support orderings represent the same
semantic case, so the primary denominators are **256 and 128**, not 1536 and
768 independent examples. Canonical representatives sort the observed action
IDs. Counts cover the entire registered finite population; no IID, binomial
or bootstrap confidence interval is claimed. Uncertainty over seeds and harder
task distributions remains unknown.

One seed-0 fit runs all **1150 updates and 588,800 presentations**, with
**physical/effective batch 512 and accumulation 1**, no tail. The fixed PCG64
seed-20260922 schedule uses 383 complete shuffled epochs plus 512 rows of
epoch 384: 1024 ordered rows receive 383 exposures and 512 receive 384.
Repetition does not increase the number of distinct semantic cases. AdamW
uses learning rate 0.0003, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01,
global L2 clip 1, mean policy CE only and no learning-rate schedule.

These settings and gates were frozen in the
[registration](artifacts/c17-binding-admission/registration.md). The user had
requested at least one million parameters and effective batch 512 before any
C16 model run. C17 retains that scale. No smaller trained comparator exists.

## Four-loop outcome and registered gates

CE is cross-entropy: the negative log-probability assigned to the correct
action, averaged over cases. Lower is better. Uniform probabilities give CE
`log(4) ≈ 1.3863`; a confidently wrong prediction can have much larger CE.
Values here are rounded from independently checked F64 scoring of retained
F32 logits. Correct counts are exact.

| Cohort/checkpoint | Canonical correct | Raw ordered correct | CE | Demonstrated correct | Omitted correct |
|---|---:|---:|---:|---:|---:|
| Initial fit | 58/256 (22.65625%) | 348/1536 | 1.586631 | 53/192 | 5/64 |
| Final fit | **256/256 (100%)** | **1536/1536** | **0.000003327** | **192/192** | **64/64** |
| Initial held-out | 31/128 (24.21875%) | 186/768 | 1.590677 | 24/96 | 7/32 |
| Final held-out | **71/128 (55.46875%)** | **426/768** | **3.670104** | **69/96 (71.875%)** | **2/32 (6.25%)** |

| Registered component | Required | Observed | Result |
|---|---:|---:|---|
| Canonical fit | ≥254/256 | 256/256 | Pass |
| Held-out overall | ≥116/128 | 71/128 | Fail |
| Held-out demonstrated | ≥87/96 | 69/96 | Fail |
| Held-out omitted | ≥29/32 | 2/32 | Fail |
| Same winner under all six support orderings, final four-loop fit/held-out | All 384 orbits | All 384 | Pass |
| Balanced deterministic ablation controls | All four streams valid | All four | Pass |

The joint scientific gate fails. Final fit has minimum true-label logit margin
11.216895; every fit mapping scores 16/16. Final held-out map counts are
`1:10/16, 3:6/16, 6:9/16, 11:7/16, 12:12/16, 17:9/16, 20:10/16, 22:8/16`.
Its prediction histogram is `[31,30,32,35]`, close to its balanced true-label
histogram `[32,32,32,32]`. Balanced predictions therefore do not imply correct
binding. Final fit predicts `[64,64,64,64]`, exactly matching labels.

The held-out failure includes strong wrong confidence. Demonstrated CE improves
from **1.562717 to 1.346083**, while omitted CE worsens from **1.674560 to
10.642166**. Omitted mean true-label margin is **−10.299360**, minimum
**−19.334417**. Overall held-out accuracy rises by 31.25 percentage points,
but CE worsens by 2.079427. The model is correct more often overall while
assigning extremely low probability to many missing-action answers. This
does not diagnose the internal cause or show that the necessary information
is absent from its representation.

## Every depth and ablation stream

Only four-loop factual results select the scientific gate. The terminal
checkpoint is also run at 1, 2 and 8 loops without further updates. These are
descriptive tests of one model trained at depth 4, not matched training runs
or proof that recurrence is beneficial.

| Final factual loops | Fit canonical / raw | Fit CE | Fit demonstrated / omitted | Held-out canonical / raw | Held-out CE | Held-out demonstrated / omitted |
|---|---|---:|---|---|---:|---|
| 1 | 232/256 / 1392/1536 | 0.318200 | 187/192 / 45/64 | 74/128 / 444/768 | 2.206074 | 69/96 / 5/32 |
| 2 | 255/256 / 1530/1536 | 0.006198 | 192/192 / 63/64 | 75/128 / 450/768 | 2.989428 | 73/96 / 2/32 |
| 4 | 256/256 / 1536/1536 | 0.000003327 | 192/192 / 64/64 | 71/128 / 426/768 | 3.670104 | 69/96 / 2/32 |
| 8 | 256/256 / 1536/1536 | 0.000018378 | 192/192 / 64/64 | 69/128 / 414/768 | 3.728989 | 66/96 / 3/32 |

All abstract streams, including initial and alternative depths, have exact
winner agreement across each orbit's six support orderings. Logits are not
bitwise identical: the maximum pairwise deviation is 2.31266e-5 for final
four-loop fit, 2.33650e-5 for final four-loop held-out, and 3.29018e-5 over all
abstract streams. Support-order invariance is distinct from equivariance to
renaming the action IDs; the latter was not established by this check.

Effect-zero sets only the three observed displacements to zero; query-zero
sets only the requested displacement to zero. Public action IDs and the query
flag remain. In each control, identical actual model inputs have balanced
four-way labels, so **any deterministic classifier must score 25% overall**.
The independent review verifies post-clamp F32 hashes and identical winners
within each such input group. These controls do not diagnose learned collapse.

| Final four-loop clamp | Canonical / raw correct | CE | Demonstrated correct | Omitted correct |
|---|---|---:|---:|---:|
| Fit effects-zero | 64/256 / 384/1536 | 7.867383 | 52/192 | 12/64 |
| Fit query-zero | 64/256 / 384/1536 | 8.039654 | 50/192 | 14/64 |
| Held-out effects-zero | 32/128 / 192/768 | 7.867383 | 26/96 | 6/32 |
| Held-out query-zero | 32/128 / 192/768 | 7.719271 | 24/96 | 8/32 |

On paired canonical cases, final-minus-initial fit gains **198 cases / 77.34375
pp**, with 58 correct at both checkpoints and none lost. Held-out gains 52
and loses 12, with 19 correct at both: net **40 cases / 31.25 pp**; 93 winners
change. Within the omitted subset, only one formerly wrong case improves while
six formerly correct cases fail, giving **−5/32 / −15.625 pp**. Factual-minus-
either-clamp accuracy is **+75 pp fit** and **+30.46875 pp held-out**. Held-out
effects-zero pairing has 20 both correct, 51 factual-only and 12 clamp-only;
query-zero has 18, 53 and 14. These are exact paired counts, not inferential CIs.
Full raw/canonical CE, margins, histograms, per-map values and paired subsets
are retained under `summaries` and `contrasts` in the numerical analysis.

## Cached visual result and external controls

The terminal four-loop binder scores **1024/1024**, with **768/768 demonstrated**
and **256/256 omitted**, CE **0.000003279** and minimum true-label margin
**11.216880**, on the retained C15 initial-attention adapter. Its action
histogram is `[256,256,256,256]`. Here raw and canonical summaries both mean
all 1024 retained rows: no six-order orbit population is fabricated.

These are the reused C12 seen 64 query groups ×16 familiar mappings, carried
through C15. Before/after agent coordinates and current goal/agent displacement
are expectations under frozen initial role attention, normalized in F64 and
rounded once to F32. True locations are used for audit only. The initial
selectors received privileged C10 role supervision and are nearly one-hot;
C15 support goals were fixed at cell63. This tests the cached adapter on
already familiar abstract cases, not fresh vision, new maps, diffuse final
selectors, or a jointly executed/trained vision-and-binding system.

The external analytic solver scores **256/256 fit, 128/128 held-out and
1024/1024 cached**, with minimum true-score margin 1 on abstract rows and
0.999995992 on cached rows. Constant action0 scores **64/256, 32/128 and
256/1024**. Always choosing the missing ID has the same 25% overall counts,
but is correct on every omitted case and no demonstrated case. Thus that
shortcut cannot pass both subset gates. These external controls are scored
separately; they supply neither learned-model logits nor a deployed policy.

## Runtime correction, hardware and profiling

[C16](2026-09-09-looped-binding-qualification-results.md) stopped before learning
because its five-update mean extrapolated one selected capture every five
updates. C17 changes runtime accounting only. The frozen Rust numerical
sources, model, data, optimizer, initializer, seed, loop count and scorers are
unchanged; the launch binary is rebuilt and separately hashed with the C17
source identity. C16 is qualification evidence, not a trained comparator.

C17 freshly qualifies its exact binary, including disposable 2- and 5-update
trials at physical512/accumulation1 with exact restore. The registered estimate
takes component maxima across those trials:

```text
u = 0.063638699 s       maximum complete ordinary-update increment
c = 3.979336581 s       selected-update residual, including cold/export work
s = 0.855256248 s       setup/post-loop residual
h = 0.239179092 s       supervisory residual
k = 0.011224810 s       checkpoint overhead
T = 1.25 × (1147u + 3c + s + h) + k
  = 107.543765855 s < the unchanged 600 s training cap.
```

The [admission record](/home/stepan/Projects/code/.tofy-runs/looped-binding-profile-amortized-20260909T203340-IST/admission.json)
is accepted. Actual [training](/home/stepan/Projects/code/.tofy-runs/looped-binding-profile-amortized-20260909T203340-IST/train-seed0/report.json)
takes **82.715978730 s model-reported**, including **81.908439426 s update-loop
time**. The [external exit](/home/stepan/Projects/code/.tofy-runs/looped-binding-profile-amortized-20260909T203340-IST/train-seed0.exit.json)
measures **83.153927212 s model phase** and **0.575514738 s finalization**.
The forecast is empirical, not a runtime upper-bound theorem. The 600-second
model watchdog and 30-minute campaign wall cap remained enforced.

All qualification, training, 15 evaluations and binding operations finish in
**216.916751095 s** of supervised execution. The separate supervised analysis
stage takes **7.825554927 s**, containing primary scoring **3.533403892 s** and
independent review **3.645823426 s**. These operation durations exclude the
pause between execution and analysis; they are not the complete wall interval
from launch through seal. Build time is **28.130182726 s**.

The RTX 5060 Laptop GPU has 8151 MiB reported memory, driver 610.57.04, CUDA
F32 and TF32 override0. Training and campaign maxima are **1246 MiB sampled
usage**, **74°C sampled temperature**, with **6905 MiB minimum sampled reserve**.
Telemetry sampling is not an allocator peak or long-run capacity guarantee.

There are **21 healthy, complete CUDA captures**, each with zero errors/four
warnings: three qualification/smoke captures, training updates2/100/1150, and
one first-batch capture per evaluation stream. All have available/bound Nsight
GPU evidence and verified raw application labels. Automatic GPU correlation
remains incomplete. Semantic spans, input/logit/loss statistics, host scalars
and declared parameter-family gradients are present; operation/activation
hotspots, allocation lifetimes, instrumented memory checkpoints and device-event
intervals remain unwired. `profiled_work` and host intervals do not establish
production throughput superiority or isolated kernel costs.

## Independent checks and provenance

The frozen independent scorer reconstructs finite rows, the entire schedule,
orbits, cached attention-derived coordinates, actual post-clamp hashes, scalar
stable CE/margins/counts, paired contrasts, external controls and exact gates.
It imports neither the primary metric implementation nor its data generator.
It agrees on **2927 floating fields**, maximum absolute difference
**1.7763568394002505e-15**, and **4994 discrete fields** exactly. It performs
no fitting or model inference. Source/build/initializer/training/gradient/device/
profile/checkpoint/cleanup acceptance is shared through the pinned
[integrity receipt](/home/stepan/Projects/code/.tofy-runs/looped-binding-profile-amortized-20260909T203340-IST/integrity.json),
whose ten checks all pass; cached producer provenance is inherited from C15.
This is independent numerical reconstruction, not an independent training
replicate or independent CUDA execution.

| Identity | Exact revision or SHA-256 |
|---|---|
| Tofy source | `9d4827d23c50aa12d8d49b36705e67dbea480371` |
| candle_graph | `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a` |
| Launch binary | `14e7649c47e385b31b9c429461db8179163efdcb2f8831bd9aedbf389d8ff9c0` |
| Registration | `f76831e2c1406cac75d0b5f21bc04736b52abba26c52c4404b0e555952f290a3` |
| B512 dataset | `bc1413652b6bf38cadde28c8f1fd3d88d9cf781ab55413300bc155e341c47295` |
| U32LE schedule | `a46eac4347bfe16d55d220240452cb67643ccd007a97d2e56da16be45cb6f51e` |
| Initial checkpoint | `1f59081cc8dd74b0b6d1ddc59e33a80fe4a98fe3c2640c74e4a3b6eecd0f1a6e` |
| Terminal checkpoint | `816d9ca78c77e42cffa2c99a683465e6423fa10de72d738197bcd88b09f3ed21` |
| Initial parameter F32-bit digest | `55d6d89a7e8a22049d074ae828cabb2f114093ec88886df67744a453c0cd364e` |
| Terminal parameter F32-bit digest | `d0484d90b3a4fda345a772332cb0fca0984611b9d51963d6b82d7faab01fb26b` |
| Numerical analysis | `76c0279c433b11019028b97c4202594951fbffdebfb83188f5e4187c0894188d` |
| Independent review | `0b8d9cb055473c6fb63c81f84468c6a0f272d88108f59aa622fc4d97b35848f7` |
| Integrity receipt | `3f84f03dacc7f2cd740fec975f7299db72a6bb481fc2e43f86a9b9d71a956d0f` |
| Outer campaign manifest | `61378b3e4ad96573aacbdbaa1fbac9619962b586895f5e49b1eae9d8431ab2a6` |

The [completed manifest](/home/stepan/Research/_runs/2026-09-09T192158Z-tofy-looped-binding-profile-amortization/completed-campaign.manifest.json)
and [verification](/home/stepan/Research/_runs/2026-09-09T192158Z-tofy-looped-binding-profile-amortization/completed-campaign-verification.json)
cover **1408 files /596,145,101 bytes, 1050 external bindings, 518 recorded PIDs
gone and 21 CUDA captures**, finalized at **20:41:21 IST**. This is point-in-time
integrity, not immutable storage. Evaluation checkpoint digests are unchanged
through all frozen evaluations. No experiment process is intentionally retained.

## Historical commands and next decision

The exact commands are preserved in
[execute.process.json](/home/stepan/Research/_runs/2026-09-09T192158Z-tofy-looped-binding-profile-amortization/operations/execute.process.json)
and [analysis-stage.process.json](/home/stepan/Research/_runs/2026-09-09T192158Z-tofy-looped-binding-profile-amortization/operations/analysis-stage.process.json).
These are historical commands; the completed root must never be reused:

```bash
cd /home/stepan/Projects/code/Tofy-binding-admission
/home/stepan/venvs/tensorboard/bin/python3 \
  /home/stepan/Research/_runs/2026-09-09T192158Z-tofy-looped-binding-profile-amortization/campaign_operator.py \
  execute \
  --campaign /home/stepan/Projects/code/.tofy-runs/looped-binding-profile-amortized-20260909T203340-IST \
  --spec-sha256 d13c0219c9f2a090bc9e3ed68bb6a193a59d1c0798716cc78e05c3774326a47b
/home/stepan/venvs/tensorboard/bin/python3 \
  /home/stepan/Research/_runs/2026-09-09T192158Z-tofy-looped-binding-profile-amortization/campaign_operator.py \
  analyze \
  --campaign /home/stepan/Projects/code/.tofy-runs/looped-binding-profile-amortized-20260909T203340-IST \
  --spec-sha256 d13c0219c9f2a090bc9e3ed68bb6a193a59d1c0798716cc78e05c3774326a47b
```

[Build provenance](/home/stepan/Research/_runs/2026-09-09T192158Z-tofy-looped-binding-profile-amortization/build.json)
records `cargo build --release --locked --offline --features
cudnn,profiling,serde_json/float_roundtrip --example action_binding_probe`, its
exact target directory and `TOFY_BUILD_COMMAND`. Python/NumPy/BLAS/Rust/Nsight
identities are in [software.json](/home/stepan/Research/_runs/2026-09-09T192158Z-tofy-looped-binding-profile-amortization/software.json).

The bounded conclusion is successful fitting and familiar-cache compatibility,
with inadequate held-out retrieval and especially missing-action complement.
A learned binder that is equivariant to renaming action IDs is a next hypothesis
under investigation, not an identified cause or selected successful recipe.
The user's request for a larger power-of-two batch is also a future design
constraint; C17 used512. Any next architecture/batch change requires a separate
claim, resource qualification and frozen comparison. No new training outcome,
general reasoning, useful-recurrence, visual, planner or ARC claim follows here.
