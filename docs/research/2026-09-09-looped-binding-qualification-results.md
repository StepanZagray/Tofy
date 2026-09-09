# C16 action binding: CUDA qualification passes, training admission stops

**September 9, 2026, machine-local IST. Classification: completed qualification
and budget-stop evidence; the registered learning experiment did not run.**

The 1,580,804-parameter ControlBinder passes the frozen CUDA check and both
disposable batch checks at physical/effective batch 512, accumulation 1. The
registered runtime estimate then exceeds the 600-second training cap, so the
operator stops before either initial evaluation, the 1,150-update fit, or any
terminal evaluation. This supplies no action-accuracy improvement, negative
learning result, native-policy failure, or ARC result. Best So Far is unchanged.

## What was registered

The narrow question was whether a small shared-depth model could learn to bind
three observed action IDs to their cardinal effects and infer the fourth effect
by exclusion. Its input is four records of seven F32 values: three observed
effect/action records and one desired-direction query. Hidden map IDs and labels
remain outside the forward input. The learned model contains a 7-to-256 input
projection, learned readout token, two shared residual attention/MLP blocks,
four attention heads, and a four-action readout; it runs four loops. The visual
core, online visual adapter, reward, value, dynamics, planner and ARC evaluator
are not executed. No pretrained LLM or LLM controller is used.

The fixed fit/held-out split has 16/8 cardinal-control mappings and 1536/768
ordered rows. Each semantic case has six support orderings, giving only
**256/128 distinct semantic cases**. The intended seed-0 fit was 1150 updates of
mean policy CE, effective batch 512, AdamW learning rate 0.0003, weight decay
0.01 and global gradient clip 1. Its 588,800 presentations repeat the finite
population: 383 complete shuffled epochs plus 512 rows of epoch 384. These
presentations were scheduled, not consumed by a main fit.

Before any C16 model work, the user requested at least one million parameters
and effective batch 512. Width and batch were increased together from the
untrained preliminary proposal. No trained smaller-model comparison exists,
so this is not a controlled size or batch ablation. All final settings and
the unchanged scientific thresholds are recorded in the
[registration](artifacts/c16-action-binding/registration.md).

The withheld scientific gates require at least 254/256 canonical fit cases,
116/128 held-out cases, including 87/96 demonstrated-action and 29/32
omitted-action cases, plus exact winning-action agreement over all six support
orderings. Deterministic effect-zero/query-zero controls must score 25% because
their identical-input groups have balanced labels. An always-missing-action
shortcut also gets 25% overall despite 100% on the omitted subset. None of these
learned-model metrics or gates was evaluated in C16.

## What actually executed

All three roots are explicitly `implementation_smoke`. Times below are the
model's reported elapsed time and the external supervisor's separately measured
model/finalization intervals; they are different timing boundaries.

| Root | Work | Physical batch / accumulation | Model report | Supervised model / finalization | Result |
|---|---|---|---|---|---|
| `qualify-b4` | Four frozen rows, zero updates | 4 / not applicable | 5.388205041 s | 5.667233640 / 0.576067213 s | Accepted |
| `smoke-b512` | Two disposable updates, 1024 presentations | 512 / 1 | 5.751586410 s | 5.977149627 / 0.671452916 s | Accepted; exact restore |
| `confirm-b512` | Five disposable updates, 2560 presentations | 512 / 1 | 5.641230562 s | 5.934344874 / 0.576696712 s | Accepted; exact restore |

The 512-row candidate is the largest allowed physical batch because the intended
effective batch is also 512. It passes directly, so no smaller candidate or
gradient accumulation is needed. Every smoke starts from the same deterministic
initializer. The input/shared-core/policy gradient norms are finite and positive;
both body and head weights change during disposable updates. Both smoke reports
then certify all restored parameters unchanged, and the restored checkpoint
SHA equals the initial checkpoint SHA. The smoke-terminal checkpoints are not
selected learned models. Pre-update CE and correctness recorded for these few
updates are not research outcomes.

The owning records are the
[four-row report](/home/stepan/Projects/code/.tofy-runs/looped-action-binding-20260909T201647-IST/qualify-b4/report.json),
[two-update report](/home/stepan/Projects/code/.tofy-runs/looped-action-binding-20260909T201647-IST/smoke-b512/report.json),
[five-update report](/home/stepan/Projects/code/.tofy-runs/looped-action-binding-20260909T201647-IST/confirm-b512/report.json),
and their corresponding external exits.

## Why the budget rule refused training

The confirmation reports 4.756724763 seconds for the whole five-update interval
and 0.010708019 seconds of checkpoint overhead. The frozen admission formula is

```text
reserved training time
  = 1.15 × (1150 / 5) × 4.756724763 + 0.010708019
  = 1258.1644078325 seconds
  > 600 seconds allowed.
```

In plain terms, the rule treats the five-update sample as 1/230 of the full fit
and adds 15% reserve. That rule is deliberately conservative, but the sample
contains a large selected-capture cost that is not repeated every five training
updates. The retained wall-time records place **4.504591671 seconds** in the
first selected update including profiling/export, and **0.252133092 seconds**
in the next four together. The approximately 4.5 seconds is not a separately
measured export-only duration. Training was registered to capture only updates
2, 100 and 1150, while the confirmation captures update 1.

If ordinary updates cost `u` and each selected capture adds `c`, a five-update
sample is approximately `5u + c`. Multiplying by 230 gives `1150u + 230c`,
although a three-capture fit would instead have a term near `1150u + 3c`.
This algebra explains the cadence mismatch under a stable-cost assumption; it
does not measure a corrected full-run time. Capture costs, warm-up, checkpoint
work and later updates need their own conservative measurement. Inner-step
host times are also not CUDA kernel times or a production throughput benchmark.

The [admission record](/home/stepan/Projects/code/.tofy-runs/looped-action-binding-20260909T201647-IST/admission.json)
has `accepted: false`. The preserved
[execute log](/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/operations/execute.stdout.log)
ends with `registered training budget cannot be met`; its
[outer exit](/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/operations/execute.exit.json)
is code 1, with cleanup successful. The source orders this admission check before
all 15 evaluation streams and `train-seed0`; none of those roots exists. The
scientific decision is therefore **not tested**, rather than a failed accuracy
gate. The frozen numerical scorers were never run on a completed C16 model.

## Device, profiling and process evidence

The device is an NVIDIA GeForce RTX 5060 Laptop GPU, driver 610.57.04, with
8151 MiB reported total VRAM. CUDA F32 and TF32 override 0 were required. The
maximum sampled usage across the accepted checks is **1086 MiB**, minimum
sampled reserve **7065 MiB**, and maximum sampled temperature **63°C**. Sampling
does not establish the allocator's instantaneous peak or long-run memory use.

Each root has one finalized candle-graph/Nsight bundle: **three healthy,
complete captures, zero errors and four warnings each**. GPU evidence is
available and provenance-bound; raw application labels match. Automatic GPU
correlation remains incomplete. The wired semantic spans, input/logit/loss
statistics, host scalars and training gradients do not supply the still-missing
operation/activation hotspots, allocation lifetimes, instrumented physical
memory checkpoints or device-event intervals. These `profiled_work` captures
cannot establish production timing superiority.

The [confirmation exit](/home/stepan/Projects/code/.tofy-runs/looped-action-binding-20260909T201647-IST/confirm-b512.exit.json)
owns its sampled resource maxima; each root's `.bound/summary.json` owns capture
health and binding. All three model operations and three binder operations
finish successfully with no survivors. The outer execute operation takes
25.074216430 seconds and records 62 owned PIDs gone; an independent exact
`/proc/<pid>` check finds all 62 absent. That count covers execution, not all
preparation/build operations. The CUDA build separately takes 27.456932230
seconds. No C16 process is intentionally retained.

## Reproduction and identities

Historical command, recorded in
[execute.process.json](/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/operations/execute.process.json).
The sealed campaign root must not be reused:

```bash
cd /home/stepan/Projects/code/Tofy-action-binding
/home/stepan/venvs/tensorboard/bin/python3 \
  /home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/campaign_operator.py \
  execute \
  --campaign /home/stepan/Projects/code/.tofy-runs/looped-action-binding-20260909T201647-IST \
  --spec-sha256 0d895bdc1b3191f1e98c2225e4655d9a9e1c9296a73d9fd07144bad892d59409
```

The build uses `cargo build --release --locked --offline --features
cudnn,profiling,serde_json/float_roundtrip --example action_binding_probe` with
the exact target directory and `TOFY_BUILD_COMMAND` stored in
[build.json](/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/build.json).
Python 3.14.7, NumPy 2.4.2, single-thread BLAS, Rust/Cargo 1.95.0 and Nsight
2026.4.1 identities are pinned in
[software.json](/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/software.json).

| Identity | Exact revision or SHA-256 |
|---|---|
| Executed Tofy source | `fcbf6d4a90793b2004dcaa32dd8ea5e318e11fc9` |
| candle_graph dependency | `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a` |
| Exact binary | `a3b0bd6f62898e6bc40aca5a386f235b465882242eb3321d29a8be54c0c72a67` |
| Registration | `f8577e55259a7484482dc0495c20e3403b1503d12be0971162f25bdec5d4407f` |
| Operator script | `ea88c35b1c30754c1982642427db851afe86537905bc9a385404892bef57c8dd` |
| B512 dataset | `bc1413652b6bf38cadde28c8f1fd3d88d9cf781ab55413300bc155e341c47295` |
| Initial parameter F32-bit digest, all three roots | `55d6d89a7e8a22049d074ae828cabb2f114093ec88886df67744a453c0cd364e` |
| Initial and both restored checkpoint files | `1f59081cc8dd74b0b6d1ddc59e33a80fe4a98fe3c2640c74e4a3b6eecd0f1a6e` |
| Five-update report | `6515904b906297b29df8d15f02f29e9c4a6d5ed5034bf1ec81c8f85cd46f9756` |
| Admission record | `a7fcc18aed080b301be5e5a020c88c78da1a03a7530d88bd8635dc38f8179adf` |
| Qualification analysis | `45326328e218c94e3067c60aea978ab138c948b0165049496428838da81f052c` |
| Completed qualification manifest | `8d007420b181c5d88a618d06ff486a13c4024f485718b40c9cd779e5711733e8` |

The [qualification analysis](/home/stepan/Projects/code/.tofy-runs/looped-action-binding-20260909T201647-IST/qualification-analysis.json)
is accepted as `implementation_qualification_only`: **seven disposable updates,
zero main-training updates and zero scientific evaluation streams**. It is not
a completed learning-analysis receipt. The
[completed qualification manifest](/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/completed-qualification.manifest.json)
and [verification](/home/stepan/Research/_runs/2026-09-09T182014Z-tofy-looped-action-binding/completed-qualification-verification.json)
were finalized at **20:27:10 IST**, covering **216 files / 93,056,179 bytes,
71 external bindings, 94 recorded PIDs gone and three CUDA captures**. The
outer count includes preparation beyond the 62 execution-owned PIDs. Its
SHA-256 was independently checked against the published pin, and all 94 PIDs
were again absent. This seal establishes point-in-time integrity, not immutable
storage.

## Decision and limits

C16 establishes that this model/batch can execute the bounded CUDA checks with
the required instrumentation and restore initialization. It leaves convergence,
held-out control transfer, omitted-action reasoning, depth effects and cached
visual robustness unknown. Even the proposed cached diagnostic reuses privileged
C15 initial role attention; it would not establish fresh visual learning or
native end-to-end action inference. No positive or negative learning verdict is
available from the disposable smoke sequence.

The next separately registered run is C17, under
`/home/stepan/Research/_runs/2026-09-09T192158Z-tofy-looped-binding-profile-amortization`.
Its correction concerns profiling cadence/amortization, using conservative
component maxima across both smokes plus supervisory overhead, while preserving
the 600-second training cap, 30-minute campaign wall cap, model, data, optimizer,
seed and scientific gates. The original
C16 refusal stays preserved; no post-hoc estimate authorizes training in its
root. The corrected admission must pass before training, and any eventual
one-seed result remains a finite-task screen rather than architecture promotion.
