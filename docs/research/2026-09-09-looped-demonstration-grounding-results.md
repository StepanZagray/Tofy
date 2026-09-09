# C15 frozen demonstration grounding — September 9, 2026 IST

**Both frozen core/head pairs correctly locate the agent and goal in every demonstration frame on this reused panel.** Each arm passes all **21/21 selector-reuse gates**, with **1024/1024** correct locations per role/frame and **3072/3072** correct demonstration displacements. This supports direct reuse of these particular role selectors on the six support frames. It does not establish learned action binding: the unchanged current policy still scores **256/1024 initial and 254/1024 final**, and final support attention assigns only about **16–19%** of its weight to the correct cells despite choosing them as winners. No Best So Far policy or ARC metric improves. [Numerical analysis](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/analysis.json), [independent review](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/independent-review.json).

## Question and fixed comparison

C15 follows the failed [C12 policy-only training screen](2026-09-09-looped-grounded-policy-results.md), [C13 query-sharpening intervention](2026-09-09-looped-query-sharpening-results.md), and [C14 fixed affine witness](2026-09-09-looped-affine-action-witness-results.md). Those results did not establish whether the existing role selectors could locate objects in demonstrations. Current-frame success alone could not answer that question: frame/time metadata and contextual attention can change the feature coordinates presented to the same selector. C15 tests this narrow empirical prerequisite before a separate learned binding experiment. It is a project diagnostic, with no paper-faithful or global-optimality claim. [Registration](/home/stepan/Research/_runs/2026-09-09T173927Z-tofy-looped-demonstration-grounding/registration.md).

Both arms use exactly the C12 seen factual panel: **64 selected training-query groups ×16 fixed fit mappings = 1024 rows**, seed `20260920`, with original training indices `floor(i*4599/63)` for `i=0..63`. Map order is `[0,2,4,5,7,8,9,10,13,14,15,16,18,19,21,23]`. These are previously observed training queries, not held-out confirmation. Initial means the C12 initial core plus its privileged C10 role-supervised head; final means the C12 terminal core plus its own terminal head. The initial head is already fitted, so initial success receives no credit as learning in C12 or C15. Initial/final contrasts change the complete core/head pair and cannot isolate a body-training effect. [Input and reference bindings](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/analysis-config.json).

The model runs its unchanged width-128, two-block core for **four shared loops**, then exposes the existing post-final-RMS features in order `before0, after0, before1, after1, before2, after2, current`. The same frozen spatial head is applied to each frame. Each batch executes the core once and the head seven times; frame 6 reuses the ordinary current output. There are **zero optimizer updates, zero fitting, no losses/backward passes and no changed parameter tensors**. Each arm uses physical batch **34**, with final batch **4**; gradient accumulation is inapplicable. This matches the parent evaluator, without a new training-capacity search. Each arm records 31 core batches and 217 spatial-head batches. [Initial report](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/frames-initial/report.json), [final report](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/frames-final/report.json).

Only public pixels and the existing public metadata enter inference. The evaluator obtains role locations from visible colors to score the learned attention winners, then compares their before/after displacement with the visible movement. Those coordinates, control maps and scoring labels are not fed into a new policy. This deterministic displacement calculation is an evaluation metric; **no hardcoded action controller is installed or credited**. Each exported row carries seven public cell grids and one complete metadata array so the scorer can reconstruct the original input bytes and hashes. [Registered input boundary](/home/stepan/Research/_runs/2026-09-09T173927Z-tofy-looped-demonstration-grounding/registration.md), [accepted integrity receipt](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/integrity.json).

## Location, displacement and action results

Every location/displacement entry below is identical in the two arms. A role's selected location is the maximum-attention cell, with the first cell winning an exact tie.

| Metric | Initial | Final |
|---|---:|---:|
| Agent correct, separately for each of all seven frames | 1024/1024 | 1024/1024 |
| Goal correct, separately for each of all seven frames | 1024/1024 | 1024/1024 |
| Both roles correct, separately for each frame | 1024/1024 | 1024/1024 |
| Both roles correct in all six support frames in the row | 1024/1024 | 1024/1024 |
| Correct displacement, separately for support pairs 0, 1 and 2 | 1024/1024 | 1024/1024 |
| Both agent locations correct, separately for each support pair | 1024/1024 | 1024/1024 |
| Correct displacements, pooled across three pairs | 3072/3072 | 3072/3072 |
| All three displacements correct in the row | 1024/1024 | 1024/1024 |
| Unchanged current policy action | 256/1024 (25%) | 254/1024 (24.8047%) |
| Registered selector-reuse components passed | 21/21 | 21/21 |

The gate requires at least 99% for both roles in each support frame, at least 98% for their joint correctness in each frame, and at least 98% for each of the three displacements. Both arms meet every component. Attention mass and confidence intervals are descriptive, not extra gate criteria. Correct displacement can occur even with wrong absolute positions; that potential shortcut is measured separately here, and both absolute positions are also correct in every pair. [All counts and gate components](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/analysis.json), `summaries`, `gates`.

The one-hot visible-role positive control has 100% location/displacement accuracy and mass 1. Uniform attention has 0% location/displacement accuracy and mass `1/64 = 0.015625`. Uniform ties select cell 0, verified to contain neither role; every true demonstration displacement is nonzero. These analytic controls validate scoring sensitivity and are not empirical policies. [Control results](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/analysis.json), `summaries/*/controls`.

## Correct winners do not mean concentrated pooling

Attention mass is the fraction of pooling weight assigned to the true cell. A selector can rank that cell first while placing most of its total weight elsewhere. Initial weights are nearly one-hot; final weights remain correctly ranked but are much more diffuse.

| Frame | Initial agent mean mass | Initial goal mean mass | Final agent mean mass | Final goal mean mass | Final agent minimum | Final goal minimum |
|---|---:|---:|---:|---:|---:|---:|
| before0 | 0.999999674 | 0.999999509 | 0.163355 | 0.187115 | 0.160906 | 0.184004 |
| after0 | 0.999999566 | 0.999999612 | 0.162760 | 0.185410 | 0.159536 | 0.182342 |
| before1 | 0.999999587 | 0.999999360 | 0.162643 | 0.185456 | 0.159627 | 0.182618 |
| after1 | 0.999999493 | 0.999999498 | 0.162162 | 0.184074 | 0.158731 | 0.181113 |
| before2 | 0.999999545 | 0.999999279 | 0.162181 | 0.184657 | 0.158739 | 0.181427 |
| after2 | 0.999999439 | 0.999999423 | 0.161734 | 0.183501 | 0.158508 | 0.180261 |
| current | 0.999999331 | 0.999999528 | 0.196755 | 0.209896 | 0.186545 | 0.200323 |

For example, final `before0` mean agent mass is **0.163354508 [0.163110398,0.163596316]**, and goal mass is **0.187114979 [0.186664412,0.187562196]**. Initial minimum mass across every frame and role is 0.999998808. All 14 final-minus-initial mean-mass contrasts are negative with intervals below zero. This documents changed soft pooling, not a mechanism explaining the earlier action-learning failure. In particular, C13 already showed that its one fixed sharpening intervention was insufficient to recover actions with the terminal body/decoder. [Full means, minima and paired intervals](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/analysis.json), [C13 limits](2026-09-09-looped-query-sharpening-results.md).

Bootstrap uses **10,000 paired whole-query-group draws**, PCG64 seed `1944`, resampling 64 groups while keeping all 16 mappings and seven frames together. Linear percentile 95% intervals are pointwise and conditional on these fixed checkpoints, maps and selected queries. Every location/displacement interval is `[1,1]`, and each corresponding final-minus-initial interval is `[0,0]`, because all observed rows are correct. These empirical intervals do **not** guarantee population perfection. No minima are bootstrapped, no simultaneous significance or training-seed uncertainty is claimed, and no best frame/checkpoint is selected.

Current action accuracy remains **25% [25%,25%] initial** and **24.8047% [24.4141%,25%] final**. Its final-minus-initial contrast is **−0.1953125 percentage points [−0.5859375,0]**. These are unchanged parent outputs, included for parity rather than a new policy comparison. [Registered uncertainty and complete contrasts](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/analysis.json).

## Integrity, runtime and profiler limits

The exact new CUDA binary first passes four-row physical-batch-4 qualification against the sealed C12 reference. Maximum errors for logits, attention, pooled features, current features and CLS are all **0**. Across both full 1024-row streams, current logits/attention/pool also have **zero maximum error** against their C12 counterparts; action and role winners match exactly. Frame 6 attention/pool match the current output exactly. All seven external integrity checks pass: source, checkpoints, zero updates, unchanged parameters, qualification, profiles and cleanup. [Qualification](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/qualification.json), [integrity](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/integrity.json).

The frozen independent reviewer imports no primary metric functions. It separately reconstructs public pixels/metadata and hashes, scalar role/displacement metrics, paired bootstrap intervals, counts and exact gates. All **445 floating fields** agree within the registered tolerance, with maximum absolute difference **1.1102230246251565e-16**; **127 discrete fields** agree exactly. Runtime/source/checkpoint/profile acceptance is shared with the pinned external receipt, and target hashes inherit the exact C12 identities rather than a new simulator reconstruction. Primary analysis takes 4.773628 s internally; independent review takes 4.561319 s. Their supervised wall times are 4.874893 s and 4.752445 s, both with accepted exits and no surviving owned processes. [Independent review](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/independent-review.json), [analysis exit](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/operations/analysis.exit.json), [review exit](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/operations/independent-review.exit.json).

| Invocation | Model-reported elapsed, s | Supervised model phase, s | Profiler finalization, s | Peak sampled VRAM, MiB | Minimum sampled reserve, MiB | Maximum temperature |
|---|---:|---:|---:|---:|---:|---:|
| Qualification, batch 4 | 4.581291 | 4.968091 | 0.570996 | 414 | 7737 | 49°C |
| Initial frames, batch 34 + tail 4 | 9.103012 | 9.556341 | 0.611955 | 990 | 7161 | 58°C |
| Final frames, batch 34 + tail 4 | 8.678822 | 8.967834 | 0.568254 | 990 | 7161 | 59°C |

The device reports 8151 MiB total memory. Qualification and the two model invocations plus binding take **30.588851 s** of operator execution wall time; analysis/review occur afterward. The release CUDA build takes **25.846733 s**. All model phases remain below their 120-second limit; the separate supervised invocation limit is 600 seconds and the registered campaign limit is 20 minutes after build. These durations include instrumented work and are not production throughput benchmarks. [Execution](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/execution.json), [build record](/home/stepan/Research/_runs/2026-09-09T173927Z-tofy-looped-demonstration-grounding/build.json), [qualification exit](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/qualify-b4.exit.json), [initial exit](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/frames-initial.exit.json), [final exit](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/frames-final.exit.json).

All three first-batch captures are structurally valid and complete, with zero errors and four recorded warnings each. Wired evidence includes semantic spans, labelled frame feature/attention/pool statistics, host tracing and matching NVTX, plus bound Nsight GPU timeline, kernel, memory-operation and runtime summaries. The raw application capture/forward labels each occur once. **Automatic CandleGraph GPU correlation remains incomplete** because it compares domain-prefixed/library labels literally; raw application-label checks pass separately. Tensor coverage is partial. Operation/activation linkage, allocation lifetimes, instrumented physical-memory checkpoints and device-event timing remain unavailable. Gradient coverage is absent by design because these are frozen forwards. Host durations are not CUDA kernel durations, and sampled telemetry is not allocator peak. `profiled_work` does not support production timing claims. [Initial bundle summary](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/frames-initial.bound/summary.json), [final capabilities and gaps](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/frames-final.bound/export-0/overview.json), [raw label verification](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/frames-final.bound/export-0/application-label-check.json).

A prelaunch combined-test import mistake loaded the historical C12 review tests instead of the intended C15 module: 25 tests ran where 35 were expected. The primary caught the mismatch before model launch, removed the lingering temporary import path and repeated validation. The independent C15 suite had separately passed its 13 tests. This was an infrastructure correction before outcomes; it changed no checkpoint, data, threshold or scoring arithmetic. [Preserved correction record](/home/stepan/Research/_runs/2026-09-09T173927Z-tofy-looped-demonstration-grounding/findings/test-import-correction.md).

## Exact provenance and historical commands

| Identity | Exact value |
|---|---|
| Campaign | `/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST` |
| Reviewed launch source | `61e1335670239627772088a2fb6c81c2df499ed2` |
| CUDA binary SHA256 | `274dd66ca13d9d6cfab140f291dfcb7f5f4ef7945d0faf6b0f821e61f26a62b3` |
| CandleGraph revision | `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a` |
| Initial core SHA256 | `4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802` |
| Initial privileged head SHA256 | `a43ad78fe7a6b1594d23c205fb191c8ac81610deaaff25198cb421d423edc678` |
| Final core SHA256 | `dc732f17a38f7bc6510dc6480a61dc4862fcb7c89ae1560f3a06515b8785d93f` |
| Final head SHA256 | `37675528ef00055f16a17e826648078ff042492787087f3604d58cf0efc60fdf` |
| Registration SHA256 | `1c84ed4567f62cf71678d9b3129091436c5ebefdff008d9604644b5ea95dbc35` |
| Analysis configuration SHA256 | `c8077d6b1ed947e8bbf0c32a33ce2e04743190fac3f40007c66b6dc4cb60eb43` |
| Numerical analysis SHA256 | `5c32aee44af42459de2db19d960f43e25afc8822115b8fd9a957744960609f03` |
| Independent review SHA256 | `ab415a7c09004094a981b26dc7e6511ef1b12a255ceaf9b774206c4ef6835fde` |
| Integrity receipt SHA256 | `93a2ee7a4e92d01bda79a3aca16b1511a5e6d6a8fa605d48cdc6fe0082a04380` |
| Qualification SHA256 | `2ece57ef7555bf43454934e60755f5e718bc297ec8cab90823e29bb66e9a997d` |
| Completed outer seal SHA256 | `6007293c5f51d263b0e3f40dcd819124efa8ffdaf2809202eda0c60142ffd40d` |

The source was committed, pushed and clean before launch, and the run owns a separately hashed binary. The completed outer seal at **19:14:20 IST** verifies **217 files / 240,074,195 bytes, 109 external bindings, 103 recorded PIDs gone and three CUDA bundles**. Classification is `completed_exploratory_grounding_evidence`. The manifest digest is retained outside the campaign root; this is point-in-time integrity, not immutable storage. [Completed manifest](/home/stepan/Research/_runs/2026-09-09T173927Z-tofy-looped-demonstration-grounding/completed-campaign.manifest.json), [verification](/home/stepan/Research/_runs/2026-09-09T173927Z-tofy-looped-demonstration-grounding/completed-campaign-verification.json).

The build used:

```bash
CARGO_TARGET_DIR=/home/stepan/Coding/Personal/.tofy-build/v6-recipe-repair/target \
  cargo build --release --locked --offline \
  --features cudnn,profiling,serde_json/float_roundtrip --example grounded_policy_probe
```

The recorded command lists below ran from `/home/stepan/Projects/code/Tofy-demonstration-grounding`. The operator sequences exact-binary qualification, both frozen arms and Nsight binding from the pinned launch specification. Each model invocation records `--config` and its exact `--config-sha256`; profiler environment and external supervision belong to the operator. These are historical commands, not instructions to reuse completed roots. [Launch specification](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/launch-spec.json), [operator process record](/home/stepan/Research/_runs/2026-09-09T173927Z-tofy-looped-demonstration-grounding/operations/execute-campaign.process.json), [analysis process](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/operations/analysis.process.json), [review process](/home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/operations/independent-review.process.json).

```bash
/home/stepan/venvs/tensorboard/bin/python3 /home/stepan/Research/_runs/2026-09-09T173927Z-tofy-looped-demonstration-grounding/campaign_operator.py execute \
  --campaign /home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST \
  --spec-sha256 44fc839f8f75687dc68f8cad9b7fa991d6d24362420f3829804a61493ab043e7

/home/stepan/venvs/tensorboard/bin/python3 /home/stepan/Research/_runs/2026-09-09T173927Z-tofy-looped-demonstration-grounding/analysis.py \
  --config /home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/analysis-config.json \
  --output /home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/analysis.json

/home/stepan/venvs/tensorboard/bin/python3 /home/stepan/Research/_runs/2026-09-09T173927Z-tofy-looped-demonstration-grounding/independent_review.py \
  --config /home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/analysis-config.json \
  --report /home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/analysis.json \
  --output /home/stepan/Projects/code/.tofy-runs/looped-demonstration-grounding-20260909T191228-IST/independent-review.json
```

## What this permits next

The specific “these selectors cannot locate demonstration objects” concern is unsupported on this panel. Direct reuse meets the registered engineering thresholds at both checkpoints. The next decision is to design and preregister a **learned before/after action-binding experiment** separately; this result does not select a training recipe or authorize automatic training.

The scope matters. Support scenes have no walls and **always place the goal at cell 63**, so perfect support-goal selection alone cannot distinguish visual role grounding from a position shortcut. The agent moves, and every predicted displacement is correct, but the score does not show that a neural policy uses those movements to bind public action IDs. Correct argmax locations also do not validate the concentration of the soft pooled representation. This reused, privileged, single-lineage diagnostic provides no fresh generalization, multi-seed promotion, architecture optimality, ARC, planner, dynamics, episode-memory or useful-recurrence result. It does not identify the cause of C12's failed action learning or prove anything about information absence elsewhere in the model.
