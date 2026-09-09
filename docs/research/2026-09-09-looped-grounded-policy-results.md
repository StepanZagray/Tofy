# C12 integrated grounded policy — September 9, 2026 IST

**The completed screen does not support training feasibility.** After 1,150 updates, terminal action accuracy is **254/1024 seen, 255/1024 familiar and 127/512 held-out**, against exactly 25% for both frozen and cleared controls. All seven registered learning gates fail; all integrity checks and the independent numerical review pass. No query is correct under every map variant. This is valid negative evidence from one training seed, with decision `training_feasibility_not_supported`; `accepted: true` means that the evidence passed validation, not that learning succeeded. [analysis](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/analysis.json), [independent numerical review](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/independent-review.json).

The current agent and goal remain the highest-attention patches on every evaluated row, **but their mean attention mass falls from almost 1 to about 0.197/0.210**. The winner stays correct while the pooling weights change substantially: roughly 80% of the weight now lies elsewhere. Retained role argmax therefore does not establish intact soft pooling. Cross-entropy improves without an action-accuracy gain; lower probability loss alone is insufficient evidence of learned support binding.

## Registered experiment and comparisons

The intervention integrates the C11-validated initial-core C10 true spatial policy into the differentiable looped forward pass. The common untrained core and the exact 1,284 F32 policy parameters are loaded before training. C10 fitted the role selectors with privileged role labels and its affine action readout using oracle-pooled features; this warm start is not grounding learned from action labels alone. C11's prior 100% result used fixed known controls. C12 instead varies the control mapping and requires demonstrations to identify the appropriate action. The [C11 result](2026-09-09-looped-cuda-readout-results.md) remains valid within its original scope. [registration](/home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning/registration.md).

The synthetic task has an 8×8-cell maze, rendered as 64×64 palette pixels, with the goal one step away. The network receives three before/after demonstrations plus the current frame and public role/action/coordinate/time metadata. Simulator controls, map IDs and role coordinates remain scoring fields. Four applications of two shared transformer blocks use width 128 and four attention heads. Only policy CE trains the body and spatial head: AdamW, learning rate 0.0003, betas 0.9/0.999, epsilon `1e-8`, weight decay 0.01 and global gradient clipping at 1.0. Ordinary policy/value/reward/successor heads are excluded from optimization. There are no auxiliary role/map losses, planner, memory, LLM controller or public pretraining. [registration](/home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning/registration.md), [campaign specification](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/campaign-spec.json).

Training seed 0 runs exactly 1,150 updates, effective batch 64, totaling 73,600 rows. Each update contains four query layouts, each paired with all 16 fit maps. The layouts use data seed `20260920` and base ID `0x47524f554e445452`. Although registration allowed duplicates, the actual audit observed **4,600 distinct training query images, zero duplicate query groups and 73,600 distinct input tuples with zero duplicate tuples**. The map variants remain paired, so 73,600 rows are not independent queries. Fit maps are `[0,2,4,5,7,8,9,10,13,14,15,16,18,19,21,23]`; held-out maps are `[1,3,6,11,12,17,20,22]`. [actual population audit](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/audit/report.json).

Each evaluation cohort has 64 query groups. Seen groups are training indices `floor(i*4599/63)` paired with 16 fit maps. Familiar and held-out cohorts share the same 64 new queries, paired with 16 fit or eight held-out maps, respectively. New queries use seed `20260921`, base `0x47524f554e444556`. The audit observed 64 distinct new queries with zero overlap against all 4,600 C12 training queries and **1,280 selected historical C8/C11 queries**. That historical boundary is not all previous Tofy exposure. Frozen factual evaluation precedes training; terminal factual and cleared evaluation use only the update-1,150 checkpoint. Clearing removes support pixels while retaining current pixels and all public metadata. [actual audit and historical bindings](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/audit/report.json), [registration](/home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning/registration.md), [accepted integrity certificate](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/certificate/integrity.json).

## Counts, controls and uncertainty

| Cohort | Condition | Correct / rows | Accuracy | Mean CE | Predictions [0,1,2,3] |
|---|---|---:|---:|---:|---|
| Seen | Frozen factual | 256/1024 | 25.0000% | 1.446887 | [192,304,224,304] |
| Seen | Terminal factual | 254/1024 | 24.8047% | 1.405918 | [0,0,98,926] |
| Seen | Terminal cleared | 256/1024 | 25.0000% | 1.405731 | [0,0,0,1024] |
| Familiar | Frozen factual | 256/1024 | 25.0000% | 1.448114 | [304,272,240,208] |
| Familiar | Terminal factual | 255/1024 | 24.9023% | 1.406028 | [0,0,116,908] |
| Familiar | Terminal cleared | 256/1024 | 25.0000% | 1.405822 | [0,0,0,1024] |
| Held-out | Frozen factual | 128/512 | 25.0000% | 1.448432 | [152,136,120,104] |
| Held-out | Terminal factual | 127/512 | 24.8047% | 1.406032 | [0,0,60,452] |
| Held-out | Terminal cleared | 128/512 | 25.0000% | 1.405822 | [0,0,0,512] |

Each cohort's labels are exactly balanced: 256 examples of each action for seen/familiar, 128 for held-out. All nine streams have **0/64 all-maps-correct query groups**. Terminal predictions use only actions 2 and 3; cleared inputs always predict action 3. These are counts from the [accepted analysis](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/analysis.json), `summaries`.

| Cohort | Terminal accuracy, 95% CI | Terminal − frozen accuracy, pp [95% CI] | Terminal − cleared accuracy, pp [95% CI] |
|---|---|---|---|
| Seen | 24.8047% [24.4141,25.0000] | −0.1953 [−0.5859,0] | −0.1953 [−0.5859,0] |
| Familiar | 24.9023% [24.4141,25.2930] | −0.0977 [−0.5859,+0.2930] | −0.0977 [−0.5859,+0.2930] |
| Held-out | 24.8047% [24.4141,25.0000] | −0.1953 [−0.5859,0] | −0.1953 [−0.5859,0] |

Intervals use 10,000 paired whole-query bootstrap draws, PCG64 seeds 1930/1931/1932 for seen/familiar/held-out. Every draw preserves all map variants and shares query selections across conditions. They are pointwise empirical intervals, not simultaneous population guarantees. Balanced labels explain why a deterministic query/action-ID-only policy scores exactly 25% on a complete group; the control's `[25%,25%]` bootstrap interval is a design property. It is not evidence of population certainty. [analysis](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/analysis.json), `contrasts`, `bootstrap`, `controls`.

All three terminal factual accuracy gates require at least 90%. Both unseen-layout cohorts additionally require a strictly positive lower 95% bound against frozen factual and a lower bound above 25 percentage points against cleared. All seven fail. Because the seen-fit prerequisite fails, the registered classification is training feasibility not supported; the run does not isolate an otherwise successful fit that failed only at transfer.

| Cohort / condition | Correct when action omitted from demonstrations | Correct when action demonstrated |
|---|---:|---:|
| Seen / frozen | 72/256 | 184/768 |
| Seen / terminal factual | 64/256 | 190/768 |
| Seen / terminal cleared | 60/256 | 196/768 |
| Familiar / frozen | 60/256 | 196/768 |
| Familiar / terminal factual | 70/256 | 185/768 |
| Familiar / terminal cleared | 64/256 | 192/768 |
| Held-out / frozen | 30/128 | 98/384 |
| Held-out / terminal factual | 36/128 | 91/384 |
| Held-out / terminal cleared | 32/128 | 96/384 |

Always choosing the undemonstrated action scores 100% on that subset, 0% on demonstrated-action queries and 25% overall. The best action-ID-only predictor on the demonstrated subset reaches 1/3. Consequently, isolated omitted-subset accuracy cannot establish learned elimination. The visible simulator oracle remains exact. [analysis](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/analysis.json), `summaries.*.subsets`, `controls`; [integrity certificate](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/certificate/integrity.json).

## Role attention and probability loss

Agent, goal and joint role argmax are correct on every row of all nine streams: 1024/1024 in each seen/familiar stream and 512/512 in each held-out stream. This selects the patch with the largest individual weight; it does not measure how concentrated the entire distribution is.

| Cohort | Frozen target mass, agent / goal | Terminal factual target mass, agent / goal | Terminal cleared target mass, agent / goal |
|---|---|---|---|
| Seen | 0.999999 / 1.000000 | 0.196755 / 0.209896 | 0.196332 / 0.210084 |
| Familiar | 0.999999 / 1.000000 | 0.197357 / 0.209706 | 0.196932 / 0.209880 |
| Held-out | 0.999999 / 1.000000 | 0.197357 / 0.209706 | 0.196932 / 0.209880 |

The displayed frozen values are rounded; neither is asserted to equal one exactly. Spatial pooling forms a weighted sum of all 64 current-patch features. A correct maximum at approximately 20% mass leaves approximately 80% on other locations, so the weighting of that sum has materially changed despite perfect role-winner accuracy. This is a separate diagnostic from action binding and does not by itself identify the cause of failure or quantify the change in pooled vectors. [analysis](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/analysis.json), `summaries.*.endpoints`; [independent role counts](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/independent-review.json), `role_correct_counts`.

Terminal-minus-frozen CE is −0.040968 seen, −0.042086 familiar and −0.042399 held-out, with all paired 95% intervals below zero. This is a positive probability-loss result despite failed categorical choices: CE rewards probability assigned to the right label even when that label is not the largest probability. Uniform four-action prediction has CE `log(4) ≈ 1.386294`; terminal CE remains about 1.406. Probability calibration was not independently measured, and these losses do not prove identical representations or disconnected gradients. Terminal factual CE is also slightly worse than cleared CE, by about 0.00019–0.00021. [analysis](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/analysis.json), `contrasts.*.*.ce`.

## Integrity, qualification and resource use

The certificate accepts data, source, initialization, numerical parity, device, gradient, profiler, unchanged-unused-head, oracle and completed-training checks. Independent scoring recomputes endpoints without importing the production analyzer and agrees on **846 floating fields**, maximum absolute discrepancy **4.440892098500626e-16**. It uses the same pinned integrity certificate; it is an independent numerical calculation of this experiment, not another experiment or an independent rerun of all provenance guards. [certificate](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/certificate/integrity.json), [review scope and comparison](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/independent-review.json).

CUDA qualification checks integrated current/CLS features, pooling, attention and logits on already accessed C8/C11 inputs, at batches 1 and 4. Disposable updates check finite nonzero body/head gradients, changes in both families, unchanged unused heads and exact saved restoration. Qualification is implementation evidence only. Physical-batch tests are `64 fail, 32 pass, 48 fail, 40 fail, 36 fail, 34 pass, 35 fail`; a five-update confirmation passes at 34. Thus each effective batch uses 34 rows plus a 30-row tail, accumulation 2. Maximality uses the preregistered monotonic-capacity assumption and the adjacent 34/35 bracket. The five failed capacity roots remain excluded infrastructure records. [qualification report](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/qualification-analysis.json), [registration](/home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning/registration.md).

The five-update projection was **2,690.905426 s**, reserving **3,094.541239 s** with the 15% margin, below the 3,600 s training cap. Measured generator/checkpoint allowance was 27.222751 s versus 403.635814 s reserved. Actual training model time was **2,113.260021 s** (35 min 13.260 s), including initialization and checkpoint work; recorded optimizer-update time was 2,111.290739 s. Nsight finalization added 0.580065 s. Maximum sampled temperature was **75°C**, peak sampled usage **7,552/8,151 MiB**, minimum reserve **599 MiB**, above the 512 MiB requirement. The model and profiler process/group cleanup records pass with no survivors. [qualification](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/qualification-analysis.json), [training report](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/train-seed0/report.json), [training lifecycle](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/train-seed0.exit.json).

All 1,150 updates have finite nonzero body, head and both query-gradient norms. Body norm ranges 0.850306–166.598663; head norm 0.307234–13.634140. Thirty body tensors and all three policy tensors change, with parameter-difference L2 norms 5.204518 and 0.650736. All ordinary unused heads retain their exact initial F32 bytes. Nonzero gradients and parameter movement establish execution, not successful learning. [certificate](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/certificate/integrity.json), `gradient_norms.train-seed0`, `parameter_changes.train-seed0`.

The campaign has 21 invocations: 16 accepted and five excluded capacity failures. Its **17 healthy captures** comprise five qualification captures, three frozen evaluations, three training captures at updates 2/100/1150 and six terminal evaluations. CPU audit correctly has an empty host trace and no model work. All wired CandleGraph, host, NVTX and Nsight evidence is retained; explicit application-label matching handles the domain-prefix correlation gap. Operation/activation linkage, allocation lifetimes, instrumented physical-memory checkpoints and device-event intervals remain unwired. Sampled VRAM is not allocator peak; `profiled_work` is not production-equivalent timing evidence. [registration](/home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning/registration.md), [certificate](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/certificate/integrity.json), [source specification and gaps](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/campaign-spec.json).

At scoring completion, cumulative qualification/training/evaluation model times were **60.463947 / 2,113.260021 / 82.101744 s**; campaign wall time was **2,417.080241 s**, below 5,400 s. Build time was 19.312657 s outside the campaign clock. Analysis-stage completion was **17:39:25 IST** on September 9. [analysis-stage accounting](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/analysis-stage.json), [build record](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/campaign-spec.json).

## Exact provenance and historical commands

| Identity | Exact value |
|---|---|
| Campaign | `/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST` |
| Source revision | `5b17246cef110a866bdb8c2d2b14919710256ff3` |
| CandleGraph revision | `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a` |
| Executed binary SHA256 | `60c677a48cdf297692d5f08c978da6926c18aa7d1a27e632162ed6416079b257` |
| Registration SHA256 | `0dccf20acd84461236e4a3468864bd90ea4762ae757831e203d18fa09559410d` |
| Campaign specification SHA256 | `b22522292a3245836094d6500d9b0da7afc34c4bcd762ad55fed854170e9d5eb` |
| Initial core SHA256 | `4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802` |
| Imported initial policy SHA256 | `a43ad78fe7a6b1594d23c205fb191c8ac81610deaaff25198cb421d423edc678` |
| Terminal core SHA256 | `dc732f17a38f7bc6510dc6480a61dc4862fcb7c89ae1560f3a06515b8785d93f` |
| Terminal policy SHA256 | `37675528ef00055f16a17e826648078ff042492787087f3604d58cf0efc60fdf` |
| Integrity certificate SHA256 | `7a48877731f54615181992ae65c6d0827a9d3466c320ed4817a2b7eacd322bde` |
| Analysis SHA256 | `01072bbb85f9bb939549fe15d361e11c647bd4017c5207d7dfca10ff374afef3` |
| Independent review SHA256 | `4d0003545319225e61d2d7cd30813528ae7ec4272137ae423232460d1efad248` |
| Completed outer seal SHA256 | `0a4ac9a8d3f46ebfca141360384a18761ea4c9fd175c43176788dd422ec7ab6f` |

These hashes were checked against the named reports. The final outer seal at **17:53:48 IST** records **1,295 files totaling 418,434,719 bytes, 63 external bindings, 1,149 recorded PIDs verified gone and 17 healthy CUDA bundles**. Its classification is `completed_single_seed_negative_evidence`. The manifest digest above is retained outside the campaign root. This is point-in-time integrity, not immutable storage. [completed campaign manifest](/home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning/completed-campaign.manifest.json).

The following are recorded historical commands, **not instructions to reuse sealed roots**. Build command and source checkout come from the [campaign specification](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/campaign-spec.json). Training and analysis commands come from their exact [training operation](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/operations/train-seed0-supervisor.process.json) and [analysis operation](/home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/operations/analysis.process.json). Their recorded working directory is `/home/stepan/Projects/code/Tofy-grounded-training`; the source was reviewed, pushed and clean before launch. The supervisor adds required profiler/environment settings and checks pinned authority before execution.

```bash
CARGO_TARGET_DIR=/home/stepan/Coding/Personal/.tofy-build/v6-recipe-repair/target cargo build --release --locked --offline --features cudnn,profiling,serde_json/float_roundtrip --example grounded_policy_probe

/home/stepan/venvs/tensorboard/bin/python3 -B /home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning/supervise.py \
  --config /home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/invocations/train-seed0.json \
  --sha256 e469e997061898fd6bf6b1e4d131b2228f163775d1c83c2d924fc5a5a32d9824 \
  --authority /home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/invocations/train-seed0-authority.json \
  --authority-sha256 cc9817415d51965b5f1bcef20a397946d784107becd8462cb66697b76f08ef6a

/home/stepan/venvs/tensorboard/bin/python3 -B /home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning/analyze.py \
  --config /home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/certificate/analysis-config.json \
  --config-sha256 f24b7a8b16b61e5829d10d91d99549cf3bcc0c917c2bd15fdca87a68cb474d7e \
  --output /home/stepan/Projects/code/.tofy-runs/looped-grounded-policy-20260909T162727-IST/analysis.json
```

## Decision and remaining uncertainty

This recipe did not learn the registered task, including its seen-training evaluation subset. The next **C13 exploratory diagnostic** compares the retained terminal unscaled head with the same head's two query vectors multiplied by **16**, using only the seen factual cohort. The final body and output decoder weights/bias remain fixed; no fitting or new population is involved. The question is whether restoring attention concentration alone is sufficient to improve action binding on those retained examples. It does not establish the original failure's cause or prove information absence if it fails. This planned check selects no new training recipe and does not revise C12's gates or decision.

The result does not prove that balanced map groups force optimization failure, that the body is disconnected, or that support information is absent. Equal predictions can coexist with input-dependent gradients, and the recorded gradients are nonzero. Loss improvement and retained role winners do not establish useful binding either. This is one seed with a privileged warm start on distance-one synthetic tasks, not an ARC score or method promotion. Dynamics, reward/value objectives, planning, episode memory and useful recurrent-depth behavior were not trained or tested; four executed loops alone do not establish that recurrence helps.
