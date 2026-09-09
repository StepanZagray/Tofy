# C13 frozen query sharpening — September 9, 2026 IST

**Restoring concentrated attention does not recover action accuracy with the fixed terminal body and decoder.** Multiplying C12's two learned attention-query vectors by 16 produces recorded F32 target mass **1 for both roles on all 1,024 rows**, yet action accuracy is **256/1024 (25%)**, versus **254/1024 (24.8047%)** without scaling. The paired improvement is **+0.1953125 percentage points, 95% CI [0,0.5859375]**; its lower bound is not strictly positive. CE worsens from **1.405918 to 1.430675**. Concentration and integrity checks pass, yielding the valid exploratory negative decision `concentration_insufficient_for_accuracy_recovery`. No Best So Far metric improves. [analysis](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/analysis.json), [independent review](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/independent-review.json).

## One fixed intervention on reused examples

[C12](2026-09-09-looped-grounded-policy-results.md) trained a recurrent body and spatial policy for 1,150 updates from a privileged C10 warm start. Role argmax stayed correct, but mean target attention mass fell to approximately 0.197/0.210. C13 asks a narrower question: is restoring concentration sufficient to recover accurate actions **with the C12 terminal core and affine output decoder fixed**? It does not retrain the model or identify the cause of C12's failure. The factor 16 was selected after observing C12 and fixed before inspecting C13 outcomes; there is no temperature sweep. [registration](/home/stepan/Research/_runs/2026-09-09T165412Z-tofy-looped-query-sharpening/registration.md).

Only the two learned 128-dimensional query vectors are scaled, by the exact power-of-two F32 transformation `queries *= 16`. The head file's header/layout, affine output weight/bias and all other bytes remain unchanged. The C12 terminal core is byte-identical. Both primary and independent prechecks validate the head transformation. Four shared loops, inputs, order, labels, public metadata and physical batch remain fixed. There are **zero optimizer updates** and no gradient capture is expected. [transformation](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/transformation.json), [independent head precheck](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/independent-head-precheck.json), [model configuration](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/config.json).

The population is exactly C12's retained seen factual cohort: **64 query groups, each paired with all 16 fixed fit maps, totaling 1,024 rows**. These are selected training-query examples, not fresh confirmation. Labels contain 256 examples of every action. The comparison reuses the retained unscaled terminal output and the same audited inputs for one sharpened forward. Physical evaluation batch is 34 with a final tail of 4; there is no gradient accumulation or new batch search. No familiar/held-out cohort, cleared-support condition, alternative multiplier, checkpoint choice or refit is eligible. The C10 warm start used privileged role supervision and an oracle-pooled affine fit; this history remains a limitation. [registration](/home/stepan/Research/_runs/2026-09-09T165412Z-tofy-looped-query-sharpening/registration.md), [pinned baseline, treatment and audit bindings](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/analysis-config.json).

## Concentration mechanism and its measured check

For probabilities `p = softmax(s)`, positive scaling obeys `softmax(16s)_i = p_i^16 / Σ_j p_j^16`. It preserves a unique maximum and concentrates mass on that winner. This algebra concerns attention weights, not whether the resulting action is correct. Current features already contain contextual computation, and the affine decoder may have adapted to diffuse pooling.

Before CUDA execution, the fixed transformation of retained attention predicted minimum agent/goal target masses **0.9999999999999998 / 0.9999999999999929**, exceeding the registered 0.99 threshold for every row. The premise passed without tuning the scale. Complete current-token feature arrays were not retained by C12, so the new pooled vectors and action logits required one actual frozen forward; old attention and already-pooled vectors alone could not reconstruct them. [premise](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/premise.json), [registration](/home/stepan/Research/_runs/2026-09-09T165412Z-tofy-looped-query-sharpening/registration.md).

| Role endpoint | Unscaled C12 terminal | Queries ×16 |
|---|---:|---:|
| Correct agent winner | 1024/1024 | 1024/1024 |
| Correct goal winner | 1024/1024 | 1024/1024 |
| Agent target mass, minimum / mean | 0.186545 / 0.196755 | 1 / 1 |
| Goal target mass, minimum / mean | 0.200323 / 0.209896 | 1 / 1 |

The treatment's ones are the recorded F32 probabilities, not a claim of exact mathematical probability one or population-perfect grounding. Actual attention matches normalized retained `p^16` with maximum absolute error **7.105427357601002e-15**, within the registered `1e-4 + 1e-4*abs(reference)` tolerance. Reconstructing logits from the new pooled vectors and unchanged affine decoder gives maximum error **3.393285644248678e-6**, within `2e-5 + 2e-5*abs(reference)`. These controls verify the intended intervention and forward calculation; they do not establish correct support binding. [analysis controls](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/analysis.json), [independent controls and baseline masses](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/independent-review.json).

## Action counts, probability loss and decision

| Endpoint | Unscaled C12 terminal | Queries ×16 |
|---|---:|---:|
| Correct / rows | 254/1024 | 256/1024 |
| Accuracy | 24.8046875% | 25.0000000% |
| Mean CE | 1.4059184908353308 | 1.4306748416357902 |
| Predictions [0,1,2,3] | [0,0,98,926] | [0,0,752,272] |
| All 16 maps correct / query groups | 0/64 | 0/64 |
| Correct when action omitted from demonstrations | 64/256 (25.0000%) | 56/256 (21.8750%) |
| Correct when action demonstrated | 190/768 (24.7396%) | 200/768 (26.0417%) |

The treatment changes many predicted actions, but the net gain is only two correct outputs. Exact 25% constant/action-ID-only accuracy follows from complete balanced map groups. Choosing the omitted action always would score 100% on the omitted subset and 25% overall, so that subset alone cannot establish learned elimination. No all-map query succeeds under either condition. [analysis](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/analysis.json), `baseline`, `treatment`, `contrasts`.

| Paired contrast | Estimate | Pointwise 95% CI |
|---|---:|---|
| Accuracy, treatment − baseline | +0.1953125 pp | [0,+0.5859375] pp |
| Accuracy, treatment − exact constant | 0 pp | [0,0] pp |
| CE, treatment − baseline | +0.0247563508 | [+0.0232952438,+0.0262318180] |

Higher CE means worse probability assigned to the correct labels on average. Thus sharper role selection alters the action distribution and worsens probability loss, without a demonstrated accuracy gain. The 25% accuracy remains far below the 90% recovery threshold. The lower accuracy-delta bound equals zero, so neither the full-recovery rule nor the partial-gain rule passes. The registered decision is therefore `concentration_insufficient_for_accuracy_recovery`; the concentration and integrity controls themselves succeeded.

The bootstrap uses **10,000 whole-query resamples, PCG64 seed 1940**, preserving all 16 map variants and sharing draws across contrasts. Linear 2.5%/97.5% quantiles are pointwise and conditional on the fixed maps, frozen checkpoint and selected seen-query population. They do not estimate uncertainty over unseen mappings, new training seeds or an independently selected population. This is one fixed exploratory contrast with no subgroup promotion. [registration](/home/stepan/Research/_runs/2026-09-09T165412Z-tofy-looped-query-sharpening/registration.md), [analysis](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/analysis.json).

## Verification, preparation history and review limits

The independent scorer agrees on **27 floating fields**, maximum absolute discrepancy **4.163336342344337e-17**. It independently checks numerical endpoints and head bytes; it shares the pinned receipt for source/core/runtime/profiler/parent-seal provenance. This is a separate calculation of the same experiment, not another experiment or a second full provenance audit. [independent review](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/independent-review.json), [integrity receipt](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/integrity-receipt.json).

A preparation import error is preserved as `failed_setup_attempt_no_model_work`: an interactive wrapper resolved R12's `independent_review` after the diagnostic changed `sys.path`. The wrapper then used an explicit file-based import of the frozen C13 verifier. The record states **zero model calls and no source/data changes**. This setup failure is excluded from model evidence; it is not silently presented as a successful first attempt. [preparation error record](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/preparation-import-error.json).

The actual Opus review also has two distinct attempts. The first timed out after **180.547192 s**, returned an empty response and supplied no usable review. A focused retry completed in **97.088147 s**; its response identifies `claude-opus-5` plus auxiliary `claude-haiku-4-5-20251001` usage, with `is_error: false`. Both cleanup records report all owned processes gone. These were advisory CLI calls during preparation, not a model controller or experimental evidence. [failed attempt](/home/stepan/Research/_runs/2026-09-09T165412Z-tofy-looped-query-sharpening/operations/opus-c13-review-01.exit.json), [completed retry](/home/stepan/Research/_runs/2026-09-09T165412Z-tofy-looped-query-sharpening/operations/opus-c13-review-02.exit.json), [actual retry response](/home/stepan/Research/_runs/2026-09-09T165412Z-tofy-looped-query-sharpening/operations/opus-c13-review-02.stdout.log).

Primary review corrected the advisor's unsupported claims before launch. Whole-query bootstrap is appropriate to the stated fixed-map conditional estimand; complete current features were absent, preventing an offline replacement of the frozen forward; and full self-attention offers a route for demonstration context into role-token states, without proving that those states encode the required mapping. The per-row concentration premise was retained, no scale ladder was adopted, and neither recovery nor failure was allowed to identify the original optimization cause. Advisor agreement is not proof. [review and primary corrections](/home/stepan/Research/_runs/2026-09-09T165412Z-tofy-looped-query-sharpening/findings/opus-c13-review.md).

## Execution, profiler coverage and final seal

The unchanged C12 CUDA binary runs from numerical source **`5b17246cef110a866bdb8c2d2b14919710256ff3`**; C13's separately reviewed operator snapshot is **`f802bef409bcf14d79d99cad0843d320a61e6ab7`**. No numerical-source rebuild or training occurs in C13. Frozen source, parent, core, affine bytes, input identities and zero updates pass verification. [launch specification](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/launch-spec.json), [model report](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/sharpened-seen/report.json), [receipt](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/integrity-receipt.json).

CUDA model phase: **9.841228191 s**; finalization: **0.574907523 s**; combined model/profiler lifecycle: **10.416136591 s**. Peak sampled memory is **990/8,151 MiB**, minimum sampled reserve **7,161 MiB**, maximum temperature **57°C**. The CPU premise outer operation takes 0.548495 s, binder 0.861310 s, analysis 0.761885 s and independent review 0.444008 s. Execution through analysis is **107.42125 s**, below the 600 s cap; earlier implementation/adviser work is excluded from that clock and reported separately above. Recorded model/profiler and outer operation cleanup checks pass with no survivors. [CUDA lifecycle](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/sharpened-seen.exit.json), [premise timing](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/operations/premise.exit.json), [binder timing](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/operations/bind.exit.json), [analysis timing](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/operations/analysis.exit.json), [review timing](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/operations/independent-review.exit.json).

One first-forward CUDA capture is structurally valid and complete, with zero health errors, four warnings and verified raw application labels. It retains CandleGraph semantic/tensor/host evidence, Chrome/NVTX, and Nsight CUDA/cuDNN/cuBLAS/OS-runtime tracing with CPU sampling. Automatic GPU correlation remains incomplete; explicit application-label matching is preserved. Operation/activation linkage, logical allocation lifetimes, instrumented physical-memory checkpoints and device-event intervals remain unwired. Zero gradient evidence is expected for frozen evaluation. Sampled VRAM is not allocator peak, and `profiled_work` is not production timing evidence. [bound capture summary](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/sharpened-seen.bound/summary.json), [profiling contract](/home/stepan/Research/_runs/2026-09-09T165412Z-tofy-looped-query-sharpening/registration.md).

The outer seal at **18:11:41 IST** verifies **95 files / 41,816,772 bytes, 86 external bindings, 50 recorded PIDs gone and one healthy CUDA bundle**. Classification is `completed_exploratory_negative_evidence`. The digest is recorded outside the campaign root; this is point-in-time verification, not immutable storage. [completed manifest](/home/stepan/Research/_runs/2026-09-09T165412Z-tofy-looped-query-sharpening/completed-campaign.manifest.json), [seal verification](/home/stepan/Research/_runs/2026-09-09T165412Z-tofy-looped-query-sharpening/completed-campaign-verification.json).

| Identity | Exact value |
|---|---|
| Campaign | `/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST` |
| CandleGraph revision | `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a` |
| Unchanged binary SHA256 | `60c677a48cdf297692d5f08c978da6926c18aa7d1a27e632162ed6416079b257` |
| Registration SHA256 | `7972d6dec2bb278d34174192a2a9aa224a0fc206af675daef11b3a0b23e5e658` |
| Launch specification SHA256 | `7017a881b124070b789b870797051492ebfcca93294e22f9c14114b02d0a0c57` |
| Frozen terminal core SHA256 | `dc732f17a38f7bc6510dc6480a61dc4862fcb7c89ae1560f3a06515b8785d93f` |
| Original terminal head SHA256 | `37675528ef00055f16a17e826648078ff042492787087f3604d58cf0efc60fdf` |
| Queries ×16 head SHA256 | `0f721fb70fbb4b77c3c8463c8ada9fa6c84c9a65fea3f305d243ed20787380a3` |
| C12 parent outer seal SHA256 | `0a4ac9a8d3f46ebfca141360384a18761ea4c9fd175c43176788dd422ec7ab6f` |
| Integrity receipt SHA256 | `7209ffb146fd7db4ad852a74ce3086692cf877c49d34c86bf02b124f258e64c6` |
| Analysis configuration SHA256 | `407e4b1ec8a94ef0b01c18836a149be1d41615749c5fa22c5c7c833c440245a8` |
| Analysis SHA256 | `9ba7ba2e9619fbfc5f5b39913c09d9fc130a1bbe4d7fd447c9336281e1e18042` |
| Independent review SHA256 | `d6dcf7a52d65a1e0a996cb0c68a004c3ca309353096c9e6ea1f7d8078da1eb13` |
| Completed outer seal SHA256 | `fbaef96bc4e520301edfb2b79e811467d226e45509a7aee078ebdfdebabcac6f` |

## Recorded commands

These are historical commands from the [evaluation operation](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/operations/evaluate.process.json) and [analysis operation](/home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/operations/analysis.process.json), with working directory `/home/stepan/Projects/code/Tofy-grounded-training`. **Do not reuse their sealed roots.** The supervisor supplies the frozen environment and profiler contract; C13 reuses C12's existing binary.

```bash
/home/stepan/venvs/tensorboard/bin/python3 -B /home/stepan/Research/_runs/2026-09-09T143629Z-tofy-looped-grounded-policy-learning/supervise.py \
  --config /home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/config.json \
  --sha256 1fbb9638963c282847fcde9c7c6d2b2db90dfa901a4e0fd316a0cee6055c3b78 \
  --authority /home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST/authority.json \
  --authority-sha256 fd2bc2e5cece75d2756cf048037a1a879d18fb1d49c65e5935d6783379247d74

/home/stepan/venvs/tensorboard/bin/python3 -B /home/stepan/Research/_runs/2026-09-09T165412Z-tofy-looped-query-sharpening/diagnostic.py \
  analyze --campaign /home/stepan/Projects/code/.tofy-runs/looped-query-sharpening-20260909T180742-IST
```

## Decision and next question

This intervention successfully concentrates attention and fails to recover correct actions with the retained terminal body and decoder on these seen examples. It does not show that the original diffusion caused C12's failure, that support information is absent, that a different decoder could not use the features, or that concentrated role states cannot carry demonstration context. It also supplies no held-out generalization, useful-recurrence, ARC, planner, dynamics or episode-memory result. The privileged warm start and selected reused population preclude promotion.

The next separately registered check is a **frozen linear action-readout witness on retained C12 features**. Its purpose is to distinguish what an alternative readout can extract from those fixed features from what the retained decoder currently predicts. No new end-to-end training recipe, architecture or guaranteed outcome is selected by C13, and the next test must not retroactively revise this negative decision.
