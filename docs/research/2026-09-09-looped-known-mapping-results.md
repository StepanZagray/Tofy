# Known controls do not rescue this looped training recipe

September9,2026 IST. Training the unchanged from-scratch looped transformer with a single known control mapping completed all 1,150 updates, but both registered spatial prerequisites failed. On 64 new layouts, factual policy scores15/64 (23.4375%), below the best constant action18/64 (28.125%). Every successor output still matches a goal-completion template. This is a completed negative synthetic experiment, not an ARC score or an architecture promotion.

## Design and provenance

[Preregistration](2026-09-09-looped-known-mapping.md). The intervention replaces the 16 variable training mappings with mapping 0 ([up,down,left,right]), retaining the same fresh query order, normal factual support, model, objective and optimizer. This removes per-episode rule inference but makes the distribution easier and reduces distinct complete inputs. It is not a fair improvement comparison with variable controls and cannot uniquely identify an internal cause.

Source `71ff87ad76400e977e8e3031afa437727af73ed5`; Candle Graph `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a`; exact CUDA/profiling binary SHA256 `ffc2df87faeab88c67244f6d87facdc8ee43ebc586f3e0840e47e9f369baa620`. Initial weights `4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802`; final update1150 weights `e379802fa8e75327061f3dcc5ae95697f83e2d2d137a69ef68daf3f902040d41`.

Campaign `/home/stepan/Projects/code/.tofy-runs/looped-known-mapping-20260909T095023-IST`. Model: 992,393 parameters, hidden 128,4 heads,2 shared transformer blocks, training loops1/2/4 cyclic, inference4, max8. AdamW learning rate 0.0003, weight decay 0.01, clipping 1; policy/reward/value/successor weights1/0.1/0.1/0.5. Seed 0, data seed 9173,73,600 samples,4,600 unique queries and inputs. Physical batch 33 plus tail 31, accumulation 2, effective 64. Each query repeats 16 consecutive times, giving only 4 distinct queries per update and one training depth per query.

Final evaluation: 64 distinct new distance-one queries, seed 20260912, IDs0x4b4e4f574e+index, one known mapping, factual and cleared support, four actions. All 64 query hashes are absent from training. Unlike unknown-control evaluation, cleared history has no 25% information ceiling: the fixed mapping permits solving from query geometry.

## Results

| Endpoint | Initial factual | Final factual | Final cleared |
|---|---:|---:|---:|
| Policy correct |18/64|15/64|12/64|
| Policy CE |1.449917|1.426638|1.426511|
| Best constant policy |18/64|18/64|18/64|
| Exact successors |0/256|64/256|64/256|
| Copy-current exact baseline |72/256|72/256|72/256|
| Blocked exact |0/72|0/72|0/72|
| Nonterminal exact |0/120|0/120|0/120|
| Terminal exact |0/64|64/64|64/64|
| Predictions equal goal template |0/256|256/256|256/256|

Factual policy always chooses action 0. Its 95% whole-query bootstrap accuracy interval is[14.0625%,34.375%]; advantage over the best constant action is−4.6875percentage points, interval[−23.4375,0]. Cleared accuracy is18.75%, interval[9.375%,28.125%]. These come from 10,000 PCG64 whole-query resamples, seed1910, with the best constant action reselected in every draw. Degenerate exact-state intervals are empirical resampling outcomes, not proofs about the population.

The final reference score 50%=64/128 also equals its own best constant baseline. It comprises only 8 queries, repeated 16 times; labels[64,0,48,16]. Those queries were trained only at updates 1–2. This does not establish useful training fit or represent the whole stream.

The successor reconstructs the goal-completion template for every action: it removes the old agent and marks the goal completed. It therefore gets all terminal frames right and every blocked/nonterminal frame wrong. Changed-pixel accuracy15872/23552 hides a split: vacated11776/11776, destinations4096/11776 (terminal destinations only). Reward recall0/64; precision is undefined because no positive is predicted at threshold 0.5. Factual Brier score 0.188147. Value MSE 2.8526e−5 is limited by constant-one training/evaluation targets; constant 1 has zero MSE and this gives no long-horizon evidence.

The factual policy gate (>=90% and lower advantage CI>25pp) and successor gate (>=75% overall and in each outcome class) both fail. Cleared/reward/value metrics cannot rescue them. Best So Far is unchanged.

## Validation and execution

All 17 Rust runner tests and 15 independent Python control/integrity fixtures pass. The analyzer was frozen before final output. Source checks bind unchanged model/task and numerical helpers. A separate reader recomputes raw outputs, simulator targets, hashes/offsets/normalization/argmax, regions and bootstrap without importing the analyzer; no material discrepancy. Every audited training query matches the C3 parent query stream, and the actual 73,600-row training stream matches its C5 audit byte for byte.

Exact-binary CUDA smoke completed 2 updates at batch 33. Main training started 09:55:55 IST and finished around 10:27 IST, supervisor elapsed 1866.025s, all 1,150 updates. Peak whole-GPU sample 7520/8151MiB, minimum reserve 631MiB, maximum 73 degrees C. These are sampled invocation measurements, not peak allocator or production performance claims. Final frozen evaluation takes 5.458s, zero optimizer updates. An initial launch using --updates0 was rejected by the global positive-count guard before model execution; its failed-infrastructure logs are retained and excluded. The successful frozen calls use unused --updates 1 but record zero actual updates and unchanged weights.

All 8 separately Nsight-bound bundles pass structural/capture/integrity and raw application-label checks (2 smoke,1 initial,4 training,1 final). Training captures updates 2, 100, 1000 plus first evaluation. Host tracing, NVTX, Nsight CUDA/cuDNN/cuBLAS/OS-runtime and CPU sampling are retained. Automatic correlation remains incomplete; operation/activation linkage, allocation lifetimes/physical checkpoints and device-event intervals remain unwired. Host spans are not kernel durations. Current profiled-work bundles do not support production timing claims. All 44 declared trainable gradient families are present in the inspected update100 capture; this does not establish helpful gradients or separate-objective alignment.

## Decision

Do not scale or repeat this recipe unchanged. First rescore 72 registered seen queries across early/middle/late training at the same four-loop inference setting, with original-depth strata disclosed. Then the concrete next candidate is cohort-preserving replay:64 distinct queries per update instead of4 repeated queries, retaining the complete input-target-depth multiset and global1/2/4 schedule. A verified finite enumeration shows batch-label variance falls14.95–19.41fold; it does not establish actual gradient variance or a learning benefit. Grouping and repetition spacing change together.

A finite readout construction shows the existing heads can express these uniform-patch targets if the upstream model supplies suitable features. It does not prove that the core learns them. Opus5 xHigh actually reviewed conditional next steps; the primary qualified its causal overclaims. Narrow primary-paper version checks found no version change in the3 previously retrieved sources; papers/advisor agreement do not prove local competence.

No pretrained LLM, public-level training or ARC evaluation was used. New-core learned persistent memory, active probing and ARC integration remain absent. The broader autonomous work continues after this completed diagnostic.

## Exact commands and owning evidence

Commands below run from the campaign root through its retained supervisor, which enables profilers and enforces resource/integrity limits:

```bash
python3 supervise.py --binary ./looped_agent_probe --name known-seed0 --mode coverage --coverage fresh --batch 33 --loops 4 --updates 1150 --data-seed 9173 --seconds 2100
python3 supervise.py --binary ./looped_agent_probe --name final-frozen --mode known-mapping --batch 1 --loops 4 --updates 1 --data-seed 20260912 --eval-episodes 64 --checkpoint ./known-seed0/final.safetensors --seconds 300
```

Never reuse these completed roots. Exact executable argv, recipe, manifests, raw arrays and intervals are retained in the [completed analysis](/home/stepan/Research/_runs/2026-09-09T083139Z-tofy-looped-known-mapping/completed-analysis.json). The [research run](/home/stepan/Research/_runs/2026-09-09T083139Z-tofy-looped-known-mapping/manifest.json) owns independent reviews, finite arguments and final campaign seal.

## Final integrity seal

All 60 recorded C5/Opus/export processes are gone. The complete 467-file campaign
contains 639,801,542 bytes and was fully rehashed twice. Its externally retained
[manifest](/home/stepan/Research/_runs/2026-09-09T083139Z-tofy-looped-known-mapping/completed-campaign.manifest.json) has SHA256 `ddc774d4b277eab80395cb01735b851e727c4fe29b793aa23e4140210f411f7f`.
This is point-in-time integrity, not immutable storage. The next evaluator uses a separate root.
