# The known-control failure also appears on seen queries

September 9, 2026 IST. The frozen C5 final checkpoint scores **25/72 factual actions**, exactly the **25/72 best constant baseline**, on registered early/middle/late training queries. All 576 successor predictions are goal-completion templates. Both seen-panel prerequisites fail. This establishes the failure on selected seen inputs at four loops; it does not reveal whether the model never learned them or later forgot them.

[Registration](2026-09-09-looped-seen-query.md). Select IDs 0–23, 2280–2303 and 4576–4599 before readout: 24 queries per temporal block, eight per original training depth 1/2/4 within each block. All 72 unique factual inputs, queries, targets and action labels match every one of their 16 repeats in the sealed C5 training stream. Evaluate initial/final checkpoints, factual/cleared support, all four actions, four inference loops, physical/effective batch one, zero optimizer updates.

| Endpoint | Initial factual | Final factual | Final cleared |
|---|---:|---:|---:|
| Policy correct |10/72|25/72|16/72|
| Policy CE |1.623162|1.377118|1.377798|
| Best constant action |25/72|25/72|25/72|
| Exact successor frames |0/288|72/288|72/288|
| Copy-current exact baseline |80/288|80/288|80/288|
| Blocked exact |0/80|0/80|0/80|
| Nonterminal exact |0/136|0/136|0/136|
| Terminal exact |0/72|72/72|72/72|
| Predictions equal to goal template |0/288|288/288|288/288|

Final factual always chooses action 0; labels are [25,10,20,17]. Its accuracy interval is [23.6111%,45.8333%], advantage over resampled best constant [−12.5,0] percentage points. Bootstrap: 10,000 whole-query PCG64 draws, seed 1911, linear percentile 95% intervals, best constant reselected within each draw. All four actions and both support conditions stay clustered; zero-denominator draws are explicit. Constant-one targets still limit value MSE (approximately 2.86e−5 versus baseline zero); reward and pixel metrics cannot rescue the primary failures.

Early/middle/late factual policy scores 9/24, 7/24, 9/24, versus their constant baselines 9/24, 8/24, 9/24. Every temporal block gets only its 24 terminal successors right out of 96 actions. These are descriptive strata, not selected subgroups or evidence of a causal recency effect.

Forty-eight queries were trained at one or two loops, so four-loop scoring does not measure their exact original-depth fitting. The 24 queries trained at four loops also fail: policy 8/24 equals their constant baseline, successor 24/96 are all terminal, copy-current 30/96. A depth mismatch alone cannot explain the complete observed pattern. No claim is made about all 4,600 training queries or intermediate learning trajectories.

## Provenance and verification

Evaluator source `ae4597b65f09adc16d7ddcff34a408b4b193ce3d`; binary `037e06bf019245fd9bf3c19ca403d536bd03f85df8a55863b68158b1c1fe5024`; Candle Graph `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a`. The unchanged parent numerical model trained at `71ff87ad76400e977e8e3031afa437727af73ed5`. Initial checkpoint `4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802`; final1150 checkpoint `e379802fa8e75327061f3dcc5ae95697f83e2d2d137a69ef68daf3f902040d41`.

All 20 CPU runner tests and 16 analyzer fixtures pass. The analyzer was frozen before any seen output. The old64-query initial evaluator reproduces all three raw files byte for byte and every native report field except elapsed durations. The independent analyzer reconstructs visible-grid targets, labels, regions, hashes, offsets, normalization and argmax; a separate raw reader confirms counts and template classification without importing it.

Campaign `/home/stepan/Projects/code/.tofy-runs/looped-known-seen-20260909T104446-IST`. Initial/final seen invocations take 6.666/6.539 seconds; each peaks at sampled whole-GPU382/8151MiB. All three separately Nsight-bound captures (legacy parity, initial seen, final seen) are structurally valid and complete for declared capabilities. Host trace, NVTX, Nsight CUDA/cuDNN/cuBLAS/OS-runtime and CPU evidence are present. Automatic correlation, operation/activation linkage, allocation lifetimes/physical checkpoints and device-event intervals remain incomplete/unwired. No production timing claim follows.

## Decision and execution

Proceed to the separately [registered cohort replay screen](/home/stepan/Projects/code/Tofy-known-replay/docs/research/2026-09-09-looped-cohort-replay.md): the same learner, examples, depths and 1,150 updates, but 64 distinct queries per update and each query revisited in 16 updates. Batch grouping and repetition spacing change jointly; no benefit is assumed. This is an optimization premise test before broader architecture changes, not an ARC improvement.

From the campaign root, execute the retained supervisor (already completed; never reuse the root):

```bash
python3 supervise.py --binary ./looped_agent_probe --name final-seen --known-seen --mode known-mapping --batch 1 --loops 4 --updates 1 --data-seed 9173 --eval-episodes 72 --checkpoint /home/stepan/Projects/code/.tofy-runs/looped-known-mapping-20260909T095023-IST/known-seed0/final.safetensors --seconds 300
```

The positive CLI updates argument is unused in this frozen mode; actual updates are zero and checkpoint hashes are unchanged. Full exact initial/audit/qualification commands and numerical evidence are in [completed analysis](/home/stepan/Research/_runs/2026-09-09T093609Z-tofy-looped-seen-replay/completed-analysis.json). No LLM, public training or ARC evaluation occurred. The new core still lacks learned persistent episode memory, active probing and ARC integration. Autonomous model work continues into the separate replay experiment.

The completed campaign contains 190 files, 518,524,316 bytes; all 27 recorded process IDs are gone. Its externally retained [manifest](/home/stepan/Research/_runs/2026-09-09T093609Z-tofy-looped-seen-replay/completed-campaign.manifest.json) was fully reverified, SHA256 `85311311e7b9b79aa51c364d52171e858dc7875891f6c8e758584200c0f87ffd`. This is point-in-time integrity, not immutable storage.
