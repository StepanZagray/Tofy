# Consumer Readout V1 preregistration

## Question

Does a learned, position-aware spatial query at the **active Q-head readout seam** improve Q-calibration proxies over the exact current global mean, while recurrence, the next-latent objective, temporal QQ, data, and optimization remain fixed?

This is not a test of whether recurrence should become spatial: revision `720bebd8` already keeps `B×C×8×8` through action conditioning, recurrence, rollout, and next-latent MSE. The earlier consumer-seam campaign had no active pooled learned consumer and therefore cannot answer this question.

## Seed-1 arms

GPU-heavy work is sequential. The preregistered seed-1 order is:

1. `consumer_readout_spatial_query`
2. `consumer_readout_global_mean`

Both arms run 1,000 updates with checkpoints and evaluation at 250, 500, 750, and 1,000. There is no efficacy early stop; only OOM, non-finite values, configuration/fingerprint mismatch, or another integrity failure may stop an arm.

The only treatment is `consumer_readout`:

- `global-mean`: exact legacy mean over the 8×8 spatial axes.
- `spatial-query`: learned position embeddings and content scores over 64 tokens, returning the same `B×C` head input.

The sole active readout-dependent training bundle is Q calibration with live gradients into the predictive state: linearly warmed Q BCE (configured maximum weight `0.1`) plus the existing fixed `0.1 × mean(sigmoid(Q) × latent_mse)` anti-hallucination term. PTRM ranking, event, reliability, prefix, rollout, and branch objectives are disabled.

## Frozen common configuration

- seed 1; `q_calibration` synthetic lesson; shuffled episodes;
- cell temporal-QQ, quantile statistic, weight `0.003`, temporal window 8, post-RMS 2×2 spatial pooling, maximum 32,768 rows;
- Q weight `0.1`, fixed latent-MSE threshold `0.05`, `stop_grad_q_y=false`;
- warm-start and residual-y recurrence; inner depth 2, outer depth 8; last-outer supervision;
- hidden width 128, action width 8; event/reliability/prefix/rollout weights zero; ensemble one;
- effective batch 1,024. Probe the spatial-query arm at `1024×1`, then `512×2`, then `256×4`; select the first stable pair and force it on both arms.

Physical batch and accumulation are part of the experiment identity because nonlinear population objectives are evaluated per microbatch.

## Integrity checks

- exact reviewed Tofy commit, candle_graph commit, release-binary SHA-256, one idle NVIDIA A40, and clean checkouts;
- identical training-population row count and provenance fingerprint across arms;
- normalized configs differ only in output path and readout topology;
- resolved experiment identity records the topology and exact resume rejects changes;
- shared named parameters initialize byte-identically (unit tested);
- global adapter exactly equals legacy pooling; spatial adapter is finite, position-sensitive, and receives gradients (unit tested);
- evaluation and board-probe population fingerprints match across arms;
- record QQ/next-latent and the complete Q-BCE-plus-surprise readout/next-latent encoder-gradient ratios separately.

## Absolute gates

Promotion remains locked until completed analysis. A candidate must satisfy all of:

- final consumer-readout and final-spatial effective-rank fraction at least 10%; retain global-mean rank as a diagnostic in both arms;
- board-probe target decoder trusted, predicted histogram improves on literal copy, and changed-patch F1 at least 0.5;
- changed-transition copy-improvement 95% lower bound greater than zero;
- shuffled-action sensitivity ratio 95% lower bound greater than one;
- Q positive-label rate in `[0.1, 0.9]`, balanced accuracy greater than 0.5 with both classes present, and Brier score below 0.25;
- H8 all finite; raw aggregate open/copy MSE at most 1; normalized median at most 1; normalized p95 at most 10; normalized CVaR95 at most 100; at least half of eligible episodes beat copy; normalized mean at most 1 as the catastrophic-tail alarm.

Main-recurrence H8 is a safety prerequisite, not direct evidence that the readout helps planning.

## Decision sequence

If spatial query passes every absolute gate and improves preregistered readout/Q metrics over global mean, repeat the paired comparison at seeds 2 and 3 with reversed/counterbalanced arm order. Only after confirmation should Tofy test a different regularizer (pressure-calibrated at common initialization), exact same-state alternate outcomes, or dense anchored action-prefix prediction.

No proxy result warrants a prediction of 100% ARC-AGI-3. Actual planning/exploitation and official hidden-evaluation evidence remain necessary.
