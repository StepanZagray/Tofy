# Results P2

P2 is implemented as a recursive latent world-model experiment. No trained metric is
recorded yet; implementation smoke tests must not be promoted to a research result.

## Frozen fields for the first experimental run

Before inspecting a full run, record here:

- training and held-out seeds;
- curriculum lesson lengths;
- model dimensions and recursion schedule;
- physical batch size and gradient accumulation;
- optimizer, learning rate, and SIGReg/event/Q weights;
- PTRM `K` values and latent-noise sweep;
- checkpoint-selection metric;
- exact train/evaluation commands.

## Best So Far

No P2 result recorded.

## Implementation validation (not a result)

The CPU smoke path completed the four ordered lessons, wrote a safetensors
checkpoint/config/report plus a `candle-graph/runtime/1` trace, evaluated held-out
synthetic transitions with deterministic `K=1` and stochastic `K=2`, and imported
the trace into `candleModelAnalyzer`. The smoke used physical batch `2` and gradient
accumulation `1`; it is not the required accelerator batch-capacity measurement for
the first real run.

```bash
cargo run --release -- p2-train \
  --lessons dynamics,sequential,falsification,retarget \
  --steps-per-lesson 1 --physical-batch 2 \
  --hidden-dim 16 --action-dim 4 \
  --inner-steps 1 --outer-steps 1 \
  --sigreg-projections 4 --sigreg-knots 3 \
  --output-dir /tmp/tofy-p2-final-smoke-v3

cargo run --release -- p2-eval \
  --checkpoint /tmp/tofy-p2-final-smoke-v3/model.safetensors \
  --train-config /tmp/tofy-p2-final-smoke-v3/config.json \
  --synthetic-episodes 1 --physical-batch 2 \
  --ptrm-k 1,2 --ptrm-noise 0.1 \
  --output /tmp/tofy-p2-final-smoke-v3/eval.json

scripts/audit_p2.sh \
  /tmp/tofy-p2-final-smoke-v3/analyzer \
  /tmp/tofy-p2-final-smoke-v3/model.safetensors \
  /tmp/tofy-p2-final-smoke-v3/runtime.json
```

All generated reports set `research_claim=false`; the local evaluator leaves
`official_rhae=null` and records `public_data_used_for_fitting=false`.

## Update rule

When a P2 metric is reported, include the exact command and separate synthetic
oracle-normalized efficiency from official ARC-AGI-3 RHAE. Public ARC games are
held-out transfer evaluation and must not be used for checkpoint selection or
hyperparameter tuning.
