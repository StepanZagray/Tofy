# P2 Phase-0 repair and geometry-v2 re-evaluation

Date: 2026-08-10

Pod: `k8mkqgocvx1gqa`

Evaluation root: `runs/p2/phase0-reeval-v10-11686b8`

Status: **all repairs validated; four preserved checkpoints re-evaluated; no
model-quality promotion**

## Outcome

The Phase-0 evaluator and ARC action-transport repairs are complete. The four
preserved geometry-v2 checkpoints were evaluated in separate `representation` and
`rollout` modes with the repaired `p2.eval_report.v10` contract. All 1,776 real
episode rows reconcile with their aggregate reports, every representation seam has
zero non-finite input rows, and the repaired evaluator preserves the prior v9
scientific metrics. A second control/update-1,000 representation run was byte-for-byte
identical.

The new seam evidence sharpens, but does not change, the completed pilot decision:

- The control retains meaningful cell-level diversity, but global spatial pooling is
  the dominant rank bottleneck. At update 2,000, dynamics post-RMS spatial rank
  fraction is `0.08175`, while post-RMS pooled rank fraction is only `0.02392`.
  The spatial representation is still below the `0.10` floor before pooling, so
  removing pooling alone would not be enough.
- The pre-RMS treatment is already low-rank at the spatial-cell seam. At update
  2,000, dynamics post-RMS spatial rank fraction is `0.02639`, then pooling reduces
  it to `0.00844`. At update 1,000, its pre-RMS pooled variance is only
  `3.68e-11`, too small to define a stable effective rank.
- The recursion does not introduce a new abrupt collapse. In each checkpoint,
  action-conditioned input, first recursion output, final prediction, and target
  spatial ranks stay close to the encoder spatial rank. The failure originates in
  the encoder geometry and is amplified by pooling.
- Control action sensitivity at update 2,000 is real, but it coexists with severe
  rollout deterioration. The treatment remains action-marginalized and worse than
  copy-forward on changed transitions. These are separate failure axes.

This supports the paired temporally centered SIGReg pilot: keep the world model and
the control's downstream geometry fixed, and change only the population regularized
by SIGReg. It does not authorize the factual-graph/executable-world-model work from
recommendation 4.

## Repair provenance

| Repair | Integrated commit | Result |
|---|---|---|
| Real per-episode rollout evidence and aggregate reconciliation | `8f2a02b6` | `p2.episode_rollout.v2` rows are emitted from actual episode measurements |
| Bounded representation-seam diagnostics | `c31c914c` | v10 reports expose encoder, action, recursion, prediction, and target seams |
| At-most-once ARC action transport | `11686b82` | ambiguous transport/5xx attempts are not automatically replayed |

The frozen evaluator binary was built at Tofy commit
`11686b82e0a5e0439ecddc7db3652c58e41d3fdc`, with SHA-256
`86e751f696987e5d75517cd7b08d80e5219ca55965b6f84b3db448ec348c5bd2`.
The synchronized `candle_graph` commit was
`c9fa15ee917f9dab96bb070cbbcd5ce8dfdb5f48`.

## Frozen evaluation contract

Each checkpoint used CUDA, evaluation seed `424242`, 64 synthetic episodes,
physical batch `1024`, deterministic PTRM `K=1`, zero PTRM noise, and one ensemble
member. Representation and rollout were run separately so the repaired evidence
contract could be validated without changing model behavior.

Command template, instantiated for both arms and updates 1,000 and 2,000:

```bash
target/release/tofy p2-eval \
  --checkpoint runs/p2/ab-sigreg-geometry-v2/seed-1/<arm>/checkpoints/step-<update>/model.safetensors \
  --train-config runs/p2/ab-sigreg-geometry-v2/seed-1/<arm>/config.json \
  --device cuda --seed 424242 --synthetic-episodes 64 --physical-batch 1024 \
  --ptrm-k 1 --ptrm-noise 0 --ensemble-members 1 \
  --eval-mode representation \
  --output runs/p2/phase0-reeval-v10-11686b8/<arm>/update-<update>/representation-a.json

target/release/tofy p2-eval \
  --checkpoint runs/p2/ab-sigreg-geometry-v2/seed-1/<arm>/checkpoints/step-<update>/model.safetensors \
  --train-config runs/p2/ab-sigreg-geometry-v2/seed-1/<arm>/config.json \
  --device cuda --seed 424242 --synthetic-episodes 64 --physical-batch 1024 \
  --ptrm-k 1 --ptrm-noise 0 --ensemble-members 1 \
  --eval-mode rollout \
  --episode-jsonl runs/p2/phase0-reeval-v10-11686b8/<arm>/update-<update>/episodes.jsonl \
  --output runs/p2/phase0-reeval-v10-11686b8/<arm>/update-<update>/rollout.json
```

## Re-evaluation metrics

| Arm | Update | Dyn. variance | Dyn. pooled rank | Dyn. spatial rank | Aggregate action ratio [lower] | Random action ratio [lower] | Changed improvement | Normalized dynamics H8 | Normalized planner H8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 1,000 | 0.002772 | 0.01383 | 0.07329 | 1.0402 [1.0225] | 1.1547 [1.0792] | 0.6665 | 1.5579 | 1.0287 |
| control | 2,000 | 0.006128 | 0.02529 | 0.08175 | 1.2095 [1.1567] | 1.7365 [1.5164] | 0.4661 | 9.6106 | 6.1610 |
| pre-RMS spatial | 1,000 | 2.18e-6 | 0.01128 | 0.02925 | 1.0000 [0.9997] | 0.9999 [0.9994] | -49.2690 | 6.1129 | 144.2573 |
| pre-RMS spatial | 2,000 | 1.95e-5 | 0.00839 | 0.02639 | 1.0004 [0.9992] | 1.0016 [0.9973] | -6.4256 | 5.9654 | 113.0891 |

Planner pooled/spatial rank fractions were respectively `0.02502/0.06990`,
`0.01653/0.07722`, `0.01431/0.02944`, and `0.00805/0.02633` in table order.
The raw dynamics H8 values were `0.10939`, `1.80823`, `0.004649`, and `0.031236`;
the tiny treatment errors remain degenerate because the learned latent target itself
collapsed.

## Evidence validation

- Four representation reports and four rollout reports use
  `p2.eval_report.v10` with the requested explicit mode.
- Each checkpoint emitted 444 `p2.episode_rollout.v2` rows: 1,776 total.
- Source/horizon row counts and raw open, closed, and copy-forward means reconcile
  within `1e-12`. Normalized means reconcile within `1e-6`, covering JSON
  serialization precision.
- Every named seam has zero non-finite rows. Effective rank is null only for the two
  update-1,000 treatment pre-RMS pooled populations whose variance is below `1e-10`;
  that is expected collapse evidence rather than missing evaluator data.
- One-step, action-intervention, changed-transition, open-rollout, copy-forward, and
  closed-loop metrics match the prior v9 reports. The repaired closed-loop report
  intentionally omits the old nonsensical normalized field.
- The repeated control/update-1,000 representation report is byte-identical, SHA-256
  `b52168ed03740957bc1841e942d62ac75140ef79a9fc3fb0be21a2c6f37427cb`.

No row or repaired metric changes the original terminal `stop_after_pilot` decision,
and no value is promoted to P2 "Best So Far".

## Authorized next experiment

Run only the paired seed-1 Phase-1B TC-SIGReg pilot:

- control: unchanged marginal SIGReg;
- treatment: subtract the within-window temporal mean from ordered, contiguous
  `W=8` latent windows before applying the same post-RMS, 2x2-pooled SIGReg geometry;
- fresh initialization, seed 1, checkpoints/evaluation at 250, 500, 750, and 1,000;
- serialized training/evaluation on the single A40;
- effective batch 1,024, using the largest stable physical batch established by a
  maximum-depth CUDA probe;
- seeds 2/3 remain unavailable until the update-1,000 gate is analyzed.

Recommendation 4 is explicitly out of scope for this run.
