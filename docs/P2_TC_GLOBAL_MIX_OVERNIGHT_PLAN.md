# P2 dual-scale TC-SIGReg overnight campaign

Date: 2026-08-10/11

Launcher: `scripts/p2_tc_global_mix_overnight.sh`

Architecture context: `docs/specs/P2_ARC_AGI_3_WORLD_MODEL_CORE_REDESIGN.md`

Preregistration authority for this campaign: this document at the reviewed Git SHA

## Research question

The rejected TC-SIGReg arm increased latent variance but reduced spatial and pooled
effective rank. This campaign tests whether that failure comes from (a) mismatch
between the cell-row training population and global pooled consumers, (b) the degree
of pressure allocated to the global population, or (c) loss of diversity caused by
2x2 spatial pooling. It does not test rollout loss, direct prefixes, or broader agent
substrate work.

## Frozen shared contract

- Fresh seed 1 for every arm; no resume or warm-start between arms.
- One `sequential` lesson, 1,000 optimizer updates.
- Effective batch 1,024. The launcher uses the largest physical A40 batch that passes
  the worst-case unpooled mixed-objective probe, minimizing accumulation.
- Post-RMS representations, temporal window `W=8`, SIGReg weight `0.003`, row cap
  32,768, random recursion depth, final-outer supervision, residual update, warm start,
  outer depth 8 and inner depth 2.
- Rollout, event, Q, prefix, and reliability loss weights are zero.
- Checkpoints and full held-out evaluations at updates 250, 500, 750, and 1,000.
- Every arm records GPU telemetry, a representative profile packet, model/optimizer
  checkpoints, factual rollout rows, action strata, and encoder gradient-pressure
  attribution immediately before the profiled update.

Only output directory, SIGReg target, global mixture, and explicitly named post-RMS
pooling geometry vary. Promotion remains locked after the campaign.

## Arms and hypotheses

| order | arm | global mix | pooling | hypothesis |
|---:|---|---:|---|---|
| 1 | `marginal-control` | 0 | 2x2 | version-matched control and reproducibility check |
| 2 | `tc-cell` | 0 | 2x2 | exact replay of the rejected cell-row TC construction |
| 3 | `tc-mix-025` | 0.25 | 2x2 | a light global constraint may restore pooled rank without erasing spatial diversity |
| 4 | `tc-mix-050` | 0.50 | 2x2 | balanced cell/global pressure may improve both populations |
| 5 | `tc-global` | 1.00 | global only | directly constraining the consumer population may repair pooled rank, with spatial rank retained as an anti-gaming gate |
| 6 | `tc-unpooled` | 0 | none | 2x2 pooling may be the spatial-rank bottleneck even before global pooling |

The mixture is a convex combination inside the existing outer SIGReg coefficient, so
mix arms do not silently increase the nominal auxiliary weight. The measured
SIGReg-to-next-latent encoder gradient-norm ratio is reported for each arm; raw loss
magnitude is not treated as gradient pressure.

## Frozen decision gates at update 1,000

An arm is a Phase-1B candidate only if all conditions hold:

- pooled variance `>= 1e-4`;
- pooled effective-rank fraction `>= 0.10`;
- spatial effective-rank fraction `>= 0.10` and no downstream seam contraction;
- aggregate and random-one-step shuffled/true action ratios `>= 1.10`, with 95% CI
  lower bounds above `1`;
- changed-transition improvement over copy-forward `>= 10%`;
- finite artifacts and factual episode reconciliation;
- dynamics and planner normalized H8 no worse than `1.25x` the paired control.

Per-action-ID, coordinate/simple, changed/no-op, hazard, exploration, and planner-source
strata are explanatory diagnostics, not replacements for the frozen aggregate gates.
Sparse action strata are never used for promotion, even when their bootstrap interval
looks decisive. Aggregate and stratum counts and weighted errors must reconcile before
the result is interpreted.

The first five arms form the target/mixture dose-response. `tc-unpooled` is a
preregistered exploratory geometry intervention, deliberately ordered last so it
cannot confound selection among the pooled mixture arms.

## Runtime and supervision

The previous two-arm campaign took 4 h 33 min. Six serialized arms are therefore
expected to take approximately 13 h 35 min, plus build and probe time. A launch around
21:15–21:45 BST should finish around 10:50–11:50 BST on 2026-08-11, comfortably after
the requested 06:30 boundary.

Each arm runs in an isolated fail-recording subshell. A failed arm is recorded and the
next preregistered arm starts, preventing one recoverable experiment failure from
leaving the GPU idle. A passed worst-case A40 probe is mandatory before the campaign.
No arm can launch seeds 2/3 automatically.

The campaign collects every frozen checkpoint before evaluation; it does not make
adaptive stop/continue decisions from partial scientific results. Technical failures
remain isolated per arm, and promotion stays locked for local analysis after the run.

## After the run

Compare trajectories and exact paired-control ratios before selecting a mechanism.
Only a fully passing arm may be replicated at seeds 2/3. If global mixing repairs
pooled rank but not spatial rank, the unpooled result decides whether to continue with
spatial geometry work. If no arm clears both rank floors, stop TC pooling sweeps and
redesign encoder/objective geometry rather than training longer.
