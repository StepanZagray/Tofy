# Archived P2 experiment and campaign docs

Superseded experiment plans, campaign preregistrations, and run analyses, kept for
the experimental record. Nothing here is a current contract: launcher scripts cited
in many of these files no longer exist, and every pinned report schema predates the
current `p2.eval_report.v18` (`src/p2/eval.rs`). The current training contract is
[ADR 0003 (world-core-v5 / foundation-v2)](../adr/0003-world-core-v5-foundation-v2.md);
active material lives in [`docs/P2.md`](../P2.md),
[`docs/P2_ARC3_TRAINING.md`](../P2_ARC3_TRAINING.md), and
[`docs/RESULTS_P2.md`](../RESULTS_P2.md).

## Experiment chain (v9 -> v17)

| Doc | What it was | Superseded by |
|---|---|---|
| [`P2_STAGE1B.md`](P2_STAGE1B.md) | v9 ARC-AGI-3-aligned curriculum spec; `eval_report_64ep_v3` | the v10–v17 chain, then ADR 0003 |
| [`P2_V10_FIXES.md`](P2_V10_FIXES.md) | v10 fixes for the v9 open-loop rollout collapse | the v12/v13 architecture chain, then ADR 0003 |
| [`P2_V12.md`](P2_V12.md) | dual-block / dual-pool architecture experiment; self-labelled superseded | [`P2_V13.md`](P2_V13.md) (architecture reverted) |
| [`P2_V13.md`](P2_V13.md) | v11 architecture with a tighter Q threshold | the v15–v17 spatial-latent line, then ADR 0003 |
| [`P2_V15.md`](P2_V15.md) | spatial grid latents; `p2.eval_report.v5` | [`P2_V16.md`](P2_V16.md) |
| [`P2_V16.md`](P2_V16.md) | dynamics-stability fixes on v15 | [`P2_V17.md`](P2_V17.md) |
| [`P2_V17.md`](P2_V17.md) | two overnight runs: stability stack vs Q/PTRM fixes | the world-core-v2/v3 campaigns, then ADR 0003 |

## Campaigns and run analyses

| Doc | What it was | Superseded by |
|---|---|---|
| [`P2_TC_GLOBAL_MIX_OVERNIGHT_PLAN.md`](P2_TC_GLOBAL_MIX_OVERNIGHT_PLAN.md) | dual-scale TC-SIGReg overnight campaign plan (2026-08-10) | the world-core-v2/v3 campaigns, then ADR 0003 |
| [`P2_WORLD_CORE_V2_CAMPAIGN.md`](P2_WORLD_CORE_V2_CAMPAIGN.md) | world-core-v2 causal campaign preregistration | treatments rejected by the frozen gates ([ADR 0002](../adr/0002-resolved-experiments-and-factual-batches.md)); ADR 0003 |
| [`P2_WORLD_CORE_V3_CAMPAIGN.md`](P2_WORLD_CORE_V3_CAMPAIGN.md) | world-core-v3 follow-up campaign | treatments rejected by the frozen gates (ADR 0002); ADR 0003 |
| [`P2_OVERNIGHT_GEOMETRY_V2_ANALYSIS.md`](P2_OVERNIGHT_GEOMETRY_V2_ANALYSIS.md) | SIGReg geometry A/B overnight run; incomplete, no decision | [`P2_GEOMETRY_V2_COMPLETED_PILOT_ANALYSIS.md`](P2_GEOMETRY_V2_COMPLETED_PILOT_ANALYSIS.md) |
| [`P2_GEOMETRY_V2_COMPLETED_PILOT_ANALYSIS.md`](P2_GEOMETRY_V2_COMPLETED_PILOT_ANALYSIS.md) | completed seed-1 geometry pilot; terminal branch A, no arm promoted | [`P2_PHASE0_REPAIR_REEVAL_2026-08-10.md`](P2_PHASE0_REPAIR_REEVAL_2026-08-10.md); ADR 0003 |
| [`P2_PHASE0_REPAIR_REEVAL_2026-08-10.md`](P2_PHASE0_REPAIR_REEVAL_2026-08-10.md) | Phase-0 evaluator repair and re-eval on `p2.eval_report.v10` | report schema v18; ADR 0003 |
| [`FULL_V4_SPEED_OPTIMIZER_RESTART_2026-08-17.md`](FULL_V4_SPEED_OPTIMIZER_RESTART_2026-08-17.md) | speed/optimizer restart preregistration against the full-v4 recipe | ADR 0003, which replaces the ADR 0001 full-v4 contract |

## Preregistrations resolved in `RESULTS_P2.md`

| Doc | What it was | Superseded by |
|---|---|---|
| [`EXPERIMENT_CONSUMER_READOUT_V1.md`](EXPERIMENT_CONSUMER_READOUT_V1.md) | spatial-query vs global-mean Q readout screen | Pressure x Grounding V1, registered to repair its failures; ADR 0003 |
| [`EXPERIMENT_PRESSURE_GROUNDING_V1.md`](EXPERIMENT_PRESSURE_GROUNDING_V1.md) | SIGReg dose x grounding factorial | negative screen, no cell promoted (`RESULTS_P2.md`, 2026-08-14); ADR 0003 |
| [`EXPERIMENT_GROUNDING_MECHANISM_V1.md`](EXPERIMENT_GROUNDING_MECHANISM_V1.md) | 2x2 grounding-mechanism follow-up | never ran — queue stopped on the parent status, zero arms trained; ADR 0003 |

## Research notes

| Doc | What it was | Superseded by |
|---|---|---|
| [`P2_VRAM_ARCHITECTURE_2026-08-05.md`](P2_VRAM_ARCHITECTURE_2026-08-05.md) | VRAM investigation of the v17-era spatial-SIGReg / randomized-depth config | ADR 0003 |
| [`TOFY_ARC3_READINESS_2026-08-05.md`](TOFY_ARC3_READINESS_2026-08-05.md) | ARC-AGI-3 readiness assessment ("do not start Stage 3 yet") on the v17 checkpoint | [`ARC_AGI_3_MODEL_BLOCKERS_AND_PLAN_2026-08-09.md`](ARC_AGI_3_MODEL_BLOCKERS_AND_PLAN_2026-08-09.md); [`docs/research/2026-08-24-foundation-improvements.md`](../research/2026-08-24-foundation-improvements.md); ADR 0003 |
| [`ARC_AGI_3_MODEL_BLOCKERS_AND_PLAN_2026-08-09.md`](ARC_AGI_3_MODEL_BLOCKERS_AND_PLAN_2026-08-09.md) | blockers and evidence-driven plan after the geometry-v2 pilot | [`docs/research/2026-08-24-foundation-improvements.md`](../research/2026-08-24-foundation-improvements.md); ADR 0003 |

## Phase results

| Doc | What it was | Superseded by |
|---|---|---|
| [`RESULTS_P0.md`](RESULTS_P0.md) | pre-P1 knowledge-transfer / VecLab / Qwen bridge results (already self-labelled archived) | [`docs/RESULTS_P1.md`](../RESULTS_P1.md) and the active [`docs/RESULTS_P2.md`](../RESULTS_P2.md) |
