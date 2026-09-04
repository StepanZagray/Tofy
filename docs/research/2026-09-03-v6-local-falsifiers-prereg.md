# Preregistration: ADR 0005 local falsifiers (2026-09-03)

Scope: the three cheap experiments ADR 0005 §5.1–5.3 that must run on the
local machine (RTX 5060 Laptop, 8 GB) before any pod GPU-hours are spent on a
v6 run. Everything here is `selection_only`/exploratory evidence: it can block
or unblock the v6 pod run, it cannot satisfy a promotion gate.

## E1 — Residual-vs-reliability AUROC (§5.3)

- Claim under test: on the latest local v5 checkpoint, the one-step composed-
  decode residual separates real ARC-AGI-3 frames from in-distribution
  synthetic frames better than the trained reliability head does.
- Checkpoint: `runs/p2/_pod_handoffs/6zp5oip7tvokfl-20260827-foundation-v2/foundation-v2/model.safetensors`
  (sha256 117f786b…, 2026-08-27). Caveat: NOT the s8 step-20480 best model;
  that checkpoint lives on the stopped pod. Conclusions transfer only as far
  as both models share the v5 recipe and the same live-query tuple.
- Real population: 25 public games, seeded uniform-random policy, 150 actions
  each, toolkit local recordings; evaluation only. Synthetic population: same
  size, fixed seed, the trainer's held-out generator incl. 30% goal dropout.
- Query tuple: all-zero goal + UNKNOWN operator (what the live policy sends).
- Metrics: AUROC(real vs synthetic) for residual and reliability; AUROC for
  "composed decode wrong" within each population; bootstrap 95% CI (1,000).
- Controls: synthetic reliability high (positive); shuffled labels ≈ 0.5
  (negative).
- Decision: residual AUROC ≥ 0.80 AND reliability AUROC ≤ 0.60 ⇒ Phase A trust
  switches to residual-derived calibration until a v6 checkpoint exists.
  Otherwise no change to Phase A trust.

## E2 — Memorization diagnostic (§5.1)

- Claim: with the v6 generator (twins, UNKNOWN operator, whole-frame content)
  the model uses the Context Window: changed-exact on the row after the
  context is higher with K = 16 than with K = 0.
- Arms: one v6 model; evaluation with the context masked (K = 0) versus the
  full context (K = 16) on 512 held-out twin-pair meta-episodes, fixed
  evaluation seed 1000002.
- Budget: local run, 4096 optimizer steps, seed 2, evaluation at 1024/2048/3072/4096.
  Batch pair measured before launch (AGENTS.md): physical 256 OOMs on the 8 GB GPU under
  the v6 context channel; physical 128 × accumulation 2 (effective 256) peaks at 5.8 GB and
  is the launched configuration. This is a smaller effective batch than the ADR 0003 recipe
  (1024); the run is a data-contract screen, not a recipe comparison.
- Launch (2026-09-03, revision 975832f9, binary sha256 in the run root):
  `tofy p2-train --recipe foundation-v2 --world-core-v6 --data-contract-v6 --device cuda
  --seed 2 --init-seed 2 --physical-batch 128 --grad-accum 2 --steps 4096
  --checkpoint-every-steps 512 --output-dir runs/p2/v6-memorization-e2-20260903`.
- Threshold: Δ ≥ 0.05 absolute at step 4096. Below threshold ⇒ DATA failure:
  the generator is not mutually exclusive; the pod run is blocked.
- Also recorded (no threshold): the same delta on the legacy streams
  (expected ≈ 0, since legacy rows carry no context), and the model-free
  `single_frame_rule_identifiable` census (must be 0).

## E3 — Adaptation falsifier (§5.2)

- Claim: Channel B (fast-weight updates per §6.2) improves prequential
  changed-exact over Channel A (context only) on held-out synthetic
  meta-episodes, without making the adapted-then-frozen model worse than the
  prior.
- Population: 256 held-out meta-episodes (levels 2–4), adapt on transitions
  1..t, score t+1..t+4, t ∈ {8, 16, 32}. Seeds 1000002 (population), 7 (adapter).
- Arms: A (context only), A+B default (reset at level), A+B carry.
- Thresholds: promote B if prequential changed-exact improves by ≥ 0.02
  absolute at every t AND frozen-after-adaptation changed-exact ≥ prior on
  the same rows. Else Channel B ships disabled (`--adapt` off by default).
- Multiplicity: three arms, three t values; thresholds are pre-set, no
  post-hoc metric changes. A single seed is a screen only.

## Stop rules and runtime

- E1 ≤ 2 h wall clock; E2 ≤ 6 h (measured: 12 steps at batch 256 ≈ 33 s
  including start-up on the local GPU; per-step timing recorded in the run);
  E3 ≤ 2 h. Any integrity failure (hash, seed, evaluator) fails closed.
- Data-access boundary: public ARC-AGI-3 frames appear only in E1 and only
  as evaluation inputs; no public frame enters any training path.

## E1 outcome (2026-09-03, selection-only)

Run: `runs/p2/residual-probe-20260903/REPORT.md`, commits 95013dc7 / adcd918c.
Checkpoint actually probed: foundation-v2 2026-08-27 EMA-best (sha256 117f786b…),
plus the final export (e5457dd4…); conclusions identical. NOT the s8 model.

| task | residual (pixel-CE) AUROC | 1 − reliability AUROC |
|---|---|---|
| real vs synthetic, all rows (n=7,500) | 0.905 [0.896, 0.914] | 0.634 [0.620, 0.647] |
| real vs synthetic, changed rows only | 0.866 | 0.494 (chance) |
| wrong-prediction within synthetic | 0.488 (chance) | 0.719 |
| shuffled-label control | 0.502 | 0.498 |

Against the preregistered rule (residual ≥ 0.80 AND reliability ≤ 0.60): the
residual condition is met; the reliability condition is **narrowly missed**
(0.634, CI excludes 0.60). The switch of Phase A trust to residual-derived
calibration is therefore NOT triggered by the rule; it is recorded as a
recommended exploratory follow-up with a fresh recording seed, because the
head is at chance on exactly the rows that matter (changed transitions).

Additional facts that change earlier statements: the "~1e-8 reliability on
real frames" observation from the s8 run is NOT reproduced on foundation-v2
(real-frame median reliability 0.511); it is s8-specific and remains
untested. Residual is a distribution-shift detector, not a per-transition
correctness signal (chance within synthetic). Only 1.2% of real changed
transitions were predicted exactly (28/2,410); 0/3,750 full-frame exact.

## Amendment (2026-09-03): recursion depth 3x3

E2 and E3 now run the v6 recipe at recursion depth 3x3 (ADR 0005 §3.5), not
2x2. The 4096-step budget and thresholds are unchanged, but the per-step cost
rises ~2.25x on the dynamics block, so the measured wall clock supersedes the
earlier 2.1 h estimate; the first attempt at 2x2 (physical 128 x accum 2)
measured 1.60 s/step. Any 2x2-vs-3x3 comparison is a separate preregistered
ablation, not part of E2: E2 is a data-contract screen and both arms of its
own comparison (K=0 vs K=16) share one checkpoint and therefore one depth.

## Amendment (2026-09-04): which statistic the §5.1 threshold applies to

Code review (2026-09-04) found that `p2-eval --context-ablation` scores the
generic held-out mixed population (legacy streams plus LearningHistories rows
with K ~ Uniform{0..16}) with and without context, not the preregistered
"512 twin-pair meta-episodes, K = 0 vs K = 16" population. Legacy rows and
K = 0 rows contribute exactly zero delta, so `overall.delta` is diluted by
roughly 2x or more. Before any result is read, the rule is fixed as follows:

- The `>= 0.05` threshold applies to the **`5-16` context-length stratum**
  delta (rows whose generated window has K >= 5, scored with their window vs
  with K = 0), which is the closest implemented proxy for the preregistered
  statistic. `overall.delta` is reported but is not the gate.
- This is a weaker test than K = 16 exactly (windows of 5..16, mean ~10);
  an exact K = 0 vs K = 16 twin-pair evaluation remains to be implemented and
  will supersede this rule when it exists.
- E2's checkpoint was trained at effective batch 256 and depth 3x3; the
  screen is a data-contract decision only.

## E2 execution record (2026-09-03/04, local RTX 5060 8 GB)

| attempt | launched | died at | cause (established) |
|---|---|---|---|
| 1 | 2026-09-03 ~14:40 | step 1024 | gate evaluation OOM; concurrent test suite on the GPU |
| 2 | 2026-09-03 22:07 | step 1024 | gate evaluation OOM with 128-row chunking (no concurrent load) |
| 3 | 2026-09-04 09:03 (resume 512) | step ~575 | concurrent CPU evaluation competing for system RAM |
| 4 | 2026-09-04 09:16 (resume 512) | step 1024 | gate evaluation OOM with 32-row chunking |
| repro | 2026-09-04 10:32 (resume 512→1032) | passed | outputs detached; GPU peak 5.7 GB at the gate |
| 5 | 2026-09-04 10:58 (resume 1032→4096) | running | same binary as the reproduction |

Root cause of 1/2/4: candle retains every op's inputs while its output lives;
the evaluator held chunk outputs, so chunking bounded nothing until the
outputs were detached. Attempt 5's checkpoints from step 1024 onward were
written by a resume launched with an absolute `output_dir`; evaluations must
use each checkpoint bundle's own `config.json`, which the evaluator enforces
by hash. E2's loss log contains the appended rows of all attempts; the
authoritative trajectory is the checkpoint chain, not the row count.

## Amendment (2026-09-04): E3 as implemented, before any result is read

`p2-eval --adaptation-falsifier` (commit eaf4b619) implements §5.2. Two facts
found during implementation bound what it can show:

1. **The as-registered arm cannot update.** §6.2 starts Channel B only once a
   level holds >= 8 unique transitions; a synthetic Learning History level has
   exactly 7 (`LEARNING_HISTORY_STEPS_PER_LEVEL = 6` movement rows plus one
   operator row). Under the registered rule the reset and carry arms therefore
   reproduce the context-only arm bit for bit with zero updates, and the
   verdict fails by construction. This is a generator/limits mismatch, not a
   model result. The registered arm is still run and reported as such. A
   **labelled deviation arm** with `--adaptation-falsifier-min-level-transitions 4`
   is the first run that actually exercises Channel B; its result is
   exploratory and cannot satisfy the §5.2 promotion rule on its own.
2. **t = 32 is unreachable** (levels in {2,3,4} give <= 28 chronological
   transitions); the rule is applied over t in {8, 16} only, and the report
   lists the skipped prefix under `verdict.skipped_prefix_lengths`.

Consequence for the generator (deferred, trajectory-changing): synthetic
levels are far shorter than live levels (tens to hundreds of actions), so the
live warm-up rule is untestable on them; a v6.1 generator should draw level
lengths from a wider range.
