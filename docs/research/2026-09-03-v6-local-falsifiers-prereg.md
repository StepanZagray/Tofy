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
- Budget: local run, physical batch 256 (accumulation 4 → effective 1024),
  4096 optimizer steps, seed 2, evaluation at 1024/2048/3072/4096.
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
