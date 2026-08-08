# Research ideas

These are unvalidated research hypotheses, not committed architecture or reported
results. Each idea should graduate through a compute-matched falsification experiment
before entering the P2 plan.

## Rolling-horizon progressive recursion

**Status:** proposed; blocked on action-conditioned dynamics and a trained plan-utility
signal. Detailed assessment:
[`research/ROLLING_HORIZON_DYNAMIC_DEPTH_2026-08-07.md`](research/ROLLING_HORIZON_DYNAMIC_DEPTH_2026-08-07.md).

### Idea

Maintain a rolling plan of future transitions `p1..pH`. Give the transition nearest
execution the most recurrent refinement and distant transitions progressively less.
After executing `p1` and observing the real next state, shift the surviving plan,
repair or invalidate affected cached latents, top each valid transition up to its new
depth target, and append a shallow new tail transition.

For an eight-step horizon and a shared two-layer block, one possible cumulative
schedule is:

```text
distance to execution h       1    2   3   4  5  6  7  8
recursion calls C(h)         128   64  32  16  8  4  2  1
effective layer depth       256  128  64  32 16  8  4  2
```

Here one recursion call means one application of the two-layer shared block. When old
`p2` becomes immediate, it receives 64 additional calls, reaching 128 calls / 256
effective layers in total.

The initial fill costs `sum C(h) = 255` calls instead of `8 * 128 = 1024`. In steady
state, surviving slot `h+1` receives `C(h)-C(h+1)` calls and the new tail receives
`C(H)`, so the work telescopes:

```text
sum_{h=1}^{H-1} [C(h)-C(h+1)] + C(H) = C(1).
```

Under perfect reuse, the distributed top-ups total `C(1)` shared-block calls per
environment step rather than `H * C(1)` calls for fresh full-depth evaluation of
every slot. This is an ideal block-call bound, not a wall-clock or model-quality
guarantee. The identity holds for any monotone cumulative schedule; geometric decay
controls when work is spent, not the ideal steady-state total.

### Required repair semantics

A future latent was refined under predicted predecessor states. Once the real next
observation arrives, continuing it is a warm start for a changed problem, not
automatically the remaining computation on the same problem. A viable cache entry
therefore needs the planned action, parent/version or conditioning input, recurrent
`(y,z)` state, applied-call count, and residual/uncertainty signals.

At each real step:

1. compare the encoded observation with the cached one-step prediction;
2. shift only the branch matching the executed action;
3. recondition and repair the surviving suffix from near to far;
4. reset a slot or suffix when observation innovation is too large;
5. append a new shallow tail;
6. spend the remaining per-action budget on the highest-value valid refinements.

For stochastic or partially observed tasks, retain a small contingent tree or PTRM
trajectory set instead of assuming one open-loop future remains valid.

### Depth allocation

Use the geometric schedule as a simple first baseline, not as a fixed architectural
law:

```text
C(h) = max(C_min, round(R * rho^(h-1))),  0 < rho < 1.
```

Compare it at equal realized compute with a deadline power curve and an adaptive
allocator. The adaptive priority for one more recursion should approximate:

```text
probability cached work survives
* value of this horizon to the current decision
* expected error or uncertainty reduction
/ measured recursion cost.
```

Candidate signals are observation innovation, recurrent residual, Q/reliability
uncertainty, and ensemble/PTRM disagreement. Keep a minimum budget for every horizon
so long-delayed consequences are not starved.

### P2 prerequisites

- Pass the existing action-shuffle gate; the readiness-v2 diagnostic provides strong
  evidence that predictions were action-marginalized
  ([results](RESULTS_P2.md#readiness-v2-action-conditioning-diagnostic-2026-08-07)).
- Add a trained reward, action-value, or equivalent task-utility signal; the current
  live policy ranks transition fidelity rather than goal value
  ([policy scope](P2_ARC3_LIVE_EVAL.md#policy-scope)).
- Expose resumable recurrent state. The current forward output retains `y` but not
  scratch `z`, and each fresh `run_recursion` initializes `z` from zero.
- Train and validate every deployed depth. Do not jump directly to 128 block calls
  from the current much smaller trained range.
- Demonstrate that additional recursion improves or preserves prediction quality and
  measure wall-clock latency; ideal block-call savings may be reduced by sequential
  execution and poor GPU occupancy.

### First falsification experiment

Use deterministic synthetic traces and fixed oracle action sequences. At matched
block-call and wall-clock budgets compare:

1. fresh full-depth replanning;
2. fresh variable-depth planning without cache reuse;
3. shifted cache without correction;
4. shifted cache with observation-gated repair/reset;
5. oracle invalidation using exact simulator state.

The decisive probe computes old `p2` from predicted `s1`, reveals exact `s1`, and
compares continued repair against a fresh recursion over the same additional-call
curve. Proceed only if repaired caches match fresh prediction quality with at least
2x fewer calls and retain at least a 2x measured latency advantage after invalidation
overhead.

### Related work

- [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) — shared two-layer
  recursion and persistent `(y,z)` refinement on one static problem.
- [Probabilistic Tiny Recursive Model](https://arxiv.org/abs/2605.19943) — stochastic
  parallel trajectories and Q-based selection; scales width rather than rolling
  horizon depth.
- [Looped World Models](https://arxiv.org/abs/2606.18208) — the closest neural
  precedent: variable-depth iterative world dynamics, adaptive exit, stability
  constraints, and cross-timestep state propagation.
- [Adaptive Computation Time](https://arxiv.org/abs/1603.08983),
  [PonderNet](https://arxiv.org/abs/2107.05407), and
  [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) — learned allocation of
  recurrent computation.
- [D* Lite](https://aaai.org/papers/00476-aaai02-072-d-lite/) — incremental
  planning that repairs reusable prior search state after changes.
- [Learning-Aided Warmstart of Model Predictive Control](https://arxiv.org/abs/2310.02918)
  — shifted plans are useful warm starts but can become invalid when uncertainty
  substantially changes the optimization problem.
- [Deep Equilibrium Optical Flow Estimation](https://arxiv.org/abs/2204.08442) —
  fixed-point reuse and correction in iterative estimation.
- [Logical Extrapolation Without Overthinking](https://arxiv.org/abs/2202.05826) —
  recurrent depth can degrade predictions unless the update is trained for stable,
  iteration-agnostic progress and retains access to the conditioning input.
