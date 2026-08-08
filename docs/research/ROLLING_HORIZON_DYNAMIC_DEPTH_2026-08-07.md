# Rolling-horizon progressive recursion

Date: 2026-08-07
Status: research assessment; no implementation or model-quality claim

## Verdict

The proposal is a good research hypothesis, but not yet a safe model change.
Its strongest part is the compute pipeline: a future transition can receive a
small amount of work early and be topped up as its execution deadline
approaches. After the horizon is filled, the top-ups telescope to one
full-depth transition per environment step, rather than one full-depth
transition for every horizon position.

The central unresolved issue is **conditioning validity**. A cached future
latent was refined under predicted predecessor states. Once the environment
returns the real next observation, continuing that latent is a warm start for
a changed problem, not automatically a continuation of the same computation.
The design therefore needs observation-conditioned repair, invalidation, and
training for variable depth and resumed computation. Without those pieces,
the nominal compute saving can preserve or amplify a wrong imagined future.

For Tofy specifically, this should follow rather than precede the current
action-conditioning fix. The readiness-v2 dynamics are action-marginalized
([`docs/RESULTS_P2.md`](../RESULTS_P2.md)), and the live policy has no trained
reward or action-value head ([`docs/P2_ARC3_LIVE_EVAL.md`](../P2_ARC3_LIVE_EVAL.md)).
A cheaper eight-step rollout is not useful planning until actions materially
change predictions and a task-utility signal can rank plans.

## Precise formulation and compute

Use one unit consistently. Let one **recursion call** mean one application of
the shared two-layer block. A maximum effective layer depth of 256 is then

```text
R = 128 recursion calls.
```

For horizon `H = 8`, the proposed geometric cumulative targets are

```text
distance to execution h       1    2   3   4  5  6  7  8
cumulative calls C(h)       128   64  32  16  8  4  2  1
effective two-layer depth   256  128  64  32 16  8  4  2
```

This resolves a unit ambiguity in the proposal: when old `p2` becomes the
immediate transition, it needs 64 additional block calls (128 additional
layers), bringing it from 64 to 128 calls, or from effective depth 128 to 256.

The first plan fill costs

```text
sum_h C(h) = 255 calls
```

versus `8 * 128 = 1024` calls if all eight positions are immediately refined
to full depth: 75.1% fewer block calls for the initial fill.

After one real action, a surviving item moves from distance `h+1` to `h` and
receives `C(h)-C(h+1)` additional calls. A new tail item receives `C(H)`.
Therefore steady-state work is

```text
sum_{h=1}^{H-1} [C(h)-C(h+1)] + C(H) = C(1) = R.
```

This identity holds for **any monotone cumulative schedule**, not just a
geometric one. Under perfect cache reuse, the steady-state comparison is 128
versus 1024 calls per real step: an 8x FLOP reduction. Schedule shape changes
when work is spent, how much early work becomes stale, initial fill cost, and
plan quality; it does not change this ideal steady-state total.

The 8x figure is not a latency guarantee. Full-depth horizon items can be
batched efficiently on a GPU, while the immediate item's top-ups remain
sequential and variable-depth batches become progressively smaller. Candidate
plans or PTRM trajectories may provide enough batch width to recover device
utilization, but wall-clock latency must be measured separately from block-call
count.

## Closest primary work

1. [Tiny Recursive Model (TRM)](https://arxiv.org/html/2510.04871) repeatedly
   refines a solution latent `y` and scratch latent `z` using a shared two-layer
   network, and carries detached `(y,z)` across deep-supervision steps. This is
   repeated refinement of one static problem, not rolling replanning after new
   environment observations.
2. [Probabilistic TRM (PTRM)](https://arxiv.org/html/2605.19943) adds Gaussian
   noise at each deep recursion step, runs `K` parallel latent trajectories,
   and selects with a Q head. It scales **width**, not horizon-dependent depth,
   and does not reuse a predicted future slot after observing a changed root.
3. [Looped World Models](https://arxiv.org/html/2606.18208) is the closest
   neural precedent. It combines parameter-shared iterative world dynamics,
   stochastic variable-depth training, adaptive early exit, spectral stability,
   and cross-timestep hidden-state propagation. It does not present the
   deadline schedule or the telescoping future-slot pipeline proposed here.
4. [Adaptive Computation Time](https://arxiv.org/abs/1603.08983),
   [PonderNet](https://arxiv.org/abs/2107.05407), and
   [Mixture-of-Recursions](https://arxiv.org/html/2507.10524) establish learned
   input-, sample-, or token-dependent compute. They support learning an exit
   rule instead of freezing a geometric curve, but do not solve temporal cache
   consistency.
5. Reusing work after the world changes is classical incremental planning.
   [D* Lite](https://ocs.aaai.org/Library/AAAI/2002/aaai02-072.php) reuses prior
   search information and repairs affected values rather than treating all old
   work as still correct. Receding-horizon control also shifts and warm-starts
   the previous solution; a study of
   [learning-aided MPC warm starts](https://arxiv.org/abs/2310.02918) explicitly
   reports that conventional shifted warm starts can fail when uncertainty
   changes the optimization problem substantially.
6. [Deep equilibrium optical flow](https://arxiv.org/abs/2204.08442) gives a
   useful iterative-estimation analogy: fixed-point reuse can save work, but the
   method also needs correction and stable recurrent dynamics.
7. [Logical extrapolation without overthinking](https://arxiv.org/abs/2202.05826)
   shows that simply applying more recurrent iterations can degrade predictions.
   Keeping the original input visible and training for progressive, iteration-
   agnostic improvement made the tested recurrent solvers self-correct after
   input perturbations.

The exact combination of deadline-scheduled future slots, cumulative top-ups,
and repair after real observations was not found in the primary sources above.
That is evidence of a distinct combination, not a novelty or patentability
claim; a formal literature review would need a broader search.

## The state-consistency problem

Suppose cached slot `p2` was computed as

```text
(y, z) <- F^64(y0, z0; x(predicted_s1, planned_a1)).
```

After executing `p1`, the environment supplies `actual_s1`. The correct new
conditioning input is

```text
x' = x(actual_s1, revised_a1).
```

Continuing `F` from the old `(y,z)` with `x'` is a warm start. The first 64
calls cannot be counted as 64 valid calls on the new problem unless the model
was trained to repair such changes and the operator is stable.

If `F` is contractive in its recurrent state with factor `kappa < 1` and
Lipschitz in its conditioning input with constant `beta`, the fixed-point
shift is bounded by

```text
||u*(x) - u*(x')|| <= beta / (1-kappa) * ||x-x'||.
```

This makes reuse principled when observation innovation is small. Tofy does
not currently establish such a contraction. Its research history includes
severe degradation when recursion was evaluated beyond trained depth, and
normalizing an iterate bounds scale without guaranteeing convergence in
direction.

There is also a dependency cascade: `p3` was conditioned on the old `p2`, and
so on. Repair must propagate from the first changed predecessor through the
suffix. In a stochastic or partially observed task, an open-loop sequence may
be the wrong cached object altogether; a small contingent tree or set of PTRM
trajectories can preserve alternative futures until the observation selects a
branch.

## Better schedule than a fixed `log2` curve

Geometric decay is a good first baseline because it strongly delays work until
the conditioning state is more reliable:

```text
C(h) = max(C_min, round(R * rho^(h-1))),  0 < rho < 1.
```

There is no theoretical reason to fix `rho = 0.5`. Compare at equal realized
compute:

- geometric `rho` in `{0.25, 0.5, 0.75}`;
- a deadline power curve
  `C(h)=C_min+(R-C_min)*((H-h)/(H-1))^alpha` for `alpha` in `{1,2,4}`;
- an adaptive allocator using measured marginal value per block call.

The adaptive priority for one more call to slot `h` should combine

```text
probability cached work survives
* value of that horizon to the current decision
* predicted loss/uncertainty reduction from one more call
/ measured call cost.
```

Useful online signals are observation innovation, recurrent residual
`||u_n-u_(n-1)||`, Q/reliability uncertainty, and ensemble/PTRM disagreement.
Give every slot a small minimum budget so delayed rewards are not starved, keep
a hard maximum, and train any learned exit gate with an explicit compute cost.
Start with fixed curves: learned halting adds a second failure mode before cache
reuse itself has been validated.

## Tofy implementation fit

This is not currently a scheduler-only change:

- [`ForwardOutput`](../../src/p2/model.rs) retains `y` but not scratch `z`.
- `run_recursion` accepts an optional `y_init` but initializes `z` to zero on
  every call.
- `forward_from_latent_with_depth` supports explicit variable depth, but starts
  a fresh transition recursion rather than resuming a saved `(x,y,z)` state.
- Current depth randomization trains only the configured range. A proposed 128
  block calls must be inside the training distribution or separately shown not
  to overthink.

A viable cache entry needs at least the action, conditioning input or parent
version, `y`, `z`, calls already applied, uncertainty/residual statistics, and
the predicted predecessor used to create it. On observation:

1. compare the real encoded state with the cached one-step prediction;
2. shift the plan only if the executed action matches;
3. recondition and repair the surviving suffix from near to far;
4. reset a slot or suffix when innovation exceeds a frozen threshold;
5. append a new shallow tail slot;
6. spend the remaining fixed per-action budget by priority.

A joint horizon latent (a plan tape with one slot per future step) may be safer
than eight independent recursions because an observed-root correction can be
propagated through the entire suffix. It is a larger architecture change and
should follow a minimal per-transition cache experiment.

## Decisive experiment

Run this first on deterministic synthetic traces with fixed oracle action
sequences, before adding action search or PTRM noise.

### Protocols

At equal block-call and wall-clock budgets compare:

1. fresh full-depth replanning;
2. fresh variable-depth planning with no reuse;
3. shifted cache with no correction;
4. shifted cache with innovation-gated repair/reset;
5. oracle cache validity, which resets exactly when a cached predecessor is
   inconsistent with the exact simulator.

Sweep geometric, power, and residual-priority schedules. Include deliberately
perturbed observations and action changes so the evaluation is not dominated
by easy exact-match shifts.

### Measurements

- one-step and horizon-1..8 latent error after each partial depth;
- error versus block calls and wall-clock latency;
- fraction of cached calls retained, repaired, and discarded;
- recovery calls after observation innovation;
- recurrence residual and whether extra calls monotonically improve error;
- action-shuffle ratio as a prerequisite;
- only after prediction gates pass: closed-loop success and action efficiency.

### Go/no-go gates

Proceed to a planner integration only if:

1. resumed-and-repaired states reach fresh-run error with at least 2x fewer
   measured calls on held-out traces;
2. correction overhead leaves at least a 2x wall-clock gain, not merely an
   idealized FLOP gain;
3. output quality does not degrade with additional calls anywhere in the
   deployed range;
4. action conditioning passes the existing `>1.1` shuffle-ratio gate;
5. long-horizon prediction and, later, closed-loop task success are not worse
   than full-depth replanning at matched total compute.

The highest-information first test is the counterfactual cache probe: compute
old `p2` under predicted `s1`, reveal the exact `s1`, then compare continued
repair from the old state against a fresh recursion from exact `s1` over a
curve of additional calls. If warm starts do not dominate fresh starts there,
the rolling architecture's core assumption is false for the current model.
