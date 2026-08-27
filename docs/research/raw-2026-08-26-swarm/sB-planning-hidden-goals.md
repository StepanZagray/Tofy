# Beyond-frontier latent planning and hidden-goal inference

## Bottom line

Deliberation should beat Tofy’s current goal-free one-step ranker, but the gain will come from explicit belief reduction and shallow verified lookahead—not from deeper unconstrained latent rollout.

The sharpest bet is:

- infer a posterior over executable, predicate-structured goals;
- separate information about the goal from information about dynamics;
- propose actions cheaply across the full action space;
- spend roughly 32–64 transition-equivalent evaluations on depth-1/2 verification;
- execute one action and replan;
- refuse imagined dominance unless it survives an error-bound or deterministic certificate.

I would not make depth-4 latent search the default at the current 51.4% changed-pixel exact accuracy. As a deliberately crude warning, independent per-step fidelity would give `0.514^4 ≈ 0.07`; the independence assumption is false, but it correctly identifies open-loop depth as dangerous.

## Blunt diagnosis of the next constraints

1. **Hidden-goal inference is now the main live-score constraint.** The live policy still supplies an all-zero goal and ranks actions by learned Q, reliability, novelty, and predicted effect—not by a posterior over objectives ([arc3_live.rs:654](/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs:654), [arc3_live.rs:688](/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs:688)). Its own contract says it is not a hidden-goal solver ([arc3_live.rs:36](/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs:36)).

2. **The fixed six-family `g19` goal language will probably become the next generalization bottleneck.** Those families are executable and valuable, but they cannot naturally express conjunction, ordering, existential object bindings, or resource-dependent conditions ([domain.rs:66](/home/stepan/Coding/Personal/Tofy/src/domain.rs:66), [domain.rs:351](/home/stepan/Coding/Personal/Tofy/src/domain.rs:351), [ADR 0004:155](/home/stepan/Coding/Personal/Tofy/docs/adr/0004-latent-planning-contract.md:155)).

3. **ADR 0004’s joint goal/member EIG can reward the wrong uncertainty.** Its JS objective mixes uncertainty about goals with uncertainty about dynamics ([ADR 0004:446](/home/stepan/Coding/Personal/Tofy/docs/adr/0004-latent-planning-contract.md:446)). An action can have high joint EIG while teaching nothing about the goal.

4. **Search lacks a dominance certificate.** Scalar reliability penalties do not prevent optimizer’s-curse exploitation when thousands of candidate policies compete. Recent theory shows that imperfect world models become more exploitable as the searched policy class grows, and derives safe-horizon restrictions under bounded error ([Imperfect World Models Are Exploitable, arXiv:2605.15960](https://arxiv.org/html/2605.15960v1)).

5. **For changed-pixel accuracy, the decoder/representation seam—not planning—is probably next.** Foreground reconstruction around 0.67 looks like the likely ceiling on further exact-pixel gains. Planning work should primarily target live score and action efficiency unless it also adversarially repairs planner-distribution errors.

Expected effects below are speculative, non-additive, and expressed as absolute points on a normalized 0–100 live score, or percentage points where the evaluator is a rate.

## Ranked proposals

| Rank | Proposal | Mode | Estimated A40 cost | Expected effect |
|---:|---|---|---:|---|
| 1 | Factor goal information from rule information | EXPLOIT | Offline/CPU, under 0.1 GPUh | Live `+1–4`; changed exact `0` |
| 2 | Copy-gate predicate-support pruning | EXPLOIT | `0.1–0.3` GPUh | Same score at `2–5x` less search, or live `+0–3` |
| 3 | Predicate-interval dominance certificates | EXPLOIT | `0.1–0.4` GPUh | Live `+1–5`; invalid-plan rate down `30–70%` |
| 4 | Retrodiction-certified local edit cache | EXPLOIT | CPU/offline | Live `+1–4`; `10–40%` exact-edge coverage |
| 5 | Amortized maximally discriminating probes | EXPLOIT | `0.2–0.6` cached GPUh | Live `+1–4`; action efficiency `+10–25%` |
| 6 | Planner CEGIS at first divergence | EXPLOIT | `0.5–1.5` GPUh | Planner-distribution changed exact `+2–6`; live `+1–5` |
| 7 | Compositional predicate-automaton goal posterior | EXPLORE | `0.5–1.5` cached GPUh | Live `+2–8`, high uncertainty |
| 8 | Causal landmarks rather than latent clustering | EXPLORE | `0.8–2.0` cached GPUh | Multi-stage/live `+2–6` |

## 1. Factor goal information from rule information — EXPLOIT

**Mechanism.** Represent goal belief and rule belief separately:

- `q_G[g]`: posterior over goal hypotheses.
- `q_R[r | g]`: posterior over four or eight dynamics/rule particles.
- `P[o | a,g,r]`: predicted distribution over a compact event alphabet containing terminal status, reward-like signals, and quantized predicate changes.

For each action compute:

```text
IG_goal(a) = I(G ; O | a, history)
IG_rule(a) = I(R ; O | G, a, history)
```

Select lexicographically: safety first, goal information second, progress third. Spend rule information only when calibrated model trust is below a fixed threshold or unknown-goal mass remains high.

**Finite argument.** The chain rule is exact:

```text
I(G,R ; O) = I(G ; O) + I(R ; O | G)
```

Therefore maximizing joint EIG does not imply maximizing goal EIG. A finite counterexample needs only two goals, two rules, and an action whose outcome identifies the rule but is conditionally independent of the goal. This is straightforward to formalize in Lean with finite probability tables.

**Cost.** No new world-model parameters. For up to 64 goals, 4 members, 64 shortlisted actions, and roughly 32 event bins, the posterior arithmetic is negligible beside a transition forward pass.

**Cheap falsifier.** Offline on existing checkpoints:

1. Compute current joint EIG and factorized EIG for identical candidate actions.
2. Use exact synthetic transitions to measure posterior entropy reduction after execution.
3. Reject if factorization fails to improve median goal-entropy reduction or actions-to-`0.9` true-goal posterior by at least 10%.

**Expected effect.** Changed-pixel exact `0`; live `+1–4` and `10–30%` fewer probes before goal concentration.

**Failure modes.** Goals and rules may be strongly coupled, making a factorized posterior miscalibrated. Detect this through held-out joint outcome NLL and the gap between full low-rank `q(G,R)` and `q(G)q(R|G)`. Retain a low-rank coupling term if independence is rejected.

## 2. Copy-gate predicate-support pruning — EXPLOIT

**Mechanism.** Compile each goal hypothesis into a critical-cell mask `R_g`. Examples include the target component boundary, cells whose color count controls the predicate, protected hazards, and resource locations.

Add a tiny action-conditioned `16x16` change-support head before full recurrence, or derive exact geometric support for coordinate actions. Score:

```text
test_mass(a) =
  sum_g q_G[g] *
  P(predicted edits intersect R_g | a)
```

Use this only to select roots. After one full transition, refine it with the existing per-pixel copy gate ([grounding.rs:51](/home/stepan/Coding/Personal/Tofy/src/p2/grounding.rs:51)). Expand depth-2 only if the predicted edit can affect a live predicate, expose a new region, or avert a protected-set violation.

**Finite argument.** If predicate `φ` depends only on cells in `R`, then:

```text
s restricted to R = s' restricted to R
implies
φ(s) = φ(s')
```

Thus an action certified not to modify `R` cannot immediately falsify or satisfy `φ`. The theorem is exact; only support prediction is empirical.

**Cost.** A single small support head is about 10–20K parameters. One thousand cached-latent updates should take roughly `0.1–0.3` A40 hours. At inference it should remove 40–80% of recursive transition calls.

**Cheap falsifier.** On exact generator episodes, calculate recall of actions that actually change any viable predicate. Require:

- at least 99% useful-action recall;
- at least 40% pruning;
- no goal family below 97% recall.

Fail closed below those thresholds.

**Expected effect.** No direct changed-pixel gain. Either equivalent performance at `2–5x` less planning compute or `+0–3` live points at the same deadline.

**Failure modes.** Long-range operators, teleportation, hidden counters, or wrongly compiled supports create false negatives. Track recall by operator and goal family; never use this head for safety rejection unless support is analytically exact.

## 3. Predicate-interval dominance certificates — EXPLOIT

**Mechanism.** Convert color logits and copy probabilities into calibrated per-cell possibility sets. Propagate those sets through predicate abstractions:

- lower and upper color counts;
- definite and possible connectivity;
- definite and possible collision;
- resource intervals;
- terminal and hazard possibility;
- goal-automaton states reachable under any admitted board.

Every plan receives a utility interval `[J_min, J_max]`. Prefer an imagined plan over the factual/reactive comparator only when:

```text
J_min(imagined plan) > J_max(comparator)
```

Otherwise fall back to a factual graph edge, a probe, or the reactive action.

**Finite argument.** Under the explicit premise that every true successor board remains inside the propagated set:

```text
J_min(plan) <= J_true(plan) <= J_max(plan)
```

Consequently, disjoint intervals prove the ordering. For calibrated one-step containment probability `1-δ`, a simple union bound gives trajectory containment of at least `1-Hδ`; no independence assumption is needed. Both statements are finite and Lean-formalizable.

**Cost.** Mostly CPU bitsets and integer intervals. Calibration on existing checkpoints should need `0.1–0.4` GPU hours at most. Abstract interpretation adds perhaps 10–20% controller time but should reduce expensive rollout depth.

**Cheap falsifier.** Offline rescore existing exact transitions and short trajectories. Measure:

- empirical set containment;
- actionable dominance rate;
- false-dominance rate;
- interval width by horizon.

Reject if H2 false dominance exceeds the preregistered safety tolerance or fewer than 10% of decisions receive actionable certificates.

**Expected effect.** Changed exact `0`; live `+1–5`, principally by eliminating attractive model hallucinations. Expect invalid or catastrophically wrong plans to fall by 30–70%.

**Failure modes.** Intervals can become vacuous, especially under distribution shift. Detect widening by depth and use a strict fallback when no candidate dominates. Marginal conformal coverage must not be misreported as simultaneous trajectory coverage.

## 4. Retrodiction-certified local edit cache — EXPLOIT

**Mechanism.** Build an episode-scoped cache of factual deterministic transitions. Each entry contains:

- a D4- and color-canonical radius-`r` neighborhood around the action locus and changed components;
- old and new local colors;
- an exact edit delta;
- a hash of the unchanged exterior;
- a reverse signature proving the edit reconstructs the predecessor.

On a new state, an entry may be reused only if its entire causal signature matches. Apply the stored delta, retrodict the old neighborhood exactly, and verify the exterior invariant. Otherwise use the neural model. Because operators are resampled, do not reuse these certificates across episodes.

This extends the exact observed graph—which is currently factual only ([agent_session.rs:1](/home/stepan/Coding/Personal/Tofy/src/p2/agent_session.rs:1), [agent_session.rs:129](/home/stepan/Coding/Personal/Tofy/src/p2/agent_session.rs:129))—across states without treating model agreement as proof.

**Finite argument.** Tofy already proves that an `r`-local update gives identical local outputs for states agreeing on the relevant neighborhood ([Locality.lean:49](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/Locality.lean:49)). Therefore cache reuse is exact under deterministic `r`-local dynamics and complete signature matching.

**Cost.** Zero training GPU cost. Runtime is hash lookup plus small board edits. Memory should remain in kilobytes to low megabytes per episode.

**Cheap falsifier.** Offline exact trajectories:

1. Sweep `r`.
2. Group equal signatures.
3. Count conflicting next-state deltas and outside-cone changes.
4. Require zero conflicts on calibration and held-out episodes before enabling reuse.

**Expected effect.** Changed exact `0`; live `+1–4`, with 10–40% of relevant transitions becoming exact rather than predicted.

**Failure modes.** Hidden global counters and nonlocal rules can alias identical neighborhoods. Any collision invalidates that radius/operator class. Increase the signature or disable reuse; never resolve a collision statistically.

## 5. Amortized maximally discriminating probes — EXPLOIT

**Mechanism.** Train a dedicated probe head, distinct from the generic action proposal head. Inputs are:

- spatial latent;
- posterior-set embedding;
- posterior-weighted predicate-critical heatmap;
- tried-action mask and remaining budget.

Outputs are seven action-type logits and a `64x64` coordinate map. An exact synthetic teacher calculates each action’s safe goal-information score, or the posterior mass split it induces. Train with:

- KL to `softmax(IG_goal / τ)`;
- pairwise margins against the runner-up;
- an auxiliary top-4 recall loss.

At runtime, evaluate all actions through this cheap head, then compute exact model-based EIG only for the top 2–4.

This is the direct analogue of amortized experimental design: DAD learns a policy that emits informative experiments in a single forward pass ([DAD, arXiv:2103.02438](https://arxiv.org/abs/2103.02438)).

**Finite argument.** If the teacher has a best-action margin `γ` and the head approximates every candidate score within `ε`, then `2ε < γ` preserves the argmax. Tofy already formalizes the corresponding ordinal-margin result ([Greedy.lean:26](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/Greedy.lean:26)).

**Cost.** Roughly 30–80K parameters. Two thousand cached-latent steps: `0.2–0.6` A40 hours. Recomputing the world-model trunk could raise this to roughly 2–3 hours and is unnecessary for the screen.

**Cheap falsifier.** Train for 1K steps and test:

- top-1/top-4 teacher recall;
- realized posterior entropy reduction;
- action efficiency;
- safety violations.

Reject if top-4 recall is below 90% or realized information is not better than the current proposal head.

**Expected effect.** Changed exact `0`; live `+1–4`, action efficiency `+10–25%`, and a major reduction from ADR 0004’s exhaustive 4,102-root Phase A search ([ADR 0004:335](/home/stepan/Coding/Personal/Tofy/docs/adr/0004-latent-planning-contract.md:335)).

**Failure modes.** Teacher bias, posterior distribution shift, and small score margins. Log the teacher margin distribution and fall back to wider model rescoring whenever predicted margins are small.

## 6. Planner CEGIS at first divergence — EXPLOIT

**Mechanism.** Let the current planner actively search for high-utility trajectories on exact synthetic games. Execute each selected plan in the exact simulator and locate its first predicted/actual divergence. Add:

- the divergent `(state, action, true successor)` transition;
- a retained “no-good” constraint for the hallucinated plan;
- a pairwise loss forcing its utility upper bound below the factual comparator.

Fine-tune only rank-4 adapters on late action-FiLM/residual blocks plus grounding heads. Use pixel CE, copy-gate loss, inverse consistency, and ordinal no-good loss. Replay every previous counterexample to prevent cyclic forgetting.

ACID shows inverse-action consistency can improve planning efficiency ([arXiv:2607.02403](https://arxiv.org/abs/2607.02403)); WAV reports adversarial world-model repair improving downstream policy quality ([arXiv:2604.01985](https://arxiv.org/abs/2604.01985)). The new step here is finite first-divergence CEGIS tied to planner-specific dominance constraints.

**Finite argument.** For a finite registered plan set, if every iteration eliminates at least one previously violating plan, retained constraints never regress, and the adapter class is realizable, the loop terminates after at most the number of violating plans. Those assumptions and the termination measure are Lean-formalizable.

**Cost.** About 20–80K trainable adapter parameters. A 256-plan mining screen followed by 1–2K cached/adapted updates should take `0.5–1.5` A40 hours.

**Cheap falsifier.** Use one checkpoint and fixed seeds:

- mine 256 counterexamples;
- train for at most 1K steps;
- rerun every retained constraint;
- measure held-out planner exploitation gap and ordinary changed-pixel exact.

Reject on any retained-constraint regression, ordinary accuracy loss over 1 point, or no reduction in first-divergence frequency.

**Expected effect.** Planner-distribution changed exact `+2–6`; live `+1–5`.

**Failure modes.** Adapter capacity may be insufficient, or the planner may endlessly expose new unrelated errors. Track unique failure clusters and constraint retention. A rising cluster count without falling severity means the planner needs certificates or shorter horizons, not more repair.

## 7. Compositional predicate-automaton goal posterior — EXPLORE

**Mechanism.** Replace the flat six-family endpoint with a bounded typed grammar over the already executable P1-style predicates:

- atoms: component count, color count, containment, contact, reachability, creation/deletion/recoloring, resource state;
- bindings: objects, colors, regions, component identities;
- combinators: `AND`, `OR`, `UNTIL`, `BEFORE`, and bounded ordered sequences.

Compile each candidate into a small deterministic automaton over observed event vectors. A grammar-production head proposes at most 64 programs and bindings from the initial board and interaction history. Actual frame deltas update the posterior through exact predicate execution with a contamination likelihood so one unexplained event does not delete the true goal. Neural dynamics is used prospectively, not to reinterpret observed history.

**Finite argument.** Let `V_t` be the surviving weighted hypothesis set. If each selected probe leaves at most fraction `ρ` of the current viable mass under every possible observation, then after `k` probes:

```text
mass(V_k) <= ρ^k * mass(V_0)
```

If the true automaton is initially represented and exact observations never eliminate consistent hypotheses, it remains in the version space. Both are finite induction arguments.

**Cost.** Around 50–120K parameters for grammar production and bindings; automaton execution is CPU bitset work. Cached training should take `0.5–1.5` A40 hours.

**Cheap falsifier.** Before training, enumerate the bounded grammar on existing exact trajectories. Require top-64 candidate recall on held-out compositional goals to beat the six-family baseline by at least 15 points. Only then train the proposal head for at most 2K updates.

**Expected effect.** Changed exact `0`; live `+2–8`. This has the largest hidden-goal upside but also the widest uncertainty.

**Failure modes.** Grammar explosion, parser aliasing, or unrepresented objectives. Track true-goal coverage before posterior accuracy. Preserve explicit unknown mass and fall back to structure-changing exploration when all programs fit poorly.

## 8. Causal landmarks rather than latent clustering — EXPLORE

**Mechanism.** Discover landmarks from exact transition graphs, not nearest-neighbor latent geometry. Candidate landmarks are states where:

- an affordance set changes;
- a graph dominator is crossed;
- a resource or irreversible predicate changes;
- many successful paths pass through the same canonical event signature.

Canonicalize under D4 and color permutation. Train a small spatial head for 16–32 landmark classes and ordinal distance buckets `0..8, infinity`. The planner searches depth 1–2 toward a landmark, then traverses learned landmark edges instead of attempting a depth-4 pixel rollout.

**Finite argument.** Suppose every successful path decomposes into `m` landmark segments, each solvable by an H-step local controller with failure probability at most `ε`. A union bound gives:

```text
P(success) >= 1 - m*ε
```

The premises—coverage, segment length, and local success—are directly measurable. If p90 landmark distance remains above the permitted local horizon, the idea is falsified before training.

**Cost.** About 80–180K parameters. Label construction is CPU graph work; 2K cached updates should take `0.8–2.0` A40 hours.

**Cheap falsifier.** Build landmarks offline and require:

- p90 successful-path segment length at most 2–4;
- low next-effect entropy per landmark;
- coverage of at least 80% of successful paths with at most 32 landmarks.

**Expected effect.** Changed exact `0`; live/multi-stage success `+2–6`.

**Failure modes.** Visually common states may not be causal. Dominator frequency can also overfit generator topology. Measure intervention-conditioned effect entropy and hold out entire operator families.

## Does model deliberation beat reactive ranking here?

**My answer is yes, with medium confidence—but only for belief-aware, shallow, verified deliberation.**

The strongest evidence is:

- A reactive policy cannot be optimal for observationally identical states governed by different hidden goals; Tofy already formalizes this impossibility ([Policy.lean:16](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/Policy.lean:16)).
- P1’s exact falsification controller materially beat sequential/reactive selection on hard cases; ADR 0004 records that evidence and its limits ([ADR 0004:300](/home/stepan/Coding/Personal/Tofy/docs/adr/0004-latent-planning-contract.md:300)).
- Thinker demonstrates learned decision-time planning in compact deterministic environments, including Sokoban-like reasoning ([Thinker, arXiv:2307.14993](https://arxiv.org/abs/2307.14993)).
- The broader planning literature finds that shallow/simple trees often capture most of the benefit unless the task requires genuinely deep reasoning ([Hamrick et al., arXiv:2011.04021](https://arxiv.org/abs/2011.04021)).
- Gumbel search makes few-simulation policy improvement plausible, but its clean guarantees rely on accurate value ordering ([Gumbel MuZero](https://openreview.net/pdf?id=bERaNdoegnO)). That is precisely why Tofy needs ordinal certificates and fallback behavior.
- GC-IDM shows that well-structured goal-conditioned planning can be amortized extremely aggressively, matching or exceeding CEM on most tested tasks at far lower compute, but it assumes a much cleaner goal interface than hidden-goal ARC provides ([arXiv:2605.08732](https://arxiv.org/html/2605.08732)).

This evidence does **not** establish that an unverified 0.5M model can profitably roll four steps through 4K actions. It supports a smaller claim: explicit posterior maintenance plus a handful of discriminating, verified simulations should beat the current goal-free ranker.

## Minimal planning compute likely to capture most of the gain

My recommended initial budget is:

1. One observation encoding.
2. Cheap probe/support scores for every legal action.
3. Full one-step transitions for the top 16.
4. Expand the top four roots with four continuations each: 16 additional transitions.
5. Apply four uncertainty members only to the final 2–4 trajectories.
6. Execute one action and replan.

That is approximately 32 ordinary transitions plus 8–32 lightweight member-step checks, or roughly **40–64 full-transition equivalents** depending on trunk sharing. Horizon is normally 2. Horizon 4 is allowed only through factual graph edges, certified local edits, or interval-certified continuation.

Preregister budgets `{0, 8, 16, 32, 64}` and horizons `{1, 2, 4}`. Define “most of the gain” as the smallest budget obtaining at least 90% of the paired live-score improvement of the largest budget, with no increase in terminal mistakes. My bet is **32 evaluations will capture most routine gains and 64 most hard-case gains**. The current 4,102-root exhaustive Phase A and 1,024-member-step Phase B budgets should be treated as teacher/oracle configurations, not the eventual deployed controller ([ADR 0004:335](/home/stepan/Coding/Personal/Tofy/docs/adr/0004-latent-planning-contract.md:335), [ADR 0004:512](/home/stepan/Coding/Personal/Tofy/docs/adr/0004-latent-planning-contract.md:512)).

The first three experiments I would fund are factorized EIG, copy-support pruning, and predicate-interval dominance. Together they test the core thesis without retraining the world model. If they fail, deeper planning is unlikely to rescue the present checkpoint.

No repository files were changed.

Status: DONE
