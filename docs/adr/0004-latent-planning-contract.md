# ADR 0004: World-core v5 latent planning contract

- Status: proposed (accepted after review)
- Date: 2026-08-25
- Amends: [ADR 0003](0003-world-core-v5-foundation-v2.md) at decision time only
- Basis: the in-repository evidence index at the end of this ADR

## Amendments (2026-08-27, pre-acceptance)

Three clauses are amended before acceptance, from the 2026-08-26 planning
review threads (T3 `ac1c8438`, T2 `b56d996c`); the original text below is
retained unmodified for history.

1. **A5 selection safety.** The exhaustive `6 + 64^2 = 4,102`-root depth-1
   sweep with an argmax finalist is demoted from deployed controller to
   teacher/oracle use only. With per-candidate model error `epsilon`, the
   maximum over ~4,102 noisy scores selects a corrupted finalist with
   probability approaching 1 (winner's curse); the union bound in the
   planning note is a worst case, not a calibration. The deployed Phase A
   controller must use a bounded candidate budget (order 40-64 evaluations,
   horizon <= 2) with a selection-aware acceptance charge; the
   selection-charge identity holds under independent sub-Gaussian score
   increments, and the correlated case is an open lemma
   (`PlanningSelection.lean` target).
2. **B2 EIG factorization.** The joint expected-information-gain score can
   reward actions whose information is entirely about mechanics/rules and
   carries zero information about the goal hypothesis (information
   chain rule: `I(A; Goal, Rule) = I(A; Rule) + I(A; Goal | Rule)`; the
   first term alone can dominate). Goal-directed selection must score the
   goal term `I(A; Goal | Rule)` separately, and rule-information may only
   be credited by an explicit, bounded exploration term.
3. **Trust gate.** Inverse-action consistency is removed as trust evidence
   for possibly irreversible edges: a predicted latent that encodes only
   the conditioned action identity passes inverse-action recovery exactly
   while being arbitrarily wrong about the state, so the check cannot
   certify transition fidelity. It may remain as a falsifier (its failure
   is disqualifying) but its success contributes nothing to the `0.02`
   trust bounds.

## Decision

Tofy will replace the bare greedy one-step confidence blend with one phased,
receding-horizon planner. The planner keeps an explicit posterior over finite
goal hypotheses, treats real transitions as immutable facts, searches short
action sequences in the patch-4 spatial Consumer Latent, selects jointly-safe
falsification probes while the goal is uncertain, and executes exactly one
environment action before observing and replanning.

The phases are cumulative rather than alternative planners:

1. **Phase A** is a rewire-only controller over the first usable
   foundation-v2 EMA checkpoint. It adds no trained parameters. It ports P1's
   falsification-only controller onto `forward_from_latent`, but puts an exact
   observed-state graph under the learned model and converts P1's hard belief
   deletion and exact safety predicates into calibrated soft evidence and
   fail-closed trust gates.
2. **Phase B** preserves Phase A's factual memory, goal posterior, safety
   contract, and fallback. A short, non-disruptive fine-tune adds an
   AlphaZero-style action proposal, a goal-conditioned progress value, a
   candidate encoder/event calibration seam, and a four-member lightweight
   epistemic ensemble. Gumbel-Top-k plus Sequential Halving replaces Phase A's
   hand-built beam, and an expected-free-energy-style score unifies progress,
   information, risk, and environment-action economy.
3. **Phase C** preserves the explicit controller and adds resettable
   in-episode belief recurrence and prequential fast adapters. Adaptation may
   affect only episode-local modules and is accepted only when exact replay of
   earlier factual transitions does not regress. Search may be amortized away
   temporarily only after explicit per-game sufficiency gates pass.

This ADR supersedes ADR 0003's `Live policy stays a searchless forward pass`
scope constraint once Phase A is activated. It does **not** change, delay, or
add a gate to the current foundation-v2 training run. ADR 0003 remains the
world-model and training contract; this ADR is its decision-time consumer and
a proposed amendment for a later head-only fine-tune.

The claim is bounded. P1's exact-simulator evidence supports falsification on
its synthetic hidden-goal distribution, especially its adversarial retargeting
slice; it does not prove that a learned latent planner will improve public
ARC-AGI-3. Phase A, Phase B, and Phase C are separately falsifiable promotions.
The cited evidence motivates the mechanisms and their boundaries; every
numeric threshold, cap, loss weight, and deadline introduced by this ADR is a
preregistered bounded engineering choice unless a source is explicitly said to
have used that exact value.

## Runtime invariants and language

- The agent is self-contained. It may use neural latent rollouts, explicit
  search, factual replay, an internal posterior, and episode-local gradient
  updates. It may not use an LLM, generated code, an external solver or tool,
  hidden metadata, internet access, or an environment reset.
- Public ARC-AGI-3 levels remain strictly held out from training, calibration,
  threshold selection, trace distillation, and checkpoint selection.
- A **Factual Memory** entry is a real `(observation, action, outcome,
  observation)` transition. A prediction never becomes factual until the
  environment returns it.
- A **Raw Observation Identity** is an exact hash over the complete returned
  frame and observation fields. It is the mandatory graph key and the fallback
  whenever latent or mechanics identity is uncertain. A separate Mechanics
  State Identity may ignore a proven presentation/status field only after
  equal outgoing factual transition signatures establish that equivalence; it
  never replaces the raw identity for audit.
- A **Goal Hypothesis** is a generator-supported goal-family template plus
  arguments instantiated from observed frame features, a legacy 19-dimensional
  goal feature, an executable observation predicate, and posterior mass.
- A **probe** is a planned prefix intended to make two or more live goal
  hypotheses predict a terminal success at the same endpoint. A nonterminal
  real outcome is therefore evidence against all claiming hypotheses.
- **Jointly safe** means every prefix passes the calibrated failure/exhaustion,
  transition-correctness, reliability, disagreement, and inverse-action gates
  for every goal hypothesis covering the protected posterior mass. It does not
  mean the learned model has proved safety.
- Q keeps ADR 0003's meaning: it ranks predicted-transition correctness. It is
  never called reward, return, progress, utility, or value, and it never adds a
  positive action-preference term. Reliability is likewise a trust signal, not
  utility. The direction of its raw logit is taken from its calibration label,
  not assumed from the sigmoid sign.
- Search uses the spatial `16 x 16` patch-4 latent. It does not use a pooled
  latent-distance objective for coordinate-sensitive planning. DINO-WM's
  spatial-token ablation (arXiv:2411.04983) and the PLDM/V-JEPA planning line
  motivate this constraint; see
  [jepa-latent-mpc.md](../research/raw-2026-08-25-planning/jepa-latent-mpc.md).
- The planner executes one environment action, observes, records the exact
  result, and replans. A selected suffix may be retained as a proposal, but is
  never executed open-loop. This is the receding-horizon pattern used by
  V-JEPA-2-AC, DINO-WM, and PLDM (arXiv:2506.09985, 2411.04983, 2502.14819) and
  limits compounding model error.

The reactive-policy impossibility and ordinal-ranking results in
[foundation-improvements.md](../research/2026-08-24-foundation-improvements.md)
§3.1 define two additional boundaries: a frame-only reactive controller cannot
solve distinct hidden goals requiring distinct actions, while a genuine value
head need only preserve the correct action ordering with margin. Better pixels
or marginal latent regularization alone cannot supply either property.

## Phase A: rewire-only latent falsification

Phase A runs on an ADR-0003 EMA checkpoint without changing any model weight.
It uses the documented `encode_state`, batched `forward_from_latent`,
`prefix_predict`, event, Q, reliability, PTRM, inverse-action, and exact decoder
seams summarized in
[p2-inference-surface.md](../research/raw-2026-08-25-planning/p2-inference-surface.md).
The current Q and reliability heads are trust gates only. PTRM is a stochastic
noise-sensitivity probe over one shared model, not an epistemic ensemble and
not EIG.

### A1. Exact observed-state graph comes first

For each Raw Observation Identity `h`, the controller stores:

```text
node[h] = {
  exact observation,
  legal/advertised action set,
  tried action keys,
  edge[action] = {next_raw_hash, exact board effect, terminal channel,
                  action cost, pre-update model predictions},
}
```

The graph is append-only and factual. An imagined latent is never inserted.
Before a learned rollout is used, the planner performs exact retrodiction:

1. If `(h, action)` is a stored edge, use the stored successor observation and
   its freshly encoded latent instead of the predicted successor.
2. Run the model prediction in parallel for audit. If its exact decoded
   gameplay frame disagrees with the stored successor, mark that edge's model
   prediction untrusted and do not continue a learned suffix through it.
3. A decoded imagined gameplay frame that exactly matches a stored frame may
   prioritize a graph lookup, but it may not snap to the factual node: the
   decoder does not predict every observation field or hidden mechanics state.
   Only an exact replayed factual edge establishes a successor Raw Observation
   Identity. Approximate latent similarity never establishes identity.
4. If a known factual path reaches a graph node with untried actions, graph
   search supplies the exact shortest prefix to that frontier.

Thus the neural planner subsumes the cheap graph explorer rather than hiding
it. When all learned branches fail trust or safety, the fallback is: follow an
exact graph path to the nearest node with an untried action; at the frontier,
take one untried, jointly nonterminal action ranked by exact novelty and the
one-step model; never intentionally repeat the same action at the same raw
observation. This follows the strong observed-state-graph ARC-AGI-3 baseline
and the DeepCubeAI lesson that puzzle search requires re-identifiable states;
continuous latent hashing is not accepted
([grid-puzzle-latent-search.md](../research/raw-2026-08-25-planning/grid-puzzle-latent-search.md),
arXiv:2512.24156 and DeepCubeAI/DeepXube arXiv:2603.23873).

### A2. Live candidate-goal proposal

P1 received a public candidate list from its generator. Live games do not.
Phase A therefore creates a bounded, auditable hypothesis set using only
mechanisms already represented by the synthetic goal vocabulary:

1. From the gameplay region of every factual frame, build a deterministic
   feature inventory: palette/color counts; 4-connected components; component
   area, bounding box, centroid, border contact, containment and D4 relations;
   repeated motifs; and transition-derived created, deleted, moved, recolored,
   toggled, contacted, or preserved components. Status/progress fields are kept
   separately and never treated as board effects.
2. Instantiate every generator-supported goal-family template against that
   inventory. The archaeology names collect-all, switch-order, avoid-hazard,
   and preserve-resource among P1's six families; the remaining templates are
   taken from the same trained generator vocabulary rather than inferred from
   those names. A template may bind a salient color, component, location,
   count, relation, or observed interaction role. No new free-form semantic
   family is invented at decision time.
3. Each candidate stores `(family, arguments, g19, predicate)`. In Phase A,
   `g19` is exactly the legacy trained 19-dimensional family feature. If two
   instances map to the same `g19`, the neural event head cannot distinguish
   them; they are grouped for neural scoring while their executable predicates
   remain separate for exact observed/decoded-frame checks. This limitation is
   explicit rather than disguised as instance-conditioned inference.
4. Deduplicate candidates by normalized predicate and feature vector. Allocate
   prior mass equally across represented families and then equally across a
   family's instances. Keep at most `M = 32` concrete candidates: at least one
   per represented family, then the most salient transition-grounded
   instances. Pruned candidates remain in a dormant ledger and can return when
   new factual features support them.
5. Add an `unknown` hypothesis with prior mass `u_0 = 0.20`. It has no success
   value. It represents “the true goal is outside or aliased by this candidate
   set,” and prevents a normalized but misspecified concrete posterior from
   being mistaken for certainty.

The family restriction is a Phase-A compatibility boundary, not a claim that
public goals lie in the generator vocabulary. Candidate proposal is the main
unproved seam identified by
[p1-falsification-archaeology.md](../research/raw-2026-08-25-planning/p1-falsification-archaeology.md)
§d and by the hidden-goal review. Phase B trains a candidate-instance encoder;
until then, high unknown mass deliberately routes to exploration.

### A3. Post-hoc calibration and fail-closed trust gates

Calibration is an evaluation pass over fixed held-out synthetic episodes, not
model retraining. Raw event, Q, reliability, inverse-action, and PTRM scores are
binned by source family, horizon, changed/no-op status, and irreversible-risk
label. Every accepted bin carries a 95% confidence bound from the existing
risk-coverage machinery. The first foundation-v2 checkpoint is deployable for
Phase A only with the following mappings and thresholds:

- The three terminal event channels are post-hoc mapped to a mutually
  exclusive distribution over `satisfied`, `failed`, `exhausted`, and
  `ordinary`; noop remains a separate Bernoulli event. A satisfaction claim
  requires calibrated probability at least `0.80` **and** a lower 95%
  precision bound at least `0.90`.
- An ordinary imagined edge is provisionally retained only when the Q and
  reliability bins each have an upper 95% exact-transition error bound at most
  `0.10`. A plan becomes executable only after every wholly imagined edge in
  its finalist prefix also passes a PTRM-disagreement bin with the same bound;
  replayed factual edges are exact and exempt. The calibration adapter converts
  the documented reliability target into “probability trustworthy”;
  raw-logit direction is not assumed.
- A possibly irreversible edge is trusted only when the corresponding Q,
  reliability, and finalist PTRM upper error bounds are each at most `0.02`.
- For failure/exhausted detection, the accepted event bin must have an upper
  95% false-safe rate at most `0.01`. Along a prefix, the conservative union
  bound on calibrated failure plus exhausted probability must be at most
  `alpha_safe = 0.02` for every protected goal hypothesis. Whenever unknown
  mass exceeds `0.05`, the goal-dropout `g = 0` query is an additional protected
  safety particle; unknown goal mass is never omitted from the risk gate.
- On a predicted changed transition, the inverse-action head must recover the
  conditioned type and, for ACTION6, place the coordinate in a calibration bin
  whose upper action-error bound is at most `0.10`. No-effect transitions are
  exempt because ADR 0003 masks them from inverse-action training.
- PTRM uses `K_ptrm = 4` only on finalists. Its pairwise latent disagreement is
  calibrated against exact decoded error. It can reject or truncate a branch;
  it cannot create an exploration bonus in Phase A.

If no calibration bin meets a threshold, the planner does not relax the
threshold after seeing live results. It truncates before that edge. If even
one-step novel edges have no accepted coverage, Phase A reduces to the exact
graph frontier controller and reports the calibration failure. This converts
P1's exact safety oracle into a bounded false-safe contract rather than a raw
sigmoid threshold. The evidence is the existing calibration surface in
[p2-inference-surface.md](../research/raw-2026-08-25-planning/p2-inference-surface.md),
MACURA's state-dependent horizon truncation (arXiv:2405.19014), COPlanner's
conservative imagined-rollout uncertainty (arXiv:2310.07220), and ACID's
inverse-action verification (arXiv:2607.02403), summarized in
[verification-amortization.md](../research/raw-2026-08-25-planning/verification-amortization.md).

### A4. Soft belief update and falsification kernel

Let concrete goal weights be `w_i`, summing to `1 - u`, and let
`C_event(g_i, prediction)` return the calibrated categorical terminal
distribution. After each real transition, the environment supplies an exact
observed channel `o` from `{satisfied, failed, exhausted, ordinary}`; exact
frame equality supplies noop/non-noop. Let `L_i` be the calibrated probability
of the observed terminal channel, multiplied by the calibrated noop likelihood
when noop is informative. Clip `L_i` to `[epsilon, 1 - epsilon]` with
`epsilon = 0.02`.

The update is tempered by `eta in [0,1]`, the previous edge's calibrated trust
coverage. An untrusted model prediction contributes no goal evidence. Concrete
hypotheses are never assigned zero mass:

```text
for each concrete goal i:
    raw_i = max(1e-4, w_i * L_i ** eta)
normalize raw_i over concrete goals

mixture_likelihood = sum_i (w_i / (1 - u)) * L_i
surprise = -log(max(epsilon, mixture_likelihood))
logit(u) = clamp(logit(u) + eta * (surprise - tau_unknown),
                 logit(0.05), logit(0.95))
w_i = (1 - u) * raw_i / sum_j raw_j
```

`tau_unknown` is fixed before deployment as the 95th percentile of mixture
surprise on held-out episodes where the true generator goal is present in the
candidate set. A candidate below `0.005` may be omitted from search compute but
stays in the ledger; the protected live set is the smallest descending set
covering at least 95% of concrete mass, plus every candidate with `w_i >=
0.02`. Exact official success ends the level; no learned mismatch causes hard
deletion. This is the P1 falsification update with a calibrated likelihood and
floor replacing exact Boolean deletion.

Probe choice remains falsification-only; Phase A does not add a progress or
latent-distance reward. For a safe plan `pi` of length `h`, candidate `i`
**claims** the endpoint only when both its executable predicate holds on the
decoded endpoint and its calibrated satisfaction lower bound meets the claim
threshold. A multi-goal probe needs at least two protected candidates to claim.
Its primary score is claimed posterior mass:

```text
claim_mass(pi) = sum_i w_i * LCB95(P(satisfied | g_i, pi_h))
```

Plans are ordered lexicographically by: larger `claim_mass`; more claiming
candidates; shorter prefix; lower summed noop probability; fewer graph repeats;
then deterministic action-key order. If no multi-goal probe exists, choose the
shortest safe single-goal probe. If none exists, use the exact graph-frontier
fallback. This preserves the winning P1 primitive and rejects the routers and
cost-aware variants that added complexity without a supported gain.

The complete decision kernel is:

```text
observe(real_transition):
    append the exact edge to Factual Memory and the observed-state graph
    compare the pre-update prediction with the exact successor (retrodiction)
    update the soft goal posterior from the exact terminal/noop channel
    regenerate template instances if new factual features appeared

decide(current_observation):
    z0 = encode_state(current_observation) once
    if an exact graph path reaches a useful untried frontier, retain it as a probe
    roots = recursively imagine every legal atomic action in one logical batch
    reject roots failing event, Q, reliability, or exact-replay gates
    build a short beam from roots; replay factual prefixes exactly
    reject a prefix immediately when any protected goal violates joint safety,
        inverse-action consistency, reliability, or adaptive horizon trust
    PTRM-check the final shortlist
    choose the highest-ranked safe multi-goal falsification probe
    else choose the shortest safe single-goal probe
    else choose the graph-frontier/unknown-goal exploration fallback
    execute only the first action; observe and replan
```

P1's evidence is bounded but material: falsification-only achieved `0.9771`
success, zero terminal failures, and `0.4629` efficiency versus `0.8750`,
`0.0542`, and `0.1205` for sequential discrimination on P1C-hard. On the
ordinary routing study its success advantage over the strongest baseline did
not pass the `+0.05` gate, while its efficiency was best. The design therefore
uses falsification under broad uncertainty but does not claim universal
superiority
([p1-falsification-archaeology.md](../research/raw-2026-08-25-planning/p1-falsification-archaeology.md)
§b).

### A5. Probe generation and compute contract

The legal atomic set contains the six non-coordinate action types plus every
ACTION6 coordinate, at most `6 + 64^2 = 4,102` actions. Phase A performs one
logical, exhaustive, recursively imagined depth-1 sweep. It may microbatch for
memory but may not proposal-prune this root. Dynamics are goal-independent, so
each action is rolled once; only the small event head is evaluated against up
to `M = 32` goal vectors.

The deeper beam is bounded and funnelled:

- maximum imagined depth `H_A = 4`, with MACURA-style earlier truncation at
  the first untrusted edge;
- beam width `B = 16`;
- per-node candidate pool `P = 64`, formed from all simple actions, top root
  actions per protected goal, graph-frontier continuations, and ACTION6 sites
  extracted from the decoded imagined frame;
- `prefix_predict` screens all `P` candidates at depths 2-4;
- only the best `b = 8` per beam node receive full recursive
  `forward_from_latent` evaluation, exact decode when needed, and all trust
  gates;
- the best `R = 8` complete probes receive `K_ptrm = 4` PTRM trajectories.

At the caps, one decision costs:

```text
1 state encode
4,102 full recursive depth-1 transition evaluations
3 * 16 * 64 = 3,072 cheap prefix-predict screens
3 * 16 * 8  =   384 full recursive deeper-edge evaluations
8 * 4 * 4   =   128 PTRM member-step evaluations
-------------------------------------------------------
4,614 full/PTRM member-steps + 3,072 prefix screens
<= (4,486 full endpoints + 3,072 prefix endpoints + 128 PTRM endpoints)
   * 32 = 245,952 small candidate-event head evaluations
```

Exact graph replay reduces this total. Decoding is limited to the shortlisted
roots, beam survivors, and hash/retrodiction checks rather than all 4,102
root latents.

The cost must be measured on the actual patch-4 checkpoint. At `C = 128`, one
two-convolution 3x3 residual block over a `16 x 16` grid is approximately
`0.151 GFLOP`; a recursive step using `N_block = outer_steps *
(inner_steps + 1)` block applications is approximately `0.151 * N_block
GFLOP` before heads and memory traffic. Thus the exhaustive root is not called
“free.” The A40 acceptance smoke must measure encode, root, beam, PTRM,
decoder, peak memory, and synchronization separately.

The target deadline is 1.0 second per environment action and the hard deadline
is 2.0 seconds. After the mandatory root sweep, the controller completes only
whole beam/PTRM batches that fit the measured remaining deadline and returns
the best result from the deepest completed layer. It reduces depth before
beam width and never weakens safety thresholds. If the exhaustive root plus
minimum calibration gates cannot complete within 2.0 seconds on the exact
launch binary, Phase A as specified is blocked pending an implementation
optimization; it is not silently replaced by the old greedy policy. Internal
model calls are reported but are not charged as RHAE environment actions.

## Phase B: trained proposal, progress, epistemic EIG, and Gumbel search

Phase B keeps Phase A's candidate posterior, factual graph, retrodiction,
joint-safety gates, one-action execution, and unknown-goal fallback. It changes
how candidate plans are proposed and ranked. It is enabled only after the
head-only fine-tune and Phase-B evaluation gates pass.

### B1. Minimal new modules

1. **Candidate-instance encoder.** A small encoder maps `(legacy family,
   template arguments, observed feature summary)` back into a 19-dimensional
   vector, preserving the existing `goal_dim = 19` model API. It is initialized
   to reproduce the legacy family feature. The existing event head is
   fine-tuned on positive and counterfactual candidate instances so candidates
   within one family can become distinguishable. Phase A does not depend on
   this module.
2. **Spatial action proposal.** From the spatial Consumer Latent, posterior
   mean goal embedding, posterior entropy, unknown mass, remaining action
   budget, and graph tried-action mask, emit seven action-type logits and one
   `64 x 64` ACTION6 coordinate map. The flat prior is factorized as
   `P(type) * P(coord | ACTION6)` and mixed with `0.05` uniform legal-action
   mass at inference. The spatial head is the standard AlphaZero solution to a
   factored coordinate action space; see
   [discrete-action-search.md](../research/raw-2026-08-25-planning/discrete-action-search.md)
   §4.
3. **Goal-conditioned progress value.** A scalar `V(z, g, belief, budget) in
   [-1, 1]` predicts synthetic episode terminal return (`+1` satisfied, `-1`
   failed/exhausted, `0` truncated), not transition correctness. Predicted
   progress of a plan is `V(z_h, g) - V(z_0, g)`. Pairwise branch ranking is
   trained explicitly because the Lean `Greedy` result requires ordinal margin,
   not exact numeric value
   ([foundation-improvements.md](../research/2026-08-24-foundation-improvements.md)
   §3.1).
4. **K = 4 lightweight epistemic members.** Freeze the EMA encoder, base
   dynamics, decoder, and existing Q/reliability heads. Each member has an
   independently initialized, rank-16 residual adapter on predicted spatial
   latents and its own progress/event readout, trained on an independent
   episode bootstrap including open-loop fragments of length 1-4. There is no
   forced-diversity loss. Member predictions are passed through the shared
   exact decoder for calibration. This is an epistemic approximation only
   after out-of-bootstrap disagreement predicts held-out error. PTRM noise is
   not accepted as a substitute because perturbing one shared model measures
   local sensitivity, not parameter/posterior uncertainty.

The ensemble choice follows Plan2Explore/PTS-BE disagreement planning
(arXiv:2005.05960, 2507.02639), reward-head disagreement under hidden utility
(DreamerV3-XP, arXiv:2510.21418), and the explicit conclusion in
[verification-amortization.md](../research/raw-2026-08-25-planning/verification-amortization.md)
that small independently bootstrapped ensembles remain better evidenced than
JEPA-native uncertainty. Its use remains conditional on calibration; agreement
among shared-trunk heads is not treated as independent proof.

### B2. EIG and action score

For candidate goal `i`, ensemble member `k`, and imagined prefix endpoint `j`,
let `p_ikj(o)` be the calibrated distribution over the four terminal channels,
and normalize concrete mass as `wbar_i = w_i / (1 - u)`. Treat `(goal i,
member k)` as particles with prior `wbar_i / K`. For the unknown particle, let
`p_k0(o)` be the member's one-step goal-dropout (`g = 0`) event distribution.
The normalized disagreement/EIG proxy is the Jensen-Shannon information:

```text
I_j = H(sum_i,k (wbar_i / K) p_ikj)
      - sum_i,k (wbar_i / K) H(p_ikj)

D_unknown = H(sum_k p_k0 / K) - sum_k H(p_k0) / K

EIG(pi) = min(1,
              ((1 - u) * sum_j I_j + u * D_unknown) / (h * log(4)))
```

In a deterministic game, disagreement about the next observed channel is
epistemic once observation aliasing and calibration error are controlled. This
quantity mixes information about the live goal particles and the bootstrapped
rule/model particles. Long-horizon member disagreement still triggers the
hard trust gate; only calibrated, accepted disagreement contributes a bonus.
The unknown particle contributes only one-step goal-dropout outcome
disagreement because it has no concrete multi-step success predicate.

For a safe plan `pi` of length `h`, define:

```text
Progress(pi) = sum_i w_i * mean_k[
                 V(z_h^k, g_i) - V(z_0, g_i)]

Risk(pi) = max over protected concrete goals and the g=0 unknown particle of
           sum_j UCB95[
             P_k(failed at j | g) + P_k(exhausted at j | g)]

ActionCost(pi) = (h + sum_j P(noop_j) + 2 * repeated_graph_edges) / H_B

beta_t = min(0.50,
             0.25 * ((1 - u) * H(wbar) / log(max(2, M)) + u))
mu_t   = 0.25 * min(4,
                    initial_action_budget / max(1, remaining_action_budget))

Score(pi) = Progress(pi)
            + beta_t * EIG(pi)
            - 4.0 * Risk(pi)
            - mu_t * ActionCost(pi)
```

`Risk(pi) <= alpha_safe = 0.02` remains a hard admission condition; the risk
term ranks admitted plans and does not buy permission to cross the threshold.
Posterior entropy and unknown mass increase the information weight; posterior
concentration shifts the same controller toward progress. `ActionCost`
penalizes long probes, predicted noops, and factual repeats, while its weight
increases as the remaining environment budget shrinks. These constants are
preregistered engineering choices, not claimed paper optima, and may be changed
only on the synthetic development split before a fresh held-out confirmation.

This scalarization is the feasible EFE/MaxInfoRL synthesis in
[hidden-goal-falsification-eig.md](../research/raw-2026-08-25-planning/hidden-goal-falsification-eig.md)
(arXiv:2504.14898, 2606.20658, 2412.12098): posterior-weighted extrinsic
progress plus epistemic information, with explicit risk and action economy.
Unlike the removed confidence blend, every positive term now refers to either
goal progress or information expected to reduce goal/rule uncertainty.

### B3. Gumbel-Top-k plus Sequential Halving

At each real decision, evaluate the proposal logits over the full legal action
set, sample `m = 16` root actions without replacement by Gumbel-Top-k, and run
four Sequential Halving rounds with `n = 64` total root-to-leaf simulations:

| round | surviving roots | new simulations per root | round simulations |
|---:|---:|---:|---:|
| 1 | 16 | 1 | 16 |
| 2 | 8 | 2 | 16 |
| 3 | 4 | 4 | 16 |
| 4 | 2 | 8 | 16 |

Each simulation has maximum depth `H_B = 4`, proposes child actions from the
same factorized prior, replays exact graph edges, and truncates at the first
uncalibrated or high-disagreement state. The rank statistic is the mean
`Score(pi)` over completed simulations. The final root uses the Gumbel
completed score `g(a) + log pi(a) + sigma(mean Score(a))`; Q is absent from
this expression. Expansion for each halving round is batched in the TransZero
style (arXiv:2509.11233).

The published operating point is deliberately conservative: EfficientZero V2
uses 8 sampled actions and 16 simulations at roughly this model scale
(arXiv:2403.00564), while Gumbel planning retains its advantage at very small
budgets (OpenReview `bERaNdoegnO`; MiniZero arXiv:2310.11305). The improvement
guarantee from Gumbel planning assumes correct value estimates; this ADR does
not transfer that guarantee to Tofy. The new progress head and closed-loop
evaluation must establish utility empirically.

The conservative maximum compute is `n * H_B * K = 64 * 4 * 4 = 1,024`
ensemble transition-member evaluations, plus one proposal forward, candidate
event/progress heads, exact replay, and selected decodes. The shared-base
implementation should be cheaper, but accounting uses the conservative number.
The same 1.0-second target and 2.0-second hard deadline apply. The controller
never lowers `m`, `n`, or a safety threshold after observing held-out results;
at runtime it truncates uncertain/deadline-exceeding simulations and completes
the current halving round before selecting from the deepest completed round.

## Phase C: prequential adaptation and belief recurrence

Phase C is episode-local and resettable. The base EMA checkpoint, encoder,
decoder, proposal, value, Q, and reliability weights are immutable during live
play. Frozen and adapted results are reported separately.

### C1. Belief-state recurrence

A small GRU consumes, after every factual action, the canonical observed
latent, action embedding and coordinate, exact board-effect summary, exact
terminal/noop channel, prediction residual, PTRM/K-ensemble disagreement, and
remaining action budget. Its state conditions the proposal and progress heads
and produces a bounded log-likelihood correction for each explicit goal
hypothesis.

The explicit posterior remains authoritative. The GRU correction is clipped to
`[-log(2), log(2)]` per transition before entering the soft Bayes update; it
cannot delete a candidate, write Factual Memory, override a terminal channel,
or establish state identity. It is trained on synthetic hidden-goal histories
with exact goal labels, shuffled-history controls, goal dropout, and a KL loss
to the exact generator posterior. This preserves an auditable belief seam
rather than replacing it with an opaque recurrent policy, consistent with the
reactive-policy impossibility in
[foundation-improvements.md](../research/2026-08-24-foundation-improvements.md)
§3.1 and the VariBAD/MAMBA boundary summarized in
[hidden-goal-falsification-eig.md](../research/raw-2026-08-25-planning/hidden-goal-falsification-eig.md).

### C2. Fast adapter

The only live trainable parameters are zero-initialized, per-game rank-4
adapters on the action-FiLM path and final predicted-latent residual, plus
scalar event/reliability calibration temperatures when their labels are
observed. The encoder, base recurrence, exact decoder, global candidate
encoder, proposal, progress value, Q, and reliability weights remain frozen.

For each newly observed transition:

1. Record its pre-update prediction and exact result first.
2. Snapshot the episode adapters.
3. Take exactly three Adam steps at learning rate `1e-3`, gradient norm clipped
   to `0.1`, over at most 64 replayed factual transitions, always including
   the newest transition. The adapter loss mirrors the applicable frozen
   foundation targets:

   ```text
   L_adapt = 1.0 L_pred_ce + 0.5 L_gate + 0.25 L_latent
             + 0.1 L_invact + 0.1 L_observed_event
   ```

4. Accept the update only if newest-transition CE improves, prior-transition
   mean CE increases by no more than 1%, no previously exact decoded
   transition becomes inexact, inverse-action accuracy does not regress, and
   event false-safe calibration does not cross its registered bound.
5. Otherwise restore the snapshot, halve the learning rate, and retry once.
   A second rejection disables parametric adaptation for the rest of the game;
   factual graph and posterior updates continue.

Adapters are reset at the official game boundary and never saved into the base
checkpoint. This is a prequential test: the transition is scored before it can
train the adapter. The trust region and revert-on-regression rule address the
planner/prior OOD failure in TD-M(PC)^2 (arXiv:2502.03550), while exact replay
uses the deterministic retrodiction advantage described in
[verification-amortization.md](../research/raw-2026-08-25-planning/verification-amortization.md).

### C3. When full amortization is safe

A GC-IDM-style direct goal-conditioned inverse policy may bypass explicit
search for one action only when all of the following hold in the current game:

- concrete posterior maximum at least `0.90`, posterior entropy at most
  `0.20 * log(M)`, and unknown mass at most `0.05`;
- the last eight factual transitions and every stored transition from the
  current raw state component replay with exact decoded parity after
  adaptation;
- proposal recall of the Phase-B search winner is at least `0.95` over the last
  16 replanning points, and direct/search top actions agree with an ordinal
  score margin larger than twice the calibrated value error;
- inverse-action, event false-safe, ensemble risk-coverage, and graph-repeat
  gates all pass; and
- the direct action is not an untried irreversible edge.

Any failed condition immediately restores Phase-B search. Even in amortized
mode the real transition is observed and checked after one action. This is the
narrow in-distribution regime in which GC-IDM's 1.5M-parameter inverse model
matched or exceeded CEM (arXiv:2605.08732); it is not assumed at first contact
with a hidden ARC-AGI-3 game. BMPC-style distillation (arXiv:2503.18871) occurs
offline on synthetic Phase-A/Phase-B traces, never by permanently learning
from held-out public levels.

## Proposed ADR-0003 amendment: training deltas

These additions are **not** prerequisites or abort gates for the current
foundation-v2 run. Phase A consumes its first qualifying EMA checkpoint as-is.
After that run is sealed, Phase B uses a separate 2,048-update head/adaptor
fine-tune from the selected EMA checkpoint. The encoder, base dynamics, exact
decoder, copy gate, existing event/Q/reliability heads, EMA bundle, and every
ADR-0003 loss and weight remain unchanged unless explicitly named below.

New heads consume stop-gradient EMA latents. They use a separate AdamW optimizer
and clip group, so their gradients cannot perturb the foundation world core or
the EP gradient-budget controller. Training data remain synthetic: the
ADR-0003 mixture, complete branch groups, sequential fragments of length at
most four, and planner traces generated offline from those episodes. No public
ARC-AGI-3 frame or action enters the fine-tune.

Add the following normalized losses to the later fine-tune:

| loss | weight | target and scope |
|---|---:|---|
| `L_candidate_event` | `0.10` | Four-way event CE plus noop BCE for proposed positive/counterfactual candidate instances; family-balanced; trains the candidate encoder and candidate-facing event readouts. |
| `L_proposal` | `0.20` | Cross-entropy/KL to the improved root distribution from exact-simulator Phase-A traces and Phase-B lazy reanalysis; factorized type and ACTION6 coordinate terms. |
| `L_progress` | `0.20` | Huber loss (`delta = 1`) from candidate-conditioned value to synthetic episode terminal return. |
| `L_progress_rank` | `0.10` | Pairwise logistic ranking on same-state factual branches using return advantage; margin `0.10`; outcome-equivalent branches excluded. |
| `L_ensemble` | `0.05` | Mean member bootstrap loss: normalized predicted-pixel CE plus `0.25` latent Huber over horizons 1-4; updates only rank-16 member adapters/readouts. |
| `L_proposal_ood` | `0.02` | `KL(U_legal || pi)` on 20% candidate-shuffled, goal-dropped, geometry/operator-augmented synthetic states; prevents a confident proposal outside candidate support. |
| `L_belief` | `0.05` | Phase-C-only KL from recurrent posterior to exact synthetic hidden-goal posterior, with shuffled-history and zero-history controls. |

The proposal target is mixed with 5% uniform legal-action mass. Phase-A traces
are generated with the hidden index concealed, exactly as at inference; the
exact simulator supplies outcomes and episode returns, not privileged inputs
to the policy. Lazy reanalysis and planner-to-prior distillation follow BMPC
(arXiv:2503.18871). OOD entropy regularization and inference-time prior
trust-regioning address the planner/policy distribution mismatch documented by
TD-M(PC)^2 (arXiv:2502.03550). Ensemble members use independent episode
bootstraps and random initialization; no loss is allowed to manufacture
disagreement.

The fine-tune checkpoint is selected by the synthetic closed-loop planability
suite below, not `L_progress`, prediction CE, or open-loop latent MSE alone.

## Evaluation and promotion contract

### Planner-aware benchmark

Create a synthetic hidden-goal suite from the v5 generator with the same
geometry/operator held-outs as ADR 0003 and three explicit strata:

1. **Ordinary hidden-goal:** the six P1 goal families, live candidate proposal,
   concealed hidden index, matched action budgets, and generator-supported
   exact oracle plans for regret/efficiency measurement.
2. **P1C-hard retargeting:** reproduce the adversarial construction in which a
   sequential controller commits to the wrong target at least three times.
   First reproduce seeds 201-208 for archaeology parity, then make all
   promotion decisions on untouched confirmation seeds 1201-1208.
3. **Misspecification and aliasing:** omit the true candidate on a preregistered
   10% of episodes; include same-looking/history-distinguishable states,
   translated and held-out operator families, irreversible hazards, and
   revisited exact frames. This tests unknown-goal exploration, recurrence,
   raw-frame hash fallback, and retrodiction rather than silently assuming
   candidate recall.

Public ARC-AGI-3 is a final held-out report only. No threshold, weight,
checkpoint, ablation choice, or phase promotion may be selected from it.

### Metrics

Prediction metrics remain diagnostics, but the primary metrics are closed-loop:

- level/episode success, terminal-failure and exhaustion rate, return, actions
  per attempted and solved episode, and oracle-normalized action efficiency;
- an RHAE-aware action ledger partitioning real actions into decisive probes,
  progress actions, exact graph replays, predicted/observed noops, repeated
  state-actions, fallback novelty actions, and terminal mistakes;
- information gain and posterior entropy reduction per **environment action**,
  time/actions to true-goal posterior `>= 0.90`, posterior NLL/Brier score,
  candidate recall, unknown-mass AUROC on true-goal-omitted episodes, and false
  hypothesis retirement/reactivation;
- closed-loop chosen-action regret against the exact simulator, plan endpoint
  success calibration, safe-prefix precision/recall, irreversible false-safe
  rate, ordinal value margin, proposal recall@16, and search/direct-policy
  agreement;
- planner-distribution one-step exactness, horizon-1/2/4 decoded fidelity,
  open-loop versus re-anchored error, Q/reliability/disagreement risk-coverage,
  inverse-action consistency, exact retrodiction parity, and graph frontier
  coverage;
- wall time p50/p95/max, full recursive steps, prefix screens, ensemble member
  steps, decodes, peak A40 memory, and deadline truncations, reported separately
  from environment actions.

Checkpoint ordering is by closed-loop success, terminal safety, then
action-efficiency under the fixed planner. Prediction loss and multi-step RMSE
may improve after closed-loop planning performance collapses; the LunarLander
RSSM result arXiv:2607.01736 in
[discrete-action-search.md](../research/raw-2026-08-25-planning/discrete-action-search.md)
§5 is the direct warning. Every saved foundation/fine-tune bundle therefore
receives the same frozen planner-aware evaluation.

### Required ablations

Use identical checkpoints, episodes, seeds, candidate proposal, action budgets,
and, for search controllers, registered model-call/deadline budgets. Report the
following grid, with factors disabled only where structurally inapplicable:

- controller: current greedy confidence blend; exact graph-only; Phase A;
  Phase B;
- falsification: on versus off (off ranks only progress/novelty under the same
  safety gates);
- EIG: on versus off in Phase B;
- maximum search depth: 1, 2, and 4;
- candidate diagnostic: live proposal versus oracle candidate inclusion (the
  oracle row diagnoses proposal recall and is never a deployable result);
- adaptation: frozen Phase B versus Phase C belief-only versus Phase C
  belief-plus-fast-adapter.

The three headline comparisons are greedy versus Phase A versus Phase B.
Graph-only is mandatory because the latent planner must justify its added
model risk. Falsification, EIG, and depth are crossed with both ordinary and
P1C-hard strata, not selected after observing one family.

Promotion thresholds are decisions, not predicted results:

- **Phase A over greedy:** lower 95% paired confidence bound on P1C-hard success
  improvement is above zero; terminal-failure upper bound does not increase;
  actions per solve improve by at least 10%; and Phase A is not worse than
  graph-only on the ordinary stratum's success lower bound.
- **Phase B over Phase A:** lower 95% paired confidence bound on
  oracle-normalized action efficiency is above zero with no success or
  terminal-safety regression; proposal recall@16 is at least `0.90`; and
  ensemble disagreement has monotone held-out risk-coverage before EIG is
  enabled.
- **Phase C over frozen Phase B:** adapted closed-loop efficiency improves with
  no exact-retrodiction, safety, or success regression. Otherwise adapters are
  rejected and belief/graph memory remain.

All promotion claims require multiple seeds and fresh held-out confirmation.
The P1C-hard historical result is prior evidence, not a Phase-A result.

## Risks and mitigations

| risk | consequence | mandatory mitigation |
|---|---|---|
| Learned-model exploitation | Search finds latent plans that score well only because the model is wrong; the large policy set makes some exploitation generic (arXiv:2605.15960). | Exact factual replay, prior trust region, Q/reliability gates, calibrated K-head disagreement, ACID inverse-action filtering, adaptive horizon truncation, depth cap 4, one-action replanning, and planner-distribution evaluation. |
| Event-head miscalibration | A false-safe prefix causes irreversible failure or the true goal loses posterior mass. | Post-hoc source/horizon calibration, 95% false-safe bounds, soft likelihood floors, protected posterior mass, hard risk threshold, and fail-closed graph fallback. No live threshold relaxation. |
| Candidate-goal proposal miss or `g19` alias | A confident posterior excludes the true goal or cannot distinguish same-family targets. | Explicit unknown mass, surprise test, dormant candidates, transition-triggered template regeneration, Phase-B instance encoder, candidate-omission benchmark, and novelty/EIG fallback. When `u` is high, set concrete progress to zero and choose a one-step untried graph-frontier action maximizing calibrated outcome disagreement/change under the same safety gates. |
| Latent aliasing and drift | Search merges distinct states or fails to recognize a revisit. | Never use continuous latent equality or a decoded-frame match as identity. Only an exact factual observation/edge supplies Raw Observation Identity; decoded equality may prioritize lookup, imagined nodes remain disposable, and mechanics aliases require factual transition-equivalence proof. |
| Compute or memory blow-up | Exhaustive coordinate roots, candidate heads, depth, or ensemble exceed the 0.1-2 s A40 envelope. | Fixed `M/B/P/b/R/K/m/n/H` caps, logical batching with bounded microbatches, actual patch-4 smoke timing, decoder shortlist, prefix screening, TransZero-style batched rounds, deadline truncation by depth, and a hard activation blocker above 2 s. |
| Proposal collapse/OOD overconfidence | Gumbel search never samples the useful coordinate. | Five-percent uniform prior mixture, OOD entropy loss, proposal recall@16 gate, exact graph frontier injection, and Phase-A/oracle-candidate diagnostics. |
| Fast-adapter overfit or public-eval contamination | Recent transitions improve while earlier mechanics or planability regress; held-out knowledge leaks into persistent weights. | Frozen base, tiny resettable modules, pre-update scoring, exact replay acceptance, one retry then disable, separate frozen/adapted reports, and no persistence or offline training on public levels. |
| False confidence from prediction metrics | A later checkpoint has lower CE/MSE but worse decisions. | Closed-loop planability selects checkpoints; prediction, calibration, semantic fidelity, action use, posterior quality, safety, and control remain independent gates (arXiv:2607.01736). |

## Consequences and non-goals

- Decision-time compute becomes an intentional capability axis. RHAE charges
  environment actions, not internal model calls, so the planner spends A40
  compute to avoid waste while reporting latency and energy honestly.
- Factual Memory, belief, and imagination become separate state classes. This
  makes retrodiction, rollback, and failure attribution possible but adds an
  explicit session/controller module around the world model.
- Phase A can fail closed to a graph explorer even with a weak first
  checkpoint. Phase B cannot be promoted merely because its proposal/value
  losses fall, and Phase C cannot be promoted merely because online CE falls.
- The design does not claim a globally optimal planner, prove ARC-AGI-3
  performance, infer arbitrary natural-language goals, synthesize executable
  rules, or solve observation aliasing from a single frame.
- The design does not use CEM/MPPI, unrestricted MCTS, a program synthesizer,
  public-level fine-tuning, open-loop environment commitments, latent state
  hashing, Q-as-value, or PTRM-as-EIG. CEM/MPPI validate latent MPC in
  continuous control (arXiv:2506.09985, 2411.04983, 2502.14819), but the chosen
  discrete contract is Gumbel-Top-k plus Sequential Halving because
  `|A| >> n` and the action space is factored.
- Horizons beyond four, full-model live fine-tuning, and permanent replacement
  of explicit search by an amortized policy require a successor ADR.

## Evidence index

- [ADR 0003](0003-world-core-v5-foundation-v2.md) defines the patch-4 spatial
  latent, FiLM action conditioning, predicted-state decoder and copy gate, EMA
  weights, branch-group data, event/Q/reliability heads, goal dropout, run
  gates, and the unchanged foundation-v2 objective.
- [p2-inference-surface.md](../research/raw-2026-08-25-planning/p2-inference-surface.md)
  defines the exact latent-in/latent-out batched API, event slots, Q's
  prediction-correctness meaning, reliability/calibration seams, PTRM noise
  ensemble and pairwise disagreement, exact decoder, and absence of a current
  planner or belief state.
- [p1-falsification-archaeology.md](../research/raw-2026-08-25-planning/p1-falsification-archaeology.md)
  reconstructs the exact belief/prune/probe/safety/commitment kernel, reports
  ordinary and P1C-hard results, and maps the transition/event/trust seams to
  P2. Its negative router and cost-aware results justify one falsification
  spine rather than a menu.
- [jepa-latent-mpc.md](../research/raw-2026-08-25-planning/jepa-latent-mpc.md)
  supplies the spatial-latent, short-rollout, receding-horizon evidence from
  V-JEPA-2-AC (arXiv:2506.09985), DINO-WM (2411.04983), and PLDM (2502.14819),
  including PLDM's roughly 2.2M-parameter, approximately 67 ms/action result.
- [discrete-action-search.md](../research/raw-2026-08-25-planning/discrete-action-search.md)
  supplies Gumbel-Top-k/Sequential Halving (OpenReview `bERaNdoegnO`),
  EfficientZero V2's 16-simulation/8-sample point (arXiv:2403.00564), the
  AlphaZero spatial head, TransZero batching (2509.11233), exhaustive depth-1
  action scoring, and the 2607.01736 planner-aware checkpoint warning.
- [hidden-goal-falsification-eig.md](../research/raw-2026-08-25-planning/hidden-goal-falsification-eig.md)
  identifies falsification with BOED/EIG; motivates explicit posterior
  particles, hypothesis likelihood updates, ensemble disagreement, posterior
  value mixing, proposal amortization, and unknown-goal exploration via DAD
  (2103.02438), PTS-BE (2507.02639), MaxInfoRL (2412.12098), Plan2Explore
  (2005.05960), and the ARC-AGI-3 exploration evidence.
- [verification-amortization.md](../research/raw-2026-08-25-planning/verification-amortization.md)
  supplies generic model-exploitation risk (arXiv:2605.15960), uncertainty
  penalties and adaptive horizons (2310.07220, 2405.19014), ACID
  cycle-consistency (2607.02403), exact deterministic retrodiction, BMPC
  distillation (2503.18871), TD-M(PC)^2 prior trust regions (2502.03550), and
  the GC-IDM full-amortization boundary (2605.08732).
- [grid-puzzle-latent-search.md](../research/raw-2026-08-25-planning/grid-puzzle-latent-search.md)
  supplies Thinker's learned-model search gain (arXiv:2307.14993),
  DeepCubeAI's discrete/re-identifiable latent requirement, the strong cheap
  observed-state-graph baseline, and the unresolved hidden-goal seam.
- [foundation-improvements.md](../research/2026-08-24-foundation-improvements.md)
  §2.5 diagnoses the current confidence blend as structurally goal-free; §3.1
  records the Lean-checked reactive-policy impossibility, greedy ordinal-margin
  sufficiency, and marginal-regularizer blindness to action independence.
