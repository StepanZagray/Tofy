The coordinator confirms the Explore child's P2 summary (archived at docs/research/raw-2026-08-25-planning/p2-inference-surface.md): `forward_from_latent` supports recursive latent imagination; a PTRM stochastic ensemble with disagreement metric exists but is diagnostic-only; the Q head ranks prediction correctness (not value); goal-conditioning features exist but live play passes zeros; the event head carries noop/satisfied/failed/exhausted predicates. Delivering the final report now, citing the clean `c2ed65dc` state throughout.

---

# Planning by falsification: reconstruction, evidence, removal, and P2/v5 compatibility

## (a) The reconstructed algorithm

**Where it lives:** branch `p1`, file `src/agents.rs` (2,500 lines; deleted from main in `55d7281b`). Read via `git show p1:src/agents.rs`. Supporting pieces: `p1:src/domain.rs` (Scenario with `candidate_goals: Vec<Goal>` + hidden `hidden_goal_index`, line 110–120), `p1:src/generator.rs` (`build_candidates`, six goal families, line 358+), `p1:src/experiment.rs` / `p1:src/report.rs` (bootstrap gating harness).

**Setting.** P1 was an exact-simulator feasibility study: the agent sees a public candidate-goal set but never the hidden index. "Planning by falsification" is the P1C agent `set_aware_parallel_planning` with `RoutingStrategy::FalsifyOnly` (`agents.rs:398,415`).

**Core loop** — `run_exact_live_controller` (`agents.rs:462`):

1. **Belief = exact-live set.** A uniform posterior over candidate indices, `alive: BTreeSet<usize>`. Pruned two ways each step:
   - `prune_impossible` (`agents.rs:1139`): drop candidates whose public viability predicate (`goal_viable_at`) fails at the current state (irreversible loss).
   - `exact_live_plans` (`agents.rs:927`): run exact `shortest_path` per live candidate under the action budget; unreachable candidates are dropped; survivors carry their full plan (`LiveGoalPlan { index, actions }`).

2. **Belief update from observation** — `update_belief_from_outcome` (`agents.rs:900`), the falsification kernel:

```rust
/// Retain exactly the candidates whose public predicates predict the observed
/// terminal channel. Positive success/failure evidence narrows the set too.
let matches_observation = if out.success { expected_success }
    else if out.failed { expected_failure }
    else { !expected_success && !expected_failure };
if !matches_observation { falsified.push(i); }
```

The load-bearing branch is the third one: a **nonterminal** observation falsifies every candidate that predicted terminal success at this state. That is what makes one probe eliminate several hypotheses.

3. **Probe selection** — `choose_falsification_probe` (`agents.rs:971`): among live plans, pick a plan whose endpoint makes **≥ 2 live goals predict success**, subject to a safety check, `simulate_probe_endpoint_successes` (`agents.rs:1063`): simulate the whole plan in the exact model; reject it if any prefix state triggers `goal_terminal_failure` or viability loss **for any live candidate** (jointly safe). Score = ProbeScore ordering (`probe_better`, `agents.rs:1042`): maximize predicted endpoint-successes, tie-break shorter plan, then lower index.

4. **Commitment.** The chosen probe becomes a `ProbeCommitment` (`agents.rs:388`) executed open-loop, aborted early if the target is falsified mid-flight, satisfied, or leaves the live set (loop body at `agents.rs:481–566`). Reaching the endpoint yields a decisive observation either way: terminal success identifies/ends, nonterminal kills all candidates that predicted success there.

5. **Fallbacks** (`select_exact_live_action`, `agents.rs:577`): one live candidate → just execute its plan; no safe multi-goal probe → safe single-goal probe (`choose_safe_single_goal_probe`, `agents.rs:1016`); nothing safe → take at most one jointly safe step (`first_action_jointly_safe`, `agents.rs:1121`) or stop rather than gamble (`stats.exhausted`, `agents.rs:539`).

**Cost function:** none learned. Objective = discriminate cheaply; ranking is (candidates-falsified-per-probe, plan length). Safety is a hard constraint, not a penalty. The `cost_aware` variant (`choose_cost_aware`, `agents.rs:735`) that compared predicted falsifications/action vs distance reduction/action **lost** to plain falsify-only.

**The competing primitive** was shared progress (`choose_shared_progress`, `agents.rs:830`): take one jointly safe action recommended by a strict majority (≥2) of live shortest plans. Routers (`RoutingStrategy`, `agents.rs:397`) mixed the two by live-set size.

## (b) Evidence it won

Source: `main:docs/RESULTS_P1.md` (kept on main precisely as the archive; report SHA-256 hashes are recorded in the doc; the `runs/p1/*.json` artifacts themselves are not committed). All runs: 480 episodes (8 seeds × 60), six goal families, 9,999 paired stratified bootstraps. Explicitly labeled **developmental** (seeds reused after inspection), except P1A/P1B which were exploratory with frozen settings.

**P1C routing study** (seeds 101–108) — eight strategies head-to-head:

| Strategy | Success | Mean actions | Oracle-norm efficiency |
|---|---:|---:|---:|
| Sequential (test one goal at a time) | 0.9313 (+0.0375 terminal-failure rate) | 20.29 | 0.5477 |
| **Falsification-only** | 0.9500 | 21.64 | **0.5564** (best) |
| Shared-progress-only | 0.9313 | 28.71 | 0.2094 |
| Proposed router (progress broad / falsify narrow) | 0.9417 | 26.70 | 0.1696 |
| Reversed router (falsify broad / progress narrow) | **0.9521** (best) | 21.95 | 0.5509 |
| Dumb alternation | 0.9458 | 23.64 | 0.2364 |
| Cost-aware | 0.9417 | 26.74 | 0.1698 |
| Capped broad progress | 0.9458 | 25.47 | 0.1813 |

Every parallel policy eliminated terminal failures. The doc's stated conclusion: the original hypothesis **reversed** — falsification is most valuable under *broad* uncertainty (one observation removes several candidates; shared progress is weak evidence and adds actions); "falsification-only has the best efficiency and is the simpler default." Honest caveat recorded: vs the strongest baseline the +0.05 CI gate **failed** (success lift +0.0188, CI [-0.0083, 0.0458]) — parity-with-small-advantage on ordinary tasks.

**P1C-hard retargeting study** (seeds 201–208; adversarial scenarios where sequential commits wrongly ≥3 times) — the decisive win:

| Agent | Success | Terminal failures | Mean actions | Efficiency |
|---|---:|---:|---:|---:|
| Sequential discrimination | 0.8750 | 0.0542 | 66.47 | 0.1205 |
| **Falsification-only** | **0.9771** | 0.0000 | 34.67 | 0.4629 |
| Broad-falsify/narrow-progress | **0.9771** | 0.0000 | 34.46 | 0.4645 |

+0.1021 success and ~+0.343 efficiency, **passing** the +0.05 gates: success CI low +0.0750, efficiency CI low +0.3137. Adding narrow-set shared progress on top of falsification added nothing (efficiency lift +0.0016, CI [-0.0018, 0.0050]) — "falsification-only remains the simpler recommendation." Biggest family gains: avoid-hazard (sequential 0.675 success / 0.325 terminal failure → 1.000 / 0.000) and preserve-resource (0.850/0.014 eff → 1.000/1.000). Weaknesses: switch-order success, collect-all efficiency. Cost: ~100× internal search work for ~half the environment actions. History context: earlier P1C policies lost badly — always-shared-progress got 0.3125 success / 0.0380 efficiency; shared-prefix+commitment 0.9229 / 0.2214.

## (c) Removal: commit and rationale

- `142245aa` (2026-08-01, "Replace the archived P0 stack with the P1 exact-simulator harness and P2 world model") landed P1+P2 together on this lineage.
- **`55d7281b`** (2026-08-01 20:29, "Keep only archived P1 results on main; leave the harness on p1") is the removal: deleted `src/agents.rs` (−2,500), `src/report.rs` (−1,188), most of `src/experiment.rs` (−1,063), keeping only `docs/RESULTS_P1.md` and the shared MDP for P2. Stated rationale is exactly that one line — archival simplification when P2 became the active tree; no claim the method failed.
- The user's memory is right that it was an agent rewrite: both commits are `Co-authored-by: Cursor <cursoragent@cursor.com>`.
- Importantly, P2 *planned* to reimplement it — `main:docs/P2.md:83–84` maps the `falsification` lesson to "combine Q, events, and PTRM rank" and `retarget` to "PTRM ranking + rollout" — but only the **training-data** side was built (`generate_p1c_falsification_episode`, pre-prune `src/p2/data.rs:1271`; `PTRM_RANK_K_FALSIFICATION`, `train.rs:58`) plus eval diagnostics (pass@K vs best-Q@K, PTRM disagreement). The inference-time planner never existed in P2; live play stayed greedy one-step. `d30d5414` (2026-08-23 prune) then removed surrounding experiment machinery but didn't touch the (never-implemented) planner.

## (d) Compatibility with current P2 (at `c2ed65dc`) / v5 reimplementation

Per the archived inference-surface summary (docs/research/raw-2026-08-25-planning/p2-inference-surface.md) plus doc/head analysis:

**Rewire-only (no retraining):**
- **Probe simulation** — P1's `sim.transition` inside `simulate_probe_endpoint_successes` maps directly to `forward_from_latent` recursive latent imagination: encode once, roll a candidate action sequence latent-in/latent-out. This is the exact structural slot for the falsification probe simulator.
- **Predicates** — P1's `goal_satisfied` / `goal_terminal_failure` / `goal_viable_at` / no-op map one-to-one onto the event head's satisfied / irreversibly-failed / exhausted / noop logits (`docs/P2.md:145–149`). The jointly-safe prefix check becomes "predicted failed/exhausted probability below threshold for all live candidates along the imagined prefix"; endpoint counting becomes summed satisfied-probabilities.
- **Belief set** — `alive: BTreeSet<usize>` plus `update_belief_from_outcome` is model-free bookkeeping over real observations; port as-is, replacing exact predicates with real observed frames where checkable and calibrated event-head outputs otherwise (thresholded, or a soft posterior instead of hard set-deletion to absorb head noise — P1's hard deletion assumed exact predicates).
- **Reliability gating** — the Q head (prediction-correctness ranker, not a value) slots in as a *trust* gate P1 never needed: discard or down-weight imagined probe prefixes whose predicted correctness is low; PTRM disagreement (already computed diagnostically) is the second gate — exactly the `K=4` at "low-Q, high-disagreement, or irreversible decisions" policy already sketched in `docs/P2.md:229–234`.

**Needs retraining / new work:**
- **Candidate conditioning at inference** — the event head is candidate-conditioned but live play passes zero goal features, so it has never been exercised with real candidate features outside the synthetic curriculum; recovering per-candidate satisfied/failed predictions on live ARC-AGI-3 requires a candidate-proposal mechanism (P1 got candidates for free from the generator — this is the genuinely missing piece) and likely fine-tuning of the event head on proposed-candidate features.
- **Plan generation** — P1's exact `shortest_path` per candidate has no analogue; the v5 replacement is forward search over imagined latents (the beam machinery P1B validated: beam 0.5542 success / 0.5536 efficiency, PASS vs one-step controls — evidence that search-over-model beats reactive even before falsification). Open-loop rollout fidelity at probe lengths is the binding constraint; the rollout loss horizon ramp (2→4→8→max) determines usable probe depth, and per the 2026-08-24 diagnosis the current foundation checkpoint's rollout quality must be re-verified before trusting multi-step probes.
- **Calibration** — P1's hard safety constraint becomes probabilistic; event-head calibration (already an eval metric) needs to be good enough that "no live candidate predicts failure along the prefix" is meaningful.

**Bottom line:** the algorithm is a ~200-line controller (belief set + probe scorer + commitment) whose three oracles — transition, predicates, safety — all have named counterparts in the c2ed65dc model (`forward_from_latent`, event head, Q+PTRM disagreement). The winning variant to reimplement is plain **falsification-only** (simplest, best efficiency, tied-best success on the hard slice); routers and cost-aware scoring were tested and are not worth carrying.

Sanity check: all commit hashes, line numbers, and metrics above were read directly from `git show`/`git grep` output on `p1`, `main:docs/RESULTS_P1.md`, `main:docs/P2.md`, and pre-prune `d30d5414^` in this session; no working-tree files were read or modified.

result: Reconstructed P1 "planning by falsification" (FalsifyOnly controller in p1:src/agents.rs — multi-goal safe probes whose nonterminal outcomes eliminate several hypotheses), its winning evidence (0.9771 vs 0.8750 success on P1C-hard, best efficiency in routing study), removal commit 55d7281b (archival simplification; planned P2 port never implemented), and a v5 rewiring map (forward_from_latent + event/Q/PTRM heads reusable; candidate proposal and latent search need new work).