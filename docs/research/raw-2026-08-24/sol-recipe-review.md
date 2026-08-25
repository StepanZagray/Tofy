# Executive verdict

Full-v4 is a respectable representation-learning baseline, but it is structurally misaligned with both its primary metric and its deployed policy. The 2–5% changed-pixel exact plateau is not surprising. The rerun is useful as a replication/seed-variance measurement; I would not expect it to escape the plateau qualitatively.

The fatal issue for the 100% target is not model size. It is that the deployed policy has no objective to pursue:

- Dynamics are explicitly goal-free.
- Q and reliability estimate transition correctness, not reward or future success.
- Live inference passes zero goal features.
- The greedy score rewards predictability, non-noop behavior, and latent movement.
- The model receives no episode history, reward history, or learned belief about hidden goals.

A perfect world model attached to the current scorer would still not be a competent hidden-objective policy.

# 1. Recipe critique

## 1.1 Fatal: the training objective and live decision rule solve different problems

The repository itself describes P2 as learning “goal-free pixel dynamics”; public candidate features only condition an auxiliary head, never dynamics ([README.md:14](/home/stepan/Coding/Personal/Tofy/README.md:14)). In the model, goal features are projected separately and enter only the event head; Q and reliability see only the predicted canonical state ([model.rs:1007](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:1007), [model.rs:1028](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:1028)).

Live evaluation makes the mismatch explicit:

- Every candidate receives an all-zero goal vector ([arc3_live.rs:649](/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs:649)).
- Candidate score is `0.25 Q + 0.30 reliability + 0.30 non-noop + 0.15 latent-effect` ([arc3_live.rs:672](/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs:672)).
- The source admits that the policy has no reward/value head and is not a hidden-goal solver ([arc3_live.rs:36](/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs:36)).

Q and reliability are trained against the same exact-decoder transition-correctness labels ([train.rs:3167](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3167), [train.rs:3200](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3200)). Thus 55% of the live score is effectively duplicate confidence in whether the world prediction is decodable. Neither term says whether the action advances, discovers, or completes a goal.

At the reported 2–5% changed-pixel exact rate, changed-transition Q labels will also be predominantly negative. The heads can minimize BCE by predicting low correctness across all useful actions, while their small residual ranking differences still influence the live policy. Calibration does not turn a fidelity score into a value function.

Verdict: this is not “approximately aligned.” It is the wrong decision objective.

## 1.2 The primary metric has no direct training loss

The V4 world loss is:

- spatial Huber between predicted and target latents;
- canonical Huber between predicted and target canonical states;
- marginal Epps–Pulley;
- exact grounding on encoded current and encoded target states.

The relevant implementation is [train.rs:3031](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3031). Specifically, the prediction term is purely latent Huber ([train.rs:3067](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3067)), while exact grounding is applied only to `encoded.current` and `encoded.next` ([train.rs:3077](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3077)).

There is a trainable exact palette decoder ([model.rs:622](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:622)), but the world loss never applies its categorical loss to `out.y`. Evaluation does decode `out.y` ([model.rs:629](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:629)). That creates a classic seam:

```text
training:  out.y ≈ encoder(next) under mean latent Huber
evaluation: argmax decoder(out.y) must equal every target pixel
```

Small latent error does not imply preservation of categorical decision margins. A prediction can be near the target in Huber distance yet lie on the wrong side of many decoder boundaries, especially at the few changed pixels.

Grounding the target representation is useful, but it is not a substitute for supervising predicted next-state logits. The ADR accurately calls latent Huber a Tofy adaptation rather than a paper-faithful pixel objective ([ADR full-v4:60](/home/stepan/Coding/Personal/Tofy/docs/adr/0001-full-v4-training-contract.md:60)).

Verdict: the plateau metric is almost entirely an out-of-objective metric.

## 1.3 Padding and changed-pixel imbalance are extreme

Training scenarios are 7×7 and held-out compositions are 8×8 ([generator.rs:255](/home/stepan/Coding/Personal/Tofy/src/generator.rs:255)). They are copied into the top-left of a 64×64 canvas and padded ([data.rs:83](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:83)). Worse, `PAD` and semantic `EMPTY` are both palette value zero ([data.rs:31](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:31)).

The exact decoder excludes the bottom status row, leaving 4,032 supervised pixels. A 7×7 board occupies only 49 of them:

- 98.78% of supervised locations are outside the semantic board.
- A normal move changes roughly two pixels, about 0.05% of the decoder target.
- Because padding and empty cells are identical, the decoder cannot infer the content rectangle from the frame.
- `exact_grounding_loss(latents, frames)` receives no provenance/content mask ([model.rs:622](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:622)), even though provenance records the exact content dimensions ([data.rs:237](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:237)).

The latent imbalance is similarly severe. The encoder uses 8×8 patches, producing an 8×8 latent grid ([model.rs:254](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:254)). An entire 7×7 or 8×8 simulator board occupies only the first latent cell. For ordinary simulator transitions, 63 of 64 latent positions are effectively background/copy positions.

This is likely one of the largest causes of the plateau. The recipe asks an averaged loss dominated by “predict empty/copy” to learn a metric dominated by the rare changed pixels.

It also creates a serious transfer defect: the recurrent block is nominally spatial, but the primary simulator curriculum does not train dynamics distributed across the 8×8 latent canvas. It trains most mechanics inside one top-left patch.

## 1.4 Full-v4 deliberately excludes the data needed to identify action effects

The repository already contains complete same-state factual branch groups: four actions from an identical current state, including marker-free coordinate branches ([data.rs:995](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:995), [data.rs:1117](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:1117)). The older architecture rationale correctly notes that single sequential expert actions do not identify counterfactual effects ([ADR world-core-v2:5](/home/stepan/Coding/Personal/Tofy/docs/adr/0001-world-core-v2.md:5)).

Full-v4 nevertheless:

- omits `factual_branches` from its five lessons ([train.rs:44](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:44));
- disables branch learning ([train.rs:631](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:631));
- excludes action recovery, outcome separation, and changed/copy margins;
- uses shortest-path expert fragments during sequential training ([data.rs:1148](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:1148)).

Random dynamics and exploration cover different actions across different states, so action IDs are not entirely absent. But they do not establish which part of the outcome is caused by action rather than state. The same-state intervention is the statistically decisive evidence, and V4 intentionally throws it away.

Weak action usage is therefore predicted by the data contract, not an anomalous optimizer failure.

## 1.5 The synthetic action semantics are far too narrow

Synthetic ACTION6 always teleports one visible agent over an otherwise empty canvas ([data.rs:842](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:842)). Synthetic ACTION5 always toggles one neighboring switch ([data.rs:899](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:899)).

This teaches “ACTION6 means teleport this token” and “ACTION5 means recolor this switch,” not the broader concept that an action is an environment-defined operator whose semantics must be inferred from observed transitions. Public interactive games can assign radically different mechanics to the same official action type.

The ACTION6 field is well constructed as an input representation—continuous relative fields and an impulse preserve coordinate information ([model.rs:944](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:944)). The failure is the impoverished distribution of operators applied to that representation.

## 1.6 Fixed recurrence is not trained for the hard propagation regime

The recipe fixes two inner and two outer steps ([train.rs:621](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:621)). The recurrence applies RMS normalization and clamping after every outer update ([model.rs:1142](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:1142)).

This can work for local deterministic dynamics. It is poorly matched to:

- flood fills, connectivity, enclosure, and propagation;
- long chains of object interactions;
- mechanics requiring a number of computational iterations proportional to board diameter;
- public boards whose dynamics span many of the 8×8 latent cells.

More importantly, the simulator curriculum largely avoids testing this capacity because the entire board sits inside one input patch. Increasing recurrence alone will not fix that data geometry, but two fixed outer iterations become a real bottleneck once the geometry is repaired.

## 1.7 Open-loop training exists, but it is weak and still misaligned

The sequential lesson receives rollout weight 0.1, ramped over the lesson ([train.rs:2044](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:2044)). Horizon progresses 2→4→8 ([train.rs:2140](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:2140)). The rollout loss is again latent Huber, with optional teacher resets ([train.rs:3801](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3801)).

Only one ordered trace is collected per optimizer update, independently of the 2,048-row one-step batch ([train.rs:4738](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:4738), [train.rs:4896](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:4896)). Consequently:

- multi-step supervision has a much smaller effective trajectory population;
- it remains categorical-metric misaligned;
- it covers expert paths, not diverse interventions;
- it trains world rollout fidelity, not the deployed greedy action score.

This is useful regularization, not a solution to the policy problem.

## 1.8 The final 8,192 updates cannot repair the world model

With 4,096 base steps:

- dynamics: 8,192;
- exploration: 8,192;
- sequential: 4,096;
- q calibration: 4,096;
- falsification: 4,096.

That totals 28,672 because dynamics and exploration receive 2× steps ([train.rs:96](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:96), [foundation_train.sh:12](/home/stepan/Coding/Personal/Tofy/scripts/foundation_train.sh:12)).

For V4, `q_calibration` and `falsification` set world and SIGReg weights to zero ([train.rs:2055](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:2055), [train.rs:2085](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:2085)). Their representations are detached.

Thus the final 29% of training can only teach observer heads to describe the already-frozen predictor. Falsification cannot make the world model more action-sensitive, despite being precisely where action/goal ambiguity is exposed.

This schedule protects world weights from observer-stage corruption, but it also prevents late lessons from correcting the failure that matters.

## 1.9 Marginal EP is not an action-faithfulness objective

Full-v4 applies Epps–Pulley to current and target canonical marginals, independently across the two time positions ([train.rs:3090](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3090), [sigreg.rs:106](/home/stepan/Coding/Personal/Tofy/src/p2/sigreg.rs:106)). It does not regularize:

- predicted states;
- transition displacements;
- conditional distributions given action;
- separation between factual outcomes;
- changed versus unchanged transitions.

EP may prevent a gross marginal collapse while leaving action-conditioned displacement nearly zero. A Gaussian-looking encoder is not an identifiable world model.

The canonical representation is also broadcast into every spatial location during action conditioning ([model.rs:902](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:902)). Because that canonical state has unit RMS, this can encourage globally duplicated information and compete with localized spatial signals. It is not necessarily a bug, but it is an unnecessary risk when action localization is already weak.

## 1.10 There is a real V4 diagnostic bug

The trainer has gradient-pressure diagnostics, but exact grounding is silently omitted. V4 sets:

- `patch_grounding_weight = 0`;
- `exact_grounding_weight = 0.1`

([train.rs:588](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:588)).

The diagnostic measures grounding gradients only when `patch_grounding_weight > 0`, and differentiates the `grounding_head` prefix ([train.rs:5029](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:5029)). It never checks `exact_grounding_weight` or `exact_grounding_head`.

Therefore V4 reports no exact-grounding pressure or cosine against the prediction/EP objectives. A central fixed weight was adopted without the intended gradient-scale evidence. Global clipping at norm 1.0 then acts on the combined gradient ([train.rs:5193](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:5193)), so unmeasured objective dominance or conflict could materially change training.

This is a diagnostic bug, not necessarily the root training bug, but it invalidates confidence in the 0.1 weighting.

## 1.11 Status handling destroys potentially necessary information

V4 always replaces the bottom row with EMPTY before encoding ([model.rs:820](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:820)). The decoder also never predicts that row.

Two problems follow:

1. Synthetic exhausted labels depend on action count/budget, but the visual counter carrying that information has been removed. The event head cannot generally infer exhaustion from board pixels and a candidate goal alone.

2. The data module itself says ARC status placement varies by game ([data.rs:695](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:695)). Hard-masking row 63 assumes both that every public game uses that row as status and that no game uses it as gameplay. Neither follows from the allowed evidence.

If status removal is retained, status must be parsed into a separate explicit input and its location must be inferred or supplied. Blindly deleting a row is unsafe transfer behavior.

## 1.12 The live policy is not quite “bare learned greedy”

It performs no rollout search, but it does contain hand-authored harness behavior:

- ACTION6 proposals use color counts, bounding boxes, centroids, corners, and a grid ([arc3_live.rs:784](/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs:784)).
- Previously tried actions at an identical observation receive a hard −1 penalty ([arc3_live.rs:696](/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs:696)).
- All tried actions are eventually cleared and retried.

Under a strict “bare forward policy” definition, that is a proposal/retry harness. More importantly, the recorded `AgentSession` is not consumed by `ModelPolicy`; model scoring uses only the current frame, candidate action, and zero goals.

The API observation exposes `state`, `levels_completed`, and `win_levels` ([arc3_live.rs:94](/home/stepan/Coding/Personal/Tofy/src/p2/arc3_live.rs:94)), but none of those signals enter the learned policy. The strongest available reward/progress signals are thrown away.

## 1.13 Evaluation is diagnostically strong but scientifically vulnerable

The evaluator does several good things:

- content dimensions and trajectory identity are persisted;
- same-state factual groups are evaluated even though V4 does not train on them;
- copy-forward and action controls are reported;
- semantic metrics are separated from latent MSE.

However:

- The synthetic “OOD” shift is predominantly known 7×7→8×8 composition, still generated by the same narrow simulator.
- The automatic foundation script runs the public live suite after every training run ([foundation_train.sh:29](/home/stepan/Coding/Personal/Tofy/scripts/foundation_train.sh:29)). If recipe decisions are made from repeated public scores, the public suite is no longer a scientific held-out set, even if its recordings never enter gradient training.
- Exact-decoder correctness conflates transition quality with decoder quality. It is appropriate as an end-to-end metric but not a clean diagnostic of which component failed.
- At the current changed accuracy, Q/reliability class balance is likely pathological; live scoring still consumes those heads even if evaluator calibration gates fail.

Muon is not the leading suspect. Its routing is internally coherent—embeddings and heads stay on AdamW while matrix-like encoder/recurrent weights use Muon ([muon.rs:18](/home/stepan/Coding/Personal/Tofy/src/p2/muon.rs:18)). But AdamW-only and Muon-only matched-budget baselines remain worthwhile because the present objective mix is not pressure-calibrated.

# 2. Strategy review

## Judgment

A 560K-parameter, synthetic-only JEPA can plausibly learn this simulator’s dynamics. It is not a plausible direct path to 100% of diverse public ARC-AGI-3 games when coupled to a reactive, reward-blind policy.

I would assign:

- reasonable probability of eventually achieving high synthetic one-step accuracy after objective/data repairs;
- low probability of broad public transfer from the current generator family;
- effectively zero probability of 100% public completion with the current live decision objective, even given perfect optimization.

## Strongest steelman

The best case is:

- ARC grids are finite and categorical.
- Many mechanics are deterministic, local, compositional, and convolution-friendly.
- Unlimited synthetic exact transitions are available.
- The recurrent environment loop supplies a fresh real observation after every action, limiting open-loop error accumulation.
- A compact world model can potentially learn reusable primitives such as movement, collision, collection, toggle, teleportation, and no-op detection.
- A greedy policy could solve some reactive tasks if visible board change is a useful exploration proxy and success is immediately observable.

Under a sufficiently broad procedural generator, a compact equivariant model plus learned goal inference might be surprisingly capable.

That steelman supports “small synthetic models deserve serious study.” It does not support the current recipe or 100%.

## Strongest refutation

There is an information-theoretic obstruction. Suppose two hidden objectives or game rules produce the same current frame and available actions but require different next actions. The current learned policy is effectively a function of:

```text
current frame + candidate action
```

It cannot choose differently between those cases. Hard-coded tried-action state only supplies a set of past attempts at exactly matching observations; it does not encode action outcomes, inferred mechanics, or hidden-goal posterior.

No amount of latent prediction accuracy resolves missing decision information. A memoryless reward-blind policy cannot be universally optimal in a partially observed hidden-objective environment.

The narrow synthetic distribution compounds the problem:

- one simulator family;
- 7×7 boards placed in the top-left;
- one ACTION5 meaning;
- one ACTION6 meaning;
- six hand-authored goal families;
- shortest-path expert trajectories;
- no learned adaptation from public episode experience.

This is closer to transfer from one small game engine than to universal interactive rule induction.

## Best approach under the hard constraints

The best plausible single-A40, no-public-training, no-search approach would be a direct recurrent agent with an auxiliary world model:

1. **A 10–30M multiscale categorical visual model**, not necessarily huge. Use local high-resolution features plus global recurrent features. Preserve categorical pixels and explicit copy/change structure.

2. **A recurrent episode belief state** over observations, actions, observed outcomes, level progress, terminal state, and available actions. A compact Transformer, state-space model, or recurrent conv/attention hybrid is adequate.

3. **A direct greedy action-value or policy head** trained on synthetic episode returns, not prediction confidence. Its input should include:
   - current observation;
   - recurrent belief;
   - candidate action and coordinate;
   - inferred goal/rule posterior;
   - information-value estimate for exploratory actions.

4. **Goal/reward inference from permitted live signals.** The signal comes from:
   - `levels_completed` changes;
   - WIN/GAME_OVER;
   - visible persistent state changes;
   - action availability changes;
   - synthetic goal labels during pretraining.

5. **In-context adaptation**, always allowed under the stated constraint:
   - minimum version: recurrent activations updated by each confirmed transition;
   - stronger version: a small adapter or rule head receives a few self-supervised gradient steps on observed `(frame, action, next frame)` tuples;
   - keep the base visual/world representation frozen during live play to control catastrophic adaptation.

6. **Counterfactual pretraining** with complete same-state action groups and diverse operator semantics. Train inverse action recovery, outcome equivalence, and changed-effect separation alongside categorical next-state prediction.

7. **Recurrence to convergence**, with shared weights, residual-based halting, and a maximum compute cap. Train across variable iteration counts and board diameters.

8. **Domain randomization over mechanics**, not just layouts:
   - palette permutations;
   - translations and scales;
   - object shapes;
   - action-to-operator assignments;
   - local/global interactions;
   - delayed effects;
   - hidden state;
   - reversible and irreversible mechanics;
   - reward/goal ambiguity.

The world model remains valuable, but as an auxiliary representation and adaptation objective. The action selected by the bare greedy pass must be the argmax of expected task return or Bayes-adaptive value—not the argmax of “I predict that this action will cause a reliable visible change.”

# 3. Prioritized plan

Effect bands below are judgmental priors, not promises. “Changed” means absolute percentage points on held-out synthetic changed-pixel exact accuracy. “Live” means approximate absolute points on a normalized 0–100 level-completion score; actual official scaling may differ.

| Rank | Change | Frontier | Expected effect | Single-A40 cost | Cheap falsification |
|---|---|---|---|---|---|
| 1 | Train a recurrent, reward-aware greedy policy/value head on synthetic episodes. Feed observation history, actions, outcomes, `levels_completed`, terminal state, and available actions. | Standard-frontier | Changed: 0–3 pp. Live: +15–40 points if synthetic task diversity is adequate. | 20–60% extra training compute; inference modest. | Freeze the world model. Train only belief/value heads for 2–4K updates. Compare greedy synthetic completion against the current reliable-effect score on unseen seeds and goal compositions. |
| 2 | Apply exact categorical next-pixel loss directly to `out.y`, with separate changed/content/unchanged reductions and an explicit copy-plus-residual decoder. | Standard-frontier | Changed: +20–50 pp. Live: +2–10 until policy alignment is fixed. | Roughly +10–35% step time and 0.5–2GB VRAM if logits are chunked/content-masked. | Fine-tune a checkpoint for 1–2K world updates with only this new loss. Require changed exact improvement without worse unchanged or copy baseline. |
| 3 | Repair spatial/data geometry: randomize board position and scale; train 7→16→32→64 content; mask padding using provenance; sample changed pixels/patches explicitly. | Standard-frontier | Changed: +15–40 pp. Live: +5–20. | Similar compute if sparse/content-cropped losses are used; possibly faster than present full-canvas CE. | Compare top-left 7×7 versus randomly translated 7×7 for 1K updates. Evaluate translated held-out boards and per-latent-cell action sensitivity. |
| 4 | Put complete factual branch groups back into every world stage. Add changed-effect separation, equivalent-effect pull, action/coordinate recovery, and balanced no-op/change sampling. | Standard-frontier | Changed: +10–30 pp. Live: +5–15. | Low-to-moderate; simulator branches are cheap. Unique-state diversity per row falls, so mix with ordinary trajectories. | Start from the same checkpoint and run 500–1,500 updates with 25% factual groups. Require improvement in held-out same-state outcome retrieval and changed-action sensitivity before longer training. |
| 5 | Add in-episode adaptation: recurrent factual memory first; then optional low-rank/adapter updates from confirmed transition prediction error. | Beyond-frontier | Changed: +0–8 pp after adaptation. Live: +10–30 on games with learnable repeated mechanics. | Recurrent memory is cheap; 1–4 adapter SGD steps/action may add 20–100% inference latency. | Build synthetic meta-test games with permuted action semantics. Compare frozen, recurrent-memory, and adapter-update policies under identical action budgets. |
| 6 | Replace the narrow ACTION5/ACTION6 lessons with a distribution of latent operators and action-semantic permutations. Pretrain inverse dynamics and operator clustering. | Standard-frontier | Changed: +10–25 pp on unseen mechanics. Live: +10–25. | Mostly generator cost; 10–30% more training diversity/steps. | Hold out entire operator families, not just seeds. Reject if action recovery improves but outcome prediction does not. |
| 7 | Scale to a 10–30M multiscale model after loss/data repairs: high-resolution local stream, low-resolution global stream, cross-scale recurrence. | Standard-frontier | Changed: +8–25 pp. Live: +5–20, conditional on alignment. | Approximately 2–5× compute; use BF16 and reduce physical batch while preserving sample budget. | Compare 0.56M, ~5M, and ~20M for the same tokens/transitions and wall-clock checkpoints. Do not scale if the smallest model still learns copy-only. |
| 8 | Use recurrence-to-convergence with shared weights, variable training depth, residual stopping, and a hard maximum iteration budget. | Beyond-frontier | Changed: +5–20 pp on propagation strata; near zero on local moves. Live: +3–15. | 2–6× recurrence compute on hard states; adaptive stopping can recover average cost. | Create held-out propagation tasks parameterized by required radius. Test whether accuracy degrades with radius under fixed depth and becomes radius-robust under convergence training. |
| 9 | Interleave curricula instead of five monolithic stages. Continue low-rate world updates during calibration/falsification using factual/categorical objectives; retain replay from ACTION5/ACTION6 lessons. | Standard-frontier | Changed: +5–15 pp. Live: +2–8. | No major additional compute. | Run 2K-update matched-budget sequential versus stratified mixture schedules. Measure early-family forgetting and gradient cosines. |
| 10 | Replace duplicated Q/reliability semantics. Use one fidelity/calibration head and spend the other capacity on reward, progress, or information value. | Standard-frontier | Changed: 0 pp. Live: +5–20 after policy training. | Negligible. | On synthetic hidden-goal episodes, compare current score against immediate-reward Q, return Q, and information-aware Q using the same frozen encoder. |
| 11 | Calibrate objectives and optimizer before another long run: fix exact-grounding pressure diagnostics; record per-loss norms/cosines across lessons; compare AdamW-only, Muon hybrid, and loss-normalized variants. | Standard-frontier | Changed: −3 to +8 pp. Live: indirect. | 3–6 short 200–500-update screens. | Promote only if the optimizer arm improves both changed accuracy and action sensitivity under matched samples and seeds. |
| 12 | Seal evaluation governance: keep public live evaluation out of routine launcher-driven model selection; promote on preregistered synthetic gates and run public scorecards sparingly. | Standard scientific practice | No direct model gain; large reduction in false confidence and adaptive benchmark overfit. | Negligible compute savings, not cost. | Record every public evaluation and recipe decision. If decisions repeatedly depend on public scores, stop calling the suite held out. |

My immediate recommendation is ranks 2–4 as the cheapest causal diagnosis, while rank 1 is designed in parallel as the necessary strategy correction. Do not spend another full 28,672-step run on architecture scaling until direct predicted-pixel supervision, content masking, and branch coverage pass cheap gates.

# 4. Theory hooks for the top five changes

## Hook 1: reactive-policy impossibility and belief sufficiency

**Setting.** Let \(G\) be a finite set of hidden goals/rules, \(O\) finite observations, and \(A\) finite actions. A reactive deterministic policy is \(\pi:O\to A\). For hidden goal \(g\), let \(Q_g(o,a)\) be the optimal finite-horizon action value.

**Assumptions.**

- There exist \(g_1,g_2\in G\) and an observation \(o\in O\) reachable under both.
- Each has a unique optimal action:
  \[
  a_1=\arg\max_a Q_{g_1}(o,a),\qquad
  a_2=\arg\max_a Q_{g_2}(o,a)
  \]
  with \(a_1\ne a_2\).

**Claim.** No reactive deterministic policy is optimal for both hidden goals.

**Proof skeleton.** \(\pi(o)\) has one value. It cannot equal both distinct unique optima.

**Positive statement.** For a finite-horizon POMDP, the posterior belief
\[
b_t(g,s)=P(g,s_t\mid o_{0:t},a_{0:t-1},r_{1:t})
\]
is a sufficient statistic for optimal control. A greedy decision
\[
a_t=\arg\max_a Q^\*(b_t,a)
\]
is optimal when \(Q^\*\) is the Bayes-optimal belief-state action value.

This formally justifies recurrent history plus a reward-aware Q head and formally rules out the current frame-only policy as a universal solver.

## Hook 2: categorical loss upper-bounds changed-pixel error

**Setting.** Let \(C\) be a finite palette and \(M\) the finite set of changed gameplay pixels. For each \(i\in M\), the decoder predicts a distribution \(p_i\) and the true class is \(y_i\). Prediction is \(\hat y_i=\arg\max_c p_i(c)\).

**Claim 1.**
\[
\mathbf 1[\hat y_i\ne y_i]
\le
\frac{-\log p_i(y_i)}{\log 2}.
\]

**Reason.** If the argmax is wrong, some other class has probability at least \(p_i(y_i)\), hence \(p_i(y_i)\le 1/2\), so \(-\log p_i(y_i)\ge\log2\).

**Claim 2.**
\[
\frac1{|M|}\sum_{i\in M}\mathbf 1[\hat y_i\ne y_i]
\le
\frac{\operatorname{CE}_M}{\log2}.
\]

**Claim 3, exact transition.**
\[
\mathbf 1[\exists i\in M:\hat y_i\ne y_i]
\le
\frac1{\log2}\sum_{i\in M}-\log p_i(y_i).
\]

Thus changed-pixel categorical CE directly controls the requested metric. Latent Huber has no analogous bound without additional decoder Lipschitz and classification-margin assumptions.

## Hook 3: counterfactual branch coverage is necessary for transition identifiability

**Setting.** Let \(S,A,Y\) be finite and let the true deterministic transition be \(F:S\times A\to Y\). Training data contains pairs \(D\subseteq S\times A\) with observed outcomes \(F(s,a)\).

**Claim 1.** If \((s,a)\notin D\) and \(|Y|\ge2\), there exist at least two transition functions \(F_1,F_2\) agreeing on all training observations while differing at \((s,a)\).

**Construction.** Set \(F_1=F_2=F\) on \(D\), choose two different outcomes for the uncovered pair, and define the rest arbitrarily.

Therefore one expert action per state does not identify the action-conditioned transition, regardless of zero training loss.

**Claim 2.** If a branch group supplies every \(a\in A_s\) for each covered state \(s\), and categorical empirical loss is zero, then the learned deterministic argmax transition equals \(F(s,a)\) for every covered pair.

This is the finite-grid justification for complete same-state factual groups.

## Hook 4: translation randomization removes the top-left shortcut

**Setting.** Let a finite group \(T\) act on grids by valid translations. Assume the data distribution is translation invariant and labels transform equivariantly. Let loss \(\ell(p,y)\) be convex in the predicted categorical distribution.

For any predictor \(f\), define the symmetrized predictor
\[
\bar f(x)=\frac1{|T|}\sum_{t\in T}t^{-1}f(tx).
\]

**Claim.**

1. \(\bar f\) is translation equivariant.
2. Its expected risk satisfies \(R(\bar f)\le R(f)\).

**Proof skeleton.** Equivariance follows by reindexing the finite group sum. The risk inequality follows from Jensen’s inequality and translation invariance of the data distribution.

This justifies randomized placement plus an equivariant architecture: a top-left-specific shortcut cannot be uniquely optimal under the symmetrized distribution.

## Hook 5: finite-hypothesis online adaptation can identify episode mechanics

**Setting.** Let \(H\) be a finite set of deterministic transition/reward hypotheses containing the true \(h^\*\). After history \(D_t\), maintain the version space
\[
V_t=\{h\in H:h\text{ is consistent with }D_t\}.
\]

Assume the selected action at each non-singleton version space produces an observation whose resulting consistent subset has size at most \(\rho |V_t|\), for fixed \(0<\rho<1\).

**Claim.**
\[
|V_t|\le \rho^t|H|,
\]
so the true hypothesis is uniquely identified after at most
\[
\left\lceil\frac{\log |H|}{-\log\rho}\right\rceil
\]
informative transitions.

Once \(V_t=\{h^\*\}\), greedy action selection from the exact finite-horizon \(Q_{h^\*}\) is optimal.

This supports in-episode rule adaptation and information-seeking action values. It also clarifies the necessary assumption: the policy must choose discriminating actions and retain their observed outcomes. Merely penalizing repeated actions does neither.

# Bottom line

Full-v4 has answered its bounded question: marginally regularized latent prediction plus target-only grounding is insufficient. Treat foundation-v1 strictly as replication.

The next decisive experiment should not be another long JEPA rerun. It should be a short 2×2 causal screen:

```text
predicted categorical loss: off / on
same-state factual branches: off / on
```

Use content-masked, changed-balanced loss; identical seeds, samples, optimizer budget, and evaluator inputs. Gate on:

- changed-pixel exact;
- unchanged-pixel exact;
- copy-forward ratio;
- same-state factual outcome retrieval;
- action sensitivity on genuinely outcome-changing pairs;
- direct predicted-logit margins.

If that does not produce a large change within 1–2K updates, the current encoder/recurrent topology is the next suspect. If it does, then build the reward-aware recurrent policy before spending on scale.

No files were modified, no processes were started, and no tests/builds were run. Research evidence was restricted to the named repository sources; no web sources or external research library were consulted. Principal commands were read-only `rg -n`, `rg --files`, `wc -l`, and numbered `nl -ba … | sed -n …` inspections over the requested files.
