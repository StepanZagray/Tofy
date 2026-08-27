# Executive verdict

The highest-value new theory result is not exotic: v5 appears to apply rare-change correction twice. The code first separates changed and unchanged pixels into conditional means, then multiplies the changed mean by `(1-p)/p`. That produces an effective per-pixel ratio of `((1-p)/p)^2`, whereas the existing copy-attractor argument derived `(1-p)/p` for a pooled loss. This is the first thing I would falsify.

My bets on the next binding constraints are:

1. **EXPLOIT — output-loss normalization and decoder allocation.** More likely than insufficient latent capacity or EP collapse.
2. **EXPLOIT — partial observability, but only if a corrected history census demonstrates a real oracle gap.** The current H1/H2 census cannot establish that.
3. **EXPLORE — planner post-selection error.** Once planning starts, optimizing over thousands of imagined candidates will invalidate marginal model-error calibration unless selection is explicitly covered.
4. **Not a binding constraint by itself — the reported 0.67 foreground reconstruction.** It does not impose a mathematical 0.67 cap on changed-transition exact accuracy because it is measured on a different pixel set.
5. **Not yet trustworthy as a certificate — inverse-action consistency.** A perfectly wrong action-code model can achieve zero inverse loss and full displacement separation.

All numerical effect sizes below are preregistration hypotheses, not evidence.

## Ranking by expected value per GPU-hour

| Rank | Proposed theorem/design consequence | Type | Cheap test cost | Expected effect |
|---:|---|---|---:|---|
| 1 | Normalization-aware rare-change weighting | EXPLOIT | 0.05 GPU-h offline; 1.3–4 GPU-h confirmation | Central estimate `+4 pp` changed-exact |
| 2 | Exact decoder/gate cap decomposition | EXPLOIT | Under 0.1 GPU-h offline | Direct `0 pp`; prevents misallocated retraining |
| 3 | Selection-valid planning and certified horizon | EXPLORE | 0.1–0.5 GPU-h offline | `+5–20` live-score points, speculative |
| 4 | Exact finite-history ambiguity ceiling | EXPLOIT | Under 0.1 GPU-h census; 1.3–2 GPU-h if warranted | `0–5 pp` changed-exact |
| 5 | Inverse-cycle insufficiency theorem | EXPLORE | Under 0.1 GPU-h | `+3–12` live-score points through safer planning |
| 6 | Factorized transition identifiability bound | EXPLOIT | CPU census; 1.5–4 GPU-h implementation screen | `+3–10 pp` held-out changed-exact |
| 7 | Minimum complete probe family | EXPLORE | CPU minutes; up to 1.5 GPU-h confirmation | `+3–12` live-score points |
| 8 | Interference-free one-shot correction | EXPLORE | Under 0.1 GPU-h replay screen | `+5–20 pp` on revisited-transition subset |

The repo’s measured A40 rate of roughly 4.72–4.76 seconds/update makes 1,000 updates about 1.32 GPU-hours and 2,000 about 2.64 GPU-hours before model-size overhead ([RESULTS_P2:702](/home/stepan/Coding/Personal/Tofy/docs/RESULTS_P2.md:702)).

---

## 1. Normalization-aware rare-change theorem

**Type: EXPLOIT**

### Statement

Let `C` and `U` be changed and unchanged pixels, with `p=|C|/(|C|+|U|)`. Consider:

```text
L_split = alpha * mean_C(loss_i) + beta * mean_U(loss_i)
```

The coefficients applied to individual changed and unchanged pixels are:

```text
changed:   alpha / |C|
unchanged: beta  / |U|
```

Therefore the per-pixel ratio is:

```text
(alpha / beta) * ((1-p) / p)
```

Consequently:

- Equal changed/unchanged aggregate weight requires `alpha=beta`.
- A desired per-pixel ratio `(1-p)/p` also requires `alpha=beta`.
- Setting `alpha/beta=(1-p)/p` produces the squared per-pixel ratio:

```text
((1-p) / p)^2
```

More generally, if `g_C` and `g_U` are the mean directional gradient magnitudes, aggregate gradient parity requires:

```text
alpha / beta = g_U / g_C
```

The class-frequency factor has already been consumed by taking conditional means.

### Why this changes v5

The ratio is calculated in [src/p2/train.rs:130-143](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:130), while `split_weighted_ce` computes `positive_weight * mean_positive + mean_negative` in [src/p2/train.rs:3459-3472](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3459). Predicted and encoded CE then use this combination in [src/p2/train.rs:3637-3644](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3637) and [src/p2/train.rs:3658-3701](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3658).

The copy-gate BCE is different: it weights pixels before a pooled reduction, so `(1-p)/p` is appropriate there ([src/p2/train.rs:3646-3652](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3646)).

At `p=0.1`, the current split CE gives changed pixels roughly `81x` the coefficient of unchanged pixels, not `9x`. At `p=0.02`, the unclamped value would be about `2401x`; even with the 64 clamp, the effective ratio remains approximately `3136x`.

### Precise mechanism

Replace the predicted and encoded split CE with either:

```text
L = 0.5 * mean_changed_CE + 0.5 * mean_unchanged_CE
```

or a measured-gradient controller:

```text
alpha / beta = clamp(EMA(g_U / g_C), 1/4, 4)
alpha + beta = 1
```

Keep the existing pooled class correction for copy-gate BCE. Log, separately for every component:

- loss scalar;
- changed and unchanged gradient norm;
- cosine with the total prediction gradient;
- changed and unchanged decoded accuracy.

### Proof and Lean assessment

This is finite-sum algebra. A Lean proof needs `Finset.sum`, cardinality identities, nonzero denominators, and `field_simp`/`ring`. Gradient magnitudes can be abstract nonnegative scalars, so no differential calculus is needed.

### Cheap falsification

Resume the same frozen checkpoint in three matched 512–1,024-step arms:

1. current squared effective weighting;
2. equal split means;
3. gradient-budgeted split means.

Reject the theorem’s design implication if the current arm wins on held-out changed-exact without a statistically or operationally significant reconstruction/gate regression.

### Expected effect

**Speculative:** central estimate `+4 pp` changed-exact, plausible range `-5 to +10 pp`. I would also expect a `+10–20 pp` improvement in target reconstruction if decoder starvation is real.

### Failure modes

- Changed pixels may deliberately need more than aggregate parity because they carry more semantic value.
- Early estimates of `g_C/g_U` may be noisy when changes are rare.
- Scalar loss contribution can disagree with useful gradient direction.

Detect these through gradient cosines and fixed held-out operator slices, not loss magnitude alone.

---

## 2. Decoder-substitution and gate-cap theorem

**Type: EXPLOIT**

### Statement

On the changed set, define:

- `P`: the predicted-latent decoder is exactly correct;
- `B`: the target-encoding decoder is exactly correct;
- `Q`: predicted and target latent decodes differ;
- `G`: the copy gate opens on every changed pixel;
- `A`: the final composed prediction is exactly correct on the changed set.

Because copying the current pixel is necessarily wrong on a changed pixel:

```text
A = P ∩ G
```

Moreover, if the two raw decodes are identical, their correctness agrees, so:

```text
P Δ B ⊆ Q
```

Therefore:

```text
Pr(B) - Pr(Q) - Pr(not G) <= Pr(A) <= Pr(B) + Pr(Q)
```

This decomposes the observed ceiling into:

1. decoder-on-target-latent failure;
2. predicted-latent distribution shift;
3. gate failure.

### What 0.67 does and does not imply

If `r` were mean pixel accuracy on the **same changed set and same example distribution**, then exact accuracy `E` satisfies:

```text
E <= r
```

because the exact-correct indicator is at most the fraction of correct pixels.

But the reported foreground reconstruction metric uses `target != EMPTY`, whereas changed-exact uses `current != target` ([src/p2/semantic_eval.rs:358-391](/home/stepan/Coding/Personal/Tofy/src/p2/semantic_eval.rs:358)). Neither set contains the other. A model can be poor on static foreground and perfect on changed pixels, or the reverse. Thus **0.67 foreground accuracy gives no nontrivial hard bound on changed-exact**.

The evaluator already computes target-reconstruction results on the changed mask ([src/p2/semantic_eval.rs:399-413](/home/stepan/Coding/Personal/Tofy/src/p2/semantic_eval.rs:399)); that exact field, not foreground pixel accuracy, is the relevant empirical decoder cap.

### Precise mechanism

Before changing the decoder, emit by source/operator/change-count:

```text
B = target-latent changed-exact
P = predicted-latent raw changed-exact
Q = predicted-vs-target decoded disagreement
G = all-changed gate-open rate
A = composed changed-exact
```

Decision rule:

- Low `B`: decoder capacity/training is binding.
- High `B`, large `Q`: dynamics produce off-manifold latents.
- High `P`, low `A`: gate is binding.
- High `B` and high `P`: do not spend compute on the decoder.

### Proof and Lean assessment

Finite Boolean-event algebra is enough. It can be formalized over `Finset` counts without probability measure theory. The counterexamples showing no foreground-to-changed bound fit on two pixels.

### Cheap falsification

Offline rescore an existing checkpoint. No optimization is necessary. If `B` is below roughly 0.65 and `Q` is small, run a 512–1,000-step decoder-only treatment. Otherwise reject “decoder first.”

### Expected effect

Direct diagnostic effect: `0 pp`.

If the decoder is confirmed binding, **speculative** recoverable changed-exact is `+5–20 pp`. If `B>0.9`, expect less than `+2 pp` from decoder work.

### Failure modes

Target and predicted latents may occupy different supports; that is why `Q` must be reported. An average disagreement metric is insufficient—use changed-set exact disagreement and change-count strata.

---

## 3. Selection-valid planning and certified-horizon theorem

**Type: EXPLORE**

The frontier now has tight worst-case horizon results for imperfect models, but those results reinforce rather than remove the selection problem: [Imperfect World Models Are Exploitable](https://arxiv.org/abs/2605.15960) derives a tight safe effective horizon under bounded total-variation error. Tofy additionally needs a finite deterministic, margin-aware certificate for its actual exhaustive root search.

### Statement A: fixed-path error

For a deterministic path of length `H`, let `epsilon_j` bound the conditional probability that imagined edge `j` is wrong given that all previous edges were correct. Then:

```text
D_H <= min(1, sum_j epsilon_j)
```

If those conditional failures are independent with common rate `epsilon`:

```text
D_H = 1 - (1-epsilon)^H
```

For terminal utilities in `[0,1]`, plan-value error is at most `D_H`. Comparing two estimated plans incurs regret at most `2D_H`.

A plan with estimated best-versus-runner-up margin `gamma_H` is certified only if:

```text
gamma_H > 2D_H
```

### Statement B: there is no universal optimal horizon

One-step exact accuracy alone cannot determine a value-optimal horizon. For any proposed fixed horizon, construct:

- an MDP whose only reward occurs one step beyond that horizon; and
- another MDP where every additional imagined step introduces model error without additional reward.

Thus the design-relevant quantity is not a globally optimal `H`, but the **longest certified horizon**:

```text
H_cert = max { H : gamma_H > 2D_H }
```

Under independent constant error:

```text
H_cert = max { H : 2 * (1 - (1-epsilon)^H) < gamma_H }
```

Under only a worst-case bound:

```text
H_cert = max { H : 2H epsilon < gamma_H }
```

### Statement C: search magnifies error

If the planner evaluates `N` candidates, each with failure probability at most `D_H`, then:

```text
Pr(any candidate is wrong) <= min(1, N D_H)
```

This is tight without dependence assumptions. Optimizing over candidates can select precisely the corrupted one.

The Phase A design evaluates 4,102 depth-one roots and roughly 4,614 recurrent steps ([ADR 0004:335-390](/home/stepan/Coding/Personal/Tofy/docs/adr/0004-latent-planning-contract.md:335)). Marginal source/horizon calibration and a per-prefix union bound ([ADR 0004:203-230](/home/stepan/Coding/Personal/Tofy/docs/adr/0004-latent-planning-contract.md:203)) do not by themselves cover selection across those roots.

For perspective, if 4,102 candidates each had independent two-percent failure probability, the probability that at least one is wrong would be effectively one. A naive family-wise two-percent certificate would require per-candidate error near `0.02/4102`, around `4.9e-6`.

### Falsification-repair corollary

If a factual replay or probe verifies edge `j`, set its `epsilon_j=0`. The path bound decreases by at most that edge’s prior contribution. A probe is decision-repairing exactly when it changes:

```text
gamma_H - 2D_H
```

from nonpositive to positive, after accounting for probe cost.

This gives falsification probes a theorem-level role: they need not improve the model globally; they only need to remove enough uncertainty from a finalist to cross the margin certificate.

### Precise mechanism

- Generate candidates with the learned model.
- Score finalists using a separately calibrated verifier or held-out bootstrap.
- Compute a path-specific `D_H` from source, operator, horizon, novelty and gate strata.
- Treat exact factual replay edges as zero-error.
- Increase horizon only while `gamma_H > 2D_H`.
- Trigger a probe only if its possible outcome partition can make that inequality decisive.
- Calibrate on the distribution of **planner-selected** edges, not merely ordinary held-out edges.

### Proof and Lean assessment

The union bounds and regret proof are elementary finite probability. The “no universal horizon” counterexamples are tiny finite deterministic MDPs. Independence products need more probability infrastructure, but the worst-case theorem can be completed using finite counts. The margin step can reuse the structure of `Greedy.lean`.

### Cheap falsification

With frozen weights and the exact simulator:

1. evaluate candidate counts `N=1,16,64,4102`;
2. measure ordinary edge error versus selected-edge error;
3. measure plan error by horizon;
4. compare `gamma_H` against actual misranking;
5. replace one disputed edge by exact replay and test whether the certificate changes as predicted.

### Expected effect

Changed-exact: `0 pp`.

**Speculative live effect:** `+5–20` absolute score points or a substantial reduction in catastrophic terminal actions. It may reduce short-term progress if certification is overly conservative.

### Failure modes

- Candidate failures can be strongly correlated, making a raw union bound too pessimistic.
- A verifier sharing the same trunk may share the same errors.
- Transition exactness does not cover value-head error.

Report empirical effective candidate count, calibrate post-selection, and keep transition and value uncertainty separate.

---

## 4. Exact finite-history ambiguity theorem

**Type: EXPLOIT**

Predictive-state work already establishes that a fixed observation window need not be sufficient and that predictive tests can be a more compact state representation; see [Predictive Representations of State](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html). The missing Tofy result is an exact finite ceiling tied to its generator.

### Statement

Let `K_k` contain:

- the last `k` public observations;
- every intervening action and public outcome/effect;
- the queried next action.

Let `Y` be the exact next board. For a finite evaluation distribution `mu`, define:

```text
C_k = sum_x max_y mu(K_k=x, Y=y)
```

Equivalently, define collision error:

```text
rho_k = sum_x (mu(K_k=x) - max_y mu(K_k=x, Y=y))
```

Then:

```text
C_k = 1 - rho_k
```

and:

1. every deterministic `k`-history predictor has exact accuracy at most `C_k`;
2. a per-key mode predictor attains `C_k`;
3. perfect prediction is possible iff `Y` is constant on every positive-mass key fiber;
4. if `K_(k+1)` genuinely refines `K_k` on the same population, then `C_(k+1) >= C_k`.

The minimal sufficient history is the least `k` for which `rho_k=0`. It may not exist for any finite `k`.

### Current measurement gap

The current H2 key includes a previous frame, current frame and current action, but omits at least the previous action/outcome fields; it also uses only contiguous rows, while H1 uses a broader population ([src/p2/semantic_eval.rs:461-545](/home/stepan/Coding/Personal/Tofy/src/p2/semantic_eval.rs:461)). Therefore the current H1/H2 numbers are neither a proper refinement comparison nor evidence of monotonic history benefit.

An empirical mode ceiling is also optimistically biased by singleton keys. The decisive census must either enumerate hidden-state aliases or estimate the mode on training fibers and score it on separate samples.

### Precise mechanism

First build a paired oracle census over simulator hidden states rendered to the same public trace:

```text
K_k -> set of exact next boards
```

Include all public fields and actions. Report `rho_k` for `k=1..16`, by operator and source.

Only if `C_k-C_1` is material should the model gain a history tensor:

```text
history: [B, k, latent_tokens + action + effect]
belief:  GRU/SSM(history) -> [B, d_belief]
```

Inject the belief through FiLM into every recurrent step. Compare against shuffled history and stateless capacity-matched controls.

### Proof and Lean assessment

Use finite weighted keys, `Finset`, maxima over finite codomains and partition refinement. The theorem is moderately straightforward. Confidence bounds for sampled estimates are separate from the finite exact theorem.

### Cheap falsification

Run the paired census offline. Reject a history architecture if the held-out lower confidence bound on `C_k-C_1` is below `2 pp`. Otherwise test frozen latent history probes, then a 1,000-step side-GRU treatment.

### Expected effect

**Speculative:** `0–5 pp` changed-exact and `+2–10` live-score points. The exact upper opportunity is the measured `C_k-C_1`; do not claim more.

### Failure modes

- Singleton bias can falsely produce a ceiling near one.
- Some mechanics may require unbounded belief rather than fixed history.
- The generator census may omit the aliases encountered live.

Use held-out fibers, explicit hidden-state pairing and an “unknown operator” slice.

---

## 5. Inverse-cycle insufficiency theorem

**Type: EXPLORE**

ACID uses action-consistency through inverse dynamics as a useful control signal ([ACID, arXiv:2607.02403](https://arxiv.org/abs/2607.02403)). It is not, however, a correctness certificate.

### Statement

For any finite action set with at least two actions, choose an injective code `c:A -> Z`. Define:

```text
F_hat(s,a) = c(a)
G_hat(z)   = inverse_c(z)
```

Then:

```text
G_hat(F_hat(s,a)) = a
```

for every state and action, so inverse-action loss is zero. If the codes are separated, displacement separation is also zero. Yet `F_hat` is independent of the true successor and can be wrong on every transition.

Therefore:

> Inverse-action recovery plus displacement separation does not imply outcome correctness, causal action use, or safe imagination.

### Why this matters

The existing separation theorem proves only that zero hinge loss separates covered displacement pairs and penalizes identical displacement ([Separation.lean:18-53](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/Separation.lean:18)). The planner contract nevertheless makes inverse-action recovery part of an irreversible-edge trust gate ([ADR 0004:219-230](/home/stepan/Coding/Personal/Tofy/docs/adr/0004-latent-planning-contract.md:219)).

That gate can accept an action-labelled hallucination.

### Precise mechanism

- Retain inverse action as an **action-use diagnostic**.
- Remove it as a sufficient safety/trust condition.
- Add an independently trained raw-outcome verifier receiving:
  - current public state;
  - action;
  - decoded predicted change set;
  - optional factual prototype distances.
- Train it on correctness labels from held-out exact simulator transitions.
- Require selective calibration specifically on planner-selected predictions.

### Proof and Lean assessment

This is an easy finite construction using `Fintype`, an embedding into `Fin |A|`, and a left inverse. A two-action/two-code explicit instance may be simplest. It should be added beside `Separation.lean`.

### Cheap falsification

On an existing checkpoint, condition successor error on inverse-action correctness:

```text
Pr(wrong successor | inverse action correct)
```

Also train or construct an action-code-only baseline. If the conditional error is close to ordinary error, inverse correctness supplies little trust information.

### Expected effect

Changed-exact: `0 pp`.

**Speculative live effect:** `+3–12` score points through fewer false-safe plans.

### Failure modes

The verifier may share representations and errors with the world model. Measure conditional calibration under actual planner selection and include deliberately wrong action-code controls.

---

## 6. Factorized transition identifiability and sample-complexity theorem

**Type: EXPLOIT**

Factored MDP theory can provide exponential sample improvements under strong realizability assumptions ([Sun et al., 2019](https://proceedings.mlr.press/v99/sun19a.html)). Recent controlled-world-model identifiability results likewise require strong excitation and model assumptions ([On Identifiability of Controlled World Models, arXiv:2607.22430](https://arxiv.org/abs/2607.22430)). Tofy needs a finite categorical theorem that first tests whether the factorization is realizable.

### Statement

Let a board contain `n` cells and `q` colors. For each labelled cell transition define a feature:

```text
phi(s,a,i) in Phi
```

and assume the successor color factors as:

```text
T(s,a)(i) = h(phi(s,a,i))
```

for an unknown `h:Phi -> Colors`.

Then:

1. Exact identification of `h` on an evaluation support is possible iff every feature value used by that support has been labelled at least once.
2. If one feature value is absent, there are at least `q` mutually indistinguishable hypotheses differing only there.
3. Under IID labelled-cell sampling where every relevant feature has probability at least `rho`:

```text
Pr(any feature unseen after N labels) <= |Phi| * exp(-rho N)
```

so it is sufficient that:

```text
N >= (log |Phi| + log(1/delta)) / rho
```

4. For a monolithic feature `phi=(s,a,i)`, the feature-domain size can be `q^n * |A| * n`.
5. For a realizable local/operator-relative factor with receptive field `m` and `K` action roles:

```text
|Phi| <= K * q^m
```

The ratio is exponential in `n-m`.

For transition rows rather than independent cell labels, the exact minimum is the set-cover number of realizable branch rows over the required feature classes. A claim about exponentially fewer **branch groups** requires proving that this cover number scales with the smaller feature domain; it does not follow merely from the architecture.

### Precise mechanism

Introduce an explicit structured edit bottleneck:

```text
local_feature_i =
  local patch tokens
  + action kind
  + coordinate relative to click
  + row/column summary
  + operator-belief vector

change_logit_i = shared_mask_head(local_feature_i)
color_logits_i = shared_edit_head(local_feature_i)
```

Retain a small global branch for nonlocal mechanics. Train with:

```text
BCE(change mask)
+ CE(edit color | changed)
+ composition consistency
```

The important difference from the current copy gate is that the edit rule is explicitly shared over feature-equivalent cells and audited for collisions.

### Realizability test

Before training, compute:

```text
C_phi = sum_u max_y P(phi=u, Y=y)
```

This is the exact best accuracy achievable by the proposed factorization. Reject any `phi` whose held-out collision error exceeds one percent, especially on PushLine and held-out SwapRegion.

The current branch sampler provides object, boundary, empty and symmetric coordinate strata, but only a small action subset per branch group ([src/p2/data.rs:2015-2176](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:2015)); that is useful coverage, not a factorized identifiability proof.

### Proof and Lean assessment

The exact-coverage result uses finite functions and agrees with the spirit of the current identifiability construction. Function counts use `Fintype.card_congr` and `Fintype.card_fun`. The exponential/coupon bound is harder; a finite-union bound can be proved first, while the exact active-query result is easy.

### Cheap falsification

- CPU-only collision census for candidate feature maps.
- Compare current head against the best collision-free structured head for at most 2,000 matched steps.
- Reject if collision-free features require nearly global boards or the factorized arm loses held-out exactness.

### Expected effect

**Speculative:** `+3–10 pp` changed-exact on held-out geometry/operators, `0–5 pp` in-distribution, and `+2–8` live-score points.

### Failure modes

- PushLine and SwapRegion may violate small-radius locality.
- Operator identity may be latent and not inferable from the proposed history.
- A learned continuous feature map does not inherit the finite theorem automatically.

Detect these through exact feature collisions broken down by effect radius and operator family.

---

## 7. Minimum complete probe-family theorem

**Type: EXPLORE**

Minimum adaptive distinguishing-sequence construction is already known to be computationally difficult in general ([Türker and Yenigün, 2016](https://doi.org/10.1016/j.infsof.2016.02.001)). Tofy’s bounded operator family is small enough to solve directly.

### Statement

Let `Theta` be a finite mechanics family. Probe `p` has deterministic public outcome `o(theta,p)`. Define the hypothesis pairs separated by `p`:

```text
S_p = { {theta,theta'} : o(theta,p) != o(theta',p) }
```

Then:

1. A nonadaptive probe set `P` identifies every mechanic iff:

```text
union_{p in P} S_p
```

contains every unordered distinct pair in `Theta`.

2. The minimum complete probe set is exactly a minimum set cover of the hypothesis-pair universe.

3. An adaptive strategy is complete iff every reachable nonsingleton version space admits a safe probe producing a nontrivial partition.

4. If every nonsingleton version space has such a probe, a strategy of worst-case depth at most `|Theta|-1` exists.

5. If a probe has at most `b` observable outcomes, every complete strategy needs worst-case depth at least:

```text
ceil(log_b |Theta|)
```

Safety and reachability must be included in the probe definition. Pairwise separability by an unreachable state is useless.

### Precise mechanism

Enumerate canonical reachable state templates and build a binary pair-by-probe matrix. Solve the small exact set-cover problem offline. Add the resulting probes to:

- training branch groups;
- the planner’s preserved root candidates;
- the proposal-head coverage loss.

The proposal head should pay a penalty if it assigns insufficient mass to every probe in the current version-space cover.

### Proof and Lean assessment

Finite-set cover equivalence is straightforward. The adaptive upper bound follows by induction because a separating probe reduces the largest surviving version space by at least one. The information lower bound uses finite cardinality and powers.

### Cheap falsification

Enumerate the five current operator families, action templates and canonical board states. Report:

- indistinguishable operator pairs;
- minimum safe cover;
- whether the existing ten-row branch groups contain that cover;
- expected adaptive depth.

If the current groups already provide a complete safe cover, make no change.

### Expected effect

**Speculative:** `+2–8 pp` held-out-operator changed-exact and `+3–12` live-score points from fewer wasted or dangerous environment probes.

### Failure modes

- The required distinguishing state may not be reachable in an episode.
- A probe may be irreversible under one surviving hypothesis.
- The real operator may be outside `Theta`.

Include safety under every surviving hypothesis and retain an explicit unknown-mechanic hypothesis.

---

## 8. Interference-free one-shot correction theorem

**Type: EXPLORE**

This theorem would change Phase C from “take three Adam steps and rollback if needed” ([ADR 0004:550-605](/home/stepan/Coding/Personal/Tofy/docs/adr/0004-latent-planning-contract.md:550)) to a correction rule with a finite no-interference certificate.

### Statement A: exact-key override

Let memory contain exact mappings for keys `x_i`. Define:

```text
f_M(x) =
  y_i, if x equals a trusted stored key x_i
  f(x), otherwise
```

Then revisited-key error is zero and predictions outside the stored key set are unchanged. This elementary theorem already justifies factual replay memory.

### Statement B: rank-one correction without old interference

Let a residual linear head predict `W phi`. Let `S` be the span of protected old features and:

```text
r = (I - projection_S) phi_new
e = y_new - W phi_new
```

If `r != 0`, define:

```text
DeltaW = e r^T / ||r||^2
```

Then:

```text
(W + DeltaW) phi_new = y_new
DeltaW s = 0 for every s in S
```

If `r=0` and `e!=0`, no linear update can both correct the new example and leave every feature in `S` unchanged.

This supplies an exact interference test before adaptation.

### Statement C: kernel residual bound

For stored keys with Gram matrix `K` and residual matrix `E`, use:

```text
f_M(x) = f(x) + E K^-1 k(x)
```

It interpolates stored corrections when `K` is invertible. On a novel point:

```text
||f_M(x)-f(x)|| <= ||E K^-1|| * ||k(x)||
```

Thus no-interference outside memory follows from zero kernel similarity; bounded interference follows from measured similarity and conditioning.

### Precise mechanism

- Keep exact raw-key factual override as the primary path.
- Store at most 64 latent/logit residuals plus 128-dimensional retrieval keys.
- Use kernel residual retrieval for approximate revisits only when:
  - Gram conditioning is below a fixed threshold;
  - leave-one-out correction succeeds;
  - the interference bound is below the planner’s error budget.
- Permit an actual adapter update only if the rank-one nullspace test passes.
- Otherwise do not adapt; retain the transition as ambiguous memory.

At patch-4 with roughly `16*16*128` fp16 latent values, 64 full residuals occupy about 4 MiB. Retrieval costs roughly two million multiply-adds per query, negligible relative to recurrent search.

### Proof and Lean assessment

Exact-key override is trivial finite case analysis. The rank-one result needs finite-dimensional inner-product spaces and orthogonal projection; moderate Lean difficulty. The full kernel inverse theorem needs `Matrix` invertibility and norm bounds and is the hardest part, so formalize the rank-one theorem first.

### Cheap falsification

Chronological offline replay comparing:

1. exact factual cache;
2. kernel residual memory;
3. the proposed three-step adapter;
4. no adaptation.

Promotion thresholds:

- `100%` exact on nonconflicting exact-key revisits;
- no more than `1 pp` regression on novel transitions;
- explicit rejection on key conflicts or ill-conditioned Gram matrices.

### Expected effect

**Speculative:** `+5–20 pp` on the revisited-transition subset and `+2–8` live-score points. Overall changed-exact gain is bounded by the revisit frequency multiplied by the repair rate.

### Failure modes

- Visually identical keys may correspond to different hidden mechanics.
- The kernel matrix may be singular.
- Latent residual interpolation may leave the decoder manifold.
- A correction may fit pixels while damaging value estimates.

Detect raw-key conflicts, condition number, leave-one-out error, decoded exactness and value-head drift separately.

---

# Audit of the existing Lean-to-design bridge

The current formalization is valuable, but several ADR-level interpretations are stronger than what the files prove.

| Formal group | What is actually proved | Overreach to avoid |
|---|---|---|
| Identifiability | Two functions can agree on observed branches and differ at one missing pair; full table agreement implies equality ([Identifiability.lean:19-53](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/Identifiability.lean:19)) | No hypothesis-count, stochastic sample complexity, restricted-class result, or guarantee that current branch groups identify the learned class |
| Copy attractor | Lower bounds for a weighted pooled total and a rare-change uniform-average construction ([CopyAttractor.lean:24-89](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/CopyAttractor.lean:24)) | Does not prove that reweighting removes the attractor in a shared neural parameterization; does not justify applying the pooled ratio to already split means |
| CE/exactness | Pointwise argmax error implies CE at least `log 2`; any-pixel error is bounded by summed CE ([CrossEntropy.lean:21-90](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/CrossEntropy.lean:21)) | Does not directly cover split means, unimix, copy-gate composition or checkpoint selection |
| Separation | Zero hinge loss separates a covered pair; action-independent equal displacement pays the margin ([Separation.lean:18-53](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/Separation.lean:18)) | Does not imply correct dynamics, semantic action grounding or inverse-cycle trustworthiness |
| Policy | One observation with two goals and distinct unique optimal actions defeats a goal-blind policy ([Policy.lean:16-30](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/Policy.lean:16)) | Not a history-sufficiency or observation-aliasing theorem |
| Greedy | Uniform value error below half the true action gap preserves a one-state unique maximizer ([Greedy.lean:17-46](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/Greedy.lean:17)) | Does not establish multistep planning safety, learned-value calibration or hidden-goal correctness |
| Marginal blindness | Replacing an action-conditioned representation with a state-only map of the same pushforward leaves any marginal regularizer unchanged ([MarginalBlindness.lean:18-54](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/MarginalBlindness.lean:18)) | Does not formalize the finite-sample EP statistic or show that such a replacement can preserve prediction loss |
| Symmetrization | Averaging a predictor over a finite group is equivariant and no worse under invariant data and convex output loss ([Symmetrization.lean:26-79](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/Symmetrization.lean:26)) | Does not prove the averaged predictor belongs to the implemented model class, that SGD finds it, or that augmentation improves finite-sample generalization |
| Mode optimality | A mode maximizes categorical point accuracy; one constructed Huber example selects a non-modal output ([ModeOptimality.lean:21-132](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/ModeOptimality.lean:21)) | The Huber counterexample is not a general theorem that every regression decoder is suboptimal |
| Locality | Abstract local operators compose with additive radius bounds ([Locality.lean:19-62](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/Locality.lean:19)) | It is not connected to the actual Candle architecture or proof that the operator families satisfy the assumed radius |

The most consequential mismatch is the copy-attractor bridge: the research argument derives a pooled per-pixel correction, while the implementation uses conditional means and then applies the correction again. That is a design bug candidate, not merely a missing proof.

# Recommended theorem order

If only three theorem groups are added next, I would choose:

1. `SplitWeighting.lean`: prove the pooled-versus-split normalization identities and audit every loss reduction.
2. `PlanningSelection.lean`: fixed-path error, candidate-union amplification, margin certificate and exact-edge repair.
3. `HistoryCeiling.lean`: finite fiber-mode ceiling, refinement monotonicity and explicit counterexamples where no fixed history suffices.

Those three would immediately settle a live training decision, a pre-planning safety decision and whether history deserves parameters.

Status: **DONE_WITH_CONCERNS** — the mathematical claims are finite and testable, but effect-size ranges remain speculative until the proposed checkpoint rescoring and matched short runs are performed. No repository files were modified.
