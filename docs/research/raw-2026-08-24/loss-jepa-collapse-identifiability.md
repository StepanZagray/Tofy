All research complete. Here are the findings.

---

# TOPIC 3 — JEPA collapse theory and what SIGReg guarantees

## 3.1 LeJEPA (Balestriero & LeCun, 2025, arXiv:2511.08544)

**Exact claims found (v3 HTML):**
- **Theorem 1**: "among distributions with a scalar-based covariance constraint, the isotropic Gaussian is the unique minimizer of the integrated square bias" for k-NN and kernel probes. Assumption: fixed total-variance constraint (Tr Cov(Z)=κ₁ or ‖Cov(Z)‖_F=κ₂). Conclusion: isotropic Gaussian embeddings minimize *worst-case downstream prediction risk over arbitrary tasks* — an optimality statement about *distributional shape*, conditioned on embeddings already carrying task information.
- **Lemma 3** (hyperspherical Cramér–Wold): ⟨u,X⟩ =ᵈ ⟨u,Y⟩ ∀u∈𝕊^{d−1} ⟺ X =ᵈ Y — justifies random-projection sketching (your 1024 projections).
- **Theorem 4**: the Epps–Pulley statistic has bounded loss/gradient/curvature (|∂EP/∂zᵢ| ≤ 4σ²/N) — a *trainability* result, nothing about content.
- **What it provably prevents**: total and dimensional/rank collapse (any degenerate covariance violates isotropy).
- **What it does NOT guarantee**: there is no theorem that SIGReg-satisfying embeddings are informative about *anything*. Informativeness comes only from the prediction loss; the loss↔downstream-accuracy link (Sec. 6.2) is explicitly empirical. **Critically for your case: an encoder whose latent displacement is action-independent can satisfy SIGReg exactly** — a Gaussian latent driven purely by state/content is still an isotropic Gaussian. SIGReg is a constraint on the marginal p(z), and actions never appear in it.

## 3.2 When does LeJEPA identify anything — Klindt, LeCun & Balestriero, "When Does LeJEPA Learn a World Model?" (2026, arXiv:2605.26379)

- **Theorem 1**: in a Gaussian world with Ornstein–Uhlenbeck transitions, any h with h(z)∼N(0,Iₙ) satisfies ℒ(h) ≥ 2(1−ρ)n with equality iff h(z)=Qz, Q orthogonal — LeJEPA is *linearly identifiable* up to rotation. **Theorem 2**: this only works if the world's latents are Gaussian. **Theorem 4**: linear-identifiable latents give identical optimal control trajectories.
- **Explicit scope limitation (Sec. D.2, "encoder versus action-conditioned dynamics")**: the theory covers the *encoder only*; it says nothing about action-conditioned p(z′|z,a), expert-data regimes, or one-action-per-state. So even the best available LeJEPA identifiability result is silent on action sensitivity.

## 3.3 Follow-ups / critiques (2026)

- **Weak-SIGReg** (Akbar, 2026, arXiv:2603.05924): argues only second-moment matching matters in practice ("Soft Batch Normalization"); makes no informativeness claim for full SIGReg either.
- **Var-JEPA** (arXiv:2603.20111), **Rectified LpJEPA** (arXiv:2602.01456), **Beyond Isotropy in JEPAs: Hamiltonian/symplectic prediction** (arXiv:2605.20107), **SIGReg as Variational Free Energy** (arXiv:2607.13612) — reinterpretations; none add an action-sensitivity guarantee.
- **A Generalization Theory for JEPA-Based World Models** (Cui et al., arXiv:2606.27014): approximation/sample-error trade-off in latent dimension; again no action-usage guarantee.
- I found no paper literally named "TC-JEPA"/"temporal centering"; the closest collapse-by-centering discussion is the DINO centering lineage, and the closest "Gaussian marginals achievable by trivial encoders" statement in print is ActSWM's "context collapse" (below).

## 3.4 Earlier collapse theory

- **Jing et al., "Understanding Dimensional Collapse in Contrastive SSL"** (ICLR 2022, arXiv:2110.09348) — dimensional collapse mechanics.
- **Sobal et al., "JEPAs Focus on Slow Features"** (2022, arXiv:2211.10831) — VICReg/SimCLR-trained JEPA world models latch onto slow/persistent features and *fail when distractor noise is fixed*; the direct ancestor of "predictor extrapolates from context, ignores the conditioning signal."
- **"How JEPA Avoids Noisy Features"** (NeurIPS 2024, arXiv:2407.03475) — deep linear self-distillation JEPAs have an implicit *low-rank* bias toward high-influence features; regularizing the marginal does not change *which* features are kept.

## 3.5 LeJEPA-for-world-models

- **LeWM / LeWorldModel** (Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026, arXiv:2603.19312): exactly your recipe (action-conditioned JEPA + SIGReg, pixels, offline). **It contains no action-sensitivity analysis at all**, and — key contrast with your setup — trains on data that "may be pseudo-expert or exploratory, as long as they sufficiently cover the environment dynamics." Follow-up: **Fast-LeWM** (arXiv:2606.26217).
- **ActSWM** (Gan et al., 2026, arXiv:2607.26712) names your exact failure mode **"context collapse"**: "autoregressive latent predictors maintain high similarity to future states while producing nearly indistinguishable futures under different action sequences" — high prediction fidelity, dead action channel, *while using SIGReg* (adopted from LeWM). This is the published confirmation that SIGReg permits action-independent dynamics.
- **ATM** (Chen, 2026, arXiv:2606.09028): a diagnosis tool (action-consistency transfer matrix) reporting that JEPA/SIGReg-regularized world models "ignore actions"; proposes an action-consistency loss.

---

# TOPIC 4 — Action-sensitivity objectives and identifiability

## 4.1 Objectives that force action dependence

- **ActSWM** (arXiv:2607.26712): (i) hinge loss ℓ_k = max(0, cos(ẑ^gt_{t+k}, ẑ^0_{t+k}) − (1−m)), m=0.3, separating rollouts under recorded vs. zeroed actions over a K-step horizon; (ii) a **frozen action readout** q_φ₀([z_t,z_{t+1}]) → a_t, gradients flowing only into the latents (Lipschitz argument, App. E: correct readout through a fixed map ⇒ latent transitions separated by positive margin). Needs only recorded-vs-zero-action counterfactual pairs, which are free — no multi-action data required.
- **Sensorimotor World Models / SMWM** (Ivashkov, Balestriero, Schölkopf, 2026, arXiv:2606.20104): ℒ = ℒ_fwd + λ·ℒ_inv with ℒ_inv = E‖â_t − a_t‖²; claims **inverse dynamics alone prevents collapse** and biases encoders to "track the controllable degrees of freedom … filter out uncontrollable distractors." Caveat stated in the paper: assumes "actions are recoverable from consecutive observations" — fails when distinct actions produce identical visible changes (relevant near walls/no-ops in grids).
- **ATM** (arXiv:2606.09028): action-consistency loss + transfer-matrix regularization as a diagnostic-turned-objective.
- **DiLA** (arXiv:2605.15725): dual-pathway structure/content bottleneck; inverse-temporal symmetry (reversed sequence ⇒ negated latent action).
- **Causal-JEPA** (Nam, Le Lidec, Maes, LeCun, Balestriero, 2026, arXiv:2602.11389): synthesizes "latent interventions with counterfactual-like effects" by object-slot masking — **no multi-action data required**; Theorem 1: predictors ignoring the influence neighborhood cannot attain minimal error.
- Also: **Dueling World Models** (arXiv:2608.06706, advantage-style action channels), **Factorized Latent Dynamics for Video JEPA** (arXiv:2605.17165, empirical auxiliary-objective study).

## 4.2 Identifiability with one expert action per state — the precise negative results

- **"On the Identifiability of Controlled World Models"** (Zhang et al., 2026, arXiv:2607.22430). The central paper for your question.
  - **Theorem 1**: latent state and controlled conditional mean identifiable up to orthogonal transform iff representation margin γ_rep(π)>0 **and transition margin ρ_tr(π)>0**, where ρ_tr(π) = λ_min(E_z[Cov_π(a|z)]) — i.e., **actions must retain variance conditional on the state**.
  - **Theorem 3 / Corollary 1 (impossibility)**: "At σ = 0, the behavior policy is deterministic given the state, and the controlled transition is not identifiable outside its support. Distinct continuous predictors attain the same on-policy risk while producing different counterfactual predictions" — **structural non-identifiability for exactly your data regime (one expert action per state)**.
  - **Error amplification**: predictors with on-policy risk δ can have counterfactual error δ/ρ_tr(π); the factor is 1/σ² and **diverges in the deterministic-policy limit**. So your model's weak action usage is not (only) an optimization artifact — with deterministic-expert data, an action-ignoring predictor is *loss-equivalent* to the true dynamics.
- **Schur, "Identifying Latent Actions and Dynamics from Offline Data via Demonstrator Diversity"** (2026, arXiv:2603.17577). **Proposition 4.1**: "observation-only data from a single behavior policy are not sufficient to identify latent actions and dynamics" (explicit counterexample); "action choice and environment stochasticity are confounded." Positive results (Thms 4.1–4.3): multiple demonstrators with "sufficiently scattered" policies restore identifiability up to permutation.
- Classical/adjacent grounding: **pessimistic model-based offline RL under partial coverage** (Uehara & Sun, arXiv:2107.06226) and **OPE under weak distributional overlap** (arXiv:2402.08201) — coverage/concentrability of the *evaluation policy's* state-action distribution is the standard positivity condition; a deterministic expert gives support on a measure-zero action slice, so any off-expert action query is extrapolation.
- Counterfactual augmentation without new environment interaction: **CAIAC — Causal Action Influence Aware Counterfactual Data Augmentation** (arXiv:2405.18917) swaps causally independent components across trajectories to manufacture same-state alternative outcomes; **Iterative Counterfactual Data Augmentation** (arXiv:2502.18249).

---

# Direct answers

**(a) Does Epps–Pulley/SIGReg permit an action-independent latent?** Yes, provably in effect. SIGReg constrains only the marginal law of z (Cramér–Wold over projections); no theorem in LeJEPA (2511.08544) or its follow-ups links isotropy to action sensitivity, and Klindt et al. (2605.26379) explicitly exclude action-conditioned dynamics from scope. ActSWM (2607.26712) documents the resulting "context collapse" empirically *in SIGReg-regularized JEPA world models*: whenever state history predicts the expert's next state well (which is exactly the one-expert-action-per-state case, where a=π(s) is a deterministic function of state), the predictor can achieve near-optimal forward loss and perfect SIGReg while its action channel is dead.

**(b) Minimal data condition for action-effect identifiability?** Conditional action variance given state: ρ_tr(π) = λ_min(E_z[Cov_π(a|z)]) > 0 (2607.22430, Thm 1), equivalently ≥2 distinct actions with positive probability from (a positive-measure set of) states — via a stochastic/perturbed policy, multiple diverse demonstrators (2603.17577, Thms 4.1–4.3), or explicit same-state counterfactual branches. With exactly one expert action per state, action effects are structurally confounded with state (2607.22430 Cor. 1; 2603.17577 Prop. 4.1), and counterfactual error is unbounded relative to on-policy error.

# Ranked objective changes supported by the literature

1. **Fix the data, not just the loss: add counterfactual action branches** — same-state multi-action groups via environment resets, ε-noised expert, or CAIAC-style causal swaps (2607.22430; 2603.17577; 2405.18917). This is the only change that removes the *identifiability* obstruction; every loss below only reshapes solutions within the observed support.
2. **Inverse-dynamics auxiliary loss** ℒ_inv = ‖q(z_t,z_{t+1}) − a_t‖² (SMWM, 2606.20104) — cheapest effective fix; forces action-discriminative features into z and alone prevents collapse; combine with (not replace) SIGReg. Caveat: needs actions recoverable from observation pairs (grid no-ops/walls break this).
3. **Zero-action (or shuffled-action) rollout hinge** ℓ_k = max(0, cos(ẑ^{gt}, ẑ^{0}) − (1−m)) over multi-step rollouts, plus a frozen action readout (ActSWM, 2607.26712) — directly targets context collapse, needs no new data, works on offline trajectories.
4. **Action-consistency/transfer-matrix regularization** (ATM, 2606.09028) — use its diagnostic to quantify your current action usage, and its loss if 2–3 are insufficient.
5. **Synthetic latent interventions** via structured masking (Causal-JEPA, 2602.11389) — counterfactual-like pressure without action-branch data; natural if your grid latent is object/slot-structured.
6. **Multi-step rollout training** (ActSWM; Fast-LeWM 2606.26217) — one-step prediction on expert data maximally rewards state-only extrapolation; longer horizons amplify action divergence.
7. Keep SIGReg as-is — no literature says to remove it; Weak-SIGReg (2603.05924) suggests at most that covariance-only matching would suffice for stability. The problem is orthogonal to it.

Sources: [arXiv:2511.08544](https://arxiv.org/abs/2511.08544), [arXiv:2605.26379](https://arxiv.org/html/2605.26379v1), [arXiv:2607.26712](https://arxiv.org/html/2607.26712v1), [arXiv:2603.19312](https://arxiv.org/html/2603.19312v1), [arXiv:2606.26217](https://arxiv.org/html/2606.26217v1), [arXiv:2606.20104](https://arxiv.org/html/2606.20104v1), [arXiv:2606.09028](https://arxiv.org/pdf/2606.09028), [arXiv:2607.22430](https://arxiv.org/html/2607.22430v2), [arXiv:2603.17577](https://arxiv.org/html/2603.17577v2), [arXiv:2602.11389](https://arxiv.org/html/2602.11389v1), [arXiv:2605.15725](https://arxiv.org/html/2605.15725v1), [arXiv:2603.05924](https://arxiv.org/html/2603.05924v1), [arXiv:2606.27014](https://arxiv.org/html/2606.27014v1), [arXiv:2407.03475](https://arxiv.org/abs/2407.03475), [arXiv:2211.10831](https://arxiv.org/abs/2211.10831), [arXiv:2110.09348](https://arxiv.org/abs/2110.09348), [arXiv:2405.18917](https://arxiv.org/html/2405.18917v1), [arXiv:2502.18249](https://arxiv.org/abs/2502.18249), [arXiv:2107.06226](https://arxiv.org/pdf/2107.06226), [arXiv:2402.08201](https://arxiv.org/pdf/2402.08201), [arXiv:2605.17165](https://arxiv.org/pdf/2605.17165), [arXiv:2608.06706](https://arxiv.org/html/2608.06706), [arXiv:2607.13612](https://arxiv.org/html/2607.13612v1), [arXiv:2603.20111](https://arxiv.org/pdf/2603.20111), [arXiv:2605.20107](https://arxiv.org/pdf/2605.20107), [arXiv:2602.01456](https://arxiv.org/html/2602.01456v1)