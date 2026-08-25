# Literature review: verification-style planning + amortization vs deliberation for Tofy

Note on one source: I could not extract numbers from "Explore Before You Solve" (PDF garbled); everything else below is verified against abstracts/HTML.

---

## Topic A — Verification-style planning with learned models

### A1. Uncertainty-aware MPC / conservative planning

**COPlanner** — arXiv [2310.07220](https://arxiv.org/abs/2310.07220), ICLR 2024. Uncertainty-aware policy-guided MPC (UP-MPC) does multi-step uncertainty estimation via ensemble disagreement; the estimate is used as a **penalty during model rollouts** (conservative) and a **bonus during real-environment action selection** (optimistic exploration). Plug-and-play over Dyna-style MBRL; significant sample-efficiency and asymptotic gains on proprioceptive + visual control. *Tofy verdict: the two-sided use (penalize in imagination, reward in exploration) maps directly onto Tofy's explore-then-plan loop; ensemble of a 1M-param JEPA is nearly free on an A40.*

**MACURA ("Trust the Model Where It Trusts Itself")** — arXiv [2405.19014](https://arxiv.org/abs/2405.19014), ICML 2024. Uses probabilistic-ensemble uncertainty to **adapt rollout length per state**: terminate imagined rollouts where the model is uncertain, roll longer where certain. Beats MBPO/M2AC in data efficiency and asymptote. *Tofy verdict: highly feasible — this is the principled version of "short horizons," giving state-dependent plan-horizon truncation instead of a fixed cap.*

**ELVIS** — arXiv [2605.04709](https://arxiv.org/pdf/2605.04709), May 2026. Dreamer-style latent MPC with a Gaussian-mixture MPPI (keeps multiple hypotheses, avoids mode averaging) and an **ensemble of latent critics whose confidence bounds gate a time-varying λ trading bootstrapping vs look-ahead** to limit compounding error. SOTA vs TD-MPC2/DreamerV3 on 14 visual DMC tasks. *Tofy verdict: the critic-ensemble gating idea (trust look-ahead only where value ensemble agrees) transfers; the visual-control machinery is heavier than needed.*

Older but load-bearing lineage: PETS-style probabilistic ensembles remain the standard epistemic signal in all of the above; nothing in 2024–26 has displaced disagreement-based penalties.

### A2. Model exploitation / planner hacks the model

**Imperfect World Models are Exploitable** — arXiv [2605.15960](https://arxiv.org/html/2605.15960v1), May 2026 (Bhamidipaty, Abel, Kochenderfer, Ramamoorthy). Formalizes exploitation as the model **mis-ranking policies** relative to the environment; proves it is **essentially unavoidable on large policy sets** and, unlike reward hacking, no condition eliminates it entirely; introduces a relaxed notion with a **calculable "safe horizon"** for risk mitigation. *Tofy verdict: directly explains the failure mode you should expect — with ~4000 actions the multi-step policy set is astronomically large, so an unpenalized optimizer over the JEPA **will** find adversarial action sequences. The safe-horizon construction is the theoretical justification for horizon capping.*

**TD-M(PC)²** — arXiv [2502.03550](https://arxiv.org/abs/2502.03550), Feb 2025. Documents a concrete exploitation mechanism inside TD-MPC2: planner-generated data vs learned policy prior mismatch → value function queried on **OOD actions → persistent value overestimation**. Fix: a policy-regularization term reducing OOD queries; no extra compute; large gains on 61-DoF humanoid. *Tofy verdict: cheap, high-value lesson — keep the planner trust-regioned toward the policy prior / training distribution.*

Established mitigation set across these works: ensemble penalties, short/adaptive horizons, frequent replanning, trust regions toward the prior. All are compatible with Tofy's 0.1–2 s budget.

### A3. Retrodiction / cycle-consistency plan verification (hottest 2026 cluster)

**ACID: Action Consistency via Inverse Dynamics** — arXiv [2607.02403](https://arxiv.org/abs/2607.02403), Jul 2026. **Decision-time** plan filter: an inverse-dynamics model infers the action backward from each predicted transition; mismatch with the conditioned action is added to the planning cost (adaptive scale-invariant weighting). Across 4 action-conditioned world models × 6 tasks it consistently improves planning and **matches baseline accuracy with substantially less planning compute**. *Tofy verdict: the single most drop-in item in this review. Tofy already has (or can cheaply add) an inverse-dynamics head; the check is one extra forward pass per plan step and directly targets "convincing but unrealizable" trajectories.*

**World Action Verifier (WAV)** — arXiv [2604.01985](https://arxiv.org/abs/2604.01985) / [OpenReview](https://openreview.net/forum?id=n2hKzcpBFW), Apr 2026 (Liu, Murphy, Finn). Training-time self-improvement: subgoal generator + sparse inverse model + forward rollout, verifying forward-inverse cycle consistency; exploits the asymmetry that verification is easier than prediction. Targets errors in **under-explored (suboptimal) action regimes**. Nine tasks incl. **MiniGrid** (discrete grid — closest to Tofy's regime): **~2× sample efficiency, >22% downstream policy improvement**. *Tofy verdict: strong fit as a training-loop augmentation for the 4000-action space, where most actions are never taken by good policies.*

**WorldCycle** — arXiv [2608.04964](https://arxiv.org/abs/2608.04964), Aug 2026. RL fine-tuning of video world models with closed action cycles: spatial closure reward (forward vs mirrored reverse) + temporal consistency across repeated cycles. **Reduces state-return drift up to 44%; ~4× composite-action accuracy.** *Tofy verdict: idea transfers (deterministic games make cycle tests exact), but the video-model RL machinery is oversized for a 1M-param latent model.*

Also relevant: **Behavior Consistency in text world models** (arXiv [2604.13824](https://arxiv.org/html/2604.13824v1)) and **ATM action-consistency diagnosis** (arXiv [2606.09028](https://arxiv.org/pdf/2606.09028)) — evidence that "state-level accuracy ≠ plan-level reliability" is now a recognized evaluation axis.

One Tofy-specific observation the literature supports indirectly: because ARC-AGI-3 is **deterministic**, replaying stored real transitions gives an *exact* retrodiction oracle — any plan prefix overlapping past experience can be verified for free, and all model error is epistemic. None of these papers get that luxury; Tofy should exploit it.

### A4. JEPA-specific uncertainty (2025–26)

Thin but existent:
- **VJEPA (Variational JEPA as Probabilistic World Models)** — arXiv [2601.14354](https://arxiv.org/abs/2601.14354), Jan 2026 (single-author, Huang). Predictive *distribution* over future latents via variational objective; credible intervals by sampling; likelihood-free in observation space. Experiments are toy (noisy-distractor filtering). *Verdict: right idea, weak empirical validation — treat as design inspiration.*
- **EB-JEPA lightweight library** — arXiv [2602.03604](https://arxiv.org/html/2602.03604v2), Feb 2026 (Terver et al., LeCun group). Frames JEPA prediction **energy as a compatibility score** — usable as a reliability signal, but not validated as a planning filter.
- **Value-guided action planning with JEPA world models** — arXiv [2601.00844](https://arxiv.org/abs/2601.00844), Dec 2025 (LeCun group, Mila World Modeling Workshop 2026). Constrains latent geometry so embedding distance ≈ negative goal-conditioned value; significantly improves JEPA planning on simple control. No uncertainty handling.

*Conclusion for A4: there is no proven JEPA-native uncertainty method yet. The evidence-backed route for Tofy is a small ensemble (3–5 seeds of the 1M model — trivially affordable) with disagreement penalties per A1, optionally augmented by prediction-energy as a secondary reliability score.*

---

## Topic B — Amortization vs deliberation

### B1. MuZero/AlphaZero evidence

**On the role of planning in model-based deep RL** — arXiv [2011.04021](https://arxiv.org/pdf/2011.04021), Hamrick et al., ICLR 2021. MuZero ablations across control/Atari/9×9 Go: search's dominant value is in **constructing policy-learning targets and shaping the data distribution**; shallow/simple planning often suffices; **eval-time search adds little unless the model is highly accurate** — and helps most in Go (deterministic, perfect information). *Tofy relevance: the caveat cuts your way — deterministic hidden-rule games are exactly the regime where their eval-time-search benefit survived.*

**Policy improvement by planning with Gumbel** (Danihelka et al., ICLR 2022; see [thesis](https://discovery.ucl.ac.uk/id/eprint/10167022/2/ivo_danihelka_thesis.pdf) and [MiniZero replication, arXiv 2310.11305](https://arxiv.org/pdf/2310.11305)). Gumbel-Top-k root sampling + sequential halving gives **guaranteed policy improvement even at tiny simulation budgets (down to n=2)** and is specifically strongest with **large action spaces and small budgets**; matches MuZero SOTA on Go/chess/Atari with far fewer simulations. *Tofy verdict: this is the search algorithm that matches Tofy's constraints almost exactly — 4000 actions, tens-not-thousands of simulations per 0.1–2 s step.*

### B2. MPC vs distilled policy at matched compute

- **TD-MPC2** — arXiv [2310.16828](https://arxiv.org/pdf/2310.16828): notably, the paper itself **does not report a clean planner-vs-policy-prior ablation** (I checked the HTML; ablations cover normalization/objectives/ensembles).
- **TD-M(PC)²** (above) supplies the missing evidence indirectly: the planner materially outperforms and *mis-trains* the prior on high-DoF tasks unless regularized — i.e., deliberation > current amortization on hard tasks, and the gap is a data-distribution artifact, not fundamental.
- **BMPC (Bootstrapped Model Predictive Control)** — arXiv [2503.18871](https://ui.adsabs.harvard.edu/abs/2025arXiv250318871W/abstract), ICLR 2025. Expert iteration for MPC: policy imitates the MPC expert (lazy reanalyze for cheap imitation), improved policy guides MPC, model-based TD improves values. Beats TD-MPC2 on high-dim locomotion with **smaller networks and comparable training time**. *Verdict: the best current template for Tofy's "self-contained policy" requirement — search at training time, distill continuously, keep cheap search at inference.*
- **Counterpoint — GC-IDM ("Latent Geometry Beyond Search")** — arXiv [2605.08732](https://arxiv.org/html/2605.08732), Jun 2026. A **1.5M-param** goal-conditioned inverse-dynamics MLP over frozen world-model embeddings replaces CEM search entirely: matches/exceeds CEM in 7/8 cells at **100–130× less compute** (e.g., 98.7% vs 67% on OGBench-Cube). *Verdict: shows amortization can crush search **when the latent geometry is well structured and the task is goal-reaching in-distribution**. ARC-AGI-3's hidden novel rules are the opposite regime at first contact — but this supports distilling per-game once the model is trusted.*

### B3. Small models + search (test-time compute for tiny nets)

- **Scaling Scaling Laws with Board Games** — arXiv [2104.03113](https://arxiv.org/pdf/2104.03113), Andy Jones, 2021. On Hex: **each 10× of train-time compute substitutes for ~15× of test-time compute**, frontier ~500 Elo per order of magnitude; a **~500k-param net (2×512) plus modest MCTS achieves perfect play on 9×9 Hex**, on par with MoHex, ~3 h on one GPU. *This is the single strongest datapoint that a Tofy-scale (~1M) network plus search can reach ceiling performance on small deterministic grid games.*
- **Scaling Laws for a Multi-Agent RL Model** — arXiv [2210.00849](https://arxiv.org/pdf/2210.00849), Neumann & Gros, 2022/23. AlphaZero on Connect Four/Pentago: Elo scales as a power law in parameters and in compute, identical exponents across games; larger models more sample-efficient. Confirms smooth capacity/compute exchange rather than capability cliffs at small scale.
- **Sokoban DRC interpretability** — arXiv [2407.15421](https://arxiv.org/html/2407.15421) and **Interpreting Emergent Planning in Model-Free RL**, arXiv [2504.01871](https://arxiv.org/abs/2504.01871), ICLR 2025. Even *model-free* recurrent nets trained on Sokoban internally learn bidirectional-search-like planning, **solve more levels given extra test-time compute**, and DRC agents outperform MuZero on Sokoban. *Verdict: in deterministic discrete puzzles, deliberation (explicit or emergent) demonstrably converts extra inference compute into solved instances at small scale.*

### B4. The deterministic-discrete sweet spot

Direct statements are scattered rather than one canonical paper, but the pattern is consistent: Hamrick et al. found eval-time search helps precisely where models are accurate (Go); MuZero's strengths are documented as discrete-action, deterministic domains with stochastic/continuous settings remaining open problems ([MuZero model-interpretation work](https://www.themoonlight.io/en/review/interpreting-the-learned-model-in-muzero-planning)); and the 2026 **ARC-AGI-3 literature itself** converges on verify-then-plan: **Executable World Models for ARC-AGI-3** (arXiv [2605.05138](https://arxiv.org/html/2605.05138v2)) builds a Python world model, **verifies it against observed transitions**, then plans through it before spending scored actions; **Tycho** (arXiv [2607.28287](https://arxiv.org/abs/2607.28287)) builds programmatic per-game models under an action-efficiency budget. Also **Explore Before You Solve** (arXiv [2605.25931](https://arxiv.org/pdf/2605.25931)) studies the speed-vs-depth trade-off on ARC-AGI-3 directly (numbers not extractable from the PDF I fetched).

---

## Bottom line for Tofy

1. **Both topics land on the same recipe, and it is feasible on one A40 at 0.1–2 s/action:** small JEPA ensemble (3–5 seeds) for disagreement-penalized, adaptively-truncated rollouts (COPlanner/MACURA), an ACID-style inverse-dynamics cycle-consistency term as a per-step plan filter, and exact replay-retrodiction against stored transitions (free, unique to deterministic games). Expect exploitation without these — it is provably generic (2605.15960), and Tofy's huge action space maximizes exposure.
2. **Keep deliberation at test time; amortize continuously.** For ~1M-param models in deterministic discrete domains, evidence (Jones; Sokoban DRC; Hamrick's Go caveat) says search buys real capability — roughly a 10×-train ≈ 15×-test exchange rate. Use Gumbel-style sequential halving to make a 4000-action root search fit tens of simulations. Distill the planner into the policy BMPC-style (with TD-M(PC)²'s OOD regularization) so the self-contained policy inherits search gains; GC-IDM suggests full amortization only becomes safe once the per-game latent geometry is verified good.
3. **JEPA-native uncertainty is not yet proven** (VJEPA is early-stage; energy-as-reliability is untested for planning) — ensembles remain the evidence-backed choice at this scale.

Sources: [2310.07220](https://arxiv.org/abs/2310.07220), [2405.19014](https://arxiv.org/abs/2405.19014), [2605.04709](https://arxiv.org/pdf/2605.04709), [2605.15960](https://arxiv.org/html/2605.15960v1), [2502.03550](https://arxiv.org/abs/2502.03550), [2607.02403](https://arxiv.org/abs/2607.02403), [2604.01985](https://arxiv.org/abs/2604.01985), [2608.04964](https://arxiv.org/abs/2608.04964), [2604.13824](https://arxiv.org/html/2604.13824v1), [2601.14354](https://arxiv.org/abs/2601.14354), [2602.03604](https://arxiv.org/html/2602.03604v2), [2601.00844](https://arxiv.org/abs/2601.00844), [2011.04021](https://arxiv.org/pdf/2011.04021), [2310.11305](https://arxiv.org/pdf/2310.11305), [Danihelka thesis](https://discovery.ucl.ac.uk/id/eprint/10167022/2/ivo_danihelka_thesis.pdf), [2310.16828](https://arxiv.org/pdf/2310.16828), [2503.18871](https://ui.adsabs.harvard.edu/abs/2025arXiv250318871W/abstract), [2605.08732](https://arxiv.org/html/2605.08732), [2104.03113](https://arxiv.org/pdf/2104.03113), [2210.00849](https://arxiv.org/pdf/2210.00849), [2407.15421](https://arxiv.org/html/2407.15421), [2504.01871](https://arxiv.org/abs/2504.01871), [2605.05138](https://arxiv.org/html/2605.05138v2), [2607.28287](https://arxiv.org/abs/2607.28287), [2605.25931](https://arxiv.org/pdf/2605.25931)