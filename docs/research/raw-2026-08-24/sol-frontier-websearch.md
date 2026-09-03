The central diagnosis is sharper than “the model needs a better recurrent architecture”:

1. Training on one expert action per state makes counterfactual action effects statistically unidentified.
2. Huber prediction in a learned latent space is poorly aligned with exact discrete grid reconstruction.
3. A single visible frame may not be Markov for ARC-AGI-3.
4. Even perfect one-step dynamics would not, by itself, produce a strong ARC-AGI-3 score: the current greedy policy still lacks goal inference, progress estimation, risk, and purposeful exploration.

The strongest near-term program is therefore: branch-complete synthetic data → direct categorical transition supervision → history-conditioned state → explicit action-effect losses → safe online adaptation and one-step goal/progress heads. More recurrence and optimizer tuning come later.

## 1. ARC-AGI-3 agents, results, and action-effect learning

The environment paper, [“ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence”](https://arxiv.org/abs/2603.24621) (arXiv:2603.24621, March 24, 2026), defines ARC-AGI-3 as an interactive benchmark requiring agents to infer controls, dynamics, goals, and strategies from sparse feedback. Humans achieved 100%, while frontier systems were below 1% at release.

The accompanying [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) describes 25 public-demo, 55 semi-private, and 55 fully private environments. Public tasks are deliberately easier and nonrepresentative. The reported release-time semi-private scores were:

- Claude Opus 4.6 Max: 0.50%
- Gemini 3.1 Pro Preview: 0.40%
- GPT-5.4 High: 0.20%
- Grok 4.20: 0.10%

Current numbers must be separated into incompatible regimes:

- ARC Prize’s verified [Claude Opus 5 result](https://arcprize.org/results/anthropic-claude-opus-5) reports 30.16% on ARC-AGI-3 as of July 24, 2026. The page exposes public-demo task results but does not present a separate public/semi-private split.
- The [GPT-5.6 Sol result](https://arcprize.org/results/openai-gpt-5-6) reports 13.33% on public tasks and 7.78% semi-private.
- The current [community leaderboard](https://arcprize.org/leaderboard/community) shows public-task systems near saturation: Tycho 100.0%, Retrodict 99.9%, and baseline1 99.0%. These are public-only, generally self-reported harness results—not comparable to no-harness semi-private evaluation.

The high-scoring public systems are overwhelmingly coding/VLM agents with explicit symbolic or executable state tracking, not small neural world models or model-free RL:

- [Tycho](https://github.com/NIMI-research/Tycho/) has an actor-controlled builder maintain an executable transition, rendering, and outcome model. It records observations, actions, and consequences, tests hypotheses against history, deliberately probes uncertain mechanics, and replans after every action.
- [Retrodict](https://github.com/ryanbbrown/Retrodict) reports 99.86% across 183 public levels, using 7,703 actions. Its defining constraint is that a proposed rule must retrodict the complete accumulated interaction history before it is trusted.
- [“Executable World Models for ARC-AGI-3 in the Era of Coding Agents”](https://arxiv.org/abs/2605.05138) (arXiv:2605.05138, May 6, 2026) reports an initial 32.58% public score and later 58.12% with GPT-5.5. It constructs Python transition models, verifies them by exact replay, prefers simpler consistent models, and plans through the resulting simulator.
- [“Do Coding Agents Need Executable World Models for Interactive Reasoning?”](https://arxiv.org/abs/2607.15439) (arXiv:2607.15439, July 16, 2026) finds that explicit verification consistently helps, although flexible executable models are not universally superior once model strength and reasoning effort are controlled.
- [“Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks”](https://arxiv.org/abs/2512.24156) (arXiv:2512.24156, December 30, 2025) builds an exact directed graph of observed states and actions and routes the agent toward untested state–action pairs. It solved a median 30/52 preview levels and placed third on the private preview.

The [ARC Prize 2026 Milestone 1 write-up](https://arcprize.org/blog/arc-prize-2026-milestone-1) likewise emphasizes executable reasoning, multimodal segmentation, reflection memory, rare-object interaction heuristics, and suppression of repeatedly unproductive actions. It does not disclose comparable final scores for those systems.

The common winning recipe is:

1. Store every factual transition.
2. Track which state–action combinations remain untested.
3. Prefer probes that discriminate among competing hypotheses.
4. Reject models that fail to replay earlier transitions exactly.
5. Replan immediately after every observation.
6. Separate “I can predict this action” from “this action advances the inferred goal.”

For Tofy, exact executable induction is infeasible at 560K parameters, but its principles are transferable: factual transition memory, branch coverage, falsification losses, novelty signatures, and explicit goal/progress prediction.

## 2. Recursive reasoners, equilibrium models, and cellular automata

[“Hierarchical Reasoning Model”](https://arxiv.org/abs/2506.21734) (HRM, arXiv:2506.21734, June 26, 2025) proposed two recurrent modules operating at different timescales, totaling 27M parameters. The paper reported approximately 40.3% ARC-AGI-1 and 5% ARC-AGI-2 with roughly 1,000 examples per task.

The independent [ARC Prize HRM analysis](https://arcprize.org/blog/hrm-analysis) substantially qualifies the architectural claims:

- Verified semi-private scores were 32% on ARC-AGI-1 and 2% on ARC-AGI-2.
- A similarly sized transformer came within roughly five percentage points.
- The clearest benefit came from repeated outer refinement, not the claimed biological hierarchy.
- Increasing training-time refinement depth mattered more than merely running more iterations at inference.
- Cross-task transfer was limited.
- Puzzle identity, training on evaluation-task distributions, and extensive augmentation were material confounds.

[“Less is More: Recursive Reasoning with Tiny Networks”](https://arxiv.org/abs/2510.04871) (TRM, arXiv:2510.04871, October 6, 2025) simplified this to a 7M-parameter, two-layer recursive network and reported 45% ARC-AGI-1 and 8% ARC-AGI-2 public performance. Its relevance to Tofy is the evidence that weight-tied refinement can provide useful effective depth without a large parameter count—not evidence that recursion automatically discovers dynamics.

[“Test-time Adaptation of Tiny Recursive Models”](https://arxiv.org/abs/2511.02886) (arXiv:2511.02886, November 4, 2025) subsequently reported 6.67% semi-private performance after task-specific full-model fine-tuning. The base model required more than 700,000 training steps over about 48 hours on four H100s, so its result does not imply that small recurrent systems are intrinsically cheap.

Early-2026 successors include:

- [“Form Follows Function: Recursive Stem Model”](https://arxiv.org/abs/2603.15641) (RSM, arXiv:2603.15641, March 3, 2026): detached warm-up iterations, final-iteration supervision, independently variable inner and outer depth, and stochastic outer depth. It reports substantial speed gains and improved very-deep Sudoku solving.
- [“Probabilistic Tiny Recursive Model”](https://arxiv.org/abs/2605.19943) (PTRM, arXiv:2605.19943, May 19, 2026): multiple noisy recurrent trajectories followed by learned selection. Its gains are interesting, but selecting among parallel futures is effectively inference-time search and conflicts with Tofy’s policy restriction.
- [“Stability and Generalization in Looped Transformers”](https://arxiv.org/abs/2604.15259) (arXiv:2604.15259, April 16, 2026) provides a key theoretical warning: weight-tied iteration without reliable input recall has a restricted fixed-point structure. Recall connections and outer normalization make stable input-dependent equilibria much more expressive.

[“Deep Equilibrium Models”](https://arxiv.org/abs/1909.01377) (arXiv:1909.01377, 2019) established root-finding over a weight-tied layer with implicit differentiation and constant activation memory. DEQs are attractive when additional computation should refine a single answer, but convergence is not semantic correctness. A stable fixed point can be a stable wrong or action-invariant prediction.

For grids, [“Learning Locally Interacting Discrete Dynamical Systems”](https://arxiv.org/abs/2404.06460) (AR-NCA, arXiv:2404.06460, April 9, 2024) shows that autoregressive neural cellular automata can learn local discrete dynamics data-efficiently and transfer across grid sizes. But [“Stability and Geometry of Attractors in Neural Cellular Automata”](https://arxiv.org/abs/2604.12720) (arXiv:2604.12720, April 14, 2026) finds that learned NCAs often converge to periodic or quasiperiodic attractors rather than fixed points.

For Tofy:

- Distinguish temporal recurrence from inference refinement. The former carries world state; the latter repeatedly solves the same transition.
- Train across randomized refinement depths. Do not expect a model trained only at depth 2 to improve safely at depth 20.
- Preserve the observation and action through explicit recall connections at every refinement step.
- Supervise intermediate or randomly selected depths, or use RSM-style stochastic outer depth.
- Monitor convergence, action sensitivity, and exact decoded state independently.
- An NCA-like local residual path is promising, but it needs a global/action-broadcast channel for ACTION6 coordinates, counters, ordering rules, and nonlocal effects.

## 3. Objectives aligned with exact discrete state prediction

For a discrete successor \(Y\):

- Squared error estimates a conditional mean.
- Huber estimates a robust conditional center.
- Categorical cross-entropy estimates a conditional distribution; argmax yields the per-variable mode.

This distinction matters whenever the same apparent state/action maps to multiple successors because of partial observability, missing history, or multimodal data. A latent Huber objective can reduce its loss by producing an “average” representation that decodes poorly to every actual successor.

The strongest immediate objective is therefore a direct 16-way categorical loss on the predicted successor grid. Because unchanged pixels dominate, use either:

- a binary changed/unchanged gate plus a 16-way new-color head, or
- a 16-way color head with per-transition normalization and separate changed/unchanged loss budgets.

A practical starting point is 50% of loss mass on changed pixels and 50% on unchanged pixels, while keeping evaluation unweighted. Only force actions apart when their factual successors differ; equivalent actions should be allowed to share predictions.

Relevant world-model precedents:

- [“Mastering Atari with Discrete World Models”](https://arxiv.org/abs/2010.02193) (DreamerV2, arXiv:2010.02193) established categorical latent world models at scale.
- [“Mastering Diverse Domains through World Models”](https://arxiv.org/abs/2301.04104) (DreamerV3, arXiv:2301.04104) combines categorical RSSM states with symlog and two-hot classification for scalar reward/value targets across more than 150 tasks.
- [“Stop Regressing: Training Value Functions via Classification for Scalable Deep RL”](https://arxiv.org/abs/2403.03950) (arXiv:2403.03950, March 6, 2024) reports broad gains from HL-Gauss-style classification of noisy scalar targets. For Tofy, the palette is already categorical, so Gaussian smoothing across color indices would be meaningless; ordinary 16-class CE is the correct analogue.
- [“Transformers are Sample-Efficient World Models”](https://arxiv.org/abs/2209.00588) (IRIS, arXiv:2209.00588) uses discrete image tokens and autoregressive token dynamics, reaching 1.046 mean human-normalized score on Atari 100K without test-time lookahead.
- [“Efficient Stochastic Transformer-based World Models for Reinforcement Learning”](https://arxiv.org/abs/2310.09615) (STORM, arXiv:2310.09615) combines stochastic discrete representations with transformer dynamics and reports 126.7% mean human-normalized Atari 100K performance.
- [“Efficient World Models with Context-Aware Tokenization”](https://arxiv.org/abs/2406.19320) (Delta-IRIS, arXiv:2406.19320) models context-dependent delta tokens rather than repeatedly reconstructing unchanged content.
- [“Masked Generative Priors Improve World Models Sequence Modelling Capabilities”](https://arxiv.org/abs/2410.07836) (GIT-STORM, arXiv:2410.07836; revised October 2025) uses a MaskGIT-like masked-token prior.
- [“Diffusion for World Modeling”](https://arxiv.org/abs/2405.12399) (DIAMOND, arXiv:2405.12399) obtains strong Atari 100K performance using diffusion-based frame generation.

Discrete diffusion is not my first recommendation. Tofy’s targets are tiny, palette-valued, and often deterministic. Diffusion adds sampling cost and makes exact prediction harder to audit. Delta-token or masked residual modeling is a much better fit: encode only which cells changed and their new values.

Independent per-pixel CE can still produce a jointly impossible grid under hidden-state ambiguity. The correct remedy is initially more history, not a heavier generative decoder. Add patch-level or autoregressive structure only if collisions remain after the state representation is made closer to Markov.

## 4. JEPA lineage and action collapse

[“LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics”](https://arxiv.org/abs/2511.08544) (arXiv:2511.08544, November 11, 2025) derives an isotropic-Gaussian representation target under its downstream-risk assumptions and introduces SIGReg, an Epps–Pulley-inspired random-projection normality regularizer. It removes the need for EMA target encoders and stop-gradient heuristics.

[“LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architectures from Pixels”](https://arxiv.org/abs/2603.19312) (arXiv:2603.19312, March 13, 2026) extends this lineage to pixel world models using next-embedding prediction plus Gaussian regularization. It reports competitive control performance and much faster latent planning with a 15M-parameter model.

The important limitation is that marginal isotropy prevents global statistical collapse; it does not guarantee that:

- actions change the representation;
- different action effects remain separated;
- spatial identity is preserved;
- the transition retains exactly decodable color information;
- a future latent cannot be predicted using action-invariant shortcuts.

Post-March evidence makes this explicit:

- [“ActSWM: Discovering and Verifying Actions in Latent World Models”](https://arxiv.org/abs/2607.26712) (arXiv:2607.26712, July 29, 2026) describes context collapse: future embeddings remain noncollapsed globally but are nearly invariant to action. It uses transition separation and action recoverability.
- [“Delta-JEPA: Learning Action-Sensitive Representations Through Latent Displacements”](https://arxiv.org/abs/2606.31232) (arXiv:2606.31232, June 30, 2026) decodes actions from latent displacement rather than from endpoint representations alone.
- [“Temporally Centered SIGReg for World Models”](https://arxiv.org/abs/2607.26924) (arXiv:2607.26924, July 2026) argues that marginal Gaussianization can compress temporally meaningful clusters; regularizing temporal residuals instead reportedly improves LIBERO performance from 53.2% to 73.6%.
- [“Subspace-Regularized JEPA”](https://arxiv.org/abs/2605.09241) (Sub-JEPA, arXiv:2605.09241, May 2026) questions whether enforcing Gaussian structure over the entire ambient representation is appropriate when useful states occupy a lower-dimensional manifold.

The best Tofy-specific fix is an identifiable-transition auxiliary objective:

\[
\Delta z = z_{t+1}-z_t,\qquad
\hat a,\hat x,\hat y = g(\Delta z).
\]

Apply inverse action/coordinate supervision only to transitions where that information is recoverable from the outcome. Alongside it, compare factual alternatives from the same state:

\[
L_{\text{sep}}
=
w(s,a,a')
\max(0,m-d(\hat z_{s,a},\hat z_{s,a'})),
\]

where \(w=0\) for outcome-equivalent actions and grows with factual successor difference. This avoids the serious mistake of pushing two no-op or equivalent actions apart merely because their action tokens differ.

Keep SIGReg as an anti-collapse regularizer, but make exact categorical transition loss primary. A useful latent world model should be shaped by the discrete task, rather than asking an isotropic latent geometry to discover exact palette semantics incidentally.

## 5. Latent-action learning

[“Genie: Generative Interactive Environments”](https://arxiv.org/abs/2402.15391) (arXiv:2402.15391, February 23, 2024) learns latent actions from unlabeled video and uses them in an autoregressive dynamics model. [Genie 2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) (December 4, 2024) extended this to action-controllable 3D environments. [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) (August 5, 2025) reports 720p generation at 24 FPS with consistency lasting several minutes, although the public evidence remains primarily an official demonstration rather than a detailed archival paper.

The robot-learning lineage is more directly informative:

- [“Learning to Act without Actions”](https://arxiv.org/abs/2312.10812) (LAPO, arXiv:2312.10812; ICLR 2024 spotlight) learns latent actions from unlabeled videos, then grounds behavior using a smaller action-labeled set.
- [“Latent Action Pretraining from Videos”](https://arxiv.org/abs/2410.11758) (LAPA, arXiv:2410.11758; ICLR 2025) uses vector-quantized latent actions to pretrain vision-language-action models before grounding them with robot action labels.
- [“Latent Action Learning Requires Supervision in the Presence of Distractors”](https://proceedings.mlr.press/v267/nikulin25a.html) (ICML 2025) shows that unsupervised latent actions readily encode distractors. Its LAOM method improved linear-probe action quality eightfold, while only 2.5% action supervision improved downstream performance by 4.2×.

Latent-action pretraining is valuable when action labels are missing but transition video is abundant. Tofy has the opposite situation: action type and ACTION6 coordinates are already known, while data diversity is limited. Replacing known actions with an ungrounded latent code would add an unnecessary identifiability problem.

A promising compromise is a jointly grounded effect code:

- infer a discrete effect token from \((s_t,s_{t+1})\);
- require it to predict known action type and coordinates when identifiable;
- condition the forward model on both known action and inferred effect class during training;
- at inference, predict effect class from state and known action;
- maintain an explicit “outcome-equivalent action” class.

That uses the LAPO/LAPA insight—dynamics may have a simpler effect vocabulary than raw control space—without discarding the action labels.

## 6. Searchless test-time training and adaptation

Test-time training was indeed decisive for static ARC:

- The ARC-AGI-3 technical report credits TTT with the 53.5% private ARC-AGI-1 breakthrough.
- The [2025 ARC Prize competition results](https://arcprize.org/competitions/2025) report NVARC at 24% on ARC-AGI-2 private, ahead of ARChitects at 16.5% and MindsAI at 12.6%.
- The [NVARC repository](https://github.com/1ytic/NVARC) describes a Qwen3-4B system trained on approximately 103,000 synthetic puzzles and 3.2 million augmentations. Its [competition write-up](https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc) used rank-256 LoRA adaptation per puzzle.

But static ARC TTT receives several solved input/output examples before predicting the test answer. ARC-AGI-3 instead provides an online stream of factual transitions, one action at a time. The correct mapping is prequential system identification:

1. Predict the next state before observing it.
2. Execute exactly one greedy action.
3. Observe the successor.
4. Add the transition to episode memory.
5. Update a small, resettable adapter.
6. Re-evaluate all earlier episode transitions and revert if retained prediction worsens.

Relevant work includes:

- [“Test-time Adaptation of Tiny Recursive Models”](https://arxiv.org/abs/2511.02886) for task-specific recursive-model fine-tuning.
- [“Out-of-Distribution Generalization in ARC: Program Synthesis versus Test-Time Fine-Tuning”](https://arxiv.org/abs/2507.15877) (arXiv:2507.15877, July 2025), which finds program synthesis stronger under controlled compositional shifts and suggests that TTFT often elicits knowledge already represented by the base model rather than inventing missing abstractions.
- [“Learning to (Learn at Test Time)”](https://arxiv.org/abs/2407.04620) (arXiv:2407.04620, July 2024), where hidden state is itself a small model updated online by a self-supervised objective.
- [“Titans: Learning to Memorize at Test Time”](https://arxiv.org/abs/2501.00663) (arXiv:2501.00663, January 2025), which uses neural long-term memory updated during inference.
- [“Meta-Learning for Physically-Constrained Neural System Identification”](https://arxiv.org/abs/2501.06167) (arXiv:2501.06167, January 2025), supporting meta-trained rapid adaptation of a restricted system-identification subnetwork.
- [“AdaPower: Test-Time Adaptation of World Foundation Models”](https://arxiv.org/abs/2512.03538) (arXiv:2512.03538, December 2025), which reports large LIBERO gains from adaptation plus memory, although its MPC component is not compatible with Tofy’s no-search contract.

For Tofy, update only a small component at first:

- action/effect embeddings;
- low-rank transition adapters;
- per-game affine modulation;
- a compact fast-weight transition memory;
- possibly the final categorical residual head.

Freeze the encoder and core dynamics until adapter-only TTT is proven insufficient. Use one to five gradient steps per observed transition, replay all episode transitions, anchor weights to the pretrained initialization, and reset at game boundaries. Meta-train this exact adaptation procedure across synthetic games.

This remains searchless. Model updates happen between actions; the chosen action is still the argmax of one-step scores.

## 7. Muon in 2026

[“Muon is Scalable for LLM Training”](https://arxiv.org/abs/2502.16982) (arXiv:2502.16982, February 24, 2025) reports roughly 2× compute efficiency in large-language-model scaling. Its successful recipe depends on carefully calibrated per-parameter update scale and weight decay, rather than simply replacing AdamW with orthogonalized updates.

The current [PyTorch Muon documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) recommends Muon for two-dimensional hidden-layer matrices, while embeddings, normalization parameters, biases, and output heads remain on AdamW. It also exposes shape scaling intended to normalize update RMS across differently shaped matrices.

Recent theory is less universally favorable:

- [“On the Convergence of Muon and Beyond”](https://arxiv.org/abs/2509.15816) (arXiv:2509.15816, September 2025; revised March 2026) derives a comparatively weak \(O(T^{-1/4})\) stochastic convergence rate for standard Muon and proposes variance-reduced variants approaching \(O(T^{-1/3})\).
- [“Newton-Muon”](https://arxiv.org/abs/2604.01472) (arXiv:2604.01472, April 1, 2026) adds input second-moment preconditioning and reports roughly 6% fewer steps and 4% wall-clock savings on nanoGPT.
- [“On MUON Optimization: From Non-Convergence to Stable Variants”](https://arxiv.org/abs/2608.04607) (arXiv:2608.04607, August 5, 2026) constructs simple stochastic settings in which practical Muon fails to converge for almost every batch size.
- [“SOAP, Muon and Beyond”](https://arxiv.org/abs/2607.20548) (arXiv:2607.20548, July 2026) emphasizes that optimizer comparisons must match update RMS; otherwise an apparent optimizer advantage can be a step-scale artifact.
- [“Normalization–Optimizer Coupling”](https://arxiv.org/abs/2604.01563) (arXiv:2604.01563, April 2026) reports a negative interaction between Muon and a specific scale-blind saturating normalizer. Its RMSNorm control does not establish a general Muon–RMSNorm pathology.

There is no convincing published evidence that Muon is superior for a 560K recurrent convolutional model. Nor is there strong evidence that it causes this particular plateau. Missing counterfactual data and an objective mismatch are much larger suspects.

The appropriate test is a tightly matched AdamW control:

- same initialization and data order;
- same effective and physical batch;
- same clipping;
- same decoupled weight decay;
- matched layerwise update/weight RMS;
- identical checkpoint selection.

Log parameter update RMS, activation RMS, recurrence Jacobian or perturbation amplification, and categorical changed-pixel accuracy. Applying Muon to convolution kernels by flattening them is defensible, but the matrix geometry it imposes is less obviously meaningful than for transformer projections.

## 8. Exploration for action-effect identification

With one expert action per state, alternative action effects are formally unidentifiable. If the dataset contains only \((s,a,y)\), then two transition functions that agree on \(T(s,a)\) but disagree arbitrarily on every \(T(s,a')\), \(a'\neq a\), have identical empirical loss. No architecture or optimizer can resolve that without additional assumptions or data.

The directly relevant exploration literature includes:

- [“Self-Supervised Exploration via Disagreement”](https://proceedings.mlr.press/v97/pathak19a.html) (ICML 2019), which uses ensemble forward-model disagreement as an intrinsic objective.
- [“Planning to Explore via Self-Supervised World Models”](https://proceedings.mlr.press/v119/sekar20a.html) (Plan2Explore, ICML 2020), which approximates information gain using disagreement among five one-step models.
- [“Active World Model Learning with Progress Curiosity”](https://proceedings.mlr.press/v119/kim20e.html) (ICML 2020), which targets learning progress rather than raw error, reducing attraction to irreducibly noisy observations.
- [“Causal Curiosity”](https://proceedings.mlr.press/v139/sontakke21a.html) (ICML 2021), which selects interventions that expose causal factors.
- [“Active Learning for Nonlinear System Identification with Guarantees”](https://arxiv.org/abs/2006.10277) (arXiv:2006.10277), which formalizes exploration of poorly covered feature directions followed by re-estimation.
- [“Curiosity in Hindsight”](https://proceedings.mlr.press/v202/jarrett23a.html) (ICML 2023), which separates useful novelty from stochastic “noisy-TV” prediction error.
- The ARC-specific [graph exploration work](https://arxiv.org/abs/2512.24156), which explicitly routes toward untested state–action edges.

For synthetic training, clone simulator state and generate grouped factual branches:

- enumerate ACTION1–ACTION5 from the same state;
- for ACTION6, stratify coordinates over objects, colors, boundaries, empty regions, symmetric counterparts, and plausible interaction sites;
- retain both no-op and state-changing cases;
- group transitions by exact state plus relevant history;
- balance action types, genuinely changed tuples, and distinct successor equivalence classes;
- distinguish “input action differs” from “simulator outcome differs.”

Exhaustively enumerating all 4,096 ACTION6 coordinates is usually unnecessary. Equivalence-class sampling is better: sample several coordinates with the same inferred semantic relation and retain at least one representative of every distinct observed outcome.

For live interaction, use a one-step system-identification bonus:

\[
U(a)=
\hat G(a)
+\beta\,\text{epistemic-disagreement}(a)
-\lambda\,\text{risk}(a)
-\rho\,\text{redundancy}(s,a)
-\kappa\,\text{action-cost}.
\]

This is still a greedy policy, not tree search. Because ARC’s RHAE metric squares action efficiency, exploration must be information-dense: prefer safe, reversible actions that discriminate among high-probability hypotheses. Record tested state–action signatures and suppress repeated probes unless the state or model uncertainty materially changed.

## Ranked interventions for Tofy

The estimates below are engineering priors, not literature-reported Tofy results. “Changed exact” means percentage-point change from the current 2–5% baseline. “Live” means RHAE percentage points under the searchless greedy policy. Ranges are not additive.

“Standard-frontier” has a direct precedent in current adjacent literature. “Beyond-frontier” combines supported components in a way not yet demonstrated for a tiny ARC-AGI-3 JEPA.

| Rank | Intervention | Status | Expected changed-exact impact | Expected live-score impact |
|---:|---|---|---:|---:|
| 1 | Branch-complete same-state counterfactual curriculum, including ACTION6 outcome-equivalence sampling | Standard-frontier | **+10 to +30 pp** | **0 to +5 pp** alone; larger once used by policy |
| 2 | Direct predicted-state 16-way CE, with changed gate/new-color factorization and balanced changed/unchanged loss | Standard-frontier | **+8 to +25 pp** | **0 to +3 pp** |
| 3 | History-conditioned belief state: 4–16 prior frames/actions plus status/resource information; first measure visible-state collision rate | Standard-frontier | **+3 to +15 pp** | **+1 to +8 pp** |
| 4 | Action-sensitive latent displacement: inverse action/coordinate head plus factual alternative-future separation only for distinct outcomes | Standard-frontier, including post-March evidence | **+5 to +15 pp** | **0 to +4 pp** |
| 5 | Meta-trained, per-game resettable fast adapter updated prequentially from observed transitions | Beyond-frontier for this regime | **+2 to +12 pp** after 5–20 informative transitions | **+1 to +8 pp** |
| 6 | Searchless goal/progress/success/risk heads; greedy one-step utility rather than dynamics magnitude alone | Beyond-frontier integration | **0 to +2 pp** | **+2 to +15 pp** |
| 7 | One-step active system identification: disagreement or learning-progress bonus, tested-action ledger, safe/reversible probe penalty | Standard-frontier | **0 to +3 pp** on the fixed offline metric | **−1 to +10 pp** depending on probe efficiency |
| 8 | Variable-depth recurrent refinement with observation/action recall, outer normalization, stochastic depth, and exact supervision at sampled depths | Standard-frontier | **+2 to +10 pp** | **0 to +3 pp** |
| 9 | Action-broadcast local residual dynamics: NCA-like spatial updates or discrete delta tokens plus a small global recurrent channel | Beyond-frontier hybrid | **+1 to +8 pp** | **0 to +3 pp** |
| 10 | Matched Muon-versus-AdamW control; correct weight decay and update-RMS scaling before considering optimizer variants | Standard-frontier hygiene | **−2 to +3 pp** | **−0.5 to +1 pp** |

The first experiment should not be architectural. Measure the Bayes-style ambiguity ceiling:

- group identical visible history/action inputs;
- count groups with multiple factual successors;
- repeat for history lengths 1, 2, 4, 8, and 16;
- stratify by action type and ACTION6 coordinate relation.

If one-frame collisions are common, no one-frame objective can reach high exact accuracy. If collisions are rare but unobserved action branches dominate, intervention 1 is decisive. Then run a factorial comparison of counterfactual coverage and categorical supervision. That separates “the model never saw alternative effects” from “the loss blurred effects it did see.”

A realistic success target for the first four interventions together is moving held-out changed-pixel exact prediction from 2–5% into roughly the **20–45%** range. That would be meaningful evidence of learned dynamics, but I would not expect a proportional ARC-AGI-3 score increase until goal/progress inference and information-efficient exploration are added. The public agents demonstrate that accurate transition modeling is necessary; they also demonstrate that verification, hypothesis management, and action choice are where game completion is won.

The durable research synthesis was also saved in [arc-agi-3-frontier-interventions.md](</home/stepan/Research/ml/tofy/arc-agi-3-frontier-interventions.md>).
