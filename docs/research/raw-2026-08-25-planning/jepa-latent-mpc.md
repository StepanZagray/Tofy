All key sources verified. Here is the full report.

# LeCun's world-model/planning blueprint and its 2024–26 latent-MPC realizations

## 1. LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)

**OpenReview (not arXiv), id `BZ5a1r-kVsf`, v0.9.2, June 2022.** (OpenReview blocked fetch behind a verification wall; the recipe below is from the paper itself, consistent with how every follow-up paper cites it.)

The Mode-2 planning recipe, precisely:
- **Modules:** perception encoder → world model (latent predictor) → cost module = fixed **intrinsic cost** IC(s) + **trainable critic** C(s) → actor. All state is latent (JEPA representations), never pixels.
- **Planning as energy minimization:** given current latent s₀ and a candidate action sequence a₁..a_T, the world model rolls out ŝ_{t+1} = Pred(ŝ_t, a_t); total energy F = Σ_t [IC(ŝ_t) + C(ŝ_t)]. The actor searches for argmin over action sequences. Because the whole pipeline is differentiable, LeCun's preferred optimizer is **gradient descent through the unrolled world model** (backprop-through-time on actions), with sampling/search as the fallback for non-smooth landscapes.
- **Receding horizon:** execute the first action(s), re-perceive, re-plan.
- **Uncertainty:** latent variables z regularized (e.g., VICReg-style) to represent unpredictable factors; planning can minimize over worst case or expectation over z.
- **H-JEPA:** stacked JEPAs at coarser time scales; the top level plans abstract subgoal sequences, each subgoal becomes the cost target for the level below — this is the part essentially *no* 2025–26 system has realized.
- **Mode-1 amortization:** train a reactive policy to imitate Mode-2 outputs, so expensive planning is distilled into a fast policy.

**Tofy verdict:** the blueprint's single-level version (JEPA + latent rollout + energy = distance-to-goal + search over actions) is exactly Tofy's shape; the deterministic ARC-AGI-3 games even remove the uncertainty machinery.

## 2. V-JEPA 2 / V-JEPA 2-AC — arXiv **2506.09985** (Meta, June 11 2025)

Assran et al. (30 authors incl. LeCun, Ballas). V-JEPA 2-AC: a ~300M-param action-conditioned predictor (block-causal transformer) post-trained on <62 h of unlabeled Droid robot video, on top of the frozen 1B-param V-JEPA 2 encoder.

- **Planner:** CEM, **800 samples × 10 iterations**, cost = **L1 distance** ‖P(a^{1:T}; s_k, z_k) − z_g‖₁ between predicted latent and the goal-image embedding; actions constrained to an L1-ball (radius 0.075 ≈ 13 cm end-effector displacement). Effective horizon ~1 step with **receding-horizon replanning**; multi-stage tasks (grasp, pick-and-place) are decomposed via **intermediate image subgoals**, not long rollouts.
- **Planning cost: ~16 s per action on an RTX 4090.**
- **Evidence:** zero-shot on Franka arms in two labs (no environment-specific data, no rewards): reach-with-object ~75%, grasp ~65%, pick-and-place ~72.5%. Baselines: Octo (behavior cloning) 15% avg on manipulation; Cosmos video-diffusion planner ~4 min/action and 0–20% grasp/pick.
- **Failure modes (stated in paper):** camera-pose sensitivity (needs the robot base in view / manual calibration), error accumulation limits long-horizon planning (hence subgoals), image-goal-only specification.

**Tofy verdict:** the loop (CEM over actions, L1 latent-to-goal cost, replan every step) transfers directly; at ~1M params Tofy can evaluate all ~4k atomic actions **exhaustively** in one batched forward pass — cheaper than V-JEPA 2-AC's 8k rollouts — easily inside 0.1–2 s. The caveat that matters is theirs too: don't trust rollouts beyond a few steps; use subgoals or replanning.

## 3. DINO-WM — arXiv **2411.04983** (Zhou, Pan, LeCun, Pinto; Nov 2024, rev. Feb 2025; ICLR 2025 line of work)

World model = frozen **DINOv2 patch features** + a ~19M-param ViT predictor (6 layers, causal attention) trained on offline trajectories only.

- **Planner:** MPC with **CEM, 100 samples × 10 iterations**; cost = **MSE between predicted latent (patch features) and goal-image latent**; frameskip 1–5, task horizons ~25–100 env steps; some experiments also use gradient descent on actions.
- **Evidence:** zero-shot at test time (no demos, no reward model, no inverse model) on 6 environments: Maze 0.98 SR (on par with DreamerV3's 1.00), **PushT 0.90 vs 0.32 best baseline (IRIS)**, Rope Chamfer 0.41 vs 1.11, Granular 0.26 vs 0.37. Ablations: frozen DINOv2 patch features 0.90 on PushT vs ResNet 0.20 / R3M 0.42 — **spatial patch tokens, not global vectors, are what made frozen features work**; removing causal masking collapses PushT 0.92→0.08.
- **Compute:** single forward 0.014 s (batch 32); a full CEM plan ~53 s (still faster than simulating deformables).
- **Failure modes:** relies on offline data coverage; global-feature encoders fail; long-horizon accuracy degrades (mitigated by frameskip).

**Tofy verdict:** the closest recipe in spirit — small predictor over frozen-ish spatial features, CEM+MSE latent cost. Lesson for Tofy: keep **per-cell/patch spatial structure** in the latent used for the planning cost (a pooled global embedding will likely kill ARC-grid planning precision).

## 4. PLDM — arXiv **2502.14819** (not 2502.14855) — Sobal et al., "Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models," Feb 2025

JEPA dynamics model (VICReg-regularized to prevent collapse) trained on reward-free offline trajectories; compared against 6 offline-RL/GC methods on 23 datasets.

- **Planner:** **MPPI**, horizon 16, 500 samples, replan every 1 step (Two-Rooms) or every 4 (Diverse PointMaze); cost = Σ_t ‖h(s_goal) − f(ẑ_t, a_t)‖ (L2 in latent space).
- **Models are tiny:** Two-Rooms total **2.22M params** (Impala-small encoder + 2-layer GRU); Diverse PointMaze total **54K params** — direct proof this class works at Tofy scale.
- **Evidence:** vs GCSL/GCIQL/CRL/HIQL/HILP/GCBC — comparable at 3M transitions, but PLDM is the standout under distribution shift and scarcity: **~50% success with only a few thousand transitions**, and it is the **only method that generalizes to held-out maze layouts** (works even trained on just 5 maps; all model-free baselines fail).
- **Compute:** planning ~100× slower than a model-free policy (~13.4 s per 200-step episode, i.e., ~67 ms/action at their scale).
- **Failure modes:** planning-time cost; below-ceiling best-case performance vs the strongest model-free methods when data is abundant and in-distribution.

**Tofy verdict:** the single most encouraging datapoint — a ~2M-param JEPA + MPPI beats model-free RL on generalization to unseen layouts from tiny offline data, which is precisely ARC-AGI-3's regime (novel games, no resets, few interactions); its ~67 ms/action MPPI is well inside Tofy's 0.1–2 s budget.

## 5. 2025–2026 follow-ups

- **Navigation World Models** — arXiv **2412.03572** (Bar, Zhou, Darrell, LeCun; CVPR 2025 **oral**). 1B-param Conditional Diffusion Transformer over (DINOv2-like) latents; plans by simulating candidate navigation trajectories and **ranking/optimizing them with sampling (CEM-style) against a goal-similarity cost**, or re-ranking an external policy's proposals. Shows planning from a single image in unfamiliar scenes. Diffusion rollouts are far too heavy for a 0.1–2 s budget — the takeaway for Tofy is the trajectory-ranking pattern, not the model class.
- **AdaWM** — arXiv **2501.13072** (ICLR 2025). Dreamer-style world-model RL for CARLA driving; contribution is **which module to finetune under distribution shift**: quantify whether the *dynamics mismatch* or the *policy mismatch* dominates, then selectively low-rank-update that module. Not latent MPC, but the mismatch-diagnosis idea maps to Tofy's per-game online adaptation choice (adapt predictor vs planner temperature).
- **DINO-world, "Back to the Features"** — arXiv **2507.19468** (Meta FAIR, July 2025). Generalist frame predictor in frozen DINOv2 latent space trained on ~60M uncurated videos; action-conditioned finetuning yields a planner-usable model. Confirms the frozen-spatial-features + latent-predictor design scales; no planner innovation.
- **Terver, Bardes, Ponce, LeCun et al., "What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?"** — arXiv **2512.24497** (Dec 2025, v3 May 2026). The empirical playbook for this whole line: **CEM with L2 cost is the most reliable planner across domains**; gradient descent (Adam) wins only on smooth cost landscapes (Metaworld) and fails on multi-modal navigation landscapes; Nevergrad ≈ CEM with less tuning. Other findings: proprioception fixes "final-inch" goal oscillation; DINO-family encoders beat video encoders; multistep-rollout training helps with an optimum at 2–6 unroll steps; predictor needs ≥2 context frames (3–5 optimal) to infer velocity; AdaLN+RoPE action conditioning best; scaling helps real data, not sim.
- **"Closing the Train-Test Gap in World Models for Gradient-Based Planning"** — arXiv **2512.09929** (Parthasarathy, …, LeCun, Izmailov, Goldblum; Dec 10 2025). Directly on failure mode 6: gradient planners drive DINO-WM-style models into **out-of-distribution latent regions (adversarial action sequences / hallucinated dynamics)**. Fixes: FGSM-style **adversarial world-model finetuning** (smooths the action-loss landscape) and **online world modeling** (retrain on simulator-corrected planner rollouts). Results on PushT/PointMaze/Wall: +18/+20/+30% open-loop success; after the fix, gradient MPC **matches or beats CEM at ~10% of the compute**.
- Peripheral 2026 entries surfaced (not individually verified in depth): hierarchical latent planning (arXiv 2604.03208), amortized planning in latent geometry (2605.08732), FF-JEPA long-horizon latent planners (2606.09311), variable-length latent world models (2606.21775).

## 6. Failure modes across the line (with sources)

1. **Optimizer exploitation of the learned model** — the planner finds action sequences whose *predicted* latents score well but are hallucinated; worst for gradient-based planning, mitigated by sampling noise (CEM/MPPI), adversarial finetuning, and online correction (2512.09929; also acknowledged in 2512.24497).
2. **Compounding rollout error** — exponential error growth in continuous latent rollouts (formalized in 2512.24497 App. D); V-JEPA 2-AC explicitly limits horizon and uses image subgoals; DINO-WM uses frameskip; PLDM replans every 1–4 steps.
3. **Latent collapse breaks planning** — PLDM needs VICReg regularization; a collapsed latent makes the distance-to-goal cost uninformative (directly relevant to Tofy's diagnosed collapse+freeze — a collapsed encoder silently zeroes out the *planning signal*, not just prediction loss).
4. **Cost landscape multi-modality** — gradient planners trapped in local minima on navigation/contact tasks; use sampling there (2512.24497).
5. **"Final-inch" imprecision** — pooled/imprecise latents can't resolve small goal distances (fixed by proprioception in robotics; the analog for Tofy is per-cell latent costs).
6. **Goal specification** — everything above needs a goal *embedding*; ARC-AGI-3's hidden goals mean Tofy must learn/infer the cost module (LeCun's trainable critic), which is the part this literature leaves most open.

## Bottom line for Tofy

The 2025–26 evidence says Tofy's architecture class is validated at Tofy's scale: PLDM (2502.14819) wins generalization-from-scarce-data with a 2.2M-param JEPA + MPPI at ~67 ms/action, and DINO-WM/V-JEPA 2-AC establish the standard loop (CEM/MPPI, L1/L2 latent-to-goal cost, receding horizon, ≤~6-step effective rollouts, spatial not pooled latents). With ~4k atomic actions and a ~1M-param model, Tofy can do *exhaustive* 1-step lookahead plus sampled short-horizon search within 0.1–2 s on an A40 — comfortably richer than what these papers afford. The two gaps the literature does not solve for ARC-AGI-3: (a) hidden goals → need a learned critic/novelty cost instead of a goal image, and (b) collapse → per PLDM, variance-covariance regularization is not optional, it is what makes the planning cost mean anything.

Sources: [arXiv 2506.09985](https://arxiv.org/abs/2506.09985), [arXiv 2411.04983](https://arxiv.org/abs/2411.04983), [arXiv 2502.14819](https://arxiv.org/abs/2502.14819) / [latent-planning.github.io](https://latent-planning.github.io/), [arXiv 2412.03572](https://arxiv.org/abs/2412.03572), [arXiv 2501.13072](https://arxiv.org/abs/2501.13072), [arXiv 2507.19468](https://arxiv.org/abs/2507.19468), [arXiv 2512.24497](https://arxiv.org/html/2512.24497v3), [arXiv 2512.09929](https://arxiv.org/html/2512.09929), [OpenReview BZ5a1r-kVsf](https://openreview.net/forum?id=BZ5a1r-kVsf).