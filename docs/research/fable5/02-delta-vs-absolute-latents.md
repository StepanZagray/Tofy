# Delta/residual vs absolute latent dynamics for autoregressive stability

> Research note for P2 (Tofy world model). Plan only — no code changes proposed here are implemented.
> Sources: repo code/history (verified directly) + primary literature (cited per claim).

## 1. What Tofy actually does today (v16)

### 1.1 Absolute prediction on a unit-RMS shell

The model predicts the **absolute** next latent `y` (`B×C×8×8` spatial grid). Per-batch
RMS normalization is applied at two points in `src/p2/model.rs`:

- encoder output: `encode_state` = `rms_norm_latent(encoder(frames))`
- dynamics output: after every outer TRM step, `y = rms_norm_latent(block(y + z))`

So every latent the model ever consumes or emits lives on the unit-RMS shell.
Autoregressive rollout (`forward_from_latent_with_depth`) feeds the normalized `y`
straight back in as the next state, adds the action bias, and re-runs the recursion
with `y`,`z` zero-initialized. There is no residual path from the previous state
latent to the predicted one except through the recursion input `x`.

### 1.2 Loss and anti-collapse mechanism

`leworld_loss` (`src/p2/train.rs`) regresses each outer-step `y` onto the **live**
(non-detached) encoding of the next frame, plus SIGReg (Epps–Pulley sketched
isotropic-Gaussian regularization, LeJEPA-style) on stacked `cur_z`/`next_z` to
prevent collapse. The only stop-gradient on `main` today is `stop_grad_event_y`
(event head reads a detached `y`; the dynamics loss target is live).

## 2. What v12 actually tested (and what it did not)

**Load-bearing finding: v12's `delta_dynamics` was algebraically a no-op.**
Recovered from the v12-era source (git checkpoint ref `291d58dc`, `src/p2/train.rs`):

```rust
// pred   = y      - sg(cur_z)
// target = next_z - sg(cur_z)   (next_z optionally detached)
// loss   = mean((pred - target)^2) = mean((y - next_z)^2)
```

The detached anchor `sg(cur_z)` is subtracted from both the prediction and the
target, so it cancels in the residual `pred − target`. Loss value and gradients were
**identical** to absolute prediction. The model itself never emitted a delta — `y`
was produced exactly as before and only the loss bookkeeping was reparameterized.

Therefore the v12→v11 regression cannot be attributed to delta dynamics. The real
changes in v12 were:

1. `stop_grad_dynamics_target` — detach `next_z` in the dynamics loss, removing the
   encoder's gradient from the dynamics objective (encoder then trained only via
   SIGReg and the event/Q heads).
2. Confounded architecture changes: dual TRM blocks (`block_z`, `block_y`),
   dual-pool encoder, `y` warm-start from `x` (`docs/P2_V12.md`).

v12 results (`p2-output-v12/chain_summary.json`): one-step MSE 0.09–2.27 (v11-control:
0.026), rollout-8 MSE 6.7e4–1.4e24 (v11-control: 0.17). The divergence signature —
one-step error merely bad but multi-step error astronomically exploding — is a latent
**magnitude/manifold escape**, which is precisely what v16's per-step RMS norm removes
by construction.

**Conclusion from internal evidence:** the question "does delta help on the spatial
grid?" is still *open*; it has never been tested. What v12 falsified (weakly, with
confounds) is stop-gradient on the dynamics target *combined with* the dual-block
architecture, in a regime without latent normalization.

## 3. What the literature says

*(Primary-source findings — see citations inline.)*

### 3.1 MuZero: absolute latent + per-step min-max scaling, no latent target at all

MuZero's dynamics function g predicts the next hidden state directly (absolutely);
there is no residual parameterization and **no latent regression target** — the
hidden state is shaped only by reward/value/policy gradients ("no direct constraints
for the hidden states to capture all information necessary to reconstruct the
original observation"; Schrittwieser et al. 2020, [arXiv:1911.08265](https://arxiv.org/abs/1911.08265), §3).
For stability it scales each hidden state to `[0,1]` per step: "we scale the hidden
state to the same range as the action input ([0, 1]): s_scaled = (s − min(s)) / (max(s) − min(s))"
(Appendix G). This is the closest published analogue of Tofy's per-step RMS norm:
**absolute prediction + per-step renormalization** is a proven-stable combination for
deep autoregressive latent rollouts (5+ steps in MuZero training, hundreds in search).

EfficientZero (Ye et al. 2021, [arXiv:2111.00210](https://arxiv.org/abs/2111.00210), §4.1)
added a SimSiam-style **self-supervised consistency loss**: predicted next latent vs
encoding of the real next observation, with **stop-gradient on the target branch** and
a projector+predictor head, exactly following SimSiam. It was the largest single
ablation gain in the paper (Atari 100k). Target is absolute, not a delta.

### 3.2 TD-MPC / TD-MPC2: absolute latent target, detached target encoder, SimNorm

TD-MPC (Hansen et al. 2022, [arXiv:2203.04955](https://arxiv.org/abs/2203.04955), §4)
regresses the predicted next latent onto the encoding of the next observation
produced by a **slow-moving EMA target encoder** (θ⁻), i.e. a stop-gradient/EMA
target, absolute space. TD-MPC2 (Hansen et al. 2024, [arXiv:2310.16828](https://arxiv.org/abs/2310.16828), §2)
keeps the absolute latent-consistency objective (detached target in the official code,
[nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2): `next_z = self.model.encode(obs[1:]).detach()`)
and introduces **SimNorm**: the latent is partitioned into groups of 8 and each group
is pushed through a softmax, embedding the state into a product of simplices. The
paper motivates it explicitly as bounding latent magnitude for **long-horizon
stability**: "naturally biases the representation towards sparsity … we find [it] to
be critical to the success of TD-MPC2" (§2, "normalization" discussion). The dynamics
MLP predicts the next latent absolutely; SimNorm is applied to its output — same
*shape* as Tofy v16's `rms_norm_latent(block(...))`, different projection surface
(simplex product vs RMS shell).

### 3.3 DreamerV2/V3: a different stabilization family

Dreamer's RSSM avoids the deterministic-regression problem entirely: latents are
stochastic (categorical in V2/V3), trained with a KL between posterior and prior with
**KL balancing** and **free bits**, plus symlog transforms in V3 (Hafner et al. 2023,
[arXiv:2301.04104](https://arxiv.org/abs/2301.04104), §3). Not directly transferable
to Tofy's deterministic TRM recursion, but it is the third data point that *no*
prominent world model uses delta latent targets.

### 3.4 Stop-gradient theory: what it is for, and when Tofy needs it

BYOL (Grill et al. 2020, [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)) and
SimSiam (Chen & He 2020, [arXiv:2011.10566](https://arxiv.org/abs/2011.10566)) show
that a network predicting its own representation collapses to a constant unless the
target branch is stop-gradded (SimSiam: "our method works even without the momentum
encoder … stop-gradient is critical", §1, §4.1 ablation), ideally with an asymmetric
predictor head. The collapse pressure exists in Tofy too: the encoder appears on both
sides of the dynamics loss. Tofy currently counters it with **SIGReg** (distributional
regularization toward an isotropic Gaussian, per LeJEPA — Balestriero & LeCun 2025,
[arXiv:2511.08544](https://arxiv.org/abs/2511.08544)) instead of stop-grad. These are
*alternative* solutions to the same failure mode. v12 stacked stop-grad *on top of*
SIGReg and starved the encoder of its main task-relevant gradient (the dynamics loss),
leaving it trained mostly by the weak event/Q signals — a plausible mechanism for the
v12 one-step regression that has nothing to do with deltas.

JEPA-family world models (V-JEPA 2, Assran et al. 2025, [arXiv:2506.09985](https://arxiv.org/abs/2506.09985);
DINO-WM, Zhou et al. 2024, [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)) sidestep
this differently: the target encoder is **frozen or EMA** and the predictor is trained
against it in absolute latent space. No external paper named "LeWorld" was found;
`leworld_loss` is Tofy's internal name (LeJEPA lineage via SIGReg).

### 3.5 Where delta prediction genuinely helps — and why that regime is not ours

The strong empirical case for delta prediction is **ground-truth-state** model-based
RL: Nagabandi et al. 2017 ([arXiv:1708.02596](https://arxiv.org/abs/1708.02596), §IV-A)
"rather than directly predicting the next state, we predict the *change* in state
over the time step duration", and PETS (Chua et al. 2018,
[arXiv:1805.12114](https://arxiv.org/abs/1805.12114)) follows the same convention.
The mechanism: physical states are smooth and slowly varying, so `s_{t+1} − s_t` is
small, near-zero-mean, and much better conditioned than `s_{t+1}` itself; the network
learns identity-plus-correction for free.

That mechanism's preconditions fail in Tofy's setting, on three counts:

1. **The latent basis is not fixed.** The encoder co-trains with the dynamics, so a
   "delta" is a difference between points whose coordinate system is itself moving.
   In ground-truth-state MBRL the basis is fixed physical units.
2. **Per-step renormalization makes deltas ill-defined.** With `y` projected to the
   unit-RMS shell every step, the informative difference is *angular*, not additive.
   An additive residual followed by renorm, `y' = rms_norm(y + Δ)`, is exactly the
   transformer residual-stream + normalization pattern (pre-LN; Xiong et al. 2020,
   [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)) — fine as an *architecture*,
   but supervising `Δ` against `next_z − cur_z` in normalized space no longer has the
   "small, well-conditioned target" property that motivated deltas in the first place
   whenever transitions are jumpy.
3. **ARC transitions are jumpy.** Grid worlds have discrete, non-smooth dynamics
   (objects appear/disappear/teleport under actions). Where the delta is not small,
   the conditioning advantage evaporates, and the residual identity path actively
   biases the model toward copy-forward — which *inflates* multi-step error when the
   environment mostly changes on action.

Note the distinction the literature keeps clean and v12 blurred: **residual
parameterization of the network output** (architecture; can help optimization) vs
**delta-space supervision target** (loss; helps only when the delta is small and the
basis is fixed). MuZero/TD-MPC/EfficientZero use neither; classical MBRL uses both,
in a fixed state basis. v12 implemented only the loss-space half, and did it in a way
that cancelled out.

## 4. Proposal for P2

### 4.1 Recommendation

**Do not retry delta-space supervision.** Both the internal evidence (v12's delta was
a no-op, so nothing was learned either way; the setting violates the preconditions
under which deltas help) and the literature consensus (MuZero, EfficientZero, TD-MPC,
TD-MPC2, DreamerV3, V-JEPA 2, DINO-WM all supervise absolute latents) point the same
way. v16's absolute-prediction + per-step RMS norm is the field-standard recipe.

**Worth testing instead** (ordered by expected information per GPU-hour):

1. **Residual y-update inside the recursion (architecture-only, absolute target).**
   Change the outer update from `y = rms_norm(block(y + z))` to
   `y = rms_norm(y + block(y + z))` in `deep_step`/`run_recursion`. This is the
   pre-LN transformer pattern: it gives the identity path (cheap copy-forward for the
   static majority of the grid) without changing the loss target. One-line change,
   checkpoint-incompatible only in behavior, not in weights.
2. **Warm-start `y` from the current state latent** instead of zeros in
   `run_recursion` (rollout passes the previous `y`; training passes `cur_z`). This
   is the "residual in spirit" variant: the recursion refines the current state
   toward the next state rather than rebuilding it from `x`. v12 bundled this with
   four other changes; it deserves an isolated test.
3. **Stop-grad/EMA dynamics target, isolated.** Only if collapse or target-chasing is
   actually observed (e.g. SIGReg loss falling while rollout MSE rises, or latent
   effective rank dropping). EfficientZero/TD-MPC2 evidence says detached targets
   work *when paired with a predictor head and a stable encoder signal*; v12's
   failure suggests Tofy's encoder currently needs the live dynamics gradient. Not
   recommended as the next experiment.

### 4.2 Minimal experiment design

Baseline: v16 config (`scripts/p2_v16_train_eval.sh`), fixed seed, identical
curriculum. One variable per run. Metrics from `p2.eval_report.v5`: one-step MSE,
open-loop rollout MSE @4/@8, closed-loop @4/@8, events acc, Q calibration.

- **E1 (residual update):** v16 + `y = rms_norm(y + block(y + z))`.
  Gate: one-step MSE ≤ v16 baseline and rollout-8 improves ≥20%; reject if one-step
  MSE degrades >10%.
- **E2 (warm-start y):** v16 + `y` initialized from the state latent in
  `run_recursion` (zeros for `z` unchanged).
  Same gate as E1. E1+E2 combined only if both individually pass or are neutral.
- **Diagnostic to log in both:** cosine similarity between consecutive rollout
  latents `cos(y_t, y_{t+1})` and between `y_t` and the encoded real next frame, per
  rollout step. This separates "copy-forward bias" (cos(y_t, y_{t+1}) → 1, real-frame
  cosine falls) from genuine tracking, which raw MSE on a unit shell hides.

Cost: each run is one v16-scale training (same batch/accum pair as v16, per AGENTS.md
hardware rule). No new CLI flags needed if the variants are tried as code toggles on
a branch; promote to a config flag only if one wins.

### 4.3 Risks

- **Residual update biases toward copy-forward** on action-heavy segments (risk for
  E1/E2). The cosine diagnostic above is the tripwire; the retarget lesson (if
  enabled) stresses exactly this.
- **Confounding, again.** v12's lesson is procedural as much as architectural: never
  bundle a normalization change, a gradient-flow change, and an architecture change
  in one run. Keep E1 and E2 strictly one-diff-from-v16.
- **RMS shell vs SimNorm.** If E1/E2 both fail and rollout drift persists, the next
  literature-backed lever is the projection surface itself (TD-MPC2's SimNorm on the
  channel dimension of the 8×8 grid), not delta targets.
- **Checkpoint compatibility:** E1/E2 change no parameter shapes (same `block`), so
  v16 checkpoints load, but behavior differs — do not resume-train across the toggle;
  train from scratch.

## 5. Answering the headline question

Delta helps when the state basis is fixed, transitions are smooth, and the delta is
small and well-conditioned (classical MBRL on physical states). It hurts — or is
meaningless — when the basis co-trains, the latent is renormalized every step, and
transitions are jumpy: Tofy's exact regime. The productive residue of the delta idea
for P2 is the **identity path** (residual update / warm-start), not delta-space
supervision, and that is what E1/E2 test.
