# Beyond-frontier in-episode mechanics learning for Tofy

## Bottom line

The next binding constraint is not the optimizer, EP regularizer, or model width. It is the training distribution:

- The episode operator affects ACTION5/ACTION6, but current temporal streams do not contain an operator-revealing transition followed by another operator-dependent query in the same episode.
- Consequently, no recurrent state, fast adapter, or history model receives the meta-learning signal “use this fact to revise later predictions.”
- The second binding constraint is the decoded-pixel seam. Given the foreground reconstruction plateau, adaptation confined to upstream FiLM/latent space cannot guarantee corrected pixels.
- The current ambiguity probe can report a ceiling of 1.0 merely because every exact visible-frame key is unique.

My best bet is a hybrid: exact canonical transition memory plus an orthogonal fast-weight writer. Memory owns factual exactness; the writer generalizes those facts through the recurrent core. Do this before a Transformer-scale history model.

## Quantitative ceiling without episode adaptation

### Finite ambiguity bound

Let `M` be the hidden episode operator and let `F_m(x,a)` be the deterministic successor under operator `m`. On any query where the four training operators produce four distinct successors, a history-free deterministic predictor satisfies:

```text
P[prediction = successor] <= max_m P(M=m) = 1/4
```

That is a 25% exact-successor ceiling on a fully separating query. More generally, if `q_sep` is the fraction of queries whose operator-conditioned successors are distinct, then:

```text
C_frozen <= 1 - 0.75 q_sep
```

This is a small finite-set theorem: partition operators by equal successor, then choose the largest outcome class. It should be straightforward to formalize in Lean. After one diagnostic fact whose outcomes are injective in `M`, the compatible version space becomes a singleton and the conditional ceiling becomes 100%, assuming the selected operator model is itself exact.

Tofy has four training families and one held-out family [data.rs](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:327), with uniform sampling from the training split [data.rs](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:1681). The 25% result assumes the intended counterfactual distribution where the visible state is held fixed while `M` is resampled. It is not a proof that the present PRNG-generated dataset has perfect statistical independence between frames and operator IDs.

### How much of v5 is exposed to this ambiguity?

Only ACTION5/ACTION6 use the sampled episode operator [data.rs](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:1517).

- Random stream: two operator actions per seven rows [data.rs](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:2576).
- Factual group: one ACTION5 plus four ACTION6 branches among ten rows [data.rs](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:2078).
- Exploration, sequential, and hazard streams use the ordinary simulator rather than `apply_episode_operator` [data.rs](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:2607).

Applying those rates to the normalized annealed mixture [data.rs](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:310):

```text
operator-bearing row share at start = 0.35 × 2/7 + 0.20 × 5/10 = 20.0%

operator-bearing row share at end
  = (0.25 × 2/7 + 0.30 × 5/10) / 0.95
  = 23.31%

schedule-average share ≈ 21.63%
```

If every operator-bearing row were family-separating and everything else were predicted perfectly, the full-stream exact ceiling would be approximately:

```text
1 - 0.2163 × 0.75 = 83.78%
```

That 83.78% is a worst-case synthetic bound, not the measured changed-exact ceiling. No-op and outcome-equivalent cases raise it. The correct answer is `1 - 0.75 q_sep`, and `q_sep` is currently unmeasured.

### Why the current ambiguity report cannot answer this

The implemented probe groups rows by exact current gameplay pixels and action, then estimates majority-successor accuracy [semantic_eval.rs](/home/stepan/Coding/Personal/Tofy/src/p2/semantic_eval.rs:461). It only computes histories one and two [semantic_eval.rs](/home/stepan/Coding/Personal/Tofy/src/p2/semantic_eval.rs:502), despite the earlier h16 intent. A singleton group contributes perfect accuracy, so randomized color, placement, geometry, and operator sampling can make the reported ceiling 1.0 without demonstrating identifiability. `repeated_groups` must therefore be treated as a validity gate, not merely a diagnostic field [semantic_eval.rs](/home/stepan/Coding/Personal/Tofy/src/p2/semantic_eval.rs:206).

The replacement probe should hold scenario, augmentation, visible frame, and action fixed while applying all operator families. For history lengths `h ∈ {0,1,2,4,8,16}`, report:

- `q_ambiguous`: more than one counterfactual successor.
- `q_separating`: all operators have distinct successors.
- Exact and changed-subframe Bayes ceilings.
- Compatible version-space size after the factual prefix.
- Rows where the prefix genuinely changes the version space.
- “Not estimable” when no repeated or matched counterfactual groups exist.

This is an offline generator census and needs no optimizer steps.

### What can be claimed about real ARC-AGI-3?

A numeric fraction of real ARC-AGI-3 mechanics that require adaptation is not publicly identifiable. The official report says the agent must infer each environment’s mechanics and win condition, and that private environments intentionally contain broader, mostly non-overlapping mechanics with greater adaptation demands. It exposes 25 public environments but keeps 110 semi-private or fully private [ARC-AGI-3 Technical Report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf).

Therefore:

- **Synthetic Tofy claim:** 100% of the episode-specific operator identity is unavailable before an informative interaction; it touches approximately 20–23.3% of v5 rows.
- **Public ARC-AGI-3 claim:** unknown. Neither public games nor published metadata provide the matched counterfactual mechanic labels needed to estimate `q_sep`.
- It would be scientifically invalid to call 21.6% “the fraction of ARC-AGI-3 mechanics requiring adaptation.”

## Frontier boundary

Current work supports the direction but does not solve literal one-fact correction at Tofy’s scale:

- TTT layers make the recurrent hidden state a model updated by self-supervised gradients, but experiments are at 125M–1.3B parameters and long language contexts [arXiv:2407.04620](https://arxiv.org/abs/2407.04620).
- World-model ICL now distinguishes environment recognition from learning and finds that context length and environment diversity control their emergence [arXiv:2509.22353](https://arxiv.org/abs/2509.22353).
- “One-shot World Models” actually condition on roughly 1,000 transitions and report difficulty transferring to complex environments [arXiv:2409.14084](https://arxiv.org/abs/2409.14084).
- Gated DeltaNet combines targeted delta writes with erasure, but demonstrates this in language and retrieval rather than exact deterministic world-model correction [arXiv:2412.06464](https://arxiv.org/abs/2412.06464).
- Elastic TTT uses Fisher anchoring to reduce forgetting, but provides stability rather than exact no-regression guarantees [arXiv:2604.07350](https://arxiv.org/abs/2604.07350).
- WorldEvolver combines factual retrieval and rules extracted from prediction errors, but operates through an LLM agent’s context [arXiv:2606.30639](https://arxiv.org/abs/2606.30639).

The beyond-frontier target is consequently: one real deterministic transition, a bounded update, exact retention of prior facts, and measurable generalization to the next same-mechanic query in a sub-1M self-contained world model.

## Ranked proposals by expected value per GPU-hour

All effect sizes below are preregisterable estimates and **speculation**, not literature-derived results.

| Rank | Proposal | Type | A40 falsification cost | Expected aggregate changed-exact lift |
|---:|---|---|---:|---:|
| 1 | Canonical causal edit memory | EXPLOIT | 0–0.1 GPU-h | +2–6 pp |
| 2 | Orthogonal one-shot fast-weight writer | EXPLOIT | 0.2–0.7 GPU-h | +2–5 pp |
| 3 | Finite operator version-space with expert adapters | EXPLOIT | 0.3–1.0 GPU-h | +3–9 pp in-bank |
| 4 | Functionally constrained TTT solve | EXPLOIT | 0–0.3 GPU-h | +0.5–3 pp |
| 5 | Tiny factual-history in-context model | EXPLORE | 0.5–2 GPU-h | +1–4 pp |
| 6 | Fact-to-plasticity hypernetwork | EXPLORE | 1–3 GPU-h | +1–5 pp |

### 1. EXPLOIT — Canonical causal edit memory

**Mechanism.** Keep the existing exact raw-state graph as highest priority—it only replays identical raw `(frame, action)` edges [ADR 0004](/home/stepan/Coding/Personal/Tofy/docs/adr/0004-latent-planning-contract.md:114). Add a distinct episode memory containing canonical transition rules:

```text
key = (
  action class,
  relative action coordinate/object role,
  D4-canonical local before-stencil,
  color-equality pattern,
  boundary mask
)

value = sparse edit template:
  destination offsets,
  constant colors or source-offset copies,
  event/noop channel
```

Store the raw sparse delta as well. On retrieval:

1. Exact raw edge: return the factual successor.
2. Unique canonical rule match: apply its sparse edit and copy untouched pixels.
3. Conflicting or uncovered rule: abstain and use the neural predictor.

This goes beyond graph caching because a fact can transfer across translations, colors, D4 transforms, and repeated objects.

**Theory/invariant.** Suppose each changed pixel is an `r`-local deterministic function of its canonical key and all query keys have unique stored values. Applying the stored rules produces the exact successor by finite array extensionality. D4/color transfer is exact when the transition commutes with the canonicalization transform. Both are Lean-formalizable finite-map statements.

**Cost.** No learned parameters. With 64–256 rules, 9×9 stencils, and sparse edits, memory should remain below roughly 256 KB per episode. Lookup is hash-table CPU work plus `O(changed pixels)` writes; negligible A40 cost.

**Cheap falsification.** Offline only. Take one factual transition, then query translated, color-permuted, D4-transformed, and repeated-object variants under the same operator. Compare raw graph, neural checkpoint, and canonical memory. Reject if unique-rule coverage is below 20%, any supposedly unique hit is wrong, or conflict rate exceeds 1%.

**Expected effect, speculation.** Eligible repeated/local-mechanic stratum: +15–40 pp exact. Aggregate changed-exact: +2–6 pp. Live RHAE: +1–4 absolute points through fewer repeated exploratory actions.

**Failure modes.** Global effects such as region swapping and line pushes violate small-radius locality; different hidden states can alias to one stencil; overlapping edits may conflict. Detect with key-conflict counts, uncovered changed pixels, overlap non-commutativity, and exact replay after every retrieval. Abstain on any failure.

### 2. EXPLOIT — Orthogonal one-shot fast-weight writer

**Mechanism.** Add a frozen slow key encoder and an episode-local matrix `W ∈ R^(16×r)`, with `r=32` or `64`. Each observed transition produces pixel/patch keys `k` and desired palette-logit residuals `v`. Maintain an orthonormal basis `Q` of previously written keys and compute:

```text
u  = k - Q Qᵀ k
W' = W + (v - Wk) uᵀ / (uᵀk)
```

If `||u||` is too small, the write is an alias: do not modify `W`; route the fact to exact memory.

Read `Wk` at two places:

- Directly into the final palette logits, where exact correction can be enforced.
- Through a bounded projection into action FiLM at every recurrence, so imagined rollouts inherit the revised mechanic.

Those are existing natural seams: action FiLM is computed centrally [model.rs](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:1021), while the primary loss already operates on exact decoded logits [train.rs](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3635).

**Theory/invariant.** Since `u` is orthogonal to every stored key `k_i`:

```text
W'k  = v
W'k_i = Wk_i
```

Thus one write exactly interpolates the new fact and causes zero regression on the stored-key span, in exact arithmetic. If canonical-equivalent future queries share the same keys, their correction is exact too. This is stronger than the approximate preservation offered by orthogonal continual-learning gradients [arXiv:1910.07104](https://arxiv.org/abs/1910.07104).

**Cost.** Approximately 10–25K slow parameters for key/output projections; 1–8K episode floats. A per-pixel read is below about 5M MACs per query. QR/update work for at most 64 facts is tiny. Expected fused inference overhead is under 2%, but Candle kernel overhead must be measured.

**Cheap falsification.** Freeze the current checkpoint. Meta-train only the key/read projections for 512–1,000 steps on temporal operator episodes. After one diagnostic fact, test later queries plus exact replay of all prior facts. Require at least 99% new-fact correction, zero lost exact facts, and at least +5 pp on the post-fact operator stratum.

**Expected effect, speculation.** Post-fact operator stratum: +10–25 pp. Aggregate changed-exact: +2–5 pp. Live RHAE: +1–5 points.

**Failure modes.** Key aliasing, exhausted rank, large unstable logit residuals, or FiLM corrections that improve the observed fact but harm rollout states. Monitor innovation norm, memory rank, residual spectral norm, old-fact exactness, and separate logit-only versus FiLM-enabled rollouts.

### 3. EXPLOIT — Finite operator version-space plus expert adapters

**Mechanism.** Train one small low-rank FiLM/logit expert for each known operator family, sharing the encoder, core, and decoder. Episode state is an explicit compatible-expert set or posterior. After every factual transition, evaluate all experts on the exact prestate/action and eliminate experts whose decoded outcome disagrees. Once one remains, run only that expert. Retain an “unknown operator” state when every expert fails.

This is mechanic inference, not the goal posterior or EIG search already planned.

**Theory/invariant.** If the diagnostic outcome map `m → F_m(s,a)` is injective and every expert exactly realizes its family, one observation leaves a singleton version space. All subsequent same-family transitions are therefore exact. For non-injective observations, the remaining version-space size gives the correct measurable ceiling. This is another finite Lean theorem.

**Cost.** Four rank-4 experts should add roughly 10–50K parameters. Diagnosis costs four core passes after an observed informative action; after selection, cost returns to one pass. A 1,000-step head/adapter campaign should fit well below one A40-hour.

**Cheap falsification.** Train on the four existing families, with family IDs used only to assign training targets. Test one-fact selection on matched states and separately on the held-out SwapRegion family. Gates: oracle expert changed-exact at least 80%, singleton selection at least 90% after a separating fact, and unknown detection AUROC at least 0.9.

**Expected effect, speculation.** Known-family operator stratum: +20–50 pp; aggregate +3–9 pp. Held-out mechanics: approximately zero unless compositional experts cover them. Live RHAE: 0–3 points because real mechanics will often be out of bank.

**Failure modes.** Expert collapse, approximate predictions preventing hard elimination, non-diagnostic first actions, and false confidence on unknown mechanics. Report pairwise expert outcome separation, oracle-expert ceiling, posterior entropy, and unknown false-accept rate.

### 4. EXPLOIT — Functionally constrained TTT instead of three Adam steps

**Mechanism.** Replace unconstrained replay optimization with a projected update on the existing rank-4 adapter. Cache Jacobian-vector products for prior factual outputs. Solve:

```text
minimize   ||J_new Δθ - residual_new||² + λ||Δθ||²

subject to J_exact Δθ = 0
           ||J_anchor Δθ||² <= ε
           ||Δθ|| <= ρ
```

`J_exact` covers previously exact factual outputs. `J_anchor` covers untried actions at stored states, preserving the slow model where evidence is absent. Apply the candidate update only after exact nonlinear replay passes.

The current Phase C specifies three Adam steps and post-hoc rollback [ADR 0004](/home/stepan/Coding/Personal/Tofy/docs/adr/0004-latent-planning-contract.md:585); this proposal chooses the no-regression subspace before updating.

**Theory/invariant.** The projected update produces zero first-order output change on protected facts. For a linear residual adapter the constraint is exact, not first-order. If the new residual lies in the remaining column space, the constrained least-squares solution corrects it in one solve.

**Cost.** No extra inference parameters. Two to six backward/JVP passes after a real fact, estimated at 10–100 ms on A40, plus QR over at most 64 constraint directions. Offline comparison should require under 0.3 GPU-hours.

**Cheap falsification.** Adapt sequentially over existing checkpoint facts without retraining. Compare current three-step Adam, Fisher/EWC anchoring, and projected solve. Stop if fewer than 80% of informative updates are feasible, if any exact transition regresses, or if accepted-update CE gain is not better than Adam.

**Expected effect, speculation.** Post-fact stratum: +3–10 pp; aggregate changed-exact +0.5–3 pp; live RHAE 0–2 points.

**Failure modes.** No useful nullspace, ill-conditioned constraints, or nonlinear FiLM drift after a first-order-safe step. Detect minimum singular value, projected residual norm, trust-region saturation, and mandatory exact post-update replay.

### 5. EXPLORE — Sub-1M factual-history in-context world model

**Mechanism.** Encode each factual transition into compact tokens rather than storing full 64×64 frames:

- action and coordinate;
- pooled before-state token;
- sparse changed-set tokens;
- after-state/effect summary;
- prediction residual and event/noop result.

Use a two-layer causal Transformer with width 96, four heads, MLP width 192, and at most 32 factual transitions. Current patch/action query tokens cross-attend to this history and emit rank-8 FiLM and palette-logit residuals.

Estimated size is 0.20–0.30M parameters, keeping the total model around 0.7–0.8M.

The indispensable data change is to create temporal meta-episodes in which a diagnostic ACTION5/ACTION6 fact is followed by different ACTION5/ACTION6 queries under the same sampled operator. Current factual groups are same-state branches rather than causal sequences [data.rs](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:2119).

**Theory/invariant.** No unconditional one-shot guarantee should be claimed. The key measurable causal invariant is:

```text
ICL gain(h)
  = accuracy(real factual history of length h)
  - accuracy(episode-shuffled history of length h)
```

The gain must appear after an informative fact, disappear when the fact is counterfactually swapped, and persist on unseen states. Reporting recognition and genuine environment learning separately follows the current world-model ICL frontier [arXiv:2509.22353](https://arxiv.org/abs/2509.22353).

**Cost.** Roughly 150K Transformer-block parameters plus tokenizers/readouts. With 32 fact tokens and a 16×16 patch grid, expected inference is under 15M additional MACs; training overhead approximately 20–40%. A 2,000-step screen should take around 0.5–2 A40-hours.

**Cheap falsification.** Train only the history module and output adapters for 2,000 steps with the slow core frozen. Compare history lengths 0, 1, 2, 4, 8; real, shuffled, and episode-swapped histories; and held-out operator compositions. Reject if one-fact gain is below 3 pp or more than 20% of the gain survives history shuffling.

**Expected effect, speculation.** Post-context operator stratum: +5–15 pp; aggregate changed-exact +1–4 pp; live RHAE +1–5 points.

**Failure modes.** Ignoring history, memorizing episode-generator correlations, recognizing only the four training families, or using changed-pixel frequency as a shortcut. Detect via matched first frames, operator permutation, history swaps, unseen compositions, and attention/history ablation.

### 6. EXPLORE — Fact-to-plasticity hypernetwork

**Mechanism.** A small writer `Hψ` consumes:

```text
(encoded prestate, action, predicted latent,
 exact successor residual, sparse board effect)
```

It emits bounded rank-1 or rank-2 updates to the action-FiLM and final latent/logit residual adapters:

```text
A_next = Project_radius(A + gate × U Vᵀ)
```

The gate depends on deterministic prediction surprise and key novelty. Meta-training loss is evaluated primarily on future same-operator queries after the write, not on reconstructing the fact that generated it. This prevents the trivial “memorize only the current edge” solution.

**Theory/invariant.** A normalized delta readout can exactly overwrite its value at one key in one update. The spectral projection yields a finite bound on the induced latent change when the frozen core is Lipschitz. Generalization from a factual residual to an unseen operator composition remains empirical and must be labeled so.

**Cost.** Approximately 50–150K writer parameters. Inference overhead below 5% after the one-time write; meta-training through the update likely adds 50–100% step cost. A 2,000-step screen is roughly 1–3 A40-hours.

**Cheap falsification.** Hold out compositions of primitive edits rather than only family labels. Compare one fact against three-step Adam, the orthogonal writer, and history conditioning. Require at least +5 pp on future held-out-composition queries, no old-fact loss, and update-cap activation below 5%.

**Expected effect, speculation.** Post-fact operator stratum: +8–20 pp; aggregate changed-exact +1–5 pp; live RHAE +1–6 points.

**Failure modes.** Writer meta-overfitting, decoder exploitation, update accumulation drift, and globally harmful changes from a local residual. Detect with action-shuffled facts, zero-residual controls, held-out compositions, spectral norms, representation variance, and exact factual replay.

## Safe online objective contract

Regardless of proposal, I would prohibit online EP, entropy minimization, and latent-only reconstruction. A one-episode sample is too small for a marginal anti-collapse statistic, and entropy minimization can confidently reinforce a wrong deterministic mechanic.

Use:

```text
L_online =
    changed-pixel CE
  + 0.5 copy-gate BCE
  + 0.1 inverse-action loss
  + 0.1 observed-event loss
  + λ anchor divergence on unobserved actions
```

with hard constraints:

- Score every transition before adaptation.
- Base, encoder, and decoder remain immutable.
- Exact factual outputs may never regress.
- Unobserved action branches stay within a fixed functional trust region.
- Every update is episode-local and reset.
- Alias, rank exhaustion, or constraint infeasibility causes abstention, not a weaker safety threshold.

## Recommended decision sequence

1. Run the matched-counterfactual ambiguity census. Do not trust the current H1/H2 ceiling without repeated-group coverage.
2. Add operator-bearing temporal meta-episodes. This is required by proposals 2, 5, and 6.
3. Test canonical edit memory and the orthogonal writer independently on the frozen foundation-v2 checkpoint.
4. Combine them only if memory remains exact and the writer adds future-query generalization.
5. Use the expert bank as an upper-bound/control for “recognition of known mechanics.”
6. Attempt the tiny history model only after one-fact causal gain is demonstrated by the cheaper mechanisms.

Research status: **DONE_WITH_CONCERNS**. The synthetic ceiling is quantitatively bounded, but the fraction of private ARC-AGI-3 mechanics requiring in-episode adaptation is not publicly measurable. No repository files were modified.
