# Executive verdict

Assuming foundation-v2 reaches 55–70% changed-pixel exact accuracy, I would bet the next wall is not raw parameter count or patch-4 latent capacity. It is the combination of:

1. A loss reduction that overweights the changed stratum more aggressively than intended.
2. A weak, linear patch-local readout that is being treated as though its 0.67 foreground reconstruction plateau were a property of the latent.
3. Hidden episode mechanics that cannot be inferred from one frame, while the base dynamics has no history-conditioned mechanics state.
4. A hard, statistically inconsistent copy-gate compositor that becomes consequential once imagined states feed planning.

Patch-4 capacity and residual EP pressure are second-tier suspects. Generic scaling would be a poor next experiment.

All projected effect sizes below are speculative priors, not evidence. Cost estimates use the repository’s observed A40 range of approximately 4.7–9 seconds/update, making 2,000 baseline updates roughly 2.6–5 GPU-hours ([RESULTS_P2, lines 694–704](/home/stepan/Coding/Personal/Tofy/docs/RESULTS_P2.md:694), [A40 batch-2048 report, lines 28–34](/home/stepan/Coding/Personal/Tofy/docs/research/raw-2026-08-24/arch-symmetry-action-scaling.md:28)).

## What is actually binding

| Rank | Candidate | Verdict |
|---:|---|---|
| 1 | Changed-pixel weighted-CE reduction | Very likely immediate constraint |
| 2 | Encoder/readout seam behind 0.67 foreground reconstruction | Likely, but “decoder ceiling” is not yet proved |
| 3 | One-frame hidden-mechanics ambiguity | Likely live-score constraint; aggregate size unmeasured |
| 4 | Hard copy-gate composition | Not responsible for the reported raw changed-exact metric, but dangerous for planning |
| 5 | Patch-4 subpixel/translation representation | Plausible geometry constraint |
| 6 | Residual EP pressure | Possible late-stage tax, probably not the main wall |
| 7 | Raw patch-4 latent capacity | Unlikely next wall |

The strongest repository evidence is:

- The prediction loss computes a mean over changed pixels and a separate mean over unchanged pixels, then multiplies the already-normalized changed mean by `(1-p)/p` ([train.rs, lines 3418–3472](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3418), [lines 3635–3652](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:3635)). At `p = 0.04`, equal per-pixel errors contribute approximately 24 times as much through the changed stratum. Conventional inverse-frequency weighting followed by a single population mean would instead balance total changed and unchanged mass. The gate BCE uses the latter, mass-normalized construction; the pixel CE does not. This is a concrete objective mismatch, not speculation.
- The “decoder” is one linear projection per 4×4 latent token to `16 positions × 16 colors`, with no cross-token spatial refinement ([grounding.rs, lines 40–130](/home/stepan/Coding/Personal/Tofy/src/p2/grounding.rs:40)). The encoder, meanwhile, collapses pixels through a stride-patch convolution before further convolutions ([model.rs, lines 430–484](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:430)).
- The headline changed-pixel exact metric uses the raw palette logits only, examines true changed pixels, and does not use the copy gate ([eval.rs, lines 2260–2303](/home/stepan/Coding/Personal/Tofy/src/p2/eval.rs:2260)). Therefore copy-gate calibration cannot explain the reported 51.4%, and that metric does not expose false edits to unchanged pixels.
- Episode operators are sampled per episode and recorded in sidecar provenance ([data.rs, lines 553–575](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:553), [lines 1681–1704](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:1681)), but the dynamics receives the current latent, action, and goal—not an inferred mechanics state ([model.rs, lines 1021–1144](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:1021)).
- The implemented ambiguity census only measures one-frame and a limited two-frame key ([semantic_eval.rs, lines 461–545](/home/stepan/Coding/Personal/Tofy/src/p2/semantic_eval.rs:461)); it does not establish the requested history-1…16 ceiling. Moreover, its reconstructed content rectangle is top-left anchored while v5 data can translate the rectangle ([semantic_eval.rs, lines 358–391](/home/stepan/Coding/Personal/Tofy/src/p2/semantic_eval.rs:358), [data.rs, lines 1359–1376](/home/stepan/Coding/Personal/Tofy/src/p2/data.rs:1359)). Treat by-source ambiguity numbers cautiously until this measurement seam is corrected.
- The ADR describes separate EP clipping, but the training path performs the adaptive EP calculation and then a combined backward/global clip ([train.rs, lines 5634–5637](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:5634), [lines 5935–5984](/home/stepan/Coding/Personal/Tofy/src/p2/train.rs:5935)). EP can therefore still rotate or consume the shared update even while its measured encoder-gradient ratio is capped.

The 0.67 target-foreground reconstruction plateau is a strong alarm, but not yet a proved decoder ceiling. A predictor could theoretically emit decoder-friendly latents different from encoded targets. In practice, the latent-matching and encoded-target losses tether those spaces, so a readout bottleneck can still constrain prediction. A frozen-encoder nonlinear decoder probe is the decisive test.

## Ranked proposals by expected value per A40-hour

### 1. Joint Bayes copy/color compositor — EXPLOIT

**Mechanism.** Replace hard `gate > 0.5` composition with a normalized posterior over final colors. Given change probability `q_i`, current color `x_i`, and raw color probabilities `r_i(c)`:

```text
P(Y_i = x_i) = (1 - q_i) + q_i r_i(x_i)
P(Y_i = c)   = q_i r_i(c), for c != x_i
```

Decode with the global 16-color argmax. Fit source-stratified temperature and bias parameters for `q_i` and `r_i` on a frozen calibration split using final-color NLL or Brier score. This changes no backbone tensors.

**Why it should work.** The final-color argmax is Bayes-optimal for per-pixel exact accuracy when these probabilities are calibrated. Hard gating is not. For example, with `q = 0.51`, current-color likelihood `0.19`, and best alternative `0.20`, hard gating selects the alternative, although the composed current-color probability is `0.49 + 0.51 × 0.19 = 0.5869`, versus `0.102` for the alternative. This is a finite theorem over a 16-element outcome space and extends the existing mode-optimality formalization ([ModeOptimality.lean, lines 19–29](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/ModeOptimality.lean:19)).

**Cost.** Zero model parameters; minutes offline or under 0.05 GPU-hour.

**Cheap falsifier.** Re-score one existing checkpoint with raw, hard-gated, and joint-Bayes decoders. Report final-color NLL, ECE, changed exact, full-transition exact, and false-change rate. Reject if final-color NLL does not improve or no composed exact metric gains at least 1 point.

**Expected effect, speculative.** No change to the current raw changed-exact headline because that metric bypasses the gate; +1–4 points on composed changed exact, +2–8 points on full-transition exact, and 0–1 live point before planning. Its larger value is preventing miscalibrated imagined edits from poisoning planning.

**Failures and early detection.** Conditional dependence between gate and color errors can defeat scalar calibration. Detect through per-source reliability diagrams and calibration gains that disappear under operator or board-size stratification.

---

### 2. Exact-cardinality changed-set likelihood: Cardinality × Where × Value — EXPLOIT

**Mechanism.** Replace independently weighted pixel CE with a normalized successor-edit distribution:

- A pooled head predicts `P(K | z_t, a_t)`, where `K` is the number of changed pixels.
- A location head emits scores `s_i`.
- Conditional on `K`, define the probability of an unordered changed set `E` as:

```text
P(E | K) = exp(sum(i in E) s_i) / e_K(exp(s_1), ..., exp(s_N))
```

Here `e_K` is the K-th elementary symmetric polynomial, computed with a streaming log-space dynamic program.
- For each selected location, a value head predicts one of the 15 colors excluding the current color.
- Copy is exact outside `E`.
- Average `CE_count - log P(E | K) + sum value_CE` per transition, not per pixel stratum.

Use `K <= 32` initially. Route larger changes through a two-level tile-count/location distribution rather than enlarging the dynamic program indefinitely.

**Why it should work.** If the true changed set is not the distribution’s MAP set, its probability is at most `1/2`; therefore:

```text
1[MAP set is wrong] <= -log P(true set) / log(2)
```

This is the direct set-level analogue of the repo’s pixel-CE exactness bound ([CrossEntropy.lean, lines 54–90](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/CrossEntropy.lean:54)). Correct count, set, and values are also necessary and sufficient for an exact successor under copy composition. Set likelihoods and explicit cardinality prediction are established ideas in [DeepSetNet, arXiv:1611.08998](https://arxiv.org/abs/1611.08998); the action-conditioned exact changed-pixel likelihood and deterministic compositor are the beyond-frontier step.

**Cost.** Approximately 5–20K additional parameters if the current color head is reused. Streaming `O(NK)` normalization should add roughly 5–15% at `N <= 1024`, `K <= 32`. A 1,000-step frozen-backbone head screen costs about 1.3–2.5 A40-hours.

**Cheap falsifier.** Freeze the EMA encoder and predictor; train only count, location, and value heads for 1,000 steps. Compare with a parameter-matched Bernoulli-mask head. Require improved set exactness, no higher false-change rate, and at least +2 points changed-color exact before joint training.

**Expected effect, speculative.** +3–8 points changed-pixel exact, +5–12 points target/predicted foreground reconstruction, and 0–3 live points before planning.

**Failures and early detection.** Large-`K` tails, spatially structured edits inadequately represented by additive location scores, or unstable Candle log-space DP. Monitor `K` coverage, NLL by `K`, set recall, unchanged false positives, and numerical parity against a small CPU enumerator.

---

### 3. Frozen nonlinear-decoder probe, then two-timescale decoder/predictor training — EXPLOIT

**Mechanism.** First test the ceiling rather than assuming it. Freeze the encoder and train a decoder consisting of:

```text
3×3 latent-grid conv, 128 -> 128
SiLU
residual connection
1×1 projection to 4×4×16 subpixel logits
```

If the probe succeeds, co-train on two timescales:

1. A decoder step fits stop-gradient encoded current/next latents.
2. A predictor step optimizes through an EMA snapshot of the decoder.
3. Update the decoder EMA only after its held-out target reconstruction remains stable.

This prevents the decoder and predictor from jointly moving the definition of a “good” latent every update.

**Why it should work.** The probe separates two hypotheses:

- If a nonlinear frozen-encoder decoder jumps from 0.67 to above roughly 0.82, the encoded target contains the information and the linear readout class is binding.
- If it does not, the encoder has discarded foreground/subpixel information and a decoder-only change cannot help.

A measurable invariant is simultaneous non-increase of target-latent validation NLL and predictor-to-target decoded KL. The alternating scheme is accepted only while both improve.

**Cost.** About 180K parameters: roughly 147K for the 3×3 convolution and 33K for the subpixel projection. Expect +15–30% step time. A 512-step frozen probe costs approximately 0.7–1.3 A40-hours; a positive probe justifies up to 2,000 joint steps.

**Cheap falsifier.** Run only the 512-step decoder probe. Reject nonlinear decoding if held-out foreground reconstruction gains under 10 points or if gains are confined to train sources.

**Expected effect, speculative and conditional on a positive probe.** +10–20 foreground-reconstruction points, +2–6 changed-exact points, and 0–2 live points.

**Failures and early detection.** Decoder memorization, moving-target co-adaptation, or predictor latents falling off the encoded-latent manifold. Detect with per-source held-out reconstruction, target-versus-predicted decode gaps, and nearest-encoded-latent distance.

---

### 4. Lossless palette register plus continuous mechanics latent — EXPLORE

**Mechanism.** Split world state into:

- `G_t`: an exact discrete 64×64 palette register.
- `h_t`: the continuous recurrent latent used for mechanics, goals, uncertainty, and planning.

Predict the structured edit `(K, E, V)` from `(G_t, h_t, a_t)` and update:

```text
G_(t+1) = apply_edits(G_t, E, V)
```

Copy outside `E` is deterministic. Continue predicting `h_(t+1)` for long-horizon reasoning, but do not ask it to reconstruct unchanged pixels. During imagination, carry both the exact register and the continuous latent. Apply EP only to `h`, never to the palette register.

This is not a learned VQ tokenizer. The grid’s native 16-color alphabet is already an exact discrete representation, so there is no codebook or decoder error. Discrete latent world models such as [DCWM, arXiv:2503.00653](https://arxiv.org/abs/2503.00653) establish the broader frontier; a lossless palette register coupled to a JEPA mechanics state goes beyond it.

**Why it should work.** A finite-array theorem is immediate:

```text
apply_edits(G, E_hat, V_hat) = G_next
iff
E_hat is the true changed set and every predicted value is correct
```

Target reconstruction becomes 100% by construction. Model capacity is spent on the conditional edit rather than restating approximately 1,000 unchanged pixels.

**Cost.** Approximately 10–50K head parameters beyond proposal 2; +5–15% compute. The `u8` grid register costs about 8 MB at batch 2048, with temporary float masks adding tens of MB. A frozen structured-head screen plus joint continuation fits within 2,000 steps, approximately 2.8–5.7 A40-hours.

**Cheap falsifier.** Train an edit head on a frozen foundation-v2 checkpoint, conditioning on the observed palette grid and predicted latent. Proceed only if composed changed-exact beats raw decoder exact by at least 3 points without increasing unchanged false edits.

**Expected effect, speculative.** +4–10 points changed exact if the readout seam is real; target reconstruction ceiling eliminated; +1–5 live points once planning consumes composed states.

**Failures and early detection.** The continuous latent and register may become inconsistent, UI/status mutations may not fit the edit contract, and long imagined rollouts may accumulate incorrect sparse edits. Track `encode(G_hat)` versus `h_hat`, residual unexplained changes, rollout edit-count drift, and re-anchor both states after every real observation.

---

### 5. Mechanics-posterior conditioning of the base world dynamics — EXPLORE

**Mechanism.** Maintain a recurrent posterior `b_t` over episode mechanics. Update it only from factual transitions:

```text
b_(t+1) = GRU(b_t, z_t, a_t, edit_summary(x_t, x_(t+1)), z_(t+1))
```

Inject `b_t` through low-rank FiLM into every recurrent dynamics step. During training:

- Distill `b_t` toward the known sidecar operator when available.
- Also train it by future-transition likelihood so the representation need not match the generator taxonomy.
- Apply prior dropout and shuffled-history controls to prevent seed or augmentation leakage.

At play time there is no operator label; only the accumulated posterior is used. This differs from ADR Phase C’s belief input to proposal/progress heads: it makes the base one-step world model history-conditioned before planning.

[HiP-RSSM, arXiv:2206.14697](https://arxiv.org/abs/2206.14697) establishes hidden-parameter recurrent world models. The proposed finite operator posterior tied to exact edit evidence and injected into a tiny JEPA predictor is the more specific step.

**Why it should work.** Let `M` be finite hidden mechanics. A one-frame predictor is bounded by the modal successor frequency within each `(x_t, a_t)` collision class. If a history partition separates every pair of mechanics that produce different next states, then the posterior equivalence class is a sufficient state and the next transition is deterministic. This partition-refinement statement is Lean-formalizable and extends the existing non-identifiability theorem ([Identifiability.lean, lines 16–53](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/Identifiability.lean:16)).

**Cost.** A 64-dimensional GRU plus rank-16 FiLM costs about 60–120K parameters and 5–10% step time. A 2,000-step continuation is approximately 2.8–5.5 A40-hours.

**Cheap falsifier.**

1. Construct counterfactual episodes sharing the exact visible frame and action under different operators.
2. Measure oracle-mechanics, inferred-history, shuffled-history, and one-frame predictors.
3. Train only posterior/FiLM parameters from the EMA checkpoint for at most 2,000 steps.

Reject if the oracle operator itself gives less than a 3-point gain on the collision-controlled slice.

**Expected effect, speculative.** +1–5 aggregate changed-exact points, +10–25 points on hidden-operator collision cases, and +3–10 live points when factual interaction reveals mechanics.

**Failures and early detection.** Some episodes contain no identifying action; history cannot manufacture missing evidence. Other risks are posterior label leakage and inadequate history length. Measure oracle-versus-inferred gaps, posterior entropy, history-shuffle degradation, and exact same-state/action cross-operator collisions.

---

### 6. Phase-equivariant, lossless patch representation — EXPLORE

**Mechanism.** Replace the learned stride-4 collapse with fixed pixel unshuffle. Preserve all 16 within-patch phases explicitly:

```text
[batch, color_embed, 64, 64]
    -> pixel_unshuffle_4
[batch, 16 × color_embed, 16, 16]
```

Project each phase lane into a fixed slice of the 128-dimensional token, then use controlled phase mixing and spatial convolutions. The decoder maps each phase lane back to its exact subpixel position. Coordinate actions are decomposed into patch coordinate plus phase coordinate.

**Why it should work.** Pixel unshuffle is bijective and exactly equivariant to single-pixel translations up to a known permutation of phase lanes:

```text
unshuffle(shift_1(x)) = phase_permute_and_patch_shift(unshuffle(x))
```

That statement is finite over the 4×4 phase group and Lean-formalizable. Learned patch embedding and subsampling are known to disrupt shift equivariance; polyphase methods explicitly target this problem ([arXiv:2306.07470](https://arxiv.org/abs/2306.07470)). This is a more exact construction because no input information is discarded before the dynamics sees it.

**Cost.** Similar activation width, roughly 50–150K additional phase-mixing parameters, and +10–25% step time. A 512-step reconstruction/translation probe costs 0.7–1.5 A40-hours; a matched 2,000-step screen costs about 3–6.3 hours.

**Cheap falsifier.** Train a parameter-matched autoencoder/predictor probe and compare:

- Accuracy by each of the 16 within-patch offsets.
- Single-pixel translation consistency.
- Coordinate-action cases near patch boundaries.
- Random-placement versus fixed-placement performance.

Reject if the worst-phase accuracy gap is already under 2 points or the equivariant representation yields under 3 points improvement on translated cases.

**Expected effect, speculative.** +2–6 aggregate changed-exact points, +5–12 points on translated/coordinate-action strata, and 0–3 live points.

**Failures and early detection.** Existing convolutions may already recover phase information, or strict phase structure may impede necessary mixing. Use per-phase confusion matrices and a parameter-matched unconstrained control.

---

### 7. Feasibility-only anti-collapse controller — EXPLOIT

**Mechanism.** Replace positive EP pressure with a constraint controller. Define minimum acceptable conditions for:

- Semantic covariance rank/eigenvalues.
- Same-state branch separation.
- Action retrieval from displacement.
- Target reconstruction.

Set the EP multiplier to zero whenever all constraints hold. Increase it only when a measured constraint is violated, using a projected dual update. Do not include EP in the task gradient merely because budget remains available.

**Why it should work.** A fixed positive regularizer generally shifts the prediction optimum whenever its gradient is nonzero. A feasibility constraint has exactly zero pressure in the interior of the acceptable set. A two-dimensional quadratic counterexample suffices to formalize this. The repository already proves that marginal anti-collapse statistics can be blind to semantic organization ([MarginalBlindness.lean, lines 18–54](/home/stepan/Coding/Personal/Tofy/formal/TofyFormal/MarginalBlindness.lean:18)); spending prediction gradient to improve such a statistic after semantic gates pass has no guaranteed benefit.

**Cost.** No new trainable parameters and potentially 1–3% faster training. A matched 1,000-step current-controller arm and 1,000-step feasibility arm total roughly 2.6–5 A40-hours.

**Cheap falsifier.** Continue the same EMA checkpoint with identical batches and schedule in two 1,000-step arms. Reject feasibility-only control if any semantic-rank, branch-separation, reconstruction, or shuffled-action gate degrades beyond preregistered tolerance.

**Expected effect, speculative.** 0–3 changed-exact points and 0–1 live point. The probability of a gain is lower than proposals 2–6, but the implementation and compute costs are small.

**Failures and early detection.** Delayed collapse after the 1,000-step screen, badly chosen feasibility thresholds, or a collapse direction invisible to the monitored statistics. Fail closed by reactivating EP if any rank/action/retrieval gate crosses its threshold.

---

### 8. Exact compositional edit algebra with sparse residuals — EXPLORE

**Mechanism.** Parse each transition into a small operation program:

```text
object/region selector
× spatial operation: move, copy, delete, expand
× displacement or extent
× palette operation
+ residual exact changed set
```

Train permutation-invariant object/region slots, operation logits, displacement, and palette-map heads. A deterministic compositor executes the factors. Pixels unexplained by the factors are handled by the exact-set residual from proposal 2. Use minimum-description-length tie-breaking when multiple parses are valid.

Object-centric dynamics and compositional adaptation are at the current frontier in [Dyn-O, arXiv:2507.03298](https://arxiv.org/abs/2507.03298) and [WLA, arXiv:2503.09911](https://arxiv.org/abs/2503.09911). Exact palette-edit algebra with a lossless residual provides a stronger exactness contract than free continuous object slots.

**Why it should work.** Under the explicit assumption that an operator factors as `T_(u,v) = Value_v ∘ Spatial_u`, identifying all factor values requires coverage proportional to `|U| + |V|`, rather than observing every one of the `|U| × |V|` combinations. This finite identification theorem is formalizable. Exact output remains guaranteed whenever the factors and residual set are exact.

**Cost.** Roughly 100–250K parameters, +10–20% model time, plus offline CPU parsing. A 1,000-step oracle-factor head and 1,000-step inferred-factor continuation cost about 3–6 A40-hours.

**Cheap falsifier.** Before training, parse the existing generated transitions. Reject if the best simple operation program leaves more than 20% of changed pixels in the residual for the median example or if parse ambiguity remains high after MDL tie-breaking. Then test whether oracle factors materially improve held-out-operator prediction.

**Expected effect, speculative.** +2–7 points aggregate changed exact, +8–20 on held-out operator compositions, and +2–8 live points.

**Failures and early detection.** Non-compositional rules, unstable object identities, ambiguous decompositions, and large paint/flood residuals. Track residual fraction, program length, oracle-versus-inferred gaps, and factor confusion by operator.

---

### 9. Ambiguity-preserving successor particles — EXPLORE

**Mechanism.** Add four shared-trunk successor-edit heads with mixture weights supplied by the mechanics posterior. Train factual conditional likelihood:

```text
-log sum(k = 1..4) pi_k P_k(edit_set, values)
```

Use same-visible-state/action counterfactual groups to align heads with distinct factual outcomes. Diversity is rewarded only when different outcomes actually occur; do not use generic ensemble disagreement. Before mechanics is identified, planning expands the top successor modes. As history accumulates, the posterior collapses their weights.

This is aleatoric hidden-mechanics modeling, not ADR Phase B’s epistemic bootstrap ensemble.

**Why it should work.** In a collision group with deterministic hidden mechanics, every single-output predictor is bounded by the largest successor-mode frequency. A `K`-component model can represent up to `K` distinct factual successors exactly. Once history identifies the mechanics, its posterior selects the appropriate deterministic component. Both statements are finite counting arguments.

**Cost.** Roughly 100–250K parameters and +20–40% head/decode compute. A 2,000-step frozen-trunk screen is approximately 3.5–7 A40-hours. At planning time, decode only the top two modes unless residual mass is high.

**Cheap falsifier.** Build a collision-controlled set across episode operators and train only the mixture heads. Require improved top-`K` successor recall, factual head diversity, and likelihood over a single-head control. Reject if utilization collapses to one head or “diverse” heads do not correspond to observed outcomes.

**Expected effect, speculative.** Only 0–2 top-1 changed-exact points without history; +2–6 after mechanics conditioning; potentially +2–8 live points because planning no longer treats an arbitrary modal successor as certain.

**Failures and early detection.** Slot collapse, permutation instability, more than four relevant mechanics modes, or branching explosion. Monitor component utilization entropy, pairwise edit-set distance tied to factual outcomes, top-`K` recall, and posterior residual mass.

## Recommended experiment order

The cheapest decisive sequence is:

1. Re-score the current checkpoint with the joint Bayes compositor.
2. Run the frozen nonlinear-decoder probe.
3. In parallel conceptually—but not in the same causal experiment—measure a corrected history-1…16 collision ceiling including previous actions and outcomes.
4. If the nonlinear probe is positive, test exact-cardinality set prediction on the frozen backbone.
5. If it is negative, skip decoder scaling and go directly to the lossless palette register or phase-equivariant encoder.
6. Promote mechanics-posterior conditioning only if the oracle-mechanics upper bound is material.
7. Use successor particles only where the corrected census proves irreducible pre-identification multimodality.

My strongest bet is proposal 2 followed by proposal 4: stop treating a deterministic sparse grid transition as 4,096 nearly independent imbalanced classifications. Represent it as an exact edit object and make copying an algebraic fact.

## Frontier and novelty boundary

V-JEPA 2 and DINO-WM establish spatial latent prediction and latent-space control ([V-JEPA 2, arXiv:2506.09985](https://arxiv.org/abs/2506.09985), [DINO-WM, arXiv:2411.04983](https://arxiv.org/abs/2411.04983)). DCWM establishes discrete latent world modeling; HiP-RSSM establishes hidden-parameter recurrent dynamics; DeepSetNet establishes set likelihood and cardinality prediction. None of those primary sources establishes the combined package proposed here: normalized exact changed-set likelihood, lossless native-palette state, history-conditioned base dynamics, and a deterministic edit compositor for tiny interactive grid worlds.

That is a bounded literature conclusion, not a proof that no similar unpublished or differently named method exists.

**Status: DONE_WITH_CONCERNS.** The main unresolved evidence gaps are the unmeasured corrected history-1…16 ambiguity ceiling and the absence of a nonlinear frozen-encoder decoder probe. No repository or research-library files were changed, per the requested read-only constraint.
