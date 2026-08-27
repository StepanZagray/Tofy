# Beyond-frontier architectures for exact deterministic grid dynamics

**Status: DONE_WITH_CONCERNS.** The current `foundation-v2` checkpoint and live run bundle are not present under checked-in `runs/`, so costs and effect sizes below are explicit engineering priors, not measured results.

## Blunt verdict

I would **not** blame the CNN-grid representation at 51.4% changed-transition exact accuracy.

My bet on the next binding constraints is:

1. **The grounding/decoder seam.** Encoded-target foreground reconstruction is only about 0.67, so the latent-to-palette map is still lossy.
2. **Independent-pixel output factorization.** Moves are being learned as many repaint decisions instead of one transport event.
3. **Support/cardinality calibration.** The model has a binary copy gate but no global assertion that exactly `N` pixels change.
4. **Patch phase and symmetry.** Stride-4 tokens create sixteen sub-patch phases that augmentation does not identify architecturally.
5. **Only then:** absence of persistent object identity and long-range recurrent computation.

There is also an important measurement caveat: the trainer’s `one_step_changed_exact` checks only factually changed pixels, so hallucinated changes elsewhere do not count against it ([eval.rs:2260](/home/stepan/Coding/Personal/Tofy/src/p2/eval.rs:2260)). It also obtains predictions from raw palette logits ([eval.rs:2294](/home/stepan/Coding/Personal/Tofy/src/p2/eval.rs:2294), [eval.rs:2355](/home/stepan/Coding/Personal/Tofy/src/p2/eval.rs:2355)), even though the model contains a separate composed copy-gate decoder ([model.rs:764](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:764)). Thus 51.4% is neither full-frame exactness nor the accuracy of the deployed copy composition.

The literature frontier already contains pointer outputs, copying, object-centric dynamics, pixel propagation, set-cardinality prediction, G-CNNs, and equilibrium layers: [Pointer Networks](https://proceedings.neurips.cc/paper_files/paper/2015/hash/29921001f2f04bd3baee84a12e98098f-Abstract.html), [CopyNet, arXiv:1603.06393](https://arxiv.org/abs/1603.06393), [DeepSetNet](https://openaccess.thecvf.com/content_iccv_2017/html/Rezatofighi_DeepSetNet_Predicting_Sets_ICCV_2017_paper.html), [PlaySlot](https://proceedings.mlr.press/v267/villar-corrales25a.html), [propagation-versus-generation video prediction](https://openaccess.thecvf.com/content_ICCV_2019/html/Gao_Disentangling_Propagation_and_Generation_for_Video_Prediction_ICCV_2019_paper.html), [G-CNNs](https://proceedings.mlr.press/v48/cohenc16.html), and [NEMON](https://proceedings.neurips.cc/paper/2021/hash/51a6ce0252d8fa6e913524bdce8db490-Abstract.html). What follows goes past that frontier by giving each architecture a finite discrete transition grammar and exact renderer.

Define `H₂k = 2000 × current_v5_step_seconds / 3600` A40-hours. This avoids fabricating wall-clock throughput; a 100-step timing gives the conversion immediately.

## Ranking by expected value per A40-hour

All accuracy effects are **speculative, additive percentage points on held-out changed-transition exact**, and are not mutually additive.

| Rank | Proposal | Class | Parameters | 2k-step screen | Expected effect |
|---:|---|---|---:|---:|---:|
| 1 | Cardinality-conditioned changed-pixel set | EXPLOIT | +0.02–0.08M | 0.5–0.7 `H₂k` | +3–8 pp |
| 2 | K-flow pointer/copy compositor | EXPLOIT | total 0.6–0.8M | 0.7–0.9 `H₂k` | +2–7 pp; +5–15 on moves |
| 3 | Hard connected-component atlas | EXPLORE | total 0.9–1.4M | zero-GPU oracle, then 0.7–1.0 `H₂k` | +3–10 pp |
| 4 | Dual static/dynamic event core | EXPLOIT | total 0.6–0.9M | 0.7–0.9 `H₂k` | +2–6 pp |
| 5 | Phase-preserving `D4 × translation` G-CNN | EXPLORE | total 0.7–1.2M | offline ensemble, then 1.4–2.0 `H₂k` | +0–3 iid, +3–8 symmetry OOD |
| 6 | Contractive fixed-point transition solver | EXPLORE | total 0.5–1.0M | depth sweep, then 1.5–3.0 `H₂k` | +0–5 pp; larger on propagation |

## 1. EXPLOIT — Cardinality-conditioned changed-pixel set

### Mechanism

From predicted latent `ŷ`, produce:

- location scores `s ∈ R^(B×4032)`;
- a count distribution `q(N)` for `N ∈ {0,…,64,overflow}`;
- color logits `v ∈ R^(B×4032×16)`.

Inference is:

1. `N̂ = argmax q`;
2. select the unique top `N̂` locations under `s`;
3. copy the current frame everywhere else;
4. write `argmax v_i` at selected locations.

Train with count CE, changed-support listwise loss, and color CE on true changed locations. No Hungarian matching is needed because top-`N` indices are unique and canonical.

This is materially different from the current independent sigmoid gate ([grounding.rs:103](/home/stepan/Coding/Personal/Tofy/src/p2/grounding.rs:103)): one global decision constrains the number of changes.

### Exactness argument

Let `Δ(x,y) = {(i,y_i) | x_i ≠ y_i}`. If count, selected support, and colors are correct, the copy-plus-edit renderer equals `y` at every pixel.

A formal error decomposition is:

```text
P(render ≠ y)
≤ P(N̂ ≠ |Δ|)
  + P(Ŝ ≠ support(Δ) | N̂ correct)
  + P(any selected color wrong | support correct)
```

This is finite, direct, and Lean-formalizable.

### Candle/A40 cost

Global pooling plus two `Linear` heads and one `1×1 Conv2d` adds roughly 20–80K parameters and under 5% training compute. Vendored Candle supplies contiguous descending sort through `sort_last_dim` ([sort.rs:258](/home/stepan/Coding/Personal/Tofy/vendor/candle-core/src/sort.rs:258)). The sort indices are nondifferentiable; training applies losses to pre-sort scores.

### Cheap falsification

Two screens on the frozen checkpoint:

- **Oracle-count screen:** use the true changed cardinality with current copy-gate scores and current color logits. No optimization.
- **Frozen-backbone head:** train only count and support heads for at most 2,000 updates.

Reject if oracle count gains less than 2 pp, count top-1 remains below 90%, or more than 5% of transitions overflow `N=64`.

### Failure modes

Large fills, whole-board recolors, and operators with hundreds of edits defeat sparse support. Monitor full-frame exactness as well as the existing changed-only metric, because the latter will not detect spurious edits.

## 2. EXPLOIT — K-flow pointer/copy compositor

### Mechanism

Use `K=4` or `K=8` transport registers. Each predicts:

- factored displacement logits `dx_k,dy_k ∈ {-63,…,63}`;
- optionally a D4 transform;
- per-destination operation logits over `COPY_SELF`, `COPY_FLOW_1…K`, and `PAINT`;
- palette logits only for `PAINT`.

For destination pixel `j`, `COPY_FLOW_k` reads the exact source palette value at `j-d_k`. Source clearing is an ordinary `PAINT(background)` operation. Conflict resolution is explicit destination priority, not duplicate scatter order.

Training labels come from a finite oracle: find the `K` displacements that cover the most changed pixels for which `target[j] = current[j-d]`, then label residual edits as paint.

This is the grid-world analogue of CopyNet selecting source tokens and pixel-propagation models warping visible content, but it predicts a compact exact transition program rather than a blended image.

### Exactness argument

For every destination pixel, the operation map selects either one source pixel or one literal color. Therefore, if displacement, operation and paint predictions are correct, the renderer is exactly correct.

More importantly, every transition expressible as at most `K` masked rigid transports plus literal paints lies in the hypothesis class. Oracle grammar coverage is measurable before training.

### Candle/A40 cost

Use `gather`, `index_select`, `where_cond`, and optionally `roll`; these are present in the vendored tensor API ([tensor.rs:989](/home/stepan/Coding/Personal/Tofy/vendor/candle-core/src/tensor.rs:989), [tensor.rs:1565](/home/stepan/Coding/Personal/Tofy/vendor/candle-core/src/tensor.rs:1565), [tensor.rs:1853](/home/stepan/Coding/Personal/Tofy/vendor/candle-core/src/tensor.rs:1853)). Expected cost is 0.1–0.3M additional parameters and 8–15% step time.

### Cheap falsification

Offline, compute:

- fraction of changed pixels covered by the best `K=1,2,4,8` displacements;
- fraction of transitions with at least 95% coverage;
- residual-paint cardinality;
- collision and source-ambiguity rates.

Then train the flow and operation heads on a frozen backbone for at most 2,000 updates. Reject `K=4` if oracle coverage is below 80% or median residual paint cardinality exceeds eight.

### Expected effect and failures

Speculative effect: +2–7 pp overall and +5–15 pp on move/teleport/push strata.

Failures are component deformation, recoloring, occlusion ordering, and repeated same-color source ambiguity. Detect them through oracle coverage, displacement top-1, operation confusion, destination collisions, and source-clearing errors.

## 3. EXPLORE — Hard connected-component atlas with exact rerendering

### Mechanism

Parse every palette frame inside the Rust agent into up to `S=32` four-connected, same-color components. Each slot stores literal data:

- exact bitmap within its bounding box;
- color, box, area, centroid and perimeter;
- pooled learned features;
- a stable lexical component index.

A small pairwise GNN predicts one operation per component:

`KEEP`, `DELETE`, `TRANSLATE`, `D4_TRANSFORM`, `RECOLOR`, `SPLIT`, plus painter priority. A residual set head handles genuinely new pixels or shapes.

Rendering begins from an exact palette buffer, deletes operated sources, applies the predicted transforms to the literal component masks, and paints residual edits. Unlike soft Slot Attention, neither masks nor colors are reconstructed from a continuous slot.

Object-centric dynamics and compositional GNNs are established in [PlaySlot](https://proceedings.mlr.press/v267/villar-corrales25a.html) and [compositional multi-object dynamics](https://proceedings.mlr.press/v205/driess23a.html); the beyond-frontier part is the hard symbolic atlas and exact renderer.

### Exactness argument

Define the grammar as an ordered list of literal-mask transforms plus residual writes. Any transition with a correct source parse and target decomposition is rendered exactly. A proof is induction over painter order: after operation `k`, every location whose final writer is at most `k` equals the target; later operations cannot affect locations outside their declared masks.

### Candle/A40 cost

CC extraction and exact painting can be bounded Rust loops inside the binary; they are not an external harness. Learned slot processing uses masked multiply, `sum`, `matmul`, broadcast, `Linear`, and small `S×S` pairwise tensors. At `S=32,d=96`, this is well below a general transformer’s cost.

Expected total: 0.9–1.4M parameters, 15–35% more step time, little inference cost relative to planning rollouts.

Candle does have differentiable `scatter` and `scatter_add` ([tensor.rs:1644](/home/stepan/Coding/Personal/Tofy/vendor/candle-core/src/tensor.rs:1644), [tensor.rs:1683](/home/stepan/Coding/Personal/Tofy/vendor/candle-core/src/tensor.rs:1683)). CUDA requires contiguous source/index/destination tensors and index dtype `u8`, `u32`, or `i64` ([cuda_backend/mod.rs:669](/home/stepan/Coding/Personal/Tofy/vendor/candle-core/src/cuda_backend/mod.rs:669)). For exact rendering, I would still use explicit occupancy planes and painter priority rather than duplicate-index scatter semantics.

### Cheap falsification

Run an offline oracle decomposition over held-out transitions:

- exact-transition coverage by component operations alone;
- changed-pixel coverage before residuals;
- median residual cardinality;
- component count overflow;
- split/merge frequency.

Promote only if component operations explain at least 90% of changed pixels and exactly cover at least 70–80% of transitions. Then train only the component-operation GNN for at most 2,000 updates.

### Expected effect and failures

Speculative effect: +3–10 pp overall, possibly +8–20 pp on rigid-component games.

Connected components are not always semantic objects. Same-color touching objects merge; one object may be disconnected; holes and background can carry semantics; occlusion can destroy identity. Detect this immediately through oracle grammar coverage rather than hoping learned slots repair the parser.

## 4. EXPLOIT — Dual static/dynamic event core

### Mechanism

Factor recurrent state into:

```text
static:  s = E_static(current_palette)
dynamic: d_0 = 0
         d_(k+1) = F(d_k, stop_gradient(s), action)
```

The static pathway is immutable during a transition. Only `d` is recurrent. Its output is an event tape: changed support, pointer transports and literal paints. The final frame is composed against the exact current palette, not decoded from `s+d`.

For latent planning, the event tape deterministically updates an internal integer palette memory and re-embeds it for the next imagined step. This remains self-contained.

The current model instead repeatedly transforms a single dense latent and only decides copy-versus-predict at the final decoder ([model.rs:1301](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:1301), [grounding.rs:40](/home/stepan/Coding/Personal/Tofy/src/p2/grounding.rs:40)).

### Exactness and measurable invariant

By construction, pixels outside predicted support cannot change. Thus:

```text
support correct  ⇒  all unchanged pixels exact
support + event payload correct  ⇒  entire next frame exact
```

A useful optimization invariant is that gradients from unchanged pixels never enter the dynamic core. This can be measured as unchanged-to-changed gradient norm and gradient cosine, rather than inferred from loss curves.

### Cost

A 64-channel static encoder plus 64–96-channel dynamic recurrence should total 0.6–0.9M parameters. Recurring only the dynamic half makes step cost roughly equal to or slightly below v5 despite the extra pathway.

### Cheap falsification

Freeze the current encoder and train a small dynamic adapter plus event head for at most 2,000 steps. Compare:

- changed exact;
- full-frame exact;
- false-positive changed pixels;
- changed/unchanged gradient norms;
- performance with static features detached versus trainable.

Reject if the dynamic support’s recall remains below the current copy gate or the static bypass reduces action-conditioned outcome retrieval.

### Expected effect and failures

Speculative effect: +2–6 pp. The chief risk is that “static” features also encode state variables needed for future dynamics. Detect this through rollout degradation and action-conditioned branch retrieval, not just one-step pixel accuracy.

## 5. EXPLORE — Phase-preserving `D4 × translation` equivariant core

### Mechanism

Replace the stride-4 encoder—which maps directly to a `16×16` grid ([model.rs:430](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:430))—with an exactly equivariant recurrent core:

1. Keep the gameplay canvas square; carry the synthetic status row in a separate scalar pathway.
2. Lift palette features into eight D4 orientation fibers.
3. Use D4 group convolutions with tied rotated/reflected kernels.
4. Do not discard patch phase. Either operate at `64×64`, or apply invertible space-to-depth and retain all sixteen `4×4` phase fibers. A one-pixel translation then acts by a known phase permutation plus a coarse-cell carry.
5. Use dilations such as `1,2,4,8,16` for long-range receptive field without downsampling.
6. Transform the spatial action field and ACTION6 coordinates under the same group representation; global action scalars remain invariant.

Candle implementation needs ordinary `conv2d`, `stack`, transpose and `flip` ([tensor.rs:2918](/home/stepan/Coding/Personal/Tofy/vendor/candle-core/src/tensor.rs:2918)). No custom CUDA or flash attention is required. [Learnable Polyphase Sampling](https://arxiv.org/abs/2210.08001) establishes that exact shift consistency can survive learned down/up-sampling; here the stronger choice is to retain every phase rather than select one.

### Provable sample-complexity gain

For a finite input set `X`, output set `Y`, and a free action of a finite group `G`, unrestricted deterministic functions number:

```text
|H_all| = |Y|^|X|
```

An equivariant function is determined by one representative per orbit:

```text
|H_equivariant| = |Y|^(|X| / |G|)
```

Therefore the leading `log |H|` term in a finite-class realizable PAC bound is divided by `|G|`.

For an ideal toroidal `64×64` canvas:

```text
|D4 × T_64| = 8 × 4096 = 32768
```

Relative to an ideal stride-4 convolution already sharing across the `16×16=256` coarse translations, the maximum incremental reduction is:

```text
32768 / 256 = 128
              = 16 missing pixel phases × 8 D4 orientations
```

That is the requested **provable upper bound**, not a prediction of 128× practical data efficiency. Boundaries, status information, non-free scenes, stabilizers, and operator asymmetries reduce the gain to the empirical mean orbit size. This limitation is consistent with formal generalization results, which prove strict gains only under explicit symmetry/distribution assumptions rather than a universal group-size multiplier ([Elesedy and Zaidi, 2021](https://proceedings.mlr.press/v139/elesedy21a.html)).

### Cost

A 32–64-channel full-resolution group core, or its phase-folded equivalent, should fit 0.7–1.2M parameters. Activation memory is approximately 2–4× the present coarse latent; expected A40 step cost is 1.4–2.0× v5 depending on dilation count.

### Cheap falsification

Before implementation:

- compute exact D4 equivariance defect over all eight transforms and conjugated actions;
- compute one-pixel translation defect for all sixteen patch phases;
- evaluate an eight-view D4 logit ensemble and a sixteen-phase translated ensemble on a fixed checkpoint.

Reject architectural D4 if the eight-view ensemble gains less than 2 pp and errors are not orientation-stratified. Reject phase preservation if translated-7×7 accuracy and one-pixel phase defect are already flat.

### Expected effect and failures

Speculative effect: 0–3 pp iid, 3–8 pp on translated/D4/operator-held-out splits.

The main risks are incorrect action conjugation, status-row leakage, boundary violations, and spending fibers on symmetry at the expense of semantic capacity. Unit tests should require near-zero layerwise equivariance defect before any training result is trusted.

## 6. EXPLORE — Contractive fixed-point recurrent transition solver

### Mechanism

Replace the unconstrained residual/RMS/clamp loop ([model.rs:1382](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:1382)) with a weight-tied map:

```text
h_(k+1) = tanh(A h_k + b(encoded_state, action))
```

Normalize every recurrent convolution so its induced infinity norm satisfies `||A||_∞ ≤ κ`, with `κ` initially 0.85–0.95. Action and observation enter additively; unrestricted FiLM scaling on recurrent state is removed or bounded.

Stop when either the discrete output is certified stable or a hard cap is reached.

### Convergence and decode certificate

Because `tanh` is 1-Lipschitz, the update is a contraction. Banach’s theorem gives a unique fixed point. From residual `r_k = ||h_(k+1)-h_k||_∞`:

```text
||h_k - h*||_∞ ≤ r_k / (1 - κ)
```

If decoder logits have certified Lipschitz bound `L`, and the current top-two palette margin at every pixel is greater than:

```text
2 L r_k / (1 - κ)
```

then no pixel’s final argmax can change at the fixed point. This certifies convergence of the deterministic answer, not its correctness.

Monotone equilibrium networks and non-Euclidean contraction networks establish the broader fixed-point frontier ([monDEQ](https://proceedings.neurips.cc/paper_files/paper/2020/hash/798d1c2813cbdf8bcdb388db0e32d496-Abstract.html), [NEMON](https://proceedings.neurips.cc/paper/2021/hash/51a6ce0252d8fa6e913524bdce8db490-Abstract.html)); the new element here is a finite palette-decision certificate used as the stopping rule.

### Cost

Parameters remain 0.5–1.0M because weights are tied. Easy transitions may stop near current depth; hard ones could require 8–16 iterations, giving 1.5–3× average compute. This is Candle-native: convolution, `tanh`, `abs`, reduction, comparison and broadcast.

### Cheap falsification

The current model already exposes evaluation with extra outer steps ([model.rs:1720](/home/stepan/Coding/Personal/Tofy/src/p2/model.rs:1720)). On the frozen checkpoint, evaluate depths `1,2,4,8,16` and record:

- residual ratios;
- pixel-decode stabilization depth;
- changed exact at every depth;
- propagation-distance and component-diameter strata.

The existing matched-compute schema reports only latent MSE ([eval.rs:584](/home/stepan/Coding/Personal/Tofy/src/p2/eval.rs:584)); semantic stabilization must be added to the analysis. Reject if extra computation does not improve the structurally hard strata or if decoded outputs oscillate.

### Expected effect and failures

Speculative effect: 0–5 pp overall, potentially +5–15 pp on long propagation/push-line transitions.

Contraction can make the dynamics too weak; `κ≈1` can make convergence too slow; a fixed point can converge perfectly to the wrong answer. Monitor certified residual, discrete stabilization, margin, iteration count and accuracy separately.

## When is the CNN grid itself binding?

My operational threshold is:

> **Start treating the grid-CNN topology as the primary bottleneck around 80–85% held-out synthetic changed-transition exact, but only after encoded-target reconstruction is at least 95% pixel-accurate and at least 97% exact on changed pixels.**

That 80–85% number is a **speculative triage threshold**, not a theorem. A more decisive criterion is a reproducible **at least 5 pp gain** from a structured renderer using the same checkpoint, parameter count and sample budget while action-conditioning and decoder-oracle gates are already green.

At the current 51.4% with 0.67 foreground reconstruction, topology is not convicted.

### Telemetry that would convict it

Use or extend these diagnostics:

- **Decoder ceiling:** compare `target_reconstruction`, predicted-next decoding, and composed copy decoding using the semantic mask schema, which already supports exact transition accuracy ([semantic_eval.rs:25](/home/stepan/Coding/Personal/Tofy/src/p2/semantic_eval.rs:25)).
- **Full-frame exactness:** add an all-gameplay-pixels exact metric alongside changed-only exact. Record false-positive unchanged edits.
- **Raw versus composed:** persist raw palette, current copy-gate composition, top-`N`, pointer-flow and oracle-render results on identical rows.
- **Patch-phase defect:** stratify by `(placement_x mod 4, placement_y mod 4)`, one-pixel translations, crossing a patch boundary, and maximum displacement.
- **D4 defect:** compare `f(gx,ga)` against `g f(x,a)` for all eight group elements, both as logit error and exact-palette disagreement.
- **Object pressure:** stratify by component count, area, perimeter, disconnectedness, split/merge, maximum translation and oracle CC-grammar coverage.
- **Action/data confounds:** retain factual branch outcome retrieval and changed-cell rows, already present in the schema ([eval.rs:686](/home/stepan/Coding/Personal/Tofy/src/p2/eval.rs:686)). A failed action selector is not a representation ceiling.
- **Recurrence:** record semantic accuracy and discrete stabilization per outer step, not merely mean residual norm, which is all the current recursion summary exposes ([eval.rs:477](/home/stepan/Coding/Personal/Tofy/src/p2/eval.rs:477)).
- **Representation probes:** current diagnostics stop at variance and effective rank ([eval.rs:400](/home/stepan/Coding/Personal/Tofy/src/p2/eval.rs:400)). Add frozen linear probes for pixel phase, component labels, coordinates, displacement, destination support and operator family.

The cheapest decisive sequence is oracle count, oracle flow coverage, oracle component grammar, D4/phase ensembles, then the depth sweep. Those five screens can identify the correct architectural seam before another full foundation run.
