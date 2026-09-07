# Looped-transformer control implementation contract

## Claim and boundary

Empirical hypothesis: a small shared-attention recurrent core can infer a new
action-to-direction permutation from three real observed calibration transitions
and use it to solve new synthetic maze layouts with an observable goal. Compare
direct action selection and bounded search on the same learned model. This is a
first prerequisite, not ARC performance, autonomous exploration, hidden-objective
inference, or evidence of universal architectural superiority.

The environment is independently generated. No ARC game assets, trajectories,
source, checkpoints trained on public games, or language-model weights enter it.
Calibration is scripted initially, counted separately and included in total
interaction cost. Later work must learn probe selection and hidden goals.

Three distinct calibration actions in a four-way permutation uniquely determine
the fourth. Their directions are recoverable from observed coordinate changes;
the current maze alone has identical pixels under all permutations. Thus the
information prerequisite holds under a no-wall, nonzero-motion calibration
construction. This does not prove a neural network will learn the inference.
Mutually exclusive action branches are separate supervised targets, never a
fabricated chronological history.

## Implementation separation

- `src/p2/looped_agent/task.rs`: pure synthetic simulator, chronological support,
  raw palette frames, leakage-free model inputs and observation-limited oracle.
- `src/p2/looped_agent/model.rs`: learned palette/patch input, shared transformer
  reasoning, policy/value/next-pixel/reward heads. No simulator calls or mechanics.
- `examples/looped_agent_probe.rs`: training, capacity smoke and closed-loop
  evaluation, honest controls, fresh roots and provenance.

Model interface, consumed by the root-owned runner:

```text
LoopedConfig { hidden: usize, heads: usize, layers: usize, max_loops: usize }
LoopedAgent::new(config: LoopedConfig, vb: VarBuilder) -> Result<Self>
LoopedAgent::forward(patches: &Tensor, metadata: &Tensor, loops: usize)
  -> Result<LoopedOutput>

patches: U32 [B, TOKENS=448, PATCH_PIXELS=64], categorical palette 0..15
metadata: F32 [B, TOKENS=448, META_DIM=10]
policy_logits: [B, ACTIONS=4]
value: [B, 1], unconstrained value logit (runner sigmoid for discounted success)
reward_logits: [B, ACTIONS=4], immediate success classifier, not task value
next_logits: [B, ACTIONS=4, PATCH_COUNT=64, PATCH_PIXELS=64, PALETTE=16]
```

The six support frames are before/after pairs in real chronology followed by
the current frame. Last 64 patch tokens belong to the current observation.
Patch ordering is row-major at both patch and within-patch levels. Metadata is
before/after/current at 0..3, action one-hot at 3..7 (zero for current), normalized
patch x/y at 7..9, support index / 3 at 9 (current uses 1). Inputs contain no
privileged direction mapping, path, oracle distance or hidden game state.

Use a learned patch embedding from all 64 categorical pixels (no average pooling
or game-specific frame parser), learned metadata projection, and attention over
spatial/context tokens. A shared small transformer stack repeats with current
input recall. Shared parameters must not grow with requested inference loops.
Use distinct readouts for policy, value, immediate success and each action's full
next frame. No token-by-token language generation. Initialization and training
are from scratch. Stochastic branching, halting and online weight adaptation are
deferred interventions, not claimed completed work.

## Evidence and launch rules

The architecture rationale is the local research library's
`2026-09-07T044851Z-tofy-looped-transformer-direction` synthesis and its primary
papers: PTRM v1, Loop-Think-Generalize v2, Guez v2, Hamrick v2 and QWM v1. This
implementation is an adaptation; none of their performance claims transfers.

Before evidence training, register exact generator/splits, model/depths,
initializations, updates/data order, objective weights, comparator budgets,
metrics/uncertainty, thresholds, checkpoint selection, stop rule, wall-clock cap,
and physical batch/accumulation selected by CUDA capacity tests. Run only a
reviewed pushed clean source, separately hashed binary, exact dependency revision,
and bounded smoke. No placeholder or untrained module can justify a long public
evaluation. Preserve negative results and failed launches.
