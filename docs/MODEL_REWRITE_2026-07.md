# World-to-Qwen model rewrite (2026-07-23)

This document is the implementation record for the clean-slate architecture
after `code_poc_1784364765`. The invariant is unchanged: a learned world model
stores and retrieves fictional knowledge, then a trainable interface supplies
that knowledge to a frozen Qwen3-1.7B-Base decoder.

This is a research-grade candidate architecture, not a state-of-the-art result
until a clean run passes the held-out generation and causal-control gates.
Existing encoder, world, compressor, adapter, and Qwen-cross-attention
checkpoints are schema-incompatible and must not be resumed.

## Evidence from the failed run

- The selected world checkpoint was good at its proxy:
  `val_prediction_mse=0.0006566337` and association top-1 `1.0` at step 6,000.
  The live weights later became unstable: global gradient norm rose from about
  `0.046` to `20.17` and then `1564.02`.
- The latest weights bridge selected teacher-forced CE `0.2584163` and semantic
  gap `0.70437014`, but evaluation covered only 4 of 300 seen tasks and none of
  the held-out tasks.
- The context channel passed 193/300 seen tasks and 0/300 held-out tasks.
  Shuffled, swapped, and zeroed controls all passed 0 tasks. This shows
  high-capacity seen-task memorization, not compositional knowledge transfer.
- The old RAG ceiling passed 0/300 on both splits because the decoder contract
  induced instruction repetition. A failed ceiling makes every downstream
  transfer ratio uninterpretable.

The preserved logs and reports are in
`artifacts/runpod-aq9dzbs2741eyw-20260723/`.

## Architecture, module by module

| Module | Failure mode | Implemented replacement |
|---|---|---|
| Encoder attention | No relative position signal; LayerNorm/GELU blocks diverged from the pretrained-decoder conventions used elsewhere. | RoPE attention, pre-RMSNorm, residual blocks, and parameter-matched SwiGLU. |
| Encoder hierarchy | All local depth preceded pooling; global context reached tokens once, after which tokens could not refine. The layer ratio described parameters more closely than compute. | Exact depth allocation `pre-local → chunk/global → global-to-token feedback → post-global local refinement`. Seven layers become 3/2/2, with no dropped remainder. |
| Chunk pooling | Mean pooling erased which tokens mattered and made padding/partial chunks fragile. | Learned attentive pooling per chunk, explicit masks, adaptive chunk size targeting about 16 chunks, and re-pooling after token refinement. |
| Latent pretraining | A shared raw representation was asked to serve as both downstream state and masked-prediction output. Chunk/global targets came from a second corrupted view. | A learned encoder predictor head maps online states into prediction space. Targets are full unmasked token/chunk/global features; prediction weights are 70/20/10. The online full-view target and SIGReg anti-collapse objective preserve LeJEPA rather than silently changing the method to EMA/data2vec. |
| SIGReg | Flattening slots into the sample axis let different slot distributions compensate, while shortest-sequence truncation discarded long-context tokens and padded chunks contaminated the statistic. | Preserve rank-3 position semantics for token/chunk/global views and regularize encoder states, not predictor outputs. Mask-aware Epps–Pulley statistics use every real token/chunk position, weight by valid sample count, and skip under-supported positions. Chunking reuses identical sampled projections and is tested for exact equality with the full all-valid computation. World training regularizes every state and next-state slot. |
| Context compressor | A single cross-attention operation made each latent query an independent shallow read. | Three-layer Perceiver/Q-Former resampler: cross-attention to source memory, latent self-attention, SwiGLU, and final RMSNorm. |
| World transition | No stochastic regularization; the failed run could continue far beyond a catastrophic gradient excursion. | Attention/MLP dropout in the action-conditioned transition, gradient-spike rejection against an EMA and absolute floor, association-aware selection, and 3,000-step early stopping. |
| World objective | Slot structure was collapsed before distribution regularization. | Raw slotwise next-state MSE plus slotwise SIGReg remains the only optimized world objective. Retrieval association remains a selection/diagnostic constraint rather than a surrogate generative loss. |
| Decoder adapter | Shallow query readout; its learned queries could act like a constant soft prompt. | Three-layer latent resampler plus salience/global mixing. The all-zero-world output is subtracted with its graph attached, preserving both `adapter(0)=0` and the derivative of the actual centered function. |
| Qwen cross-attention | Every site used full 2048-wide Q/K/V/O projections: about 16.8M parameters per site and about 117M across seven sites. | Qwen stays frozen; every fourth layer receives a gated cross-attention site with a default 512-wide bottleneck. A site is about 4.2M parameters and seven sites about 29.4M, before gates/adapter. |
| Latent-language interface | Code CE had to discover both the world/Qwen coordinate transform and code generation at once. | A BLIP-2-style first stage aligns adapter slots to frozen Qwen target-token embeddings using fine-grained token-to-best-slot cosine plus symmetric pooled InfoNCE. World states are frozen in this stage. |
| Generation contract | Qwen was asked for complete source under repetitive negative instructions, while the evaluator expected exact API-bearing code. | Qwen sees a natural task comment and the actual Go function scaffold. Only the function body and closing brace are supervised/generated. A lexical brace scanner stops after the outer body closes. |
| Truncation | Long examples could silently discard supervised target tokens. | The prompt is truncated only after reserving the complete target plus EOS; an example that cannot fit is rejected. |
| Checkpoint selection | Teacher-forced CE and a semantic margin could qualify a checkpoint that did not generate. A substring API check could still accept malformed code or comments. | Periodic greedy autoregressive validation runs the real Go build and hidden harness for matched and wrong conditioning across the full function-disjoint validation set. Tree-sitter Go AST validation requires an exact package selector call. Eligibility requires the teacher-forced semantic gap, matched suite-pass rate, and matched-over-wrong advantage; behavior dominates the selection score. |
| Experimental ladder | Bridge training could start even when the frozen decoder could not use explicit documentation, and the pipeline could finish after a failed held-out transfer. | A mandatory RAG preflight runs separate seen and held-out splits before learned model training. Final knowledge-in-weights evaluation requires held-out matched pass rate plus positive advantages over shuffled, swapped, and zero controls. |
| Runtime controls | RAG/floor modes unnecessarily loaded encoder/world/bridge checkpoints. | Frozen-decoder controls load only Qwen. Zero conditioning is created in the model dtype, including BF16 CUDA execution. |

## Encoder depth and cost

The hierarchy deliberately does not call a `3/2/2` split a compute split.
A self-attention/SwiGLU block has roughly the same parameters at token and
chunk resolution, while its attention FLOPs scale quadratically with sequence
length. Therefore:

```text
3 token-local blocks
→ learned chunk pooling
→ 2 chunk/global blocks
→ global-to-token cross-attention
→ 2 token-local refinement blocks
```

allocates 3/7, 2/7, and 2/7 of the repeated block parameters, but token stages
still dominate execution. The two final local blocks are specifically reserved
for token representations to act on global information rather than merely
receiving one terminal projection.

## Training and attribution rules

1. Train the new encoder from scratch; its RoPE blocks, hierarchical names,
   predictor, and projections changed.
2. Train the world stage from that exact encoder. Reject a run on a gradient
   spike and preserve the best atomic world/encoder pair.
3. Before encoder/world training, the RAG ceiling must pass independently on
   seen and held-out tasks.
4. Keep the world frozen by default during latent-language alignment and
   generative bridge training. Unfreezing is an explicit practical ablation,
   not the default claim.
5. Qualify checkpoints with compile-and-harness autoregressive generation,
   then run matched, wrong, shuffled, swapped, and zero controls. The final
   pipeline fails rather than reports success if held-out matched pass rate or
   causal advantage misses its floor.
6. Report held-out suite pass and its fraction of the RAG ceiling. A high seen
   score alone is not knowledge transfer.

## Research basis

- Hierarchical latent bottlenecks and iterative query refinement:
  [Perceiver](https://arxiv.org/abs/2103.03206),
  [Perceiver IO](https://arxiv.org/abs/2107.14795).
- Frozen language model plus learned query/interface pretraining:
  [BLIP-2](https://arxiv.org/abs/2301.12597).
- Gated cross-attention into a frozen language model:
  [Flamingo](https://arxiv.org/abs/2204.14198).
- Masked prediction from contextualized latent targets:
  [data2vec](https://arxiv.org/abs/2202.03555) and
  [LeJEPA](https://arxiv.org/abs/2511.08544).
- World-state prediction and SIGReg:
  [LeWorldModel](https://arxiv.org/abs/2603.19312).
- Parameter-efficient gated feed-forward blocks:
  [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).

These papers motivate components; they do not by themselves establish that
their combination is state of the art. The held-out causal experiment is the
evidence required for that claim.
