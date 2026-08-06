# v10 fixes (post-v9 rollout collapse)

## Root cause (v9)

Open-loop rollout loss compares predicted latents against `encode(next_frame)` step
by step. v9 introduced **per-step render style alternation** (BlockScaleHud vs ArcPad)
inside multi-step traces:

1. `sample_from_transition` keyed style off `actions_used`, so every retarget step
   flipped style.
2. `generate_plan_fragments` keyed style off step index (50% flip rate).
3. Full rollout weight on retarget (4096 steps, horizon up to 8) amplified drift.

The encoder must map two pixel layouts to the same latent dynamics; open-loop cannot
when targets alternate every step. Training rollout loss hit **4038** (sequential) and
**4994** (retarget) vs v8 **1666** / **0.104**; eval 8-step MSE exploded.

Q threshold 0.05 and exploration lesson were fine; render mixing in rollout paths was not.

## v10 changes

| Fix | Implementation |
|-----|----------------|
| Trace-consistent render | `render_style_for_trace(seed, ep, allow_arc_pad)` — one style per episode |
| Rollout paths Hud-only | `sequential`, `retarget`, `sample_from_transition` → `allow_arc_pad=false` |
| ArcPad transfer only | `exploration`, `p1c_falsification` → `allow_arc_pad=true` per episode |
| One-step dynamics mix | `random_one_step` keeps per-step `render_style_for_episode` (no rollout) |
| Revert falsification data | lesson → `p1c_falsification` (not `hypothesis_probe`) |
| Cap retarget rollout | weight × **0.25**, max horizon **4** |
| Keep v9 wins | exploration lesson, `q_mse_threshold=0.05`, PTRM rank k=4 on falsification |

## Train command (v10)

```bash
cargo run --release --features cudnn -- p2-train \
  --device cuda --hidden-dim 128 --action-dim 32 \
  --physical-batch 1024 --grad-accum 1 --steps-per-lesson 4096 \
  --q-mse-threshold 0.05 \
  --checkpoint-every-steps 100 --output-dir p2-output-v10
```

## Expected vs v9

- Sequential/retarget train rollout loss should return to v8 scale (<2000 / <1).
- Eval rollout-8 should be O(1–10), not 10¹¹.
- Q remains unsaturated; events should recover toward 93%+.
