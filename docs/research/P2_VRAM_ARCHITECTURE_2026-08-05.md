# P2 VRAM: why a 430K-param model fills an 8GB GPU

Status: investigation only (2026-08-05). Grounded in current `main` sources and
the failing readiness run (`p2-output-readiness`, batch 256, spatial SIGReg,
randomized depth 1..=8).

## Executive summary

Parameter count (~430K → **~1.7 MiB** weights) is **not** the VRAM driver.
Training peak is dominated by **activation storage** proportional to:

```
physical_batch × (pixel_one_hot + 2×encode + recursion_graph + sigreg_inflation)
```

With readiness defaults (`physical_batch=256`, `--sigreg-spatial`,
`--randomize-depth`, `--outer-steps 8`, `--inner-steps 2`, `--warm-start-y`,
`--residual-y-update`), a **single optimizer step** can legitimately allocate
**1–3 GiB** of activations; with cuDNN workspaces, **Adam duplicate buffers**,
and CUDA caching-allocator retention, **6–7 GiB sustained** on an 8 GiB card is
expected—not a mystery leak from “model too large.”

Several implementation choices **amplify** memory far beyond what the parameter
count suggests. The largest are architectural, not hyperparameter tuning.

## Observed symptoms (this session)

| Config | Result |
|--------|--------|
| Fresh, batch 64, short run | OK |
| Fresh, batch 128, 200 steps | OK (~4.5 min) |
| Fresh, batch 256, 20–50 steps | CUDA OOM after ~10–60 s |
| Resume batch 256 from step 2500 | OOM or NaN; stuck eval processes also consumed ~500 MiB |
| Readiness script + `--sigreg-spatial` + depth 1..=8 | OOM at batch 256 on RTX 5060 8GB |

Sources: local runs 2026-08-05; logs under `p2-output-readiness/repair-logs/`.

## Parameter vs activation budget

### Weights (negligible)

| Item | Size (430K params, F32) |
|------|-------------------------|
| Model weights | ~1.7 MiB |
| Adam m + v | ~3.4 MiB |
| **Total trainable** | **~5 MiB** |

### Activations (dominant) — batch B=256

| Tensor / path | Shape | F32 size |
|---------------|-------|----------|
| `frames` one-hot | B×16×64×64 | **67 MiB** |
| `next_frames` one-hot | B×16×64×64 | **67 MiB** |
| Latent `cur_z` / `next_z` | B×128×8×8 | 8 MiB each |
| Dynamics `x`, `y`, `z` | B×128×8×8 | 8 MiB each (many alive in graph) |
| SIGReg stack (spatial) | **(B·H·W·2)×128** = **32768×128** | **16 MiB** (+ matmul/backward temps) |

Encoder convs run at **8×8** after the patch stride (`GridEncoder` in
`src/p2/model.rs`); the **pixel one-hot** and **double encode** dominate input-side
memory, not mid-encoder 64×64 feature maps.

## Root causes (ranked by impact)

### 1. Full one-hot pixel tensors at 64×64 (≈16× overshoot)

`frames_to_one_hot` (`src/p2/train.rs`) builds categorical indices then:

```rust
indices.broadcast_eq(&channels)?.to_dtype(DType::F32)  // B×16×64×64 F32
```

For ARC, each cell is **one of 16** colors. Storing **16 float planes** per pixel
is ~16× larger than `B×1×64×64` integer indices + a small embedding lookup.

Both `frames` and `next_frames` live on GPU for the whole forward/backward.
At B=256 this alone is **~134 MiB** before autograd saves encoder intermediates.

**Fix direction:** index tensor `B×64×64` (u8/i32) + `nn::Embedding(16, …)` or
gather-based encoder first layer; keeps semantics, drops input activation ~16×.

### 2. `--sigreg-spatial` inflates SIGReg batch by H×W (×64 here)

`stack_latents_for_sigreg` (`src/p2/train.rs`):

```rust
// spatial: reshape to (B*H*W, C), cat cur+next → (2*B*H*W, C)
```

With B=256, H=W=8: **effective SIGReg batch = 32 768**, dim=128.

`sigreg_epps_pulley_seeded` (`src/p2/sigreg.rs`) then runs matmul
`(batch*time, dim) @ (dim, num_slices)` and cos/sin over knot grid—all
**connected to `cur_z` and `next_z` in autograd**.

This is intentional per `docs/P2_V17.md` / fable5 note 05 (per-cell statistic
instead of flattened B×4096). It trades **statistical resolution for VRAM**:
SIGReg cost scales **linearly in B×H×W**, not in parameter count.

**Fix directions (smallest first):**

- Cap SIGReg rows (random subsample of grid cells per step).
- Pool spatially before SIGReg (e.g. 4×4 → 2×2) for the statistic only.
- Run SIGReg on `B×C` pooled latents during dynamics; enable spatial only in
  later lessons.

### 3. Deep supervision over **every** outer recursion step

`leworld_loss` (`src/p2/train.rs`):

```rust
for step in &out.steps {
    let mse = step.y.sub(&next_z)?.sqr()?.mean_all()?;
    // accumulates into next_latent loss
}
```

`run_recursion` (`src/p2/model.rs`) **clones and stores** each outer `y` in
`steps: Vec<StepOutput>`:

```rust
steps.push(StepOutput { y: y.clone(), ... });
```

With `--randomize-depth` and `--outer-steps 8`, a single forward can unroll up to
**8 outer × (inner + 1) block forwards** (each `GridResidualBlock` = 2× Conv2d
3×3 on B×128×8×8). Autograd must retain **all** outer intermediates because
each contributes to `next_latent`.

Expected sampled depth ≈ 4.5 outer × 1.5 inner → ~20+ conv blocks **in one
graph**. This matches TRM-style “supervise all depths” but memory scales with
**max outer_steps**, not with hidden_dim.

**Fix directions:**

- Train with fixed outer=2 for memory profiling; add depth randomization only
  after batch fits.
- Last-step-only loss option (keep depth randomization for forward, detach
  earlier `y` from loss).
- Gradient checkpointing inside `deep_step` / `run_recursion` (recompute on
  backward).

### 4. Double encoder pass every step

`leworld_loss` always:

```rust
let cur_z = model.encode_state(&batch.frames)?;
let next_z = model.encode_state(&batch.next_frames)?;
```

With `--warm-start-y`, `forward_from_encoded_state` avoids a **third** encode
(uses `add_action(cur_state, …)`). Without warm-start, `encode_x` would encode
again—**triple** encode.

Two full encoder backward paths per microbatch is fixed cost **~2×** encoder
activation memory. Sharing a single encode pass for `cur` when the transition
uses `next` of the previous sample would halve this (requires batch layout
change).

### 5. `CheckpointAdamW` duplicates optimizer state on GPU

`CheckpointAdamW::new` (`src/p2/train.rs`):

- Clones **every model `Var`** into `optimizer.vars[].var` (second param copy).
- Allocates `first_moment` / `second_moment` on each `AdamVariable`.
- **Also** inserts duplicate moments into `moments: VarMap` for checkpoint I/O.

Roughly **~3× parameter-sized GPU memory** for optimizer bookkeeping (~5 MiB
for this model—small absolute, but wrong pattern at scale).

**Resume bug:** `load()` fills `moments` VarMap only; `step()` reads
`AdamVariable.first_moment` / `second_moment`, which are **not** reloaded from
checkpoint. Moments on disk may not match tensors used in training after resume.

**Fix direction:** single source of truth—optimizer steps directly on varmap vars;
moments only in one buffer set; load/sync into the tensors `step()` uses.

### 6. `--bf16-conv` non-functional (cannot halve conv activations)

`GridEncoder::forward(..., bf16: true)` casts **activations** to BF16 but conv
**weights stay F32** → `dtype mismatch in conv2d` at runtime. The flag cannot
currently reduce conv workspace.

**Fix direction:** cast weights to BF16 for conv paths or use mixed-precision
wrapper; keep norms/losses in F32.

### 7. Secondary contributors

| Issue | Where | Effect |
|-------|-------|--------|
| `grad_accum>1` | `train.rs` microbatch loop | Peak ≈ one microbatch graph, but longer live allocator pressure |
| PTRM ranking loss | `ptrm_ranking_loss` | K full recursions when lesson enables it (not dynamics) |
| Open-loop rollout | `open_loop_latent_loss` | Extra encode + forward per horizon step (sequential lesson) |
| `y.clone()` every outer step | `run_recursion` | Keeps dead tensors in graph until backward |
| CUDA caching allocator | driver | Freed tensors don’t return to OS; repeated OOM retries look like “leak” |
| Stuck `p2-eval` | ops | Competes for VRAM (observed 506 MiB) |

## Why batch 256 is near the 8GB cliff

Rough peak model (one microbatch, worst-case depth 8/2):

| Component | Estimate |
|-----------|----------|
| One-hot inputs + encoder (×2) | 200–400 MiB |
| Recursion graph (8 outer, 2 inner) | 400–1200 MiB |
| SIGReg spatial (32k rows) + backward | 100–400 MiB |
| cuDNN workspaces | 200–800 MiB |
| Optimizer + allocator overhead | 200–500 MiB |
| **Total** | **~1.5–3.5 GiB peak/step** |

Sustained **6–7 GiB** reported by `nvidia-smi` matches allocator retention +
worst-case depth draws + grad_accum=2–4, not a parameter explosion.

Readiness **effective batch** 512 (=256×2 or 128×4) does **not** multiply peak
memory—only `physical_batch` sets activation batch. Smaller physical batch is
the correct immediate mitigation.

## Recommended experiment order

1. **Measure** — log peak `cudaMemGetInfo` per phase (encode / recurse / SIGReg /
   backward) once; confirms table above on this GPU.
2. **Operational** — `physical_batch=128`, `grad_accum=4` (same effective 512);
   record in `docs/RESULTS_P2.md`.
3. **Quick code wins**
   - Fix `CheckpointAdamW` dedupe + resume moment sync.
   - SIGReg spatial subsample cap (e.g. 4096 rows).
   - Fix `--bf16-conv` weight casting.
4. **Architectural**
   - Integer pixel indices + embedding (largest input win).
   - Optional last-step-only recursion loss or checkpointing.
5. **Re-profile** largest stable physical batch after each change.

## Relation to prior research

- `docs/research/fable5/05-recursion-sigreg-q-stability.md` — depth randomization
  + spatial SIGReg are **training-stability** tools; note 05 already warns v15
  spatial latents exploded under weak SIGReg. This doc adds: the **same flags
  multiply VRAM** by design.
- `docs/P2_V17.md` — explicitly says re-measure batch at `--outer-steps 8`
  (~2× step cost vs depth 2). VRAM follows the same curve.

## Conclusion

The model is “small” in **parameters** but “large” in **activated tensors**:

- 64×64×16 one-hot pixels,
- 32k-row spatial SIGReg,
- up to 8 supervised outer recursions per step,

all at batch 256. None of this requires a bug to explain 6–7 GiB on an 8 GiB
GPU; several **implementation choices are suboptimal** (one-hot frames, duplicate
optimizer tensors, broken bf16, possible resume moment desync) and should be
fixed—but the dominant lever is **activation architecture**, not hidden_dim or
parameter count.
