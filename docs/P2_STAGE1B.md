# P2 Stage 1b — ARC-AGI-3 aligned curriculum (v9)

## Motivation

v8 (`p2-output-v8/`, 20 480 steps) was stable but misaligned with real ARC-AGI-3 in
three ways seen in local eval (`eval_report_64ep_v3.json`):

| Metric | v8 | Issue |
|--------|-----|-------|
| one-step latent MSE | 0.057 | Worse than v2 (0.025); ArcPad render under-trained |
| Q accuracy | 99% (`saturated: true`) | Threshold 0.25 trivial vs ~0.06 MSE |
| q_oracle_rank @ k=4 | 24% | PTRM cannot rank trajectories — planner-critical |
| events | 93.7% | 763 shared errors across v2/v7/v8 (event head plateau) |
| 8-step rollout MSE | 1.02 | v8 win; keep rollout ramp |

Real ARC-AGI-3 (technical report + Tycho/OPIUM papers) adds constraints v8 never
trained on:

1. **Early goal ignorance** — first interactions are exploratory; objectives are inferred
   from HUD/mechanics, not supplied as 19-dim vectors.
2. **Action economy** — RHAE squares `(human/ai_actions)`; retries and RESET accumulate.
3. **Render domain** — official frames are 64×64 padded grids; synthetic training mixed
   BlockScaleHud and ArcPad but plan/falsification paths used Hud-only.
4. **Hypothesis testing** — safe probes that discriminate among competing candidates,
   then sequential pursuit after evidence (P1 mechanism → Stage 3 planner).

Stage 1b keeps Stage 1 boundaries (`docs/ARC_AGI_3_PLAN.md`): dynamics stay goal-free;
beliefs update only from real transitions at inference; no ARC public fitting.

## Paper-informed design choices

| Source | Takeaway for v9 |
|--------|-----------------|
| [ARC-AGI-3 technical report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) | Interactive skill acquisition under scored actions; levels share mechanics |
| [Tycho (2607.28287)](https://arxiv.org/html/2607.28287v1) | Separate decision vs animation frames; RESET costs one scored action; executable models + verification |
| [LeJEPA / SIGReg (P2 baseline)](https://github.com/) | Keep SIGReg on encoder; representation before auxiliary heads |
| P1 falsification (`docs/RESULTS_P1.md`) | Safe multi-goal probes beat broad progress on hard ambiguity |
| v8 eval | Tighten Q threshold; train PTRM ranking on falsification batches |

Deferred (Stage 3+, not v9): closed-loop planner, hypothesis generation from pixels,
official online RHAE.

## v9 curriculum (6 lessons × 4096 steps = 24 576)

| Lesson | Curriculum kind | Purpose |
|--------|-----------------|---------|
| `dynamics` | `random_one_step` | Goal-free local transitions (unchanged) |
| `exploration` | `exploration` | 8–12 random legal steps, `goal_features=0`, event labels masked |
| `sequential` | `sequential` | Multi-step plan fragments + rollout ramp 2→8 |
| `q_calibration` | `sequential` | Warm event/Q heads on plan fragments |
| `falsification` | `hypothesis_probe` | P1C: exploration prefix + reset + multi-candidate probe |
| `retarget` | `p1c_hard_retarget` | False leads + retarget (unchanged) |

### Loss schedule changes vs v8

| Lesson | next_latent | rollout | event | Q | PTRM rank |
|--------|-------------|---------|-------|---|-----------|
| exploration | yes | no | **masked** | no | no |
| falsification | yes | no | ramp | ramp | **yes (k=4)** |

Other training changes:

- **`q_mse_threshold`**: `0.25 → 0.05` (matches ~v8 one-step MSE scale; fixes Q saturation)
- **Render mix**: `sequential`, `p1c_falsification`, `plan_fragments` use `render_style_for_episode` (50% ArcPad)
- **Default lessons** updated in CLI/train config

## Frozen v9 run config

```bash
cargo run --release --features cudnn -- p2-train \
  --device cuda --hidden-dim 128 --action-dim 32 \
  --physical-batch 1024 --grad-accum 1 --steps-per-lesson 4096 \
  --q-mse-threshold 0.05 \
  --checkpoint-every-steps 100 --output-dir p2-output-v9
```

Eval (after train):

```bash
cargo run --release --features cudnn -- p2-eval \
  --checkpoint p2-output-v9/model.safetensors \
  --train-config p2-output-v9/config.json \
  --device cuda --synthetic-episodes 64 --physical-batch 64 \
  --ptrm-k 1,2,4,8 --output p2-output-v9/eval_report_64ep_v3.json
```

## Success gates (pre-registered)

Improve vs v8 on held-out synthetic (64 ep):

1. `q.saturated == false` OR `q.balanced_accuracy > 0.55`
2. `q_oracle_rank_accuracy` @ k=4 **≥ 0.35**
3. one-step MSE **≤ 0.045** (ArcPad mix)
4. events accuracy **≥ 0.94** (no regression)
5. 8-step rollout MSE **≤ 1.5** (no collapse)

Stage 3 planner remains blocked until gates 1–3 pass on two seeds.

## Baseline reference (v8, do not overwrite)

```
one_step=0.057  rollout_4=0.43  rollout_8=1.02
events=0.937  q_saturated=true  q_oracle_rank@4=0.24
```
