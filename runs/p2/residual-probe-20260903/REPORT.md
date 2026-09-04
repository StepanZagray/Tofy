# ADR 0005 §5.3 — Residual-vs-reliability AUROC probe (2026-09-03)

Classification: **frozen-checkpoint rescoring, selection-only / exploratory** (single
checkpoint family, single random-policy seed). No training happened. Real ARC frames were
used only as probe inputs (ADR 0003/0005 invariant: public levels are evaluation only).

## Claim under test (preregistered before running)

On the latest local v5 checkpoint, the one-step composed-decode residual separates real
ARC-AGI-3 frames from in-distribution synthetic frames (AUROC clearly above 0.5, CI
excluding 0.5) while the learned reliability head does not (AUROC near 0.5). Falsifier:
if the residual AUROC CI includes 0.5, the representation carries no OOD signal and only
training-distribution breadth can help (research finding "Cheapest decisive local
experiment"). Secondary questions: does either score predict "composed decode wrong"
within each set; is the real-frame reliability ~1e-8 as previously observed live?

Comparator class: two scalar scores per transition (residual = composed-decode changed-pixel
count / mixture pixel cross-entropy vs actual next frame; head = `sigmoid(reliability_logit)`,
`sigmoid(q_logit)` recorded alongside). Uncertainty: 1,000-resample bootstrap 95% CIs.
Controls: shuffled-label negative control (expect ~0.5); positive control = synthetic
in-distribution rows decoded by the same path (expect high in-distribution exactness and
`residual == 0 <=> composed_exact` by construction).

## Checkpoint caveat (read first)

- The brief names `foundation-v2/model.safetensors` with sha256 `117f786b…`. Locally,
  `117f786b…` is **`foundation-v2/checkpoints/best/ema.safetensors`** (the EMA "best"
  weights, the file the 2026-08-27 live eval `arc3_live_report.json` actually used);
  `foundation-v2/model.safetensors` (final step 24576 export) is `e5457dd4…`. Both were
  probed; **the EMA-best file (`117f786b…`) is treated as primary** because it matches the
  brief's hash and the live-eval provenance. Conclusions are the same for both.
- This is foundation-v2 (2026-08-27, seed 2, 24,576 steps, 460,625 params, recipe
  `foundation_v2`, world_core_v4/v5 topology). It is **NOT** the later s8 step-20480 model,
  which is unreachable (pod stopped). The "~1e-8 reliability on real frames" observation
  that motivated §5.3 came from that later model; it is **not reproduced** here (see
  reliability distribution), so this probe cannot confirm or refute the shortcut hypothesis
  for the s8 model — only for foundation-v2.
- The checkpoint predates `operator_conditioning_proj` (added 2026-08-28, commit
  `36fe9e96`). The probe's loader leaves exactly those two tensors at their zero
  initialization; the projection is additive at the action seam, so this is numerically
  identical to the trained topology. Any other tensor mismatch still fails closed.
  (`legacy_operator_projection_zeroed: true` in both summaries.)
- Goal features only enter the event head (`heads()`: `event_in = cat(readout, goal_h)`);
  q/reliability heads and the latent do not see them. The `--conditioning eval` control arm
  (row goals + row operator token) therefore produced **bit-identical** residual, q and
  reliability on all 7,500 rows, including the 554 synthetic rows with a non-zero goal.
  "Live-policy conditioning" (zero goal + UNKNOWN operator) is thus not a confound here.

## Setup

- Code: branch `feat/v6-residual-probe` (worktree of `feat/adaptation-v6` @ `12ab24a2`),
  new CLI `tofy p2-residual-probe` (`src/p2/residual_probe.rs`), scripts
  `scripts/p2_residual_probe_record.py`, `scripts/p2_residual_probe_analyze.py`.
  Build: `cargo build --release --features cuda --bin tofy` (RTX 5060 Laptop 8 GB, CUDA
  13.3, sibling `candle_graph` @ `0c2fae7`, binary sha256 prefix `f56b09df4764bdc7`).
- Real frames: 25 public games downloaded via the toolkit (anonymous download path), driven
  **locally** (OFFLINE mode, no scorecard API play) with a seeded uniform-random policy over
  `available_actions \ {RESET}` (ACTION6 with x,y ~ U{0..63}), 150 actions per game, master
  seed 20260903 (per-game seed `20260903*1000 + index`), terminal states reset and
  continued. Toolkit JSONL recordings under `recordings/` (130 MB, not committed).
  Imported with the existing `arc3::import_recordings_dir` (settled frame layer; RESET rows
  skipped; `goal_features = zeros`, `noop = None`).
- Synthetic comparator: the offline evaluator's own held-out in-distribution population,
  `compose_mixed_stream_batch(MixedStreamConfig{batch_size: 3750, seed: iid_seed(3) +
  V5_HOLDOUT_SEED_DOMAIN, schedule: foundation_v2_stream_schedule}, progress=1.0,
  V5DataSplit::UnseenSeed7x7)` — identical construction to `eval::foundation_v2_v5_holdout_gates`
  lane 0 ("unseen_seed_7x7"). 3,750 rows: RandomOneStep 991, FactualBranches 1180,
  Exploration 790, SequentialFragments 592, HazardOneStep 197; **1,183 rows (31.5%) had the
  generator's goal dropout applied** (3,196 rows total had an all-zero goal at source).
  Rows were deterministically shuffled before chunking (physical batch 64) only so that no
  chunk is an all-factual partial branch group; per-row metrics are order-independent.
- Per row: encode current frame → `forward_from_latent_with_operator_conditioning` (eval
  recursion depth, no noise, no context) → `composed_gameplay_decode` (deployed copy-gate
  composition, 63x64 gameplay area; status/64th row excluded for both sources as in
  `eval::gameplay_pixels`) → `residual_changed_pixels = #(pred != actual next)`,
  `residual_pixel_ce = mean_pixels -log[(1-g)·1[cur==next] + g·softmax(logits)[next]]`
  (mixture form of the copy-gate composition), `decoder_pixel_ce` (decoder only),
  `reliability = sigmoid(reliability_logit)`, `q = sigmoid(q_logit)`, `actual_noop`
  (gameplay area unchanged), `composed_exact` (whole gameplay area), `composed_changed_exact`
  (trainer's `one_step_changed_exact` label: correct on the pixels that actually changed).

## Populations

| | rows | gameplay no-op | changed | composed full-exact | changed-exact (trainer label) |
|---|---|---|---|---|---|
| real (25 games x 150) | 3750 | 1340 (35.7%) | 2410 | 0 / 3750 (0.0%) | 28 / 2410 (1.2%) [ema]; 31 (1.3%) [final] |
| synthetic UnseenSeed7x7 | 3750 | 1123 (29.9%) | 2627 | 100 / 3750 (2.7%) | 0.403 [ema]; 0.407 [final] |

Recorder-side "frame changed" count was 3,146/3,750 because it compared every animation
layer and all 64 rows; the probe's no-op flag uses the imported settled layer and the 63-row
gameplay area, giving 2,410 changed. The generator's `noop` label agreed with the probe's
gameplay-area flag on 3750/3750 synthetic rows.

Trainer reference: the in-run gate on its own reserved population reported
`one_step_changed_exact = 0.534` (best 0.537). The probe's 0.40 on the evaluator's
unseen-seed population is the same order (different population, harder held-out seed,
and includes FactualBranches at 0.23), so the decode path is behaving as trained.
Synthetic full-frame exactness is only 2.7% because the composed decode makes a median of
23 spurious pixel edits per frame even on no-op rows (copy gate mean ~0.45); this is a
property of the checkpoint, not of the probe.

## Results — EMA-best checkpoint `117f786b…` (primary)

AUROC, positive class = real (or = wrong), 1,000 bootstrap resamples, 95% CI.

| Task | Score | AUROC [95% CI] | n (pos) |
|---|---|---|---|
| real vs synthetic (all rows) | residual pixel-CE | **0.905 [0.896, 0.914]** | 7500 (3750) |
| | residual changed pixels | **0.903 [0.893, 0.912]** | |
| | 1 − reliability | 0.634 [0.620, 0.647] | |
| | 1 − q | 0.634 [0.621, 0.647] | |
| real vs synthetic, changed rows only | residual pixel-CE | 0.866 [0.852, 0.878] | 5037 (2410) |
| | residual changed pixels | 0.863 [0.850, 0.876] | |
| | 1 − reliability | 0.494 [0.477, 0.511] | |
| real vs synthetic, no-op rows only (ideal residual = 0 for both) | residual pixel-CE | 0.998 [0.996, 1.000] | 2463 (1340) |
| | residual changed pixels | 0.998 [0.996, 1.000] | |
| | 1 − reliability | 0.895 [0.882, 0.908] | |
| predict changed-exact WRONG within real | residual pixel-CE | 0.646 [0.533, 0.757] | 2410 (2382) |
| | residual changed pixels | 0.602 [0.492, 0.713] | |
| | 1 − reliability | 0.518 [0.426, 0.608] | |
| predict changed-exact WRONG within synthetic | residual pixel-CE | 0.488 [0.466, 0.509] | 2627 (1568) |
| | residual changed pixels | 0.488 [0.466, 0.510] | |
| | 1 − reliability | **0.719 [0.700, 0.738]** | |
| predict full-frame WRONG within synthetic | residual changed pixels | 1.000 (tautological: residual==0 ⇔ exact) | 3750 (3650) |
| | 1 − reliability | 0.988 [0.982, 0.994] | |
| predict full-frame WRONG within real | any | undefined — all 3750 real rows are wrong | 3750 (3750) |
| **negative control** (shuffled labels) | residual pixel-CE | 0.502 [0.490, 0.515] | 7500 |
| | 1 − reliability | 0.498 [0.485, 0.511] | |

Final-export checkpoint `e5457dd4…` (`analysis.json`): real-vs-synthetic residual
0.905 [0.896, 0.914] / 0.903; reliability 0.615 [0.601, 0.629]; changed-only reliability
0.471 [0.455, 0.488]; no-op-only residual 0.998, reliability 0.888 [0.875, 0.902];
within-synthetic changed-wrong: reliability 0.720 [0.701, 0.740], residual 0.49; negative
control 0.501 / 0.497. Same picture.

### Reliability distribution on real frames (is it ~1e-8?)

**No.** EMA-best: real reliability min 0.137, p05 0.213, p25 0.292, median **0.511**,
p75 0.703, p95 0.801, max 0.850; fraction below 1e-3: 0.0; fraction above 0.5: 52%.
Synthetic in-distribution: median 0.661, p25 0.562, p95 0.792, above 0.5: 77%.
Final export: real median 0.529, synthetic median 0.678. This matches the 2026-08-27 live
report on the same EMA weights (chosen-action `reliability_probability` p05–p95 0.24–0.80,
median 0.42, 3,577 decisions). The ~1e-8 collapse was observed on the later s8 model and is
**not a property of foundation-v2**.

Per game (EMA-best; `changed-exact` = trainer label on changed rows):

| game | changed rows | changed-exact | rel median | pixel-CE median | no-op rate |
|---|---|---|---|---|---|
| ar25 | 136 | 0.000 | 0.746 | 0.607 | 0.09 |
| bp35 | 101 | 0.000 | 0.704 | 0.604 | 0.33 |
| cd82 | 49 | 0.000 | 0.607 | 1.133 | 0.67 |
| cn04 | 132 | 0.038 | 0.661 | 0.757 | 0.12 |
| dc22 | 69 | 0.000 | 0.748 | 0.235 | 0.54 |
| ft09 | 15 | 0.000 | 0.250 | 0.128 | 0.90 |
| g50t | 78 | 0.000 | 0.562 | 0.458 | 0.48 |
| ka59 | 98 | 0.000 | 0.644 | 0.649 | 0.35 |
| lf52 | 150 | 0.000 | 0.628 | 0.323 | 0.00 |
| lp85 | 4 | 0.000 | 0.265 | 0.164 | 0.97 |
| ls20 | 150 | 0.000 | 0.787 | 0.415 | 0.00 |
| m0r0 | 115 | 0.104 | 0.670 | 0.690 | 0.23 |
| r11l | 148 | 0.000 | 0.255 | 0.448 | 0.01 |
| re86 | 150 | 0.000 | 0.421 | 0.751 | 0.00 |
| s5i5 | 4 | 0.000 | 0.260 | 0.153 | 0.97 |
| sb26 | 50 | 0.000 | 0.256 | 0.795 | 0.67 |
| sc25 | 108 | 0.093 | 0.445 | 0.457 | 0.28 |
| sk48 | 110 | 0.000 | 0.715 | 0.417 | 0.27 |
| sp80 | 103 | 0.010 | 0.653 | 0.583 | 0.31 |
| su15 | 1 | 0.000 | 0.511 | 0.698 | 0.99 |
| tn36 | 150 | 0.000 | 0.275 | 0.143 | 0.00 |
| tr87 | 150 | 0.000 | 0.798 | 0.393 | 0.00 |
| tu93 | 71 | 0.000 | 0.496 | 0.422 | 0.53 |
| vc33 | 150 | 0.000 | 0.297 | 0.080 | 0.00 |
| wa30 | 118 | 0.000 | 0.531 | 1.053 | 0.21 |

Note the head is anti-informative across games: ls20/tr87/ar25 get median reliability
0.75–0.80 while being wrong on 100% of changed transitions; the games where the head is low
(ft09, lp85, s5i5, vc33, tn36, r11l — all ACTION6-only "click" games) are the ones whose
frames the decode copies best (lowest pixel-CE), not the ones it predicts correctly.

## Decision per §5.3

**Residual separates; the reliability head does not (or only weakly and inconsistently).**
Residual AUROC 0.905 [0.896, 0.914] overall, 0.866 on changed rows, 0.998 on no-op rows;
reliability 0.634 [0.620, 0.647] overall, chance (0.494) on changed rows, 0.895 on no-op rows.
Negative control ≈ 0.50 in both. Under §5.3 this supports **Phase A trust using
residual-derived calibration until a v6 checkpoint exists**, with these bounds on what that
means:

1. **What residual detects is distribution shift, not per-transition correctness.** Within
   the real set the residual predicts changed-exact wrongness only at 0.646 [0.533, 0.757]
   (barely 28 positives-complement), and within the synthetic set it is at chance
   (0.488) while the reliability head reaches 0.719 [0.700, 0.738]. A residual-derived trust
   signal should therefore be used as an **"is this frame in-distribution at all"** gate
   (where it is decisive), not as a substitute for the head's in-distribution ranking.
2. **Content-density confound.** Real frames are dense 64x64 scenes; synthetic content is
   7x7 in a padded frame, so more pixels can be wrong on real frames. The no-op stratum
   (ideal residual is exactly zero for both sources) controls for this and still separates at
   0.998 with a median of 305 spurious edits on real no-ops vs 23 on synthetic; so the
   separation is not only density. It remains a shift detector between *these two*
   populations, not a validated OOD detector for unseen synthetic families.
3. **The real set is almost entirely "wrong".** 0/3750 full-frame exact, 28/2410
   changed-exact (1.2%). Any calibration fitted on real frames has essentially one class;
   the §5.3 remedy can only down-weight trust on real frames wholesale, which the residual
   does (median residual 590 px vs 23 px).
4. **Not the s8 model.** The motivating ~1e-8 observation is not reproduced on foundation-v2
   (real median 0.51). The shortcut hypothesis for s8 remains untested; if s8 becomes
   reachable, rerun with `--checkpoint <s8 ema/model> --train-config <its config.json>`.
5. Single random-policy seed, single checkpoint family, random-policy transitions (heavy
   on no-ops for click games); selection-only. A promotion decision on the calibration
   itself needs a fresh recording seed and a held-out game split.

## Commands (exact)

```
# worktree
git -C /home/stepan/Coding/Personal/.tofy-build/Tofy worktree add \
  /home/stepan/Coding/Personal/.tofy-build/wt-probe -b feat/v6-residual-probe feat/adaptation-v6
cd /home/stepan/Coding/Personal/.tofy-build/wt-probe
cargo build --release --features cuda --bin tofy

# 1. real frames (evaluation only; key read from ../Tofy/.env into ARC_API_KEY, never printed)
/home/stepan/Coding/Personal/Tofy/.venv/bin/python scripts/p2_residual_probe_record.py \
  --out runs/p2/residual-probe-20260903 --actions 150 --seed 20260903

# 2+3. probe (synthetic comparator generated inside; size defaults to the real row count)
CK=/home/stepan/Coding/Personal/Tofy/runs/p2/_pod_handoffs/6zp5oip7tvokfl-20260827-foundation-v2/foundation-v2
R=runs/p2/residual-probe-20260903
./target/release/tofy p2-residual-probe --checkpoint $CK/checkpoints/best/ema.safetensors \
  --train-config $CK/checkpoints/best/config.json --arc-recordings-dir $R/recordings \
  --device cuda --physical-batch 64 --conditioning live \
  --output-jsonl $R/probe_rows_ema.jsonl --output-summary $R/probe_summary_ema.json
./target/release/tofy p2-residual-probe --checkpoint $CK/model.safetensors \
  --train-config $CK/config.json --arc-recordings-dir $R/recordings --device cuda \
  --physical-batch 64 --output-jsonl $R/probe_rows.jsonl --output-summary $R/probe_summary.json
./target/release/tofy p2-residual-probe ... --conditioning eval \
  --output-jsonl $R/probe_rows_evalcond.jsonl --output-summary $R/probe_summary_evalcond.json

# 4. analysis
/home/stepan/Coding/Personal/Tofy/.venv/bin/python scripts/p2_residual_probe_analyze.py \
  --rows $R/probe_rows_ema.jsonl --out $R/analysis_ema.json
/home/stepan/Coding/Personal/Tofy/.venv/bin/python scripts/p2_residual_probe_analyze.py \
  --rows $R/probe_rows.jsonl --out $R/analysis.json
```

## Artifacts

Committed (small): `REPORT.md`, `recordings_summary.json`, `probe_summary_ema.json`,
`probe_summary.json`, `probe_summary_evalcond.json`, `analysis_ema.json`, `analysis.json`.
Local only (gitignored `runs/`): `recordings/` (toolkit JSONL, 130 MB),
`environment_files/`, `probe_rows*.jsonl` (7,500 rows each), `record.*.log`.

## Correction (2026-09-04)

The decision wording above overstates the outcome. Against the preregistered rule
(residual AUROC >= 0.80 AND reliability AUROC <= 0.60) the reliability condition
was missed (0.634, CI excludes 0.60), so the switch was NOT triggered. Independent
audit further noted that using the 25 public games to decide Phase A trust would
violate the project's held-out policy; this report is evaluation-only evidence
and no policy or threshold is selected from it.
