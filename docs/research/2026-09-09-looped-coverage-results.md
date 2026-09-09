# Complete episode coverage does not rescue rule inference

September 9, 2026 IST. The registered single-seed comparison fails every policy
promotion gate. Fresh procedural episodes produce **25.10% familiar / 25.78%
held-out mapping accuracy**, versus **24.22% / 26.95%** for repeated episodes and
**25%** with cleared history. This is a completed negative screen at the named
budget and recipe, not proof that looped transformers or diverse data cannot work.
There is no ARC-AGI-3 score improvement: the new core still lacks learned episode
memory, active probing and an ARC adapter. No public ARC data or pretrained LLM
weights/controller/teacher were used.

## Registered comparison and integrity

[Registration and execution amendments](2026-09-08-looped-layout-coverage-screen.md).
Both from-scratch models have 992,393 parameters, width 128, four attention heads,
two shared transformer blocks, a 1/2/4 training-loop cycle and four evaluation loops.
Final-only RMS, input recall, AdamW lr 0.0003/wd 0.01/clip 1 and all four loss weights
remain matched. Each arm completes exactly 1,150 updates /73,600 examples, physical
33 plus tail 31, accumulation 2/effective 64, initialization seed 0/data 9173. Checkpoints
are retained every 25 updates; the final 1150 checkpoint is used regardless of peaks.

Fixed coverage repeats eight episode IDs:128 unique complete inputs and eight query
frames. Fresh coverage uses 4,600 episode IDs:73,600 unique complete inputs and 4,600
query frames. Both expose each of 16 training mappings 4,600 times in the same order.
The intervention changes complete episodes, including calibration order, omitted
action and positions, not query geometry alone. Initial weights and the first128
inputs are identical; the first two full training updates match numerically exactly.
Every actual training-stream row matches its sealed preflight audit byte for byte.

Both frozen readouts use 64 layouts/data 20260909, batch 1/four loops, zero updates,
all 16 familiar and eight held-out mappings:1,536 policy rows and 512 prediction action
tuples per checkpoint. Input/query/label populations match exactly between arms,
query hashes are absent from both training streams, and parent weight hashes remain
unchanged. Positive/negative controls verify oracle factual1/wrong-real0/follows1 and
exactly 25% for cleared fixed-action control. Calibration identifies the bijection
from three distinct visible moves plus elimination; raw spatial sums being invariant
does not prove the nonlinear model is blind.

## Outcomes

| Frozen policy population | Fixed episodes | Fresh episodes | Fresh minus fixed,95% CI |
|---|---:|---:|---:|
| New layouts, familiar mappings |248/1024=24.22%|257/1024=25.10%|+0.88pp [−3.22,+4.98]|
| New layouts, held-out mappings |138/512=26.95%|132/512=25.78%|−1.17pp [−4.49,+2.15]|
| Cleared history, either split |25%|25%|0pp|

Intervals use 10,000 paired whole-layout bootstrap samples, explicit NumPy PCG64
seed 1908 and linear percentile quantiles. They quantify layout variation at one
initialization seed; they do not capture seed uncertainty. The primary held-out
lower bound fails the required+10pp, both fresh accuracies fail 75%, and both fresh
factual-minus-cleared lower bounds fail+25pp. Those last differences are+0.10pp
[−0.88,+0.98] and+0.78pp [0.00,+1.76]. No checkpoint, metric, seed subset or depth was
selected after observing outcomes. No multi-seed confirmation or promotion is justified.

On the common128-example reference set, fixed ends 113/128=88.28%, CE 0.300638;
fresh ends 33/128=25.78%, CE 1.386924 (uniform four-action CE is ln 4≈1.386294).
For fresh coverage this set was seen once, so it is not the fresh arm's whole
training population. The fixed control demonstrates substantial finite fitting but
does not reproduce the earlier b22ce068 first-pass93.75% at 1150. That old run is not
an intervention comparator: its clipping reduction order was undefined. Intermediate
reference peaks are descriptive only and cannot replace the registered final result.

Wrong support changes fixed-policy decisions on 626/1024 and 312/512 rows, but fresh
on only 52/1024 and 22/512. Fresh maximum probability changes are below 0.000177 in
both splits; its decisions are weakly context-sensitive, not provably disconnected.
Fixed wrong-support accuracy under the real target is 277/1024 and 115/512; fresh is
253/1024 and 125/512. Follows-presented-rule accuracy equals factual accuracy by the
within-split permutation construction for every deterministic policy; subtracting
those quantities is invalid and was never used as a gate.

## Independent prediction and value boundaries

Each split has64 prediction queries/256 action tuples, half one-step and half
distance 2–10. Prediction geometry and value targets therefore extend beyond the
one-step-only training distribution.

| Metric | Fixed familiar / held-out | Fresh familiar / held-out |
|---|---:|---:|
| Exact successor frames |0/256 /0/256|32/256 /32/256|
| Copy-current baseline |70/256 /70/256|70/256 /70/256|
| Vacated-agent pixels correct |11904/11904 both|11904/11904 both|
| Destination pixels correct |6018/11904 /5322/11904|2048/11904 both|
| Internally inconsistent predicted patches |4812 /4614|0 /0|
| Reward true positives |7/32 /9/32|0/32 /0/32|
| Reward false positives |51/224 /46/224|0/224 /0/224|

Fresh coverage improves patch consistency and raw frame exactness, but remains below
copy and fails every non-reward successor. An **exploratory** post-result cross-tab
finds its 32 exact cases are exactly the 32 reward-positive cases in each split.
Both checkpoints already get all 4096/4096 changed pixels correct on those terminal
rows. The exact-frame gain therefore fixes errors in otherwise unchanged pixels, not
new terminal-destination placement. Fresh non-reward cases get 9856/19712 changed
pixels correct, exactly the vacated half, and 0/224 whole frames. This is consistent
with a goal-completion shortcut, but raw predicted states were not logged: it does
not prove that all actions produce identical predictions.
Correctly vacating the query agent is not evidence of decoding calibration transitions
or correctly selecting the alternative destination. Neither arm validates planning.

Mixed-horizon value MSE is 0.010726/0.011750 fixed and 0.012448/0.012448 fresh, versus
constant 0.9 at 0.009087. All training value targets are 1, because the source target is
DISCOUNT^(distance−1). This mixed-horizon result cannot establish an in-distribution
value-fit failure. The preregistered descriptive one-step rule-row check, using the
already-sigmoided logged values without another sigmoid, gives MSE 0.001212/0.000814
fixed and 0.00001352/0.00001352 fresh versus the perfect constant 1 baseline0. Fresh
thus fits the constant target, which supplies no evidence of useful long-horizon value.

## Reproducibility, capacity and execution evidence

Before the valid pair, an initial fixed arm at 273f2c6c was stopped at 645 updates
(last checkpoint 625) after matching initial weights and first losses but differing
clipping norms. A direct CPU regression on identical named gradients in 64 separately
randomized VarMaps reproduced changing norms/scales. Sorting floating parameters by
name in `clip_gradients_gpu_with_stats` fixes the unordered F32 reduction. Ten
optimizer tests and six runner tests pass. This repairs order dependence, not F32
precision or every possible cause of historical trajectory divergence.

Two independent six-update CUDA replicas at the repaired source match all 384 input
rows, initial reference predictions, every loss/norm/scale and final weights exactly;
clipping is exercised. This proves only that bounded replay, not 1150-update determinism.
Physical34 initially passed short checks but the long fixed arm exceeded the 512MiB
whole-GPU reserve at 7732/8151MiB; it was stopped at 626 updates (last625). Whole-GPU
telemetry does not localize the increase to model allocations. Physical35 failed CUDA
allocation. The registered capacity fallback requalified33 and restarted BOTH arms,
fresh then fixed; no interrupted or mixed-batch arm enters the quality comparison.
The supervisor now enforces the same reserve immediately rather than only at exit.

Completed training elapsed 1,624.24s fresh and 1,625.22s fixed, about54m9s combined;
the two interrupted arms additionally consumed about30m17s, plus qualification,
evaluation, compilation and analysis overhead. Valid full-arm sampled peaks were
7508/8151MiB fresh and 7444/8151MiB fixed, maximum 68°C/69°C. These are supervised
elapsed times and sampled whole-GPU usage, not production timing or allocator peaks.

All wired profiler planes are enabled: first evaluation and updates 2/100/1000,
semantic spans, labelled tensor statistics, host scalars/traces and manifest-checked
44-parameter/five-family gradients, NVTX and Nsight CUDA/cuBLAS/cuDNN/OS-runtime with
process-tree CPU sampling. Official Nsight reports are attached in separately
published, semantically verified bundles after producer exit. Raw application-label
checks pass. CG0.10.1 automatic NVTX correlation remains incomplete due literal domain
prefixes/library labels; no per-phase attribution or clock alignment is claimed.
Operation/activation linkage, allocator lifetimes and device-event intervals remain
unwired. All captures are `profiled_work`, so they cannot support production timing
verdicts. Producer and CLI retain explicit `serde_json/float_roundtrip`.

Opus 5 xHigh supplied independent code/localization advice. Its unsupported causal
interpretations were qualified by source checks; advice is not evidence of benefit.
Fable review was rejected twice for exhausted account quota, with no model usage.
Astra independently reviewed clipping, qualification and final numerical conclusions.

## Exact provenance and commands

Model source `713e672dfc8e7eb316b8cb622c4b2569f839d761`, clean and pushed at launch;
Candle Graph `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a`. Launch binary SHA-256
`68ddaf33f94e2e9b31c348912b8d3ce6c957a1fe1e8cdb93b99daaa8e48fd1ea`;
Candle Graph CLI `54a9dfd118c26b165efde74d02b9a03ec8db290d3dcb299a69cbdf1a9569ef8a`.
NVIDIA RTX5060 Laptop 8GiB, driver610.57.04; model/task bytes unchanged during the
coverage experiment. Initial weight SHA
`4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802`.

- fixed: final weights `83a88e881f5262a8bb47e7e742b9ef483b8a928b92666e4034f0c46d47f71804`; training manifest `a01b92971e6ee039f647f34bdc7a763f34d7683bef1677ac9fc75b5c481f45c2`; frozen manifest `8a00e16a63dc438fc29b713553229e7ea7abb34ffef134bee83b59112c822996`.
- fresh: final weights `81365192586b628177b05cfb600f485489703ad3fbd96047f5057b11e8aa77c0`; training manifest `245585d45b9b6e5c9951bb67e99ca7ed8627de1faf6f36181164df761acf798d`; frozen manifest `9f4fb4b41da6111dff3c35a99fb9df8342e716d9a2a80200c90fc1f392f04754`.

Build:

```bash
CARGO_TARGET_DIR=/home/stepan/Coding/Personal/.tofy-build/v6-recipe-repair/target \
TOFY_BUILD_COMMAND='CARGO_TARGET_DIR=/home/stepan/Coding/Personal/.tofy-build/v6-recipe-repair/target cargo build --release --locked --offline --features cudnn,profiling,serde_json/float_roundtrip --example looped_agent_probe' \
cargo build --release --locked --offline --features cudnn,profiling,serde_json/float_roundtrip --example looped_agent_probe
```

Actual supervised commands below use their recorded, sealed roots; never reuse them.
The wrapper records the complete Nsight command, environment, PIDs and binary hash.

```bash
python3 /home/stepan/Projects/code/.tofy-runs/looped-coverage-b33-20260909T0019-IST/supervise.py \
  --binary /home/stepan/Projects/code/.tofy-runs/looped-coverage-b33-20260909T0019-IST/looped_agent_probe \
  --name fresh-seed0 --mode coverage --coverage fresh --updates 1150 --batch 33 --seconds 2100
python3 /home/stepan/Projects/code/.tofy-runs/looped-coverage-b33-20260909T0019-IST/supervise.py \
  --binary /home/stepan/Projects/code/.tofy-runs/looped-coverage-b33-20260909T0019-IST/looped_agent_probe \
  --name fixed-seed0 --mode coverage --coverage fixed --updates 1150 --batch 33 --seconds 2100
python3 /home/stepan/Projects/code/.tofy-runs/looped-coverage-b33-20260909T0019-IST/supervise.py \
  --binary /home/stepan/Projects/code/.tofy-runs/looped-coverage-b33-20260909T0019-IST/looped_agent_probe \
  --name frozen-fixed --mode inspect --batch 1 --seconds 300 --eval-episodes 64 --data-seed 20260909 \
  --checkpoint /home/stepan/Projects/code/.tofy-runs/looped-coverage-b33-20260909T0019-IST/fixed-seed0/final.safetensors
python3 /home/stepan/Projects/code/.tofy-runs/looped-coverage-b33-20260909T0019-IST/supervise.py \
  --binary /home/stepan/Projects/code/.tofy-runs/looped-coverage-b33-20260909T0019-IST/looped_agent_probe \
  --name frozen-fresh --mode inspect --batch 1 --seconds 300 --eval-episodes 64 --data-seed 20260909 \
  --checkpoint /home/stepan/Projects/code/.tofy-runs/looped-coverage-b33-20260909T0019-IST/fresh-seed0/final.safetensors
```

[Registered analysis JSON](/home/stepan/Research/_runs/2026-09-08T221611Z-tofy-looped-episode-coverage/paired-analysis.json), [supplementary scope/cross-tab](/home/stepan/Research/_runs/2026-09-08T221611Z-tofy-looped-episode-coverage/supplementary-analysis.json), [research run](/home/stepan/Research/_runs/2026-09-08T221611Z-tofy-looped-episode-coverage/manifest.json). The analyzer's17 CPU control/integrity fixtures pass. The research run retains all exact commands, failed launches, receipts and externally stored campaign manifest digests.

## Decision and next falsifiable experiment

Reject **coverage alone at this seed/budget/recipe** as a useful rule-inference gain.
Retain the shared-depth architecture direction; do not increase loops, rewrite
normalization or reweight losses solely from these data. The finite fit is possible,
but fresh-data optimization and correct support-to-action binding remain unresolved.

Next preregister a bounded frozen successor diagnostic: hold each query fixed,
sweep complete observed control mappings, retain actual predicted pixels/probabilities,
and score genuinely different simulator outcomes separately for demonstrated versus
bijection-inferred actions. Include oracle/copy controls and a fresh confirmation
query population for the exploratory terminal-case pattern. No learned probe or
optimizer step: a fitted probe could perform the missing computation itself.
Probability sensitivity, correct alternative-state prediction and internal mechanism
localization must remain distinct. That diagnostic can falsify a rule-invariant
successor shortcut; it cannot by itself identify where inside the transformer a
failure originates. Any representation or loss intervention requires a separate
registered comparison, with per-loss gradient pressure if objectives change.

Final campaign:972 files /689224877 bytes, externally stored point-in-time manifest
`a1ea2d7b92c27aa12aa72a3087ca19160cc4d5739a07cbedf593bae93061327c`. All 129 recorded process IDs are gone; GPU compute-app query is empty.
