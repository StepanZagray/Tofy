# Learned spatial pooling fails the fresh routing screen

At source `06a8d76fada203c5b1a11a45782ed777df27ff4c`, the fixed policy-only learned spatial readout scores **51/256 (19.92%)** on initial features and **68/256 (26.56%)** on C7-final features. Best constant is **77/256 (30.08%)**. Every learned-routing success gate fails; geometry and both shuffled-label controls pass. This is completed negative evidence for this recipe, not an ARC evaluation or a proof that role information is absent. [Frozen analysis](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/completed-analysis.json), `fresh_metrics`, `primary_decision`.

## Population and recipe

Six heads fit the same 512 C8 training queries, with raw frozen four-loop features and no core updates. Each receives 1,000 full-batch F32 AdamW updates, learning rate0.003, no weight decay, global clip1, deterministic seed0 and shared initial bytes within head family. Physical/effective batch512, accumulation1 is the entire fitting population. Two learned spatial queries pool64 current tokens into a linear action head (1,284 parameters); the CLS comparator is128→10 SiLU→4 (1,334 parameters). Both frozen cores also receive a spatial shuffled-label control using the same PCG64seed1915 query permutation. This changes supervised readout fitting; it does not isolate a cause of the earlier end-to-end failure. [Registration](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/registration.md), [sealed fitting inputs](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/fit-cache-seal.json).

All six final heads were sealed before generating the256 evaluation queries: known controls, distance one, factual support, seed20260916 and episode base0x524541444f5554. The new queries overlap none of5,432 retained queries, contain all four labels [77,56,53,70], and share exactly the same feature/input identities across arms. No public-level training or pretrained model is involved. [Head seal](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/final-heads.manifest.json), [generation authorization](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/fresh-audit-authorization.json), [audit](/home/stepan/Projects/code/.tofy-runs/looped-learned-readout-20260909T124221-IST/fresh-audit/report.json).

## All results

| Frozen core / readout | Fit correct /512 | Fresh correct /256 | Fresh CE |
|---|---:|---:|---:|
| initial spatial | 158 | 51 | 1.456557 |
| initial cls | 163 | 49 | 1.460618 |
| initial spatial null | 129 true / 170 shuffled | 67 | 1.397500 |
| final spatial | 187 | 68 | 1.385302 |
| final cls | 148 | 54 | 1.418346 |
| final spatial null | 121 true / 161 shuffled | 64 | 1.398162 |

Native policy remains constant: initial action1 scores56/256; final action0 scores77/256, exactly best constant. Geometry independently predicts256/256 labels and targets. Null fresh scores67/256 and64/256 pass the registered coarse ≤50% control. A single permutation does not estimate a null distribution. [Complete measurements](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/completed-analysis.json), `native_fresh`, `geometry`, `fits`.

Initial spatial accuracy interval is[15.23%,25.00%]; final[21.09%,32.03%]. Their constant advantages are−10.16pp [−17.97,−4.30] and−3.52pp [−11.72,+2.73]. Spatial-minus-CLS gains are+0.78pp [−1.17,+2.73] initially and+5.47pp [−1.17,+12.11] finally. None meets the ≥90% absolute gate or registered lower-bound advantages of25pp over constant and10pp over CLS. [Analysis](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/completed-analysis.json), `fresh_metrics`, `spatial_minus_cls`.

Preserve the positive contrast: **final-minus-initial spatial accuracy improves6.64pp [0.39,12.89]**, with CE change−0.07126 [−0.11753,−0.02632]. CLS accuracy improves1.95pp [−3.52,7.42]. These pointwise empirical paired-query intervals use10,000 PCG64seed1916 resamples and reselect best constant in each draw. One model/head initialization and one synthetic distribution do not establish robust training benefit or a unique architectural mechanism. [Analysis](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/completed-analysis.json), `final_minus_initial_true_heads`, `bootstrap`.

## Verification and resources

Eight new-head and29 existing example tests, two profile tests,26 independent analyzer fixtures and29 launch-guard fixtures pass. Strict Clippy has zero warnings after reviewed mechanical cleanup of pre-existing legacy warnings. Exact CUDA binaries were qualified with two-update spatial/CLS smokes and byte-identical legacy outputs. All twelve full fit/evaluation heads independently reconstruct from weights and cached features within2.19e-6, with zero prediction disagreements; counts, CE and bootstrap intervals agree. That independent reader did not revalidate geometry/profiler provenance; the frozen analyzer checks those separately. [Qualification](/home/stepan/Projects/code/.tofy-runs/looped-learned-readout-20260909T124221-IST/launch-qualification.json), [independent numeric evidence](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/findings/independent_numerics.json).

The six fits used37.291s total supervised model phase and5.562s supervisor-plus-binding finalization, within separate600s budgets. Full head fitting peaks at414MiB sampled GPU allocation, leaving7,737MiB reserve. The entire512-row physical batch preserves the registered optimizer schedule. Four qualification, six fitting, two extraction and six evaluation invocations produced18 healthy bound Candle Graph/Nsight captures; no operation/activation linkage, allocation-lifetime, instrumented physical-memory, device-event or complete automatic-correlation coverage is claimed. Host spans are not kernel durations; these profiled captures cannot establish production speed. [Budget](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/completed-analysis.json), `fit_budget_seconds`; [capture example](/home/stepan/Projects/code/.tofy-runs/looped-learned-readout-20260909T124221-IST/fit-final-spatial.bound/summary.json).

Source: `06a8d76fada203c5b1a11a45782ed777df27ff4c`; Candle Graph: `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a`.
Head binary SHA256 `fecbc1b91099ddc9b39ac9b6361ee8072c26563f6b710e06c29ec7b3adeef20e`; extractor `220daa93fecf43fa66ed681577fef81efb40bc058c1de39d52ed4a7afa87b728`.
Analysis SHA256 `7ca5e9130bb18e1f8ead24c6bbd978fef8a64855ccab4e7f2f4f63dbefd7681c`; outer campaign manifest `ea0fb1e773400363e52e3b6966600fde5d8e0ffe37a14547e49772d2186fe477`. The finalized tree has1,161 files and354,241,308 bytes; all286 recorded campaign/review PIDs are gone. The seal is point-in-time integrity, not immutable storage. [Cleanup](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/process-cleanup.json), [seal verification](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/seal-verification.json).

## Exact build and evaluation

```bash
CARGO_TARGET_DIR=/home/stepan/Coding/Personal/.tofy-build/v6-recipe-repair/target \
  cargo build --release --locked --offline \
  --features cudnn,profiling,serde_json/float_roundtrip \
  --example learned_readout_probe --example looped_agent_probe
```

The pinned supervisor derives every CLI, checks source/binary/cache identities, enforces deadlines/reserve and launches the required profilers. [Six exact fitting invocations](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/fit-invocations.json) and [six exact evaluation invocations](/home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/eval-invocations.json) retain config paths and digests. For example, the completed spatial-final fit used:

```bash
python3 -B /home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/supervise_stage.py \
  --invocation /home/stepan/Projects/code/.tofy-runs/looped-learned-readout-20260909T124221-IST/invocations/fit-final-spatial.json \
  --invocation-sha256 955c65366c2f9b1975fa44fb2b705320e5c9131ede2b3c4b50ece68a5b3c89b3
```

Run roots are never reusable. Frozen rescoring is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -B /home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/readout_analysis.py \
  --campaign /home/stepan/Projects/code/.tofy-runs/looped-learned-readout-20260909T124221-IST \
  --source 06a8d76fada203c5b1a11a45782ed777df27ff4c \
  --head-binary fecbc1b91099ddc9b39ac9b6361ee8072c26563f6b710e06c29ec7b3adeef20e \
  --extractor-binary 220daa93fecf43fa66ed681577fef81efb40bc058c1de39d52ed4a7afa87b728 \
  --output /home/stepan/Research/_runs/2026-09-09T113612Z-tofy-looped-learned-readout/REQUIRES-NEW-OUTPUT-NAME.json
```

## Decision and next falsifier

Reject scaling this fixed policy-only pooling recipe. Neither a failed small nonlinear head nor sharper attention proves information absence or optimizer optimality. C8's successful true-role selection can exploit coordinates supplied by the routing oracle even when object identity is not recoverable from features. Check that missing premise next: fixed F64 linear role queries trained with synthetic role labels, followed by the already frozen C8 affine policy. Inference receives features only; privileged fitting supervision and reuse of this C9 panel are explicitly selection-only. [C10 registration](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/registration.md).

The shared-depth looped core remains. Successor fidelity, useful value/planning, learned episode memory, active probing and new-core ARC integration remain unestablished or absent. No public score or architecture promotion follows from C9.
