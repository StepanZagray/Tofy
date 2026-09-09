# Frozen role selectors supply a finite spatial-readout witness

C10 learns shared linear agent/goal selectors from **privileged synthetic role labels during fitting**, then routes raw frozen features without true-role indices or a color parser at inference. Both the initial and C7-final cores score **256/256** joint role localization, hard-policy actions and soft-policy actions. These are **selection-only** results on C9's already inspected panel. Initial success predates C7 training; this is not a policy-only learning result or an ARC score. [Owning report](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/evidence-01/report.json), `fit`, `evaluation`, `decision`.

## Population, fitting and readout

The four fixed F64 ridge fits use exactly the first 512 C8 queries: seed 20260915, episode IDs `0x46454154555245 + i`, factual known controls, distance one, and raw `[512,64,128]` post-RMS current tokens from four-loop extraction. Agent/goal labels come from colors 2/3 in these fitting sidecars only. Each fit uses 32,768 token rows, training-only feature means and population standard deviations clamped below at `1e-6`, unpenalized intercept and ridge coefficient `0.01`. Core forwards and optimizer updates are both zero. Both cores receive true targets and one shared query-level PCG64 seed-1917 permutation of role pairs; no settings are selected from results. [Frozen registration](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/registration.md), “Fixed recipe”; [input bindings](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/evidence-01/report.json), `input_provenance`.

The C8 action readouts are frozen, including their means, scales, coefficients and `label_mean` intercepts. They already received supervised action fitting in C8. All four selectors, their fixed scales and both action readouts were externally sealed before any evaluation feature or role-sidecar read. Evaluation reuses C9's 256 queries, seed 20260916, episode IDs `0x524541444f5554 + i`; it is not fresh confirmation for C10. Fit action counts are `[119,136,133,124]`, evaluation counts `[77,56,53,70]`, with best constant **77/256 (30.078125%)**. [Fit seal](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/evidence-01.fit-seal.json); [chronology and populations](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/evidence-01/report.json), `chronology`, `populations`.

## Every evaluation arm

| Frozen core / fitting targets | Agent /256 | Goal /256 | Joint /256 | Hard policy /256 | Soft policy /256 |
|---|---:|---:|---:|---:|---:|
| Initial / true | 256 | 256 | 256 | 256 | 256 |
| Initial / permuted | 39 | 0 | 0 | 58 | 56 |
| C7 final / true | 256 | 256 | 256 | 256 | 256 |
| C7 final / permuted | 0 | 1 | 0 | 85 | 56 |

The **39/256 initial-null agent localizations** must not be summarized as no localization signal; its joint localization is zero. Both null cores pass the registered joint ≤10% and both hard/soft action ≤50% controls. The frozen oracle-role policies score 256/256 on both cores. Ordinary native policy scores 56/256 initially and 77/256 finally on this same panel. [Complete endpoints](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/evidence-01/report.json), `evaluation`, `populations`; [independent reconstruction](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/findings/independent_raw_review.json), `evaluation`, `controls`.

Both true selectors also localize all 512 fitting role pairs and score 512/512 hard and soft actions. Initial-null true/fitted joint localization is 0/512 and 2/512; final-null is 0/512 against both. Their hard/soft fitting policy counts are 144/136 initially and 133/136 finally. All fitting role-position distributions, competing-token margins, tie counts and fitted-versus-true endpoints remain in the [report](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/evidence-01/report.json), `fit`, `populations`.

| Core / targets | Minimum agent fitting margin | Minimum goal fitting margin | Fixed agent alpha | Fixed goal alpha | Finite fit witness |
|---|---:|---:|---:|---:|---|
| Initial / true | 0.9798337660 | 0.9610010801 | 18.32825619 | 18.68743403 | Yes |
| Initial / permuted | -0.0274287023 | -0.0286584755 | 1 | 1 | No |
| C7 final / true | 0.9118188134 | 0.9860829430 | 19.69540880 | 18.21210316 | Yes |
| C7 final / permuted | -0.0261528987 | -0.0281644374 | 1 | 1 | No |

For a positive minimum role margin `m`, the fixed rule is `alpha = log(63*(1-epsilon)/epsilon)/m`, with `epsilon=1e-6`. In real arithmetic, the target softmax mass is at least `1/(1+63*exp(-alpha*m)) = 1-epsilon` on the fitting panel. Both true fits pass the preregistered F64/F32 numerical mass checks. Ineligible null margins leave alpha at 1 and imply no mass guarantee. A failed ridge fit is not a proof of linear infeasibility. [Registration](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/registration.md), “Exact claim and decision” and final numerical clarification; [mass checks](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/evidence-01/report.json), `fit`.

The soft selector has an explicit real-arithmetic translation into C9's two-query head: `q = sqrt(128)*alpha*raw_role_coefficients`, with one row per role, and the folded C8 affine policy. Hard argmax selection is a separate diagnostic. F32 CPU checks do not validate a Rust/CUDA transplant. [Pinned implementation](/home/stepan/Projects/code/Tofy-role-witness/docs/research/artifacts/2026-09-09-looped-role-witness/role_selector_witness.py), `fit_selector`, `apply_selector`, `run`; [independent check](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/findings/independent_raw_review.json), `normal_equation_residuals`, `limits`.

## Registered gates and uncertainty

Both true cores meet the separate selection-only routing conjunction: joint localization ≥95%, hard and soft policy ≥90%, both lower 95% advantages over resampled best constant >25 percentage points, oracle policy ≥90%, and all null/numerical controls. Both hard/soft advantages are **69.921875 percentage points**, interval **[64.453125,73.046875]**. Every true accuracy interval is empirical `[1,1]`, which does not guarantee 100% population accuracy. [Report](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/evidence-01/report.json), `evaluation`, `decision`; [independent intervals](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/findings/independent_raw_review.json), `evaluation`.

Intervals use 10,000 paired whole-query PCG64 seed-1918 resamples, linear 95% percentiles and best-constant reselection in every draw. They are pointwise, not simultaneous. The finite fitting witness and evaluation support are reported separately. One fixed null permutation is a coarse control, not a null-distribution estimate.

## Provenance, verification and execution

- Source: `82ac8cb6ea3a06ae13d836d526c47994cd675d18`.
- Initial core SHA256: `4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802`.
- C7-final core SHA256: `a983626a888279cbb31651c6abcd1ebf4f808c8927a29a8349d03ffc35bd4e5a`.
- Registration SHA256: `411afa3b56fb71f5cb78494b9b91911cec19313080d98ff7474b52b129568dfb`.
- Frozen diagnostic SHA256: `a49151b2cd1294aa86fc72433755feb92e3d6b855796e8ee157e2402b51ac748`.
- Report SHA256: `9543b45637ce4cc363d1d1c0dfaf993590e82cb7207ffafbd5fad406a98a98bc`.
- Four-fit seal SHA256: `9276fc0ee6adcadf3143d57c7c8ebc483db5de075b548120add2bf641b951a5e`.
- Independent report SHA256: `dc5777b871cc9508af27e0a966e20c831799b25adf22c6d167315ba40a576281`.

The [source freeze](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/source-freeze.json) records 16 diagnostic and 10 supervisor fixtures. The independent reader imports no experiment implementation and performs no new fit or solve. It checks the four stored normal equations against the raw fitting prefix, every retained fitting/evaluation score, pool, role, margin and policy prediction, both precision paths, all intervals and controls. Maximum residual is **7.21e-16**, maximum compared F64 difference **2.14e-13**, and retained F32 selector arrays agree exactly. It rehashes the full C8/C9 parent seals and checks source/fit chronology. These are independent calculations on the same evidence, not an independent data replication. [Independent script](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/findings/independent_raw_review.py); [verification record](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/findings/independent_raw_review.json).

The diagnostic took **1.908296 seconds**, total reported wall time 1.911851 seconds; its supervisor completed in 2.181406 seconds. Python 3.14.7, NumPy 2.4.2 and one OpenBLAS thread are recorded. The registered bounds were 60 seconds diagnostic / 90 seconds external wall time. C10 performs four closed-form role fits on CPU, with no GPU work, model forwards or optimizer updates. NumPy/BLAS is not Candle Graph-instrumented. Existing frozen extractions retain their Candle Graph/Nsight bundles and known operation/activation, allocation/physical-memory, device-event and correlation gaps; no GPU or production-timing claim follows. [Runtime and limits](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/evidence-01/report.json), `runtime`, `elapsed_seconds`, `limits`; [accepted supervisor exit](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/evidence-01.exit.json).

The [external evidence seal](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/completed-campaign.manifest.json) was recorded **September 9, 2026, 14:16:52 IST** and binds **24 files / 54,473,654 bytes**, plus 13 source/lifecycle bindings. SHA256: `76565de6508b903ff19537a7f63352f1d649debef89abf1773b93096f8b727e8`. The evidence process PID 1038745 and supervisor 1038742 are gone; independent-review PID 1050104 is also gone. This is point-in-time integrity, not immutable storage. The sealed evidence and its bindings are unchanged by publication.

Historical supervised invocation; its output root is sealed and must not be reused:

```bash
python3 -B /home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/supervise_cpu.py \
  --config /home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/launch-config.json \
  --sha256 046ceaa68049eb58a6035a21342c8d62c419af6e85bf3f3a5a596f82beeb2b46
```

The [configuration](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/launch-config.json) retains the exact Python executable, command, working tree, source and frozen input hashes. The separate independent review used `timeout --signal=TERM --kill-after=3 120 python -B findings/independent_raw_review.py` from the C10 research root and completed in 0.897414 seconds.

## Decision and next check

C10 establishes a finite linear role-selection witness and successful frozen action composition on the reused development panel. It does not show that policy-only supervision discovers these parameters, isolate AdamW or any unique cause of C9 failure, demonstrate useful recurrence, or promote the architecture. Initial success cannot be credited to C7. New-core episode memory, active probing and ARC integration remain absent; successor fidelity and planning remain separate unmet prerequisites.

Next is the separately prepared [C11 confirmation](/home/stepan/Research/_runs/2026-09-09T131652Z-tofy-looped-cuda-readout-confirmation): fixed transplanted heads in Rust/CUDA on three new 256-query episode panels, preserving the shared-depth core and performing no refit. Those panels provide the new data needed to test transfer and actual runtime compatibility; this C10 publication supplies no C11 result. [Research synthesis](/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/synthesis.md); [canonical insight](/home/stepan/Research/ml/tofy/insights/looped-role-selector-witness-82ac8cb6.md).
