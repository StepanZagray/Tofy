# C14 frozen affine action witness — September 9, 2026 IST

**Neither registered true-label ridge arm recovers accurate actions from the retained pooling seam.** Final C12 features score **273/1024 seen, 262/1024 familiar and 125/512 held-out**; initial features score **288/1024, 245/1024 and 126/512**. The permuted-label arm scores **241/1024, 250/1024 and 122/512 against true labels**. Controls and independent numerical verification pass, while both true-arm witness gates fail: `registered_affine_witness_not_supported`. All results are exploratory because every population was already observed. No Best So Far or ARC metric improves. [analysis](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/analysis.json), [independent review](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/independent-review.json).

Two limited positive contrasts must not be lost in that conclusion: initial-feature seen accuracy exceeds its native/constant comparator on the same fitting cohort, and final-feature accuracy exceeds initial-feature accuracy on familiar queries. Neither reaches the absolute witness gates or establishes improved held-out behavior. A failed fit under this one objective and penalty is not a proof that the features lack action information.

## Fixed problem and population boundary

C14 follows [C12's failed policy-only training screen](2026-09-09-looped-grounded-policy-results.md) and [C13's failed frozen concentration intervention](2026-09-09-looped-query-sharpening-results.md). It uses only the six retained C12 factual streams: initial/final features for seen, familiar and held-out cohorts. **C13 sharpened features are not used.** Each decoder receives only the saved 256-dimensional pooled vector; labels, logits, attention, map/control IDs, role coordinates, cells, hashes and addresses are not appended to the feature matrix. [registration](/home/stepan/Research/_runs/2026-09-09T171342Z-tofy-looped-affine-action-witness/registration.md), [configuration and stream bindings](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/config.json).

The fit population is C12's 64 selected seen-training query groups ×16 fixed fit maps = **1,024 rows**. Familiar uses 64 retained queries ×16 fit maps; held-out uses those same 64 queries ×8 held-out maps. Familiar and held-out are paired views of the same layouts, not independent replications. All were previously inspected, including the transfer populations; there is no fresh confirmation. The C10 privileged role-supervised/oracle-pooled warm start remains part of feature provenance. [registration](/home/stepan/Research/_runs/2026-09-09T171342Z-tofy-looped-affine-action-witness/registration.md).

Three arms are fitted and retained without selection:

- `final_true`: terminal C12 pooled features and true fitting labels.
- `initial_true`: frozen-initial C12 pooled features and true labels, using the same procedure.
- `final_permuted`: terminal features with labels permuted within each of the 64 fitting groups, sequential PCG64 seed 1941 permutations of 16 positions. Four labels per action per group are preserved; all evaluation labels remain true.

There are **three CPU statistical fits, zero neural forwards, zero GPU work and zero neural optimizer updates**. Means and population standard deviations (`ddof=0`) come only from seen fitting features; scale is `max(std,1e-6)`. All 256 dimensions are retained. No transfer-dependent preprocessing, lambda/feature search, whitening, checkpoint selection or additional training occurs. [analysis environment and data-access declaration](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/analysis.json), [registration](/home/stepan/Research/_runs/2026-09-09T171342Z-tofy-looped-affine-action-witness/registration.md).

For standardized features (Z), one-hot labels (Y) and (n=1024), the fixed objective is

\[
\frac1n\lVert ZB+\mathbf1b^\top-Y\rVert_F^2+0.01\lVert B\rVert_F^2,
\qquad b=(0.25,0.25,0.25,0.25).
\]

The intercept is unpenalized. Positive ridge penalty makes the normal matrix `ZᵀZ/n + 0.01I` positive definite, giving a unique solution to this **fixed squared-loss problem** in real arithmetic. This is not a maximum-accuracy solution or C12's joint neural AdamW/CE objective. Positive feature scaling and centering preserve the affine function class in the original pooled vector; they change the fitting geometry and effective regularization, not representational capacity at this seam. Thus a difference from the retained decoder cannot be credited simply to a more expressive output layer. [registered derivation](/home/stepan/Research/_runs/2026-09-09T171342Z-tofy-looped-affine-action-witness/registration.md), [mechanism and limits](/home/stepan/Research/_runs/2026-09-09T171342Z-tofy-looped-affine-action-witness/findings/mechanism-check.md).

## Counts and controls

All action scores below use true labels, including the permuted arm. Labels are exactly balanced: 256/action in seen/familiar and 128/action in held-out. Every arm/cohort has **0/64 all-maps-correct query groups**.

| Arm / cohort | Correct / rows | Accuracy | Predictions [0,1,2,3] | Omitted-action correct | Demonstrated-action correct |
|---|---:|---:|---|---:|---:|
| final_true / seen | 273/1024 | 26.6602% | [245,272,432,75] | 75/256 | 198/768 |
| final_true / familiar | 262/1024 | 25.5859% | [241,214,475,94] | 59/256 | 203/768 |
| final_true / held-out | 125/512 | 24.4141% | [128,108,235,41] | 33/128 | 92/384 |
| initial_true / seen | 288/1024 | 28.1250% | [266,247,437,74] | 84/256 | 204/768 |
| initial_true / familiar | 245/1024 | 23.9258% | [282,197,423,122] | 57/256 | 188/768 |
| initial_true / held-out | 126/512 | 24.6094% | [154,96,206,56] | 32/128 | 94/384 |
| final_permuted / seen | 241/1024 | 23.5352% | [175,71,378,400] | 56/256 | 185/768 |
| final_permuted / familiar | 250/1024 | 24.4141% | [190,30,322,482] | 60/256 | 190/768 |
| final_permuted / held-out | 122/512 | 23.8281% | [109,12,159,232] | 29/128 | 93/384 |

The permuted arm scores **269/1024 against its fitting labels**, versus **241/1024 against true labels** on those same rows. The true arms' fitting-label and true-label counts coincide at 273/1024 and 288/1024. The null's familiar/held-out scores are below the coarse 50% control ceiling; one fixed permutation is not a null distribution and cannot prove absence of leakage. [analysis](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/analysis.json), `summaries`, `fit_counts`, `gates`.

For each fitted arm and cohort, replacing every map variant's pooled vector with its query-group mean yields identical within-group predictions and exactly 25% accuracy: 256/1024 or 128/512. This negative control works by construction because every complete group has balanced labels. It is a repeated representation control, not cleared-observation model behavior. The synthetic linear positive control and balanced group-only negative control preceded actual fits; overall `controls_valid` is true. [group-mean checks](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/analysis.json), `group_mean_controls`; [source freeze](/home/stepan/Research/_runs/2026-09-09T171342Z-tofy-looped-affine-action-witness/source-freeze.json).

No CE or calibration endpoint is reported: ridge outputs are uncalibrated real-valued scores and may be negative. The squared training objective below must not be read as neural cross-entropy or action accuracy.

## Paired contrasts and the registered decision

Here, “native” means the retained C12 spatial head's affine output on its corresponding initial/final pooled features, not the unused ordinary policy head. Initial native counts are256/1024,256/1024,128/512; final native counts are254/1024,255/1024,127/512. The constant comparator is exactly 25%. All entries are percentage-point differences with pointwise95% intervals.

| Arm / cohort | Fitted − native, pp [95% CI] | Fitted − constant, pp [95% CI] |
|---|---|---|
| final_true / seen | +1.8555 [−0.8789,+4.5898] | +1.6602 [−0.9766,+4.2969] |
| final_true / familiar | +0.6836 [−1.2695,+2.7344] | +0.5859 [−1.3672,+2.5391] |
| final_true / held-out | −0.3906 [−2.7344,+1.7578] | −0.5859 [−2.9297,+1.5625] |
| initial_true / seen | +3.1250 [+0.2930,+6.0547] | +3.1250 [+0.2930,+6.0547] |
| initial_true / familiar | −1.0742 [−3.1250,+0.9766] | −1.0742 [−3.1250,+0.9766] |
| initial_true / held-out | −0.3906 [−2.3438,+1.5625] | −0.3906 [−2.3438,+1.5625] |
| final_permuted / seen | −1.2695 [−3.3203,+0.7812] | −1.4648 [−3.5156,+0.5859] |
| final_permuted / familiar | −0.4883 [−2.4414,+1.4648] | −0.5859 [−2.5391,+1.3672] |
| final_permuted / held-out | −0.9766 [−3.7109,+1.9531] | −1.1719 [−3.9062,+1.5625] |

| Final true − initial true | Accuracy difference, pp [95% CI] |
|---|---|
| Seen | −1.4648 [−3.6133,+0.6836] |
| Familiar | +1.6602 [+0.0977,+3.2227] |
| Held-out | −0.1953 [−2.1484,+1.9531] |

The initial-feature **seen** contrast is a real positive pointwise result on the fitting population; it is not transfer evidence. The final-minus-initial **familiar** contrast also excludes zero. Familiar accuracy is 25.5859% final versus 23.9258% initial, and final does not beat its own native/constant comparator with a positive lower bound. This comparison combines changed body features, learned queries/pooling and newly fitted decoders; it does not isolate a benefit of core training. The held-out contrast remains near zero. [all contrasts and native counts](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/analysis.json), `contrasts`, `native`.

Bootstrap uses 10,000 whole-query resamples: PCG64 seed 1942 for seen and 1943 for both familiar and held-out, sharing fresh-query indices across those two cohorts and draws across arms/contrasts. Linear 2.5%/97.5% quantiles condition on fixed maps, selected queries and checkpoints. They are descriptive pointwise intervals, with no simultaneous or training-seed uncertainty guarantee. [registration](/home/stepan/Research/_runs/2026-09-09T171342Z-tofy-looped-affine-action-witness/registration.md), [independent draw identities](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/independent-review.json).

Each true arm required at least 90% on all three cohorts, positive lower 95% transfer advantages over native, and lower transfer advantages over constant greater than 25 percentage points, with valid controls. Neither arm approaches the absolute thresholds; none of its transfer advantages over native or constant has a positive lower bound. Both witness gates fail while controls pass. No arm is selected as a recovered model.

## Fit numerics and feature dispersion

| Arm | Penalized squared objective | Relative normal-equation residual | Normal-matrix condition number | Scale-clamped dimensions |
|---|---:|---:|---:|---:|
| final_true | 0.7489784351153376 | 4.3044257662382126e-14 | 12137.232240395 | 0/256 |
| initial_true | 0.7481142387179883 | 3.883901744327409e-14 | 8805.088148468 | 0/256 |
| final_permuted | 0.7498103623798239 | 6.218689291023551e-14 | 12137.232240395 | 0/256 |

The balanced constant score vector 0.25 has this squared objective 0.75. All three fits reduce it slightly; this is not evidence of high action accuracy. Every intercept is exactly `[0.25,0.25,0.25,0.25]`. Full means, scales, coefficients and intercepts are retained in [parameters.json](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/parameters.json). All relative residuals are far below the 1e-9 gate.

| Seen-feature std/scale quantile | Initial | Final (both final arms) |
|---|---:|---:|
| Minimum | 0.019263498 | 0.000856675 |
| 1% | 0.028880817 | 0.001207270 |
| 10% | 0.046178286 | 0.001685978 |
| 50% | 0.076211350 | 0.002804617 |
| 90% | 0.115269194 | 0.005362850 |
| 99% | 0.145102849 | 0.006976555 |
| Maximum | 0.159600623 | 0.009578232 |

Std and scale are equal here because no dimension reaches the 1e-6 floor. Final pooled features have much smaller raw spread, but the probe standardizes each arm using its own fitting population. Raw shrinkage alone therefore does not explain the failed fit; regularization and feature-spectrum differences remain possible influences. [fit preprocessing records](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/parameters.json).

| Features / cohort | Within-query, across-map RMS | Between-query RMS | Within / between |
|---|---:|---:|---:|
| Initial / seen | 0.002108790 | 0.082652773 | 2.5514% |
| Initial / familiar | 0.002095459 | 0.081479722 | 2.5718% |
| Initial / held-out | 0.002095543 | 0.081479087 | 2.5719% |
| Final / seen | 0.000037846 | 0.003551359 | 1.0657% |
| Final / familiar | 0.000036589 | 0.003499675 | 1.0455% |
| Final / held-out | 0.000036376 | 0.003499675 | 1.0394% |

Map-dependent variation is **nonzero** and smaller than between-query variation at both checkpoints. These raw-coordinate RMS values are descriptive and scale-dependent. They do not establish correct action-effect representation, causal sensitivity, information absence or a change attributable to the core alone: the pooled seam also depends on the learned query vectors. [reported dispersion](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/analysis.json), `dispersion`.

Independent augmented least squares solves the stacked system `[Z/sqrt(n); sqrt(.01)I]` against `[(Y-b)/sqrt(n); 0]`. Coefficients and all evaluated scores agree within the registered `1e-9 + 1e-7*abs(reference)` tolerance, and every action argmax agrees exactly. Its relative residuals range from 6.22e-14 to 1.96e-13. All **120 compared floating report fields match exactly**, maximum difference 0. This exact report agreement does not mean the two numerical solvers produce bit-identical coefficients. The review shares parent provenance bindings; it independently verifies the fits and scoring, not a new experiment. [independent review](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/independent-review.json).

## Runtime, profiling scope and provenance

Primary analysis including three fits takes **1.454417 s internally / 1.742292 s under its supervisor**. Independent review takes **0.948799 / 1.127540 s**. Both 120-second operation deadlines pass with return code 0, no errors and all owned PIDs/process groups gone. The primary and independent process PIDs were 1521204 and 1521281. [analysis lifecycle](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/operations/analysis.exit.json), [review lifecycle](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/operations/independent-review.exit.json), [primary timing](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/analysis.json), [review timing](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/independent-review.json).

The environment is CPU F64, Python 3.14.7, NumPy 2.4.2 and one OpenBLAS/OMP/MKL thread. C12's physical 34 feature producer owns the retained CUDA provenance. **There is no new Candle model workload or CUDA capture in C14**, so zero new model profiling is expected by design. C12's operation/activation linkage, allocation-lifetime, instrumented physical-memory, device-event and automatic correlation gaps remain; C14 makes no model-performance or GPU-cost claim from these CPU timings. [configuration environment](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/config.json), [runtime preflight](/home/stepan/Research/_runs/2026-09-09T171342Z-tofy-looped-affine-action-witness/findings/runtime-preflight.json).

| Identity | Exact value |
|---|---|
| Campaign | `/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST` |
| Operator revision | `05ac192226cb93f3a90ac716b982316dc17cff21` |
| C12 feature-producer revision | `5b17246cef110a866bdb8c2d2b14919710256ff3` |
| Original producer binary SHA256 | `60c677a48cdf297692d5f08c978da6926c18aa7d1a27e632162ed6416079b257` |
| CandleGraph revision | `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a` |
| Registration SHA256 | `2a6ba26a4f16a637a2d16ed07e90817b5632193396ddd968cd30b5a853a87b78` |
| Configuration SHA256 | `2a9dc53a8164d7972a1209af7365890ee116dc1058cae0af618ccdab64014497` |
| Three fitted parameter records SHA256 | `53d331ca9f976dcf4e3d21ad285df766a3c7e9a11619bdb714a5b172821ecd7f` |
| Within-group permutation SHA256 | `b022f734782a07d80452ec87980603873e58456fb0d6b49fd1fa13e21fe5ce77` |
| Analysis SHA256 | `6b14710df9f41e3ec143c6f5d4c2c49e377d579b85046c2916c61499524d2eb8` |
| Independent review SHA256 | `c9bf47b22e5e5a53d4f3fb4cb690694aaa4824b73f5d531994eb05852a3c42f3` |
| OpenBLAS library SHA256 | `c0f0784c075afdeb2d57cb78e6225221f7c97ef8d03e512b3c98e105054e73c2` |
| C12 parent outer seal SHA256 | `0a4ac9a8d3f46ebfca141360384a18761ea4c9fd175c43176788dd422ec7ab6f` |
| C13 parent outer seal SHA256 | `fbaef96bc4e520301edfb2b79e811467d226e45509a7aee078ebdfdebabcac6f` |
| Completed C14 outer seal SHA256 | `9926967421887885079f91a7c6f4f3cbc5a23266ad5f305493cb6159820c99ff` |

Exact original initial/final core and policy hashes, six input streams, audits and frozen operator/source bindings are retained in the [configuration](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/config.json). The completed outer seal at **18:33:44 IST** verifies **12 files / 232,108 bytes, 84 external bindings and four recorded PIDs gone**, with zero new CUDA bundles because C14 executes no model workload. Analysis plus independent review takes **2.8865 s** of campaign execution wall time, below the 600-second limit. Classification is `completed_exploratory_negative_evidence`; the digest is recorded outside the campaign root. This is point-in-time verification, not immutable storage. [completed manifest](/home/stepan/Research/_runs/2026-09-09T171342Z-tofy-looped-affine-action-witness/completed-campaign.manifest.json), [seal and execution verification](/home/stepan/Research/_runs/2026-09-09T171342Z-tofy-looped-affine-action-witness/completed-campaign-verification.json).

## Recorded commands and decision

These exact historical argument lists come from the [analysis process record](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/operations/analysis.process.json) and [independent-review process record](/home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/operations/independent-review.process.json). The working directory was `/home/stepan/Projects/code/Tofy-grounded-diagnostics`; the reviewed operator source was pushed and clean before launch. Do not reuse the completed output roots.

```bash
/home/stepan/venvs/tensorboard/bin/python3 -B /home/stepan/Research/_runs/2026-09-09T171342Z-tofy-looped-affine-action-witness/analyze.py \
  --config /home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/config.json \
  --config-sha256 2a9dc53a8164d7972a1209af7365890ee116dc1058cae0af618ccdab64014497 \
  --output /home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/analysis.json

/home/stepan/venvs/tensorboard/bin/python3 -B /home/stepan/Research/_runs/2026-09-09T171342Z-tofy-looped-affine-action-witness/independent_review.py \
  --config /home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/config.json \
  --config-sha256 2a9dc53a8164d7972a1209af7365890ee116dc1058cae0af618ccdab64014497 \
  --report /home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/analysis.json \
  --report-sha256 6b14710df9f41e3ec143c6f5d4c2c49e377d579b85046c2916c61499524d2eb8 \
  --output /home/stepan/Projects/code/.tofy-runs/looped-affine-action-witness-20260909T182856-IST/independent-review.json
```

After C12's failed training screen, C13's failed concentration recovery and this failed fixed affine witness, the working direction is to reassess how demonstrations represent actions and their visual effects before committing to more training. The fixed penalty and feature-spectrum effects leave the affine function class unexhausted, so the next step is to choose the smallest discriminating check before redesigning the representation. The next architecture, interface and training recipe remain unselected. This report does not authorize an automatic lambda sweep or another run.

C14 rules out success of this registered fitting procedure on these fixed pooled features and reused populations. It does **not** rule out another affine optimization objective or penalty, nonlinear decoding, information elsewhere in the model, or a different estimation population. Initial/final contrasts combine body and query/pooling changes plus fitted readout changes, so they do not isolate core learning. No fresh confirmation, multi-seed promotion, ARC, planner, dynamics, memory or useful-recurrence claim follows.
