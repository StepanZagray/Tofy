# C14 affine action witness

Registered September9,2026 at18:14 IST, before reading feature-value statistics or fitting probes. Source C12 5b17246cef110a866bdb8c2d2b14919710256ff3; frozen C12 outer seal0a4ac9a8d3f46ebfca141360384a18761ea4c9fd175c43176788dd422ec7ab6f. C13 operatorf802bef4 and outerfbaef96bc4e520301edfb2b79e811467d226e45509a7aee078ebdfdebabcac6f establish sharpening alone did not restore action accuracy. No C13 features are used here.

## Exact hypothesis and decision scope

An affine decoder fit only on C12 seen pooled256 can score>=90% on seen, familiar and heldout-map cohorts, exceeding the native spatial decoder on both transfers by a strictly positive lower paired95% accuracy interval and exceeding the exact25% query-only baseline by a lower bound>25percentage points. This is a feature-readout existence witness under one specified fitting procedure, not a better model, global optimality, information-absence or original-training-cause claim. All panels were already observed; every result is exploratory. A fresh independent confirmation and actual model interface qualification are required before promotion.

## Mathematical claim before compute

For fixed finite training features standardized only using their training population mean and std, with scale=max(std,1e-6), define Z and one-hot Y. Minimize (1/n)||ZB + 1b^T - Y||_F^2 + .01||B||_F^2, summed across four output columns and averaged over n=1024 rows. The intercept is unpenalized. In real arithmetic Z has column mean0, b=mean(Y)=(.25,.25,.25,.25), and B=(Z^T Z/n + .01I)^(-1) Z^T(Y-b)/n. The Hessian in B is positive definite because lambda>0, hence a unique minimum for this fixed squared-loss problem. It does not optimize zero-one action accuracy or C12 joint AdamW/CE. Numerical residuals and an independent augmented least-squares solve verify the implementation.

All feature scales are strictly positive, so the fitted standardized decoder is still affine in the original256 features: preprocessing does not enlarge the native affine function class. For feature vectors equal within every complete balanced query group, sum_map(Y-b)=0, therefore the exact feature-label cross-product is0 and the ridge B=0. Any deterministic head evaluated on a repeated group-mean feature also makes identical decisions across the16 or8 maps, giving exactly25%; this is the query-only negative control. A failed probe cannot imply absent information: nonlinear decoding, information outside this pooling seam, finite64-group estimation, and regularization/feature-spectrum effects remain counterexamples.

## Inputs, arms and leakage boundary

Retained C12 streams only: frozen-initial factual and terminal-final factual for seen1024, familiar1024 and heldout512 rows. Familiar and heldout share64query layouts, not independent replications. Seen uses64training groups, each16fit maps; same complete ordering, labels and audits as C12. Only pooled[256] enters feature matrices. Logits, attention, cells, map/control IDs, labels, roles, hashes and row addresses remain validation/target fields. Reconstruct labels independently from visible cells plus the audited bijection; never append them to X.

Three fitted arms, reported without selection:
1. final_true: terminal factual pooled256, true fitting labels.
2. initial_true: frozen-initial factual pooled256, same true labels/procedure.
3. final_permuted: terminal factual pooled256, labels independently permuted within each of64fitting groups by sequential PCG64 seed1941 permutations of16 row positions. Preserve four labels/action/group; transfer labels remain true.

Fit uses only seen rows, train mean/std(ddof0), fixedlambda.01 and scale floor1e-6. No evaluation-derived centering, scaling, feature selection, tuning, checkpoint/arm selection or early stopping. Keep all256dimensions. Native C12 logits give no-fit comparators. For each fitted arm/cohort, group-average that cohort's pooled features and repeat the identical vector across every map in the group; score with the unchanged fitted decoder and require exactly25% and identical within-group predictions. This is an algebraic invariance control, not cleared-observation behavior or a trained policy.

## Numerics, diagnostics and controls

CPU F64 NumPy with one OpenBLAS/OMP/MKL thread, fixed pinned Python/NumPy/BLAS environment. Reject nonfinite values, incomplete rows, changed hashes/audits/source or wrong schema. Normal-equation relative residual norm<=1e-9, use max(||RHS||,float64 tiny) denominator; exact-zero RHS must yield zero solution. Require independent augmented least squares using [Z/sqrt(n);sqrt(.01)I] and [(Y-b)/sqrt(n);0] to match coefficients and all evaluated scores elementwise atol1e-9+rtol1e-7, and all action argmax exactly. If a numerical tie changes argmax, fail integrity; do not choose a tolerance after results. Report training objective, residual, std/scale quantiles0,.01,.1,.5,.9,.99,1, clamped dimensions, normal-matrix condition number and within-group vs between-group feature RMS dispersion. These are descriptive, not outcome-selected gates.

Synthetic positive control verifies a known linearly encoded4-action feature across disjoint examples; negative balanced group-only features must give25%. Test affine-score reconstruction, training-only preprocessing, wrong labels/hashes and controlled corruption before real fits. Actual final_permuted transfer accuracy must be<=50% on both cohorts as a coarse pipeline gate; one permutation is not a null distribution and passing it does not prove absence of leakage. If it fails, report inconclusive_failed_control with all numbers intact.

No CE/calibration metric: ridge scores are uncalibrated and may be negative. Report true-label correct/rows, accuracy, action histogram, all-maps-correct query groups, omitted/demonstrated subsets, fitting-label and true-label fit counts for the permuted arm. Report each true arm minus its native comparator, each minus constant25%, and final_true minus initial_true; the last is a joint core/query/readout contrast, not isolated core learning.

## Uncertainty, gates and stop rules

Use10000whole-query bootstrap resamples, PCG64 seed1942 for seen and1943 for fresh; the same fresh query indices for familiar and heldout, and shared indices across all arms/contrasts. Linear pointwise95%quantiles; intervals condition on the fixed map set, selected queries and checkpoints. They do not estimate unseen-map-population or training-seed uncertainty. No simultaneous/multiple-method promotion claim. All secondary contrasts are descriptive.

Each true arm passes witness gates only if all integrity/null gates pass, seen/familiar/heldout accuracy>=.90, and both transfer lowerCI gains over native>0 and constant>.25. If both pass: affine_witness_already_present_initially. If final alone: final_affine_witness_supported_exploratorily. If initial alone: initial_only_affine_witness. If none: registered_affine_witness_not_supported. If null/numerics fail: inconclusive_failed_control or failed_integrity. No additional lambda, whitening, architecture or training run may launch automatically from this result; analyze it first.

## Execution and provenance

Zero accelerator/model forwards and zero neural optimizer updates. This is a closed-form CPU statistical probe; C12's existing full wired CUDA captures own feature-producer evidence. There is no new Candle model invocation to profile. Source/operation/activation/memory/device-event gaps inherited from C12 remain. Record separate probe fits, exact source snapshots, scripts, Python/NumPy/BLAS identities, original binary/core/head hashes, source streams/audits, configuration and profiler scope.

Launch only after reviewed pushed clean operator commit, verified C12/C13 seals and positive/negative tests. Each analysis and independent-review operation<=120s; overall execution<=600s from first real-fit invocation, excluding implementation/review/publication. New never-reused campaign root, tracked PIDs, all cleanup before point-in-time sealing and external digest. Source mismatches or numerical/control failures stop this diagnostic without changing its recipe. Record positive and negative evidence and the next falsifiable decision in the library and RESULTS_P2.
