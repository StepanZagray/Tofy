# Balanced fixed-set fitting diagnostic

Register before implementation or fitting outcomes. The first 256-update screen
failed useful rule use. Frozen source 3684e4c7 then showed CPU/CUDA probability
agreement within 1.5e-7 and small, nonzero support effects around 6e-5, against a
roughly .094 winning-action margin. This rules out the registered backend mismatch
on the tested rows, but does not explain optimization or representation limits.

## Bounded claim and mathematical control

Can the unchanged 992,393-parameter RMS-normalized shared transformer and its
existing optimizer fit correct history-dependent actions on 128 fixed examples
within 1,200 updates? This tests optimizer reachability of a fitting solution under
this recipe. It cannot prove global expressivity, generalization, the cause of
the first failure, or ARC readiness.

Construct eight layout IDs 0..7 times all sixteen training-rule IDs from the
existing task generator, seed `9173 XOR TRAIN_TAG`, distance exactly one. Reuse the
same 128 samples in layout-major/rule-major order. Each 64-example optimizer batch
contains four complete layouts. For a given layout, query pixels are identical
across rules, the optimal action is unique, and every action is correct four times.
Only the observed calibration varies with the rule; no rule ID is a model input.

For cleared support, any deterministic model has the same probability vector q
for each layout's sixteen inputs. Its mean CE is
`H(uniform_4,q) = ln(4) + KL(uniform_4 || q) >= ln(4)`; its argmax accuracy is
exactly 25%. This analytic lower bound and identical-input control make an extra
300-update blind training arm unnecessary. Evaluate cleared inputs at every fit
check and fail integrity if CE < ln(4)-0.0001 or accuracy differs from 25%.

## Invariants, budget and stopping

- Exact model.rs/task.rs bytes unchanged from source 2d194f89. Named initialization
  seed 0 must reproduce the first screen's initial checkpoint hash. No warm start.
- Hidden 128, four heads, two shared blocks, training loops 1/2/4 repeating,
  maximum inference loops eight; evaluate the fitting gate at four loops.
- AdamW .0003, weight decay .01, default pinned Candle beta/epsilon, global gradient
  norm clip 1.0. Policy/value/reward/dynamics weights 1/.1/.1/.5, unchanged losses.
- Effective batch 64, physical 33, accumulation two with 31 examples in the final
  microbatch. Reconfirm a two-update exact-binary CUDA fit smoke before the run.
- Fixed examples replace fresh examples, all goals are one step, and repetitions
  increase. Name these changes; do not claim a single-factor comparison with the
  original screen. This is a diagnostic, not a treatment-promotion experiment.
- At update zero and every 25 updates, evaluate all 128 memorized examples with
  true and cleared support. Report policy CE, unique-label accuracy, distinct
  selected actions and prediction hash. Retain the complete log, not a best slice.
- Stop at the first registered check with true accuracy >=90% and CE <=.35, with
  the cleared control passing. Otherwise stop after 1,200 updates or 35 minutes
  including generation, training and fit checks. No automatic extension.
- Expected training time from the screen's measured loop schedule is roughly
  27 minutes, plus fit checks, if the full budget is needed. No public evaluation
  or learned-tree search is included. A new binary hash is mandatory because the
  runner changes; source and model hashes remain explicit.

## Decisions

PASS supports local fitting capacity and warrants a new generalization experiment;
it does not prove that more fresh-data updates alone will work. If accuracy remains
20..30% and CE within .05 of ln(4) at the cap, record a marginal-output failure of
this optimization recipe; representation and optimization remain possible causes.
Intermediate results are inconclusive, with no automatic longer run. A failed
cleared-control, hash, source, initialization or backend gate is invalid evidence.

After a failure or intermediate result, choose one subsequent diagnostic from the
observed failure: objective competition, pairwise transition representation, or
optimization. Do not change several factors or infer the winning remedy from
advisor agreement. All fit artifacts are exploratory fitting diagnostics, not
the previous screen's held-out evidence and not completed ARC capability.
