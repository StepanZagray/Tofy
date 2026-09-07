# Four replay updates per factual observation: registered optimization screen

This is a fresh empirical test after the512-update prerequisite FAIL and exact
frozen separability diagnostic atfe5571d3. One seed cannot reach micro F1>=0.70
at any threshold on its frozen scores. The existing CNN can represent the
registered local generator using11/9/1 hidden channels within width16; this
constructive existence result does not imply optimizer convergence. Thus test
additional optimization before changing representation. No public game data,
goal/value signal, remote click effect or connected public controller is present.

Claim: four optimizer updates per factual observation can satisfy the original
local action-effect accuracy gates on three fresh procedural environments,
using exactly the same observation stream as one update. This explicitly spends
four times the optimizer updates; it is a quality-versus-compute test, not a
matched-compute improvement or proof of optimality. The unknown is whether this
small increase in replay optimization is sufficient, not whether extra steps
are mathematically necessary. Source review and tests must confirm that the
factual transition is recorded once and that prequential prediction precedes
all updates for that observation.

Distribution and preprocessing are exactly the original online-effect screen:
fixed per-environment two-category palette and movement permutation, independent
3x3-square positions, action1..4 one-pixel movement,5/7 noop,6 clicked-pixel erasure
only inside the square. Training centers have even x+y; held-out centers odd.
No new data generator, augmentation, mask, architecture, threshold or loss.

Fresh model/data seeds (10,201),(11,202),(12,203), all used. For each seed run,
in fixed order:

1. Conditioned, one optimizer update per observation:512 updates.
2. Conditioned, four optimizer updates per observation:2048 updates.
3. Constant ACTION5 input, four updates per observation:2048 updates.

Every arm has the same512 factual observations/targets in the same order and
identical initialized parameters. Data identity excludes the update multiplier
and input arm. Initial fingerprints use the same true-action probes for all.
The constant arm is a matched-2048-update action-independent comparator; its
action ranking must tie at0.5 by construction, while its spatial F1 is empirical.
The one-update conditioned arm isolates the optimizer schedule on fresh seeds.

Width16, three3x3 and one1x1 convolutions, F32, existing balanced BCE, AdamW
lr0.001/weight decay0, default betas, no clipping or learning-rate schedule.
Replay capacity256, latest64 rows per update with original ramp1..64. Intended
effective/physical batch64, accumulation1. The four updates for each observation
use the same selected replay rows; no extra observations or replay insertions.
Record every optimizer step or explicit first/last/count summaries without
calling a last-step value an average. Record actual observations versus updates.

Only the final model is evaluated: same64 held-out frames and seven actual
action outcomes per frame,448 tuples,288 changed/160 noop,1568 changed pixels,
704 genuinely changed-vs-noop action pairs. Preserve oracle/copy controls,
per-action confusion, ACTION6 changed32/noop32, full observation/data counts,
action replacement identity, gradients, and raw final score capture. All gates
use the original fixed probability threshold0.5; threshold search is not a gate.

Required on every four-update conditioned seed:

- Micro changed F1>=0.70 and >=0.15 gain over paired four-update constant input.
- Changed-vs-noop action ranking>=0.90, noop FP pixel rate<=0.001, click F1>=0.50.
- Relative to one-update conditioned input: if its F1<0.70, require improvement
  >=0.05; otherwise require no degradation larger than0.01. This conditional
  ceiling-safe rule is fixed before results and does not select a seed subset.

All three must pass. Report each arm/seed, min/max and paired differences; no
population confidence claim from three environments or best-seed selection.
There is one preregistered treatment. No multiple threshold/checkpoint search.
The same-budget constant comparison and the unequal-compute optimization
comparison answer different claims and must remain separately labeled.

Launch reviewed pushed clean source, record exact sibling revision, feature
flags/build command, binary SHA256, generator/configuration, hardware/software,
batch/accumulation and seeds in never-reused roots. First run the exact cuDNN
binary on a two-observation/four-update CUDA smoke (eight total updates).
Repeat bounded physical16/32/64 capacity probes under multiplier4 and seed999
to verify the largest stable batch up to intended effective64; their quality
is selection-only. No simultaneous GPU job. Abort on nonfinite values, wrong
source/device/data identity, observation/update count, zero unexpected gradient
or failed evaluator/oracle/control integrity. Fail closed; do not change a root.

Maximum300 seconds per arm/capacity test,20 minutes total GPU stage including
smokes; expected optimizer time from the measured one-update runs is about
three minutes, but overhead will be measured. An external watchdog owns exact
training/telemetry PIDs. Stop and verify children before sealing, hash all final
artifacts and preserve manifest digests outside the roots. Stop on a failed
integrity check; no automatic restarts that hide failed provenance.

PASS admits only this CNN's separately registered online-control component
test on development games, not ARC promotion. FAIL requires analysis before
one new intervention; no automatic extra epochs, wider model or loss sweep.
Neither outcome blocks the independent pretrained-agent development screen or
ends the autonomous public-performance campaign. The22 reserved games remain
unseen by policy inference and absent from pretraining.
