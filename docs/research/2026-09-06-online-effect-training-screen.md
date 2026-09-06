# Online effect learning: registered procedural prerequisite

Registered before the first non-fixture optimizer run. The hypothesis is local:
a fresh four-convolution action-conditioned CNN learns local visible effects
more accurately than the same initialized CNN with constant action input.
There is no public ARC data or pretraining checkpoint in either arm. This is
an empirical prerequisite for later online exploration, not evidence of ARC
goal understanding, an optimal controller, or a predicted prize score.

## Distribution and comparison

Each procedural environment has a fixed pair of distinct palette categories
and a fixed random permutation of four directional inputs. A 3x3 foreground
square is positioned independently in a 64x64 background for each transition.
Inputs1–4 move it one pixel according to that permutation;5/7 are noops;
input6 changes its clicked pixel only when the click lies inside the square.
Positions stay away from boundaries. There is no goal, reward, timer, animation,
obstacle, hidden state or cross-level transfer. This deliberately tests the
model's local receptive field; remote click effects are outside its capability.
No public game assets, trajectories, identifiers or mechanics are imported.
Training centers have even x+y and evaluation centers have odd x+y, so current
frames cannot overlap across the two streams. This spatial split and the fixed
per-environment palette pair are set before the first optimizer experiment.

Conditioned and constant arms share exact initial model seed, generator seed,
ordered factual transitions, optimizer and updates. The constant arm receives
ACTION5 for every input, retaining the actual next-frame target. Actual input
ids and coordinates still determine the simulator outcome and dataset hash.
This is a trained action-independent comparator, not an inference-only shuffle.
Report the number of replaced tuples and genuinely outcome-different pairs.
Compare initial prediction fingerprints using identical true-action probes for
both arms; arm-specific input fingerprints would not check matched initialization.
Under deterministic evaluation, the constant-input arm must tie all seven
action scores on a given frame: its inputs are identical. Its action-ranking
control is therefore0.5 by construction. This local property does not prove
that the conditioned CNN can learn accurate change maps; absolute held-out F1,
coordinate use and noop precision are the unknown empirical prerequisites.
The constant arm is still trained to measure spatial-prediction performance
under the matched sample/optimizer budget, not to discover that ranking tie.

Paired model/data seeds: (0,101),(1,102),(2,103). Exactly512 online observations
and512 optimizer updates per arm/environment. Final checkpoint only, no selection.
Hidden width16, three3x3 convolutions plus a1x1 output, F32, deterministic
initialization, AdamW learning rate0.001, weight decay0, no clipping or schedule.
Replay retains256 rows; each update uses the most recent64, ramping from1 to64
as observations arrive. Intended full effective batch64, accumulation1. Match
that exposure schedule across arms. The objective balances changed/unchanged
pixels over the physical batch, using the existing class when one is absent.

## Capacity, checks and compute

Build and launch only reviewed pushed clean source with the cuDNN feature;
hash the exact example binary and record sibling revision, command, host/device,
configuration and generated-data identity in each never-reused root.
First run a two-update non-ARC CUDA smoke on that binary. Then measure physical
batches16,32,64 on the same generated capacity stream (model/data seed999),
using batch+8 observations so every candidate reaches full batch. These are
selection-only resource probes; their model-quality values are discarded.
The largest stable batch up to the intended effective64 is selected. If64
fails or cannot fit with reasonable headroom, stop and amend the accumulation
implementation before the comparison; do not silently change sample exposure.
Keep the full intended effective64 after the ramp, with accumulation1 if64 fits.

Maximum300 seconds per capacity/arm run, external hard watchdog and exact PID
cleanup. Maximum45 minutes GPU time including smokes, capacity and six arms.
Record elapsed update durations and peak device memory; no simultaneous GPU job.
Stop an arm on nonfinite loss/gradient/parameters, zero unexpected gradient,
wrong data identity, timing cap or failed source/device/provenance checks.
Failed runs are infrastructure/implementation evidence, never promotion data.

## Frozen evaluation and decision

Score prequential predictions before each factual observation trains the model.
Then freeze weights and evaluate64 separately generated frames, all seven
actual action outcomes per frame. Alternate click at the object center and
an outside coordinate. The evaluation seed construction is fixed in the
reviewed generator before execution and differs from the training stream.
Both arms receive identical frames, actions and targets. Score every frame;
no palette/seed subset, threshold, output or checkpoint selection after results.

Primary prerequisites, required on all three conditioned seeds:

- Changed-pixel micro F1 at least0.70 and at least0.15 higher than the paired
  constant arm; fixed0.5 probability threshold.
- Ranking accuracy at least0.90 between genuinely changed and noop action
  outcomes from the same frame; score is summed change probability, ties0.5.
- False-positive pixel rate on fully noop outcomes at most0.001.
- Report ACTION6 changed-pixel F1 separately; require at least0.50, so movement
  success cannot hide missing coordinate use. Noop action rows have no positive
  class and are reported as such, not excluded from false-positive metrics.

Also report precision/recall, per-action confusion counts, prequential balanced
BCE, gradient norms, physical batch, action-conditioned/outcome-changing counts,
and copy-zero control (changeF1=0 on this populated positive class). This is a
three-seed prerequisite screen; report each pair and min/max, not a global
confidence claim or a multiple-hypothesis search. Gradient cosine measurement
is not applicable: there is one loss, and the intervention changes only input.

PASS admits one separately registered public-development online-control test;
it does not promote model quality on ARC or change Best So Far. FAIL triggers
analysis of a named premise before one new intervention; no automatic longer
training. These stage stop rules do not end the autonomous public-performance
campaign. The22 reserved public games remain untouched by policy inference.
