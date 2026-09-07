# Frozen-score separability after the failed online effect prerequisite

This diagnostic follows the completed three-seed FAIL at3a925df0. It asks
whether a single decision threshold could recover accurate masks from those
same final scores. It does not change the old0.5 gate or promote a model.
The population, optimizer, architecture, input arms,512-update budget and
final-only evaluation remain exactly those in the original training screen.
There is no public data, goal, value objective or connected public controller.

For fixed positive class weights, minimizing conditional weighted BCE gives
q = w1*p / (w1*p + w0*(1-p)). Thus q >=0.5 tests p >=w0/(w1+w0).
When weights are inverse class prevalence, the nominal probability correction
has threshold q >=1-prevalence for a posterior0.5 decision. This derivation
assumes fixed weights, sufficient capacity and population minimization. Here
weights vary per replay batch, training is finite, and deterministic targets
still have the correct0/1 optimum for any positive weights. Consequently the
loss can contribute to broad masks but the current evidence does not prove a
calibration cause. Aggregate confusion counts do not establish a ring shape.

Implement only optional final held-out score capture in the probe. Require the
learner source SHA256 to remain
`97425bd6c099b84cb18cdc832c7a6caa445bcea05cad8996bd29ebc5d048ba4e`.
Commit, review, push, rebuild with cuDNN, hash the separate launch binary, and
run a two-update CUDA smoke including score capture. The existing capacity
qualification64/accumulation1 remains applicable because architecture, dtype,
training buffers and effective schedule are unchanged; verify the actual64
batch and memory in each new run. Capture occurs only after training.

Reproduce conditioned seeds (0,101),(1,102),(2,103),512 updates each, saving all
448 probability maps with factual changed-pixel indices, action tuples and
frame identities. Do not retrain the constant arm: its existing complete
matched evidence is only a reference and no new intervention comparison is
being made. Before inspecting threshold results, require exact equality to
each old conditioned report for generated-data SHA256, initial and final
prediction SHA256, held_out, prequential and totals. Source/binary/path/timing
and new capture metadata necessarily differ and are excluded from that equality.
If exact equality fails, stop this frozen-rescoring claim and analyze; do not
relax the criterion after seeing differences. Save all failed roots.

Before parsing model maps, test the independent rescoring function on oracle,
copy-zero and tied-score fixtures. Require its threshold0.5 confusion counts
to match the original evaluator exactly for the full population and each action.
Also require448 unique frame/action tuples,4096 finite probabilities in[0,1]
per tuple,1568 positives,288 changed/160 noop outcomes, and reported map hashes.

Frozen diagnostics, all explicitly exploratory with no promotion claim:

- Full precision-recall curve grouped at equal score values, non-interpolated
  average precision, and maximum attainable micro F1 across thresholds.
- F1, precision, recall, noop false-positive rate and ACTION6 F1 at fixed
  thresholds0.5,0.9,0.99 and1-training_prevalence. The prevalence is computed
  from all512 factual observations, not held-out labels. The last threshold is
  only an approximate fixed-weight correction, not a calibrated estimator here.
- Report ACTION6's own maximum F1 separately, and whether a common threshold
  can satisfy F1>=0.70, click F1>=0.50 and noop FP<=0.001 for each seed and
  across all three seeds. Separate per-action maxima do not imply a common
  threshold exists. Ranking scores are unchanged; retain their old values.

Do not use the optimum threshold as a new result. If no threshold works,
threshold-only repair is insufficient for these frozen scores; objective,
optimization or representation changes could still change their ordering.
If a common threshold works, threshold separability is supported only on this
development fixture. Either branch requires analysis and a newly registered
single intervention, with fresh seeds and the original absolute gates, before
any prerequisite promotion. Do not auto-chain longer training or choose a seed.

Maximum300 seconds per reproduction,20 minutes total GPU preparation/reproduction;
expected measured optimizer time is about25 seconds plus capture/build overhead.
No concurrent GPU model load. Exact PID watchdog/telemetry, new roots, clean
source checks, finalized manifest hashes and external digests remain mandatory.
Keep the independent language-model resource/development route moving; it has
no dependency on this CNN's outcome. The22 reserved public games remain untouched.
