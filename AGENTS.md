## Scope
- These rules apply to this repository.
- Read this file before making edits.

## Chat
- If the user asks for explanation only, do not modify files.
- Prefer concrete, implementation-level explanations.

## Edits
- Use repository-native tooling and normal editor operations for file changes.
- Keep changes minimal and targeted; avoid unrelated refactors.
- Do not preserve backward compatibility unless the user explicitly asks for it.
- Write as little code as possible while still solving the problem clearly.
- Always use available hardware to its safe maximum without reducing quality.
  For accelerator training, measure the largest stable physical batch, then
  minimize gradient accumulation while preserving the intended effective batch
  and optimizer schedule. Record the selected batch/accumulation pair in the
  experiment results.

## Validation
- After code edits, run lint/compile checks when possible.
- Resolve all errors and warnings.

## Research
- Always use the globally installed `research` skill (the local-memory workflow under
  `~/.agents/skills/research/`) when analyzing or comparing experiments or experiment results,
  and whenever the user asks to research, or produce a source-backed report.
- Follow the skill's global-library workflow and use the `ml/tofy` scope for Tofy research.
- For every completed experiment analysis, preserve a concise insight under
  `ml/tofy/insights/` with exact run identifiers and revisions, positive and negative results,
  metric or causal confounds, the resulting decision, and the next falsifiable experiment.
  Link it from `ml/tofy/insights/INDEX.md` and the Tofy scope index.

## Scientific experiment contract
- Define the exact claim before implementing or launching an experiment. Specify the task or
  environment distribution, objective, comparator class, compute budget, and meaning of
  "better", "optimal", or "proved". An empirical run can falsify or support a bounded claim;
  it cannot prove global method optimality or guarantee ARC-AGI performance.
- Before spending compute, investigate whether the claim can be derived mathematically. State
  assumptions, try to construct counterexamples, and separate proved local properties from
  unsupported system-level conclusions. When a proof is unavailable, label the proposal as an
  empirical hypothesis and test the weakest prerequisite with the cheapest decisive diagnostic.
- Prefer current primary sources and check for newer work before relying on a method. Record the
  exact paper/specification version, theorem assumptions, and evidence scope. Do not treat paper
  popularity, agent agreement, or a related benchmark result as proof for Tofy.
- Call an experiment "paper-faithful" only after recording a fidelity matrix covering the
  representation seam, preprocessing, sample/population construction, temporal or spatial
  statistics, projection distribution/count, loss and reduction, caps, optimizer, clipping,
  model class, data-generating process, action coverage, evaluator, and hardware/software
  environment. Any material mismatch must be named; classify the method as an adaptation and do
  not transfer the paper's theorem or empirical claim without a new justification.
- Preregister the intervention, baseline, invariants, seeds, checkpoints, metrics, uncertainty
  calculation, multiplicity policy, promotion/rejection thresholds, stop rule, maximum runtime,
  data-access boundary, and next decision. Never choose a metric, checkpoint, seed subset, or
  threshold after observing results without labeling the analysis exploratory and confirming it
  in a fresh run.
- Prefer premise checks and frozen-checkpoint rescoring before retraining. Validate evaluator
  sensitivity with positive/negative controls, confirm the tested population contains the needed
  classes or genuinely changed pairs, and keep semantic grounding, non-collapse, action use,
  rollout fidelity, and planner/Q validity as independent gates.

## Fair experiment execution
- Change one causal factor at a time when feasible. Match initialization, data order, update and
  token/sample budget, effective and physical batch, optimizer schedule, clipping, checkpoint
  selection, evaluator episodes, and hardware across arms. Record unavoidable differences and
  measure component gradient pressure/cosines when a loss intervention competes with other
  objectives.
- Use multiple seeds for promotion claims. A single seed is a screen only. Report absolute
  performance, copy/null/control baselines, confidence intervals where appropriate, and negative
  or contradictory outcomes; do not promote a method merely because it is less bad than another
  failing arm.
- Freeze and validate evaluators before the treatment comparison. For evaluator-only changes,
  require identical checkpoints and inputs and, when possible, exact parity for all legacy report
  fields and episode streams so the new metric is the only changed quantity.
- For conditioning interventions, report total rows, eligible rows, genuinely changed tuples,
  and outcome-changing tuples. Distinguish action-tuple sensitivity from correct alternative-state
  prediction; a changed input with the same simulator outcome is not a causal positive example.
- Estimate runtime from measured stage durations and reserve time for integrity checks and
  analysis. Auto-sequence a dependent experiment only when its branch can be selected by a
  preregistered machine-checkable rule; otherwise stop after the parent result for analysis.

## Experiment provenance and launch safety
- Launch only from a reviewed, pushed commit in a clean checkout. Record the exact Tofy revision,
  sibling dependency revisions, build command and feature flags, binary SHA-256, configuration,
  hardware, physical batch/accumulation pair, source-run identity, and evaluation seed in the run.
- Run a bounded device smoke test on the exact launch binary before a long campaign. Verify the
  required accelerator backend is compiled (P2 RunPod builds normally require
  `cargo build --release --locked --features cudnn,profiling`), the expected device opens, and one minimal
  evaluation reaches its metric/integrity checks. A successful CPU build is not a CUDA preflight.
- Before an automatic handoff, test repository fetch authentication and exact-commit availability,
  verify the parent run is sealed and its manifest passes, and never replace a binary still used by
  an active process. Use a separately hashed binary or wait for the parent to exit.
- Give every run a never-reused root and explicit lifecycle state such as `running`,
  `complete_pending_analysis`, or `failed_integrity_or_evaluation`. Fail closed on hash, source,
  evaluator, or environment mismatches. Preserve failed-launch provenance, but exclude it from
  model evidence and never silently reuse a partial root.
- Classify artifacts as completed evidence, selection-only, exploratory, implementation smoke, or
  failed infrastructure/integrity. Only completed evidence may satisfy a registered promotion gate;
  selection-only or exploratory findings require fresh held-out confirmation.
- Stop telemetry and child processes before sealing results. Hash the finalized artifact tree and
  verify it. Treat this as point-in-time integrity, not immutable storage; record the manifest
  digest outside the run root during analysis. Keep intentionally active remote sessions named and
  supervised; terminate and verify cleanup of every other process started by an agent.

## Delegation
- Capable primary agents should delegate bounded, independent coding, documentation, analysis, and research work when doing so improves speed or supplies an independent review.
- Keep delegated tasks narrow, safe, and verifiable. The primary agent owns scope, safety, integration decisions, review of every delegated change, and final validation.
- Never claim a delegated model was used if the CLI or account rejected it; record the failure and continue with an available agent or locally.

## RunPod SSH
- RunPod pods may reject non-PTY SSH commands with `Error: Your SSH client doesn't support PTY`.
- To connect on the first try, use the Connect-tab user/host and allocate a TTY:
  `ssh -tt <pod-user>@ssh.runpod.io -i ~/.ssh/id_ed25519`
- After connecting, run inspection commands inside the session instead of using one-shot `ssh ... 'command'`.
- Deploy repository changes to RunPod through a reviewed Git commit: push
  locally, then fetch/pull or check out that exact commit on the pod. Do not
  send repository files directly with `scp`, `rsync`, tar-over-SSH, terminal
  pastes, or similar mechanisms unless Git transfer is genuinely impossible
  or a non-repository artifact is strictly required. State the necessity
  before using an exception.

## Results Tracking
- Keep `docs/RESULTS_P2.md` up to date for active P2 metrics (`docs/RESULTS.md` indexes phase logs; `docs/RESULTS_P1.md` is archived P1; `docs/RESULTS_P0.md` is archived pre-P1).
- The P1 exact-simulator harness lives only on the `p1` git branch; do not reintroduce `p1a` / `p1b` / `p1c` CLI or agents on `main`.
- When a better metric is reported, update the "Best So Far" section with the new metric and exact command used.

## Candle execution profiler (P2)

Sibling crate: [`../candle_graph`](../candle_graph). **Guide:**
[`docs/CANDLE_GRAPH.md`](docs/CANDLE_GRAPH.md).

- Use `candle_graph` whenever applicable to training, evaluation, model inspection, numerical
  debugging, and performance investigation. It records executed, explicitly instrumented work;
  it cannot reconstruct unrecorded operations from Rust source. Prefer verified runtime bundles
  and focused queries when assessing what the model actually did.
- Train with **all supported profilers enabled**. Keep the `candle-graph` dependency's `all`
  feature (Candle helpers plus HTML viewer), and build Tofy with `profiling` plus the required
  device backend; CUDA/cuDNN training normally uses
  `cargo build --release --locked --features cudnn,profiling`. The dependency's `all` feature
  alone does not activate Tofy's NVTX/Chrome tracing or collect every evidence plane.
- Set `TOFY_PERF_TRACE` to a unique run-owned JSON path to enable Chrome/Perfetto host tracing;
  the `profiling` build also enables the matching NVTX ranges. For the legacy training loop,
  set `TOFY_P2_STEP_PROFILE` to a positive reporting interval (e.g. `100`). That interval controls
  reporting, not sampling: phase synchronization runs on every update. Foundation-v2 does not
  consume this variable; inspect its candle-graph phase spans instead. See
  [`src/perf.rs`](src/perf.rs) for the activation controls.
- Explicitly preregister representative capture updates within the planned training budget:
  legacy uses `--profile-update 2`; Foundation-v2 requires a nonempty `--profile-updates` list
  (e.g. `--profile-updates 2,100,1000` for a sufficiently long run). Foundation-v2's empty default
  captures no training updates, and `--profile-update` does not replace its list. Select reachable
  updates on short runs and resumes. Keep `--profile-eval true` for supported evaluators.
  Enable all available evidence on the selected invocations; candle-graph intentionally stays
  inactive on other invocations. Record capture cadence and profiler overhead in the run budget
  and match profiling settings across comparison arms.
- On NVIDIA training, also collect Nsight Systems evidence with supported CUDA, NVTX, OS-runtime,
  cuDNN/cuBLAS tracing and CPU sampling. Retain the `.nsys-rep`, supported official
  `nsys stats --format csv` reports, and a matching `capture-manifest.json`. Bind them to the
  corresponding trace using `CaptureRun::with_nsight_dir` before publication, or publish a new
  separate bundle with `cargo candle-graph report TRACE --nsight-dir DIR --bundle NEW_BUNDLE`.
  Tofy's current trainer does not attach Nsight automatically; never add files to a finalized
  bundle. Follow the sibling [runtime guide](../candle_graph/docs/runtime-analysis-guide.md)
  and [CLI reference](../candle_graph/docs/cli-reference.md) for the evidence contract.
- Before a long run, verify an actual capture from the exact launch binary, its enabled
  profilers, and its recorded capabilities. Enable every wired evidence plane and disclose
  unavailable tools, permissions, unsupported backends, and unwired probes before launch;
  do not silently disable profilers for speed or describe partial coverage as full profiling.
  Current Tofy captures include semantic spans, labelled tensor/seam statistics, host scalars,
  and manifest-checked training gradients, but do not record operations/activation hotspots,
  logical allocation lifetimes, physical-memory checkpoints, or device-event intervals.
  Those need producer instrumentation; enabling Cargo features cannot supply them. Host span
  durations are not CUDA kernel durations, and tensor footprints are not peak VRAM measurements.

```bash
cargo candle-graph protocol
cargo candle-graph campaign-status --manifest runs/p2/example/profile/campaign.json
cargo candle-graph overview runs/p2/example/profile/update-000000000002
cargo candle-graph query runs/p2/example/profile/update-000000000002 --kind labels
cargo candle-graph query runs/p2/example/profile/update-000000000002 \
  --kind tensor-stats --label-prefix loss/ --limit 20
cargo p2-view runs/p2/example/profile/update-000000000002 \
  --output /tmp/tofy-update-2-viewer.html
```

Start with `overview`; check `health.structurally_valid`, `health.capture_complete`, capabilities,
and gaps. Discover labels before querying; collection queries return 50 rows by default. Follow
`result.next_offset` with `--offset`, or deliberately export `--all --output FILE`. Keep finalized
bundles immutable: regenerate viewers outside the bundle. The Measurements tab exposes scalar
values, seam statistics, and gradient states with declared family expectations.

Current Tofy captures declare `profiled_work`, so timing comparisons are ineligible by design.
`compare` can still provide diagnostic numerical comparisons; a timing verdict requires compatible
production-equivalent captures and at least five independent bundles per cohort. Missing Nsight,
operation, and memory-lifetime evidence are stated gaps, not zero cost or a training error.

`.cargo/config.toml` aliases: `candle-graph`, `p2-view`.
