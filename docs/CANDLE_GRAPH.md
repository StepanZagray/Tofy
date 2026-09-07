# candle-graph evidence in Tofy P2

Legacy training targets one configurable, one-based representative optimizer update
(`--profile-update`, default `2`, after one warm-up update). Foundation-v2 uses the unique,
one-based `--profile-updates` list (`profile_updates` in config) and publishes one independent
bundle for every selected optimizer update. Its default list is empty, so it captures no training
updates unless explicitly configured; `--profile-update` is not a fallback for Foundation-v2.
Select updates reachable within the run or resume budget. Candle-graph instrumentation is inactive
on all other updates. A resume reconciles complete bundles through candle-graph's deep bundle
verification and planned-capture identity. If publication beat
the trainer checkpoint, `CaptureRun::begin` returns `AlreadyPublished`; Tofy records the update and
continues. A completed selected update with no publishable bundle still fails closed.

## Enable the profilers

Follow the repository's [profiling policy](../AGENTS.md#candle-execution-profiler-p2). The
`candle-graph` dependency already enables `all`, which supplies Candle helpers and the HTML viewer;
it does not activate Tofy's other profilers or create missing instrumentation.

| Profiler | Activation | Scope |
| --- | --- | --- |
| Candle-graph | Legacy `--profile-update 2`; Foundation-v2 e.g. `--profile-updates 2,100,1000` when those updates are reachable | Selected invocations only |
| Chrome/Perfetto host trace | Build with `profiling`; set `TOFY_PERF_TRACE` to a unique run-owned JSON path | Emitted `tracing` spans; host evidence only |
| NVTX | Build with `profiling` | Semantic ranges matching candle-graph labels |
| Legacy phase timings | Positive `TOFY_P2_STEP_PROFILE`, e.g. `100` | Synchronizes phases every update and reports averages at that interval; unused by Foundation-v2 |
| Nsight Systems | External capture plus official CSV export | GPU evidence retained separately, then bound to the corresponding trace |

CUDA/cuDNN training normally builds with `cargo build --release --locked --features cudnn,profiling`.
Check profiler availability and an actual capture on the exact launch binary before a long run.
Record capture cadence and overhead; enabling all available profilers does not require a
candle-graph bundle for every update.

## Artifact bundle

Every completed or paused run publishes `OUTPUT/evidence_manifest.json`
(`tofy/p2/evidence/1`) as the run-level entry point. It records comparison invariants and treatment,
terminal state, source/binary provenance, a digest of ordered gradient-pressure samples, and
SHA-256/byte-length bindings for the exported model, exact resume checkpoint, config, report, and
immutable representative evidence. Run-owned paths are relative, so the bundle remains relocatable.
The manifest excludes itself; Foundation-v2 binds every file in each finalized candle-graph bundle,
while the legacy structured profile field retains its trace binding.

Foundation-v2 also appends one record per optimizer update to `OUTPUT/loss_log.jsonl`. Each line
contains the update number, every realized foundation-v2 loss scalar, pre-clip gradient norm,
gradient clip scale, and the WSD learning rate used for that update. The writer is buffered and
durably flushed with every checkpoint and when training exits.

At training start Tofy writes the preregistered plan as
`OUTPUT/profile/campaign.json` (`candle-graph/campaign/1`). When update 2 is selected, its capture
is published atomically under:

```text
OUTPUT/profile/
├── campaign.json       # candle-graph/campaign/1
└── update-000000000002/
    ├── bundle.json     # candle-graph/bundle/1 content manifest
    ├── trace.jsonl     # current trace schema; query `candle-graph protocol`
    ├── evidence.json   # current evidence schema; query `candle-graph protocol`
    ├── report.md       # bounded evidence report
    ├── viewer.html     # Overview + Execution graph + Timings + Measurements + Memory + GPU
    └── nsight/         # only in a bundle explicitly published with Nsight inputs
```

`train_report.json` and resumable trainer state carry the legacy structured `profile` status plus
Foundation-v2's ordered published-bundle list. Every published bundle forces a durable checkpoint;
on resume, `CaptureRun`/`reconcile_published_bundle` verifies bundle content and capture identity
before repairing bookkeeping. Tofy does not write evidence files, probe for a pair of expected
files, or rename profile directories itself. A caught profiled-step error calls `publish_failed`,
leaving a verified diagnostic bundle that `campaign-status` reports as `failed_run`.

The trace contains labelled tensor metadata and GPU-reduced `tensor_stats` only for the four
mechanism seams: `seam/out_y`, `seam/current_canonical`, `seam/predicted_canonical`, and
`seam/gate_logits`. Loss terms, pre-clip gradient norm, clip scale/flag, learning rate, current EP
weight, and gate-cadence copy-bypass alpha are recorded with `record_scalar` from values already on
the host. These scalar events share the tensor-statistics plane without launching reduction kernels
or adding readbacks. `GradientCapturePlan` binds a complete exact manifest to the recorded
`world`, `observers`, `exact_decoder`, and `auxiliary_decoders` families; the root states whether
the capture is `vb/pre_clip` or `vb/post_clip`.

CUDA captures report `measured_region_device_synchronized=true`, derived from the resolved Candle
device. Selected phases wrapped in `synchronized_phase` also synchronize before and after their
work; these include forward/loss, backward, gradient inspection/clipping, and optimizer phases.
The trace retains `timing_mode=host` because this does not establish that every nested span is
individually synchronized. Phase times include synchronization and probe overhead; they are not
individual kernel timings. Foundation-v2 closes the measured region before EMA updates, loss-log
writes, gate evaluation, bundle publication, and the profile-forced checkpoint.
Nsight supplies kernel durations when retained in the bundle. The capture contract declares the
instrumented update as `profiled_work`, labelled-subset tensor coverage, and complete gradient
coverage. Operations/activation hotspots, logical allocation lifetimes, physical-memory
checkpoints, and device intervals remain unavailable because Tofy does not record those planes.
Tensor metadata is not allocation-lifetime evidence.

## Post-training evaluation capture

`p2-eval` and `p2-arc3-eval` enable evaluation profiling by default. The local toolkit command
`p2-arc3-bridge` defaults it off; pass `--profile-eval true` in either bridge mode to enable it.
A full foundation-v2 `p2-eval` publishes the fixed unseen-seed V5 gate-support population pass
under the report output directory:

```text
REPORT_PARENT/profile/
├── eval-campaign.json
└── eval-000000000001/
```

The capture has phase `infer`, tag `phase=eval`, and spans for encode, forward, decode, and host
metric reduction. It records the changed/full/composed exactness variants, content and padding
false-edit rates, shuffled-action ratio, and foreground metrics already computed by the evaluator.

The ARC recording evaluator and toolkit bridge capture the first candidate-scoring forward for
each game when profiling is enabled and publish `profile/arc3-<game_id>/` bundles listed by
`profile/arc3-campaign.json`. The chosen action's score,
Q probability, reliability probability, no-op probability, and predicted effect are scalar events.
Later decisions retain the uninstrumented pacing path. Use an explicit `--recordings-dir` for
toolkit-schema replay recordings. Bridge `drive` requires `--output`; bridge `serve` uses its
output parent as the profiling anchor, or the current directory when no output was supplied.

Inspect the evaluation campaigns with:

```bash
cargo candle-graph campaign-status --manifest RUN/profile/eval-campaign.json
cargo candle-graph campaign-status --manifest RUN/profile/arc3-campaign.json
```

## Agent workflow

Bind to the installed protocol first, then inspect the run manifest and bounded bundle overview:

```bash
cargo candle-graph protocol
sed -n '1,240p' runs/p2/example/evidence_manifest.json
cargo candle-graph overview runs/p2/example/profile/update-000000000002
cargo candle-graph campaign-status \
  --manifest runs/p2/example/profile/campaign.json
cargo candle-graph query runs/p2/example/profile/update-000000000002 --kind labels
cargo candle-graph query runs/p2/example/profile/update-000000000002 \
  --kind gradients --label-prefix vb/pre_clip/ --limit 20
cargo candle-graph query runs/p2/example/profile/update-000000000002 \
  --kind tensor-stats --label-prefix loss/
cargo candle-graph series \
  --manifest runs/p2/example/profile/campaign.json --label-prefix loss/
```

Collection queries expose `total`, `matched`, `displayed`, `truncated`, and `next_offset`.
The default page is 50 rows; pass the returned `next_offset` as `--offset` to continue. Label
discovery returns `{kind, label, events}` rows, so repeated labels are visible before fetching
their observations. Use `--all --output FILE` only when a full export is needed.

Compare an explicit baseline for a diagnostic numerical readout:

```bash
cargo candle-graph compare \
  --baseline runs/p2/baseline/profile/update-000000000002 \
  --candidate runs/p2/candidate/profile/update-000000000002
```

These singleton cohorts cannot yield an eligible timing verdict. Current Tofy capture contracts
also declare `profiled_work`, so adding repeats alone does not make them timing-eligible. A timing
verdict requires at least five compatible independent bundles per cohort whose producer declares
production-equivalent coverage. `compare --require-eligible` fails on an ineligible result after
writing its typed reasons; numerical observations do not establish causal attribution.

## Human workflow

Open the already-published `viewer.html`, or regenerate from the verified bundle to a path outside
that immutable bundle:

```bash
cargo p2-view runs/p2/example/profile/update-000000000002 \
  --output /tmp/tofy-update-2-viewer.html
```

Use **Measurements** to inspect scalar values (losses, optimizer settings, evaluation metrics),
tensor statistics at the four mechanism seams, and gradients. Search any recorded label or family,
filter gradient states, and follow the recorded-span link into the graph. Family expectations
remain beside missing/zero gradients; an inactive family can legitimately have no attached
gradient. Non-finite statistics are marked explicitly instead of showing their serialized zero
placeholders as measurements. Previously published HTML must be regenerated to include this view.

## Optional Nsight capture

`CaptureRun::with_nsight_dir` is the supported publication seam for a flat directory of official
Nsight artifacts; it binds those files into `bundle.json` before atomic publication. Tofy's trainer
currently calls `publish()` without supplying this directory. Building with `profiling` and
running under `nsys` does not automatically add Nsight evidence to its training bundles.

After external Nsight capture and CSV export finish, retain the `.nsys-rep`, supported reports,
and matching `capture-manifest.json` in one flat directory, then publish a separate bundle:

```bash
cargo candle-graph verify RUN/profile/update-000000000002 --semantic
cargo candle-graph report RUN/profile/update-000000000002/trace.jsonl \
  --nsight-dir NSIGHT_DIR --bundle RUN/profile/update-000000000002-with-nsight
cargo candle-graph verify RUN/profile/update-000000000002-with-nsight --semantic
cargo candle-graph query RUN/profile/update-000000000002-with-nsight --kind gpu-status
cargo candle-graph query RUN/profile/update-000000000002-with-nsight --kind gpu-correlation
```

Use a new destination and preserve the original bundle and sealed run artifacts. If the run is
already sealed, publish outside its artifact tree. The existing campaign and evidence manifest
still reference the original capture; record the separate bundle's path and digest in analysis.
The sibling [publication guide](../../candle_graph/docs/runtime-analysis-guide.md#10-publish-through-capturerun)
describes this workflow. Do not mutate or regenerate files inside an already-published bundle.
Candle and NVTX use exact labels such as
`tofy.p2/update-000000000002/forward`, allowing `nvtx_gpu_proj_trace` to connect semantic phases to
GPU work. Global kernel/runtime summaries remain explicitly global.

## Full-update capacity gate

The capacity probe runs production hidden width, worst-case fixed recursion depth, spatial SIGReg,
PTRM ranking, and active auxiliary losses. Use accumulation locally while keeping effective batch
512; the L40S acceptance run must use physical 512 with accumulation 1:

```bash
# 8 GiB development GPU
TOFY_VRAM_PROBE=1 TOFY_VRAM_PHYSICAL_BATCH=64 TOFY_VRAM_GRAD_ACCUM=8 \
  cargo test --release --features cudnn --test p2_vram_probe -- --ignored --nocapture

# L40S 48 GiB acceptance gate
TOFY_VRAM_PROBE=1 TOFY_VRAM_PHYSICAL_BATCH=512 TOFY_VRAM_GRAD_ACCUM=1 \
  cargo test --release --features cudnn --test p2_vram_probe -- --ignored --nocapture
```

Run the same command with `TOFY_VRAM_LESSON=retarget` to cover the open-loop branch. Candle-graph
records tensor facts, not allocator high-water; record peak process VRAM from Nsight or
`nvidia-smi` beside the result.
