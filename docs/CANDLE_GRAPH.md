# candle-graph evidence in Tofy P2

Legacy training targets one configurable, one-based representative optimizer update
(`--profile-update`, default `2`, after one warm-up update). Foundation-v2 accepts a unique,
one-based `profile_updates` list and publishes one independent bundle for every selected optimizer
update; instrumentation is inactive on all other updates. A resume reconciles complete bundles
through candle-graph's deep bundle verification and planned-capture identity. If publication beat
the trainer checkpoint, `CaptureRun::begin` returns `AlreadyPublished`; Tofy records the update and
continues. A completed selected update with no publishable bundle still fails closed.

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
`OUTPUT/profile/campaign.json` (`candle-graph/campaign/1`). The default update 2 capture is then
published atomically under:

```text
OUTPUT/profile/
├── campaign.json       # candle-graph/campaign/1
└── update-000000000002/
    ├── bundle.json     # candle-graph/bundle/1 content manifest
    ├── trace.jsonl     # current trace schema; query `candle-graph protocol`
    ├── evidence.json   # candle-graph/evidence/4
    ├── report.md       # bounded evidence report
    ├── viewer.html     # Evidence + Trace + Span costs + Memory + GPU
    └── nsight/         # present only when official Nsight inputs were supplied
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

The root measured region is device-synchronized once before and once after the complete update.
Generation, staging, forward, backward, gradient inspection, optimizer, and metrics retain typed
semantic spans; the trace therefore uses `timing_mode=host` and records
`measured_region_device_synchronized=true`, rather than overstating every nested duration as
synchronized. This flag is derived from the resolved Candle device, not from its display label.
Nsight supplies kernel durations when retained in the bundle. The capture contract declares the
instrumented update as `profiled_work`, labelled-subset tensor coverage, and complete gradient
coverage. Operations, logical/physical allocation lifetime, and device intervals remain honestly
declared unavailable because Tofy does not record those planes. Tensor metadata is not misrepresented as
allocation lifetime.

## Post-training evaluation capture

`p2-eval`, `p2-arc3-eval`, and `p2-arc3-live-eval` enable evaluation profiling by default.
Pass `--profile-eval false` to disable it. A full foundation-v2 `p2-eval` publishes the fixed
unseen-seed V5 gate-support population pass under the report output directory:

```text
REPORT_PARENT/profile/
├── eval-campaign.json
└── eval-000000000001/
```

The capture has phase `infer`, tag `phase=eval`, and spans for encode, forward, decode, and host
metric reduction. It records the changed/full/composed exactness variants, content and padding
false-edit rates, shuffled-action ratio, and foreground metrics already computed by the evaluator.

Both ARC evaluators capture only the first candidate-scoring forward for each game and publish
`profile/arc3-<game_id>/` bundles listed by `profile/arc3-campaign.json`. The chosen action's score,
Q probability, reliability probability, no-op probability, and predicted effect are scalar events.
Later live decisions retain the uninstrumented pacing path. Live evaluation also writes replayable
toolkit-schema JSONL atomically under `--recordings-dir`; when omitted, that directory defaults to
`<report output parent>/recordings`.

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
cargo candle-graph query runs/p2/example/profile/update-000000000002 --kind gradients
cargo candle-graph query runs/p2/example/profile/update-000000000002 \
  --kind tensor-stats --label-prefix loss/
cargo candle-graph series \
  --manifest runs/p2/example/profile/campaign.json --label-prefix loss/
```

Compare an explicit baseline:

```bash
cargo candle-graph compare \
  --baseline runs/p2/baseline/profile/update-000000000002 \
  --candidate runs/p2/candidate/profile/update-000000000002
```

## Human workflow

Open the already-published `viewer.html`, or regenerate from the verified bundle to a path outside
that immutable bundle:

```bash
cargo p2-view runs/p2/example/profile/update-000000000002 \
  --output /tmp/tofy-update-2-viewer.html
```

## Optional Nsight capture

`CaptureRun::with_nsight_dir` is the supported publication seam for a flat directory of official
Nsight artifacts; it binds those files into `bundle.json` before atomic publication. Do not mutate
or regenerate files inside an already-published bundle. Candle and NVTX use exact labels such as
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
