# candle-graph evidence in Tofy P2

Legacy training targets one configurable, one-based representative optimizer update
(`--profile-update`, default `2`, after one warm-up update). Foundation-v2 accepts a unique,
one-based `profile_updates` list and publishes one independent bundle for every selected optimizer
update; instrumentation is inactive on all other updates. A resume fails closed when a selected
update is already complete but its publication is absent from trainer state.

## Artifact bundle

Every completed or paused run publishes `OUTPUT/evidence_manifest.json`
(`tofy/p2/evidence/1`) as the run-level entry point. It records comparison invariants and treatment,
terminal state, source/binary provenance, a digest of ordered gradient-pressure samples, and
SHA-256/byte-length bindings for the exported model, exact resume checkpoint, config, report, and
immutable representative trace. Run-owned paths are relative, so the bundle remains relocatable.
The manifest excludes itself and excludes derived profile reports and optional Nsight output,
because the Nsight wrapper can regenerate those after training; they remain reproducible from the
bound trace and explicitly listed profile paths in `train_report.json`.

The default update 2 profile publishes atomically under:

```text
OUTPUT/profile/update-000000000002/
├── application.jsonl   # candle-graph/trace/10
├── evidence.json       # candle-graph/evidence/4
├── EVIDENCE.md         # bounded repair/research handoff
├── viewer.html         # Evidence + Trace + Span costs + Memory + GPU
└── nsight/             # optional raw .nsys-rep, CSV reports, status
```

`train_report.json` and resumable trainer state carry the legacy structured `profile` status plus
Foundation-v2's ordered published-bundle list. Every published bundle forces a durable checkpoint.
Its trace contains ordinary tensor metadata and numerical `tensor_stats` for each Foundation-v2
loss scalar and the `out_y`, current/predicted canonical, and copy-gate-logit mechanism seams.

The root measured region is device-synchronized once before and once after the complete update.
Generation, staging, forward, backward, gradient inspection, optimizer, and metrics retain typed
semantic spans; the trace therefore uses `timing_mode=host` and records
`measured_region_device_synchronized=true`, rather than overstating every nested duration as
synchronized. This flag is derived from the resolved Candle device, not from its display label.
Nsight supplies kernel durations. Gradient norms use one batched device read. Batch frames, loss storage, and every
parameter gradient are captured. Tensor metadata is not misrepresented as allocation lifetime.

## Agent workflow

Start with the run manifest, then the bounded packet—not raw JSONL:

```bash
sed -n '1,240p' runs/p2/example/evidence_manifest.json
sed -n '1,220p' runs/p2/example/profile/update-000000000002/EVIDENCE.md
cargo candle-graph summary runs/p2/example/profile/update-000000000002/application.jsonl
cargo candle-graph query runs/p2/example/profile/update-000000000002/application.jsonl --kind gradients
cargo candle-graph query runs/p2/example/profile/update-000000000002/application.jsonl --kind tensors
```

Compare an explicit baseline:

```bash
cargo candle-graph compare \
  runs/p2/baseline/profile/update-000000000002/application.jsonl \
  runs/p2/candidate/profile/update-000000000002/application.jsonl
```

## Human workflow

Open the already-published `viewer.html`, or regenerate with a baseline/Nsight directory:

```bash
cargo p2-view runs/p2/example/profile/update-000000000002 \
  --baseline runs/p2/baseline/profile/update-000000000002/application.jsonl \
  --output runs/p2/example/profile/update-000000000002/viewer.html
```

## Optional Nsight capture

Use the wrapper; normal training succeeds when Nsight is absent or fails in `auto` mode:

```bash
P2_NSYS=auto P2_PROFILE_UPDATE=2 scripts/p2_profile_nsys.sh runs/p2/example -- \
  cargo run --release --features cudnn,profiling -- p2-train \
  --device cuda --output-dir runs/p2/example --profile-update 2 ...
```

The wrapper retains `.nsys-rep`, exports official CSV reports, and regenerates the same
`evidence.json`, `EVIDENCE.md`, and `viewer.html`. Candle and NVTX use exact labels such as
`tofy.p2/update-000000000002/forward`, allowing `nvtx_gpu_proj_trace` to connect semantic phases to
GPU work. Global kernel/runtime summaries remain explicitly global.

`P2_NSYS=off|auto|require`: `auto` never changes the training command's exit status and reruns
normally if the profiler ends without a child result; `require` fails when Nsight evidence cannot
be produced. A previously published representative bundle bypasses capture on resume. Nsight
augmentation is assembled beside the published bundle and exchanged atomically only after JSON,
Markdown, and HTML all succeed.

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
