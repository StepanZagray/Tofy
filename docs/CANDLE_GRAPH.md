# candle-graph integration (Tofy P2)

Tofy links the sibling [`../candle_graph`](../candle_graph) crate and emits a **step-0
execution trace** on every `p2-train` run.

## What training writes

| File | When | Schema |
| --- | --- | --- |
| `profile.jsonl` | first optimizer update (global step 0) | `candle-graph/trace/4` |
| `train_report.json` | train end / pause | `p2.train_report.v3` — field `profile_trace` |

The trace includes nested spans for `generate`, `stage`, `forward`, `backward`, plus
parameter gradient facts under VarBuilder root `vb`. There is **no per-step overhead**
after step 0.

Resumed runs skip re-emission when `profile_emitted` is already set in the checkpoint.

## View the graph

From the Tofy repo root:

```bash
cargo p2-view runs/p2/v15/profile.jsonl --output runs/p2/v15/model.html
```

Or via the sibling CLI:

```bash
cargo candle-graph view runs/p2/v15/profile.jsonl --output runs/p2/v15/model.html
cargo candle-graph summary runs/p2/v15/profile.jsonl
cargo candle-graph query runs/p2/v15/profile.jsonl --kind slowest
```

## Cargo aliases (`.cargo/config.toml`)

| Alias | Command |
| --- | --- |
| `cargo candle-graph …` | Sibling `cargo-candle-graph` binary (`--features all`) |
| `cargo p2-view …` | Tofy wrapper around `candle-graph view` |

## Other profiling (not candle-graph)

| Tool | Enable | Use |
| --- | --- | --- |
| Phase ms | `TOFY_P2_STEP_PROFILE=N` | Per-step generate/forward/backward breakdown |
| Chrome trace | `TOFY_PERF_TRACE=path` + `--features profiling` | Perfetto timeline |
| GPU | `nsys profile …` | CUDA kernels and stalls |

See [`src/perf.rs`](../src/perf.rs) and [`docs/P2.md`](P2.md).

## Legacy removed (v0.4 candle-graph)

- Static Rust analysis (`cargo candle-graph check/view --path .`)
- `runtime.json` (`candle-graph/runtime/2`)
- `cargo p2-audit` / analyzer feature / `model-ir.json` bundles

Old checkpoints may still carry a `runtime_trace` blob in `trainer_state.json`; resume
treats that as `profile_emitted = true` without rewriting `profile.jsonl`.

## Docs

| Doc | Contents |
| --- | --- |
| [`../candle_graph/README.md`](../candle_graph/README.md) | Trace protocol + CLI |
| [`../candle_graph/docs/runtime-analysis-guide.md`](../candle_graph/docs/runtime-analysis-guide.md) | Probe design |
