# OOM Testing

OOM behavior is not a compiler property and not a normal unit-test property. It depends on the GPU, driver, CUDA allocator, dtype, exact batch shape, sequence length, and how long the training loop runs.

Use short manual `cargo run --release -- ...` stages when you need a quick correctness smoke. Use `cargo run --release -- --sustained-oom-probe ...` for VRAM and batch decisions.

## Trusted Probe

Run the sustained probe from the repo root:

```bash
cargo run --release -- --sustained-oom-probe --stage all
```

This runs the release binary with the current 8 GB proof-of-concept shapes:

- latent: `12x2`, `seq_len=256`, `dim=640`, `layers=7`
- world: warmup `64x1`, then `64x2`, `seq_len=256`, `bridge_dim=640`, `planner_slots=64`
- code decoder: `6x4`, `max_seq=128`, `dim=640`

The probe samples `nvidia-smi` externally every `0.1s`, writes per-stage logs and `*.vram_samples.jsonl`, and restores `local_models/` afterward.

By default it fails if:

- the process exits non-zero
- CUDA reports out-of-memory
- minimum free VRAM drops below `512 MB`
- late-run VRAM growth exceeds `512 MB`

The late-growth check matters because a one-step probe previously passed world `96x1`, while a real run later climbed to `7643 MB` and OOMed around step `5900`.

## Stage-Specific Probes

World only:

```bash
cargo run --release -- --sustained-oom-probe --stage world
```

Decoder only:

```bash
cargo run --release -- --sustained-oom-probe --stage decoder
```

Latent only:

```bash
cargo run --release -- --sustained-oom-probe --stage latent
```

To validate the `--sustained-oom-probe` implementation itself without doing a real memory decision:

```bash
cargo run --release -- --sustained-oom-probe --stage all --quick
```

Treat `--quick` as a shallow sanity check only.

## 80 GB Cloud Profile Probe

Run the sustained probe manually before launching the long 80 GB pipeline:

```bash
cargo run --release -- --sustained-oom-probe --stage all --dim 2048 --layers 7 --heads 16 --bridge-dim 2048 --planner-slots 128 --decoder-max-vocab 32000
```

That profile uses `DIM=2048`, `BRIDGE_DIM=2048`, `LAYERS=7`, `HEADS=16`,
`NUM_LATENT_TOKENS=128`, encoder/world context `256`, and code-decoder context
`128`. The probe defaults to at least `4096 MB` free headroom and allows up to
`2048 MB` late-run growth; override with `TOFY_80GB_MIN_HEADROOM_MB` and
`TOFY_80GB_MAX_LATE_GROWTH_MB` if a specific cloud GPU needs different margins.

## 48 GB A40 Profile Probe

Run the sustained probe manually before launching the long 48 GB pipeline:

```bash
cargo run --release -- --sustained-oom-probe --stage all --dim 1536 --layers 7 --heads 12 --bridge-dim 1536 --planner-slots 96 --decoder-max-vocab 24000
```

That profile uses `DIM=1536`, `BRIDGE_DIM=1536`, `LAYERS=7`, `HEADS=12`,
`NUM_LATENT_TOKENS=96`, encoder/world context `256`, and code-decoder context
`128`. The probe defaults to at least `3072 MB` free headroom and allows up to
`1536 MB` late-run growth; override with `TOFY_48GB_MIN_HEADROOM_MB` and
`TOFY_48GB_MAX_LATE_GROWTH_MB` if needed.

## High-Level World Stage

HWM is part of the standard training pipeline. The profile defaults train the
high-world stage after the low-level world model: `12000` steps for `8gb`,
`36000` for `48gb`, and `120000` for `80gb`. It is not toggled on separately:
the model uses the high-world checkpoint automatically when the profile run
creates it.

The high-level world model reuses frozen encoder/planner memory and trains only
the macro-action encoder plus high-level transition, so it should be smaller
than decoder training but still depends on `WORLD_BATCH`, `WORLD_GRAD_ACCUM`,
and the planner segment-batch settings.

## Testing Candidate Batches

Keep effective batch fixed while changing microbatch/accumulation. For example, to test whether decoder `24x1` is safe:

```bash
cargo run --release -- --sustained-oom-probe \
  --stage decoder \
  --decoder-batch 24 \
  --decoder-accum 1
```

To test the current safer decoder default:

```bash
cargo run --release -- --sustained-oom-probe \
  --stage decoder \
  --decoder-batch 6 \
  --decoder-accum 4 \
  --decoder-max-seq 128
```

To test world without warmup:

```bash
cargo run --release -- --sustained-oom-probe \
  --stage world \
  --world-warmup-steps 0
```

## Interpreting Results

A passing sustained probe means the tested shape is operationally safe under the sampled conditions. It does not prove training quality.

A failing probe with `oom=false` can still be a real failure if `min_free_mb` is too low or `late_growth_mb` is high. That means the shape is too close to the edge for long training, even if the short process exited successfully.

Do not use one-step fit probes to set defaults. They are useful only for checking that the code path starts.
