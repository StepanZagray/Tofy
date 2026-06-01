# OOM Testing

OOM behavior is not a compiler property and not a normal unit-test property. It depends on the GPU, driver, CUDA allocator, dtype, exact batch shape, sequence length, and how long the training loop runs.

Use short manual `cargo run --release -- ...` stages when you need a quick correctness smoke. Use `cargo run --release -- --sustained-oom-probe ...` for VRAM and batch decisions.

## Trusted Probe

Run the sustained probe from the repo root:

```bash
cargo run --release -- --sustained-oom-probe --stage all
```

This runs the release binary with the current 8 GB proof-of-concept shapes:

- latent: `16x16`, `seq_len=256`, `dim=640`, `layers=7`
- world: warmup `32x1`, then `32x8`, `seq_len=256`, `bridge_dim=640`, `context_slots=64`
- code decoder: `8x16`, `max_seq=192`, `dim=640`, `ff_dim=3072`, `max_vocab=24000`

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

## Short Max-VRAM Probe

Before renting a GPU for a long run, use the short max-VRAM probe:

```bash
cargo run --release -- --max-vram-probe --profile 80gb --stage all
# or, for the L40S/A40 profile:
cargo run --release -- --max-vram-probe --profile 48gb --stage all
```

This uses `config/model_profiles.json`, runs latent, world, high-world, and
decoder probes. Each measured stage is capped at 200 steps because VRAM peaks
stabilized early in recent profile probes. It forces warmup transitions after
10 steps and uses frequent world/high-world/decoder logging so
validation/checkpoint paths are exercised during the short run.

The output includes `peak_used_mb`, `min_free_mb`, per-stage
`*.vram_samples.jsonl`, and `summary.json` in the probe directory. The summary
also includes rough full-run VRAM estimates by multiplying measured peaks by
historical stage growth factors from saved runs. It is still an empirical check
rather than a formal guarantee: allocator behavior, driver version, background
processes, and longer-run checkpoint cadence can still change the exact peak.

## 80 GB Default Profile Probe

Run the sustained probe manually before launching the long 80 GB pipeline:

```bash
cargo run --release -- --sustained-oom-probe --profile 80gb --stage all
```

The probe loads the current `80gb` shape from `config/model_profiles.json`.
That profile uses encoder `32x16` (`512` effective), frozen-encoder world
`96x6` (`576` effective), decoder `16x16` (`256` effective), and Go feedback
`16x16` (`256` effective). Decoder effective batch stays at `256`, but the
microbatch stays small because the decoder is much wider.
Prefer `--max-vram-probe --profile 80gb` for the first rental check, then run
the sustained probe only after the short probe passes.

## 48 GB L40S/A40 Profile Probe

Run the same flow with `--profile 48gb` on 48 GB pods. That profile uses
encoder `48x11` (`528` effective), frozen-encoder world `128x4` (`512`
effective), decoder `32x8` (`256` effective), and Go feedback `32x8` (`256`
effective).

## High-Level World Stage

HWM is part of the standard training pipeline. The profile defaults train the
high-world stage after the low-level world model: `12000` steps for `8gb` and
`2000` for `48gb`. It is not toggled on separately:
the model uses the high-world checkpoint automatically when the profile run
creates it.

The macro-action state transition reuses frozen encoder/context compressor and trains only
the macro-action encoder plus high-level transition, so it should be smaller
than decoder training but still depends on `WORLD_BATCH`, `WORLD_GRAD_ACCUM`,
and the context segment-batch settings.

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
