# OOM Testing

OOM behavior is not a compiler property and not a normal unit-test property. It depends on the GPU, driver, CUDA allocator, dtype, exact batch shape, sequence length, and how long the training loop runs.

Use `runtime_smoke_tests.sh` for correctness smoke. Use `sustained_oom_probe.py` for VRAM/batch decisions.

## Trusted Probe

Run the sustained probe from the repo root:

```bash
./scripts/sustained_oom_probe.py --stage all
```

This runs the release binary with the current 8 GB proof-of-concept shapes:

- latent: `32x1`, `seq_len=256`, `dim=640`, `layers=7`
- world: warmup `64x1`, then `64x2`, `seq_len=256`, `bridge_dim=640`, `planner_slots=64`
- code decoder: `12x2`, `max_seq=224`, `dim=640`

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
./scripts/sustained_oom_probe.py --stage world
```

Decoder only:

```bash
./scripts/sustained_oom_probe.py --stage decoder
```

Latent only:

```bash
./scripts/sustained_oom_probe.py --stage latent
```

To validate the probe script itself without doing a real memory decision:

```bash
./scripts/sustained_oom_probe.py --stage all --quick
```

Treat `--quick` as a script smoke test only.

## Testing Candidate Batches

Keep effective batch fixed while changing microbatch/accumulation. For example, to test whether decoder `24x1` is safe:

```bash
./scripts/sustained_oom_probe.py \
  --stage decoder \
  --decoder-batch 24 \
  --decoder-accum 1
```

To test the current safer decoder default:

```bash
./scripts/sustained_oom_probe.py \
  --stage decoder \
  --decoder-batch 12 \
  --decoder-accum 2
```

To test world without warmup:

```bash
./scripts/sustained_oom_probe.py \
  --stage world \
  --world-warmup-steps 0
```

## Interpreting Results

A passing sustained probe means the tested shape is operationally safe under the sampled conditions. It does not prove training quality.

A failing probe with `oom=false` can still be a real failure if `min_free_mb` is too low or `late_growth_mb` is high. That means the shape is too close to the edge for long training, even if the short process exited successfully.

Do not use one-step fit probes to set defaults. They are useful only for checking that the code path starts.
