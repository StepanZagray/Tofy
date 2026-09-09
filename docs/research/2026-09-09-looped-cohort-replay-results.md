# Cohort replay did not repair spatial learning

The registered C7 replay screen completed all 1,150 updates and fails every
continuation gate. On the reused 64-query development panel, factual policy is
**15/64 (23.4375%)**, identical to C5, versus **18/64 (28.125%)** best constant.
All 512 factual and cleared successor predictions are the same goal-completion
template. This rejects this ordering recipe for scaling; it does not prove that
other optimization changes or the looped architecture cannot work.

## What changed

C5 placed four distinct queries, each repeated 16 times, in an effective batch of
64. C7 uses 64 distinct queries per update and revisits each query in 16 different
updates. The full 73,600 input/target/training-depth tuples are unchanged. All
4,600 queries retain their original depth, and the global 1/2/4 loop schedule is
unchanged. The actual training stream exactly matches the prior CPU audit and
the complete tuple multiset matches the retained C5 artifacts. Grouping and repeat
spacing change jointly; their separate effects are not identified.

Initialization, 992,393 parameters, losses, optimizer, clipping, sample budget
and physical 33/tail 31/accumulation 2/effective 64 are unchanged. No model,
objective, LLM or ARC-data intervention occurred. Final checkpoint 1,150 was
selected in advance; intermediate checkpoints cannot establish a better result.

## Registered readout

| Metric | C5 final | C7 replay final | Control |
|---|---:|---:|---:|
| Factual policy | 15/64 | 15/64 | Best constant 18/64 |
| Cleared policy | 12/64 | 15/64 | Best constant 18/64 |
| Factual policy CE | 1.426638 | 1.397652 | Uniform 1.386294 |
| Exact factual successor | 64/256 | 64/256 | Copy 72/256 |
| Exact blocked successor | 0/72 | 0/72 | Copy 72/72 |
| Exact nonterminal successor | 0/120 | 0/120 | Copy 0/120 |
| Exact terminal successor | 64/64 | 64/64 | Goal template 64/64 |
| Predictions equal goal template | 512/512 | 512/512 | Both conditions |

Replay always selects action 0 in both conditions. All 64 factual decisions match
C5. The paired factual accuracy gain is **0 percentage points**, bootstrap
interval **[0, 0]**. This degenerate empirical interval reflects identical tested
decisions, not a population proof of equivalence. Factual accuracy interval is
[14.0625%, 34.375%]; advantage over a resampled best constant is -4.6875 points,
interval [-25, 0]. Paired CE change is -0.028986, interval [-0.090372, 0.029548].

Intervals use 10,000 paired whole-query NumPy PCG64 resamples, seed 1912. All
actions and both conditions stay clustered and the best constant is reselected
in each draw. The panel already influenced intervention selection: this is
**selection-only evidence**, not fresh confirmation or method promotion.

Reward has zero predicted positives and zero recall at threshold 0.5; precision
is undefined, factual Brier 0.187628. Factual value MSE is 0.000015441 versus zero
for the constant prediction 1. All value targets equal 1, so a lower MSE cannot
establish long-horizon value. Final eight-query reference accuracy is 64/128,
exactly its constant-action baseline. Pixel accuracy 97.65625% comes entirely
from the goal template and must not be described as useful transition learning.

## Provenance and execution

- Campaign: `/home/stepan/Projects/code/.tofy-runs/looped-known-replay-20260909T110206-IST`.
- Source: `8cec1006b49e3c118ec9979f4f5e393e5cd7dc29`.
- Candle Graph dependency: `1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a`.
- Binary SHA256: `e68e27c44bf332bc5b24e904e8f33d464f80a6a662a34fba91d90b3b250df914`.
- Initial weights: `4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802`.
- Final weights: `a983626a888279cbb31651c6abcd1ebf4f808c8927a29a8349d03ffc35bd4e5a`.
- C5 comparator final: `e379802fa8e75327061f3dcc5ae95697f83e2d2d137a69ef68daf3f902040d41`.
- Training manifest: `14f34e34a12db5ae7ddfbfd6c89ce2a34ac515d43fb07a5d89f7d5ccedb131ff`.
- Final evaluation manifest: `f6bbbd6deb956a12d342f997326d9d12bedfc034a7c7d2d85e595a45124e6b38`.

The clean reviewed commit was pushed before launch. Build:

```bash
cargo build --release --locked --offline \
  --features cudnn,profiling,serde_json/float_roundtrip \
  --example looped_agent_probe
```

Exact model training arguments, using the separately hashed campaign binary:

```bash
./looped_agent_probe --known-mapping --known-replay --seed 0 \
  --mode coverage --coverage fresh --loops 4 --device cuda:0 \
  --batch 33 --effective-batch 64 --max-seconds 2100 --updates 1150 \
  --eval-episodes 64 --data-seed 9173 \
  --profile-updates 2,100,1000 --profile-eval true \
  --output-dir replay-seed0
```

Exact final frozen model arguments:

```bash
./looped_agent_probe --known-mapping --seed 0 --mode known-mapping \
  --loops 4 --device cuda:0 --batch 1 --effective-batch 1 \
  --max-seconds 300 --updates 1 --eval-episodes 64 --data-seed 20260912 \
  --checkpoint replay-seed0/final.safetensors --profile-eval true \
  --output-dir final-frozen
```

The frozen mode performs **zero optimizer updates**; `--updates 1` satisfies the
generic CLI guard. Retained metadata contains absolute executable/output paths
and exact argument order. Both invocations ran through the campaign supervisor
with unique host traces and Nsight CUDA/NVTX/cuDNN/cuBLAS/OS-runtime/CPU sampling.

The full pipeline ran from **11:17 to 11:44 IST on September 9, 2026**. Training
supervision elapsed 1,631.943 seconds; final frozen evaluation 5.468 seconds.
On the RTX 5060 Laptop GPU, sampled whole-device peak was 7,584/8,151 MiB,
minimum reserve 567 MiB, maximum temperature 69°C. Timing is operational context,
not a production-speed comparison. All 24 Rust tests, 17 analyzer fixtures,
legacy raw parity, stream integrity and eight Nsight-bound captures pass.
The three-update smoke explicitly exercised loops 1/2/4 before full training.

Operation/activation linkage, allocation lifetimes, physical-memory checkpoints,
device-event intervals and complete automatic NVTX correlation remain unavailable.
Host spans are not CUDA kernel durations; sampled whole-GPU usage is not an
allocator peak measurement. These are declared evidence gaps, not zero costs.

An independent reader reproduced the raw counts and bootstrap without importing
the analysis helpers. All 61 recorded process IDs exited before sealing. The
completed campaign contains 462 files and 651,026,399 bytes, fully rehashed against
the externally retained manifest SHA256
`35b575573ba94bd2ba67a16cf86758d85560613c91892a536036b2306d897070`.
This is point-in-time integrity, not immutable storage.

## Decision

Do not scale this ordering recipe. The next registered diagnostic exports frozen
CLS and current-patch features, then tests prespecified CPU linear readouts with
geometry and permuted-label controls on separate synthetic fitting/evaluation
panels. A positive probe supports recoverability under its representation; a
negative probe cannot establish that information is absent. Task-specific patch
selection is a diagnostic, not a general controller.

The shared-depth recurrent core still lacks learned episode memory, active
probing and ARC integration. There is no new Best So Far metric or ARC score.
The broad autonomous model-improvement task remains active.

[Registration](2026-09-09-looped-cohort-replay.md) ·
[Complete frozen analysis](/home/stepan/Research/_runs/2026-09-09T094928Z-tofy-looped-cohort-replay/completed-analysis.json)
