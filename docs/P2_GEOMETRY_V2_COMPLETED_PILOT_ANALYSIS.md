# P2 geometry-isolation completed pilot analysis

Date analyzed: 2026-08-09

Pod: `k8mkqgocvx1gqa`

Experiment: `runs/p2/ab-sigreg-geometry-v2`

Status: **completed seed-1 pilot; terminal branch A, `stop_after_pilot`; no arm promoted**

## Executive conclusion

The recovered SIGReg geometry-isolation pilot completed cleanly and produced a valid
preregistered decision. Both seed-1 arms trained from fresh initialization through
updates 1,000 and 2,000, all eight train/evaluation phases exited zero, every artifact
manifest verified, and a local replay of the unchanged gate reproduced terminal
branch A. Seeds 2 and 3 and the 4,000-update extension were correctly not run.

Neither geometry solved representation collapse. The control learned measurable
action dependence by update 2,000, but its effective-rank fraction remained only
`0.02529` versus the `0.10` floor and its dynamics horizon-8 MSE deteriorated from
`0.1094` to `1.8082`. The `pre-rms-spatial` treatment drove absolute one-step MSE
down to `0.0006168`, but this is not a positive result: downstream variance and
effective rank failed, shuffled-action ratios stayed at approximately `1.00`, and
the learned changed-transition MSE remained `7.43x` copy-forward. Its low raw error
is therefore dominated by a low-scale, action-marginalized representation.

The geometry treatment was operationally cheap and stable. Its synchronized
candle-graph update was only `0.85%` slower than control, full training throughput
was within `0.7%`, and all batch-1,024 CUDA evaluations succeeded in isolation.
This closes the prior CUDA failure as an operational recovery success, but the
scientific hypothesis failed its frozen gate.

## Provenance and local preservation

| Field | Verified value |
|---|---|
| Tofy commit | `17fbfdffba917bcef23f5bdea86aa83561524272` |
| `candle_graph` commit | `c9fa15ee917f9dab96bb070cbbcd5ce8dfdb5f48` |
| Release binary SHA-256 | `6540482ef354143e4d959ec0ef13d1febf3086c434886d10282fa7e399f5cb65` |
| Shared comparison-contract SHA-256 | `b9c7f2e5cd509977dbfdd6d29445539cab9050cca03239591cc546e3ece62c92` |
| Control config SHA-256 | `a089d993ea1ab17c08e83b964f7e4945f2d0e923971e55f2ba7667b1a3b8ca4c` |
| Treatment config SHA-256 | `0757964361b53d96523d707c92d6e92b9ef975db58ed6e7bc82bd8f9e121c715` |
| GPU | NVIDIA A40, 46,068 MiB |
| Physical / accumulated batch | `1024 / 1` |
| Remote result files | 121 |
| Remote file-manifest SHA-256 | `8639b10c37a5116111c9b32cbfbf33882d279eb6e70d9161b41ca975f6e4b7a4` |
| Transfer archive SHA-256 | `08a47f004f62da65dcd7e86d7a6817a1e1440006081788ac37b3d4f0cf4afc47` |
| Supervisor exit | `0` |

The pod-specific handoff is preserved under
`runs/p2/_pod_handoffs/k8mkqgocvx1gqa-20260809/`. Its archive checksum verifies,
and all 121 result files verify against the manifest in
`tofy-results-k8mkqgocvx1gqa-20260809/FILES.sha256`. The extracted artifacts,
supervisor log/status, transfer archive, provenance, and local verification transcript
are retained together.

The gate was rerun locally against the copied artifacts. After normalizing only the
pod/local root strings, its complete JSON result matched the copied gate result and
again selected `stop_after_pilot`.

Two requested independent Cursor reviews (Composer 2.5 and GPT-5.6 Luna High) were
attempted twice and rejected before execution because the external-agent account was
out of usage. No independent review is claimed. The conclusions below were checked
directly against the manifests, gate implementation, reports, phase logs, telemetry,
and profiler packets.

## Frozen experiment contract

Both arms used seed 1, dynamics-only training, physical batch 1,024, accumulation 1,
2,000 updates, shuffled episodes, randomized recursion to 8 outer by 2 inner steps,
final-outer-only supervision, a 32,768-row SIGReg cap, and the same 413,798-parameter
model. Held-out evaluation used seed 424242, 64 synthetic episodes, deterministic
PTRM `K=1`, no PTRM noise, and one ensemble member at updates 1,000 and 2,000.

- `control`: post-RMS, pooled 2x2 spatial cells.
- `pre-rms-spatial`: pre-RMS, unpooled 8x8 spatial cells, deterministically
  subsampled to the same row cap.

All reports have `research_claim=false` and use no public data for fitting. This is a
synthetic representation diagnostic, not a public ARC score or research claim.

## Exact outcome

| Arm | Update | Aggregate action ratio [95% CI] | Random-one-step ratio [95% CI] | Changed improvement [95% CI] | Variance | Rank fraction | Dynamics H8 | Hard pass |
|---|---:|---|---|---|---:|---:|---:|---|
| control | 1,000 | 1.0402 [1.0225, 1.0620] | 1.1547 [1.0792, 1.2378] | 0.6665 [0.6187, 0.7054] | 0.002772 | 0.01383 | 0.1094 | no |
| control | 2,000 | 1.2095 [1.1567, 1.2695] | 1.7365 [1.5164, 1.9980] | 0.4661 [0.4133, 0.5153] | 0.006128 | 0.02529 | 1.8082 | no |
| pre-RMS spatial | 1,000 | 1.0000 [0.9997, 1.0003] | 0.9999 [0.9994, 1.0005] | -49.2690 [-52.4334, -45.9926] | 2.18e-6 | 0.01128 | 0.00465 | no |
| pre-RMS spatial | 2,000 | 1.0004 [0.9992, 1.0018] | 1.0016 [0.9973, 1.0064] | -6.4256 [-6.6567, -6.1754] | 1.95e-5 | 0.00839 | 0.03124 | no |

All four reports were artifact-valid and had valid, non-near-bound SIGReg statistics.
The hard gate failed as follows:

- Control passed both action gates and changed-transition improvement at update 2,000,
  but failed noncollapse because effective-rank fraction was `0.02529 < 0.10`.
- Treatment failed both action gates, changed-transition improvement, and noncollapse
  at both updates. At update 2,000 its variance was `5.14x` below the `1e-4` floor
  and its rank fraction was `11.9x` below the `0.10` floor.
- Neither arm met the credible monotonic-approach rule. Control's changed-transition
  improvement decreased from `0.6665` to `0.4661`; treatment never reached the
  required action or noncollapse thresholds.
- Treatment did not materially improve control on the three required gate metrics.

The preregistered gate therefore selected branch A and stopped after seed 1.

## Interpretation

### Control: action conditioning emerged, but the representation and rollout degraded

From update 1,000 to 2,000, control aggregate action ratio improved from `1.0402` to
`1.2095` and random-one-step ratio from `1.1547` to `1.7365`. These confidence
intervals exclude 1.0 at update 2,000, so the effect is real within this held-out
sample. However:

- one-step dynamics MSE worsened `0.02394 -> 0.11371` (`4.75x`);
- planner one-step MSE worsened `0.02910 -> 0.14898` (`5.12x`);
- dynamics H8 worsened `0.1094 -> 1.8082` (`16.5x`);
- normalized dynamics H8 worsened `1.558 -> 9.611`;
- changed-transition improvement declined by 0.2004; and
- rank fraction improved but remained far below the noncollapse floor.

Action sensitivity, noncollapse, and rollout stability are therefore separate axes.
The control's action-shuffle success does not rescue its representation or rollout.

### Treatment: tiny absolute MSE is degenerate

Treatment one-step dynamics MSE fell `0.001307 -> 0.0006168`, and at update 2,000 it
was `184x` lower than control. That number cannot be read as a better world model:

- shuffled and true actions produced indistinguishable error (`~1.00` ratios);
- learned changed-transition MSE was `7.43x` copy-forward at update 2,000;
- global pooled variance and rank both failed;
- dynamics H8 increased `6.72x` from update 1,000 to 2,000;
- normalized dynamics H8 remained about `6`; and
- normalized planner H8 was `113.1`, over `18x` worse than control at update 2,000.

The treatment made the latent target easier primarily by shrinking/degenerating the
representation. The gate's noncollapse and intervention metrics correctly prevented
this raw-MSE improvement from being promoted.

Because the treatment changes both normalization placement and pooling relative to
control, this pilot rejects the combined construction but does not identify which of
those two changes caused the failure.

## Runtime, GPU, and profiler evidence

| Arm | Train 0-1k | Eval 1k | Train 1k-2k | Eval 2k | Total |
|---|---:|---:|---:|---:|---:|
| control | 4,521s | 978s | 4,512s | 1,014s | 11,025s |
| pre-RMS spatial | 4,543s | 974s | 4,550s | 970s | 11,037s |

Treatment was only 12 seconds (`0.11%`) slower end-to-end. Training throughput was
`13.28` updates/min for control and `13.20` updates/min for treatment. The run used
the largest verified physical batch, 1,024, with no gradient accumulation.

| Arm/window | Mean sampled GPU util. | Zero-util. samples | Mean / peak memory |
|---|---:|---:|---:|
| control train | 90.46% | 8.07% | 14,433 / 14,866 MiB |
| treatment train | 93.06% | 6.30% | 14,544 / 14,994 MiB |
| control eval | 3.40% | 68.18% | 1,806 / 17,127 MiB |
| treatment eval | 4.16% | 68.75% | 1,777 / 17,223 MiB |

Training used the A40 effectively. Evaluation alternated CPU-heavy synthetic sample
generation and summaries with short saturated CUDA bursts, explaining the low sampled
average. Before another pod campaign, persisting the frozen synthetic sample set and
batching/pipelining rollout evaluation are the clearest utilization improvements.
They require a newly frozen evaluation contract.

Both candle-graph packets are `TRUSTED`, captured at update 2 after one warmup update,
with synchronized host totals of `4720.71 ms` (control) and `4760.75 ms` (treatment),
a treatment delta of `+0.85%`. Both contain complete semantic span coverage for the
measured root and 28 gradient facts. Neither contains timed operation evidence,
allocation/device-memory events, or Nsight data, so the packets do not support
kernel-level or VRAM attribution.

## Recommended next work

Do not run seeds 2/3, extend these arms to 4,000, or restart the full curriculum.
The frozen stop decision is conclusive for this combined geometry treatment.

Before another expensive run:

1. Add layer-aligned evaluation for the exact populations involved: pre-RMS cells,
   post-RMS cells, pooled downstream latent, and predicted next latent. Report
   variance, effective rank, and action intervention at every seam.
2. Add scale-normalized/fixed-anchor prediction diagnostics so a shrinking learned
   target cannot look good solely through raw latent MSE.
3. Isolate the two geometry changes instead of changing both at once: compare
   post-RMS unpooled cells and pre-RMS pooled cells in a short factorial mechanism
   test with early checkpoints before committing to 2,000 updates.
4. Evaluate at shorter intervals (for example 250/500/750/1,000) because control
   action dependence increased while rollout quality deteriorated sharply.
5. Optimize the evaluator locally, then preregister any changed evaluation contract.

No model-quality metric from this pilot replaces the existing P2 "Best So Far"
entries. The useful result is a validated negative mechanism test and a fully
recovered unattended execution path.

## Pod shutdown

The experiment ended with supervisor exit 0, no trainer/evaluator/tmux process
remaining, and no GPU allocation. The complete result tree, supervisor evidence,
archive, and checksums are verified locally. The pod can be stopped without losing
any artifact required for this analysis.
