# Tofy patches on candle-core 0.11.0

Wired from the repo root with:

```toml
[patch.crates-io]
candle-core = { path = "vendor/candle-core" }
```

Prefer `--features cudnn` only when measurements show it beats `--features cuda`.
Stock candle cudnn is a regression; with these patches cudnn wins on the P2 hot path.

## 1. cuDNN Conv2D backward

Upstream only implements cuDNN *forward* convolution. Weight gradients are rewritten
as another forward `conv2d` with batch/channel axes swapped and stride/dilation
exchanged. On P2's encoder (`3x3` stride-2 at batch 1024) that rewrite becomes a
`32x32` dilated filter and is ~3× slower than im2col.

This tree adds `ConvBackwardFilter` / `ConvBackwardData` launches (via cudarc) and
routes `Op::Conv2D` backprop through them when the `cudnn` feature is on and the
tensors live on CUDA with contiguous layouts (non-contiguous activations fall back to
im2col automatically).

## 2. Skip dead Conv input gradients

Stock candle always materializes `grad_arg` for Conv{1,2}D even when `arg` is a
non-tracking leaf (one-hot pixels). The backward walk now skips input-grad work
unless `arg.id()` is in the `sorted_nodes` needs-grad set. Weight gradients are
unchanged; intermediate activations still receive input grads so earlier layers train.

## Measured (RTX 5060 Laptop)

Encoder c1 microbench (`tests/cuda_conv_probe.rs`):

| path | leaf-input fwd+bwd | notes |
|------|--------------------|-------|
| `--features cuda` | higher | im2col |
| `--features cudnn` + patches | lowest | real bwd filter + skip leaf input grad |

End-to-end `p2-train` (hidden 128, batch 1024): see `docs/RESULTS_P2.md`.
