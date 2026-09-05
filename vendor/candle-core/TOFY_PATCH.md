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

## 3. BF16 Conv2D accumulation contract

BF16 cuDNN forward, backward-data, and backward-filter use BF16 tensor/filter descriptors,
an FP32 convolution compute descriptor, and explicit `CUDNN_TENSOR_OP_MATH`. Results remain
BF16 tensors: cuDNN accumulates internally in FP32 and rounds once when storing `y`, `dX`, or
`dW`. In particular, convolution gradients are BF16 before a differentiable cast returns them
to an F32 master tensor.

The non-cuDNN CUDA forward path is im2col plus BF16 cuBLAS GEMM with FP32 compute by default;
its direct fallback CUDA kernel also has an FP32 accumulator. Backward operators use the same
FP32-accumulating primitives and store BF16 outputs. CPU Conv2D and the ConvTranspose2D used by
its backward-data path convert BF16 operands to F32 internally, accumulate in F32, and convert
the stored result back to BF16. Existing behavior for every other dtype is unchanged.

Run the A40 kernel/timing probe with:

```console
cargo run --release --features cudnn --example conv_kernel_probe -- --warmup 20 --iters 100
```

The ignored parity tests can exercise either CUDA implementation explicitly:

```console
cargo test --no-default-features --features cuda --test bf16_conv_parity -- --ignored
cargo test --no-default-features --features cudnn --test bf16_conv_parity -- --ignored
```

The probe binary is named `conv_kernel_probe`, benchmarks the Foundation-V2 recurrent shape,
and synchronizes every forward/backward sample. Candle has no convolution-level TF32 toggle, so
the F32 arm measures cuDNN's existing default math behavior; use an Nsight Systems trace of this
binary to identify the selected F32 and BF16 kernels.

## 4. Deterministic cuDNN Conv2D backward algorithms

The cuDNN safe wrapper's backward algorithm picker returns the fastest heuristic candidate
without filtering for determinism. NVIDIA's cuDNN 9.25 headers classify backward-filter
algorithms 0 and 3 and backward-data algorithm 0 as nondeterministic. This tree therefore pins
backward-filter and backward-data to their documented deterministic `ALGO_1` variants. Forward
convolution remains on its existing configured or heuristic path.

This local choice establishes only the algorithms requested from cuDNN. Whole-training
bit-determinism must still be verified empirically because other CUDA operations can remain
nondeterministic.
