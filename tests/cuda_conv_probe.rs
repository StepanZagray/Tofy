//! Isolates the three convolution calls the P2 encoder's first layer generates, so the
//! cuDNN-on vs cuDNN-off decision rests on per-op numbers instead of whole-step totals.
//!
//! Upstream candle 0.11.0 only routes *forward* conv2d through cuDNN. Its stock conv2d
//! backward rewrites the weight gradient as a pathological forward `conv2d` (batch/channel
//! swap + stride/dilation swap), which is ~3× slower than im2col on the P2 c1 shape.
//! Tofy patches this via `[patch.crates-io]` → `vendor/candle-core`, which calls cudarc's
//! `ConvBackwardFilter` / `ConvBackwardData`. The "as conv2d" row below still exercises the
//! old rewrite for comparison; `full fwd+bwd through backprop` uses the patched path when
//! built with `--features cudnn`.
//!
//! Run under each backend and compare:
//!   cargo test --release --features cuda  --test cuda_conv_probe -- --ignored --nocapture
//!   cargo test --release --features cudnn --test cuda_conv_probe -- --ignored --nocapture

#![cfg(feature = "cuda")]

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

/// P2 encoder layer c1 at the real training batch: 16->16 channels, 3x3, stride 2, pad 1.
const BATCH: usize = 1024;
const CHANNELS: usize = 16;
const SIDE: usize = 64;
const KERNEL: usize = 3;
const PADDING: usize = 1;
const STRIDE: usize = 2;
const DILATION: usize = 1;

fn bench(label: &str, device: &Device, reps: usize, mut f: impl FnMut() -> Result<Tensor>) {
    // One warmup so descriptor/plan setup and lazy module loads are not in the sample.
    match f() {
        Ok(_) => {}
        Err(err) => {
            println!("{label:<38} FAILED: {err}");
            return;
        }
    }
    if device.synchronize().is_err() {
        println!("{label:<38} FAILED: synchronize");
        return;
    }
    let start = Instant::now();
    for _ in 0..reps {
        if f().is_err() {
            println!("{label:<38} FAILED mid-run");
            return;
        }
    }
    if device.synchronize().is_err() {
        println!("{label:<38} FAILED: synchronize");
        return;
    }
    let ms = start.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("{label:<38} {ms:>9.2} ms/call");
}

#[test]
#[ignore]
fn encoder_c1_conv_ops() -> Result<()> {
    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(err) => {
            println!("no CUDA device, skipping: {err}");
            return Ok(());
        }
    };
    let backend = if cfg!(feature = "cudnn") {
        "cudnn"
    } else {
        "cuda (im2col)"
    };
    let out_side = (SIDE + 2 * PADDING - KERNEL) / STRIDE + 1;
    println!("backend = {backend}; c1 out {out_side}x{out_side}");

    let x = Tensor::randn(0f32, 1.0, (BATCH, CHANNELS, SIDE, SIDE), &device)?;
    let w = Tensor::randn(0f32, 1.0, (CHANNELS, CHANNELS, KERNEL, KERNEL), &device)?;
    let grad = Tensor::randn(0f32, 1.0, (BATCH, CHANNELS, out_side, out_side), &device)?;

    bench("forward conv2d", &device, 20, || {
        x.conv2d(&w, PADDING, STRIDE, DILATION, 1).map_err(Into::into)
    });

    // backprop.rs:300-302 — note stride and dilation are swapped relative to the forward op.
    let xt = x.transpose(0, 1)?;
    let gt = grad.transpose(0, 1)?;
    bench("backward weight-grad (as conv2d)", &device, 20, || {
        xt.conv2d(&gt, PADDING, DILATION, STRIDE, 1)
            .map_err(Into::into)
    });

    // backprop.rs:290-297 — the input gradient; never has a cuDNN path.
    let out_size = (out_side - 1) * STRIDE + DILATION * (KERNEL - 1) + 1 - 2 * PADDING;
    let out_padding = SIDE - out_size;
    bench("backward input-grad (conv_transpose2d)", &device, 20, || {
        grad.conv_transpose2d(&w, PADDING, out_padding, STRIDE, DILATION)
            .map_err(Into::into)
    });

    // What the whole layer costs end to end.
    // Leaf-input case (matches P2 pixels): weight grad only; input grad is dead work
    // that the vendored candle patch skips when `arg` is not in the grad graph.
    let x_leaf = Tensor::randn(0f32, 1.0, (BATCH, CHANNELS, SIDE, SIDE), &device)?;
    let wv = candle_core::Var::from_tensor(&w)?;
    bench("full fwd+bwd (leaf input)", &device, 10, || {
        let y = x_leaf.conv2d(wv.as_tensor(), PADDING, STRIDE, DILATION, 1)?;
        let loss = y.sqr()?.sum_all()?;
        let _ = loss.backward()?;
        Ok(loss)
    });

    // Both ends trainable: still pays for input grad (needed to reach earlier layers).
    let xv = candle_core::Var::from_tensor(&Tensor::randn(
        0f32,
        1.0,
        (BATCH, CHANNELS, SIDE, SIDE),
        &device,
    )?)?;
    bench("full fwd+bwd (var input)", &device, 10, || {
        let y = xv.as_tensor().conv2d(wv.as_tensor(), PADDING, STRIDE, DILATION, 1)?;
        let loss = y.sqr()?.sum_all()?;
        let _ = loss.backward()?;
        Ok(loss)
    });

    // Numerical sanity: weight-grad via backprop should be finite and non-zero.
    // With `--features cudnn`, also check that the patched ConvBackwardFilter path
    // agrees with the stock transpose+forward rewrite (still numerically valid, just slow).
    let x_fixed = x_leaf.contiguous()?;
    let w_fixed = wv.as_tensor().contiguous()?;
    let y = x_fixed.conv2d(&w_fixed, PADDING, STRIDE, DILATION, 1)?;
    let loss = y.sqr()?.mean_all()?;
    let grads = loss.backward()?;
    let gw = grads.get(&wv).expect("missing weight grad");
    let gw_sum = gw.abs()?.sum_all()?.to_scalar::<f32>()?;
    println!("weight-grad L1 after one step: {gw_sum:.6} (must be finite and > 0)");
    assert!(gw_sum.is_finite() && gw_sum > 0.0);
    // Leaf input must not receive a stored gradient under the skip-dead-input-grad patch.
    assert!(
        grads.get(&x_leaf).is_none(),
        "leaf input unexpectedly received a gradient"
    );

    #[cfg(feature = "cudnn")]
    {
        let y2 = x_fixed.conv2d(&w_fixed, PADDING, STRIDE, DILATION, 1)?;
        let n = y2.elem_count() as f64;
        let dy = (y2 * (2.0f64 / n))?;
        let ref_gw = x_fixed
            .transpose(0, 1)?
            .conv2d(&dy.transpose(0, 1)?, PADDING, DILATION, STRIDE, 1)?
            .transpose(0, 1)?;
        let (_, _, k0, k1) = w_fixed.dims4()?;
        let (_, _, g0, g1) = ref_gw.dims4()?;
        let ref_gw = if g0 != k0 || g1 != k1 {
            ref_gw.narrow(2, 0, k0)?.narrow(3, 0, k1)?
        } else {
            ref_gw
        };
        let diff = (gw - ref_gw)?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("patched vs rewrite weight-grad max|diff|: {diff:.6e}");
        assert!(
            diff < 1e-3,
            "cuDNN ConvBackwardFilter disagrees with rewrite path: max|diff|={diff}"
        );
    }

    let _ = DType::F32;
    Ok(())
}

/// Sweeps every cuDNN forward algorithm against the weight-gradient shape.
///
/// candle's backprop calls plain `conv2d`, which leaves `cudnn_fwd_algo = None` and so takes
/// whatever `pick_algorithm()` (cudarc conv.rs:213, i.e. cuDNN's forward heuristic) returns.
/// If some explicit algorithm beats that pick, the fix is a one-line candle patch swapping
/// `conv2d` for `conv2d_with_algo`. If none of them are competitive, the weight gradient has
/// to go through cuDNN's dedicated `cudnnConvolutionBackwardFilter` entry point instead —
/// cudarc exposes it as `ConvBackwardFilter`, candle simply never calls it.
#[test]
#[ignore]
fn cudnn_algo_sweep_for_weight_grad() -> Result<()> {
    use candle_core::conv::CudnnFwdAlgo::*;

    let device = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(err) => {
            println!("no CUDA device, skipping: {err}");
            return Ok(());
        }
    };
    if !cfg!(feature = "cudnn") {
        println!("built without the cudnn feature; algo choice is ignored, skipping");
        return Ok(());
    }

    let out_side = (SIDE + 2 * PADDING - KERNEL) / STRIDE + 1;
    let x = Tensor::randn(0f32, 1.0, (BATCH, CHANNELS, SIDE, SIDE), &device)?;
    let grad = Tensor::randn(0f32, 1.0, (BATCH, CHANNELS, out_side, out_side), &device)?;
    let xt = x.transpose(0, 1)?;
    let gt = grad.transpose(0, 1)?;

    println!("weight-grad shape: input {:?} filter {:?} pad {PADDING} stride {DILATION} dilation {STRIDE}",
        xt.dims(), gt.dims());

    bench("  heuristic (what candle uses)", &device, 10, || {
        xt.conv2d(&gt, PADDING, DILATION, STRIDE, 1)
            .map_err(Into::into)
    });
    for (name, algo) in [
        ("ImplicitGemm", ImplicitGemm),
        ("ImplicitPrecompGemm", ImplicitPrecompGemm),
        ("Gemm", Gemm),
        ("Direct", Direct),
        ("Fft", Fft),
        ("FftTiling", FftTiling),
        ("Winograd", Winograd),
        ("WinogradNonFused", WinogradNonFused),
    ] {
        bench(&format!("  {name}"), &device, 10, || {
            xt.conv2d_with_algo(&gt, PADDING, DILATION, STRIDE, 1, Some(algo))
                .map_err(Into::into)
        });
    }
    Ok(())
}
