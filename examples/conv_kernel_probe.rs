use anyhow::{ensure, Result};
use candle_core::{DType, Device, Var};
use clap::Parser;
use std::hint::black_box;
use std::time::{Duration, Instant};

const BATCH: usize = 1_024;
const CHANNELS: usize = 128;
const SPATIAL: usize = 16;
const KERNEL: usize = 3;

#[derive(Debug, Parser)]
#[command(
    name = "conv-kernel-probe",
    about = "Time Foundation-V2's recurrent Conv2D forward and both gradients"
)]
struct Args {
    /// Untimed, device-synchronized steps per dtype.
    #[arg(long, default_value_t = 20)]
    warmup: usize,

    /// Timed, device-synchronized steps per dtype.
    #[arg(long, default_value_t = 100)]
    iters: usize,
}

fn step(input: &Var, kernel: &Var, device: &Device) -> Result<()> {
    let output = input.conv2d(kernel, 1, 1, 1, 1)?;
    // Candle seeds a non-scalar backward with an all-ones output gradient, avoiding an unrelated
    // reduction kernel in the timed region while exercising both convolution gradients.
    let gradients = output.backward()?;
    device.synchronize()?;
    black_box(gradients);
    Ok(())
}

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort_unstable();
    let middle = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[middle]
    } else {
        (samples[middle - 1] + samples[middle]) / 2
    }
}

fn benchmark(dtype: DType, args: &Args, device: &Device) -> Result<Duration> {
    let input = Var::randn_f64(0.0, 1.0, (BATCH, CHANNELS, SPATIAL, SPATIAL), dtype, device)?;
    let kernel = Var::randn_f64(
        0.0,
        0.05,
        (CHANNELS, CHANNELS, KERNEL, KERNEL),
        dtype,
        device,
    )?;

    for _ in 0..args.warmup {
        step(&input, &kernel, device)?;
    }

    let mut samples = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        device.synchronize()?;
        let started = Instant::now();
        step(&input, &kernel, device)?;
        samples.push(started.elapsed());
    }
    Ok(median(&mut samples))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.iters > 0, "--iters must be greater than zero");

    let device = Device::new_cuda(0)?;
    ensure!(device.supports_bf16(), "cuda:0 does not support BF16");
    println!(
        "shape=NCHW[{BATCH},{CHANNELS},{SPATIAL},{SPATIAL}] kernel=[{CHANNELS},{CHANNELS},{KERNEL},{KERNEL}] padding=1 stride=1 warmup={} iters={}",
        args.warmup, args.iters
    );
    println!(
        "f32 uses cuDNN's existing default math behavior; Candle exposes no Conv2D TF32 toggle"
    );

    for (label, dtype) in [("f32", DType::F32), ("bf16-fp32-accum", DType::BF16)] {
        let elapsed = benchmark(dtype, &args, &device)?;
        println!(
            "mode={label} median_step_ms={:.3}",
            elapsed.as_secs_f64() * 1e3
        );
    }
    Ok(())
}
