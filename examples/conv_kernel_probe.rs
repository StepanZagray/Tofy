use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor, Var};
use clap::{Parser, ValueEnum};
use std::hint::black_box;
use std::time::{Duration, Instant};

const BATCH: usize = 1_024;
const CHANNELS: usize = 128;
const SPATIAL: usize = 16;
const KERNEL: usize = 3;

#[derive(Debug, Parser)]
#[command(
    name = "conv-kernel-probe",
    about = "Time native-dtype Conv2D or Foundation-V2's full recurrent cast-island autograd path"
)]
struct Args {
    /// Graph to benchmark. `native-dtype` is the historical one-convolution probe; `cast-island`
    /// reproduces one complete GridResidualBlock with F32 roots and the model's BF16 casts.
    #[arg(long, value_enum, default_value = "native-dtype")]
    mode: Mode,

    /// Untimed, device-synchronized steps per dtype.
    #[arg(long, default_value_t = 20)]
    warmup: usize,

    /// Timed, device-synchronized steps per dtype.
    #[arg(long, default_value_t = 100)]
    iters: usize,

    /// Required F32/BF16 speedup for `--mode cast-island`.
    #[arg(long, default_value_t = 1.3)]
    min_speedup: f64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    NativeDtype,
    CastIsland,
}

fn native_dtype_step(input: &Var, kernel: &Var, device: &Device) -> Result<()> {
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

fn benchmark_native_dtype(dtype: DType, args: &Args, device: &Device) -> Result<Duration> {
    device.set_seed(0xB_F16_C02)?;
    let input = Var::randn_f64(0.0, 1.0, (BATCH, CHANNELS, SPATIAL, SPATIAL), dtype, device)?;
    let kernel = Var::randn_f64(
        0.0,
        0.05,
        (CHANNELS, CHANNELS, KERNEL, KERNEL),
        dtype,
        device,
    )?;

    for _ in 0..args.warmup {
        native_dtype_step(&input, &kernel, device)?;
    }

    let mut samples = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        device.synchronize()?;
        let started = Instant::now();
        native_dtype_step(&input, &kernel, device)?;
        samples.push(started.elapsed());
    }
    Ok(median(&mut samples))
}

struct BlockVars {
    input: Var,
    c1_weight: Var,
    c1_bias: Var,
    c2_weight: Var,
    c2_bias: Var,
    film_gamma: Var,
    film_beta: Var,
}

impl BlockVars {
    fn new(device: &Device) -> Result<Self> {
        // Both arms use F32 roots. In the treatment arm only the operands of each convolution
        // cross to BF16; the residual state, biases, SiLU, FiLM, and returned delta stay F32.
        device.set_seed(0xB_F16_C02)?;
        Ok(Self {
            input: Var::randn_f64(
                0.0,
                1.0,
                (BATCH, CHANNELS, SPATIAL, SPATIAL),
                DType::F32,
                device,
            )?,
            c1_weight: Var::randn_f64(
                0.0,
                0.05,
                (CHANNELS, CHANNELS, KERNEL, KERNEL),
                DType::F32,
                device,
            )?,
            c1_bias: Var::zeros(CHANNELS, DType::F32, device)?,
            c2_weight: Var::randn_f64(
                0.0,
                0.05,
                (CHANNELS, CHANNELS, KERNEL, KERNEL),
                DType::F32,
                device,
            )?,
            c2_bias: Var::zeros(CHANNELS, DType::F32, device)?,
            film_gamma: Var::ones((BATCH, CHANNELS, 1, 1), DType::F32, device)?,
            film_beta: Var::zeros((BATCH, CHANNELS, 1, 1), DType::F32, device)?,
        })
    }
}

fn conv_product_f32(input: &Tensor, weight: &Tensor, bias: &Tensor, bf16: bool) -> Result<Tensor> {
    let output = if bf16 {
        input
            .to_dtype(DType::BF16)?
            .conv2d(&weight.to_dtype(DType::BF16)?, 1, 1, 1, 1)?
            .to_dtype(DType::F32)?
    } else {
        input.conv2d(weight, 1, 1, 1, 1)?
    };
    output
        .broadcast_add(&bias.reshape((1, CHANNELS, 1, 1))?)
        .map_err(Into::into)
}

fn recurrent_block(vars: &BlockVars, bf16: bool) -> Result<Tensor> {
    let hidden = conv_product_f32(&vars.input, &vars.c1_weight, &vars.c1_bias, bf16)?.silu()?;
    let hidden = hidden
        .broadcast_mul(&vars.film_gamma)?
        .broadcast_add(&vars.film_beta)?;
    let delta = conv_product_f32(&hidden, &vars.c2_weight, &vars.c2_bias, bf16)?;
    vars.input.add(&delta).map_err(Into::into)
}

#[derive(Clone, Copy)]
struct BlockSample {
    forward: Duration,
    backward: Duration,
    total: Duration,
}

fn cast_island_step(vars: &BlockVars, bf16: bool, device: &Device) -> Result<BlockSample> {
    device.synchronize()?;
    let total_started = Instant::now();
    let forward_started = Instant::now();
    let output = recurrent_block(vars, bf16)?;
    device.synchronize()?;
    let forward = forward_started.elapsed();

    let backward_started = Instant::now();
    // A non-scalar root makes Candle seed an all-ones F32 output gradient. Backprop then traverses
    // the exact BF16-output -> F32-state cast and reaches every F32 Var root through autograd.
    let gradients = output.backward()?;
    device.synchronize()?;
    let backward = backward_started.elapsed();
    let total = total_started.elapsed();

    for (root, label) in [
        (vars.input.as_tensor(), "input"),
        (vars.c1_weight.as_tensor(), "c1_weight"),
        (vars.c1_bias.as_tensor(), "c1_bias"),
        (vars.c2_weight.as_tensor(), "c2_weight"),
        (vars.c2_bias.as_tensor(), "c2_bias"),
        (vars.film_gamma.as_tensor(), "film_gamma"),
        (vars.film_beta.as_tensor(), "film_beta"),
    ] {
        let grad = gradients
            .get(root)
            .unwrap_or_else(|| panic!("missing full-autograd gradient for {label}"));
        ensure!(grad.dtype() == DType::F32, "{label} gradient is not F32");
    }
    black_box(gradients);
    Ok(BlockSample {
        forward,
        backward,
        total,
    })
}

fn benchmark_cast_island(bf16: bool, args: &Args, device: &Device) -> Result<BlockSample> {
    let vars = BlockVars::new(device)?;
    for _ in 0..args.warmup {
        black_box(cast_island_step(&vars, bf16, device)?);
    }

    let mut forward = Vec::with_capacity(args.iters);
    let mut backward = Vec::with_capacity(args.iters);
    let mut total = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        let sample = cast_island_step(&vars, bf16, device)?;
        forward.push(sample.forward);
        backward.push(sample.backward);
        total.push(sample.total);
    }
    Ok(BlockSample {
        forward: median(&mut forward),
        backward: median(&mut backward),
        total: median(&mut total),
    })
}

fn as_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.iters > 0, "--iters must be greater than zero");
    ensure!(
        args.min_speedup.is_finite() && args.min_speedup > 0.0,
        "--min-speedup must be finite and greater than zero"
    );

    let device = Device::new_cuda(0)?;
    ensure!(device.supports_bf16(), "cuda:0 does not support BF16");
    println!(
        "f32 uses cuDNN's existing default math behavior; Candle exposes no Conv2D TF32 toggle"
    );

    match args.mode {
        Mode::NativeDtype => {
            println!(
                "graph=native-dtype-conv shape=NCHW[{BATCH},{CHANNELS},{SPATIAL},{SPATIAL}] kernel=[{CHANNELS},{CHANNELS},{KERNEL},{KERNEL}] padding=1 stride=1 warmup={} iters={}",
                args.warmup, args.iters
            );
            for (label, dtype) in [("f32", DType::F32), ("bf16-fp32-accum", DType::BF16)] {
                let elapsed = benchmark_native_dtype(dtype, &args, &device)?;
                println!("precision={label} median_step_ms={:.3}", as_ms(elapsed));
            }
        }
        Mode::CastIsland => {
            println!(
                "graph=full-autograd-grid-residual-block roots=f32 convs=2 shape=NCHW[{BATCH},{CHANNELS},{SPATIAL},{SPATIAL}] kernel=[{CHANNELS},{CHANNELS},{KERNEL},{KERNEL}] padding=1 stride=1 warmup={} iters={} min_speedup={:.3}",
                args.warmup, args.iters, args.min_speedup
            );
            let f32 = benchmark_cast_island(false, &args, &device)?;
            let bf16 = benchmark_cast_island(true, &args, &device)?;
            for (label, sample) in [("f32", f32), ("bf16-fp32-accum", bf16)] {
                println!(
                    "precision={label} median_forward_ms={:.3} median_backward_ms={:.3} median_step_ms={:.3}",
                    as_ms(sample.forward),
                    as_ms(sample.backward),
                    as_ms(sample.total)
                );
            }
            let speedup = f32.total.as_secs_f64() / bf16.total.as_secs_f64();
            println!("full_autograd_speedup={speedup:.3}x");
            ensure!(
                speedup >= args.min_speedup,
                "full-autograd cast-island speedup {speedup:.3}x is below required {:.3}x",
                args.min_speedup
            );
        }
    }
    Ok(())
}
