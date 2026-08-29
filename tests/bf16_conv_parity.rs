use anyhow::Result;
use candle_core::{DType, Device, Tensor, Var};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// BF16 rounds each operand and stored result at roughly 0.78% spacing near one. The aggregate
// bounds allow several such roundings while remaining tight enough to catch reduced-precision
// accumulation or a broken backward implementation.
const FORWARD_RELATIVE_L2_TOLERANCE: f64 = 0.02;
const BACKWARD_RELATIVE_L2_TOLERANCE: f64 = 0.05;

const INPUT_SHAPE: (usize, usize, usize, usize) = (2, 3, 5, 5);
const KERNEL_SHAPE: (usize, usize, usize, usize) = (4, 3, 3, 3);

struct Evaluation {
    output: Vec<f32>,
    input_grad: Vec<f32>,
    kernel_grad: Vec<f32>,
}

fn samples() -> (Vec<f32>, Vec<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(0xB_F16_C02);
    let input = (0..2 * 3 * 5 * 5)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect();
    let kernel = (0..4 * 3 * 3 * 3)
        .map(|_| rng.random_range(-0.5f32..0.5))
        .collect();
    (input, kernel)
}

fn variable(
    values: &[f32],
    shape: impl Into<candle_core::Shape>,
    dtype: DType,
    device: &Device,
) -> Result<Var> {
    let tensor = Tensor::from_vec(values.to_vec(), shape, device)?.to_dtype(dtype)?;
    Ok(Var::from_tensor(&tensor)?)
}

fn f32_values(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?)
}

fn evaluate(dtype: DType, device: &Device) -> Result<Evaluation> {
    let (input_values, kernel_values) = samples();
    let input = variable(&input_values, INPUT_SHAPE, dtype, device)?;
    let kernel = variable(&kernel_values, KERNEL_SHAPE, dtype, device)?;
    let output = input.conv2d(&kernel, 1, 1, 1, 1)?;
    assert_eq!(output.dtype(), dtype);

    // Keep loss arithmetic in F32 so this isolates operand/output quantization and convolution.
    let loss = output.to_dtype(DType::F32)?.sqr()?.sum_all()?;
    let gradients = loss.backward()?;
    device.synchronize()?;
    let input_grad = gradients.get(&input).expect("input gradient");
    let kernel_grad = gradients.get(&kernel).expect("kernel gradient");
    // Conv backward follows Candle's same-dtype storage contract. A surrounding cast then
    // promotes these gradients to the dtype of an F32 master, as checked separately below.
    assert_eq!(input_grad.dtype(), dtype);
    assert_eq!(kernel_grad.dtype(), dtype);
    Ok(Evaluation {
        output: f32_values(&output)?,
        input_grad: f32_values(input_grad)?,
        kernel_grad: f32_values(kernel_grad)?,
    })
}

fn relative_l2(actual: &[f32], reference: &[f32]) -> f64 {
    assert_eq!(actual.len(), reference.len());
    let (squared_error, squared_reference) = actual.iter().zip(reference).fold(
        (0.0f64, 0.0f64),
        |(error, norm), (&actual, &reference)| {
            let delta = f64::from(actual) - f64::from(reference);
            (
                error + delta * delta,
                norm + f64::from(reference) * f64::from(reference),
            )
        },
    );
    (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt()
}

fn assert_finite(values: &[f32], label: &str) {
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} contains a non-finite value"
    );
}

fn assert_parity(device: &Device) -> Result<()> {
    let reference = evaluate(DType::F32, device)?;
    let bf16 = evaluate(DType::BF16, device)?;

    assert_finite(&bf16.output, "BF16 output");
    assert_finite(&bf16.input_grad, "BF16 input gradient");
    assert_finite(&bf16.kernel_grad, "BF16 kernel gradient");

    let forward_error = relative_l2(&bf16.output, &reference.output);
    let input_grad_error = relative_l2(&bf16.input_grad, &reference.input_grad);
    let kernel_grad_error = relative_l2(&bf16.kernel_grad, &reference.kernel_grad);
    assert!(
        forward_error <= FORWARD_RELATIVE_L2_TOLERANCE,
        "forward relative L2 error {forward_error} exceeds {FORWARD_RELATIVE_L2_TOLERANCE}"
    );
    assert!(
        input_grad_error <= BACKWARD_RELATIVE_L2_TOLERANCE,
        "input-gradient relative L2 error {input_grad_error} exceeds {BACKWARD_RELATIVE_L2_TOLERANCE}"
    );
    assert!(
        kernel_grad_error <= BACKWARD_RELATIVE_L2_TOLERANCE,
        "kernel-gradient relative L2 error {kernel_grad_error} exceeds {BACKWARD_RELATIVE_L2_TOLERANCE}"
    );
    Ok(())
}

fn assert_cast_gradient_flow(device: &Device) -> Result<()> {
    let (input_values, kernel_values) = samples();
    let input_master = Var::from_vec(input_values, INPUT_SHAPE, device)?;
    let kernel_master = Var::from_vec(kernel_values, KERNEL_SHAPE, device)?;
    let input = input_master.to_dtype(DType::BF16)?;
    let kernel = kernel_master.to_dtype(DType::BF16)?;
    let loss = input
        .conv2d(&kernel, 1, 1, 1, 1)?
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_all()?;
    let gradients = loss.backward()?;
    device.synchronize()?;

    for (master, label) in [
        (input_master.as_tensor(), "input master"),
        (kernel_master.as_tensor(), "kernel master"),
    ] {
        let gradient = gradients.get(master).expect("F32 master gradient");
        assert_eq!(gradient.dtype(), DType::F32, "{label} gradient dtype");
        let values = f32_values(gradient)?;
        assert_finite(&values, label);
        assert!(
            values.iter().any(|value| *value != 0.0),
            "{label} gradient is all zero"
        );
    }
    Ok(())
}

#[test]
fn cpu_bf16_conv_matches_f32_forward_and_backward() -> Result<()> {
    assert_parity(&Device::Cpu)
}

#[test]
fn cpu_cast_chain_reaches_f32_masters() -> Result<()> {
    assert_cast_gradient_flow(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device with BF16 support"]
fn cuda_bf16_conv_matches_f32_forward_and_backward() -> Result<()> {
    assert_parity(&Device::new_cuda(0)?)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device with BF16 support"]
fn cuda_cast_chain_reaches_f32_masters() -> Result<()> {
    assert_cast_gradient_flow(&Device::new_cuda(0)?)
}
