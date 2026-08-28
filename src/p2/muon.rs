//! Muon optimizer (DeepSeek-V4 / Jordan et al. 2024 / Liu et al. 2025).
//!
//! Matrix-momentum updates with hybrid Newton–Schulz orthogonalization and
//! shape rescaling so AdamW-tuned learning rates transfer directly.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor, D};

/// Muon-is-Scalable RMS matching coefficient.
pub const MUON_RMS_SCALE: f64 = 0.2;

const NS_FAST: (f64, f64, f64) = (3.4445, -4.7750, 2.0315);
const NS_LOCK: (f64, f64, f64) = (2.0, -1.5, 0.5);

/// Foundation-v2 keeps tiny/degenerate projections on Adam. Whitening a matrix
/// with either fan below eight creates a disproportionate fixed-magnitude step.
const MUON_MIN_SIDE: usize = 8;

/// Name fragments that stay on AdamW per DeepSeek-V4 §2.4:
/// embeddings, prediction heads, norm scales, biases.
const ADAMW_NAME_FRAGMENTS: &[&str] = &[
    "embed", "emb", // pixel_emb, action_emb
    "head", "decoder", "gate", "norm", "ln", "bias", "pos", "token", "lm_head",
];

/// Leading `[rows, cols]` for a weight tensor (conv/linear flattened on trailing dims).
pub fn weight_matrix_dims(shape: &[usize]) -> Option<(usize, usize)> {
    match shape.len() {
        0 | 1 => None,
        2 => Some((shape[0], shape[1])),
        n if n >= 3 => {
            let rows = shape[0];
            let cols: usize = shape[1..].iter().product();
            Some((rows, cols))
        }
        _ => None,
    }
}

/// Whether a parameter should use Muon vs AdamW (DeepSeek-V4 hybrid routing).
pub fn uses_muon(name: &str, shape: &[usize]) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }
    let lower = name.to_ascii_lowercase();
    if ADAMW_NAME_FRAGMENTS.iter().any(|frag| lower.contains(frag)) {
        return false;
    }
    let Some((rows, cols)) = weight_matrix_dims(shape) else {
        return false;
    };
    rows >= MUON_MIN_SIDE && cols >= MUON_MIN_SIDE
}

/// Reshape conv/linear weights to a 2D matrix for Muon.
pub fn matrix_view(t: &Tensor) -> Result<Tensor> {
    match t.rank() {
        1 => t.unsqueeze(0).map_err(Into::into),
        2 => Ok(t.clone()),
        r if r >= 3 => {
            let dims = t.dims();
            let rows = dims[0];
            let cols: usize = dims[1..].iter().product();
            t.reshape((rows, cols)).map_err(Into::into)
        }
        _ => bail!("matrix_view: empty tensor"),
    }
}

/// DeepSeek-V4 hybrid Newton–Schulz (10 steps: 8 fast + 2 lock).
pub fn hybrid_newton_schulz(g: &Tensor) -> Result<Tensor> {
    let g = g.to_dtype(DType::F32)?;
    let mut x = matrix_view(&g)?;
    let (rows, cols) = x.dims2()?;
    let transposed = rows > cols;
    if transposed {
        x = x.transpose(0, 1)?.contiguous()?;
    }
    let fro = x.sqr()?.sum_all()?.sqrt()?.maximum(1e-7)?;
    x = x.broadcast_div(&fro)?;
    for (a, b, c) in std::iter::repeat_n(NS_FAST, 8).chain(std::iter::repeat_n(NS_LOCK, 2)) {
        let a_mat = x.matmul(&x.transpose(D::Minus1, D::Minus2)?)?;
        let ax = a_mat.matmul(&x)?;
        let b_term = ax.affine(b, 0.0)?;
        let c_term = a_mat.matmul(&ax)?.affine(c, 0.0)?;
        x = x.affine(a, 0.0)?.add(&b_term)?.add(&c_term)?;
    }
    if transposed {
        x = x.transpose(0, 1)?.contiguous()?;
    }
    Ok(x)
}

/// Muon-is-Scalable RMS matching: `U * 0.2 * sqrt(max(fan_in, fan_out))`.
/// `gamma` remains explicit for checkpointed experiment configuration.
pub fn muon_shape_rescale(update: &Tensor, gamma: f64) -> Result<Tensor> {
    let (n, m) = update.dims2()?;
    let scale = (n.max(m) as f64).sqrt() * gamma;
    update.affine(scale, 0.0).map_err(Into::into)
}

/// One Muon step on a 2D gradient: Nesterov + hybrid NS + shape rescale.
pub fn muon_update(
    grad: &Tensor,
    momentum: &Tensor,
    beta: f64,
    lr: f64,
    gamma: f64,
) -> Result<(Tensor, Tensor)> {
    let g = matrix_view(grad)?;
    let new_m = momentum
        .affine(beta, 0.0)?
        .add(&g.affine(1.0 - beta, 0.0)?)?;
    let nesterov = new_m.affine(beta, 0.0)?.add(&g.affine(1.0 - beta, 0.0)?)?;
    let ortho = hybrid_newton_schulz(&nesterov)?;
    let update = muon_shape_rescale(&ortho, gamma)?;
    let delta = update.affine(-lr, 0.0)?;
    Ok((new_m, delta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn newton_schulz_output_is_finite() -> Result<()> {
        let device = Device::Cpu;
        let g = Tensor::randn(0f32, 1.0, (32, 16), &device)?;
        let u = hybrid_newton_schulz(&g)?;
        assert!(u.to_vec2::<f32>()?.iter().flatten().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn uses_muon_routing() {
        // AdamW: embeddings
        assert!(!uses_muon("action_emb.weight", &[8, 32]));
        assert!(!uses_muon("pixel_emb.weight", &[16, 8]));
        // AdamW: biases
        assert!(!uses_muon("block.c1.bias", &[128]));
        // AdamW: prediction / auxiliary heads (incl. 128×1 logits)
        assert!(!uses_muon("event_head.weight", &[4, 256]));
        assert!(!uses_muon("q_head.weight", &[1, 128]));
        assert!(!uses_muon("reliability_head.weight", &[1, 128]));
        assert!(!uses_muon("prefix_head.weight", &[128, 256]));
        assert!(!uses_muon("grounding.decoder.weight", &[16, 128]));
        assert!(!uses_muon("grounding.copy_gate.weight", &[1, 128]));
        assert!(!uses_muon("action_decoder.weight", &[8, 128]));
        // Muon: hidden conv / block weights
        assert!(uses_muon("block.c1.weight", &[128, 128, 3, 3]));
        assert!(uses_muon("encoder.patch.weight", &[32, 8, 4, 4]));
        // Muon: state/input projections into hidden dim
        assert!(uses_muon("action_proj.weight", &[128, 8]));
        assert!(uses_muon("operator_conditioning_proj.weight", &[128, 54]));
        assert!(!uses_muon("operator_conditioning_proj.bias", &[128]));
        assert!(uses_muon("goal_proj.weight", &[128, 64]));
        assert!(!uses_muon("coord_proj.weight", &[128, 2]));
        assert!(!uses_muon("spatial_action_proj.weight", &[128, 4, 1, 1]));
    }

    #[test]
    fn weight_matrix_dims_flattens_conv() {
        assert_eq!(weight_matrix_dims(&[128, 128, 3, 3]), Some((128, 1152)));
        assert_eq!(weight_matrix_dims(&[1, 128]), Some((1, 128)));
        assert!(!uses_muon("q_head.weight", &[1, 128]));
    }

    fn assert_tensor_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
        Ok(())
    }

    #[test]
    fn two_step_nesterov_matches_deepseek_algorithm_one() -> Result<()> {
        let device = Device::Cpu;
        let beta = 0.8;
        let lr = 0.3;
        let gamma = MUON_RMS_SCALE;
        let g1 = Tensor::new(&[[1f32, 2.0], [3.0, 5.0]], &device)?;
        let g2 = Tensor::new(&[[-2f32, 1.0], [4.0, -3.0]], &device)?;
        let zero = Tensor::zeros((2, 2), DType::F32, &device)?;

        let (m1, delta1) = muon_update(&g1, &zero, beta, lr, gamma)?;
        let expected_m1 = g1.affine(1.0 - beta, 0.0)?;
        let expected_n1 = expected_m1
            .affine(beta, 0.0)?
            .add(&g1.affine(1.0 - beta, 0.0)?)?;
        let expected_delta1 =
            muon_shape_rescale(&hybrid_newton_schulz(&expected_n1)?, gamma)?.affine(-lr, 0.0)?;
        assert_tensor_close(&m1, &expected_m1, 1e-6)?;
        assert_tensor_close(&delta1, &expected_delta1, 1e-6)?;

        let (m2, delta2) = muon_update(&g2, &m1, beta, lr, gamma)?;
        let expected_m2 = expected_m1
            .affine(beta, 0.0)?
            .add(&g2.affine(1.0 - beta, 0.0)?)?;
        let expected_n2 = expected_m2
            .affine(beta, 0.0)?
            .add(&g2.affine(1.0 - beta, 0.0)?)?;
        let expected_delta2 =
            muon_shape_rescale(&hybrid_newton_schulz(&expected_n2)?, gamma)?.affine(-lr, 0.0)?;
        assert_tensor_close(&m2, &expected_m2, 1e-6)?;
        assert_tensor_close(&delta2, &expected_delta2, 1e-6)?;

        let old_nesterov = expected_m2.add(&g2.affine(beta, 0.0)?)?;
        let old_delta =
            muon_shape_rescale(&hybrid_newton_schulz(&old_nesterov)?, gamma)?.affine(-lr, 0.0)?;
        let corrected = delta2.flatten_all()?.to_vec1::<f32>()?;
        let old = old_delta.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            corrected
                .iter()
                .zip(old)
                .any(|(corrected, old)| (corrected - old).abs() > 1e-4),
            "non-collinear second step must distinguish the old Nesterov formula"
        );
        Ok(())
    }
}
