//! Muon optimizer (DeepSeek-V4 / Jordan et al. 2024 / Liu et al. 2025).
//!
//! Matrix-momentum updates with hybrid Newton–Schulz orthogonalization and
//! shape rescaling so AdamW-tuned learning rates transfer directly.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor, D};

/// DeepSeek-V4 rescales orthogonalized updates to this target RMS (Algorithm 1, γ).
pub const MUON_RMS_SCALE: f64 = 0.18;

const NS_FAST: (f64, f64, f64) = (3.4445, -4.7750, 2.0315);
const NS_LOCK: (f64, f64, f64) = (2.0, -1.5, 0.5);

/// Minimum rows/cols for Muon orthogonalization (DeepSeek-V4: undefined on vectors).
const MUON_MIN_SIDE: usize = 2;

/// Name fragments that stay on AdamW per DeepSeek-V4 §2.4:
/// embeddings, prediction heads, norm scales, biases.
const ADAMW_NAME_FRAGMENTS: &[&str] = &[
    "embed",
    "emb", // pixel_emb, action_emb
    "head",
    "norm",
    "ln",
    "bias",
    "pos",
    "token",
    "lm_head",
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
    if ADAMW_NAME_FRAGMENTS
        .iter()
        .any(|frag| lower.contains(frag))
    {
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
    let fro = x.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
    let fro = fro.max(1e-7);
    x = x.affine(1.0 / fro, 0.0)?;
    for (a, b, c) in std::iter::repeat_n(NS_FAST, 8).chain(std::iter::repeat_n(NS_LOCK, 2)) {
        let a_mat = x.matmul(&x.transpose(D::Minus1, D::Minus2)?)?;
        let b_term = a_mat.matmul(&x)?.affine(b, 0.0)?;
        let c_term = a_mat
            .matmul(&a_mat)?
            .matmul(&x)?
            .affine(c, 0.0)?;
        x = x.affine(a, 0.0)?.add(&b_term)?.add(&c_term)?;
    }
    if transposed {
        x = x.transpose(0, 1)?.contiguous()?;
    }
    Ok(x)
}

/// Shape rescale from DeepSeek-V4 Algorithm 1: `U * sqrt(max(n,m)) * γ`.
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
    _weight_decay: f64,
    gamma: f64,
) -> Result<(Tensor, Tensor)> {
    let g = matrix_view(grad)?;
    let new_m = momentum
        .affine(beta, 0.0)?
        .add(&g.affine(1.0 - beta, 0.0)?)?;
    let nesterov = new_m.add(&g.affine(beta, 0.0)?)?;
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
        // Muon: hidden conv / block weights
        assert!(uses_muon("block.c1.weight", &[128, 128, 3, 3]));
        assert!(uses_muon("encoder.patch.weight", &[32, 8, 4, 4]));
        // Muon: state/input projections into hidden dim
        assert!(uses_muon("action_proj.weight", &[128, 8]));
        assert!(uses_muon("goal_proj.weight", &[128, 64]));
        assert!(uses_muon("coord_proj.weight", &[128, 2]));
    }

    #[test]
    fn weight_matrix_dims_flattens_conv() {
        assert_eq!(weight_matrix_dims(&[128, 128, 3, 3]), Some((128, 1152)));
        assert_eq!(weight_matrix_dims(&[1, 128]), Some((1, 128)));
        assert!(!uses_muon("q_head.weight", &[1, 128]));
    }
}
