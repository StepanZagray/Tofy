use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

/// JEPA predictor: maps context latent to target latent.
/// Modern practice: bottleneck (dim → hidden → dim) + pre-norm for stability.
/// Bottleneck forces abstraction; LayerNorm stabilizes training.
#[allow(dead_code)]
pub struct Predictor {
    ln: nn::LayerNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

#[allow(dead_code)]
impl Predictor {
    /// hidden_dim: bottleneck size (e.g. dim/4). Use max(dim/4, 32) for small dim.
    pub fn new(vb: VarBuilder<'_>, dim: usize, hidden_dim: usize) -> Result<Self> {
        let ln = nn::layer_norm(dim, 1e-5, vb.pp("ln"))?;
        let fc1 = nn::linear(dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = nn::linear(hidden_dim, dim, vb.pp("fc2"))?;
        Ok(Self { ln, fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ln.forward(x)?;
        let h = self.fc1.forward(&x)?.relu()?;
        Ok(self.fc2.forward(&h)?)
    }
}
