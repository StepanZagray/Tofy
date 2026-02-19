use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

/// Lightweight projection from JEPA latent to decoder conditioning vector.
pub struct DecoderBridge {
    proj: nn::Linear,
}

impl DecoderBridge {
    pub fn new(vb: VarBuilder<'_>, in_dim: usize, out_dim: usize) -> Result<Self> {
        let proj = nn::linear(in_dim, out_dim, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.proj.forward(x)?)
    }
}
