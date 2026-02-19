use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use super::shared::EncoderBackbone;

/// Online encoder: updated directly by gradients/optimizer each step.
pub struct OnlineEncoder {
    inner: EncoderBackbone,
}

impl OnlineEncoder {
    pub fn new(
        vb: VarBuilder<'_>,
        vocab_size: usize,
        dim: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: EncoderBackbone::new(vb, vocab_size, dim, num_layers, num_heads)?,
        })
    }

    pub fn forward_sequence(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward_sequence(x)
    }

    #[allow(dead_code)]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}
