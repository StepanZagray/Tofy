use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use super::shared::{EncoderBackbone, EncoderFeatures};

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

    pub fn forward_features(&self, x: &Tensor) -> Result<EncoderFeatures> {
        self.inner.forward_features(x)
    }

    /// Effective chunk size the backbone will use for a given sequence length.
    pub fn chunk_size_for_seq_len(&self, seq_len: usize) -> usize {
        self.inner.chunk_size_for_seq_len(seq_len)
    }
}
