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

    /// Reports block allocation separately from the sequence-dependent
    /// attention score work, avoiding the false implication that a layer
    /// fraction is also a compute fraction.
    pub fn attention_work_summary(&self, seq_len: usize) -> String {
        self.inner.attention_work_summary(seq_len)
    }

    /// Prediction-space head used by masked latent pretraining. Downstream
    /// consumers continue to use the encoder representation before this head.
    pub fn predict_states(&self, states: &Tensor) -> Result<Tensor> {
        self.inner.predict_states(states)
    }
}
