use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{self as nn, Module, VarBuilder};

use super::attention::MultiHeadAttention;

struct TransitionBlock {
    attn: MultiHeadAttention,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
}

impl TransitionBlock {
    fn new(vb: VarBuilder<'_>, dim: usize, num_heads: usize, ff_dim: usize) -> Result<Self> {
        Ok(Self {
            attn: MultiHeadAttention::new(vb.pp("attn"), dim, num_heads)?,
            ln1: nn::layer_norm(dim, 1e-5, vb.pp("ln1"))?,
            ln2: nn::layer_norm(dim, 1e-5, vb.pp("ln2"))?,
            ff1: nn::linear(dim, ff_dim, vb.pp("ff1"))?,
            ff2: nn::linear(ff_dim, dim, vb.pp("ff2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = crate::util::layer_norm_diff(&self.ln1, x)?;
        let x = (x + self.attn.forward(&normed)?)?;
        let ff = self.ff2.forward(
            &self
                .ff1
                .forward(&crate::util::layer_norm_diff(&self.ln2, &x)?)?
                .gelu()?,
        )?;
        Ok((x + ff)?)
    }
}

/// Predicts next latent-slot sequence from current slots (unconditioned residual predictor).
pub struct ActionStateTransition {
    blocks: Vec<TransitionBlock>,
    delta_ln: nn::LayerNorm,
    delta_proj: nn::Linear,
}

impl ActionStateTransition {
    pub fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        let num_blocks = 6;
        let num_heads = transition_heads(dim);
        let ff_dim = (dim * 4).max(320);
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            blocks.push(TransitionBlock::new(
                vb.pp(format!("block_{}", i)),
                dim,
                num_heads,
                ff_dim,
            )?);
        }
        let delta_ln = nn::layer_norm(dim, 1e-5, vb.pp("delta_ln"))?;
        let delta_proj = nn::linear(dim, dim, vb.pp("delta_proj"))?;
        Ok(Self {
            blocks,
            delta_ln,
            delta_proj,
        })
    }

    pub fn forward_delta(&self, state_slots: &Tensor) -> Result<Tensor> {
        let mut hidden = state_slots.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        self.delta_proj
            .forward(&crate::util::layer_norm_diff(&self.delta_ln, &hidden)?)?
            .tanh()
            .map_err(Into::into)
    }

    pub fn forward(&self, state_slots: &Tensor) -> Result<Tensor> {
        let delta = self.forward_delta(state_slots)?;
        (state_slots + delta).map_err(Into::into)
    }

    pub fn forward_one(&self, state_slots: &Tensor) -> Result<Tensor> {
        self.forward(state_slots)
    }
}

fn transition_heads(dim: usize) -> usize {
    [16, 8, 4, 2]
        .into_iter()
        .find(|heads| dim.is_multiple_of(*heads))
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn forward_preserves_batch_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let transition = ActionStateTransition::new(vb, 16)?;
        let state = Tensor::zeros((2, 3, 16), DType::F32, &device)?;
        let out = transition.forward(&state)?;
        assert_eq!(out.dims(), state.dims());
        Ok(())
    }

    #[test]
    fn raw_prediction_loss_reaches_transition_parameters() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let transition =
            ActionStateTransition::new(VarBuilder::from_varmap(&varmap, DType::F32, &device), 16)?;
        let state = Tensor::randn(0f32, 1f32, (2, 3, 16), &device)?;
        let target = Tensor::randn(0f32, 1f32, (2, 3, 16), &device)?;
        let loss = crate::model::prediction_loss(&transition.forward(&state)?, &target)?;
        let grads = loss.backward()?;
        let variables = varmap.all_vars();
        assert!(!variables.is_empty());
        assert!(variables
            .iter()
            .all(|variable| grads.get(variable).is_some()));
        Ok(())
    }
}
