use anyhow::{bail, Result};
use candle_core::Tensor;
use candle_nn::{self as nn, Init, Module, VarBuilder};

use super::attention::MultiHeadAttention;

const NUM_ACTIONS: usize = 4;

struct TransitionBlock {
    attn: MultiHeadAttention,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
    modulation: nn::Linear,
    dim: usize,
}

impl TransitionBlock {
    fn new(vb: VarBuilder<'_>, dim: usize, num_heads: usize, ff_dim: usize) -> Result<Self> {
        let modulation = nn::Linear::new(
            vb.get_with_hints((6 * dim, dim), "modulation.weight", Init::Const(0.0))?,
            Some(vb.get_with_hints(6 * dim, "modulation.bias", Init::Const(0.0))?),
        );
        Ok(Self {
            attn: MultiHeadAttention::new(vb.pp("attn"), dim, num_heads)?,
            ln1: nn::layer_norm(dim, 1e-6, vb.pp("ln1"))?,
            ln2: nn::layer_norm(dim, 1e-6, vb.pp("ln2"))?,
            ff1: nn::linear(dim, ff_dim, vb.pp("ff1"))?,
            ff2: nn::linear(ff_dim, dim, vb.pp("ff2"))?,
            modulation,
            dim,
        })
    }

    fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&scale.affine(1.0, 1.0)?)?
            .broadcast_add(shift)
            .map_err(Into::into)
    }

    fn forward(&self, x: &Tensor, action: &Tensor) -> Result<Tensor> {
        let modulation = self.modulation.forward(&action.silu()?)?;
        let parts = modulation.chunk(6, candle_core::D::Minus1)?;
        if parts.len() != 6 || parts.iter().any(|part| part.dim(2).ok() != Some(self.dim)) {
            bail!("AdaLN modulation produced an invalid shape")
        }

        let normed = crate::util::layer_norm_diff(&self.ln1, x)?;
        let attended = self
            .attn
            .forward(&Self::modulate(&normed, &parts[0], &parts[1])?)?
            .broadcast_mul(&parts[2])?;
        let x = (x + attended)?;

        let normed = crate::util::layer_norm_diff(&self.ln2, &x)?;
        let ff = self.ff2.forward(
            &self
                .ff1
                .forward(&Self::modulate(&normed, &parts[3], &parts[4])?)?
                .gelu()?,
        )?;
        (x + ff.broadcast_mul(&parts[5])?).map_err(Into::into)
    }
}

/// LeWorldModel-style action-conditioned latent predictor.
///
/// Discrete Tofy actions condition every transformer block through AdaLN-Zero,
/// so the predictor begins action-agnostic and learns action effects gradually.
pub struct ActionStateTransition {
    action_embed: nn::Embedding,
    blocks: Vec<TransitionBlock>,
    output_norm: nn::LayerNorm,
    output_proj: nn::Linear,
}

impl ActionStateTransition {
    pub fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        let num_blocks = 6;
        let num_heads = transition_heads(dim);
        let ff_dim = (dim * 4).max(320);
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            blocks.push(TransitionBlock::new(
                vb.pp(format!("block_{i}")),
                dim,
                num_heads,
                ff_dim,
            )?);
        }
        Ok(Self {
            action_embed: nn::embedding(NUM_ACTIONS, dim, vb.pp("action_embed"))?,
            blocks,
            output_norm: nn::layer_norm(dim, 1e-6, vb.pp("output_norm"))?,
            output_proj: nn::linear(dim, dim, vb.pp("output_proj"))?,
        })
    }

    pub fn forward(&self, state_slots: &Tensor, action_labels: &Tensor) -> Result<Tensor> {
        let (batch, slots, dim) = state_slots.dims3()?;
        if action_labels.dims() != [batch] {
            bail!(
                "action labels must have shape [{batch}], got {:?}",
                action_labels.dims()
            )
        }
        let action = self
            .action_embed
            .forward(action_labels)?
            .unsqueeze(1)?
            .broadcast_as((batch, slots, dim))?;
        let mut hidden = state_slots.clone();
        for block in &self.blocks {
            hidden = block.forward(&hidden, &action)?;
        }
        Ok(self
            .output_proj
            .forward(&crate::util::layer_norm_diff(&self.output_norm, &hidden)?)?)
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
        let actions = Tensor::from_vec(vec![0u32, 3], 2, &device)?;
        let out = transition.forward(&state, &actions)?;
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
        let actions = Tensor::from_vec(vec![1u32, 3], 2, &device)?;
        let loss = crate::model::prediction_loss(&transition.forward(&state, &actions)?, &target)?;
        let grads = loss.backward()?;
        let variables = varmap.all_vars();
        assert!(!variables.is_empty());
        assert!(variables
            .iter()
            .any(|variable| grads.get(variable).is_some()));
        Ok(())
    }
}
