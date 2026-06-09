use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{self as nn, Module, VarBuilder};

use super::action_classifier_head::NUM_ACTIONS;
use super::attention::MultiHeadAttention;

struct ActionConditionedBlock {
    attn: MultiHeadAttention,
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
    action_mod: nn::Linear,
    dim: usize,
}

impl ActionConditionedBlock {
    fn new(vb: VarBuilder<'_>, dim: usize, num_heads: usize, ff_dim: usize) -> Result<Self> {
        Ok(Self {
            attn: MultiHeadAttention::new(vb.pp("attn"), dim, num_heads)?,
            ln1: nn::layer_norm(dim, 1e-5, vb.pp("ln1"))?,
            ln2: nn::layer_norm(dim, 1e-5, vb.pp("ln2"))?,
            ff1: nn::linear(dim, ff_dim, vb.pp("ff1"))?,
            ff2: nn::linear(ff_dim, dim, vb.pp("ff2"))?,
            action_mod: nn::linear(dim, dim * 4, vb.pp("action_mod"))?,
            dim,
        })
    }

    fn modulate(&self, x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
        let scale = scale.tanh()?.affine(0.25, 1.0)?;
        let shift = shift.tanh()?.affine(0.25, 0.0)?;
        Ok(x.broadcast_mul(&scale)?.broadcast_add(&shift)?)
    }

    fn forward(&self, x: &Tensor, action_vec: &Tensor) -> Result<Tensor> {
        let modulation = self.action_mod.forward(action_vec)?.unsqueeze(1)?;
        let attn_scale = modulation.narrow(2, 0, self.dim)?;
        let attn_shift = modulation.narrow(2, self.dim, self.dim)?;
        let ff_scale = modulation.narrow(2, self.dim * 2, self.dim)?;
        let ff_shift = modulation.narrow(2, self.dim * 3, self.dim)?;

        let normed = self.modulate(&self.ln1.forward(x)?, &attn_scale, &attn_shift)?;
        let x = (x + self.attn.forward(&normed)?)?;
        let normed = self.modulate(&self.ln2.forward(&x)?, &ff_scale, &ff_shift)?;
        let ff = self.ff2.forward(&self.ff1.forward(&normed)?.gelu()?)?;
        Ok((x + ff)?)
    }
}

/// Predicts next latent-slot sequence conditioned on current slots and an action id.
pub struct ActionStateTransition {
    action_embed: nn::Embedding,
    blocks: Vec<ActionConditionedBlock>,
    delta_ln: nn::LayerNorm,
    delta_proj: nn::Linear,
    delta_gate: nn::Linear,
    dim: usize,
}

impl ActionStateTransition {
    pub fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        let action_embed = nn::embedding(NUM_ACTIONS, dim, vb.pp("action_embed"))?;
        let num_blocks = 6;
        let num_heads = transition_heads(dim);
        let ff_dim = (dim * 4).max(320);
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            blocks.push(ActionConditionedBlock::new(
                vb.pp(format!("block_{}", i)),
                dim,
                num_heads,
                ff_dim,
            )?);
        }
        let delta_ln = nn::layer_norm(dim, 1e-5, vb.pp("delta_ln"))?;
        let delta_proj = nn::linear(dim, dim, vb.pp("delta_proj"))?;
        let delta_gate = nn::linear(dim, dim, vb.pp("delta_gate"))?;
        Ok(Self {
            action_embed,
            blocks,
            delta_ln,
            delta_proj,
            delta_gate,
            dim,
        })
    }

    fn action_vec(&self, state_slots: &Tensor, action_labels: &[u32]) -> Result<Tensor> {
        let (batch, _, _) = state_slots.dims3()?;
        let mut action_ids = if action_labels.len() == 1 {
            vec![action_labels[0]; batch]
        } else if action_labels.len() == batch {
            action_labels.to_vec()
        } else {
            anyhow::bail!(
                "action label count {} must be 1 or match batch size {}",
                action_labels.len(),
                batch
            );
        };
        let strict_labels = std::env::var("TOFY_STRICT_ACTION_LABELS")
            .ok()
            .is_none_or(|value| value == "1" || value.eq_ignore_ascii_case("true"));
        for action_id in &mut action_ids {
            if *action_id >= NUM_ACTIONS as u32 {
                if strict_labels {
                    anyhow::bail!("invalid action label {action_id}; expected 0..{}", NUM_ACTIONS);
                }
                *action_id = (NUM_ACTIONS - 1) as u32;
            }
        }
        let action_ids = Tensor::from_vec(action_ids, (batch,), state_slots.device())?;
        self.action_embed.forward(&action_ids).map_err(Into::into)
    }

    pub fn forward_delta(&self, state_slots: &Tensor, action_labels: &[u32]) -> Result<Tensor> {
        let (batch, slots, _) = state_slots.dims3()?;
        let action_vec = self.action_vec(state_slots, action_labels)?;
        let action_bias = action_vec
            .unsqueeze(1)?
            .broadcast_as((batch, slots, self.dim))?;
        let mut hidden = state_slots.broadcast_add(&action_bias)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &action_vec)?;
        }
        let delta = self
            .delta_proj
            .forward(&self.delta_ln.forward(&hidden)?)?
            .tanh()?;
        let gate = nn::ops::sigmoid(&self.delta_gate.forward(&action_vec)?)?
            .unsqueeze(1)?
            .broadcast_as((batch, slots, self.dim))?;
        delta.broadcast_mul(&gate).map_err(Into::into)
    }

    pub fn forward(&self, state_slots: &Tensor, action_labels: &[u32]) -> Result<Tensor> {
        let delta = self.forward_delta(state_slots, action_labels)?;
        (state_slots + delta).map_err(Into::into)
    }

    pub fn forward_one(&self, state_slots: &Tensor, action_label: u32) -> Result<Tensor> {
        self.forward(state_slots, &[action_label])
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
    fn forward_one_applies_same_action_to_entire_batch() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let transition = ActionStateTransition::new(vb, 16)?;
        let state = Tensor::zeros((2, 3, 16), DType::F32, &device)?;
        let single = transition.forward_one(&state, 1)?;
        let batched = transition.forward(&state, &[1, 1])?;
        let single = single.flatten_all()?.to_vec1::<f32>()?;
        let batched = batched.flatten_all()?.to_vec1::<f32>()?;
        let diff = single
            .iter()
            .zip(batched.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();
        assert!(diff < 1e-5, "forward_one diff {diff}");
        Ok(())
    }

    #[test]
    fn action_label_count_mismatch_is_rejected() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let transition = ActionStateTransition::new(vb, 16)?;
        let state = Tensor::zeros((2, 3, 16), DType::F32, &device)?;

        let err = transition
            .forward(&state, &[])
            .expect_err("empty label list should be rejected");

        assert!(err.to_string().contains("action label count"));
        Ok(())
    }
}
