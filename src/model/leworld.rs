use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{self as nn, Module, ModuleT, VarBuilder};

use super::ActionStateTransition;

/// The post-normalization MLP used by both sides of LeWorldModel.
///
/// Tofy's compressor ends in RMSNorm, so SIGReg cannot directly control its
/// output scale. This projector creates the unconstrained latent space on
/// which prediction and Gaussian regularization are optimized.
struct WorldProjector {
    input: nn::Linear,
    norm: nn::BatchNorm,
    output: nn::Linear,
    hidden_dim: usize,
}

impl WorldProjector {
    fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        let hidden_dim = 2048usize.max(dim);
        Ok(Self {
            input: nn::linear(dim, hidden_dim, vb.pp("input"))?,
            // Candle's BatchNorm computes statistics in F32. Keeping its
            // parameters and running statistics in BF16 causes mixed-dtype
            // updates during training.
            norm: nn::batch_norm(hidden_dim, 1e-5, vb.pp("norm").to_dtype(DType::F32))?,
            output: nn::linear(hidden_dim, dim, vb.pp("output"))?,
            hidden_dim,
        })
    }

    fn forward_t(&self, slots: &Tensor, train: bool) -> Result<Tensor> {
        let (batch, positions, _) = slots.dims3()?;
        let hidden = self
            .input
            .forward(slots)?
            .reshape((batch * positions, self.hidden_dim))?;
        let normalized = self
            .norm
            .forward_t(&hidden.to_dtype(DType::F32)?, train)?
            .to_dtype(slots.dtype())?;
        self.output
            .forward(&normalized.gelu()?)?
            .reshape((batch, positions, ()))
            .map_err(Into::into)
    }
}

/// End-to-end LeWorldModel core adapted to Tofy's discrete retrieval actions.
pub struct LeWorldModel {
    encoder_projector: WorldProjector,
    transition: ActionStateTransition,
    predictor_projector: WorldProjector,
}

impl LeWorldModel {
    pub fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        Ok(Self {
            encoder_projector: WorldProjector::new(vb.pp("encoder_projector"), dim)?,
            transition: ActionStateTransition::new(vb.pp("action_state_transition"), dim)?,
            predictor_projector: WorldProjector::new(vb.pp("predictor_projector"), dim)?,
        })
    }

    pub fn encode(&self, raw_slots: &Tensor, train: bool) -> Result<Tensor> {
        self.encoder_projector.forward_t(raw_slots, train)
    }

    pub fn encode_pair(
        &self,
        raw_state: &Tensor,
        raw_next: &Tensor,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let batch = raw_state.dim(0)?;
        let combined = Tensor::cat(&[raw_state, raw_next], 0)?;
        let encoded = self.encode(&combined, train)?;
        Ok((
            encoded.narrow(0, 0, batch)?,
            encoded.narrow(0, batch, batch)?,
        ))
    }

    pub fn predict(
        &self,
        encoded_state: &Tensor,
        action_labels: &Tensor,
        train: bool,
    ) -> Result<Tensor> {
        let predicted = self
            .transition
            .forward_t(encoded_state, action_labels, train)?;
        self.predictor_projector.forward_t(&predicted, train)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn model_projects_and_predicts_slot_states() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = LeWorldModel::new(VarBuilder::from_varmap(&varmap, DType::F32, &device), 16)?;
        let raw = Tensor::randn(0f32, 1f32, (2, 3, 16), &device)?;
        let actions = Tensor::from_vec(vec![3u32, 3], 2, &device)?;
        let encoded = model.encode(&raw, true)?;
        let predicted = model.predict(&encoded, &actions, true)?;
        assert_eq!(encoded.dims(), raw.dims());
        assert_eq!(predicted.dims(), raw.dims());
        Ok(())
    }
}
