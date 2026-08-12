//! Exact latent transition consumed by recurrence, probes, and factual objectives.

use crate::p2::model::pool_latent;
use anyhow::{ensure, Result};
use candle_core::Tensor;

/// One aligned current/predicted/target latent population.
///
/// Constructing this value once keeps consumer-facing objectives from quietly
/// switching to a projector or another representation seam.
#[derive(Clone, Debug)]
pub struct ConsumerTransition {
    current: Tensor,
    predicted: Tensor,
    target: Tensor,
}

impl ConsumerTransition {
    pub fn try_new(current: Tensor, predicted: Tensor, target: Tensor) -> Result<Self> {
        ensure!(
            current.dims() == predicted.dims() && current.dims() == target.dims(),
            "consumer latent shapes must match: current={:?}, predicted={:?}, target={:?}",
            current.dims(),
            predicted.dims(),
            target.dims()
        );
        ensure!(current.rank() == 4, "consumer latents must be BxCxHxW");
        ensure!(current.dim(0)? > 0, "consumer transition batch is empty");
        Ok(Self {
            current,
            predicted,
            target,
        })
    }

    pub fn current(&self) -> &Tensor {
        &self.current
    }
    pub fn predicted(&self) -> &Tensor {
        &self.predicted
    }
    pub fn target(&self) -> &Tensor {
        &self.target
    }

    pub fn batch_len(&self) -> Result<usize> {
        self.current.dim(0).map_err(Into::into)
    }

    pub fn pooled_predicted_displacement(&self) -> Result<Tensor> {
        pool_latent(&self.predicted.sub(&self.current)?)
    }

    pub fn pooled_target_displacement(&self) -> Result<Tensor> {
        pool_latent(&self.target.sub(&self.current)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn rejects_misaligned_consumer_populations() -> Result<()> {
        let device = Device::Cpu;
        let current = Tensor::zeros((2, 3, 8, 8), DType::F32, &device)?;
        let predicted = Tensor::zeros((1, 3, 8, 8), DType::F32, &device)?;
        assert!(ConsumerTransition::try_new(current.clone(), predicted, current).is_err());
        Ok(())
    }
}
