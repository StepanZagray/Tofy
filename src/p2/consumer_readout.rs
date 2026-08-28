//! One owned seam from the final spatial prediction to planning-head features.
//!
//! Recurrence, SIGReg, branch learning, and board probes intentionally do not
//! use this module. It exists only for heads that require one `B×C` readout.

use anyhow::{ensure, Result};
use candle_core::{DType, Tensor, D};
use candle_nn::{embedding, linear, ops::softmax, Embedding, Linear, Module, VarBuilder};

use crate::p2::experiment::ConsumerReadoutTopology;
use crate::p2::model::pool_latent;

pub struct ConsumerReadout {
    topology: ConsumerReadoutTopology,
    position_embedding: Option<Embedding>,
    position_value_embedding: Option<Embedding>,
    query_score: Option<Linear>,
    position_indices: Option<Tensor>,
    spatial_side: usize,
}

impl ConsumerReadout {
    pub fn new(
        topology: ConsumerReadoutTopology,
        channels: usize,
        spatial_side: usize,
        positional_value_readout: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let spatial_tokens = spatial_side * spatial_side;
        let position_indices = (topology == ConsumerReadoutTopology::SpatialQuery)
            .then(|| Tensor::arange(0u32, spatial_tokens as u32, vb.device()))
            .transpose()?;
        let (position_embedding, position_value_embedding, query_score) = match topology {
            ConsumerReadoutTopology::GlobalMean => (None, None, None),
            ConsumerReadoutTopology::SpatialQuery => (
                Some(embedding(
                    spatial_tokens,
                    channels,
                    vb.pp("position_embedding"),
                )?),
                positional_value_readout
                    .then(|| embedding(spatial_tokens, channels, vb.pp("position_value_embedding")))
                    .transpose()?,
                Some(linear(channels, 1, vb.pp("query_score"))?),
            ),
        };
        Ok(Self {
            topology,
            position_embedding,
            position_value_embedding,
            query_score,
            position_indices,
            spatial_side,
        })
    }

    /// Read a `B×C×S×S` prediction into the `B×C` interface expected by
    /// planning heads. The spatial-query adapter scores position-augmented
    /// tokens and, when enabled, returns position-augmented values too.
    pub fn forward(&self, spatial: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = spatial.dims4()?;
        ensure!(
            height == self.spatial_side && width == self.spatial_side,
            "consumer readout requires BxCx{}x{}, got BxCx{height}x{width}",
            self.spatial_side,
            self.spatial_side,
        );
        if self.topology == ConsumerReadoutTopology::GlobalMean {
            return pool_latent(spatial);
        }

        let spatial_tokens = self.spatial_side * self.spatial_side;
        let tokens = spatial
            .permute((0, 2, 3, 1))?
            .reshape((batch, spatial_tokens, channels))?;
        let positions = self
            .position_indices
            .as_ref()
            .expect("spatial-query adapter owns position indices");
        let position_embedding = self
            .position_embedding
            .as_ref()
            .expect("spatial-query adapter owns position embeddings")
            .forward(positions)?
            .to_dtype(DType::F32)?;
        let scored_tokens = tokens
            .to_dtype(DType::F32)?
            .broadcast_add(&position_embedding.unsqueeze(0)?)?;
        let logits = self
            .query_score
            .as_ref()
            .expect("spatial-query adapter owns a query scorer")
            .forward(&scored_tokens)?;
        let weights = softmax(&logits.squeeze(D::Minus1)?, 1)?.unsqueeze(D::Minus1)?;
        let values = match &self.position_value_embedding {
            Some(position_values) => tokens
                .to_dtype(DType::F32)?
                .broadcast_add(&position_values.forward(positions)?.to_dtype(DType::F32)?)?,
            None => tokens.to_dtype(DType::F32)?,
        };
        values.broadcast_mul(&weights)?.sum(1).map_err(Into::into)
    }

    pub fn topology(&self) -> ConsumerReadoutTopology {
        self.topology
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::train::reinit_varmap_deterministic;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn global_mean_is_the_legacy_pool_exactly() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let readout = ConsumerReadout::new(
            ConsumerReadoutTopology::GlobalMean,
            3,
            8,
            false,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        let spatial =
            Tensor::arange(0f32, (2 * 3 * 8 * 8) as f32, &device)?.reshape((2, 3, 8, 8))?;
        assert_eq!(
            readout.forward(&spatial)?.to_vec2::<f32>()?,
            pool_latent(&spatial)?.to_vec2::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn spatial_query_returns_finite_batch_channel_rows() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let readout = ConsumerReadout::new(
            ConsumerReadoutTopology::SpatialQuery,
            4,
            8,
            false,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        let spatial = Tensor::ones((2, 4, 8, 8), DType::F32, &device)?;
        let output = readout.forward(&spatial)?;
        assert_eq!(output.dims2()?, (2, 4));
        assert!(output
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .all(f32::is_finite));
        Ok(())
    }

    #[test]
    fn spatial_query_is_position_sensitive_and_receives_gradients() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let readout = ConsumerReadout::new(
            ConsumerReadoutTopology::SpatialQuery,
            4,
            8,
            false,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        reinit_varmap_deterministic(&varmap, 17)?;
        let values = (0..4 * 8 * 8)
            .map(|index| index as f32 / 64.0)
            .collect::<Vec<_>>();
        let spatial = Tensor::from_vec(values, (1, 4, 8, 8), &device)?;
        let shifted = Tensor::cat(&[&spatial.narrow(3, 1, 7)?, &spatial.narrow(3, 0, 1)?], 3)?;
        assert_ne!(
            readout.forward(&spatial)?.to_vec2::<f32>()?,
            readout.forward(&shifted)?.to_vec2::<f32>()?
        );

        let gradients = readout.forward(&spatial)?.sqr()?.mean_all()?.backward()?;
        let data = varmap.data().lock().unwrap();
        let nonzero = data
            .iter()
            .filter(|(name, _)| {
                name.starts_with("query_score.") || name.starts_with("position_embedding.")
            })
            .filter_map(|(_, var)| gradients.get(var.as_tensor()))
            .map(|gradient| {
                gradient
                    .abs()?
                    .sum_all()?
                    .to_scalar::<f32>()
                    .map_err(Into::into)
            })
            .collect::<Result<Vec<_>>>()?;
        assert!(!nonzero.is_empty());
        assert!(nonzero.into_iter().any(|value| value > 0.0));
        Ok(())
    }
}
