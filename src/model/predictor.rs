use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{self as nn, VarBuilder};

pub struct Predictor {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Predictor {
    pub fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        let fc1 = nn::linear(dim, dim, vb.pp("fc1"))?;
        let fc2 = nn::linear(dim, dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?.relu()?;
        Ok(self.fc2.forward(&h)?)
    }
}
