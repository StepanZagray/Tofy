use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{VarBuilder, VarMap};

use super::shared::EncoderBackbone;

/// Teacher encoder: updated by EMA from the online encoder (no optimizer updates).
pub struct TeacherEncoder {
    inner: EncoderBackbone,
}

impl TeacherEncoder {
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

    pub fn embed_tokens(&self, ids: &Tensor) -> Result<Tensor> {
        self.inner.embed_tokens(ids)
    }
}

/// Copy variables that exist in both varmaps from src to dst.
pub fn copy_matching_vars(src: &VarMap, dst: &mut VarMap) -> Result<()> {
    let src_data = src.data().lock().unwrap();
    let dst_data = dst.data().lock().unwrap();
    for (name, dst_var) in dst_data.iter() {
        if let Some(src_var) = src_data.get(name) {
            dst_var.set(src_var.as_tensor())?;
        }
    }
    Ok(())
}

/// EMA update for variables that exist in both varmaps: dst = decay * dst + (1 - decay) * src.
pub fn ema_update_matching_vars(src: &VarMap, dst: &mut VarMap, decay: f64) -> Result<()> {
    let one_minus = 1.0 - decay;
    let src_data = src.data().lock().unwrap();
    let dst_data = dst.data().lock().unwrap();
    for (name, dst_var) in dst_data.iter() {
        if let Some(src_var) = src_data.get(name) {
            let dst_scaled = (dst_var.as_tensor().clone() / (1.0 / decay))?;
            let src_scaled = (src_var.as_tensor().clone() / (1.0 / one_minus))?;
            let blended = dst_scaled.broadcast_add(&src_scaled)?;
            dst_var.set(&blended)?;
        }
    }
    Ok(())
}
