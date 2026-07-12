use anyhow::Result;
use candle_core::{Module, Tensor, D};
use candle_nn::{self as nn, VarBuilder};

use crate::model::attention::CrossAttention;
use crate::util;

struct ReconstructionBlock {
    norm: nn::LayerNorm,
    attention: CrossAttention,
    ff_norm: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
}

impl ReconstructionBlock {
    fn new(vb: VarBuilder<'_>, dim: usize) -> Result<Self> {
        Ok(Self {
            norm: nn::layer_norm(dim, 1e-5, vb.pp("norm"))?,
            attention: CrossAttention::new(vb.pp("attention"), dim, dim, 8)?,
            ff_norm: nn::layer_norm(dim, 1e-5, vb.pp("ff_norm"))?,
            ff1: nn::linear(dim, dim * 4, vb.pp("ff1"))?,
            ff2: nn::linear(dim * 4, dim, vb.pp("ff2"))?,
        })
    }

    fn forward(&self, x: &Tensor, slots: &Tensor) -> Result<Tensor> {
        let x = (x + self
            .attention
            .forward(&util::layer_norm_diff(&self.norm, x)?, slots)?)?;
        let ff = self
            .ff1
            .forward(&util::layer_norm_diff(&self.ff_norm, &x)?)?
            .gelu()?;
        (x + self.ff2.forward(&ff)?).map_err(Into::into)
    }
}

/// Training-only decoder which forces slots to retain token-level document information.
pub struct KnowledgeReconstructionHead {
    positions: nn::Embedding,
    input_proj: nn::Linear,
    blocks: [ReconstructionBlock; 2],
    output_norm: nn::RmsNorm,
    output: nn::Linear,
    max_seq: usize,
}

impl KnowledgeReconstructionHead {
    pub fn new(vb: VarBuilder<'_>, slot_dim: usize, vocab: usize, max_seq: usize) -> Result<Self> {
        let dim = slot_dim.max(64);
        Ok(Self {
            positions: nn::embedding(max_seq, dim, vb.pp("positions"))?,
            input_proj: nn::linear(slot_dim, dim, vb.pp("input_proj"))?,
            blocks: [
                ReconstructionBlock::new(vb.pp("blocks.0"), dim)?,
                ReconstructionBlock::new(vb.pp("blocks.1"), dim)?,
            ],
            output_norm: nn::rms_norm(dim, 1e-6, vb.pp("output_norm"))?,
            output: nn::linear(dim, vocab, vb.pp("output"))?,
            max_seq,
        })
    }

    pub fn forward(&self, slots: &Tensor, sequence_len: usize) -> Result<Tensor> {
        let batch = slots.dim(0)?;
        let sequence_len = sequence_len.min(self.max_seq);
        let ids = Tensor::arange(0u32, sequence_len as u32, slots.device())?.unsqueeze(0)?;
        let positions = self.positions.forward(&ids)?;
        let dim = positions.dim(2)?;
        let mut x = positions.broadcast_as((batch, sequence_len, dim))?;
        let memory = self.input_proj.forward(slots)?;
        for block in &self.blocks {
            x = block.forward(&x, &memory)?;
        }
        self.output
            .forward(&self.output_norm.forward_diff(&x)?)
            .map_err(Into::into)
    }
}

pub fn association_loss(task_slots: &Tensor, doc_slots: &Tensor) -> Result<Tensor> {
    let logits = association_logits(task_slots, doc_slots)?;
    let labels = Tensor::arange(0u32, logits.dim(0)? as u32, logits.device())?;
    let task_to_doc = nn::loss::cross_entropy(&logits, &labels)?;
    let doc_to_task = nn::loss::cross_entropy(&logits.t()?, &labels)?;
    task_to_doc
        .broadcast_add(&doc_to_task)?
        .affine(0.5, 0.0)
        .map_err(Into::into)
}

fn association_logits(task_slots: &Tensor, doc_slots: &Tensor) -> Result<Tensor> {
    let (batch, slots, dim) = task_slots.dims3()?;
    if doc_slots.dims3()? != (batch, slots, dim) {
        anyhow::bail!("association tensors must have identical [batch, slots, dim] shapes");
    }
    // Preserve slot identity. Mean pooling erased most of the compressor's
    // structured signal and left retrieval at chance.
    let task = task_slots.reshape((batch, slots * dim))?;
    let docs = doc_slots.reshape((batch, slots * dim))?;
    let task_norm = task
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-6, f64::INFINITY)?;
    let doc_norm = docs
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-6, f64::INFINITY)?;
    let task = task.broadcast_div(&task_norm)?;
    let docs = docs.broadcast_div(&doc_norm)?;
    task.matmul(&docs.t()?)?
        .affine(1.0 / 0.07, 0.0)
        .map_err(Into::into)
}

pub fn association_top1_accuracy(task_slots: &Tensor, doc_slots: &Tensor) -> Result<f32> {
    let predictions = association_logits(task_slots, doc_slots)?
        .argmax(D::Minus1)?
        .to_vec1::<u32>()?;
    let correct = predictions
        .iter()
        .enumerate()
        .filter(|(index, prediction)| **prediction as usize == *index)
        .count();
    Ok(correct as f32 / predictions.len().max(1) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn association_top1_is_perfect_for_distinct_identical_pairs() -> Result<()> {
        let slots = Tensor::from_vec(
            vec![1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            (2, 2, 2),
            &Device::Cpu,
        )?;
        assert_eq!(association_top1_accuracy(&slots, &slots)?, 1.0);
        Ok(())
    }
}
