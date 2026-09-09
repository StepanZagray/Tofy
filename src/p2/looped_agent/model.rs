//! Learned looped-transformer controller for the synthetic prerequisite task.

use super::{ACTIONS, META_DIM, OBSERVED_FRAMES, PALETTE, PATCH_COUNT, PATCH_PIXELS, TOKENS};
use anyhow::{bail, ensure, Result};
use candle_core::{DType, Tensor, D};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

const PIXEL_EMBED_DIM: usize = 8;
const MLP_EXPANSION: usize = 4;
const NORM_EPSILON: f64 = 1e-5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopedConfig {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub max_loops: usize,
}

impl LoopedConfig {
    fn validate(&self) -> Result<()> {
        ensure!(self.hidden > 0, "LoopedConfig.hidden must be positive");
        ensure!(self.heads > 0, "LoopedConfig.heads must be positive");
        ensure!(self.layers > 0, "LoopedConfig.layers must be positive");
        ensure!(
            self.max_loops > 0,
            "LoopedConfig.max_loops must be positive"
        );
        ensure!(
            self.hidden.is_multiple_of(self.heads),
            "LoopedConfig.hidden ({}) must be divisible by heads ({})",
            self.hidden,
            self.heads
        );
        Ok(())
    }
}

pub struct LoopedOutput {
    pub policy_logits: Tensor,
    pub value: Tensor,
    pub reward_logits: Tensor,
    pub next_logits: Tensor,
}

/// The exact post-final-RMS tensors consumed by the ordinary linear heads.
/// Retaining these handles adds no model operations, parameters, or detachment.
/// This also retains the full normalized state and its upstream autograd graph
/// until the handles are dropped; support frames are not eagerly copied.
pub struct LoopedFeatures {
    /// Policy/value/reward input, shaped [batch, hidden].
    pub cls: Tensor,
    /// Four successor heads' shared input, shaped [batch, PATCH_COUNT, hidden].
    pub current: Tensor,
    state: Tensor,
}

impl LoopedFeatures {
    /// Returns one exact post-final-RMS frame, shaped [batch, PATCH_COUNT, hidden].
    /// Indices 0..6 are before/after pairs for support transitions 0, 1, and 2;
    /// index 6 is the current frame. Indices >= OBSERVED_FRAMES return an error.
    /// Patches retain their input order, and the result is contiguous: support
    /// frames are packed on demand when a batched slice has gaps. Frame 6 reuses
    /// `current`. No values are detached; a shared-storage slice can keep the full
    /// state allocation alive even after this feature object is dropped.
    pub fn frame(&self, index: usize) -> Result<Tensor> {
        ensure!(
            index < OBSERVED_FRAMES,
            "frame index must be in 0..{OBSERVED_FRAMES}, got {index}"
        );
        if index == OBSERVED_FRAMES - 1 {
            return Ok(self.current.clone());
        }
        Ok(self
            .state
            .narrow(1, 1 + index * PATCH_COUNT, PATCH_COUNT)?
            .contiguous()?)
    }
}

struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn new(hidden: usize, heads: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            query: linear(hidden, hidden, vb.pp("query"))?,
            key: linear(hidden, hidden, vb.pp("key"))?,
            value: linear(hidden, hidden, vb.pp("value"))?,
            output: linear(hidden, hidden, vb.pp("output"))?,
            heads,
            head_dim: hidden / heads,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, tokens, hidden) = input.dims3()?;
        let split_heads = |tensor: Tensor| -> Result<Tensor> {
            Ok(tensor
                .reshape((batch, tokens, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };

        let query = split_heads(self.query.forward(input)?)?;
        let key = split_heads(self.key.forward(input)?)?;
        let value = split_heads(self.value.forward(input)?)?;
        let scores = query
            .matmul(&key.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attended = weights
            .matmul(&value)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, tokens, hidden))?;
        self.output.forward(&attended).map_err(Into::into)
    }
}

struct TransformerBlock {
    attention: SelfAttention,
    mlp_in: Linear,
    mlp_out: Linear,
}

impl TransformerBlock {
    fn new(hidden: usize, heads: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            attention: SelfAttention::new(hidden, heads, vb.pp("attention"))?,
            mlp_in: linear(hidden, hidden * MLP_EXPANSION, vb.pp("mlp_in"))?,
            mlp_out: linear(hidden * MLP_EXPANSION, hidden, vb.pp("mlp_out"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let normalized = rms_norm(input)?;
        let state = input.add(&self.attention.forward(&normalized)?)?;
        let normalized = rms_norm(&state)?;
        let update = self
            .mlp_out
            .forward(&self.mlp_in.forward(&normalized)?.silu()?)?;
        state.add(&update).map_err(Into::into)
    }
}

/// A from-scratch controller whose one learned transformer stack is reused at every loop.
pub struct LoopedAgent {
    config: LoopedConfig,
    palette_embedding: Embedding,
    patch_projection: Linear,
    metadata_projection: Linear,
    readout_token: Tensor,
    blocks: Vec<TransformerBlock>,
    policy_head: Linear,
    value_head: Linear,
    reward_head: Linear,
    next_heads: Vec<Linear>,
}

impl LoopedAgent {
    pub fn new(config: LoopedConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let mut blocks = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            blocks.push(TransformerBlock::new(
                config.hidden,
                config.heads,
                vb.pp(format!("block_{layer}")),
            )?);
        }
        let mut next_heads = Vec::with_capacity(ACTIONS);
        for action in 0..ACTIONS {
            next_heads.push(linear(
                config.hidden,
                PATCH_PIXELS * PALETTE,
                vb.pp(format!("next_head_{action}")),
            )?);
        }

        Ok(Self {
            palette_embedding: embedding(PALETTE, PIXEL_EMBED_DIM, vb.pp("palette_embedding"))?,
            patch_projection: linear(
                PATCH_PIXELS * PIXEL_EMBED_DIM,
                config.hidden,
                vb.pp("patch_projection"),
            )?,
            metadata_projection: linear(META_DIM, config.hidden, vb.pp("metadata_projection"))?,
            readout_token: vb.get((1, 1, config.hidden), "readout_token")?,
            policy_head: linear(config.hidden, ACTIONS, vb.pp("policy_head"))?,
            value_head: linear(config.hidden, 1, vb.pp("value_head"))?,
            reward_head: linear(config.hidden, ACTIONS, vb.pp("reward_head"))?,
            blocks,
            next_heads,
            config,
        })
    }

    pub fn config(&self) -> &LoopedConfig {
        &self.config
    }

    pub fn forward(
        &self,
        patches: &Tensor,
        metadata: &Tensor,
        loops: usize,
    ) -> Result<LoopedOutput> {
        self.forward_with_features(patches, metadata, loops)
            .map(|(output, _)| output)
    }

    /// Runs the same ordinary heads once and also retains their input tensors.
    pub fn forward_with_features(
        &self,
        patches: &Tensor,
        metadata: &Tensor,
        loops: usize,
    ) -> Result<(LoopedOutput, LoopedFeatures)> {
        let features = self.forward_features(patches, metadata, loops)?;
        let batch = patches.dim(0)?;
        let mut next_by_action = Vec::with_capacity(ACTIONS);
        for head in &self.next_heads {
            next_by_action.push(head.forward(&features.current)?.reshape((
                batch,
                PATCH_COUNT,
                PATCH_PIXELS,
                PALETTE,
            ))?);
        }

        let output = LoopedOutput {
            policy_logits: self.policy_head.forward(&features.cls)?,
            value: self.value_head.forward(&features.cls)?,
            reward_logits: self.reward_head.forward(&features.cls)?,
            next_logits: Tensor::stack(&next_by_action, 1)?,
        };
        Ok((output, features))
    }

    /// Differentiable post-RMS body features, without executing ordinary heads.
    pub fn forward_features(
        &self,
        patches: &Tensor,
        metadata: &Tensor,
        loops: usize,
    ) -> Result<LoopedFeatures> {
        self.validate_inputs(patches, metadata, loops)?;
        let batch = patches.dim(0)?;

        // Embedding retains the within-patch axis. Flattening concatenates the
        // 64 position-specific embeddings; it does not pool away any pixel.
        let patch_features = self.palette_embedding.forward(patches)?.reshape((
            batch,
            TOKENS,
            PATCH_PIXELS * PIXEL_EMBED_DIM,
        ))?;
        let patch_features = self.patch_projection.forward(&patch_features)?;
        let metadata_features = self.metadata_projection.forward(metadata)?;
        let observed_tokens = patch_features.add(&metadata_features)?;
        let readout = self
            .readout_token
            .broadcast_as((batch, 1, self.config.hidden))?;
        let recalled_input = Tensor::cat(&[&readout, &observed_tokens], 1)?;

        let mut state = Tensor::zeros_like(&recalled_input)?;
        for _ in 0..loops {
            state = state.add(&recalled_input)?;
            for block in &self.blocks {
                state = block.forward(&state)?;
            }
        }
        let state = rms_norm(&state)?;

        // Narrowing leaves gaps between batches; CUDA linear readouts require
        // packed rows even though the batch-one layout appears contiguous.
        let readout = state.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?;
        let current_start = 1 + TOKENS - PATCH_COUNT;
        let current_patches = state.narrow(1, current_start, PATCH_COUNT)?.contiguous()?;
        Ok(LoopedFeatures {
            cls: readout,
            current: current_patches,
            state,
        })
    }

    fn validate_inputs(&self, patches: &Tensor, metadata: &Tensor, loops: usize) -> Result<()> {
        ensure!(
            (1..=self.config.max_loops).contains(&loops),
            "loops must be in 1..={}, got {loops}",
            self.config.max_loops
        );
        ensure!(
            patches.rank() == 3,
            "patches must have rank 3 [B, {TOKENS}, {PATCH_PIXELS}], got rank {} with shape {:?}",
            patches.rank(),
            patches.dims()
        );
        ensure!(
            metadata.rank() == 3,
            "metadata must have rank 3 [B, {TOKENS}, {META_DIM}], got rank {} with shape {:?}",
            metadata.rank(),
            metadata.dims()
        );
        let (patch_batch, patch_tokens, patch_pixels) = patches.dims3()?;
        let (metadata_batch, metadata_tokens, metadata_dim) = metadata.dims3()?;
        ensure!(patch_batch > 0, "input batch must be positive");
        ensure!(
            (patch_tokens, patch_pixels) == (TOKENS, PATCH_PIXELS),
            "patches must have shape [B, {TOKENS}, {PATCH_PIXELS}], got {:?}",
            patches.dims()
        );
        ensure!(
            (metadata_tokens, metadata_dim) == (TOKENS, META_DIM),
            "metadata must have shape [B, {TOKENS}, {META_DIM}], got {:?}",
            metadata.dims()
        );
        ensure!(
            patch_batch == metadata_batch,
            "patches and metadata batch sizes differ: {patch_batch} vs {metadata_batch}"
        );
        ensure!(
            patches.dtype() == DType::U32,
            "patches must use U32 categorical palette indices, got {:?}",
            patches.dtype()
        );
        ensure!(
            metadata.dtype() == DType::F32,
            "metadata must use F32, got {:?}",
            metadata.dtype()
        );
        ensure!(
            patches.device().same_device(metadata.device()),
            "patches and metadata must be on the same device"
        );
        let max_palette = patches.max_all()?.to_scalar::<u32>()? as usize;
        if max_palette >= PALETTE {
            bail!("patch palette indices must be in 0..{PALETTE}, found {max_palette}");
        }
        Ok(())
    }
}

fn rms_norm(input: &Tensor) -> Result<Tensor> {
    let scale = input
        .sqr()?
        .mean_keepdim(D::Minus1)?
        .clamp(NORM_EPSILON * NORM_EPSILON, f64::INFINITY)?
        .sqrt()?;
    input.broadcast_div(&scale).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    fn tiny_config() -> LoopedConfig {
        LoopedConfig {
            hidden: 16,
            heads: 2,
            layers: 1,
            max_loops: 4,
        }
    }

    fn inputs(device: &Device) -> Result<(Tensor, Tensor)> {
        Ok((
            Tensor::zeros((1, TOKENS, PATCH_PIXELS), DType::U32, device)?,
            Tensor::zeros((1, TOKENS, META_DIM), DType::F32, device)?,
        ))
    }

    fn model(device: &Device) -> Result<(LoopedAgent, VarMap)> {
        let vars = VarMap::new();
        let model = LoopedAgent::new(
            tiny_config(),
            VarBuilder::from_varmap(&vars, DType::F32, device),
        )?;
        Ok((model, vars))
    }

    fn parameter_names_and_count(vars: &VarMap) -> (Vec<String>, usize) {
        let parameters = vars.data().lock().unwrap();
        let names = parameters.keys().cloned().collect();
        let count = parameters.values().map(|var| var.elem_count()).sum();
        (names, count)
    }

    // Frozen numerical body from 8cec1006; independent of the new forwarding API.
    fn legacy_forward(
        model: &LoopedAgent,
        patches: &Tensor,
        metadata: &Tensor,
        loops: usize,
    ) -> Result<LoopedOutput> {
        model.validate_inputs(patches, metadata, loops)?;
        let batch = patches.dim(0)?;

        // Embedding retains the within-patch axis. Flattening concatenates the
        // 64 position-specific embeddings; it does not pool away any pixel.
        let patch_features = model.palette_embedding.forward(patches)?.reshape((
            batch,
            TOKENS,
            PATCH_PIXELS * PIXEL_EMBED_DIM,
        ))?;
        let patch_features = model.patch_projection.forward(&patch_features)?;
        let metadata_features = model.metadata_projection.forward(metadata)?;
        let observed_tokens = patch_features.add(&metadata_features)?;
        let readout = model
            .readout_token
            .broadcast_as((batch, 1, model.config.hidden))?;
        let recalled_input = Tensor::cat(&[&readout, &observed_tokens], 1)?;

        let mut state = Tensor::zeros_like(&recalled_input)?;
        for _ in 0..loops {
            state = state.add(&recalled_input)?;
            for block in &model.blocks {
                state = block.forward(&state)?;
            }
        }
        let state = rms_norm(&state)?;

        // Narrowing leaves gaps between batches; CUDA linear readouts require
        // packed rows even though the batch-one layout appears contiguous.
        let readout = state.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?;
        let current_start = 1 + TOKENS - PATCH_COUNT;
        let current_patches = state.narrow(1, current_start, PATCH_COUNT)?.contiguous()?;
        let mut next_by_action = Vec::with_capacity(ACTIONS);
        for head in &model.next_heads {
            next_by_action.push(head.forward(&current_patches)?.reshape((
                batch,
                PATCH_COUNT,
                PATCH_PIXELS,
                PALETTE,
            ))?);
        }

        Ok(LoopedOutput {
            policy_logits: model.policy_head.forward(&readout)?,
            value: model.value_head.forward(&readout)?,
            reward_logits: model.reward_head.forward(&readout)?,
            next_logits: Tensor::stack(&next_by_action, 1)?,
        })
    }

    #[test]
    fn feature_api_preserves_legacy_outputs_parameters_and_gradients() -> Result<()> {
        fn values(tensor: &Tensor) -> Result<Vec<f32>> {
            Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
        }
        fn objective(output: &LoopedOutput) -> Result<Tensor> {
            Ok(output
                .policy_logits
                .sqr()?
                .mean_all()?
                .add(&output.value.sqr()?.mean_all()?)?
                .add(&output.reward_logits.sqr()?.mean_all()?)?
                .add(&output.next_logits.sqr()?.mean_all()?)?)
        }
        let device = Device::Cpu;
        let (model, vars) = model(&device)?;
        let named = vars
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect::<Vec<_>>();
        let before = named
            .iter()
            .map(|(_, v)| values(v))
            .collect::<Result<Vec<_>>>()?;
        let patches = Tensor::from_vec(
            (0..2 * TOKENS * PATCH_PIXELS)
                .map(|i| (i % 5) as u32)
                .collect::<Vec<_>>(),
            (2, TOKENS, PATCH_PIXELS),
            &device,
        )?;
        let metadata = Tensor::from_vec(
            (0..2 * TOKENS * META_DIM)
                .map(|i| (i % 7) as f32 / 7.0)
                .collect::<Vec<_>>(),
            (2, TOKENS, META_DIM),
            &device,
        )?;
        for loops in [1, 4] {
            let legacy = legacy_forward(&model, &patches, &metadata, loops)?;
            let ordinary = model.forward(&patches, &metadata, loops)?;
            let (exported, features) = model.forward_with_features(&patches, &metadata, loops)?;
            assert_eq!(features.cls.dims(), &[2, model.config.hidden]);
            assert_eq!(
                features.current.dims(),
                &[2, PATCH_COUNT, model.config.hidden]
            );
            // Reconstruct every head from the exported handles, also testing
            // that these handles retain the original upstream gradient graph.
            let next = model
                .next_heads
                .iter()
                .map(|head| {
                    head.forward(&features.current)?.reshape((
                        2,
                        PATCH_COUNT,
                        PATCH_PIXELS,
                        PALETTE,
                    ))
                })
                .collect::<candle_core::Result<Vec<_>>>()?;
            let reconstructed = LoopedOutput {
                policy_logits: model.policy_head.forward(&features.cls)?,
                value: model.value_head.forward(&features.cls)?,
                reward_logits: model.reward_head.forward(&features.cls)?,
                next_logits: Tensor::stack(&next, 1)?,
            };
            let expected = objective(&legacy)?.backward()?;
            for actual in [&ordinary, &exported, &reconstructed] {
                for (a, b) in [
                    (&legacy.policy_logits, &actual.policy_logits),
                    (&legacy.value, &actual.value),
                    (&legacy.reward_logits, &actual.reward_logits),
                    (&legacy.next_logits, &actual.next_logits),
                ] {
                    assert_eq!(values(a)?, values(b)?);
                }
                let gradients = objective(actual)?.backward()?;
                for (name, var) in &named {
                    let a = expected
                        .get(var)
                        .unwrap_or_else(|| panic!("missing legacy gradient {name}"));
                    let b = gradients
                        .get(var)
                        .unwrap_or_else(|| panic!("missing feature gradient {name}"));
                    let av = values(a)?;
                    let bv = values(b)?;
                    assert!(av.iter().chain(&bv).all(|x| x.is_finite()));
                    assert_eq!(av, bv, "gradient parity for {name} at depth {loops}");
                }
            }
        }
        assert_eq!(
            before,
            named
                .iter()
                .map(|(_, v)| values(v))
                .collect::<Result<Vec<_>>>()?
        );
        assert_eq!(named.len(), vars.data().lock().unwrap().len());
        Ok(())
    }

    #[test]
    fn current_frame_is_exact_for_single_and_multiple_batches() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = model(&device)?;
        for batch in [1, 3] {
            let patches = Tensor::from_vec(
                (0..batch * TOKENS * PATCH_PIXELS)
                    .map(|i| (i % PALETTE) as u32)
                    .collect::<Vec<_>>(),
                (batch, TOKENS, PATCH_PIXELS),
                &device,
            )?;
            let metadata = Tensor::zeros((batch, TOKENS, META_DIM), DType::F32, &device)?;
            for loops in [1, 4] {
                let features = model.forward_features(&patches, &metadata, loops)?;
                let current = features.frame(6)?;
                assert_eq!(current.dims(), &[batch, PATCH_COUNT, model.config.hidden]);
                assert!(current.is_contiguous());
                assert_eq!(
                    current.flatten_all()?.to_vec1::<f32>()?,
                    features.current.flatten_all()?.to_vec1::<f32>()?
                );
                assert_eq!(
                    current.flatten_all()?.to_vec1::<f32>()?,
                    features
                        .state
                        .narrow(1, 1 + 6 * PATCH_COUNT, PATCH_COUNT)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn frame_slices_preserve_order_and_reject_out_of_range() -> Result<()> {
        let device = Device::Cpu;
        let (batch, hidden) = (2, 3);
        // Each coordinate is unique, including the CLS row and batch boundary.
        let state = Tensor::arange(0f32, (batch * (1 + TOKENS) * hidden) as f32, &device)?
            .reshape((batch, 1 + TOKENS, hidden))?;
        let features = LoopedFeatures {
            cls: state.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?,
            current: state
                .narrow(1, 1 + 6 * PATCH_COUNT, PATCH_COUNT)?
                .contiguous()?,
            state,
        };
        for frame_index in 0..7 {
            let frame = features.frame(frame_index)?;
            assert_eq!(frame.dims(), &[batch, PATCH_COUNT, hidden]);
            assert!(frame.is_contiguous());
            let values = frame.to_vec3::<f32>()?;
            for (b, patches) in values.iter().enumerate() {
                for (p, channels) in patches.iter().enumerate() {
                    for (h, value) in channels.iter().enumerate() {
                        let offset =
                            (b * (1 + TOKENS) + 1 + frame_index * PATCH_COUNT + p) * hidden + h;
                        assert_eq!(*value, offset as f32);
                    }
                }
            }
        }
        assert!(features.frame(7).is_err());
        assert!(features.frame(usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn every_support_frame_retains_body_gradients() -> Result<()> {
        let device = Device::Cpu;
        let (model, vars) = model(&device)?;
        let patches = Tensor::from_vec(
            (0..2 * TOKENS * PATCH_PIXELS)
                .map(|i| ((i / PATCH_PIXELS) % PALETTE) as u32)
                .collect::<Vec<_>>(),
            (2, TOKENS, PATCH_PIXELS),
            &device,
        )?;
        let metadata = Tensor::from_vec(
            (0..2 * TOKENS * META_DIM)
                .map(|i| (i % 7) as f32 / 7.0)
                .collect::<Vec<_>>(),
            (2, TOKENS, META_DIM),
            &device,
        )?;
        let features = model.forward_features(&patches, &metadata, 2)?;
        let parameters = vars.data().lock().unwrap();
        for index in 0..6 {
            // A single-channel objective avoids the constant RMS squared norm.
            let gradients = features
                .frame(index)?
                .narrow(2, 0, 1)?
                .mean_all()?
                .backward()?;
            for name in [
                "palette_embedding.weight",
                "patch_projection.weight",
                "metadata_projection.weight",
                "block_0.attention.query.weight",
            ] {
                let values = gradients
                    .get(&parameters[name])
                    .unwrap_or_else(|| panic!("missing frame {index} gradient for {name}"))
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert!(values.iter().all(|v| v.is_finite()));
                assert!(
                    values.iter().any(|&v| v != 0.0),
                    "zero frame {index} gradient for {name}"
                );
            }
            assert!(gradients.get(&parameters["policy_head.weight"]).is_none());
            assert!(gradients.get(&parameters["next_head_0.weight"]).is_none());
        }
        Ok(())
    }

    #[test]
    fn output_shapes_match_contract() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = model(&device)?;
        let (patches, metadata) = inputs(&device)?;
        let output = model.forward(&patches, &metadata, 1)?;
        assert_eq!(output.policy_logits.dims(), &[1, ACTIONS]);
        assert_eq!(output.value.dims(), &[1, 1]);
        assert_eq!(output.reward_logits.dims(), &[1, ACTIONS]);
        assert_eq!(
            output.next_logits.dims(),
            &[1, ACTIONS, PATCH_COUNT, PATCH_PIXELS, PALETTE]
        );
        Ok(())
    }

    #[test]
    fn batched_readouts_match_separate_rows() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = model(&device)?;
        let (patches, metadata) = inputs(&device)?;
        let other = Tensor::ones((1, TOKENS, PATCH_PIXELS), DType::U32, &device)?;
        let batch = model.forward(
            &Tensor::cat(&[&patches, &other], 0)?,
            &Tensor::cat(&[&metadata, &metadata], 0)?,
            2,
        )?;
        for (i, input) in [&patches, &other].into_iter().enumerate() {
            let single = model.forward(input, &metadata, 2)?;
            for (joined, separate) in [
                (&batch.policy_logits, &single.policy_logits),
                (&batch.next_logits, &single.next_logits),
            ] {
                let error = joined
                    .narrow(0, i, 1)?
                    .sub(separate)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                assert!(error < 1e-4, "batched prediction differs: {error}");
            }
        }
        Ok(())
    }

    #[test]
    fn inference_depth_reuses_the_same_named_parameters() -> Result<()> {
        let device = Device::Cpu;
        let (model, vars) = model(&device)?;
        let (patches, metadata) = inputs(&device)?;
        let before = parameter_names_and_count(&vars);
        model.forward(&patches, &metadata, 1)?;
        let after_one = parameter_names_and_count(&vars);
        model.forward(&patches, &metadata, 4)?;
        let after_four = parameter_names_and_count(&vars);
        assert_eq!(before, after_one);
        assert_eq!(after_one, after_four);
        assert!(after_four
            .0
            .iter()
            .any(|name| name == "block_0.attention.query.weight"));
        assert!(!after_four.0.iter().any(|name| name.contains("loop_")));
        Ok(())
    }

    #[test]
    fn supervised_multihead_loss_reaches_shared_attention_and_palette() -> Result<()> {
        let device = Device::Cpu;
        let (model, vars) = model(&device)?;
        let (patches, metadata) = inputs(&device)?;
        let output = model.forward(&patches, &metadata, 2)?;
        let losses = [
            output
                .policy_logits
                .sub(&Tensor::full(
                    0.25f32,
                    output.policy_logits.shape(),
                    &device,
                )?)?
                .sqr()?
                .mean_all()?,
            output
                .value
                .sub(&Tensor::full(-0.5f32, output.value.shape(), &device)?)?
                .sqr()?
                .mean_all()?,
            output
                .reward_logits
                .sub(&Tensor::full(
                    0.75f32,
                    output.reward_logits.shape(),
                    &device,
                )?)?
                .sqr()?
                .mean_all()?,
            output
                .next_logits
                .sub(&Tensor::full(0.1f32, output.next_logits.shape(), &device)?)?
                .sqr()?
                .mean_all()?,
        ];
        let loss = losses[0]
            .add(&losses[1])?
            .add(&losses[2])?
            .add(&losses[3])?;
        let gradients = loss.backward()?;
        let parameters = vars.data().lock().unwrap();
        let names_with_gradients = parameters
            .iter()
            .filter(|(_, var)| gradients.get(var).is_some())
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        for name in ["palette_embedding.weight", "block_0.attention.query.weight"] {
            let var = parameters
                .get(name)
                .unwrap_or_else(|| panic!("missing parameter {name}"));
            let values = gradients
                .get(var)
                .unwrap_or_else(|| {
                    panic!("missing gradient for {name}; attached: {names_with_gradients:?}")
                })
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(
                values.iter().any(|value| *value != 0.0),
                "gradient for {name} was zero"
            );
        }
        Ok(())
    }

    #[test]
    fn within_patch_pixel_positions_are_not_averaged_away() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = model(&device)?;
        let (_, metadata) = inputs(&device)?;
        let mut first = vec![0u32; TOKENS * PATCH_PIXELS];
        let current_patch = (TOKENS - PATCH_COUNT) * PATCH_PIXELS;
        first[current_patch] = 1;
        first[current_patch + 1] = 2;
        let mut swapped = first.clone();
        swapped[current_patch] = 2;
        swapped[current_patch + 1] = 1;
        let first = Tensor::from_vec(first, (1, TOKENS, PATCH_PIXELS), &device)?;
        let swapped = Tensor::from_vec(swapped, (1, TOKENS, PATCH_PIXELS), &device)?;
        let first_output = model.forward(&first, &metadata, 1)?.next_logits;
        let swapped_output = model.forward(&swapped, &metadata, 1)?.next_logits;
        let difference = first_output
            .sub(&swapped_output)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(difference.is_finite() && difference > 0.0);
        Ok(())
    }
}
