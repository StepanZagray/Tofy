//! TRM-inspired pixel world model with PTRM stochastic inference.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor, D};
use candle_nn::{
    conv2d, embedding, linear, Conv2d, Conv2dConfig, Embedding, Linear, Module, VarBuilder,
};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::p2::consumer_readout::ConsumerReadout;
use crate::p2::data::TransitionSample;
use crate::p2::experiment::ConsumerReadoutTopology;
use crate::p2::grounding::{
    ExactPatchGrounding, PatchGroundingLoss, PatchGroundingMode, PatchHistogramGrounding,
};
use crate::p2::representation::RepresentationSeam;

/// Fixed observation resolution required by the pixel encoder.
pub const FRAME_SIDE: usize = 64;
/// Embedded palette feature width.
pub const PIXEL_EMB_DIM: usize = 8;
/// Number of discrete ARC palette values accepted by the pixel embedding.
pub const PALETTE_SIZE: usize = 16;
/// Inclusive action ID range `0..=6`.
/// Embedding rows for official action ids `0..=7` (`0` unused; `1..=7` = ACTION1..ACTION7).
pub const ACTION_VOCAB: usize = 8;

/// Six family indicators plus public parameters/order slots.
pub const DEFAULT_GOAL_DIM: usize = 19;
/// Default event head width: noop / goal_satisfied / goal_failed / exhausted.
pub const DEFAULT_NUM_EVENTS: usize = 4;

pub const EVENT_NOOP: usize = 0;
pub const EVENT_GOAL_SATISFIED: usize = 1;
pub const EVENT_GOAL_FAILED: usize = 2;
pub const EVENT_EXHAUSTED: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecursionDepth {
    pub inner_steps: usize,
    pub outer_steps: usize,
}

/// One encoder pass shared by normalized dynamics and experimental SIGReg geometry.
pub struct TrainingEncodedPair {
    pub current: Tensor,
    pub next: Tensor,
    pub current_raw: Tensor,
    pub next_raw: Tensor,
    pub projected_sigreg: Option<Tensor>,
}

impl RecursionDepth {
    pub fn from_config(cfg: &ModelConfig) -> Self {
        Self {
            inner_steps: cfg.inner_steps,
            outer_steps: cfg.outer_steps,
        }
    }
}

/// Training vs eval behavior inside [`WorldModel::run_recursion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecursionOpts {
    /// Populate [`ForwardOutput::recursion_probes`] (eval diagnostics; syncs GPU).
    pub record_probes: bool,
    /// Retain every outer `y` in [`ForwardOutput::steps`] (needed for TRM deep supervision).
    pub store_intermediate_steps: bool,
}

impl RecursionOpts {
    pub const EVAL: Self = Self {
        record_probes: true,
        store_intermediate_steps: true,
    };

    pub fn training(supervise_last_outer_only: bool) -> Self {
        Self {
            record_probes: false,
            store_intermediate_steps: !supervise_last_outer_only,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub frame_side: usize,
    pub hidden_dim: usize,
    pub action_dim: usize,
    pub goal_dim: usize,
    pub inner_steps: usize,
    pub outer_steps: usize,
    pub num_events: usize,
    /// Pre-LN residual on dynamics: `y = rms_norm(y + block(y + z))`.
    #[serde(default)]
    pub residual_y_update: bool,
    /// Initialize recursion `y` from the incoming state latent instead of zeros.
    #[serde(default)]
    pub warm_start_y: bool,
    /// Run conv encoder/decoder paths in BF16 (norms/losses stay F32).
    #[serde(default)]
    pub bf16_conv: bool,
    /// Feed pre-RMS pooled encoder features through a learned SIGReg projector.
    #[serde(default)]
    pub sigreg_projector: bool,
    /// Projected embedding width used only when `sigreg_projector` is enabled.
    #[serde(default = "default_sigreg_projector_dim")]
    pub sigreg_projector_dim: usize,
    /// Use a spatial ACTION6 coordinate field instead of the legacy coordinate broadcast.
    #[serde(default)]
    pub spatial_action_field: bool,
    /// Preserve the global ACTION6 coordinate broadcast and add the spatial
    /// field as a bounded residual. False preserves historical V2 semantics.
    #[serde(default)]
    pub spatial_action_residual: bool,
    #[serde(default = "default_spatial_action_residual_scale")]
    pub spatial_action_residual_scale: f64,
    /// Instantiate the action-faithful world-core-v2 heads. V2 checkpoints are
    /// intentionally incompatible with legacy training checkpoints.
    #[serde(default)]
    pub world_core_v2: bool,
    /// Explicit experiment schema marker. V3 retains the V2 heads/topology.
    #[serde(default)]
    pub world_core_v3: bool,
    /// Paper-grounded successor recipe. Unlike V2/V3, V4 does not enable the
    /// experimental factual-branch heads.
    #[serde(default)]
    pub world_core_v4: bool,
    /// Readout used only by Q/event/reliability planning heads.
    #[serde(default)]
    pub consumer_readout: ConsumerReadoutTopology,
}

fn default_sigreg_projector_dim() -> usize {
    128
}

fn default_spatial_action_residual_scale() -> f64 {
    0.25
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            frame_side: FRAME_SIDE,
            hidden_dim: 128,
            action_dim: 32,
            goal_dim: DEFAULT_GOAL_DIM,
            inner_steps: 2,
            outer_steps: 2,
            num_events: DEFAULT_NUM_EVENTS,
            residual_y_update: false,
            warm_start_y: false,
            bf16_conv: false,
            sigreg_projector: false,
            sigreg_projector_dim: default_sigreg_projector_dim(),
            spatial_action_field: false,
            spatial_action_residual: false,
            spatial_action_residual_scale: default_spatial_action_residual_scale(),
            world_core_v2: false,
            world_core_v3: false,
            world_core_v4: false,
            consumer_readout: ConsumerReadoutTopology::GlobalMean,
        }
    }
}

/// Per-outer-step recursion diagnostics (detached scalars for eval logging).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursionStepProbe {
    pub outer_step: usize,
    pub mean_residual_norm: f64,
    pub mean_latent_norm: f64,
    pub mean_amplification: f64,
}

pub const PREFIX_HORIZONS: [usize; 5] = [1, 2, 4, 8, 16];

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        if self.frame_side != FRAME_SIDE {
            bail!(
                "ModelConfig.frame_side must be {FRAME_SIDE}, got {}",
                self.frame_side
            );
        }
        if self.hidden_dim == 0
            || self.action_dim == 0
            || self.goal_dim == 0
            || self.inner_steps == 0
            || self.outer_steps == 0
            || self.num_events == 0
        {
            bail!("ModelConfig dims/steps must be positive");
        }
        if self.sigreg_projector && self.sigreg_projector_dim < 2 {
            bail!("sigreg_projector_dim must be >= 2 when the projector is enabled");
        }
        if self.world_core_v3 && !self.world_core_v2 {
            bail!("world_core_v3 requires the world_core_v2 base topology");
        }
        if self.world_core_v4 && (self.world_core_v2 || self.world_core_v3) {
            bail!("world_core_v4 cannot be composed with V2/V3");
        }
        if self.world_core_v4
            && (!self.spatial_action_field
                || self.consumer_readout != ConsumerReadoutTopology::SpatialQuery)
        {
            bail!("world_core_v4 requires spatial action fields and SpatialQuery readout");
        }
        if self.spatial_action_residual && (!self.world_core_v3 || !self.spatial_action_field) {
            bail!("spatial_action_residual requires world_core_v3 and spatial_action_field");
        }
        if !self.spatial_action_residual_scale.is_finite()
            || self.spatial_action_residual_scale <= 0.0
            || self.spatial_action_residual_scale > 1.0
        {
            bail!("spatial_action_residual_scale must be finite and in (0,1]");
        }
        Ok(())
    }
}

fn standard_normal(rng: &mut rand::rngs::StdRng) -> f32 {
    let u1 = rng.random_range(f32::EPSILON..1.0f32);
    let u2 = rng.random_range(0.0f32..1.0f32);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn seeded_gaussian_like(template: &Tensor, sigma: f64, seed: u64) -> Result<Tensor> {
    let shape = template.dims();
    if shape.is_empty() {
        return Ok(template.zeros_like()?);
    }
    let batch = shape[0];
    let per_sample = template.elem_count() / batch.max(1);
    let scale = sigma as f32;
    let mut data = vec![0f32; template.elem_count()];
    data.par_chunks_mut(per_sample)
        .enumerate()
        .for_each(|(b, row)| {
            let sample_seed = seed.wrapping_add((b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut rng = rand::rngs::StdRng::seed_from_u64(sample_seed);
            for value in row {
                *value = standard_normal(&mut rng) * scale;
            }
        });
    Tensor::from_vec(data, shape, template.device()).map_err(Into::into)
}

/// Side length of square input patches (`64 / PATCH_SIZE` grid).
pub const PATCH_SIZE: usize = 8;
/// Spatial latent grid side (matches patch grid; dynamics stay on `B×C×8×8`).
pub const LATENT_GRID: usize = FRAME_SIDE / PATCH_SIZE;

/// Flatten `B×C×H×W` (or pass through `B×D`) for SIGReg / identifiability.
pub fn flatten_latent(z: &Tensor) -> Result<Tensor> {
    match z.rank() {
        4 => {
            let (b, c, h, w) = z.dims4()?;
            Ok(z.reshape((b, c * h * w))?)
        }
        2 => Ok(z.clone()),
        r => bail!("flatten_latent expected rank 2 or 4, got {r}"),
    }
}

/// Global mean pool to `B×C` for Q/event heads.
pub fn pool_latent(z: &Tensor) -> Result<Tensor> {
    match z.rank() {
        4 => Ok(z.mean(D::Minus1)?.mean(D::Minus1)?),
        2 => Ok(z.clone()),
        r => bail!("pool_latent expected rank 2 or 4, got {r}"),
    }
}

const LATENT_RMS_EPS: f32 = 1e-6;

/// Per-batch RMS normalization on `B×C×H×W` latents (scale control, no extra params).
pub fn rms_norm_latent(z: &Tensor) -> Result<Tensor> {
    match z.rank() {
        4 => {
            let (batch, _, _, _) = z.dims4()?;
            let flat = z.flatten_from(1)?;
            let rms = flat
                .sqr()?
                .mean_keepdim(D::Minus1)?
                .sqrt()?
                .clamp(LATENT_RMS_EPS as f64, f64::INFINITY)?;
            let scale = rms.reshape((batch, 1, 1, 1))?;
            z.broadcast_div(&scale).map_err(Into::into)
        }
        2 => {
            let (batch, _) = z.dims2()?;
            let rms = z
                .sqr()?
                .mean_keepdim(D::Minus1)?
                .sqrt()?
                .clamp(LATENT_RMS_EPS as f64, f64::INFINITY)?;
            let scale = rms.reshape((batch, 1))?;
            z.broadcast_div(&scale).map_err(Into::into)
        }
        r => bail!("rms_norm_latent expected rank 2 or 4, got {r}"),
    }
}

/// Per-sample mean squared error → `B×1`.
pub fn latent_mse_per_sample(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    let diff = pred.sub(target)?.sqr()?;
    match diff.rank() {
        4 => Ok(diff
            .flatten_from(1)?
            .mean(D::Minus1)?
            .unsqueeze(D::Minus1)?),
        2 => Ok(diff.mean_keepdim(D::Minus1)?),
        r => bail!("latent_mse_per_sample expected rank 2 or 4, got {r}"),
    }
}

/// Conv residual block on spatial latents: `h + Conv(h)`.
struct GridResidualBlock {
    c1: Conv2d,
    c2: Conv2d,
}

impl GridResidualBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            c1: conv2d(channels, channels, 3, cfg, vb.pp("c1"))?,
            c2: conv2d(channels, channels, 3, cfg, vb.pp("c2"))?,
        })
    }

    fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let delta = self.c2.forward(&self.c1.forward(h)?.silu()?)?;
        h.add(&delta).map_err(Into::into)
    }
}

struct GridEncoder {
    patch: Conv2d,
    c2: Conv2d,
    c3: Conv2d,
    proj: Conv2d,
}

impl GridEncoder {
    fn new(cell_dim: usize, vb: VarBuilder) -> Result<Self> {
        if !FRAME_SIDE.is_multiple_of(PATCH_SIZE) {
            bail!("FRAME_SIDE must be divisible by PATCH_SIZE");
        }
        if LATENT_GRID != FRAME_SIDE / PATCH_SIZE {
            bail!("LATENT_GRID mismatch");
        }
        let patch_cfg = Conv2dConfig {
            stride: PATCH_SIZE,
            ..Default::default()
        };
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            patch: conv2d(PIXEL_EMB_DIM, 32, PATCH_SIZE, patch_cfg, vb.pp("patch"))?,
            c2: conv2d(32, 64, 3, conv_cfg, vb.pp("c2"))?,
            c3: conv2d(64, 64, 3, conv_cfg, vb.pp("c3"))?,
            proj: conv2d(64, cell_dim, 1, Default::default(), vb.pp("proj"))?,
        })
    }

    /// `frames`: palette indices `B×1×H×W` (u8/f32) or embedded `B×PIXEL_EMB_DIM×H×W`.
    fn forward(&self, frames: &Tensor, bf16: bool) -> Result<Tensor> {
        let (b, c, h, w) = frames.dims4()?;
        if h != FRAME_SIDE || w != FRAME_SIDE {
            bail!("expected frames spatial {FRAME_SIDE}x{FRAME_SIDE}, got {h}x{w}");
        }
        let input = if c == 1 {
            bail!("use WorldModel::encode_state for index tensors");
        } else if c != PIXEL_EMB_DIM {
            bail!(
                "expected embedded frames Bx{PIXEL_EMB_DIM}x{FRAME_SIDE}x{FRAME_SIDE}, got {b}x{c}x{h}x{w}"
            );
        } else if bf16 {
            frames.to_dtype(DType::BF16)?
        } else {
            frames.clone()
        };
        let mut h = self.patch.forward(&input)?.silu()?;
        h = self.c2.forward(&h)?.silu()?;
        h = self.c3.forward(&h)?.silu()?;
        h = self.proj.forward(&h)?;
        if bf16 {
            h.to_dtype(DType::F32).map_err(Into::into)
        } else {
            Ok(h)
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepOutput {
    pub y: Tensor,
    pub event_logits: Option<Tensor>,
    pub q_logit: Option<Tensor>,
}

#[derive(Debug, Clone)]
pub struct ForwardOutput {
    /// Trace at each outer recursion step.
    pub steps: Vec<StepOutput>,
    pub y: Tensor,
    pub event_logits: Tensor,
    pub q_logit: Tensor,
    pub reliability_logit: Tensor,
    #[allow(clippy::type_complexity)]
    pub recursion_probes: Vec<RecursionStepProbe>,
}

#[derive(Debug, Clone)]
pub struct LatentRecursionOutput {
    /// Retained outer-step latents according to [`RecursionOpts`].
    pub steps: Vec<Tensor>,
    pub y: Tensor,
    #[allow(clippy::type_complexity)]
    pub recursion_probes: Vec<RecursionStepProbe>,
}

/// Detached tensors from one forward pass, keyed by the stable evaluation seams.
#[derive(Debug, Clone)]
pub struct RepresentationDiagnosticOutput {
    pub seams: BTreeMap<RepresentationSeam, Tensor>,
}

#[derive(Debug, Clone)]
pub struct PtrmTrajectory {
    pub steps: Vec<StepOutput>,
    pub y: Tensor,
    pub event_logits: Tensor,
    pub q_logit: Tensor,
}

#[derive(Debug, Clone)]
pub struct PtrmRankingTrajectory {
    pub y: Tensor,
    pub q_logit: Tensor,
}

#[derive(Debug, Clone)]
pub struct PtrmOutput {
    /// All K stochastic trajectories (retained).
    pub trajectories: Vec<PtrmTrajectory>,
    /// Maximum-Q trajectory index for each batch item.
    pub best_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct PtrmConfig {
    pub k: usize,
    pub sigma: f64,
    pub seed: Option<u64>,
}

/// Per-sample index of the trajectory with the highest Q logit.
pub fn best_q_indices(q_logits: &[Tensor]) -> Result<Vec<usize>> {
    if q_logits.is_empty() {
        bail!("best_q_indices requires at least one Q logit tensor");
    }
    let first = q_logits[0].to_dtype(DType::F32)?;
    let (batch, width) = first.dims2()?;
    if width != 1 {
        bail!("Q logits must have shape Bx1");
    }
    let mut best_indices = vec![0usize; batch];
    let mut best_values = first.squeeze(1)?.to_vec1::<f32>()?;
    for (trajectory, q) in q_logits.iter().enumerate().skip(1) {
        let q = q.to_dtype(DType::F32)?;
        if q.dims2()? != (batch, 1) {
            bail!("all Q logits must share shape Bx1");
        }
        for (sample, value) in q.squeeze(1)?.to_vec1::<f32>()?.into_iter().enumerate() {
            if value > best_values[sample] {
                best_values[sample] = value;
                best_indices[sample] = trajectory;
            }
        }
    }
    Ok(best_indices)
}

pub struct WorldModel {
    config: ModelConfig,
    pixel_emb: Embedding,
    encoder: GridEncoder,
    action_emb: Embedding,
    action_proj: Linear,
    coord_proj: Linear,
    /// Optional `B×4×8×8` ACTION6 coordinate field projection.
    spatial_action_proj: Option<Conv2d>,
    goal_proj: Linear,
    block: GridResidualBlock,
    event_head: Linear,
    /// PTRM trajectory ranking score.
    q_head: Linear,
    /// Calibrated reliability / error prediction (Phase D).
    reliability_head: Linear,
    consumer_readout: ConsumerReadout,
    /// Direct one-step prefix delta from pooled state + action.
    prefix_head: Linear,
    spatial_prefix_head: Option<Conv2d>,
    action_decoder: Option<Linear>,
    coordinate_decoder: Option<Linear>,
    /// Optional pre-RMS `B×C` → `B×D` projection used only by SIGReg.
    sigreg_projector: Option<Linear>,
    /// Present in every arm; its loss is switched by the training contract.
    patch_histogram_grounding: PatchHistogramGrounding,
    exact_patch_grounding: Option<ExactPatchGrounding>,
}

impl WorldModel {
    pub fn new(cfg: ModelConfig, vb: VarBuilder) -> Result<Self> {
        cfg.validate()?;
        let sigreg_projector = cfg
            .sigreg_projector
            .then(|| {
                linear(
                    cfg.hidden_dim,
                    cfg.sigreg_projector_dim,
                    vb.pp("sigreg_projector"),
                )
            })
            .transpose()?;
        let spatial_action_proj = (cfg.world_core_v2 || cfg.spatial_action_field)
            .then(|| {
                conv2d(
                    4,
                    cfg.hidden_dim,
                    1,
                    Default::default(),
                    vb.pp("spatial_action_proj"),
                )
            })
            .transpose()?;
        let spatial_prefix_head = cfg
            .world_core_v2
            .then(|| {
                conv2d(
                    cfg.hidden_dim * 2,
                    cfg.hidden_dim,
                    3,
                    Conv2dConfig {
                        padding: 1,
                        ..Default::default()
                    },
                    vb.pp("spatial_prefix_head"),
                )
            })
            .transpose()?;
        let action_decoder = cfg
            .world_core_v2
            .then(|| linear(cfg.hidden_dim, ACTION_VOCAB, vb.pp("action_decoder")))
            .transpose()?;
        let coordinate_decoder = cfg
            .world_core_v2
            .then(|| linear(cfg.hidden_dim, 2, vb.pp("coordinate_decoder")))
            .transpose()?;
        Ok(Self {
            pixel_emb: embedding(PALETTE_SIZE, PIXEL_EMB_DIM, vb.pp("pixel_emb"))?,
            encoder: GridEncoder::new(cfg.hidden_dim, vb.pp("encoder"))?,
            action_emb: embedding(ACTION_VOCAB, cfg.action_dim, vb.pp("action_emb"))?,
            action_proj: linear(cfg.action_dim, cfg.hidden_dim, vb.pp("action_proj"))?,
            coord_proj: linear(2, cfg.hidden_dim, vb.pp("coord_proj"))?,
            spatial_action_proj,
            goal_proj: linear(cfg.goal_dim, cfg.hidden_dim, vb.pp("goal_proj"))?,
            block: GridResidualBlock::new(cfg.hidden_dim, vb.pp("block"))?,
            event_head: linear(cfg.hidden_dim * 2, cfg.num_events, vb.pp("event_head"))?,
            q_head: linear(cfg.hidden_dim, 1, vb.pp("q_head"))?,
            reliability_head: linear(cfg.hidden_dim, 1, vb.pp("reliability_head"))?,
            consumer_readout: ConsumerReadout::new(
                cfg.consumer_readout,
                cfg.hidden_dim,
                vb.pp("consumer_readout"),
            )?,
            prefix_head: linear(cfg.hidden_dim * 2, cfg.hidden_dim, vb.pp("prefix_head"))?,
            spatial_prefix_head,
            action_decoder,
            coordinate_decoder,
            sigreg_projector,
            patch_histogram_grounding: PatchHistogramGrounding::new(
                cfg.hidden_dim,
                vb.pp("grounding_head"),
            )?,
            exact_patch_grounding: cfg
                .world_core_v4
                .then(|| ExactPatchGrounding::new(cfg.hidden_dim, vb.pp("exact_grounding_head")))
                .transpose()?,
            config: cfg,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn patch_histogram_grounding_loss(
        &self,
        predicted: &Tensor,
        target: &Tensor,
        samples: &[TransitionSample],
        mode: PatchGroundingMode,
    ) -> Result<PatchGroundingLoss> {
        self.patch_histogram_grounding
            .loss(predicted, target, samples, mode)
    }

    pub fn exact_grounding_loss(&self, latents: &Tensor, frames: &Tensor) -> Result<Tensor> {
        self.exact_patch_grounding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("exact grounding requires world-core-v4"))?
            .loss(latents, frames)
    }

    /// Decode a spatial state into `B×63×64×16` palette logits. This is the
    /// canonical semantic seam used by Full V4 evaluation; it never includes
    /// the synthetic status row.
    pub fn exact_gameplay_logits(&self, latents: &Tensor) -> Result<Tensor> {
        self.exact_patch_grounding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("exact grounding requires world-core-v4"))?
            .gameplay_logits(latents)
    }

    pub fn exact_transition_correctness(
        &self,
        predicted: &Tensor,
        current_frames: &Tensor,
        next_frames: &Tensor,
    ) -> Result<Tensor> {
        self.exact_patch_grounding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("exact grounding requires world-core-v4"))?
            .transition_correctness(predicted, current_frames, next_frames)
    }

    /// Full V4's single canonical BxC state. It is both regularized and
    /// consumed by transition/observer heads; this is a Tofy adaptation, not
    /// the separate loss-only projectors used by LeWorldModel.
    pub fn canonical_representation(&self, spatial: &Tensor) -> Result<Tensor> {
        rms_norm_latent(&self.consumer_readout.forward(spatial)?)
    }

    /// Encode palette-index frames into the shared latent space.
    pub fn encode_state(&self, frames: &Tensor) -> Result<Tensor> {
        let embedded = self.embed_frames(frames)?;
        rms_norm_latent(&self.encoder.forward(&embedded, self.config.bf16_conv)?)
    }

    fn encode_state_pair_raw(
        &self,
        frames: &Tensor,
        next_frames: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let batch = frames.dim(0)?;
        if next_frames.dim(0)? != batch {
            bail!(
                "encode_state_pair: batch mismatch {} vs {}",
                batch,
                next_frames.dim(0)?
            );
        }
        let both = Tensor::cat(&[frames, next_frames], 0)?;
        let embedded = self.embed_frames(&both)?;
        let encoded = self.encoder.forward(&embedded, self.config.bf16_conv)?;
        let current = encoded.narrow(0, 0, batch)?;
        let next = encoded.narrow(0, batch, batch)?;
        Ok((current, next))
    }

    /// Encode current and next frames in one conv pass (`2B` batch), then split.
    pub fn encode_state_pair(
        &self,
        frames: &Tensor,
        next_frames: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (current, next) = self.encode_state_pair_raw(frames, next_frames)?;
        Ok((rms_norm_latent(&current)?, rms_norm_latent(&next)?))
    }

    /// Return normalized dynamics latents together with their pre-RMS source tensors
    /// and optional `T×B×D` projector embeddings used only by SIGReg treatments.
    pub fn encode_state_pair_for_training(
        &self,
        frames: &Tensor,
        next_frames: &Tensor,
    ) -> Result<TrainingEncodedPair> {
        let (current_raw, next_raw) = self.encode_state_pair_raw(frames, next_frames)?;
        let projected = self
            .sigreg_projector
            .as_ref()
            .map(|projector| -> Result<Tensor> {
                let current = projector.forward(&pool_latent(&current_raw)?)?;
                let next = projector.forward(&pool_latent(&next_raw)?)?;
                Ok(Tensor::stack(&[current, next], 0)?)
            })
            .transpose()?;
        Ok(TrainingEncodedPair {
            current: rms_norm_latent(&current_raw)?,
            next: rms_norm_latent(&next_raw)?,
            current_raw,
            next_raw,
            projected_sigreg: projected,
        })
    }

    /// Capture every named representation seam from one deterministic batch forward.
    ///
    /// This intentionally exposes only detached tensors, rather than the trainable
    /// encoder/recurrence internals, so evaluation can diagnose representations
    /// without expanding the model's public training surface.
    pub fn representation_diagnostic(
        &self,
        frames: &Tensor,
        next_frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        _goal_features: &Tensor,
    ) -> Result<RepresentationDiagnosticOutput> {
        let (current_raw, target_raw) = self.encode_state_pair_raw(frames, next_frames)?;
        let current = rms_norm_latent(&current_raw)?;
        let target = rms_norm_latent(&target_raw)?;
        let x = self.add_action(&current, actions, action_coords)?;
        let y_init = self.config.warm_start_y.then(|| current.clone());
        let recursion = self.run_latent_recursion(
            &x,
            0.0,
            None,
            RecursionDepth::from_config(&self.config),
            y_init,
            RecursionOpts {
                record_probes: false,
                store_intermediate_steps: true,
            },
        )?;
        let outer_one = recursion
            .steps
            .first()
            .expect("configured outer_steps is positive")
            .clone();
        let prediction = recursion.y;
        let seams = BTreeMap::from([
            (
                RepresentationSeam::EncoderPreRmsPooled,
                pool_latent(&current_raw)?.detach(),
            ),
            (
                RepresentationSeam::EncoderPostRmsPooled,
                pool_latent(&current)?.detach(),
            ),
            (
                RepresentationSeam::EncoderPreRmsSpatial,
                current_raw.detach(),
            ),
            (RepresentationSeam::EncoderPostRmsSpatial, current.detach()),
            (
                RepresentationSeam::ActionConditionedInputSpatial,
                x.detach(),
            ),
            (
                RepresentationSeam::RecursionOuterOneSpatial,
                outer_one.detach(),
            ),
            (
                RepresentationSeam::PredictionFinalPooled,
                pool_latent(&prediction)?.detach(),
            ),
            (
                RepresentationSeam::PredictionFinalConsumerReadout,
                if self.config.world_core_v4 {
                    self.canonical_representation(&prediction)?.detach()
                } else {
                    self.consumer_readout.forward(&prediction)?.detach()
                },
            ),
            (
                RepresentationSeam::PredictionFinalSpatial,
                prediction.detach(),
            ),
            (
                RepresentationSeam::TargetPostRmsPooled,
                pool_latent(&target)?.detach(),
            ),
            (RepresentationSeam::TargetPostRmsSpatial, target.detach()),
        ]);
        Ok(RepresentationDiagnosticOutput { seams })
    }

    fn embed_frames(&self, frames: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = frames.dims4()?;
        if h != FRAME_SIDE || w != FRAME_SIDE {
            bail!("embed_frames: expected {FRAME_SIDE}x{FRAME_SIDE}, got {h}x{w}");
        }
        let embedded = if c == PIXEL_EMB_DIM {
            frames.clone()
        } else if c == 1 {
            let idx = frames.squeeze(1)?.to_dtype(DType::U32)?;
            let flat = idx.flatten_all()?;
            let emb = self.pixel_emb.forward(&flat)?;
            emb.reshape((b, h, w, PIXEL_EMB_DIM))?
                .permute((0, 3, 1, 2))?
                .contiguous()?
        } else {
            bail!("embed_frames: expected 1 or {PIXEL_EMB_DIM} channels, got {c}");
        };
        if !(self.config.world_core_v2 || self.config.world_core_v4) {
            return Ok(embedded);
        }

        // The synthetic status strip advances with the action budget even when
        // the board has not changed. World-core-v2 models board dynamics only:
        // replace that strip with the learned EMPTY embedding before the patch
        // encoder, so branch-equivalence and copy objectives cannot conflict
        // with an observation-side counter. Status remains available to the
        // separately supervised event labels rather than leaking into z.
        let board = embedded.narrow(2, 0, FRAME_SIDE - 1)?;
        let empty_indices = Tensor::zeros((b * w,), DType::U32, frames.device())?;
        let empty = self
            .pixel_emb
            .forward(&empty_indices)?
            .reshape((b, 1, w, PIXEL_EMB_DIM))?
            .permute((0, 3, 1, 2))?
            .contiguous()?;
        Tensor::cat(&[&board, &empty], 2).map_err(Into::into)
    }

    fn add_action(
        &self,
        state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
    ) -> Result<Tensor> {
        let b = state.dim(0)?;
        if state.dims4()? != (b, self.config.hidden_dim, LATENT_GRID, LATENT_GRID) {
            bail!(
                "state must be Bx{}x{LATENT_GRID}x{LATENT_GRID}, got {:?}",
                self.config.hidden_dim,
                state.dims()
            );
        }
        let actions = match actions.rank() {
            1 => actions.clone(),
            2 if actions.dim(1)? == 1 => actions.reshape((b,))?,
            rank => bail!("actions must be shape [B] or [B,1], got rank {rank}"),
        };
        if actions.dim(0)? != b {
            bail!(
                "action batch {} does not match state batch {b}",
                actions.dim(0)?
            );
        }
        if action_coords.dims2()? != (b, 2) {
            bail!("action_coords must have shape [B,2]");
        }
        let action = self
            .action_proj
            .forward(&self.action_emb.forward(&actions)?)?;
        let action_bias = action.reshape((b, self.config.hidden_dim, 1, 1))?;
        let conditioned = state.broadcast_add(&action_bias)?;
        let mut coords = self.coord_proj.forward(action_coords)?;
        if self.config.world_core_v2 {
            let coordinate_active = actions
                .eq(6u32)?
                .to_dtype(coords.dtype())?
                .reshape((b, 1))?
                .broadcast_as(coords.dims())?;
            coords = coords.mul(&coordinate_active)?;
        }
        let coord_bias = coords.reshape((b, self.config.hidden_dim, 1, 1))?;
        let conditioned = if self.config.world_core_v4 {
            let canonical =
                self.canonical_representation(state)?
                    .reshape((b, self.config.hidden_dim, 1, 1))?;
            conditioned.broadcast_add(&canonical)?
        } else {
            conditioned
        };
        if !self.config.spatial_action_field {
            return conditioned.broadcast_add(&coord_bias).map_err(Into::into);
        }

        let field = self.spatial_action_field(&actions, action_coords)?;
        let mut projection = self
            .spatial_action_proj
            .as_ref()
            .expect("spatial_action_proj is present when spatial_action_field is enabled")
            .forward(&field)?;
        if self.config.spatial_action_residual {
            let active = actions
                .eq(6u32)?
                .to_dtype(projection.dtype())?
                .reshape((b, 1, 1, 1))?
                .broadcast_as(projection.dims())?;
            projection = projection.mul(&active)?;
            conditioned
                .broadcast_add(&coord_bias)?
                .add(&projection.affine(self.config.spatial_action_residual_scale, 0.0)?)
                .map_err(Into::into)
        } else {
            conditioned.add(&projection).map_err(Into::into)
        }
    }

    /// ACTION6 coordinate conditioning over the latent grid.
    ///
    /// Coordinates are normalized to `[0, 1]`. The four channels are a localized
    /// impulse, relative x/y offsets, and an ACTION6-active mask. Simple actions
    /// produce an all-zero field even though their placeholder coordinates are
    /// still shape-checked by [`Self::add_action`].
    fn spatial_action_field(&self, actions: &Tensor, action_coords: &Tensor) -> Result<Tensor> {
        let b = actions.dim(0)?;
        if action_coords.dims2()? != (b, 2) {
            bail!("action_coords must have shape [B,2]");
        }
        let coords = action_coords.to_dtype(DType::F32)?;
        let x = coords.narrow(1, 0, 1)?.reshape((b, 1, 1, 1))?;
        let y = coords.narrow(1, 1, 1)?.reshape((b, 1, 1, 1))?;
        let axis = Tensor::arange(0f32, LATENT_GRID as f32, coords.device())?
            .affine(1.0 / (LATENT_GRID - 1) as f64, 0.0)?;
        let grid_x = axis.reshape((1, 1, 1, LATENT_GRID))?;
        let grid_y = axis.reshape((1, 1, LATENT_GRID, 1))?;
        let dx = grid_x
            .broadcast_sub(&x)?
            .broadcast_as((b, 1, LATENT_GRID, LATENT_GRID))?;
        let dy = grid_y
            .broadcast_sub(&y)?
            .broadcast_as((b, 1, LATENT_GRID, LATENT_GRID))?;
        let active = actions
            .eq(6u32)?
            .to_dtype(DType::F32)?
            .reshape((b, 1, 1, 1))?
            .broadcast_as((b, 1, LATENT_GRID, LATENT_GRID))?;
        let impulse = dx
            .sqr()?
            .add(&dy.sqr()?)?
            .affine(-16.0, 0.0)?
            .exp()?
            .broadcast_mul(&active)?;
        let dx = dx.broadcast_mul(&active)?;
        let dy = dy.broadcast_mul(&active)?;
        Tensor::cat(&[&impulse, &dx, &dy, &active], 1).map_err(Into::into)
    }

    /// Encode state pixels and action IDs into the dynamics input `x` (goal-free).
    pub fn encode_x(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
    ) -> Result<Tensor> {
        let state = self.encode_state(frames)?;
        self.add_action(&state, actions, action_coords)
    }

    fn project_goal(&self, goal_features: &Tensor) -> Result<Tensor> {
        let (b, g) = goal_features.dims2()?;
        if g != self.config.goal_dim {
            bail!(
                "goal_features dim {g} != config.goal_dim {}",
                self.config.goal_dim
            );
        }
        let _ = b;
        Ok(self.goal_proj.forward(goal_features)?)
    }

    /// Shared transition prep: encode state/action, project goals, optional warm-start `y`.
    fn prepare_transition(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let state = if self.config.warm_start_y {
            Some(self.encode_state(frames)?)
        } else {
            None
        };
        let x = match &state {
            Some(s) => self.add_action(s, actions, action_coords)?,
            None => self.encode_x(frames, actions, action_coords)?,
        };
        let goal_h = self.project_goal(goal_features)?;
        Ok((x, goal_h, state))
    }

    fn heads(&self, y: &Tensor, goal_h: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let readout = if self.config.world_core_v4 {
            self.canonical_representation(y)?
        } else {
            self.consumer_readout.forward(y)?
        };
        let event_in = Tensor::cat(&[&readout, goal_h], D::Minus1)?;
        let event_logits = self.event_head.forward(&event_in)?;
        let q_logit = self.q_head.forward(&readout)?;
        let reliability_logit = self.reliability_head.forward(&readout)?;
        Ok((event_logits, q_logit, reliability_logit))
    }

    /// Event logits from detached or live `y` (training may stop-grad events only).
    pub fn event_logits_from(&self, y: &Tensor, goal_features: &Tensor) -> Result<Tensor> {
        let goal_h = self.project_goal(goal_features)?;
        let readout = if self.config.world_core_v4 {
            self.canonical_representation(y)?
        } else {
            self.consumer_readout.forward(y)?
        };
        let event_in = Tensor::cat(&[&readout, &goal_h], D::Minus1)?;
        self.event_head.forward(&event_in).map_err(Into::into)
    }

    /// Observer-only V4 head seams. Callers detach `canonical` before these
    /// methods so SpatialQuery and the world core remain frozen.
    pub fn event_logits_from_canonical(
        &self,
        canonical: &Tensor,
        goal_features: &Tensor,
    ) -> Result<Tensor> {
        let goal_h = self.project_goal(goal_features)?;
        let event_in = Tensor::cat(&[canonical, &goal_h], D::Minus1)?;
        self.event_head.forward(&event_in).map_err(Into::into)
    }

    pub fn q_logit_from_canonical(&self, canonical: &Tensor) -> Result<Tensor> {
        self.q_head.forward(canonical).map_err(Into::into)
    }

    pub fn reliability_logit_from_canonical(&self, canonical: &Tensor) -> Result<Tensor> {
        self.reliability_head.forward(canonical).map_err(Into::into)
    }

    /// Inject noise into `z`, with `sigma` interpreted **relative to the current
    /// latent scale**.
    ///
    /// `run_recursion` renormalises `y` to unit RMS every outer step, so an
    /// absolute sigma was attenuated to a ~3% perturbation by the time it
    /// reached the measured output: every PTRM trajectory collapsed onto the
    /// same answer and `pass@k` was flat in `k`. Scaling by the tensor's own RMS
    /// keeps the injected diversity meaningful at any latent scale. The scale
    /// stays on-device so this adds no stream synchronisation.
    fn maybe_noise_z(&self, z: &Tensor, sigma: f64, noise_seed: Option<u64>) -> Result<Tensor> {
        if sigma == 0.0 {
            return Ok(z.clone());
        }
        let eps = match noise_seed {
            Some(seed) => seeded_gaussian_like(z, sigma, seed)?,
            None => z.randn_like(0.0, sigma)?,
        };
        let ones = vec![1usize; z.rank()];
        let rms = z.sqr()?.mean_all()?.sqrt()?.reshape(ones)?;
        let eps = eps.broadcast_mul(&rms)?;
        z.add(&eps).map_err(Into::into)
    }

    /// One deep recursion: `inner_steps` z-updates then one y-update.
    /// Noise (when enabled) is injected into `z` before every z refinement.
    #[allow(clippy::too_many_arguments)]
    fn deep_step(
        &self,
        x: &Tensor,
        y: &Tensor,
        z: &Tensor,
        inner_steps: usize,
        sigma: f64,
        noise_seed_base: Option<u64>,
        noise_counter: &mut u64,
    ) -> Result<(Tensor, Tensor)> {
        if inner_steps == 0 || inner_steps > self.config.inner_steps {
            bail!(
                "inner_steps must be in 1..={}, got {inner_steps}",
                self.config.inner_steps
            );
        }
        let mut z = z.clone();
        let mut y = y.clone();
        for _ in 0..inner_steps {
            let step_seed = noise_seed_base.map(|s| s.wrapping_add(*noise_counter));
            *noise_counter = noise_counter.wrapping_add(1);
            z = self.maybe_noise_z(&z, sigma, step_seed)?;
            let inp = x.add(&y)?.add(&z)?;
            z = self.block.forward(&inp)?;
        }
        let inp = if self.config.world_core_v4 {
            x.add(&y)?.add(&z)?
        } else {
            y.add(&z)?
        };
        y = self.block.forward(&inp)?;
        Ok((y, z))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_latent_recursion(
        &self,
        x: &Tensor,
        sigma: f64,
        noise_seed_base: Option<u64>,
        depth: RecursionDepth,
        y_init: Option<Tensor>,
        opts: RecursionOpts,
    ) -> Result<LatentRecursionOutput> {
        if depth.inner_steps == 0 || depth.inner_steps > self.config.inner_steps {
            bail!(
                "inner_steps must be in 1..={}, got {}",
                self.config.inner_steps,
                depth.inner_steps
            );
        }
        if depth.outer_steps == 0 {
            bail!("outer_steps must be >= 1");
        }
        let (b, c, hh, ww) = x.dims4()?;
        let device = x.device();
        let mut y = match y_init {
            Some(y0) => y0,
            None => Tensor::zeros((b, c, hh, ww), x.dtype(), device)?,
        };
        let mut z = Tensor::zeros((b, c, hh, ww), x.dtype(), device)?;
        let mut steps = Vec::with_capacity(depth.outer_steps);
        let mut probes = if opts.record_probes {
            Vec::with_capacity(depth.outer_steps)
        } else {
            Vec::new()
        };
        let mut noise_counter = 0u64;
        for outer_idx in 0..depth.outer_steps {
            let y_before = if opts.record_probes {
                Some(y.clone())
            } else {
                None
            };
            let (ny, nz) = self.deep_step(
                x,
                &y,
                &z,
                depth.inner_steps,
                sigma,
                noise_seed_base,
                &mut noise_counter,
            )?;
            y = if self.config.residual_y_update {
                rms_norm_latent(&y.add(&ny)?)?
            } else {
                rms_norm_latent(&ny)?
            };
            y = y.clamp(-32.0, 32.0)?;
            z = nz;
            if let Some(y_before) = y_before {
                probes.push(Self::probe_step(&y_before, &y, outer_idx)?);
            }
            let is_last = outer_idx + 1 == depth.outer_steps;
            if opts.store_intermediate_steps || is_last {
                steps.push(y.clone());
            }
        }
        Ok(LatentRecursionOutput {
            y: steps
                .last()
                .expect("outer_steps >= 1 guaranteed by validate")
                .clone(),
            steps,
            recursion_probes: probes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_recursion(
        &self,
        x: &Tensor,
        goal_h: &Tensor,
        sigma: f64,
        noise_seed_base: Option<u64>,
        depth: RecursionDepth,
        y_init: Option<Tensor>,
        opts: RecursionOpts,
    ) -> Result<ForwardOutput> {
        let latent = self.run_latent_recursion(x, sigma, noise_seed_base, depth, y_init, opts)?;
        self.attach_heads(latent, goal_h)
    }

    fn attach_heads(
        &self,
        latent: LatentRecursionOutput,
        goal_h: &Tensor,
    ) -> Result<ForwardOutput> {
        let (event_logits, q_logit, reliability_logit) = self.heads(&latent.y, goal_h)?;
        let last = latent.steps.len() - 1;
        let steps = latent
            .steps
            .into_iter()
            .enumerate()
            .map(|(index, y)| StepOutput {
                y,
                event_logits: (index == last).then(|| event_logits.clone()),
                q_logit: (index == last).then(|| q_logit.clone()),
            })
            .collect();
        Ok(ForwardOutput {
            y: latent.y,
            event_logits,
            q_logit,
            reliability_logit,
            steps,
            recursion_probes: latent.recursion_probes,
        })
    }

    /// Q logit from a (possibly detached) latent state.
    pub fn q_logit_from_y(&self, y: &Tensor) -> Result<Tensor> {
        let readout = if self.config.world_core_v4 {
            self.canonical_representation(y)?
        } else {
            self.consumer_readout.forward(y)?
        };
        self.q_head.forward(&readout).map_err(Into::into)
    }

    pub fn reliability_logit_from_y(&self, y: &Tensor) -> Result<Tensor> {
        let readout = if self.config.world_core_v4 {
            self.canonical_representation(y)?
        } else {
            self.consumer_readout.forward(y)?
        };
        self.reliability_head.forward(&readout).map_err(Into::into)
    }

    /// Direct prefix prediction: residual delta on spatial latent from `(z, action)`.
    pub fn prefix_predict(
        &self,
        state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
    ) -> Result<Tensor> {
        let b = state.dim(0)?;
        if let Some(spatial_prefix_head) = &self.spatial_prefix_head {
            let conditioned = self.add_action(state, actions, action_coords)?;
            let fused = Tensor::cat(&[state, &conditioned], 1)?;
            let delta = spatial_prefix_head.forward(&fused)?;
            return rms_norm_latent(&state.add(&delta)?);
        }
        let pooled = pool_latent(state)?;
        let actions = match actions.rank() {
            1 => actions.clone(),
            2 if actions.dim(1)? == 1 => actions.reshape((b,))?,
            rank => bail!("actions must be shape [B] or [B,1], got rank {rank}"),
        };
        let action = self
            .action_proj
            .forward(&self.action_emb.forward(&actions)?)?;
        let coords = self.coord_proj.forward(action_coords)?;
        let fused = Tensor::cat(&[&pooled, &action.add(&coords)?], D::Minus1)?;
        let delta = self.prefix_head.forward(&fused)?;
        let (_, c, hh, ww) = state.dims4()?;
        let delta = delta.reshape((b, c, 1, 1))?.broadcast_as((b, c, hh, ww))?;
        rms_norm_latent(&state.add(&delta)?)
    }

    /// Recover action identity and ACTION6 coordinates from a predicted latent
    /// displacement. This head is available only in world-core-v2 and is
    /// trained on factual, board-effect-bearing branches.
    pub fn decode_action_displacement(&self, displacement: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, channels) = displacement.dims2()?;
        if channels != self.config.hidden_dim {
            bail!(
                "action displacement width {channels} != hidden_dim {}",
                self.config.hidden_dim
            );
        }
        let action_decoder = self
            .action_decoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("action decoder requires world_core_v2"))?;
        let coordinate_decoder = self
            .coordinate_decoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("coordinate decoder requires world_core_v2"))?;
        Ok((
            action_decoder.forward(displacement)?,
            candle_nn::ops::sigmoid(&coordinate_decoder.forward(displacement)?)?,
        ))
    }

    fn probe_step(
        y_before: &Tensor,
        y_after: &Tensor,
        outer_idx: usize,
    ) -> Result<RecursionStepProbe> {
        let residual = y_after.sub(y_before)?.detach();
        let res_norm = residual.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()? as f64;
        let lat_norm = y_after
            .detach()
            .sqr()?
            .mean_all()?
            .sqrt()?
            .to_scalar::<f32>()? as f64;
        let before_norm = y_before
            .detach()
            .sqr()?
            .mean_all()?
            .sqrt()?
            .to_scalar::<f32>()? as f64;
        let amplification = if before_norm > 1e-8 {
            lat_norm / before_norm
        } else {
            1.0
        };
        Ok(RecursionStepProbe {
            outer_step: outer_idx,
            mean_residual_norm: res_norm,
            mean_latent_norm: lat_norm,
            mean_amplification: amplification,
        })
    }

    /// Deterministic forward with explicit recursion depth (training/eval).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_depth(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        depth: RecursionDepth,
        z_noise_sigma: f64,
        noise_seed: Option<u64>,
    ) -> Result<ForwardOutput> {
        let (x, goal_h, y_init) =
            self.prepare_transition(frames, actions, action_coords, goal_features)?;
        self.run_recursion(
            &x,
            &goal_h,
            z_noise_sigma,
            noise_seed,
            depth,
            y_init,
            RecursionOpts::EVAL,
        )
    }

    /// Like [`Self::forward_with_depth`] but reuses a pre-encoded current state tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_from_encoded_state(
        &self,
        cur_state: &Tensor,
        _frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        depth: RecursionDepth,
        z_noise_sigma: f64,
        noise_seed: Option<u64>,
        recursion: RecursionOpts,
    ) -> Result<ForwardOutput> {
        let x = self.add_action(cur_state, actions, action_coords)?;
        let goal_h = self.project_goal(goal_features)?;
        let y_init = if self.config.warm_start_y {
            Some(cur_state.clone())
        } else {
            None
        };
        self.run_recursion(
            &x,
            &goal_h,
            z_noise_sigma,
            noise_seed,
            depth,
            y_init,
            recursion,
        )
    }

    /// Training recursion with no observer heads or goal projection.
    ///
    /// Loss code applies only the heads active for the current lesson to the
    /// returned latent. The transition computation and retained outer-step
    /// latents are identical to [`Self::forward_from_encoded_state`].
    #[allow(clippy::too_many_arguments)]
    pub fn training_latents_from_encoded_state(
        &self,
        cur_state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        depth: RecursionDepth,
        z_noise_sigma: f64,
        noise_seed: Option<u64>,
        recursion: RecursionOpts,
    ) -> Result<LatentRecursionOutput> {
        let x = self.add_action(cur_state, actions, action_coords)?;
        let y_init = self.config.warm_start_y.then(|| cur_state.clone());
        self.run_latent_recursion(&x, z_noise_sigma, noise_seed, depth, y_init, recursion)
    }

    /// Deterministic forward (no noise, no learned halting).
    pub fn forward(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
    ) -> Result<ForwardOutput> {
        self.forward_with_depth(
            frames,
            actions,
            action_coords,
            goal_features,
            RecursionDepth::from_config(&self.config),
            0.0,
            None,
        )
    }

    /// Deterministic matched-compute ablation with extra recursion and unchanged
    /// weights. This is evaluation-only; training uses the configured depth.
    pub fn forward_with_outer_steps(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        outer_steps: usize,
    ) -> Result<ForwardOutput> {
        let (x, goal_h, y_init) =
            self.prepare_transition(frames, actions, action_coords, goal_features)?;
        self.run_recursion(
            &x,
            &goal_h,
            0.0,
            None,
            RecursionDepth {
                inner_steps: self.config.inner_steps,
                outer_steps,
            },
            y_init,
            RecursionOpts::EVAL,
        )
    }

    /// Autoregressive latent transition used by multi-step rollouts.
    pub fn forward_from_latent(
        &self,
        state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
    ) -> Result<ForwardOutput> {
        self.forward_from_latent_with_depth(
            state,
            actions,
            action_coords,
            goal_features,
            RecursionDepth::from_config(&self.config),
        )
    }

    pub fn forward_from_latent_with_depth(
        &self,
        state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        depth: RecursionDepth,
    ) -> Result<ForwardOutput> {
        let x = self.add_action(state, actions, action_coords)?;
        let goal_h = self.project_goal(goal_features)?;
        let y_init = if self.config.warm_start_y {
            Some(state.clone())
        } else {
            None
        };
        self.run_recursion(
            &x,
            &goal_h,
            0.0,
            None,
            depth,
            y_init,
            RecursionOpts::training(false),
        )
    }

    /// Goal-free latent transition for training losses that consume only `y`.
    pub fn predict_latent_with_depth(
        &self,
        state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        depth: RecursionDepth,
    ) -> Result<Tensor> {
        let x = self.add_action(state, actions, action_coords)?;
        let y_init = self.config.warm_start_y.then(|| state.clone());
        Ok(self
            .run_latent_recursion(&x, 0.0, None, depth, y_init, RecursionOpts::training(true))?
            .y)
    }

    /// PTRM forward: K stochastic trajectories with Gaussian noise on z at
    /// every inner recursion step. When `seed` is set, noise uses a
    /// deterministic `StdRng` (mixed seed/trajectory plus injection counter), which
    /// works on CPU where `Device::set_seed` is unsupported.
    pub fn forward_ptrm(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        ptrm: PtrmConfig,
    ) -> Result<PtrmOutput> {
        self.forward_ptrm_with_depth(
            frames,
            actions,
            action_coords,
            goal_features,
            RecursionDepth::from_config(&self.config),
            ptrm,
        )
    }

    pub fn forward_ptrm_with_depth(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        depth: RecursionDepth,
        ptrm: PtrmConfig,
    ) -> Result<PtrmOutput> {
        if ptrm.k == 0 {
            bail!("PTRM requires K >= 1");
        }
        if !ptrm.sigma.is_finite() || ptrm.sigma < 0.0 {
            bail!("PTRM sigma must be finite and non-negative");
        }
        let (x, goal_h, y_init) =
            self.prepare_transition(frames, actions, action_coords, goal_features)?;
        self.forward_ptrm_prepared(&x, &goal_h, y_init, depth, ptrm)
    }

    /// PTRM from precomputed transition tensors (no frame encode).
    pub fn forward_ptrm_prepared(
        &self,
        x: &Tensor,
        goal_h: &Tensor,
        y_init: Option<Tensor>,
        depth: RecursionDepth,
        ptrm: PtrmConfig,
    ) -> Result<PtrmOutput> {
        if ptrm.k == 0 {
            bail!("PTRM requires K >= 1");
        }
        if !ptrm.sigma.is_finite() || ptrm.sigma < 0.0 {
            bail!("PTRM sigma must be finite and non-negative");
        }
        let latent_trajectories = self.ptrm_latent_trajectories(x, y_init, depth, ptrm)?;
        let mut trajectories = Vec::with_capacity(ptrm.k);
        let mut q_logits = Vec::with_capacity(ptrm.k);
        for latent in latent_trajectories {
            let out = self.attach_heads(latent, goal_h)?;
            q_logits.push(out.q_logit.clone());
            trajectories.push(PtrmTrajectory {
                steps: out.steps,
                y: out.y,
                event_logits: out.event_logits,
                q_logit: out.q_logit,
            });
        }
        let best_indices = best_q_indices(&q_logits)?;
        Ok(PtrmOutput {
            trajectories,
            best_indices,
        })
    }

    fn ptrm_latent_trajectories(
        &self,
        x: &Tensor,
        y_init: Option<Tensor>,
        depth: RecursionDepth,
        ptrm: PtrmConfig,
    ) -> Result<Vec<LatentRecursionOutput>> {
        let mut trajectories = Vec::with_capacity(ptrm.k);
        for traj in 0..ptrm.k {
            // Trajectory 0 is the deterministic member. Previously every
            // trajectory (including 0) was perturbed, so the K-set never
            // contained the noise-free answer and the reported `k=1` row was a
            // noisy sample mislabelled as deterministic — the deterministic-vs-
            // PTRM ablation the design requires could not be read off it.
            // Keeping member 0 clean also makes the training-time ranking label
            // meaningful: it asks whether a perturbation beats the default.
            let sigma = if traj == 0 { 0.0 } else { ptrm.sigma };
            let noise_base = ptrm
                .seed
                .map(|s| s ^ (traj as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let out = self.run_latent_recursion(
                x,
                sigma,
                noise_base,
                depth,
                y_init.clone(),
                RecursionOpts::training(false),
            )?;
            trajectories.push(out);
        }
        Ok(trajectories)
    }

    /// Training-only PTRM trajectories: dynamics plus Q, with no observer heads,
    /// goal projection, or host-side best-index selection.
    pub fn ptrm_ranking_trajectories_from_encoded(
        &self,
        state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        depth: RecursionDepth,
        ptrm: PtrmConfig,
    ) -> Result<Vec<PtrmRankingTrajectory>> {
        if ptrm.k == 0 {
            bail!("PTRM requires K >= 1");
        }
        if !ptrm.sigma.is_finite() || ptrm.sigma < 0.0 {
            bail!("PTRM sigma must be finite and non-negative");
        }
        let x = self.add_action(state, actions, action_coords)?;
        let y_init = self.config.warm_start_y.then(|| state.clone());
        self.ptrm_latent_trajectories(&x, y_init, depth, ptrm)?
            .into_iter()
            .map(|trajectory| {
                let q_logit = self.q_logit_from_y(&trajectory.y)?;
                Ok(PtrmRankingTrajectory {
                    y: trajectory.y,
                    q_logit,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::train::reinit_varmap_deterministic;
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn tiny_cfg() -> ModelConfig {
        ModelConfig {
            frame_side: FRAME_SIDE,
            hidden_dim: 32,
            action_dim: 8,
            goal_dim: 6,
            inner_steps: 2,
            outer_steps: 2,
            num_events: DEFAULT_NUM_EVENTS,
            spatial_action_field: false,
            ..Default::default()
        }
    }

    fn make_model(device: &Device) -> Result<(WorldModel, VarMap)> {
        let cfg = tiny_cfg();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = WorldModel::new(cfg, vb)?;
        Ok((model, varmap))
    }

    #[test]
    fn readout_variants_preserve_shared_parameter_initialization() -> Result<()> {
        let device = Device::Cpu;
        let global_vars = VarMap::new();
        let global_model = WorldModel::new(
            tiny_cfg(),
            VarBuilder::from_varmap(&global_vars, DType::F32, &device),
        )?;
        reinit_varmap_deterministic(&global_vars, 41)?;

        let mut spatial_cfg = tiny_cfg();
        spatial_cfg.consumer_readout = ConsumerReadoutTopology::SpatialQuery;
        let spatial_vars = VarMap::new();
        let spatial_model = WorldModel::new(
            spatial_cfg,
            VarBuilder::from_varmap(&spatial_vars, DType::F32, &device),
        )?;
        reinit_varmap_deterministic(&spatial_vars, 41)?;

        let global = global_vars.data().lock().unwrap();
        let spatial = spatial_vars.data().lock().unwrap();
        for (name, tensor) in global.iter() {
            let counterpart = spatial
                .get(name)
                .unwrap_or_else(|| panic!("spatial arm is missing shared parameter {name}"));
            assert_eq!(
                tensor.as_tensor().flatten_all()?.to_vec1::<f32>()?,
                counterpart.as_tensor().flatten_all()?.to_vec1::<f32>()?,
                "shared initialization differs for {name}"
            );
        }
        assert!(spatial
            .keys()
            .any(|name| name.starts_with("consumer_readout.")));
        assert_eq!(
            global_model.consumer_readout.topology(),
            ConsumerReadoutTopology::GlobalMean
        );
        assert_eq!(
            spatial_model.consumer_readout.topology(),
            ConsumerReadoutTopology::SpatialQuery
        );
        Ok(())
    }

    fn sample_batch(
        device: &Device,
        batch: usize,
        goal_dim: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let frames = Tensor::from_vec(
            (0..batch * FRAME_SIDE * FRAME_SIDE)
                .map(|i| (i % PALETTE_SIZE) as u8)
                .collect::<Vec<_>>(),
            (batch, 1, FRAME_SIDE, FRAME_SIDE),
            device,
        )?;
        let actions = Tensor::from_vec(
            (0..batch)
                .map(|i| (i % ACTION_VOCAB) as u32)
                .collect::<Vec<_>>(),
            (batch,),
            device,
        )?;
        let goals = Tensor::randn(0f32, 1.0, (batch, goal_dim), device)?;
        let coords = Tensor::zeros((batch, 2), DType::F32, device)?;
        Ok((frames, actions, coords, goals))
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        let d = a
            .to_dtype(DType::F32)?
            .sub(&b.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(d.into_iter().fold(0.0f32, f32::max))
    }

    #[test]
    fn action6_spatial_field_is_nonuniform_and_coordinate_sensitive() -> Result<()> {
        let device = Device::Cpu;
        let cfg = ModelConfig {
            spatial_action_field: true,
            ..tiny_cfg()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg, vb)?;
        let actions = Tensor::from_vec(vec![6u32], (1,), &device)?;
        let upper_left = Tensor::from_vec(vec![0.0f32, 0.0], (1, 2), &device)?;
        let lower_right = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &device)?;

        let field = model.spatial_action_field(&actions, &upper_left)?;
        assert_eq!(field.dims(), &[1, 4, LATENT_GRID, LATENT_GRID]);
        let impulse = field.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
        let impulse_min = impulse.iter().copied().fold(f32::INFINITY, f32::min);
        let impulse_max = impulse.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            impulse_max - impulse_min > 0.5,
            "ACTION6 impulse should vary across the latent grid, got range {}",
            impulse_max - impulse_min
        );

        let shifted = model.spatial_action_field(&actions, &lower_right)?;
        assert!(
            max_abs_diff(&field, &shifted)? > 0.5,
            "ACTION6 field should change with coordinates"
        );
        Ok(())
    }

    #[test]
    fn non_action6_has_no_spatial_coordinate_field() -> Result<()> {
        let device = Device::Cpu;
        let cfg = ModelConfig {
            spatial_action_field: true,
            ..tiny_cfg()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg, vb)?;
        let actions = Tensor::from_vec(vec![5u32], (1,), &device)?;
        let coords = Tensor::from_vec(vec![0.25f32, 0.75], (1, 2), &device)?;

        let field = model.spatial_action_field(&actions, &coords)?;
        assert!(
            field
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|value| value.abs() < 1e-6),
            "non-ACTION6 spatial field must be all zeros"
        );
        Ok(())
    }

    #[test]
    fn disabled_spatial_action_field_preserves_uniform_coordinate_broadcast() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = make_model(&device)?;
        let state = Tensor::zeros(
            (1, model.config.hidden_dim, LATENT_GRID, LATENT_GRID),
            DType::F32,
            &device,
        )?;
        let actions = Tensor::from_vec(vec![6u32], (1,), &device)?;
        let coords = Tensor::from_vec(vec![0.25f32, 0.75], (1, 2), &device)?;

        let conditioned = model.add_action(&state, &actions, &coords)?;
        let origin = conditioned
            .narrow(2, 0, 1)?
            .narrow(3, 0, 1)?
            .broadcast_as(conditioned.dims())?;
        assert!(
            max_abs_diff(&conditioned, &origin)? < 1e-6,
            "legacy conditioning must be uniform across the latent grid"
        );
        Ok(())
    }

    #[test]
    fn v3_spatial_residual_preserves_global_coordinate_bias() -> Result<()> {
        let device = Device::Cpu;
        let cfg = ModelConfig {
            world_core_v2: true,
            world_core_v3: true,
            spatial_action_field: true,
            spatial_action_residual: true,
            spatial_action_residual_scale: 0.25,
            ..tiny_cfg()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg, vb)?;
        let state = Tensor::zeros(
            (1, model.config.hidden_dim, LATENT_GRID, LATENT_GRID),
            DType::F32,
            &device,
        )?;
        let actions = Tensor::from_vec(vec![6u32], (1,), &device)?;
        let coords = Tensor::from_vec(vec![0.25f32, 0.75], (1, 2), &device)?;

        let action = model
            .action_proj
            .forward(&model.action_emb.forward(&actions)?)?
            .reshape((1, model.config.hidden_dim, 1, 1))?;
        let conditioned = state.broadcast_add(&action)?;
        let global =
            model
                .coord_proj
                .forward(&coords)?
                .reshape((1, model.config.hidden_dim, 1, 1))?;
        let spatial = model
            .spatial_action_proj
            .as_ref()
            .expect("V3 has a spatial projection")
            .forward(&model.spatial_action_field(&actions, &coords)?)?;
        let expected = conditioned
            .broadcast_add(&global)?
            .add(&spatial.affine(0.25, 0.0)?)?;
        let actual = model.add_action(&state, &actions, &coords)?;
        assert!(max_abs_diff(&actual, &expected)? < 1e-6);
        Ok(())
    }

    #[test]
    fn v3_spatial_residual_ignores_placeholder_coordinates_for_simple_actions() -> Result<()> {
        let device = Device::Cpu;
        let cfg = ModelConfig {
            world_core_v2: true,
            world_core_v3: true,
            spatial_action_field: true,
            spatial_action_residual: true,
            ..tiny_cfg()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg, vb)?;
        let state = Tensor::zeros(
            (1, model.config.hidden_dim, LATENT_GRID, LATENT_GRID),
            DType::F32,
            &device,
        )?;
        let actions = Tensor::from_vec(vec![5u32], (1,), &device)?;
        let first = Tensor::from_vec(vec![0.0f32, 0.0], (1, 2), &device)?;
        let second = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &device)?;
        assert!(
            max_abs_diff(
                &model.add_action(&state, &actions, &first)?,
                &model.add_action(&state, &actions, &second)?,
            )? < 1e-6
        );
        Ok(())
    }

    #[test]
    fn representation_diagnostic_returns_every_named_detached_seam() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = make_model(&device)?;
        let (frames, actions, coords, goals) = sample_batch(&device, 2, model.config.goal_dim)?;
        let next = frames.clone();
        let output = model.representation_diagnostic(&frames, &next, &actions, &coords, &goals)?;
        assert_eq!(output.seams.len(), 11);
        for seam in [
            RepresentationSeam::EncoderPreRmsPooled,
            RepresentationSeam::EncoderPostRmsPooled,
            RepresentationSeam::EncoderPreRmsSpatial,
            RepresentationSeam::EncoderPostRmsSpatial,
            RepresentationSeam::ActionConditionedInputSpatial,
            RepresentationSeam::RecursionOuterOneSpatial,
            RepresentationSeam::PredictionFinalPooled,
            RepresentationSeam::PredictionFinalConsumerReadout,
            RepresentationSeam::PredictionFinalSpatial,
            RepresentationSeam::TargetPostRmsPooled,
            RepresentationSeam::TargetPostRmsSpatial,
        ] {
            assert!(output.seams.contains_key(&seam), "missing {seam:?}");
        }
        assert_eq!(
            output.seams[&RepresentationSeam::EncoderPostRmsPooled].dims2()?,
            (2, model.config.hidden_dim)
        );
        assert_eq!(
            output.seams[&RepresentationSeam::PredictionFinalSpatial].dims4()?,
            (2, model.config.hidden_dim, LATENT_GRID, LATENT_GRID)
        );
        Ok(())
    }

    #[test]
    fn rms_norm_latent_unit_scale() -> Result<()> {
        let device = Device::Cpu;
        let z = Tensor::full(3.0f32, (2, 4, 8, 8), &device)?;
        let normed = rms_norm_latent(&z)?;
        for batch in 0..2 {
            let slice = normed.i(batch)?.flatten_all()?.to_vec1::<f32>()?;
            let mean_sq = slice.iter().map(|v| v * v).sum::<f32>() / slice.len() as f32;
            assert!(
                (mean_sq.sqrt() - 1.0).abs() < 1e-5,
                "batch {batch} expected unit RMS, got {}",
                mean_sq.sqrt()
            );
        }
        Ok(())
    }

    #[test]
    fn prefix_rollout_stays_on_unit_rms_latent_support() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = make_model(&device)?;
        let (frames, actions, coords, _) = sample_batch(&device, 2, model.config.goal_dim)?;
        let mut latent = model.encode_state(&frames)?;
        for _ in 0..PREFIX_HORIZONS[PREFIX_HORIZONS.len() - 1] {
            latent = model.prefix_predict(&latent, &actions, &coords)?;
        }
        let rms = latent
            .sqr()?
            .flatten_from(1)?
            .mean(D::Minus1)?
            .sqrt()?
            .to_vec1::<f32>()?;
        assert!(
            rms.iter().all(|value| (*value - 1.0).abs() < 1e-4),
            "prefix rollout left normalized latent support: {rms:?}"
        );
        Ok(())
    }

    #[test]
    fn forward_output_shapes() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = make_model(&device)?;
        let (frames, actions, coords, goals) = sample_batch(&device, 3, model.config.goal_dim)?;
        let out = model.forward(&frames, &actions, &coords, &goals)?;
        assert_eq!(out.steps.len(), model.config.outer_steps);
        assert_eq!(
            out.y.dims(),
            &[3, model.config.hidden_dim, LATENT_GRID, LATENT_GRID]
        );
        assert_eq!(out.event_logits.dims(), &[3, model.config.num_events]);
        assert_eq!(out.q_logit.dims(), &[3, 1]);
        for (idx, step) in out.steps.iter().enumerate() {
            assert_eq!(
                step.y.dims(),
                &[3, model.config.hidden_dim, LATENT_GRID, LATENT_GRID]
            );
            let is_last = idx + 1 == out.steps.len();
            if is_last {
                assert_eq!(
                    step.event_logits.as_ref().unwrap().dims(),
                    &[3, model.config.num_events]
                );
                assert_eq!(step.q_logit.as_ref().unwrap().dims(), &[3, 1]);
            } else {
                assert!(step.event_logits.is_none());
                assert!(step.q_logit.is_none());
            }
        }
        Ok(())
    }

    #[test]
    fn encode_state_pair_matches_separate_encodes() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = make_model(&device)?;
        let (frames, _, _, _) = sample_batch(&device, 4, model.config.goal_dim)?;
        let next_frames = Tensor::ones(frames.dims(), DType::U8, &device)?;
        let (cur, next) = model.encode_state_pair(&frames, &next_frames)?;
        let cur_solo = model.encode_state(&frames)?;
        let next_solo = model.encode_state(&next_frames)?;
        assert!(max_abs_diff(&cur, &cur_solo)? < 1e-5);
        assert!(max_abs_diff(&next, &next_solo)? < 1e-5);
        Ok(())
    }

    #[test]
    fn world_core_v2_latent_is_invariant_to_status_strip() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = tiny_cfg();
        cfg.world_core_v2 = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg, vb)?;
        let mut first = vec![0u8; FRAME_SIDE * FRAME_SIDE];
        first[10 * FRAME_SIDE + 10] = 3;
        let mut second = first.clone();
        for (column, pixel) in second[(FRAME_SIDE - 1) * FRAME_SIDE..]
            .iter_mut()
            .enumerate()
        {
            *pixel = (column % PALETTE_SIZE) as u8;
        }
        let first = Tensor::from_vec(first, (1, 1, FRAME_SIDE, FRAME_SIDE), &device)?;
        let second = Tensor::from_vec(second, (1, 1, FRAME_SIDE, FRAME_SIDE), &device)?;

        let first_latent = model.encode_state(&first)?;
        let second_latent = model.encode_state(&second)?;

        assert_eq!(max_abs_diff(&first_latent, &second_latent)?, 0.0);
        Ok(())
    }

    #[test]
    fn sigreg_projector_uses_pre_rms_features_and_returns_time_batch_embeddings() -> Result<()> {
        let device = Device::Cpu;
        let cfg = ModelConfig {
            hidden_dim: 8,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            sigreg_projector: true,
            sigreg_projector_dim: 6,
            spatial_action_field: false,
            ..ModelConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg, vb)?;
        let current = Tensor::zeros((3, 1, FRAME_SIDE, FRAME_SIDE), DType::U8, &device)?;
        let next = Tensor::ones((3, 1, FRAME_SIDE, FRAME_SIDE), DType::U8, &device)?;
        let pair = model.encode_state_pair_for_training(&current, &next)?;
        assert_eq!(pair.current.dims(), &[3, 8, LATENT_GRID, LATENT_GRID]);
        assert_eq!(pair.next.dims(), &[3, 8, LATENT_GRID, LATENT_GRID]);
        assert_eq!(pair.current_raw.dims(), pair.current.dims());
        assert_eq!(pair.next_raw.dims(), pair.next.dims());
        assert_eq!(
            pair.projected_sigreg.expect("projector enabled").dims(),
            &[2, 3, 6]
        );
        let names = varmap.data().lock().unwrap();
        assert!(names.contains_key("sigreg_projector.weight"));
        assert!(names.contains_key("sigreg_projector.bias"));
        Ok(())
    }

    #[test]
    fn recurrence_output_is_nonzero_at_init() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = make_model(&device)?;
        let (frames, actions, coords, goals) = sample_batch(&device, 2, model.config.goal_dim)?;
        let cur_z = model.encode_state(&frames)?;
        let depth = RecursionDepth::from_config(model.config());
        let out = model.forward_from_latent_with_depth(&cur_z, &actions, &coords, &goals, depth)?;
        let y_norm = out.y.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        assert!(
            y_norm > 1e-4,
            "recurrence should move y away from zero at init, got norm {y_norm}"
        );
        Ok(())
    }

    #[test]
    fn shallow_depth_forward_is_valid() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = make_model(&device)?;
        let (frames, actions, coords, goals) = sample_batch(&device, 2, model.config.goal_dim)?;
        let out = model.forward_with_outer_steps(&frames, &actions, &coords, &goals, 1)?;
        assert_eq!(out.steps.len(), 1);
        Ok(())
    }

    #[test]
    fn ptrm_k1_sigma0_matches_deterministic() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = make_model(&device)?;
        let (frames, actions, coords, goals) = sample_batch(&device, 2, model.config.goal_dim)?;
        let depth = RecursionDepth::from_config(model.config());
        let det = model.forward(&frames, &actions, &coords, &goals)?;
        let ptrm = model.forward_ptrm_with_depth(
            &frames,
            &actions,
            &coords,
            &goals,
            depth,
            PtrmConfig {
                k: 1,
                sigma: 0.0,
                seed: Some(0),
            },
        )?;
        assert_eq!(ptrm.trajectories.len(), 1);
        assert_eq!(ptrm.best_indices, vec![0; 2]);
        let traj = &ptrm.trajectories[0];
        assert!(max_abs_diff(&det.y, &traj.y)? < 1e-5);
        assert!(max_abs_diff(&det.event_logits, &traj.event_logits)? < 1e-5);
        assert!(max_abs_diff(&det.q_logit, &traj.q_logit)? < 1e-5);
        Ok(())
    }

    #[test]
    fn ptrm_retains_all_trajectories_and_best_q() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = make_model(&device)?;
        let (frames, actions, coords, goals) = sample_batch(&device, 2, model.config.goal_dim)?;
        let out = model.forward_ptrm(
            &frames,
            &actions,
            &coords,
            &goals,
            PtrmConfig {
                k: 3,
                sigma: 0.1,
                seed: Some(123),
            },
        )?;
        assert_eq!(out.trajectories.len(), 3);
        let qs: Vec<Tensor> = out.trajectories.iter().map(|t| t.q_logit.clone()).collect();
        assert_eq!(best_q_indices(&qs)?, out.best_indices);
        assert_eq!(out.best_indices.len(), 2);
        Ok(())
    }

    #[test]
    fn ptrm_ranking_path_matches_full_trajectory_outputs_exactly() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = make_model(&device)?;
        let (frames, actions, coords, goals) = sample_batch(&device, 2, model.config.goal_dim)?;
        let depth = RecursionDepth::from_config(model.config());
        let config = PtrmConfig {
            k: 3,
            sigma: 0.1,
            seed: Some(123),
        };
        let full =
            model.forward_ptrm_with_depth(&frames, &actions, &coords, &goals, depth, config)?;
        let current = model.encode_state(&frames)?;
        let ranking = model
            .ptrm_ranking_trajectories_from_encoded(&current, &actions, &coords, depth, config)?;
        assert_eq!(full.trajectories.len(), ranking.len());
        for (full, ranking) in full.trajectories.iter().zip(&ranking) {
            assert_eq!(
                full.y.flatten_all()?.to_vec1::<f32>()?,
                ranking.y.flatten_all()?.to_vec1::<f32>()?
            );
            assert_eq!(
                full.q_logit.flatten_all()?.to_vec1::<f32>()?,
                ranking.q_logit.flatten_all()?.to_vec1::<f32>()?
            );
        }
        Ok(())
    }

    #[test]
    fn best_q_selection_is_per_sample() -> Result<()> {
        let device = Device::Cpu;
        let a = Tensor::from_vec(vec![2f32, -1.0], (2, 1), &device)?;
        let b = Tensor::from_vec(vec![1f32, 3.0], (2, 1), &device)?;
        assert_eq!(best_q_indices(&[a, b])?, vec![0, 1]);
        Ok(())
    }

    #[test]
    fn backward_produces_nonempty_finite_grads() -> Result<()> {
        let device = Device::Cpu;
        let (model, varmap) = make_model(&device)?;
        let (frames, actions, coords, goals) = sample_batch(&device, 2, model.config.goal_dim)?;
        let out = model.forward(&frames, &actions, &coords, &goals)?;
        let target = Tensor::zeros_like(&out.y)?;
        let loss = out.y.sub(&target)?.sqr()?.mean_all()?;
        let grads = loss.backward()?;
        let mut seen = 0usize;
        for var in varmap.all_vars() {
            if let Some(g) = grads.get(&var) {
                let flat = g.flatten_all()?.to_vec1::<f32>()?;
                assert!(flat.iter().all(|v| v.is_finite()));
                if flat.iter().any(|v| *v != 0.0) {
                    seen += 1;
                }
            }
        }
        assert!(seen > 0, "expected nonempty finite parameter gradients");
        Ok(())
    }

    fn warm_start_model(device: &Device) -> Result<(WorldModel, VarMap)> {
        let mut cfg = tiny_cfg();
        cfg.warm_start_y = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = WorldModel::new(cfg, vb)?;
        Ok((model, varmap))
    }

    #[test]
    fn matched_outer_steps_equals_normal_with_warm_start() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = warm_start_model(&device)?;
        let (frames, actions, coords, goals) = sample_batch(&device, 2, model.config.goal_dim)?;
        let depth = model.config().outer_steps;
        let normal = model.forward(&frames, &actions, &coords, &goals)?;
        let matched = model.forward_with_outer_steps(&frames, &actions, &coords, &goals, depth)?;
        assert!(max_abs_diff(&normal.y, &matched.y)? < 1e-5);
        Ok(())
    }

    #[test]
    fn training_latents_match_all_head_recursion_exactly() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = make_model(&device)?;
        let (frames, actions, coords, goals) = sample_batch(&device, 3, model.config.goal_dim)?;
        let current = model.encode_state(&frames)?;
        let depth = RecursionDepth::from_config(model.config());
        let opts = RecursionOpts::training(false);
        let full = model.forward_from_encoded_state(
            &current,
            &frames,
            &actions,
            &coords,
            &goals,
            depth,
            0.1,
            Some(73),
            opts,
        )?;
        let latent = model.training_latents_from_encoded_state(
            &current,
            &actions,
            &coords,
            depth,
            0.1,
            Some(73),
            opts,
        )?;
        assert_eq!(
            full.y.flatten_all()?.to_vec1::<f32>()?,
            latent.y.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(full.steps.len(), latent.steps.len());
        for (with_heads, without_heads) in full.steps.iter().zip(&latent.steps) {
            assert_eq!(
                with_heads.y.flatten_all()?.to_vec1::<f32>()?,
                without_heads.flatten_all()?.to_vec1::<f32>()?
            );
        }
        Ok(())
    }

    #[test]
    fn seeded_noise_is_invariant_to_batch_layout() -> Result<()> {
        let device = Device::Cpu;
        let template1 = Tensor::zeros((1, 4, 2, 2), DType::F32, &device)?;
        let template2 = Tensor::zeros((2, 4, 2, 2), DType::F32, &device)?;
        let n1 = seeded_gaussian_like(&template1, 1.0, 42)?;
        let n2 = seeded_gaussian_like(&template2, 1.0, 42)?;
        let v1 = n1.flatten_all()?.to_vec1::<f32>()?;
        let v2 = n2.i(0)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(v1, v2);
        Ok(())
    }

    #[test]
    fn contiguous_seeded_noise_matches_rowwise_reference_exactly() -> Result<()> {
        let device = Device::Cpu;
        let template = Tensor::zeros((4, 3, 2, 2), DType::F32, &device)?;
        let actual = seeded_gaussian_like(&template, 0.125, 91)?;
        let mut rows = Vec::new();
        for batch in 0..4u64 {
            let sample_seed = 91u64.wrapping_add(batch.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut rng = rand::rngs::StdRng::seed_from_u64(sample_seed);
            let data = (0..12)
                .map(|_| standard_normal(&mut rng) * 0.125)
                .collect::<Vec<_>>();
            rows.push(Tensor::from_vec(data, (3, 2, 2), &device)?);
        }
        let reference = Tensor::stack(&rows, 0)?;
        assert_eq!(
            actual.flatten_all()?.to_vec1::<f32>()?,
            reference.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
}
