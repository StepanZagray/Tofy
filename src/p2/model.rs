//! TRM-inspired pixel world model with PTRM stochastic inference.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor, D};
use candle_nn::{
    conv2d, embedding, linear, Conv2d, Conv2dConfig, Embedding, Init, Linear, Module, VarBuilder,
    VarMap,
};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::p2::consumer_readout::ConsumerReadout;
use crate::p2::data::TransitionSample;
use crate::p2::experiment::ConsumerReadoutTopology;
use crate::p2::grounding::{
    DecodeComposition, ExactPatchGrounding, PatchGroundingLoss, PatchGroundingMode,
    PatchHistogramGrounding,
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
/// UNKNOWN plus the five synthetic episode-operator families.
pub const OPERATOR_FAMILY_VOCAB: usize = 6;
/// One family one-hot and three independent 16-color one-hots.
pub const OPERATOR_CONDITION_DIM: usize = OPERATOR_FAMILY_VOCAB + 3 * PALETTE_SIZE;
/// Family token used for held-out or unavailable operator provenance.
pub const OPERATOR_FAMILY_UNKNOWN: usize = 0;

/// Six family indicators plus public parameters/order slots.
pub const DEFAULT_GOAL_DIM: usize = 19;
/// Default event head width: noop / goal_satisfied / goal_failed / exhausted.
pub const DEFAULT_NUM_EVENTS: usize = 4;

/// Initial copy-bypass interpolation weight. This must remain small enough to
/// preserve the latent-copy prior, but nonzero so the candidate path receives
/// prediction gradient from the first update.
pub const COPY_BYPASS_INITIAL_ALPHA: f64 = 0.02;

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
    /// Side length of one square encoder/decoder patch. Foundation-v2 uses 4;
    /// patch 8 remains available for legacy comparisons.
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
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
    /// Run only the recurrent block's convolution products in BF16. Inputs
    /// and F32 master weights are cast at use time; every convolution result
    /// returns to F32 before bias, activation, FiLM, or residual/state math.
    #[serde(default)]
    pub bf16_recurrent_core: bool,
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
    /// Instantiate the legacy action-faithful world-core-v2 topology. The
    /// inverse-action heads are also present on V5 so foundation-v2 can train
    /// the ADR 0003 inverse-action objective without composing V2 and V4.
    #[serde(default)]
    pub world_core_v2: bool,
    /// Explicit experiment schema marker. V3 retains the V2 heads/topology.
    #[serde(default)]
    pub world_core_v3: bool,
    /// Paper-grounded successor recipe. Unlike V2/V3, V4 does not enable the
    /// experimental factual-branch heads.
    #[serde(default)]
    pub world_core_v4: bool,
    /// Foundation-v2 objective marker. Reuses the V4 exact decoder while
    /// enabling only the inverse-action heads required by ADR 0003.
    #[serde(default)]
    pub world_core_v5: bool,
    /// Readout used only by Q/event/reliability planning heads.
    #[serde(default)]
    pub consumer_readout: ConsumerReadoutTopology,
    /// Copy-bypass gated outer update: the legacy candidate
    /// `l = clamp(rms_norm(y + ny))` is interpolated as `y' = y + a*(l - y)`
    /// with a scalar gate `a` initialized to [`COPY_BYPASS_INITIAL_ALPHA`].
    /// `a = 0` is exact latent copy for any state in the clamp envelope;
    /// `a = 1` reproduces the legacy update algebraically (tested to 1e-6 in
    /// f32), so the treatment contains the baseline as a parameter setting.
    /// The interpolation is re-clamped to the legacy envelope. Requires
    /// `residual_y_update && warm_start_y`.
    #[serde(default)]
    pub copy_bypass_gate: bool,
    /// Initialize the copy-gate bias to `logit(p)` for this expected
    /// changed-pixel rate so composition starts as calibrated copy.
    /// `None` keeps candle's default uniform bias (a ~50/50 gate). Note the
    /// counterargument: under the class-balanced gate BCE the uninformative
    /// optimum is 0.5, so a negative bias is transient and is a
    /// preregistered choice, not a prescribed default.
    #[serde(default)]
    pub copy_gate_bias_prior: Option<f64>,
    /// Scale the ACTION6 Gaussian impulse to the latent grid (sigma = one
    /// cell) instead of fixed normalized units; the fixed -16 exponent
    /// blurred neighbor-cell contrast from 0.72 to 0.93 under patch 4.
    #[serde(default)]
    pub grid_scaled_action_impulse: bool,
    /// Composition rule for the deployed gameplay decode.
    #[serde(default)]
    pub decode_composition: DecodeComposition,
    /// Native-grid positional-value canonical readout. New-run only: enabling
    /// adds position-value embeddings, so loading a checkpoint without them
    /// fails closed.
    #[serde(default)]
    pub positional_value_readout: bool,
}

fn default_sigreg_projector_dim() -> usize {
    128
}

fn default_patch_size() -> usize {
    PATCH_SIZE
}

fn default_spatial_action_residual_scale() -> f64 {
    0.25
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            frame_side: FRAME_SIDE,
            patch_size: default_patch_size(),
            hidden_dim: 128,
            action_dim: 32,
            goal_dim: DEFAULT_GOAL_DIM,
            inner_steps: 2,
            outer_steps: 2,
            num_events: DEFAULT_NUM_EVENTS,
            residual_y_update: false,
            warm_start_y: false,
            bf16_conv: false,
            bf16_recurrent_core: false,
            sigreg_projector: false,
            sigreg_projector_dim: default_sigreg_projector_dim(),
            spatial_action_field: false,
            spatial_action_residual: false,
            spatial_action_residual_scale: default_spatial_action_residual_scale(),
            world_core_v2: false,
            world_core_v3: false,
            world_core_v4: false,
            world_core_v5: false,
            consumer_readout: ConsumerReadoutTopology::GlobalMean,
            copy_bypass_gate: false,
            copy_gate_bias_prior: None,
            grid_scaled_action_impulse: false,
            decode_composition: DecodeComposition::default(),
            positional_value_readout: false,
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
    /// Batch-mean cosine between the state before and after this outer step.
    /// The diagnosed F1 mechanism is a per-step rotation of the state
    /// direction; a copy-consistent update keeps this near 1.
    #[serde(default = "default_step_cosine")]
    pub mean_step_cosine: f64,
}

fn default_step_cosine() -> f64 {
    f64::NAN
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
        if !matches!(self.patch_size, PATCH_SIZE | LEGACY_PATCH_SIZE) {
            bail!(
                "ModelConfig.patch_size must be 4 or 8, got {}",
                self.patch_size
            );
        }
        if !FRAME_SIDE.is_multiple_of(self.patch_size) {
            bail!(
                "frame side {FRAME_SIDE} must be divisible by patch size {}",
                self.patch_size
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
        if self.copy_bypass_gate && !(self.residual_y_update && self.warm_start_y) {
            bail!(
                "copy_bypass_gate requires residual_y_update and warm_start_y: \
                 the zero-gate fixpoint must be the warm-started current state"
            );
        }
        if let Some(prior) = self.copy_gate_bias_prior {
            if !(prior.is_finite() && prior > 0.0 && prior < 1.0) {
                bail!("copy_gate_bias_prior must be a probability in (0, 1), got {prior}");
            }
        }
        if self.positional_value_readout
            && self.consumer_readout != ConsumerReadoutTopology::SpatialQuery
        {
            bail!(
                "positional_value_readout requires the SpatialQuery consumer readout; \
                 the GlobalMean topology would silently ignore it"
            );
        }
        if self.grid_scaled_action_impulse && !self.spatial_action_field {
            bail!(
                "grid_scaled_action_impulse requires spatial_action_field=true; \
                 without it the treatment silently does nothing while the \
                 contract labels the run a treatment arm"
            );
        }
        if self.decode_composition != DecodeComposition::LegacyHardGate && !self.world_core_v4 {
            bail!(
                "decode_composition treatments require the world-core-v4 exact \
                 decoder; without it the flag silently does nothing"
            );
        }
        if self.world_core_v3 && !self.world_core_v2 {
            bail!("world_core_v3 requires the world_core_v2 base topology");
        }
        if self.world_core_v4 && (self.world_core_v2 || self.world_core_v3) {
            bail!("world_core_v4 cannot be composed with V2/V3");
        }
        if self.world_core_v5 && !self.world_core_v4 {
            bail!("world_core_v5 requires the world_core_v4 exact-decoder topology");
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

    pub fn latent_grid(&self) -> usize {
        FRAME_SIDE / self.patch_size
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

fn zero_initialized_linear(in_dim: usize, out_dim: usize, vb: VarBuilder<'_>) -> Result<Linear> {
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", Init::Const(0.0))?;
    let bias = vb.get_with_hints(out_dim, "bias", Init::Const(0.0))?;
    Ok(Linear::new(weight, Some(bias)))
}

/// Restore FiLM's identity initialization after any generic model-wide
/// reinitializer. Tofy's name-seeded initializer intentionally overwrites all
/// weights, so foundation-v2 training calls this once immediately afterward.
pub fn zero_action_film_projections(varmap: &VarMap) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    let mut matched = 0usize;
    for (name, var) in data
        .iter()
        .filter(|(name, _)| name.starts_with("action_film_"))
    {
        var.set(&Tensor::zeros(var.shape(), var.dtype(), var.device())?)
            .map_err(|error| anyhow::anyhow!("zero {name}: {error}"))?;
        matched += 1;
    }
    if matched != 4 {
        bail!("expected four action FiLM parameters, found {matched}");
    }
    Ok(())
}

/// Restore the operator pathway's identity-preserving zero initialization
/// after the deterministic model-wide reinitializer. Legacy topologies do not
/// instantiate this projection and are a deliberate no-op.
pub fn zero_operator_conditioning_projection(varmap: &VarMap) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    let mut matched = 0usize;
    for (name, var) in data
        .iter()
        .filter(|(name, _)| name.starts_with("operator_conditioning_proj."))
    {
        var.set(&Tensor::zeros(var.shape(), var.dtype(), var.device())?)
            .map_err(|error| anyhow::anyhow!("zero {name}: {error}"))?;
        matched += 1;
    }
    if matched != 0 && matched != 2 {
        bail!("expected zero or two operator-conditioning parameters, found {matched}");
    }
    Ok(())
}

/// UNKNOWN family token with a neutral all-zero color triple.
pub fn unknown_operator_conditioning(batch: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut values = vec![0f32; batch * OPERATOR_CONDITION_DIM];
    for row in 0..batch {
        values[row * OPERATOR_CONDITION_DIM + OPERATOR_FAMILY_UNKNOWN] = 1.0;
    }
    Tensor::from_vec(values, (batch, OPERATOR_CONDITION_DIM), device).map_err(Into::into)
}

/// Restore the copy-bypass gate's small nonzero initialization after any
/// generic model-wide reinitializer (which xavier-initializes every non-bias
/// tensor). A no-op when the gate is absent; every fresh-init training path
/// calls this next to [`zero_action_film_projections`].
pub fn init_copy_bypass_gate(varmap: &VarMap) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    for (name, var) in data
        .iter()
        .filter(|(name, _)| name.ends_with("y_copy_bypass_alpha"))
    {
        let initial = Tensor::zeros(var.shape(), var.dtype(), var.device())?
            .affine(0.0, COPY_BYPASS_INITIAL_ALPHA)?;
        var.set(&initial)
            .map_err(|error| anyhow::anyhow!("initialize {name}: {error}"))?;
    }
    Ok(())
}

/// Restore the configured copy-gate bias prior after a generic reinitializer.
/// The reinitializer zeroes every `*bias` tensor, which would silently turn
/// the calibrated-copy prior into a 50/50 gate and degenerate the treatment
/// arm into its control. Must be called on every fresh-init path when the
/// prior is configured; fails loudly if the gate tensor is absent.
pub fn restore_copy_gate_bias_prior(varmap: &VarMap, prior: Option<f64>) -> Result<()> {
    let Some(prior) = prior else {
        return Ok(());
    };
    if !(prior.is_finite() && prior > 0.0 && prior < 1.0) {
        bail!("copy_gate_bias_prior must be a probability in (0, 1), got {prior}");
    }
    let logit = ((prior / (1.0 - prior)).ln()) as f32;
    let data = varmap.data().lock().unwrap();
    let mut matched = 0usize;
    for (name, var) in data
        .iter()
        .filter(|(name, _)| name.ends_with("copy_gate.bias"))
    {
        var.set(&Tensor::full(logit, var.shape().dims(), var.device())?)
            .map_err(|error| anyhow::anyhow!("restore {name}: {error}"))?;
        matched += 1;
    }
    if matched == 0 {
        bail!("copy_gate_bias_prior is configured but no copy-gate bias tensor exists");
    }
    Ok(())
}

/// Default side length of square input patches (`64 / PATCH_SIZE` grid).
pub const PATCH_SIZE: usize = 4;
/// Retained legacy patch side for explicit patch-8 comparisons.
pub const LEGACY_PATCH_SIZE: usize = 8;
/// Default spatial latent grid side. Configured patch-8 models use an 8×8 grid.
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

#[derive(Clone)]
struct ActionFilm {
    gamma: Tensor,
    beta: Tensor,
}

impl ActionFilm {
    fn neutral_like(latent: &Tensor) -> Result<Self> {
        let (batch, channels, _, _) = latent.dims4()?;
        let zeros = Tensor::zeros((batch, channels, 1, 1), latent.dtype(), latent.device())?;
        Ok(Self {
            gamma: zeros.ones_like()?,
            beta: zeros,
        })
    }
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

    fn conv_product_f32(&self, conv: &Conv2d, input: &Tensor) -> Result<Tensor> {
        let config = *conv.config();
        let input = input.to_dtype(DType::BF16)?;
        let weight = conv.weight().to_dtype(DType::BF16)?;
        // Candle's generic CPU conv lowers through a matmul implementation
        // that currently rejects BF16. Round both already-quantized operands
        // back to F32 there to emulate BF16 operands with F32 accumulation;
        // CUDA keeps the actual BF16 convolution required by the treatment.
        let (input, weight) = if input.device().is_cpu() {
            (input.to_dtype(DType::F32)?, weight.to_dtype(DType::F32)?)
        } else {
            (input, weight)
        };
        let output = input.conv2d_with_algo(
            &weight,
            config.padding,
            config.stride,
            config.dilation,
            config.groups,
            config.cudnn_fwd_algo,
        )?;
        let output = output.to_dtype(DType::F32)?;
        match conv.bias() {
            None => Ok(output),
            Some(bias) => {
                let channels = bias.dims1()?;
                output
                    .broadcast_add(&bias.reshape((1, channels, 1, 1))?)
                    .map_err(Into::into)
            }
        }
    }

    fn forward(&self, h: &Tensor, film: &ActionFilm, bf16: bool) -> Result<Tensor> {
        if !bf16 {
            let hidden = self.c1.forward(h)?.silu()?;
            let hidden = hidden
                .broadcast_mul(&film.gamma)?
                .broadcast_add(&film.beta)?;
            let delta = self.c2.forward(&hidden)?;
            return h.add(&delta).map_err(Into::into);
        }

        // Bias stays on the F32 side of each cast island. This avoids BF16
        // bias quantization and returns c1 to F32 before SiLU/FiLM, while c2
        // returns its delta to F32 before the residual addition.
        let hidden = self.conv_product_f32(&self.c1, h)?.silu()?;
        let hidden = hidden
            .broadcast_mul(&film.gamma)?
            .broadcast_add(&film.beta)?;
        let delta = self.conv_product_f32(&self.c2, &hidden)?;
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
    fn new(cell_dim: usize, patch_size: usize, vb: VarBuilder) -> Result<Self> {
        if !FRAME_SIDE.is_multiple_of(patch_size) {
            bail!("FRAME_SIDE must be divisible by patch_size");
        }
        let patch_cfg = Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let conv_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            patch: conv2d(PIXEL_EMB_DIM, 32, patch_size, patch_cfg, vb.pp("patch"))?,
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
    pub reliability_logit: Tensor,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseAInferenceCheck {
    pub passed: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseAInferenceCapabilities {
    pub patch4_grid: PhaseAInferenceCheck,
    pub spatial_prefix_faithful: PhaseAInferenceCheck,
    pub action_faithful_ptrm: PhaseAInferenceCheck,
    pub composed_decode_available: PhaseAInferenceCheck,
    pub null_action_row_present: PhaseAInferenceCheck,
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
    positional_value_readout: bool,
    pixel_emb: Embedding,
    encoder: GridEncoder,
    action_emb: Embedding,
    action_proj: Linear,
    /// Zero-initialized additive projection of family and operator colors.
    operator_conditioning_proj: Option<Linear>,
    action_film_gamma: Linear,
    action_film_beta: Linear,
    coord_proj: Linear,
    /// Optional `B×4×grid×grid` ACTION6 coordinate field projection.
    spatial_action_proj: Option<Conv2d>,
    goal_proj: Linear,
    block: GridResidualBlock,
    /// Scalar copy-bypass gate `a` (small nonzero init) when enabled.
    y_copy_bypass_alpha: Option<Tensor>,
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
        let positional = cfg.positional_value_readout;
        Self::new_with_positional_value_readout(cfg, positional, vb)
    }

    /// Build with position-aware readout values. The default constructor keeps
    /// existing checkpoint parameter names and the legacy pooled readout.
    pub fn new_with_positional_value_readout(
        cfg: ModelConfig,
        positional_value_readout: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
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
        let action_decoder = (cfg.world_core_v2 || cfg.world_core_v5)
            .then(|| linear(cfg.hidden_dim, ACTION_VOCAB, vb.pp("action_decoder")))
            .transpose()?;
        let coordinate_decoder = (cfg.world_core_v2 || cfg.world_core_v5)
            .then(|| linear(cfg.hidden_dim, 2, vb.pp("coordinate_decoder")))
            .transpose()?;
        Ok(Self {
            pixel_emb: embedding(PALETTE_SIZE, PIXEL_EMB_DIM, vb.pp("pixel_emb"))?,
            encoder: GridEncoder::new(cfg.hidden_dim, cfg.patch_size, vb.pp("encoder"))?,
            action_emb: embedding(ACTION_VOCAB, cfg.action_dim, vb.pp("action_emb"))?,
            action_proj: linear(cfg.action_dim, cfg.hidden_dim, vb.pp("action_proj"))?,
            operator_conditioning_proj: cfg
                .world_core_v5
                .then(|| {
                    zero_initialized_linear(
                        OPERATOR_CONDITION_DIM,
                        cfg.hidden_dim,
                        vb.pp("operator_conditioning_proj"),
                    )
                })
                .transpose()?,
            action_film_gamma: zero_initialized_linear(
                cfg.action_dim,
                cfg.hidden_dim,
                vb.pp("action_film_gamma"),
            )?,
            action_film_beta: zero_initialized_linear(
                cfg.action_dim,
                cfg.hidden_dim,
                vb.pp("action_film_beta"),
            )?,
            coord_proj: linear(2, cfg.hidden_dim, vb.pp("coord_proj"))?,
            spatial_action_proj,
            goal_proj: linear(cfg.goal_dim, cfg.hidden_dim, vb.pp("goal_proj"))?,
            block: GridResidualBlock::new(cfg.hidden_dim, vb.pp("block"))?,
            // Created only when the flag is on so default-off checkpoints keep
            // their exact parameter set. No ".weight" suffix: the tiny gate
            // must route to AdamW, never Muon.
            y_copy_bypass_alpha: cfg
                .copy_bypass_gate
                .then(|| {
                    vb.get_with_hints(
                        (1usize, 1usize, 1usize, 1usize),
                        "y_copy_bypass_alpha",
                        candle_nn::Init::Const(COPY_BYPASS_INITIAL_ALPHA),
                    )
                })
                .transpose()?,
            event_head: linear(cfg.hidden_dim * 2, cfg.num_events, vb.pp("event_head"))?,
            q_head: linear(cfg.hidden_dim, 1, vb.pp("q_head"))?,
            reliability_head: linear(cfg.hidden_dim, 1, vb.pp("reliability_head"))?,
            consumer_readout: ConsumerReadout::new(
                cfg.consumer_readout,
                cfg.hidden_dim,
                if positional_value_readout {
                    cfg.latent_grid()
                } else {
                    FRAME_SIDE / LEGACY_PATCH_SIZE
                },
                positional_value_readout,
                vb.pp("consumer_readout"),
            )?,
            prefix_head: linear(cfg.hidden_dim * 2, cfg.hidden_dim, vb.pp("prefix_head"))?,
            spatial_prefix_head,
            action_decoder,
            coordinate_decoder,
            sigreg_projector,
            patch_histogram_grounding: PatchHistogramGrounding::new(
                cfg.hidden_dim,
                cfg.patch_size,
                vb.pp("grounding_head"),
            )?,
            exact_patch_grounding: cfg
                .world_core_v4
                .then(|| {
                    ExactPatchGrounding::new(
                        cfg.hidden_dim,
                        cfg.patch_size,
                        cfg.copy_gate_bias_prior,
                        cfg.decode_composition,
                        vb.pp("exact_grounding_head"),
                    )
                })
                .transpose()?,
            positional_value_readout,
            config: cfg,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Learned scalar for the optional latent copy-bypass treatment.
    pub fn copy_bypass_alpha(&self) -> Result<Option<f64>> {
        self.y_copy_bypass_alpha
            .as_ref()
            .map(|alpha| {
                let value = alpha
                    .detach()
                    .to_dtype(DType::F32)?
                    .reshape(())?
                    .to_scalar::<f32>()
                    .map(f64::from)
                    .map_err(anyhow::Error::from)?;
                if !value.is_finite() {
                    bail!("copy-bypass alpha is non-finite");
                }
                Ok(value)
            })
            .transpose()
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
        self.exact_gameplay_logits_trainable(latents)
    }

    /// Decode predicted latents without severing the predictor/encoder graph.
    /// Foundation-v2's primary pixel loss must use this explicit seam.
    pub fn exact_gameplay_logits_trainable(&self, latents: &Tensor) -> Result<Tensor> {
        self.exact_patch_grounding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("exact grounding requires world-core-v4"))?
            .gameplay_logits(latents)
    }

    /// Detached exact logits for observer labels and diagnostics only.
    pub fn exact_gameplay_logits_detached(&self, latents: &Tensor) -> Result<Tensor> {
        Ok(self.exact_gameplay_logits_trainable(latents)?.detach())
    }

    /// Per-pixel probability that evaluation should use the predicted colour
    /// instead of copying the current observation.
    pub fn exact_copy_gate(&self, predicted: &Tensor) -> Result<Tensor> {
        self.exact_patch_grounding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("copy gate requires world-core-v4"))?
            .copy_gate(predicted)
    }

    /// Trainable per-pixel copy/change logits for the balanced gate BCE loss.
    pub fn exact_copy_gate_logits_trainable(&self, predicted: &Tensor) -> Result<Tensor> {
        self.exact_patch_grounding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("copy gate requires world-core-v4"))?
            .copy_gate_logits(predicted)
    }

    /// Discrete gameplay decode used by evaluation, composed per the
    /// configured [`DecodeComposition`] (legacy hard gate by default).
    pub fn composed_gameplay_decode(
        &self,
        predicted: &Tensor,
        current_frames: &Tensor,
    ) -> Result<Tensor> {
        self.exact_patch_grounding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("composed decode requires world-core-v4"))?
            .compose_gameplay_pixels(predicted, current_frames)
    }

    /// Compose precomputed exact-decoder logits and copy-gate probabilities.
    /// This avoids repeating both projections when an observer target shares
    /// the primary prediction latent.
    pub fn composed_gameplay_decode_from_parts(
        &self,
        gameplay_logits: &Tensor,
        copy_gate: &Tensor,
        current_pixels: &Tensor,
    ) -> Result<Tensor> {
        self.exact_patch_grounding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("composed decode requires world-core-v4"))?
            .compose_gameplay_pixels_from_parts(gameplay_logits, copy_gate, current_pixels)
    }

    /// Compatibility seam for observer labels. This intentionally uses the
    /// deployed copy-gate composition rather than raw decoder colours.
    pub fn exact_transition_correctness(
        &self,
        predicted: &Tensor,
        current_frames: &Tensor,
        next_frames: &Tensor,
    ) -> Result<Tensor> {
        self.composed_transition_correctness(predicted, current_frames, next_frames)
    }

    pub fn raw_decoder_transition_correctness(
        &self,
        predicted: &Tensor,
        current_frames: &Tensor,
        next_frames: &Tensor,
    ) -> Result<Tensor> {
        self.exact_patch_grounding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("exact grounding requires world-core-v4"))?
            .raw_decoder_transition_correctness(predicted, current_frames, next_frames)
    }

    pub fn composed_transition_correctness(
        &self,
        predicted: &Tensor,
        current_frames: &Tensor,
        next_frames: &Tensor,
    ) -> Result<Tensor> {
        self.exact_patch_grounding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("exact grounding requires world-core-v4"))?
            .composed_transition_correctness(predicted, current_frames, next_frames)
    }

    /// Full V4's single canonical BxC state. It is both regularized and
    /// consumed by transition/observer heads; this is a Tofy adaptation, not
    /// the separate loss-only projectors used by LeWorldModel.
    pub fn canonical_representation(&self, spatial: &Tensor) -> Result<Tensor> {
        rms_norm_latent(&self.read_consumer(spatial)?)
    }

    /// Preserve the established 8×8 SpatialQuery interface while the v5
    /// dynamics grid becomes 16×16. Patch-4 tokens are averaged in exact 2×2
    /// groups before the canonical readout; patch-8 models pass through.
    fn read_consumer(&self, spatial: &Tensor) -> Result<Tensor> {
        if self.positional_value_readout {
            return self.consumer_readout.forward(spatial).map_err(Into::into);
        }
        let (_, _, height, width) = spatial.dims4()?;
        let readout_grid = FRAME_SIDE / LEGACY_PATCH_SIZE;
        if height == readout_grid && width == readout_grid {
            return self.consumer_readout.forward(spatial).map_err(Into::into);
        }
        if height != width || !height.is_multiple_of(readout_grid) {
            bail!(
                "consumer spatial grid must downsample exactly to {readout_grid}x{readout_grid}, got {height}x{width}"
            );
        }
        self.consumer_readout
            .forward(&spatial.avg_pool2d(height / readout_grid)?)
            .map_err(Into::into)
    }

    /// Encode palette-index frames into the shared latent space.
    pub fn encode_state(&self, frames: &Tensor) -> Result<Tensor> {
        let embedded = self.embed_frames(frames, false)?;
        rms_norm_latent(&self.encoder.forward(&embedded, self.config.bf16_conv)?)
    }

    fn encode_state_pair_raw(
        &self,
        frames: &Tensor,
        next_frames: &Tensor,
        status_already_empty: bool,
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
        let embedded = self.embed_frames(&both, status_already_empty)?;
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
        let (current, next) = self.encode_state_pair_raw(frames, next_frames, false)?;
        Ok((rms_norm_latent(&current)?, rms_norm_latent(&next)?))
    }

    /// Return normalized dynamics latents together with their pre-RMS source tensors
    /// and optional `T×B×D` projector embeddings used only by SIGReg treatments.
    pub fn encode_state_pair_for_training(
        &self,
        frames: &Tensor,
        next_frames: &Tensor,
    ) -> Result<TrainingEncodedPair> {
        self.encode_state_pair_for_training_impl(frames, next_frames, false)
    }

    /// Paired training encode for host-staged EMPTY status rows.
    pub fn encode_state_pair_for_training_staged(
        &self,
        frames: &Tensor,
        next_frames: &Tensor,
    ) -> Result<TrainingEncodedPair> {
        self.encode_state_pair_for_training_impl(frames, next_frames, true)
    }

    fn encode_state_pair_for_training_impl(
        &self,
        frames: &Tensor,
        next_frames: &Tensor,
        status_already_empty: bool,
    ) -> Result<TrainingEncodedPair> {
        let (current_raw, next_raw) =
            self.encode_state_pair_raw(frames, next_frames, status_already_empty)?;
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
        goal_features: &Tensor,
    ) -> Result<RepresentationDiagnosticOutput> {
        let operator_conditioning = unknown_operator_conditioning(frames.dim(0)?, frames.device())?;
        self.representation_diagnostic_with_operator_conditioning(
            frames,
            next_frames,
            actions,
            action_coords,
            goal_features,
            &operator_conditioning,
        )
    }

    pub fn representation_diagnostic_with_operator_conditioning(
        &self,
        frames: &Tensor,
        next_frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        _goal_features: &Tensor,
        operator_conditioning: &Tensor,
    ) -> Result<RepresentationDiagnosticOutput> {
        let (current_raw, target_raw) = self.encode_state_pair_raw(frames, next_frames, false)?;
        let current = rms_norm_latent(&current_raw)?;
        let target = rms_norm_latent(&target_raw)?;
        let (x, film) = self.prepare_action_conditioning(
            &current,
            None,
            actions,
            action_coords,
            Some(operator_conditioning),
        )?;
        let y_init = self.config.warm_start_y.then(|| current.clone());
        let recursion = self.run_latent_recursion(
            &x,
            &film,
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
                    self.read_consumer(&prediction)?.detach()
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

    fn embed_frames(&self, frames: &Tensor, status_already_empty: bool) -> Result<Tensor> {
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
        if !(self.config.world_core_v2 || self.config.world_core_v4)
            || (status_already_empty && c == 1)
        {
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
        let operator_conditioning = unknown_operator_conditioning(state.dim(0)?, state.device())?;
        self.add_action_with_canonical_and_operator(
            state,
            None,
            actions,
            action_coords,
            &operator_conditioning,
        )
    }

    fn action_embedding(&self, actions: &Tensor, batch: usize) -> Result<(Tensor, Tensor)> {
        let actions = match actions.rank() {
            1 => actions.clone(),
            2 if actions.dim(1)? == 1 => actions.reshape((batch,))?,
            rank => bail!("actions must be shape [B] or [B,1], got rank {rank}"),
        };
        if actions.dim(0)? != batch {
            bail!(
                "action batch {} does not match latent batch {batch}",
                actions.dim(0)?
            );
        }
        let embedding = self.action_emb.forward(&actions)?;
        Ok((actions, embedding))
    }

    fn action_film_from_embedding(&self, embedding: &Tensor, batch: usize) -> Result<ActionFilm> {
        Ok(ActionFilm {
            gamma: self
                .action_film_gamma
                .forward(embedding)?
                .affine(1.0, 1.0)?
                .reshape((batch, self.config.hidden_dim, 1, 1))?,
            beta: self.action_film_beta.forward(embedding)?.reshape((
                batch,
                self.config.hidden_dim,
                1,
                1,
            ))?,
        })
    }

    /// Shared action-conditioning implementation. Full V4 training supplies
    /// the already-required canonical state so action conditioning, SIGReg,
    /// and the prediction objective reuse one autograd node.
    fn add_action_with_canonical_and_operator(
        &self,
        state: &Tensor,
        canonical: Option<&Tensor>,
        actions: &Tensor,
        action_coords: &Tensor,
        operator_conditioning: &Tensor,
    ) -> Result<Tensor> {
        let b = state.dim(0)?;
        let (actions, embedding) = self.action_embedding(actions, b)?;
        self.add_action_with_embedding(
            state,
            canonical,
            &actions,
            &embedding,
            action_coords,
            operator_conditioning,
        )
    }

    fn add_action_with_embedding(
        &self,
        state: &Tensor,
        canonical: Option<&Tensor>,
        actions: &Tensor,
        action_embedding: &Tensor,
        action_coords: &Tensor,
        operator_conditioning: &Tensor,
    ) -> Result<Tensor> {
        let b = state.dim(0)?;
        let latent_grid = self.config.latent_grid();
        if state.dims4()? != (b, self.config.hidden_dim, latent_grid, latent_grid) {
            bail!(
                "state must be Bx{}x{}x{}, got {:?}",
                self.config.hidden_dim,
                latent_grid,
                latent_grid,
                state.dims()
            );
        }
        if action_coords.dims2()? != (b, 2) {
            bail!("action_coords must have shape [B,2]");
        }
        if operator_conditioning.dims2()? != (b, OPERATOR_CONDITION_DIM) {
            bail!("operator_conditioning must have shape [B,{OPERATOR_CONDITION_DIM}]");
        }
        let action = self.action_proj.forward(action_embedding)?;
        let action_bias = action.reshape((b, self.config.hidden_dim, 1, 1))?;
        let mut conditioned = state.broadcast_add(&action_bias)?;
        if let Some(projection) = &self.operator_conditioning_proj {
            let operator_bias = projection.forward(operator_conditioning)?.reshape((
                b,
                self.config.hidden_dim,
                1,
                1,
            ))?;
            conditioned = conditioned.broadcast_add(&operator_bias)?;
        }
        let coord_bias = if !self.config.spatial_action_field || self.config.spatial_action_residual
        {
            let mut coords = self.coord_proj.forward(action_coords)?;
            if self.config.world_core_v2 {
                let coordinate_active = actions
                    .eq(6u32)?
                    .to_dtype(coords.dtype())?
                    .reshape((b, 1))?
                    .broadcast_as(coords.dims())?;
                coords = coords.mul(&coordinate_active)?;
            }
            Some(coords.reshape((b, self.config.hidden_dim, 1, 1))?)
        } else {
            None
        };
        let conditioned = if self.config.world_core_v4 {
            let canonical = match canonical {
                Some(canonical) => canonical.clone(),
                None => self.canonical_representation(state)?,
            }
            .reshape((b, self.config.hidden_dim, 1, 1))?;
            conditioned.broadcast_add(&canonical)?
        } else {
            conditioned
        };
        if !self.config.spatial_action_field {
            return conditioned
                .broadcast_add(coord_bias.as_ref().expect("non-spatial coordinate bias"))
                .map_err(Into::into);
        }

        let field = self.spatial_action_field(actions, action_coords)?;
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
                .broadcast_add(
                    coord_bias
                        .as_ref()
                        .expect("residual spatial coordinate bias"),
                )?
                .add(&projection.affine(self.config.spatial_action_residual_scale, 0.0)?)
                .map_err(Into::into)
        } else {
            conditioned.add(&projection).map_err(Into::into)
        }
    }

    /// One action-embedding gather per recurrence: conditioning and FiLM share
    /// the same embedding, and callers without episode-operator provenance get
    /// the UNKNOWN conditioning row.
    fn prepare_action_conditioning(
        &self,
        state: &Tensor,
        canonical: Option<&Tensor>,
        actions: &Tensor,
        action_coords: &Tensor,
        operator_conditioning: Option<&Tensor>,
    ) -> Result<(Tensor, ActionFilm)> {
        let batch = state.dim(0)?;
        let unknown;
        let operator_conditioning = match operator_conditioning {
            Some(conditioning) => conditioning,
            None => {
                unknown = unknown_operator_conditioning(batch, state.device())?;
                &unknown
            }
        };
        let (actions, embedding) = self.action_embedding(actions, batch)?;
        let conditioned = self.add_action_with_embedding(
            state,
            canonical,
            &actions,
            &embedding,
            action_coords,
            operator_conditioning,
        )?;
        let film = self.action_film_from_embedding(&embedding, batch)?;
        Ok((conditioned, film))
    }

    /// ACTION6 coordinate conditioning over the latent grid.
    ///
    /// Coordinates are normalized to `[0, 1]`. The four channels are a localized
    /// impulse, relative x/y offsets, and an ACTION6-active mask. These raw
    /// channels are zero for simple actions, but the following biased 1x1
    /// projection may still contribute a learned constant spatial offset.
    /// Placeholder coordinates remain shape-checked by [`Self::add_action`].
    fn spatial_action_field(&self, actions: &Tensor, action_coords: &Tensor) -> Result<Tensor> {
        let b = actions.dim(0)?;
        if action_coords.dims2()? != (b, 2) {
            bail!("action_coords must have shape [B,2]");
        }
        let coords = action_coords.to_dtype(DType::F32)?;
        let latent_grid = self.config.latent_grid();
        let x = coords.narrow(1, 0, 1)?.reshape((b, 1, 1, 1))?;
        let y = coords.narrow(1, 1, 1)?.reshape((b, 1, 1, 1))?;
        let axis = Tensor::arange(0f32, latent_grid as f32, coords.device())?
            .affine(1.0 / (latent_grid - 1) as f64, 0.0)?;
        let grid_x = axis.reshape((1, 1, 1, latent_grid))?;
        let grid_y = axis.reshape((1, 1, latent_grid, 1))?;
        let dx = grid_x
            .broadcast_sub(&x)?
            .broadcast_as((b, 1, latent_grid, latent_grid))?;
        let dy = grid_y
            .broadcast_sub(&y)?
            .broadcast_as((b, 1, latent_grid, latent_grid))?;
        let active = actions
            .eq(6u32)?
            .to_dtype(DType::F32)?
            .reshape((b, 1, 1, 1))?
            .broadcast_as((b, 1, latent_grid, latent_grid))?;
        // Legacy -16 is fixed in normalized units (sigma ~0.18, ~11 board
        // pixels): sharp on the 8x8 grid but nearly flat over a +-2-cell
        // neighborhood at patch 4. Grid scaling pins sigma to one latent
        // cell: exponent -(d_cells^2)/2 = -((grid-1)^2/2)*(dx^2+dy^2).
        let impulse_coeff = if self.config.grid_scaled_action_impulse {
            -(((latent_grid - 1) * (latent_grid - 1)) as f64) / 2.0
        } else {
            -16.0
        };
        let impulse = dx
            .sqr()?
            .add(&dy.sqr()?)?
            .affine(impulse_coeff, 0.0)?
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

    fn prepare_transition_with_operator_conditioning(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        operator_conditioning: &Tensor,
    ) -> Result<(Tensor, ActionFilm, Tensor, Option<Tensor>)> {
        let encoded = self.encode_state(frames)?;
        let (x, film) = self.prepare_action_conditioning(
            &encoded,
            None,
            actions,
            action_coords,
            Some(operator_conditioning),
        )?;
        let state = self.config.warm_start_y.then(|| encoded.clone());
        let goal_h = self.project_goal(goal_features)?;
        Ok((x, film, goal_h, state))
    }

    fn heads(&self, y: &Tensor, goal_h: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let readout = if self.config.world_core_v4 {
            self.canonical_representation(y)?
        } else {
            self.read_consumer(y)?
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
            self.read_consumer(y)?
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
        z: Option<&Tensor>,
        film: &ActionFilm,
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
        let mut z = z.cloned();
        let mut y = y.clone();
        let xy = if self.config.world_core_v4 {
            Some(x.add(&y)?)
        } else {
            None
        };
        for _ in 0..inner_steps {
            let step_seed = noise_seed_base.map(|s| s.wrapping_add(*noise_counter));
            *noise_counter = noise_counter.wrapping_add(1);
            z = z
                .map(|z| self.maybe_noise_z(&z, sigma, step_seed))
                .transpose()?;
            let inp = match (&xy, &z) {
                (Some(xy), Some(z)) => xy.add(z)?,
                (Some(xy), None) => xy.clone(),
                (None, Some(z)) => x.add(&y)?.add(z)?,
                (None, None) => x.add(&y)?,
            };
            z = Some(
                self.block
                    .forward(&inp, film, self.config.bf16_recurrent_core)?,
            );
        }
        let z = z.expect("inner_steps >= 1 initializes z");
        let inp = if self.config.world_core_v4 {
            xy.expect("Full V4 x+y was prepared").add(&z)?
        } else {
            y.add(&z)?
        };
        y = self
            .block
            .forward(&inp, film, self.config.bf16_recurrent_core)?;
        Ok((y, z))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_latent_recursion(
        &self,
        x: &Tensor,
        film: &ActionFilm,
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
        // At zero noise the mathematical initial state is absent: the first
        // refinement consumes x+y directly and materializes z from its output.
        let mut z = (sigma != 0.0)
            .then(|| Tensor::zeros((b, c, hh, ww), x.dtype(), device))
            .transpose()?;
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
                z.as_ref(),
                film,
                depth.inner_steps,
                sigma,
                noise_seed_base,
                &mut noise_counter,
            )?;
            let candidate = if self.config.residual_y_update {
                rms_norm_latent(&y.add(&ny)?)?
            } else {
                rms_norm_latent(&ny)?
            };
            let candidate = candidate.clamp(-32.0, 32.0)?;
            y = match &self.y_copy_bypass_alpha {
                // y' = y + a*(l - y): a=0 is exact latent copy for any state
                // inside the clamp envelope; a=1 reproduces the legacy update
                // algebraically (tested to 1e-6 in f32). Fresh runs use a
                // small nonzero a so prediction gradients reach l immediately.
                // The gate is unconstrained, so the interpolation is re-clamped
                // to keep the legacy activation envelope if a leaves [0, 1].
                Some(alpha) => y
                    .add(&candidate.sub(&y)?.broadcast_mul(alpha)?)?
                    .clamp(-32.0, 32.0)?,
                None => candidate,
            };
            z = Some(nz);
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
        film: &ActionFilm,
        goal_h: &Tensor,
        sigma: f64,
        noise_seed_base: Option<u64>,
        depth: RecursionDepth,
        y_init: Option<Tensor>,
        opts: RecursionOpts,
    ) -> Result<ForwardOutput> {
        let latent =
            self.run_latent_recursion(x, film, sigma, noise_seed_base, depth, y_init, opts)?;
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
            self.read_consumer(y)?
        };
        self.q_head.forward(&readout).map_err(Into::into)
    }

    pub fn reliability_logit_from_y(&self, y: &Tensor) -> Result<Tensor> {
        let readout = if self.config.world_core_v4 {
            self.canonical_representation(y)?
        } else {
            self.read_consumer(y)?
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
        if self.config.world_core_v4 {
            return self.predict_latent_with_depth(
                state,
                actions,
                action_coords,
                RecursionDepth {
                    inner_steps: 1,
                    outer_steps: 1,
                },
            );
        }
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
    /// displacement. This head is available in world-core-v2 and the
    /// foundation-v2 exact-decoder topology, and is trained only on factual,
    /// board-effect-bearing branches.
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
            .ok_or_else(|| anyhow::anyhow!("action decoder requires an action-faithful model"))?;
        let coordinate_decoder = self.coordinate_decoder.as_ref().ok_or_else(|| {
            anyhow::anyhow!("coordinate decoder requires an action-faithful model")
        })?;
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
        // Per-sample cosine over flattened C*H*W, averaged across the batch.
        let batch = y_before.dim(0)?;
        let a = y_before.detach().reshape((batch, ()))?;
        let b = y_after.detach().reshape((batch, ()))?;
        let dot = a.mul(&b)?.sum(1)?;
        let norm_a = a.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f64::INFINITY)?;
        let norm_b = b.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f64::INFINITY)?;
        let cosine = dot
            .div(&norm_a)?
            .div(&norm_b)?
            .mean_all()?
            .to_scalar::<f32>()? as f64;
        Ok(RecursionStepProbe {
            outer_step: outer_idx,
            mean_residual_norm: res_norm,
            mean_latent_norm: lat_norm,
            mean_amplification: amplification,
            mean_step_cosine: cosine,
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
        let operator_conditioning = unknown_operator_conditioning(frames.dim(0)?, frames.device())?;
        self.forward_with_depth_and_operator_conditioning(
            frames,
            actions,
            action_coords,
            goal_features,
            &operator_conditioning,
            depth,
            z_noise_sigma,
            noise_seed,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_depth_and_operator_conditioning(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        operator_conditioning: &Tensor,
        depth: RecursionDepth,
        z_noise_sigma: f64,
        noise_seed: Option<u64>,
    ) -> Result<ForwardOutput> {
        let (x, film, goal_h, y_init) = self.prepare_transition_with_operator_conditioning(
            frames,
            actions,
            action_coords,
            goal_features,
            operator_conditioning,
        )?;
        self.run_recursion(
            &x,
            &film,
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
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        depth: RecursionDepth,
        z_noise_sigma: f64,
        noise_seed: Option<u64>,
        recursion: RecursionOpts,
    ) -> Result<ForwardOutput> {
        let operator_conditioning =
            unknown_operator_conditioning(cur_state.dim(0)?, cur_state.device())?;
        self.forward_from_encoded_state_with_operator_conditioning(
            cur_state,
            frames,
            actions,
            action_coords,
            goal_features,
            &operator_conditioning,
            depth,
            z_noise_sigma,
            noise_seed,
            recursion,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_from_encoded_state_with_operator_conditioning(
        &self,
        cur_state: &Tensor,
        _frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        operator_conditioning: &Tensor,
        depth: RecursionDepth,
        z_noise_sigma: f64,
        noise_seed: Option<u64>,
        recursion: RecursionOpts,
    ) -> Result<ForwardOutput> {
        let (x, film) = self.prepare_action_conditioning(
            cur_state,
            None,
            actions,
            action_coords,
            Some(operator_conditioning),
        )?;
        let goal_h = self.project_goal(goal_features)?;
        let y_init = if self.config.warm_start_y {
            Some(cur_state.clone())
        } else {
            None
        };
        self.run_recursion(
            &x,
            &film,
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
        let operator_conditioning =
            unknown_operator_conditioning(cur_state.dim(0)?, cur_state.device())?;
        self.training_latents_from_encoded_state_with_operator_conditioning(
            cur_state,
            actions,
            action_coords,
            &operator_conditioning,
            depth,
            z_noise_sigma,
            noise_seed,
            recursion,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn training_latents_from_encoded_state_with_operator_conditioning(
        &self,
        cur_state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        operator_conditioning: &Tensor,
        depth: RecursionDepth,
        z_noise_sigma: f64,
        noise_seed: Option<u64>,
        recursion: RecursionOpts,
    ) -> Result<LatentRecursionOutput> {
        let (x, film) = self.prepare_action_conditioning(
            cur_state,
            None,
            actions,
            action_coords,
            Some(operator_conditioning),
        )?;
        let y_init = self.config.warm_start_y.then(|| cur_state.clone());
        self.run_latent_recursion(
            &x,
            &film,
            z_noise_sigma,
            noise_seed,
            depth,
            y_init,
            recursion,
        )
    }

    /// Full V4 training recursion with a caller-owned canonical current state.
    /// This keeps the canonical representation as the single consumed state
    /// across conditioning and objective terms.
    #[allow(clippy::too_many_arguments)]
    pub fn full_v4_training_latents_from_encoded_state(
        &self,
        cur_state: &Tensor,
        current_canonical: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        depth: RecursionDepth,
        z_noise_sigma: f64,
        noise_seed: Option<u64>,
        recursion: RecursionOpts,
    ) -> Result<LatentRecursionOutput> {
        let operator_conditioning =
            unknown_operator_conditioning(cur_state.dim(0)?, cur_state.device())?;
        self.full_v4_training_latents_from_encoded_state_with_operator_conditioning(
            cur_state,
            current_canonical,
            actions,
            action_coords,
            &operator_conditioning,
            depth,
            z_noise_sigma,
            noise_seed,
            recursion,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn full_v4_training_latents_from_encoded_state_with_operator_conditioning(
        &self,
        cur_state: &Tensor,
        current_canonical: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        operator_conditioning: &Tensor,
        depth: RecursionDepth,
        z_noise_sigma: f64,
        noise_seed: Option<u64>,
        recursion: RecursionOpts,
    ) -> Result<LatentRecursionOutput> {
        if !self.config.world_core_v4 {
            bail!("Full V4 training recursion requires world_core_v4");
        }
        let (x, film) = self.prepare_action_conditioning(
            cur_state,
            Some(current_canonical),
            actions,
            action_coords,
            Some(operator_conditioning),
        )?;
        let y_init = self.config.warm_start_y.then(|| cur_state.clone());
        self.run_latent_recursion(
            &x,
            &film,
            z_noise_sigma,
            noise_seed,
            depth,
            y_init,
            recursion,
        )
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

    pub fn forward_with_operator_conditioning(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        operator_conditioning: &Tensor,
    ) -> Result<ForwardOutput> {
        self.forward_with_depth_and_operator_conditioning(
            frames,
            actions,
            action_coords,
            goal_features,
            operator_conditioning,
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
        let operator_conditioning = unknown_operator_conditioning(frames.dim(0)?, frames.device())?;
        self.forward_with_outer_steps_and_operator_conditioning(
            frames,
            actions,
            action_coords,
            goal_features,
            &operator_conditioning,
            outer_steps,
        )
    }

    pub fn forward_with_outer_steps_and_operator_conditioning(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        operator_conditioning: &Tensor,
        outer_steps: usize,
    ) -> Result<ForwardOutput> {
        let (x, film, goal_h, y_init) = self.prepare_transition_with_operator_conditioning(
            frames,
            actions,
            action_coords,
            goal_features,
            operator_conditioning,
        )?;
        self.run_recursion(
            &x,
            &film,
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

    pub fn forward_from_latent_with_operator_conditioning(
        &self,
        state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        operator_conditioning: &Tensor,
    ) -> Result<ForwardOutput> {
        self.forward_from_latent_with_depth_and_operator_conditioning(
            state,
            actions,
            action_coords,
            goal_features,
            operator_conditioning,
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
        let operator_conditioning = unknown_operator_conditioning(state.dim(0)?, state.device())?;
        self.forward_from_latent_with_depth_and_operator_conditioning(
            state,
            actions,
            action_coords,
            goal_features,
            &operator_conditioning,
            depth,
        )
    }

    pub fn forward_from_latent_with_depth_and_operator_conditioning(
        &self,
        state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        operator_conditioning: &Tensor,
        depth: RecursionDepth,
    ) -> Result<ForwardOutput> {
        let (x, film) = self.prepare_action_conditioning(
            state,
            None,
            actions,
            action_coords,
            Some(operator_conditioning),
        )?;
        let goal_h = self.project_goal(goal_features)?;
        let y_init = if self.config.warm_start_y {
            Some(state.clone())
        } else {
            None
        };
        self.run_recursion(
            &x,
            &film,
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
        let (x, film) = self.prepare_action_conditioning(state, None, actions, action_coords, None)?;
        let y_init = self.config.warm_start_y.then(|| state.clone());
        Ok(self
            .run_latent_recursion(
                &x,
                &film,
                0.0,
                None,
                depth,
                y_init,
                RecursionOpts::training(true),
            )?
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

    pub fn forward_ptrm_with_operator_conditioning(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        operator_conditioning: &Tensor,
        ptrm: PtrmConfig,
    ) -> Result<PtrmOutput> {
        self.forward_ptrm_with_depth_and_operator_conditioning(
            frames,
            actions,
            action_coords,
            goal_features,
            operator_conditioning,
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
        let operator_conditioning = unknown_operator_conditioning(frames.dim(0)?, frames.device())?;
        self.forward_ptrm_with_depth_and_operator_conditioning(
            frames,
            actions,
            action_coords,
            goal_features,
            &operator_conditioning,
            depth,
            ptrm,
        )
    }

    pub fn forward_ptrm_with_depth_and_operator_conditioning(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        operator_conditioning: &Tensor,
        depth: RecursionDepth,
        ptrm: PtrmConfig,
    ) -> Result<PtrmOutput> {
        if ptrm.k == 0 {
            bail!("PTRM requires K >= 1");
        }
        if !ptrm.sigma.is_finite() || ptrm.sigma < 0.0 {
            bail!("PTRM sigma must be finite and non-negative");
        }
        let (x, film, goal_h, y_init) = self.prepare_transition_with_operator_conditioning(
            frames,
            actions,
            action_coords,
            goal_features,
            operator_conditioning,
        )?;
        self.forward_ptrm_prepared_with_film(&x, &film, &goal_h, y_init, depth, ptrm)
    }

    /// Action-faithful PTRM from an already encoded latent state.
    pub fn forward_ptrm_from_latent(
        &self,
        state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        depth: RecursionDepth,
        ptrm: PtrmConfig,
    ) -> Result<PtrmOutput> {
        let operator_conditioning = unknown_operator_conditioning(state.dim(0)?, state.device())?;
        self.forward_ptrm_from_latent_with_operator_conditioning(
            state,
            actions,
            action_coords,
            goal_features,
            &operator_conditioning,
            depth,
            ptrm,
        )
    }

    pub fn forward_ptrm_from_latent_with_operator_conditioning(
        &self,
        state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
        operator_conditioning: &Tensor,
        depth: RecursionDepth,
        ptrm: PtrmConfig,
    ) -> Result<PtrmOutput> {
        let (x, film) = self.prepare_action_conditioning(
            state,
            None,
            actions,
            action_coords,
            Some(operator_conditioning),
        )?;
        let goal_h = self.project_goal(goal_features)?;
        let y_init = self.config.warm_start_y.then(|| state.clone());
        self.forward_ptrm_prepared_with_film(&x, &film, &goal_h, y_init, depth, ptrm)
    }

    /// Legacy PTRM from a precomputed action-conditioned tensor. Because this
    /// seam does not receive action IDs, it retains identity FiLM.
    pub fn forward_ptrm_prepared(
        &self,
        x: &Tensor,
        goal_h: &Tensor,
        y_init: Option<Tensor>,
        depth: RecursionDepth,
        ptrm: PtrmConfig,
    ) -> Result<PtrmOutput> {
        if self.config.world_core_v5 {
            bail!("world-core-v5 PTRM requires action-aware preparation");
        }
        let film = ActionFilm::neutral_like(x)?;
        self.forward_ptrm_prepared_with_film(x, &film, goal_h, y_init, depth, ptrm)
    }

    fn forward_ptrm_prepared_with_film(
        &self,
        x: &Tensor,
        film: &ActionFilm,
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
        let latent_trajectories = self.ptrm_latent_trajectories(x, film, y_init, depth, ptrm)?;
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
                reliability_logit: out.reliability_logit,
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
        film: &ActionFilm,
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
                film,
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
        let (x, film) = self.prepare_action_conditioning(state, None, actions, action_coords, None)?;
        let y_init = self.config.warm_start_y.then(|| state.clone());
        self.ptrm_latent_trajectories(&x, &film, y_init, depth, ptrm)?
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

    /// Read-only Phase A deployment capability probe.
    pub fn phase_a_inference_capabilities(&self) -> PhaseAInferenceCapabilities {
        let check = |passed, reason: &str| PhaseAInferenceCheck {
            passed,
            reason: (!passed).then(|| reason.to_string()),
        };
        PhaseAInferenceCapabilities {
            patch4_grid: check(
                self.config.patch_size == PATCH_SIZE,
                "requires the canonical patch-4 latent grid",
            ),
            spatial_prefix_faithful: check(
                self.config.world_core_v4 && self.spatial_prefix_head.is_none(),
                "requires the world-core-v4 recurrence prefix path",
            ),
            action_faithful_ptrm: check(
                self.config.world_core_v5,
                "requires world-core-v5 action-aware PTRM preparation",
            ),
            composed_decode_available: check(
                self.exact_patch_grounding.is_some(),
                "requires the world-core-v4 exact decoder and copy gate",
            ),
            // Structural only: an id-0 embedding row exists. Whether the NULL
            // action was actually in the training range is a checkpoint
            // property this probe cannot verify.
            null_action_row_present: check(ACTION_VOCAB > 0, "action embedding has no id-0 row"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::train::reinit_varmap_deterministic;
    use candle_core::{DType, Device, IndexOp, Tensor, Var};
    use candle_nn::optim::{Optimizer, SGD};
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
    fn grid_scaled_impulse_rejects_world_core_without_spatial_field() {
        let config = ModelConfig {
            world_core_v2: true,
            spatial_action_field: false,
            grid_scaled_action_impulse: true,
            ..ModelConfig::default()
        };
        let error = config
            .validate()
            .expect_err("an impulse without its spatial field must be rejected");
        assert!(error
            .to_string()
            .contains("grid_scaled_action_impulse requires spatial_action_field=true"));
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
    fn zero_initialized_operator_conditioning_is_bit_identical_to_unknown_input() -> Result<()> {
        let device = Device::Cpu;
        let cfg = ModelConfig {
            world_core_v4: true,
            world_core_v5: true,
            spatial_action_field: true,
            consumer_readout: ConsumerReadoutTopology::SpatialQuery,
            ..tiny_cfg()
        };
        let varmap = VarMap::new();
        let model = WorldModel::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device))?;
        reinit_varmap_deterministic(&varmap, 97)?;
        zero_action_film_projections(&varmap)?;
        zero_operator_conditioning_projection(&varmap)?;
        let (frames, actions, coords, goals) = sample_batch(&device, 2, model.config.goal_dim)?;
        let baseline = model.forward(&frames, &actions, &coords, &goals)?;
        let mut values = vec![0f32; 2 * OPERATOR_CONDITION_DIM];
        for row in 0..2 {
            let base = row * OPERATOR_CONDITION_DIM;
            values[base + 2] = 1.0;
            values[base + OPERATOR_FAMILY_VOCAB + 7] = 1.0;
            values[base + OPERATOR_FAMILY_VOCAB + PALETTE_SIZE + 11] = 1.0;
            values[base + OPERATOR_FAMILY_VOCAB + 2 * PALETTE_SIZE + 4] = 1.0;
        }
        let attached = Tensor::from_vec(values, (2, OPERATOR_CONDITION_DIM), &device)?;
        let conditioned = model
            .forward_with_operator_conditioning(&frames, &actions, &coords, &goals, &attached)?;
        for (name, left, right) in [
            ("latent", &baseline.y, &conditioned.y),
            ("events", &baseline.event_logits, &conditioned.event_logits),
            ("q", &baseline.q_logit, &conditioned.q_logit),
            (
                "reliability",
                &baseline.reliability_logit,
                &conditioned.reliability_logit,
            ),
        ] {
            assert_eq!(
                left.flatten_all()?.to_vec1::<f32>()?,
                right.flatten_all()?.to_vec1::<f32>()?,
                "fresh output differs at {name}"
            );
        }
        let parameters = varmap.data().lock().unwrap();
        for (name, parameter) in parameters
            .iter()
            .filter(|(name, _)| name.starts_with("operator_conditioning_proj."))
        {
            assert!(
                parameter
                    .as_tensor()
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|value| *value == 0.0),
                "{name} was not zero initialized"
            );
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
    fn staged_empty_status_pair_matches_post_embedding_replacement_exactly() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = tiny_cfg();
        cfg.world_core_v2 = true;
        let varmap = VarMap::new();
        let model = WorldModel::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device))?;
        reinit_varmap_deterministic(&varmap, 73)?;

        let mut current = (0..2 * FRAME_SIDE * FRAME_SIDE)
            .map(|index| (index % PALETTE_SIZE) as u8)
            .collect::<Vec<_>>();
        let mut next = current
            .iter()
            .map(|value| (usize::from(*value) + 3) as u8 % PALETTE_SIZE as u8)
            .collect::<Vec<_>>();
        let original_current =
            Tensor::from_vec(current.clone(), (2, 1, FRAME_SIDE, FRAME_SIDE), &device)?;
        let original_next =
            Tensor::from_vec(next.clone(), (2, 1, FRAME_SIDE, FRAME_SIDE), &device)?;
        for frame in current.chunks_mut(FRAME_SIDE * FRAME_SIDE) {
            frame[(FRAME_SIDE - 1) * FRAME_SIDE..].fill(0);
        }
        for frame in next.chunks_mut(FRAME_SIDE * FRAME_SIDE) {
            frame[(FRAME_SIDE - 1) * FRAME_SIDE..].fill(0);
        }
        let staged_current = Tensor::from_vec(current, (2, 1, FRAME_SIDE, FRAME_SIDE), &device)?;
        let staged_next = Tensor::from_vec(next, (2, 1, FRAME_SIDE, FRAME_SIDE), &device)?;

        let (legacy_current, legacy_next) =
            model.encode_state_pair(&original_current, &original_next)?;
        let staged = model.encode_state_pair_for_training_staged(&staged_current, &staged_next)?;
        assert_eq!(
            legacy_current.flatten_all()?.to_vec1::<f32>()?,
            staged.current.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            legacy_next.flatten_all()?.to_vec1::<f32>()?,
            staged.next.flatten_all()?.to_vec1::<f32>()?
        );
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
    fn absent_initial_z_is_bit_identical_to_explicit_zero_at_zero_noise() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = tiny_cfg();
        cfg.inner_steps = 2;
        cfg.world_core_v4 = true;
        cfg.spatial_action_field = true;
        cfg.consumer_readout = ConsumerReadoutTopology::SpatialQuery;
        let varmap = VarMap::new();
        let model = WorldModel::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device))?;
        reinit_varmap_deterministic(&varmap, 89)?;
        let grid = model.config.latent_grid();
        let shape = (2, model.config.hidden_dim, grid, grid);
        let elem_count = shape.0 * shape.1 * shape.2 * shape.3;
        let x = Tensor::arange(0f32, elem_count as f32, &device)?
            .affine(1.0 / 1024.0, -1.0)?
            .reshape(shape)?;
        let y = Tensor::zeros(shape, DType::F32, &device)?;
        let actions = Tensor::new(&[1u32, 6], &device)?;
        let (_, action_embedding) = model.action_embedding(&actions, 2)?;
        let film = model.action_film_from_embedding(&action_embedding, 2)?;

        let mut optimized_counter = 0;
        let optimized =
            model.deep_step(&x, &y, None, &film, 2, 0.0, None, &mut optimized_counter)?;

        let mut legacy_z = Tensor::zeros(shape, DType::F32, &device)?;
        let mut legacy_y = y.clone();
        let xy = x.add(&legacy_y)?;
        for _ in 0..2 {
            let input = xy.add(&legacy_z)?;
            legacy_z = model.block.forward(&input, &film, false)?;
        }
        legacy_y = model.block.forward(&xy.add(&legacy_z)?, &film, false)?;

        assert_eq!(
            optimized.0.flatten_all()?.to_vec1::<f32>()?,
            legacy_y.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            optimized.1.flatten_all()?.to_vec1::<f32>()?,
            legacy_z.flatten_all()?.to_vec1::<f32>()?
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

    #[test]
    fn bf16_recurrent_flag_off_is_bit_exact_forward_and_backward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let block = GridResidualBlock::new(
            4,
            VarBuilder::from_varmap(&varmap, DType::F32, &device).pp("block"),
        )?;
        reinit_varmap_deterministic(&varmap, 211)?;
        let input = Var::from_tensor(
            &Tensor::arange(0f32, 64f32, &device)?
                .affine(1.0 / 32.0, -1.0)?
                .reshape((1, 4, 4, 4))?,
        )?;
        let film = ActionFilm {
            gamma: Tensor::from_vec(vec![1.0f32, 0.75, 1.25, 0.5], (1, 4, 1, 1), &device)?,
            beta: Tensor::from_vec(vec![0.0f32, 0.1, -0.2, 0.3], (1, 4, 1, 1), &device)?,
        };

        let actual = block.forward(input.as_tensor(), &film, false)?;
        let expected_hidden = block.c1.forward(input.as_tensor())?.silu()?;
        let expected_hidden = expected_hidden
            .broadcast_mul(&film.gamma)?
            .broadcast_add(&film.beta)?;
        let expected = input.as_tensor().add(&block.c2.forward(&expected_hidden)?)?;
        assert_eq!(
            actual.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );

        let actual_grads = actual.sqr()?.sum_all()?.backward()?;
        let expected_grads = expected.sqr()?.sum_all()?.backward()?;
        for (name, var) in varmap.data().lock().unwrap().iter() {
            let actual = actual_grads
                .get(var)
                .unwrap_or_else(|| panic!("missing actual gradient for {name}"));
            let expected = expected_grads
                .get(var)
                .unwrap_or_else(|| panic!("missing reference gradient for {name}"));
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?,
                "flag-off gradient drifted for {name}"
            );
        }
        assert_eq!(
            actual_grads
                .get(&input)
                .expect("actual input gradient")
                .flatten_all()?
                .to_vec1::<f32>()?,
            expected_grads
                .get(&input)
                .expect("reference input gradient")
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn bf16_recurrent_core_updates_f32_masters_with_finite_gradients() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let block = GridResidualBlock::new(
            4,
            VarBuilder::from_varmap(&varmap, DType::F32, &device).pp("block"),
        )?;
        reinit_varmap_deterministic(&varmap, 223)?;
        let input = Tensor::arange(0f32, 64f32, &device)?
            .affine(1.0 / 32.0, -1.0)?
            .reshape((1, 4, 4, 4))?;
        let film = ActionFilm::neutral_like(&input)?;
        let output = block.forward(&input, &film, true)?;
        assert_eq!(output.dtype(), DType::F32);
        assert!(output
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite()));
        let grads = output.sqr()?.mean_all()?.backward()?;

        let mut masters = Vec::new();
        for (name, var) in varmap.data().lock().unwrap().iter() {
            assert_eq!(var.dtype(), DType::F32, "{name} is not an F32 master");
            if name == "block.c1.weight" || name == "block.c2.weight" {
                let grad = grads
                    .get(var)
                    .unwrap_or_else(|| panic!("missing gradient for {name}"));
                assert_eq!(grad.dtype(), DType::F32, "{name} gradient is not F32");
                let values = grad.flatten_all()?.to_vec1::<f32>()?;
                assert!(values.iter().all(|value| value.is_finite()));
                assert!(values.iter().any(|value| *value != 0.0));
                masters.push((name.clone(), var.clone(), var.as_tensor().copy()?));
            }
        }
        assert_eq!(masters.len(), 2);
        let mut optimizer = SGD::new(varmap.all_vars(), 1e-3)?;
        optimizer.step(&grads)?;
        for (name, var, before) in masters {
            assert_eq!(var.dtype(), DType::F32, "{name} changed dtype after update");
            let after = var.as_tensor().flatten_all()?.to_vec1::<f32>()?;
            assert!(after.iter().all(|value| value.is_finite()));
            assert_ne!(before.flatten_all()?.to_vec1::<f32>()?, after);
        }
        let updated = block.forward(&input, &film, true)?;
        assert!(updated
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| value.is_finite()));
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
    fn full_v4_cached_canonical_recursion_matches_uncached_path_exactly() -> Result<()> {
        let device = Device::Cpu;
        let cfg = ModelConfig {
            world_core_v4: true,
            spatial_action_field: true,
            consumer_readout: ConsumerReadoutTopology::SpatialQuery,
            residual_y_update: true,
            warm_start_y: true,
            ..tiny_cfg()
        };
        let varmap = VarMap::new();
        let model = WorldModel::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device))?;
        let (frames, actions, coords, _) = sample_batch(&device, 3, model.config.goal_dim)?;
        let current = model.encode_state(&frames)?;
        let current_canonical = model.canonical_representation(&current)?;
        let depth = RecursionDepth::from_config(model.config());
        let opts = RecursionOpts::training(true);
        let uncached = model.training_latents_from_encoded_state(
            &current, &actions, &coords, depth, 0.0, None, opts,
        )?;
        let cached = model.full_v4_training_latents_from_encoded_state(
            &current,
            &current_canonical,
            &actions,
            &coords,
            depth,
            0.0,
            None,
            opts,
        )?;
        assert_eq!(
            uncached.y.flatten_all()?.to_vec1::<f32>()?,
            cached.y.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(uncached.steps.len(), cached.steps.len());
        for (uncached, cached) in uncached.steps.iter().zip(&cached.steps) {
            assert_eq!(
                uncached.flatten_all()?.to_vec1::<f32>()?,
                cached.flatten_all()?.to_vec1::<f32>()?
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
