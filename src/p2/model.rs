//! TRM-inspired pixel world model with PTRM stochastic inference.

use anyhow::{bail, Result};
use candle_core::{DType, Tensor, D};
use candle_nn::{
    conv2d, embedding, linear, Conv2d, Conv2dConfig, Embedding, Linear, Module, VarBuilder,
};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Fixed observation resolution required by the pixel encoder.
pub const FRAME_SIDE: usize = 64;
/// ARC colors are categorical, so frames are supplied as 16 one-hot channels.
pub const PIXEL_CHANNELS: usize = 16;
/// Inclusive action ID range `0..=6`.
pub const ACTION_VOCAB: usize = 7;
/// Six family indicators plus public parameters/order slots.
pub const DEFAULT_GOAL_DIM: usize = 19;
/// Default event head width: noop / goal_satisfied / goal_failed / exhausted.
pub const DEFAULT_NUM_EVENTS: usize = 4;

pub const EVENT_NOOP: usize = 0;
pub const EVENT_GOAL_SATISFIED: usize = 1;
pub const EVENT_GOAL_FAILED: usize = 2;
pub const EVENT_EXHAUSTED: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub frame_side: usize,
    pub hidden_dim: usize,
    pub action_dim: usize,
    pub goal_dim: usize,
    pub inner_steps: usize,
    pub outer_steps: usize,
    pub num_events: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            frame_side: FRAME_SIDE,
            hidden_dim: 128,
            action_dim: 32,
            goal_dim: DEFAULT_GOAL_DIM,
            inner_steps: 4,
            outer_steps: 4,
            num_events: DEFAULT_NUM_EVENTS,
        }
    }
}

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
        Ok(())
    }
}

fn standard_normal(rng: &mut rand::rngs::StdRng) -> f32 {
    let u1 = rng.random_range(f32::EPSILON..1.0f32);
    let u2 = rng.random_range(0.0f32..1.0f32);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn seeded_gaussian_like(template: &Tensor, sigma: f64, seed: u64) -> Result<Tensor> {
    let n = template.elem_count();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(n);
    let scale = sigma as f32;
    for _ in 0..n {
        data.push(standard_normal(&mut rng) * scale);
    }
    Ok(Tensor::from_vec(data, template.shape(), template.device())?.to_dtype(template.dtype())?)
}

/// Shared two-linear residual block reused for z and y refinement.
struct ResidualBlock {
    lin1: Linear,
    lin2: Linear,
}

impl ResidualBlock {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lin1: linear(dim, dim, vb.pp("lin1"))?,
            lin2: linear(dim, dim, vb.pp("lin2"))?,
        })
    }

    fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let delta = self.lin2.forward(&self.lin1.forward(h)?.silu()?)?;
        h.add(&delta).map_err(Into::into)
    }
}

struct PixelEncoder {
    c1: Conv2d,
    c2: Conv2d,
    c3: Conv2d,
    c4: Conv2d,
    proj: Linear,
}

impl PixelEncoder {
    fn new(hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        Ok(Self {
            c1: conv2d(PIXEL_CHANNELS, 16, 3, cfg, vb.pp("c1"))?,
            c2: conv2d(16, 32, 3, cfg, vb.pp("c2"))?,
            c3: conv2d(32, 64, 3, cfg, vb.pp("c3"))?,
            c4: conv2d(64, 64, 3, cfg, vb.pp("c4"))?,
            // 64×64 → 4×4 after four stride-2 convs.
            proj: linear(64 * 4 * 4, hidden_dim, vb.pp("proj"))?,
        })
    }

    fn forward(&self, frames: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = frames.dims4()?;
        if c != PIXEL_CHANNELS || h != FRAME_SIDE || w != FRAME_SIDE {
            bail!(
                "expected frames Bx{PIXEL_CHANNELS}x{FRAME_SIDE}x{FRAME_SIDE}, got {b}x{c}x{h}x{w}"
            );
        }
        let h = self.c1.forward(frames)?.silu()?;
        let h = self.c2.forward(&h)?.silu()?;
        let h = self.c3.forward(&h)?.silu()?;
        let h = self.c4.forward(&h)?.silu()?;
        let (b, ch, hh, ww) = h.dims4()?;
        let flat = h.reshape((b, ch * hh * ww))?;
        Ok(self.proj.forward(&flat)?)
    }
}

#[derive(Debug, Clone)]
pub struct StepOutput {
    pub y: Tensor,
    pub event_logits: Tensor,
    pub q_logit: Tensor,
}

#[derive(Debug, Clone)]
pub struct ForwardOutput {
    /// Trace at each outer recursion step.
    pub steps: Vec<StepOutput>,
    pub y: Tensor,
    pub event_logits: Tensor,
    pub q_logit: Tensor,
}

#[derive(Debug, Clone)]
pub struct PtrmTrajectory {
    pub steps: Vec<StepOutput>,
    pub y: Tensor,
    pub event_logits: Tensor,
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
    encoder: PixelEncoder,
    action_emb: Embedding,
    action_proj: Linear,
    coord_proj: Linear,
    goal_proj: Linear,
    block: ResidualBlock,
    event_head: Linear,
    q_head: Linear,
}

impl WorldModel {
    pub fn new(cfg: ModelConfig, vb: VarBuilder) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            encoder: PixelEncoder::new(cfg.hidden_dim, vb.pp("encoder"))?,
            action_emb: embedding(ACTION_VOCAB, cfg.action_dim, vb.pp("action_emb"))?,
            action_proj: linear(cfg.action_dim, cfg.hidden_dim, vb.pp("action_proj"))?,
            coord_proj: linear(2, cfg.hidden_dim, vb.pp("coord_proj"))?,
            goal_proj: linear(cfg.goal_dim, cfg.hidden_dim, vb.pp("goal_proj"))?,
            block: ResidualBlock::new(cfg.hidden_dim, vb.pp("block"))?,
            event_head: linear(cfg.hidden_dim * 2, cfg.num_events, vb.pp("event_head"))?,
            q_head: linear(cfg.hidden_dim, 1, vb.pp("q_head"))?,
            config: cfg,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Encode a categorical one-hot frame into the shared latent space.
    pub fn encode_state(&self, frames: &Tensor) -> Result<Tensor> {
        self.encoder.forward(frames)
    }

    fn add_action(
        &self,
        state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
    ) -> Result<Tensor> {
        let (b, h) = state.dims2()?;
        if h != self.config.hidden_dim {
            bail!("state latent dim {h} != {}", self.config.hidden_dim);
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
        let coords = self.coord_proj.forward(action_coords)?;
        state.add(&action)?.add(&coords).map_err(Into::into)
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

    fn heads(&self, y: &Tensor, goal_h: &Tensor) -> Result<(Tensor, Tensor)> {
        let event_in = Tensor::cat(&[y, goal_h], D::Minus1)?;
        let event_logits = self.event_head.forward(&event_in)?;
        let q_logit = self.q_head.forward(y)?;
        Ok((event_logits, q_logit))
    }

    fn maybe_noise_z(&self, z: &Tensor, sigma: f64, noise_seed: Option<u64>) -> Result<Tensor> {
        if sigma == 0.0 {
            return Ok(z.clone());
        }
        let eps = match noise_seed {
            Some(seed) => seeded_gaussian_like(z, sigma, seed)?,
            None => z.randn_like(0.0, sigma)?,
        };
        z.add(&eps).map_err(Into::into)
    }

    /// One deep recursion: `inner_steps` z-updates then one y-update.
    /// Noise (when enabled) is injected into `z` before every z refinement.
    fn deep_step(
        &self,
        x: &Tensor,
        y: &Tensor,
        z: &Tensor,
        sigma: f64,
        noise_seed_base: Option<u64>,
        noise_counter: &mut u64,
    ) -> Result<(Tensor, Tensor)> {
        let mut z = z.clone();
        let mut y = y.clone();
        for _ in 0..self.config.inner_steps {
            let step_seed = noise_seed_base.map(|s| s.wrapping_add(*noise_counter));
            *noise_counter = noise_counter.wrapping_add(1);
            z = self.maybe_noise_z(&z, sigma, step_seed)?;
            let inp = x.add(&y)?.add(&z)?;
            z = self.block.forward(&inp)?;
        }
        let inp = y.add(&z)?;
        y = self.block.forward(&inp)?;
        Ok((y, z))
    }

    fn run_recursion(
        &self,
        x: &Tensor,
        goal_h: &Tensor,
        sigma: f64,
        noise_seed_base: Option<u64>,
        outer_steps: usize,
    ) -> Result<ForwardOutput> {
        if outer_steps == 0 {
            bail!("outer_steps override must be >= 1");
        }
        let (b, h) = x.dims2()?;
        let device = x.device();
        let mut y = Tensor::zeros((b, h), x.dtype(), device)?;
        let mut z = Tensor::zeros((b, h), x.dtype(), device)?;
        let mut steps = Vec::with_capacity(outer_steps);
        let mut noise_counter = 0u64;
        for _ in 0..outer_steps {
            let (ny, nz) = self.deep_step(x, &y, &z, sigma, noise_seed_base, &mut noise_counter)?;
            y = ny;
            z = nz;
            let (event_logits, q_logit) = self.heads(&y, goal_h)?;
            steps.push(StepOutput {
                y: y.clone(),
                event_logits,
                q_logit,
            });
        }
        let last = steps
            .last()
            .expect("outer_steps >= 1 guaranteed by validate")
            .clone();
        Ok(ForwardOutput {
            y: last.y.clone(),
            event_logits: last.event_logits.clone(),
            q_logit: last.q_logit.clone(),
            steps,
        })
    }

    /// Deterministic forward (no noise, no learned halting).
    pub fn forward(
        &self,
        frames: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
    ) -> Result<ForwardOutput> {
        let x = self.encode_x(frames, actions, action_coords)?;
        let goal_h = self.project_goal(goal_features)?;
        self.run_recursion(&x, &goal_h, 0.0, None, self.config.outer_steps)
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
        let x = self.encode_x(frames, actions, action_coords)?;
        let goal_h = self.project_goal(goal_features)?;
        self.run_recursion(&x, &goal_h, 0.0, None, outer_steps)
    }

    /// Autoregressive latent transition used by multi-step rollouts.
    pub fn forward_from_latent(
        &self,
        state: &Tensor,
        actions: &Tensor,
        action_coords: &Tensor,
        goal_features: &Tensor,
    ) -> Result<ForwardOutput> {
        let x = self.add_action(state, actions, action_coords)?;
        let goal_h = self.project_goal(goal_features)?;
        self.run_recursion(&x, &goal_h, 0.0, None, self.config.outer_steps)
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
        if ptrm.k == 0 {
            bail!("PTRM requires K >= 1");
        }
        if !ptrm.sigma.is_finite() || ptrm.sigma < 0.0 {
            bail!("PTRM sigma must be finite and non-negative");
        }
        let x = self.encode_x(frames, actions, action_coords)?;
        let goal_h = self.project_goal(goal_features)?;
        let mut trajectories = Vec::with_capacity(ptrm.k);
        let mut q_logits = Vec::with_capacity(ptrm.k);
        for traj in 0..ptrm.k {
            let noise_base = ptrm
                .seed
                .map(|s| s ^ (traj as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let out =
                self.run_recursion(&x, &goal_h, ptrm.sigma, noise_base, self.config.outer_steps)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
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
        }
    }

    fn make_model(device: &Device) -> Result<(WorldModel, VarMap)> {
        let cfg = tiny_cfg();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let model = WorldModel::new(cfg, vb)?;
        Ok((model, varmap))
    }

    fn sample_batch(
        device: &Device,
        batch: usize,
        goal_dim: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let frames = Tensor::randn(
            0f32,
            1.0,
            (batch, PIXEL_CHANNELS, FRAME_SIDE, FRAME_SIDE),
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
    fn forward_output_shapes() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = make_model(&device)?;
        let (frames, actions, coords, goals) = sample_batch(&device, 3, model.config.goal_dim)?;
        let out = model.forward(&frames, &actions, &coords, &goals)?;
        assert_eq!(out.steps.len(), model.config.outer_steps);
        assert_eq!(out.y.dims(), &[3, model.config.hidden_dim]);
        assert_eq!(out.event_logits.dims(), &[3, model.config.num_events]);
        assert_eq!(out.q_logit.dims(), &[3, 1]);
        for step in &out.steps {
            assert_eq!(step.y.dims(), &[3, model.config.hidden_dim]);
            assert_eq!(step.event_logits.dims(), &[3, model.config.num_events]);
            assert_eq!(step.q_logit.dims(), &[3, 1]);
        }
        Ok(())
    }

    #[test]
    fn ptrm_k1_sigma0_matches_deterministic() -> Result<()> {
        let device = Device::Cpu;
        let (model, _) = make_model(&device)?;
        let (frames, actions, coords, goals) = sample_batch(&device, 2, model.config.goal_dim)?;
        let det = model.forward(&frames, &actions, &coords, &goals)?;
        let ptrm = model.forward_ptrm(
            &frames,
            &actions,
            &coords,
            &goals,
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
}
