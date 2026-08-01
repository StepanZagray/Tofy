//! P2 LeWorld / TRM training on synthetic curriculum only.

use crate::domain::Split;
use crate::p2::data::{
    generate_curriculum, ArcFrame, TransitionSample, FRAME_SIDE, GOAL_FEATURES_DIM,
};
use crate::p2::model::{ModelConfig, WorldModel, ACTION_VOCAB, DEFAULT_NUM_EVENTS, PIXEL_CHANNELS};
use crate::p2::sigreg::sigreg_epps_pulley_seeded;
use anyhow::{bail, Context, Result};
use candle_core::{backprop::GradStore, DType, Device, Tensor, D};
use candle_nn::init::FanInOut;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::{ops, VarBuilder, VarMap};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Default sequential lesson order. Never pulls ARC public recordings.
pub const DEFAULT_LESSONS: &[&str] = &["dynamics", "sequential", "falsification", "retarget"];

pub const TRAIN_REPORT_SCHEMA: &str = "p2.train_report.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub seed: u64,
    pub lessons: Vec<String>,
    pub steps_per_lesson: usize,
    pub physical_batch: usize,
    /// Must be recorded. Initial implementation requires `1` (no fake accumulation).
    pub grad_accum: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub sigreg_projections: usize,
    pub sigreg_knots: usize,
    pub sigreg_weight: f64,
    pub event_weight: f64,
    pub q_weight: f64,
    /// Weight for open-loop latent error on sequential/retarget lessons.
    pub rollout_weight: f64,
    /// Frozen MSE threshold for Q-correctness targets.
    pub q_mse_threshold: f64,
    pub hidden_dim: usize,
    pub action_dim: usize,
    pub inner_steps: usize,
    pub outer_steps: usize,
    /// `"cpu"` or `"cuda"` / `"cuda:N"`.
    pub device: String,
    pub output_dir: PathBuf,
    /// Optional safetensors path to resume weights (optimizer state is not restored).
    pub resume: Option<PathBuf>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            seed: 1,
            lessons: DEFAULT_LESSONS.iter().map(|s| (*s).to_string()).collect(),
            steps_per_lesson: 2,
            physical_batch: 2,
            grad_accum: 1,
            lr: 1e-3,
            weight_decay: 0.01,
            sigreg_projections: 8,
            sigreg_knots: 5,
            sigreg_weight: 0.01,
            event_weight: 0.1,
            q_weight: 0.1,
            rollout_weight: 0.1,
            q_mse_threshold: 0.5,
            hidden_dim: 32,
            action_dim: 8,
            inner_steps: 1,
            outer_steps: 1,
            device: "cpu".into(),
            output_dir: PathBuf::from("runs/p2/smoke"),
            resume: None,
        }
    }
}

impl TrainConfig {
    pub fn validate(&self) -> Result<()> {
        if self.steps_per_lesson == 0 {
            bail!("steps_per_lesson must be > 0");
        }
        if self.physical_batch < 2 {
            bail!("physical_batch must be >= 2 (SIGReg needs batch >= 2)");
        }
        if self.grad_accum != 1 {
            bail!(
                "grad_accum must be 1 in the initial P2 trainer (got {}); \
                 accumulation is recorded but not implemented",
                self.grad_accum
            );
        }
        if self.lessons.is_empty() {
            bail!("at least one lesson is required");
        }
        for lesson in &self.lessons {
            lesson_to_curriculum(lesson)?;
        }
        if !(self.lr.is_finite() && self.lr > 0.0) {
            bail!("lr must be finite and > 0");
        }
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            bail!("weight_decay must be finite and >= 0");
        }
        for (name, weight) in [
            ("sigreg_weight", self.sigreg_weight),
            ("event_weight", self.event_weight),
            ("q_weight", self.q_weight),
            ("rollout_weight", self.rollout_weight),
        ] {
            if !(weight.is_finite() && weight >= 0.0) {
                bail!("{name} must be finite and >= 0");
            }
        }
        if self.sigreg_projections == 0 || self.sigreg_knots < 3 {
            bail!("sigreg_projections >= 1 and sigreg_knots >= 3 required");
        }
        if !(self.q_mse_threshold.is_finite() && self.q_mse_threshold >= 0.0) {
            bail!("q_mse_threshold must be finite and >= 0");
        }
        Ok(())
    }

    pub fn model_config(&self) -> ModelConfig {
        ModelConfig {
            frame_side: FRAME_SIDE,
            hidden_dim: self.hidden_dim,
            action_dim: self.action_dim,
            goal_dim: GOAL_FEATURES_DIM,
            inner_steps: self.inner_steps,
            outer_steps: self.outer_steps,
            num_events: DEFAULT_NUM_EVENTS,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LessonLossMeans {
    pub total: f64,
    pub next_latent: f64,
    pub rollout: f64,
    pub sigreg: f64,
    pub event: f64,
    pub q: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LessonReport {
    pub lesson: String,
    pub curriculum: String,
    pub steps: usize,
    pub mean_losses: LessonLossMeans,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainReport {
    pub schema: String,
    pub seed: u64,
    pub physical_batch: usize,
    pub grad_accum: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub parameter_count: usize,
    pub device: String,
    pub lessons: Vec<LessonReport>,
    pub checkpoint: PathBuf,
    pub config_path: PathBuf,
    /// Analyzer-compatible `candle-graph/runtime/1` trace from the first update.
    pub runtime_trace: PathBuf,
    /// Smoke / scaffolding only; not a research result.
    pub research_claim: bool,
}

/// Map a lesson name to a `generate_curriculum` kind. Never ARC recordings.
pub fn lesson_to_curriculum(lesson: &str) -> Result<&'static str> {
    match lesson {
        "dynamics" => Ok("random_one_step"),
        "sequential" => Ok("sequential"),
        "falsification" => Ok("p1c_falsification"),
        "retarget" => Ok("p1c_hard_retarget"),
        other => bail!("unknown lesson {other}"),
    }
}

pub fn resolve_device(spec: &str) -> Result<Device> {
    let spec = spec.trim();
    if spec.eq_ignore_ascii_case("cpu") {
        return Ok(Device::Cpu);
    }
    if spec.eq_ignore_ascii_case("cuda") {
        return Device::new_cuda(0).context("open cuda:0");
    }
    if let Some(rest) = spec
        .strip_prefix("cuda:")
        .or_else(|| spec.strip_prefix("CUDA:"))
    {
        let ordinal: usize = rest.parse().context("parse cuda ordinal")?;
        return Device::new_cuda(ordinal).with_context(|| format!("open cuda:{ordinal}"));
    }
    bail!("unsupported device {spec:?}; use cpu, cuda, or cuda:N");
}

fn stable_name_seed(master: u64, name: &str) -> u64 {
    let mut h = master ^ 0x9E37_79B9_7F4A_7C15;
    for &b in name.as_bytes() {
        h = h
            .wrapping_mul(0x0000_0100_0000_01B3)
            .wrapping_add(u64::from(b));
    }
    h
}

fn xavier_uniform_vec(shape: &[usize], seed: u64) -> Vec<f32> {
    let shape_obj = candle_core::Shape::from(shape.to_vec());
    let fan_in = FanInOut::FanIn.for_shape(&shape_obj).max(1);
    let fan_out = FanInOut::FanOut.for_shape(&shape_obj).max(1);
    let bound = (6.0f64 / (fan_in + fan_out) as f64).sqrt() as f32;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = shape.iter().product::<usize>();
    (0..n).map(|_| rng.random_range(-bound..=bound)).collect()
}

/// Deterministic reinitialization: zero biases, Xavier-like weights from
/// `hash(name) ⊕ master_seed`. Works on CPU where `Device::set_seed` is unsupported.
pub fn reinit_varmap_deterministic(varmap: &VarMap, master_seed: u64) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    let mut names: Vec<String> = data.keys().cloned().collect();
    names.sort();
    for name in names {
        let var = data
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("missing var {name}"))?;
        let shape = var.shape().dims().to_vec();
        let n = var.elem_count();
        let seed = stable_name_seed(master_seed, &name);
        let is_bias = name.rsplit('.').next() == Some("bias") || name.ends_with("bias");
        let values = if is_bias {
            vec![0f32; n]
        } else {
            xavier_uniform_vec(&shape, seed)
        };
        let t = Tensor::from_vec(values, shape.as_slice(), var.device())?.to_dtype(var.dtype())?;
        var.set(&t)?;
    }
    Ok(())
}

pub fn parameter_count(varmap: &VarMap) -> usize {
    varmap.all_vars().iter().map(|v| v.elem_count()).sum()
}

/// Convert palette frames to categorical `B×16×64×64` one-hot tensors.
pub fn frames_to_one_hot(frames: &[ArcFrame], device: &Device) -> Result<Tensor> {
    let b = frames.len();
    if b == 0 {
        bail!("frames_to_one_hot requires at least one frame");
    }
    let mut data = vec![0f32; b * PIXEL_CHANNELS * FRAME_SIDE * FRAME_SIDE];
    for (bi, frame) in frames.iter().enumerate() {
        ensure_fixed_frame(frame)?;
        let base = bi * PIXEL_CHANNELS * FRAME_SIDE * FRAME_SIDE;
        for (i, &pix) in frame.pixels.iter().enumerate() {
            if pix as usize >= PIXEL_CHANNELS {
                bail!("palette value {pix} out of 0..{PIXEL_CHANNELS}");
            }
            let y = i / FRAME_SIDE;
            let x = i % FRAME_SIDE;
            let idx = base + (pix as usize) * FRAME_SIDE * FRAME_SIDE + y * FRAME_SIDE + x;
            data[idx] = 1.0;
        }
    }
    Tensor::from_vec(data, (b, PIXEL_CHANNELS, FRAME_SIDE, FRAME_SIDE), device).map_err(Into::into)
}

fn ensure_fixed_frame(frame: &ArcFrame) -> Result<()> {
    if frame.width as usize != FRAME_SIDE || frame.height as usize != FRAME_SIDE {
        bail!(
            "expected {FRAME_SIDE}x{FRAME_SIDE} frame, got {}x{}",
            frame.width,
            frame.height
        );
    }
    if frame.pixels.len() != FRAME_SIDE * FRAME_SIDE {
        bail!("frame pixel length mismatch");
    }
    Ok(())
}

/// Event targets (`B×4`) and mask (`B×4`) from `Option<bool>` labels.
pub fn event_targets_and_mask(
    samples: &[TransitionSample],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let b = samples.len();
    let mut targets = vec![0f32; b * DEFAULT_NUM_EVENTS];
    let mut mask = vec![0f32; b * DEFAULT_NUM_EVENTS];
    for (i, s) in samples.iter().enumerate() {
        let row = i * DEFAULT_NUM_EVENTS;
        for (j, opt) in [s.noop, s.goal_satisfied, s.goal_failed, s.exhausted]
            .into_iter()
            .enumerate()
        {
            if let Some(v) = opt {
                targets[row + j] = if v { 1.0 } else { 0.0 };
                mask[row + j] = 1.0;
            }
        }
    }
    let targets = Tensor::from_vec(targets, (b, DEFAULT_NUM_EVENTS), device)?;
    let mask = Tensor::from_vec(mask, (b, DEFAULT_NUM_EVENTS), device)?;
    Ok((targets, mask))
}

pub struct BatchTensors {
    pub frames: Tensor,
    pub next_frames: Tensor,
    pub actions: Tensor,
    /// Normalized `(x,y)` for ACTION6, zeros for simple actions.
    pub action_coords: Tensor,
    pub goals: Tensor,
    pub event_targets: Tensor,
    pub event_mask: Tensor,
}

pub fn batch_from_samples(samples: &[TransitionSample], device: &Device) -> Result<BatchTensors> {
    if samples.is_empty() {
        bail!("empty batch");
    }
    let currents: Vec<ArcFrame> = samples.iter().map(|s| s.current.clone()).collect();
    let nexts: Vec<ArcFrame> = samples.iter().map(|s| s.next.clone()).collect();
    let frames = frames_to_one_hot(&currents, device)?;
    let next_frames = frames_to_one_hot(&nexts, device)?;
    let actions: Vec<u32> = samples
        .iter()
        .map(|s| {
            let id = s.action.id as u32;
            if id as usize >= ACTION_VOCAB {
                bail!("action id {id} out of ACTION_VOCAB={ACTION_VOCAB}");
            }
            Ok(id)
        })
        .collect::<Result<Vec<_>>>()?;
    let actions = Tensor::from_vec(actions, (samples.len(),), device)?;
    let mut coords = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        match (sample.action.x, sample.action.y) {
            (Some(x), Some(y)) => {
                coords.push(f32::from(x) / 63.0);
                coords.push(f32::from(y) / 63.0);
            }
            (None, None) => {
                coords.push(0.0);
                coords.push(0.0);
            }
            _ => bail!("action coordinate pair is incomplete"),
        }
    }
    let action_coords = Tensor::from_vec(coords, (samples.len(), 2), device)?;
    let mut goals = Vec::with_capacity(samples.len() * GOAL_FEATURES_DIM);
    for s in samples {
        goals.extend_from_slice(&s.goal_features.values);
    }
    let goals = Tensor::from_vec(goals, (samples.len(), GOAL_FEATURES_DIM), device)?;
    let (event_targets, event_mask) = event_targets_and_mask(samples, device)?;
    Ok(BatchTensors {
        frames,
        next_frames,
        actions,
        action_coords,
        goals,
        event_targets,
        event_mask,
    })
}

fn collect_batch(
    curriculum: &str,
    seed: u64,
    start_episode: u64,
    batch: usize,
    split: Split,
) -> Result<Vec<TransitionSample>> {
    let mut out = Vec::with_capacity(batch);
    let mut ep = start_episode;
    let limit = start_episode.saturating_add(50_000);
    while out.len() < batch {
        if ep > limit {
            bail!("failed to collect batch={batch} from curriculum {curriculum}");
        }
        let samples = generate_curriculum(curriculum, seed, ep, split)?;
        for s in samples {
            out.push(s);
            if out.len() == batch {
                break;
            }
        }
        ep = ep.wrapping_add(1);
    }
    Ok(out)
}

fn masked_bce_with_logits(logits: &Tensor, targets: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let mask_sum = mask.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    if mask_sum <= 0.0 {
        return Tensor::zeros((), DType::F32, logits.device()).map_err(Into::into);
    }
    let p = ops::sigmoid(logits)?;
    let ones = Tensor::ones_like(targets)?;
    let eps = 1e-7f64;
    let left = (targets * p.affine(1.0, eps)?.log()?)?;
    let right = ((ones.sub(targets)?) * p.affine(-1.0, 1.0 + eps)?.log()?)?;
    let elem = (left + right)?.neg()?;
    let weighted = (elem * mask)?;
    weighted
        .sum_all()?
        .affine(1.0 / mask_sum as f64, 0.0)
        .map_err(Into::into)
}

fn ensure_finite(name: &str, t: &Tensor) -> Result<f32> {
    let v = t.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    if !v.is_finite() {
        bail!("{name} is not finite: {v}");
    }
    Ok(v)
}

#[derive(Debug, Clone)]
pub struct LossBreakdown {
    pub total: Tensor,
    pub next_latent: Tensor,
    pub sigreg: Tensor,
    pub event: Tensor,
    pub q: Tensor,
}

/// LeWorld loss: deep-supervised next-latent MSE + seeded SIGReg on stacked
/// current/next encoder embeddings (no detach/EMA) + masked event BCE + Q BCE.
pub fn leworld_loss(
    model: &WorldModel,
    batch: &BatchTensors,
    cfg: &TrainConfig,
    sigreg_seed: u64,
) -> Result<LossBreakdown> {
    let out = model.forward(
        &batch.frames,
        &batch.actions,
        &batch.action_coords,
        &batch.goals,
    )?;
    let cur_z = model.encode_state(&batch.frames)?;
    let next_z = model.encode_state(&batch.next_frames)?;

    let mut pred_acc: Option<Tensor> = None;
    for step in &out.steps {
        let mse = step.y.sub(&next_z)?.sqr()?.mean_all()?;
        pred_acc = Some(match pred_acc {
            None => mse,
            Some(acc) => acc.add(&mse)?,
        });
    }
    let n_steps = out.steps.len().max(1) as f64;
    let next_latent = pred_acc
        .ok_or_else(|| anyhow::anyhow!("no outer steps"))?
        .affine(1.0 / n_steps, 0.0)?;

    let stack = Tensor::stack(&[cur_z.clone(), next_z.clone()], 0)?;
    let sigreg = sigreg_epps_pulley_seeded(
        &stack,
        cfg.sigreg_projections,
        cfg.sigreg_knots,
        sigreg_seed,
    )?;

    let event = masked_bce_with_logits(&out.event_logits, &batch.event_targets, &batch.event_mask)?;

    let per = out.y.sub(&next_z)?.sqr()?.mean_keepdim(D::Minus1)?.detach();
    let q_targets = per.lt(cfg.q_mse_threshold)?.to_dtype(DType::F32)?;
    let q = candle_nn::loss::binary_cross_entropy_with_logit(&out.q_logit, &q_targets)?;

    let total = next_latent
        .add(&sigreg.affine(cfg.sigreg_weight, 0.0)?)?
        .add(&event.affine(cfg.event_weight, 0.0)?)?
        .add(&q.affine(cfg.q_weight, 0.0)?)?;

    ensure_finite("next_latent", &next_latent)?;
    ensure_finite("sigreg", &sigreg)?;
    ensure_finite("event", &event)?;
    ensure_finite("q", &q)?;
    ensure_finite("total", &total)?;

    Ok(LossBreakdown {
        total,
        next_latent,
        sigreg,
        event,
        q,
    })
}

fn collect_rollout_trace(
    curriculum: &str,
    seed: u64,
    start_episode: u64,
    split: Split,
) -> Result<Vec<TransitionSample>> {
    for offset in 0..1024u64 {
        let samples =
            generate_curriculum(curriculum, seed, start_episode.wrapping_add(offset), split)?;
        if samples.len() >= 2 {
            return Ok(samples);
        }
    }
    bail!("failed to find a multi-step trace for curriculum {curriculum}")
}

/// Open-loop latent rollout from the first real frame; later inputs are predicted
/// latents, never teacher-forced frames. Candidate changes affect only event heads.
pub fn open_loop_latent_loss(
    model: &WorldModel,
    samples: &[TransitionSample],
    device: &Device,
    horizon: usize,
) -> Result<Tensor> {
    if samples.len() < 2 || horizon < 2 {
        bail!("open-loop loss requires at least two ordered transitions");
    }
    let first = batch_from_samples(&[samples[0].clone()], device)?;
    let mut latent = model.encode_state(&first.frames)?;
    let mut total: Option<Tensor> = None;
    let mut steps = 0usize;
    for sample in samples.iter().take(horizon) {
        let batch = batch_from_samples(std::slice::from_ref(sample), device)?;
        let predicted = model.forward_from_latent(
            &latent,
            &batch.actions,
            &batch.action_coords,
            &batch.goals,
        )?;
        latent = predicted.y;
        let target = model.encode_state(&batch.next_frames)?;
        // Robustify early open-loop training, where an untrained recursive model
        // can otherwise produce a single enormous horizon loss.
        let mse = candle_nn::loss::huber(&latent, &target, 1.0)?;
        total = Some(match total {
            None => mse,
            Some(acc) => acc.add(&mse)?,
        });
        steps += 1;
    }
    total
        .ok_or_else(|| anyhow::anyhow!("open-loop trace was empty"))?
        .affine(1.0 / steps as f64, 0.0)
        .map_err(Into::into)
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
    }
    let json = serde_json::to_string_pretty(value).context("serialize json")?;
    let tmp = {
        let mut os = path.as_os_str().to_owned();
        os.push(".tmp");
        PathBuf::from(os)
    };
    fs::write(&tmp, &json).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| {
        format!(
            "rename {} -> {} (atomic replace)",
            tmp.display(),
            path.display()
        )
    })?;
    Ok(())
}

pub fn save_checkpoint(varmap: &VarMap, cfg: &TrainConfig, report: &TrainReport) -> Result<()> {
    fs::create_dir_all(&cfg.output_dir)
        .with_context(|| format!("create {}", cfg.output_dir.display()))?;
    let weights = cfg.output_dir.join("model.safetensors");
    let weights_tmp = cfg.output_dir.join("model.safetensors.tmp");
    varmap
        .save(&weights_tmp)
        .with_context(|| format!("save {}", weights_tmp.display()))?;
    fs::rename(&weights_tmp, &weights)
        .with_context(|| format!("rename {} -> {}", weights_tmp.display(), weights.display()))?;
    write_json_atomic(&cfg.output_dir.join("config.json"), cfg)?;
    write_json_atomic(&cfg.output_dir.join("train_report.json"), report)?;
    Ok(())
}

pub fn load_weights(varmap: &mut VarMap, path: &Path) -> Result<()> {
    varmap
        .load(path)
        .with_context(|| format!("load weights {}", path.display()))
}

pub fn load_train_config(path: &Path) -> Result<TrainConfig> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).context("parse TrainConfig")
}

fn active_cargo_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    if cfg!(feature = "cuda") {
        features.push("cuda");
    }
    if cfg!(feature = "cudnn") {
        features.push("cudnn");
    }
    if cfg!(feature = "metal") {
        features.push("metal");
    }
    features
}

fn runtime_trace(
    varmap: &VarMap,
    grads: &GradStore,
    total_loss: &Tensor,
    cfg: &TrainConfig,
) -> Result<serde_json::Value> {
    let data = varmap.data().lock().unwrap();
    let mut names: Vec<_> = data.keys().cloned().collect();
    names.sort();
    let mut gradient_facts = Vec::with_capacity(names.len());
    for (event_index, name) in names.into_iter().enumerate() {
        let var = data
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("missing var {name}"))?;
        let (state, norm) = if let Some(grad) = grads.get(var.as_tensor()) {
            let norm = grad
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_all()?
                .sqrt()?
                .to_scalar::<f32>()? as f64;
            if !norm.is_finite() {
                ("non_finite", None)
            } else if norm == 0.0 {
                ("zero", None)
            } else {
                ("present", Some(norm))
            }
        } else {
            ("missing", None)
        };
        gradient_facts.push(serde_json::json!({
            "event_id": format!("first-step-gradient-{event_index}"),
            "root": "vb",
            "key": name,
            "state": state,
            "norm": norm,
        }));
    }
    Ok(serde_json::json!({
        "schema": "candle-graph/runtime/1",
        "run": {
            "entrypoint": "p2::train::leworld_loss",
            "profile": cfg.device,
            "cargo_features": active_cargo_features(),
            "cfg": [],
        },
        "tensors": [{
            "event_id": "first-step-total-loss",
            "source": "p2::train::leworld_loss total",
            "shape": total_loss.dims(),
            "dtype": format!("{:?}", total_loss.dtype()),
            "device": cfg.device,
            "contiguous": total_loss.is_contiguous(),
            "requires_grad": true,
        }],
        "operations": [],
        "gradients": gradient_facts,
    }))
}

/// Train lessons in order without resetting optimizer or model.
pub fn train(cfg: &TrainConfig) -> Result<TrainReport> {
    cfg.validate()?;
    let device = resolve_device(&cfg.device)?;
    let model_cfg = cfg.model_config();
    model_cfg.validate()?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = WorldModel::new(model_cfg, vb)?;
    if let Some(path) = &cfg.resume {
        load_weights(&mut varmap, path)?;
    } else {
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
    }
    let param_count = parameter_count(&varmap);

    let adam = ParamsAdamW {
        lr: cfg.lr,
        weight_decay: cfg.weight_decay,
        ..ParamsAdamW::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), adam)?;

    let mut lesson_reports = Vec::with_capacity(cfg.lessons.len());
    let mut global_step = 0u64;
    let mut first_step_trace = None;

    for lesson in &cfg.lessons {
        let curriculum = lesson_to_curriculum(lesson)?;
        let mut sums = LessonLossMeans {
            total: 0.0,
            next_latent: 0.0,
            rollout: 0.0,
            sigreg: 0.0,
            event: 0.0,
            q: 0.0,
        };
        for step in 0..cfg.steps_per_lesson {
            let samples = collect_batch(
                curriculum,
                cfg.seed,
                global_step,
                cfg.physical_batch,
                Split::Train,
            )?;
            let batch = batch_from_samples(&samples, &device)?;
            let sigreg_seed = cfg.seed.wrapping_add(global_step);
            let losses = leworld_loss(&model, &batch, cfg, sigreg_seed)?;
            let rollout = if matches!(lesson.as_str(), "sequential" | "retarget") {
                let trace = collect_rollout_trace(curriculum, cfg.seed, global_step, Split::Train)?;
                open_loop_latent_loss(&model, &trace, &device, 4)?
            } else {
                Tensor::zeros((), DType::F32, &device)?
            };
            let total = losses
                .total
                .add(&rollout.affine(cfg.rollout_weight, 0.0)?)?;
            ensure_finite("rollout", &rollout)?;
            ensure_finite("total_with_rollout", &total)?;
            let grads = total.backward()?;
            if first_step_trace.is_none() {
                first_step_trace = Some(runtime_trace(&varmap, &grads, &total, cfg)?);
            }
            opt.step(&grads)?;

            sums.total += ensure_finite("total", &total)? as f64;
            sums.next_latent += ensure_finite("next_latent", &losses.next_latent)? as f64;
            sums.rollout += ensure_finite("rollout", &rollout)? as f64;
            sums.sigreg += ensure_finite("sigreg", &losses.sigreg)? as f64;
            sums.event += ensure_finite("event", &losses.event)? as f64;
            sums.q += ensure_finite("q", &losses.q)? as f64;
            global_step = global_step.wrapping_add(1);
            let _ = step;
        }
        let n = cfg.steps_per_lesson as f64;
        lesson_reports.push(LessonReport {
            lesson: lesson.clone(),
            curriculum: curriculum.to_string(),
            steps: cfg.steps_per_lesson,
            mean_losses: LessonLossMeans {
                total: sums.total / n,
                next_latent: sums.next_latent / n,
                rollout: sums.rollout / n,
                sigreg: sums.sigreg / n,
                event: sums.event / n,
                q: sums.q / n,
            },
        });
    }

    let report = TrainReport {
        schema: TRAIN_REPORT_SCHEMA.into(),
        seed: cfg.seed,
        physical_batch: cfg.physical_batch,
        grad_accum: cfg.grad_accum,
        lr: cfg.lr,
        weight_decay: cfg.weight_decay,
        parameter_count: param_count,
        device: cfg.device.clone(),
        lessons: lesson_reports,
        checkpoint: cfg.output_dir.join("model.safetensors"),
        config_path: cfg.output_dir.join("config.json"),
        runtime_trace: cfg.output_dir.join("runtime.json"),
        research_claim: false,
    };
    save_checkpoint(&varmap, cfg, &report)?;
    write_json_atomic(
        &report.runtime_trace,
        &first_step_trace.ok_or_else(|| anyhow::anyhow!("training produced no runtime trace"))?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Split;
    use crate::p2::data::{ArcAction, GoalFeatures};

    fn toy_frame(fill: u8) -> ArcFrame {
        ArcFrame::new(
            FRAME_SIDE as u16,
            FRAME_SIDE as u16,
            vec![fill; FRAME_SIDE * FRAME_SIDE],
        )
        .unwrap()
    }

    fn toy_sample(pix: u8) -> TransitionSample {
        TransitionSample {
            current: toy_frame(pix),
            next: toy_frame((pix + 1) % 16),
            action: ArcAction::new(1, None, None).unwrap(),
            goal_features: GoalFeatures::zeros(),
            noop: Some(false),
            goal_satisfied: Some(false),
            goal_failed: None,
            exhausted: Some(false),
            split: Split::Train,
            family: "test".into(),
            seed: 0,
            episode_id: 0,
        }
    }

    #[test]
    fn one_hot_conversion_and_event_mask() -> Result<()> {
        let device = Device::Cpu;
        let mut coordinate_sample = toy_sample(7);
        coordinate_sample.action = ArcAction::new(6, Some(63), Some(21))?;
        let samples = vec![toy_sample(3), coordinate_sample];
        let batch = batch_from_samples(&samples, &device)?;
        assert_eq!(batch.frames.dims(), &[2, 16, 64, 64]);
        let f0 = batch.frames.get(0)?;
        // Channel 3 should be all ones for first frame.
        let ch3 = f0.get(3)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(ch3.iter().all(|v| *v == 1.0));
        let ch0 = f0.get(0)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(ch0.iter().all(|v| *v == 0.0));

        let targets = batch.event_targets.to_vec2::<f32>()?;
        let mask = batch.event_mask.to_vec2::<f32>()?;
        // goal_failed is None → mask 0
        assert_eq!(mask[0][2], 0.0);
        assert_eq!(mask[0][0], 1.0);
        assert_eq!(targets[0][0], 0.0);
        assert_eq!(
            batch.action_coords.to_vec2::<f32>()?,
            vec![vec![0.0, 0.0], vec![1.0, 21.0 / 63.0],]
        );
        Ok(())
    }

    #[test]
    fn deterministic_init_repeats() -> Result<()> {
        let device = Device::Cpu;
        let cfg = ModelConfig {
            frame_side: FRAME_SIDE,
            hidden_dim: 16,
            action_dim: 4,
            goal_dim: GOAL_FEATURES_DIM,
            inner_steps: 1,
            outer_steps: 1,
            num_events: DEFAULT_NUM_EVENTS,
        };
        let mut maps = Vec::new();
        for _ in 0..2 {
            let map = VarMap::new();
            let vb = VarBuilder::from_varmap(&map, DType::F32, &device);
            let _model = WorldModel::new(cfg.clone(), vb)?;
            reinit_varmap_deterministic(&map, 42)?;
            maps.push(map);
        }
        let a = maps[0].data().lock().unwrap();
        let b = maps[1].data().lock().unwrap();
        let mut names: Vec<_> = a.keys().cloned().collect();
        names.sort();
        for name in names {
            let va = a
                .get(&name)
                .unwrap()
                .as_tensor()
                .flatten_all()?
                .to_vec1::<f32>()?;
            let vb = b
                .get(&name)
                .unwrap()
                .as_tensor()
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(va, vb, "mismatch at {name}");
            if name.ends_with("bias") {
                assert!(va.iter().all(|v| *v == 0.0), "bias not zero: {name}");
            }
        }
        Ok(())
    }

    #[test]
    fn one_optimizer_step_changes_finite_loss() -> Result<()> {
        let cfg = TrainConfig {
            steps_per_lesson: 1,
            lessons: vec!["dynamics".into()],
            physical_batch: 2,
            grad_accum: 1,
            output_dir: std::env::temp_dir().join(format!("tofy-p2-train-{}", std::process::id())),
            ..TrainConfig::default()
        };
        let _ = fs::remove_dir_all(&cfg.output_dir);

        let device = resolve_device(&cfg.device)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg.model_config(), vb)?;
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
        let mut opt = AdamW::new_lr(varmap.all_vars(), cfg.lr)?;

        let samples = collect_batch(
            "random_one_step",
            cfg.seed,
            0,
            cfg.physical_batch,
            Split::Train,
        )?;
        let batch = batch_from_samples(&samples, &device)?;
        let before = leworld_loss(&model, &batch, &cfg, cfg.seed)?;
        let v0 = ensure_finite("before", &before.total)?;
        opt.backward_step(&before.total)?;
        let after = leworld_loss(&model, &batch, &cfg, cfg.seed)?;
        let v1 = ensure_finite("after", &after.total)?;
        assert!(v0.is_finite() && v1.is_finite());
        assert!(
            v1 < v0 || (v1 - v0).abs() > 1e-8,
            "expected loss to decrease or change, got {v0} -> {v1}"
        );
        let _ = fs::remove_dir_all(&cfg.output_dir);
        Ok(())
    }

    #[test]
    fn open_loop_loss_is_finite_and_backpropagates() -> Result<()> {
        let cfg = TrainConfig {
            hidden_dim: 16,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            ..TrainConfig::default()
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(cfg.model_config(), vb)?;
        reinit_varmap_deterministic(&varmap, cfg.seed)?;
        let trace = collect_rollout_trace("sequential", cfg.seed, 0, Split::Train)?;
        let loss = open_loop_latent_loss(&model, &trace, &device, 4)?;
        assert!(ensure_finite("open_loop", &loss)?.is_finite());
        let grads = loss.backward()?;
        assert!(varmap.all_vars().iter().any(|var| grads.get(var).is_some()));
        Ok(())
    }

    #[test]
    fn report_serialization_roundtrip() -> Result<()> {
        let report = TrainReport {
            schema: TRAIN_REPORT_SCHEMA.into(),
            seed: 1,
            physical_batch: 2,
            grad_accum: 1,
            lr: 1e-3,
            weight_decay: 0.01,
            parameter_count: 10,
            device: "cpu".into(),
            lessons: vec![],
            checkpoint: PathBuf::from("m.safetensors"),
            config_path: PathBuf::from("c.json"),
            runtime_trace: PathBuf::from("runtime.json"),
            research_claim: false,
        };
        let s = serde_json::to_string(&report)?;
        let back: TrainReport = serde_json::from_str(&s)?;
        assert_eq!(back.schema, TRAIN_REPORT_SCHEMA);
        assert_eq!(back.grad_accum, 1);
        assert!(!back.research_claim);
        Ok(())
    }
}
