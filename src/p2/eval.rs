//! P2 world-model evaluation (synthetic held-out + ARC recording transfer).

use crate::domain::Split;
use crate::p2::arc3::import_recordings_dir;
use crate::p2::data::{generate_curriculum, TransitionSample};
use crate::p2::model::{PtrmConfig, WorldModel, EVENT_GOAL_FAILED};
use crate::p2::train::{
    batch_from_samples, load_train_config, resolve_device, BatchTensors, TrainConfig,
};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{ops, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const EVAL_REPORT_SCHEMA: &str = "p2.eval_report.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    pub checkpoint: PathBuf,
    pub train_config: PathBuf,
    pub seed: u64,
    pub synthetic_episodes: usize,
    pub physical_batch: usize,
    pub ptrm_k: Vec<usize>,
    pub ptrm_noise: f64,
    pub q_mse_threshold: f64,
    pub device: String,
    pub arc_recordings_dir: Option<PathBuf>,
    pub output: PathBuf,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            checkpoint: PathBuf::from("runs/p2/smoke/model.safetensors"),
            train_config: PathBuf::from("runs/p2/smoke/config.json"),
            seed: 2,
            synthetic_episodes: 4,
            physical_batch: 2,
            ptrm_k: vec![1, 2, 4],
            ptrm_noise: 0.1,
            q_mse_threshold: 0.5,
            device: "cpu".into(),
            arc_recordings_dir: None,
            output: PathBuf::from("runs/p2/smoke/eval_report.json"),
        }
    }
}

impl EvalConfig {
    pub fn validate(&self) -> Result<()> {
        if self.physical_batch == 0 {
            bail!("physical_batch must be > 0");
        }
        if self.ptrm_k.is_empty() || self.ptrm_k.contains(&0) {
            bail!("ptrm_k must contain only positive values");
        }
        let unique: BTreeSet<_> = self.ptrm_k.iter().copied().collect();
        if unique.len() != self.ptrm_k.len() {
            bail!("ptrm_k values must be unique");
        }
        if !(self.ptrm_noise.is_finite() && self.ptrm_noise >= 0.0) {
            bail!("ptrm_noise must be finite and >= 0");
        }
        if !(self.q_mse_threshold.is_finite() && self.q_mse_threshold >= 0.0) {
            bail!("q_mse_threshold must be finite and >= 0");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetrics {
    pub labeled: usize,
    pub accuracy: Option<f64>,
    pub bce: Option<f64>,
    pub hazard_failure_labeled: usize,
    pub hazard_false_negatives: usize,
    pub hazard_false_negative_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMetrics {
    pub n: usize,
    pub brier: Option<f64>,
    pub accuracy: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtrmKMetrics {
    pub k: usize,
    pub noise: f64,
    pub n: usize,
    pub pass_at_k: f64,
    pub best_q_at_k: f64,
    pub disagreement: f64,
    /// Deterministic model-trajectory evaluations per transition.
    pub trajectory_evaluations_per_transition: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutMetrics {
    pub n4: usize,
    pub mse_4: Option<f64>,
    pub n8: usize,
    pub mse_8: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedComputeMetrics {
    pub ptrm_k_equivalent: usize,
    pub outer_steps: usize,
    pub n: usize,
    pub one_step_latent_mse: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitEval {
    pub source: String,
    pub n_samples: usize,
    pub one_step_latent_mse: Option<f64>,
    pub events: EventMetrics,
    pub q: QMetrics,
    pub ptrm: Vec<PtrmKMetrics>,
    pub deterministic_matched_compute: Vec<MatchedComputeMetrics>,
    pub rollout: Option<RolloutMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub schema: String,
    pub seed: u64,
    pub checkpoint: PathBuf,
    pub device: String,
    pub q_mse_threshold: f64,
    pub ptrm_k: Vec<usize>,
    pub ptrm_noise: f64,
    /// Official ARC-AGI-3 RHAE is not computed here.
    pub official_rhae: Option<f64>,
    /// Public ARC recordings/games were not used for fitting.
    pub public_data_used_for_fitting: bool,
    pub synthetic: SplitEval,
    pub arc3: Option<SplitEval>,
    /// Smoke / scaffolding only; not a research result.
    pub research_claim: bool,
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
    }
    let json = serde_json::to_string_pretty(value).context("serialize eval report")?;
    let tmp = {
        let mut os = path.as_os_str().to_owned();
        os.push(".tmp");
        PathBuf::from(os)
    };
    fs::write(&tmp, &json).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

pub fn load_model(
    train_cfg: &TrainConfig,
    weights: &Path,
    device: &Device,
) -> Result<(WorldModel, VarMap)> {
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = WorldModel::new(train_cfg.model_config(), vb)?;
    varmap
        .load(weights)
        .with_context(|| format!("load {}", weights.display()))?;
    Ok((model, varmap))
}

fn collect_synthetic(seed: u64, episodes: usize, kinds: &[&str]) -> Result<Vec<TransitionSample>> {
    let mut out = Vec::new();
    for (i, kind) in kinds.iter().enumerate() {
        for ep in 0..episodes {
            let episode_id = (i as u64).wrapping_mul(1_000_003).wrapping_add(ep as u64);
            out.extend(generate_curriculum(
                kind,
                seed,
                episode_id,
                Split::HeldOutComposition,
            )?);
        }
    }
    Ok(out)
}

fn chunks(samples: &[TransitionSample], batch: usize) -> Vec<&[TransitionSample]> {
    if samples.is_empty() || batch == 0 {
        return Vec::new();
    }
    samples.chunks(batch).collect()
}

fn per_sample_mse(pred: &Tensor, target: &Tensor) -> Result<Vec<f32>> {
    let mse = pred.sub(target)?.sqr()?.mean(D::Minus1)?;
    mse.to_dtype(DType::F32)?
        .to_vec1::<f32>()
        .map_err(Into::into)
}

fn mean(xs: &[f32]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(xs.iter().map(|v| *v as f64).sum::<f64>() / xs.len() as f64)
}

fn eval_events(
    logits: &Tensor,
    targets: &Tensor,
    mask: &Tensor,
) -> Result<(usize, Option<f64>, Option<f64>)> {
    let logits = logits.to_dtype(DType::F32)?;
    let targets = targets.to_dtype(DType::F32)?;
    let mask = mask.to_dtype(DType::F32)?;
    let (b, e) = logits.dims2()?;
    let logit_v = logits.flatten_all()?.to_vec1::<f32>()?;
    let tgt_v = targets.flatten_all()?.to_vec1::<f32>()?;
    let mask_v = mask.flatten_all()?.to_vec1::<f32>()?;
    let mut labeled = 0usize;
    let mut correct = 0usize;
    let mut bce_sum = 0f64;
    for i in 0..b * e {
        if mask_v[i] <= 0.0 {
            continue;
        }
        labeled += 1;
        let p = 1.0 / (1.0 + (-logit_v[i]).exp());
        let y = tgt_v[i];
        let pred = if p >= 0.5 { 1.0 } else { 0.0 };
        if (pred - y).abs() < 0.5 {
            correct += 1;
        }
        let p = p.clamp(1e-7, 1.0 - 1e-7);
        bce_sum += -(y as f64 * (p as f64).ln() + (1.0 - y as f64) * (1.0 - p as f64).ln());
    }
    if labeled == 0 {
        Ok((0, None, None))
    } else {
        Ok((
            labeled,
            Some(correct as f64 / labeled as f64),
            Some(bce_sum / labeled as f64),
        ))
    }
}

fn eval_q(q_logit: &Tensor, mse: &[f32], threshold: f64) -> Result<QMetrics> {
    let probs = ops::sigmoid(q_logit)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if probs.len() != mse.len() {
        bail!("Q batch {} != mse {}", probs.len(), mse.len());
    }
    let mut brier = 0f64;
    let mut correct = 0usize;
    for (p, m) in probs.iter().zip(mse.iter()) {
        let y = if (*m as f64) < threshold { 1.0 } else { 0.0 };
        let p = *p as f64;
        brier += (p - y).powi(2);
        let pred = if p >= 0.5 { 1.0 } else { 0.0 };
        if (pred - y).abs() < 0.5 {
            correct += 1;
        }
    }
    let n = probs.len();
    Ok(QMetrics {
        n,
        brier: (n > 0).then_some(brier / n as f64),
        accuracy: (n > 0).then_some(correct as f64 / n as f64),
    })
}

fn ptrm_metrics(
    model: &WorldModel,
    batch: &BatchTensors,
    next_z: &Tensor,
    ks: &[usize],
    noise: f64,
    threshold: f64,
    seed: u64,
) -> Result<Vec<PtrmKMetrics>> {
    let mut out = Vec::with_capacity(ks.len());
    for &k in ks {
        if k == 0 {
            continue;
        }
        let effective_noise = if k == 1 { 0.0 } else { noise };
        let ptrm = model.forward_ptrm(
            &batch.frames,
            &batch.actions,
            &batch.action_coords,
            &batch.goals,
            PtrmConfig {
                k,
                sigma: effective_noise,
                seed: Some(seed),
            },
        )?;
        let b = next_z.dim(0)?;
        let mut pass = 0usize;
        let mut best_q_pass = 0usize;
        let mut disagree_acc = 0f64;
        for sample in 0..b {
            let mut any_pass = false;
            let mut ys = Vec::with_capacity(k);
            for traj in &ptrm.trajectories {
                let y = traj.y.get(sample)?;
                let t = next_z.get(sample)?;
                let mse = y.sub(&t)?.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
                if mse < threshold {
                    any_pass = true;
                }
                ys.push(y.flatten_all()?.to_vec1::<f32>()?);
            }
            if any_pass {
                pass += 1;
            }
            let best = ptrm.best_indices[sample];
            let y_best = ptrm.trajectories[best].y.get(sample)?;
            let t = next_z.get(sample)?;
            let mse_best = y_best.sub(&t)?.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
            if mse_best < threshold {
                best_q_pass += 1;
            }
            // Mean pairwise L2 disagreement across trajectory latents.
            let mut dsum = 0f64;
            let mut dpairs = 0usize;
            for i in 0..k {
                for j in (i + 1)..k {
                    let mut s = 0f64;
                    for (a, b) in ys[i].iter().zip(ys[j].iter()) {
                        let d = (*a - *b) as f64;
                        s += d * d;
                    }
                    dsum += s.sqrt();
                    dpairs += 1;
                }
            }
            if dpairs > 0 {
                disagree_acc += dsum / dpairs as f64;
            }
        }
        out.push(PtrmKMetrics {
            k,
            noise: effective_noise,
            n: b,
            pass_at_k: pass as f64 / b as f64,
            best_q_at_k: best_q_pass as f64 / b as f64,
            disagreement: disagree_acc / b as f64,
            trajectory_evaluations_per_transition: k,
        });
    }
    Ok(out)
}

/// Group by stable episode identity. `family` can change during retarget traces.
fn group_rollouts(samples: &[TransitionSample]) -> BTreeMap<(u64, u64), Vec<&TransitionSample>> {
    let mut map: BTreeMap<(u64, u64), Vec<&TransitionSample>> = BTreeMap::new();
    for s in samples {
        map.entry((s.seed, s.episode_id)).or_default().push(s);
    }
    map
}

fn eval_rollouts(
    model: &WorldModel,
    samples: &[TransitionSample],
    device: &Device,
) -> Result<RolloutMetrics> {
    let groups = group_rollouts(samples);
    let mut mse4 = Vec::new();
    let mut mse8 = Vec::new();
    for ((_seed, _ep), steps) in groups {
        if steps.len() < 4 {
            continue;
        }
        // Encode first current frame, then roll latent transitions in order.
        let first = batch_from_samples(&[steps[0].clone()], device)?;
        let mut latent = model.encode_state(&first.frames)?;
        for (idx, sample) in steps.iter().enumerate() {
            let batch = batch_from_samples(&[(*sample).clone()], device)?;
            let out = model.forward_from_latent(
                &latent,
                &batch.actions,
                &batch.action_coords,
                &batch.goals,
            )?;
            latent = out.y;
            let target = model.encode_state(&batch.next_frames)?;
            let mse = latent.sub(&target)?.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
            let step_no = idx + 1;
            if step_no == 4 {
                mse4.push(mse as f32);
            }
            if step_no == 8 {
                mse8.push(mse as f32);
            }
        }
    }
    Ok(RolloutMetrics {
        n4: mse4.len(),
        mse_4: mean(&mse4),
        n8: mse8.len(),
        mse_8: mean(&mse8),
    })
}

fn eval_sample_set(
    model: &WorldModel,
    samples: &[TransitionSample],
    source: &str,
    cfg: &EvalConfig,
    device: &Device,
    with_rollout: bool,
) -> Result<SplitEval> {
    if samples.is_empty() {
        return Ok(SplitEval {
            source: source.into(),
            n_samples: 0,
            one_step_latent_mse: None,
            events: EventMetrics {
                labeled: 0,
                accuracy: None,
                bce: None,
                hazard_failure_labeled: 0,
                hazard_false_negatives: 0,
                hazard_false_negative_rate: None,
            },
            q: QMetrics {
                n: 0,
                brier: None,
                accuracy: None,
            },
            ptrm: Vec::new(),
            deterministic_matched_compute: Vec::new(),
            rollout: None,
        });
    }

    let mut mse_all = Vec::new();
    let mut event_labeled = 0usize;
    let mut event_correct = 0f64;
    let mut event_bce = 0f64;
    let mut hazard_failure_labeled = 0usize;
    let mut hazard_false_negatives = 0usize;
    let mut q_brier = 0f64;
    let mut q_acc = 0f64;
    let mut q_n = 0usize;
    let mut ptrm_acc: BTreeMap<usize, (f64, f64, f64, usize)> = BTreeMap::new();
    let mut matched_acc: BTreeMap<usize, (f64, usize, usize)> = BTreeMap::new();

    let batch_size = cfg.physical_batch.max(1);
    for (bi, chunk) in chunks(samples, batch_size).into_iter().enumerate() {
        // Drop trailing incomplete batch for SIGReg-free eval; still evaluate if len>=1.
        if chunk.is_empty() {
            continue;
        }
        let batch = batch_from_samples(chunk, device)?;
        let out = model.forward(
            &batch.frames,
            &batch.actions,
            &batch.action_coords,
            &batch.goals,
        )?;
        let next_z = model.encode_state(&batch.next_frames)?;
        let mses = per_sample_mse(&out.y, &next_z)?;
        mse_all.extend(mses.iter().copied());

        let (lab, acc, bce) =
            eval_events(&out.event_logits, &batch.event_targets, &batch.event_mask)?;
        if lab > 0 {
            event_labeled += lab;
            event_correct += acc.unwrap_or(0.0) * lab as f64;
            event_bce += bce.unwrap_or(0.0) * lab as f64;
        }
        let event_logits = out.event_logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        for (sample, logits) in chunk.iter().zip(event_logits.iter()) {
            if sample.family == "avoid_hazard_reach_marker" && sample.goal_failed == Some(true) {
                hazard_failure_labeled += 1;
                if logits[EVENT_GOAL_FAILED] < 0.0 {
                    hazard_false_negatives += 1;
                }
            }
        }

        let q = eval_q(&out.q_logit, &mses, cfg.q_mse_threshold)?;
        q_brier += q.brier.unwrap_or(0.0) * q.n as f64;
        q_acc += q.accuracy.unwrap_or(0.0) * q.n as f64;
        q_n += q.n;

        for &k in &cfg.ptrm_k {
            let outer_steps = model
                .config()
                .outer_steps
                .checked_mul(k)
                .ok_or_else(|| anyhow::anyhow!("matched-compute outer_steps overflow"))?;
            let deterministic = model.forward_with_outer_steps(
                &batch.frames,
                &batch.actions,
                &batch.action_coords,
                &batch.goals,
                outer_steps,
            )?;
            let values = per_sample_mse(&deterministic.y, &next_z)?;
            let entry = matched_acc.entry(k).or_insert((0.0, 0, outer_steps));
            entry.0 += values.iter().map(|value| f64::from(*value)).sum::<f64>();
            entry.1 += values.len();
        }

        let ptrm = ptrm_metrics(
            model,
            &batch,
            &next_z,
            &cfg.ptrm_k,
            cfg.ptrm_noise,
            cfg.q_mse_threshold,
            cfg.seed.wrapping_add(bi as u64),
        )?;
        for m in ptrm {
            let e = ptrm_acc.entry(m.k).or_insert((0.0, 0.0, 0.0, 0));
            e.0 += m.pass_at_k * m.n as f64;
            e.1 += m.best_q_at_k * m.n as f64;
            e.2 += m.disagreement * m.n as f64;
            e.3 += m.n;
        }
    }

    let events = if event_labeled == 0 {
        EventMetrics {
            labeled: 0,
            accuracy: None,
            bce: None,
            hazard_failure_labeled,
            hazard_false_negatives,
            hazard_false_negative_rate: None,
        }
    } else {
        EventMetrics {
            labeled: event_labeled,
            accuracy: Some(event_correct / event_labeled as f64),
            bce: Some(event_bce / event_labeled as f64),
            hazard_failure_labeled,
            hazard_false_negatives,
            hazard_false_negative_rate: (hazard_failure_labeled > 0)
                .then_some(hazard_false_negatives as f64 / hazard_failure_labeled as f64),
        }
    };
    let q = QMetrics {
        n: q_n,
        brier: (q_n > 0).then_some(q_brier / q_n as f64),
        accuracy: (q_n > 0).then_some(q_acc / q_n as f64),
    };
    let ptrm = ptrm_acc
        .into_iter()
        .map(|(k, (p, bq, d, n))| PtrmKMetrics {
            k,
            noise: if k == 1 { 0.0 } else { cfg.ptrm_noise },
            n,
            pass_at_k: if n == 0 { 0.0 } else { p / n as f64 },
            best_q_at_k: if n == 0 { 0.0 } else { bq / n as f64 },
            disagreement: if n == 0 { 0.0 } else { d / n as f64 },
            trajectory_evaluations_per_transition: k,
        })
        .collect();
    let deterministic_matched_compute = matched_acc
        .into_iter()
        .map(|(k, (sum, n, outer_steps))| MatchedComputeMetrics {
            ptrm_k_equivalent: k,
            outer_steps,
            n,
            one_step_latent_mse: (n > 0).then_some(sum / n as f64),
        })
        .collect();

    let rollout = if with_rollout {
        Some(eval_rollouts(model, samples, device)?)
    } else {
        None
    };

    Ok(SplitEval {
        source: source.into(),
        n_samples: samples.len(),
        one_step_latent_mse: mean(&mse_all),
        events,
        q,
        ptrm,
        deterministic_matched_compute,
        rollout,
    })
}

/// Full evaluation: synthetic held-out (+ optional ARC recordings dir).
pub fn evaluate(cfg: &EvalConfig) -> Result<EvalReport> {
    cfg.validate()?;
    let train_cfg = load_train_config(&cfg.train_config)?;
    if (cfg.q_mse_threshold - train_cfg.q_mse_threshold).abs() > f64::EPSILON {
        bail!(
            "evaluation q_mse_threshold={} differs from frozen training threshold={}",
            cfg.q_mse_threshold,
            train_cfg.q_mse_threshold
        );
    }
    let device = resolve_device(&cfg.device)?;
    let (model, _varmap) = load_model(&train_cfg, &cfg.checkpoint, &device)?;

    let synthetic_samples = collect_synthetic(
        cfg.seed,
        cfg.synthetic_episodes,
        &[
            "random_one_step",
            "sequential",
            "p1c_falsification",
            "p1c_hard_retarget",
        ],
    )?;
    // Rollout metrics use sequential + retarget traces only.
    let mut rollout_samples = Vec::new();
    for (kind_index, kind) in ["sequential", "p1c_hard_retarget"].into_iter().enumerate() {
        for ep in 0..cfg.synthetic_episodes {
            let episode_id = (kind_index as u64)
                .wrapping_mul(1_000_003)
                .wrapping_add(ep as u64);
            rollout_samples.extend(generate_curriculum(
                kind,
                cfg.seed,
                episode_id,
                Split::HeldOutComposition,
            )?);
        }
    }

    let mut synthetic = eval_sample_set(
        &model,
        &synthetic_samples,
        "synthetic_held_out",
        cfg,
        &device,
        false,
    )?;
    synthetic.rollout = Some(eval_rollouts(&model, &rollout_samples, &device)?);

    let arc3 = if let Some(dir) = &cfg.arc_recordings_dir {
        let samples = import_recordings_dir(dir)?;
        Some(eval_sample_set(
            &model,
            &samples,
            "arc3_recordings_transfer",
            cfg,
            &device,
            false,
        )?)
    } else {
        None
    };

    let report = EvalReport {
        schema: EVAL_REPORT_SCHEMA.into(),
        seed: cfg.seed,
        checkpoint: cfg.checkpoint.clone(),
        device: cfg.device.clone(),
        q_mse_threshold: cfg.q_mse_threshold,
        ptrm_k: cfg.ptrm_k.clone(),
        ptrm_noise: cfg.ptrm_noise,
        official_rhae: None,
        public_data_used_for_fitting: false,
        synthetic,
        arc3,
        research_claim: false,
    };
    write_json_atomic(&cfg.output, &report)?;
    Ok(report)
}

/// ARC-only transfer eval helper.
pub fn evaluate_arc3(cfg: &EvalConfig) -> Result<EvalReport> {
    if cfg.arc_recordings_dir.is_none() {
        bail!("p2-arc3-eval requires --arc-recordings-dir");
    }
    evaluate(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::train::{
        reinit_varmap_deterministic, save_checkpoint, TrainReport, TRAIN_REPORT_SCHEMA,
    };

    #[test]
    fn tiny_eval_and_report_schema() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("tofy-p2-eval-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir)?;

        let train_cfg = TrainConfig {
            output_dir: dir.clone(),
            steps_per_lesson: 1,
            lessons: vec!["dynamics".into()],
            physical_batch: 2,
            grad_accum: 1,
            hidden_dim: 16,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            ..TrainConfig::default()
        };

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let _model = WorldModel::new(train_cfg.model_config(), vb)?;
        reinit_varmap_deterministic(&varmap, train_cfg.seed)?;
        let report = TrainReport {
            schema: TRAIN_REPORT_SCHEMA.into(),
            seed: train_cfg.seed,
            physical_batch: train_cfg.physical_batch,
            grad_accum: 1,
            lr: train_cfg.lr,
            weight_decay: train_cfg.weight_decay,
            parameter_count: 1,
            device: "cpu".into(),
            lessons: vec![],
            checkpoint: dir.join("model.safetensors"),
            config_path: dir.join("config.json"),
            runtime_trace: dir.join("runtime.json"),
            research_claim: false,
        };
        save_checkpoint(&varmap, &train_cfg, &report)?;

        let eval_cfg = EvalConfig {
            checkpoint: dir.join("model.safetensors"),
            train_config: dir.join("config.json"),
            seed: 3,
            synthetic_episodes: 1,
            physical_batch: 2,
            ptrm_k: vec![1, 2],
            ptrm_noise: 0.0,
            q_mse_threshold: train_cfg.q_mse_threshold,
            device: "cpu".into(),
            arc_recordings_dir: None,
            output: dir.join("eval.json"),
        };
        let eval = evaluate(&eval_cfg)?;
        assert_eq!(eval.schema, EVAL_REPORT_SCHEMA);
        assert!(eval.official_rhae.is_none());
        assert!(!eval.public_data_used_for_fitting);
        assert!(!eval.research_claim);
        assert!(eval.synthetic.n_samples > 0);
        assert!(eval
            .synthetic
            .one_step_latent_mse
            .is_some_and(f64::is_finite));
        assert_eq!(
            eval.synthetic
                .ptrm
                .iter()
                .find(|row| row.k == 1)
                .map(|row| row.noise),
            Some(0.0)
        );
        assert_eq!(eval.synthetic.deterministic_matched_compute.len(), 2);
        let text = fs::read_to_string(&eval_cfg.output)?;
        let back: EvalReport = serde_json::from_str(&text)?;
        assert_eq!(back.schema, EVAL_REPORT_SCHEMA);

        let recordings = dir.join("empty-recordings");
        fs::create_dir_all(&recordings)?;
        let mut arc_cfg = eval_cfg.clone();
        arc_cfg.synthetic_episodes = 0;
        arc_cfg.arc_recordings_dir = Some(recordings);
        arc_cfg.output = dir.join("arc-eval.json");
        let arc_eval = evaluate_arc3(&arc_cfg)?;
        assert_eq!(arc_eval.synthetic.one_step_latent_mse, None);
        assert_eq!(arc_eval.arc3.as_ref().map(|split| split.n_samples), Some(0));
        let _: EvalReport = serde_json::from_str(&fs::read_to_string(&arc_cfg.output)?)?;

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }
}
