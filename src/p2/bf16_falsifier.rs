//! Frozen-checkpoint numerics and throughput falsifiers for the recurrent-core
//! BF16 treatment. These tools never mutate the supplied run directory.

use crate::p2::data::{
    compose_mixed_stream_batch, foundation_v2_stream_schedule, MixedStreamConfig, V5DataSplit,
    FRAME_SIDE,
};
use crate::p2::eval::load_model;
use crate::p2::train::{
    batch_from_samples, benchmark_bf16_recurrent_core, foundation_v2_rollout_falsifier,
    load_train_config, resolve_device, Bf16BenchmarkReport, TrainConfig,
};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const BF16_DRIFT_SCHEMA: &str = "p2.bf16_recurrent_core_drift.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bf16DriftReport {
    pub schema: String,
    pub checkpoint: PathBuf,
    pub train_config: PathBuf,
    pub device: String,
    pub seed: u64,
    pub batch_size: usize,
    pub latent_elements: usize,
    pub latent_max_abs_drift: f64,
    pub logit_elements: usize,
    pub logit_max_abs_drift: f64,
    pub changed_pixels: usize,
    pub changed_pixel_prediction_flips: usize,
    pub changed_pixel_prediction_flip_rate: f64,
    pub content_pixels: usize,
    pub composed_decode_flips: usize,
    pub composed_decode_flip_rate: f64,
    pub f32_rollout_loss: f64,
    pub bf16_rollout_loss: f64,
    pub f32_rollout_fragments: usize,
    pub bf16_rollout_fragments: usize,
}

struct ArmOutputs {
    latent: Vec<f32>,
    logits: Vec<f32>,
    raw_predictions: Vec<u32>,
    composed_predictions: Vec<u32>,
    rollout_loss: f64,
    rollout_fragments: usize,
}

fn ensure_baseline_config(cfg: &TrainConfig) -> Result<()> {
    if cfg.recipe != crate::p2::experiment::TrainingRecipe::FoundationV2 {
        bail!("BF16 falsifier requires a foundation-v2 training config");
    }
    if cfg.bf16_recurrent_core {
        bail!("BF16 falsifier requires an F32 baseline config with bf16_recurrent_core=false");
    }
    Ok(())
}

fn arm_outputs(
    cfg: &TrainConfig,
    checkpoint: &Path,
    device: &Device,
    mixed: &crate::p2::data::MixedStreamBatch,
) -> Result<ArmOutputs> {
    let (model, varmap) = load_model(cfg, checkpoint, device)?;
    let transitions = mixed.transitions().cloned().collect::<Vec<_>>();
    let batch = batch_from_samples(&transitions, device)?;
    let (rollout_loss, rollout_fragments) = foundation_v2_rollout_falsifier(&model, mixed, device)?;
    let out = model.forward_with_operator_conditioning(
        &batch.model_frames,
        &batch.actions,
        &batch.action_coords,
        &batch.goals,
        &batch.operator_conditioning,
    )?;
    let logits = model.exact_gameplay_logits(&out.y)?;
    let raw_predictions = logits.argmax(D::Minus1)?;
    let composed_predictions = model.composed_gameplay_decode(&out.y, &batch.frames)?;
    if device.is_cuda() {
        device.synchronize()?;
    }

    let latent = f32_values(&out.y, "recurrent latent")?;
    let logits = f32_values(&logits, "exact decoder logits")?;
    let raw_predictions = raw_predictions.flatten_all()?.to_vec1::<u32>()?;
    let composed_predictions = composed_predictions.flatten_all()?.to_vec1::<u32>()?;
    drop(model);
    drop(varmap);
    Ok(ArmOutputs {
        latent,
        logits,
        raw_predictions,
        composed_predictions,
        rollout_loss,
        rollout_fragments,
    })
}

fn f32_values(tensor: &Tensor, name: &str) -> Result<Vec<f32>> {
    let values = tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if values.iter().any(|value| !value.is_finite()) {
        bail!("BF16 falsifier {name} contains a non-finite value");
    }
    Ok(values)
}

fn max_abs_drift(left: &[f32], right: &[f32], name: &str) -> Result<f64> {
    if left.len() != right.len() {
        bail!(
            "BF16 falsifier {name} length mismatch: {} vs {}",
            left.len(),
            right.len()
        );
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| f64::from((left - right).abs()))
        .fold(0.0f64, f64::max))
}

fn flip_metrics(
    mixed: &crate::p2::data::MixedStreamBatch,
    f32_raw: &[u32],
    bf16_raw: &[u32],
    f32_composed: &[u32],
    bf16_composed: &[u32],
) -> Result<(usize, usize, f64, usize, usize, f64)> {
    let gameplay_pixels = (FRAME_SIDE - 1) * FRAME_SIDE;
    let expected = mixed.samples().len() * gameplay_pixels;
    for (name, values) in [
        ("F32 raw predictions", f32_raw),
        ("BF16 raw predictions", bf16_raw),
        ("F32 composed predictions", f32_composed),
        ("BF16 composed predictions", bf16_composed),
    ] {
        if values.len() != expected {
            bail!("{name} has {} pixels, expected {expected}", values.len());
        }
    }

    let mut changed_pixels = 0usize;
    let mut changed_flips = 0usize;
    let mut content_pixels = 0usize;
    let mut composed_flips = 0usize;
    for (row, sample) in mixed.samples().iter().enumerate() {
        for pixel in 0..gameplay_pixels {
            if sample.content_mask.values[pixel] == 0 {
                continue;
            }
            let index = row * gameplay_pixels + pixel;
            content_pixels += 1;
            composed_flips += usize::from(f32_composed[index] != bf16_composed[index]);
            if sample.transition.current.pixels[pixel] != sample.transition.next.pixels[pixel] {
                changed_pixels += 1;
                changed_flips += usize::from(f32_raw[index] != bf16_raw[index]);
            }
        }
    }
    if changed_pixels == 0 || content_pixels == 0 {
        bail!(
            "BF16 drift population lacks support: changed_pixels={changed_pixels} content_pixels={content_pixels}"
        );
    }
    Ok((
        changed_pixels,
        changed_flips,
        changed_flips as f64 / changed_pixels as f64,
        content_pixels,
        composed_flips,
        composed_flips as f64 / content_pixels as f64,
    ))
}

pub fn compare_bf16_recurrent_core(
    train_config: &Path,
    checkpoint: &Path,
    device_spec: &str,
    seed: u64,
    batch_size: usize,
) -> Result<Bf16DriftReport> {
    let baseline_cfg = load_train_config(train_config)?;
    ensure_baseline_config(&baseline_cfg)?;
    baseline_cfg.validate()?;
    if batch_size < crate::p2::data::FACTUAL_BRANCHES_PER_GROUP {
        bail!(
            "BF16 drift batch_size must be at least {}",
            crate::p2::data::FACTUAL_BRANCHES_PER_GROUP
        );
    }
    let device = resolve_device(device_spec)?;
    let mixed = compose_mixed_stream_batch(
        &MixedStreamConfig {
            batch_size,
            seed,
            schedule: foundation_v2_stream_schedule,
            ..MixedStreamConfig::default()
        },
        1.0,
        0,
        V5DataSplit::UnseenSeed7x7,
    )?;

    let f32 = arm_outputs(&baseline_cfg, checkpoint, &device, &mixed)
        .context("run frozen F32 falsifier arm")?;
    let mut mixed_cfg = baseline_cfg.clone();
    mixed_cfg.bf16_recurrent_core = true;
    mixed_cfg.validate()?;
    let bf16 = arm_outputs(&mixed_cfg, checkpoint, &device, &mixed)
        .context("run frozen BF16 recurrent-core falsifier arm")?;
    let (
        changed_pixels,
        changed_flips,
        changed_flip_rate,
        content_pixels,
        composed_flips,
        composed_flip_rate,
    ) = flip_metrics(
        &mixed,
        &f32.raw_predictions,
        &bf16.raw_predictions,
        &f32.composed_predictions,
        &bf16.composed_predictions,
    )?;

    Ok(Bf16DriftReport {
        schema: BF16_DRIFT_SCHEMA.into(),
        checkpoint: checkpoint.to_path_buf(),
        train_config: train_config.to_path_buf(),
        device: device_spec.into(),
        seed,
        batch_size,
        latent_elements: f32.latent.len(),
        latent_max_abs_drift: max_abs_drift(&f32.latent, &bf16.latent, "latent")?,
        logit_elements: f32.logits.len(),
        logit_max_abs_drift: max_abs_drift(&f32.logits, &bf16.logits, "logit")?,
        changed_pixels,
        changed_pixel_prediction_flips: changed_flips,
        changed_pixel_prediction_flip_rate: changed_flip_rate,
        content_pixels,
        composed_decode_flips: composed_flips,
        composed_decode_flip_rate: composed_flip_rate,
        f32_rollout_loss: f32.rollout_loss,
        bf16_rollout_loss: bf16.rollout_loss,
        f32_rollout_fragments: f32.rollout_fragments,
        bf16_rollout_fragments: bf16.rollout_fragments,
    })
}

pub fn run_bf16_benchmark(
    train_config: &Path,
    checkpoint: &Path,
    device_spec: &str,
    warmup_updates: usize,
    measured_updates: usize,
) -> Result<Bf16BenchmarkReport> {
    let cfg = load_train_config(train_config)?;
    ensure_baseline_config(&cfg)?;
    benchmark_bf16_recurrent_core(
        &cfg,
        checkpoint,
        device_spec,
        warmup_updates,
        measured_updates,
    )
}

pub fn write_json_report(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(value).context("serialize BF16 falsifier report")?;
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));
    fs::write(&tmp, format!("{json}\n")).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_abs_drift_rejects_shape_drift_and_measures_values() -> Result<()> {
        assert_eq!(max_abs_drift(&[1.0, -2.0], &[1.25, -1.5], "test")?, 0.5);
        assert!(max_abs_drift(&[1.0], &[1.0, 2.0], "test").is_err());
        Ok(())
    }
}
