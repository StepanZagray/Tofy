//! Frozen-checkpoint numerics and throughput falsifiers for the recurrent-core
//! BF16 treatment. These tools never mutate the supplied run directory.

use crate::p2::data::{
    compose_mixed_stream_batch, foundation_v2_stream_schedule, MixedStreamBatch, MixedStreamConfig,
    V5DataSplit, FRAME_SIDE,
};
use crate::p2::eval::load_model;
use crate::p2::train::{
    batch_from_samples, benchmark_bf16_recurrent_core, foundation_v2_rollout_falsifier,
    load_train_config, resolve_device, Bf16BenchmarkReport, TrainConfig,
};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Report schema. v2 (Wave 22) adds the unchanged-content raw strata,
/// per-arm / directional raw false-edit counts, and [`Bf16DriftIdentity`].
/// Every v1 field keeps its name and meaning.
pub const BF16_DRIFT_SCHEMA: &str = "p2.bf16_recurrent_core_drift.v2";
/// Domain tag folded into [`Bf16DriftIdentity::identity_root`].
pub const BF16_DRIFT_IDENTITY_DOMAIN: &str = "tofy.p2.bf16_recurrent_core_drift.identity.v2";
/// Domain tag folded into [`Bf16DriftIdentity::population_sha256`].
pub const BF16_DRIFT_POPULATION_DOMAIN: &str = "tofy.p2.bf16_recurrent_core_drift.population.v2";
/// Domain tag folded into [`Bf16DriftIdentity::falsifier_config_sha256`].
pub const BF16_DRIFT_CONFIG_DOMAIN: &str = "tofy.p2.bf16_recurrent_core_drift.config.v2";

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
    /// Content pixels whose factual value did not change (`current == next`).
    pub unchanged_pixels: usize,
    /// Raw (pre-copy-gate) F32/BF16 argmax disagreement on unchanged content.
    /// The composed parity above can hide these when the copy gate restores
    /// the current value in both arms.
    pub unchanged_pixel_prediction_flips: usize,
    pub unchanged_pixel_prediction_flip_rate: f64,
    /// Per-arm raw false edits: unchanged content where the raw argmax is not
    /// the (unchanged) target value.
    pub f32_unchanged_raw_false_edits: usize,
    pub bf16_unchanged_raw_false_edits: usize,
    /// Unchanged content where F32 was correct and BF16 was not.
    pub raw_false_edits_introduced_by_bf16: usize,
    /// Unchanged content where BF16 was correct and F32 was not.
    pub raw_false_edits_resolved_by_bf16: usize,
    /// Per-arm raw correctness on changed content, for directionality of the
    /// changed-pixel flips.
    pub f32_changed_pixel_raw_correct: usize,
    pub bf16_changed_pixel_raw_correct: usize,
    /// Cryptographic identity of the inputs this report was computed from.
    pub identity: Bf16DriftIdentity,
}

/// Content-addressed identity of one drift comparison. Paths and argv are kept
/// as descriptive context only; `identity_root` binds the digests so a saved
/// report can be checked against an exact rerun after relocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bf16DriftIdentity {
    pub domain: String,
    /// SHA-256 of the checkpoint bytes, hashed immediately before the F32 arm
    /// and verified unchanged after the BF16 arm.
    pub checkpoint_sha256: String,
    /// SHA-256 of the exact training-config bytes that were parsed.
    pub train_config_sha256: String,
    pub evaluator_binary: PathBuf,
    pub evaluator_binary_sha256: String,
    pub evaluator_package_version: String,
    pub command: Vec<String>,
    pub command_sha256: String,
    /// SHA-256 over the falsifier's own parameters (seed, batch, device,
    /// split, schedule, progress, batch index).
    pub falsifier_config_sha256: String,
    pub population_rows: usize,
    /// Domain-separated SHA-256 over the ordered `V5Sample` rows (transition,
    /// content mask, V5 provenance) both arms consumed.
    pub population_sha256: String,
    /// Domain-separated SHA-256 over the digests above, in fixed order.
    pub identity_root: String,
}

/// Digest leaves that enter [`Bf16DriftIdentity::identity_root`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bf16DriftIdentityLeaves<'a> {
    pub checkpoint_sha256: &'a str,
    pub train_config_sha256: &'a str,
    pub evaluator_binary_sha256: &'a str,
    pub evaluator_package_version: &'a str,
    pub falsifier_config_sha256: &'a str,
    pub population_rows: usize,
    pub population_sha256: &'a str,
}

/// Parameters of one drift comparison that select the population and arms.
#[derive(Debug, Clone, Serialize)]
struct Bf16FalsifierConfig<'a> {
    schema: &'a str,
    seed: u64,
    batch_size: usize,
    device: &'a str,
    split: &'a str,
    stream_schedule: &'a str,
    progress: f32,
    batch_index: u64,
}

fn sha256_hex(digest: Sha256) -> String {
    format!("sha256:{:x}", digest.finalize())
}

fn digest_framed(digest: &mut Sha256, role: &str, value: &[u8]) {
    digest.update((role.len() as u64).to_le_bytes());
    digest.update(role.as_bytes());
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value);
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(sha256_hex(digest))
}

fn raw_bytes_sha256(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    sha256_hex(digest)
}

fn bytes_sha256(domain: &str, bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest_framed(&mut digest, "domain", domain.as_bytes());
    digest_framed(&mut digest, "bytes", bytes);
    sha256_hex(digest)
}

/// `domain || row_count || (ordinal || framed canonical JSON row)*` over the
/// ordered mixed-stream rows. `V5Sample` serializes with a fixed field order,
/// so the same rows always produce the same bytes.
fn population_sha256(mixed: &MixedStreamBatch) -> Result<String> {
    let mut digest = Sha256::new();
    digest_framed(
        &mut digest,
        "domain",
        BF16_DRIFT_POPULATION_DOMAIN.as_bytes(),
    );
    digest.update((mixed.samples().len() as u64).to_le_bytes());
    for (ordinal, sample) in mixed.samples().iter().enumerate() {
        digest.update((ordinal as u64).to_le_bytes());
        let row = serde_json::to_vec(sample).context("serialize BF16 population row")?;
        digest_framed(&mut digest, "row", &row);
    }
    Ok(sha256_hex(digest))
}

/// Fixed-order, domain-separated root over the identity leaves. Paths, argv,
/// and the output location are deliberately excluded so relocating the same
/// bytes preserves the root.
pub fn bf16_drift_identity_root(leaves: &Bf16DriftIdentityLeaves<'_>) -> String {
    let mut digest = Sha256::new();
    digest_framed(&mut digest, "domain", BF16_DRIFT_IDENTITY_DOMAIN.as_bytes());
    digest_framed(
        &mut digest,
        "checkpoint_sha256",
        leaves.checkpoint_sha256.as_bytes(),
    );
    digest_framed(
        &mut digest,
        "train_config_sha256",
        leaves.train_config_sha256.as_bytes(),
    );
    digest_framed(
        &mut digest,
        "evaluator_binary_sha256",
        leaves.evaluator_binary_sha256.as_bytes(),
    );
    digest_framed(
        &mut digest,
        "evaluator_package_version",
        leaves.evaluator_package_version.as_bytes(),
    );
    digest_framed(
        &mut digest,
        "falsifier_config_sha256",
        leaves.falsifier_config_sha256.as_bytes(),
    );
    digest_framed(
        &mut digest,
        "population_rows",
        &(leaves.population_rows as u64).to_le_bytes(),
    );
    digest_framed(
        &mut digest,
        "population_sha256",
        leaves.population_sha256.as_bytes(),
    );
    sha256_hex(digest)
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
    if cfg.world_core_v6 || cfg.data_contract_v6 {
        bail!(
            "BF16 falsifier is defined only for the Foundation-v2/V5 population; +             world_core_v6/data_contract_v6 configs require a separately registered v6 population"
        );
    }
    Ok(())
}

fn arm_outputs(
    cfg: &TrainConfig,
    checkpoint: &Path,
    device: &Device,
    mixed: &MixedStreamBatch,
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

/// One population row as seen by the flip reducer: the content mask and the
/// factual current/next pixels over the same gameplay pixel range.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FlipRow<'a> {
    pub content_mask: &'a [u8],
    pub current: &'a [u8],
    pub next: &'a [u8],
}

/// Pixel-level F32/BF16 comparison strata.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Bf16FlipMetrics {
    pub changed_pixels: usize,
    pub changed_flips: usize,
    pub changed_flip_rate: f64,
    pub content_pixels: usize,
    pub composed_flips: usize,
    pub composed_flip_rate: f64,
    pub unchanged_pixels: usize,
    pub unchanged_flips: usize,
    pub unchanged_flip_rate: f64,
    pub f32_unchanged_raw_false_edits: usize,
    pub bf16_unchanged_raw_false_edits: usize,
    pub raw_false_edits_introduced_by_bf16: usize,
    pub raw_false_edits_resolved_by_bf16: usize,
    pub f32_changed_raw_correct: usize,
    pub bf16_changed_raw_correct: usize,
}

/// Reduce raw and composed predictions of both arms over every content pixel.
///
/// Raw flips are stratified by whether the factual pixel changed. Composed
/// parity spans all content. Every stratum fails closed on an empty
/// denominator: a population without support cannot certify anything.
pub(crate) fn flip_metrics_rows(
    rows: &[FlipRow<'_>],
    pixels_per_row: usize,
    f32_raw: &[u32],
    bf16_raw: &[u32],
    f32_composed: &[u32],
    bf16_composed: &[u32],
) -> Result<Bf16FlipMetrics> {
    let expected = rows.len() * pixels_per_row;
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
    for (row_index, row) in rows.iter().enumerate() {
        for (name, values) in [
            ("content mask", row.content_mask),
            ("current frame", row.current),
            ("next frame", row.next),
        ] {
            if values.len() < pixels_per_row {
                bail!(
                    "BF16 falsifier row {row_index} {name} has {} pixels, expected at least {pixels_per_row}",
                    values.len()
                );
            }
        }
    }

    let mut metrics = Bf16FlipMetrics {
        changed_pixels: 0,
        changed_flips: 0,
        changed_flip_rate: 0.0,
        content_pixels: 0,
        composed_flips: 0,
        composed_flip_rate: 0.0,
        unchanged_pixels: 0,
        unchanged_flips: 0,
        unchanged_flip_rate: 0.0,
        f32_unchanged_raw_false_edits: 0,
        bf16_unchanged_raw_false_edits: 0,
        raw_false_edits_introduced_by_bf16: 0,
        raw_false_edits_resolved_by_bf16: 0,
        f32_changed_raw_correct: 0,
        bf16_changed_raw_correct: 0,
    };
    for (row_index, row) in rows.iter().enumerate() {
        for pixel in 0..pixels_per_row {
            if row.content_mask[pixel] == 0 {
                continue;
            }
            let index = row_index * pixels_per_row + pixel;
            metrics.content_pixels += 1;
            metrics.composed_flips += usize::from(f32_composed[index] != bf16_composed[index]);
            let target = u32::from(row.next[pixel]);
            let raw_flip = f32_raw[index] != bf16_raw[index];
            let f32_correct = f32_raw[index] == target;
            let bf16_correct = bf16_raw[index] == target;
            if row.current[pixel] != row.next[pixel] {
                metrics.changed_pixels += 1;
                metrics.changed_flips += usize::from(raw_flip);
                metrics.f32_changed_raw_correct += usize::from(f32_correct);
                metrics.bf16_changed_raw_correct += usize::from(bf16_correct);
            } else {
                metrics.unchanged_pixels += 1;
                metrics.unchanged_flips += usize::from(raw_flip);
                metrics.f32_unchanged_raw_false_edits += usize::from(!f32_correct);
                metrics.bf16_unchanged_raw_false_edits += usize::from(!bf16_correct);
                metrics.raw_false_edits_introduced_by_bf16 +=
                    usize::from(f32_correct && !bf16_correct);
                metrics.raw_false_edits_resolved_by_bf16 +=
                    usize::from(!f32_correct && bf16_correct);
            }
        }
    }
    if metrics.changed_pixels == 0 || metrics.unchanged_pixels == 0 || metrics.content_pixels == 0 {
        bail!(
            "BF16 drift population lacks support: changed_pixels={} unchanged_pixels={} content_pixels={}",
            metrics.changed_pixels,
            metrics.unchanged_pixels,
            metrics.content_pixels
        );
    }
    metrics.changed_flip_rate = metrics.changed_flips as f64 / metrics.changed_pixels as f64;
    metrics.composed_flip_rate = metrics.composed_flips as f64 / metrics.content_pixels as f64;
    metrics.unchanged_flip_rate = metrics.unchanged_flips as f64 / metrics.unchanged_pixels as f64;
    Ok(metrics)
}

fn flip_metrics(
    mixed: &MixedStreamBatch,
    f32_raw: &[u32],
    bf16_raw: &[u32],
    f32_composed: &[u32],
    bf16_composed: &[u32],
) -> Result<Bf16FlipMetrics> {
    let gameplay_pixels = (FRAME_SIDE - 1) * FRAME_SIDE;
    let rows = mixed
        .samples()
        .iter()
        .map(|sample| FlipRow {
            content_mask: &sample.content_mask.values,
            current: &sample.transition.current.pixels,
            next: &sample.transition.next.pixels,
        })
        .collect::<Vec<_>>();
    flip_metrics_rows(
        &rows,
        gameplay_pixels,
        f32_raw,
        bf16_raw,
        f32_composed,
        bf16_composed,
    )
}

const BF16_DRIFT_SPLIT: V5DataSplit = V5DataSplit::UnseenSeed7x7;
const BF16_DRIFT_PROGRESS: f32 = 1.0;
const BF16_DRIFT_BATCH_INDEX: u64 = 0;

pub fn compare_bf16_recurrent_core(
    train_config: &Path,
    checkpoint: &Path,
    device_spec: &str,
    seed: u64,
    batch_size: usize,
) -> Result<Bf16DriftReport> {
    // Parse the config from the exact bytes that are hashed, so the recorded
    // digest is the digest of what both arms actually consumed.
    let train_config_bytes =
        fs::read(train_config).with_context(|| format!("read {}", train_config.display()))?;
    let train_config_sha256 = raw_bytes_sha256(&train_config_bytes);
    let baseline_cfg: TrainConfig =
        serde_json::from_slice(&train_config_bytes).context("parse TrainConfig")?;
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
        BF16_DRIFT_PROGRESS,
        BF16_DRIFT_BATCH_INDEX,
        BF16_DRIFT_SPLIT,
    )?;
    let population_sha256 = population_sha256(&mixed)?;
    let falsifier_config_sha256 = bytes_sha256(
        BF16_DRIFT_CONFIG_DOMAIN,
        &serde_json::to_vec(&Bf16FalsifierConfig {
            schema: BF16_DRIFT_SCHEMA,
            seed,
            batch_size,
            device: device_spec,
            split: "UnseenSeed7x7",
            stream_schedule: "foundation_v2",
            progress: BF16_DRIFT_PROGRESS,
            batch_index: BF16_DRIFT_BATCH_INDEX,
        })
        .context("serialize BF16 falsifier config")?,
    );

    // Both executable and checkpoint are mutable paths. Hash both immediately
    // before the first arm and verify them after the last one so the report
    // never attributes a two-arm comparison to bytes that changed mid-run.
    let evaluator_binary = std::env::current_exe().context("resolve falsifier binary")?;
    let evaluator_binary_sha256 = file_sha256(&evaluator_binary)?;
    let checkpoint_sha256 = file_sha256(checkpoint)?;
    let f32 = arm_outputs(&baseline_cfg, checkpoint, &device, &mixed)
        .context("run frozen F32 falsifier arm")?;
    let mut mixed_cfg = baseline_cfg.clone();
    mixed_cfg.bf16_recurrent_core = true;
    mixed_cfg.validate()?;
    let bf16 = arm_outputs(&mixed_cfg, checkpoint, &device, &mixed)
        .context("run frozen BF16 recurrent-core falsifier arm")?;
    let checkpoint_after = file_sha256(checkpoint)?;
    if checkpoint_after != checkpoint_sha256 {
        bail!(
            "checkpoint {} changed during the BF16 falsifier: {checkpoint_sha256} before, {checkpoint_after} after",
            checkpoint.display()
        );
    }
    let evaluator_binary_after = file_sha256(&evaluator_binary)?;
    if evaluator_binary_after != evaluator_binary_sha256 {
        bail!(
            "evaluator binary {} changed during the BF16 falsifier: {evaluator_binary_sha256} before, {evaluator_binary_after} after",
            evaluator_binary.display()
        );
    }
    let flips = flip_metrics(
        &mixed,
        &f32.raw_predictions,
        &bf16.raw_predictions,
        &f32.composed_predictions,
        &bf16.composed_predictions,
    )?;

    let evaluator_package_version = env!("CARGO_PKG_VERSION").to_string();
    let command = std::env::args().collect::<Vec<_>>();
    let command_sha256 =
        raw_bytes_sha256(&serde_json::to_vec(&command).context("serialize falsifier argv")?);
    let identity_root = bf16_drift_identity_root(&Bf16DriftIdentityLeaves {
        checkpoint_sha256: &checkpoint_sha256,
        train_config_sha256: &train_config_sha256,
        evaluator_binary_sha256: &evaluator_binary_sha256,
        evaluator_package_version: &evaluator_package_version,
        falsifier_config_sha256: &falsifier_config_sha256,
        population_rows: mixed.samples().len(),
        population_sha256: &population_sha256,
    });

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
        changed_pixels: flips.changed_pixels,
        changed_pixel_prediction_flips: flips.changed_flips,
        changed_pixel_prediction_flip_rate: flips.changed_flip_rate,
        content_pixels: flips.content_pixels,
        composed_decode_flips: flips.composed_flips,
        composed_decode_flip_rate: flips.composed_flip_rate,
        f32_rollout_loss: f32.rollout_loss,
        bf16_rollout_loss: bf16.rollout_loss,
        f32_rollout_fragments: f32.rollout_fragments,
        bf16_rollout_fragments: bf16.rollout_fragments,
        unchanged_pixels: flips.unchanged_pixels,
        unchanged_pixel_prediction_flips: flips.unchanged_flips,
        unchanged_pixel_prediction_flip_rate: flips.unchanged_flip_rate,
        f32_unchanged_raw_false_edits: flips.f32_unchanged_raw_false_edits,
        bf16_unchanged_raw_false_edits: flips.bf16_unchanged_raw_false_edits,
        raw_false_edits_introduced_by_bf16: flips.raw_false_edits_introduced_by_bf16,
        raw_false_edits_resolved_by_bf16: flips.raw_false_edits_resolved_by_bf16,
        f32_changed_pixel_raw_correct: flips.f32_changed_raw_correct,
        bf16_changed_pixel_raw_correct: flips.bf16_changed_raw_correct,
        identity: Bf16DriftIdentity {
            domain: BF16_DRIFT_IDENTITY_DOMAIN.into(),
            checkpoint_sha256,
            train_config_sha256,
            evaluator_binary,
            evaluator_binary_sha256,
            evaluator_package_version,
            command,
            command_sha256,
            falsifier_config_sha256,
            population_rows: mixed.samples().len(),
            population_sha256,
            identity_root,
        },
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

/// Sidecar path carrying the SHA-256 of the exact bytes written to `path`.
pub fn report_sha256_sidecar_path(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_owned();
    os.push(".sha256");
    PathBuf::from(os)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));
    fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Write a pretty-printed JSON report atomically, followed by a
/// `<path>.sha256` sidecar in `sha256sum` format over the exact report bytes.
pub fn write_json_report(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(value).context("serialize BF16 falsifier report")?;
    let bytes = format!("{json}\n");
    write_atomic(path, bytes.as_bytes())?;
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let sidecar = format!("{:x}  {file_name}\n", Sha256::digest(bytes.as_bytes()));
    write_atomic(&report_sha256_sidecar_path(path), sidecar.as_bytes())
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

    /// Wave 22 finding 15 witness: two content pixels, current `[0,2]`, next
    /// `[1,2]`, F32 raw `[1,2]`, BF16 raw `[1,3]`, both composed `[1,2]`.
    /// The v1 strata (changed raw, all-content composed) are both zero while
    /// BF16 made a raw false edit on the unchanged pixel.
    #[test]
    fn unchanged_content_raw_false_edit_is_visible() -> Result<()> {
        let rows = [FlipRow {
            content_mask: &[1, 1],
            current: &[0, 2],
            next: &[1, 2],
        }];
        let metrics = flip_metrics_rows(&rows, 2, &[1, 2], &[1, 3], &[1, 2], &[1, 2])?;
        assert_eq!(metrics.changed_pixels, 1);
        assert_eq!(metrics.changed_flips, 0);
        assert_eq!(metrics.changed_flip_rate, 0.0);
        assert_eq!(metrics.content_pixels, 2);
        assert_eq!(metrics.composed_flips, 0);
        assert_eq!(metrics.composed_flip_rate, 0.0);
        assert_eq!(metrics.unchanged_pixels, 1);
        assert_eq!(metrics.unchanged_flips, 1);
        assert_eq!(metrics.unchanged_flip_rate, 1.0);
        assert_eq!(metrics.f32_unchanged_raw_false_edits, 0);
        assert_eq!(metrics.bf16_unchanged_raw_false_edits, 1);
        assert_eq!(metrics.raw_false_edits_introduced_by_bf16, 1);
        assert_eq!(metrics.raw_false_edits_resolved_by_bf16, 0);
        assert_eq!(metrics.f32_changed_raw_correct, 1);
        assert_eq!(metrics.bf16_changed_raw_correct, 1);

        // Directional control: swap the arms and the edit is "resolved".
        let swapped = flip_metrics_rows(&rows, 2, &[1, 3], &[1, 2], &[1, 2], &[1, 2])?;
        assert_eq!(swapped.f32_unchanged_raw_false_edits, 1);
        assert_eq!(swapped.bf16_unchanged_raw_false_edits, 0);
        assert_eq!(swapped.raw_false_edits_introduced_by_bf16, 0);
        assert_eq!(swapped.raw_false_edits_resolved_by_bf16, 1);

        // Wrong-to-different-wrong: a flip that is neither introduced nor resolved.
        let both_wrong = flip_metrics_rows(&rows, 2, &[1, 4], &[1, 3], &[1, 2], &[1, 2])?;
        assert_eq!(both_wrong.unchanged_flips, 1);
        assert_eq!(both_wrong.f32_unchanged_raw_false_edits, 1);
        assert_eq!(both_wrong.bf16_unchanged_raw_false_edits, 1);
        assert_eq!(both_wrong.raw_false_edits_introduced_by_bf16, 0);
        assert_eq!(both_wrong.raw_false_edits_resolved_by_bf16, 0);
        Ok(())
    }

    #[test]
    fn flip_reducer_respects_mask_and_fails_closed_without_support() -> Result<()> {
        // Masked pixel is ignored even when the arms disagree there.
        let rows = [FlipRow {
            content_mask: &[1, 1, 0],
            current: &[0, 2, 5],
            next: &[1, 2, 5],
        }];
        let metrics = flip_metrics_rows(&rows, 3, &[1, 2, 0], &[1, 2, 9], &[1, 2, 0], &[1, 2, 9])?;
        assert_eq!(metrics.content_pixels, 2);
        assert_eq!(metrics.composed_flips, 0);
        assert_eq!(metrics.unchanged_flips, 0);

        // No unchanged content: fail closed.
        let only_changed = [FlipRow {
            content_mask: &[1],
            current: &[0],
            next: &[1],
        }];
        let err = flip_metrics_rows(&only_changed, 1, &[1], &[1], &[1], &[1]).unwrap_err();
        assert!(err.to_string().contains("unchanged_pixels=0"), "{err:#}");

        // No changed content: fail closed (v1 behaviour preserved).
        let only_unchanged = [FlipRow {
            content_mask: &[1],
            current: &[3],
            next: &[3],
        }];
        let err = flip_metrics_rows(&only_unchanged, 1, &[3], &[3], &[3], &[3]).unwrap_err();
        assert!(err.to_string().contains("changed_pixels=0"), "{err:#}");

        // Prediction length mismatch is rejected.
        assert!(flip_metrics_rows(&rows, 3, &[1, 2], &[1, 2, 9], &[1, 2, 0], &[1, 2, 9]).is_err());
        Ok(())
    }

    #[test]
    fn identity_root_binds_each_leaf_and_ignores_paths() {
        let leaves = Bf16DriftIdentityLeaves {
            checkpoint_sha256: "sha256:aa",
            train_config_sha256: "sha256:bb",
            evaluator_binary_sha256: "sha256:cc",
            evaluator_package_version: "0.1.0",
            falsifier_config_sha256: "sha256:dd",
            population_rows: 4,
            population_sha256: "sha256:ee",
        };
        let root = bf16_drift_identity_root(&leaves);
        assert!(root.starts_with("sha256:"));
        assert_eq!(root, bf16_drift_identity_root(&leaves.clone()));

        type LeafMutation = (
            &'static str,
            Box<dyn Fn(&mut Bf16DriftIdentityLeaves<'static>)>,
        );
        let mutations: [LeafMutation; 7] = [
            (
                "checkpoint",
                Box::new(|l| l.checkpoint_sha256 = "sha256:a0"),
            ),
            (
                "train_config",
                Box::new(|l| l.train_config_sha256 = "sha256:b0"),
            ),
            (
                "binary",
                Box::new(|l| l.evaluator_binary_sha256 = "sha256:c0"),
            ),
            (
                "version",
                Box::new(|l| l.evaluator_package_version = "0.2.0"),
            ),
            (
                "falsifier_config",
                Box::new(|l| l.falsifier_config_sha256 = "sha256:d0"),
            ),
            ("rows", Box::new(|l| l.population_rows = 5)),
            (
                "population",
                Box::new(|l| l.population_sha256 = "sha256:e0"),
            ),
        ];
        for (name, mutate) in &mutations {
            let mut mutated = leaves.clone();
            mutate(&mut mutated);
            assert_ne!(bf16_drift_identity_root(&mutated), root, "{name}");
        }

        // Framing: moving bytes between adjacent leaves must not collide.
        let shifted = Bf16DriftIdentityLeaves {
            checkpoint_sha256: "sha256:a",
            train_config_sha256: "asha256:bb",
            ..leaves.clone()
        };
        assert_ne!(bf16_drift_identity_root(&shifted), root);

        // The root is a pure function of the leaves; paths and argv live in
        // the descriptive identity fields only, so relocating the same bytes
        // (different `checkpoint`/`train_config`/`command`) keeps the root.
        let relocated = Bf16DriftIdentity {
            domain: BF16_DRIFT_IDENTITY_DOMAIN.into(),
            checkpoint_sha256: leaves.checkpoint_sha256.into(),
            train_config_sha256: leaves.train_config_sha256.into(),
            evaluator_binary: PathBuf::from("/elsewhere/tofy"),
            evaluator_binary_sha256: leaves.evaluator_binary_sha256.into(),
            evaluator_package_version: leaves.evaluator_package_version.into(),
            command: vec!["tofy".into(), "--output".into(), "/other/drift.json".into()],
            command_sha256: "sha256:ff".into(),
            falsifier_config_sha256: leaves.falsifier_config_sha256.into(),
            population_rows: leaves.population_rows,
            population_sha256: leaves.population_sha256.into(),
            identity_root: root.clone(),
        };
        assert_eq!(
            bf16_drift_identity_root(&Bf16DriftIdentityLeaves {
                checkpoint_sha256: &relocated.checkpoint_sha256,
                train_config_sha256: &relocated.train_config_sha256,
                evaluator_binary_sha256: &relocated.evaluator_binary_sha256,
                evaluator_package_version: &relocated.evaluator_package_version,
                falsifier_config_sha256: &relocated.falsifier_config_sha256,
                population_rows: relocated.population_rows,
                population_sha256: &relocated.population_sha256,
            }),
            relocated.identity_root
        );
    }

    #[test]
    fn bytes_sha256_is_domain_separated_and_framed() {
        assert_ne!(bytes_sha256("a", b"b"), bytes_sha256("", b"ab"));
        assert_ne!(bytes_sha256("ab", b""), bytes_sha256("a", b"b"));
        assert_eq!(bytes_sha256("x", b"y"), bytes_sha256("x", b"y"));
        assert_eq!(
            raw_bytes_sha256(b"config bytes"),
            format!("sha256:{:x}", Sha256::digest(b"config bytes"))
        );
    }

    #[test]
    fn bf16_falsifier_rejects_v6_configs_until_a_v6_population_is_registered() {
        let mut cfg = TrainConfig {
            recipe: crate::p2::experiment::TrainingRecipe::FoundationV2,
            ..TrainConfig::default()
        };
        assert!(ensure_baseline_config(&cfg).is_ok());

        cfg.world_core_v6 = true;
        let err = ensure_baseline_config(&cfg).unwrap_err();
        assert!(err.to_string().contains("Foundation-v2/V5"), "{err:#}");

        cfg.world_core_v6 = false;
        cfg.data_contract_v6 = true;
        let err = ensure_baseline_config(&cfg).unwrap_err();
        assert!(err.to_string().contains("Foundation-v2/V5"), "{err:#}");
    }

    #[test]
    fn json_report_sidecar_matches_written_bytes() -> Result<()> {
        let dir = std::env::temp_dir().join(format!(
            "tofy-bf16-falsifier-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let path = dir.join("nested").join("drift.json");
        let result = (|| -> Result<()> {
            write_json_report(
                &path,
                &serde_json::json!({"schema": BF16_DRIFT_SCHEMA, "n": 1}),
            )?;
            let bytes = fs::read(&path)?;
            assert!(bytes.ends_with(b"\n"));
            let sidecar = fs::read_to_string(report_sha256_sidecar_path(&path))?;
            let expected = format!("{:x}  drift.json\n", Sha256::digest(&bytes));
            assert_eq!(sidecar, expected);
            assert!(!path.with_extension("json.tmp").exists());
            Ok(())
        })();
        let _ = fs::remove_dir_all(&dir);
        result
    }

    #[test]
    fn v1_report_field_names_are_preserved_in_v2() -> Result<()> {
        let report = Bf16DriftReport {
            schema: BF16_DRIFT_SCHEMA.into(),
            checkpoint: PathBuf::from("ckpt"),
            train_config: PathBuf::from("cfg"),
            device: "cpu".into(),
            seed: 1,
            batch_size: 2,
            latent_elements: 0,
            latent_max_abs_drift: 0.0,
            logit_elements: 0,
            logit_max_abs_drift: 0.0,
            changed_pixels: 1,
            changed_pixel_prediction_flips: 0,
            changed_pixel_prediction_flip_rate: 0.0,
            content_pixels: 2,
            composed_decode_flips: 0,
            composed_decode_flip_rate: 0.0,
            f32_rollout_loss: 0.0,
            bf16_rollout_loss: 0.0,
            f32_rollout_fragments: 0,
            bf16_rollout_fragments: 0,
            unchanged_pixels: 1,
            unchanged_pixel_prediction_flips: 1,
            unchanged_pixel_prediction_flip_rate: 1.0,
            f32_unchanged_raw_false_edits: 0,
            bf16_unchanged_raw_false_edits: 1,
            raw_false_edits_introduced_by_bf16: 1,
            raw_false_edits_resolved_by_bf16: 0,
            f32_changed_pixel_raw_correct: 1,
            bf16_changed_pixel_raw_correct: 1,
            identity: Bf16DriftIdentity {
                domain: BF16_DRIFT_IDENTITY_DOMAIN.into(),
                checkpoint_sha256: "sha256:aa".into(),
                train_config_sha256: "sha256:bb".into(),
                evaluator_binary: PathBuf::from("tofy"),
                evaluator_binary_sha256: "sha256:cc".into(),
                evaluator_package_version: "0.1.0".into(),
                command: vec![],
                command_sha256: "sha256:dd".into(),
                falsifier_config_sha256: "sha256:ee".into(),
                population_rows: 1,
                population_sha256: "sha256:ff".into(),
                identity_root: "sha256:00".into(),
            },
        };
        let value = serde_json::to_value(&report)?;
        for key in [
            "schema",
            "checkpoint",
            "train_config",
            "device",
            "seed",
            "batch_size",
            "latent_elements",
            "latent_max_abs_drift",
            "logit_elements",
            "logit_max_abs_drift",
            "changed_pixels",
            "changed_pixel_prediction_flips",
            "changed_pixel_prediction_flip_rate",
            "content_pixels",
            "composed_decode_flips",
            "composed_decode_flip_rate",
            "f32_rollout_loss",
            "bf16_rollout_loss",
            "f32_rollout_fragments",
            "bf16_rollout_fragments",
        ] {
            assert!(
                value.get(key).is_some(),
                "v1 field {key} missing from v2 report"
            );
        }
        assert_eq!(value["schema"], "p2.bf16_recurrent_core_drift.v2");
        assert_eq!(value["identity"]["identity_root"], "sha256:00");
        Ok(())
    }
}
