//! Frozen semantic-access audit for spatial checkpoint representations.
//!
//! Model weights are loaded only for feature extraction. Decoder selection is
//! episode-disjoint and the outer-final partition is scored once after refit.

use crate::gpu_lock::GpuSessionGuard;
use crate::p2::board_probe::{
    histograms_for_frames, FixedBoardProbe, FIXED_TARGET_DECODER_MSE_CEILING, PALETTE_SIZE,
    PATCHES_PER_SIDE, PATCH_COUNT,
};
use crate::p2::eval::collect_frozen_board_probe_population;
use crate::p2::train::{load_train_config, reinit_varmap_deterministic, resolve_device};
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const SEMANTIC_ACCESS_SCHEMA: &str = "p2.semantic_access.v1";

#[derive(Debug, Clone)]
pub struct SemanticAccessConfig {
    pub checkpoint: PathBuf,
    pub train_config: PathBuf,
    pub seed: u64,
    pub synthetic_episodes: usize,
    pub physical_batch: usize,
    pub device: String,
    pub hidden_dim: usize,
    pub max_epochs: usize,
    pub patience: usize,
    pub decoder_batch: usize,
    pub permutation_seeds: Vec<u64>,
    pub output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitManifest {
    pub train_episodes: usize,
    pub selection_episodes: usize,
    pub final_episodes: usize,
    pub train_rows: usize,
    pub selection_rows: usize,
    pub final_rows: usize,
    pub rule: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderScore {
    pub name: String,
    pub selection_mse: Option<f64>,
    pub final_mse: f64,
    pub selected_epochs: Option<usize>,
    pub parameter_count: usize,
    pub permutation_final_mse: Vec<f64>,
    pub permutation_p_value: Option<f64>,
    pub beats_every_permutation: Option<bool>,
    pub passes_fixed_ceiling: bool,
    pub trusted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlScores {
    pub observable_positive_mse: f64,
    pub global_marginal_mse: f64,
    pub position_source_marginal_mse: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAccessReport {
    pub schema: String,
    pub checkpoint: PathBuf,
    pub checkpoint_sha256: String,
    pub train_config: PathBuf,
    pub train_config_sha256: String,
    pub population_seed: u64,
    pub synthetic_episodes_per_source: usize,
    pub population_fingerprint: String,
    pub target: String,
    pub protocol: SemanticAccessProtocol,
    pub split: SplitManifest,
    pub controls: ControlScores,
    pub decoders: Vec<DecoderScore>,
    pub any_bounded_decoder_trusted: bool,
    pub real_target_final_scores_per_decoder: usize,
    pub final_partition_used_for_decoder_selection: bool,
    pub model_weights_updated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAccessProtocol {
    pub device: String,
    pub physical_batch: usize,
    pub decoder_hidden: usize,
    pub decoder_max_epochs: usize,
    pub decoder_patience: usize,
    pub decoder_batch: usize,
    pub permutation_seeds: Vec<u64>,
    pub permutation_unit: String,
    pub permutation_movable_row_fraction: f64,
    pub singleton_episode_strata: usize,
    pub parameter_cap: usize,
    pub trust_hierarchy: String,
}

#[derive(Clone)]
struct AuditRows {
    local: Vec<Vec<f32>>,
    contextual: Vec<Vec<f32>>,
    targets: Vec<[f32; PALETTE_SIZE]>,
    source: Vec<String>,
    episode: Vec<u64>,
    patch: Vec<usize>,
}

#[derive(Default)]
struct Partitions {
    train_frames: Vec<usize>,
    selection_frames: Vec<usize>,
    final_frames: Vec<usize>,
}

struct ProbeMlp {
    first: Linear,
    second: Linear,
}

struct StandardizedRows {
    rows: Vec<Vec<f32>>,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl ProbeMlp {
    fn new(input: usize, hidden: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            first: linear(input, hidden, vb.pp("first"))?,
            second: linear(hidden, PALETTE_SIZE, vb.pp("second"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.second
            .forward(&self.first.forward(input)?.relu()?)
            .map_err(Into::into)
    }
}

pub fn run_semantic_access_audit(cfg: &SemanticAccessConfig) -> Result<SemanticAccessReport> {
    validate_config(cfg)?;
    ensure!(
        !cfg.output.exists(),
        "audit output already exists: {}",
        cfg.output.display()
    );
    let checkpoint_sha256 = sha256_file(&cfg.checkpoint)?;
    let train_config_sha256 = sha256_file(&cfg.train_config)?;
    let train_cfg = load_train_config(&cfg.train_config)?;
    let _gpu_guard = if cfg.device == "cuda" || cfg.device.starts_with("cuda:") {
        Some(GpuSessionGuard::acquire(&train_cfg.output_dir)?)
    } else {
        None
    };
    let population = collect_frozen_board_probe_population(
        &cfg.checkpoint,
        &cfg.train_config,
        cfg.seed,
        cfg.synthetic_episodes,
        cfg.physical_batch,
        &cfg.device,
    )?;
    ensure!(
        population.samples.len() == population.source_by_sample.len(),
        "source labels do not align with frames"
    );
    let local = population.target_rows.as_rows().to_vec();
    ensure!(
        local.len() == population.samples.len() * PATCH_COUNT,
        "latent rows do not align with frames"
    );
    let contextual = contextual_features(&local)?;
    let targets = histograms_for_frames(
        &population
            .samples
            .iter()
            .map(|sample| sample.next.clone())
            .collect::<Vec<_>>(),
    )?;
    let mut source = Vec::with_capacity(local.len());
    let mut episode = Vec::with_capacity(local.len());
    let mut patch = Vec::with_capacity(local.len());
    for (frame_index, sample) in population.samples.iter().enumerate() {
        for patch_index in 0..PATCH_COUNT {
            source.push(population.source_by_sample[frame_index].clone());
            episode.push(sample.episode_id);
            patch.push(patch_index);
        }
    }
    let rows = AuditRows {
        local,
        contextual,
        targets,
        source,
        episode,
        patch,
    };
    let partitions = partitions(&population.samples);
    validate_partitions(&partitions)?;
    let train_rows = frame_rows(&partitions.train_frames);
    let selection_rows = frame_rows(&partitions.selection_frames);
    let final_rows = frame_rows(&partitions.final_frames);
    let fit_rows = train_rows
        .iter()
        .chain(&selection_rows)
        .copied()
        .collect::<Vec<_>>();
    let device = resolve_device(&cfg.device)?;
    let permutation_coverage = permutation_coverage(&rows);
    ensure!(
        permutation_coverage.movable_row_fraction >= 0.95,
        "only {:.3} of rows belong to movable episode strata",
        permutation_coverage.movable_row_fraction
    );

    let train_latents = select_rows(&rows.local, &train_rows);
    let train_frames = partitions
        .train_frames
        .iter()
        .map(|&i| population.samples[i].next.clone())
        .collect::<Vec<_>>();
    let ridge_selection_model = FixedBoardProbe::fit(&train_latents, &train_frames)?;
    let ridge_selection = mse(
        &ridge_selection_model.predict_histograms(&select_rows(&rows.local, &selection_rows))?,
        &select_targets(&rows.targets, &selection_rows),
    )?;
    let fit_latents = select_rows(&rows.local, &fit_rows);
    let fit_frames = partitions
        .train_frames
        .iter()
        .chain(&partitions.selection_frames)
        .map(|&i| population.samples[i].next.clone())
        .collect::<Vec<_>>();
    let ridge = FixedBoardProbe::fit(&fit_latents, &fit_frames)?;
    let ridge_final = mse(
        &ridge.predict_histograms(&select_rows(&rows.local, &final_rows))?,
        &select_targets(&rows.targets, &final_rows),
    )?;
    let mut decoders = vec![DecoderScore {
        name: "local_ridge_fixed_lambda_1e-2".into(),
        selection_mse: Some(ridge_selection),
        final_mse: ridge_final,
        selected_epochs: None,
        parameter_count: ridge.input_dim * PALETTE_SIZE + ridge.input_dim * 2 + PALETTE_SIZE,
        permutation_final_mse: Vec::new(),
        permutation_p_value: None,
        beats_every_permutation: None,
        passes_fixed_ceiling: ridge_final <= FIXED_TARGET_DECODER_MSE_CEILING,
        // Ridge remains the registered descriptive comparator. Inferential
        // trust is reserved for the preregistered MLP hierarchy with nulls.
        trusted: false,
    }];

    for (name, features) in [
        ("local_mlp", &rows.local),
        ("contextual_3x3_global_mlp", &rows.contextual),
    ] {
        let real = select_and_refit_mlp(
            features,
            &rows.targets,
            &rows.targets,
            &train_rows,
            &selection_rows,
            &fit_rows,
            &final_rows,
            cfg,
            &device,
            stable_seed(cfg.seed, name),
        )?;
        let mut permutation_final_mse = Vec::with_capacity(cfg.permutation_seeds.len());
        for &permutation_seed in &cfg.permutation_seeds {
            // A valid restricted randomization null permutes the entire frozen
            // population, then repeats selection and final scoring against the
            // permuted targets. The final partition still never selects epochs.
            let permuted = permute_targets(&rows, permutation_seed)?;
            let score = select_and_refit_mlp(
                features,
                &permuted,
                &permuted,
                &train_rows,
                &selection_rows,
                &fit_rows,
                &final_rows,
                cfg,
                &device,
                stable_seed(cfg.seed, name),
            )?;
            permutation_final_mse.push(score.final_mse);
        }
        let worse_or_equal = permutation_final_mse
            .iter()
            .filter(|&&value| value <= real.final_mse)
            .count();
        let p_value = (worse_or_equal + 1) as f64 / (permutation_final_mse.len() + 1) as f64;
        let beats_all = permutation_final_mse
            .iter()
            .all(|&value| real.final_mse < value);
        let passes = real.final_mse <= FIXED_TARGET_DECODER_MSE_CEILING;
        decoders.push(DecoderScore {
            name: name.into(),
            selection_mse: Some(real.selection_mse),
            final_mse: real.final_mse,
            selected_epochs: Some(real.epochs),
            parameter_count: mlp_parameter_count(features[0].len(), cfg.hidden_dim),
            permutation_final_mse,
            permutation_p_value: Some(p_value),
            beats_every_permutation: Some(beats_all),
            passes_fixed_ceiling: passes,
            trusted: false,
        });
    }

    let controls = controls(&rows, &fit_rows, &final_rows)?;
    ensure!(
        controls.observable_positive_mse <= FIXED_TARGET_DECODER_MSE_CEILING,
        "observable positive control failed"
    );
    // Two confirmatory decoder families split alpha equally (Bonferroni).
    let marginal_ceiling = controls
        .global_marginal_mse
        .min(controls.position_source_marginal_mse);
    decoders[1].trusted = decoder_inferential_pass(&decoders[1], marginal_ceiling);
    decoders[2].trusted = decoder_inferential_pass(&decoders[2], marginal_ceiling);
    ensure!(
        sha256_file(&cfg.checkpoint)? == checkpoint_sha256
            && sha256_file(&cfg.train_config)? == train_config_sha256,
        "checkpoint or training config changed during audit"
    );
    let report = SemanticAccessReport {
        schema: SEMANTIC_ACCESS_SCHEMA.into(),
        checkpoint: cfg.checkpoint.clone(),
        checkpoint_sha256,
        train_config: cfg.train_config.clone(),
        train_config_sha256,
        population_seed: cfg.seed,
        synthetic_episodes_per_source: cfg.synthetic_episodes,
        population_fingerprint: population.population_fingerprint,
        target: "16_colour_pixel_counts_per_8x8_patch_status_row_excluded".into(),
        protocol: SemanticAccessProtocol {
            device: cfg.device.clone(),
            physical_batch: cfg.physical_batch,
            decoder_hidden: cfg.hidden_dim,
            decoder_max_epochs: cfg.max_epochs,
            decoder_patience: cfg.patience,
            decoder_batch: cfg.decoder_batch,
            permutation_seeds: cfg.permutation_seeds.clone(),
            permutation_unit: "whole_episode_derangement_within_generator_source_length_and_frozen_split_preserving_transition_order_and_patch_site; singleton_strata_fixed; movable_rows_at_least_95_percent".into(),
            permutation_movable_row_fraction: permutation_coverage.movable_row_fraction,
            singleton_episode_strata: permutation_coverage.singleton_strata,
            parameter_cap: 100_000,
            trust_hierarchy: "two_confirmatory_decoder_families_bonferroni_alpha_0.025_each; fixed_ceiling, beats_all_nulls_and_both_marginals".into(),
        },
        split: SplitManifest {
            train_episodes: unique_episodes(&rows, &train_rows),
            selection_episodes: unique_episodes(&rows, &selection_rows),
            final_episodes: unique_episodes(&rows, &final_rows),
            train_rows: train_rows.len(),
            selection_rows: selection_rows.len(),
            final_rows: final_rows.len(),
            rule: "outer_final: episode_id%3!=0; development: episode_id%3==0; selection: episode_id%9==0; train: episode_id%9 in {3,6}".into(),
        },
        controls,
        any_bounded_decoder_trusted: decoders.iter().any(|decoder| decoder.trusted),
        decoders,
        real_target_final_scores_per_decoder: 1,
        final_partition_used_for_decoder_selection: false,
        model_weights_updated: false,
    };
    write_json_create_new(&cfg.output, &report)?;
    Ok(report)
}

fn decoder_inferential_pass(decoder: &DecoderScore, marginal_ceiling: f64) -> bool {
    decoder.passes_fixed_ceiling
        && decoder.beats_every_permutation == Some(true)
        && decoder
            .permutation_p_value
            .is_some_and(|value| value <= 0.025)
        && decoder.final_mse < marginal_ceiling
}

fn validate_config(cfg: &SemanticAccessConfig) -> Result<()> {
    ensure!(
        cfg.checkpoint.is_file(),
        "missing checkpoint {}",
        cfg.checkpoint.display()
    );
    ensure!(
        cfg.train_config.is_file(),
        "missing train config {}",
        cfg.train_config.display()
    );
    ensure!(
        cfg.synthetic_episodes >= 9,
        "audit needs at least nine episodes per source"
    );
    ensure!(
        cfg.physical_batch > 0 && cfg.decoder_batch > 0,
        "batch sizes must be positive"
    );
    ensure!(
        cfg.hidden_dim > 0 && cfg.max_epochs > 0 && cfg.patience > 0,
        "decoder budget must be positive"
    );
    ensure!(
        cfg.permutation_seeds.len() >= 39,
        "at least 39 permutations are required for Bonferroni p<=0.025 resolution"
    );
    ensure!(
        cfg.permutation_seeds
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            == cfg.permutation_seeds.len(),
        "permutation seeds must be unique"
    );
    Ok(())
}

fn contextual_features(local: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    ensure!(
        !local.is_empty() && local.len().is_multiple_of(PATCH_COUNT),
        "invalid local grid rows"
    );
    let channels = local[0].len();
    ensure!(
        channels > 0 && local.iter().all(|row| row.len() == channels),
        "inconsistent local channels"
    );
    let mut output = Vec::with_capacity(local.len());
    for frame_start in (0..local.len()).step_by(PATCH_COUNT) {
        let mut global = vec![0.0f32; channels];
        for row in &local[frame_start..frame_start + PATCH_COUNT] {
            for (sum, value) in global.iter_mut().zip(row) {
                *sum += *value / PATCH_COUNT as f32;
            }
        }
        for patch_index in 0..PATCH_COUNT {
            let y = patch_index / PATCHES_PER_SIDE;
            let x = patch_index % PATCHES_PER_SIDE;
            let mut neighborhood = vec![0.0f32; channels];
            let mut count = 0f32;
            for ny in y.saturating_sub(1)..=(y + 1).min(PATCHES_PER_SIDE - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(PATCHES_PER_SIDE - 1) {
                    for (sum, value) in neighborhood
                        .iter_mut()
                        .zip(&local[frame_start + ny * PATCHES_PER_SIDE + nx])
                    {
                        *sum += *value;
                    }
                    count += 1.0;
                }
            }
            for value in &mut neighborhood {
                *value /= count;
            }
            let mut row = Vec::with_capacity(channels * 3);
            row.extend_from_slice(&local[frame_start + patch_index]);
            row.extend(neighborhood);
            row.extend_from_slice(&global);
            output.push(row);
        }
    }
    Ok(output)
}

fn partitions(samples: &[crate::p2::data::TransitionSample]) -> Partitions {
    let mut result = Partitions::default();
    for (index, sample) in samples.iter().enumerate() {
        if sample.episode_id % 3 != 0 {
            result.final_frames.push(index);
        } else if sample.episode_id % 9 == 0 {
            result.selection_frames.push(index);
        } else {
            result.train_frames.push(index);
        }
    }
    result
}

fn validate_partitions(parts: &Partitions) -> Result<()> {
    ensure!(
        !parts.train_frames.is_empty()
            && !parts.selection_frames.is_empty()
            && !parts.final_frames.is_empty(),
        "episode split contains an empty partition"
    );
    let mut all = BTreeSet::new();
    for index in parts
        .train_frames
        .iter()
        .chain(&parts.selection_frames)
        .chain(&parts.final_frames)
    {
        ensure!(all.insert(*index), "frame leaked across audit partitions");
    }
    Ok(())
}

fn frame_rows(frames: &[usize]) -> Vec<usize> {
    frames
        .iter()
        .flat_map(|&frame| frame * PATCH_COUNT..(frame + 1) * PATCH_COUNT)
        .collect()
}

fn select_rows(rows: &[Vec<f32>], indices: &[usize]) -> Vec<Vec<f32>> {
    indices.iter().map(|&index| rows[index].clone()).collect()
}

fn select_targets(rows: &[[f32; PALETTE_SIZE]], indices: &[usize]) -> Vec<[f32; PALETTE_SIZE]> {
    indices.iter().map(|&index| rows[index]).collect()
}

fn mse(predictions: &[[f32; PALETTE_SIZE]], targets: &[[f32; PALETTE_SIZE]]) -> Result<f64> {
    ensure!(
        !predictions.is_empty() && predictions.len() == targets.len(),
        "incompatible score rows"
    );
    Ok(predictions
        .iter()
        .zip(targets)
        .flat_map(|(p, t)| p.iter().zip(t))
        .map(|(p, t)| {
            let d = f64::from(*p - *t);
            d * d
        })
        .sum::<f64>()
        / (predictions.len() * PALETTE_SIZE) as f64)
}

struct MlpScore {
    selection_mse: f64,
    final_mse: f64,
    epochs: usize,
}

#[allow(clippy::too_many_arguments)]
fn select_and_refit_mlp(
    features: &[Vec<f32>],
    true_targets: &[[f32; PALETTE_SIZE]],
    fit_targets: &[[f32; PALETTE_SIZE]],
    train: &[usize],
    selection: &[usize],
    fit: &[usize],
    final_rows: &[usize],
    cfg: &SemanticAccessConfig,
    device: &Device,
    seed: u64,
) -> Result<MlpScore> {
    let train_standardized = standardized(features, train)?;
    let selection_x = apply_standardization(
        features,
        selection,
        &train_standardized.mean,
        &train_standardized.std,
    );
    let train_y = flattened_targets(fit_targets, train);
    let selection_y = flattened_targets(fit_targets, selection);
    let (best_epoch, selection_mse) = select_epochs(
        &train_standardized.rows,
        &train_y,
        &selection_x,
        &selection_y,
        cfg,
        device,
        seed,
    )?;
    let fit_standardized = standardized(features, fit)?;
    let final_x = apply_standardization(
        features,
        final_rows,
        &fit_standardized.mean,
        &fit_standardized.std,
    );
    let fit_y = flattened_targets(fit_targets, fit);
    let final_y = flattened_targets(true_targets, final_rows);
    let final_mse = refit_and_score(
        &fit_standardized.rows,
        &fit_y,
        &final_x,
        &final_y,
        cfg,
        device,
        seed,
        best_epoch,
    )?;
    Ok(MlpScore {
        selection_mse,
        final_mse,
        epochs: best_epoch,
    })
}

fn standardized(features: &[Vec<f32>], indices: &[usize]) -> Result<StandardizedRows> {
    ensure!(!indices.is_empty(), "empty standardization population");
    let dim = features[0].len();
    let mut mean = vec![0f64; dim];
    for &index in indices {
        for (j, &value) in features[index].iter().enumerate() {
            mean[j] += value as f64;
        }
    }
    for value in &mut mean {
        *value /= indices.len() as f64;
    }
    let mut std = vec![0f64; dim];
    for &index in indices {
        for (j, &value) in features[index].iter().enumerate() {
            let d = value as f64 - mean[j];
            std[j] += d * d;
        }
    }
    for value in &mut std {
        *value = (*value / indices.len() as f64).sqrt().max(1e-6);
    }
    let mean = mean.into_iter().map(|v| v as f32).collect::<Vec<_>>();
    let std = std.into_iter().map(|v| v as f32).collect::<Vec<_>>();
    Ok(StandardizedRows {
        rows: apply_standardization(features, indices, &mean, &std),
        mean,
        std,
    })
}

fn apply_standardization(
    features: &[Vec<f32>],
    indices: &[usize],
    mean: &[f32],
    std: &[f32],
) -> Vec<Vec<f32>> {
    indices
        .iter()
        .map(|&index| {
            features[index]
                .iter()
                .enumerate()
                .map(|(j, value)| (*value - mean[j]) / std[j])
                .collect()
        })
        .collect()
}

fn flattened_targets(targets: &[[f32; PALETTE_SIZE]], indices: &[usize]) -> Vec<f32> {
    indices.iter().flat_map(|&index| targets[index]).collect()
}

fn select_epochs(
    train_x: &[Vec<f32>],
    train_y: &[f32],
    val_x: &[Vec<f32>],
    val_y: &[f32],
    cfg: &SemanticAccessConfig,
    device: &Device,
    seed: u64,
) -> Result<(usize, f64)> {
    let (model, varmap, mut optimizer) = new_mlp(train_x[0].len(), cfg, device, seed)?;
    let mut best_epoch = 1;
    let mut best = f64::INFINITY;
    let mut stale = 0usize;
    for epoch in 1..=cfg.max_epochs {
        train_epoch(
            &model,
            &mut optimizer,
            train_x,
            train_y,
            cfg.decoder_batch,
            device,
            stable_seed(seed, &format!("epoch-{epoch}")),
        )?;
        let score = score_mlp(&model, val_x, val_y, cfg.decoder_batch, device)?;
        if score + 1e-12 < best {
            best = score;
            best_epoch = epoch;
            stale = 0;
        } else {
            stale += 1;
        }
        if stale >= cfg.patience {
            break;
        }
    }
    drop(varmap);
    Ok((best_epoch, best))
}

#[allow(clippy::too_many_arguments)]
fn refit_and_score(
    train_x: &[Vec<f32>],
    train_y: &[f32],
    test_x: &[Vec<f32>],
    test_y: &[f32],
    cfg: &SemanticAccessConfig,
    device: &Device,
    seed: u64,
    epochs: usize,
) -> Result<f64> {
    let (model, _varmap, mut optimizer) = new_mlp(train_x[0].len(), cfg, device, seed)?;
    for epoch in 1..=epochs {
        train_epoch(
            &model,
            &mut optimizer,
            train_x,
            train_y,
            cfg.decoder_batch,
            device,
            stable_seed(seed, &format!("refit-{epoch}")),
        )?;
    }
    score_mlp(&model, test_x, test_y, cfg.decoder_batch, device)
}

fn new_mlp(
    input: usize,
    cfg: &SemanticAccessConfig,
    device: &Device,
    seed: u64,
) -> Result<(ProbeMlp, VarMap, AdamW)> {
    ensure!(
        mlp_parameter_count(input, cfg.hidden_dim) <= 100_000,
        "decoder exceeds 100k parameter cap"
    );
    let varmap = VarMap::new();
    let model = ProbeMlp::new(
        input,
        cfg.hidden_dim,
        VarBuilder::from_varmap(&varmap, DType::F32, device),
    )?;
    reinit_varmap_deterministic(&varmap, seed)?;
    let optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 1e-3,
            weight_decay: 1e-4,
            ..ParamsAdamW::default()
        },
    )?;
    Ok((model, varmap, optimizer))
}

fn train_epoch(
    model: &ProbeMlp,
    optimizer: &mut AdamW,
    features: &[Vec<f32>],
    targets: &[f32],
    batch: usize,
    device: &Device,
    seed: u64,
) -> Result<()> {
    let mut order = (0..features.len()).collect::<Vec<_>>();
    order.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    for indices in order.chunks(batch) {
        let x = Tensor::from_vec(
            indices
                .iter()
                .flat_map(|&i| features[i].iter().copied())
                .collect::<Vec<_>>(),
            (indices.len(), features[0].len()),
            device,
        )?;
        let y = Tensor::from_vec(
            indices
                .iter()
                .flat_map(|&i| {
                    targets[i * PALETTE_SIZE..(i + 1) * PALETTE_SIZE]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>(),
            (indices.len(), PALETTE_SIZE),
            device,
        )?;
        let loss = candle_nn::loss::mse(&model.forward(&x)?, &y)?;
        optimizer.backward_step(&loss)?;
    }
    Ok(())
}

fn score_mlp(
    model: &ProbeMlp,
    features: &[Vec<f32>],
    targets: &[f32],
    batch: usize,
    device: &Device,
) -> Result<f64> {
    let mut sum = 0f64;
    let mut count = 0usize;
    for start in (0..features.len()).step_by(batch) {
        let end = (start + batch).min(features.len());
        let x = Tensor::from_vec(
            features[start..end]
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            (end - start, features[0].len()),
            device,
        )?;
        let predicted = model
            .forward(&x)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (p, t) in predicted
            .iter()
            .zip(&targets[start * PALETTE_SIZE..end * PALETTE_SIZE])
        {
            let d = f64::from(*p - *t);
            sum += d * d;
            count += 1;
        }
    }
    ensure!(count > 0 && sum.is_finite(), "non-finite decoder score");
    Ok(sum / count as f64)
}

type EpisodePermutationGroups = BTreeMap<(String, usize, u8), Vec<Vec<usize>>>;

struct PermutationCoverage {
    movable_row_fraction: f64,
    singleton_strata: usize,
}

fn episode_permutation_groups(rows: &AuditRows) -> EpisodePermutationGroups {
    let mut episodes: BTreeMap<(String, u64), Vec<usize>> = BTreeMap::new();
    for frame in 0..rows.targets.len() / PATCH_COUNT {
        let index = frame * PATCH_COUNT;
        episodes
            .entry((rows.source[index].clone(), rows.episode[index]))
            .or_default()
            .push(frame);
    }
    let mut groups: BTreeMap<(String, usize, u8), Vec<Vec<usize>>> = BTreeMap::new();
    for ((source, episode), frames) in episodes {
        groups
            .entry((source, frames.len(), split_class(episode)))
            .or_default()
            .push(frames);
    }
    groups
}

fn permutation_coverage(rows: &AuditRows) -> PermutationCoverage {
    let groups = episode_permutation_groups(rows);
    let movable_frames = groups
        .values()
        .filter(|episodes| episodes.len() >= 2)
        .flat_map(|episodes| episodes.iter())
        .map(Vec::len)
        .sum::<usize>();
    let total_frames = rows.targets.len() / PATCH_COUNT;
    PermutationCoverage {
        movable_row_fraction: movable_frames as f64 / total_frames.max(1) as f64,
        singleton_strata: groups
            .values()
            .filter(|episodes| episodes.len() == 1)
            .count(),
    }
}

fn permute_targets(rows: &AuditRows, seed: u64) -> Result<Vec<[f32; PALETTE_SIZE]>> {
    let mut output = rows.targets.clone();
    let groups = episode_permutation_groups(rows);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for episode_sequences in groups.values() {
        if episode_sequences.len() == 1 {
            continue;
        }
        let mut shuffled = episode_sequences.clone();
        for _ in 0..64 {
            shuffled.shuffle(&mut rng);
            if episode_sequences
                .iter()
                .zip(&shuffled)
                .all(|(a, b)| a[0] != b[0])
            {
                break;
            }
        }
        if episode_sequences
            .iter()
            .zip(&shuffled)
            .any(|(a, b)| a[0] == b[0])
        {
            shuffled.clone_from(episode_sequences);
            shuffled.rotate_left(1);
        }
        ensure!(
            episode_sequences
                .iter()
                .zip(&shuffled)
                .all(|(a, b)| a[0] != b[0]),
            "failed to construct episode derangement"
        );
        for (destination_episode, source_episode) in episode_sequences.iter().zip(&shuffled) {
            for (&destination_frame, &source_frame) in
                destination_episode.iter().zip(source_episode)
            {
                for patch in 0..PATCH_COUNT {
                    output[destination_frame * PATCH_COUNT + patch] =
                        rows.targets[source_frame * PATCH_COUNT + patch];
                }
            }
        }
    }
    Ok(output)
}

fn split_class(episode_id: u64) -> u8 {
    if !episode_id.is_multiple_of(3) {
        2
    } else if episode_id.is_multiple_of(9) {
        1
    } else {
        0
    }
}

fn controls(rows: &AuditRows, fit: &[usize], final_rows: &[usize]) -> Result<ControlScores> {
    let final_targets = select_targets(&rows.targets, final_rows);
    let fit_targets = select_targets(&rows.targets, fit);
    let observable_fit = fit_targets
        .iter()
        .map(|target| target.to_vec())
        .collect::<Vec<_>>();
    let observable_final = final_targets
        .iter()
        .map(|target| target.to_vec())
        .collect::<Vec<_>>();
    let observable_probe = FixedBoardProbe::fit_histograms(&observable_fit, &fit_targets)?;
    let positive = mse(
        &observable_probe.predict_histograms(&observable_final)?,
        &final_targets,
    )?;
    let mut global = [0f32; PALETTE_SIZE];
    for &index in fit {
        for (colour, value) in global.iter_mut().enumerate() {
            *value += rows.targets[index][colour] / fit.len() as f32;
        }
    }
    let global_predictions = vec![global; final_rows.len()];
    let global_marginal_mse = mse(&global_predictions, &final_targets)?;
    let mut sums: BTreeMap<(String, usize), ([f64; PALETTE_SIZE], usize)> = BTreeMap::new();
    for &index in fit {
        let entry = sums
            .entry((rows.source[index].clone(), rows.patch[index]))
            .or_insert(([0f64; PALETTE_SIZE], 0));
        for c in 0..PALETTE_SIZE {
            entry.0[c] += rows.targets[index][c] as f64;
        }
        entry.1 += 1;
    }
    let predictions = final_rows
        .iter()
        .map(|&index| {
            let (sum, n) = sums
                .get(&(rows.source[index].clone(), rows.patch[index]))
                .expect("fit stratum");
            let mut value = [0f32; PALETTE_SIZE];
            for c in 0..PALETTE_SIZE {
                value[c] = (sum[c] / *n as f64) as f32;
            }
            value
        })
        .collect::<Vec<_>>();
    Ok(ControlScores {
        observable_positive_mse: positive,
        global_marginal_mse,
        position_source_marginal_mse: mse(&predictions, &final_targets)?,
    })
}

fn unique_episodes(rows: &AuditRows, indices: &[usize]) -> usize {
    indices
        .iter()
        .map(|&i| (rows.source[i].clone(), rows.episode[i]))
        .collect::<BTreeSet<_>>()
        .len()
}
fn mlp_parameter_count(input: usize, hidden: usize) -> usize {
    input * hidden + hidden + hidden * PALETTE_SIZE + PALETTE_SIZE
}
fn stable_seed(seed: u64, name: &str) -> u64 {
    name.bytes().fold(seed ^ 0x9e3779b97f4a7c15, |h, b| {
        h.wrapping_mul(1099511628211).wrapping_add(b as u64)
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn write_json_create_new(path: &Path, report: &SemanticAccessReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let staging = path.with_extension("json.staging");
    ensure!(
        !staging.exists(),
        "audit staging output already exists: {}",
        staging.display()
    );
    let bytes = serde_json::to_vec_pretty(report)?;
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    use std::io::Write;
    let mut file = options.open(&staging)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&staging, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contextual_features_have_bounded_three_part_width() -> Result<()> {
        let rows = (0..PATCH_COUNT)
            .map(|i| vec![i as f32, 1.0])
            .collect::<Vec<_>>();
        let contextual = contextual_features(&rows)?;
        assert_eq!(contextual.len(), PATCH_COUNT);
        assert_eq!(contextual[0].len(), 6);
        Ok(())
    }

    #[test]
    fn parameter_cap_rejects_legacy_hidden_size_for_contextual_decoder() {
        assert!(mlp_parameter_count(128 * 3, 128) < 100_000);
        assert!(mlp_parameter_count(128 * 3, 256) > 100_000);
    }

    #[test]
    fn permutation_deranges_whole_episodes_and_preserves_patch_sites() -> Result<()> {
        let mut targets = Vec::new();
        let mut source = Vec::new();
        let mut episode = Vec::new();
        let mut patch = Vec::new();
        for frame in 0..6 {
            for site in 0..PATCH_COUNT {
                let mut target = [0f32; PALETTE_SIZE];
                target[0] = frame as f32;
                target[1] = site as f32;
                targets.push(target);
                source.push("one_source".into());
                episode.push((frame * 9) as u64);
                patch.push(site);
            }
        }
        let rows = AuditRows {
            local: vec![vec![0.0]; targets.len()],
            contextual: vec![vec![0.0]; targets.len()],
            targets,
            source,
            episode,
            patch,
        };
        let permuted = permute_targets(&rows, 17)?;
        for frame in 0..6 {
            let mapped_frame = permuted[frame * PATCH_COUNT][0] as usize;
            assert_ne!(mapped_frame, frame);
            for site in 0..PATCH_COUNT {
                let target = permuted[frame * PATCH_COUNT + site];
                assert_eq!(target[0] as usize, mapped_frame);
                assert_eq!(target[1] as usize, site);
            }
        }
        Ok(())
    }
}
