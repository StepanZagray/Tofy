//! Deterministic nonlinear coarse semantic-seam audit.
//!
//! Frozen Tofy representations are mapped through presealed sparse ReLU features.
//! Only closed-form ridge readouts are fitted, so optimizer budget and patience
//! cannot censor the evaluator. Selection persists the exact fitted state; a
//! fresh final population is materialized only after every arm and replay seal passes.

use crate::gpu_lock::GpuSessionGuard;
use crate::p2::board_probe::{histograms_for_frames, FixedBoardProbe, PALETTE_SIZE, PATCH_COUNT};
use crate::p2::eval::{
    collect_frozen_board_probe_population_partition_with_predictions, SemanticAccessPopulation,
};
use crate::p2::train::load_train_config;
use anyhow::{ensure, Context, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const SCHEMA: &str = "p2.semantic_access.fixed_coarse.v2";
const STATE_SCHEMA: &str = "p2.semantic_access.fixed_coarse.selection_state.v1";
const CAMPAIGN_SEAL_SCHEMA: &str = "p2.semantic_access.fixed_coarse.selection_seal.v1";
pub const POPULATION_SEED: u64 = 424_244;
pub const FINAL_POPULATION_SEED: u64 = 424_245;
pub const SYNTHETIC_EPISODES: usize = 64;
const FEATURE_WIDTH: usize = 256;
const INPUTS_PER_FEATURE: usize = 8;
const FEATURE_SEEDS: [u64; 3] = [
    0x000f_4958_4544_0001,
    0x000f_4958_4544_0002,
    0x000f_4958_4544_0003,
];
const PARAMETER_CAP: usize = 100_000;
const CONTROL_MSE_CEILING: f64 = 0.04;
const CONTROL_MIN_REDUCTION: f64 = 0.90;
const CONTROL_MIN_ABSOLUTE_IMPROVEMENT: f64 = 0.01;
const CONTROL_INTERACTION_SCALE: f32 = 32.0;

#[derive(Debug, Clone)]
pub struct Config {
    pub checkpoint: PathBuf,
    pub train_config: PathBuf,
    pub physical_batch: usize,
    pub device: String,
    pub selection_only: bool,
    pub required_population_fingerprint: Option<String>,
    pub selection_reference: Option<PathBuf>,
    pub selection_reference_sha256: Option<String>,
    pub fitted_state_output: Option<PathBuf>,
    pub fitted_state_reference: Option<PathBuf>,
    pub fitted_state_reference_sha256: Option<String>,
    pub campaign_selection_seal: Option<PathBuf>,
    pub campaign_selection_seal_sha256: Option<String>,
    pub arm: Option<String>,
    pub final_population_seed: Option<u64>,
    pub final_access_marker: Option<PathBuf>,
    pub output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Protocol {
    pub device: String,
    pub physical_batch: usize,
    pub feature_map: String,
    pub feature_width: usize,
    pub inputs_per_feature: usize,
    pub feature_seeds: Vec<u64>,
    pub seed_aggregation: String,
    pub ridge: f64,
    pub parameter_cap: usize,
    pub learned_parameter_count_per_family: usize,
    pub fixed_nonzero_coefficient_count_per_family: usize,
    pub observable_control_mse_ceiling: f64,
    pub observable_control_min_fractional_reduction: f64,
    pub observable_control_min_absolute_improvement: f64,
    pub observable_control_interaction_scale: f64,
    pub model_weights_frozen: bool,
    pub optimizer_used: bool,
    pub inferential_claims_enabled: bool,
    pub selection_rule: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SplitManifest {
    pub train_frames: usize,
    pub selection_frames: usize,
    pub final_frames: usize,
    pub train_rows: usize,
    pub selection_rows: usize,
    pub final_rows: usize,
    pub rule: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SeedScore {
    pub seed: u64,
    pub feature_map_sha256: String,
    pub mse: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Qualification {
    pub ridge_selection_mse: f64,
    pub ensemble_selection_mse: f64,
    pub fractional_reduction: f64,
    pub per_seed: Vec<SeedScore>,
    pub passed: bool,
    pub failure_reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteSelectionDiagnostic {
    pub fit: String,
    pub ridge_selection_mse: f64,
    pub ensemble_selection_mse: f64,
    pub fractional_reduction: f64,
    pub per_seed: Vec<SeedScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EpisodeScore {
    pub source: String,
    pub episode_id: u64,
    pub patch_rows: usize,
    pub mse: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteScore {
    pub route: String,
    pub ridge_final_mse: f64,
    pub ensemble_final_mse: f64,
    pub per_seed_final: Vec<SeedScore>,
    pub final_episode_mse: Vec<EpisodeScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FamilyReport {
    pub name: String,
    pub input_dim: usize,
    pub learned_parameter_count: usize,
    pub fixed_nonzero_coefficient_count: usize,
    pub qualification: Qualification,
    pub route_selection_diagnostics: Vec<RouteSelectionDiagnostic>,
    pub routes: Vec<RouteScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Report {
    pub schema: String,
    pub checkpoint: PathBuf,
    pub checkpoint_sha256: String,
    pub train_config: PathBuf,
    pub train_config_sha256: String,
    pub population_seed: u64,
    pub synthetic_episodes_per_source: usize,
    pub population_fingerprint: String,
    pub source_population_fingerprint: String,
    pub target_latents_sha256: String,
    pub predicted_latents_sha256: String,
    pub fitted_state_sha256: Option<String>,
    pub selection_reference_sha256: Option<String>,
    pub final_population_seed: Option<u64>,
    pub final_population_fingerprint: Option<String>,
    pub final_target_latents_sha256: Option<String>,
    pub final_predicted_latents_sha256: Option<String>,
    pub target: String,
    pub protocol: Protocol,
    pub split: SplitManifest,
    pub evaluator_status: String,
    pub execution_phase: String,
    pub families: Vec<FamilyReport>,
    pub model_weights_updated: bool,
    pub final_partition_used_for_decoder_selection: bool,
    pub final_partition_scored: bool,
    pub descriptive_seam_interpretation_permitted: bool,
    pub model_level_conclusion_permitted: bool,
    pub next_stage: String,
}

#[derive(Default)]
struct Partitions {
    train_frames: Vec<usize>,
    selection_frames: Vec<usize>,
    final_frames: Vec<usize>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
struct Standardization {
    mean: Vec<f32>,
    std: Vec<f32>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
struct SparseFeature {
    indices: Vec<usize>,
    signs: Vec<f32>,
    bias: f32,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
struct SparseReluMap {
    input_dim: usize,
    seed: u64,
    features: Vec<SparseFeature>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
struct SeedFit {
    map: SparseReluMap,
    residual: FixedBoardProbe,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
struct FittedEnsemble {
    base: FixedBoardProbe,
    input_standardization: Standardization,
    seeds: Vec<SeedFit>,
}

struct EnsemblePrediction {
    base: Vec<[f32; PALETTE_SIZE]>,
    per_seed: Vec<Vec<[f32; PALETTE_SIZE]>>,
    ensemble: Vec<[f32; PALETTE_SIZE]>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
struct FittedFamilyState {
    name: String,
    target_fit: FittedEnsemble,
    predicted_fit: FittedEnsemble,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
struct SelectionState {
    schema: String,
    checkpoint_sha256: String,
    train_config_sha256: String,
    source_population_fingerprint: String,
    population_fingerprint: String,
    target_latents_sha256: String,
    predicted_latents_sha256: String,
    split: SplitManifest,
    families: Vec<FittedFamilyState>,
    payload_sha256: String,
}

#[derive(Debug, Deserialize)]
struct CampaignArmSeal {
    selection_report_sha256: String,
    fitted_state_sha256: String,
    replay_selection_report_sha256: String,
    replay_fitted_state_sha256: String,
    checkpoint_sha256: String,
    train_config_sha256: String,
    evaluator_status: String,
}

#[derive(Debug, Deserialize)]
struct CampaignSelectionSeal {
    schema: String,
    final_arms: Vec<String>,
    final_population_seeds: Vec<u64>,
    synthetic_episodes_per_source: usize,
    arms: BTreeMap<String, CampaignArmSeal>,
}

pub fn run(cfg: &Config) -> Result<Report> {
    validate_config(cfg)?;
    ensure!(
        !cfg.output.exists(),
        "output already exists: {}",
        cfg.output.display()
    );
    if cfg.selection_only {
        run_selection(cfg)
    } else {
        run_final(cfg)
    }
}

fn run_selection(cfg: &Config) -> Result<Report> {
    let checkpoint_sha256 = sha256_file(&cfg.checkpoint)?;
    let train_config_sha256 = sha256_file(&cfg.train_config)?;
    let train_cfg = load_train_config(&cfg.train_config)?;
    let _gpu_guard = if cfg.device == "cuda" || cfg.device.starts_with("cuda:") {
        Some(GpuSessionGuard::acquire(&train_cfg.output_dir)?)
    } else {
        None
    };
    let population = collect_frozen_board_probe_population_partition_with_predictions(
        &cfg.checkpoint,
        &cfg.train_config,
        POPULATION_SEED,
        SYNTHETIC_EPISODES,
        cfg.physical_batch,
        &cfg.device,
        SemanticAccessPopulation::SelectionFit,
    )?;
    ensure!(
        cfg.required_population_fingerprint
            .as_ref()
            .is_some_and(|required| required == &population.source_population_fingerprint),
        "population fingerprint does not match the checksum-verified B1b population"
    );
    ensure!(
        population.samples.len() == population.source_by_sample.len(),
        "source labels do not align with frames"
    );
    let target_local = population.target_rows.as_rows().to_vec();
    let predicted_local = population
        .predicted_rows
        .as_ref()
        .context("fixed coarse audit omitted predicted-next rows")?
        .as_rows()
        .to_vec();
    ensure!(
        target_local.len() == predicted_local.len()
            && target_local.len() == population.samples.len() * PATCH_COUNT,
        "target/predicted latent rows do not align with frames"
    );
    let target_contextual = contextual_features(&target_local)?;
    let predicted_contextual = contextual_features(&predicted_local)?;
    let target_latents_sha256 = sha256_rows(&target_local);
    let predicted_latents_sha256 = sha256_rows(&predicted_local);
    let targets = histograms_for_frames(
        &population
            .samples
            .iter()
            .map(|sample| sample.next.clone())
            .collect::<Vec<_>>(),
    )?;
    let parts = partitions(&population.samples);
    validate_selection_partitions(&parts)?;
    let train = frame_rows(&parts.train_frames);
    let selection = frame_rows(&parts.selection_frames);
    let fit = train.iter().chain(&selection).copied().collect::<Vec<_>>();
    let family_data = [
        (
            "local",
            &target_local,
            &predicted_local,
            &target_local,
            &predicted_local,
        ),
        (
            "contextual_3x3_global",
            &target_contextual,
            &predicted_contextual,
            &target_local,
            &predicted_local,
        ),
    ];

    let mut families = Vec::new();
    for (name, target_features, _predicted_features, target_base, _predicted_base) in
        family_data.iter().copied()
    {
        let qualification = qualify_same_path(
            target_features[0].len(),
            name == "contextual_3x3_global",
            &targets,
            &train,
            &selection,
        )?;
        families.push(FamilyReport {
            name: name.into(),
            input_dim: target_features[0].len(),
            learned_parameter_count: learned_parameter_count(target_base[0].len()),
            fixed_nonzero_coefficient_count: fixed_nonzero_coefficient_count(),
            qualification,
            route_selection_diagnostics: Vec::new(),
            routes: Vec::new(),
        });
    }
    let controls_qualified = families.iter().all(|family| family.qualification.passed);
    if controls_qualified {
        for (index, (_name, target_features, predicted_features, target_base, predicted_base)) in
            family_data.iter().copied().enumerate()
        {
            let target_fit = fit_ensemble(target_features, target_base, &targets, &train)?;
            let predicted_fit = fit_ensemble(predicted_features, predicted_base, &targets, &train)?;
            families[index]
                .route_selection_diagnostics
                .push(selection_diagnostic(
                    "target_next_fit",
                    &target_fit,
                    target_features,
                    target_base,
                    &targets,
                    &selection,
                )?);
            families[index]
                .route_selection_diagnostics
                .push(selection_diagnostic(
                    "predicted_next_fit",
                    &predicted_fit,
                    predicted_features,
                    predicted_base,
                    &targets,
                    &selection,
                )?);
        }
    }
    let diagnostics_finite = controls_qualified
        && families.iter().all(|family| {
            family.route_selection_diagnostics.len() == 2
                && family.route_selection_diagnostics.iter().all(|diagnostic| {
                    diagnostic.ridge_selection_mse.is_finite()
                        && diagnostic.ensemble_selection_mse.is_finite()
                        && diagnostic
                            .per_seed
                            .iter()
                            .all(|score| score.mse.is_finite())
                })
        });
    let evaluator_status = if !controls_qualified {
        "control_invalid"
    } else if !diagnostics_finite {
        "non_finite_selection_diagnostic"
    } else {
        "qualified"
    };
    let split = SplitManifest {
        train_frames: parts.train_frames.len(),
        selection_frames: parts.selection_frames.len(),
        final_frames: 0,
        train_rows: train.len(),
        selection_rows: selection.len(),
        final_rows: 0,
        rule: "selection population only: selection episode_id%9==0; train episode_id%9 in {3,6}; no final rows encoded".into(),
    };
    let fitted_state_sha256 = if evaluator_status == "qualified" {
        let mut fitted_families = Vec::with_capacity(family_data.len());
        for (name, target_features, predicted_features, target_base, predicted_base) in family_data
        {
            fitted_families.push(FittedFamilyState {
                name: name.into(),
                target_fit: fit_ensemble(target_features, target_base, &targets, &fit)?,
                predicted_fit: fit_ensemble(predicted_features, predicted_base, &targets, &fit)?,
            });
        }
        let mut state = SelectionState {
            schema: STATE_SCHEMA.into(),
            checkpoint_sha256: checkpoint_sha256.clone(),
            train_config_sha256: train_config_sha256.clone(),
            source_population_fingerprint: population.source_population_fingerprint.clone(),
            population_fingerprint: population.population_fingerprint.clone(),
            target_latents_sha256: target_latents_sha256.clone(),
            predicted_latents_sha256: predicted_latents_sha256.clone(),
            split: split.clone(),
            families: fitted_families,
            payload_sha256: String::new(),
        };
        state.payload_sha256 = selection_state_payload_sha256(&state);
        let path = cfg
            .fitted_state_output
            .as_deref()
            .context("qualified selection requires --fitted-state-output")?;
        write_json_create_new(path, &state)?;
        Some(sha256_file(path)?)
    } else {
        None
    };
    ensure!(
        sha256_file(&cfg.checkpoint)? == checkpoint_sha256
            && sha256_file(&cfg.train_config)? == train_config_sha256,
        "checkpoint or training config changed during audit"
    );
    let report = Report {
        schema: SCHEMA.into(),
        checkpoint: cfg.checkpoint.clone(),
        checkpoint_sha256,
        train_config: cfg.train_config.clone(),
        train_config_sha256,
        population_seed: POPULATION_SEED,
        synthetic_episodes_per_source: SYNTHETIC_EPISODES,
        population_fingerprint: population.population_fingerprint,
        source_population_fingerprint: population.source_population_fingerprint,
        target_latents_sha256,
        predicted_latents_sha256,
        fitted_state_sha256,
        selection_reference_sha256: None,
        final_population_seed: None,
        final_population_fingerprint: None,
        final_target_latents_sha256: None,
        final_predicted_latents_sha256: None,
        target: "descriptive_16_colour_counts_per_8x8_patch_status_row_excluded".into(),
        protocol: Protocol {
            device: cfg.device.clone(),
            physical_batch: cfg.physical_batch,
            feature_map: "rand-0.9_chacha8_seeded_mixed_sparse_relu_quadratic_v2".into(),
            feature_width: FEATURE_WIDTH,
            inputs_per_feature: INPUTS_PER_FEATURE,
            feature_seeds: FEATURE_SEEDS.to_vec(),
            seed_aggregation: "arithmetic_mean_predictions_no_seed_selection".into(),
            ridge: crate::p2::board_probe::RIDGE,
            parameter_cap: PARAMETER_CAP,
            learned_parameter_count_per_family: learned_parameter_count(target_local[0].len()),
            fixed_nonzero_coefficient_count_per_family: fixed_nonzero_coefficient_count(),
            observable_control_mse_ceiling: CONTROL_MSE_CEILING,
            observable_control_min_fractional_reduction: CONTROL_MIN_REDUCTION,
            observable_control_min_absolute_improvement: CONTROL_MIN_ABSOLUTE_IMPROVEMENT,
            observable_control_interaction_scale: f64::from(CONTROL_INTERACTION_SCALE),
            model_weights_frozen: true,
            optimizer_used: false,
            inferential_claims_enabled: false,
            selection_rule: "episode-disjoint evaluator qualification; no tuned hyperparameters; final scored once after all-arm qualification".into(),
        },
        split,
        evaluator_status: evaluator_status.into(),
        execution_phase: "selection_only".into(),
        families,
        model_weights_updated: false,
        final_partition_used_for_decoder_selection: false,
        final_partition_scored: false,
        descriptive_seam_interpretation_permitted: false,
        model_level_conclusion_permitted: false,
        next_stage: if evaluator_status != "qualified" {
            "repair_deterministic_evaluator_only_without_final_access"
        } else {
            "seal_byte_identical_all_arm_replay_then_run_fresh_final_panel"
        }
        .into(),
    };
    write_json_create_new(&cfg.output, &report)?;
    Ok(report)
}

fn run_final(cfg: &Config) -> Result<Report> {
    let selection_path = cfg
        .selection_reference
        .as_deref()
        .context("final phase requires --selection-reference")?;
    let selection_sha = cfg
        .selection_reference_sha256
        .as_deref()
        .context("final phase requires --selection-reference-sha256")?;
    let state_path = cfg
        .fitted_state_reference
        .as_deref()
        .context("final phase requires --fitted-state-reference")?;
    let state_sha = cfg
        .fitted_state_reference_sha256
        .as_deref()
        .context("final phase requires --fitted-state-reference-sha256")?;
    let campaign_seal_path = cfg
        .campaign_selection_seal
        .as_deref()
        .context("final phase requires --campaign-selection-seal")?;
    let campaign_seal_sha = cfg
        .campaign_selection_seal_sha256
        .as_deref()
        .context("final phase requires --campaign-selection-seal-sha256")?;
    let arm = cfg.arm.as_deref().context("final phase requires --arm")?;
    let final_seed = cfg
        .final_population_seed
        .context("final phase requires --final-population-seed")?;

    verify_file_sha(selection_path, selection_sha, "selection report")?;
    verify_file_sha(state_path, state_sha, "fitted state")?;
    verify_file_sha(
        campaign_seal_path,
        campaign_seal_sha,
        "campaign selection seal",
    )?;
    let selection: Report = read_json(selection_path)?;
    let state: SelectionState = read_json(state_path)?;
    let campaign_seal: CampaignSelectionSeal = read_json(campaign_seal_path)?;
    verify_final_preflight(
        cfg,
        arm,
        final_seed,
        selection_sha,
        state_sha,
        &selection,
        &state,
        &campaign_seal,
    )?;

    let checkpoint_sha256 = sha256_file(&cfg.checkpoint)?;
    let train_config_sha256 = sha256_file(&cfg.train_config)?;
    let train_cfg = load_train_config(&cfg.train_config)?;
    let _gpu_guard = if cfg.device == "cuda" || cfg.device.starts_with("cuda:") {
        Some(GpuSessionGuard::acquire(&train_cfg.output_dir)?)
    } else {
        None
    };
    create_final_access_marker(
        cfg.final_access_marker
            .as_deref()
            .context("final phase requires --final-access-marker")?,
        arm,
        final_seed,
        selection_sha,
        state_sha,
        campaign_seal_sha,
    )?;

    let population = collect_frozen_board_probe_population_partition_with_predictions(
        &cfg.checkpoint,
        &cfg.train_config,
        final_seed,
        SYNTHETIC_EPISODES,
        cfg.physical_batch,
        &cfg.device,
        SemanticAccessPopulation::FreshFinal,
    )?;
    ensure!(
        population.samples.len() == population.source_by_sample.len(),
        "source labels do not align with fresh final frames"
    );
    let target_local = population.target_rows.as_rows().to_vec();
    let predicted_local = population
        .predicted_rows
        .as_ref()
        .context("fresh final population omitted predicted-next rows")?
        .as_rows()
        .to_vec();
    ensure!(
        target_local.len() == predicted_local.len()
            && target_local.len() == population.samples.len() * PATCH_COUNT,
        "fresh final target/predicted rows do not align"
    );
    let target_contextual = contextual_features(&target_local)?;
    let predicted_contextual = contextual_features(&predicted_local)?;
    let targets = histograms_for_frames(
        &population
            .samples
            .iter()
            .map(|sample| sample.next.clone())
            .collect::<Vec<_>>(),
    )?;
    let final_rows = (0..targets.len()).collect::<Vec<_>>();
    let mut row_source = Vec::with_capacity(targets.len());
    let mut row_episode = Vec::with_capacity(targets.len());
    for (frame, sample) in population.samples.iter().enumerate() {
        for _ in 0..PATCH_COUNT {
            row_source.push(population.source_by_sample[frame].clone());
            row_episode.push(sample.episode_id);
        }
    }
    let family_data = [
        (
            "local",
            &target_local,
            &predicted_local,
            &target_local,
            &predicted_local,
        ),
        (
            "contextual_3x3_global",
            &target_contextual,
            &predicted_contextual,
            &target_local,
            &predicted_local,
        ),
    ];
    let mut families = selection.families.clone();
    for (index, (name, target_features, predicted_features, target_base, predicted_base)) in
        family_data.iter().copied().enumerate()
    {
        let fitted = state
            .families
            .get(index)
            .context("fitted-state family count changed")?;
        ensure!(
            fitted.name == name && families[index].name == name,
            "fitted-state family order changed"
        );
        families[index].routes = vec![
            score_route(
                "true_next_encoder_fit",
                &fitted.target_fit,
                target_features,
                target_base,
                &targets,
                &final_rows,
                &row_source,
                &row_episode,
            )?,
            score_route(
                "target_fit_transfer_to_predicted_next",
                &fitted.target_fit,
                predicted_features,
                predicted_base,
                &targets,
                &final_rows,
                &row_source,
                &row_episode,
            )?,
            score_route(
                "predicted_next_refit",
                &fitted.predicted_fit,
                predicted_features,
                predicted_base,
                &targets,
                &final_rows,
                &row_source,
                &row_episode,
            )?,
        ];
    }
    ensure!(
        sha256_file(&cfg.checkpoint)? == checkpoint_sha256
            && sha256_file(&cfg.train_config)? == train_config_sha256,
        "checkpoint or training config changed during final audit"
    );
    let mut split = state.split.clone();
    split.final_frames = population.samples.len();
    split.final_rows = final_rows.len();
    split.rule = format!(
        "selection seed {POPULATION_SEED} train+selection fit; every row from fresh final seed {final_seed} scored exactly once"
    );
    let report = Report {
        schema: SCHEMA.into(),
        checkpoint: cfg.checkpoint.clone(),
        checkpoint_sha256,
        train_config: cfg.train_config.clone(),
        train_config_sha256,
        population_seed: POPULATION_SEED,
        synthetic_episodes_per_source: SYNTHETIC_EPISODES,
        population_fingerprint: state.population_fingerprint.clone(),
        source_population_fingerprint: state.source_population_fingerprint.clone(),
        target_latents_sha256: state.target_latents_sha256.clone(),
        predicted_latents_sha256: state.predicted_latents_sha256.clone(),
        fitted_state_sha256: Some(state_sha.into()),
        selection_reference_sha256: Some(selection_sha.into()),
        final_population_seed: Some(final_seed),
        final_population_fingerprint: Some(population.population_fingerprint),
        final_target_latents_sha256: Some(sha256_rows(&target_local)),
        final_predicted_latents_sha256: Some(sha256_rows(&predicted_local)),
        target: selection.target,
        protocol: selection.protocol,
        split,
        evaluator_status: "qualified".into(),
        execution_phase: "final_score".into(),
        families,
        model_weights_updated: false,
        final_partition_used_for_decoder_selection: false,
        final_partition_scored: true,
        descriptive_seam_interpretation_permitted: true,
        model_level_conclusion_permitted: false,
        next_stage:
            "analyze_paired_population_seed_transfer_then_run_coordinate_aware_exact_cell_sentinel"
                .into(),
    };
    write_json_create_new(&cfg.output, &report)?;
    Ok(report)
}

fn validate_config(cfg: &Config) -> Result<()> {
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
    ensure!(cfg.physical_batch > 0, "physical batch must be positive");
    if cfg.selection_only {
        ensure!(
            cfg.required_population_fingerprint
                .as_ref()
                .is_some_and(|value| value.starts_with("sha256:")),
            "selection requires the checksum-verified source-population fingerprint"
        );
        let state_output = cfg
            .fitted_state_output
            .as_deref()
            .context("selection requires --fitted-state-output")?;
        ensure!(!state_output.exists(), "fitted-state output already exists");
    } else {
        ensure!(
            cfg.selection_reference.is_some(),
            "final scoring requires --selection-reference"
        );
        ensure!(
            cfg.selection_reference_sha256.is_some(),
            "final scoring requires --selection-reference-sha256"
        );
        ensure!(
            cfg.fitted_state_reference.is_some(),
            "final scoring requires --fitted-state-reference"
        );
        ensure!(
            cfg.fitted_state_reference_sha256.is_some(),
            "final scoring requires --fitted-state-reference-sha256"
        );
        ensure!(
            cfg.campaign_selection_seal.is_some(),
            "final scoring requires --campaign-selection-seal"
        );
        ensure!(
            cfg.campaign_selection_seal_sha256.is_some(),
            "final scoring requires --campaign-selection-seal-sha256"
        );
        ensure!(cfg.arm.is_some(), "final scoring requires --arm");
        ensure!(
            cfg.final_population_seed.is_some(),
            "final scoring requires --final-population-seed"
        );
        let marker = cfg
            .final_access_marker
            .as_deref()
            .context("final scoring requires --final-access-marker")?;
        ensure!(
            !marker.exists(),
            "final-access marker already exists; retry forbidden"
        );
    }
    ensure!(
        learned_parameter_count(128) + fixed_nonzero_coefficient_count() <= PARAMETER_CAP,
        "deterministic evaluator exceeds parameter cap"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_final_preflight(
    cfg: &Config,
    arm: &str,
    final_seed: u64,
    selection_sha: &str,
    state_sha: &str,
    selection: &Report,
    state: &SelectionState,
    campaign_seal: &CampaignSelectionSeal,
) -> Result<()> {
    ensure!(
        selection.schema == SCHEMA,
        "selection.schema mismatch: expected={SCHEMA} actual={}",
        selection.schema
    );
    ensure!(
        selection.execution_phase == "selection_only",
        "selection.execution_phase mismatch: expected=selection_only actual={}",
        selection.execution_phase
    );
    ensure!(
        selection.evaluator_status == "qualified",
        "selection.evaluator_status mismatch: expected=qualified actual={}",
        selection.evaluator_status
    );
    ensure!(
        !selection.final_partition_scored,
        "selection.final_partition_scored must be false"
    );
    ensure!(
        selection.fitted_state_sha256.as_deref() == Some(state_sha),
        "selection.fitted_state_sha256 mismatch: expected={state_sha} actual={:?}",
        selection.fitted_state_sha256
    );
    ensure!(
        state.schema == STATE_SCHEMA,
        "state.schema mismatch: expected={STATE_SCHEMA} actual={}",
        state.schema
    );
    let actual_payload_sha = selection_state_payload_sha256(state);
    ensure!(
        state.payload_sha256 == actual_payload_sha,
        "state.payload_sha256 mismatch: expected={} actual={actual_payload_sha}",
        state.payload_sha256
    );
    ensure!(
        campaign_seal.schema == CAMPAIGN_SEAL_SCHEMA,
        "campaign_seal.schema mismatch: expected={CAMPAIGN_SEAL_SCHEMA} actual={}",
        campaign_seal.schema
    );
    ensure!(
        campaign_seal.final_population_seeds == vec![424_245, 424_246, 424_247],
        "campaign_seal.final_population_seeds mismatch: expected=[424245,424246,424247] actual={:?}",
        campaign_seal.final_population_seeds
    );
    ensure!(
        campaign_seal.final_arms == vec!["S0G1", "ScurG1"],
        "campaign_seal.final_arms mismatch: expected=[S0G1,ScurG1] actual={:?}",
        campaign_seal.final_arms
    );
    ensure!(
        campaign_seal.final_population_seeds.contains(&final_seed),
        "final_population_seed {final_seed} is not sealed"
    );
    ensure!(
        campaign_seal.synthetic_episodes_per_source == SYNTHETIC_EPISODES,
        "campaign_seal.synthetic_episodes_per_source mismatch"
    );
    let expected_arms = ["S0G0", "S0G1", "ScalG0", "ScalG1", "ScurG0", "ScurG1"];
    ensure!(
        ["S0G1", "ScurG1"].contains(&arm),
        "arm {arm} is not in the sealed endpoint final panel"
    );
    ensure!(
        campaign_seal.arms.len() == expected_arms.len()
            && expected_arms
                .iter()
                .all(|name| campaign_seal.arms.contains_key(*name)),
        "campaign seal must bind all six registered arms"
    );
    ensure!(
        campaign_seal
            .arms
            .values()
            .all(|entry| entry.evaluator_status == "qualified"),
        "campaign seal contains an unqualified arm"
    );
    ensure!(
        campaign_seal.arms.values().all(|entry| {
            entry.selection_report_sha256 == entry.replay_selection_report_sha256
                && entry.fitted_state_sha256 == entry.replay_fitted_state_sha256
        }),
        "campaign seal contains a non-identical selection replay"
    );
    let arm_seal = campaign_seal
        .arms
        .get(arm)
        .with_context(|| format!("arm {arm} missing from campaign seal"))?;
    ensure!(
        arm_seal.selection_report_sha256 == selection_sha,
        "campaign_seal.arms.{arm}.selection_report_sha256 mismatch"
    );
    ensure!(
        arm_seal.fitted_state_sha256 == state_sha,
        "campaign_seal.arms.{arm}.fitted_state_sha256 mismatch"
    );
    ensure!(
        arm_seal.replay_selection_report_sha256 == arm_seal.selection_report_sha256,
        "campaign_seal.arms.{arm} selection replay is not byte-identical"
    );
    ensure!(
        arm_seal.replay_fitted_state_sha256 == arm_seal.fitted_state_sha256,
        "campaign_seal.arms.{arm} fitted-state replay is not byte-identical"
    );
    let checkpoint_sha = sha256_file(&cfg.checkpoint)?;
    let train_config_sha = sha256_file(&cfg.train_config)?;
    ensure!(
        arm_seal.checkpoint_sha256 == checkpoint_sha,
        "campaign_seal.arms.{arm}.checkpoint_sha256 mismatch"
    );
    ensure!(
        arm_seal.train_config_sha256 == train_config_sha,
        "campaign_seal.arms.{arm}.train_config_sha256 mismatch"
    );
    ensure!(
        selection.checkpoint_sha256 == checkpoint_sha && state.checkpoint_sha256 == checkpoint_sha,
        "checkpoint identity differs across report/state/current input"
    );
    ensure!(
        selection.train_config_sha256 == train_config_sha
            && state.train_config_sha256 == train_config_sha,
        "train-config identity differs across report/state/current input"
    );
    ensure!(
        selection.source_population_fingerprint == state.source_population_fingerprint,
        "source_population_fingerprint differs between report and state"
    );
    ensure!(
        selection.population_fingerprint == state.population_fingerprint,
        "population_fingerprint differs between report and state"
    );
    ensure!(
        selection.target_latents_sha256 == state.target_latents_sha256,
        "target_latents_sha256 differs between report and state"
    );
    ensure!(
        selection.predicted_latents_sha256 == state.predicted_latents_sha256,
        "predicted_latents_sha256 differs between report and state"
    );
    ensure!(
        selection.split == state.split,
        "split differs between report and state"
    );
    ensure!(
        selection.families.len() == state.families.len(),
        "family count differs between report and state"
    );
    ensure!(
        selection.population_seed == POPULATION_SEED
            && selection.synthetic_episodes_per_source == SYNTHETIC_EPISODES
            && selection.protocol.feature_width == FEATURE_WIDTH
            && selection.protocol.inputs_per_feature == INPUTS_PER_FEATURE
            && selection.protocol.feature_seeds == FEATURE_SEEDS
            && selection.protocol.ridge == crate::p2::board_probe::RIDGE
            && selection.protocol.model_weights_frozen
            && !selection.protocol.optimizer_used,
        "selection protocol differs from the compiled sealed protocol"
    );
    for (report_family, state_family) in selection.families.iter().zip(&state.families) {
        ensure!(
            report_family.name == state_family.name,
            "family order differs: report={} state={}",
            report_family.name,
            state_family.name
        );
    }
    Ok(())
}

fn verify_file_sha(path: &Path, expected: &str, label: &str) -> Result<()> {
    ensure!(
        expected.len() == 64
            && expected
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "{label} SHA-256 must be 64 lowercase hexadecimal characters"
    );
    let actual = sha256_file(path)?;
    ensure!(
        actual == expected,
        "{label} SHA-256 mismatch: expected={expected} actual={actual}"
    );
    Ok(())
}

fn selection_state_payload_sha256(state: &SelectionState) -> String {
    let mut payload = state.clone();
    payload.payload_sha256.clear();
    let encoded = serde_json::to_vec(&payload).expect("selection state is serializable");
    format!("sha256:{:x}", Sha256::digest(encoded))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))
}

fn create_final_access_marker(
    path: &Path,
    arm: &str,
    final_seed: u64,
    selection_sha: &str,
    state_sha: &str,
    campaign_seal_sha: &str,
) -> Result<()> {
    let marker = serde_json::json!({
        "schema": "p2.semantic_access.fixed_coarse.final_access.v1",
        "arm": arm,
        "final_population_seed": final_seed,
        "selection_report_sha256": selection_sha,
        "fitted_state_sha256": state_sha,
        "campaign_selection_seal_sha256": campaign_seal_sha
    });
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    let mut file = options
        .open(path)
        .with_context(|| format!("create one-shot final-access marker {}", path.display()))?;
    file.write_all(&serde_json::to_vec_pretty(&marker)?)?;
    file.sync_all()?;
    Ok(())
}

fn qualify_same_path(
    input_dim: usize,
    contextual: bool,
    targets: &[[f32; PALETTE_SIZE]],
    train: &[usize],
    selection: &[usize],
) -> Result<Qualification> {
    let local_dim = if contextual { input_dim / 3 } else { input_dim };
    ensure!(
        !contextual || local_dim * 3 == input_dim,
        "invalid contextual width"
    );
    ensure!(local_dim > PALETTE_SIZE + 2, "control input is too narrow");
    let mut control_targets = targets.to_vec();
    let observable_local = targets
        .iter()
        .enumerate()
        .map(|(index, target)| {
            // Derive the control coordinates from row-local observables only. This avoids
            // leaking global ordering, episode, source, arm, or partition membership.
            let mut digest = Sha256::new();
            digest.update(b"tofy-fixed-control-v1");
            for value in target {
                digest.update(value.to_bits().to_le_bytes());
            }
            let bytes = digest.finalize();
            let u_word = u64::from_le_bytes(bytes[0..8].try_into().expect("fixed digest width"));
            let v_word = u64::from_le_bytes(bytes[8..16].try_into().expect("fixed digest width"));
            let u = 2.0 * (u_word as f64 / u64::MAX as f64) as f32 - 1.0;
            let v = 2.0 * (v_word as f64 / u64::MAX as f64) as f32 - 1.0;
            let mut row = vec![0.0; local_dim];
            for colour in 0..PALETTE_SIZE {
                row[colour] = target[colour] / 64.0;
            }
            for (channel, value) in row.iter_mut().enumerate().skip(PALETTE_SIZE) {
                *value = if channel.is_multiple_of(2) { u } else { v };
            }
            control_targets[index][0] += CONTROL_INTERACTION_SCALE * u * v;
            row
        })
        .collect::<Vec<_>>();
    let observable_features = if contextual {
        contextual_features(&observable_local)?
    } else {
        observable_local.clone()
    };
    let fitted = fit_ensemble(
        &observable_features,
        &observable_local,
        &control_targets,
        train,
    )?;
    let predicted = predict_ensemble(
        &fitted,
        &select_rows(&observable_features, selection),
        &select_rows(&observable_local, selection),
    )?;
    let selection_targets = select_targets(&control_targets, selection);
    let ridge_selection_mse = mse(&predicted.base, &selection_targets)?;
    let ensemble_selection_mse = mse(&predicted.ensemble, &selection_targets)?;
    let per_seed = seed_scores(&fitted, &predicted.per_seed, &selection_targets)?;
    let fractional_reduction = 1.0 - ensemble_selection_mse / ridge_selection_mse.max(1e-12);
    let mut failure_reasons = Vec::new();
    if !ridge_selection_mse.is_finite()
        || !ensemble_selection_mse.is_finite()
        || per_seed.iter().any(|score| !score.mse.is_finite())
    {
        failure_reasons.push("non_finite_control_score".into());
    }
    if ensemble_selection_mse > CONTROL_MSE_CEILING {
        failure_reasons.push("observable_control_above_fixed_mse_ceiling".into());
    }
    if per_seed.iter().any(|score| score.mse > CONTROL_MSE_CEILING) {
        failure_reasons.push("observable_control_seed_above_fixed_mse_ceiling".into());
    }
    if fractional_reduction < CONTROL_MIN_REDUCTION {
        failure_reasons.push("observable_control_insufficient_fractional_reduction".into());
    }
    if ridge_selection_mse - ensemble_selection_mse < CONTROL_MIN_ABSOLUTE_IMPROVEMENT {
        failure_reasons.push("observable_control_insufficient_absolute_improvement".into());
    }
    Ok(Qualification {
        ridge_selection_mse,
        ensemble_selection_mse,
        fractional_reduction,
        per_seed,
        passed: failure_reasons.is_empty(),
        failure_reasons,
    })
}

fn fit_ensemble(
    features: &[Vec<f32>],
    base_features: &[Vec<f32>],
    targets: &[[f32; PALETTE_SIZE]],
    indices: &[usize],
) -> Result<FittedEnsemble> {
    let fit_features = select_rows(features, indices);
    let fit_base_features = select_rows(base_features, indices);
    let fit_targets = select_targets(targets, indices);
    let base = FixedBoardProbe::fit_histograms(&fit_base_features, &fit_targets)?;
    let base_predictions = base.predict_histograms(&fit_base_features)?;
    let residual_targets = residual_targets(&fit_targets, &base_predictions);
    let input_standardization = feature_standardization(&fit_features)?;
    let standardized = apply_standardization(
        &fit_features,
        &input_standardization.mean,
        &input_standardization.std,
    );
    let mut seeds = Vec::with_capacity(FEATURE_SEEDS.len());
    for seed in FEATURE_SEEDS {
        let map = SparseReluMap::new(features[0].len(), seed)?;
        let mapped = map.apply(&standardized)?;
        let residual = FixedBoardProbe::fit_histograms(&mapped, &residual_targets)?;
        seeds.push(SeedFit { map, residual });
    }
    Ok(FittedEnsemble {
        base,
        input_standardization,
        seeds,
    })
}

fn predict_ensemble(
    fitted: &FittedEnsemble,
    features: &[Vec<f32>],
    base_features: &[Vec<f32>],
) -> Result<EnsemblePrediction> {
    ensure!(
        !features.is_empty() && features.len() == base_features.len(),
        "invalid prediction rows"
    );
    let base = fitted.base.predict_histograms(base_features)?;
    let standardized = apply_standardization(
        features,
        &fitted.input_standardization.mean,
        &fitted.input_standardization.std,
    );
    let mut per_seed = Vec::with_capacity(fitted.seeds.len());
    for seed_fit in &fitted.seeds {
        let mapped = seed_fit.map.apply(&standardized)?;
        let residual = seed_fit.residual.predict_histograms(&mapped)?;
        per_seed.push(add_predictions(&base, &residual)?);
    }
    let ensemble = average_predictions(&per_seed)?;
    Ok(EnsemblePrediction {
        base,
        per_seed,
        ensemble,
    })
}

fn selection_diagnostic(
    fit_name: &str,
    fitted: &FittedEnsemble,
    features: &[Vec<f32>],
    base_features: &[Vec<f32>],
    targets: &[[f32; PALETTE_SIZE]],
    selection: &[usize],
) -> Result<RouteSelectionDiagnostic> {
    let selection_targets = select_targets(targets, selection);
    let prediction = predict_ensemble(
        fitted,
        &select_rows(features, selection),
        &select_rows(base_features, selection),
    )?;
    let ridge_selection_mse = mse(&prediction.base, &selection_targets)?;
    let ensemble_selection_mse = mse(&prediction.ensemble, &selection_targets)?;
    Ok(RouteSelectionDiagnostic {
        fit: fit_name.into(),
        ridge_selection_mse,
        ensemble_selection_mse,
        fractional_reduction: 1.0 - ensemble_selection_mse / ridge_selection_mse.max(1e-12),
        per_seed: seed_scores(fitted, &prediction.per_seed, &selection_targets)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn score_route(
    route: &str,
    fitted: &FittedEnsemble,
    features: &[Vec<f32>],
    base_features: &[Vec<f32>],
    targets: &[[f32; PALETTE_SIZE]],
    final_rows: &[usize],
    row_source: &[String],
    row_episode: &[u64],
) -> Result<RouteScore> {
    let final_targets = select_targets(targets, final_rows);
    let prediction = predict_ensemble(
        fitted,
        &select_rows(features, final_rows),
        &select_rows(base_features, final_rows),
    )?;
    Ok(RouteScore {
        route: route.into(),
        ridge_final_mse: mse(&prediction.base, &final_targets)?,
        ensemble_final_mse: mse(&prediction.ensemble, &final_targets)?,
        per_seed_final: seed_scores(fitted, &prediction.per_seed, &final_targets)?,
        final_episode_mse: episode_scores(
            final_rows,
            row_source,
            row_episode,
            &prediction.ensemble,
            &final_targets,
        )?,
    })
}

impl SparseReluMap {
    fn new(input_dim: usize, seed: u64) -> Result<Self> {
        ensure!(
            input_dim >= INPUTS_PER_FEATURE,
            "feature input is too narrow"
        );
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut features = Vec::with_capacity(FEATURE_WIDTH);
        for _ in 0..FEATURE_WIDTH {
            let mut indices = Vec::with_capacity(INPUTS_PER_FEATURE);
            while indices.len() < INPUTS_PER_FEATURE {
                let index = rng.random_range(0..input_dim);
                if !indices.contains(&index) {
                    indices.push(index);
                }
            }
            let signs = (0..INPUTS_PER_FEATURE)
                .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
                .collect();
            let bias = rng.random_range(-1.0_f32..1.0_f32);
            features.push(SparseFeature {
                indices,
                signs,
                bias,
            });
        }
        Ok(Self {
            input_dim,
            seed,
            features,
        })
    }

    fn apply(&self, rows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        ensure!(
            !rows.is_empty() && rows.iter().all(|row| row.len() == self.input_dim),
            "feature-map input width mismatch"
        );
        let normalization = (INPUTS_PER_FEATURE as f32).sqrt().recip();
        Ok(rows
            .par_iter()
            .map(|row| {
                self.features
                    .iter()
                    .enumerate()
                    .map(|(feature_index, feature)| {
                        if feature_index % 2 == 1 {
                            let midpoint = INPUTS_PER_FEATURE / 2;
                            let left = feature.indices[..midpoint]
                                .iter()
                                .zip(&feature.signs[..midpoint])
                                .map(|(&index, &sign)| row[index] * sign)
                                .sum::<f32>()
                                / (midpoint as f32).sqrt();
                            let right = feature.indices[midpoint..]
                                .iter()
                                .zip(&feature.signs[midpoint..])
                                .map(|(&index, &sign)| row[index] * sign)
                                .sum::<f32>()
                                / ((INPUTS_PER_FEATURE - midpoint) as f32).sqrt();
                            return left * right;
                        }
                        let projected = feature
                            .indices
                            .iter()
                            .zip(&feature.signs)
                            .map(|(&index, &sign)| row[index] * sign)
                            .sum::<f32>()
                            * normalization
                            + feature.bias;
                        projected.max(0.0)
                    })
                    .collect()
            })
            .collect())
    }

    fn sha256(&self) -> String {
        let mut digest = Sha256::new();
        digest.update((self.input_dim as u64).to_le_bytes());
        digest.update(self.seed.to_le_bytes());
        for feature in &self.features {
            for (&index, &sign) in feature.indices.iter().zip(&feature.signs) {
                digest.update((index as u64).to_le_bytes());
                digest.update(sign.to_bits().to_le_bytes());
            }
            digest.update(feature.bias.to_bits().to_le_bytes());
        }
        format!("sha256:{:x}", digest.finalize())
    }
}

fn seed_scores(
    fitted: &FittedEnsemble,
    predictions: &[Vec<[f32; PALETTE_SIZE]>],
    targets: &[[f32; PALETTE_SIZE]],
) -> Result<Vec<SeedScore>> {
    ensure!(
        fitted.seeds.len() == predictions.len(),
        "seed predictions do not align"
    );
    fitted
        .seeds
        .iter()
        .zip(predictions)
        .map(|(fit, prediction)| {
            Ok(SeedScore {
                seed: fit.map.seed,
                feature_map_sha256: fit.map.sha256(),
                mse: mse(prediction, targets)?,
            })
        })
        .collect()
}

fn residual_targets(
    targets: &[[f32; PALETTE_SIZE]],
    base: &[[f32; PALETTE_SIZE]],
) -> Vec<[f32; PALETTE_SIZE]> {
    targets
        .iter()
        .zip(base)
        .map(|(target, prediction)| {
            let mut residual = [0.0; PALETTE_SIZE];
            for colour in 0..PALETTE_SIZE {
                residual[colour] = target[colour] - prediction[colour];
            }
            residual
        })
        .collect()
}

fn add_predictions(
    base: &[[f32; PALETTE_SIZE]],
    residual: &[[f32; PALETTE_SIZE]],
) -> Result<Vec<[f32; PALETTE_SIZE]>> {
    ensure!(
        base.len() == residual.len(),
        "base/residual rows do not align"
    );
    base.iter()
        .zip(residual)
        .map(|(base, residual)| {
            let mut combined = [0.0; PALETTE_SIZE];
            for colour in 0..PALETTE_SIZE {
                combined[colour] = base[colour] + residual[colour];
            }
            ensure!(
                combined.iter().all(|value| value.is_finite()),
                "non-finite prediction"
            );
            Ok(combined)
        })
        .collect()
}

fn average_predictions(
    predictions: &[Vec<[f32; PALETTE_SIZE]>],
) -> Result<Vec<[f32; PALETTE_SIZE]>> {
    ensure!(!predictions.is_empty(), "empty prediction ensemble");
    let rows = predictions[0].len();
    ensure!(
        rows > 0
            && predictions
                .iter()
                .all(|prediction| prediction.len() == rows),
        "ensemble rows do not align"
    );
    let mut result = vec![[0.0; PALETTE_SIZE]; rows];
    for prediction in predictions {
        for (output, row) in result.iter_mut().zip(prediction) {
            for colour in 0..PALETTE_SIZE {
                output[colour] += row[colour] / predictions.len() as f32;
            }
        }
    }
    ensure!(
        result.iter().flatten().all(|value| value.is_finite()),
        "non-finite ensemble"
    );
    Ok(result)
}

fn feature_standardization(features: &[Vec<f32>]) -> Result<Standardization> {
    ensure!(!features.is_empty(), "empty standardization population");
    let dim = features[0].len();
    ensure!(
        dim > 0 && features.iter().all(|row| row.len() == dim),
        "inconsistent feature width"
    );
    let mut mean = vec![0.0f64; dim];
    for row in features {
        for (index, value) in row.iter().enumerate() {
            mean[index] += f64::from(*value);
        }
    }
    for value in &mut mean {
        *value /= features.len() as f64;
    }
    let mut std = vec![0.0f64; dim];
    for row in features {
        for (index, value) in row.iter().enumerate() {
            let delta = f64::from(*value) - mean[index];
            std[index] += delta * delta;
        }
    }
    for value in &mut std {
        *value = (*value / features.len() as f64).sqrt().max(1e-6);
    }
    Ok(Standardization {
        mean: mean.into_iter().map(|value| value as f32).collect(),
        std: std.into_iter().map(|value| value as f32).collect(),
    })
}

fn apply_standardization(features: &[Vec<f32>], mean: &[f32], std: &[f32]) -> Vec<Vec<f32>> {
    features
        .par_iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(index, value)| (*value - mean[index]) / std[index])
                .collect()
        })
        .collect()
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
            let y = patch_index / 8;
            let x = patch_index % 8;
            let mut neighborhood = vec![0.0f32; channels];
            let mut count = 0.0f32;
            for ny in y.saturating_sub(1)..=(y + 1).min(7) {
                for nx in x.saturating_sub(1)..=(x + 1).min(7) {
                    for (sum, value) in neighborhood
                        .iter_mut()
                        .zip(&local[frame_start + ny * 8 + nx])
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
        if !sample.episode_id.is_multiple_of(3) {
            result.final_frames.push(index);
        } else if sample.episode_id.is_multiple_of(9) {
            result.selection_frames.push(index);
        } else {
            result.train_frames.push(index);
        }
    }
    result
}

fn validate_selection_partitions(parts: &Partitions) -> Result<()> {
    ensure!(
        !parts.train_frames.is_empty() && !parts.selection_frames.is_empty(),
        "selection population contains an empty train or selection partition"
    );
    ensure!(
        parts.final_frames.is_empty(),
        "selection collector encoded forbidden final rows"
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
        .flat_map(|(prediction, target)| prediction.iter().zip(target))
        .map(|(prediction, target)| {
            let delta = f64::from(*prediction - *target);
            delta * delta
        })
        .sum::<f64>()
        / (predictions.len() * PALETTE_SIZE) as f64)
}

fn episode_scores(
    indices: &[usize],
    row_source: &[String],
    row_episode: &[u64],
    predictions: &[[f32; PALETTE_SIZE]],
    targets: &[[f32; PALETTE_SIZE]],
) -> Result<Vec<EpisodeScore>> {
    ensure!(
        indices.len() == predictions.len()
            && predictions.len() == targets.len()
            && row_source.len() == row_episode.len(),
        "episode scoring rows do not align"
    );
    let mut grouped: BTreeMap<(String, u64), (f64, usize)> = BTreeMap::new();
    for ((prediction, target), &index) in predictions.iter().zip(targets).zip(indices) {
        let squared_error = prediction
            .iter()
            .zip(target)
            .map(|(prediction, target)| {
                let delta = f64::from(*prediction - *target);
                delta * delta
            })
            .sum::<f64>();
        let entry = grouped
            .entry((row_source[index].clone(), row_episode[index]))
            .or_insert((0.0, 0));
        entry.0 += squared_error;
        entry.1 += 1;
    }
    Ok(grouped
        .into_iter()
        .map(|((source, episode_id), (sum, patch_rows))| EpisodeScore {
            source,
            episode_id,
            patch_rows,
            mse: sum / (patch_rows * PALETTE_SIZE) as f64,
        })
        .collect())
}

fn learned_parameter_count(base_dim: usize) -> usize {
    base_dim * PALETTE_SIZE
        + PALETTE_SIZE
        + FEATURE_SEEDS.len() * (FEATURE_WIDTH * PALETTE_SIZE + PALETTE_SIZE)
}

fn fixed_nonzero_coefficient_count() -> usize {
    FEATURE_SEEDS.len() * FEATURE_WIDTH * (INPUTS_PER_FEATURE + 1)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_rows(rows: &[Vec<f32>]) -> String {
    let mut digest = Sha256::new();
    digest.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        digest.update((row.len() as u64).to_le_bytes());
        for value in row {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    format!("sha256:{:x}", digest.finalize())
}

fn write_json_create_new<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let staging = path.with_extension("json.staging");
    ensure!(
        !staging.exists(),
        "staging output already exists: {}",
        staging.display()
    );
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    let mut file = options.open(&staging)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.sync_all()?;
    fs::rename(staging, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_maps_are_deterministic_and_seed_distinct() -> Result<()> {
        let first = SparseReluMap::new(128, FEATURE_SEEDS[0])?;
        let repeated = SparseReluMap::new(128, FEATURE_SEEDS[0])?;
        let second = SparseReluMap::new(128, FEATURE_SEEDS[1])?;
        assert_eq!(first.sha256(), repeated.sha256());
        assert_ne!(first.sha256(), second.sha256());
        assert_eq!(first.features.len(), FEATURE_WIDTH);
        assert!(first
            .features
            .iter()
            .all(|feature| feature.indices.len() == INPUTS_PER_FEATURE));
        Ok(())
    }

    #[test]
    fn local_and_contextual_have_equal_evaluator_capacity() {
        assert_eq!(learned_parameter_count(128), learned_parameter_count(128));
        assert_eq!(fixed_nonzero_coefficient_count(), 6_912);
        assert!(learned_parameter_count(128) + fixed_nonzero_coefficient_count() < PARAMETER_CAP);
    }

    #[test]
    fn contextual_width_and_count_are_stable() -> Result<()> {
        let rows = (0..PATCH_COUNT)
            .map(|index| vec![index as f32, 1.0])
            .collect::<Vec<_>>();
        let contextual = contextual_features(&rows)?;
        assert_eq!(contextual.len(), PATCH_COUNT);
        assert!(contextual.iter().all(|row| row.len() == 6));
        Ok(())
    }

    #[test]
    fn fixed_ensemble_fits_nonlinear_observable() -> Result<()> {
        let rows = 2_048usize;
        let mut features = Vec::with_capacity(rows);
        let mut targets = Vec::with_capacity(rows);
        for index in 0..rows {
            let u = 2.0 * (index % 47) as f32 / 46.0 - 1.0;
            let v = 2.0 * (index.wrapping_mul(11) % 53) as f32 / 52.0 - 1.0;
            let row = (0usize..128)
                .map(|channel| if channel.is_multiple_of(2) { u } else { v })
                .collect::<Vec<_>>();
            let mut target = [0.0; PALETTE_SIZE];
            target[0] = CONTROL_INTERACTION_SCALE * u * v;
            features.push(row);
            targets.push(target);
        }
        let train = (0..1_536).collect::<Vec<_>>();
        let held_out = (1_536..rows).collect::<Vec<_>>();
        let fitted = fit_ensemble(&features, &features, &targets, &train)?;
        let prediction = predict_ensemble(
            &fitted,
            &select_rows(&features, &held_out),
            &select_rows(&features, &held_out),
        )?;
        let held_out_targets = select_targets(&targets, &held_out);
        let ridge = mse(&prediction.base, &held_out_targets)?;
        let ensemble = mse(&prediction.ensemble, &held_out_targets)?;
        let per_seed = prediction
            .per_seed
            .iter()
            .map(|rows| mse(rows, &held_out_targets))
            .collect::<Result<Vec<_>>>()?;
        let reduction = 1.0 - ensemble / ridge.max(1e-12);
        assert!(
            ensemble <= CONTROL_MSE_CEILING
                && per_seed.iter().all(|mse| *mse <= CONTROL_MSE_CEILING)
                && reduction >= CONTROL_MIN_REDUCTION
                && ridge - ensemble >= CONTROL_MIN_ABSOLUTE_IMPROVEMENT,
            "ridge={ridge} ensemble={ensemble} per_seed={per_seed:?} reduction={reduction}"
        );
        Ok(())
    }

    #[test]
    fn selection_state_round_trip_preserves_fit_bits_and_payload_seal() -> Result<()> {
        let rows = (0usize..640)
            .map(|row| {
                (0usize..128)
                    .map(|channel| ((row * 17 + channel * 13) % 101) as f32 / 50.0 - 1.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let targets = rows
            .iter()
            .map(|row| {
                let mut target = [0.0; PALETTE_SIZE];
                target[0] = row[3] * row[71];
                target[1] = row[11].max(0.0);
                target
            })
            .collect::<Vec<_>>();
        let fit_rows = (0..512).collect::<Vec<_>>();
        let fitted = fit_ensemble(&rows, &rows, &targets, &fit_rows)?;
        let mut state = SelectionState {
            schema: STATE_SCHEMA.into(),
            checkpoint_sha256: "a".repeat(64),
            train_config_sha256: "b".repeat(64),
            source_population_fingerprint: format!("sha256:{}", "c".repeat(64)),
            population_fingerprint: format!("sha256:{}", "d".repeat(64)),
            target_latents_sha256: format!("sha256:{}", "e".repeat(64)),
            predicted_latents_sha256: format!("sha256:{}", "f".repeat(64)),
            split: SplitManifest {
                train_frames: 3,
                selection_frames: 2,
                final_frames: 0,
                train_rows: 192,
                selection_rows: 128,
                final_rows: 0,
                rule: "test".into(),
            },
            families: vec![FittedFamilyState {
                name: "local".into(),
                target_fit: fitted.clone(),
                predicted_fit: fitted,
            }],
            payload_sha256: String::new(),
        };
        state.payload_sha256 = selection_state_payload_sha256(&state);
        let encoded = serde_json::to_vec_pretty(&state)?;
        let decoded: SelectionState = serde_json::from_slice(&encoded)?;
        assert_eq!(
            state.payload_sha256,
            selection_state_payload_sha256(&decoded)
        );
        assert!(state == decoded);

        let held_out = (512..640).collect::<Vec<_>>();
        let before = predict_ensemble(
            &state.families[0].target_fit,
            &select_rows(&rows, &held_out),
            &select_rows(&rows, &held_out),
        )?;
        let after = predict_ensemble(
            &decoded.families[0].target_fit,
            &select_rows(&rows, &held_out),
            &select_rows(&rows, &held_out),
        )?;
        assert_eq!(before.ensemble, after.ensemble);
        Ok(())
    }

    #[test]
    fn selection_state_payload_seal_detects_float_mutation() -> Result<()> {
        let rows = (0usize..300)
            .map(|row| vec![(row % 17) as f32 / 16.0; 128])
            .collect::<Vec<_>>();
        let mut targets = vec![[0.0; PALETTE_SIZE]; rows.len()];
        for (row, target) in targets.iter_mut().enumerate() {
            target[0] = (row % 11) as f32 / 10.0;
        }
        let fit_rows = (0..rows.len()).collect::<Vec<_>>();
        let fitted = fit_ensemble(&rows, &rows, &targets, &fit_rows)?;
        let mut state = SelectionState {
            schema: STATE_SCHEMA.into(),
            checkpoint_sha256: "a".repeat(64),
            train_config_sha256: "b".repeat(64),
            source_population_fingerprint: format!("sha256:{}", "c".repeat(64)),
            population_fingerprint: format!("sha256:{}", "d".repeat(64)),
            target_latents_sha256: format!("sha256:{}", "e".repeat(64)),
            predicted_latents_sha256: format!("sha256:{}", "f".repeat(64)),
            split: SplitManifest {
                train_frames: 1,
                selection_frames: 1,
                final_frames: 0,
                train_rows: 1,
                selection_rows: 1,
                final_rows: 0,
                rule: "test".into(),
            },
            families: vec![FittedFamilyState {
                name: "local".into(),
                target_fit: fitted.clone(),
                predicted_fit: fitted,
            }],
            payload_sha256: String::new(),
        };
        state.payload_sha256 = selection_state_payload_sha256(&state);
        let sealed = state.payload_sha256.clone();
        state.families[0].target_fit.input_standardization.mean[0] = f32::from_bits(
            state.families[0].target_fit.input_standardization.mean[0].to_bits() + 1,
        );
        assert_ne!(sealed, selection_state_payload_sha256(&state));
        Ok(())
    }
}
