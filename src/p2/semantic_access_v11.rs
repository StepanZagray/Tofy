//! Semantic Access V1.1: qualified, descriptive target/prediction seam audit.
//!
//! The scientific protocol is intentionally sealed. Model weights are frozen;
//! only bounded decoders are optimized. A same-path observable control must
//! qualify before any checkpoint result is emitted.

use crate::gpu_lock::GpuSessionGuard;
use crate::p2::board_probe::{histograms_for_frames, FixedBoardProbe, PALETTE_SIZE, PATCH_COUNT};
use crate::p2::eval::collect_frozen_board_probe_population_with_predictions;
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
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const SCHEMA: &str = "p2.semantic_access.v1_1_stage_b1b";
pub const POPULATION_SEED: u64 = 424_244;
pub const SYNTHETIC_EPISODES: usize = 64;
const HIDDEN: usize = 64;
const MAX_STEPS: usize = 4_800;
const EVAL_EVERY: usize = 25;
const PATIENCE_EVALS: usize = 8;
const LEARNING_RATE: f64 = 1e-3;
const WEIGHT_DECAY: f64 = 1e-4;
const PARAMETER_CAP: usize = 100_000;
const DECODER_BATCH: usize = 4_096;
const CONTROL_MSE_CEILING: f64 = 0.04;
const CONTROL_MIN_REDUCTION: f64 = 0.90;
const CONTROL_MIN_ABSOLUTE_IMPROVEMENT: f64 = 0.01;

#[derive(Debug, Clone)]
pub struct Config {
    pub checkpoint: PathBuf,
    pub train_config: PathBuf,
    pub physical_batch: usize,
    pub device: String,
    pub selection_only: bool,
    pub forbidden_population_fingerprint: Option<String>,
    pub selection_reference: Option<PathBuf>,
    pub output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Protocol {
    pub device: String,
    pub physical_batch: usize,
    pub decoder_batch: usize,
    pub decoder_gradient_accumulation: usize,
    pub decoder_hidden: usize,
    pub max_optimizer_steps: usize,
    pub evaluate_every_steps: usize,
    pub patience_evaluations: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub parameter_cap: usize,
    pub observable_control_mse_ceiling: f64,
    pub observable_control_min_fractional_reduction: f64,
    pub observable_control_min_absolute_improvement: f64,
    pub model_weights_frozen: bool,
    pub inferential_claims_enabled: bool,
    pub selection_rule: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub struct SelectionPoint {
    pub optimizer_step: usize,
    pub selection_mse: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Qualification {
    pub initial_selection_mse: f64,
    pub best_selection_mse: f64,
    pub fractional_reduction: f64,
    pub selected_optimizer_steps: usize,
    pub reached_step_budget: bool,
    pub decoder_seed: u64,
    pub selection_curve: Vec<SelectionPoint>,
    pub passed: bool,
    pub failure_reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteScore {
    pub route: String,
    pub ridge_selection_mse: Option<f64>,
    pub residual_selection_mse: Option<f64>,
    pub ridge_final_mse: f64,
    pub residual_final_mse: f64,
    pub selected_optimizer_steps: usize,
    pub selection_stopped_optimizer_steps: usize,
    pub selection_converged_before_budget: bool,
    pub final_episode_mse: Vec<EpisodeScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeScore {
    pub source: String,
    pub episode_id: u64,
    pub patch_rows: usize,
    pub mse: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteSelectionDiagnostic {
    pub fit: String,
    pub ridge_selection_mse: f64,
    pub initial_residual_selection_mse: f64,
    pub best_residual_selection_mse: f64,
    pub selected_optimizer_steps: usize,
    pub stopped_optimizer_steps: usize,
    pub converged_before_budget: bool,
    pub decoder_seed: u64,
    pub examples_consumed: usize,
    pub selection_curve: Vec<SelectionPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyReport {
    pub name: String,
    pub input_dim: usize,
    pub parameter_count: usize,
    pub qualification: Qualification,
    pub route_selection_diagnostics: Vec<RouteSelectionDiagnostic>,
    pub routes: Vec<RouteScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub schema: String,
    pub checkpoint: PathBuf,
    pub checkpoint_sha256: String,
    pub train_config: PathBuf,
    pub train_config_sha256: String,
    pub population_seed: u64,
    pub synthetic_episodes_per_source: usize,
    pub population_fingerprint: String,
    pub target_latents_sha256: String,
    pub predicted_latents_sha256: String,
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

struct ProbeMlp {
    first: Linear,
    second: Linear,
}

impl ProbeMlp {
    fn new(input: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            first: linear(input, HIDDEN, vb.pp("first"))?,
            second: linear(HIDDEN, PALETTE_SIZE, vb.pp("second"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.second
            .forward(&self.first.forward(input)?.relu()?)
            .map_err(Into::into)
    }
}

struct Standardization {
    rows: Vec<Vec<f32>>,
    mean: Vec<f32>,
    std: Vec<f32>,
}

struct Selection {
    best_step: usize,
    initial_mse: f64,
    best_mse: f64,
    stopped_step: usize,
    curve: Vec<SelectionPoint>,
}

struct FittedRoute {
    ridge: FixedBoardProbe,
    feature_mean: Vec<f32>,
    feature_std: Vec<f32>,
    residual_scale: [f32; PALETTE_SIZE],
    model: ProbeMlp,
    _varmap: VarMap,
    selection: Selection,
    ridge_selection_mse: f64,
    seed: u64,
}

fn selection_diagnostic(name: &str, fitted: &FittedRoute) -> RouteSelectionDiagnostic {
    RouteSelectionDiagnostic {
        fit: name.into(),
        ridge_selection_mse: fitted.ridge_selection_mse,
        initial_residual_selection_mse: fitted.selection.initial_mse,
        best_residual_selection_mse: fitted.selection.best_mse,
        selected_optimizer_steps: fitted.selection.best_step,
        stopped_optimizer_steps: fitted.selection.stopped_step,
        converged_before_budget: fitted.selection.stopped_step < MAX_STEPS,
        decoder_seed: fitted.seed,
        examples_consumed: fitted.selection.stopped_step * DECODER_BATCH,
        selection_curve: fitted.selection.curve.clone(),
    }
}

pub fn run(cfg: &Config) -> Result<Report> {
    validate_config(cfg)?;
    ensure!(
        !cfg.output.exists(),
        "output already exists: {}",
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
    let population = collect_frozen_board_probe_population_with_predictions(
        &cfg.checkpoint,
        &cfg.train_config,
        POPULATION_SEED,
        SYNTHETIC_EPISODES,
        cfg.physical_batch,
        &cfg.device,
    )?;
    if let Some(forbidden) = &cfg.forbidden_population_fingerprint {
        ensure!(
            &population.population_fingerprint != forbidden,
            "freshness check failed before evaluator fitting: population fingerprint matches forbidden prior audit"
        );
    }
    ensure!(
        population.samples.len() == population.source_by_sample.len(),
        "source labels do not align with frames"
    );
    let target_local = population.target_rows.as_rows().to_vec();
    let predicted_local = population
        .predicted_rows
        .as_ref()
        .context("V1.1 population omitted predicted-next rows")?
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
    let mut row_source = Vec::with_capacity(targets.len());
    let mut row_episode = Vec::with_capacity(targets.len());
    for (frame, sample) in population.samples.iter().enumerate() {
        for _ in 0..PATCH_COUNT {
            row_source.push(population.source_by_sample[frame].clone());
            row_episode.push(sample.episode_id);
        }
    }
    let parts = partitions(&population.samples);
    validate_partitions(&parts)?;
    let train = frame_rows(&parts.train_frames);
    let selection = frame_rows(&parts.selection_frames);
    let final_rows = frame_rows(&parts.final_frames);
    let fit = train.iter().chain(&selection).copied().collect::<Vec<_>>();
    let device = resolve_device(&cfg.device)?;

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
    // Stage A is global and fail-closed: final rows are untouched until every
    // registered family has passed the same ridge-plus-residual fitting path.
    let mut families = Vec::new();
    for (name, target_features, _, _, _) in family_data {
        let seed = stable_seed(POPULATION_SEED, name);
        let qualification = qualify_same_path(
            target_features[0].len(),
            name == "contextual_3x3_global",
            &targets,
            &train,
            &selection,
            &device,
            seed,
        )?;
        families.push(FamilyReport {
            name: name.into(),
            input_dim: target_features[0].len(),
            parameter_count: parameter_count(target_features[0].len()),
            qualification,
            route_selection_diagnostics: Vec::new(),
            routes: Vec::new(),
        });
    }
    let controls_qualified = families.iter().all(|family| family.qualification.passed);
    let mut fitted = Vec::new();
    if controls_qualified {
        for (index, (name, target_features, predicted_features, target_base, predicted_base)) in
            family_data.iter().enumerate()
        {
            // Identical initialization and minibatch randomness for the paired
            // target/predicted refits; only the feature seam differs.
            let seed = stable_seed(stable_seed(POPULATION_SEED, name), "matched-fit");
            let target_fit = fit_residual_route(
                target_features,
                target_base,
                &targets,
                &train,
                &selection,
                &fit,
                &device,
                seed,
                false,
            )?;
            let predicted_fit = fit_residual_route(
                predicted_features,
                predicted_base,
                &targets,
                &train,
                &selection,
                &fit,
                &device,
                seed,
                false,
            )?;
            families[index].route_selection_diagnostics = vec![
                selection_diagnostic("target_next_fit", &target_fit),
                selection_diagnostic("predicted_next_fit", &predicted_fit),
            ];
            fitted.push((target_fit, predicted_fit));
        }
    }
    let routes_converged = controls_qualified
        && fitted.iter().all(|(target, predicted)| {
            target.selection.stopped_step < MAX_STEPS
                && predicted.selection.stopped_step < MAX_STEPS
        });
    let evaluator_status = if !controls_qualified {
        "control_invalid"
    } else if !routes_converged {
        "route_selection_budget_censored"
    } else {
        "qualified"
    };
    if !cfg.selection_only {
        verify_selection_reference(
            cfg.selection_reference
                .as_deref()
                .context("final phase requires a phase-1 selection reference")?,
            &families,
            &population.population_fingerprint,
            &checkpoint_sha256,
            evaluator_status,
        )?;
    }
    // Stage B final access occurs only after both controls and every real-route
    // selection have converged before the fixed optimizer budget.
    if routes_converged && !cfg.selection_only {
        for (
            index,
            (
                (_, target_features, predicted_features, target_base, predicted_base),
                (target_fit, predicted_fit),
            ),
        ) in family_data.iter().zip(&fitted).enumerate()
        {
            families[index].routes = vec![
                score_route(
                    "true_next_encoder_fit",
                    target_fit,
                    target_features,
                    target_base,
                    &targets,
                    &final_rows,
                    &row_source,
                    &row_episode,
                    &device,
                    true,
                )?,
                score_route(
                    "target_fit_transfer_to_predicted_next",
                    target_fit,
                    predicted_features,
                    predicted_base,
                    &targets,
                    &final_rows,
                    &row_source,
                    &row_episode,
                    &device,
                    false,
                )?,
                score_route(
                    "predicted_next_refit",
                    predicted_fit,
                    predicted_features,
                    predicted_base,
                    &targets,
                    &final_rows,
                    &row_source,
                    &row_episode,
                    &device,
                    true,
                )?,
            ];
        }
    }
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
        target_latents_sha256,
        predicted_latents_sha256,
        target: "descriptive_16_colour_counts_per_8x8_patch_status_row_excluded".into(),
        protocol: Protocol {
            device: cfg.device.clone(),
            physical_batch: cfg.physical_batch,
            decoder_batch: DECODER_BATCH,
            decoder_gradient_accumulation: 1,
            decoder_hidden: HIDDEN,
            max_optimizer_steps: MAX_STEPS,
            evaluate_every_steps: EVAL_EVERY,
            patience_evaluations: PATIENCE_EVALS,
            learning_rate: LEARNING_RATE,
            weight_decay: WEIGHT_DECAY,
            parameter_cap: PARAMETER_CAP,
            observable_control_mse_ceiling: CONTROL_MSE_CEILING,
            observable_control_min_fractional_reduction: CONTROL_MIN_REDUCTION,
            observable_control_min_absolute_improvement: CONTROL_MIN_ABSOLUTE_IMPROVEMENT,
            model_weights_frozen: true,
            inferential_claims_enabled: false,
            selection_rule: "episode-disjoint selection; refit for selected optimizer steps; final scored once; no null p-values".into(),
        },
        split: SplitManifest {
            train_frames: parts.train_frames.len(),
            selection_frames: parts.selection_frames.len(),
            final_frames: parts.final_frames.len(),
            train_rows: train.len(),
            selection_rows: selection.len(),
            final_rows: final_rows.len(),
            rule: "final: episode_id%3!=0; selection: episode_id%9==0; train: episode_id%9 in {3,6}".into(),
        },
        evaluator_status: evaluator_status.into(),
        execution_phase: if cfg.selection_only {
            "selection_only"
        } else {
            "final_score"
        }
        .into(),
        families,
        model_weights_updated: false,
        final_partition_used_for_decoder_selection: false,
        final_partition_scored: routes_converged && !cfg.selection_only,
        descriptive_seam_interpretation_permitted: routes_converged && !cfg.selection_only,
        model_level_conclusion_permitted: false,
        next_stage: if routes_converged && cfg.selection_only {
            "run_registered_final_phase_only_after_all_six_arms_qualify"
        } else if routes_converged {
            "run_exact_cell_coordinate_aware_seam_audit_then_choose_model_intervention"
        } else if controls_qualified {
            "extend_or_repair_route_selection_without_accessing_final_rows"
        } else {
            "repair_evaluator_only; no_checkpoint_or_model_conclusion"
        }
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
    ensure!(
        cfg.forbidden_population_fingerprint
            .as_ref()
            .is_some_and(|value| value.starts_with("sha256:")),
        "V1.1 requires the checksum-verified prior population fingerprint"
    );
    ensure!(
        cfg.selection_only || cfg.selection_reference.is_some(),
        "final scoring requires --selection-reference from the all-arm qualification phase"
    );
    Ok(())
}

fn verify_selection_reference(
    path: &Path,
    families: &[FamilyReport],
    population_fingerprint: &str,
    checkpoint_sha256: &str,
    evaluator_status: &str,
) -> Result<()> {
    let reference: Report = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read selection reference {}", path.display()))?,
    )?;
    ensure!(
        reference.schema == SCHEMA
            && reference.execution_phase == "selection_only"
            && !reference.final_partition_scored
            && reference.population_fingerprint == population_fingerprint
            && reference.checkpoint_sha256 == checkpoint_sha256
            && reference.evaluator_status == evaluator_status
            && reference.families.len() == families.len(),
        "selection reference identity or status mismatch"
    );
    for (current, prior) in families.iter().zip(&reference.families) {
        ensure!(
            current.name == prior.name
                && current.qualification == prior.qualification
                && current.route_selection_diagnostics == prior.route_selection_diagnostics,
            "selection diagnostics changed before final scoring for family {}",
            current.name
        );
    }
    Ok(())
}

fn qualify_same_path(
    input_dim: usize,
    contextual: bool,
    targets: &[[f32; PALETTE_SIZE]],
    train: &[usize],
    selection: &[usize],
    device: &Device,
    seed: u64,
) -> Result<Qualification> {
    let local_dim = if contextual { input_dim / 3 } else { input_dim };
    ensure!(
        !contextual || local_dim * 3 == input_dim,
        "contextual control width is not three local views"
    );
    ensure!(
        local_dim >= PALETTE_SIZE + 2,
        "observable control needs two nonlinear channels"
    );
    let mut control_targets = targets.to_vec();
    let observable_local = targets
        .iter()
        .enumerate()
        .map(|(index, target)| {
            let mut row = vec![0.0; local_dim];
            for colour in 0..PALETTE_SIZE.min(local_dim) {
                row[colour] = target[colour] / 64.0;
            }
            let u = 2.0 * (index % 17) as f32 / 16.0 - 1.0;
            let v = 2.0 * (index.wrapping_mul(7) % 19) as f32 / 18.0 - 1.0;
            row[PALETTE_SIZE] = u;
            row[PALETTE_SIZE + 1] = v;
            // The target is fully observable, but the bilinear term cannot be
            // solved by the ridge baseline and must exercise the residual MLP.
            control_targets[index][0] += 8.0 * u * v;
            row
        })
        .collect::<Vec<_>>();
    let observable_features = if contextual {
        contextual_features(&observable_local)?
    } else {
        observable_local.clone()
    };
    let fit = train.iter().chain(selection).copied().collect::<Vec<_>>();
    let fitted = fit_residual_route(
        &observable_features,
        &observable_local,
        &control_targets,
        train,
        selection,
        &fit,
        device,
        seed,
        true,
    )?;
    let selection_result = &fitted.selection;
    let reduction = 1.0 - selection_result.best_mse / selection_result.initial_mse.max(1e-12);
    let mut failure_reasons = Vec::new();
    if !selection_result.best_mse.is_finite() {
        failure_reasons.push("non_finite_score".into());
    }
    if selection_result.best_mse > CONTROL_MSE_CEILING {
        failure_reasons.push("observable_control_above_fixed_mse_ceiling".into());
    }
    if reduction < CONTROL_MIN_REDUCTION {
        failure_reasons.push("observable_control_insufficient_loss_reduction".into());
    }
    if selection_result.initial_mse - selection_result.best_mse < CONTROL_MIN_ABSOLUTE_IMPROVEMENT {
        failure_reasons.push("observable_control_insufficient_absolute_improvement".into());
    }
    if selection_result.best_step == 0 {
        failure_reasons.push("observable_control_did_not_exercise_optimizer".into());
    }
    if selection_result.stopped_step == MAX_STEPS {
        failure_reasons.push("observable_control_did_not_converge_before_step_budget".into());
    }
    Ok(Qualification {
        initial_selection_mse: selection_result.initial_mse,
        best_selection_mse: selection_result.best_mse,
        fractional_reduction: reduction,
        selected_optimizer_steps: selection_result.best_step,
        reached_step_budget: selection_result.stopped_step == MAX_STEPS,
        decoder_seed: seed,
        selection_curve: selection_result.curve.clone(),
        passed: failure_reasons.is_empty(),
        failure_reasons,
    })
}

#[allow(clippy::too_many_arguments)]
fn fit_residual_route(
    features: &[Vec<f32>],
    ridge_features: &[Vec<f32>],
    targets: &[[f32; PALETTE_SIZE]],
    train: &[usize],
    selection: &[usize],
    fit: &[usize],
    device: &Device,
    seed: u64,
    stop_when_control_qualified: bool,
) -> Result<FittedRoute> {
    let train_features = select_rows(features, train);
    let train_ridge_features = select_rows(ridge_features, train);
    let train_targets = select_targets(targets, train);
    let selection_features = select_rows(features, selection);
    let selection_ridge_features = select_rows(ridge_features, selection);
    let selection_targets = select_targets(targets, selection);
    let ridge_select = FixedBoardProbe::fit_histograms(&train_ridge_features, &train_targets)?;
    let train_base = ridge_select.predict_histograms(&train_ridge_features)?;
    let selection_base = ridge_select.predict_histograms(&selection_ridge_features)?;
    let ridge_selection_mse = mse(&selection_base, &selection_targets)?;
    let train_standardized = standardized(&train_features)?;
    let selection_standardized = apply_standardization(
        &selection_features,
        &train_standardized.mean,
        &train_standardized.std,
    );
    let scale = residual_scale(&train_targets, &train_base);
    let selected = select_steps(
        &train_standardized.rows,
        &train_targets,
        &train_base,
        &selection_standardized,
        &selection_targets,
        &selection_base,
        &scale,
        device,
        seed,
        stop_when_control_qualified,
    )?;

    let fit_features = select_rows(features, fit);
    let fit_ridge_features = select_rows(ridge_features, fit);
    let fit_targets = select_targets(targets, fit);
    let ridge = FixedBoardProbe::fit_histograms(&fit_ridge_features, &fit_targets)?;
    let fit_base = ridge.predict_histograms(&fit_ridge_features)?;
    let fit_standardized = standardized(&fit_features)?;
    let fit_scale = residual_scale(&fit_targets, &fit_base);
    let (model, varmap, mut optimizer) = new_mlp(features[0].len(), device, seed)?;
    train_exact_steps(
        &model,
        &mut optimizer,
        &fit_standardized.rows,
        &fit_targets,
        &fit_base,
        &fit_scale,
        DECODER_BATCH,
        device,
        seed,
        selected.best_step,
    )?;
    Ok(FittedRoute {
        ridge,
        feature_mean: fit_standardized.mean,
        feature_std: fit_standardized.std,
        residual_scale: fit_scale,
        model,
        _varmap: varmap,
        selection: selected,
        ridge_selection_mse,
        seed,
    })
}

#[allow(clippy::too_many_arguments)]
fn score_route(
    name: &str,
    fit: &FittedRoute,
    features: &[Vec<f32>],
    ridge_features: &[Vec<f32>],
    targets: &[[f32; PALETTE_SIZE]],
    final_rows: &[usize],
    row_source: &[String],
    row_episode: &[u64],
    device: &Device,
    include_selection: bool,
) -> Result<RouteScore> {
    let final_features = select_rows(features, final_rows);
    let final_ridge_features = select_rows(ridge_features, final_rows);
    let final_targets = select_targets(targets, final_rows);
    let base = fit.ridge.predict_histograms(&final_ridge_features)?;
    let standardized = apply_standardization(&final_features, &fit.feature_mean, &fit.feature_std);
    let predictions = predict_combined(
        &fit.model,
        &standardized,
        &base,
        &fit.residual_scale,
        DECODER_BATCH,
        device,
    )?;
    let residual_final_mse = mse(&predictions, &final_targets)?;
    Ok(RouteScore {
        route: name.into(),
        ridge_selection_mse: include_selection.then_some(fit.ridge_selection_mse),
        residual_selection_mse: include_selection.then_some(fit.selection.best_mse),
        ridge_final_mse: mse(&base, &final_targets)?,
        residual_final_mse,
        selected_optimizer_steps: fit.selection.best_step,
        selection_stopped_optimizer_steps: fit.selection.stopped_step,
        selection_converged_before_budget: fit.selection.stopped_step < MAX_STEPS,
        final_episode_mse: episode_scores(
            final_rows,
            row_source,
            row_episode,
            &predictions,
            &final_targets,
        )?,
    })
}

#[allow(clippy::too_many_arguments)]
fn select_steps(
    train_x: &[Vec<f32>],
    train_y: &[[f32; PALETTE_SIZE]],
    train_base: &[[f32; PALETTE_SIZE]],
    selection_x: &[Vec<f32>],
    selection_y: &[[f32; PALETTE_SIZE]],
    selection_base: &[[f32; PALETTE_SIZE]],
    scale: &[f32; PALETTE_SIZE],
    device: &Device,
    seed: u64,
    stop_when_control_qualified: bool,
) -> Result<Selection> {
    let (model, _varmap, mut optimizer) = new_mlp(train_x[0].len(), device, seed)?;
    let initial_mse = score_combined(
        &model,
        selection_x,
        selection_y,
        selection_base,
        scale,
        DECODER_BATCH,
        device,
    )?;
    let mut best_mse = initial_mse;
    let mut best_step = 0;
    let mut stale = 0usize;
    let mut stopped_step = 0usize;
    let mut curve = vec![SelectionPoint {
        optimizer_step: 0,
        selection_mse: initial_mse,
    }];
    let mut order = (0..train_x.len()).collect::<Vec<_>>();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    order.shuffle(&mut rng);
    let mut cursor = 0usize;
    for step in 1..=MAX_STEPS {
        let indices = next_batch(&mut order, &mut cursor, DECODER_BATCH, &mut rng);
        train_batch(
            &model,
            &mut optimizer,
            train_x,
            train_y,
            train_base,
            scale,
            &indices,
            device,
        )?;
        stopped_step = step;
        if step % EVAL_EVERY == 0 || step == MAX_STEPS {
            let score = score_combined(
                &model,
                selection_x,
                selection_y,
                selection_base,
                scale,
                DECODER_BATCH,
                device,
            )?;
            curve.push(SelectionPoint {
                optimizer_step: step,
                selection_mse: score,
            });
            let meaningful_improvement = (best_mse.abs() * 1e-4).max(1e-8);
            if score + meaningful_improvement < best_mse {
                best_mse = score;
                best_step = step;
                stale = 0;
            } else {
                stale += 1;
            }
            let reduction = 1.0 - best_mse / initial_mse.max(1e-12);
            if stop_when_control_qualified
                && best_mse <= CONTROL_MSE_CEILING
                && reduction >= CONTROL_MIN_REDUCTION
            {
                break;
            }
            if stale >= PATIENCE_EVALS {
                break;
            }
        }
    }
    Ok(Selection {
        best_step,
        initial_mse,
        best_mse,
        stopped_step,
        curve,
    })
}

#[allow(clippy::too_many_arguments)]
fn train_exact_steps(
    model: &ProbeMlp,
    optimizer: &mut AdamW,
    features: &[Vec<f32>],
    targets: &[[f32; PALETTE_SIZE]],
    base: &[[f32; PALETTE_SIZE]],
    scale: &[f32; PALETTE_SIZE],
    batch: usize,
    device: &Device,
    seed: u64,
    steps: usize,
) -> Result<()> {
    let mut order = (0..features.len()).collect::<Vec<_>>();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    order.shuffle(&mut rng);
    let mut cursor = 0usize;
    for _ in 0..steps {
        let indices = next_batch(&mut order, &mut cursor, batch, &mut rng);
        train_batch(
            model, optimizer, features, targets, base, scale, &indices, device,
        )?;
    }
    Ok(())
}

fn next_batch(
    order: &mut [usize],
    cursor: &mut usize,
    batch: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    let count = batch.min(order.len());
    let mut indices = Vec::with_capacity(count);
    while indices.len() < count {
        let available = order.len() - *cursor;
        let take = available.min(count - indices.len());
        indices.extend_from_slice(&order[*cursor..*cursor + take]);
        *cursor += take;
        if *cursor == order.len() && indices.len() < count {
            order.shuffle(rng);
            *cursor = 0;
        }
    }
    indices
}

#[allow(clippy::too_many_arguments)]
fn train_batch(
    model: &ProbeMlp,
    optimizer: &mut AdamW,
    features: &[Vec<f32>],
    targets: &[[f32; PALETTE_SIZE]],
    base: &[[f32; PALETTE_SIZE]],
    scale: &[f32; PALETTE_SIZE],
    indices: &[usize],
    device: &Device,
) -> Result<()> {
    let x = Tensor::from_vec(
        indices
            .iter()
            .flat_map(|&i| features[i].iter().copied())
            .collect::<Vec<_>>(),
        (indices.len(), features[0].len()),
        device,
    )?;
    let normalized = indices
        .iter()
        .flat_map(|&i| (0..PALETTE_SIZE).map(move |c| (targets[i][c] - base[i][c]) / scale[c]))
        .collect::<Vec<_>>();
    let y = Tensor::from_vec(normalized, (indices.len(), PALETTE_SIZE), device)?;
    let loss = candle_nn::loss::mse(&model.forward(&x)?, &y)?;
    let loss_value = loss.to_scalar::<f32>()?;
    ensure!(loss_value.is_finite(), "non-finite decoder training loss");
    optimizer.backward_step(&loss)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn score_combined(
    model: &ProbeMlp,
    features: &[Vec<f32>],
    targets: &[[f32; PALETTE_SIZE]],
    base: &[[f32; PALETTE_SIZE]],
    scale: &[f32; PALETTE_SIZE],
    batch: usize,
    device: &Device,
) -> Result<f64> {
    let predictions = predict_combined(model, features, base, scale, batch, device)?;
    mse(&predictions, targets)
}

fn predict_combined(
    model: &ProbeMlp,
    features: &[Vec<f32>],
    base: &[[f32; PALETTE_SIZE]],
    scale: &[f32; PALETTE_SIZE],
    batch: usize,
    device: &Device,
) -> Result<Vec<[f32; PALETTE_SIZE]>> {
    ensure!(
        !features.is_empty() && features.len() == base.len(),
        "invalid scoring rows"
    );
    let mut predictions = Vec::with_capacity(features.len());
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
        let residual = model
            .forward(&x)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (within_batch, base_row) in base[start..end].iter().enumerate() {
            let mut prediction = [0.0; PALETTE_SIZE];
            for colour in 0..PALETTE_SIZE {
                let offset = within_batch * PALETTE_SIZE + colour;
                prediction[colour] = base_row[colour] + residual[offset] * scale[colour];
            }
            ensure!(
                prediction.iter().all(|value| value.is_finite()),
                "non-finite decoder prediction"
            );
            predictions.push(prediction);
        }
    }
    Ok(predictions)
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
        ensure!(
            index < row_source.len(),
            "episode row index is out of bounds"
        );
        let squared_error = prediction
            .iter()
            .zip(target)
            .map(|(p, t)| {
                let difference = f64::from(*p - *t);
                difference * difference
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

fn new_mlp(input: usize, device: &Device, seed: u64) -> Result<(ProbeMlp, VarMap, AdamW)> {
    ensure!(
        parameter_count(input) <= PARAMETER_CAP,
        "decoder exceeds parameter cap"
    );
    let varmap = VarMap::new();
    let model = ProbeMlp::new(input, VarBuilder::from_varmap(&varmap, DType::F32, device))?;
    reinit_varmap_deterministic(&varmap, seed)?;
    zero_output_layer(&varmap)?;
    let optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            weight_decay: WEIGHT_DECAY,
            ..ParamsAdamW::default()
        },
    )?;
    Ok((model, varmap, optimizer))
}

fn zero_output_layer(varmap: &VarMap) -> Result<()> {
    let data = varmap.data().lock().unwrap();
    let mut matched = 0;
    for (name, var) in data.iter().filter(|(name, _)| name.starts_with("second.")) {
        let zero = Tensor::zeros(var.shape(), var.dtype(), var.device())?;
        var.set(&zero)
            .with_context(|| format!("zero residual output {name}"))?;
        matched += 1;
    }
    ensure!(matched == 2, "expected residual output weight and bias");
    Ok(())
}

fn residual_scale(
    targets: &[[f32; PALETTE_SIZE]],
    base: &[[f32; PALETTE_SIZE]],
) -> [f32; PALETTE_SIZE] {
    let mut sum_sq = [0.0f64; PALETTE_SIZE];
    for (target, prediction) in targets.iter().zip(base) {
        for colour in 0..PALETTE_SIZE {
            let residual = f64::from(target[colour] - prediction[colour]);
            sum_sq[colour] += residual * residual;
        }
    }
    let mut scale = [0.0; PALETTE_SIZE];
    for colour in 0..PALETTE_SIZE {
        scale[colour] = (sum_sq[colour] / targets.len().max(1) as f64)
            .sqrt()
            .max(1e-3) as f32;
    }
    scale
}

fn standardized(features: &[Vec<f32>]) -> Result<Standardization> {
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
            let difference = f64::from(*value) - mean[index];
            std[index] += difference * difference;
        }
    }
    for value in &mut std {
        *value = (*value / features.len() as f64).sqrt().max(1e-6);
    }
    let mean = mean
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    let std = std
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    Ok(Standardization {
        rows: apply_standardization(features, &mean, &std),
        mean,
        std,
    })
}

fn apply_standardization(features: &[Vec<f32>], mean: &[f32], std: &[f32]) -> Vec<Vec<f32>> {
    features
        .iter()
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
        .flat_map(|(prediction, target)| prediction.iter().zip(target))
        .map(|(prediction, target)| {
            let difference = f64::from(*prediction - *target);
            difference * difference
        })
        .sum::<f64>()
        / (predictions.len() * PALETTE_SIZE) as f64)
}

fn parameter_count(input: usize) -> usize {
    input * HIDDEN + HIDDEN + HIDDEN * PALETTE_SIZE + PALETTE_SIZE
}

fn stable_seed(seed: u64, name: &str) -> u64 {
    name.bytes()
        .fold(seed ^ 0x9e37_79b9_7f4a_7c15, |hash, byte| {
            hash.wrapping_mul(1_099_511_628_211)
                .wrapping_add(u64::from(byte))
        })
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

fn write_json_create_new(path: &Path, report: &Report) -> Result<()> {
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
    file.write_all(&serde_json::to_vec_pretty(report)?)?;
    file.sync_all()?;
    fs::rename(staging, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contextual_width_and_count_are_stable() -> Result<()> {
        let rows = (0..PATCH_COUNT)
            .map(|index| vec![index as f32, 1.0])
            .collect::<Vec<_>>();
        let contextual = contextual_features(&rows)?;
        assert_eq!(contextual.len(), PATCH_COUNT);
        assert_eq!(contextual[0].len(), 6);
        Ok(())
    }

    #[test]
    fn both_sealed_decoder_families_fit_under_cap() {
        assert!(parameter_count(128) < PARAMETER_CAP);
        assert!(parameter_count(128 * 3) < PARAMETER_CAP);
    }

    #[test]
    fn residual_scale_never_collapses() {
        let targets = vec![[1.0; PALETTE_SIZE]; 3];
        assert_eq!(residual_scale(&targets, &targets), [1e-3; PALETTE_SIZE]);
    }

    #[test]
    fn same_path_observable_control_qualifies_on_cpu() -> Result<()> {
        let targets = (0..512)
            .map(|index| {
                let mut target = [0.0; PALETTE_SIZE];
                target[index % PALETTE_SIZE] = 32.0;
                target[(index * 7 + 3) % PALETTE_SIZE] += 32.0;
                target
            })
            .collect::<Vec<_>>();
        let train = (0..384).collect::<Vec<_>>();
        let selection = (384..512).collect::<Vec<_>>();
        let qualification =
            qualify_same_path(128, false, &targets, &train, &selection, &Device::Cpu, 17)?;
        assert!(qualification.passed, "{qualification:?}");
        let contextual =
            qualify_same_path(384, true, &targets, &train, &selection, &Device::Cpu, 19)?;
        assert!(contextual.passed, "{contextual:?}");
        Ok(())
    }
}
