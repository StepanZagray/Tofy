//! Bounded non-ARC prerequisite probe for the online visible-effect learner.
//!
//! This measures whether action conditioning helps predict local procedural
//! pixel changes. It does not evaluate ARC games or imply an ARC performance gain.

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, ValueEnum};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use tofy::p2::data::{ArcAction, ArcFrame, FRAME_SIDE};
use tofy::p2::online_effect::{
    ChangePrediction, EffectMetrics, EffectUpdate, OnlineEffectConfig, OnlineEffectLearner,
};
use tofy::p2::train::resolve_device;

const HIDDEN_CHANNELS: usize = 16;
const REPLAY_CAPACITY: usize = 256;
const LEARNING_RATE: f64 = 1e-3;
const HELD_OUT_FRAMES: usize = 64;
const SMOKE_HELD_OUT_FRAMES: usize = 2;
const FRAME_PIXELS: usize = FRAME_SIDE * FRAME_SIDE;
const TRAIN_STREAM_TAG: u64 = 0x5452_4149_4E5F_7631;
const HELD_OUT_STREAM_TAG: u64 = 0x4845_4C44_5F76_3101;
const GENERATOR_SCHEMA: &str = "online-effect-procedural-v1";
const COMPARATOR_ID: &str = "constant-action5-input-same-seed-stream-v1";
const COPY_ZERO_ID: &str = "fixed-copy-zero-change-map-v1";
const ORACLE_ID: &str = "factual-change-map-oracle-v1";
const INITIAL_PROBE_ID: &str = "held-out-true-actions-before-training-v1";
const HELD_OUT_PREDICTIONS_ID: &str = "online-effect-held-out-predictions-v1";
const HELD_OUT_PREDICTIONS_FILE: &str = "held-out-predictions.jsonl";
const EMBEDDED_SOURCE_REVISION: &str = env!("TOFY_EMBEDDED_SOURCE_REVISION");
const EMBEDDED_SOURCE_DIRTY: &str = env!("TOFY_EMBEDDED_SOURCE_DIRTY");
const EMBEDDED_SOURCE_PUSHED: &str = env!("TOFY_EMBEDDED_SOURCE_PUSHED");
const EMBEDDED_CANDLE_GRAPH_REVISION: &str = env!("TOFY_EMBEDDED_CANDLE_GRAPH_REVISION");
const EMBEDDED_CANDLE_GRAPH_DIRTY: &str = env!("TOFY_EMBEDDED_CANDLE_GRAPH_DIRTY");
const EMBEDDED_CANDLE_GRAPH_PUSHED: &str = env!("TOFY_EMBEDDED_CANDLE_GRAPH_PUSHED");
const EMBEDDED_CARGO_FEATURES: &str = env!("TOFY_EMBEDDED_CARGO_FEATURES");
const EMBEDDED_BUILD_COMMAND: &str = env!("TOFY_EMBEDDED_BUILD_COMMAND");

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Arm {
    Conditioned,
    Constant,
}

impl Arm {
    fn as_str(self) -> &'static str {
        match self {
            Self::Conditioned => "conditioned",
            Self::Constant => "constant",
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "online-effect-probe",
    about = "Train and evaluate the online effect learner on a bounded procedural stream"
)]
struct Args {
    /// Candle device: cpu, cuda, or cuda:N.
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// Fresh model initialization seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Procedural generator seed, independent of the model seed.
    #[arg(long, default_value_t = 0)]
    data_seed: u64,

    /// Number of online observations.
    #[arg(long, default_value_t = 256)]
    steps: usize,

    /// Maximum physical replay batch; the effective batch ramps from one.
    #[arg(long, default_value_t = 64)]
    batch: usize,

    /// Optimizer updates over the same selected replay batch per observation.
    #[arg(long, default_value_t = 1)]
    updates_per_observation: usize,

    /// Actual actions, or constant ACTION5 inputs with unchanged true targets.
    #[arg(long, value_enum)]
    arm: Arm,

    /// New output directory; an existing path is rejected.
    #[arg(long, value_name = "NEW")]
    output_dir: PathBuf,

    /// Run a two-update, two-frame implementation fixture.
    #[arg(long)]
    smoke: bool,

    /// Save raw held-out predictions and factual changed-pixel indices as JSONL.
    #[arg(long)]
    save_heldout_predictions: bool,

    /// Hard wall-clock bound for generation, training, and evaluation.
    #[arg(long, default_value_t = 300)]
    max_seconds: u64,
}

#[derive(Clone, Copy)]
struct Direction {
    dx: i8,
    dy: i8,
}

#[derive(Clone, Copy)]
struct FrameSpec {
    background: u8,
    object: u8,
    center_x: u8,
    center_y: u8,
}

struct Transition {
    current: ArcFrame,
    action: ArcAction,
    next: ArcFrame,
}

struct HeldOutFrame {
    current: ArcFrame,
    outcomes: Vec<Transition>,
}

struct GeneratedData {
    movement: [Direction; 4],
    background: u8,
    object: u8,
    training: Vec<Transition>,
    held_out: Vec<HeldOutFrame>,
    sha256: String,
}

#[derive(Default)]
struct Confusion {
    true_positive: u64,
    true_negative: u64,
    false_positive: u64,
    false_negative: u64,
}

impl Confusion {
    fn total(&self) -> u64 {
        self.true_positive + self.true_negative + self.false_positive + self.false_negative
    }

    fn changed(&self) -> u64 {
        self.true_positive + self.false_negative
    }

    fn predicted_changed(&self) -> u64 {
        self.true_positive + self.false_positive
    }

    fn precision(&self) -> Option<f64> {
        (self.predicted_changed() > 0)
            .then(|| self.true_positive as f64 / self.predicted_changed() as f64)
    }

    fn recall(&self) -> Option<f64> {
        (self.changed() > 0).then(|| self.true_positive as f64 / self.changed() as f64)
    }

    fn f1(&self) -> Option<f64> {
        let denominator = 2 * self.true_positive + self.false_positive + self.false_negative;
        (denominator > 0).then(|| 2.0 * self.true_positive as f64 / denominator as f64)
    }

    fn accuracy(&self) -> f64 {
        (self.true_positive + self.true_negative) as f64 / self.total() as f64
    }
}

struct Evaluation {
    value: Value,
    prediction_sha256: String,
}

#[derive(Default)]
struct ActionEvaluation {
    confusion: Confusion,
    changed_tuples: u64,
    noop_tuples: u64,
    noop_false_positive_pixels: u64,
    noop_pixels: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    ensure!(
        !args.output_dir.exists(),
        "--output-dir must be new: {}",
        args.output_dir.display()
    );
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create output directory {}", args.output_dir.display()))?;

    let started = Instant::now();
    let exact_args = std::env::args().collect::<Vec<_>>();
    let provenance = provenance();
    match run(&args, &exact_args, &provenance, started) {
        Ok(report) => {
            write_json(&args.output_dir.join("report.json"), &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Err(error) => {
            let report = json!({
                "schema": "online-effect-probe-report-v1",
                "status": "failed",
                "evidence_class": if args.smoke { "implementation_smoke" } else { "prerequisite_probe" },
                "error": format!("{error:#}"),
                "elapsed_seconds": started.elapsed().as_secs_f64(),
                "exact_args": exact_args,
                "device_requested": args.device,
                "model_seed": args.seed,
                "data_seed": args.data_seed,
                "arm": args.arm.as_str(),
                "optimizer_updates_per_observation": args.updates_per_observation,
                "save_heldout_predictions": args.save_heldout_predictions,
                "held_out_predictions_artifact_identity": HELD_OUT_PREDICTIONS_ID,
                "provenance": provenance,
            });
            write_json(&args.output_dir.join("report.json"), &report)
                .context("write failed run status")?;
            Err(error)
        }
    }
}

fn validate_args(args: &Args) -> Result<()> {
    ensure!(args.steps > 0, "--steps must be greater than zero");
    ensure!((1..=64).contains(&args.batch), "--batch must be in 1..=64");
    ensure!(
        (1..=16).contains(&args.updates_per_observation),
        "--updates-per-observation must be in 1..=16"
    );
    ensure!(
        args.max_seconds > 0,
        "--max-seconds must be greater than zero"
    );
    Ok(())
}

fn run(args: &Args, exact_args: &[String], provenance: &Value, started: Instant) -> Result<Value> {
    validate_launch_integrity(args)?;
    let deadline = Duration::from_secs(args.max_seconds);
    let steps = if args.smoke { 2 } else { args.steps };
    let held_out_frames = if args.smoke {
        SMOKE_HELD_OUT_FRAMES
    } else {
        HELD_OUT_FRAMES
    };
    let expected_optimizer_updates = steps
        .checked_mul(args.updates_per_observation)
        .context("expected optimizer update count overflow")?;
    let data = generate_data(args.data_seed, steps, held_out_frames)?;
    check_deadline(started, deadline, "procedural generation")?;

    let metadata = json!({
        "schema": "online-effect-probe-metadata-v1",
        "claim_boundary": "action conditioning improves local visible-change prediction on this non-ARC procedural generator",
        "evidence_class": if args.smoke { "implementation_smoke" } else { "prerequisite_probe" },
        "exact_args": exact_args,
        "device_requested": args.device,
        "model_seed": args.seed,
        "data_seed": args.data_seed,
        "arm": args.arm.as_str(),
        "generator_schema": GENERATOR_SCHEMA,
        "generated_data_sha256": data.sha256,
        "generated_data_hash_excludes_arm": true,
        "generated_data_hash_excludes_optimizer_updates_per_observation": true,
        "generator_seed_construction": seed_construction(args.data_seed),
        "environment_palette": {"background": data.background, "object": data.object},
        "position_split": {"training_center_parity": "even", "held_out_center_parity": "odd"},
        "movement_action_permutation": movement_json(&data.movement),
        "training_steps": steps,
        "expected_optimizer_updates": expected_optimizer_updates,
        "held_out_frames": held_out_frames,
        "held_out_action_tuples": held_out_frames * 7,
        "save_heldout_predictions": args.save_heldout_predictions,
        "held_out_predictions_artifact": held_out_artifact_declaration(args),
        "model": {
            "hidden_channels": HIDDEN_CHANNELS,
            "replay_capacity": REPLAY_CAPACITY,
            "physical_batch": args.batch,
            "optimizer_updates_per_observation": args.updates_per_observation,
            "learning_rate": LEARNING_RATE,
            "gradient_accumulation": 1,
            "gradient_clipping": null,
        },
        "comparator_identity": COMPARATOR_ID,
        "copy_zero_identity": COPY_ZERO_ID,
        "initial_prediction_probe_identity": INITIAL_PROBE_ID,
        "initial_prediction_probe_uses_true_actions": true,
        "max_seconds": args.max_seconds,
        "provenance": provenance,
    });
    write_json(&args.output_dir.join("metadata.json"), &metadata)?;

    let device = resolve_device(&args.device)?;
    let device_resolved = format!("{device:?}");
    let config = OnlineEffectConfig {
        hidden_channels: HIDDEN_CHANNELS,
        replay_capacity: REPLAY_CAPACITY,
        physical_batch: args.batch,
        optimizer_updates_per_observation: args.updates_per_observation,
        learning_rate: LEARNING_RATE,
    };
    let mut learner = OnlineEffectLearner::new(args.seed, &device, config)?;
    check_deadline(started, deadline, "learner initialization")?;

    let initial_prediction_sha256 = prediction_fingerprint(
        &learner,
        &data.held_out,
        Arm::Conditioned,
        started,
        deadline,
    )?;
    let updates_path = args.output_dir.join("updates.jsonl");
    let mut updates = BufWriter::new(
        File::create(&updates_path)
            .with_context(|| format!("create update log {}", updates_path.display()))?,
    );
    let mut prequential = Confusion::default();
    let mut prequential_bce_sum = 0.0;
    let mut training_replaced_tuples = 0u64;

    for (index, transition) in data.training.iter().enumerate() {
        check_deadline(started, deadline, "online training")?;
        let model_action = model_action(args.arm, &transition.action)?;
        let input_replaced = model_action != transition.action;
        training_replaced_tuples += u64::from(input_replaced);
        let update_started = Instant::now();
        let observed = learner.observe(&transition.current, &model_action, &transition.next)?;
        check_deadline(started, deadline, "online training")?;
        ensure!(
            observed.updates.len() == args.updates_per_observation,
            "learner returned {} optimizer updates for one observation, expected {}",
            observed.updates.len(),
            args.updates_per_observation
        );
        accumulate_metrics(&mut prequential, &observed.pre_update_metrics);
        prequential_bce_sum += observed.pre_update_metrics.balanced_bce;
        let row = json!({
            "event": "update",
            "stream_index": index,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "update_seconds": update_started.elapsed().as_secs_f64(),
            "arm": args.arm.as_str(),
            "true_action": action_json(&transition.action),
            "model_input_action": action_json(&model_action),
            "model_input_actually_replaced": input_replaced,
            "prequential": effect_metrics_json(&observed.pre_update_metrics),
            "optimizer": {
                "statistic": "final_update_for_observation",
                "update": observed.update.update,
                "batch_size": observed.update.batch_size,
                "loss": observed.update.loss,
                "changed_pixels": observed.update.changed_pixels,
                "unchanged_pixels": observed.update.unchanged_pixels,
                "changed_weight": observed.update.changed_weight,
                "unchanged_weight": observed.update.unchanged_weight,
                "gradient_l2": observed.update.gradient_l2,
            },
            "optimizer_updates_for_observation": observed
                .updates
                .iter()
                .map(effect_update_json)
                .collect::<Vec<_>>(),
            "totals": {
                "observations": observed.totals.observations,
                "optimizer_updates": observed.totals.optimizer_updates,
                "replay_len": observed.totals.replay_len,
                "observed_changed_pixels": observed.totals.observed_changed_pixels,
                "observed_unchanged_pixels": observed.totals.observed_unchanged_pixels,
            },
        });
        serde_json::to_writer(&mut updates, &row)?;
        updates.write_all(b"\n")?;
        updates.flush()?;
    }

    let predictions_path = args.output_dir.join(HELD_OUT_PREDICTIONS_FILE);
    let mut predictions_writer = args
        .save_heldout_predictions
        .then(|| {
            File::create(&predictions_path)
                .map(BufWriter::new)
                .with_context(|| {
                    format!(
                        "create held-out prediction artifact {}",
                        predictions_path.display()
                    )
                })
        })
        .transpose()?;
    let capture = predictions_writer
        .as_mut()
        .map(|writer| writer as &mut dyn Write);
    let mut evaluation = evaluate(
        &learner,
        &data.held_out,
        args.arm,
        started,
        deadline,
        capture,
    )?;
    if let Some(writer) = predictions_writer.as_mut() {
        writer
            .flush()
            .context("flush held-out prediction artifact")?;
    }
    drop(predictions_writer);
    let predictions_artifact = if args.save_heldout_predictions {
        json!({
            "identity": HELD_OUT_PREDICTIONS_ID,
            "path": HELD_OUT_PREDICTIONS_FILE,
            "enabled": true,
            "rows": data.held_out.len() * 7,
            "sha256": file_sha256(&predictions_path)?,
            "input_arm_identity": input_arm_identity(args.arm),
        })
    } else {
        held_out_artifact_declaration(args)
    };
    let oracle = evaluate_with_predictor(
        &data.held_out,
        Arm::Conditioned,
        started,
        deadline,
        None,
        oracle_predictions,
    )?;
    let copy_zero = evaluate_with_predictor(
        &data.held_out,
        Arm::Conditioned,
        started,
        deadline,
        None,
        |_, actions| Ok(copy_zero_predictions(actions)),
    )?;
    let held_out_object = evaluation
        .value
        .as_object_mut()
        .context("held-out evaluation is not a JSON object")?;
    held_out_object.insert(
        "oracle_positive_control".into(),
        control_json(ORACLE_ID, oracle.value),
    );
    held_out_object.insert(
        "copy_zero_negative_control".into(),
        control_json(COPY_ZERO_ID, copy_zero.value),
    );
    let totals = learner.totals();
    ensure!(
        totals.observations == steps as u64
            && totals.optimizer_updates == expected_optimizer_updates as u64,
        "learner totals do not match requested updates"
    );
    ensure!(
        totals.replay_len == steps.min(REPLAY_CAPACITY),
        "learner replay length does not match bounded stream"
    );
    let report = json!({
        "schema": "online-effect-probe-report-v1",
        "status": "complete",
        "evidence_class": if args.smoke { "implementation_smoke" } else { "prerequisite_probe" },
        "claim_boundary": "action conditioning improves local visible-change prediction on this non-ARC procedural generator",
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "exact_args": exact_args,
        "device_requested": args.device,
        "device_resolved": device_resolved,
        "model_seed": args.seed,
        "data_seed": args.data_seed,
        "arm": args.arm.as_str(),
        "training_steps": steps,
        "physical_batch": args.batch,
        "optimizer_updates_per_observation": args.updates_per_observation,
        "expected_optimizer_updates": expected_optimizer_updates,
        "generated_data_sha256": data.sha256,
        "generated_data_hash_excludes_arm": true,
        "generated_data_hash_excludes_optimizer_updates_per_observation": true,
        "generator_seed_construction": seed_construction(args.data_seed),
        "environment_palette": {"background": data.background, "object": data.object},
        "position_split": {"training_center_parity": "even", "held_out_center_parity": "odd"},
        "initial_prediction_sha256": initial_prediction_sha256,
        "initial_prediction_probe_identity": INITIAL_PROBE_ID,
        "initial_prediction_probe_uses_true_actions": true,
        "final_prediction_sha256": evaluation.prediction_sha256,
        "training_model_input_actually_replaced_tuples": training_replaced_tuples,
        "save_heldout_predictions": args.save_heldout_predictions,
        "held_out_predictions_artifact": predictions_artifact,
        "prequential": {
            "balanced_bce_mean": prequential_bce_sum / steps as f64,
            "confusion": confusion_json(&prequential),
        },
        "learner_totals": {
            "observations": totals.observations,
            "optimizer_updates": totals.optimizer_updates,
            "replay_len": totals.replay_len,
            "observed_changed_pixels": totals.observed_changed_pixels,
            "observed_unchanged_pixels": totals.observed_unchanged_pixels,
        },
        "held_out": evaluation.value,
        "comparator_identity": COMPARATOR_ID,
        "copy_zero_identity": COPY_ZERO_ID,
        "provenance": provenance,
        "limitations": [
            "non-ARC procedural frames only",
            "one fixed movement permutation per data seed",
            "local one-step visible effects only",
            "no reward, task value, planning, or public-level performance claim",
            "smoke mode uses two updates and two held-out frames",
        ],
    });
    ensure_finite_json(&report)?;
    Ok(report)
}

fn generate_data(data_seed: u64, steps: usize, held_out_frames: usize) -> Result<GeneratedData> {
    let mut environment_rng = ChaCha8Rng::seed_from_u64(data_seed);
    let background = environment_rng.random_range(0..16u8);
    let mut object = environment_rng.random_range(0..15u8);
    if object >= background {
        object += 1;
    }
    let mut movement = [
        Direction { dx: 0, dy: -1 },
        Direction { dx: 0, dy: 1 },
        Direction { dx: -1, dy: 0 },
        Direction { dx: 1, dy: 0 },
    ];
    movement.shuffle(&mut environment_rng);

    let mut train_rng = ChaCha8Rng::seed_from_u64(data_seed ^ TRAIN_STREAM_TAG);
    let mut training = Vec::with_capacity(steps);
    for step in 0..steps {
        let spec = random_frame_spec(&mut train_rng, background, object, 0)?;
        let action_id = (step % 7 + 1) as u8;
        let click_inside = (step / 7) % 2 == 0;
        training.push(transition(spec, action_id, click_inside, &movement)?);
    }

    let mut held_out_rng = ChaCha8Rng::seed_from_u64(data_seed ^ HELD_OUT_STREAM_TAG);
    let mut held_out = Vec::with_capacity(held_out_frames);
    for frame_index in 0..held_out_frames {
        let spec = random_frame_spec(&mut held_out_rng, background, object, 1)?;
        let current = render(spec)?;
        let outcomes = (1..=7)
            .map(|action_id| transition(spec, action_id, frame_index % 2 == 0, &movement))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            outcomes.iter().all(|outcome| outcome.current == current),
            "held-out outcomes do not share a current frame"
        );
        held_out.push(HeldOutFrame { current, outcomes });
    }

    let sha256 = generated_data_hash(background, object, &movement, &training, &held_out);
    Ok(GeneratedData {
        movement,
        background,
        object,
        training,
        held_out,
        sha256,
    })
}

fn random_frame_spec(
    rng: &mut ChaCha8Rng,
    background: u8,
    object: u8,
    center_parity: u8,
) -> Result<FrameSpec> {
    for _ in 0..128 {
        let center_x = rng.random_range(2..62u8);
        let center_y = rng.random_range(2..62u8);
        if (center_x + center_y) % 2 == center_parity {
            return Ok(FrameSpec {
                background,
                object,
                center_x,
                center_y,
            });
        }
    }
    bail!("failed to sample center parity {center_parity} within 128 draws")
}

fn transition(
    spec: FrameSpec,
    action_id: u8,
    click_inside: bool,
    movement: &[Direction; 4],
) -> Result<Transition> {
    let current = render(spec)?;
    let action = if action_id == 6 {
        let (x, y) = if click_inside {
            (spec.center_x, spec.center_y)
        } else {
            (0, 0)
        };
        ArcAction::new(6, Some(x), Some(y))?
    } else {
        ArcAction::new(action_id, None, None)?
    };
    let next = match action_id {
        1..=4 => {
            let direction = movement[usize::from(action_id - 1)];
            render(FrameSpec {
                center_x: (spec.center_x as i16 + i16::from(direction.dx)) as u8,
                center_y: (spec.center_y as i16 + i16::from(direction.dy)) as u8,
                ..spec
            })?
        }
        5 | 7 => current.clone(),
        6 => {
            let mut pixels = current.pixels.to_vec();
            let x = action.x.context("generated ACTION6 lacks x")?;
            let y = action.y.context("generated ACTION6 lacks y")?;
            if x.abs_diff(spec.center_x) <= 1 && y.abs_diff(spec.center_y) <= 1 {
                pixels[usize::from(y) * FRAME_SIDE + usize::from(x)] = spec.background;
            }
            ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, pixels)?
        }
        _ => bail!("generator action {action_id} is outside 1..=7"),
    };
    Ok(Transition {
        current,
        action,
        next,
    })
}

fn render(spec: FrameSpec) -> Result<ArcFrame> {
    let mut pixels = vec![spec.background; FRAME_PIXELS];
    for y in spec.center_y - 1..=spec.center_y + 1 {
        for x in spec.center_x - 1..=spec.center_x + 1 {
            pixels[usize::from(y) * FRAME_SIDE + usize::from(x)] = spec.object;
        }
    }
    ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, pixels)
}

fn model_action(arm: Arm, actual: &ArcAction) -> Result<ArcAction> {
    match arm {
        Arm::Conditioned => Ok(actual.clone()),
        Arm::Constant => ArcAction::new(5, None, None),
    }
}

fn input_arm_identity(arm: Arm) -> &'static str {
    match arm {
        Arm::Conditioned => "actual-action-tuple-v1",
        Arm::Constant => COMPARATOR_ID,
    }
}

fn held_out_artifact_declaration(args: &Args) -> Value {
    json!({
        "identity": HELD_OUT_PREDICTIONS_ID,
        "path": HELD_OUT_PREDICTIONS_FILE,
        "enabled": args.save_heldout_predictions,
        "input_arm_identity": input_arm_identity(args.arm),
    })
}

fn prediction_fingerprint(
    learner: &OnlineEffectLearner,
    held_out: &[HeldOutFrame],
    arm: Arm,
    started: Instant,
    deadline: Duration,
) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(b"online-effect-predictions-v1");
    for frame in held_out {
        check_deadline(started, deadline, "prediction fingerprint")?;
        let actions = frame
            .outcomes
            .iter()
            .map(|outcome| model_action(arm, &outcome.action))
            .collect::<Result<Vec<_>>>()?;
        let predictions = learner.predict_batch(&frame.current, &actions)?;
        hash_predictions(&mut hasher, &predictions)?;
    }
    Ok(hex_digest(hasher.finalize()))
}

fn evaluate(
    learner: &OnlineEffectLearner,
    held_out: &[HeldOutFrame],
    arm: Arm,
    started: Instant,
    deadline: Duration,
    capture: Option<&mut dyn Write>,
) -> Result<Evaluation> {
    evaluate_with_predictor(
        held_out,
        arm,
        started,
        deadline,
        capture,
        |frame, actions| learner.predict_batch(&frame.current, actions),
    )
}

fn evaluate_with_predictor<F>(
    held_out: &[HeldOutFrame],
    arm: Arm,
    started: Instant,
    deadline: Duration,
    mut capture: Option<&mut dyn Write>,
    mut predictor: F,
) -> Result<Evaluation>
where
    F: FnMut(&HeldOutFrame, &[ArcAction]) -> Result<Vec<ChangePrediction>>,
{
    let mut confusion = Confusion::default();
    let mut no_op_false_positives = 0u64;
    let mut no_op_pixels = 0u64;
    let mut changed_loss = 0.0;
    let mut unchanged_loss = 0.0;
    let mut changed_pixels = 0u64;
    let mut unchanged_pixels = 0u64;
    let mut changed_tuples = 0u64;
    let mut noop_tuples = 0u64;
    let mut ranking_pairs = 0u64;
    let mut ranking_true_outcome_different_pairs = 0u64;
    let mut ranking_points = 0.0;
    let mut action_evaluations: [ActionEvaluation; 7] =
        std::array::from_fn(|_| ActionEvaluation::default());
    let mut held_out_replaced_tuples = 0u64;
    let mut fingerprint = Sha256::new();
    fingerprint.update(b"online-effect-predictions-v1");

    for (frame_index, frame) in held_out.iter().enumerate() {
        check_deadline(started, deadline, "held-out evaluation")?;
        let actions = frame
            .outcomes
            .iter()
            .map(|outcome| model_action(arm, &outcome.action))
            .collect::<Result<Vec<_>>>()?;
        let predictions = predictor(frame, &actions)?;
        ensure!(
            predictions.len() == frame.outcomes.len(),
            "held-out prediction count mismatch"
        );
        hash_predictions(&mut fingerprint, &predictions)?;
        if let Some(writer) = capture.as_deref_mut() {
            write_prediction_rows(writer, frame_index, frame, &actions, &predictions, arm)?;
        }

        let mut changed_scores = Vec::new();
        let mut noop_scores = Vec::new();
        for (outcome, prediction) in frame.outcomes.iter().zip(&predictions) {
            let action_evaluation = &mut action_evaluations[usize::from(outcome.action.id - 1)];
            if prediction.action != outcome.action {
                held_out_replaced_tuples += 1;
            }
            ensure!(
                prediction.probabilities.len() == FRAME_PIXELS,
                "held-out prediction has {} pixels, expected {FRAME_PIXELS}",
                prediction.probabilities.len()
            );
            let target_changed = outcome
                .current
                .pixels
                .iter()
                .zip(outcome.next.pixels.iter())
                .any(|(before, after)| before != after);
            let score = prediction
                .probabilities
                .iter()
                .map(|&probability| f64::from(probability))
                .sum::<f64>();
            if target_changed {
                changed_tuples += 1;
                action_evaluation.changed_tuples += 1;
                changed_scores.push((score, &outcome.next));
            } else {
                noop_tuples += 1;
                action_evaluation.noop_tuples += 1;
                noop_scores.push((score, &outcome.next));
            }

            for ((before, after), &probability) in outcome
                .current
                .pixels
                .iter()
                .zip(outcome.next.pixels.iter())
                .zip(&prediction.probabilities)
            {
                ensure!(
                    probability.is_finite() && (0.0..=1.0).contains(&probability),
                    "held-out probability is outside finite [0,1]"
                );
                let target = before != after;
                let predicted = probability >= 0.5;
                match (predicted, target) {
                    (true, true) => {
                        confusion.true_positive += 1;
                        action_evaluation.confusion.true_positive += 1;
                    }
                    (false, false) => {
                        confusion.true_negative += 1;
                        action_evaluation.confusion.true_negative += 1;
                    }
                    (true, false) => {
                        confusion.false_positive += 1;
                        action_evaluation.confusion.false_positive += 1;
                    }
                    (false, true) => {
                        confusion.false_negative += 1;
                        action_evaluation.confusion.false_negative += 1;
                    }
                }
                let p = f64::from(probability).clamp(1e-12, 1.0 - 1e-12);
                if target {
                    changed_pixels += 1;
                    changed_loss -= p.ln();
                } else {
                    unchanged_pixels += 1;
                    unchanged_loss -= (1.0 - p).ln();
                    if !target_changed {
                        no_op_pixels += 1;
                        action_evaluation.noop_pixels += 1;
                        if predicted {
                            no_op_false_positives += 1;
                            action_evaluation.noop_false_positive_pixels += 1;
                        }
                    }
                }
            }
        }
        for (changed_score, changed_next) in &changed_scores {
            for (noop_score, noop_next) in &noop_scores {
                ranking_pairs += 1;
                if changed_next.pixels == noop_next.pixels {
                    continue;
                }
                ranking_true_outcome_different_pairs += 1;
                ranking_points += if changed_score > noop_score {
                    1.0
                } else if changed_score == noop_score {
                    0.5
                } else {
                    0.0
                };
            }
        }
    }

    ensure!(changed_pixels > 0, "held-out set has no changed pixels");
    ensure!(unchanged_pixels > 0, "held-out set has no unchanged pixels");
    ensure!(no_op_pixels > 0, "held-out set has no no-op pixels");
    ensure!(ranking_pairs > 0, "held-out set has no ranking pairs");
    ensure!(
        ranking_true_outcome_different_pairs > 0,
        "held-out set has no true outcome-different ranking pairs"
    );
    let balanced_bce =
        0.5 * (changed_loss / changed_pixels as f64 + unchanged_loss / unchanged_pixels as f64);
    let ranking_accuracy = ranking_points / ranking_true_outcome_different_pairs as f64;
    let no_op_false_positive_pixel_rate = no_op_false_positives as f64 / no_op_pixels as f64;
    ensure!(
        balanced_bce.is_finite()
            && ranking_accuracy.is_finite()
            && no_op_false_positive_pixel_rate.is_finite(),
        "held-out metrics are non-finite"
    );

    let value = json!({
        "frames": held_out.len(),
        "total_action_tuples": changed_tuples + noop_tuples,
        "eligible_action_tuples": changed_tuples + noop_tuples,
        "genuinely_changed_tuples": changed_tuples,
        "noop_tuples": noop_tuples,
        "outcome_different_tuples_vs_action5": changed_tuples,
        "changed_vs_noop_ranking_eligible_pairs": ranking_pairs,
        "true_outcome_different_changed_vs_noop_action_pairs": ranking_true_outcome_different_pairs,
        "model_input_actually_replaced_tuples": held_out_replaced_tuples,
        "micro_change_f1": confusion.f1(),
        "micro_change_precision": confusion.precision(),
        "micro_change_recall": confusion.recall(),
        "pixel_accuracy": confusion.accuracy(),
        "balanced_bce": balanced_bce,
        "no_op_false_positive_pixel_rate": no_op_false_positive_pixel_rate,
        "per_frame_changed_vs_noop_ranking": ranking_accuracy,
        "ranking_tie_credit": 0.5,
        "confusion": confusion_json(&confusion),
        "per_actual_action": action_evaluations
            .iter()
            .enumerate()
            .map(|(index, evaluation)| action_evaluation_json((index + 1) as u8, evaluation))
            .collect::<Vec<_>>(),
        "input_arm_identity": input_arm_identity(arm),
        "action_independent_comparator_identity": COMPARATOR_ID,
    });
    ensure_finite_json(&value)?;
    Ok(Evaluation {
        value,
        prediction_sha256: hex_digest(fingerprint.finalize()),
    })
}

fn write_prediction_rows(
    writer: &mut dyn Write,
    frame_index: usize,
    frame: &HeldOutFrame,
    input_actions: &[ArcAction],
    predictions: &[ChangePrediction],
    arm: Arm,
) -> Result<()> {
    ensure!(
        frame.outcomes.len() == input_actions.len() && input_actions.len() == predictions.len(),
        "capture tuple count mismatch"
    );
    let input_frame_sha256 = frame_fingerprint(&frame.current);
    for (tuple_index, ((outcome, input_action), prediction)) in frame
        .outcomes
        .iter()
        .zip(input_actions)
        .zip(predictions)
        .enumerate()
    {
        ensure!(
            prediction.action == *input_action,
            "captured prediction action does not match model input action"
        );
        ensure!(
            prediction.probabilities.len() == FRAME_PIXELS
                && prediction
                    .probabilities
                    .iter()
                    .all(|value| value.is_finite()),
            "captured probabilities must contain {FRAME_PIXELS} finite values"
        );
        let actual_changed_pixel_indices = outcome
            .current
            .pixels
            .iter()
            .zip(outcome.next.pixels.iter())
            .enumerate()
            .filter_map(|(index, (before, after))| (before != after).then_some(index))
            .collect::<Vec<_>>();
        let row = json!({
            "schema": HELD_OUT_PREDICTIONS_ID,
            "frame_index": frame_index,
            "tuple_index": tuple_index,
            "input_frame_sha256": input_frame_sha256,
            "actual_action": action_json(&outcome.action),
            "model_input_action": action_json(input_action),
            "model_input_actually_replaced": input_action != &outcome.action,
            "model_input_identity": input_arm_identity(arm),
            "actual_changed_pixel_indices": actual_changed_pixel_indices,
            "probabilities": prediction.probabilities,
        });
        serde_json::to_writer(&mut *writer, &row).context("write held-out prediction row")?;
        writer
            .write_all(b"\n")
            .context("terminate held-out prediction row")?;
    }
    Ok(())
}

fn frame_fingerprint(frame: &ArcFrame) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"online-effect-input-frame-v1");
    hasher.update(frame.width.to_le_bytes());
    hasher.update(frame.height.to_le_bytes());
    hasher.update(frame.pixels.as_ref());
    hex_digest(hasher.finalize())
}

fn oracle_predictions(
    frame: &HeldOutFrame,
    actions: &[ArcAction],
) -> Result<Vec<ChangePrediction>> {
    ensure!(
        actions.len() == frame.outcomes.len(),
        "oracle action count mismatch"
    );
    frame
        .outcomes
        .iter()
        .zip(actions)
        .map(|(outcome, action)| {
            ensure!(
                action == &outcome.action,
                "oracle control requires true action inputs"
            );
            Ok(ChangePrediction {
                action: action.clone(),
                probabilities: outcome
                    .current
                    .pixels
                    .iter()
                    .zip(outcome.next.pixels.iter())
                    .map(|(before, after)| if before != after { 1.0 } else { 0.0 })
                    .collect(),
            })
        })
        .collect()
}

fn copy_zero_predictions(actions: &[ArcAction]) -> Vec<ChangePrediction> {
    actions
        .iter()
        .cloned()
        .map(|action| ChangePrediction {
            action,
            probabilities: vec![0.0; FRAME_PIXELS],
        })
        .collect()
}

fn control_json(identity: &str, mut metrics: Value) -> Value {
    metrics
        .as_object_mut()
        .expect("evaluation metrics are a JSON object")
        .insert("identity".into(), json!(identity));
    metrics
}

fn accumulate_metrics(confusion: &mut Confusion, metrics: &EffectMetrics) {
    confusion.true_positive += metrics.true_positive as u64;
    confusion.true_negative += metrics.true_negative as u64;
    confusion.false_positive += metrics.false_positive as u64;
    confusion.false_negative += metrics.false_negative as u64;
}

fn effect_metrics_json(metrics: &EffectMetrics) -> Value {
    json!({
        "balanced_bce": metrics.balanced_bce,
        "changed_pixels": metrics.changed_pixels,
        "unchanged_pixels": metrics.unchanged_pixels,
        "predicted_changed_pixels": metrics.predicted_changed_pixels,
        "true_positive": metrics.true_positive,
        "true_negative": metrics.true_negative,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "accuracy": metrics.accuracy,
    })
}

fn effect_update_json(update: &EffectUpdate) -> Value {
    json!({
        "update": update.update,
        "batch_size": update.batch_size,
        "loss": update.loss,
        "changed_pixels": update.changed_pixels,
        "unchanged_pixels": update.unchanged_pixels,
        "changed_weight": update.changed_weight,
        "unchanged_weight": update.unchanged_weight,
        "gradient_l2": update.gradient_l2,
    })
}

fn confusion_json(confusion: &Confusion) -> Value {
    json!({
        "true_positive": confusion.true_positive,
        "true_negative": confusion.true_negative,
        "false_positive": confusion.false_positive,
        "false_negative": confusion.false_negative,
        "changed_pixels": confusion.changed(),
        "unchanged_pixels": confusion.total() - confusion.changed(),
        "predicted_changed_pixels": confusion.predicted_changed(),
        "precision": confusion.precision(),
        "recall": confusion.recall(),
        "micro_change_f1": confusion.f1(),
        "accuracy": confusion.accuracy(),
    })
}

fn action_json(action: &ArcAction) -> Value {
    json!({"id": action.id, "x": action.x, "y": action.y})
}

fn movement_json(movement: &[Direction; 4]) -> Value {
    Value::Array(
        movement
            .iter()
            .enumerate()
            .map(|(index, direction)| {
                json!({
                    "action": index + 1,
                    "dx": direction.dx,
                    "dy": direction.dy,
                })
            })
            .collect(),
    )
}

fn action_evaluation_json(action_id: u8, evaluation: &ActionEvaluation) -> Value {
    json!({
        "action": action_id,
        "tuples": evaluation.changed_tuples + evaluation.noop_tuples,
        "genuinely_changed_tuples": evaluation.changed_tuples,
        "noop_tuples": evaluation.noop_tuples,
        "micro_change_f1": evaluation.confusion.f1(),
        "micro_change_precision": evaluation.confusion.precision(),
        "micro_change_recall": evaluation.confusion.recall(),
        "pixel_accuracy": evaluation.confusion.accuracy(),
        "no_op_false_positive_pixel_rate": (evaluation.noop_pixels > 0).then(|| {
            evaluation.noop_false_positive_pixels as f64 / evaluation.noop_pixels as f64
        }),
        "confusion": confusion_json(&evaluation.confusion),
    })
}

fn seed_construction(data_seed: u64) -> Value {
    json!({
        "rng": "ChaCha8Rng::seed_from_u64",
        "environment_seed": data_seed,
        "environment_draw_order": "background palette, distinct object palette, movement permutation",
        "training_seed_expression": "data_seed XOR 0x545241494e5f7631",
        "training_seed": data_seed ^ TRAIN_STREAM_TAG,
        "held_out_seed_expression": "data_seed XOR 0x48454c445f763101",
        "held_out_seed": data_seed ^ HELD_OUT_STREAM_TAG,
        "training_and_held_out_streams_distinct": true,
    })
}

fn generated_data_hash(
    background: u8,
    object: u8,
    movement: &[Direction; 4],
    training: &[Transition],
    held_out: &[HeldOutFrame],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(GENERATOR_SCHEMA.as_bytes());
    hasher.update([background, object]);
    for direction in movement {
        hasher.update(direction.dx.to_le_bytes());
        hasher.update(direction.dy.to_le_bytes());
    }
    hasher.update((training.len() as u64).to_le_bytes());
    for row in training {
        hash_transition(&mut hasher, row);
    }
    hasher.update((held_out.len() as u64).to_le_bytes());
    for frame in held_out {
        hasher.update(frame.current.pixels.as_ref());
        for outcome in &frame.outcomes {
            hash_transition(&mut hasher, outcome);
        }
    }
    hex_digest(hasher.finalize())
}

fn hash_transition(hasher: &mut Sha256, transition: &Transition) {
    hasher.update(transition.current.width.to_le_bytes());
    hasher.update(transition.current.height.to_le_bytes());
    hasher.update(transition.current.pixels.as_ref());
    hasher.update([transition.action.id]);
    hasher.update([transition.action.x.unwrap_or(u8::MAX)]);
    hasher.update([transition.action.y.unwrap_or(u8::MAX)]);
    hasher.update(transition.next.width.to_le_bytes());
    hasher.update(transition.next.height.to_le_bytes());
    hasher.update(transition.next.pixels.as_ref());
}

fn hash_predictions(hasher: &mut Sha256, predictions: &[ChangePrediction]) -> Result<()> {
    hasher.update((predictions.len() as u64).to_le_bytes());
    for prediction in predictions {
        hasher.update([prediction.action.id]);
        hasher.update([prediction.action.x.unwrap_or(u8::MAX)]);
        hasher.update([prediction.action.y.unwrap_or(u8::MAX)]);
        ensure!(
            prediction.probabilities.len() == FRAME_PIXELS,
            "prediction fingerprint received wrong pixel count"
        );
        for &probability in &prediction.probabilities {
            ensure!(
                probability.is_finite(),
                "prediction fingerprint received non-finite probability"
            );
            hasher.update(probability.to_bits().to_le_bytes());
        }
    }
    Ok(())
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn check_deadline(started: Instant, deadline: Duration, stage: &str) -> Result<()> {
    if started.elapsed() >= deadline {
        bail!(
            "--max-seconds deadline exceeded during {stage}: {:.3}s >= {:.3}s",
            started.elapsed().as_secs_f64(),
            deadline.as_secs_f64()
        );
    }
    Ok(())
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    ensure_finite_json(value)?;
    let mut file = BufWriter::new(
        File::create(path).with_context(|| format!("create JSON file {}", path.display()))?,
    );
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("open artifact {}", path.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("read artifact {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_digest(hasher.finalize()))
}

fn ensure_finite_json(value: &Value) -> Result<()> {
    match value {
        Value::Array(values) => {
            for value in values {
                ensure_finite_json(value)?;
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                ensure_finite_json(value)?;
            }
        }
        Value::Number(number) => ensure!(
            number.as_f64().is_some(),
            "JSON report contains a non-finite number"
        ),
        _ => {}
    }
    Ok(())
}

fn validate_launch_integrity(args: &Args) -> Result<()> {
    ensure!(
        EMBEDDED_SOURCE_DIRTY == "false",
        "binary was compiled from a dirty Tofy source tree"
    );
    ensure!(
        EMBEDDED_SOURCE_PUSHED == "true",
        "compiled Tofy source revision is not confirmed pushed"
    );
    ensure!(
        EMBEDDED_CANDLE_GRAPH_DIRTY == "false",
        "binary was compiled from a dirty candle_graph source tree"
    );
    ensure!(
        EMBEDDED_CANDLE_GRAPH_PUSHED == "true",
        "compiled candle_graph source revision is not confirmed pushed"
    );
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let runtime_source_revision = command_output(manifest_dir, &["rev-parse", "HEAD"])
        .context("read runtime Tofy revision")?;
    let runtime_source_status = command_output(
        manifest_dir,
        &["status", "--short", "--untracked-files=all"],
    )
    .context("read runtime Tofy status")?;
    ensure!(
        runtime_source_status.is_empty(),
        "runtime Tofy tree is dirty"
    );
    ensure!(
        runtime_source_revision == EMBEDDED_SOURCE_REVISION,
        "runtime Tofy revision {runtime_source_revision} does not match compiled revision {EMBEDDED_SOURCE_REVISION}"
    );

    let candle_graph_dir = manifest_dir.join("..").join("candle_graph");
    let runtime_candle_graph_revision = command_output(&candle_graph_dir, &["rev-parse", "HEAD"])
        .context("read runtime candle_graph revision")?;
    let runtime_candle_graph_status = command_output(
        &candle_graph_dir,
        &["status", "--short", "--untracked-files=all"],
    )
    .context("read runtime candle_graph status")?;
    ensure!(
        runtime_candle_graph_status.is_empty(),
        "runtime candle_graph tree is dirty"
    );
    ensure!(
        runtime_candle_graph_revision == EMBEDDED_CANDLE_GRAPH_REVISION,
        "runtime candle_graph revision {runtime_candle_graph_revision} does not match compiled revision {EMBEDDED_CANDLE_GRAPH_REVISION}"
    );
    if args.device.trim().starts_with("cuda") {
        ensure!(
            EMBEDDED_CARGO_FEATURES
                .split(',')
                .any(|feature| feature == "cudnn"),
            "CUDA probe requires a binary compiled with the cudnn feature"
        );
    }
    Ok(())
}

fn provenance() -> Value {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let git_revision = command_output(manifest_dir, &["rev-parse", "HEAD"]);
    let git_status = command_output(
        manifest_dir,
        &["status", "--short", "--untracked-files=all"],
    );
    let module_path = manifest_dir.join("src/p2/online_effect.rs");
    let module_source_sha256 = fs::read(&module_path).ok().map(|bytes| {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex_digest(hasher.finalize())
    });
    let candle_graph_dir = manifest_dir.join("..").join("candle_graph");
    json!({
        "embedded": {
            "source_revision": EMBEDDED_SOURCE_REVISION,
            "source_dirty": EMBEDDED_SOURCE_DIRTY,
            "source_pushed": EMBEDDED_SOURCE_PUSHED,
            "candle_graph_revision": EMBEDDED_CANDLE_GRAPH_REVISION,
            "candle_graph_dirty": EMBEDDED_CANDLE_GRAPH_DIRTY,
            "candle_graph_pushed": EMBEDDED_CANDLE_GRAPH_PUSHED,
            "cargo_features": EMBEDDED_CARGO_FEATURES,
            "build_command": EMBEDDED_BUILD_COMMAND,
        },
        "runtime": {
            "source_revision": git_revision,
            "source_dirty": git_status.as_ref().map(|status| !status.is_empty()),
            "candle_graph_revision": command_output(&candle_graph_dir, &["rev-parse", "HEAD"]),
            "candle_graph_dirty": command_output(
                &candle_graph_dir,
                &["status", "--short", "--untracked-files=all"],
            ).map(|status| !status.is_empty()),
        },
        "module_path": "src/p2/online_effect.rs",
        "module_source_sha256": module_source_sha256,
    })
}

fn command_output(directory: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn number(value: &Value, key: &str) -> f64 {
        value[key]
            .as_f64()
            .unwrap_or_else(|| panic!("missing {key}"))
    }

    fn count(value: &Value, key: &str) -> u64 {
        value[key]
            .as_u64()
            .unwrap_or_else(|| panic!("missing {key}"))
    }

    fn object_center(frame: &ArcFrame, object: u8) -> (u8, u8) {
        let coordinates = frame
            .pixels
            .iter()
            .enumerate()
            .filter(|(_, pixel)| **pixel == object)
            .map(|(index, _)| ((index % FRAME_SIDE) as u8, (index / FRAME_SIDE) as u8))
            .collect::<Vec<_>>();
        assert_eq!(coordinates.len(), 9);
        (
            (coordinates.iter().map(|(x, _)| u16::from(*x)).sum::<u16>() / 9) as u8,
            (coordinates.iter().map(|(_, y)| u16::from(*y)).sum::<u16>() / 9) as u8,
        )
    }

    #[test]
    fn generator_fixes_palette_and_separates_position_splits() -> Result<()> {
        let data = generate_data(101, 512, HELD_OUT_FRAMES)?;
        assert_ne!(data.background, data.object);
        let allowed = BTreeSet::from([data.background, data.object]);
        let mut training_centers = BTreeSet::new();
        for transition in &data.training {
            assert!(transition
                .current
                .pixels
                .iter()
                .all(|pixel| allowed.contains(pixel)));
            let center = object_center(&transition.current, data.object);
            assert_eq!((center.0 + center.1) % 2, 0);
            training_centers.insert(center);
        }
        let mut held_out_centers = BTreeSet::new();
        for frame in &data.held_out {
            assert!(frame
                .current
                .pixels
                .iter()
                .all(|pixel| allowed.contains(pixel)));
            let center = object_center(&frame.current, data.object);
            assert_eq!((center.0 + center.1) % 2, 1);
            held_out_centers.insert(center);
        }
        assert!(training_centers.is_disjoint(&held_out_centers));
        assert_eq!(
            data.sha256,
            generate_data(101, 512, HELD_OUT_FRAMES)?.sha256
        );
        Ok(())
    }

    #[test]
    fn evaluator_distinguishes_oracle_and_copy_zero_controls() -> Result<()> {
        let data = generate_data(101, 1, HELD_OUT_FRAMES)?;
        let oracle = evaluate_with_predictor(
            &data.held_out,
            Arm::Conditioned,
            Instant::now(),
            Duration::from_secs(30),
            None,
            oracle_predictions,
        )?;
        let copy_zero = evaluate_with_predictor(
            &data.held_out,
            Arm::Conditioned,
            Instant::now(),
            Duration::from_secs(30),
            None,
            |_, actions| Ok(copy_zero_predictions(actions)),
        )?;

        for metrics in [&oracle.value, &copy_zero.value] {
            assert_eq!(count(metrics, "total_action_tuples"), 448);
            assert_eq!(count(metrics, "genuinely_changed_tuples"), 288);
            assert_eq!(count(metrics, "noop_tuples"), 160);
            assert_eq!(
                count(
                    metrics,
                    "true_outcome_different_changed_vs_noop_action_pairs"
                ),
                704
            );
            assert_eq!(count(&metrics["confusion"], "changed_pixels"), 1_568);
            assert!(count(&metrics["confusion"], "unchanged_pixels") > 0);
            let action6 = &metrics["per_actual_action"][5];
            assert_eq!(count(action6, "genuinely_changed_tuples"), 32);
            assert_eq!(count(action6, "noop_tuples"), 32);
            assert_eq!(count(&action6["confusion"], "changed_pixels"), 32);
        }

        assert_eq!(number(&oracle.value, "micro_change_f1"), 1.0);
        assert_eq!(
            number(&oracle.value, "per_frame_changed_vs_noop_ranking"),
            1.0
        );
        assert_eq!(
            number(&oracle.value, "no_op_false_positive_pixel_rate"),
            0.0
        );
        assert_eq!(number(&copy_zero.value, "micro_change_f1"), 0.0);
        assert_eq!(
            number(&copy_zero.value, "per_frame_changed_vs_noop_ranking"),
            0.5
        );
        assert_eq!(
            number(&copy_zero.value, "no_op_false_positive_pixel_rate"),
            0.0
        );
        Ok(())
    }

    #[test]
    fn held_out_capture_preserves_metrics_and_records_rescorable_rows() -> Result<()> {
        let data = generate_data(101, 1, SMOKE_HELD_OUT_FRAMES)?;
        let baseline = evaluate_with_predictor(
            &data.held_out,
            Arm::Conditioned,
            Instant::now(),
            Duration::from_secs(30),
            None,
            oracle_predictions,
        )?;
        let mut captured = Vec::new();
        let with_capture = evaluate_with_predictor(
            &data.held_out,
            Arm::Conditioned,
            Instant::now(),
            Duration::from_secs(30),
            Some(&mut captured),
            oracle_predictions,
        )?;
        assert_eq!(baseline.value, with_capture.value);
        assert_eq!(baseline.prediction_sha256, with_capture.prediction_sha256);

        let rows = captured
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(serde_json::from_slice::<Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(rows.len(), SMOKE_HELD_OUT_FRAMES * 7);
        for row in &rows {
            let frame_index = count(row, "frame_index") as usize;
            let tuple_index = count(row, "tuple_index") as usize;
            assert_eq!(row["schema"], HELD_OUT_PREDICTIONS_ID);
            assert_eq!(
                row["input_frame_sha256"],
                frame_fingerprint(&data.held_out[frame_index].current)
            );
            assert_eq!(
                count(&row["actual_action"], "id"),
                data.held_out[frame_index].outcomes[tuple_index].action.id as u64
            );
            assert_eq!(row["model_input_actually_replaced"], false);
            assert_eq!(
                row["model_input_identity"],
                input_arm_identity(Arm::Conditioned)
            );
            let probabilities = row["probabilities"]
                .as_array()
                .context("captured probabilities are not an array")?;
            assert_eq!(probabilities.len(), FRAME_PIXELS);
            let changed = row["actual_changed_pixel_indices"]
                .as_array()
                .context("captured target is not an array")?
                .iter()
                .map(|index| index.as_u64().context("target index is not an integer"))
                .collect::<Result<BTreeSet<_>>>()?;
            assert!(changed.iter().all(|index| *index < FRAME_PIXELS as u64));
            for (index, probability) in probabilities.iter().enumerate() {
                assert_eq!(
                    probability.as_f64().context("probability is not numeric")?,
                    if changed.contains(&(index as u64)) {
                        1.0
                    } else {
                        0.0
                    }
                );
            }
        }

        let mut constant_capture = Vec::new();
        evaluate_with_predictor(
            &data.held_out,
            Arm::Constant,
            Instant::now(),
            Duration::from_secs(30),
            Some(&mut constant_capture),
            |_, actions| Ok(copy_zero_predictions(actions)),
        )?;
        let first_constant: Value = serde_json::from_slice(
            constant_capture
                .split(|byte| *byte == b'\n')
                .next()
                .context("constant capture is empty")?,
        )?;
        assert_eq!(count(&first_constant["actual_action"], "id"), 1);
        assert_eq!(count(&first_constant["model_input_action"], "id"), 5);
        assert_eq!(first_constant["model_input_actually_replaced"], true);
        assert_eq!(first_constant["model_input_identity"], COMPARATOR_ID);
        Ok(())
    }

    #[test]
    fn cli_defaults_to_one_bounded_update_per_observation() -> Result<()> {
        let defaults = Args::try_parse_from([
            "online-effect-probe",
            "--arm",
            "conditioned",
            "--output-dir",
            "/tmp/unused-online-effect-defaults",
        ])?;
        assert_eq!(defaults.updates_per_observation, 1);
        validate_args(&defaults)?;

        let repeated = Args::try_parse_from([
            "online-effect-probe",
            "--arm",
            "constant",
            "--output-dir",
            "/tmp/unused-online-effect-repeated",
            "--updates-per-observation",
            "16",
        ])?;
        assert_eq!(repeated.updates_per_observation, 16);
        validate_args(&repeated)?;

        let mut invalid = repeated;
        invalid.updates_per_observation = 17;
        assert!(validate_args(&invalid).is_err());
        Ok(())
    }
}
