//! C12: grounded spatial policy trained through the shared-depth recurrent body.
#[path = "grounded_policy/data.rs"]
mod data;
#[path = "grounded_policy/engine.rs"]
mod engine;
#[path = "grounded_policy/evidence.rs"]
mod evidence;

use anyhow::{ensure, Context, Result};
use candle_core::{Device, Tensor};
use clap::Parser;
use evidence::{file_hash, write_json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
    time::Instant,
};
use tofy::p2::looped_agent::{
    profile::LoopedCapture,
    task::{self, Inputs},
};

const INITIAL_CORE: &str = "4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802";
const INITIAL_HEAD: &str = "a43ad78fe7a6b1594d23c205fb191c8ac81610deaaff25198cb421d423edc678";
const INITIAL_IMPORT: &str = "d2f9a7a99917d180646a5e28e26acd92cbd3fe53919fa009da3e11747e0c158e";
const EFFECTIVE: usize = 64;
const UPDATES: usize = 1150;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Audit,
    Qualify,
    BatchSmoke,
    Train,
    EvalInitial,
    EvalFinal,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema: String,
    source_revision: String,
    registration: PathBuf,
    registration_sha256: String,
    mode: Mode,
    output_dir: PathBuf,
    physical_batch: usize,
    updates: usize,
    max_seconds: u64,
    core_checkpoint: PathBuf,
    core_sha256: String,
    head_checkpoint: PathBuf,
    head_sha256: String,
    import_manifest: PathBuf,
    audit_root: Option<PathBuf>,
    audit_manifest_sha256: Option<String>,
    history: Vec<data::HistorySource>,
    cohort: Option<data::Cohort>,
    cleared: bool,
}
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    config_sha256: String,
}
impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "looped-grounded-policy-config-v1",
            "unknown config schema"
        );
        ensure!(
            self.output_dir.is_absolute() && !self.output_dir.exists(),
            "run root must be new and absolute"
        );
        ensure!(
            self.source_revision == env!("TOFY_EMBEDDED_SOURCE_REVISION"),
            "configured source differs from binary"
        );
        ensure!(
            (1..=64).contains(&self.physical_batch),
            "physical batch outside registered range"
        );
        ensure!(
            self.registration.is_absolute()
                && file_hash(&self.registration)? == self.registration_sha256,
            "registration binding differs"
        );
        ensure!(
            self.core_checkpoint.is_absolute() && self.head_checkpoint.is_absolute(),
            "absolute checkpoint paths required"
        );
        ensure!(
            file_hash(&self.core_checkpoint)? == self.core_sha256
                && file_hash(&self.head_checkpoint)? == self.head_sha256,
            "checkpoint hash differs"
        );
        if self.mode != Mode::EvalFinal {
            ensure!(
                self.core_sha256 == INITIAL_CORE && self.head_sha256 == INITIAL_HEAD,
                "exact registered initialization required"
            );
        }
        ensure!(
            file_hash(&self.import_manifest)? == INITIAL_IMPORT,
            "privileged warm-start import binding differs"
        );
        let training = matches!(self.mode, Mode::Train | Mode::BatchSmoke);
        let evaluation = matches!(self.mode, Mode::EvalInitial | Mode::EvalFinal);
        ensure!(
            self.updates
                == if self.mode == Mode::Train {
                    UPDATES
                } else if self.mode == Mode::BatchSmoke {
                    self.updates
                } else {
                    0
                },
            "update budget differs"
        );
        ensure!(
            self.mode != Mode::BatchSmoke || matches!(self.updates, 2 | 5),
            "batch qualification requires two or five disposable updates"
        );
        ensure!(
            self.mode != Mode::Qualify || matches!(self.physical_batch, 1 | 4),
            "parity requires registered batches one or four"
        );
        ensure!(
            evaluation == self.cohort.is_some(),
            "only evaluation selects a cohort"
        );
        ensure!(
            !self.cleared || self.mode == Mode::EvalFinal,
            "cleared support only at terminal evaluation"
        );
        ensure!(
            !training || !self.cleared,
            "training must use factual support"
        );
        ensure!(
            self.max_seconds > 0
                && self.max_seconds
                    <= if self.mode == Mode::Train {
                        3600
                    } else if self.mode == Mode::BatchSmoke || evaluation {
                        600
                    } else {
                        120
                    },
            "registered invocation budget differs"
        );
        ensure!(
            (self.mode == Mode::Audit) != self.history.is_empty(),
            "history only in CPU audit"
        );
        ensure!(
            (self.mode == Mode::Train || evaluation)
                == (self.audit_root.is_some() && self.audit_manifest_sha256.is_some()),
            "training/evaluation require sealed audit"
        );
        ensure!(
            self.audit_root.is_some() == self.audit_manifest_sha256.is_some(),
            "incomplete audit binding"
        );
        Ok(())
    }
    fn deadline(&self, started: Instant) -> Result<()> {
        ensure!(
            started.elapsed().as_secs_f64() < self.max_seconds as f64,
            "registered model deadline exceeded"
        );
        Ok(())
    }
}

fn load_model(config: &Config, device: &Device) -> Result<engine::Model> {
    let model = engine::Model::new(device)?;
    model.load_core(&config.core_checkpoint)?;
    if config.mode == Mode::EvalFinal {
        model.load_head(&config.head_checkpoint)?;
    } else {
        let bytes = fs::read(&config.head_checkpoint)?;
        ensure!(bytes.len() == 5136, "canonical head byte count differs");
        let values = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "nonfinite imported parameters"
        );
        let shapes = [
            ("queries", vec![2, 128]),
            ("output.weight", vec![4, 256]),
            ("output.bias", vec![4]),
        ];
        let vars = model
            .head_vars
            .data()
            .lock()
            .map_err(|_| anyhow::anyhow!("head lock poisoned"))?;
        let mut offset = 0;
        for (name, shape) in shapes {
            let count = shape.iter().product::<usize>();
            vars.get(name)
                .context("missing canonical head tensor")?
                .set(&Tensor::from_vec(
                    values[offset..offset + count].to_vec(),
                    shape,
                    device,
                )?)?;
            offset += count;
        }
        ensure!(offset == values.len(), "unconsumed import bytes");
    }
    Ok(model)
}

fn old_inputs(count: usize) -> Result<Vec<Inputs>> {
    (0..count)
        .map(|i| {
            let episode =
                task::episode_with_permutation(20260915, 0x46454154555245 + i as u64, 0, 1, 1)?;
            task::inputs(&episode.support, &episode.maze.render())
        })
        .collect()
}
fn labels_old(count: usize) -> Result<Vec<usize>> {
    (0..count)
        .map(|i| {
            let episode =
                task::episode_with_permutation(20260915, 0x46454154555245 + i as u64, 0, 1, 1)?;
            let sample = task::sample(&episode)?;
            sample
                .policy
                .iter()
                .position(|&x| x == 1.0)
                .context("old smoke requires unique action")
        })
        .collect()
}

fn run(config: &Config, started: Instant) -> Result<Value> {
    let source = evidence::provenance()?;
    write_json(
        &config.output_dir.join("metadata.json"),
        &json!({"schema":"looped-grounded-policy-v1","config":config,"provenance":source,
        "loops":4,"hidden":128,"layers":2,"heads":4,"max_loops":8,"effective_batch":64,"seed":0,
        "objective":"policy_cross_entropy_only","privileged_role_warm_start":true,
        "deferred":["reward","value","dynamics","planner","episode_memory","ARC_evaluation","useful_depth_test"]}),
    )?;
    if config.mode == Mode::Audit {
        let report = data::audit(
            &config.output_dir,
            &config.history,
            started,
            config.max_seconds,
        )?;
        return Ok(
            json!({"status":"complete_pending_analysis","classification":"population_audit","optimizer_updates":0,"model_forwards":0,"audit":report,"elapsed_seconds":started.elapsed().as_secs_f64()}),
        );
    }
    if let Some(root) = &config.audit_root {
        evidence::verify_root(root, config.audit_manifest_sha256.as_deref().unwrap())?;
        let audit: Value = serde_json::from_slice(&fs::read(root.join("report.json"))?)?;
        ensure!(
            audit["status"] == "complete_pending_analysis"
                && audit["classification"] == "population_audit"
                && audit["optimizer_updates"] == 0
                && audit["model_forwards"] == 0
                && audit["audit"]["training_rows"] == 73600
                && audit["audit"]["new_unique_queries"] == 64
                && audit["audit"]["new_query_overlap"] == 0
                && audit["audit"]["artifacts"]
                    .as_array()
                    .is_some_and(|a| a.len() == 7),
            "incomplete or failed population audit"
        );
    }
    ensure!(
        std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() == Ok("0"),
        "TF32 override must be zero"
    );
    #[cfg(feature = "cudnn")]
    ensure!(
        !candle_core::cuda_backend::gemm_reduced_precision_f32(),
        "reduced precision F32 GEMM forbidden"
    );
    let device = Device::new_cuda(0)?;
    let model = load_model(config, &device)?;
    config.deadline(started)?;
    let report = match config.mode {
        Mode::Qualify => qualify(config, &model, &device, started)?,
        Mode::BatchSmoke | Mode::Train => train(config, &model, &device, started)?,
        Mode::EvalInitial | Mode::EvalFinal => evaluate(config, &model, &device, started)?,
        Mode::Audit => unreachable!(),
    };
    device.synchronize()?;
    config.deadline(started)?;
    Ok(report)
}

fn capture(
    config: &Config,
    model: &engine::Model,
    device: &Device,
    step: usize,
    training: bool,
    actual_batch: usize,
) -> Result<LoopedCapture> {
    LoopedCapture::begin_grounded_policy(
        &config
            .output_dir
            .with_extension("profiles")
            .join(format!("update-{step:012}")),
        step as u64,
        device,
        training.then_some((&model.core_vars, &model.head_vars)),
        actual_batch,
        if training { EFFECTIVE } else { actual_batch },
        4,
    )
}

fn qualify(
    config: &Config,
    model: &engine::Model,
    device: &Device,
    started: Instant,
) -> Result<Value> {
    let frozen = model.frozen()?;
    let rows = old_inputs(4)?;
    let mut output = BufWriter::new(File::create(
        config.output_dir.join("qualification-rows.jsonl"),
    )?);
    for (i, chunk) in rows.chunks(config.physical_batch).enumerate() {
        config.deadline(started)?;
        let cap = (i == 0)
            .then(|| capture(config, model, device, 1, false, chunk.len()))
            .transpose()?;
        let forward = measured_forward(&frozen, chunk, device, cap.as_ref())?;
        let logits = forward.policy.logits.to_vec2::<f32>()?;
        let attention = forward.policy.attention.to_vec3::<f32>()?;
        let pooled = forward.policy.pooled.to_vec2::<f32>()?;
        let current = forward.features.current.to_vec3::<f32>()?;
        let cls = forward.features.cls.to_vec2::<f32>()?;
        ensure!(
            logits
                .iter()
                .flatten()
                .chain(attention.iter().flatten().flatten())
                .chain(pooled.iter().flatten())
                .chain(current.iter().flatten().flatten())
                .chain(cls.iter().flatten())
                .all(|x| x.is_finite()),
            "nonfinite qualification outputs"
        );
        for j in 0..chunk.len() {
            let row = json!({"index":i*config.physical_batch+j,"logits":logits[j],"attention":attention[j],"pooled":pooled[j],"current":current[j],"cls":cls[j]});
            serde_json::to_writer(&mut output, &row)?;
            output.write_all(b"\n")?;
        }
        if let Some(cap) = cap {
            cap.finish()?;
        }
    }
    output.flush()?;
    Ok(
        json!({"status":"complete_pending_analysis","classification":"implementation_smoke","optimizer_updates":0,"input_rows":4,"physical_batch":config.physical_batch,"elapsed_seconds":started.elapsed().as_secs_f64()}),
    )
}

fn measured_forward(
    model: &engine::FrozenModel,
    rows: &[Inputs],
    device: &Device,
    cap: Option<&LoopedCapture>,
) -> Result<engine::Forward> {
    device.synchronize()?;
    let measured = cap.map(LoopedCapture::measurement);
    let phase = cap.map(|c| c.phase("forward", Some(candle_graph::ExecutionStep::Forward)));
    let result = (|| {
        let out = model.forward(rows)?;
        if let (Some(c), Some(p)) = (cap, &phase) {
            c.record_tensor_stats(p, "features/current", &out.features.current)?;
            c.record_tensor_stats(p, "policy/logits", &out.policy.logits)?;
            c.record_tensor_stats(p, "policy/attention", &out.policy.attention)?;
            c.record_tensor_stats(p, "policy/pooled", &out.policy.pooled)?;
        }
        Ok(out)
    })();
    let sync = device.synchronize();
    drop(phase);
    drop(measured);
    sync?;
    result
}

// Training and evaluation use the data module's audit identity stream. No audit
// field except Inputs is passed to forward; labels enter CE/scoring separately.
fn train(
    config: &Config,
    model: &engine::Model,
    device: &Device,
    started: Instant,
) -> Result<Value> {
    let initial = model.snapshot()?;
    let mut optimizer = model.optimizer()?;
    let mut log = BufWriter::new(File::create(config.output_dir.join("updates.jsonl"))?);
    let mut stream = BufWriter::new(File::create(
        config.output_dir.join("training-stream.jsonl"),
    )?);
    let old = if config.mode == Mode::BatchSmoke {
        Some((old_inputs(64)?, labels_old(64)?))
    } else {
        None
    };
    let mut audit = if config.mode == Mode::Train {
        Some(
            BufReader::new(File::open(
                config
                    .audit_root
                    .as_ref()
                    .unwrap()
                    .join("training-audit.jsonl"),
            )?)
            .lines(),
        )
    } else {
        None
    };
    let update_started = Instant::now();
    let mut last = Value::Null;
    for update in 1..=config.updates {
        config.deadline(started)?;
        let (inputs, labels) = if let Some((inputs, labels)) = &old {
            (inputs.clone(), labels.clone())
        } else {
            let rows = data::training_batch(update - 1)?;
            for row in &rows {
                let identity = row.audit_json.clone();
                let expected: Value = serde_json::from_str(
                    &audit
                        .as_mut()
                        .unwrap()
                        .next()
                        .context("audit truncated")??,
                )?;
                ensure!(
                    identity == expected,
                    "training tuple differs from frozen audit at update {update}"
                );
                serde_json::to_writer(&mut stream, &identity)?;
                stream.write_all(b"\n")?;
            }
            (
                rows.iter().map(|r| r.inputs.clone()).collect(),
                rows.iter().map(|r| r.correct_action).collect(),
            )
        };
        let selected = if config.mode == Mode::Train {
            [2, 100, 1150].contains(&update)
        } else {
            update == 1
        };
        let cap = selected
            .then(|| capture(config, model, device, update, true, config.physical_batch))
            .transpose()?;
        let metrics = engine::train_update(
            model,
            &inputs,
            &labels,
            config.physical_batch,
            &mut optimizer,
            cap.as_ref(),
        )?;
        if config.mode == Mode::BatchSmoke {
            ensure!(
                metrics.body_gradient_norm > 0.0 && metrics.head_gradient_norm > 0.0,
                "qualified policy must have nonzero body and head gradients"
            );
        }
        if let Some(cap) = cap {
            cap.finish()?;
        }
        last = serde_json::to_value(metrics)?;
        let record = json!({"update":update,"metrics":last,"elapsed_seconds":started.elapsed().as_secs_f64()});
        serde_json::to_writer(&mut log, &record)?;
        log.write_all(b"\n")?;
        log.flush()?;
        if update == 1 || update.is_multiple_of(100) || update == config.updates {
            println!("{}", record);
        }
    }
    let updates_elapsed = update_started.elapsed().as_secs_f64();
    if let Some(audit) = &mut audit {
        ensure!(audit.next().is_none(), "unconsumed training audit rows");
    }
    stream.flush()?;
    log.flush()?;
    let changes = model.change_audit(&initial)?;
    ensure!(
        changes.unused_heads_unchanged
            && !changes.changed_body_names.is_empty()
            && !changes.changed_head_names.is_empty(),
        "body/head update or unused-head integrity gate failed"
    );
    let checkpoint_started = Instant::now();
    model.save_pair(
        &config.output_dir.join("final-core.safetensors"),
        &config.output_dir.join("final-head.safetensors"),
    )?;
    let checkpoint_seconds = checkpoint_started.elapsed().as_secs_f64();
    let mut restored = Value::Null;
    if config.mode == Mode::BatchSmoke {
        model.restore(&initial)?;
        let restoration = model.change_audit(&initial)?;
        ensure!(
            restoration.all_parameters_unchanged && restoration.unused_heads_unchanged,
            "qualification restore differs"
        );
        restored = serde_json::to_value(restoration)?;
    }
    config.deadline(started)?;
    Ok(
        json!({"status":"complete_pending_analysis","classification":if config.mode == Mode::Train {"single_seed_screen"} else {"implementation_smoke"},
        "optimizer_updates":config.updates,"physical_batch":config.physical_batch,"effective_batch":64,"accumulation":64usize.div_ceil(config.physical_batch),
        "input_rows":config.updates*64,"updates_elapsed_seconds":updates_elapsed,"checkpoint_seconds":checkpoint_seconds,"last_update":last,"changes":changes,"restored_changes":restored,
        "final_core_sha256":file_hash(&config.output_dir.join("final-core.safetensors"))?,"final_head_sha256":file_hash(&config.output_dir.join("final-head.safetensors"))?,"elapsed_seconds":started.elapsed().as_secs_f64()}),
    )
}

fn evaluate(
    config: &Config,
    model: &engine::Model,
    device: &Device,
    started: Instant,
) -> Result<Value> {
    let cohort = config.cohort.context("evaluation cohort missing")?;
    ensure!(
        cohort != data::Cohort::Training,
        "evaluation cannot select full training stream"
    );
    let frozen = model.frozen()?;
    let mut output = BufWriter::new(File::create(
        config.output_dir.join("evaluation-rows.jsonl"),
    )?);
    let audit_name = format!(
        "{}-{}-audit.jsonl",
        cohort.name(),
        if config.cleared { "cleared" } else { "factual" }
    );
    let mut expected = BufReader::new(File::open(
        config.audit_root.as_ref().unwrap().join(audit_name),
    )?)
    .lines();
    let mut rows = Vec::new();
    for group in 0..64 {
        rows.extend(data::group(cohort, group, config.cleared)?);
    }
    for (i, chunk) in rows.chunks(config.physical_batch).enumerate() {
        config.deadline(started)?;
        for row in chunk {
            let audited: Value =
                serde_json::from_str(&expected.next().context("evaluation audit truncated")??)?;
            ensure!(
                row.audit_json.clone() == audited,
                "evaluation input identity differs from sealed audit"
            );
        }
        let inputs = chunk.iter().map(|r| r.inputs.clone()).collect::<Vec<_>>();
        let cap = (i == 0)
            .then(|| capture(config, model, device, 1, false, chunk.len()))
            .transpose()?;
        let out = measured_forward(&frozen, &inputs, device, cap.as_ref())?;
        let logits = out.policy.logits.to_vec2::<f32>()?;
        let attention = out.policy.attention.to_vec3::<f32>()?;
        let pooled = out.policy.pooled.to_vec2::<f32>()?;
        ensure!(
            logits
                .iter()
                .flatten()
                .chain(attention.iter().flatten().flatten())
                .chain(pooled.iter().flatten())
                .all(|x| x.is_finite()),
            "nonfinite evaluation outputs"
        );
        for j in 0..chunk.len() {
            ensure!(
                chunk[j].agent_patch < 64 && chunk[j].goal_patch < 64,
                "invalid scoring role index"
            );
            let mut row = chunk[j].audit_json.clone();
            row["checkpointstage"] = json!(if config.mode == Mode::EvalInitial {
                "frozen"
            } else {
                "final"
            });
            row["logits"] = json!(logits[j]);
            row["attention"] = json!(attention[j]);
            row["pooled"] = json!(pooled[j]);
            serde_json::to_writer(&mut output, &row)?;
            output.write_all(b"\n")?;
        }
        if let Some(cap) = cap {
            cap.finish()?;
        }
    }
    ensure!(
        expected.next().is_none(),
        "unconsumed evaluation audit rows"
    );
    output.flush()?;
    Ok(
        json!({"status":"complete_pending_analysis","optimizer_updates":0,"cohort":cohort,"cleared":config.cleared,"input_rows":rows.len(),"physical_batch":config.physical_batch,"elapsed_seconds":started.elapsed().as_secs_f64()}),
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.config.is_absolute() && file_hash(&args.config)? == args.config_sha256,
        "configuration binding differs"
    );
    let config: Config = serde_json::from_slice(&fs::read(&args.config)?)?;
    config.validate()?;
    let guard = tofy::perf::install()?;
    ensure!(
        cfg!(feature = "profiling") && guard.is_some(),
        "profiling and unique TOFY_PERF_TRACE required"
    );
    fs::create_dir(&config.output_dir)?;
    write_json(
        &config.output_dir.join("launch.json"),
        &json!({"status":"running","pid":std::process::id(),"exact_args":std::env::args().collect::<Vec<_>>(),"config_sha256":args.config_sha256,"binary_sha256":file_hash(&std::env::current_exe()?)?}),
    )?;
    let started = Instant::now();
    let result = run(&config, started);
    drop(guard);
    match &result {
        Ok(report) => write_json(&config.output_dir.join("report.json"), report)?,
        Err(error) => write_json(
            &config.output_dir.join("report.json"),
            &json!({"status":"failed_integrity_or_evaluation","error":format!("{error:#}"),"elapsed_seconds":started.elapsed().as_secs_f64()}),
        )?,
    }
    evidence::bind_profiles(&config.output_dir)?;
    evidence::seal(&config.output_dir)?;
    result.map(|_| ())
}
