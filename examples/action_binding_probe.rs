//! Standalone action-binding screen; no vision core or simulator executes here.
#[path = "action_binding/engine.rs"]
mod engine;
#[path = "grounded_policy/evidence.rs"]
mod evidence;

use anyhow::{ensure, Context, Result};
use candle_core::Device;
use clap::Parser;
use engine::{MAX_EFFECTIVE, MIN_EFFECTIVE};
use evidence::{file_hash, write_json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};
use tofy::p2::looped_agent::{binding::ModelKind, profile::LoopedCapture};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
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
    dataset: PathBuf,
    dataset_sha256: String,
    mode: Mode,
    output_dir: PathBuf,
    model_kind: ModelKind,
    physical_batch: usize,
    effective_batch: usize,
    schedule_presentations: usize,
    schedule_sha256: String,
    profile_updates: Vec<usize>,
    updates: usize,
    max_seconds: u64,
    checkpoint: Option<PathBuf>,
    checkpoint_sha256: Option<String>,
    cohort: Option<String>,
    loops: usize,
    cleared: bool,
    query_cleared: bool,
}
impl Config {
    fn validate_mode(&self) -> Result<()> {
        ensure!(
            self.effective_batch.is_power_of_two()
                && (MIN_EFFECTIVE..=MAX_EFFECTIVE).contains(&self.effective_batch)
                && self.physical_batch.is_power_of_two()
                && self.physical_batch <= self.effective_batch,
            "invalid power-of-two physical/effective batch"
        );
        ensure!(
            self.schedule_presentations > 0
                && self.schedule_sha256.len() == 64
                && self
                    .schedule_sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "invalid schedule identity"
        );
        ensure!(
            self.max_seconds > 0
                && self.max_seconds <= if self.mode == Mode::Train { 600 } else { 120 },
            "model budget differs"
        );
        let evaluation = matches!(self.mode, Mode::EvalInitial | Mode::EvalFinal);
        ensure!(
            evaluation == self.cohort.is_some(),
            "cohort only in evaluation"
        );
        ensure!(
            (self.mode == Mode::EvalFinal) == self.checkpoint.is_some()
                && self.checkpoint.is_some() == self.checkpoint_sha256.is_some(),
            "checkpoint only for final evaluation"
        );
        ensure!(
            self.updates
                == match self.mode {
                    Mode::Train => self.schedule_presentations.div_ceil(self.effective_batch),
                    Mode::BatchSmoke => self.updates,
                    _ => 0,
                },
            "update count differs"
        );
        ensure!(
            self.mode != Mode::BatchSmoke || matches!(self.updates, 2 | 5),
            "smoke requires2or5 updates"
        );
        match self.mode {
            Mode::Train => ensure!(
                self.profile_updates.len() == 3
                    && self.profile_updates[0] == 2
                    && self.profile_updates[2] == self.updates
                    && self.profile_updates.windows(2).all(|p| p[0] < p[1]),
                "training requires reachable captures at2, middle and final update"
            ),
            Mode::BatchSmoke => ensure!(
                self.profile_updates == [1]
                    && self.updates.checked_mul(self.effective_batch)
                        == Some(self.schedule_presentations),
                "smoke requires full candidate batches and capture1"
            ),
            _ => ensure!(
                self.profile_updates.is_empty(),
                "frozen modes capture first batch, no updates"
            ),
        }
        ensure!(
            !(self.cleared && self.query_cleared),
            "combined clearing forbidden"
        );
        if self.mode == Mode::EvalFinal {
            ensure!(
                matches!(
                    self.cohort.as_deref(),
                    Some("fit" | "heldout" | "cached_visual")
                ),
                "unknown cohort"
            );
            ensure!(
                [1, 2, 4, 8].contains(&self.loops),
                "unregistered evaluation depth"
            );
            if self.cleared || self.query_cleared || self.cohort.as_deref() == Some("cached_visual")
            {
                ensure!(self.loops == 4, "control/cached evaluation requires4loops");
            }
            ensure!(
                self.cohort.as_deref() != Some("cached_visual")
                    || !(self.cleared || self.query_cleared),
                "cached visual controls not registered"
            );
        } else {
            ensure!(
                self.loops == 4 && !self.cleared && !self.query_cleared,
                "only final evaluation changes depth/inputs"
            );
            ensure!(
                self.mode != Mode::EvalInitial
                    || matches!(self.cohort.as_deref(), Some("fit" | "heldout")),
                "initial evaluation requires abstract cohort"
            );
        }
        ensure!(
            self.mode != Mode::Qualify || self.physical_batch == 4,
            "qualification requiresbatch4"
        );
        Ok(())
    }
    fn validate(&self) -> Result<()> {
        self.validate_mode()?;
        ensure!(
            self.schema == "looped-action-binding-config-v2",
            "unknown config schema"
        );
        ensure!(
            self.source_revision == env!("TOFY_EMBEDDED_SOURCE_REVISION"),
            "source differs from binary"
        );
        ensure!(
            self.output_dir.is_absolute()
                && !self.output_dir.exists()
                && !self
                    .output_dir
                    .file_name()
                    .context("output basename missing")?
                    .to_string_lossy()
                    .contains('.'),
            "new absolute dotless output required"
        );
        ensure!(
            self.registration.is_absolute()
                && file_hash(&self.registration)? == self.registration_sha256,
            "registration differs"
        );
        ensure!(
            self.dataset.is_absolute() && file_hash(&self.dataset)? == self.dataset_sha256,
            "dataset differs"
        );
        if let Some(path) = &self.checkpoint {
            ensure!(
                path.is_absolute() && file_hash(path)? == *self.checkpoint_sha256.as_ref().unwrap(),
                "checkpoint differs"
            );
        }
        Ok(())
    }
    fn deadline(&self, started: Instant) -> Result<()> {
        ensure!(
            started.elapsed().as_secs_f64() < self.max_seconds as f64,
            "model deadline exceeded"
        );
        Ok(())
    }
    fn source_kind(&self) -> &'static str {
        if self.cohort.as_deref() == Some("cached_visual") {
            "precomputed_visual"
        } else {
            "abstract_effects"
        }
    }
    fn intervention(&self) -> engine::Intervention {
        engine::Intervention {
            cleared: self.cleared,
            query_cleared: self.query_cleared,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ScheduleKind {
    Training,
    Smoke,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Schedule {
    kind: ScheduleKind,
    effective_batch: usize,
    presentations: usize,
    indices_sha256: String,
}

fn schedule_hash(updates: &[Vec<usize>]) -> String {
    let mut hash = Sha256::new();
    for &index in updates.iter().flatten() {
        hash.update((index as u32).to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

struct Dataset {
    fit: Vec<engine::Row>,
    heldout: Vec<engine::Row>,
    cached_visual: Vec<engine::Row>,
    updates: Vec<Vec<usize>>,
    schedule: Schedule,
}
impl Dataset {
    fn parse(value: Value, config: &Config) -> Result<Self> {
        config.validate_mode()?;
        ensure!(
            value["schema"] == "looped-action-binding-data-v2",
            "unknown dataset schema"
        );
        let rows = |name: &str, count: usize| -> Result<Vec<engine::Row>> {
            let rows = value[name].as_array().context("missing dataset cohort")?;
            ensure!(
                rows.len() == count,
                "dataset cohort row count differs: {name}"
            );
            rows.iter()
                .enumerate()
                .map(|(index, row)| {
                    ensure!(
                        row["index"].as_u64() == Some(index as u64),
                        "dataset row order differs"
                    );
                    engine::Row::parse(row.clone())
                })
                .collect()
        };
        let updates: Vec<Vec<usize>> = serde_json::from_value(value["updates"].clone())?;
        let schedule: Schedule = serde_json::from_value(value["schedule"].clone())?;
        ensure!(
            schedule.effective_batch == config.effective_batch
                && schedule.presentations == config.schedule_presentations
                && schedule.indices_sha256 == config.schedule_sha256
                && (schedule.kind == ScheduleKind::Smoke) == (config.mode == Mode::BatchSmoke),
            "schedule/config identity differs"
        );
        ensure!(
            updates.len() == schedule.presentations.div_ceil(schedule.effective_batch)
                && updates.iter().enumerate().all(|(i, u)| {
                    let expected = if i + 1 == updates.len() {
                        (schedule.presentations - 1) % schedule.effective_batch + 1
                    } else {
                        schedule.effective_batch
                    };
                    u.len() == expected && u.iter().all(|&index| index < 1536)
                })
                && schedule_hash(&updates) == schedule.indices_sha256,
            "invalid fixed update stream"
        );
        Ok(Self {
            fit: rows("fit", 1536)?,
            heldout: rows("heldout", 768)?,
            cached_visual: rows("cached_visual", 1024)?,
            updates,
            schedule,
        })
    }
    fn cohort(&self, name: &str) -> Result<&[engine::Row]> {
        match name {
            "fit" => Ok(&self.fit),
            "heldout" => Ok(&self.heldout),
            "cached_visual" => Ok(&self.cached_visual),
            _ => anyhow::bail!("unknown cohort"),
        }
    }
    fn update(&self, index: usize) -> Result<Vec<engine::Row>> {
        self.updates
            .get(index)
            .context("update outside frozen schedule")?
            .iter()
            .map(|&i| {
                self.fit
                    .get(i)
                    .cloned()
                    .context("fit index outside dataset")
            })
            .collect()
    }
}

fn capture(
    config: &Config,
    model: &engine::Model,
    step: usize,
    training: bool,
    rows: usize,
) -> Result<LoopedCapture> {
    LoopedCapture::begin_binding(
        &config
            .output_dir
            .with_extension("profiles")
            .join(format!("update-{step:012}")),
        step as u64,
        &model.device,
        training.then_some(&model.vars),
        if training {
            config.physical_batch.min(rows)
        } else {
            rows
        },
        rows,
        config.loops,
        config.source_kind(),
        config.model_kind,
    )
}

fn evaluated_row(
    row: &engine::Row,
    logits: &[f32],
    model_input_sha256: &str,
    config: &Config,
) -> Value {
    let mut output = row.raw.clone();
    output["logits"] = json!(logits);
    output["stage"] = json!(if config.mode == Mode::EvalFinal {
        "final"
    } else {
        "initial"
    });
    output["loops"] = json!(config.loops);
    output["cleared"] = json!(config.cleared);
    output["query_cleared"] = json!(config.query_cleared);
    output["model_input_sha256"] = json!(model_input_sha256);
    output["model_kind"] = json!(config.model_kind);
    output
}

fn evaluate(
    config: &Config,
    data: &Dataset,
    model: &engine::Model,
    started: Instant,
) -> Result<Value> {
    let before = model.snapshot()?;
    let frozen = model.frozen()?;
    let qualify = config.mode == Mode::Qualify;
    let rows = if qualify {
        &data.fit[..4]
    } else {
        data.cohort(
            config
                .cohort
                .as_deref()
                .context("evaluation cohort missing")?,
        )?
    };
    let filename = if qualify {
        "qualification-rows.jsonl"
    } else {
        "evaluation-rows.jsonl"
    };
    let mut output = BufWriter::new(File::create(config.output_dir.join(filename))?);
    let mut total_ce = 0.0;
    let mut correct = 0;
    for (index, chunk) in rows.chunks(config.physical_batch).enumerate() {
        config.deadline(started)?;
        let cap = (index == 0)
            .then(|| capture(config, model, 1, false, chunk.len()))
            .transpose()?;
        let result = engine::evaluate(
            &frozen,
            chunk,
            config.loops,
            config.intervention(),
            &model.device,
            cap.as_ref(),
        )?;
        total_ce += result.mean_ce * chunk.len() as f64;
        correct += result.correct;
        for ((row, logits), input_hash) in chunk
            .iter()
            .zip(&result.logits)
            .zip(&result.model_input_sha256)
        {
            serde_json::to_writer(&mut output, &evaluated_row(row, logits, input_hash, config))?;
            output.write_all(b"\n")?;
        }
        if let Some(cap) = cap {
            cap.finish()?;
        }
    }
    output.flush()?;
    let changes = model.changes(&before)?;
    ensure!(
        changes.all_parameters_unchanged,
        "evaluation changed parameters"
    );
    Ok(
        json!({"status":"complete_pending_analysis","classification":if qualify { "implementation_smoke" } else { "single_seed_screen" },"optimizer_updates":0,"input_rows":rows.len(),"physical_batch":config.physical_batch,"actual_physical_batch":config.physical_batch.min(rows.len()),"microbatches":rows.len().div_ceil(config.physical_batch),"tail_batch":(rows.len()-1)%config.physical_batch+1,"cohort":config.cohort,"loops":config.loops,"cleared":config.cleared,"query_cleared":config.query_cleared,"input_source":config.source_kind(),"executed_vision_core_forwards":0,"mean_ce":total_ce / rows.len() as f64,"correct":correct,"changes":changes,"evaluated_parameter_sha256":engine::parameter_digest(&before),"elapsed_seconds":started.elapsed().as_secs_f64()}),
    )
}

fn train(
    config: &Config,
    data: &Dataset,
    model: &engine::Model,
    started: Instant,
) -> Result<Value> {
    let before = model.snapshot()?;
    let mut optimizer = model.optimizer()?;
    let mut log = BufWriter::new(File::create(config.output_dir.join("updates.jsonl"))?);
    let mut stream = BufWriter::new(File::create(
        config.output_dir.join("training-stream.jsonl"),
    )?);
    let update_started = Instant::now();
    let mut last = Value::Null;
    for update in 1..=config.updates {
        config.deadline(started)?;
        let rows = data.update(update - 1)?;
        for (slot, row) in rows.iter().enumerate() {
            serde_json::to_writer(
                &mut stream,
                &json!({"update":update,"slot":slot,"dataset_index":data.updates[update-1][slot],"row":row.raw}),
            )?;
            stream.write_all(b"\n")?;
        }
        let selected = config.profile_updates.contains(&update);
        let cap = selected
            .then(|| capture(config, model, update, true, rows.len()))
            .transpose()?;
        let metrics = engine::train_update(
            model,
            &rows,
            config.physical_batch,
            config.effective_batch,
            &mut optimizer,
            cap.as_ref(),
        )?;
        if let Some(cap) = cap {
            cap.finish()?;
        }
        last = serde_json::to_value(metrics)?;
        let record = json!({"update":update,"metrics":last,"elapsed_seconds":started.elapsed().as_secs_f64()});
        serde_json::to_writer(&mut log, &record)?;
        log.write_all(b"\n")?;
        log.flush()?;
        if update == 1 || update.is_multiple_of(100) || update == config.updates {
            println!("{record}");
        }
    }
    let updates_elapsed = update_started.elapsed().as_secs_f64();
    log.flush()?;
    stream.flush()?;
    let changes = model.changes(&before)?;
    ensure!(
        !changes.changed_body_names.is_empty() && !changes.changed_head_names.is_empty(),
        "body/head did not change"
    );
    let final_snapshot = model.snapshot()?;
    let final_parameter_sha256 = engine::parameter_digest(&final_snapshot);
    let final_shared_core_parameter_sha256 = engine::shared_core_parameter_digest(&final_snapshot);
    let checkpoint_started = Instant::now();
    let final_path = config.output_dir.join("final.safetensors");
    model.save(&final_path)?;
    let checkpoint_seconds = checkpoint_started.elapsed().as_secs_f64();
    let mut restoration = Value::Null;
    let mut restored_sha256 = Value::Null;
    if config.mode == Mode::BatchSmoke {
        model.restore(&before)?;
        let restored = config.output_dir.join("restored.safetensors");
        model.save(&restored)?;
        restoration = serde_json::to_value(model.changes(&before)?)?;
        restored_sha256 = json!(file_hash(&restored)?);
    }
    Ok(
        json!({"status":"complete_pending_analysis","classification":if config.mode == Mode::Train { "single_seed_screen" } else { "implementation_smoke" },"optimizer_updates":config.updates,"input_rows":data.schedule.presentations,"physical_batch":config.physical_batch,"effective_batch":config.effective_batch,"accumulation":config.effective_batch.div_ceil(config.physical_batch),"actual_physical_batch":config.physical_batch.min(data.updates[0].len()),"final_update_rows":data.updates.last().unwrap().len(),"final_update_physical_batch":config.physical_batch.min(data.updates.last().unwrap().len()),"tail_update_rows":if data.updates.last().unwrap().len()<config.effective_batch {json!(data.updates.last().unwrap().len())} else {Value::Null},"profile_updates":config.profile_updates,"loops":4,"cleared":false,"query_cleared":false,"input_source":"abstract_effects","executed_vision_core_forwards":0,"changes":changes,"restored_changes":restoration,"restored_sha256":restored_sha256,"updates_elapsed_seconds":updates_elapsed,"checkpoint_seconds":checkpoint_seconds,"last_update":last,"final_sha256":file_hash(&final_path)?,"final_parameter_sha256":final_parameter_sha256,"final_shared_core_parameter_sha256":final_shared_core_parameter_sha256,"elapsed_seconds":started.elapsed().as_secs_f64()}),
    )
}

fn run(config: &Config, started: Instant) -> Result<Value> {
    let source = evidence::provenance()?;
    ensure!(
        std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() == Ok("0"),
        "TF32 override must be zero"
    );
    #[cfg(feature = "cudnn")]
    ensure!(
        !candle_core::cuda_backend::gemm_reduced_precision_f32(),
        "reduced precision F32 forbidden"
    );
    ensure!(
        fs::metadata(&config.dataset)?.len() <= 64 * 1024 * 1024,
        "dataset exceeds bound"
    );
    let dataset = Dataset::parse(serde_json::from_slice(&fs::read(&config.dataset)?)?, config)?;
    let device = Device::new_cuda(0)?;
    let model = engine::Model::new(&device, config.model_kind)?;
    let initial_snapshot = model.snapshot()?;
    let initial_parameter_sha256 = engine::parameter_digest(&initial_snapshot);
    let initial_shared_core_parameter_sha256 =
        engine::shared_core_parameter_digest(&initial_snapshot);
    let initial_path = config.output_dir.join("initial.safetensors");
    model.save(&initial_path)?;
    if let Some(path) = &config.checkpoint {
        model.load(path)?;
    }
    let starting_snapshot = model.snapshot()?;
    let starting_parameter_sha256 = engine::parameter_digest(&starting_snapshot);
    let starting_shared_core_parameter_sha256 =
        engine::shared_core_parameter_digest(&starting_snapshot);
    write_json(
        &config.output_dir.join("metadata.json"),
        &json!({"schema":"looped-action-binding-v2","config":config,"provenance":source,"parameter_count":config.model_kind.parameter_count(),"model_kind":config.model_kind,"parameter_digest_schema":engine::DIGEST_SCHEMA,"initial_parameter_sha256":initial_parameter_sha256,"input_source":config.source_kind(),"seed":0,"objective":"policy_cross_entropy_only","executed_vision_core_forwards":0,"effective_batch":if matches!(config.mode, Mode::Train | Mode::BatchSmoke) { config.effective_batch } else { config.physical_batch },"deferred":["online_visual_adapter","reward","value","dynamics","planner","ARC_evaluation"]}),
    )?;
    config.deadline(started)?;
    let mut report = if matches!(config.mode, Mode::Train | Mode::BatchSmoke) {
        train(config, &dataset, &model, started)?
    } else {
        evaluate(config, &dataset, &model, started)?
    };
    device.synchronize()?;
    ensure!(
        file_hash(&config.dataset)? == config.dataset_sha256,
        "dataset changed during invocation"
    );
    if let Some(path) = &config.checkpoint {
        ensure!(
            file_hash(path)? == *config.checkpoint_sha256.as_ref().unwrap(),
            "checkpoint changed during invocation"
        );
    }
    config.deadline(started)?;
    report["parameter_count"] = json!(config.model_kind.parameter_count());
    report["model_kind"] = json!(config.model_kind);
    report["requested_physical_batch"] = json!(config.physical_batch);
    report["parameter_digest_schema"] = json!(engine::DIGEST_SCHEMA);
    report["initial_parameter_sha256"] = json!(initial_parameter_sha256);
    report["starting_parameter_sha256"] = json!(starting_parameter_sha256);
    report["initial_shared_core_parameter_sha256"] = json!(initial_shared_core_parameter_sha256);
    report["starting_shared_core_parameter_sha256"] = json!(starting_shared_core_parameter_sha256);
    report["initial_sha256"] = json!(file_hash(&initial_path)?);
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    Ok(report)
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    config_sha256: String,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.config.is_absolute() && file_hash(&args.config)? == args.config_sha256,
        "configuration differs"
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

#[cfg(test)]
mod tests {
    use super::*;
    fn config(effective: usize, presentations: usize, mode: Mode) -> Config {
        let updates = presentations.div_ceil(effective);
        Config {
            schema: "looped-action-binding-config-v2".into(),
            source_revision: "fixture".into(),
            registration: "/fixture/registration".into(),
            registration_sha256: String::new(),
            dataset: "/fixture/data".into(),
            dataset_sha256: String::new(),
            mode,
            output_dir: "/fixture/new".into(),
            model_kind: ModelKind::Legacy,
            physical_batch: effective,
            effective_batch: effective,
            schedule_presentations: presentations,
            schedule_sha256: "0".repeat(64),
            profile_updates: match mode {
                Mode::Train => vec![2, 1 + updates / 2, updates],
                Mode::BatchSmoke => vec![1],
                _ => vec![],
            },
            updates: if matches!(mode, Mode::Train | Mode::BatchSmoke) {
                updates
            } else {
                0
            },
            max_seconds: if mode == Mode::Train { 600 } else { 120 },
            checkpoint: None,
            checkpoint_sha256: None,
            cohort: None,
            loops: 4,
            cleared: false,
            query_cleared: false,
        }
    }
    fn dataset_value(c: &mut Config) -> Value {
        // Synthetic fixed stream only; no registered dataset/checkpoint is read.
        let flat = (0..c.schedule_presentations)
            .map(|i| i % 1536)
            .collect::<Vec<_>>();
        let updates = flat
            .chunks(c.effective_batch)
            .map(|x| x.to_vec())
            .collect::<Vec<_>>();
        c.schedule_sha256 = schedule_hash(&updates);
        let rows = |count| {
            (0..count)
                .map(|i| engine::tests::row(i).raw)
                .collect::<Vec<_>>()
        };
        json!({"schema":"looped-action-binding-data-v2","fit":rows(1536),"heldout":rows(768),
            "cached_visual":rows(1024),"updates":updates,
            "schedule":{"kind":if c.mode==Mode::BatchSmoke{"smoke"}else{"training"},
                "effective_batch":c.effective_batch,"presentations":c.schedule_presentations,
                "indices_sha256":c.schedule_sha256}})
    }
    #[test]
    fn mode_guards_bind_model_batch_budget_and_capture_reachability() -> Result<()> {
        let mut c = config(1024, 4608, Mode::Train);
        c.validate_mode()?;
        for e in [512, 1024, 2048, 4096, 8192, 16384, 32768] {
            let mut c = config(e, e * 5, Mode::Train);
            for p in [1, e / 2, e] {
                c.physical_batch = p;
                c.validate_mode()?;
            }
        }
        for e in [0, 256, 513, 65536] {
            let mut bad = config(1024, 4608, Mode::Train);
            bad.effective_batch = e;
            assert!(bad.validate_mode().is_err());
        }
        for p in [0, 3, 2048] {
            c.physical_batch = p;
            assert!(c.validate_mode().is_err());
        }
        c.physical_batch = 1024;
        for captures in [vec![], vec![1, 3, 5], vec![2, 2, 5], vec![2, 3, 6]] {
            c.profile_updates = captures;
            assert!(c.validate_mode().is_err());
        }
        c.profile_updates = vec![2, 3, 5];
        c.updates = 4;
        assert!(c.validate_mode().is_err());
        c.updates = 5;
        c.query_cleared = true;
        assert!(c.validate_mode().is_err());
        c.query_cleared = false;
        let mut value = serde_json::to_value(&c)?;
        value["model_kind"] = json!("unknown");
        assert!(serde_json::from_value::<Config>(value).is_err());
        for n in [2, 5] {
            config(32768, 32768 * n, Mode::BatchSmoke).validate_mode()?;
        }
        assert!(config(1024, 1536, Mode::BatchSmoke)
            .validate_mode()
            .is_err());
        c = config(1024, 4608, Mode::EvalInitial);
        c.cohort = Some("fit".into());
        c.validate_mode()?;
        c.cohort = Some("cached_visual".into());
        assert!(c.validate_mode().is_err());
        c.mode = Mode::EvalFinal;
        c.checkpoint = Some("/fixture/final".into());
        c.checkpoint_sha256 = Some("0".repeat(64));
        c.validate_mode()?;
        c.loops = 8;
        assert!(c.validate_mode().is_err());
        c.cohort = Some("heldout".into());
        for loops in [1, 2, 4, 8] {
            c.loops = loops;
            c.validate_mode()?;
        }
        c.query_cleared = true;
        assert!(c.validate_mode().is_err());
        c.loops = 4;
        c.validate_mode()?;
        c.cleared = true;
        assert!(c.validate_mode().is_err());
        c.query_cleared = false;
        c.validate_mode()?;
        c.max_seconds = 121;
        assert!(c.validate_mode().is_err());
        Ok(())
    }
    #[test]
    fn larger_schedules_preserve_the_complete_stream_and_explicit_tail() -> Result<()> {
        let mut identity = None;
        for effective in [512, 1024, 2048, 32768] {
            let mut c = config(effective, 588800, Mode::Train);
            let value = dataset_value(&mut c);
            let data = Dataset::parse(value, &c)?;
            assert_eq!(data.updates.len(), 588800usize.div_ceil(effective));
            assert_eq!(
                data.updates.last().unwrap().len(),
                (588800 - 1) % effective + 1
            );
            for (i, &index) in data.updates.iter().flatten().enumerate() {
                assert_eq!(index, i % 1536);
            }
            assert_eq!(data.updates.iter().map(Vec::len).sum::<usize>(), 588800);
            if let Some(expected) = &identity {
                assert_eq!(&data.schedule.indices_sha256, expected);
            } else {
                identity = Some(data.schedule.indices_sha256);
            }
        }
        Ok(())
    }
    #[test]
    fn smoke_uses_full_candidate_batches_and_preserves_raw_export() -> Result<()> {
        let mut c = config(32768, 5 * 32768, Mode::BatchSmoke);
        let value = dataset_value(&mut c);
        let data = Dataset::parse(value, &c)?;
        assert!(data.updates.iter().all(|u| u.len() == 32768));
        assert_eq!(data.update(4)?.len(), 32768);
        assert!(data.update(5).is_err());
        c.mode = Mode::EvalFinal;
        c.model_kind = ModelKind::Equivariant;
        c.query_cleared = true;
        let row = &data.fit[7];
        let before = row.raw.clone();
        let output = evaluated_row(row, &[0.; 4], "actual-input-hash", &c);
        for (key, value) in before.as_object().unwrap() {
            assert_eq!(&output[key], value);
        }
        assert_eq!(output["stage"], "final");
        assert_eq!(output["model_kind"], "equivariant");
        assert_eq!(output["query_cleared"], true);
        assert_eq!(row.raw, before);
        Ok(())
    }
    #[test]
    fn malformed_schedule_population_or_kind_fails_before_model() {
        for corruption in 0..8 {
            let mut c = config(1024, 4608, Mode::Train);
            let mut value = dataset_value(&mut c);
            match corruption {
                0 => {
                    value["fit"].as_array_mut().unwrap().pop();
                }
                1 => {
                    value["updates"].as_array_mut().unwrap().pop();
                }
                2 => value["updates"][0][0] = json!(1536),
                3 => value["fit"][0]["index"] = json!(1),
                4 => value["updates"][0] = json!(vec![0; 512]),
                5 => value["schedule"]["kind"] = json!("smoke"),
                6 => value["schedule"]["presentations"] = json!(4609),
                _ => value["updates"][0][0] = json!(1),
            }
            assert!(Dataset::parse(value, &c).is_err());
        }
    }
}
