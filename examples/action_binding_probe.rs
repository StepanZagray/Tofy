//! Standalone action-binding screen; no vision core or simulator executes here.
#[path = "action_binding/engine.rs"]
mod engine;
#[path = "grounded_policy/evidence.rs"]
mod evidence;

use anyhow::{ensure, Context, Result};
use candle_core::Device;
use clap::Parser;
use engine::EFFECTIVE;
use evidence::{file_hash, write_json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};
use tofy::p2::looped_agent::{binding::PARAMETERS, profile::LoopedCapture};

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
    physical_batch: usize,
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
            self.physical_batch.is_power_of_two() && self.physical_batch <= EFFECTIVE,
            "physical batch must be a power of two in 1..={EFFECTIVE}"
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
                    Mode::Train => 1150,
                    Mode::BatchSmoke => self.updates,
                    _ => 0,
                },
            "update count differs"
        );
        ensure!(
            self.mode != Mode::BatchSmoke || matches!(self.updates, 2 | 5),
            "smoke requires2or5 updates"
        );
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
            self.schema == "looped-action-binding-config-v1",
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

struct Dataset {
    fit: Vec<engine::Row>,
    heldout: Vec<engine::Row>,
    cached_visual: Vec<engine::Row>,
    updates: Vec<Vec<usize>>,
}
impl Dataset {
    fn parse(value: Value) -> Result<Self> {
        ensure!(
            value["schema"] == "looped-action-binding-data-v1",
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
        ensure!(
            updates.len() == 1150
                && updates
                    .iter()
                    .all(|u| u.len() == EFFECTIVE && u.iter().all(|&i| i < 1536)),
            "invalid fixed update stream"
        );
        Ok(Self {
            fit: rows("fit", 1536)?,
            heldout: rows("heldout", 768)?,
            cached_visual: rows("cached_visual", 1024)?,
            updates,
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
        rows,
        if training { EFFECTIVE } else { rows },
        config.loops,
        config.source_kind(),
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
        json!({"status":"complete_pending_analysis","classification":if qualify { "implementation_smoke" } else { "single_seed_screen" },"optimizer_updates":0,"input_rows":rows.len(),"physical_batch":config.physical_batch,"cohort":config.cohort,"loops":config.loops,"cleared":config.cleared,"query_cleared":config.query_cleared,"input_source":config.source_kind(),"executed_vision_core_forwards":0,"mean_ce":total_ce / rows.len() as f64,"correct":correct,"changes":changes,"evaluated_parameter_sha256":engine::parameter_digest(&before),"elapsed_seconds":started.elapsed().as_secs_f64()}),
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
        let selected = if config.mode == Mode::Train {
            [2, 100, 1150].contains(&update)
        } else {
            update == 1
        };
        let cap = selected
            .then(|| capture(config, model, update, true, config.physical_batch))
            .transpose()?;
        let metrics = engine::train_update(
            model,
            &rows,
            config.physical_batch,
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
    let final_parameter_sha256 = engine::parameter_digest(&model.snapshot()?);
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
        json!({"status":"complete_pending_analysis","classification":if config.mode == Mode::Train { "single_seed_screen" } else { "implementation_smoke" },"optimizer_updates":config.updates,"input_rows":config.updates*EFFECTIVE,"physical_batch":config.physical_batch,"effective_batch":EFFECTIVE,"accumulation":EFFECTIVE.div_ceil(config.physical_batch),"loops":4,"cleared":false,"query_cleared":false,"input_source":"abstract_effects","executed_vision_core_forwards":0,"changes":changes,"restored_changes":restoration,"restored_sha256":restored_sha256,"updates_elapsed_seconds":updates_elapsed,"checkpoint_seconds":checkpoint_seconds,"last_update":last,"final_sha256":file_hash(&final_path)?,"final_parameter_sha256":final_parameter_sha256,"elapsed_seconds":started.elapsed().as_secs_f64()}),
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
    let dataset = Dataset::parse(serde_json::from_slice(&fs::read(&config.dataset)?)?)?;
    let device = Device::new_cuda(0)?;
    let model = engine::Model::new(&device)?;
    let initial_parameter_sha256 = engine::parameter_digest(&model.snapshot()?);
    let initial_path = config.output_dir.join("initial.safetensors");
    model.save(&initial_path)?;
    if let Some(path) = &config.checkpoint {
        model.load(path)?;
    }
    let starting_parameter_sha256 = engine::parameter_digest(&model.snapshot()?);
    write_json(
        &config.output_dir.join("metadata.json"),
        &json!({"schema":"looped-action-binding-v1","config":config,"provenance":source,"parameter_count":PARAMETERS,"parameter_digest_schema":engine::DIGEST_SCHEMA,"initial_parameter_sha256":initial_parameter_sha256,"input_source":config.source_kind(),"seed":0,"objective":"policy_cross_entropy_only","executed_vision_core_forwards":0,"effective_batch":if matches!(config.mode, Mode::Train | Mode::BatchSmoke) { EFFECTIVE } else { config.physical_batch },"deferred":["online_visual_adapter","reward","value","dynamics","planner","ARC_evaluation"]}),
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
    report["parameter_count"] = json!(PARAMETERS);
    report["parameter_digest_schema"] = json!(engine::DIGEST_SCHEMA);
    report["initial_parameter_sha256"] = json!(initial_parameter_sha256);
    report["starting_parameter_sha256"] = json!(starting_parameter_sha256);
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
    fn config() -> Config {
        Config {
            schema: "looped-action-binding-config-v1".into(),
            source_revision: "fixture".into(),
            registration: "/fixture/registration".into(),
            registration_sha256: String::new(),
            dataset: "/fixture/data".into(),
            dataset_sha256: String::new(),
            mode: Mode::Train,
            output_dir: "/fixture/new".into(),
            physical_batch: EFFECTIVE,
            updates: 1150,
            max_seconds: 600,
            checkpoint: None,
            checkpoint_sha256: None,
            cohort: None,
            loops: 4,
            cleared: false,
            query_cleared: false,
        }
    }
    #[test]
    fn mode_guards_keep_training_and_controls_separate() -> Result<()> {
        let mut c = config();
        c.validate_mode()?;
        for physical in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
            c.physical_batch = physical;
            c.validate_mode()?;
        }
        for physical in [0, 3, 511, 513, 1024] {
            c.physical_batch = physical;
            assert!(c.validate_mode().is_err());
        }
        c.physical_batch = EFFECTIVE;
        c.updates = 1149;
        assert!(c.validate_mode().is_err());
        c.updates = 1150;
        c.query_cleared = true;
        assert!(c.validate_mode().is_err());
        c.query_cleared = false;
        c.mode = Mode::BatchSmoke;
        c.max_seconds = 120;
        for n in [2, 5] {
            c.updates = n;
            c.validate_mode()?;
        }
        c.updates = 3;
        assert!(c.validate_mode().is_err());
        c.mode = Mode::EvalInitial;
        c.updates = 0;
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
    fn dataset_value() -> Value {
        let rows = |count| {
            (0..count)
                .map(|i| engine::tests::row(i).raw)
                .collect::<Vec<_>>()
        };
        json!({"schema":"looped-action-binding-data-v1","fit":rows(1536),"heldout":rows(768),"cached_visual":rows(1024),"updates":vec![(0..EFFECTIVE).rev().collect::<Vec<_>>();1150]})
    }
    #[test]
    fn dataset_replay_and_export_preserve_audit_boundaries() -> Result<()> {
        let data = Dataset::parse(dataset_value())?;
        for index in [0, 1149] {
            let batch = data.update(index)?;
            assert_eq!(batch.len(), EFFECTIVE);
            for (slot, row) in batch.iter().enumerate() {
                assert_eq!(row.raw, data.fit[EFFECTIVE - 1 - slot].raw);
            }
        }
        assert!(data.update(1150).is_err());
        let mut c = config();
        c.mode = Mode::EvalFinal;
        c.loops = 4;
        c.query_cleared = true;
        let row = &data.fit[7];
        let before = row.raw.clone();
        let exported = evaluated_row(row, &[0.0; 4], "actual-clamped-tensor-hash", &c);
        for (key, value) in before.as_object().unwrap() {
            assert_eq!(&exported[key], value);
        }
        assert_eq!(exported["stage"], "final");
        assert_eq!(exported["query_cleared"], true);
        assert_eq!(exported["model_input_sha256"], "actual-clamped-tensor-hash");
        assert_eq!(row.raw, before);
        Ok(())
    }
    #[test]
    fn bad_schedule_or_population_fails_before_model_construction() {
        for kind in 0..6 {
            let mut value = dataset_value();
            match kind {
                0 => {
                    value["fit"].as_array_mut().unwrap().pop();
                }
                1 => {
                    value["updates"].as_array_mut().unwrap().pop();
                }
                2 => value["updates"][0][0] = json!(1536),
                3 => value["fit"][0]["index"] = json!(1),
                4 => value["fit"][0]["input_sha256"] = json!("0".repeat(64)),
                _ => value["updates"][0] = json!(vec![0; 64]),
            }
            assert!(Dataset::parse(value).is_err());
        }
    }
}
