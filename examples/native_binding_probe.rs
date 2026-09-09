//! Frozen online composition, with a separate CPU public-input audit.
#[path = "native_binding/data.rs"]
mod data;
#[path = "native_binding/engine.rs"]
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
    collections::HashSet,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    process::Command,
    time::Instant,
};
use tofy::p2::looped_agent::{native_binding, profile::LoopedCapture};

const CORE: &str = "4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802";
const HEAD: &str = "a43ad78fe7a6b1594d23c205fb191c8ac81610deaaff25198cb421d423edc678";
const IMPORT: &str = "d2f9a7a99917d180646a5e28e26acd92cbd3fe53919fa009da3e11747e0c158e";
const BINDER: &str = "d2deeba7b0fafc2386c9a19d39bc99b7534b91a0155d66e77792a5c4037bd717";
const HISTORY: [&str; 6] = [
    "09559eb4dd1b933de724da81aef3bb80547e010a00e38e39de31c9f72fab1c6d",
    "bd72fd9707759dd19074a22b9d62899419b28711722b3617c21cc277a6a25c0d",
    "09a7aa423b17f32bbae14ae9178a15d5fff8428936afdf23ebc4f09878a6ddca",
    "61ae87a267cc545b05884dc412160c0313a0c2bc1015cfa2323bde256f91eaeb",
    "55f200b3a136476896d57f284a7d5db4e8f8fc2b7b4fa54ef309d1837f3c5910",
    "fbb137383a16a27d72141ef66028c1c9cb470afccc2b03864ec0a961a94d73f2",
];
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Audit,
    Qualify,
    BatchSmoke,
    Evaluate,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Control {
    Factual,
    UniformAttention,
}
impl Control {
    fn native(self) -> native_binding::Control {
        match self {
            Self::Factual => native_binding::Control::Factual,
            Self::UniformAttention => native_binding::Control::UniformAttention,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Factual => "factual",
            Self::UniformAttention => "uniform_attention",
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct History {
    path: PathBuf,
    sha256: String,
}
impl History {
    fn verify(&self) -> Result<()> {
        ensure!(
            self.path.is_absolute() && file_hash(&self.path)? == self.sha256,
            "historical exclusion binding differs"
        );
        Ok(())
    }
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
    core_loops: usize,
    binder_loops: usize,
    control: Control,
    panel_seed: u64,
    panel_tag: u64,
    query_groups: usize,
    history: Vec<History>,
    audit_root: Option<PathBuf>,
    audit_manifest_sha256: Option<String>,
    core_checkpoint: PathBuf,
    core_sha256: String,
    head_checkpoint: PathBuf,
    head_sha256: String,
    import_manifest: PathBuf,
    binder_checkpoint: PathBuf,
    binder_sha256: String,
}
impl Config {
    fn mode_guards(&self) -> Result<()> {
        ensure!(
            self.schema == "looped-native-binding-config-v1"
                && self.updates == 0
                && self.core_loops == 4
                && self.binder_loops == 4
                && self.panel_seed == 20260923
                && self.panel_tag == 0x4e415449564542
                && self.query_groups == 32
                && (1..=120).contains(&self.max_seconds),
            "registered mode/population/depth/budget differs"
        );
        ensure!(
            self.mode == Mode::Evaluate || self.control == Control::Factual,
            "only evaluation permits uniform control"
        );
        let audit = self.mode == Mode::Audit;
        ensure!(
            self.audit_root.is_some() == self.audit_manifest_sha256.is_some()
                && audit != self.audit_root.is_some(),
            "audit binding differs"
        );
        ensure!(
            if audit {
                self.physical_batch == 1
                    && self.history.len() == 6
                    && self.history.iter().map(|h| h.sha256.as_str()).eq(HISTORY)
                    && self
                        .history
                        .iter()
                        .map(|h| &h.path)
                        .collect::<HashSet<_>>()
                        .len()
                        == 6
            } else {
                self.history.is_empty()
                    && if self.mode == Mode::Qualify {
                        self.physical_batch == 4
                    } else {
                        [32, 64, 128, 256, 512, 1024].contains(&self.physical_batch)
                    }
            },
            "batch/exclusion contract differs"
        );
        ensure!(
            self.core_sha256 == CORE && self.head_sha256 == HEAD && self.binder_sha256 == BINDER,
            "frozen checkpoint selection differs"
        );
        Ok(())
    }
    fn validate(&self) -> Result<()> {
        self.mode_guards()?;
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
                    .context("missing basename")?
                    .to_string_lossy()
                    .contains('.'),
            "new absolute dotless root required"
        );
        ensure!(
            self.registration.is_absolute()
                && file_hash(&self.registration)? == self.registration_sha256,
            "registration hash differs"
        );
        if self.mode != Mode::Audit {
            self.verify_checkpoints()?;
        }
        Ok(())
    }
    fn verify_checkpoints(&self) -> Result<()> {
        for (p, h) in [
            (&self.core_checkpoint, CORE),
            (&self.head_checkpoint, HEAD),
            (&self.import_manifest, IMPORT),
            (&self.binder_checkpoint, BINDER),
        ] {
            ensure!(
                p.is_absolute() && file_hash(p)? == h,
                "frozen checkpoint/import bytes differ"
            );
        }
        Ok(())
    }
    fn deadline(&self, started: Instant) -> Result<()> {
        ensure!(
            started.elapsed().as_secs_f64() < self.max_seconds as f64,
            "invocation deadline exceeded"
        );
        Ok(())
    }
}

fn cpu_provenance() -> Result<Value> {
    for (root, revision, dirty, pushed) in [
        (
            PathBuf::from(env!("CARGO_MANIFEST_DIR")),
            env!("TOFY_EMBEDDED_SOURCE_REVISION"),
            env!("TOFY_EMBEDDED_SOURCE_DIRTY"),
            env!("TOFY_EMBEDDED_SOURCE_PUSHED"),
        ),
        (
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../candle_graph"),
            env!("TOFY_EMBEDDED_CANDLE_GRAPH_REVISION"),
            env!("TOFY_EMBEDDED_CANDLE_GRAPH_DIRTY"),
            env!("TOFY_EMBEDDED_CANDLE_GRAPH_PUSHED"),
        ),
    ] {
        ensure!(
            dirty == "false" && pushed == "true",
            "audit build must use clean pushed source"
        );
        let git = |args: &[&str]| -> Result<String> {
            let out = Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(args)
                .output()?;
            ensure!(out.status.success(), "source provenance failed");
            Ok(String::from_utf8(out.stdout)?.trim().into())
        };
        ensure!(
            git(&["rev-parse", "HEAD"])? == revision
                && git(&["status", "--porcelain", "--untracked-files=all"])?.is_empty(),
            "audit source drift"
        );
        git(&["merge-base", "--is-ancestor", "HEAD", "@{upstream}"])?;
    }
    Ok(
        json!({"device":"cpu","gpu":null,"source_revision":env!("TOFY_EMBEDDED_SOURCE_REVISION"),"candle_graph_revision":env!("TOFY_EMBEDDED_CANDLE_GRAPH_REVISION"),"binary_sha256":file_hash(&std::env::current_exe()?)?,"features":env!("TOFY_EMBEDDED_CARGO_FEATURES")}),
    )
}

fn selected_indices(mode: Mode, physical: usize, rows: usize) -> Vec<usize> {
    let count = match mode {
        Mode::Qualify => 4,
        Mode::BatchSmoke => physical,
        Mode::Evaluate => rows,
        Mode::Audit => 0,
    };
    (0..count).map(|i| i % rows).collect()
}
fn run(config: &Config, started: Instant) -> Result<Value> {
    if config.mode == Mode::Audit {
        write_json(
            &config.output_dir.join("metadata.json"),
            &json!({"schema":"looped-native-binding-v1","config":config,"device":"cpu","provenance":cpu_provenance()?,"objective":"public_input_audit","optimizer_updates":0,"model_forwards":0}),
        )?;
        return data::audit(config, started);
    }
    // Validate all public rows and labels on CPU before CUDA device construction.
    let rows = data::load(config)?;
    let public_inputs = rows
        .iter()
        .map(data::Row::inputs)
        .collect::<Result<Vec<_>>>()?;
    let indices = selected_indices(config.mode, config.physical_batch, rows.len());
    let source = evidence::provenance()?;
    ensure!(
        std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() == Ok("0"),
        "strict F32 override required"
    );
    #[cfg(feature = "cudnn")]
    ensure!(
        !candle_core::cuda_backend::gemm_reduced_precision_f32(),
        "reduced F32 precision forbidden"
    );
    let device = Device::new_cuda(0)?;
    let model = engine::Model::load(config, &device)?;
    let before = model.identity()?;
    write_json(&config.output_dir.join("parameters-before.json"), &before)?;
    write_json(
        &config.output_dir.join("metadata.json"),
        &json!({"schema":"looped-native-binding-v1","config":config,"provenance":source,"device":"cuda:0","objective":"frozen_native_composition","optimizer_updates":0,"privileged_role_warm_start":true,"input_source":"fresh_public_images","core_loops":4,"binder_loops":4,"ordinary_core_heads":"stored_not_executed","selector_output":"executed_discarded","parameter_identity":before,"deferred":["training","dynamics","reward","value","planner","ARC_evaluation"]}),
    )?;
    let mut file = BufWriter::new(File::create(
        config.output_dir.join("evaluation-rows.jsonl"),
    )?);
    let mut correct = 0;
    let mut total_ce = 0.;
    for (batch, chunk) in indices.chunks(config.physical_batch).enumerate() {
        config.deadline(started)?;
        let inputs = chunk.iter().map(|&i| &public_inputs[i]).collect::<Vec<_>>();
        let patches = Tensor::from_vec(
            inputs
                .iter()
                .flat_map(|x| x.patches.iter().copied())
                .collect::<Vec<_>>(),
            (chunk.len(), 448, 64),
            &device,
        )?;
        let metadata = Tensor::from_vec(
            inputs
                .iter()
                .flat_map(|x| x.metadata.iter().copied())
                .collect::<Vec<_>>(),
            (chunk.len(), 448, 10),
            &device,
        )?;
        let capture = (batch == 0)
            .then(|| {
                LoopedCapture::begin_native_binding(
                    &config
                        .output_dir
                        .with_extension("profiles")
                        .join("update-000000000001"),
                    1,
                    &device,
                    chunk.len(),
                    config.control.name(),
                )
            })
            .transpose()?;
        let output = model.forward(
            &patches,
            &metadata,
            config.control,
            &device,
            capture.as_ref(),
        )?;
        let attention = output
            .attention
            .reshape((chunk.len() * 7, 2, 64))?
            .to_vec3::<f32>()?;
        let records = output.records.to_vec3::<f32>()?;
        let logits = output.logits.to_vec2::<f32>()?;
        for (slot, &i) in chunk.iter().enumerate() {
            let row = &rows[i];
            let l = &logits[slot];
            let prediction = (0..4)
                .max_by(|&a, &b| l[a].total_cmp(&l[b]).then_with(|| b.cmp(&a)))
                .unwrap();
            correct += usize::from(prediction == row.policy_label);
            let max = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            total_ce += max + (l.iter().map(|&v| (f64::from(v) - max).exp()).sum::<f64>()).ln()
                - f64::from(l[row.policy_label]);
            let mut exported = serde_json::to_value(row)?;
            exported["evaluation_index"] = json!(batch * config.physical_batch + slot);
            exported["learned_attention"] = json!(attention[slot * 7..(slot + 1) * 7]);
            exported["adapter_records"] = json!(records[slot]);
            exported["logits"] = json!(l);
            exported["prediction"] = json!(prediction);
            exported["control"] = json!(config.control);
            serde_json::to_writer(&mut file, &exported)?;
            file.write_all(b"\n")?;
        }
        if let Some(c) = capture {
            c.finish()?;
        }
    }
    file.flush()?;
    device.synchronize()?;
    let after = model.identity()?;
    ensure!(before == after, "frozen parameters changed");
    write_json(&config.output_dir.join("parameters-after.json"), &after)?;
    config.verify_checkpoints()?;
    evidence::verify_root(
        config.audit_root.as_ref().unwrap(),
        config.audit_manifest_sha256.as_deref().unwrap(),
    )?;
    config.deadline(started)?;
    let batches = indices.len().div_ceil(config.physical_batch);
    Ok(
        json!({"status":"complete_pending_analysis","classification":if config.mode==Mode::Evaluate{"frozen_native_confirmation"}else{"implementation_smoke"},"optimizer_updates":0,"input_rows":indices.len(),"panel_rows":rows.len(),"unique_input_rows":indices.iter().collect::<HashSet<_>>().len(),"repeated_input_rows":indices.len()-indices.iter().collect::<HashSet<_>>().len(),"physical_batch":config.physical_batch,"actual_physical_batch":config.physical_batch.min(indices.len()),"microbatches":batches,"tail_batch":(indices.len()-1)%config.physical_batch+1,"core_loops":4,"binder_loops":4,"control":config.control,"model_forwards":batches,"core_forward_batches":batches,"selector_forward_batches":7*batches,"binder_forward_batches":batches,"mean_ce":total_ce/indices.len() as f64,"correct":correct,"privileged_role_warm_start":true,"current_frame_input_bitwise_equal":true,"current_frame_max_absolute_difference":0.0,"changes":{"all_parameters_unchanged":true,"unused_heads_unchanged":true},"parameter_digests_before":before,"parameter_digests_after":after,"core_sha256":CORE,"head_sha256":HEAD,"import_manifest_sha256":IMPORT,"binder_sha256":BINDER,"elapsed_seconds":started.elapsed().as_secs_f64()}),
    )
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
        "config binding differs"
    );
    let config: Config = serde_json::from_slice(&fs::read(&args.config)?)?;
    config.validate()?;
    let trace = tofy::perf::install()?;
    ensure!(
        cfg!(feature = "profiling") && trace.is_some(),
        "profiling/unique host trace required"
    );
    fs::create_dir(&config.output_dir)?;
    write_json(
        &config.output_dir.join("launch.json"),
        &json!({"status":"running","pid":std::process::id(),"exact_args":std::env::args().collect::<Vec<_>>(),"config_sha256":args.config_sha256,"binary_sha256":file_hash(&std::env::current_exe()?)?}),
    )?;
    let started = Instant::now();
    let result = run(&config, started);
    drop(trace);
    match &result {
        Ok(report) => write_json(&config.output_dir.join("report.json"), report)?,
        Err(e) => write_json(
            &config.output_dir.join("report.json"),
            &json!({"status":"failed_integrity_or_evaluation","error":format!("{e:#}"),"elapsed_seconds":started.elapsed().as_secs_f64()}),
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
            schema: "looped-native-binding-config-v1".into(),
            source_revision: String::new(),
            registration: "/fixture/reg".into(),
            registration_sha256: String::new(),
            mode: Mode::Qualify,
            output_dir: "/fixture/new".into(),
            physical_batch: 4,
            updates: 0,
            max_seconds: 120,
            core_loops: 4,
            binder_loops: 4,
            control: Control::Factual,
            panel_seed: 20260923,
            panel_tag: 0x4e415449564542,
            query_groups: 32,
            history: vec![],
            audit_root: Some("/fixture/audit".into()),
            audit_manifest_sha256: Some(String::new()),
            core_checkpoint: "/fixture/core".into(),
            core_sha256: CORE.into(),
            head_checkpoint: "/fixture/head".into(),
            head_sha256: HEAD.into(),
            import_manifest: "/fixture/import".into(),
            binder_checkpoint: "/fixture/binder".into(),
            binder_sha256: BINDER.into(),
        }
    }
    #[test]
    fn mode_population_checkpoint_and_control_guards() -> Result<()> {
        let mut c = config();
        c.mode_guards()?;
        c.mode = Mode::BatchSmoke;
        c.physical_batch = 1024;
        c.mode_guards()?;
        c.control = Control::UniformAttention;
        assert!(c.mode_guards().is_err());
        c.mode = Mode::Evaluate;
        c.mode_guards()?;
        c.panel_seed += 1;
        assert!(c.mode_guards().is_err());
        c.panel_seed -= 1;
        c.updates = 1;
        assert!(c.mode_guards().is_err());
        c.updates = 0;
        c.binder_sha256 = CORE.into();
        assert!(c.mode_guards().is_err());
        Ok(())
    }
    #[test]
    fn capacity_smoke_uses_requested_batch_and_reports_repetitions() {
        let indices = selected_indices(Mode::BatchSmoke, 1024, 768);
        assert_eq!(indices.len(), 1024);
        assert_eq!(indices.iter().collect::<HashSet<_>>().len(), 768);
        assert_eq!(indices[768], 0);
        assert_eq!(selected_indices(Mode::Evaluate, 1024, 768).len(), 768);
        assert_eq!(selected_indices(Mode::Qualify, 4, 768), vec![0, 1, 2, 3]);
    }
}
