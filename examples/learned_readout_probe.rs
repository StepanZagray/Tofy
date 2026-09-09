//! C9: train only a small policy readout on sealed, frozen feature prefixes.
#[path = "learned_readout_probe/cache.rs"]
mod cache;
#[path = "learned_readout_probe/head.rs"]
mod head;
#[path = "learned_readout_probe/import.rs"]
mod import;

use anyhow::{ensure, Context, Result};
use cache::{Cache, Core, Row, FIT_ROWS};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::{Parser, ValueEnum};
use head::{Head, Kind, Output};
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
use tofy::p2::looped_agent::profile::LoopedCapture;
use tofy::p2::optimizer::clip_gradients_gpu_with_stats;

const SCHEMA: &str = "looped-learned-readout-v1";
const UPDATES: usize = 1000;

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
enum Mode {
    Fit,
    Smoke,
    Evaluate,
    EvaluateImported,
}
#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long, value_enum)]
    kind: Kind,
    #[arg(long, value_enum)]
    core_checkpoint: Core,
    #[arg(long)]
    cache_dir: PathBuf,
    #[arg(long)]
    cache_manifest_sha256: String,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value = "cuda:0")]
    device: String,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    permuted_labels: bool,
    #[arg(long)]
    label_permutation: Option<PathBuf>,
    #[arg(long)]
    label_permutation_sha256: Option<String>,
    #[arg(long)]
    head_dir: Option<PathBuf>,
    #[arg(long)]
    head_manifest_sha256: Option<String>,
    /// C11 canonical fixed parameters and their actual producer recipe.
    #[arg(long)]
    import_dir: Option<PathBuf>,
    #[arg(long)]
    import_manifest_sha256: Option<String>,
    /// Qualification reads only the already accessed C8 fitting cache.
    #[arg(long)]
    import_qualification: bool,
    #[arg(long)]
    confirmation_panel: Option<u8>,
    #[arg(long, default_value_t = 100)]
    max_seconds: u64,
}

impl Args {
    fn evaluates(&self) -> bool {
        matches!(self.mode, Mode::Evaluate | Mode::EvaluateImported)
    }
    fn validate(&self) -> Result<()> {
        ensure!(self.seed == 0, "C9 fixes initialization seed zero");
        ensure!(
            self.max_seconds > 0 && self.max_seconds <= if self.evaluates() { 60 } else { 100 },
            "registered per-invocation time limit exceeded"
        );
        ensure!(
            !self.permuted_labels || self.kind == Kind::Spatial,
            "only the spatial head has a permuted-label arm"
        );
        ensure!(
            self.cache_dir.is_absolute() && self.output_dir.is_absolute(),
            "cache/output paths must be absolute"
        );
        cache::hash_text(&self.cache_manifest_sha256)?;
        if self.mode == Mode::EvaluateImported {
            ensure!(self.import_dir.as_ref().is_some_and(|p| p.is_absolute())
                && self.import_manifest_sha256.is_some()
                && (self.import_qualification != self.confirmation_panel.is_some())
                && self.confirmation_panel.is_none_or(|i| i < 3)
                && self.head_dir.is_none() && self.head_manifest_sha256.is_none()
                && self.label_permutation.is_none() && self.label_permutation_sha256.is_none()
                && !self.permuted_labels, "imported evaluation requires a sealed import and exactly qualification or one closed confirmation panel; fitting/legacy head flags are forbidden");
            cache::hash_text(
                self.import_manifest_sha256
                    .as_deref()
                    .context("import hash")?,
            )?;
            return Ok(());
        }
        ensure!(
            self.import_dir.is_none()
                && self.import_manifest_sha256.is_none()
                && !self.import_qualification
                && self.confirmation_panel.is_none(),
            "import/panel flags require evaluate-imported mode"
        );
        if self.mode == Mode::Evaluate {
            ensure!(
                self.head_dir.is_some()
                    && self.head_manifest_sha256.is_some()
                    && self.label_permutation.is_none()
                    && self.label_permutation_sha256.is_none(),
                "evaluation requires a sealed head and forbids a fitting permutation input"
            );
        } else {
            ensure!(
                self.head_dir.is_none()
                    && self.head_manifest_sha256.is_none()
                    && self.label_permutation.is_some()
                    && self.label_permutation_sha256.as_deref() == Some(cache::PERMUTATION_SHA),
                "fit/smoke require the registered permutation and fresh head initialization"
            );
        }
        Ok(())
    }
    fn updates(&self) -> usize {
        match self.mode {
            Mode::Fit => UPDATES,
            Mode::Smoke => 2,
            Mode::Evaluate | Mode::EvaluateImported => 0,
        }
    }
    fn deadline(&self, started: Instant) -> Result<()> {
        ensure!(
            started.elapsed().as_secs_f64() < self.max_seconds as f64,
            "registered readout runtime limit reached"
        );
        Ok(())
    }
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn git(path: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(path)
        .args(args)
        .output()?;
    ensure!(output.status.success(), "git provenance command failed");
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}
fn provenance(args: &Args) -> Result<Value> {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"));
    let sibling = source.join("../candle_graph");
    for (path, revision, dirty, pushed) in [
        (
            source,
            env!("TOFY_EMBEDDED_SOURCE_REVISION"),
            env!("TOFY_EMBEDDED_SOURCE_DIRTY"),
            env!("TOFY_EMBEDDED_SOURCE_PUSHED"),
        ),
        (
            sibling.as_path(),
            env!("TOFY_EMBEDDED_CANDLE_GRAPH_REVISION"),
            env!("TOFY_EMBEDDED_CANDLE_GRAPH_DIRTY"),
            env!("TOFY_EMBEDDED_CANDLE_GRAPH_PUSHED"),
        ),
    ] {
        ensure!(
            dirty == "false" && pushed == "true",
            "readout build source must be clean and pushed"
        );
        ensure!(
            git(path, &["rev-parse", "HEAD"])? == revision
                && git(path, &["status", "--porcelain", "--untracked-files=all"])?.is_empty(),
            "build/runtime source mismatch"
        );
        git(
            path,
            &["merge-base", "--is-ancestor", "HEAD", "@{upstream}"],
        )?;
    }
    ensure!(
        env!("TOFY_EMBEDDED_CANDLE_GRAPH_REVISION") == "1ea5cc5c5998a1c2bd388eb158653dd0d1b7452a",
        "registered profiler revision mismatch"
    );
    ensure!(
        env!("TOFY_EMBEDDED_BUILD_COMMAND") != "unknown",
        "build command provenance missing"
    );
    if args.device.starts_with("cuda") {
        ensure!(
            env!("TOFY_EMBEDDED_CARGO_FEATURES")
                .split(',')
                .any(|x| x == "cudnn"),
            "CUDA requires cudnn build"
        );
    }
    let gpu = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ])
        .output()
        .ok();
    Ok(
        json!({"source_revision":env!("TOFY_EMBEDDED_SOURCE_REVISION"),"candle_graph_revision":env!("TOFY_EMBEDDED_CANDLE_GRAPH_REVISION"),
        "binary_sha256":cache::file_hash(&std::env::current_exe()?)?,"features":env!("TOFY_EMBEDDED_CARGO_FEATURES"),"build_command":env!("TOFY_EMBEDDED_BUILD_COMMAND"),
        "gpu":gpu.map(|x|String::from_utf8_lossy(&x.stdout).trim().to_owned())}),
    )
}

fn finite(tensor: &Tensor) -> Result<Vec<f32>> {
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "nonfinite readout tensor"
    );
    Ok(values)
}
fn entropy(output: &Output) -> Result<Option<f32>> {
    output
        .attention
        .as_ref()
        .map(|a| -> Result<f32> {
            Ok(a.mul(&a.clamp(1e-30, f64::INFINITY)?.log()?)?
                .sum(candle_core::D::Minus1)?
                .mean_all()?
                .neg()?
                .to_scalar()?)
        })
        .transpose()
}
fn record_output(
    capture: &LoopedCapture,
    phase: &tofy::p2::looped_agent::profile::LoopedRange<'_>,
    head: &Head,
    output: &Output,
    loss: &Tensor,
) -> Result<()> {
    for (name, tensor) in [
        ("readout/pooled", &output.pooled),
        ("readout/logits", &output.logits),
        ("loss/policy_ce", loss),
    ] {
        capture.record_tensor_stats(phase, name, tensor)?;
    }
    if let Some(attention) = &output.attention {
        capture.record_tensor_stats(phase, "readout/attention", attention)?;
    }
    if let Some(value) = entropy(output)? {
        capture.record_scalar(phase, "readout/attention_entropy", value as f64)?;
    }
    if let Some(value) = head.query_separation()? {
        capture.record_scalar(phase, "readout/query_separation", value as f64)?;
    }
    Ok(())
}

fn update(
    head: &Head,
    vars: &VarMap,
    features: &Tensor,
    labels: &Tensor,
    optimizer: &mut AdamW,
    capture: Option<&LoopedCapture>,
) -> Result<Value> {
    let forward = capture.map(|c| {
        c.phase(
            "micro-0/forward",
            Some(candle_graph::ExecutionStep::Forward),
        )
    });
    let output = head.forward(features)?;
    let loss = head::cross_entropy(&output.logits, labels)?;
    let ce = loss.to_scalar::<f32>()?;
    ensure!(ce.is_finite(), "nonfinite policy loss");
    if let (Some(c), Some(phase)) = (capture, forward.as_ref()) {
        record_output(c, phase, head, &output, &loss)?;
    }
    drop(forward);
    let backward = capture.map(|c| {
        c.phase(
            "micro-0/backward",
            Some(candle_graph::ExecutionStep::Backward),
        )
    });
    let mut grads = loss.backward()?;
    ensure!(
        !features.track_op() && grads.get(features).is_none(),
        "frozen cache received a gradient"
    );
    drop(backward);
    let inspect = capture.map(|c| c.phase("gradient-inspection-and-clip", None));
    for (name, var) in head::named(vars) {
        let gradient = grads
            .get(&var)
            .with_context(|| format!("missing head gradient {name}"))?;
        if let (Some(c), Some(phase)) = (capture, inspect.as_ref()) {
            let norm = gradient.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            ensure!(norm.is_finite(), "nonfinite head parameter gradient");
            c.record_scalar(phase, &format!("gradient_norm/{name}"), norm as f64)?;
        }
    }
    if let Some(c) = capture {
        c.record_gradients(
            inspect.as_ref().context("missing inspection phase")?,
            &grads,
        )?;
    }
    let clipped = clip_gradients_gpu_with_stats(&mut grads, vars, 1.0)?;
    if let (Some(c), Some(phase)) = (capture, inspect.as_ref()) {
        c.record_scalar(
            phase,
            "gradient/global_pre_clip_norm",
            clipped.pre_clip_norm,
        )?;
        c.record_scalar(phase, "gradient/clip_scale", clipped.scale)?;
    }
    drop(inspect);
    let phase = capture.map(|c| c.phase("optimizer", Some(candle_graph::ExecutionStep::Optimizer)));
    optimizer.step(&grads)?;
    drop(phase);
    Ok(json!({"pre_update_ce":ce,"pre_clip_norm":clipped.pre_clip_norm,"clip_scale":clipped.scale}))
}

fn final_forward(
    args: &Args,
    head: &Head,
    features: &Tensor,
    labels: &Tensor,
    device: &Device,
) -> Result<Output> {
    if !args.evaluates() {
        return head.forward(features);
    }
    let capture = LoopedCapture::begin_readout(
        &args
            .output_dir
            .with_extension("profiles")
            .join("evaluation-000001"),
        1,
        device,
        None,
        features.dim(0)?,
        args.kind.name(),
    )?;
    device.synchronize()?;
    let measured = capture.measurement();
    let phase = capture.phase("forward", Some(candle_graph::ExecutionStep::Forward));
    let result = (|| -> Result<Output> {
        let out = head.forward(features)?;
        let loss = head::cross_entropy(&out.logits, labels)?;
        record_output(&capture, &phase, head, &out, &loss)?;
        Ok(out)
    })();
    let synchronized = device.synchronize();
    drop(phase);
    drop(measured);
    synchronized?;
    let output = result?;
    capture.finish()?;
    Ok(output)
}

fn predictions(root: &Path, rows: &[Row], fitted_labels: &[u32], output: &Output) -> Result<Value> {
    write_predictions(root, rows, fitted_labels, output, false)
}

fn write_predictions(
    root: &Path,
    rows: &[Row],
    fitted_labels: &[u32],
    output: &Output,
    imported: bool,
) -> Result<Value> {
    let logits = finite(&output.logits)?;
    ensure!(
        logits.len() == rows.len() * 4 && fitted_labels.len() == rows.len(),
        "wrong prediction population"
    );
    finite(&output.pooled)?;
    if let Some(a) = &output.attention {
        finite(a)?;
    }
    let pooled_width = if output.attention.is_some() { 256 } else { 10 };
    if imported {
        ensure!(
            output.pooled.dims() == [rows.len(), pooled_width],
            "wrong imported pooled shape"
        );
        let mut arrays = vec![("pooled.f32", &output.pooled)];
        if let Some(attention) = &output.attention {
            ensure!(
                attention.dims() == [rows.len(), 2, 64],
                "wrong imported attention shape"
            );
            for weights in finite(attention)?.chunks_exact(64) {
                ensure!(
                    weights.iter().all(|&x| (0.0..=1.0).contains(&x))
                        && (weights.iter().map(|&x| x as f64).sum::<f64>() - 1.0).abs() <= 1e-5,
                    "invalid imported attention normalization"
                );
            }
            arrays.push(("attention.f32", attention));
        }
        for (name, tensor) in arrays {
            let mut file = BufWriter::new(File::create_new(root.join(name))?);
            for value in finite(tensor)? {
                file.write_all(&value.to_le_bytes())?;
            }
            file.flush()?;
            ensure!(
                root.join(name).metadata()?.len() == (tensor.elem_count() * 4) as u64,
                "wrong imported raw array byte count"
            );
        }
    }
    let mut raw = BufWriter::new(File::create_new(root.join("logits.f32"))?);
    let mut labels = BufWriter::new(File::create_new(root.join("labels.u32"))?);
    let mut ids = BufWriter::new(File::create_new(root.join("episode-ids.u64"))?);
    let mut records = BufWriter::new(File::create_new(root.join("predictions.jsonl"))?);
    let mut true_correct = 0;
    let mut fitted_correct = 0;
    let mut true_ce = 0.0f64;
    let mut fitted_ce = 0.0f64;
    for (i, (row, values)) in rows.iter().zip(logits.chunks_exact(4)).enumerate() {
        let action = (1..4).fold(0, |best, a| if values[a] > values[best] { a } else { best });
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let logsum = values
            .iter()
            .map(|&x| (x as f64 - max).exp())
            .sum::<f64>()
            .ln()
            + max;
        true_correct += usize::from(action == row.correct_action as usize);
        fitted_correct += usize::from(action == fitted_labels[i] as usize);
        true_ce += logsum - values[row.correct_action as usize] as f64;
        fitted_ce += logsum - values[fitted_labels[i] as usize] as f64;
        for value in values {
            raw.write_all(&value.to_le_bytes())?;
        }
        labels.write_all(&row.correct_action.to_le_bytes())?;
        labels.write_all(&fitted_labels[i].to_le_bytes())?;
        ids.write_all(&row.episode_id.to_le_bytes())?;
        let mut record = json!({"identity":row,"true_action":row.correct_action,"fitted_action":fitted_labels[i],"prediction":action,
            "logits":{"file":"logits.f32","dtype":"F32LE","byte_offset":i*16,"byte_length":16,"shape":[4]},
            "labels":{"file":"labels.u32","dtype":"U32LE","byte_offset":i*8,"byte_length":8,"order":["true","fitted"]},
            "episode_id":{"file":"episode-ids.u64","dtype":"U64LE","byte_offset":i*8,"byte_length":8}});
        if imported {
            record["pooled"] = json!({"file":"pooled.f32","dtype":"F32LE","shape":[pooled_width],"byte_offset":i*pooled_width*4,"byte_length":pooled_width*4});
            record["attention"] = if output.attention.is_some() {
                json!({"file":"attention.f32","dtype":"F32LE","shape":[2,64],"byte_offset":i*512,"byte_length":512})
            } else {
                Value::Null
            };
        }
        serde_json::to_writer(&mut records, &record)?;
        records.write_all(b"\n")?;
    }
    raw.flush()?;
    labels.flush()?;
    ids.flush()?;
    records.flush()?;
    ensure!(
        root.join("logits.f32").metadata()?.len() == (rows.len() * 16) as u64
            && root.join("labels.u32").metadata()?.len() == (rows.len() * 8) as u64
            && root.join("episode-ids.u64").metadata()?.len() == (rows.len() * 8) as u64,
        "readout output byte count mismatch"
    );
    Ok(
        json!({"rows":rows.len(),"true_correct":true_correct,"fitted_correct":fitted_correct,"true_accuracy":true_correct as f64/rows.len() as f64,
        "fitted_accuracy":fitted_correct as f64/rows.len() as f64,"true_ce":true_ce/rows.len() as f64,"fitted_ce":fitted_ce/rows.len() as f64,"attention_entropy":entropy(output)?}),
    )
}

fn annotate_import(args: &Args, source: &Value, document: &mut Value) {
    document["schema"] = json!(import::SCHEMA);
    document["import_source"] = source.clone();
    document["arm"] = source["manifest"]["arm"].clone();
    document["optimizer"] = Value::Null;
    document["optimizer_updates"] = json!(0);
    document["cuda_gemm_reduced_precision_f32"] =
        json!(import::gemm_reduced_precision(&args.device));
    document["nvidia_tf32_override"] = json!(std::env::var("NVIDIA_TF32_OVERRIDE").ok());
    document["import_qualification"] = json!(args.import_qualification);
    document["confirmation_panel"] = json!(args.confirmation_panel);
    document["evidence_class"] = json!(if args.import_qualification {
        "implementation_smoke"
    } else {
        "frozen_imported_readout_confirmation"
    });
    document["partition"] = json!(if args.import_qualification {
        "fit"
    } else {
        "confirmation_eval"
    });
    document["objective"] = json!("frozen policy inference; labels enter scoring only");
    document["claim_boundary"] = json!("fixed synthetic readout confirmation; C10 originally used privileged role fitting and a frozen C8 affine policy; initial-core capability is not C7 learning; no policy-only learnability or architecture/ARC promotion");
    for key in [
        "head_source",
        "permuted_labels",
        "label_permutation",
        "label_permutation_sha256",
    ] {
        document.as_object_mut().expect("report object").remove(key);
    }
}

fn run(args: &Args, started: Instant) -> Result<Value> {
    let provenance = provenance(args)?;
    if args.mode == Mode::EvaluateImported {
        ensure!(
            import::gemm_reduced_precision(&args.device) != Some(true),
            "C11 requires strict F32 GEMM"
        );
    }
    let imported = if args.mode == Mode::EvaluateImported {
        Some(import::Imported::load(
            args.import_dir.as_deref().context("import root")?,
            args.import_manifest_sha256
                .as_deref()
                .context("import seal")?,
            args.core_checkpoint,
            args.kind,
        )?)
    } else {
        None
    };
    let import_source = imported.as_ref().map(|source| json!({"root":args.import_dir,"manifest_sha256":args.import_manifest_sha256,"manifest":source.manifest}));
    let cache = if let Some(panel) = args.confirmation_panel {
        cache::load_confirmation(
            &args.cache_dir,
            &args.cache_manifest_sha256,
            args.core_checkpoint,
            args.kind,
            panel,
        )?
    } else {
        cache::load(
            &args.cache_dir,
            &args.cache_manifest_sha256,
            args.core_checkpoint,
            args.kind,
            args.mode == Mode::Evaluate,
        )?
    };
    let Cache {
        manifest: cache_manifest,
        rows,
        values,
    } = cache;
    let true_labels = rows
        .iter()
        .map(|row| row.correct_action)
        .collect::<Vec<_>>();
    let permutation = if args.evaluates() {
        None
    } else {
        Some(cache::permutation(
            args.label_permutation
                .as_ref()
                .context("permutation path")?,
            args.label_permutation_sha256
                .as_deref()
                .context("permutation hash")?,
        )?)
    };
    let fitted_labels = if args.permuted_labels && !args.evaluates() {
        permutation
            .as_ref()
            .context("permutation")?
            .iter()
            .map(|&i| true_labels[i])
            .collect::<Vec<_>>()
    } else {
        true_labels.clone()
    };
    let mut head_source = None;
    if args.mode == Mode::Evaluate {
        let root = args.head_dir.as_ref().context("head root")?;
        let seal = cache::verified_manifest(
            root,
            args.head_manifest_sha256.as_deref().context("head seal")?,
            false,
        )?;
        let report: Value = serde_json::from_slice(&fs::read(root.join("report.json"))?)?;
        let metadata: Value = serde_json::from_slice(&fs::read(root.join("metadata.json"))?)?;
        ensure!(
            report["schema"] == SCHEMA
                && report["status"] == "complete_pending_analysis"
                && report["optimizer_updates"] == UPDATES
                && report["evidence_class"] == "learned_readout_screen"
                && report["head_kind"] == args.kind.name()
                && report["core_checkpoint_sha256"] == args.core_checkpoint.checkpoint()
                && report["permuted_labels"] == args.permuted_labels,
            "sealed head kind/checkpoint/arm mismatch or incomplete fitting"
        );
        for key in ["source_revision", "candle_graph_revision", "binary_sha256"] {
            ensure!(
                metadata["provenance"][key] == provenance[key],
                "head fit/evaluation executable provenance mismatch: {key}"
            );
        }
        head_source = Some(
            json!({"root":root,"manifest_sha256":args.head_manifest_sha256,"checkpoint_sha256":seal["files"]["final.safetensors"]}),
        );
    }
    let mut metadata = json!({"schema":SCHEMA,"status":"running","exact_args":std::env::args().collect::<Vec<_>>(),"provenance":provenance,
        "model":{"type":"learned_readout","head_kind":args.kind,"parameters":args.kind.parameters(),"cached_width":128,"head_recurrence":false},
        "head_kind":args.kind,"core_checkpoint":args.core_checkpoint,"core_checkpoint_sha256":args.core_checkpoint.checkpoint(),
        "cache":{"root":args.cache_dir,"manifest_sha256":args.cache_manifest_sha256,"manifest":cache_manifest},"head_source":head_source,
        "seed":args.seed,"permuted_labels":args.permuted_labels,"label_permutation":args.label_permutation,"label_permutation_sha256":args.label_permutation_sha256,
        "physical_batch":rows.len(),"effective_batch":rows.len(),"accumulation":1,"executed_core_forwards":0,"core_optimizer_updates":0,"cached_extraction_loops":4,
        "optimizer":{"kind":"AdamW","updates":args.updates(),"learning_rate":0.003,"weight_decay":0.0,"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"clip_norm":1.0,"sorted_parameters":true},
        "objective":"mean policy cross-entropy only","feature_preprocessing":"none; raw cached F32","profile_updates":if args.evaluates() {vec![]}else{vec![2]},
        "claim_boundary":"learned cached-feature readout screen; no true-role routing, core training, architecture promotion or ARC claim"});
    if let Some(source) = &import_source {
        annotate_import(args, source, &mut metadata);
    }
    write_json(&args.output_dir.join("metadata.json"), &metadata)?;
    let device = tofy::p2::train::resolve_device(&args.device)?;
    let features = cache::frozen_tensor(values, args.kind, rows.len(), &device)?;
    let original_features = cache::digest(
        &finite(&features)?
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let label_tensor = Tensor::new(fitted_labels.as_slice(), &device)?;
    let mut vars = VarMap::new();
    let head = Head::new(
        args.kind,
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )?;
    head::initialize(&vars, args.seed)?;
    if let Some(imported) = &imported {
        imported.apply(&vars, &device)?;
    }
    if let Some(source) = &head_source {
        vars.load(
            args.head_dir
                .as_ref()
                .context("head root")?
                .join("final.safetensors"),
        )?;
        ensure!(
            source["checkpoint_sha256"].as_str().is_some(),
            "missing sealed head checkpoint digest"
        );
    }
    let names = head::named(&vars);
    ensure!(
        names.iter().map(|(_, v)| v.elem_count()).sum::<usize>() == args.kind.parameters(),
        "head parameter count mismatch"
    );
    vars.save(args.output_dir.join("initial.safetensors"))?;
    let initial_hash = cache::file_hash(&args.output_dir.join("initial.safetensors"))?;
    if let Some(imported) = &imported {
        imported.verify_saved(&args.output_dir.join("initial.safetensors"))?;
    }
    if let Some(source) = &head_source {
        ensure!(
            source["checkpoint_sha256"] == initial_hash,
            "loaded head checkpoint bytes differ"
        );
    }
    let mut updates = BufWriter::new(File::create_new(args.output_dir.join("updates.jsonl"))?);
    let mut losses = BufWriter::new(File::create_new(args.output_dir.join("losses.jsonl"))?);
    if !args.evaluates() {
        ensure!(
            rows.len() == FIT_ROWS,
            "fit batch must contain all 512 cached examples"
        );
        let mut optimizer = AdamW::new(
            names.iter().map(|(_, var)| var.clone()).collect(),
            ParamsAdamW {
                lr: 0.003,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
        )?;
        for step in 1..=args.updates() {
            args.deadline(started)?;
            let capture = if step == 2 {
                Some(LoopedCapture::begin_readout(
                    &args
                        .output_dir
                        .with_extension("profiles")
                        .join("update-000000000002"),
                    2,
                    &device,
                    Some(&vars),
                    rows.len(),
                    args.kind.name(),
                )?)
            } else {
                None
            };
            // Operational update timing includes selected instrumentation, but
            // excludes capture publication. It is not production timing evidence.
            device.synchronize()?;
            let update_started = Instant::now();
            let measurement = capture.as_ref().map(LoopedCapture::measurement);
            let result = update(
                &head,
                &vars,
                &features,
                &label_tensor,
                &mut optimizer,
                capture.as_ref(),
            );
            let synchronized = device.synchronize();
            drop(measurement);
            let update_elapsed = update_started.elapsed().as_secs_f64();
            synchronized?;
            let mut record = result?;
            if let Some(capture) = capture {
                capture.finish()?;
            }
            record["update"] = json!(step);
            record["rows"] = json!(rows.len());
            record["elapsed_seconds"] = json!(update_elapsed);
            record["profiled"] = json!(step == 2);
            record["frozen_features_gradient_absent"] = json!(true);
            serde_json::to_writer(&mut updates, &record)?;
            updates.write_all(b"\n")?;
            if step % 100 == 0 || args.mode == Mode::Smoke {
                serde_json::to_writer(&mut losses, &record)?;
                losses.write_all(b"\n")?;
            }
        }
    }
    updates.flush()?;
    losses.flush()?;
    drop(updates);
    drop(losses);
    for (_, var) in &names {
        finite(var)?;
    }
    vars.save(args.output_dir.join("final.safetensors"))?;
    let final_hash = cache::file_hash(&args.output_dir.join("final.safetensors"))?;
    if args.evaluates() {
        ensure!(initial_hash == final_hash, "frozen head parameters changed");
    }
    args.deadline(started)?;
    let output = final_forward(args, &head, &features, &label_tensor, &device)?;
    let predictions = if imported.is_some() {
        write_predictions(&args.output_dir, &rows, &fitted_labels, &output, true)?
    } else {
        predictions(&args.output_dir, &rows, &fitted_labels, &output)?
    };
    if let Some(imported) = &imported {
        imported.verify(&vars)?;
        imported.verify_saved(&args.output_dir.join("final.safetensors"))?;
        let reloaded = import::Imported::load(
            args.import_dir.as_deref().context("import root")?,
            args.import_manifest_sha256
                .as_deref()
                .context("import seal")?,
            args.core_checkpoint,
            args.kind,
        )?;
        reloaded.verify(&vars)?;
        ensure!(
            cache::file_hash(&args.output_dir.join("initial.safetensors"))? == final_hash,
            "imported saved head changed during inference"
        );
    }
    ensure!(
        original_features
            == cache::digest(
                &finite(&features)?
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>()
            ),
        "frozen feature bytes changed"
    );
    ensure!(
        cache::verified_manifest(&args.cache_dir, &args.cache_manifest_sha256, true)?
            == cache_manifest,
        "cache changed during head invocation"
    );
    if let Some(source) = &head_source {
        let path = args.head_dir.as_ref().context("head root")?;
        ensure!(
            cache::verified_manifest(
                path,
                args.head_manifest_sha256.as_deref().context("head seal")?,
                false
            )?["files"]["final.safetensors"]
                == source["checkpoint_sha256"],
            "source head changed"
        );
    }
    args.deadline(started)?;
    let mut report = json!({"schema":SCHEMA,"status":"complete_pending_analysis","evidence_class":match args.mode {Mode::Fit=>"learned_readout_screen",Mode::Smoke=>"implementation_smoke",Mode::Evaluate=>"frozen_learned_readout_evaluation",Mode::EvaluateImported=>"frozen_imported_readout_confirmation"},
        "head_kind":args.kind,"model_type":"learned_readout","parameters":args.kind.parameters(),"parameter_names":names.iter().map(|(name,_)|name).collect::<Vec<_>>(),
        "core_checkpoint_sha256":args.core_checkpoint.checkpoint(),"permuted_labels":args.permuted_labels,"seed":0,
        "optimizer_updates":args.updates(),"executed_core_forwards":0,"core_optimizer_updates":0,"head_forwards":args.updates()+1,
        "frozen_feature_gradient_checks":args.updates(),"frozen_feature_bytes_unchanged":true,"cached_extraction_loops":4,
        "physical_batch":rows.len(),"effective_batch":rows.len(),"accumulation":1,"input_rows":rows.len(),"partition":if args.mode==Mode::Evaluate {"fresh_eval"}else{"fit"},
        "cache_manifest_sha256":args.cache_manifest_sha256,"core_source":cache_manifest["source"],"head_source":head_source,
        "initial_checkpoint_sha256":initial_hash,"final_checkpoint_sha256":final_hash,"query_separation":head.query_separation()?,
        "predictions":predictions,"elapsed_seconds":started.elapsed().as_secs_f64(),"method_promotion":false});
    if let Some(source) = &import_source {
        annotate_import(args, source, &mut report);
        report["canonical_parameter_bytes_unchanged"] = json!(true);
        report["saved_canonical_parameter_roundtrip"] = json!(true);
    }
    Ok(report)
}

fn bind_profiles(root: &Path) -> Result<()> {
    fn files(root: &Path, path: &Path, out: &mut serde_json::Map<String, Value>) -> Result<()> {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() {
                files(root, &path, out)?;
            } else {
                ensure!(kind.is_file(), "nonregular profile artifact");
                out.insert(
                    path.strip_prefix(root)?.to_string_lossy().into_owned(),
                    json!(cache::file_hash(&path)?),
                );
            }
        }
        Ok(())
    }
    let profile_root = root.with_extension("profiles");
    let mut hashes = serde_json::Map::new();
    if profile_root.exists() {
        files(&profile_root, &profile_root, &mut hashes)?;
    }
    let host = std::env::var_os("TOFY_PERF_TRACE")
        .map(PathBuf::from)
        .map(|path| -> Result<Value> { Ok(json!({"sha256":cache::file_hash(&path)?,"path":path})) })
        .transpose()?;
    write_json(
        &root.join("profiles.json"),
        &json!({"root":profile_root,"files":hashes,"host_trace":host,"nsight":"external capture and export; bind in a separate bundle after profiler exit"}),
    )
}
fn seal(root: &Path) -> Result<()> {
    let mut files = serde_json::Map::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        ensure!(entry.file_type()?.is_file(), "nonregular run artifact");
        files.insert(
            entry
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("non-UTF8 artifact"))?,
            json!(cache::file_hash(&entry.path())?),
        );
    }
    write_json(
        &root.join("manifest.json"),
        &json!({"schema":"looped-agent-artifacts-v1","files":files}),
    )?;
    let digest = cache::file_hash(&root.join("manifest.json"))?;
    cache::verified_manifest(root, &digest, false)?;
    fs::write(
        root.with_extension("manifest.sha256"),
        format!("{digest}\n"),
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    args.validate()?;
    let host = std::env::var_os("TOFY_PERF_TRACE").map(PathBuf::from);
    ensure!(
        !args.output_dir.exists()
            && !args.output_dir.with_extension("profiles").exists()
            && !args.output_dir.with_extension("manifest.sha256").exists()
            && host
                .as_ref()
                .is_some_and(|p| p == &args.output_dir.with_extension("host.json") && !p.exists()),
        "readout output/evidence root must be new"
    );
    let perf = tofy::perf::install()?;
    ensure!(
        cfg!(feature = "profiling") && perf.is_some(),
        "readout invocation requires profiling and TOFY_PERF_TRACE at OUTPUT.host.json"
    );
    fs::create_dir(&args.output_dir)?;
    write_json(
        &args.output_dir.join("launch.json"),
        &json!({"status":"running","pid":std::process::id(),"exact_args":std::env::args().collect::<Vec<_>>(),
        "source_revision":env!("TOFY_EMBEDDED_SOURCE_REVISION"),"binary_sha256":cache::file_hash(&std::env::current_exe()?)?}),
    )?;
    let started = Instant::now();
    let result = run(&args, started);
    drop(perf);
    let report = match &result {
        Ok(report) => report.clone(),
        Err(error) => {
            json!({"status":"failed_integrity_or_evaluation","error":format!("{error:#}"),"elapsed_seconds":started.elapsed().as_secs_f64()})
        }
    };
    write_json(&args.output_dir.join("report.json"), &report)?;
    bind_profiles(&args.output_dir)?;
    seal(&args.output_dir)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    result.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_graph::trace::{analyze_health, GradientState};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub(super) struct TestDir(pub PathBuf);
    impl TestDir {
        pub fn new(name: &str) -> Result<Self> {
            let path = std::env::temp_dir().join(format!(
                "tofy-readout-{name}-{}-{}",
                std::process::id(),
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
            ));
            fs::create_dir(&path)?;
            Ok(Self(path))
        }
    }
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn args() -> Args {
        Args::try_parse_from([
            "readout",
            "--mode",
            "fit",
            "--kind",
            "spatial",
            "--core-checkpoint",
            "initial",
            "--cache-dir",
            "/cache",
            "--cache-manifest-sha256",
            &"1".repeat(64),
            "--output-dir",
            "/output",
            "--label-permutation",
            "/permutation.json",
            "--label-permutation-sha256",
            cache::PERMUTATION_SHA,
        ])
        .unwrap()
    }

    #[test]
    fn cli_fixes_updates_and_separates_fit_smoke_and_frozen_evaluation() -> Result<()> {
        let mut a = args();
        a.validate()?;
        assert_eq!(a.updates(), 1000);
        a.mode = Mode::Smoke;
        a.validate()?;
        assert_eq!(a.updates(), 2);
        a.seed = 1;
        assert!(a.validate().is_err());
        a.seed = 0;
        a.kind = Kind::Cls;
        a.permuted_labels = true;
        assert!(a.validate().is_err());
        a.permuted_labels = false;
        a.mode = Mode::Evaluate;
        a.max_seconds = 60;
        a.head_dir = Some("/head".into());
        a.head_manifest_sha256 = Some("2".repeat(64));
        assert!(a.validate().is_err()); // fitting inputs cannot enter evaluation
        a.label_permutation = None;
        a.label_permutation_sha256 = None;
        a.validate()?;
        assert_eq!(a.updates(), 0);
        a.max_seconds = 61;
        assert!(a.validate().is_err());
        a.mode = Mode::Fit;
        assert!(a.validate().is_err()); // no fitting from a supplied head
        Ok(())
    }

    #[test]
    fn imported_cli_is_zero_update_and_separates_qualification_confirmation_and_legacy(
    ) -> Result<()> {
        let mut a = args();
        a.import_dir = Some("/import".into());
        assert!(a.validate().is_err());
        a.mode = Mode::EvaluateImported;
        a.max_seconds = 60;
        a.import_manifest_sha256 = Some("a".repeat(64));
        a.import_qualification = true;
        assert!(a.validate().is_err());
        a.label_permutation = None;
        a.label_permutation_sha256 = None;
        a.validate()?;
        assert_eq!(a.updates(), 0);
        assert!(a.evaluates());
        a.confirmation_panel = Some(0);
        assert!(a.validate().is_err());
        a.import_qualification = false;
        for panel in 0..3 {
            a.confirmation_panel = Some(panel);
            a.validate()?;
        }
        a.confirmation_panel = Some(3);
        assert!(a.validate().is_err());
        a.confirmation_panel = None;
        assert!(a.validate().is_err());
        a.import_qualification = true;
        a.permuted_labels = true;
        assert!(a.validate().is_err());
        a.permuted_labels = false;
        a.head_dir = Some("/legacy".into());
        assert!(a.validate().is_err());
        a.head_dir = None;
        a.max_seconds = 61;
        assert!(a.validate().is_err());
        a.max_seconds = 60;
        let mut metadata = json!({"optimizer":{"kind":"AdamW"},"permuted_labels":false});
        annotate_import(
            &a,
            &json!({"manifest":{"arm":"c10_true","source_kind":"c10_role_ridge"}}),
            &mut metadata,
        );
        assert!(metadata["optimizer"].is_null());
        assert_eq!(metadata["optimizer_updates"], 0);
        assert_eq!(metadata["evidence_class"], "implementation_smoke");
        assert!(metadata.get("permuted_labels").is_none());
        for mode in [Mode::Fit, Mode::Smoke, Mode::Evaluate] {
            a.mode = mode;
            assert!(a.validate().is_err());
        }
        Ok(())
    }

    #[test]
    fn two_artificial_updates_capture_only_head_gradients_and_freeze_inputs() -> Result<()> {
        for kind in [Kind::Spatial, Kind::Cls] {
            let dir = TestDir::new(kind.name())?;
            let vars = VarMap::new();
            let model = Head::new(
                kind,
                VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
            )?;
            head::initialize(&vars, 0)?;
            let initial = dir.0.join("initial.safetensors");
            vars.save(&initial)?;
            let repeated = VarMap::new();
            let _ = Head::new(
                kind,
                VarBuilder::from_varmap(&repeated, DType::F32, &Device::Cpu),
            )?;
            head::initialize(&repeated, 0)?;
            repeated.save(dir.0.join("repeat.safetensors"))?;
            assert_eq!(
                cache::file_hash(&initial)?,
                cache::file_hash(&dir.0.join("repeat.safetensors"))?
            );
            let values = (0..kind.shape(8).iter().product())
                .map(|i| ((i * 17 % 139) as f32 - 69.0) / 31.0)
                .collect::<Vec<_>>();
            let features = cache::frozen_tensor(values.clone(), kind, 8, &Device::Cpu)?;
            let labels = Tensor::new(&[0u32, 1, 2, 3, 3, 2, 1, 0], &Device::Cpu)?;
            let mut optimizer = AdamW::new(
                head::named(&vars).into_iter().map(|(_, v)| v).collect(),
                ParamsAdamW {
                    lr: 0.003,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.0,
                },
            )?;
            update(&model, &vars, &features, &labels, &mut optimizer, None)?;
            let capture_path = dir.0.join("update-2");
            let capture = LoopedCapture::begin_readout(
                &capture_path,
                2,
                &Device::Cpu,
                Some(&vars),
                8,
                kind.name(),
            )?;
            {
                let _measurement = capture.measurement();
                let record = update(
                    &model,
                    &vars,
                    &features,
                    &labels,
                    &mut optimizer,
                    Some(&capture),
                )?;
                assert!(record["pre_clip_norm"].as_f64().unwrap() > 0.0);
            }
            capture.finish()?;
            let trace = candle_graph::parse_trace(capture_path.join("trace.jsonl"))?;
            let health = analyze_health(&trace);
            assert!(
                health.structurally_valid && health.capture_complete,
                "{health:?}"
            );
            let contract = trace
                .run
                .capture_contract
                .gradient_contract
                .as_ref()
                .unwrap();
            assert_eq!(contract.families.len(), 2);
            assert_eq!(
                contract
                    .expected
                    .iter()
                    .map(|p| p.key.as_str())
                    .collect::<Vec<_>>(),
                head::named(&vars)
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>()
            );
            assert!(trace
                .gradients
                .iter()
                .all(|g| g.state == GradientState::Present));
            assert_eq!(trace.gradients.len(), head::named(&vars).len());
            assert_eq!(trace.run.tags["executed_core_forwards"], "0");
            assert_eq!(trace.run.tags["cached_extraction_loops"], "4");
            assert_eq!(trace.run.tags["loops"], "not_applicable");
            assert_eq!(finite(&features)?, values);
            vars.save(dir.0.join("final.safetensors"))?;
            assert_ne!(
                cache::file_hash(&initial)?,
                cache::file_hash(&dir.0.join("final.safetensors"))?
            );
            let mut restored = VarMap::new();
            let restored_head = Head::new(
                kind,
                VarBuilder::from_varmap(&restored, DType::F32, &Device::Cpu),
            )?;
            restored.load(dir.0.join("final.safetensors"))?;
            assert_eq!(
                finite(&model.forward(&features)?.logits)?,
                finite(&restored_head.forward(&features)?.logits)?
            );
            // Frozen evaluation publishes no active gradient families.
            for mode in [Mode::Evaluate, Mode::EvaluateImported] {
                let mut a = args();
                a.mode = mode;
                a.kind = kind;
                a.output_dir = dir.0.join(format!("eval-{mode:?}"));
                let before = finite(&restored_head.forward(&features)?.logits)?;
                let measured = final_forward(&a, &restored_head, &features, &labels, &Device::Cpu)?;
                assert_eq!(finite(&measured.logits)?, before);
                let eval = candle_graph::parse_trace(
                    a.output_dir
                        .with_extension("profiles")
                        .join("evaluation-000001/trace.jsonl"),
                )?;
                assert!(eval.run.capture_contract.gradient_contract.is_none());
                assert!(eval.gradients.is_empty());
                assert!(analyze_health(&eval).capture_complete);
            }
        }
        Ok(())
    }

    #[test]
    fn prediction_offsets_roundtrip_and_null_labels_are_scored_separately() -> Result<()> {
        let dir = TestDir::new("binary")?;
        let rows = (0..2)
            .map(|i| Row {
                input_index: i,
                episode_id: 91 + i as u64,
                partition: "fit".into(),
                input_sha256: cache::digest(&[i as u8]),
                query_sha256: cache::digest(&[i as u8, 9]),
                label_sha256: cache::digest(&(i as u32).to_le_bytes()),
                correct_action: i as u32,
            })
            .collect::<Vec<_>>();
        let raw = vec![-0f32, 0., 0., 0., -2., 1., 3., 0.];
        let output = Output {
            logits: Tensor::from_vec(raw.clone(), (2, 4), &Device::Cpu)?,
            pooled: Tensor::zeros((2, 10), DType::F32, &Device::Cpu)?,
            attention: None,
        };
        let metrics = predictions(&dir.0, &rows, &[2, 2], &output)?;
        assert_eq!(metrics["true_correct"], 1); // first-argmax tie is action zero
        assert_eq!(metrics["fitted_correct"], 1);
        assert!(metrics["true_ce"].as_f64().unwrap() > metrics["fitted_ce"].as_f64().unwrap());
        let bytes = fs::read(dir.0.join("logits.f32"))?;
        assert_eq!(cache::decode_f32(&bytes)?, raw);
        let records = fs::read_to_string(dir.0.join("predictions.jsonl"))?
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<serde_json::Result<Vec<_>>>()?;
        let labels = fs::read(dir.0.join("labels.u32"))?;
        let ids = fs::read(dir.0.join("episode-ids.u64"))?;
        for (i, row) in records.iter().enumerate() {
            let offset = row["logits"]["byte_offset"].as_u64().unwrap() as usize;
            assert_eq!(
                cache::decode_f32(&bytes[offset..offset + 16])?,
                raw[i * 4..i * 4 + 4]
            );
            let offset = row["labels"]["byte_offset"].as_u64().unwrap() as usize;
            assert_eq!(
                u32::from_le_bytes(labels[offset..offset + 4].try_into()?),
                i as u32
            );
            assert_eq!(
                u32::from_le_bytes(labels[offset + 4..offset + 8].try_into()?),
                2
            );
            let offset = row["episode_id"]["byte_offset"].as_u64().unwrap() as usize;
            assert_eq!(
                u64::from_le_bytes(ids[offset..offset + 8].try_into()?),
                91 + i as u64
            );
        }
        seal(&dir.0)?;
        let hash = cache::file_hash(&dir.0.join("manifest.json"))?;
        cache::verified_manifest(&dir.0, &hash, false)?;
        fs::write(dir.0.join("logits.f32"), b"corrupt")?;
        assert!(cache::verified_manifest(&dir.0, &hash, false).is_err());
        fs::remove_file(dir.0.with_extension("manifest.sha256"))?;
        Ok(())
    }
}
