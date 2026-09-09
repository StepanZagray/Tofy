//! Synthetic prerequisite for the from-scratch looped agent. Never loads ARC games.
#[path = "looped_agent_probe/counterfactual.rs"]
mod counterfactual;
#[path = "looped_agent_probe/known_mapping.rs"]
mod known_mapping;
#[path = "looped_agent_probe/known_replay.rs"]
mod known_replay;

use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::{Parser, ValueEnum};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tofy::p2::looped_agent::model::{LoopedAgent, LoopedConfig, LoopedOutput};
use tofy::p2::looped_agent::profile::LoopedCapture;
use tofy::p2::looped_agent::task::{self, Frame, Inputs, Sample, Split, Transition};
use tofy::p2::looped_agent::{ACTIONS, META_DIM, PALETTE, PATCH_COUNT, PATCH_PIXELS, TOKENS};
use tofy::p2::optimizer::{accumulate_parameter_gradients, clip_gradients_gpu_with_stats};
use tofy::p2::train::resolve_device;

const PIXELS: usize = PATCH_COUNT * PATCH_PIXELS;
const TRAIN_TAG: u64 = 0x545241494e;
const EVAL_TAG: u64 = 0x4556414c;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum Mode {
    Smoke,
    Train,
    Evaluate,
    Inspect,
    Counterfactual,
    CounterfactualAudit,
    KnownMapping,
    KnownMappingAudit,
    Fit,
    FitSmoke,
    Coverage,
    CoverageAudit,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
enum Coverage {
    Fixed,
    Fresh,
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
    /// Balanced one-step episode stream for the matched coverage comparison.
    #[arg(long, value_enum, default_value = "fixed")]
    coverage: Coverage,
    /// Spatial prerequisite: train and evaluate only the known permutation zero.
    #[arg(long)]
    known_mapping: bool,
    /// Frozen known-mapping diagnosis on a fixed subset of training queries.
    #[arg(long)]
    known_seen: bool,
    /// Reorder known fresh queries within their original training-depth cohorts.
    #[arg(long)]
    known_replay: bool,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value = "cuda:0")]
    device: String,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 9173)]
    data_seed: u64,
    #[arg(long, default_value_t = 128)]
    hidden: usize,
    #[arg(long, default_value_t = 4)]
    heads: usize,
    #[arg(long, default_value_t = 2)]
    layers: usize,
    #[arg(long, default_value_t = 4)]
    loops: usize,
    /// Maximum inference depth, including untrained depth extrapolation.
    #[arg(long, default_value_t = 8)]
    max_loops: usize,
    #[arg(long, default_value_t = 8)]
    batch: usize,
    #[arg(long, default_value_t = 64)]
    effective_batch: usize,
    #[arg(long, default_value_t = 256)]
    updates: usize,
    #[arg(long, default_value_t = 0.0003)]
    learning_rate: f64,
    #[arg(long, default_value_t = 32)]
    eval_episodes: usize,
    #[arg(long, default_value_t = 32)]
    step_cap: usize,
    #[arg(long, default_value_t = 1800)]
    max_seconds: u64,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Include two-step search through categorical learned successor frames.
    #[arg(long)]
    search: bool,
    /// One-based optimizer updates with full supported runtime evidence.
    #[arg(long, value_delimiter = ',', default_value = "2")]
    profile_updates: Vec<usize>,
    /// Capture the first representative evaluation forward.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    profile_eval: bool,
}

impl Args {
    fn accumulation(&self) -> usize {
        self.effective_batch.div_ceil(self.batch)
    }
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn file_hash(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn git(path: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(path)
        .args(args)
        .output()?;
    ensure!(
        output.status.success(),
        "git provenance command failed: {args:?}"
    );
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
            "build source must be clean and pushed: {}",
            path.display()
        );
        ensure!(
            git(path, &["rev-parse", "HEAD"])? == revision,
            "runtime/build revision mismatch"
        );
        ensure!(
            git(path, &["status", "--porcelain", "--untracked-files=all"])?.is_empty(),
            "runtime tree is dirty"
        );
        git(
            path,
            &["merge-base", "--is-ancestor", "HEAD", "@{upstream}"],
        )?;
    }
    if args.device.starts_with("cuda") {
        ensure!(
            env!("TOFY_EMBEDDED_CARGO_FEATURES")
                .split(',')
                .any(|x| x == "cudnn"),
            "CUDA launch requires --features cudnn"
        );
    }
    ensure!(
        env!("TOFY_EMBEDDED_BUILD_COMMAND") != "unknown",
        "record TOFY_BUILD_COMMAND at compilation"
    );
    let gpu = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ])
        .output()
        .ok();
    Ok(json!({
        "source_revision": env!("TOFY_EMBEDDED_SOURCE_REVISION"),
        "candle_graph_revision": env!("TOFY_EMBEDDED_CANDLE_GRAPH_REVISION"),
        "features": env!("TOFY_EMBEDDED_CARGO_FEATURES"),
        "build_command": env!("TOFY_EMBEDDED_BUILD_COMMAND"),
        "binary_sha256": file_hash(&std::env::current_exe()?)?,
        "gpu": gpu.map(|x| String::from_utf8_lossy(&x.stdout).trim().to_owned()),
        "checkpoint": args.checkpoint.as_ref().map(|p| -> Result<Value> { Ok(json!({"path": p, "sha256": file_hash(p)?})) }).transpose()?,
    }))
}

fn deadline(args: &Args, started: Instant) -> Result<()> {
    ensure!(
        started.elapsed().as_secs() < args.max_seconds,
        "registered wall-clock limit reached"
    );
    Ok(())
}

fn tensors(rows: &[Inputs], device: &Device) -> Result<(Tensor, Tensor)> {
    let patches = rows
        .iter()
        .flat_map(|r| r.patches.iter().copied())
        .collect::<Vec<_>>();
    let metadata = rows
        .iter()
        .flat_map(|r| r.metadata.iter().copied())
        .collect::<Vec<_>>();
    Ok((
        Tensor::from_vec(patches, (rows.len(), TOKENS, PATCH_PIXELS), device)?,
        Tensor::from_vec(metadata, (rows.len(), TOKENS, META_DIM), device)?,
    ))
}

fn predict(
    model: &LoopedAgent,
    rows: &[Inputs],
    loops: usize,
    device: &Device,
) -> Result<LoopedOutput> {
    let (pixels, metadata) = tensors(rows, device)?;
    model.forward(&pixels, &metadata, loops)
}

fn inspected_predict(
    args: &Args,
    model: &LoopedAgent,
    input: Inputs,
    device: &Device,
) -> Result<LoopedOutput> {
    let capture = LoopedCapture::begin_eval(
        &args
            .output_dir
            .with_extension("profiles")
            .join("evaluation-000001"),
        1,
        device,
        1,
        1,
        args.loops,
    )?;
    device.synchronize()?;
    let measured = capture.measurement();
    let phase = capture.phase("forward", Some(candle_graph::ExecutionStep::Forward));
    let out = predict(model, &[input], args.loops, device)?;
    capture.record_tensor_stats(&phase, "readout/policy_logits", &out.policy_logits)?;
    capture.record_tensor_stats(&phase, "readout/value", &out.value)?;
    capture.record_tensor_stats(&phase, "readout/reward_logits", &out.reward_logits)?;
    capture.record_tensor_stats(&phase, "readout/next_logits", &out.next_logits)?;
    device.synchronize()?;
    drop(phase);
    drop(measured);
    capture.finish()?;
    Ok(out)
}

fn bce(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // Stable BCE: max(x,0) - x*y + log(1+exp(-abs(x))).
    Ok((logits.relu()? - logits.mul(targets)?)?
        .add(&logits.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?
        .mean_all()?)
}

fn losses(out: &LoopedOutput, samples: &[Sample], device: &Device) -> Result<(Tensor, [f32; 4])> {
    let n = samples.len();
    let policy = Tensor::from_vec(
        samples.iter().flat_map(|s| s.policy).collect::<Vec<_>>(),
        (n, ACTIONS),
        device,
    )?;
    let policy_loss = candle_nn::ops::log_softmax(&out.policy_logits, D::Minus1)?
        .mul(&policy)?
        .sum(D::Minus1)?
        .mean_all()?
        .neg()?;
    let value_targets = Tensor::from_vec(
        samples.iter().map(|s| s.value).collect::<Vec<_>>(),
        (n, 1),
        device,
    )?;
    let value_loss = (candle_nn::ops::sigmoid(&out.value)? - value_targets)?
        .sqr()?
        .mean_all()?;
    let rewards = Tensor::from_vec(
        samples.iter().flat_map(|s| s.rewards).collect::<Vec<_>>(),
        (n, ACTIONS),
        device,
    )?;
    let reward_loss = bce(&out.reward_logits, &rewards)?;
    let mut targets = Vec::with_capacity(n * ACTIONS * PIXELS);
    let mut weights = Vec::with_capacity(targets.capacity());
    for s in samples {
        let changed = s
            .next
            .iter()
            .enumerate()
            .filter(|(i, p)| **p != s.current[*i % PIXELS])
            .count();
        let unchanged = ACTIONS * PIXELS - changed;
        for (i, &p) in s.next.iter().enumerate() {
            targets.push(p);
            weights.push(if p != s.current[i % PIXELS] {
                0.5 / changed.max(1) as f32
            } else {
                0.5 / unchanged.max(1) as f32
            });
        }
    }
    let rows = n * ACTIONS * PIXELS;
    let targets = Tensor::from_vec(targets, (rows, 1), device)?;
    let weights = Tensor::from_vec(weights, (rows, 1), device)?;
    let state_loss =
        candle_nn::ops::log_softmax(&out.next_logits.reshape((rows, PALETTE))?, D::Minus1)?
            .gather(&targets, 1)?
            .mul(&weights)?
            .sum_all()?
            .affine(-1.0 / n as f64, 0.0)?;
    let metrics = [
        policy_loss.to_scalar::<f32>()?,
        value_loss.to_scalar::<f32>()?,
        reward_loss.to_scalar::<f32>()?,
        state_loss.to_scalar::<f32>()?,
    ];
    ensure!(
        metrics.iter().all(|x| x.is_finite()),
        "non-finite objective"
    );
    let total = (&policy_loss + (value_loss * 0.1)?)?
        .add(&(reward_loss * 0.1)?)?
        .add(&(state_loss * 0.5)?)?;
    Ok((total, metrics))
}

fn initialize(varmap: &VarMap, seed: u64) -> Result<()> {
    let mut vars = varmap
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(n, v)| (n.clone(), v.clone()))
        .collect::<Vec<_>>();
    vars.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, var) in vars {
        let mut hash = Sha256::new();
        hash.update(seed.to_le_bytes());
        hash.update(name.as_bytes());
        let digest = hash.finalize();
        let mut rng = ChaCha8Rng::seed_from_u64(u64::from_le_bytes(digest[..8].try_into()?));
        let shape = var.dims();
        let values = if name.ends_with("bias") {
            vec![0.0; var.elem_count()]
        } else if shape.len() == 1 && name.ends_with("weight") {
            vec![1.0; var.elem_count()]
        } else {
            let bound = (6.0
                / (shape.last().copied().unwrap_or(1) + shape.first().copied().unwrap_or(1))
                    as f32)
                .sqrt();
            (0..var.elem_count())
                .map(|_| rng.random_range(-bound..bound))
                .collect()
        };
        var.set(&Tensor::from_vec(values, var.shape(), var.device())?)?;
    }
    Ok(())
}

fn frozen(varmap: &VarMap, config: &LoopedConfig, device: &Device) -> Result<LoopedAgent> {
    let weights: HashMap<_, _> = varmap
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().detach()))
        .collect();
    LoopedAgent::new(
        config.clone(),
        VarBuilder::from_tensors(weights, DType::F32, device),
    )
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .unwrap()
        .0
}

fn save_fitting_checkpoint(vars: &VarMap, root: &Path, update: usize) -> Result<Value> {
    let name = format!("checkpoint-{update:06}.safetensors");
    let path = root.join(&name);
    ensure!(!path.exists(), "checkpoint must be new");
    let temporary = path.with_extension("tmp");
    vars.save(&temporary)?;
    fs::rename(&temporary, &path)?;
    Ok(json!({"file":name,"sha256":file_hash(&path)?,"weights_only":true}))
}

fn fixed_samples(args: &Args) -> Result<Vec<(usize, Sample)>> {
    let mut samples = Vec::new();
    for layout in 0..8 {
        let mut reference = None;
        let mut labels = [0; ACTIONS];
        for scheduled_rule in task::permutation_ids(Split::Train) {
            let rule = if args.known_mapping {
                0
            } else {
                scheduled_rule
            };
            let ep =
                task::episode_with_permutation(args.data_seed ^ TRAIN_TAG, layout, rule, 1, 1)?;
            let sample = task::sample(&ep)?;
            if let Some(current) = &reference {
                ensure!(
                    current == &sample.current,
                    "fixed-set query pixels vary with rule"
                );
            } else {
                reference = Some(sample.current.clone());
            }
            let label = argmax(&sample.policy);
            ensure!(
                sample.policy[label] == 1.0,
                "fixed-set label must be unique"
            );
            labels[label] += 1;
            samples.push((rule, sample));
        }
        ensure!(
            args.known_mapping || labels == [4; ACTIONS],
            "fixed-set labels must be balanced within each layout"
        );
    }
    Ok(samples)
}

fn coverage_sample(args: &Args, id: u64) -> Result<(u64, usize, Sample)> {
    let rules = task::permutation_ids(Split::Train);
    let index = id / rules.len() as u64;
    let episode = if args.known_replay {
        known_replay::episode_id(id)
    } else if args.coverage == Coverage::Fixed {
        index % 8
    } else {
        index
    };
    // Known-rule slots share the original generator; replay changes only their order.
    let rule = if args.known_mapping {
        0
    } else {
        rules[id as usize % rules.len()]
    };
    let ep = task::episode_with_permutation(args.data_seed ^ TRAIN_TAG, episode, rule, 1, 1)?;
    Ok((episode, rule, task::sample(&ep)?))
}

fn input_hash(input: &Inputs) -> String {
    let mut hash = Sha256::new();
    for p in &input.patches {
        hash.update(p.to_le_bytes());
    }
    for m in &input.metadata {
        hash.update(m.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn query_hash(pixels: &[u32]) -> String {
    let mut hash = Sha256::new();
    for p in pixels {
        hash.update(p.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn coverage_row(id: u64, episode: u64, rule: usize, sample: &Sample) -> Value {
    let mut targets = Sha256::new();
    for p in &sample.next {
        targets.update(p.to_le_bytes());
    }
    for p in sample.policy {
        targets.update(p.to_le_bytes());
    }
    for p in sample.rewards {
        targets.update(p.to_le_bytes());
    }
    targets.update(sample.value.to_le_bytes());
    json!({"id":id,"episode_index":episode,"rule":rule,"input_sha256":input_hash(&sample.inputs),"query_sha256":query_hash(&sample.current),"targets_sha256":format!("{:x}",targets.finalize()),"correct_action":argmax(&sample.policy)})
}

fn coverage_audit(args: &Args, started: Instant) -> Result<Value> {
    if args.known_mapping {
        return known_mapping::coverage_audit(args, started);
    }
    let mut log = BufWriter::new(File::create(args.output_dir.join("training-stream.jsonl"))?);
    let mut inputs = HashSet::new();
    let mut queries = HashSet::new();
    let mut episodes = HashSet::new();
    let mut rules = [0usize; 24];
    for id in 0..(args.updates * args.effective_batch) as u64 {
        deadline(args, started)?;
        let (episode, rule, sample) = coverage_sample(args, id)?;
        let row = coverage_row(id, episode, rule, &sample);
        inputs.insert(row["input_sha256"].as_str().unwrap().to_owned());
        queries.insert(row["query_sha256"].as_str().unwrap().to_owned());
        episodes.insert(episode);
        rules[rule] += 1;
        serde_json::to_writer(&mut log, &row)?;
        log.write_all(b"\n")?;
    }
    log.flush()?;
    let mut evaluation = BufWriter::new(File::create(
        args.output_dir.join("evaluation-queries.jsonl"),
    )?);
    for split in [Split::Train, Split::HeldOut] {
        for layout in 0..64 {
            deadline(args, started)?;
            let mut correct = [0usize; ACTIONS];
            for rule in task::permutation_ids(split) {
                let ep = task::episode_with_permutation(
                    20260909 ^ EVAL_TAG ^ 0x52554c45,
                    layout,
                    rule,
                    1,
                    1,
                )?;
                let sample = task::sample(&ep)?;
                let query = query_hash(&sample.current);
                ensure!(
                    !queries.contains(&query),
                    "evaluation query overlaps training"
                );
                let truth = argmax(&task::oracle(&ep.support, &ep.maze.render())?.0);
                ensure!(
                    sample.policy[truth] == 1.0,
                    "oracle control disagrees with target"
                );
                correct[truth] += 1;
                let changed = task::permutations()[rule].map(|d| (d + 1) % ACTIONS);
                let wrong_id = task::permutations()
                    .iter()
                    .position(|p| *p == changed)
                    .context("missing changed rule")?;
                ensure!(
                    task::permutation_ids(split).contains(&wrong_id),
                    "wrong rule crosses split"
                );
                let wrong = task::episode_with_permutation(
                    20260909 ^ EVAL_TAG ^ 0x52554c45,
                    layout,
                    wrong_id,
                    1,
                    1,
                )?;
                ensure!(
                    ep.maze.render() == wrong.maze.render(),
                    "oracle query parity"
                );
                let presented = argmax(&task::oracle(&wrong.support, &wrong.maze.render())?.0);
                ensure!(
                    truth != presented,
                    "oracle counterfactual not outcome changing"
                );
                serde_json::to_writer(
                    &mut evaluation,
                    &json!({"split":split,"layout":layout,"rule":rule,"query_sha256":query,"true_action":truth,"wrong_action":presented,"cleared_fixed_action":0}),
                )?;
                evaluation.write_all(b"\n")?;
            }
            ensure!(
                correct == [task::permutation_ids(split).len() / ACTIONS; ACTIONS],
                "blind control not balanced"
            );
        }
        for id in 0..64 {
            deadline(args, started)?;
            let (min, max) = if id % 2 == 0 { (1, 1) } else { (2, 10) };
            let ep = task::episode(20260909 ^ EVAL_TAG ^ 0x50524544, id, split, min, max)?;
            let sample = task::sample(&ep)?;
            let query = query_hash(&sample.current);
            ensure!(
                !queries.contains(&query),
                "prediction evaluation query overlaps training"
            );
            serde_json::to_writer(
                &mut evaluation,
                &json!({"split":split,"episode":id,"kind":"prediction","query_sha256":query,"input_sha256":input_hash(&sample.inputs)}),
            )?;
            evaluation.write_all(b"\n")?;
        }
    }
    evaluation.flush()?;
    Ok(
        json!({"status":"complete_pending_analysis","evidence_class":"data_audit","coverage":args.coverage,"optimizer_updates":0,"rows":args.updates*args.effective_batch,"unique_inputs":inputs.len(),"unique_query_frames":queries.len(),"episode_ids":episodes.len(),"rule_counts":rules,"evaluation_data_seed":20260909,"evaluation_layouts":64,"evaluation_query_overlap":0,"oracle_factual_accuracy":1.0,"oracle_wrong_under_real":0.0,"oracle_follows_presented":1.0,"cleared_fixed_action_accuracy":0.25,"elapsed_seconds":started.elapsed().as_secs_f64()}),
    )
}

fn fitting_check(
    args: &Args,
    model: &LoopedAgent,
    samples: &[(usize, Sample)],
    device: &Device,
    started: Instant,
    update: usize,
) -> Result<Value> {
    let mut arms = Vec::new();
    for clear in [false, true] {
        let mut ce = 0.0f64;
        let mut correct = 0;
        let mut actions = std::collections::BTreeSet::new();
        let mut hash = Sha256::new();
        for (sample_index, (_, sample)) in samples.iter().enumerate() {
            deadline(args, started)?;
            let mut input = sample.inputs.clone();
            if clear {
                input.patches[..(TOKENS - PATCH_COUNT) * PATCH_PIXELS].fill(0);
                input.metadata[..(TOKENS - PATCH_COUNT) * META_DIM].fill(0.0);
            }
            let out = if args.profile_eval && update == 0 && !clear && sample_index == 0 {
                inspected_predict(args, model, input, device)?
            } else {
                predict(model, &[input], args.loops, device)?
            };
            let logits = out.policy_logits.flatten_all()?.to_vec1::<f32>()?;
            ensure!(
                logits.iter().all(|v| v.is_finite()),
                "non-finite fitting readout"
            );
            for v in &logits {
                hash.update(v.to_le_bytes());
            }
            let label = argmax(&sample.policy);
            let action = argmax(&logits);
            actions.insert(action);
            correct += usize::from(action == label);
            ce -= f64::from(
                candle_nn::ops::log_softmax(&out.policy_logits, D::Minus1)?
                    .flatten_all()?
                    .to_vec1::<f32>()?[label],
            );
        }
        ce /= samples.len() as f64;
        if clear && !args.known_mapping {
            ensure!(
                correct * 4 == samples.len() && ce >= 4.0f64.ln() - 0.0001,
                "fixed-set blind information bound failed"
            );
        }
        arms.push(json!({"cleared":clear,"examples":samples.len(),"policy_ce":ce,"accuracy":correct as f64/samples.len() as f64,"distinct_actions":actions.len(),"action_ids":actions,"prediction_sha256":format!("{:x}",hash.finalize())}));
    }
    let pass = arms[0]["accuracy"]
        .as_f64()
        .context("missing fit accuracy")?
        >= 0.9
        && arms[0]["policy_ce"].as_f64().context("missing fit loss")? <= 0.35;
    let mut report = json!({"update":update,"loops":args.loops,"arms":arms,"fit_gate_pass":pass,"elapsed_seconds":started.elapsed().as_secs_f64()});
    if args.known_mapping {
        report["known_mapping"] = known_mapping::population();
        report["policy_controls"] =
            known_mapping::label_controls(samples.iter().map(|(_, sample)| sample))?;
    }
    Ok(report)
}

#[derive(Clone, Copy, Debug)]
enum Controller {
    Direct,
    Cleared,
    Search,
    Random,
    Oracle,
}

fn observed_input(support: &[Transition], frame: &Frame, clear: bool) -> Result<Inputs> {
    let mut row = task::inputs(support, frame)?;
    if clear {
        row.patches[..(TOKENS - PATCH_COUNT) * PATCH_PIXELS].fill(0);
        row.metadata[..(TOKENS - PATCH_COUNT) * META_DIM].fill(0.0);
    }
    Ok(row)
}

fn search_action(
    model: &LoopedAgent,
    support: &[Transition],
    root: &LoopedOutput,
    loops: usize,
    device: &Device,
) -> Result<usize> {
    // These are imagined branches. They never become observed support transitions.
    let next = root
        .next_logits
        .argmax(D::Minus1)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let first_rows = next
        .chunks_exact(PIXELS)
        .map(|pixels| task::inputs(support, &task::unpatchify(pixels)?))
        .collect::<Result<Vec<_>>>()?;
    let first = predict(model, &first_rows, loops, device)?;
    let grandchildren = first
        .next_logits
        .argmax(D::Minus1)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let second_rows = grandchildren
        .chunks_exact(PIXELS)
        .map(|pixels| task::inputs(support, &task::unpatchify(pixels)?))
        .collect::<Result<Vec<_>>>()?;
    // Bound inference memory by four nodes per call, independently of training batch.
    let mut values = Vec::new();
    for batch in second_rows.chunks(ACTIONS) {
        values.extend(
            candle_nn::ops::sigmoid(&predict(model, batch, loops, device)?.value)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        );
    }
    let r0 = candle_nn::ops::sigmoid(&root.reward_logits)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let r1 = candle_nn::ops::sigmoid(&first.reward_logits)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut scores = [0.0; ACTIONS];
    for a in 0..ACTIONS {
        let child = (0..ACTIONS)
            .map(|b| {
                let i = a * ACTIONS + b;
                r1[i] + (1.0 - r1[i]) * task::DISCOUNT * values[i]
            })
            .fold(f32::NEG_INFINITY, f32::max);
        scores[a] = r0[a] + (1.0 - r0[a]) * task::DISCOUNT * child;
    }
    ensure!(
        scores.iter().all(|x| x.is_finite()),
        "non-finite search value"
    );
    Ok(argmax(&scores))
}

fn wilson(wins: usize, n: usize) -> [f64; 2] {
    let n = n as f64;
    let p = wins as f64 / n;
    let z2 = 1.959963984540054_f64.powi(2);
    let center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n);
    let radius = (z2 * (p * (1.0 - p) / n + z2 / (4.0 * n * n))).sqrt() / (1.0 + z2 / n);
    [center - radius, center + radius]
}

fn evaluate(args: &Args, model: &LoopedAgent, device: &Device, started: Instant) -> Result<Value> {
    let count = if args.mode == Mode::Smoke {
        1
    } else {
        args.eval_episodes
    };
    let cap = if args.mode == Mode::Smoke {
        2
    } else {
        args.step_cap
    };
    let mut log = BufWriter::new(File::create(args.output_dir.join("episodes.jsonl"))?);
    let mut reports = Vec::new();
    let mut depths = vec![1, 2, args.loops, args.max_loops];
    depths.retain(|&d| d <= args.max_loops);
    depths.sort();
    depths.dedup();
    if args.mode == Mode::Smoke {
        depths = vec![args.loops];
    }
    for split in [Split::Train, Split::HeldOut] {
        let mut arms = depths
            .iter()
            .map(|&d| (Controller::Direct, d))
            .collect::<Vec<_>>();
        arms.extend([
            (Controller::Cleared, args.loops),
            (Controller::Random, 0),
            (Controller::Oracle, 0),
        ]);
        if args.search {
            arms.push((Controller::Search, args.loops));
        }
        for (controller, loops) in arms {
            let mut wins = 0;
            let mut total_steps = 0;
            let mut efficiency = 0.0;
            let arm_start = Instant::now();
            for id in 0..count {
                deadline(args, started)?;
                let mut episode =
                    task::episode(args.data_seed ^ EVAL_TAG, id as u64, split, 2, 10)?;
                let (_, optimal) = task::oracle(&episode.support, &episode.maze.render())?;
                let mut rng = ChaCha8Rng::seed_from_u64(args.data_seed ^ EVAL_TAG ^ id as u64);
                let mut actions = Vec::new();
                for _ in 0..cap {
                    deadline(args, started)?;
                    let frame = episode.maze.render();
                    let action = match controller {
                        Controller::Random => rng.random_range(0..ACTIONS),
                        Controller::Oracle => argmax(&task::oracle(&episode.support, &frame)?.0),
                        _ => {
                            let row = observed_input(
                                &episode.support,
                                &frame,
                                matches!(controller, Controller::Cleared),
                            )?;
                            let out = predict(model, &[row], loops, device)?;
                            if matches!(controller, Controller::Search) {
                                search_action(model, &episode.support, &out, loops, device)?
                            } else {
                                argmax(&out.policy_logits.flatten_all()?.to_vec1::<f32>()?)
                            }
                        }
                    };
                    actions.push(action);
                    if episode.maze.step(action)? {
                        break;
                    }
                }
                let won = episode.maze.done();
                wins += usize::from(won);
                total_steps += actions.len();
                if won {
                    efficiency += (optimal + 3) as f64 / (actions.len() + 3) as f64;
                }
                let row = json!({"split":split,"controller":format!("{controller:?}"),"loops":loops,"episode":id,"permutation_id":episode.permutation_id,"won":won,"query_actions":actions,"calibration_actions":3,"optimal_query_actions":optimal});
                serde_json::to_writer(&mut log, &row)?;
                log.write_all(b"\n")?;
                log.flush()?;
            }
            reports.push(json!({"split":split,"controller":format!("{controller:?}"),"loops":loops,"episodes":count,"wins":wins,"win_rate":wins as f64/count as f64,"win_rate_wilson95":wilson(wins,count),"query_actions":total_steps,"total_actions_including_calibration":total_steps+3*count,"success_weighted_efficiency":efficiency/count as f64,"seconds":arm_start.elapsed().as_secs_f64(),"imagined_nodes_per_action":if matches!(controller,Controller::Search) {20} else {0}}));
        }
    }
    Ok(json!(reports))
}

fn prediction_metrics(
    args: &Args,
    model: &LoopedAgent,
    device: &Device,
    started: Instant,
) -> Result<Value> {
    let count = if args.mode == Mode::Smoke {
        1
    } else {
        args.eval_episodes
    };
    let mut reports = Vec::new();
    let mut rows_log = BufWriter::new(File::create(args.output_dir.join("predictions.jsonl"))?);
    for split in [Split::Train, Split::HeldOut] {
        let mut exact = 0;
        let mut copy_exact = 0;
        let mut changed_correct = 0;
        let mut changed_total = 0;
        let mut vacated_total = 0;
        let mut vacated_correct = 0;
        let mut destination_total = 0;
        let mut destination_correct = 0;
        let mut inconsistent_patches = 0;
        let mut policy_correct = 0;
        let mut reward_positive = 0;
        let mut reward_tp = 0;
        let mut reward_fp = 0;
        let mut value_squared = 0.0;
        let mut constant_value_squared = 0.0;
        for id in 0..count {
            deadline(args, started)?;
            let (min, max) = if id.is_multiple_of(2) {
                (1, 1)
            } else {
                (2, 10)
            };
            let ep = task::episode(
                args.data_seed ^ EVAL_TAG ^ 0x50524544,
                id as u64,
                split,
                min,
                max,
            )?;
            let sample = task::sample(&ep)?;
            let out = predict(
                model,
                std::slice::from_ref(&sample.inputs),
                args.loops,
                device,
            )?;
            let pixels = out
                .next_logits
                .argmax(D::Minus1)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            let action = argmax(&out.policy_logits.flatten_all()?.to_vec1::<f32>()?);
            policy_correct += usize::from(sample.policy[action] > 0.0);
            let value = candle_nn::ops::sigmoid(&out.value)?
                .flatten_all()?
                .to_vec1::<f32>()?[0];
            value_squared += f64::from((value - sample.value).powi(2));
            constant_value_squared += f64::from((0.9 - sample.value).powi(2));
            let reward = candle_nn::ops::sigmoid(&out.reward_logits)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for a in 0..ACTIONS {
                let real = &sample.next[a * PIXELS..(a + 1) * PIXELS];
                let predicted = &pixels[a * PIXELS..(a + 1) * PIXELS];
                let same = real == predicted;
                exact += usize::from(same);
                copy_exact += usize::from(real == sample.current);
                let mut changed = 0;
                let mut correct = 0;
                for p in 0..PIXELS {
                    if real[p] != sample.current[p] {
                        changed += 1;
                        correct += usize::from(real[p] == predicted[p]);
                        if sample.current[p] == 2 && real[p] == 0 {
                            vacated_total += 1;
                            vacated_correct += usize::from(real[p] == predicted[p]);
                        } else {
                            ensure!(
                                [0, 3].contains(&sample.current[p]) && [2, 4].contains(&real[p]),
                                "unexpected changed pixel in diagnostic decomposition"
                            );
                            destination_total += 1;
                            destination_correct += usize::from(real[p] == predicted[p]);
                        }
                    }
                }
                inconsistent_patches += predicted
                    .chunks_exact(PATCH_PIXELS)
                    .filter(|patch| patch.iter().any(|p| *p != patch[0]))
                    .count();
                changed_total += changed;
                changed_correct += correct;
                let positive = sample.rewards[a] > 0.5;
                reward_positive += usize::from(positive);
                reward_tp += usize::from(positive && reward[a] >= 0.5);
                reward_fp += usize::from(!positive && reward[a] >= 0.5);
                serde_json::to_writer(
                    &mut rows_log,
                    &json!({"split":split,"episode":id,"permutation_id":ep.permutation_id,"input_sha256":input_hash(&sample.inputs),"query_sha256":query_hash(&sample.current),"action":a,"frame_exact":same,"copy_exact":real==sample.current,"changed_pixels":changed,"changed_correct":correct,"reward_target":sample.rewards[a],"reward_probability":reward[a],"policy_correct":sample.policy[action]>0.0}),
                )?;
                rows_log.write_all(b"\n")?;
            }
        }
        ensure!(
            vacated_total + destination_total == changed_total,
            "changed-pixel decomposition is incomplete"
        );
        reports.push(json!({"split":split,"rows":count,"action_tuples":count*ACTIONS,"next_frame_exact":exact,"copy_frame_exact":copy_exact,"changed_pixels":changed_total,"changed_correct":changed_correct,"changed_accuracy":changed_correct as f64/changed_total.max(1) as f64,"policy_optimal_action_accuracy":policy_correct as f64/count as f64,"value_mse":value_squared/count as f64,"constant_value_prediction":0.9,"constant_value_mse":constant_value_squared/count as f64,"reward_positive":reward_positive,"reward_true_positive":reward_tp,"reward_false_positive":reward_fp,"decomposition":{"vacated_pixels":vacated_total,"vacated_correct":vacated_correct,"destination_pixels":destination_total,"destination_correct":destination_correct,"inconsistent_predicted_patches":inconsistent_patches}}));
    }
    rows_log.flush()?;
    Ok(json!(reports))
}

fn rule_probe(
    args: &Args,
    model: &LoopedAgent,
    device: &Device,
    started: Instant,
) -> Result<Value> {
    // Paired one-step queries remove path planning as a possible confound.
    // Each identical image occurs under all balanced rules in a split. A
    // context-free deterministic policy therefore gets exactly one quarter.
    let layouts = if args.mode == Mode::Smoke {
        1
    } else {
        args.eval_episodes
    };
    let permutations = task::permutations();
    let mut log = BufWriter::new(File::create(args.output_dir.join("rule-probe.jsonl"))?);
    let mut reports = Vec::new();
    for split in [Split::Train, Split::HeldOut] {
        let mut true_correct = 0;
        let mut clear_correct = 0;
        let mut wrong_correct = 0;
        let mut follows_wrong_rule = 0;
        let mut rows = 0;
        for layout in 0..layouts {
            for rule in task::permutation_ids(split) {
                deadline(args, started)?;
                let seed = args.data_seed ^ EVAL_TAG ^ 0x52554c45;
                let ep = task::episode_with_permutation(seed, layout as u64, rule, 1, 1)?;
                let frame = ep.maze.render();
                let truth = argmax(&task::oracle(&ep.support, &frame)?.0);
                let changed_rule = permutations[rule].map(|direction| (direction + 1) % ACTIONS);
                let wrong_id = permutations
                    .iter()
                    .position(|p| *p == changed_rule)
                    .context("missing counterfactual rule")?;
                ensure!(
                    task::permutation_ids(split).contains(&wrong_id),
                    "rule probe crosses split"
                );
                let wrong = task::episode_with_permutation(seed, layout as u64, wrong_id, 1, 1)?;
                ensure!(wrong.maze.render() == frame, "paired query pixels changed");
                let wrong_truth = argmax(&task::oracle(&wrong.support, &frame)?.0);
                ensure!(
                    truth != wrong_truth,
                    "counterfactual rule did not change optimal action"
                );
                let inputs = [
                    observed_input(&ep.support, &frame, false)?,
                    observed_input(&ep.support, &frame, true)?,
                    observed_input(&wrong.support, &frame, false)?,
                ];
                // Each call is B=1 so this control also runs in the smallest capacity smoke.
                let mut actions = Vec::new();
                let mut logits = Vec::new();
                let mut probabilities = Vec::new();
                let mut values = Vec::new();
                let mut rewards = Vec::new();
                let mut input_hashes = Vec::new();
                for input in inputs {
                    input_hashes.push(input_hash(&input));
                    let output = if args.profile_eval
                        && split == Split::Train
                        && rows == 0
                        && actions.is_empty()
                    {
                        inspected_predict(args, model, input, device)?
                    } else {
                        predict(model, &[input], args.loops, device)?
                    };
                    let policy = output.policy_logits.flatten_all()?.to_vec1::<f32>()?;
                    ensure!(
                        policy.iter().all(|p| p.is_finite()),
                        "non-finite policy logit"
                    );
                    actions.push(argmax(&policy));
                    logits.push(policy);
                    probabilities.push(
                        candle_nn::ops::softmax(&output.policy_logits, D::Minus1)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                    );
                    values.push(
                        candle_nn::ops::sigmoid(&output.value)?
                            .flatten_all()?
                            .to_vec1::<f32>()?[0],
                    );
                    rewards.push(
                        candle_nn::ops::sigmoid(&output.reward_logits)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                    );
                }
                rows += 1;
                true_correct += usize::from(actions[0] == truth);
                clear_correct += usize::from(actions[1] == truth);
                wrong_correct += usize::from(actions[2] == truth);
                follows_wrong_rule += usize::from(actions[2] == wrong_truth);
                serde_json::to_writer(
                    &mut log,
                    &json!({"split":split,"layout":layout,"rule":rule,"wrong_rule":wrong_id,"correct_action":truth,"wrong_rule_action":wrong_truth,"query_sha256":query_hash(&task::patchify(&frame)?),"predictions_true_clear_wrong":actions,"input_sha256_true_clear_wrong":input_hashes,"policy_logits_true_clear_wrong":logits,"policy_probabilities_true_clear_wrong":probabilities,"value_true_clear_wrong":values,"reward_probabilities_true_clear_wrong":rewards}),
                )?;
                log.write_all(b"\n")?;
            }
        }
        ensure!(
            clear_correct * 4 == rows,
            "cleared-context control violated balanced blind bound"
        );
        reports.push(json!({"split":split,"layouts":layouts,"rows":rows,"eligible_rows":rows,"changed_rule_tuples":rows,"outcome_changing_queries":rows,"true_context_correct":true_correct,"true_context_accuracy":true_correct as f64/rows as f64,"cleared_correct":clear_correct,"cleared_accuracy":clear_correct as f64/rows as f64,"wrong_context_correct_under_real_rule":wrong_correct,"wrong_context_follows_presented_rule":follows_wrong_rule}));
    }
    log.flush()?;
    Ok(json!(reports))
}

fn run(args: &Args, started: Instant) -> Result<Value> {
    let provenance = provenance(args)?;
    let config = LoopedConfig {
        hidden: args.hidden,
        heads: args.heads,
        layers: args.layers,
        max_loops: args.max_loops,
    };
    let mut metadata = json!({"status":"running","schema":task::SCHEMA,"exact_args":std::env::args().collect::<Vec<_>>(),"provenance":provenance,"model":config,"seed":args.seed,"data_seed":args.data_seed,"physical_batch":args.batch,"accumulation":args.accumulation(),"effective_batch":args.effective_batch,"objectives":{"policy":1.0,"value":0.1,"reward":0.1,"balanced_categorical_dynamics":0.5},"training_rule_ids":task::permutation_ids(Split::Train),"held_out_rule_ids":task::permutation_ids(Split::HeldOut),"boundary":"scripted three-action calibration and visible objective; no active probe selection, hidden objectives, ARC data, or pretrained language weights"});
    if args.known_mapping {
        known_mapping::annotate(&mut metadata);
        known_mapping::annotate_seen(args, &mut metadata);
        known_replay::annotate(args, &mut metadata);
    }
    write_json(&args.output_dir.join("metadata.json"), &metadata)?;
    if args.mode == Mode::CoverageAudit {
        return coverage_audit(args, started);
    }
    if args.mode == Mode::CounterfactualAudit {
        return counterfactual::audit(args, started);
    }
    if args.mode == Mode::KnownMappingAudit {
        return known_mapping::audit(args, started, &HashSet::new());
    }
    let device = resolve_device(&args.device)?;
    let mut vars = VarMap::new();
    let model = LoopedAgent::new(
        config.clone(),
        VarBuilder::from_varmap(&vars, DType::F32, &device),
    )?;
    initialize(&vars, args.seed)?;
    if let Some(path) = &args.checkpoint {
        vars.load(path)?;
    }
    let parameters = vars
        .all_vars()
        .iter()
        .map(|v| v.elem_count())
        .sum::<usize>();
    let updates = match args.mode {
        Mode::Smoke | Mode::FitSmoke => 2,
        Mode::Train | Mode::Fit | Mode::Coverage => args.updates,
        Mode::Evaluate | Mode::Inspect | Mode::Counterfactual | Mode::KnownMapping => 0,
        Mode::CoverageAudit | Mode::CounterfactualAudit | Mode::KnownMappingAudit => {
            unreachable!("data audit returns before model construction")
        }
    };
    let mut optimizer = AdamW::new(
        vars.all_vars(),
        ParamsAdamW {
            lr: args.learning_rate,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;
    let mut log = BufWriter::new(File::create(args.output_dir.join("updates.jsonl"))?);
    vars.save(args.output_dir.join("initial.safetensors"))?;
    let fixed = if matches!(args.mode, Mode::Fit | Mode::FitSmoke | Mode::Coverage) {
        ensure!(
            file_hash(&args.output_dir.join("initial.safetensors"))?
                == "4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802",
            "fixed fit initialization differs from the original screen"
        );
        Some(fixed_samples(args)?)
    } else {
        None
    };
    let mut fit_log = if fixed.is_some() {
        Some(BufWriter::new(File::create(
            args.output_dir.join("fit.jsonl"),
        )?))
    } else {
        None
    };
    let mut fit_result = None;
    if let Some(samples) = &fixed {
        let mut hash = Sha256::new();
        for (rule, s) in samples {
            hash.update((*rule as u64).to_le_bytes());
            for p in &s.inputs.patches {
                hash.update(p.to_le_bytes());
            }
            for m in &s.inputs.metadata {
                hash.update(m.to_le_bytes());
            }
            for p in &s.next {
                hash.update(p.to_le_bytes());
            }
            for p in s.policy {
                hash.update(p.to_le_bytes());
            }
        }
        let mut fixed_set = json!({"layouts":8,"rules":task::permutation_ids(Split::Train),"examples":samples.len(),"order":"layout-major then increasing training rule ID; repeated cyclically","distance":1,"sha256":format!("{:x}",hash.finalize())});
        if args.known_mapping {
            let queries: HashSet<_> = samples
                .iter()
                .map(|(_, sample)| query_hash(&sample.current))
                .collect();
            ensure!(
                queries.len() == 8,
                "known reference requires eight unique queries"
            );
            fixed_set["rules"] = json!([0]);
            fixed_set["unique_query_frames"] = json!(queries.len());
            fixed_set["order"] = json!("eight original layouts, sixteen identical known-rule samples per layout; repeated cyclically");
            fixed_set["known_mapping"] = known_mapping::population();
            fixed_set["policy_controls"] =
                known_mapping::label_controls(samples.iter().map(|(_, sample)| sample))?;
            known_replay::annotate(args, &mut fixed_set);
        }
        write_json(&args.output_dir.join("fixed-set.json"), &fixed_set)?;
        let mut row = fitting_check(
            args,
            &frozen(&vars, &config, &device)?,
            samples,
            &device,
            started,
            0,
        )?;
        known_replay::annotate(args, &mut row);
        let writer = fit_log.as_mut().context("missing fit log")?;
        serde_json::to_writer(&mut *writer, &row)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        fit_result = Some(row);
    }
    let mut stream_hash = Sha256::new();
    let mut coverage_log = if args.mode == Mode::Coverage {
        Some(BufWriter::new(File::create(
            args.output_dir.join("training-stream.jsonl"),
        )?))
    } else {
        None
    };
    let train_start = Instant::now();
    let mut updates_done = 0;
    for update in 0..updates {
        deadline(args, started)?;
        let update_start = Instant::now();
        let loops = if matches!(args.mode, Mode::Smoke | Mode::FitSmoke) {
            args.loops
        } else {
            [1, 2, args.loops][update % 3].min(args.loops)
        };
        let capture = if args.profile_updates.contains(&(update + 1)) {
            Some(LoopedCapture::begin(
                &args
                    .output_dir
                    .with_extension("profiles")
                    .join(format!("update-{:012}", update + 1)),
                (update + 1) as u64,
                &device,
                &vars,
                args.batch,
                args.effective_batch,
                loops,
            )?)
        } else {
            None
        };
        if capture.is_some() {
            device.synchronize()?;
        }
        let measured = capture.as_ref().map(LoopedCapture::measurement);
        let mut grads = None;
        let mut means = [0.0f32; 4];
        for micro in 0..args.accumulation() {
            deadline(args, started)?;
            let micro_batch = args.batch.min(args.effective_batch - micro * args.batch);
            let fraction = micro_batch as f64 / args.effective_batch as f64;
            let mut samples = Vec::with_capacity(micro_batch);
            for b in 0..micro_batch {
                let id = (update * args.effective_batch + micro * args.batch + b) as u64;
                let (permutation_id, sample) = if args.mode == Mode::Coverage {
                    let (episode, rule, sample) = coverage_sample(args, id)?;
                    let log = coverage_log.as_mut().context("missing coverage log")?;
                    let mut row = if args.known_mapping {
                        known_mapping::training_row(id, episode, &sample)
                    } else {
                        coverage_row(id, episode, rule, &sample)
                    };
                    if args.known_replay {
                        known_replay::annotate_row(id, episode, &mut row);
                    }
                    serde_json::to_writer(&mut *log, &row)?;
                    log.write_all(b"\n")?;
                    (rule, sample)
                } else if let Some(pool) = &fixed {
                    pool[id as usize % pool.len()].clone()
                } else {
                    let (min, max) = if id.is_multiple_of(2) {
                        (1, 1)
                    } else {
                        (2, 10)
                    };
                    let ep = task::episode(args.data_seed ^ TRAIN_TAG, id, Split::Train, min, max)?;
                    (ep.permutation_id, task::sample(&ep)?)
                };
                stream_hash.update(id.to_le_bytes());
                stream_hash.update((permutation_id as u64).to_le_bytes());
                for pixel in &sample.inputs.patches {
                    stream_hash.update(pixel.to_le_bytes());
                }
                samples.push(sample);
            }
            let inputs = samples.iter().map(|s| s.inputs.clone()).collect::<Vec<_>>();
            if capture.is_some() {
                device.synchronize()?;
            }
            let forward = capture.as_ref().map(|c| {
                c.phase(
                    &format!("micro-{micro}/forward"),
                    Some(candle_graph::ExecutionStep::Forward),
                )
            });
            let out = predict(&model, &inputs, loops, &device)?;
            let (loss, metrics) = losses(&out, &samples, &device)?;
            if let (Some(c), Some(p)) = (&capture, &forward) {
                c.record_tensor_stats(
                    p,
                    &format!("micro-{micro}/policy_logits"),
                    &out.policy_logits,
                )?;
                c.record_tensor_stats(p, &format!("micro-{micro}/value"), &out.value)?;
                c.record_tensor_stats(
                    p,
                    &format!("micro-{micro}/reward_logits"),
                    &out.reward_logits,
                )?;
                c.record_tensor_stats(p, &format!("micro-{micro}/next_logits"), &out.next_logits)?;
                for (name, value) in ["policy", "value", "reward", "dynamics"]
                    .into_iter()
                    .zip(metrics)
                {
                    c.record_scalar(p, &format!("micro-{micro}/loss/{name}"), f64::from(value))?;
                }
                device.synchronize()?;
            }
            drop(forward);
            for i in 0..4 {
                means[i] += metrics[i] * fraction as f32;
            }
            let backward = capture.as_ref().map(|c| {
                c.phase(
                    &format!("micro-{micro}/backward"),
                    Some(candle_graph::ExecutionStep::Backward),
                )
            });
            accumulate_parameter_gradients(&mut grads, (loss * fraction)?.backward()?, &vars)?;
            if capture.is_some() {
                device.synchronize()?;
            }
            drop(backward);
        }
        let mut grads = grads.context("missing gradients")?;
        let clipping = capture
            .as_ref()
            .map(|c| c.phase("gradient-inspection-and-clip", None));
        if let (Some(c), Some(p)) = (&capture, &clipping) {
            c.record_gradients(p, &grads)?;
        }
        let clip = clip_gradients_gpu_with_stats(&mut grads, &vars, 1.0)?;
        if let (Some(c), Some(p)) = (&capture, &clipping) {
            c.record_scalar(p, "gradient/pre_clip_l2", clip.pre_clip_norm)?;
            c.record_scalar(p, "gradient/clip_scale", clip.scale)?;
            device.synchronize()?;
        }
        drop(clipping);
        ensure!(
            clip.pre_clip_norm > 0.0 && clip.pre_clip_norm.is_finite(),
            "invalid gradient norm"
        );
        let optimizer_phase = capture
            .as_ref()
            .map(|c| c.phase("optimizer", Some(candle_graph::ExecutionStep::Optimizer)));
        optimizer.step(&grads)?;
        drop(grads);
        device.synchronize()?;
        drop(optimizer_phase);
        drop(measured);
        if let Some(capture) = capture {
            capture.finish()?;
        }
        updates_done = update + 1;
        let row = json!({"update":update+1,"loops":loops,"losses":means,"gradient_l2":clip.pre_clip_norm,"clip_scale":clip.scale,"update_seconds":update_start.elapsed().as_secs_f64(),"elapsed_seconds":started.elapsed().as_secs_f64()});
        serde_json::to_writer(&mut log, &row)?;
        log.write_all(b"\n")?;
        log.flush()?;
        if (update + 1) % 16 == 0 || update == 0 {
            println!("{}", row);
        }
        if let Some(samples) = &fixed {
            if args.mode == Mode::FitSmoke
                || updates_done.is_multiple_of(25)
                || updates_done == updates
            {
                let mut row = fitting_check(
                    args,
                    &frozen(&vars, &config, &device)?,
                    samples,
                    &device,
                    started,
                    updates_done,
                )?;
                known_replay::annotate(args, &mut row);
                row["checkpoint"] = save_fitting_checkpoint(&vars, &args.output_dir, updates_done)?;
                let pass = row["fit_gate_pass"]
                    .as_bool()
                    .context("missing fitting gate")?;
                let writer = fit_log.as_mut().context("missing fit log")?;
                serde_json::to_writer(&mut *writer, &row)?;
                writer.write_all(b"\n")?;
                writer.flush()?;
                println!("{}", row);
                fit_result = Some(row);
                if pass && args.mode == Mode::Fit {
                    break;
                }
            }
        }
    }
    let train_seconds = train_start.elapsed().as_secs_f64();
    drop(log);
    drop(optimizer);
    drop(model);
    drop(fit_log);
    if let Some(mut log) = coverage_log {
        log.flush()?;
    }
    vars.save(args.output_dir.join("final.safetensors"))?;
    if let Some(fit) = fit_result {
        let mut report = json!({"status":"complete_pending_analysis","evidence_class":if args.mode==Mode::FitSmoke {"implementation_smoke"} else if args.mode==Mode::Coverage {"coverage_screen"} else {"fitting_diagnostic"},"coverage":args.coverage,"claim_boundary":"fixed reference fitting only; frozen paired evaluation required for generalization; no ARC claim","parameters":parameters,"optimizer_updates":updates_done,"requested_updates":updates,"physical_batch":args.batch,"accumulation":args.accumulation(),"effective_batch":args.effective_batch,"training_and_fit_check_seconds":train_seconds,"elapsed_seconds":started.elapsed().as_secs_f64(),"fit":fit,"training_stream_sha256":format!("{:x}",stream_hash.finalize()),"final_checkpoint_sha256":file_hash(&args.output_dir.join("final.safetensors"))?});
        if args.known_mapping {
            known_mapping::annotate(&mut report);
            report["evidence_class"] = json!(if args.mode == Mode::FitSmoke {
                "implementation_smoke"
            } else {
                "known_mapping_spatial_prerequisite"
            });
            report["claim_boundary"] = json!("single-seed fixed-known-control spatial prerequisite; fitted reference is not generalization; no variable-rule comparison or promotion");
            known_replay::annotate(args, &mut report);
            if args.known_replay {
                report["evidence_class"] = json!(if args.updates == 3 {
                    "implementation_smoke"
                } else {
                    "known_mapping_replay_order_screen"
                });
                report["training_stream_file_sha256"] =
                    json!(file_hash(&args.output_dir.join("training-stream.jsonl"))?);
            }
        }
        return Ok(report);
    }
    let model = frozen(&vars, &config, &device)?;
    if args.mode == Mode::KnownMapping {
        let evaluation = known_mapping::evaluate(args, &model, &device, started)?;
        let mut report = json!({
            "status": "complete_pending_analysis", "evidence_class": "frozen_known_mapping_diagnostic",
            "known_mapping": known_mapping::population(), "optimizer_updates": 0,
            "parameters": parameters, "physical_batch": args.batch, "effective_batch": args.effective_batch,
            "accumulation": args.accumulation(), "evaluation": evaluation,
            "final_checkpoint_sha256": file_hash(&args.output_dir.join("final.safetensors"))?,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "claim_boundary": "single-seed known spatial prerequisite; no variable-rule comparison, ARC claim or promotion"
        });
        known_mapping::annotate_seen(args, &mut report);
        return Ok(report);
    }
    if args.mode == Mode::Counterfactual {
        let counterfactual = counterfactual::evaluate(args, &model, &device, started)?;
        return Ok(json!({
            "status": "complete_pending_analysis",
            "evidence_class": "frozen_checkpoint_diagnostic",
            "claim_boundary": "synthetic successor counterfactuals only; no optimizer updates or ARC claim",
            "parameters": parameters,
            "optimizer_updates": updates,
            "physical_batch": args.batch,
            "accumulation": args.accumulation(),
            "effective_batch": args.effective_batch,
            "counterfactual": counterfactual,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
            "final_checkpoint_sha256": file_hash(&args.output_dir.join("final.safetensors"))?,
        }));
    }
    let predictions = prediction_metrics(args, &model, &device, started)?;
    let rule_probe = rule_probe(args, &model, &device, started)?;
    let episodes = if args.mode == Mode::Inspect {
        json!([])
    } else {
        evaluate(args, &model, &device, started)?
    };
    Ok(
        json!({"status":"complete_pending_analysis","evidence_class":if args.mode==Mode::Smoke {"implementation_smoke"} else if args.mode==Mode::Inspect {"frozen_checkpoint_diagnostic"} else {"exploratory"},"claim_boundary":"synthetic calibrated control prerequisite, not ARC performance","parameters":parameters,"optimizer_updates":updates,"physical_batch":args.batch,"accumulation":args.accumulation(),"effective_batch":args.effective_batch,"training_seconds":train_seconds,"elapsed_seconds":started.elapsed().as_secs_f64(),"training_stream_sha256":format!("{:x}",stream_hash.finalize()),"predictions":predictions,"rule_probe":rule_probe,"episodes":episodes,"final_checkpoint_sha256":file_hash(&args.output_dir.join("final.safetensors"))?}),
    )
}

fn seal(root: &Path) -> Result<()> {
    let mut names = fs::read_dir(root)?
        .map(|e| Ok(e?.path()))
        .collect::<Result<Vec<_>>>()?;
    names.sort();
    let mut hashes = serde_json::Map::new();
    for path in names {
        ensure!(path.is_file(), "unexpected directory in run root");
        hashes.insert(
            path.file_name()
                .context("missing artifact name")?
                .to_string_lossy()
                .to_string(),
            json!(file_hash(&path)?),
        );
    }
    let manifest = root.join("manifest.json");
    write_json(
        &manifest,
        &json!({"schema":"looped-agent-artifacts-v1","files":hashes}),
    )?;
    let saved: Value = serde_json::from_slice(&fs::read(&manifest)?)?;
    for (name, hash) in saved["files"].as_object().context("bad manifest")? {
        ensure!(
            file_hash(&root.join(name))? == hash.as_str().context("bad artifact digest")?,
            "artifact hash mismatch"
        );
    }
    fs::write(
        root.with_extension("manifest.sha256"),
        format!("{}\n", file_hash(&manifest)?),
    )?;
    Ok(())
}

fn bind_profiles(root: &Path) -> Result<()> {
    fn files(root: &Path, path: &Path, hashes: &mut serde_json::Map<String, Value>) -> Result<()> {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() {
                files(root, &path, hashes)?;
            } else {
                ensure!(kind.is_file(), "unexpected profile artifact type");
                hashes.insert(
                    path.strip_prefix(root)?.to_string_lossy().into_owned(),
                    json!(file_hash(&path)?),
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
    let host_trace = std::env::var_os("TOFY_PERF_TRACE")
        .map(PathBuf::from)
        .map(|path| -> Result<Value> { Ok(json!({"sha256":file_hash(&path)?,"path":path})) })
        .transpose()?;
    write_json(
        &root.join("profiles.json"),
        &json!({"root":profile_root,"files":hashes,"host_trace":host_trace,"nsight":"external capture and export; bind in a separate bundle after profiler exit"}),
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    known_mapping::validate_args(&args)?;
    let perf_guard = tofy::perf::install()?;
    ensure!(
        args.loops > 0 && args.max_loops >= args.loops,
        "inference depth must cover positive training depth"
    );
    ensure!(
        args.batch > 0
            && args.effective_batch >= args.batch
            && args.updates > 0
            && args.eval_episodes > 0
            && args.step_cap > 0
            && args.max_seconds > 0,
        "counts must be positive"
    );
    ensure!(
        args.learning_rate > 0.0 && args.learning_rate.is_finite(),
        "invalid learning rate"
    );
    ensure!(
        !matches!(args.mode, Mode::Counterfactual | Mode::KnownMapping)
            || (args.batch == 1 && args.effective_batch == 1),
        "counterfactual evaluation requires physical/effective batch one"
    );
    ensure!(
        !matches!(
            args.mode,
            Mode::Evaluate | Mode::Inspect | Mode::Counterfactual | Mode::KnownMapping
        ) || args.checkpoint.is_some(),
        "evaluation requires a checkpoint"
    );
    ensure!(
        matches!(
            args.mode,
            Mode::Evaluate | Mode::Inspect | Mode::Counterfactual | Mode::KnownMapping
        ) || args.checkpoint.is_none(),
        "training uses fresh initialization; resume needs optimizer provenance"
    );
    if !matches!(
        args.mode,
        Mode::Evaluate
            | Mode::Inspect
            | Mode::Counterfactual
            | Mode::CoverageAudit
            | Mode::CounterfactualAudit
            | Mode::KnownMapping
            | Mode::KnownMappingAudit
    ) {
        let count = if matches!(args.mode, Mode::Smoke | Mode::FitSmoke) {
            2
        } else {
            args.updates
        };
        ensure!(
            !args.profile_updates.is_empty()
                && args.profile_updates.iter().all(|&u| u > 0 && u <= count),
            "profile updates must be nonempty and reachable"
        );
        ensure!(
            cfg!(feature = "profiling") && perf_guard.is_some(),
            "training requires the profiling feature and TOFY_PERF_TRACE"
        );
    }
    fs::create_dir(&args.output_dir).context("run root must be new, with an existing parent")?;
    write_json(
        &args.output_dir.join("launch.json"),
        &json!({
            "status":"running", "pid":std::process::id(), "exact_args":std::env::args().collect::<Vec<_>>(),
            "source_revision":env!("TOFY_EMBEDDED_SOURCE_REVISION"),
            "binary_sha256":file_hash(&std::env::current_exe()?)?
        }),
    )?;
    let started = Instant::now();
    let result = run(&args, started);
    drop(perf_guard);
    match &result {
        Ok(report) => {
            write_json(&args.output_dir.join("report.json"), report)?;
            println!("{}", serde_json::to_string_pretty(report)?);
        }
        Err(error) => write_json(
            &args.output_dir.join("report.json"),
            &json!({"status":"failed_integrity_or_evaluation","error":format!("{error:#}"),"elapsed_seconds":started.elapsed().as_secs_f64()}),
        )?,
    }
    bind_profiles(&args.output_dir)?;
    seal(&args.output_dir)?;
    result.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_sampler_preserves_fixed_problem_and_only_expands_episode_ids() -> Result<()> {
        let mut args = Args::parse_from([
            "probe",
            "--mode",
            "coverage-audit",
            "--output-dir",
            "unused",
        ]);
        let fixed = fixed_samples(&args)?;
        for id in 0..256 {
            let (episode, rule, sample) = coverage_sample(&args, id)?;
            let (old_rule, old) = &fixed[id as usize % fixed.len()];
            assert_eq!(rule, *old_rule);
            assert_eq!(input_hash(&sample.inputs), input_hash(&old.inputs));
            assert_eq!(sample.current, old.current);
            assert_eq!(sample.next, old.next);
            assert_eq!(sample.policy, old.policy);
            assert_eq!(sample.value, old.value);
            assert_eq!(sample.rewards, old.rewards);
            assert!(episode < 8);
        }
        args.coverage = Coverage::Fresh;
        for episode in 0..16 {
            let mut labels = [0; ACTIONS];
            let mut query = None;
            for offset in 0..16 {
                let (actual, rule, sample) = coverage_sample(&args, episode * 16 + offset)?;
                assert_eq!(actual, episode);
                assert!(task::permutation_ids(Split::Train).contains(&rule));
                let current = query_hash(&sample.current);
                if let Some(ref expected) = query {
                    assert_eq!(expected, &current);
                }
                query = Some(current);
                labels[argmax(&sample.policy)] += 1;
                if episode < 8 {
                    assert_eq!(
                        input_hash(&sample.inputs),
                        input_hash(&fixed[(episode * 16 + offset) as usize].1.inputs)
                    );
                }
            }
            assert_eq!(labels, [4; ACTIONS]);
        }
        Ok(())
    }

    #[test]
    fn fitting_checkpoint_preserves_weights_and_rejects_overwrite() -> Result<()> {
        let root = std::env::temp_dir().join(format!("tofy-fit-save-{}", std::process::id()));
        fs::create_dir(&root)?;
        let result = (|| -> Result<()> {
            let vars = VarMap::new();
            let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
            let weight = vb.get((2, 3), "weight")?;
            let before = weight.to_vec2::<f32>()?;
            let saved = save_fitting_checkpoint(&vars, &root, 25)?;
            let path = root.join(saved["file"].as_str().unwrap());
            let restored = candle_core::safetensors::load(&path, &Device::Cpu)?;
            assert_eq!(before, restored["weight"].to_vec2::<f32>()?);
            assert_eq!(before, weight.to_vec2::<f32>()?);
            assert_eq!(saved["sha256"], file_hash(&path)?);
            assert!(save_fitting_checkpoint(&vars, &root, 25).is_err());
            assert!(!path.with_extension("tmp").exists());
            Ok(())
        })();
        fs::remove_dir_all(&root)?;
        result
    }

    #[test]
    fn fixed_set_preserves_the_blind_information_bound() -> Result<()> {
        let args = Args::parse_from([
            "probe",
            "--mode",
            "smoke",
            "--output-dir",
            "unused",
            "--device",
            "cpu",
            "--hidden",
            "16",
            "--heads",
            "2",
            "--layers",
            "1",
            "--loops",
            "1",
            "--max-loops",
            "1",
            "--profile-eval",
            "false",
        ]);
        let samples = fixed_samples(&args)?;
        assert_eq!(samples.len(), 128);
        let vars = VarMap::new();
        let config = LoopedConfig {
            hidden: 16,
            heads: 2,
            layers: 1,
            max_loops: 1,
        };
        let _model = LoopedAgent::new(
            config.clone(),
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
        )?;
        initialize(&vars, 57)?;
        let report = fitting_check(
            &args,
            &frozen(&vars, &config, &Device::Cpu)?,
            &samples,
            &Device::Cpu,
            Instant::now(),
            0,
        )?;
        assert_eq!(report["arms"][1]["accuracy"], json!(0.25));
        assert!(report["arms"][1]["policy_ce"].as_f64().unwrap() >= 4.0f64.ln() - 0.0001);
        Ok(())
    }

    #[test]
    fn paired_rule_evaluator_obeys_exact_blind_bound() -> Result<()> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("tofy-rule-probe-{}-{unique}", std::process::id()));
        fs::create_dir(&root)?;
        let args = Args::parse_from([
            "probe",
            "--mode",
            "smoke",
            "--output-dir",
            root.to_str().unwrap(),
            "--device",
            "cpu",
            "--hidden",
            "16",
            "--heads",
            "2",
            "--layers",
            "1",
            "--loops",
            "1",
            "--max-loops",
            "1",
            "--profile-eval",
            "false",
        ]);
        let vars = VarMap::new();
        let config = LoopedConfig {
            hidden: 16,
            heads: 2,
            layers: 1,
            max_loops: 1,
        };
        let _model = LoopedAgent::new(
            config.clone(),
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
        )?;
        initialize(&vars, 57)?;
        let result = rule_probe(
            &args,
            &frozen(&vars, &config, &Device::Cpu)?,
            &Device::Cpu,
            Instant::now(),
        );
        fs::remove_dir_all(&root)?;
        let report = result?;
        assert_eq!(report[0]["cleared_accuracy"], json!(0.25));
        assert_eq!(report[1]["cleared_accuracy"], json!(0.25));
        assert_eq!(report[1]["outcome_changing_queries"], json!(8));
        Ok(())
    }

    fn categorical_logits(labels: &[u32], shape: &[usize], classes: usize) -> Result<Tensor> {
        let mut logits = vec![-8.0f32; labels.len() * classes];
        for (i, &label) in labels.iter().enumerate() {
            logits[i * classes + label as usize] = 8.0;
        }
        Ok(Tensor::from_vec(logits, shape, &Device::Cpu)?)
    }

    #[test]
    fn objective_rejects_copy_on_real_changed_successors() -> Result<()> {
        let ep = task::episode(23, 0, Split::Train, 1, 1)?;
        let sample = task::sample(&ep)?;
        let action = sample.policy.iter().position(|&p| p > 0.0).unwrap();
        let mut out = LoopedOutput {
            policy_logits: categorical_logits(&[action as u32], &[1, ACTIONS], ACTIONS)?,
            value: Tensor::full(8.0f32, (1, 1), &Device::Cpu)?,
            reward_logits: Tensor::from_vec(
                sample
                    .rewards
                    .iter()
                    .map(|&r| if r > 0.0 { 8.0f32 } else { -8.0 })
                    .collect(),
                (1, ACTIONS),
                &Device::Cpu,
            )?,
            next_logits: categorical_logits(
                &sample.next,
                &[1, ACTIONS, PATCH_COUNT, PATCH_PIXELS, PALETTE],
                PALETTE,
            )?,
        };
        let (_, correct) = losses(&out, std::slice::from_ref(&sample), &Device::Cpu)?;
        assert!(
            correct.iter().all(|&x| x < 0.001),
            "perfect predictions: {correct:?}"
        );
        out.next_logits = categorical_logits(
            &sample.current.repeat(ACTIONS),
            &[1, ACTIONS, PATCH_COUNT, PATCH_PIXELS, PALETTE],
            PALETTE,
        )?;
        let (_, copy) = losses(&out, &[sample], &Device::Cpu)?;
        assert!(
            copy[3] > 7.0,
            "copy must fail on changed outcomes: {copy:?}"
        );
        Ok(())
    }

    #[test]
    fn named_initialization_and_frozen_predictions_are_reproducible() -> Result<()> {
        let config = LoopedConfig {
            hidden: 16,
            heads: 2,
            layers: 1,
            max_loops: 1,
        };
        let ep = task::episode(23, 0, Split::Train, 1, 1)?;
        let input = task::sample(&ep)?.inputs;
        let mut first = None;
        for _ in 0..2 {
            let vars = VarMap::new();
            let train = LoopedAgent::new(
                config.clone(),
                VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
            )?;
            initialize(&vars, 57)?;
            let expected = predict(&train, std::slice::from_ref(&input), 1, &Device::Cpu)?
                .policy_logits
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = predict(
                &frozen(&vars, &config, &Device::Cpu)?,
                std::slice::from_ref(&input),
                1,
                &Device::Cpu,
            )?
            .policy_logits
            .flatten_all()?
            .to_vec1::<f32>()?;
            assert_eq!(actual, expected);
            if let Some(previous) = &first {
                assert_eq!(&actual, previous);
            } else {
                first = Some(actual);
            }
        }
        Ok(())
    }
}
