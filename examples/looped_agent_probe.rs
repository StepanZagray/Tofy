//! Synthetic prerequisite for the from-scratch looped agent. Never loads ARC games.
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::{Parser, ValueEnum};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tofy::p2::looped_agent::model::{LoopedAgent, LoopedConfig, LoopedOutput};
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
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
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
                    }
                }
                changed_total += changed;
                changed_correct += correct;
                let positive = sample.rewards[a] > 0.5;
                reward_positive += usize::from(positive);
                reward_tp += usize::from(positive && reward[a] >= 0.5);
                reward_fp += usize::from(!positive && reward[a] >= 0.5);
                serde_json::to_writer(
                    &mut rows_log,
                    &json!({"split":split,"episode":id,"permutation_id":ep.permutation_id,"action":a,"frame_exact":same,"copy_exact":real==sample.current,"changed_pixels":changed,"changed_correct":correct,"reward_target":sample.rewards[a],"reward_probability":reward[a],"policy_correct":sample.policy[action]>0.0}),
                )?;
                rows_log.write_all(b"\n")?;
            }
        }
        reports.push(json!({"split":split,"rows":count,"action_tuples":count*ACTIONS,"next_frame_exact":exact,"copy_frame_exact":copy_exact,"changed_pixels":changed_total,"changed_correct":changed_correct,"changed_accuracy":changed_correct as f64/changed_total.max(1) as f64,"policy_optimal_action_accuracy":policy_correct as f64/count as f64,"value_mse":value_squared/count as f64,"constant_value_prediction":0.9,"constant_value_mse":constant_value_squared/count as f64,"reward_positive":reward_positive,"reward_true_positive":reward_tp,"reward_false_positive":reward_fp}));
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
                for input in inputs {
                    actions.push(argmax(
                        &predict(model, &[input], args.loops, device)?
                            .policy_logits
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                    ));
                }
                rows += 1;
                true_correct += usize::from(actions[0] == truth);
                clear_correct += usize::from(actions[1] == truth);
                wrong_correct += usize::from(actions[2] == truth);
                follows_wrong_rule += usize::from(actions[2] == wrong_truth);
                serde_json::to_writer(
                    &mut log,
                    &json!({"split":split,"layout":layout,"rule":rule,"wrong_rule":wrong_id,"correct_action":truth,"wrong_rule_action":wrong_truth,"predictions_true_clear_wrong":actions}),
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
    write_json(
        &args.output_dir.join("metadata.json"),
        &json!({"status":"running","schema":task::SCHEMA,"exact_args":std::env::args().collect::<Vec<_>>(),"provenance":provenance,"model":config,"seed":args.seed,"data_seed":args.data_seed,"physical_batch":args.batch,"accumulation":args.accumulation(),"effective_batch":args.effective_batch,"objectives":{"policy":1.0,"value":0.1,"reward":0.1,"balanced_categorical_dynamics":0.5},"training_rule_ids":task::permutation_ids(Split::Train),"held_out_rule_ids":task::permutation_ids(Split::HeldOut),"boundary":"scripted three-action calibration and visible objective; no active probe selection, hidden objectives, ARC data, or pretrained language weights"}),
    )?;
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
        Mode::Smoke => 2,
        Mode::Train => args.updates,
        Mode::Evaluate => 0,
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
    let mut stream_hash = Sha256::new();
    let train_start = Instant::now();
    for update in 0..updates {
        deadline(args, started)?;
        let update_start = Instant::now();
        let loops = if args.mode == Mode::Smoke {
            args.loops
        } else {
            [1, 2, args.loops][update % 3].min(args.loops)
        };
        let mut grads = None;
        let mut means = [0.0f32; 4];
        for micro in 0..args.accumulation() {
            deadline(args, started)?;
            let micro_batch = args.batch.min(args.effective_batch - micro * args.batch);
            let fraction = micro_batch as f64 / args.effective_batch as f64;
            let mut samples = Vec::with_capacity(micro_batch);
            for b in 0..micro_batch {
                let id = (update * args.effective_batch + micro * args.batch + b) as u64;
                let (min, max) = if id.is_multiple_of(2) {
                    (1, 1)
                } else {
                    (2, 10)
                };
                let ep = task::episode(args.data_seed ^ TRAIN_TAG, id, Split::Train, min, max)?;
                let sample = task::sample(&ep)?;
                stream_hash.update(id.to_le_bytes());
                stream_hash.update((ep.permutation_id as u64).to_le_bytes());
                for pixel in &sample.inputs.patches {
                    stream_hash.update(pixel.to_le_bytes());
                }
                samples.push(sample);
            }
            let inputs = samples.iter().map(|s| s.inputs.clone()).collect::<Vec<_>>();
            let out = predict(&model, &inputs, loops, &device)?;
            let (loss, metrics) = losses(&out, &samples, &device)?;
            for i in 0..4 {
                means[i] += metrics[i] * fraction as f32;
            }
            accumulate_parameter_gradients(&mut grads, (loss * fraction)?.backward()?, &vars)?;
        }
        let mut grads = grads.context("missing gradients")?;
        let clip = clip_gradients_gpu_with_stats(&mut grads, &vars, 1.0)?;
        ensure!(
            clip.pre_clip_norm > 0.0 && clip.pre_clip_norm.is_finite(),
            "invalid gradient norm"
        );
        optimizer.step(&grads)?;
        device.synchronize()?;
        let row = json!({"update":update+1,"loops":loops,"losses":means,"gradient_l2":clip.pre_clip_norm,"clip_scale":clip.scale,"update_seconds":update_start.elapsed().as_secs_f64(),"elapsed_seconds":started.elapsed().as_secs_f64()});
        serde_json::to_writer(&mut log, &row)?;
        log.write_all(b"\n")?;
        log.flush()?;
        if (update + 1) % 16 == 0 || update == 0 {
            println!("{}", row);
        }
    }
    let train_seconds = train_start.elapsed().as_secs_f64();
    drop(log);
    drop(optimizer);
    drop(model);
    vars.save(args.output_dir.join("final.safetensors"))?;
    let model = frozen(&vars, &config, &device)?;
    let predictions = prediction_metrics(args, &model, &device, started)?;
    let rule_probe = rule_probe(args, &model, &device, started)?;
    let episodes = evaluate(args, &model, &device, started)?;
    Ok(
        json!({"status":"complete_pending_analysis","evidence_class":if args.mode==Mode::Smoke {"implementation_smoke"} else {"exploratory"},"claim_boundary":"synthetic calibrated control prerequisite, not ARC performance","parameters":parameters,"optimizer_updates":updates,"physical_batch":args.batch,"accumulation":args.accumulation(),"effective_batch":args.effective_batch,"training_seconds":train_seconds,"elapsed_seconds":started.elapsed().as_secs_f64(),"training_stream_sha256":format!("{:x}",stream_hash.finalize()),"predictions":predictions,"rule_probe":rule_probe,"episodes":episodes,"final_checkpoint_sha256":file_hash(&args.output_dir.join("final.safetensors"))?}),
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

fn main() -> Result<()> {
    let args = Args::parse();
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
        args.mode != Mode::Evaluate || args.checkpoint.is_some(),
        "evaluation requires a checkpoint"
    );
    ensure!(
        args.mode == Mode::Evaluate || args.checkpoint.is_none(),
        "training uses fresh initialization; resume needs optimizer provenance"
    );
    fs::create_dir(&args.output_dir).context("run root must be new, with an existing parent")?;
    write_json(
        &args.output_dir.join("launch.json"),
        &json!({
            "status":"running", "exact_args":std::env::args().collect::<Vec<_>>(),
            "source_revision":env!("TOFY_EMBEDDED_SOURCE_REVISION"),
            "binary_sha256":file_hash(&std::env::current_exe()?)?
        }),
    )?;
    let started = Instant::now();
    let result = run(&args, started);
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
    seal(&args.output_dir)?;
    result.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

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
