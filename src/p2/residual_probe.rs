//! `p2-residual-probe` — ADR 0005 §5.3 residual-vs-reliability probe.
//!
//! Frozen-checkpoint rescoring only. For every transition in (a) a directory of
//! official-toolkit ARC-AGI-3 recordings (EVALUATION ONLY; nothing here trains)
//! and (b) a held-out in-distribution synthetic population drawn through the
//! same `compose_mixed_stream_batch` path the offline evaluator uses, the model
//! is run with the conditioning the live policy sends — all-zero goal features
//! and the UNKNOWN operator token — and the following are recorded per row:
//! composed-decode residual against the actual next frame (changed-pixel count
//! and mean per-pixel cross-entropy), `sigmoid(reliability_logit)`,
//! `sigmoid(q_logit)`, whether the actual transition was a gameplay no-op, and
//! the source label. Output is JSONL plus a summary JSON; the AUROC analysis
//! lives in `scripts/p2_residual_probe_analyze.py`.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops;
use clap::Args;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;

use crate::p2::arc3::import_recordings_dir;
use crate::p2::data::{
    compose_mixed_stream_batch, foundation_v2_stream_schedule, MixedStreamConfig, TransitionSample,
    V5DataSplit, FRAME_SIDE, GOAL_FEATURES_DIM,
};
use crate::p2::model::{unknown_operator_conditioning, WorldModel, PALETTE_SIZE};
use crate::p2::train::{
    batch_from_samples, load_train_config, load_varmap_exact, resolve_device,
    FOUNDATION_V2_GATE_SEED,
};
use candle_nn::{VarBuilder, VarMap};

/// Same eval-domain constant as `eval::foundation_v2_v5_holdout_gates` so the
/// synthetic comparator is the evaluator's `unseen_seed_7x7` population.
const V5_HOLDOUT_SEED_DOMAIN: u64 = 0x5645_4C35_1D00_0000;
const GAMEPLAY_LEN: usize = (FRAME_SIDE - 1) * FRAME_SIDE;

#[derive(Debug, Clone, Args)]
pub struct P2ResidualProbeArgs {
    #[arg(long)]
    pub checkpoint: PathBuf,
    #[arg(long)]
    pub train_config: PathBuf,
    /// Directory of official-toolkit JSONL recordings (real frames, eval only).
    #[arg(long)]
    pub arc_recordings_dir: PathBuf,
    /// Synthetic comparator size. Defaults to the number of real transitions.
    #[arg(long)]
    pub synthetic_rows: Option<usize>,
    /// Evaluator iid seed; the population seed is `iid_seed + V5_HOLDOUT_SEED_DOMAIN`.
    #[arg(long, default_value_t = 3)]
    pub iid_seed: u64,
    #[arg(long, default_value_t = 64)]
    pub physical_batch: usize,
    #[arg(long, default_value = "cuda")]
    pub device: String,
    /// `live`: all-zero goal + UNKNOWN operator (what the live policy sends).
    /// `eval`: each row's own goal features + operator conditioning, as the
    /// offline evaluator uses (control arm; identical for ARC rows).
    #[arg(long, default_value = "live")]
    pub conditioning: String,
    #[arg(long)]
    pub output_jsonl: PathBuf,
    #[arg(long)]
    pub output_summary: PathBuf,
}

#[derive(Debug, Serialize)]
struct ProbeRow {
    source: &'static str,
    game_id: String,
    family: String,
    stream: Option<String>,
    operator_family: Option<String>,
    /// Synthetic rows: whether the generator's goal dropout zeroed the goal
    /// before the probe zeroed every goal. Real rows: always true (no goal).
    goal_zero_at_source: bool,
    action_id: u8,
    actual_noop: bool,
    /// Generator label where available (`None` for ARC recordings).
    generator_noop: Option<bool>,
    actual_changed_pixels: usize,
    residual_changed_pixels: usize,
    residual_pixel_ce: f64,
    decoder_pixel_ce: f64,
    composed_exact: bool,
    composed_changed_exact: Option<bool>,
    copy_gate_mean: f64,
    reliability: f64,
    q: f64,
}

#[derive(Debug, Default, Serialize)]
struct SourceSummary {
    rows: usize,
    noop_rows: usize,
    composed_exact_rows: usize,
    reliability_mean: f64,
    reliability_median: f64,
    residual_pixel_ce_mean: f64,
    residual_changed_pixels_mean: f64,
}

#[derive(Debug, Serialize)]
struct ProbeSummary {
    checkpoint: PathBuf,
    checkpoint_sha256: String,
    train_config: PathBuf,
    device: String,
    conditioning: String,
    /// True when the checkpoint predates `operator_conditioning_proj` (added
    /// 2026-08-28, commit 36fe9e96) and that additive projection was left at
    /// its zero initialization, which is numerically identical to the trained
    /// topology (the projection did not exist, so it contributed nothing).
    legacy_operator_projection_zeroed: bool,
    synthetic_population: BTreeMap<String, serde_json::Value>,
    real: SourceSummary,
    synthetic: SourceSummary,
    real_rows_per_game: BTreeMap<String, usize>,
}

pub fn run_p2_residual_probe(args: P2ResidualProbeArgs) -> Result<()> {
    let live_conditioning = match args.conditioning.as_str() {
        "live" => true,
        "eval" => false,
        other => bail!("--conditioning must be live or eval, got {other:?}"),
    };
    let train_cfg = load_train_config(&args.train_config)?;
    let device = resolve_device(&args.device)?;
    let (model, _varmap, legacy_operator_projection_zeroed) =
        load_model_legacy_operator_compat(&train_cfg, &args.checkpoint, &device)?;

    // (a) real frames: toolkit recordings -> TransitionSample (goal already zero).
    let real = import_recordings_dir(&args.arc_recordings_dir)?;
    if real.is_empty() {
        bail!(
            "no transitions imported from {}",
            args.arc_recordings_dir.display()
        );
    }

    // (b) synthetic comparator: the evaluator's unseen-seed in-distribution population.
    let synthetic_rows = args.synthetic_rows.unwrap_or(real.len());
    let population_seed = args.iid_seed.wrapping_add(V5_HOLDOUT_SEED_DOMAIN);
    if population_seed == FOUNDATION_V2_GATE_SEED || population_seed == train_cfg.seed {
        bail!("synthetic comparator seed collides with a training/gate seed");
    }
    let batch = compose_mixed_stream_batch(
        &MixedStreamConfig {
            batch_size: synthetic_rows,
            seed: population_seed,
            schedule: foundation_v2_stream_schedule,
            ..MixedStreamConfig::default()
        },
        1.0,
        0,
        V5DataSplit::UnseenSeed7x7,
    )?;
    let mut synthetic: Vec<(TransitionSample, Option<String>, Option<String>, bool)> = batch
        .samples()
        .iter()
        .map(|s| {
            (
                s.transition.clone(),
                Some(format!("{:?}", s.provenance.stream)),
                Some(format!("{:?}", s.provenance.operator.family)),
                s.provenance.goal_dropped,
            )
        })
        .collect();
    let goal_dropped_rows = synthetic.iter().filter(|s| s.3).count();
    // Deterministic shuffle so no physical chunk is an all-factual partial group
    // (batch_from_samples would otherwise try to rebuild complete branch groups).
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(population_seed ^ 0x0050_524f_4245);
    synthetic.shuffle(&mut rng);

    let mut out = fs::File::create(&args.output_jsonl)
        .with_context(|| format!("create {}", args.output_jsonl.display()))?;
    let mut real_rows = Vec::new();
    let mut synth_rows = Vec::new();

    let real_meta: Vec<(TransitionSample, Option<String>, Option<String>, bool)> =
        real.into_iter().map(|s| (s, None, None, true)).collect();
    for (source, rows, sink) in [
        ("real", &real_meta, &mut real_rows),
        ("synthetic", &synthetic, &mut synth_rows),
    ] {
        for chunk in rows.chunks(args.physical_batch) {
            let samples: Vec<TransitionSample> = chunk.iter().map(|c| c.0.clone()).collect();
            let probe = probe_batch(&model, &samples, &device, live_conditioning)?;
            for (i, (sample, stream, operator, goal_zero)) in chunk.iter().enumerate() {
                let row = ProbeRow {
                    source,
                    game_id: sample
                        .family
                        .strip_prefix("arc3:")
                        .unwrap_or(&sample.family)
                        .to_string(),
                    family: sample.family.clone(),
                    stream: stream.clone(),
                    operator_family: operator.clone(),
                    goal_zero_at_source: *goal_zero
                        || sample.goal_features.values.iter().all(|v| *v == 0.0),
                    action_id: sample.action.id,
                    actual_noop: probe.actual_changed[i] == 0,
                    generator_noop: sample.noop,
                    actual_changed_pixels: probe.actual_changed[i],
                    residual_changed_pixels: probe.residual_changed[i],
                    residual_pixel_ce: probe.composed_ce[i],
                    decoder_pixel_ce: probe.decoder_ce[i],
                    composed_exact: probe.residual_changed[i] == 0,
                    composed_changed_exact: probe.changed_exact[i],
                    copy_gate_mean: probe.gate_mean[i],
                    reliability: probe.reliability[i],
                    q: probe.q[i],
                };
                serde_json::to_writer(&mut out, &row)?;
                out.write_all(b"\n")?;
                sink.push(row);
            }
        }
    }
    out.flush()?;

    let mut per_game = BTreeMap::new();
    for row in &real_rows {
        *per_game.entry(row.game_id.clone()).or_insert(0usize) += 1;
    }
    let mut synthetic_population = BTreeMap::new();
    synthetic_population.insert("split".into(), serde_json::json!("UnseenSeed7x7"));
    synthetic_population.insert("seed".into(), serde_json::json!(population_seed));
    synthetic_population.insert("rows".into(), serde_json::json!(synthetic.len()));
    synthetic_population.insert(
        "goal_dropped_rows_at_source".into(),
        serde_json::json!(goal_dropped_rows),
    );
    synthetic_population.insert(
        "schedule".into(),
        serde_json::json!("foundation_v2_stream_schedule(progress=1.0)"),
    );
    let mut streams = BTreeMap::new();
    for row in &synth_rows {
        *streams
            .entry(row.stream.clone().unwrap_or_default())
            .or_insert(0usize) += 1;
    }
    synthetic_population.insert("stream_counts".into(), serde_json::json!(streams));

    let summary = ProbeSummary {
        checkpoint_sha256: sha256_file(&args.checkpoint)?,
        checkpoint: args.checkpoint.clone(),
        train_config: args.train_config.clone(),
        device: args.device.clone(),
        conditioning: if live_conditioning {
            "live: goal=all-zero(19), operator=UNKNOWN token, no context, eval recursion depth"
                .into()
        } else {
            "eval: row goal features + row operator conditioning, no context, eval recursion depth"
                .into()
        },
        legacy_operator_projection_zeroed,
        synthetic_population,
        real: summarize(&real_rows),
        synthetic: summarize(&synth_rows),
        real_rows_per_game: per_game,
    };
    fs::write(
        &args.output_summary,
        serde_json::to_string_pretty(&summary)?,
    )?;
    println!(
        "p2-residual-probe complete real={} synthetic={} real_rel_median={:.3e} synth_rel_median={:.3e}",
        summary.real.rows,
        summary.synthetic.rows,
        summary.real.reliability_median,
        summary.synthetic.reliability_median
    );
    Ok(())
}

/// `eval::load_model`, except that a checkpoint written before the operator
/// conditioning projection existed may omit exactly `operator_conditioning_proj.*`;
/// those two tensors stay at their zero initialization (an additive no-op).
/// Any other name/shape/dtype mismatch still fails closed.
fn load_model_legacy_operator_compat(
    train_cfg: &crate::p2::train::TrainConfig,
    weights: &std::path::Path,
    device: &Device,
) -> Result<(WorldModel, VarMap, bool)> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = WorldModel::new(train_cfg.model_config(), vb)?;
    match load_varmap_exact(&varmap, weights) {
        Ok(()) => Ok((model, varmap, false)),
        Err(error) => {
            let mmap = unsafe { candle_core::safetensors::MmapedSafetensors::new(weights)? };
            let checkpoint_names: std::collections::BTreeSet<String> =
                mmap.tensors().into_iter().map(|(name, _)| name).collect();
            let expected: Vec<(String, candle_core::Var)> = {
                let data = varmap.data().lock().unwrap();
                data.iter().map(|(n, v)| (n.clone(), v.clone())).collect()
            };
            let expected_names: std::collections::BTreeSet<String> =
                expected.iter().map(|(n, _)| n.clone()).collect();
            let missing: Vec<&String> = expected_names.difference(&checkpoint_names).collect();
            let extra: Vec<&String> = checkpoint_names.difference(&expected_names).collect();
            let only_operator_proj = !missing.is_empty()
                && extra.is_empty()
                && missing
                    .iter()
                    .all(|name| name.starts_with("operator_conditioning_proj."));
            if !only_operator_proj {
                return Err(error);
            }
            eprintln!(
                "warning: checkpoint {} predates operator_conditioning_proj; leaving {:?} at zero init (additive no-op)",
                weights.display(),
                missing
            );
            for (name, var) in expected {
                if name.starts_with("operator_conditioning_proj.") {
                    continue;
                }
                let tensor = mmap
                    .load(&name, device)
                    .with_context(|| format!("load model tensor {name}"))?;
                if tensor.dims() != var.shape().dims() || tensor.dtype() != var.dtype() {
                    bail!("model checkpoint shape/dtype mismatch for {name}");
                }
                var.set(&tensor)?;
            }
            Ok((model, varmap, true))
        }
    }
}

struct BatchProbe {
    actual_changed: Vec<usize>,
    residual_changed: Vec<usize>,
    changed_exact: Vec<Option<bool>>,
    composed_ce: Vec<f64>,
    decoder_ce: Vec<f64>,
    gate_mean: Vec<f64>,
    reliability: Vec<f64>,
    q: Vec<f64>,
}

/// One frozen forward pass with live-policy conditioning, then residuals.
fn probe_batch(
    model: &WorldModel,
    samples: &[TransitionSample],
    device: &Device,
    live_conditioning: bool,
) -> Result<BatchProbe> {
    let n = samples.len();
    let batch = batch_from_samples(samples, device)?;
    let (goals, operator) = if live_conditioning {
        (
            Tensor::zeros((n, GOAL_FEATURES_DIM), DType::F32, device)?,
            unknown_operator_conditioning(n, device)?,
        )
    } else {
        (batch.goals.clone(), batch.operator_conditioning.clone())
    };
    let current = model.encode_state(&batch.frames)?;
    let output = model.forward_from_latent_with_operator_conditioning(
        &current,
        &batch.actions,
        &batch.action_coords,
        &goals,
        &operator,
    )?;
    let y = output.y.detach();

    let composed = model
        .composed_gameplay_decode(&y, &batch.frames)?
        .reshape((n, GAMEPLAY_LEN))?
        .to_dtype(DType::U8)?
        .to_vec2::<u8>()?;
    // Probabilistic residual: mixture p = (1-g)*onehot(current) + g*softmax(logits),
    // scored against the actual next pixel (decoder-only CE reported alongside).
    let logits = model.exact_gameplay_logits(&y)?.detach(); // B×63×64×16
    let log_probs = ops::log_softmax(&logits, D::Minus1)?;
    let gate = model.exact_copy_gate(&y)?.detach(); // B×63×64
    let next_u32 = batch
        .next_frames
        .narrow(2, 0, FRAME_SIDE - 1)?
        .squeeze(1)?
        .to_dtype(DType::U32)?;
    let cur_u32 = batch
        .frames
        .narrow(2, 0, FRAME_SIDE - 1)?
        .squeeze(1)?
        .to_dtype(DType::U32)?;
    let lp_next = log_probs
        .gather(&next_u32.unsqueeze(D::Minus1)?, D::Minus1)?
        .squeeze(D::Minus1)?; // B×63×64
    let p_next = lp_next.exp()?;
    let same = cur_u32.eq(&next_u32)?.to_dtype(DType::F32)?;
    let mix = gate
        .affine(-1.0, 1.0)?
        .mul(&same)?
        .add(&gate.mul(&p_next)?)?
        .clamp(1e-9f64, 1.0f64)?;
    let composed_ce = mix
        .log()?
        .neg()?
        .reshape((n, GAMEPLAY_LEN))?
        .mean(D::Minus1)?
        .to_vec1::<f32>()?;
    let decoder_ce = lp_next
        .neg()?
        .reshape((n, GAMEPLAY_LEN))?
        .mean(D::Minus1)?
        .to_vec1::<f32>()?;
    let gate_mean = gate
        .reshape((n, GAMEPLAY_LEN))?
        .mean(D::Minus1)?
        .to_vec1::<f32>()?;
    let reliability = ops::sigmoid(&output.reliability_logit.detach())?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let q = ops::sigmoid(&output.q_logit.detach())?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if reliability.len() != n || q.len() != n {
        bail!(
            "head output rows {} / {} do not match batch {n}",
            reliability.len(),
            q.len()
        );
    }
    debug_assert_eq!(PALETTE_SIZE, logits.dim(3)?);

    let mut actual_changed = Vec::with_capacity(n);
    let mut residual_changed = Vec::with_capacity(n);
    let mut changed_exact = Vec::with_capacity(n);
    for (sample, prediction) in samples.iter().zip(&composed) {
        let cur = &sample.current.pixels[..GAMEPLAY_LEN.min(sample.current.pixels.len())];
        let nxt = &sample.next.pixels[..GAMEPLAY_LEN.min(sample.next.pixels.len())];
        if cur.len() != GAMEPLAY_LEN
            || nxt.len() != GAMEPLAY_LEN
            || prediction.len() != GAMEPLAY_LEN
        {
            bail!("gameplay pixel width mismatch");
        }
        let mut changed = 0usize;
        let mut residual = 0usize;
        let mut exact_on_changed = true;
        for ((c, a), p) in cur.iter().zip(nxt).zip(prediction) {
            if p != a {
                residual += 1;
            }
            if c != a {
                changed += 1;
                exact_on_changed &= p == a;
            }
        }
        actual_changed.push(changed);
        residual_changed.push(residual);
        changed_exact.push((changed > 0).then_some(exact_on_changed));
    }
    Ok(BatchProbe {
        actual_changed,
        residual_changed,
        changed_exact,
        composed_ce: composed_ce.iter().map(|v| f64::from(*v)).collect(),
        decoder_ce: decoder_ce.iter().map(|v| f64::from(*v)).collect(),
        gate_mean: gate_mean.iter().map(|v| f64::from(*v)).collect(),
        reliability: reliability.iter().map(|v| f64::from(*v)).collect(),
        q: q.iter().map(|v| f64::from(*v)).collect(),
    })
}

fn summarize(rows: &[ProbeRow]) -> SourceSummary {
    if rows.is_empty() {
        return SourceSummary::default();
    }
    let n = rows.len() as f64;
    let mut rel: Vec<f64> = rows.iter().map(|r| r.reliability).collect();
    rel.sort_by(|a, b| a.total_cmp(b));
    SourceSummary {
        rows: rows.len(),
        noop_rows: rows.iter().filter(|r| r.actual_noop).count(),
        composed_exact_rows: rows.iter().filter(|r| r.composed_exact).count(),
        reliability_mean: rel.iter().sum::<f64>() / n,
        reliability_median: rel[rel.len() / 2],
        residual_pixel_ce_mean: rows.iter().map(|r| r.residual_pixel_ce).sum::<f64>() / n,
        residual_changed_pixels_mean: rows
            .iter()
            .map(|r| r.residual_changed_pixels as f64)
            .sum::<f64>()
            / n,
    }
}

fn sha256_file(path: &std::path::Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
