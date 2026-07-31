use anyhow::{bail, Context, Result};
use candle_core::{DType, Module, Tensor, D};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::tasks::bridge::{rag_ceiling_for_function, BridgeRuntime};
use crate::tasks::veclab::{
    attach_docs, load_docs_map, load_task_rows, VeclabTaskRow, FUNCTION_COUNT, SEEN_FUNCTION_MAX,
};
use crate::util;

fn conditioning_l2(left: &Tensor, right: &Tensor) -> Result<f32> {
    let delta = left.broadcast_sub(right)?.sqr()?.mean_all()?;
    Ok(util::scalar_f32(&delta)?.sqrt())
}

fn swap_docs_api(docs: &str, from: &str, to: &str) -> String {
    docs.replace(&format!("veclab.{from}"), &format!("veclab.{to}"))
        .replace(from, to)
}

fn corruption_variants(
    row: &VeclabTaskRow,
    alt_function_id: usize,
    docs: &BTreeMap<usize, String>,
) -> Vec<(String, VeclabTaskRow)> {
    let mut out = Vec::new();
    let wrong = VeclabTaskRow {
        function_id: alt_function_id,
        docs: docs
            .get(&alt_function_id)
            .cloned()
            .unwrap_or_else(|| row.docs.clone()),
        ..row.clone()
    };
    out.push(("wrong_function".into(), wrong));
    if let Some(api) = row
        .docs
        .split_whitespace()
        .find_map(|token| token.strip_prefix("veclab."))
        .map(|name| name.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
    {
        if let Some(alt_api) = docs
            .get(&alt_function_id)
            .and_then(|doc| {
                doc.split_whitespace()
                    .find_map(|token| token.strip_prefix("veclab."))
                    .map(|name| name.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            })
            .filter(|name| name != &api)
        {
            out.push((
                "docs_real_swap".into(),
                VeclabTaskRow {
                    docs: swap_docs_api(&row.docs, &api, &alt_api),
                    ..row.clone()
                },
            ));
        }
        out.push((
            "docs_fake_swap".into(),
            VeclabTaskRow {
                docs: swap_docs_api(&row.docs, &api, "NotARealFn"),
                ..row.clone()
            },
        ));
    }
    out
}

fn run_conditioning_corruption_probe(args: &[String]) -> Result<bool> {
    if args.len() < 8 {
        bail!("usage: --run-conditioning-corruption-probe <qwen_dir> <bridge> <encoder> <vocab> <world> <tasks> <report.json>");
    }
    let runtime = BridgeRuntime::load(
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
        Path::new(&args[5]),
        Path::new(&args[6]),
    )?;
    let mut rows = load_task_rows(Path::new(&args[7]))?;
    let docs = load_docs_map(Path::new("data/fictional/veclab_docs.txt"))?;
    attach_docs(&mut rows, &docs);
    let sample_count = std::env::var("TOFY_CORRUPTION_PROBE_ROWS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(60usize);
    let code_rows = rows
        .into_iter()
        .filter(|row| row.completion.trim_start().starts_with("package solution"))
        .collect::<Vec<_>>();
    if code_rows.is_empty() {
        bail!("conditioning corruption probe requires code rows with package solution completions");
    }
    let mut by_function: BTreeMap<usize, VeclabTaskRow> = BTreeMap::new();
    for row in code_rows {
        by_function.entry(row.function_id).or_insert(row);
    }
    let function_ids = by_function.keys().copied().collect::<Vec<_>>();
    let selected = function_ids
        .iter()
        .copied()
        .take(sample_count)
        .filter_map(|function_id| by_function.remove(&function_id))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        bail!("conditioning corruption probe could not select any functions");
    }
    let mut split_metrics: BTreeMap<String, BTreeMap<String, Vec<f32>>> = BTreeMap::new();
    for row in &selected {
        let alt_function_id = function_ids
            .iter()
            .copied()
            .find(|id| *id != row.function_id && *id % 7 != row.function_id % 7)
            .or_else(|| function_ids.iter().copied().find(|id| *id != row.function_id))
            .context("conditioning corruption probe requires an alternate function id")?;
        let matched = runtime.conditioning(std::slice::from_ref(row))?;
        let split = if row.function_id <= SEEN_FUNCTION_MAX {
            "seen"
        } else {
            "heldout"
        };
        for (label, corrupted) in corruption_variants(row, alt_function_id, &docs) {
            let corrupted_cond = runtime.conditioning(std::slice::from_ref(&corrupted))?;
            let distance = conditioning_l2(&matched, &corrupted_cond)?;
            split_metrics
                .entry(split.to_string())
                .or_default()
                .entry(label)
                .or_default()
                .push(distance);
        }
    }
    let mean = |values: &[f32]| {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    };
    let mut report_splits = serde_json::Map::new();
    for (split, variants) in &split_metrics {
        let rag_ceiling = if split == "heldout" {
            rag_ceiling_for_function(SEEN_FUNCTION_MAX + 1)
        } else {
            rag_ceiling_for_function(1)
        };
        let mut variant_report = serde_json::Map::new();
        for (label, distances) in variants {
            let average = mean(distances);
            variant_report.insert(
                label.clone(),
                serde_json::json!({
                    "mean_l2": average,
                    "samples": distances.len(),
                }),
            );
        }
        report_splits.insert(
            split.clone(),
            serde_json::json!({
                "rag_ceiling": rag_ceiling,
                "variants": variant_report,
            }),
        );
    }
    let report_path = PathBuf::from(&args[8]);
    let report = serde_json::json!({
        "schema_version": 1,
        "arm": "conditioning_corruption_probe",
        "bridge_model": args[3],
        "tensor": "decoder_conditioning_post_adapter",
        "output_slots": runtime.output_slots(),
        "functions": selected.len(),
        "splits": report_splits,
    });
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary_report = PathBuf::from(format!("{}.tmp", report_path.to_string_lossy()));
    fs::write(
        &temporary_report,
        format!("{}\n", serde_json::to_string_pretty(&report)?),
    )?;
    fs::rename(temporary_report, &report_path)?;
    for (split, variants) in &split_metrics {
        let rag_ceiling = rag_ceiling_for_function(if split == "heldout" {
            SEEN_FUNCTION_MAX + 1
        } else {
            1
        });
        for (label, distances) in variants {
            println!(
                "conditioning_corruption_probe {split}/{label}: mean_l2={:.4} samples={} rag_ceiling={rag_ceiling:.4}",
                mean(distances),
                distances.len(),
            );
        }
    }
    println!(
        "conditioning_corruption_probe report={}",
        report_path.display()
    );
    Ok(true)
}

fn probe_batch(
    runtime: &BridgeRuntime,
    probe: &candle_nn::Linear,
    rows: &[VeclabTaskRow],
) -> Result<(Tensor, f32)> {
    let slots = runtime.conditioning(rows)?.detach();
    let logits = probe.forward(&slots.mean(D::Minus2)?)?;
    let labels = Tensor::from_vec(
        rows.iter()
            .map(|row| (row.function_id - 1) as u32)
            .collect(),
        (rows.len(),),
        &runtime.device,
    )?;
    let loss = candle_nn::loss::cross_entropy(&logits, &labels)?;
    let predicted = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let correct = predicted
        .iter()
        .zip(rows)
        .filter(|(prediction, row)| **prediction as usize + 1 == row.function_id)
        .count();
    Ok((loss, correct as f32 / rows.len().max(1) as f32))
}

fn evaluate(
    runtime: &BridgeRuntime,
    probe: &candle_nn::Linear,
    rows: &[VeclabTaskRow],
    batch: usize,
) -> Result<f32> {
    let mut weighted = 0f32;
    for chunk in rows.chunks(batch.max(1)) {
        weighted += probe_batch(runtime, probe, chunk)?.1 * chunk.len() as f32;
    }
    Ok(weighted / rows.len().max(1) as f32)
}

pub fn try_run(args: &[String]) -> Result<bool> {
    if matches!(
        args.get(1).map(String::as_str),
        Some("--run-conditioning-corruption-probe" | "run-conditioning-corruption-probe")
    ) {
        return run_conditioning_corruption_probe(args);
    }
    if !matches!(
        args.get(1).map(String::as_str),
        Some("--train-channel-probe" | "train-channel-probe")
    ) {
        return Ok(false);
    }
    if args.len() < 10 {
        bail!("usage: --train-channel-probe <qwen_dir> <bridge> <encoder> <vocab> <world> <seen_tasks> <heldout_tasks> <output> [steps]");
    }
    let runtime = BridgeRuntime::load(
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
        Path::new(&args[5]),
        Path::new(&args[6]),
    )?;
    let mut rows = load_task_rows(Path::new(&args[7]))?;
    rows.extend(load_task_rows(Path::new(&args[8]))?);
    let docs = load_docs_map(Path::new("data/fictional/veclab_docs.txt")).unwrap_or_default();
    attach_docs(&mut rows, &docs);
    let mut per_function = HashMap::new();
    let (validation, training): (Vec<_>, Vec<_>) = rows.into_iter().partition(|row| {
        let index = per_function.entry(row.function_id).or_insert(0usize);
        let is_validation = *index % 5 == 0;
        *index += 1;
        is_validation
    });
    let train_ids = training
        .iter()
        .map(|row| row.function_id)
        .collect::<HashSet<_>>();
    let val_ids = validation
        .iter()
        .map(|row| row.function_id)
        .collect::<HashSet<_>>();
    if train_ids.len() != FUNCTION_COUNT || val_ids.len() != FUNCTION_COUNT {
        bail!(
            "channel probe requires train and validation examples for all {FUNCTION_COUNT} functions (got train={} val={})",
            train_ids.len(),
            val_ids.len()
        );
    }
    let output = PathBuf::from(&args[9]);
    let steps = args
        .get(10)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000usize);
    let batch = std::env::var("TOFY_PROBE_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32usize);
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &runtime.device);
    let slot_dim = runtime.conditioning(&training[..1])?.dim(2)?;
    let probe = candle_nn::linear(slot_dim, FUNCTION_COUNT, vb.pp("channel_probe"))?;
    let mut optimizer = candle_nn::AdamW::new_lr(vars.all_vars(), 1e-3)?;
    for step in 0..steps {
        let start = (step * batch) % training.len();
        let rows = (0..batch.min(training.len()))
            .map(|offset| training[(start + offset) % training.len()].clone())
            .collect::<Vec<_>>();
        let (loss, _) = probe_batch(&runtime, &probe, &rows)?;
        optimizer.backward_step(&loss)?;
    }
    let seen_validation = validation
        .iter()
        .filter(|row| row.function_id <= crate::tasks::veclab::SEEN_FUNCTION_MAX)
        .cloned()
        .collect::<Vec<_>>();
    let heldout_validation = validation
        .iter()
        .filter(|row| row.function_id > crate::tasks::veclab::SEEN_FUNCTION_MAX)
        .cloned()
        .collect::<Vec<_>>();
    let seen_accuracy = evaluate(&runtime, &probe, &seen_validation, batch)?;
    let heldout_accuracy = evaluate(&runtime, &probe, &heldout_validation, batch)?;
    util::save_varmap_atomic(&vars, &output)?;
    let report_path = output.with_extension("json");
    let report = serde_json::json!({
        "schema_version": 1,
        "arm": "channel_probe",
        "bridge_model": args[3],
        "steps": steps,
        "batch": batch,
        "seen_validation_tasks": seen_validation.len(),
        "heldout_validation_tasks": heldout_validation.len(),
        "seen_accuracy": seen_accuracy,
        "heldout_accuracy": heldout_accuracy,
    });
    let temporary_report = PathBuf::from(format!("{}.tmp", report_path.to_string_lossy()));
    fs::write(
        &temporary_report,
        format!("{}\n", serde_json::to_string_pretty(&report)?),
    )?;
    fs::rename(temporary_report, &report_path)?;
    println!(
        "channel_probe held-out-paraphrase seen_accuracy={seen_accuracy:.4} heldout_accuracy={heldout_accuracy:.4} report={}",
        report_path.display()
    );
    Ok(true)
}
