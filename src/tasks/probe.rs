use anyhow::{bail, Result};
use candle_core::{DType, Module, Tensor, D};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use std::path::{Path, PathBuf};

use crate::tasks::bridge::BridgeRuntime;
use crate::tasks::veclab::{
    attach_docs, load_docs_map, load_task_rows, VeclabTaskRow, FUNCTION_COUNT,
};
use crate::util;

fn probe_batch(
    runtime: &BridgeRuntime,
    probe: &candle_nn::Linear,
    rows: &[VeclabTaskRow],
) -> Result<(Tensor, f32)> {
    let slots = runtime.conditioning(rows)?;
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
    let mut seen = load_task_rows(Path::new(&args[7]))?;
    let mut heldout = load_task_rows(Path::new(&args[8]))?;
    let docs = load_docs_map(Path::new("data/fictional/veclab_docs.txt")).unwrap_or_default();
    attach_docs(&mut seen, &docs);
    attach_docs(&mut heldout, &docs);
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
    let slot_dim = runtime.conditioning(&seen[..1])?.dim(2)?;
    let probe = candle_nn::linear(slot_dim, FUNCTION_COUNT, vb.pp("channel_probe"))?;
    let mut optimizer = candle_nn::AdamW::new_lr(vars.all_vars(), 1e-3)?;
    for step in 0..steps {
        let start = (step * batch) % seen.len();
        let rows = (0..batch.min(seen.len()))
            .map(|offset| seen[(start + offset) % seen.len()].clone())
            .collect::<Vec<_>>();
        let (loss, _) = probe_batch(&runtime, &probe, &rows)?;
        optimizer.backward_step(&loss)?;
    }
    let seen_accuracy = evaluate(&runtime, &probe, &seen, batch)?;
    let heldout_accuracy = evaluate(&runtime, &probe, &heldout, batch)?;
    util::save_varmap_atomic(&vars, &output)?;
    println!(
        "channel_probe seen_accuracy={seen_accuracy:.4} heldout_accuracy={heldout_accuracy:.4}"
    );
    Ok(true)
}
