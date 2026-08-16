//! Focused CUDA stress for the device-resident representation row collector.
//!
//! Run with:
//! `TOFY_CUDA_COLLECTOR_STRESS_ITERS=50 cargo test --release --locked --features cudnn \
//!   --test cuda_representation_collector -- --ignored --nocapture`

#![cfg(feature = "cuda")]

use anyhow::Result;
use candle_core::{Device, Tensor};
use tofy::p2::representation::{RepresentationRowCollector, RepresentationSeamMetrics};

const ROWS_PER_BATCH: usize = 32_768;
const DIMENSION: usize = 32;
const ROW_CAP: usize = 8_192;
const BATCHES_PER_ITERATION: usize = 8;

fn collect_rows(device: &Device) -> Result<RepresentationSeamMetrics> {
    let mut collector = RepresentationRowCollector::new(424250, 7, ROW_CAP);
    for batch in 0..BATCHES_PER_ITERATION {
        let row_start = batch * ROWS_PER_BATCH;
        let rows = Tensor::arange(
            row_start as f32,
            (row_start + ROWS_PER_BATCH) as f32,
            device,
        )?
        .reshape((ROWS_PER_BATCH, 1))?
        .broadcast_as((ROWS_PER_BATCH, DIMENSION))?
        .contiguous()?;
        collector.collect_rows(
            &rows,
            (row_start..row_start + ROWS_PER_BATCH)
                .map(|row| row as u64)
                .collect(),
        )?;

        // Force short-lived allocations onto the same stream between collector updates.
        // This makes premature source/index lifetime reuse much easier to observe.
        for _ in 0..8 {
            let churn = Tensor::from_vec(vec![u32::MAX; ROW_CAP], (ROW_CAP,), device)?;
            drop(churn);
        }
    }
    collector.summarize()
}

#[test]
#[ignore]
fn repeated_cuda_collector_keeps_selected_indices_in_bounds() -> Result<()> {
    let device = Device::new_cuda(0)
        .map_err(|error| anyhow::anyhow!("required CUDA device 0 did not open: {error}"))?;
    let iterations = std::env::var("TOFY_CUDA_COLLECTOR_STRESS_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);
    let expected = collect_rows(&Device::Cpu)?;

    for iteration in 0..iterations {
        let summary = collect_rows(&device)?;
        assert_eq!(summary.rows_seen, ROWS_PER_BATCH * BATCHES_PER_ITERATION);
        assert_eq!(summary.rows_used, ROW_CAP);
        assert_eq!(summary.non_finite_rows, 0);
        assert_eq!(summary.mean_rms, expected.mean_rms);
        assert_eq!(summary.mean_variance, expected.mean_variance);
        assert_eq!(summary.effective_rank, expected.effective_rank);
        device.synchronize()?;
        if iteration % 25 == 0 || iteration + 1 == iterations {
            println!("collector CUDA stress iteration {iteration} passed");
        }
    }
    Ok(())
}
