use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
use std::time::Instant;
use tofy::p2::data::{
    foundation_v2_stream_schedule, MixedStreamBatch, MixedStreamConfig, OperatorFamilySplit,
};
use tofy::p2::prefetch::MixedStreamBatchPrefetcher;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

const BATCH_ROWS: usize = 1_024;
const TOTAL_STEPS: usize = 24_576;
const DEFAULT_MEASURED_BATCHES: usize = 8;

fn batch_hash(batch: &MixedStreamBatch) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(batch)?)))
}

fn main() -> Result<()> {
    let measured_batches = std::env::var("TOFY_COMPOSE_BENCH_BATCHES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_MEASURED_BATCHES);
    let worker_counts = std::env::var("TOFY_COMPOSE_BENCH_WORKERS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::parse)
                .collect::<std::result::Result<Vec<usize>, _>>()
        })
        .transpose()?
        .unwrap_or_else(|| vec![1, 4, 8]);
    ensure!(measured_batches > 0, "measured batch count must be > 0");
    ensure!(!worker_counts.is_empty(), "worker list must not be empty");
    let config = MixedStreamConfig {
        batch_size: BATCH_ROWS,
        seed: 2,
        schedule: foundation_v2_stream_schedule,
        goal_dropout_probability: 0.3,
        operator_families: OperatorFamilySplit::default(),
        symmetry_augmentation: true,
        data_contract_v6: false,
        synthetic_shards_dir: None,
    };
    println!(
        "compose benchmark: rows={} measured_batches={} rayon_threads={}",
        BATCH_ROWS,
        measured_batches,
        rayon::current_num_threads()
    );

    let mut reference_hashes = None;
    for worker_count in worker_counts {
        let mut prefetcher =
            MixedStreamBatchPrefetcher::new(config.clone(), TOTAL_STEPS, 0, worker_count)?;
        let started = Instant::now();
        let mut hashes = Vec::with_capacity(measured_batches);
        for expected_index in 0..measured_batches as u64 {
            let (batch_index, prepared) = prefetcher.recv_next()?;
            ensure!(batch_index == expected_index, "out-of-order batch");
            let (batch, _host, _) = prepared.into_parts();
            ensure!(batch.samples().len() == BATCH_ROWS, "wrong row count");
            hashes.push(batch_hash(&batch)?);
        }
        let elapsed = started.elapsed();
        let batches_per_second = measured_batches as f64 / elapsed.as_secs_f64();
        if let Some(reference) = &reference_hashes {
            ensure!(&hashes == reference, "worker count changed composed bytes");
        } else {
            reference_hashes = Some(hashes);
        }
        println!(
            "workers={worker_count} elapsed={:.3}s batches/sec={batches_per_second:.3} ms/batch={:.1}",
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() * 1_000.0 / measured_batches as f64,
        );
        prefetcher.shutdown();
    }
    println!("OK: all worker counts produced identical SHA-256 sequences");
    Ok(())
}
