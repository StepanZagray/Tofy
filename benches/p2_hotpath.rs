//! CPU-side P2 hot-path microbenchmarks via the crate's public API.
//!
//! These isolate curriculum generation and the frame/batch tensor packing that
//! feeds training, without touching CUDA. Criterion sample sizes stay low
//! because a single 1024-frame pack is already expensive.

use candle_core::Device;
use criterion::{criterion_group, criterion_main, Criterion};
use tofy::domain::Split;
use tofy::p2::data::{generate_curriculum, ArcFrame, TransitionSample};
use tofy::p2::train::{batch_from_samples, frames_to_one_hot};

const BATCH: usize = 1024;

/// Collect at least `n` transition samples by walking curriculum episodes.
fn collect_samples(kind: &str, n: usize) -> Vec<TransitionSample> {
    let mut out = Vec::with_capacity(n);
    let mut episode = 0u64;
    while out.len() < n {
        let mut batch = generate_curriculum(kind, 42, episode, Split::Train)
            .unwrap_or_else(|e| panic!("generate_curriculum({kind}): {e}"));
        out.append(&mut batch);
        episode += 1;
    }
    out.truncate(n);
    out
}

fn bench_generate_curriculum(c: &mut Criterion) {
    let mut group = c.benchmark_group("generate_curriculum");
    // Curriculum generation is scenario+render heavy; keep wall time bounded.
    group.sample_size(20);

    for kind in ["random_one_step", "p1c_falsification"] {
        group.bench_function(kind, |b| {
            let mut episode = 0u64;
            b.iter(|| {
                let samples = generate_curriculum(kind, 7, episode, Split::Train)
                    .expect("generate_curriculum");
                episode = episode.wrapping_add(1);
                std::hint::black_box(samples)
            });
        });
    }
    group.finish();
}

fn bench_frames_to_one_hot(c: &mut Criterion) {
    let samples = collect_samples("random_one_step", BATCH);
    let frames: Vec<ArcFrame> = samples.iter().map(|s| s.current.clone()).collect();
    let device = Device::Cpu;

    let mut group = c.benchmark_group("frames_to_one_hot");
    group.sample_size(10);
    group.bench_function("batch_1024_cpu", |b| {
        b.iter(|| {
            let tensor = frames_to_one_hot(&frames, &device).expect("frames_to_one_hot");
            std::hint::black_box(tensor)
        });
    });
    group.finish();
}

fn bench_batch_from_samples(c: &mut Criterion) {
    let samples = collect_samples("random_one_step", BATCH);
    let device = Device::Cpu;

    let mut group = c.benchmark_group("batch_from_samples");
    group.sample_size(10);
    group.bench_function("batch_1024_cpu", |b| {
        b.iter(|| {
            let batch = batch_from_samples(&samples, &device).expect("batch_from_samples");
            std::hint::black_box(batch)
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_generate_curriculum,
    bench_frames_to_one_hot,
    bench_batch_from_samples
);
criterion_main!(benches);
