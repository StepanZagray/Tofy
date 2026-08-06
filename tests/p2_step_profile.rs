//! Host-side cost of one P2 training step, and the batch-overlap that motivates
//! the trainer's episode cache.
//!
//! Ignored by default: these are profiling probes, not assertions. Run with
//! `cargo test --release --test p2_step_profile -- --ignored --nocapture --test-threads=1`.
//!
//! The cached/parallel steady-state counterpart lives in the library tests:
//! `cargo test --release --lib episode_cache_steady_state -- --ignored --nocapture`.

use anyhow::Result;
use candle_core::Device;
use std::time::Instant;
use tofy::domain::Split;
use tofy::p2::data::{generate_curriculum, TransitionSample};
use tofy::p2::train::{batch_from_samples, frames_to_one_hot};

const BATCH: usize = 1024;

/// The pre-optimization batch collector: one episode at a time, no reuse across steps.
fn sequential_collect_batch(
    curriculum: &str,
    seed: u64,
    start_episode: u64,
) -> Result<Vec<TransitionSample>> {
    let mut out = Vec::with_capacity(BATCH);
    let mut ep = start_episode;
    while out.len() < BATCH {
        for s in generate_curriculum(curriculum, seed, ep, Split::Train)? {
            out.push(s);
            if out.len() == BATCH {
                break;
            }
        }
        ep = ep.wrapping_add(1);
    }
    Ok(out)
}

#[test]
#[ignore]
fn host_side_step_phases() -> Result<()> {
    let device = Device::Cpu;
    for curriculum in ["random_one_step", "sequential", "p1c_falsification"] {
        let (mut gen_ms, mut onehot_ms, mut stage_ms) = (0.0, 0.0, 0.0);
        let reps = 3;

        for step in 0..reps {
            let t0 = Instant::now();
            let samples = sequential_collect_batch(curriculum, 1, step as u64)?;
            gen_ms += t0.elapsed().as_secs_f64() * 1e3;

            let currents: Vec<_> = samples.iter().map(|s| s.current.clone()).collect();
            let t1 = Instant::now();
            let _ = frames_to_one_hot(&currents, &device)?;
            let _ = frames_to_one_hot(&currents, &device)?; // frames + next_frames
            onehot_ms += t1.elapsed().as_secs_f64() * 1e3;

            let t2 = Instant::now();
            let _ = batch_from_samples(&samples, &device)?;
            stage_ms += t2.elapsed().as_secs_f64() * 1e3;
        }

        println!(
            "{curriculum:<20} sequential_generate={:>8.1}ms  one_hot(x2)={:>7.1}ms  \
             batch_from_samples={:>7.1}ms",
            gen_ms / reps as f64,
            onehot_ms / reps as f64,
            stage_ms / reps as f64,
        );
    }
    Ok(())
}

/// Consecutive steps advance the episode window by one, so nearly every episode in a
/// batch was already built for the previous step. This is what the trainer's episode
/// cache exploits — and it is also why per-step batch diversity is far lower than
/// `physical_batch` suggests.
#[test]
#[ignore]
fn samples_per_episode_and_batch_overlap() -> Result<()> {
    for curriculum in [
        "random_one_step",
        "sequential",
        "p1c_falsification",
        "p1c_hard_retarget",
    ] {
        let per_ep: Vec<usize> = (0..8)
            .map(|ep| {
                generate_curriculum(curriculum, 1, ep, Split::Train)
                    .map(|s| s.len())
                    .unwrap_or(0)
            })
            .collect();
        let mean: f64 = per_ep.iter().sum::<usize>() as f64 / per_ep.len() as f64;
        let episodes_needed = (BATCH as f64 / mean).ceil();
        println!(
            "{curriculum:<20} samples/episode={per_ep:?} mean={mean:.1} \
             episodes_per_step={episodes_needed:.0} \
             overlap_with_next_step={:.1}%",
            (episodes_needed - 1.0) / episodes_needed * 100.0
        );
    }
    Ok(())
}
