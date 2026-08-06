//! Performance tooling: Chrome Trace timelines, P2 step-phase timing, and
//! external CPU+GPU profiling via NVIDIA Nsight Systems.
//!
//! ## `TOFY_PERF_TRACE` (Chrome / Perfetto)
//!
//! When the `profiling` feature is enabled and `TOFY_PERF_TRACE` points at an
//! output path, [`install`] registers a `tracing-chrome` layer that writes
//! nested span events as Chrome Trace Event JSON. Open that JSON in
//! <https://ui.perfetto.dev> or `chrome://tracing` to inspect the timeline.
//!
//! Produce a trace (call [`install`] once at process start, then run any
//! binary that emits `tracing` spans):
//!
//! ```bash
//! TOFY_PERF_TRACE=/tmp/tofy-perf.json cargo run --release --features profiling -- p2-train --help
//! ```
//!
//! Without the feature (or when the env var is unset) install is a no-op so
//! normal builds stay unaffected. Spans themselves live behind `tracing` macros
//! that compile to no-ops when no subscriber is installed. This path is
//! host-side only; it does not record CUDA kernels or memcpy.
//!
//! ## `TOFY_P2_STEP_PROFILE` (phase ms print)
//!
//! Set to a report interval (e.g. `100`) in the P2 train loop to print per-step
//! millisecond averages for `generate`, `stage`, `forward`, `backward`,
//! `optimizer`, `metrics`, and `checkpoint`. Each boundary syncs the device
//! before timing so numbers reflect real GPU work (opt-in because syncs cost
//! throughput). See `StepProfile` in `p2::train`.
//!
//! ## Nsight Systems (`nsys`) — CPU + GPU
//!
//! For unified CPU and CUDA timelines, wrap a release train binary with NVIDIA
//! Nsight Systems. Prefer this when optimizing GPU utilization, launch/sync
//! stalls, or cuDNN/cuBLAS vs host work. Day-to-day phase timings can stay on
//! `TOFY_P2_STEP_PROFILE` / `TOFY_PERF_TRACE`.
//!
//! ```bash
//! nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --sample=cpu \
//!   --output=tofy-train \
//!   cargo run --release --features cudnn -- p2-train --device cuda ...
//! ```
//!
//! Open the `.nsys-rep` in the Nsight Systems GUI. Use Nsight Compute (`ncu`)
//! only after `nsys` identifies a hot kernel.
//!
//! ## Host RSS (`alloc`)
//!
//! Long Rayon episode generation on glibc can retain multi-giB anonymous RSS from
//! arena fragmentation even when live allocations are small. At process start,
//! [`crate::alloc::init`] caps glibc arenas when `MALLOC_ARENA_MAX` is unset.
//! During P2 training, `TOFY_MALLOC_TRIM_EVERY` (default `100`) calls `malloc_trim`.
//! Shell scripts may also export `MALLOC_ARENA_MAX=2`. For overnight runs:
//! `cargo build --release --features jemalloc` uses jemalloc as the global allocator.
//!
//! Prefer `--features cudnn` only when the vendored candle patch makes it faster
//! than im2col; stock candle cudnn can regress conv backward. See
//! [`vendor/candle-core/TOFY_PATCH.md`](../vendor/candle-core/TOFY_PATCH.md).

use anyhow::Result;

/// RAII handle that keeps the Chrome Trace writer alive until drop.
///
/// Dropping the guard flushes and closes the JSON file so Perfetto/chrome
/// tracing see a complete event stream.
pub struct PerfGuard {
    #[cfg(feature = "profiling")]
    _flush: tracing_chrome::FlushGuard,
}

/// Install a Chrome Trace subscriber when `TOFY_PERF_TRACE` is set.
///
/// Returns `Ok(None)` when the env var is unset, or always under
/// `cfg(not(feature = "profiling"))`, so callers can ignore the result
/// without branching on the feature flag.
pub fn install() -> Result<Option<PerfGuard>> {
    #[cfg(feature = "profiling")]
    {
        install_profiling()
    }
    #[cfg(not(feature = "profiling"))]
    {
        Ok(None)
    }
}

#[cfg(feature = "profiling")]
fn install_profiling() -> Result<Option<PerfGuard>> {
    use anyhow::Context;
    use std::path::PathBuf;
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let Some(path) = std::env::var_os("TOFY_PERF_TRACE") else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let (chrome_layer, flush) = ChromeLayerBuilder::new().file(&path).build();
    tracing_subscriber::registry()
        .with(chrome_layer)
        .try_init()
        .context("failed to install tracing-chrome subscriber for TOFY_PERF_TRACE")?;
    Ok(Some(PerfGuard { _flush: flush }))
}
