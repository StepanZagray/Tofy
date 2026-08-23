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
//! The representative update uses identical candle-graph and NVTX semantic labels.
//! Capture and normalize it by wrapping the training binary directly:
//!
//! ```bash
//! nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --sample=cpu -- \
//!   cargo run --release --features cudnn,profiling -- p2-train \
//!   --device cuda --output-dir runs/p2/example --profile-update 2 ...
//! ```
//!
//! The bundle retains `.nsys-rep`, official CSV reports, agent evidence, and unified HTML.
//!
//! ## Host RSS (`alloc`)
//!
//! Long Rayon episode generation on glibc can retain multi-giB anonymous RSS from
//! arena fragmentation even when live allocations are small. During P2 training,
//! `TOFY_MALLOC_TRIM_EVERY` (default `100`) calls `malloc_trim`. Do not cap glibc
//! arenas for the allocation-heavy worker pool; the L40S measurement in `alloc`
//! showed a 15x throughput regression. For overnight runs, jemalloc remains available.
//!
//! Prefer `--features cudnn` only when the vendored candle patch makes it faster
//! than im2col; stock candle cudnn can regress conv backward. See
//! [`vendor/candle-core/TOFY_PATCH.md`](../vendor/candle-core/TOFY_PATCH.md).

use anyhow::Result;

/// Semantic NVTX range used to project GPU work onto Tofy's training phases.
pub struct NvtxRange {
    #[cfg(feature = "profiling")]
    _guard: nvtx::RangeGuard,
}

impl NvtxRange {
    #[inline]
    pub fn new(label: &str) -> Self {
        #[cfg(feature = "profiling")]
        {
            Self {
                _guard: nvtx::range!("{}", label),
            }
        }

        #[cfg(not(feature = "profiling"))]
        {
            let _ = label;
            Self {}
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::NvtxRange;

    #[test]
    fn nvtx_range_is_safe_to_construct_and_drop() {
        let _range = NvtxRange::new("p2.forward");
    }

    #[cfg(not(feature = "profiling"))]
    #[test]
    fn nvtx_range_is_zero_sized_without_profiling() {
        assert_eq!(std::mem::size_of::<NvtxRange>(), 0);
    }
}
