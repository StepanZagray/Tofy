//! Host heap tuning for long P2 training runs.
//!
//! Overnight training with Rayon episode generation on glibc can retain multi-giB of
//! anonymous RSS from per-thread malloc arenas even when live allocations are modest.
//! Periodic `malloc_trim` targets that fragmentation path. GPU VRAM is separate; this
//! module only affects process RSS.
//!
//! Arena *count* is deliberately left at the glibc default. P2 training runs ~100
//! allocation-heavy threads (the Rayon episode pool plus the prefetch workers), and
//! funnelling those through a couple of arenas serialises every malloc in the process.
//! Measured on an L40S pod at `physical_batch=512`, `MALLOC_ARENA_MAX=2` cost 15x
//! throughput -- 1.9 vs 28.6 optimizer steps/min -- and left the GPU idle ~90% of the
//! time waiting on batch generation. Hosts that genuinely need the RSS ceiling can set
//! `MALLOC_ARENA_MAX`, which glibc reads on its own.

/// Install host allocator tuning before the first heap allocation.
///
/// Arena count is left to glibc (see the module docs). This stays as the single hook
/// that is guaranteed to run before `main` allocates, so future tuning has one home.
pub fn init() {}

/// Return freed heap pages to the OS where glibc supports it (no-op elsewhere).
pub fn trim_host_heap() {
    #[cfg(target_os = "linux")]
    unsafe {
        libc::malloc_trim(0);
    }
}

/// Optimizer steps between `trim_host_heap` calls (`0` disables). Default: 100.
pub fn trim_interval_from_env() -> usize {
    std::env::var("TOFY_MALLOC_TRIM_EVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_interval_default() {
        let prev = std::env::var("TOFY_MALLOC_TRIM_EVERY").ok();
        std::env::remove_var("TOFY_MALLOC_TRIM_EVERY");
        assert_eq!(trim_interval_from_env(), 100);
        if let Some(v) = prev {
            std::env::set_var("TOFY_MALLOC_TRIM_EVERY", v);
        }
    }
}
