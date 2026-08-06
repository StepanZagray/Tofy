//! Host heap tuning for long P2 training runs.
//!
//! Overnight training with Rayon episode generation on glibc can retain multi-giB of
//! anonymous RSS from per-thread malloc arenas even when live allocations are modest.
//! `MALLOC_ARENA_MAX`, periodic `malloc_trim`, and optional jemalloc target that
//! fragmentation path. GPU VRAM is separate; this module only affects process RSS.

/// Glibc mallopt parameter: cap per-thread arena count (see `malloc.h`).
#[cfg(target_os = "linux")]
const M_ARENA_MAX: i32 = 8;

/// Install host allocator tuning before the first heap allocation.
pub fn init() {
    #[cfg(target_os = "linux")]
    limit_glibc_arenas();
}

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

#[cfg(target_os = "linux")]
fn limit_glibc_arenas() {
    if std::env::var_os("MALLOC_ARENA_MAX").is_some() {
        return;
    }
  // Must run before the first malloc; `main` calls `init` first.
    unsafe {
        libc::mallopt(M_ARENA_MAX, 2);
    }
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
