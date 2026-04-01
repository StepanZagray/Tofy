//! Shared helpers.

use anyhow::Result;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

/// Format parameter count: <1M → k, <1B → M, ≥1B → B.
pub fn format_params(n: usize) -> String {
    const K: usize = 1_000;
    const M: usize = 1_000_000;
    const B: usize = 1_000_000_000;
    if n < M {
        format!("{:.1}k", n as f64 / K as f64)
    } else if n < B {
        format!("{:.2}M", n as f64 / M as f64)
    } else {
        format!("{:.2}B", n as f64 / B as f64)
    }
}

pub fn create_run_dir(stage: &str) -> Result<String> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let path = format!("runs/{stage}/{stamp}");
    fs::create_dir_all(&path)?;
    Ok(path)
}

