use anyhow::{bail, Result};
use std::path::PathBuf;

pub struct Config {
    pub data_path: PathBuf,
    pub steps: usize,
    pub batch_size: usize,
    pub dim: usize,
    pub max_ctx: usize,
    pub max_tgt: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub lr: f64,
    pub log_every: usize,
}

impl Config {
    pub fn from_args() -> Result<Self> {
        let args: Vec<String> = std::env::args().collect();
        if args.len() < 2 {
            bail!(
                "usage: {} <data_path> [steps] [batch] [dim] [max_ctx] [max_tgt] [num_layers] [num_heads]",
                args[0]
            );
        }
        Ok(Self {
            data_path: PathBuf::from(&args[1]),
            steps: args.get(2).and_then(|v| v.parse().ok()).unwrap_or(10000),
            batch_size: args.get(3).and_then(|v| v.parse().ok()).unwrap_or(16),
            dim: args.get(4).and_then(|v| v.parse().ok()).unwrap_or(256),
            max_ctx: args.get(5).and_then(|v| v.parse().ok()).unwrap_or(64),
            max_tgt: args.get(6).and_then(|v| v.parse().ok()).unwrap_or(32),
            num_layers: args.get(7).and_then(|v| v.parse().ok()).unwrap_or(4),
            num_heads: args.get(8).and_then(|v| v.parse().ok()).unwrap_or(4),
            lr: 3e-4, // Lower LR for transformer training
            log_every: 100,
        })
    }
}
