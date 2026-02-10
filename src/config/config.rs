use anyhow::Result;
use std::path::PathBuf;

pub struct Config {
    pub data_path: PathBuf,
    /// If set, load encoder weights from this path (e.g. previous latent checkpoint) before training.
    pub init_encoder_path: Option<PathBuf>,
    pub steps: usize,
    pub batch_size: usize,
    pub dim: usize,
    pub max_seq: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_vocab: usize,
    pub lr: f64,
    pub log_every: usize,
}

impl Config {
    /// Parse config from slice starting with data_path (for --latent)
    pub fn from_args_after(args: &[String]) -> Result<Self> {
        if args.is_empty() {
            anyhow::bail!(
                "usage: --latent <data_path> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [max_vocab]"
            );
        }
        Ok(Self {
            data_path: PathBuf::from(&args[0]),
            init_encoder_path: None,
            steps: args.get(1).and_then(|v| v.parse().ok()).unwrap_or(10000),
            batch_size: args.get(2).and_then(|v| v.parse().ok()).unwrap_or(16),
            dim: args.get(3).and_then(|v| v.parse().ok()).unwrap_or(256),
            max_seq: args.get(4).and_then(|v| v.parse().ok()).unwrap_or(72),
            num_layers: args.get(5).and_then(|v| v.parse().ok()).unwrap_or(4),
            num_heads: args.get(6).and_then(|v| v.parse().ok()).unwrap_or(4),
            max_vocab: args.get(7).and_then(|v| v.parse().ok()).unwrap_or(32_000),
            lr: 3e-4,
            log_every: 100,
        })
    }
}
