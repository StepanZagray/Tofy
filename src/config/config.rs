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
    /// Max number of masked spans per sample (JEPA). Default 3.
    pub max_spans_per_sample: usize,
    /// Max span length in tokens per mask. Default 32 for paragraph-style; use smaller for short phrases.
    pub max_span_len: usize,
    /// Cap on fraction of valid (non-pad) context that can be masked (e.g. 0.25 = at most 1/4). Defence against masking most of context.
    pub max_masked_ratio: f64,
    /// True when data is one paragraph per line (e.g. Wikipedia cache); allows single-token lines.
    pub is_paragraph_data: bool,
    pub lr: f64,
    pub log_every: usize,
}

impl Config {
    /// Parse config from slice starting with data_path (for --latent)
    pub fn from_args_after(args: &[String]) -> Result<Self> {
        if args.is_empty() {
            anyhow::bail!(
                "usage: --latent <data_path> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [max_vocab] [max_spans] [max_span_len] [max_masked_ratio]"
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
            max_spans_per_sample: args.get(8).and_then(|v| v.parse().ok()).unwrap_or(3),
            max_span_len: args.get(9).and_then(|v| v.parse().ok()).unwrap_or(32),
            max_masked_ratio: args.get(10).and_then(|v| v.parse().ok()).unwrap_or(0.25),
            is_paragraph_data: false, // set by caller when using hub:...wikipedia
            lr: 3e-4,
            log_every: 100,
        })
    }
}
