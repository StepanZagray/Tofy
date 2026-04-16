use anyhow::Result;
use candle_core::DType;
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
    pub lambda: f64,
    pub lr: f64,
    pub log_every: usize,
    pub grad_accum_steps: usize,
    pub grad_accum_warmup_steps: usize,
    pub grad_accum_warmup_value: usize,
    pub batch_warmup_steps: usize,
    pub batch_warmup_value: usize,
    pub resume: bool,
    pub train_dtype: DType,
    pub latent_context_segments: usize,
    pub latent_recent_full_segments: usize,
    pub latent_history_ratio: f64,
}

impl Config {
    /// Parse config from slice starting with data_path (for --latent)
    pub fn from_args_after(args: &[String]) -> Result<Self> {
        if args.is_empty() {
            anyhow::bail!(
                "usage: --latent <data_path> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [max_vocab] [max_spans] [max_span_len] [max_masked_ratio] [lambda] [--grad-accum <int>] [--resume]"
            );
        }
        let mut grad_accum_steps = 1usize;
        let mut resume = std::env::var("TOFY_RESUME")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let train_dtype = std::env::var("TOFY_TRAIN_DTYPE")
            .ok()
            .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
                "f16" | "float16" | "fp16" => Some(DType::F16),
                "bf16" => Some(DType::BF16),
                "f32" | "float32" | "fp32" => Some(DType::F32),
                _ => None,
            })
            .unwrap_or(DType::F32);
        let mut filtered = Vec::new();
        let mut i = 0usize;
        while i < args.len() {
            if args[i] == "--grad-accum" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--grad-accum requires integer"))?;
                grad_accum_steps = value
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--grad-accum must be integer"))?;
                i += 2;
                continue;
            }
            if args[i] == "--resume" {
                resume = true;
                i += 1;
                continue;
            }
            filtered.push(args[i].clone());
            i += 1;
        }
        let steps = filtered
            .get(1)
            .and_then(|v| v.parse().ok())
            .unwrap_or(10000);
        let batch_size = filtered.get(2).and_then(|v| v.parse().ok()).unwrap_or(12);
        let grad_accum_steps = grad_accum_steps.max(1);
        let batch_warmup_value = std::env::var("TOFY_LATENT_WARMUP_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(batch_size)
            .max(1);
        let grad_accum_warmup_value = std::env::var("TOFY_LATENT_WARMUP_GRAD_ACCUM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1usize)
            .max(1)
            .min(grad_accum_steps);
        let warmup_is_active =
            batch_warmup_value != batch_size || grad_accum_warmup_value < grad_accum_steps;
        let grad_accum_warmup_steps = std::env::var("TOFY_LATENT_WARMUP_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(if warmup_is_active { steps / 5 } else { 0 })
            .min(steps);
        let batch_warmup_steps = grad_accum_warmup_steps;
        Ok(Self {
            data_path: PathBuf::from(&filtered[0]),
            init_encoder_path: None,
            steps,
            batch_size,
            dim: filtered.get(3).and_then(|v| v.parse().ok()).unwrap_or(768),
            max_seq: filtered.get(4).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_layers: filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(9),
            num_heads: filtered.get(6).and_then(|v| v.parse().ok()).unwrap_or(8),
            max_vocab: filtered
                .get(7)
                .and_then(|v| v.parse().ok())
                .unwrap_or(32_000),
            max_spans_per_sample: filtered.get(8).and_then(|v| v.parse().ok()).unwrap_or(3),
            max_span_len: filtered.get(9).and_then(|v| v.parse().ok()).unwrap_or(32),
            max_masked_ratio: filtered
                .get(10)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.25),
            is_paragraph_data: false, // set by caller when using hub:...wikipedia
            lambda: filtered.get(11).and_then(|v| v.parse().ok()).unwrap_or(0.2),
            lr: 3e-4,
            log_every: 100,
            grad_accum_steps,
            grad_accum_warmup_steps,
            grad_accum_warmup_value,
            batch_warmup_steps,
            batch_warmup_value,
            resume,
            train_dtype,
            latent_context_segments: std::env::var("TOFY_LATENT_CONTEXT_SEGMENTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1usize)
                .max(1),
            latent_recent_full_segments: std::env::var("TOFY_LATENT_RECENT_FULL_SEGMENTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1usize)
                .max(1),
            latent_history_ratio: std::env::var("TOFY_LATENT_HISTORY_RATIO")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.25f64)
                .clamp(0.0, 0.5),
        })
    }
}
