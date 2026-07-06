use anyhow::{bail, Result};
use candle_core::DType;
use std::path::PathBuf;

use super::common::{env_usize, parse_train_dtype};

#[derive(Debug, Clone)]
pub struct WorldTrainConfig {
    pub encoder_model_path: PathBuf,
    pub encoder_vocab_path: PathBuf,
    pub data_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub steps: usize,
    pub batch_size: usize,
    pub dim: usize,
    pub max_seq: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub bridge_dim: usize,
    pub num_latent_tokens: usize,
    pub lambda: f64,
    pub lr: f64,
    pub log_every: usize,
    pub grad_accum_steps: usize,
    pub grad_accum_warmup_steps: usize,
    pub grad_accum_warmup_value: usize,
    pub batch_warmup_steps: usize,
    pub batch_warmup_value: usize,
    pub resume: bool,
    pub train_encoder: bool,
    pub encoder_output_path: Option<PathBuf>,
    pub train_dtype: DType,
}

impl WorldTrainConfig {
    pub fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 3 {
            bail!(
                "usage: --train-world-knowledge <encoder_model.safetensors> <encoder_vocab.txt> <data_path> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [bridge_dim] [num_slots] [--lambda <float>] [--lr <float>] [--grad-accum <int>] [--output <path>] [--encoder-output <path>] [--freeze-encoder] [--resume]"
            );
        }
        let mut lr_override = None;
        let mut lambda_override = None;
        let mut grad_accum_steps = 1usize;
        let mut output_path = None;
        let mut encoder_output_path = None;
        let mut train_encoder = true;
        let mut resume = std::env::var("TOFY_RESUME")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let mut filtered = Vec::new();
        let mut i = 0usize;
        while i < args.len() {
            match args[i].as_str() {
                "--lr" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--lr requires float"))?;
                    lr_override = Some(
                        value
                            .parse()
                            .map_err(|_| anyhow::anyhow!("--lr must be float"))?,
                    );
                    i += 2;
                }
                "--lambda" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--lambda requires float"))?;
                    let lambda: f64 = value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--lambda must be float"))?;
                    lambda_override = Some(lambda.clamp(0.0, 1.0));
                    i += 2;
                }
                "--grad-accum" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--grad-accum requires integer"))?;
                    grad_accum_steps = value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--grad-accum must be integer"))?;
                    i += 2;
                }
                "--resume" => {
                    resume = true;
                    i += 1;
                }
                "--freeze-encoder" => {
                    train_encoder = false;
                    i += 1;
                }
                "--output" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--output requires path"))?;
                    output_path = Some(PathBuf::from(value));
                    i += 2;
                }
                "--encoder-output" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--encoder-output requires path"))?;
                    encoder_output_path = Some(PathBuf::from(value));
                    i += 2;
                }
                _ => {
                    filtered.push(args[i].clone());
                    i += 1;
                }
            }
        }
        let steps = filtered
            .get(3)
            .and_then(|v| v.parse().ok())
            .unwrap_or(60_000);
        let batch_size = filtered.get(4).and_then(|v| v.parse().ok()).unwrap_or(24);
        let grad_accum_steps = grad_accum_steps.max(1);
        let batch_warmup_value = std::env::var("TOFY_WORLD_WARMUP_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(batch_size)
            .max(1);
        let grad_accum_warmup_value = std::env::var("TOFY_WORLD_WARMUP_GRAD_ACCUM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1usize)
            .max(1)
            .min(grad_accum_steps);
        let warmup_is_active =
            batch_warmup_value != batch_size || grad_accum_warmup_value < grad_accum_steps;
        let grad_accum_warmup_steps = std::env::var("TOFY_WORLD_WARMUP_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(if warmup_is_active { steps / 5 } else { 0 })
            .min(steps);
        let batch_warmup_steps = grad_accum_warmup_steps;
        Ok(Self {
            encoder_model_path: PathBuf::from(&filtered[0]),
            encoder_vocab_path: PathBuf::from(&filtered[1]),
            data_path: PathBuf::from(&filtered[2]),
            output_path,
            steps,
            batch_size,
            dim: filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(768),
            max_seq: filtered.get(6).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_layers: filtered.get(7).and_then(|v| v.parse().ok()).unwrap_or(9),
            num_heads: filtered.get(8).and_then(|v| v.parse().ok()).unwrap_or(8),
            bridge_dim: filtered.get(9).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_latent_tokens: filtered.get(10).and_then(|v| v.parse().ok()).unwrap_or(64),
            lambda: lambda_override.unwrap_or(0.2),
            lr: lr_override.unwrap_or(2e-4),
            log_every: env_usize("TOFY_WORLD_LOG_EVERY", 100),
            grad_accum_steps,
            grad_accum_warmup_steps,
            grad_accum_warmup_value,
            batch_warmup_steps,
            batch_warmup_value,
            resume,
            train_encoder,
            encoder_output_path,
            train_dtype: parse_train_dtype(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct WorldEvalConfig {
    pub encoder_model_path: PathBuf,
    pub encoder_vocab_path: PathBuf,
    pub model_path: PathBuf,
    pub data_arg: String,
    pub eval_steps: usize,
    pub batch_size: usize,
    pub dim: usize,
    pub max_seq: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub bridge_dim: usize,
    pub num_latent_tokens: usize,
}

impl WorldEvalConfig {
    pub fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 4 {
            bail!(
                "usage: --eval-world <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:dataset_id> [eval_steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots]"
            );
        }
        Ok(Self {
            encoder_model_path: PathBuf::from(&args[0]),
            encoder_vocab_path: PathBuf::from(&args[1]),
            model_path: PathBuf::from(&args[2]),
            data_arg: args[3].clone(),
            eval_steps: args.get(4).and_then(|v| v.parse().ok()).unwrap_or(200),
            batch_size: args.get(5).and_then(|v| v.parse().ok()).unwrap_or(32),
            dim: args.get(6).and_then(|v| v.parse().ok()).unwrap_or(768),
            max_seq: args.get(7).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_layers: args.get(8).and_then(|v| v.parse().ok()).unwrap_or(9),
            num_heads: args.get(9).and_then(|v| v.parse().ok()).unwrap_or(8),
            bridge_dim: args.get(10).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_latent_tokens: args.get(11).and_then(|v| v.parse().ok()).unwrap_or(64),
        })
    }
}
