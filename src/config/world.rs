use anyhow::{bail, Result};
use candle_core::DType;
use std::path::PathBuf;

use crate::model::DecoderKind;

fn parse_train_dtype() -> DType {
    std::env::var("TOFY_TRAIN_DTYPE")
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "f16" | "float16" | "fp16" => Some(DType::F16),
            "bf16" => Some(DType::BF16),
            "f32" | "float32" | "fp32" => Some(DType::F32),
            _ => None,
        })
        .unwrap_or(DType::F32)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
        .max(1)
}

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
    pub action_loss_weight: f64,
    pub train_encoder: bool,
    pub encoder_output_path: Option<PathBuf>,
    pub train_dtype: DType,
}

impl WorldTrainConfig {
    pub fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 3 {
            bail!(
                "usage: --train-world <encoder_model.safetensors> <encoder_vocab.txt> <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--lambda <float>] [--lr <float>] [--grad-accum <int>] [--output <path>] [--encoder-output <path>] [--freeze-encoder] [--resume]"
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
        let mut action_loss_weight = None;
        let mut filtered = Vec::new();
        let mut i = 0usize;
        while i < args.len() {
            if args[i] == "--lr" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--lr requires float"))?;
                let lr: f64 = value
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--lr must be float, got {:?}", value))?;
                lr_override = Some(lr);
                i += 2;
                continue;
            }
            if args[i] == "--lambda" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--lambda requires float"))?;
                let lambda: f64 = value
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--lambda must be float, got {:?}", value))?;
                lambda_override = Some(lambda.clamp(0.0, 1.0));
                i += 2;
                continue;
            }
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
            if args[i] == "--freeze-encoder" {
                train_encoder = false;
                i += 1;
                continue;
            }
            if args[i] == "--output" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--output requires path"))?;
                output_path = Some(PathBuf::from(value));
                i += 2;
                continue;
            }
            if args[i] == "--encoder-output" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--encoder-output requires path"))?;
                encoder_output_path = Some(PathBuf::from(value));
                i += 2;
                continue;
            }
            if args[i] == "--action-loss-weight" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--action-loss-weight requires float"))?;
                let parsed: f64 = value.parse().map_err(|_| {
                    anyhow::anyhow!("--action-loss-weight must be float, got {:?}", value)
                })?;
                action_loss_weight = Some(parsed.max(0.0));
                i += 2;
                continue;
            }
            filtered.push(args[i].clone());
            i += 1;
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
            action_loss_weight: action_loss_weight.unwrap_or(0.0),
            train_encoder,
            encoder_output_path,
            train_dtype: parse_train_dtype(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct HighWorldTrainConfig {
    pub encoder_model_path: PathBuf,
    pub encoder_vocab_path: PathBuf,
    pub world_model_path: PathBuf,
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
    pub macro_min_len: usize,
    pub macro_max_len: usize,
    pub lambda: f64,
    pub lr: f64,
    pub log_every: usize,
    pub grad_accum_steps: usize,
    pub resume: bool,
    pub train_dtype: DType,
}

impl HighWorldTrainConfig {
    pub fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 4 {
            bail!(
                "usage: --train-high-world <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--macro-min-len <int>] [--macro-max-len <int>] [--lambda <float>] [--lr <float>] [--grad-accum <int>] [--output <path>] [--resume]"
            );
        }
        let mut lr_override = None;
        let mut lambda_override = None;
        let mut grad_accum_steps = 1usize;
        let mut output_path = None;
        let mut macro_min_len = env_usize("TOFY_HWM_MACRO_MIN_LEN", 2);
        let mut macro_max_len = env_usize("TOFY_HWM_MACRO_MAX_LEN", 4);
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
                    let parsed: f64 = value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--lambda must be float"))?;
                    lambda_override = Some(parsed.clamp(0.0, 1.0));
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
                "--macro-min-len" => {
                    macro_min_len = args
                        .get(i + 1)
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(macro_min_len)
                        .max(1);
                    i += 2;
                }
                "--macro-max-len" => {
                    macro_max_len = args
                        .get(i + 1)
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(macro_max_len)
                        .max(1);
                    i += 2;
                }
                "--resume" => {
                    resume = true;
                    i += 1;
                }
                "--output" | "--high-world-output" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--output requires path"))?;
                    output_path = Some(PathBuf::from(value));
                    i += 2;
                }
                _ => {
                    filtered.push(args[i].clone());
                    i += 1;
                }
            }
        }
        if macro_min_len > macro_max_len {
            std::mem::swap(&mut macro_min_len, &mut macro_max_len);
        }
        Ok(Self {
            encoder_model_path: PathBuf::from(&filtered[0]),
            encoder_vocab_path: PathBuf::from(&filtered[1]),
            world_model_path: PathBuf::from(&filtered[2]),
            data_path: PathBuf::from(&filtered[3]),
            output_path,
            steps: filtered
                .get(4)
                .and_then(|v| v.parse().ok())
                .unwrap_or(20_000),
            batch_size: filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(24),
            dim: filtered.get(6).and_then(|v| v.parse().ok()).unwrap_or(768),
            max_seq: filtered.get(7).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_layers: filtered.get(8).and_then(|v| v.parse().ok()).unwrap_or(9),
            num_heads: filtered.get(9).and_then(|v| v.parse().ok()).unwrap_or(8),
            bridge_dim: filtered.get(10).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_latent_tokens: filtered.get(11).and_then(|v| v.parse().ok()).unwrap_or(64),
            macro_min_len,
            macro_max_len,
            lambda: lambda_override.unwrap_or(0.2),
            lr: lr_override.unwrap_or(2e-4),
            log_every: env_usize("TOFY_HIGH_WORLD_LOG_EVERY", 100),
            grad_accum_steps: grad_accum_steps.max(1),
            resume,
            train_dtype: parse_train_dtype(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct OrchestratorTrainConfig {
    pub encoder_model_path: PathBuf,
    pub encoder_vocab_path: PathBuf,
    pub world_model_path: PathBuf,
    pub data_path: PathBuf,
    pub steps: usize,
    pub batch_size: usize,
    pub dim: usize,
    pub max_seq: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub bridge_dim: usize,
    pub num_latent_tokens: usize,
    pub lr: f64,
    pub log_every: usize,
    pub grad_accum_steps: usize,
    pub resume: bool,
    pub tune_planner: bool,
    pub output_path: Option<PathBuf>,
    pub train_dtype: DType,
}

impl OrchestratorTrainConfig {
    pub fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 4 {
            bail!(
                "usage: --train-orchestrator <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--lr <float>] [--grad-accum <int>] [--freeze-planner] [--output <path>] [--resume]"
            );
        }
        let mut lr_override = None;
        let mut grad_accum_steps = 1usize;
        let mut resume = std::env::var("TOFY_RESUME")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let mut tune_planner = true;
        let mut output_path = None;
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
                "--freeze-planner" => {
                    tune_planner = false;
                    i += 1;
                }
                "--output" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--output requires path"))?;
                    output_path = Some(PathBuf::from(value));
                    i += 2;
                }
                _ => {
                    filtered.push(args[i].clone());
                    i += 1;
                }
            }
        }
        Ok(Self {
            encoder_model_path: PathBuf::from(&filtered[0]),
            encoder_vocab_path: PathBuf::from(&filtered[1]),
            world_model_path: PathBuf::from(&filtered[2]),
            data_path: PathBuf::from(&filtered[3]),
            steps: filtered
                .get(4)
                .and_then(|v| v.parse().ok())
                .unwrap_or(20_000),
            batch_size: filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(24),
            dim: filtered.get(6).and_then(|v| v.parse().ok()).unwrap_or(768),
            max_seq: filtered.get(7).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_layers: filtered.get(8).and_then(|v| v.parse().ok()).unwrap_or(9),
            num_heads: filtered.get(9).and_then(|v| v.parse().ok()).unwrap_or(8),
            bridge_dim: filtered.get(10).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_latent_tokens: filtered.get(11).and_then(|v| v.parse().ok()).unwrap_or(64),
            lr: lr_override.unwrap_or(2e-4),
            log_every: env_usize("TOFY_ORCHESTRATOR_LOG_EVERY", 100),
            grad_accum_steps: grad_accum_steps.max(1),
            resume,
            tune_planner,
            output_path,
            train_dtype: parse_train_dtype(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct DecoderTrainConfig {
    pub encoder_model_path: PathBuf,
    pub encoder_vocab_path: PathBuf,
    pub world_model_path: PathBuf,
    pub data_path: PathBuf,
    pub steps: usize,
    pub batch_size: usize,
    pub max_seq: usize,
    pub dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub bridge_dim: usize,
    pub num_latent_tokens: usize,
    pub lr: f64,
    pub log_every: usize,
    pub grad_accum_steps: usize,
    pub resume: bool,
    pub train_dtype: DType,
    pub syntax_loss_weight: f64,
    pub signature_loss_weight: f64,
    pub structure_loss_weight: f64,
    pub conditioning_loss_weight: f64,
    pub conditioning_margin: f64,
    pub init_decoder_path: Option<PathBuf>,
    pub decoder_kind: DecoderKind,
    pub decoder_vocab_path: Option<PathBuf>,
    pub decoder_max_vocab: usize,
    pub decoder_output_path: Option<PathBuf>,
    pub decoder_dim: usize,
    pub decoder_layers: usize,
    pub decoder_heads: usize,
    pub decoder_ff_dim: usize,
}

impl DecoderTrainConfig {
    pub fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 4 {
            bail!(
                "usage: --train-decoder <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:id> [steps] ... [--decoder-kind <text|code>] [--decoder-vocab <path>] [--decoder-max-vocab <int>] [--lr <float>] [--grad-accum <int>] [--conditioning-loss-weight <float>] [--conditioning-margin <float>] [--init-decoder <path>] [--decoder-output <path>] [--resume]"
            );
        }
        let mut init_decoder_path = None;
        let mut decoder_output_path = None;
        let mut decoder_vocab_path = None;
        let mut decoder_kind = None;
        let mut lr_override = None;
        let mut grad_accum_steps = 1usize;
        let mut resume = std::env::var("TOFY_RESUME")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let mut decoder_max_vocab = None;
        let mut decoder_dim = None;
        let mut decoder_layers = None;
        let mut decoder_heads = None;
        let mut decoder_ff_dim = None;
        let mut conditioning_loss_weight = None;
        let mut conditioning_margin = None;
        let mut filtered = Vec::new();
        let mut i = 0usize;
        while i < args.len() {
            if args[i] == "--init-decoder" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--init-decoder requires path"))?;
                init_decoder_path = Some(PathBuf::from(value));
                i += 2;
                continue;
            }
            if args[i] == "--decoder-output" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--decoder-output requires path"))?;
                decoder_output_path = Some(PathBuf::from(value));
                i += 2;
                continue;
            }
            if args[i] == "--decoder-vocab" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--decoder-vocab requires path"))?;
                decoder_vocab_path = Some(PathBuf::from(value));
                i += 2;
                continue;
            }
            if args[i] == "--decoder-max-vocab" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--decoder-max-vocab requires integer"))?;
                decoder_max_vocab = Some(
                    value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--decoder-max-vocab must be integer"))?,
                );
                i += 2;
                continue;
            }
            if args[i] == "--decoder-dim" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--decoder-dim requires integer"))?;
                decoder_dim = Some(
                    value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--decoder-dim must be integer"))?,
                );
                i += 2;
                continue;
            }
            if args[i] == "--decoder-layers" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--decoder-layers requires integer"))?;
                decoder_layers = Some(
                    value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--decoder-layers must be integer"))?,
                );
                i += 2;
                continue;
            }
            if args[i] == "--decoder-heads" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--decoder-heads requires integer"))?;
                decoder_heads = Some(
                    value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--decoder-heads must be integer"))?,
                );
                i += 2;
                continue;
            }
            if args[i] == "--decoder-ff-dim" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--decoder-ff-dim requires integer"))?;
                decoder_ff_dim = Some(
                    value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--decoder-ff-dim must be integer"))?,
                );
                i += 2;
                continue;
            }
            if args[i] == "--decoder-kind" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--decoder-kind requires text|code"))?;
                decoder_kind = DecoderKind::from_flag(value);
                if decoder_kind.is_none() {
                    bail!("--decoder-kind must be one of: text, code");
                }
                i += 2;
                continue;
            }
            if args[i] == "--lr" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--lr requires float"))?;
                lr_override = Some(
                    value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--lr must be float"))?,
                );
                i += 2;
                continue;
            }
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
            if args[i] == "--conditioning-loss-weight" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--conditioning-loss-weight requires float"))?;
                conditioning_loss_weight =
                    Some(value.parse().map_err(|_| {
                        anyhow::anyhow!("--conditioning-loss-weight must be float")
                    })?);
                i += 2;
                continue;
            }
            if args[i] == "--conditioning-margin" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--conditioning-margin requires float"))?;
                conditioning_margin = Some(
                    value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--conditioning-margin must be float"))?,
                );
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
        let decoder_kind = decoder_kind.unwrap_or(DecoderKind::CodeSpecialist);
        Ok(Self {
            decoder_kind,
            encoder_model_path: PathBuf::from(&filtered[0]),
            encoder_vocab_path: PathBuf::from(&filtered[1]),
            world_model_path: PathBuf::from(&filtered[2]),
            data_path: PathBuf::from(&filtered[3]),
            steps: filtered
                .get(4)
                .and_then(|v| v.parse().ok())
                .unwrap_or(40_000),
            batch_size: filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(8),
            max_seq: filtered
                .get(6)
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| {
                    if decoder_kind == DecoderKind::CodeSpecialist {
                        192
                    } else {
                        128
                    }
                }),
            dim: filtered.get(7).and_then(|v| v.parse().ok()).unwrap_or(768),
            num_layers: filtered.get(8).and_then(|v| v.parse().ok()).unwrap_or(9),
            num_heads: filtered.get(9).and_then(|v| v.parse().ok()).unwrap_or(8),
            bridge_dim: filtered.get(10).and_then(|v| v.parse().ok()).unwrap_or(256),
            num_latent_tokens: filtered.get(11).and_then(|v| v.parse().ok()).unwrap_or(64),
            lr: lr_override.unwrap_or(3e-4),
            log_every: env_usize("TOFY_DECODER_LOG_EVERY", 100),
            grad_accum_steps: grad_accum_steps.max(1),
            resume,
            train_dtype: parse_train_dtype(),
            syntax_loss_weight: std::env::var("TOFY_DECODER_SYNTAX_LOSS_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.35f64)
                .max(0.0),
            signature_loss_weight: std::env::var("TOFY_DECODER_SIGNATURE_LOSS_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.45f64)
                .max(0.0),
            structure_loss_weight: std::env::var("TOFY_DECODER_STRUCTURE_LOSS_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.9f64)
                .max(0.0),
            conditioning_loss_weight: conditioning_loss_weight
                .or_else(|| {
                    std::env::var("TOFY_DECODER_CONDITIONING_LOSS_WEIGHT")
                        .ok()
                        .and_then(|v| v.parse().ok())
                })
                .unwrap_or(0.30f64)
                .max(0.0),
            conditioning_margin: conditioning_margin
                .or_else(|| {
                    std::env::var("TOFY_DECODER_CONDITIONING_MARGIN")
                        .ok()
                        .and_then(|v| v.parse().ok())
                })
                .unwrap_or(0.10f64)
                .max(0.0),
            init_decoder_path,
            decoder_vocab_path,
            decoder_max_vocab: decoder_max_vocab.unwrap_or_else(|| {
                if decoder_kind == DecoderKind::CodeSpecialist {
                    32_000
                } else {
                    16_000
                }
            }),
            decoder_output_path,
            decoder_dim: decoder_dim
                .or_else(|| {
                    std::env::var("TOFY_DECODER_DIM")
                        .ok()
                        .and_then(|v| v.parse().ok())
                })
                .unwrap_or(640),
            decoder_layers: decoder_layers
                .or_else(|| {
                    std::env::var("TOFY_DECODER_LAYERS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                })
                .unwrap_or(6),
            decoder_heads: decoder_heads
                .or_else(|| {
                    std::env::var("TOFY_DECODER_HEADS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                })
                .unwrap_or(8),
            decoder_ff_dim: decoder_ff_dim
                .or_else(|| {
                    std::env::var("TOFY_DECODER_FF_DIM")
                        .ok()
                        .and_then(|v| v.parse().ok())
                })
                .unwrap_or(2560),
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

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub encoder_model_path: PathBuf,
    pub encoder_vocab_path: PathBuf,
    pub world_model_path: PathBuf,
    pub high_world_model_path: Option<PathBuf>,
    pub bind: String,
    pub dim: usize,
    pub max_seq: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub bridge_dim: usize,
    pub num_latent_tokens: usize,
    pub debug: bool,
}

impl ServeConfig {
    pub fn from_args_after(args: &[String]) -> Result<Self> {
        let debug = args.iter().any(|arg| arg == "--debug");
        let mut high_world_model_path = None;
        let mut positional = Vec::new();
        let mut i = 0usize;
        while i < args.len() {
            match args[i].as_str() {
                "--debug" => i += 1,
                "--high-world-model" => {
                    let value = args
                        .get(i + 1)
                        .ok_or_else(|| anyhow::anyhow!("--high-world-model requires path"))?;
                    high_world_model_path = Some(PathBuf::from(value));
                    i += 2;
                }
                _ => {
                    positional.push(args[i].as_str());
                    i += 1;
                }
            }
        }
        if positional.len() < 3 {
            bail!(
                "usage: --serve <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> [bind] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--high-world-model <override>] [--debug]"
            );
        }
        Ok(Self {
            encoder_model_path: PathBuf::from(positional[0]),
            encoder_vocab_path: PathBuf::from(positional[1]),
            world_model_path: PathBuf::from(positional[2]),
            high_world_model_path,
            bind: positional
                .get(3)
                .copied()
                .unwrap_or("0.0.0.0:8080")
                .to_string(),
            dim: positional
                .get(4)
                .and_then(|v| v.parse().ok())
                .unwrap_or(768),
            max_seq: positional
                .get(5)
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            num_layers: positional.get(6).and_then(|v| v.parse().ok()).unwrap_or(9),
            num_heads: positional.get(7).and_then(|v| v.parse().ok()).unwrap_or(8),
            bridge_dim: positional
                .get(8)
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            num_latent_tokens: positional.get(9).and_then(|v| v.parse().ok()).unwrap_or(64),
            debug,
        })
    }
}
