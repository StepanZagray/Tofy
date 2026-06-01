use anyhow::{bail, Result};
use candle_core::DType;
use std::path::PathBuf;

use crate::model::DecoderKind;

use super::common::{env_usize, parse_train_dtype};

const DEFAULT_DECODER_WARMUP_STEPS: usize = 500;
const DEFAULT_DECODER_SYNTAX_LOSS_WEIGHT: f64 = 0.05;
const DEFAULT_DECODER_SIGNATURE_LOSS_WEIGHT: f64 = 0.15;
const DEFAULT_DECODER_STRUCTURE_LOSS_WEIGHT: f64 = 0.05;
const DEFAULT_DECODER_CONDITIONING_LOSS_WEIGHT: f64 = 0.20;
const DEFAULT_DECODER_CONDITIONING_MARGIN: f64 = 0.10;
const DEFAULT_CODE_DECODER_MTP_LOSS_WEIGHT: f64 = 0.0;
const DEFAULT_CODE_DECODER_MTP_MAX_AHEAD: usize = 4;

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
    pub grad_accum_warmup_steps: usize,
    pub grad_accum_warmup_value: usize,
    pub batch_warmup_steps: usize,
    pub batch_warmup_value: usize,
    pub resume: bool,
    pub train_dtype: DType,
    pub syntax_loss_weight: f64,
    pub signature_loss_weight: f64,
    pub structure_loss_weight: f64,
    pub conditioning_loss_weight: f64,
    pub conditioning_margin: f64,
    pub mtp_loss_weight: f64,
    pub mtp_max_ahead: usize,
    pub init_decoder_path: Option<PathBuf>,
    pub decoder_kind: DecoderKind,
    pub decoder_vocab_path: Option<PathBuf>,
    pub decoder_max_vocab: usize,
    pub decoder_output_path: Option<PathBuf>,
    pub decoder_dim: usize,
    pub decoder_layers: usize,
    pub decoder_heads: usize,
    pub decoder_ff_dim: usize,
    pub build_conditioned_cache: bool,
    pub from_conditioned_cache: bool,
}

impl DecoderTrainConfig {
    pub fn from_args_after(args: &[String]) -> Result<Self> {
        if args.len() < 4 {
            bail!(
                "usage: --train-decoder <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:id> [steps] ... [--decoder-kind <text|code>] [--decoder-vocab <path>] [--decoder-max-vocab <int>] [--lr <float>] [--grad-accum <int>] [--conditioning-loss-weight <float>] [--conditioning-margin <float>] [--mtp-loss-weight <float>] [--mtp-max-ahead <int>] [--init-decoder <path>] [--decoder-output <path>] [--build-conditioned-cache] [--from-conditioned-cache] [--resume]"
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
        let mut mtp_loss_weight = None;
        let mut mtp_max_ahead = None;
        let mut build_conditioned_cache = false;
        let mut from_conditioned_cache = false;
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
            if args[i] == "--mtp-loss-weight" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--mtp-loss-weight requires float"))?;
                mtp_loss_weight = Some(
                    value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--mtp-loss-weight must be float"))?,
                );
                i += 2;
                continue;
            }
            if args[i] == "--mtp-max-ahead" {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--mtp-max-ahead requires integer"))?;
                mtp_max_ahead = Some(
                    value
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--mtp-max-ahead must be integer"))?,
                );
                i += 2;
                continue;
            }
            if args[i] == "--resume" {
                resume = true;
                i += 1;
                continue;
            }
            if args[i] == "--build-conditioned-cache" {
                build_conditioned_cache = true;
                i += 1;
                continue;
            }
            if args[i] == "--from-conditioned-cache" {
                from_conditioned_cache = true;
                i += 1;
                continue;
            }
            filtered.push(args[i].clone());
            i += 1;
        }
        let decoder_kind = decoder_kind.unwrap_or(DecoderKind::CodeSpecialist);
        let steps = filtered
            .get(4)
            .and_then(|v| v.parse().ok())
            .unwrap_or(40_000);
        let batch_size = filtered.get(5).and_then(|v| v.parse().ok()).unwrap_or(8);
        let grad_accum_steps = grad_accum_steps.max(1);
        let batch_warmup_value = std::env::var("TOFY_DECODER_WARMUP_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(batch_size)
            .min(batch_size.max(1))
            .max(1);
        let grad_accum_warmup_value = std::env::var("TOFY_DECODER_WARMUP_GRAD_ACCUM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(grad_accum_steps)
            .max(1);
        let warmup_is_active =
            batch_warmup_value != batch_size || grad_accum_warmup_value != grad_accum_steps;
        let grad_accum_warmup_steps = std::env::var("TOFY_DECODER_WARMUP_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(if warmup_is_active {
                DEFAULT_DECODER_WARMUP_STEPS
            } else {
                0
            })
            .min(steps);
        let batch_warmup_steps = grad_accum_warmup_steps;
        Ok(Self {
            decoder_kind,
            encoder_model_path: PathBuf::from(&filtered[0]),
            encoder_vocab_path: PathBuf::from(&filtered[1]),
            world_model_path: PathBuf::from(&filtered[2]),
            data_path: PathBuf::from(&filtered[3]),
            steps,
            batch_size,
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
            grad_accum_steps,
            grad_accum_warmup_steps,
            grad_accum_warmup_value,
            batch_warmup_steps,
            batch_warmup_value,
            resume,
            train_dtype: parse_train_dtype(),
            syntax_loss_weight: std::env::var("TOFY_DECODER_SYNTAX_LOSS_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_DECODER_SYNTAX_LOSS_WEIGHT)
                .max(0.0),
            signature_loss_weight: std::env::var("TOFY_DECODER_SIGNATURE_LOSS_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_DECODER_SIGNATURE_LOSS_WEIGHT)
                .max(0.0),
            structure_loss_weight: std::env::var("TOFY_DECODER_STRUCTURE_LOSS_WEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_DECODER_STRUCTURE_LOSS_WEIGHT)
                .max(0.0),
            conditioning_loss_weight: conditioning_loss_weight
                .or_else(|| {
                    std::env::var("TOFY_DECODER_CONDITIONING_LOSS_WEIGHT")
                        .ok()
                        .and_then(|v| v.parse().ok())
                })
                .unwrap_or(DEFAULT_DECODER_CONDITIONING_LOSS_WEIGHT)
                .max(0.0),
            conditioning_margin: conditioning_margin
                .or_else(|| {
                    std::env::var("TOFY_DECODER_CONDITIONING_MARGIN")
                        .ok()
                        .and_then(|v| v.parse().ok())
                })
                .unwrap_or(DEFAULT_DECODER_CONDITIONING_MARGIN)
                .max(0.0),
            mtp_loss_weight: mtp_loss_weight
                .or_else(|| {
                    std::env::var("TOFY_DECODER_MTP_LOSS_WEIGHT")
                        .ok()
                        .and_then(|v| v.parse().ok())
                })
                .unwrap_or_else(|| {
                    if decoder_kind == DecoderKind::CodeSpecialist {
                        DEFAULT_CODE_DECODER_MTP_LOSS_WEIGHT
                    } else {
                        0.0
                    }
                })
                .max(0.0),
            mtp_max_ahead: mtp_max_ahead
                .or_else(|| {
                    std::env::var("TOFY_DECODER_MTP_MAX_AHEAD")
                        .ok()
                        .and_then(|v| v.parse().ok())
                })
                .unwrap_or_else(|| {
                    if decoder_kind == DecoderKind::CodeSpecialist {
                        DEFAULT_CODE_DECODER_MTP_MAX_AHEAD
                    } else {
                        1
                    }
                })
                .max(1),
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
            build_conditioned_cache,
            from_conditioned_cache,
        })
    }
}
