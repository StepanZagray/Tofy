use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::cli::resolve_data_path;
use crate::config::WorldTrainConfig;
use crate::data::{
    count_raw_world_rows, count_raw_world_rows_split, encode_world_examples, CachedWorldStream,
    RawWorldStream, TokenizationMode, WorldExample, DEFAULT_STREAM_SHUFFLE_BUFFER,
};
use crate::model::vocab::vocab_signature;
use crate::model::{
    association_loss, association_top1_accuracy, flatten_latent_slots, load_vocab_from_file,
    sigreg_epps_pulley, ActionStateTransition, ContextCompressor, KnowledgeReconstructionHead,
    OnlineEncoder,
};
use crate::tasks::world_context::{
    context_slots_from_world_pair_batch, env_bool, env_f64, env_usize,
};
use crate::tasks::world_support::masked_cross_entropy;
use crate::util;

const HELDOUT_SPLIT_MODULUS: usize = 20;
const HELDOUT_SPLIT_REMAINDER: usize = 0;
const TOKEN_CACHE_MANIFEST_VERSION: u32 = 8;

type WorldConfig = WorldTrainConfig;

fn reconstruction_targets(
    batch: &[WorldExample],
    pad_id: u32,
    max_seq: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    reconstruction_targets_with_mode(batch, pad_id, max_seq, true, device)
}

fn reconstruction_targets_with_mode(
    batch: &[WorldExample],
    pad_id: u32,
    max_seq: usize,
    stochastic: bool,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut labels = vec![pad_id; batch.len() * max_seq];
    let mut mask = vec![0f32; batch.len() * max_seq];
    for (row, example) in batch.iter().enumerate() {
        let tokens = &example.next_tokens[..example.next_tokens.len().min(max_seq)];
        let start = row * max_seq;
        labels[start..start + tokens.len()].copy_from_slice(tokens);
        // A stochastic half-token objective is the masked-doc reconstruction signal.
        for index in 0..tokens.len() {
            mask[start + index] = if if stochastic {
                rand::random::<f32>() < 0.5
            } else {
                (row + index) % 2 == 0
            } {
                1.0
            } else {
                0.0
            };
        }
        if !tokens.is_empty() && mask[start..start + tokens.len()].iter().all(|v| *v == 0.0) {
            mask[start] = 1.0;
        }
    }
    Ok((
        Tensor::from_vec(labels, (batch.len(), max_seq), device)?,
        Tensor::from_vec(mask, (batch.len(), max_seq), device)?,
    ))
}

struct WorldLogSnapshot {
    loss_val: f32,
    recon_val: f32,
    assoc_val: f32,
    sigreg_val: f32,
    assoc_top1: f32,
    duplicate_fn_in_batch: usize,
}

fn duplicate_function_ids(batch: &[WorldExample], vocab: &crate::model::Vocab) -> usize {
    let mut seen = std::collections::HashSet::new();
    for example in batch {
        let text = vocab.decode_ids_lossy(&example.state_tokens);
        if let Some(id) = crate::tasks::veclab::parse_fn_tag(&text) {
            if !seen.insert(id) {
                return 1;
            }
        }
    }
    0
}

#[allow(clippy::too_many_arguments)]
fn world_validation_metric(
    batches: usize,
    batch_size: usize,
    raw_stream: &mut Option<RawWorldStream>,
    cached_stream: &mut Option<CachedWorldStream>,
    encoder: &OnlineEncoder,
    compressor: &ContextCompressor,
    transition: &ActionStateTransition,
    reconstruction: &KnowledgeReconstructionHead,
    vocab: &crate::model::Vocab,
    config: &WorldConfig,
    context_segments: usize,
    recent_full_segments: usize,
    use_transition: bool,
    recon_weight: f64,
    assoc_weight: f64,
    sigreg_slices: usize,
    sigreg_points: usize,
    device: &Device,
) -> Result<Option<WorldLogSnapshot>> {
    if raw_stream.is_none() && cached_stream.is_none() {
        return Ok(None);
    }
    let mut total = WorldLogSnapshot {
        loss_val: 0.0,
        recon_val: 0.0,
        assoc_val: 0.0,
        sigreg_val: 0.0,
        assoc_top1: 0.0,
        duplicate_fn_in_batch: 0,
    };
    for _ in 0..batches.max(1) {
        let batch = if let Some(stream) = cached_stream.as_mut() {
            stream.next_batch(batch_size)?
        } else {
            encode_world_examples(
                &raw_stream
                    .as_mut()
                    .expect("validation stream checked")
                    .next_batch(batch_size)?,
                vocab,
            )
        };
        let (mut state_slots, mut next_slots) = context_slots_from_world_pair_batch(
            encoder,
            compressor,
            &batch,
            vocab.pad_id,
            config.max_seq,
            context_segments,
            recent_full_segments,
            true,
            device,
        )?;
        if use_transition {
            state_slots = transition.forward(&state_slots)?;
            next_slots = transition.forward(&next_slots)?;
        }
        let (labels, mask) =
            reconstruction_targets_with_mode(&batch, vocab.pad_id, config.max_seq, false, device)?;
        let recon = masked_cross_entropy(
            &reconstruction.forward(&next_slots, config.max_seq)?,
            &labels,
            &mask,
        )?;
        let assoc = association_loss(&state_slots, &next_slots)?;
        let sigreg = sigreg_epps_pulley(
            &flatten_latent_slots(&state_slots)?,
            sigreg_slices,
            sigreg_points,
        )?
        .affine(0.5, 0.0)?
        .broadcast_add(
            &sigreg_epps_pulley(
                &flatten_latent_slots(&next_slots)?,
                sigreg_slices,
                sigreg_points,
            )?
            .affine(0.5, 0.0)?,
        )?;
        let loss = recon
            .affine(recon_weight, 0.0)?
            .broadcast_add(&assoc.affine(assoc_weight, 0.0)?)?
            .broadcast_add(&sigreg.affine(config.lambda, 0.0)?)?;
        total.loss_val += util::scalar_f32(&loss)?;
        total.recon_val += util::scalar_f32(&recon)?;
        total.assoc_val += util::scalar_f32(&assoc)?;
        total.sigreg_val += util::scalar_f32(&sigreg)?;
        total.assoc_top1 += association_top1_accuracy(&state_slots, &next_slots)?;
        total.duplicate_fn_in_batch += duplicate_function_ids(&batch, vocab);
    }
    let scale = 1.0 / batches.max(1) as f32;
    total.loss_val *= scale;
    total.recon_val *= scale;
    total.assoc_val *= scale;
    total.sigreg_val *= scale;
    total.assoc_top1 *= scale;
    Ok(Some(total))
}

#[derive(Debug, Deserialize)]
struct CacheManifestSource {
    path: String,
    len: u64,
    content_hash: String,
}

#[derive(Debug, Deserialize)]
struct CacheTokenManifest {
    #[serde(default)]
    version: u32,
    kind: String,
    source: CacheManifestSource,
    tokenizer: String,
    max_seq: usize,
    action_filter: Option<u32>,
    vocab_signature: String,
    token_cache_path: String,
}

fn token_cache_path(kind: &str) -> Option<PathBuf> {
    if !env_bool("TOFY_USE_TOKEN_CACHE", true) {
        return None;
    }
    let cache_dir = std::env::var("TOFY_CACHE_DIR").unwrap_or_else(|_| "data/cache".to_string());
    let path = PathBuf::from(cache_dir).join(format!("{kind}.tokens.bin"));
    path.exists().then_some(path)
}

fn cache_dir() -> PathBuf {
    PathBuf::from(std::env::var("TOFY_CACHE_DIR").unwrap_or_else(|_| "data/cache".to_string()))
}

fn load_cache_token_manifest(path: &Path) -> Result<CacheTokenManifest> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn source_fingerprint_matches(path: &Path, source: &CacheManifestSource) -> Result<bool> {
    if source.path != path.to_string_lossy() {
        return Ok(false);
    }
    let metadata = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if source.len != metadata.len() {
        return Ok(false);
    }
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = 0xcbf29ce484222325u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        for byte in &buf[..read] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(source.content_hash == format!("{hash:016x}"))
}

fn compatible_world_cache_path(
    data_path: &Path,
    max_seq: usize,
    encoder_vocab_sig: &str,
) -> Result<Option<PathBuf>> {
    let cache_path = match token_cache_path("world") {
        Some(path) => path,
        None => return Ok(None),
    };
    let manifest_path = cache_dir().join("world_tokens.manifest.json");
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest = load_cache_token_manifest(&manifest_path)
        .with_context(|| format!("load world cache manifest from {:?}", manifest_path))?;
    if manifest.version != TOKEN_CACHE_MANIFEST_VERSION
        || manifest.kind != "world"
        || manifest.tokenizer != TokenizationMode::Default.as_str()
        || manifest.max_seq < max_seq
        || manifest.action_filter.is_some()
        || manifest.vocab_signature != encoder_vocab_sig
        || manifest.token_cache_path != cache_path.to_string_lossy()
        || !source_fingerprint_matches(data_path, &manifest.source)?
    {
        return Ok(None);
    }
    Ok(Some(cache_path))
}

fn default_world_encoder_path(model_path: &Path) -> PathBuf {
    let raw = model_path.to_string_lossy();
    if let Some(prefix) = raw.strip_suffix(".safetensors") {
        PathBuf::from(format!("{prefix}.encoder.safetensors"))
    } else {
        PathBuf::from(format!("{raw}.encoder.safetensors"))
    }
}

fn world_batch_size_for_step(step: usize, config: &WorldConfig) -> usize {
    if config.batch_warmup_steps > 0
        && step <= config.batch_warmup_steps
        && config.batch_warmup_value != config.batch_size
    {
        config.batch_warmup_value.max(1)
    } else {
        config.batch_size.max(1)
    }
}

fn world_grad_accum_for_step(step: usize, config: &WorldConfig) -> usize {
    if config.grad_accum_warmup_steps > 0
        && step <= config.grad_accum_warmup_steps
        && config.grad_accum_warmup_value < config.grad_accum_steps
    {
        config.grad_accum_warmup_value.max(1)
    } else {
        config.grad_accum_steps.max(1)
    }
}

pub fn try_run_train(args: &[String]) -> Result<bool> {
    if args.len() < 5
        || (args[1] != "--train-world-knowledge" && args[1] != "train-world-knowledge")
    {
        return Ok(false);
    }
    let data_path = resolve_data_path(&args[4])?.path;
    let mut args_for_cfg = args[2..].to_vec();
    args_for_cfg[2] = data_path.to_string_lossy().to_string();
    let cfg = WorldConfig::from_args_after(&args_for_cfg)?;
    run_world_training(cfg)?;
    Ok(true)
}

fn run_world_training(config: WorldConfig) -> Result<()> {
    if config.data_path.to_string_lossy().contains("veclab") {
        let rows = crate::tasks::veclab::load_task_rows(&config.data_path)?;
        let heldout_gold = rows
            .iter()
            .filter(|row| {
                row.function_id > crate::tasks::veclab::SEEN_FUNCTION_MAX
                    && row.completion.contains("package solution")
            })
            .count();
        println!(
            "World split stats: rows={} heldout_gold_rows={heldout_gold}",
            rows.len()
        );
        if heldout_gold != 0 {
            anyhow::bail!("held-out veclab gold completions are forbidden in world training");
        }
        if config
            .data_path
            .to_string_lossy()
            .contains("veclab_knowledge")
        {
            crate::tasks::prepare_veclab::print_split_stats(Path::new("data/fictional"))?;
        }
    }
    let device = match Device::new_cuda(0) {
        Ok(d) => {
            tracing::info!("using device: CUDA(0)");
            d
        }
        Err(e) => {
            tracing::warn!("CUDA not available: {}", e);
            Device::Cpu
        }
    };

    let encoder_vocab = load_vocab_from_file(&config.encoder_vocab_path)?;
    if config.data_path.to_string_lossy().contains("veclab") {
        crate::tasks::veclab::print_vocab_identifier_sanity(
            &encoder_vocab,
            Path::new("data/fictional/veclab_docs.txt"),
        )?;
    }
    let encoder_vocab_sig = vocab_signature(&encoder_vocab);
    let train_dtype = util::resolve_train_dtype(&device, config.train_dtype);
    let row_count = count_raw_world_rows(&config.data_path)?;
    let val_row_count = count_raw_world_rows_split(
        &config.data_path,
        Some(HELDOUT_SPLIT_MODULUS),
        HELDOUT_SPLIT_REMAINDER,
    )
    .unwrap_or(0);
    let train_row_count = row_count.saturating_sub(val_row_count);
    let mut world_stream = RawWorldStream::with_split(
        &config.data_path,
        DEFAULT_STREAM_SHUFFLE_BUFFER,
        Some(HELDOUT_SPLIT_MODULUS),
        HELDOUT_SPLIT_REMAINDER,
        true,
    )?;
    let mut val_stream = if val_row_count > 0 {
        Some(RawWorldStream::with_split(
            &config.data_path,
            1,
            Some(HELDOUT_SPLIT_MODULUS),
            HELDOUT_SPLIT_REMAINDER,
            false,
        )?)
    } else {
        None
    };
    let world_cache_path =
        compatible_world_cache_path(&config.data_path, config.max_seq, &encoder_vocab_sig)?;
    let mut cached_world_stream = if let Some(cache_path) = world_cache_path.as_ref() {
        println!("Token cache: using world training cache {:?}", cache_path);
        Some(CachedWorldStream::with_split(
            cache_path,
            DEFAULT_STREAM_SHUFFLE_BUFFER,
            Some(HELDOUT_SPLIT_MODULUS),
            HELDOUT_SPLIT_REMAINDER,
            true,
        )?)
    } else {
        println!("Token cache: no world cache found; using raw tokenization stream");
        None
    };
    let mut cached_val_stream = if val_row_count > 0 {
        if let Some(cache_path) = world_cache_path.as_ref() {
            Some(CachedWorldStream::with_split(
                cache_path,
                1,
                Some(HELDOUT_SPLIT_MODULUS),
                HELDOUT_SPLIT_REMAINDER,
                false,
            )?)
        } else {
            None
        }
    } else {
        None
    };
    let vocab_size = encoder_vocab.id_to_token.len();

    let mut encoder_varmap = VarMap::new();
    let encoder = if config.train_encoder {
        let encoder_vb = VarBuilder::from_varmap(&encoder_varmap, train_dtype, &device);
        let encoder = OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            vocab_size,
            config.dim,
            config.num_layers,
            config.num_heads,
        )?;
        util::load_varmap_checked(&mut encoder_varmap, &config.encoder_model_path)?;
        util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;
        encoder
    } else {
        let mut frozen_encoder_varmap = VarMap::new();
        let encoder_vb = VarBuilder::from_varmap(&frozen_encoder_varmap, train_dtype, &device);
        let loaded_encoder = OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            vocab_size,
            config.dim,
            config.num_layers,
            config.num_heads,
        )?;
        util::load_varmap_checked(&mut frozen_encoder_varmap, &config.encoder_model_path)?;
        util::cast_varmap_dtype(&mut frozen_encoder_varmap, train_dtype)?;
        let frozen_encoder_tensors = util::frozen_tensors_from_varmap(&frozen_encoder_varmap)?;
        drop(loaded_encoder);
        let encoder_vb = VarBuilder::from_tensors(frozen_encoder_tensors, train_dtype, &device);
        OnlineEncoder::new(
            encoder_vb.pp("encoder"),
            vocab_size,
            config.dim,
            config.num_layers,
            config.num_heads,
        )?
    };

    let mut world_varmap = VarMap::new();
    let world_vb = VarBuilder::from_varmap(&world_varmap, train_dtype, &device);
    let context_compressor = ContextCompressor::new(
        world_vb.pp("context_compressor"),
        config.dim,
        config.bridge_dim,
        config.num_latent_tokens,
    )?;
    let transition =
        ActionStateTransition::new(world_vb.pp("action_state_transition"), config.bridge_dim)?;
    let reconstruction = KnowledgeReconstructionHead::new(
        world_vb.pp("knowledge_reconstruction"),
        config.bridge_dim,
        vocab_size,
        config.max_seq,
    )?;

    let model_path = config
        .output_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("local_models/model_world.safetensors"));
    let encoder_model_path = config
        .encoder_output_path
        .clone()
        .unwrap_or_else(|| default_world_encoder_path(&model_path));
    let resume_stage = util::resume_stage_name("world");
    let train_checkpoint_path =
        util::checkpoint_sidecar_path(&model_path, &resume_stage, "train.safetensors");
    let encoder_train_checkpoint_path =
        util::checkpoint_sidecar_path(&encoder_model_path, &resume_stage, "train.safetensors");
    let optimizer_checkpoint_path =
        util::checkpoint_sidecar_path(&model_path, &resume_stage, "optimizer.safetensors");
    let resume_state_path =
        util::checkpoint_sidecar_path(&model_path, &resume_stage, "resume.json");
    let mut resume_state = util::TrainingResumeState::new(&resume_stage);
    if config.resume && train_checkpoint_path.exists() {
        util::load_varmap_checked(&mut world_varmap, &train_checkpoint_path)?;
        util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;
        println!("Resuming world weights from {:?}", train_checkpoint_path);
    } else if config.resume && model_path.exists() {
        util::load_varmap_checked(&mut world_varmap, &model_path)?;
        util::cast_varmap_dtype(&mut world_varmap, train_dtype)?;
        println!("Resuming world weights from best export {:?}", model_path);
    }
    if config.train_encoder {
        if config.resume && encoder_train_checkpoint_path.exists() {
            util::load_varmap_checked(&mut encoder_varmap, &encoder_train_checkpoint_path)?;
            util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;
        } else if config.resume && encoder_model_path.exists() {
            util::load_varmap_checked(&mut encoder_varmap, &encoder_model_path)?;
            util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;
        }
    }

    let mut named_train_vars = util::named_train_vars(&world_varmap)?;
    if config.train_encoder {
        named_train_vars.extend(util::named_train_vars(&encoder_varmap)?);
        named_train_vars.sort_by(|a, b| a.name.cmp(&b.name));
    }
    let train_vars = named_train_vars
        .iter()
        .map(|entry| entry.var.clone())
        .collect::<Vec<_>>();
    let mut opt = util::TrainOptimizer::new_lr_named(named_train_vars, config.lr)?;
    if config.resume {
        if let Some(state) = util::load_resume_state(&resume_state_path, &resume_stage)? {
            resume_state = state;
        }
        if optimizer_checkpoint_path.exists() {
            opt.load_state(&optimizer_checkpoint_path)?;
            if resume_state.step == 0 {
                resume_state.step = opt.step_t();
            }
        }
    }

    let use_transition = env_bool("TOFY_KNOWLEDGE_USE_TRANSITION", false);
    let recon_weight = env_f64("TOFY_WORLD_RECON_LOSS_WEIGHT", 0.5).max(0.0);
    let assoc_weight = env_f64("TOFY_WORLD_ASSOC_LOSS_WEIGHT", 0.5).max(0.0);

    println!("Training world knowledge model (reconstruction + association + SIGReg)");
    println!("Rows: train {} | val {}", train_row_count, val_row_count);

    let mut best_metric = resume_state.best_metric;
    let mut saved_checkpoint = resume_state.saved_checkpoint;
    let start_step = if config.resume {
        resume_state.step.min(config.steps)
    } else {
        0
    };

    let run_dir = util::create_run_dir("world")?;
    let mut tb = util::AsyncSummaryWriter::new(&run_dir);
    let sigreg_slices = env_usize("TOFY_SIGREG_SLICES", 1024);
    let sigreg_points = env_usize("TOFY_SIGREG_POINTS", 17);
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 4);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let clip_norm = env_f64("TOFY_WORLD_CLIP_NORM", 1.0).max(0.0);
    let val_batches = env_usize("TOFY_WORLD_VAL_BATCHES", 8);
    let checkpoint_every = env_usize("TOFY_CHECKPOINT_EVERY", 500);

    for step in (start_step + 1)..=config.steps {
        let mut accumulated_grads = None;
        let mut log_snapshot = None;
        let batch_size = world_batch_size_for_step(step, &config);
        let grad_accum_steps = world_grad_accum_for_step(step, &config);

        for micro_step in 0..grad_accum_steps {
            let batch = if let Some(ref mut cached_stream) = cached_world_stream {
                cached_stream.next_batch(batch_size)?
            } else {
                let raw_batch = world_stream.next_batch(batch_size)?;
                encode_world_examples(&raw_batch, &encoder_vocab)
            };
            let (mut state_slots, mut next_slots) = context_slots_from_world_pair_batch(
                &encoder,
                &context_compressor,
                &batch,
                encoder_vocab.pad_id,
                config.max_seq,
                context_segments,
                recent_full_segments,
                !config.train_encoder,
                &device,
            )?;
            if use_transition {
                state_slots = transition.forward(&state_slots)?;
                next_slots = transition.forward(&next_slots)?;
            }
            let (labels, recon_mask) =
                reconstruction_targets(&batch, encoder_vocab.pad_id, config.max_seq, &device)?;
            let recon_logits = reconstruction.forward(&next_slots, config.max_seq)?;
            let reconstruction_loss = masked_cross_entropy(&recon_logits, &labels, &recon_mask)?;
            let association = association_loss(&state_slots, &next_slots)?;
            let state_sigreg = sigreg_epps_pulley(
                &flatten_latent_slots(&state_slots)?,
                sigreg_slices,
                sigreg_points,
            )?;
            let next_sigreg = sigreg_epps_pulley(
                &flatten_latent_slots(&next_slots)?,
                sigreg_slices,
                sigreg_points,
            )?;
            let sigreg_loss = state_sigreg
                .affine(0.5, 0.0)?
                .broadcast_add(&next_sigreg.affine(0.5, 0.0)?)?;
            let loss = reconstruction_loss
                .affine(recon_weight, 0.0)?
                .broadcast_add(&association.affine(assoc_weight, 0.0)?)?
                .broadcast_add(&sigreg_loss.affine(config.lambda, 0.0)?)?;

            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &loss,
                grad_accum_steps,
            )?;

            if step % config.log_every == 0 && micro_step + 1 == grad_accum_steps {
                let loss_val = util::scalar_f32(&loss)?;
                let sigreg_val = util::scalar_f32(&sigreg_loss)?;
                log_snapshot = Some(WorldLogSnapshot {
                    loss_val,
                    recon_val: util::scalar_f32(&reconstruction_loss)?,
                    assoc_val: util::scalar_f32(&association)?,
                    sigreg_val,
                    assoc_top1: association_top1_accuracy(&state_slots, &next_slots)?,
                    duplicate_fn_in_batch: duplicate_function_ids(&batch, &encoder_vocab),
                });
            }
        }

        let scheduled_lr = util::scheduled_lr(config.lr, step, config.steps);
        opt.set_learning_rate(scheduled_lr);
        if let Some(grad_norm) =
            util::clip_accumulated_gradients_device(&mut accumulated_grads, &train_vars, clip_norm)?
        {
            if step % config.log_every == 0 {
                tb.add_scalar("grad/global_norm", util::scalar_f32(&grad_norm)?, step);
            }
        }
        util::optimizer_step_from_accumulated(&mut opt, &mut accumulated_grads)?;

        if step % config.log_every == 0 {
            let snap = log_snapshot.context("world grad accumulation produced no log snapshot")?;
            tb.add_scalar("loss/total", snap.loss_val, step);
            tb.add_scalar("loss/reconstruction", snap.recon_val, step);
            tb.add_scalar("loss/association", snap.assoc_val, step);
            tb.add_scalar("loss/sigreg", snap.sigreg_val, step);
            tb.add_scalar("assoc/top1_acc", snap.assoc_top1, step);
            tb.add_scalar(
                "data/duplicate_fn_in_batch",
                snap.duplicate_fn_in_batch as f32,
                step,
            );
            tb.add_scalar("schedule/lr", scheduled_lr as f32, step);
            let validation = world_validation_metric(
                val_batches,
                batch_size,
                &mut val_stream,
                &mut cached_val_stream,
                &encoder,
                &context_compressor,
                &transition,
                &reconstruction,
                &encoder_vocab,
                &config,
                context_segments,
                recent_full_segments,
                use_transition,
                recon_weight,
                assoc_weight,
                sigreg_slices,
                sigreg_points,
                &device,
            )?;
            let val_selection = validation.as_ref().map(|val| {
                tb.add_scalar("val/total", val.loss_val, step);
                tb.add_scalar("val/reconstruction", val.recon_val, step);
                tb.add_scalar("val/association", val.assoc_val, step);
                tb.add_scalar("val/sigreg", val.sigreg_val, step);
                tb.add_scalar("val/assoc_top1_acc", val.assoc_top1, step);
                tb.add_scalar(
                    "val/duplicate_fn_in_batch",
                    val.duplicate_fn_in_batch as f32,
                    step,
                );
                val.recon_val + val.assoc_val + 0.2 * val.sigreg_val
            });
            let selection_metric =
                val_selection.unwrap_or(snap.recon_val + snap.assoc_val + 0.2 * snap.sigreg_val);

            if selection_metric < best_metric {
                best_metric = selection_metric;
                util::save_varmap_atomic(&world_varmap, &model_path)?;
                if config.train_encoder {
                    util::save_varmap_atomic(&encoder_varmap, &encoder_model_path)?;
                }
                saved_checkpoint = true;
                println!(
                    "step {step}/{} total {:.4} recon {:.4} assoc {:.4} sigreg {:.4} val_sel {:.4} [saved best]",
                    config.steps, snap.loss_val, snap.recon_val, snap.assoc_val, snap.sigreg_val, selection_metric
                );
            } else {
                println!(
                    "step {step}/{} total {:.4} recon {:.4} assoc {:.4} sigreg {:.4} val_sel {:.4}",
                    config.steps,
                    snap.loss_val,
                    snap.recon_val,
                    snap.assoc_val,
                    snap.sigreg_val,
                    selection_metric
                );
            }
            tb.flush();
        }

        if step % checkpoint_every == 0 {
            util::save_varmap_atomic(&world_varmap, &train_checkpoint_path)?;
            if config.train_encoder {
                util::save_varmap_atomic(&encoder_varmap, &encoder_train_checkpoint_path)?;
            }
            opt.save_state(&optimizer_checkpoint_path)?;
            util::save_resume_state(
                &resume_state_path,
                &util::TrainingResumeState {
                    stage: resume_stage.clone(),
                    step,
                    best_metric,
                    best_aux_metric: best_metric,
                    saved_checkpoint,
                },
            )?;
        }
    }

    if !saved_checkpoint {
        util::save_varmap_atomic(&world_varmap, &model_path)?;
        if config.train_encoder {
            util::save_varmap_atomic(&encoder_varmap, &encoder_model_path)?;
        }
    }
    util::save_varmap_atomic(&world_varmap, &train_checkpoint_path)?;
    if config.train_encoder {
        util::save_varmap_atomic(&encoder_varmap, &encoder_train_checkpoint_path)?;
    }
    opt.save_state(&optimizer_checkpoint_path)?;
    util::save_resume_state(
        &resume_state_path,
        &util::TrainingResumeState {
            stage: resume_stage,
            step: config.steps,
            best_metric,
            best_aux_metric: best_metric,
            saved_checkpoint,
        },
    )?;
    tb.finish()?;
    println!("World model saved to {:?}", model_path);
    Ok(())
}
