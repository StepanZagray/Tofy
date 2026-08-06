use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::cli::resolve_data_path;
use crate::config::WorldTrainConfig;
use crate::data::{
    count_raw_world_rows, encode_world_examples, CachedWorldStream, RawWorldStream,
    TokenizationMode, WorldExample, DEFAULT_STREAM_SHUFFLE_BUFFER,
};
use crate::model::vocab::vocab_signature;
use crate::model::{
    association_top1_accuracy, load_vocab_from_file, prediction_loss, sigreg_epps_pulley_seeded,
    ContextCompressor, LeWorldModel, OnlineEncoder,
};
use crate::tasks::veclab::model_visible_task;
use crate::tasks::world_context::{
    context_slots_from_world_pair_batch, env_bool, env_f64, env_usize,
};
use crate::util;

const TOKEN_CACHE_MANIFEST_VERSION: u32 = 8;

type WorldConfig = WorldTrainConfig;

struct WorldLogSnapshot {
    loss_val: f32,
    prediction_val: f32,
    sigreg_val: f32,
    assoc_top1: f32,
    duplicate_fn_in_batch: usize,
}

fn leworld_loss(prediction: &Tensor, sigreg: &Tensor, lambda: f64) -> Result<Tensor> {
    prediction
        .to_dtype(DType::F32)?
        .broadcast_add(&sigreg.to_dtype(DType::F32)?.affine(lambda, 0.0)?)?
        .to_dtype(DType::F32)
        .map_err(Into::into)
}

fn action_labels(batch: &[WorldExample], device: &Device) -> Result<Tensor> {
    Tensor::from_vec(
        batch.iter().map(|row| row.action_label).collect::<Vec<_>>(),
        batch.len(),
        device,
    )
    .map_err(Into::into)
}

fn world_sigreg_embeddings(state_slots: &Tensor, next_slots: &Tensor) -> Result<Tensor> {
    let state = state_slots.mean(1)?.unsqueeze(1)?;
    let next = next_slots.mean(1)?.unsqueeze(1)?;
    Tensor::cat(&[state, next], 1).map_err(Into::into)
}

fn validation_path(train_path: &Path) -> Option<PathBuf> {
    let name = train_path.file_name()?.to_str()?;
    let val_name = name.replace("_train.txt", "_val.txt");
    (val_name != name).then(|| train_path.with_file_name(val_name))
}

fn next_unique_batch(
    batch_size: usize,
    mut raw_stream: Option<&mut RawWorldStream>,
    mut cached_stream: Option<&mut CachedWorldStream>,
    vocab: &crate::model::Vocab,
) -> Result<Vec<WorldExample>> {
    let mut batch = Vec::with_capacity(batch_size);
    let mut function_ids = std::collections::HashSet::new();
    let max_attempts = batch_size.saturating_mul(32).max(32);
    let mut attempts = 0usize;
    while attempts < max_attempts {
        let request = (batch_size - batch.len()).max(1);
        let examples = if let Some(stream) = cached_stream.as_deref_mut() {
            stream.next_batch(request)?
        } else {
            let raw = raw_stream
                .as_deref_mut()
                .context("raw world stream unavailable")?
                .next_batch(request)?;
            encode_world_examples(&raw, vocab)
        };
        attempts += examples.len();
        for mut example in examples {
            let text = vocab.decode_ids_lossy(&example.state_tokens);
            let function_id = crate::tasks::veclab::parse_fn_tag(&text)
                .context("world knowledge row is missing a decodable [fn:NNN] tag")?;
            if function_ids.insert(function_id) {
                example.state_tokens = vocab.encode_boundless(model_visible_task(&text));
                batch.push(example);
                if batch.len() == batch_size {
                    return Ok(batch);
                }
            }
        }
    }
    anyhow::bail!(
        "could not assemble a world batch of {batch_size} unique function IDs after {max_attempts} rows"
    )
}

#[allow(clippy::too_many_arguments)]
fn world_validation_metric(
    batches: usize,
    batch_size: usize,
    raw_stream: &mut Option<RawWorldStream>,
    cached_stream: &mut Option<CachedWorldStream>,
    encoder: &OnlineEncoder,
    compressor: &ContextCompressor,
    world: &LeWorldModel,
    vocab: &crate::model::Vocab,
    config: &WorldConfig,
    context_segments: usize,
    recent_full_segments: usize,
    sigreg_slices: usize,
    sigreg_points: usize,
    device: &Device,
) -> Result<Option<WorldLogSnapshot>> {
    if raw_stream.is_none() && cached_stream.is_none() {
        return Ok(None);
    }
    let mut total = WorldLogSnapshot {
        loss_val: 0.0,
        prediction_val: 0.0,
        sigreg_val: 0.0,
        assoc_top1: 0.0,
        duplicate_fn_in_batch: 0,
    };
    for batch_index in 0..batches.max(1) {
        let batch = next_unique_batch(
            batch_size,
            raw_stream.as_mut(),
            cached_stream.as_mut(),
            vocab,
        )?;
        let (raw_state, raw_next) = context_slots_from_world_pair_batch(
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
        let (state_slots, next_slots) = world.encode_pair(&raw_state, &raw_next, false)?;
        let actions = action_labels(&batch, device)?;
        let predicted_slots = world.predict(&state_slots, &actions, false)?;
        let prediction = prediction_loss(&predicted_slots, &next_slots)?;
        let encoded_observations = world_sigreg_embeddings(&state_slots, &next_slots)?;
        let sigreg = sigreg_epps_pulley_seeded(
            &encoded_observations,
            sigreg_slices,
            sigreg_points,
            0x5641_4c49_4441_5445u64.wrapping_add(batch_index as u64),
        )?;
        let loss = leworld_loss(&prediction, &sigreg, config.lambda)?;
        total.loss_val += util::scalar_f32(&loss)?;
        total.prediction_val += util::scalar_f32(&prediction)?;
        total.sigreg_val += util::scalar_f32(&sigreg)?;
        total.assoc_top1 += association_top1_accuracy(&predicted_slots, &next_slots)?;
    }
    let scale = 1.0 / batches.max(1) as f32;
    total.loss_val *= scale;
    total.prediction_val *= scale;
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
        let mut rows = 0usize;
        let mut heldout_gold = 0usize;
        for line in fs::read_to_string(&config.data_path)?.lines() {
            let Some(row) = crate::data::data::raw_world_example_from_line_with_mode(
                line,
                TokenizationMode::Default,
            ) else {
                continue;
            };
            rows += 1;
            let function_id = crate::tasks::veclab::parse_fn_tag(&row.state_text).unwrap_or(0);
            if function_id > crate::tasks::veclab::SEEN_FUNCTION_MAX
                && row.next_text.contains("package solution")
            {
                heldout_gold += 1;
            }
        }
        println!(
            "World split stats: rows={} heldout_gold_rows={heldout_gold}",
            rows
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
    let train_row_count = count_raw_world_rows(&config.data_path)?;
    let val_path = validation_path(&config.data_path)
        .filter(|path| path.exists())
        .context("world knowledge training requires a sibling *_val.txt split")?;
    let val_row_count = count_raw_world_rows(&val_path)?;
    let mut world_stream = RawWorldStream::with_split(
        &config.data_path,
        DEFAULT_STREAM_SHUFFLE_BUFFER,
        None,
        0,
        false,
    )?;
    let mut val_stream = Some(RawWorldStream::with_split(&val_path, 1, None, 0, false)?);
    let world_cache_path =
        compatible_world_cache_path(&config.data_path, config.max_seq, &encoder_vocab_sig)?;
    let mut cached_world_stream = if let Some(cache_path) = world_cache_path.as_ref() {
        println!("Token cache: using world training cache {:?}", cache_path);
        Some(CachedWorldStream::with_split(
            cache_path,
            DEFAULT_STREAM_SHUFFLE_BUFFER,
            None,
            0,
            false,
        )?)
    } else {
        println!("Token cache: no world cache found; using raw tokenization stream");
        None
    };
    let mut cached_val_stream = None;
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
    let world = LeWorldModel::new(world_vb, config.bridge_dim)?;

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
    named_train_vars.retain(|entry| {
        !entry.name.ends_with("running_mean") && !entry.name.ends_with("running_var")
    });
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

    println!("Training LeWorldModel (next-embedding MSE + SIGReg, end-to-end)");
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
            let batch = next_unique_batch(
                batch_size,
                Some(&mut world_stream),
                cached_world_stream.as_mut(),
                &encoder_vocab,
            )?;
            let (raw_state, raw_next) = context_slots_from_world_pair_batch(
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
            let (state_slots, next_slots) = world.encode_pair(&raw_state, &raw_next, true)?;
            let actions = action_labels(&batch, &device)?;
            let predicted_slots = world.predict(&state_slots, &actions, true)?;
            let prediction = prediction_loss(&predicted_slots, &next_slots)?;
            let encoded_observations = world_sigreg_embeddings(&state_slots, &next_slots)?;
            let sigreg_loss = sigreg_epps_pulley_seeded(
                &encoded_observations,
                sigreg_slices,
                sigreg_points,
                (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ micro_step as u64,
            )?;
            let loss = leworld_loss(&prediction, &sigreg_loss, config.lambda)?;

            util::accumulate_scaled_gradients(&mut accumulated_grads, &train_vars, &loss, 1)?;

            if step % config.log_every == 0 && micro_step + 1 == grad_accum_steps {
                let loss_val = util::scalar_f32(&loss)?;
                let sigreg_val = util::scalar_f32(&sigreg_loss)?;
                log_snapshot = Some(WorldLogSnapshot {
                    loss_val,
                    prediction_val: util::scalar_f32(&prediction)?,
                    sigreg_val,
                    assoc_top1: association_top1_accuracy(&predicted_slots, &next_slots)?,
                    duplicate_fn_in_batch: 0,
                });
            }
        }
        util::scale_accumulated_gradients(
            &mut accumulated_grads,
            &train_vars,
            1.0 / grad_accum_steps as f64,
        )?;

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
            tb.add_scalar("loss/prediction", snap.prediction_val, step);
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
                &world,
                &encoder_vocab,
                &config,
                context_segments,
                recent_full_segments,
                sigreg_slices,
                sigreg_points,
                &device,
            )?;
            let val_selection = validation.as_ref().map(|val| {
                tb.add_scalar("val/total", val.loss_val, step);
                tb.add_scalar("val/prediction", val.prediction_val, step);
                tb.add_scalar("val/sigreg", val.sigreg_val, step);
                tb.add_scalar("val/assoc_top1_acc", val.assoc_top1, step);
                tb.add_scalar(
                    "val/duplicate_fn_in_batch",
                    val.duplicate_fn_in_batch as f32,
                    step,
                );
                val.loss_val
            });
            let selection_metric = val_selection.unwrap_or(snap.loss_val);

            if selection_metric < best_metric {
                best_metric = selection_metric;
                util::save_varmap_atomic(&world_varmap, &model_path)?;
                if config.train_encoder {
                    util::save_varmap_atomic(&encoder_varmap, &encoder_model_path)?;
                }
                saved_checkpoint = true;
                println!(
                    "step {step}/{} total {:.4} prediction {:.4} sigreg {:.4} assoc_top1 {:.3} val_sel {:.4} [saved best]",
                    config.steps, snap.loss_val, snap.prediction_val, snap.sigreg_val, snap.assoc_top1, selection_metric
                );
            } else {
                println!(
                    "step {step}/{} total {:.4} prediction {:.4} sigreg {:.4} assoc_top1 {:.3} val_sel {:.4}",
                    config.steps,
                    snap.loss_val,
                    snap.prediction_val,
                    snap.sigreg_val,
                    snap.assoc_top1,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leworld_loss_accepts_mixed_precision_scalars() -> Result<()> {
        let device = Device::Cpu;
        let prediction = Tensor::new(1.0f32, &device)?;
        let sigreg = Tensor::new(4.0f32, &device)?.to_dtype(DType::BF16)?;
        let loss = leworld_loss(&prediction, &sigreg, 0.125)?;

        assert_eq!(loss.dtype(), DType::F32);
        let value = util::scalar_f32(&loss)?;
        assert!((value - 1.5).abs() < 1e-5, "loss={value}");
        Ok(())
    }
}
