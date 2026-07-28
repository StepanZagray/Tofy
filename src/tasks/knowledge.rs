use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::cli::resolve_data_path;
use crate::config::WorldTrainConfig;
use crate::data::{
    count_raw_world_rows, encode_world_examples, CachedWorldStream, RawWorldStream,
    TokenizationMode, WorldExample, DEFAULT_STREAM_SHUFFLE_BUFFER,
};
use crate::model::vocab::vocab_signature;
use crate::model::{
    association_top1_accuracy, load_vocab_from_file, prediction_loss,
    sigreg_epps_pulley_chunked_seeded, sigreg_epps_pulley_linearization_chunked_seeded,
    sigreg_linear_surrogate, ContextCompressor, LeWorldModel, OnlineEncoder,
};
use crate::tasks::veclab::model_visible_task;
use crate::tasks::world_context::{
    context_slots_from_world_pair_batch, env_bool, env_f64, env_usize,
};
use crate::util;

const TOKEN_CACHE_MANIFEST_VERSION: u32 = 8;
const WORLD_ENCODER_RUNNING_STATS: [&str; 2] = [
    "encoder_projector.norm.running_mean",
    "encoder_projector.norm.running_var",
];

type WorldConfig = WorldTrainConfig;

struct WorldLogSnapshot {
    loss_val: f32,
    prediction_val: f32,
    sigreg_val: f32,
    assoc_top1: f32,
    duplicate_fn_in_batch: usize,
}

fn snapshot_world_encoder_running_stats(world_varmap: &VarMap) -> Result<Vec<Tensor>> {
    let vars = world_varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("world VarMap lock poisoned"))?;
    WORLD_ENCODER_RUNNING_STATS
        .iter()
        .map(|name| {
            vars.get(*name)
                .with_context(|| format!("missing world BatchNorm buffer {name}"))?
                .as_tensor()
                .detach()
                .copy()
                .map_err(Into::into)
        })
        .collect()
}

fn restore_world_encoder_running_stats(
    world_varmap: &mut VarMap,
    snapshot: &[Tensor],
) -> Result<()> {
    if snapshot.len() != WORLD_ENCODER_RUNNING_STATS.len() {
        bail!(
            "world BatchNorm snapshot has {} tensors, expected {}",
            snapshot.len(),
            WORLD_ENCODER_RUNNING_STATS.len()
        );
    }
    world_varmap
        .set(WORLD_ENCODER_RUNNING_STATS.iter().zip(snapshot.iter()))
        .map_err(Into::into)
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
    Tensor::cat(&[state_slots, next_slots], 1).map_err(Into::into)
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
    let mut sigreg_batches = Vec::with_capacity(batches.max(1));
    for _batch_index in 0..batches.max(1) {
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
        sigreg_batches.push(encoded_observations.detach());
        total.prediction_val += util::scalar_f32(&prediction)?;
        total.assoc_top1 += association_top1_accuracy(&predicted_slots, &next_slots)?;
    }
    let scale = 1.0 / batches.max(1) as f32;
    total.prediction_val *= scale;
    total.assoc_top1 *= scale;
    let sigreg_refs = sigreg_batches.iter().collect::<Vec<_>>();
    let pooled_sigreg = sigreg_epps_pulley_chunked_seeded(
        &Tensor::cat(&sigreg_refs, 0)?,
        sigreg_slices,
        sigreg_points,
        env_usize("TOFY_SIGREG_POSITION_CHUNK", 8),
        0x5641_4c49_4441_5445,
    )?;
    total.sigreg_val = util::scalar_f32(&pooled_sigreg)?;
    total.loss_val = util::scalar_f32(&leworld_loss(
        &Tensor::new(total.prediction_val, device)?,
        &pooled_sigreg,
        config.lambda,
    )?)?;
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
    let device = Device::new_cuda(0)
        .context("world knowledge training requires an available CUDA device 0")?;
    tracing::info!("using device: CUDA(0)");

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
    let mut resume_metadata = util::ResumeCheckpointMetadata::default();
    if config.resume {
        let loaded_state = util::load_resume_state(&resume_state_path, &resume_stage)?;
        resume_metadata = util::load_resume_checkpoint_metadata(&resume_state_path)?;
        let mut weight_paths = vec![train_checkpoint_path.as_path()];
        if config.train_encoder {
            weight_paths.push(encoder_train_checkpoint_path.as_path());
        }
        util::validate_resume_checkpoint_tuple(
            loaded_state.as_ref(),
            &resume_metadata,
            &weight_paths,
            &optimizer_checkpoint_path,
        )?;
        if let Some(state) = loaded_state {
            resume_state = state;
        }
        if resume_state.step > 0 {
            util::load_varmap_checked(&mut world_varmap, &train_checkpoint_path)?;
            // The world projectors deliberately keep BatchNorm parameters and
            // running statistics in F32 while the rest of the model uses the
            // configured training dtype. VarMap::load targets the already
            // constructed mixed-dtype variables; a blanket cast would attempt
            // an invalid BF16 copy into those F32 variables on CUDA.
            println!("Resuming world weights from {:?}", train_checkpoint_path);
        } else if model_path.exists() {
            bail!(
                "cannot --resume world training from best export {} without a complete train/optimizer/resume tuple",
                model_path.display()
            );
        }
    }
    if config.train_encoder && config.resume && resume_state.step > 0 {
        util::load_varmap_checked(&mut encoder_varmap, &encoder_train_checkpoint_path)?;
        util::cast_varmap_dtype(&mut encoder_varmap, train_dtype)?;
    }

    let mut named_train_vars = util::named_train_vars(&world_varmap)?;
    named_train_vars.retain(|entry| {
        !entry.name.ends_with("running_mean") && !entry.name.ends_with("running_var")
    });
    if config.train_encoder {
        let mut encoder_train_vars = util::named_train_vars(&encoder_varmap)?;
        // The masked-pretraining predictor is intentionally retained in the
        // checkpoint schema, but the world objective never calls it. Exclude
        // disconnected weights from optimizer/master-state allocation.
        encoder_train_vars.retain(|entry| !entry.name.contains("predictor_"));
        named_train_vars.extend(encoder_train_vars);
        named_train_vars.sort_by(|a, b| a.name.cmp(&b.name));
    }
    let train_vars = named_train_vars
        .iter()
        .map(|entry| entry.var.clone())
        .collect::<Vec<_>>();
    let mut opt = util::TrainOptimizer::new_lr_named(named_train_vars, config.lr)?;
    if config.resume && resume_state.step > 0 {
        opt.load_state(&optimizer_checkpoint_path)?;
        if opt.step_t() != resume_state.step {
            bail!(
                "world optimizer loaded at step {}, expected {}",
                opt.step_t(),
                resume_state.step
            );
        }
    }
    resume_metadata.validate_and_set_batch_schedule(config.batch_size, config.grad_accum_steps)?;

    println!("Training LeWorldModel (next-embedding MSE + SIGReg, end-to-end)");
    println!("Rows: train {} | val {}", train_row_count, val_row_count);
    println!(
        "World SIGReg pools all gradient-accumulation microbatches: {} independent examples per optimizer step",
        config.batch_size * config.grad_accum_steps
    );

    let mut best_metric = resume_state.best_metric;
    let mut saved_checkpoint = resume_state.saved_checkpoint;
    let start_step = if config.resume { resume_state.step } else { 0 };

    let run_dir = util::create_run_dir("world")?;
    let mut tb = util::AsyncSummaryWriter::new(&run_dir);
    let sigreg_slices = env_usize("TOFY_SIGREG_SLICES", 1024);
    let sigreg_points = env_usize("TOFY_SIGREG_POINTS", 17);
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 4);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let clip_norm = env_f64("TOFY_WORLD_CLIP_NORM", 1.0).max(0.0);
    let val_batches = env_usize("TOFY_WORLD_VAL_BATCHES", 8);
    let checkpoint_every = env_usize("TOFY_CHECKPOINT_EVERY", 500);
    let health_log_every = env_usize("TOFY_WORLD_HEALTH_LOG_EVERY", 0);
    let gradient_spike_ratio = env_f64("TOFY_WORLD_GRAD_SPIKE_RATIO", 50.0).max(2.0) as f32;
    let gradient_absolute_floor =
        env_f64("TOFY_WORLD_GRAD_SPIKE_FLOOR", clip_norm.max(1.0) * 20.0).max(1.0) as f32;
    let early_stop_patience = env_usize("TOFY_WORLD_EARLY_STOP_PATIENCE", 3_000);
    let min_association = env_f64("TOFY_WORLD_MIN_ASSOCIATION", 0.9).clamp(0.0, 1.0) as f32;
    let association_penalty = env_f64("TOFY_WORLD_ASSOCIATION_PENALTY", 1.0).max(0.0) as f32;
    let mut gradient_norm_ema: Option<f32> = None;
    let mut last_improvement_step = resume_metadata
        .last_improvement_step
        .unwrap_or(start_step)
        .min(start_step);
    let mut completed_step = start_step;

    let mut stopped_early = false;
    for step in (start_step + 1)..=config.steps {
        let step_started = Instant::now();
        let mut accumulated_grads = None;
        let mut log_snapshot = None;
        let batch_size = world_batch_size_for_step(step, &config);
        let grad_accum_steps = world_grad_accum_for_step(step, &config);
        let mut microbatches = Vec::with_capacity(grad_accum_steps);
        let mut cached_sigreg = Vec::with_capacity(grad_accum_steps);
        let capture_log = step % config.log_every == 0;
        let mut prediction_sum = 0.0f32;
        let mut association_sum = 0.0f32;

        for _micro_step in 0..grad_accum_steps {
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
            cached_sigreg.push(encoded_observations.detach());
            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &prediction,
                grad_accum_steps,
            )?;

            if capture_log {
                prediction_sum += util::scalar_f32(&prediction)?;
                association_sum += association_top1_accuracy(&predicted_slots, &next_slots)?;
            }
            microbatches.push(batch);
        }
        if capture_log {
            let divisor = grad_accum_steps as f32;
            let prediction_val = prediction_sum / divisor;
            log_snapshot = Some(WorldLogSnapshot {
                loss_val: prediction_val,
                prediction_val,
                sigreg_val: 0.0,
                assoc_top1: association_sum / divisor,
                duplicate_fn_in_batch: 0,
            });
        }

        let sigreg_position_chunk = env_usize("TOFY_SIGREG_POSITION_CHUNK", 8);
        let sigreg_slice_chunk = env_usize("TOFY_SIGREG_SLICE_CHUNK", 128);
        let sigreg_seed = (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x574f_524c_4400_0001;
        let pooled_sigreg = {
            let refs = cached_sigreg.iter().collect::<Vec<_>>();
            let pooled = Tensor::cat(&refs, 0)?;
            sigreg_epps_pulley_linearization_chunked_seeded(
                &pooled,
                sigreg_slices,
                sigreg_points,
                sigreg_position_chunk,
                sigreg_slice_chunk,
                sigreg_seed,
            )?
        };
        // Replay must use training batch statistics so its gradients match the
        // first pass, but it must not advance persistent BatchNorm statistics
        // a second time.
        let encoder_running_stats = snapshot_world_encoder_running_stats(&world_varmap)?;
        let sigreg_rows = cached_sigreg[0].dim(0)?;
        for (micro_step, batch) in microbatches.iter().enumerate() {
            let (raw_state, raw_next) = context_slots_from_world_pair_batch(
                &encoder,
                &context_compressor,
                batch,
                encoder_vocab.pad_id,
                config.max_seq,
                context_segments,
                recent_full_segments,
                !config.train_encoder,
                &device,
            )?;
            let (state_slots, next_slots) = world.encode_pair(&raw_state, &raw_next, true)?;
            let live_observations = world_sigreg_embeddings(&state_slots, &next_slots)?;
            let sigreg_surrogate = sigreg_linear_surrogate(
                &live_observations,
                &pooled_sigreg.input_gradient,
                micro_step * sigreg_rows,
            )?;
            util::accumulate_scaled_gradients(
                &mut accumulated_grads,
                &train_vars,
                &sigreg_surrogate.affine(config.lambda, 0.0)?,
                1,
            )?;
        }
        restore_world_encoder_running_stats(&mut world_varmap, &encoder_running_stats)?;
        if let Some(snapshot) = log_snapshot.as_mut() {
            snapshot.sigreg_val = pooled_sigreg.value;
            snapshot.loss_val =
                snapshot.prediction_val + config.lambda as f32 * pooled_sigreg.value;
        }

        let scheduled_lr = util::scheduled_lr(config.lr, step, config.steps);
        opt.set_learning_rate(scheduled_lr);
        let grad_norm = util::clip_accumulated_gradients_device(
            &mut accumulated_grads,
            &train_vars,
            clip_norm,
        )?;
        if let Some(grad_norm) = grad_norm {
            let norm = util::scalar_f32(&grad_norm)?;
            if !norm.is_finite() {
                bail!("world gradient became non-finite at step {step}");
            }
            if let Some(ema) = gradient_norm_ema {
                let maximum = (ema * gradient_spike_ratio).max(gradient_absolute_floor);
                if norm > maximum {
                    bail!(
                        "world gradient spike rejected at step {step}: norm={norm:.6} maximum={maximum:.6} ema={ema:.6}; best checkpoint remains intact"
                    );
                }
                gradient_norm_ema = Some(0.95 * ema + 0.05 * norm);
            } else {
                gradient_norm_ema = Some(norm);
            }
            if step % config.log_every == 0 {
                tb.add_scalar("grad/global_norm", norm, step);
            }
        }
        util::optimizer_step_from_accumulated(&mut opt, &mut accumulated_grads)?;
        // Candle launches CUDA optimizer kernels asynchronously. Fence the
        // step so temporary Muon/AdamW workspaces cannot accumulate across
        // iterations and the health timer measures completed GPU work.
        device.synchronize()?;
        completed_step = step;
        if health_log_every > 0 && step % health_log_every == 0 {
            println!(
                "World health: step {step}/{} physical_batch={batch_size} grad_accum={grad_accum_steps} effective_batch={} step_seconds={:.3}",
                config.steps,
                batch_size * grad_accum_steps,
                step_started.elapsed().as_secs_f64()
            );
        }

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
                val.loss_val + (min_association - val.assoc_top1).max(0.0) * association_penalty
            });
            let selection_metric = val_selection.unwrap_or(snap.loss_val);

            if selection_metric < best_metric {
                best_metric = selection_metric;
                last_improvement_step = step;
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
            if early_stop_patience > 0
                && step.saturating_sub(last_improvement_step) >= early_stop_patience
            {
                println!(
                    "World early stopping at step {step}: no held-out improvement for {early_stop_patience} steps; best={best_metric:.6}"
                );
                stopped_early = true;
                break;
            }
        }

        if step % checkpoint_every == 0 {
            let checkpoint_id = util::new_resume_checkpoint_id(&resume_stage, step);
            resume_metadata.checkpoint_id = Some(checkpoint_id.clone());
            resume_metadata.last_improvement_step = Some(last_improvement_step);
            util::save_varmap_resume_checkpoint_atomic(
                &world_varmap,
                &train_checkpoint_path,
                &checkpoint_id,
            )?;
            if config.train_encoder {
                util::save_varmap_resume_checkpoint_atomic(
                    &encoder_varmap,
                    &encoder_train_checkpoint_path,
                    &checkpoint_id,
                )?;
            }
            util::save_optimizer_resume_checkpoint_atomic(
                &opt,
                &optimizer_checkpoint_path,
                &checkpoint_id,
            )?;
            util::save_resume_state_with_metadata(
                &resume_state_path,
                &util::TrainingResumeState {
                    stage: resume_stage.clone(),
                    step,
                    best_metric,
                    best_aux_metric: best_metric,
                    saved_checkpoint,
                    terminal: None,
                },
                &resume_metadata,
            )?;
        }
    }

    if !saved_checkpoint {
        util::save_varmap_atomic(&world_varmap, &model_path)?;
        if config.train_encoder {
            util::save_varmap_atomic(&encoder_varmap, &encoder_model_path)?;
        }
        saved_checkpoint = true;
    }
    let checkpoint_id = util::new_resume_checkpoint_id(&resume_stage, completed_step);
    resume_metadata.checkpoint_id = Some(checkpoint_id.clone());
    resume_metadata.last_improvement_step = Some(last_improvement_step);
    util::save_varmap_resume_checkpoint_atomic(
        &world_varmap,
        &train_checkpoint_path,
        &checkpoint_id,
    )?;
    if config.train_encoder {
        util::save_varmap_resume_checkpoint_atomic(
            &encoder_varmap,
            &encoder_train_checkpoint_path,
            &checkpoint_id,
        )?;
    }
    util::save_optimizer_resume_checkpoint_atomic(
        &opt,
        &optimizer_checkpoint_path,
        &checkpoint_id,
    )?;
    util::save_resume_state_with_metadata(
        &resume_state_path,
        &util::TrainingResumeState {
            stage: resume_stage,
            step: completed_step,
            best_metric,
            best_aux_metric: best_metric,
            saved_checkpoint,
            terminal: Some(if stopped_early {
                util::TrainingTerminal::EarlyStopped
            } else {
                util::TrainingTerminal::TargetReached
            }),
        },
        &resume_metadata,
    )?;
    tb.finish()?;
    println!("World model saved to {:?}", model_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::LeWorldModel;
    use candle_nn::VarBuilder;

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

    #[test]
    fn world_sigreg_replay_does_not_persist_batch_norm_updates() -> Result<()> {
        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        let world = LeWorldModel::new(VarBuilder::from_varmap(&varmap, DType::F32, &device), 16)?;
        let raw_state = Tensor::randn(0f32, 1f32, (2, 3, 16), &device)?;
        let raw_next = Tensor::randn(2f32, 1f32, (2, 3, 16), &device)?;

        world.encode_pair(&raw_state, &raw_next, true)?;
        let after_prediction = snapshot_world_encoder_running_stats(&varmap)?;
        world.encode_pair(&raw_state, &raw_next, true)?;
        let after_replay = snapshot_world_encoder_running_stats(&varmap)?;
        assert_ne!(
            after_prediction[0].to_vec1::<f32>()?,
            after_replay[0].to_vec1::<f32>()?,
            "the test replay must exercise a BatchNorm running-stat update"
        );

        restore_world_encoder_running_stats(&mut varmap, &after_prediction)?;
        let restored = snapshot_world_encoder_running_stats(&varmap)?;
        for (expected, actual) in after_prediction.iter().zip(restored.iter()) {
            assert_eq!(expected.to_vec1::<f32>()?, actual.to_vec1::<f32>()?);
        }
        Ok(())
    }
}
