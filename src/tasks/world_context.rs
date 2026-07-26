#![allow(dead_code)]

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};

use crate::data::WorldExample;
use crate::model::encoders::EncoderFeatures;
use crate::model::{ContextCompressor, OnlineEncoder};
use crate::util;

fn world_post_state_token_sequences(batch: &[WorldExample]) -> Vec<Vec<u32>> {
    batch.iter().map(|row| row.next_tokens.clone()).collect()
}

pub(crate) fn default_context_hybrid_exact_tail(
    max_seq: usize,
    recent_full_segments: usize,
) -> usize {
    let max_seq = max_seq.max(1);
    let min_chunk = 16usize.min(max_seq);
    let target_chunk = max_seq.div_ceil(16).max(min_chunk);
    let chunk = target_chunk.min(max_seq).max(1);
    let chunk_slots = max_seq.div_ceil(chunk).max(1);
    max_seq
        .saturating_add(chunk_slots)
        .saturating_add(6)
        .saturating_mul(recent_full_segments.max(1))
}
fn context_compressor_mask_from_lengths(
    features: &EncoderFeatures,
    token_lengths: &[usize],
) -> Result<Tensor> {
    let (batch, token_slots, _) = features.token_states.dims3()?;
    let chunk_slots = features.chunk_states.dim(1)?;
    let global_slots = features.global_states.dim(1)?;
    let chunk_size = token_slots.div_ceil(chunk_slots.max(1)).max(1);
    let total_slots = token_slots + chunk_slots + global_slots + 2;
    let mut mask_buf: Vec<f32> = Vec::with_capacity(batch * total_slots);
    for b in 0..batch {
        let token_len = token_lengths
            .get(b)
            .copied()
            .unwrap_or(token_slots)
            .min(token_slots);
        let chunk_len = if token_len == 0 {
            0
        } else {
            token_len.div_ceil(chunk_size).min(chunk_slots)
        };
        mask_buf.extend((0..token_slots).map(|idx| if idx < token_len { 1.0f32 } else { 0.0f32 }));
        mask_buf.extend((0..chunk_slots).map(|idx| if idx < chunk_len { 1.0f32 } else { 0.0f32 }));
        mask_buf.extend(std::iter::repeat_n(1.0f32, global_slots + 2));
    }
    Tensor::from_vec(
        mask_buf,
        (batch, total_slots),
        features.token_states.device(),
    )
    .map_err(Into::into)
}

fn planner_forward_encoder_masked(
    context_compressor: &ContextCompressor,
    features: &EncoderFeatures,
    token_lengths: &[usize],
) -> Result<Tensor> {
    let planner = features.planner_summary()?;
    let routing = features.routing_summary()?;
    let memory = Tensor::cat(
        &[
            features.token_states.clone(),
            features.chunk_states.clone(),
            features.global_states.clone(),
            planner,
            routing,
        ],
        1,
    )?;
    let mask = context_compressor_mask_from_lengths(features, token_lengths)?;
    context_compressor.forward_masked(&memory, Some(&mask))
}

fn maybe_detach_features(features: EncoderFeatures, detach: bool) -> EncoderFeatures {
    if detach {
        features.detached()
    } else {
        features
    }
}

fn context_segment_ranges(
    total_tokens: usize,
    max_seq: usize,
    max_segments: usize,
) -> Vec<(usize, usize)> {
    let max_seq = max_seq.max(1);
    let max_segments = max_segments.max(1);
    let keep_tokens = max_seq.saturating_mul(max_segments);
    let start = total_tokens.saturating_sub(keep_tokens);
    let mut ranges = Vec::new();
    let mut cursor = start;
    while cursor < total_tokens {
        let end = (cursor + max_seq).min(total_tokens);
        ranges.push((cursor, end));
        cursor = end;
    }
    if ranges.is_empty() {
        ranges.push((0, 0));
    }
    ranges
}

fn context_compressor_segment_batch_from_features(
    features: &EncoderFeatures,
    token_lengths: &[usize],
    include_tokens: bool,
) -> Result<(Tensor, Tensor)> {
    let planner = features.planner_summary()?;
    let routing = features.routing_summary()?;
    let batch = features.token_states.dim(0)?;
    let token_slots = features.token_states.dim(1)?;
    let chunk_slots = features.chunk_states.dim(1)?;
    let global_slots = features.global_states.dim(1)?;
    let chunk_size = token_slots.div_ceil(chunk_slots.max(1));
    let mask_slots = if include_tokens {
        token_slots + chunk_slots + global_slots + 2
    } else {
        chunk_slots + global_slots + 2
    };
    let mut mask_buf = Vec::with_capacity(batch * mask_slots);
    for b in 0..batch {
        let token_len = token_lengths
            .get(b)
            .copied()
            .unwrap_or(token_slots)
            .min(token_slots);
        let valid_chunks = if token_len == 0 {
            0
        } else {
            token_len.div_ceil(chunk_size).min(chunk_slots)
        };
        if include_tokens {
            mask_buf.extend((0..token_slots).map(|idx| if idx < token_len { 1.0 } else { 0.0 }));
        }
        mask_buf.extend((0..chunk_slots).map(|idx| if idx < valid_chunks { 1.0 } else { 0.0 }));
        mask_buf.extend(std::iter::repeat_n(1.0f32, global_slots + 2));
    }

    let memory = if include_tokens {
        Tensor::cat(
            &[
                features.token_states.clone(),
                features.chunk_states.clone(),
                features.global_states.clone(),
                planner,
                routing,
            ],
            1,
        )?
    } else {
        Tensor::cat(
            &[
                features.chunk_states.clone(),
                features.global_states.clone(),
                planner,
                routing,
            ],
            1,
        )?
    };
    let mask = util::from_vec_like(mask_buf, (batch, mask_slots), &memory)?;
    Ok((memory, mask))
}

fn recursive_memory_retain(
    segment_idx: usize,
    total_segments: usize,
    recent_full_segments: usize,
) -> f64 {
    let remaining = total_segments.saturating_sub(segment_idx + 1);
    if remaining < recent_full_segments.max(1) {
        0.42
    } else {
        0.72
    }
}

struct PlannerSegmentRecord {
    sample_idx: usize,
    segment_idx: usize,
    total_segments: usize,
    recent_full_segments: usize,
    token_len: usize,
    include_tokens: bool,
}

fn append_padded_segment(
    out: &mut Vec<u32>,
    tokens: &[u32],
    start: usize,
    end: usize,
    max_seq: usize,
    pad_id: u32,
) -> usize {
    let row_start = out.len();
    let token_len = if start < end {
        let len = (end - start).min(max_seq);
        out.extend(tokens[start..end].iter().take(len).copied());
        len
    } else {
        0
    };
    while out.len() - row_start < max_seq {
        out.push(pad_id);
    }
    token_len
}

fn make_tail_token_batch(
    token_sequences: &[&[u32]],
    max_seq: usize,
    pad_id: u32,
) -> (Vec<u32>, Vec<usize>) {
    let mut input_buf = Vec::with_capacity(token_sequences.len() * max_seq);
    let mut token_lengths = Vec::with_capacity(token_sequences.len());
    for tokens in token_sequences {
        let ranges = context_segment_ranges(tokens.len(), max_seq, 1);
        let (start, end) = ranges[0];
        let token_len = append_padded_segment(&mut input_buf, tokens, start, end, max_seq, pad_id);
        token_lengths.push(token_len);
    }
    (input_buf, token_lengths)
}

fn make_segment_token_batch(
    token_sequences: &[&[u32]],
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    pad_id: u32,
) -> (Vec<u32>, Vec<PlannerSegmentRecord>, Vec<Vec<usize>>) {
    let mut input_buf = Vec::with_capacity(token_sequences.len() * context_segments * max_seq);
    let mut records = Vec::with_capacity(token_sequences.len() * context_segments);
    let mut records_by_sample = Vec::with_capacity(token_sequences.len());

    for (sample_idx, tokens) in token_sequences.iter().enumerate() {
        let segments = context_segment_ranges(tokens.len(), max_seq, context_segments);
        let sample_recent_full_segments = recent_full_segments.min(segments.len()).max(1);
        let mut sample_records = Vec::with_capacity(segments.len());
        for (segment_idx, (start, end)) in segments.iter().copied().enumerate() {
            let token_len =
                append_padded_segment(&mut input_buf, tokens, start, end, max_seq, pad_id);
            let include_tokens = segment_idx + sample_recent_full_segments >= segments.len();
            let record_idx = records.len();
            records.push(PlannerSegmentRecord {
                sample_idx,
                segment_idx,
                total_segments: segments.len(),
                recent_full_segments: sample_recent_full_segments,
                token_len,
                include_tokens,
            });
            sample_records.push(record_idx);
        }
        records_by_sample.push(sample_records);
    }

    (input_buf, records, records_by_sample)
}

fn select_encoder_features(
    features: &EncoderFeatures,
    record_indices: &[usize],
) -> Result<EncoderFeatures> {
    let index_values = record_indices
        .iter()
        .map(|idx| *idx as u32)
        .collect::<Vec<_>>();
    let indexes = Tensor::from_vec(
        index_values,
        (record_indices.len(),),
        features.token_states.device(),
    )?;
    Ok(EncoderFeatures {
        token_states: features
            .token_states
            .contiguous()?
            .index_select(&indexes, 0)?,
        chunk_states: features
            .chunk_states
            .contiguous()?
            .index_select(&indexes, 0)?,
        global_states: features
            .global_states
            .contiguous()?
            .index_select(&indexes, 0)?,
        pooled_queries: features
            .pooled_queries
            .contiguous()?
            .index_select(&indexes, 0)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn context_slots_from_token_sequences(
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    token_sequences: &[&[u32]],
    pad_id: u32,
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    device: &Device,
) -> Result<Tensor> {
    context_slots_from_token_sequences_with_detach(
        encoder,
        context_compressor,
        token_sequences,
        pad_id,
        max_seq,
        context_segments,
        recent_full_segments,
        true,
        device,
    )
}

#[allow(clippy::too_many_arguments)]
fn context_slots_from_token_sequences_with_detach(
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    token_sequences: &[&[u32]],
    pad_id: u32,
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    detach_encoder: bool,
    device: &Device,
) -> Result<Tensor> {
    if token_sequences.is_empty() {
        bail!("context slot batch is empty");
    }
    let max_seq = max_seq.max(1);
    let context_segments = context_segments.max(1);
    let segment_batch_limit = env_usize("TOFY_CONTEXT_SEGMENT_BATCH", 64);
    let hybrid_context = env_bool("TOFY_CONTEXT_HYBRID_MEMORY", true);
    let hybrid_exact_tail = env_usize(
        "TOFY_CONTEXT_HYBRID_EXACT_TAIL",
        default_context_hybrid_exact_tail(max_seq, recent_full_segments),
    );
    let hybrid_block_size = env_usize("TOFY_CONTEXT_HYBRID_BLOCK_SIZE", 16);
    let hybrid_retrieval_slots = env_usize("TOFY_CONTEXT_RETRIEVAL_SLOTS", 8);

    if context_segments == 1 {
        let mut chunk_slots =
            Vec::with_capacity(token_sequences.len().div_ceil(segment_batch_limit));
        for chunk in token_sequences.chunks(segment_batch_limit) {
            let (input_buf, token_lengths) = make_tail_token_batch(chunk, max_seq, pad_id);
            let input_ids = Tensor::from_vec(input_buf, (chunk.len(), max_seq), device)?;
            let features =
                maybe_detach_features(encoder.forward_features(&input_ids)?, detach_encoder);
            chunk_slots.push(planner_forward_encoder_masked(
                context_compressor,
                &features,
                &token_lengths,
            )?);
        }
        let refs = chunk_slots.iter().collect::<Vec<_>>();
        return Tensor::cat(&refs, 0).map_err(Into::into);
    }

    let (input_buf, records, records_by_sample) = make_segment_token_batch(
        token_sequences,
        max_seq,
        context_segments,
        recent_full_segments,
        pad_id,
    );
    let mut sample_slots = Vec::with_capacity(token_sequences.len());
    let mut memory_by_record: Vec<Option<(Tensor, Tensor)>> =
        (0..records.len()).map(|_| None).collect();
    for chunk_start in (0..records.len()).step_by(segment_batch_limit) {
        let chunk_end = (chunk_start + segment_batch_limit).min(records.len());
        let chunk_len = chunk_end - chunk_start;
        let offset = chunk_start * max_seq;
        let end = chunk_end * max_seq;
        let input_ids = Tensor::from_vec(
            input_buf[offset..end].to_vec(),
            (chunk_len, max_seq),
            device,
        )?;
        let features = maybe_detach_features(encoder.forward_features(&input_ids)?, detach_encoder);
        let mut summary_indices = Vec::new();
        let mut full_indices = Vec::new();
        for local_idx in 0..chunk_len {
            if records[chunk_start + local_idx].include_tokens {
                full_indices.push(local_idx);
            } else {
                summary_indices.push(local_idx);
            }
        }
        for (include_tokens, local_indices) in [(false, summary_indices), (true, full_indices)] {
            if local_indices.is_empty() {
                continue;
            }
            let selected = select_encoder_features(&features, &local_indices)?;
            let token_lengths = local_indices
                .iter()
                .map(|idx| records[chunk_start + *idx].token_len)
                .collect::<Vec<_>>();
            let (memory, mask) = context_compressor_segment_batch_from_features(
                &selected,
                &token_lengths,
                include_tokens,
            )?;
            for (group_pos, local_idx) in local_indices.iter().copied().enumerate() {
                let record_idx = chunk_start + local_idx;
                memory_by_record[record_idx] = Some((
                    memory.narrow(0, group_pos, 1)?,
                    mask.narrow(0, group_pos, 1)?,
                ));
            }
        }
    }

    for sample_records in &records_by_sample {
        let mut memory_refs = Vec::with_capacity(sample_records.len());
        let mut mask_refs = Vec::with_capacity(sample_records.len());
        for record_idx in sample_records {
            let (memory, mask) = memory_by_record[*record_idx]
                .as_ref()
                .context("missing context compressor for segment record")?;
            memory_refs.push(memory);
            mask_refs.push(mask);
        }
        let memory = Tensor::cat(&memory_refs, 1)?;
        let mask = Tensor::cat(&mask_refs, 1)?;
        if hybrid_context && sample_records.len() > 1 {
            sample_slots.push(context_compressor.forward_hybrid_masked(
                &memory,
                Some(&mask),
                hybrid_exact_tail,
                hybrid_block_size,
                hybrid_retrieval_slots,
            )?);
        } else {
            sample_slots.push(context_compressor.forward_masked(&memory, Some(&mask))?);
        }
    }

    let refs = sample_slots.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 0).map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn context_slots_from_world_pair_batch(
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    batch: &[WorldExample],
    pad_id: u32,
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    detach_encoder: bool,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if batch.is_empty() {
        bail!("world pair batch is empty");
    }
    let post_state_sequences = world_post_state_token_sequences(batch);
    let mut token_sequences = Vec::with_capacity(batch.len() * 2);
    token_sequences.extend(batch.iter().map(|row| row.state_tokens.as_slice()));
    token_sequences.extend(post_state_sequences.iter().map(Vec::as_slice));
    let slots = context_slots_from_token_sequences_with_detach(
        encoder,
        context_compressor,
        &token_sequences,
        pad_id,
        max_seq,
        context_segments,
        recent_full_segments,
        detach_encoder,
        device,
    )?;
    let batch_size = batch.len();
    let state_slots = slots.narrow(0, 0, batch_size)?;
    let next_slots = slots.narrow(0, batch_size, batch_size)?;
    Ok((state_slots, next_slots))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn context_slots_from_world_states(
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    batch: &[WorldExample],
    pad_id: u32,
    max_seq: usize,
    context_segments: usize,
    recent_full_segments: usize,
    device: &Device,
) -> Result<Tensor> {
    if batch.is_empty() {
        bail!("world state batch is empty");
    }
    let token_sequences = batch
        .iter()
        .map(|row| row.state_tokens.as_slice())
        .collect::<Vec<_>>();
    // Bridge training may unfreeze the world compressor/predictor, but the
    // encoder is always a fixed checkpoint in this stage. Detach encoder
    // features before the compressor to avoid retaining its activation graph.
    context_slots_from_token_sequences_with_detach(
        encoder,
        context_compressor,
        &token_sequences,
        pad_id,
        max_seq,
        context_segments,
        recent_full_segments,
        true,
        device,
    )
}
pub(crate) fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
        .max(1)
}

pub(crate) fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub(crate) fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(default)
}
