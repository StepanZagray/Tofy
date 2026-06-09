use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::ops;
use std::collections::HashSet;

use crate::data::{
    encode_text_with_vocab_mode, make_decoder_batch_from_slice, RawWorldExample, TokenizationMode,
    WorldExample, ACTION_CODE, ACTION_DONE, ACTION_FETCH_DOCS, CODE_CONTROL_TOKENS, CODE_EOS_TOKEN,
};
use crate::model::{
    flatten_latent_slots, mean_cosine_similarity, prediction_loss, tensor_rms,
    ActionStateTransition, CodeDecoder, ContextCompressor, DecoderConditioningAdapter,
    OnlineEncoder, Vocab,
};
use crate::tasks::world::{
    context_slots_from_token_sequences, decoder_tokenization_mode, env_bool, env_usize,
};
use crate::util;

#[derive(Clone, Copy, Default)]
pub(crate) struct ActionMetrics {
    pub(crate) accuracy: f32,
    pub(crate) balanced_accuracy: f32,
    pub(crate) macro_f1: f32,
    pub(crate) code_precision: f32,
    pub(crate) code_recall: f32,
    pub(crate) code_f1: f32,
    pub(crate) code_rate: f32,
    pub(crate) pred_code_rate: f32,
    pub(crate) done_precision: f32,
    pub(crate) done_recall: f32,
    pub(crate) done_f1: f32,
    pub(crate) done_rate: f32,
    pub(crate) pred_done_rate: f32,
    pub(crate) fetch_docs_precision: f32,
    pub(crate) fetch_docs_recall: f32,
    pub(crate) fetch_docs_f1: f32,
    pub(crate) fetch_docs_rate: f32,
    pub(crate) pred_fetch_docs_rate: f32,
}

pub(crate) struct WorldBatchMetrics {
    pub(crate) total_loss: f32,
    pub(crate) transition_loss: f32,
    pub(crate) sigreg_loss: f32,
    pub(crate) action_loss: f32,
    pub(crate) inverse_loss: f32,
    pub(crate) action_metrics: ActionMetrics,
    pub(crate) inverse_action_metrics: ActionMetrics,
    pub(crate) transition_cosine: f32,
}

pub(crate) struct DecoderBatchMetrics {
    pub(crate) loss: f32,
    pub(crate) raw_loss: f32,
    pub(crate) ablated_loss: f32,
    pub(crate) conditioning_gain: f32,
    pub(crate) zero_gain: f32,
    pub(crate) shuffled_loss: f32,
    pub(crate) shuffle_gain: f32,
    pub(crate) hard_negative_gain: f32,
    pub(crate) syntax_loss: f32,
    pub(crate) signature_loss: f32,
    pub(crate) structure_loss: f32,
    pub(crate) perplexity: f32,
    pub(crate) active_tokens: f32,
    pub(crate) active_frac: f32,
    pub(crate) world_rms: f32,
    pub(crate) oov_rate: f32,
    pub(crate) token_accuracy: f32,
    pub(crate) identifier_accuracy: f32,
    pub(crate) delimiter_balance_rate: f32,
    pub(crate) syntax_token_accuracy: f32,
    pub(crate) function_skeleton_rate: f32,
    pub(crate) signature_token_accuracy: f32,
    pub(crate) signature_exact_rate: f32,
    pub(crate) function_name_token_accuracy: f32,
    pub(crate) function_name_exact_rate: f32,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct DecoderConditioningGains {
    pub(crate) conditioning_gain: f32,
    pub(crate) zero_gain: f32,
    pub(crate) shuffle_gain: f32,
    pub(crate) hard_negative_gain: f32,
}

pub(crate) fn decoder_conditioning_gains(
    raw_loss: f32,
    ablated_loss: f32,
    shuffled_loss: f32,
    hard_mismatch_loss: f32,
    compute_conditioning_metrics: bool,
) -> DecoderConditioningGains {
    if !compute_conditioning_metrics {
        return DecoderConditioningGains::default();
    }
    let conditioning_gain = ablated_loss - raw_loss;
    DecoderConditioningGains {
        conditioning_gain,
        zero_gain: conditioning_gain,
        shuffle_gain: shuffled_loss - raw_loss,
        hard_negative_gain: ablated_loss.min(shuffled_loss).min(hard_mismatch_loss) - raw_loss,
    }
}

pub(crate) fn shuffled_conditioning_latent(world_latent: &Tensor) -> Result<Tensor> {
    slot_shuffled_conditioning_latent(world_latent)
        .or_else(|_| shifted_conditioning_latent(world_latent, 1))
}

pub(crate) fn slot_shuffled_conditioning_latent(world_latent: &Tensor) -> Result<Tensor> {
    use rand::seq::SliceRandom;
    let (batch, slots, _) = world_latent.dims3()?;
    if slots <= 1 {
        return shifted_conditioning_latent(world_latent, 1);
    }
    let device = world_latent.device();
    let mut rng = rand::rng();
    let mut out_rows = Vec::with_capacity(batch);
    for row in 0..batch {
        let mut perm = (0..slots as u32).collect::<Vec<_>>();
        perm.shuffle(&mut rng);
        let indices = Tensor::from_vec(perm, (slots,), device)?;
        let shuffled = world_latent
            .narrow(0, row, 1)?
            .squeeze(0)?
            .index_select(&indices, 0)?
            .unsqueeze(0)?;
        out_rows.push(shuffled);
    }
    Tensor::cat(&out_rows, 0).map_err(Into::into)
}

pub(crate) fn hard_mismatched_conditioning_latent(world_latent: &Tensor) -> Result<Tensor> {
    let (batch, _, _) = world_latent.dims3()?;
    shifted_conditioning_latent(world_latent, (batch / 2).max(1))
}

fn shifted_conditioning_latent(world_latent: &Tensor, offset: usize) -> Result<Tensor> {
    let (batch, _, _) = world_latent.dims3()?;
    if batch <= 1 {
        return world_latent.affine(0.0, 0.0).map_err(Into::into);
    }
    let offset = offset % batch;
    if offset == 0 {
        return Ok(world_latent.clone());
    }
    let tail = world_latent.narrow(0, offset, batch - offset)?;
    let head = world_latent.narrow(0, 0, offset)?;
    Tensor::cat(&[&tail, &head], 0).map_err(Into::into)
}

pub(crate) struct DecoderPredictionMetrics {
    pub(crate) token_accuracy: f32,
    pub(crate) identifier_accuracy: f32,
    pub(crate) delimiter_balance_rate: f32,
    pub(crate) syntax_token_accuracy: f32,
    pub(crate) function_skeleton_rate: f32,
    pub(crate) signature_token_accuracy: f32,
    pub(crate) signature_exact_rate: f32,
    pub(crate) function_name_token_accuracy: f32,
    pub(crate) function_name_exact_rate: f32,
}

fn action_focal_gamma() -> f32 {
    std::env::var("TOFY_ACTION_FOCAL_GAMMA")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(2.0)
        .max(0.0)
}

pub(crate) fn action_cross_entropy(
    logits: &Tensor,
    labels: &[u32],
    device: &Device,
) -> Result<Tensor> {
    let b = logits.dim(0)?;
    let n_classes = logits.dim(1)?;
    if labels.len() != b {
        anyhow::bail!(
            "action label count {} must match logits batch {}",
            labels.len(),
            b
        );
    }
    let log_probs = ops::log_softmax(logits, 1)?;
    let probs = ops::softmax(logits, 1)?;
    let class_weights = balanced_class_weights(labels, n_classes);
    let sample_labels = labels.to_vec();
    let indices = Tensor::from_vec(
        sample_labels.iter().map(|&x| x as i64).collect::<Vec<_>>(),
        (b,),
        device,
    )?
    .unsqueeze(1)?;
    let nll = log_probs
        .gather(&indices, 1)?
        .squeeze(1)?
        .affine(-1.0, 0.0)?;
    let mut sample_weights = sample_labels
        .iter()
        .map(|&label| class_weights.get(label as usize).copied().unwrap_or(1.0))
        .collect::<Vec<_>>();
    let focal_gamma = action_focal_gamma();
    if focal_gamma > 0.0 {
        let p_t = probs.gather(&indices, 1)?.squeeze(1)?;
        let focal_scale = p_t
            .affine(-1.0, 1.0)?
            .powf(focal_gamma as f64)?
            .to_vec1::<f32>()?;
        for (weight, scale) in sample_weights.iter_mut().zip(focal_scale) {
            *weight *= scale.max(1e-6);
        }
    }
    let sample_weights = util::from_vec_like(sample_weights, (b,), &nll)?;
    let weighted_nll = nll.broadcast_mul(&sample_weights)?;
    let normalizer = sample_weights.sum_all()?.clamp(1e-8, 1e10)?;
    Ok(weighted_nll.sum_all()?.broadcast_div(&normalizer)?)
}

fn balanced_class_weights(labels: &[u32], n_classes: usize) -> Vec<f32> {
    let mut counts = vec![0usize; n_classes];
    for &label in labels {
        if let Some(count) = counts.get_mut(label as usize) {
            *count += 1;
        }
    }
    let present = counts.iter().filter(|&&count| count > 0).count().max(1) as f32;
    let total = labels.len().max(1) as f32;
    counts
        .into_iter()
        .map(|count| {
            if count == 0 {
                1.0
            } else {
                (total / (present * count as f32)).clamp(0.5, 12.0)
            }
        })
        .collect()
}

fn predicted_positive_rate(confusion: &[Vec<usize>], positive_label: usize) -> f32 {
    let pred_total = confusion
        .iter()
        .map(|row| row.get(positive_label).copied().unwrap_or(0))
        .sum::<usize>() as f32;
    let total = confusion
        .iter()
        .map(|row| row.iter().sum::<usize>())
        .sum::<usize>()
        .max(1) as f32;
    pred_total / total
}

fn class_prf(confusion: &[Vec<usize>], label: usize) -> (f32, f32, f32, f32) {
    let tp = confusion
        .get(label)
        .and_then(|row| row.get(label))
        .copied()
        .unwrap_or(0) as f32;
    let true_total = confusion
        .get(label)
        .map(|row| row.iter().sum::<usize>())
        .unwrap_or(0) as f32;
    let pred_total = confusion
        .iter()
        .map(|row| row.get(label).copied().unwrap_or(0))
        .sum::<usize>() as f32;
    let precision = if pred_total > 0.0 {
        tp / pred_total
    } else {
        0.0
    };
    let recall = if true_total > 0.0 {
        tp / true_total
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let total = confusion
        .iter()
        .map(|row| row.iter().sum::<usize>())
        .sum::<usize>()
        .max(1) as f32;
    (precision, recall, f1, true_total / total)
}

pub(crate) fn world_sigreg_loss(
    state_sigreg: &Tensor,
    next_sigreg: &Tensor,
    pred_sigreg: &Tensor,
) -> Result<Tensor> {
    let pred_weight = std::env::var("TOFY_WORLD_SIGREG_PRED_WEIGHT")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.6)
        .clamp(0.0, 1.0);
    let other_weight = (1.0 - pred_weight) * 0.5;
    pred_sigreg
        .affine(pred_weight, 0.0)?
        .broadcast_add(&state_sigreg.affine(other_weight, 0.0)?)?
        .broadcast_add(&next_sigreg.affine(other_weight, 0.0)?)
        .map_err(Into::into)
}

pub(crate) fn world_selection_score(metrics: &WorldBatchMetrics) -> f32 {
    let code_rate_gap = (metrics.action_metrics.pred_code_rate - metrics.action_metrics.code_rate)
        .abs();
    metrics.transition_loss
        + 0.2 * metrics.sigreg_loss
        + 0.1 * metrics.action_loss
        + 0.05 * metrics.inverse_loss
        + 0.1 * code_rate_gap
        - 0.05 * metrics.action_metrics.macro_f1
        - 0.02 * metrics.transition_cosine
}

pub(crate) fn slot_delta_slots(next_slots: &Tensor, state_slots: &Tensor) -> Result<Tensor> {
    next_slots.broadcast_sub(state_slots).map_err(Into::into)
}

pub(crate) fn compute_action_metrics(logits: &Tensor, labels: &[u32]) -> Result<ActionMetrics> {
    let rows = util::vec2_f32(logits)?;
    if labels.len() != rows.len() {
        anyhow::bail!(
            "action metric label count {} must match logits batch {}",
            labels.len(),
            rows.len()
        );
    }
    let n_classes = rows.first().map(|row| row.len()).unwrap_or(0).max(1);
    let mut confusion = vec![vec![0usize; n_classes]; n_classes];
    let mut correct = 0usize;
    let mut total = 0usize;

    for (row, &label) in rows.iter().zip(labels.iter()) {
        let pred = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        if pred == label {
            correct += 1;
        }
        if let Some(row_counts) = confusion.get_mut(label as usize) {
            if let Some(cell) = row_counts.get_mut(pred as usize) {
                *cell += 1;
            }
        }
        total += 1;
    }

    let mut recall_sum = 0.0f32;
    let mut f1_sum = 0.0f32;
    let mut present = 0usize;
    for label in 0..n_classes {
        let (_precision, recall, f1, rate) = class_prf(&confusion, label);
        if rate > 0.0 {
            recall_sum += recall;
            f1_sum += f1;
            present += 1;
        }
    }
    let balanced_accuracy = if present == 0 {
        0.0
    } else {
        recall_sum / present as f32
    };
    let macro_f1 = if present == 0 {
        0.0
    } else {
        f1_sum / present as f32
    };
    let (code_precision, code_recall, code_f1, code_rate) =
        class_prf(&confusion, ACTION_CODE as usize);
    let (done_precision, done_recall, done_f1, done_rate) =
        class_prf(&confusion, ACTION_DONE as usize);
    let (fetch_docs_precision, fetch_docs_recall, fetch_docs_f1, fetch_docs_rate) =
        class_prf(&confusion, ACTION_FETCH_DOCS as usize);

    Ok(ActionMetrics {
        accuracy: correct as f32 / total.max(1) as f32,
        balanced_accuracy,
        macro_f1,
        code_precision,
        code_recall,
        code_f1,
        code_rate,
        pred_code_rate: predicted_positive_rate(&confusion, ACTION_CODE as usize),
        done_precision,
        done_recall,
        done_f1,
        done_rate,
        pred_done_rate: predicted_positive_rate(&confusion, ACTION_DONE as usize),
        fetch_docs_precision,
        fetch_docs_recall,
        fetch_docs_f1,
        fetch_docs_rate,
        pred_fetch_docs_rate: predicted_positive_rate(&confusion, ACTION_FETCH_DOCS as usize),
    })
}

pub(crate) fn raw_examples_oov_rate(
    rows: &[RawWorldExample],
    vocab: &Vocab,
    mode: TokenizationMode,
) -> f32 {
    let mut total = 0usize;
    let mut oov = 0usize;
    for row in rows {
        let state_ids = encode_text_with_vocab_mode(&row.state_text, vocab, mode);
        let next_ids = encode_text_with_vocab_mode(&row.next_text, vocab, mode);
        total += state_ids.len() + next_ids.len();
        oov += state_ids
            .iter()
            .chain(next_ids.iter())
            .filter(|&&id| id == vocab.unk_id)
            .count();
    }
    if total == 0 {
        0.0
    } else {
        oov as f32 / total as f32
    }
}

pub(crate) fn encoded_examples_oov_rate(rows: &[WorldExample], unk_id: u32) -> f32 {
    let mut total = 0usize;
    let mut oov = 0usize;
    for row in rows {
        total += row.state_tokens.len() + row.next_tokens.len();
        oov += row
            .state_tokens
            .iter()
            .chain(row.next_tokens.iter())
            .filter(|&&id| id == unk_id)
            .count();
    }
    if total == 0 {
        0.0
    } else {
        oov as f32 / total as f32
    }
}

fn is_identifier_token(token: &str) -> bool {
    !token.is_empty()
        && token != "<num_lit>"
        && token != "<str_lit>"
        && token.chars().all(|ch| ch.is_ascii_alphanumeric())
        && token
            .chars()
            .next()
            .map(|ch| ch.is_ascii_alphabetic())
            .unwrap_or(false)
}

fn delimiter_balance_for_tokens(tokens: &[String]) -> bool {
    let mut round = 0i32;
    let mut square = 0i32;
    let mut curly = 0i32;
    for token in tokens {
        match token.as_str() {
            "(" => round += 1,
            ")" => round -= 1,
            "[" => square += 1,
            "]" => square -= 1,
            "{" => curly += 1,
            "}" => curly -= 1,
            _ => {}
        }
        if round < 0 || square < 0 || curly < 0 {
            return false;
        }
    }
    round == 0 && square == 0 && curly == 0
}

fn decode_active_tokens(ids: &[u32], mask: &[f32], vocab: &Vocab) -> Vec<String> {
    ids.iter()
        .zip(mask.iter())
        .filter_map(|(&id, &m)| {
            if m <= 0.0 {
                None
            } else {
                Some(
                    vocab
                        .id_to_token
                        .get(id as usize)
                        .cloned()
                        .unwrap_or_else(|| "<unk>".to_string()),
                )
            }
        })
        .collect()
}

fn is_syntax_token(token: &str) -> bool {
    matches!(
        token,
        "{" | "}"
            | "("
            | ")"
            | "["
            | "]"
            | ";"
            | ","
            | ":"
            | "="
            | "=="
            | "<nl>"
            | "<indent_tab>"
            | "func"
            | "package"
            | "import"
            | "struct"
            | "return"
            | "type"
            | "var"
            | "const"
            | "if"
            | "else"
            | "for"
            | "range"
            | "switch"
            | "case"
            | "default"
            | "map"
    )
}

fn syntax_weight_for_token(token: &str) -> f32 {
    if matches!(
        token,
        "{" | "}" | "(" | ")" | "[" | "]" | ";" | "func" | "struct" | "type"
    ) {
        2.0
    } else if is_syntax_token(token) {
        1.5
    } else {
        1.0
    }
}

fn is_type_like_token(token: &str) -> bool {
    matches!(
        token,
        "string"
            | "bool"
            | "byte"
            | "rune"
            | "error"
            | "int"
            | "int8"
            | "int16"
            | "int32"
            | "int64"
            | "uint"
            | "uint8"
            | "uint16"
            | "uint32"
            | "uint64"
            | "float32"
            | "float64"
            | "complex64"
            | "complex128"
    )
}

fn importance_weight_for_token(token: &str) -> f32 {
    if matches!(
        token,
        "func" | "type" | "struct" | ":" | "(" | ")" | "{" | "}" | "," | ":="
    ) {
        2.4
    } else if is_type_like_token(token) {
        2.0
    } else if is_syntax_token(token) {
        1.6
    } else {
        1.0
    }
}

pub(crate) fn syntax_weight_mask(
    target: &Tensor,
    mask: &Tensor,
    vocab: &Vocab,
    _device: &Device,
) -> Result<Tensor> {
    let target_ids = target.reshape((target.elem_count(),))?.to_vec1::<u32>()?;
    let mask_values = util::vec1_f32(&mask.reshape((mask.elem_count(),))?)?;
    let weights = target_ids
        .iter()
        .zip(mask_values.iter())
        .map(|(&id, &m)| {
            if m <= 0.0 {
                0.0
            } else {
                vocab
                    .id_to_token
                    .get(id as usize)
                    .map(|token| syntax_weight_for_token(token))
                    .unwrap_or(1.0)
            }
        })
        .collect::<Vec<_>>();
    let mask_like = mask.reshape((mask.elem_count(),))?;
    util::from_vec_like(weights, (target.elem_count(),), &mask_like)
}

fn signature_span_indices(ids: &[u32], mask: &[f32], vocab: &Vocab) -> Vec<usize> {
    let mut spans = Vec::new();
    let mut idx = 0usize;
    while idx < ids.len().min(mask.len()) {
        if mask[idx] <= 0.0 {
            idx += 1;
            continue;
        }
        let token = vocab
            .id_to_token
            .get(ids[idx] as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unk>");
        if !matches!(token, "func" | "type") {
            idx += 1;
            continue;
        }
        let start = idx;
        let go_type_decl = token == "type";
        let mut end = None;
        idx += 1;
        while idx < ids.len().min(mask.len()) {
            let inner = vocab
                .id_to_token
                .get(ids[idx] as usize)
                .map(|s| s.as_str())
                .unwrap_or("<unk>");
            if mask[idx] > 0.0
                && ((!go_type_decl && inner == "{") || (go_type_decl && inner == "}"))
            {
                end = Some(idx);
                idx += 1;
                break;
            }
            idx += 1;
        }
        if let Some(end) = end {
            spans.extend(start..=end);
        }
    }
    spans
}

fn function_name_span_indices(ids: &[u32], mask: &[f32], vocab: &Vocab) -> Vec<usize> {
    let mut seen_func = false;
    let mut positions = Vec::new();
    for (idx, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
        if m <= 0.0 {
            continue;
        }
        let token = vocab
            .id_to_token
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unk>");
        if !seen_func {
            if token == "func" {
                seen_func = true;
            }
            continue;
        }
        if token == "(" {
            break;
        }
        if token == "<nl>" {
            continue;
        }
        positions.push(idx);
    }
    positions
}

pub(crate) fn signature_weight_mask(
    target: &Tensor,
    mask: &Tensor,
    vocab: &Vocab,
    _device: &Device,
) -> Result<Tensor> {
    let target_rows = target.to_vec2::<u32>()?;
    let mask_rows = util::vec2_f32(mask)?;
    let seq_len = target.dim(1)?;
    let mut weights = Vec::with_capacity(target.elem_count());
    for (target_row, mask_row) in target_rows.iter().zip(mask_rows.iter()) {
        let signature_positions = signature_span_indices(target_row, mask_row, vocab)
            .into_iter()
            .collect::<HashSet<_>>();
        for (idx, &m) in mask_row.iter().enumerate().take(seq_len) {
            if m <= 0.0 {
                weights.push(0.0);
            } else if signature_positions.contains(&idx) {
                weights.push(2.5);
            } else {
                weights.push(1.0);
            }
        }
    }
    util::from_vec_like(weights, (target.elem_count(),), mask)
}

pub(crate) fn structure_weight_mask(
    target: &Tensor,
    mask: &Tensor,
    vocab: &Vocab,
    _device: &Device,
) -> Result<Tensor> {
    let target_rows = target.to_vec2::<u32>()?;
    let mask_rows = util::vec2_f32(mask)?;
    let seq_len = target.dim(1)?;
    let mut weights = Vec::with_capacity(target.elem_count());
    for (target_row, mask_row) in target_rows.iter().zip(mask_rows.iter()) {
        let signature_positions = signature_span_indices(target_row, mask_row, vocab)
            .into_iter()
            .collect::<HashSet<_>>();
        let name_positions = function_name_span_indices(target_row, mask_row, vocab)
            .into_iter()
            .collect::<HashSet<_>>();
        let last_brace = target_row
            .iter()
            .zip(mask_row.iter())
            .enumerate()
            .filter_map(|(idx, (&id, &m))| {
                if m <= 0.0 {
                    return None;
                }
                let token = vocab.id_to_token.get(id as usize).map(|s| s.as_str())?;
                if token == "}" {
                    Some(idx)
                } else {
                    None
                }
            })
            .next_back();
        for (idx, &m) in mask_row.iter().enumerate().take(seq_len) {
            if m <= 0.0 {
                weights.push(0.0);
            } else if name_positions.contains(&idx) {
                weights.push(4.0);
            } else if signature_positions.contains(&idx) {
                weights.push(2.8);
            } else if last_brace == Some(idx) {
                weights.push(2.2);
            } else {
                weights.push(1.0);
            }
        }
    }
    util::from_vec_like(weights, (target.elem_count(),), mask)
}

pub(crate) fn importance_weight_mask(
    target: &Tensor,
    mask: &Tensor,
    vocab: &Vocab,
    _device: &Device,
) -> Result<Tensor> {
    let target_rows = target.to_vec2::<u32>()?;
    let mask_rows = util::vec2_f32(mask)?;
    let seq_len = target.dim(1)?;
    let mut weights = Vec::with_capacity(target.elem_count());
    for (target_row, mask_row) in target_rows.iter().zip(mask_rows.iter()) {
        let signature_positions = signature_span_indices(target_row, mask_row, vocab)
            .into_iter()
            .collect::<HashSet<_>>();
        let name_positions = function_name_span_indices(target_row, mask_row, vocab)
            .into_iter()
            .collect::<HashSet<_>>();
        for (idx, (&id, &m)) in target_row
            .iter()
            .zip(mask_row.iter())
            .enumerate()
            .take(seq_len)
        {
            if m <= 0.0 {
                weights.push(0.0);
                continue;
            }
            let token = vocab
                .id_to_token
                .get(id as usize)
                .map(|s| s.as_str())
                .unwrap_or("<unk>");
            let weight = if name_positions.contains(&idx) {
                4.5
            } else if signature_positions.contains(&idx) {
                3.0
            } else {
                importance_weight_for_token(token)
            };
            weights.push(weight);
        }
    }
    util::from_vec_like(weights, (target.elem_count(),), mask)
}

fn go_function_skeleton_for_tokens(tokens: &[String]) -> bool {
    let has_func = tokens.iter().any(|token| token == "func");
    let has_parens =
        tokens.iter().any(|token| token == "(") && tokens.iter().any(|token| token == ")");
    let has_body =
        tokens.iter().any(|token| token == "{") && tokens.iter().any(|token| token == "}");
    has_func && has_parens && has_body && delimiter_balance_for_tokens(tokens)
}

pub(crate) fn decoder_prediction_metrics(
    logits: &Tensor,
    target: &Tensor,
    mask: &Tensor,
    vocab: &Vocab,
) -> Result<DecoderPredictionMetrics> {
    let pred = logits.argmax(candle_core::D::Minus1)?;
    let pred_rows = pred.to_vec2::<u32>()?;
    let target_rows = target.to_vec2::<u32>()?;
    let mask_rows = util::vec2_f32(mask)?;
    let mut total = 0usize;
    let mut correct = 0usize;
    let mut ident_total = 0usize;
    let mut ident_correct = 0usize;
    let mut balanced = 0usize;
    let mut syntax_total = 0usize;
    let mut syntax_correct = 0usize;
    let mut function_skeletons = 0usize;
    let mut signature_total = 0usize;
    let mut signature_correct = 0usize;
    let mut signature_exact = 0usize;
    let mut function_name_total = 0usize;
    let mut function_name_correct = 0usize;
    let mut function_name_exact = 0usize;
    for ((pred_row, target_row), mask_row) in pred_rows
        .iter()
        .zip(target_rows.iter())
        .zip(mask_rows.iter())
    {
        let signature_positions = signature_span_indices(target_row, mask_row, vocab);
        let function_name_positions = function_name_span_indices(target_row, mask_row, vocab);
        let mut row_signature_ok = !signature_positions.is_empty();
        let mut row_function_name_ok = !function_name_positions.is_empty();
        for (idx, ((&pred_id, &target_id), &m)) in pred_row
            .iter()
            .zip(target_row.iter())
            .zip(mask_row.iter())
            .enumerate()
        {
            if m <= 0.0 {
                continue;
            }
            total += 1;
            if pred_id == target_id {
                correct += 1;
            }
            let token = vocab
                .id_to_token
                .get(target_id as usize)
                .map(|s| s.as_str())
                .unwrap_or("<unk>");
            if is_identifier_token(token) {
                ident_total += 1;
                if pred_id == target_id {
                    ident_correct += 1;
                }
            }
            if is_syntax_token(token) {
                syntax_total += 1;
                if pred_id == target_id {
                    syntax_correct += 1;
                }
            }
            if signature_positions.contains(&idx) {
                signature_total += 1;
                if pred_id == target_id {
                    signature_correct += 1;
                } else {
                    row_signature_ok = false;
                }
            }
            if function_name_positions.contains(&idx) {
                function_name_total += 1;
                if pred_id == target_id {
                    function_name_correct += 1;
                } else {
                    row_function_name_ok = false;
                }
            }
        }
        let active_pred_tokens = decode_active_tokens(pred_row, mask_row, vocab);
        if delimiter_balance_for_tokens(&active_pred_tokens) {
            balanced += 1;
        }
        if go_function_skeleton_for_tokens(&active_pred_tokens) {
            function_skeletons += 1;
        }
        if row_signature_ok {
            signature_exact += 1;
        }
        if row_function_name_ok {
            function_name_exact += 1;
        }
    }

    let row_count = pred_rows.len().max(1) as f32;
    Ok(DecoderPredictionMetrics {
        token_accuracy: correct as f32 / total.max(1) as f32,
        identifier_accuracy: ident_correct as f32 / ident_total.max(1) as f32,
        delimiter_balance_rate: balanced as f32 / row_count,
        syntax_token_accuracy: syntax_correct as f32 / syntax_total.max(1) as f32,
        function_skeleton_rate: function_skeletons as f32 / row_count,
        signature_token_accuracy: signature_correct as f32 / signature_total.max(1) as f32,
        signature_exact_rate: signature_exact as f32 / row_count,
        function_name_token_accuracy: function_name_correct as f32
            / function_name_total.max(1) as f32,
        function_name_exact_rate: function_name_exact as f32 / row_count,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_world_encoded_batch(
    batch: &[WorldExample],
    encoder_vocab: &Vocab,
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    transition: &ActionStateTransition,
    action_classifier_head: &crate::model::NextActionClassifier,
    inverse_action_head: &crate::model::NextActionClassifier,
    max_seq: usize,
    lambda: f64,
    action_loss_weight: f64,
    inverse_loss_weight: f64,
    device: &Device,
) -> Result<WorldBatchMetrics> {
    let sigreg_slices = env_usize("TOFY_SIGREG_SLICES", 1024);
    let sigreg_points = env_usize("TOFY_SIGREG_POINTS", 17);
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 4);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_context_compressor = env_bool("TOFY_RECURSIVE_CONTEXT_COMPRESSION", false);
    let action_labels = batch.iter().map(|row| row.action_label).collect::<Vec<_>>();
    let (state_slots, next_slots) = crate::tasks::world::context_slots_from_world_pair_sequences(
        encoder,
        context_compressor,
        batch,
        encoder_vocab.pad_id,
        max_seq,
        context_segments,
        recent_full_segments,
        recursive_context_compressor,
        device,
    )?;
    let pred_slots = transition.forward(&state_slots, &action_labels)?;
    let fixed_next_slots = next_slots.detach();
    let pred_loss = prediction_loss(&pred_slots, &fixed_next_slots)?;
    let state_sigreg = crate::model::sigreg_epps_pulley(
        &flatten_latent_slots(&state_slots)?,
        sigreg_slices,
        sigreg_points,
    )?;
    let next_sigreg = crate::model::sigreg_epps_pulley(
        &flatten_latent_slots(&next_slots)?,
        sigreg_slices,
        sigreg_points,
    )?;
    let pred_sigreg = crate::model::sigreg_epps_pulley(
        &flatten_latent_slots(&pred_slots)?,
        sigreg_slices,
        sigreg_points,
    )?;
    let sigreg_loss = world_sigreg_loss(&state_sigreg, &next_sigreg, &pred_sigreg)?;
    let action_logits = action_classifier_head.forward(&state_slots)?;
    let action_loss = action_cross_entropy(&action_logits, &action_labels, device)?;
    let (inverse_loss, inverse_action_metrics) = if inverse_loss_weight > 0.0 {
        let true_delta_slots = slot_delta_slots(&fixed_next_slots, &state_slots.detach())?;
        let pred_delta_slots = slot_delta_slots(&pred_slots, &state_slots)?;
        let inverse_logits_true = inverse_action_head.forward(&true_delta_slots)?;
        let inverse_logits_pred = inverse_action_head.forward(&pred_delta_slots)?;
        let inverse_true_loss = action_cross_entropy(&inverse_logits_true, &action_labels, device)?;
        let inverse_pred_loss = action_cross_entropy(&inverse_logits_pred, &action_labels, device)?;
        (
            inverse_true_loss
                .broadcast_add(&inverse_pred_loss)?
                .affine(0.5, 0.0)?,
            compute_action_metrics(&inverse_logits_pred, &action_labels)?,
        )
    } else {
        (pred_loss.affine(0.0, 0.0)?, ActionMetrics::default())
    };
    let total_loss = pred_loss
        .broadcast_add(&sigreg_loss.affine(lambda, 0.0)?)?
        .broadcast_add(&action_loss.affine(action_loss_weight, 0.0)?)?
        .broadcast_add(&inverse_loss.affine(inverse_loss_weight, 0.0)?)?;
    Ok(WorldBatchMetrics {
        total_loss: util::scalar_f32(&total_loss)?,
        transition_loss: util::scalar_f32(&pred_loss)?,
        sigreg_loss: util::scalar_f32(&sigreg_loss)?,
        action_loss: util::scalar_f32(&action_loss)?,
        inverse_loss: util::scalar_f32(&inverse_loss)?,
        action_metrics: compute_action_metrics(&action_logits, &action_labels)?,
        inverse_action_metrics,
        transition_cosine: util::scalar_f32(&mean_cosine_similarity(&pred_slots, &next_slots)?)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_decoder_batch(
    raw_batch: &[RawWorldExample],
    encoder_vocab: &Vocab,
    decoder_vocab: &Vocab,
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    transition: &ActionStateTransition,
    decoder_conditioning_adapter: &DecoderConditioningAdapter,
    decoder: &CodeDecoder,
    decoder_kind: crate::model::DecoderKind,
    decoder_action_label: u32,
    compute_conditioning_metrics: bool,
    max_seq: usize,
    device: &Device,
) -> Result<DecoderBatchMetrics> {
    let encoder_batch = crate::data::encode_world_examples(raw_batch, encoder_vocab);
    let decoder_batch = crate::data::encode_world_examples_with_mode(
        raw_batch,
        decoder_vocab,
        decoder_tokenization_mode(decoder_kind),
    );
    let oov_rate = raw_examples_oov_rate(
        raw_batch,
        decoder_vocab,
        decoder_tokenization_mode(decoder_kind),
    );
    evaluate_decoder_encoded_batch(
        &encoder_batch,
        &decoder_batch,
        oov_rate,
        encoder_vocab,
        decoder_vocab,
        encoder,
        context_compressor,
        transition,
        decoder_conditioning_adapter,
        decoder,
        decoder_action_label,
        compute_conditioning_metrics,
        max_seq,
        device,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_decoder_cached_batch(
    cached_batch: &[crate::data::CachedDecoderExample],
    encoder_vocab: &Vocab,
    decoder_vocab: &Vocab,
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    transition: &ActionStateTransition,
    decoder_conditioning_adapter: &DecoderConditioningAdapter,
    decoder: &CodeDecoder,
    _decoder_kind: crate::model::DecoderKind,
    decoder_action_label: u32,
    compute_conditioning_metrics: bool,
    max_seq: usize,
    device: &Device,
) -> Result<DecoderBatchMetrics> {
    let encoder_batch = cached_batch
        .iter()
        .map(|row| row.encoder.clone())
        .collect::<Vec<_>>();
    let decoder_batch = cached_batch
        .iter()
        .map(|row| row.decoder.clone())
        .collect::<Vec<_>>();
    let oov_rate = encoded_examples_oov_rate(&decoder_batch, decoder_vocab.unk_id);
    evaluate_decoder_encoded_batch(
        &encoder_batch,
        &decoder_batch,
        oov_rate,
        encoder_vocab,
        decoder_vocab,
        encoder,
        context_compressor,
        transition,
        decoder_conditioning_adapter,
        decoder,
        decoder_action_label,
        compute_conditioning_metrics,
        max_seq,
        device,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_decoder_encoded_batch(
    encoder_batch: &[WorldExample],
    decoder_batch: &[WorldExample],
    oov_rate: f32,
    encoder_vocab: &Vocab,
    decoder_vocab: &Vocab,
    encoder: &OnlineEncoder,
    context_compressor: &ContextCompressor,
    transition: &ActionStateTransition,
    decoder_conditioning_adapter: &DecoderConditioningAdapter,
    decoder: &CodeDecoder,
    decoder_action_label: u32,
    compute_conditioning_metrics: bool,
    max_seq: usize,
    device: &Device,
) -> Result<DecoderBatchMetrics> {
    let context_segments = env_usize("TOFY_WORLD_CONTEXT_SEGMENTS", 4);
    let recent_full_segments = env_usize("TOFY_WORLD_RECENT_FULL_SEGMENTS", 1);
    let recursive_context_compressor = env_bool("TOFY_RECURSIVE_CONTEXT_COMPRESSION", false);
    let rollout_steps = env_usize("TOFY_WORLD_TRAIN_ROLLOUT_STEPS", 1);
    let state_tokens = encoder_batch
        .iter()
        .map(|row| row.state_tokens.as_slice())
        .collect::<Vec<_>>();
    let state_slots = context_slots_from_token_sequences(
        encoder,
        context_compressor,
        &state_tokens,
        encoder_vocab.pad_id,
        max_seq,
        context_segments,
        recent_full_segments,
        recursive_context_compressor,
        device,
    )?;
    let decoder_action_labels = vec![decoder_action_label; encoder_batch.len()];
    let steps = rollout_steps.max(1);
    let mut next_context_slots = state_slots;
    for _ in 0..steps {
        next_context_slots = transition.forward(&next_context_slots, &decoder_action_labels)?;
    }
    let adapter_action_labels = vec![decoder_action_label; decoder_batch.len()];
    let world_latent = decoder_conditioning_adapter
        .forward_with_actions(&next_context_slots.detach(), &adapter_action_labels)?;
    let (dec_input, dec_target, loss_mask) =
        make_decoder_batch_from_slice(decoder_batch, max_seq, decoder_vocab.pad_id, device)?;
    let logits = decoder.forward(&dec_input, &world_latent)?;
    let importance_mask = importance_weight_mask(&dec_target, &loss_mask, decoder_vocab, device)?;
    let raw_loss = masked_cross_entropy(&logits, &dec_target, &loss_mask)?;
    let syntax_mask = syntax_weight_mask(&dec_target, &loss_mask, decoder_vocab, device)?;
    let signature_mask = signature_weight_mask(&dec_target, &loss_mask, decoder_vocab, device)?;
    let structure_mask = structure_weight_mask(&dec_target, &loss_mask, decoder_vocab, device)?;
    let loss = masked_weighted_cross_entropy(&logits, &dec_target, &importance_mask)?;
    let raw_loss_val = util::scalar_f32(&raw_loss)?;
    let syntax_loss = masked_weighted_cross_entropy(&logits, &dec_target, &syntax_mask)?;
    let signature_loss = masked_weighted_cross_entropy(&logits, &dec_target, &signature_mask)?;
    let structure_loss = masked_weighted_cross_entropy(&logits, &dec_target, &structure_mask)?;
    let loss_val = util::scalar_f32(&loss)?;
    let ablated_loss_val = if compute_conditioning_metrics {
        let zero_world_latent = world_latent.affine(0.0, 0.0)?;
        let ablated_logits = decoder.forward(&dec_input, &zero_world_latent)?;
        let ablated_loss = masked_cross_entropy(&ablated_logits, &dec_target, &loss_mask)?;
        util::scalar_f32(&ablated_loss)?
    } else {
        loss_val
    };
    let shuffled_loss_val = if compute_conditioning_metrics {
        let shuffled_world_latent = shuffled_conditioning_latent(&world_latent)?;
        let shuffled_logits = decoder.forward(&dec_input, &shuffled_world_latent)?;
        let shuffled_loss = masked_cross_entropy(&shuffled_logits, &dec_target, &loss_mask)?;
        util::scalar_f32(&shuffled_loss)?
    } else {
        loss_val
    };
    let hard_mismatch_loss_val = if compute_conditioning_metrics {
        let hard_mismatch_world_latent = hard_mismatched_conditioning_latent(&world_latent)?;
        let hard_mismatch_logits = decoder.forward(&dec_input, &hard_mismatch_world_latent)?;
        let hard_mismatch_loss =
            masked_cross_entropy(&hard_mismatch_logits, &dec_target, &loss_mask)?;
        util::scalar_f32(&hard_mismatch_loss)?
    } else {
        loss_val
    };
    let syntax_loss_val = util::scalar_f32(&syntax_loss)?;
    let signature_loss_val = util::scalar_f32(&signature_loss)?;
    let structure_loss_val = util::scalar_f32(&structure_loss)?;
    let active_tokens = util::scalar_f32(&loss_mask.sum_all()?)?;
    let total_tokens = (decoder_batch.len().max(1) * max_seq * 2) as f32;
    let prediction_metrics =
        decoder_prediction_metrics(&logits, &dec_target, &loss_mask, decoder_vocab)?;
    let gains = decoder_conditioning_gains(
        raw_loss_val,
        ablated_loss_val,
        shuffled_loss_val,
        hard_mismatch_loss_val,
        compute_conditioning_metrics,
    );
    Ok(DecoderBatchMetrics {
        loss: loss_val,
        raw_loss: raw_loss_val,
        ablated_loss: ablated_loss_val,
        conditioning_gain: gains.conditioning_gain,
        zero_gain: gains.zero_gain,
        shuffled_loss: shuffled_loss_val,
        shuffle_gain: gains.shuffle_gain,
        hard_negative_gain: gains.hard_negative_gain,
        syntax_loss: syntax_loss_val,
        signature_loss: signature_loss_val,
        structure_loss: structure_loss_val,
        perplexity: raw_loss_val.exp(),
        active_tokens,
        active_frac: active_tokens / total_tokens.max(1.0),
        world_rms: util::scalar_f32(&tensor_rms(&world_latent)?)?,
        oov_rate,
        token_accuracy: prediction_metrics.token_accuracy,
        identifier_accuracy: prediction_metrics.identifier_accuracy,
        delimiter_balance_rate: prediction_metrics.delimiter_balance_rate,
        syntax_token_accuracy: prediction_metrics.syntax_token_accuracy,
        function_skeleton_rate: prediction_metrics.function_skeleton_rate,
        signature_token_accuracy: prediction_metrics.signature_token_accuracy,
        signature_exact_rate: prediction_metrics.signature_exact_rate,
        function_name_token_accuracy: prediction_metrics.function_name_token_accuracy,
        function_name_exact_rate: prediction_metrics.function_name_exact_rate,
    })
}

fn label_smoothing_eps() -> f32 {
    std::env::var("TOFY_LABEL_SMOOTHING")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.05)
        .clamp(0.0, 0.2)
}

pub(crate) fn masked_weighted_cross_entropy(
    logits: &Tensor,
    target: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    let logits_flat = logits.reshape((b * t, v))?;
    let target_flat = target
        .reshape((b * t,))?
        .to_dtype(candle_core::DType::U32)?;
    let log_probs = ops::log_softmax(&logits_flat, candle_core::D::Minus1)?;
    let gathered = log_probs
        .gather(&target_flat.unsqueeze(1)?, 1)?
        .squeeze(1)?
        .affine(-1.0, 0.0)?;
    let eps = label_smoothing_eps();
    let nll_per = if eps > 0.0 {
        let uniform = log_probs.mean(candle_core::D::Minus1)?.affine(-1.0, 0.0)?;
        gathered
            .affine((1.0 - eps) as f64, 0.0)?
            .broadcast_add(&uniform.affine(eps as f64, 0.0)?)?
    } else {
        gathered
    };
    let mask_flat = mask.reshape((b * t,))?.to_dtype(nll_per.dtype())?;
    let sum_nll = (nll_per.broadcast_mul(&mask_flat)?).sum_all()?;
    let sum_mask = mask_flat.sum_all()?.clamp(1e-8, 1e10)?;
    Ok(sum_nll.broadcast_div(&sum_mask)?)
}

pub(crate) fn masked_cross_entropy(
    logits: &Tensor,
    target: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    masked_weighted_cross_entropy(logits, target, mask)
}

pub(crate) fn forbidden_output_probability_loss(
    logits: &Tensor,
    mask: &Tensor,
    vocab: &Vocab,
) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    let forbidden_ids = vocab
        .id_to_token
        .iter()
        .enumerate()
        .filter_map(|(idx, token)| forbidden_decoder_token(token).then_some(idx as u32))
        .collect::<Vec<_>>();
    if forbidden_ids.is_empty() {
        return logits.sum_all()?.affine(0.0, 0.0).map_err(Into::into);
    }
    let rows = b * t;
    let probs = ops::softmax(&logits.reshape((rows, v))?, candle_core::D::Minus1)?;
    let mut gather_ids = Vec::with_capacity(rows * forbidden_ids.len());
    for _ in 0..rows {
        gather_ids.extend(forbidden_ids.iter().copied());
    }
    let ids = Tensor::from_vec(gather_ids, (rows, forbidden_ids.len()), logits.device())?;
    let forbidden_prob = probs.gather(&ids, 1)?.sum(1)?;
    let mask_flat = mask.reshape((rows,))?.to_dtype(forbidden_prob.dtype())?;
    let weighted = forbidden_prob.broadcast_mul(&mask_flat)?.sum_all()?;
    let normalizer = mask_flat.sum_all()?.clamp(1e-8, 1e10)?;
    weighted.broadcast_div(&normalizer).map_err(Into::into)
}

fn forbidden_decoder_token(token: &str) -> bool {
    if matches!(token, "<pad>" | "<unk>" | "<mask>") {
        return true;
    }
    if CODE_CONTROL_TOKENS
        .iter()
        .any(|control| *control != CODE_EOS_TOKEN && token == *control)
    {
        return true;
    }
    if token.contains("```") {
        return true;
    }
    matches!(
        token,
        "Here"
            | "here"
            | "explanation"
            | "Explanation"
            | "markdown"
            | "Markdown"
            | "assistant"
            | "Assistant"
            | "user"
            | "User"
            | "Compiler"
            | "compiler"
            | "feedback"
            | "Feedback"
            | "Previous"
            | "previous"
            | "attempt"
            | "Attempt"
            | "corrected"
            | "Corrected"
    )
}

pub(crate) fn decoder_selection_score(
    metrics: &DecoderBatchMetrics,
    syntax_loss_weight: f64,
    signature_loss_weight: f64,
    structure_loss_weight: f64,
) -> f32 {
    let reward_proxy = decoder_reward_proxy(metrics);
    metrics.loss
        + 0.20 * (0.05 - metrics.conditioning_gain).max(0.0)
        + 0.25 * (0.05 - metrics.shuffle_gain).max(0.0)
        + 0.25 * (0.05 - metrics.hard_negative_gain).max(0.0)
        + (syntax_loss_weight as f32 * 0.5 * metrics.syntax_loss)
        + (signature_loss_weight as f32 * 0.5 * metrics.signature_loss)
        + (structure_loss_weight as f32 * 0.7 * metrics.structure_loss)
        - 0.08 * metrics.syntax_token_accuracy
        - 0.08 * metrics.signature_token_accuracy
        - 0.10 * metrics.function_name_token_accuracy
        - 0.06 * metrics.delimiter_balance_rate
        - 0.06 * metrics.function_skeleton_rate
        - 0.08 * metrics.signature_exact_rate
        - 0.12 * metrics.function_name_exact_rate
        - 0.04 * metrics.conditioning_gain.max(0.0)
        - 0.04 * metrics.shuffle_gain.max(0.0)
        - 0.04 * metrics.hard_negative_gain.max(0.0)
        - 0.35 * reward_proxy
}

pub(crate) fn decoder_reward_proxy(metrics: &DecoderBatchMetrics) -> f32 {
    let syntax_quality = 0.20 * metrics.syntax_token_accuracy
        + 0.15 * metrics.delimiter_balance_rate
        + 0.15 * metrics.function_skeleton_rate;
    let signature_quality = 0.15 * metrics.signature_token_accuracy
        + 0.10 * metrics.signature_exact_rate
        + 0.10 * metrics.function_name_token_accuracy
        + 0.10 * metrics.function_name_exact_rate;
    let token_quality = 0.10 * metrics.token_accuracy + 0.05 * metrics.identifier_accuracy;
    let conditioning_quality = 0.05
        * (metrics
            .conditioning_gain
            .min(metrics.shuffle_gain)
            .min(metrics.hard_negative_gain)
            .max(0.0)
            / 0.25)
            .clamp(0.0, 1.0);
    (syntax_quality + signature_quality + token_quality + conditioning_quality).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::{
        action_cross_entropy, compute_action_metrics, decoder_conditioning_gains,
        decoder_reward_proxy, forbidden_decoder_token, function_name_span_indices,
        signature_span_indices,
    };
    use crate::data::CODE_EOS_TOKEN;
    use crate::model::Vocab;
    use candle_core::{DType, Device, Tensor};

    fn vocab_with(tokens: &[&str]) -> (Vocab, Vec<u32>) {
        let mut vocab = Vocab::new();
        let ids = tokens
            .iter()
            .map(|token| vocab.add_token(token))
            .collect::<Vec<_>>();
        (vocab, ids)
    }

    #[test]
    fn forbidden_decoder_tokens_keep_code_eos_available() {
        assert!(forbidden_decoder_token("<pad>"));
        assert!(forbidden_decoder_token("<unk>"));
        assert!(forbidden_decoder_token("<mask>"));
        assert!(forbidden_decoder_token("<fim_prefix>"));
        assert!(forbidden_decoder_token("<fim_suffix>"));
        assert!(forbidden_decoder_token("<fim_middle>"));
        assert!(!forbidden_decoder_token(CODE_EOS_TOKEN));
    }

    #[test]
    fn action_cross_entropy_rejects_label_batch_mismatch() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::zeros((2, 4), DType::F32, &device)?;

        let err = action_cross_entropy(&logits, &[0], &device)
            .expect_err("label count mismatch should fail");

        assert!(err.to_string().contains("action label count"));
        Ok(())
    }

    #[test]
    fn action_metrics_reject_label_batch_mismatch() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::zeros((2, 4), DType::F32, &device)?;

        let err = match compute_action_metrics(&logits, &[0]) {
            Ok(_) => anyhow::bail!("label count mismatch should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("action metric label count"));
        Ok(())
    }

    #[test]
    fn signature_spans_include_go_type_and_func_declarations() {
        let tokens = [
            "type",
            "Interval",
            "struct",
            "{",
            "Start",
            "int",
            "}",
            "func",
            "MergeIntervals",
            "(",
            ")",
            "{",
            "return",
            "nil",
            "}",
        ];
        let (vocab, ids) = vocab_with(&tokens);
        let mask = vec![1.0; ids.len()];

        let spans = signature_span_indices(&ids, &mask, &vocab);

        assert!(spans.contains(&0));
        assert!(spans.contains(&6));
        assert!(spans.contains(&7));
        assert!(spans.contains(&11));
        assert!(!spans.contains(&12));
    }

    #[test]
    fn function_name_span_handles_go_func_keyword() {
        let tokens = ["func", "ParseSize", "(", "input", "string", ")", "{"];
        let (vocab, ids) = vocab_with(&tokens);
        let mask = vec![1.0; ids.len()];

        let positions = function_name_span_indices(&ids, &mask, &vocab);

        assert_eq!(positions, vec![1]);
    }

    #[test]
    fn decoder_reward_proxy_prefers_structurally_correct_predictions() {
        let weak = super::DecoderBatchMetrics {
            loss: 2.0,
            raw_loss: 2.0,
            ablated_loss: 2.0,
            conditioning_gain: 0.0,
            zero_gain: 0.0,
            shuffled_loss: 2.0,
            shuffle_gain: 0.0,
            hard_negative_gain: 0.0,
            syntax_loss: 2.0,
            signature_loss: 2.0,
            structure_loss: 2.0,
            perplexity: 7.0,
            active_tokens: 1.0,
            active_frac: 0.5,
            world_rms: 1.0,
            oov_rate: 0.2,
            token_accuracy: 0.2,
            identifier_accuracy: 0.1,
            delimiter_balance_rate: 0.2,
            syntax_token_accuracy: 0.2,
            function_skeleton_rate: 0.0,
            signature_token_accuracy: 0.1,
            signature_exact_rate: 0.0,
            function_name_token_accuracy: 0.0,
            function_name_exact_rate: 0.0,
        };
        let weak_score = decoder_reward_proxy(&weak);
        let strong = super::DecoderBatchMetrics {
            token_accuracy: 0.9,
            identifier_accuracy: 0.8,
            delimiter_balance_rate: 1.0,
            syntax_token_accuracy: 0.9,
            function_skeleton_rate: 1.0,
            signature_token_accuracy: 0.85,
            signature_exact_rate: 0.7,
            function_name_token_accuracy: 0.8,
            function_name_exact_rate: 0.7,
            conditioning_gain: 0.2,
            shuffle_gain: 0.2,
            hard_negative_gain: 0.2,
            ..weak
        };

        assert!(decoder_reward_proxy(&strong) > weak_score);
    }

    #[test]
    fn decoder_conditioning_gains_compare_against_raw_ce() {
        let gains = decoder_conditioning_gains(2.0, 2.6, 2.4, 2.3, true);

        assert!((gains.conditioning_gain - 0.6).abs() < 1e-6);
        assert!((gains.shuffle_gain - 0.4).abs() < 1e-6);
        assert!((gains.hard_negative_gain - 0.3).abs() < 1e-6);
        assert_eq!(
            decoder_conditioning_gains(2.0, 2.6, 2.4, 2.3, false).conditioning_gain,
            0.0
        );
    }
}
