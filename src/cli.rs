use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::data::{ensure_hub_dataset_cached, ensure_hub_wikipedia_cached};

#[derive(Debug, Clone)]
pub struct ResolvedDataPath {
    pub path: PathBuf,
    pub is_wikipedia: bool,
}

pub fn resolve_data_path(data_arg: &str) -> Result<ResolvedDataPath> {
    if !data_arg.starts_with("hub:") {
        return Ok(ResolvedDataPath {
            path: PathBuf::from(data_arg),
            is_wikipedia: false,
        });
    }

    let dataset_id = data_arg.strip_prefix("hub:").unwrap_or(data_arg);
    let is_wikipedia = dataset_id.to_ascii_lowercase().contains("wikipedia");
    let cache_dir = std::env::var("TOFY_HUB_CACHE_DIR")
        .or_else(|_| std::env::var("TOFY_DATA_DIR").map(|dir| format!("{dir}/hub")))
        .unwrap_or_else(|_| "data".to_string());
    let cache_dir = Path::new(&cache_dir);
    let path = if is_wikipedia {
        ensure_hub_wikipedia_cached(dataset_id, cache_dir)?
    } else {
        ensure_hub_dataset_cached(dataset_id, cache_dir)?
    };

    Ok(ResolvedDataPath { path, is_wikipedia })
}

pub fn print_usage(program: &str) {
    eprintln!("usage (choose one):");
    eprintln!("  Training (learn from data):");
    eprintln!(
        "    {program} train <8gb|48gb> [--resume [latest|run_id|runs/path]] [--with-code-eval]"
    );
    eprintln!(
        "    {program} --latent <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [max_vocab] [max_spans] [max_span_len] [max_masked_ratio] [lambda] [--grad-accum <int>] [--output <path>] [--resume]"
    );
    eprintln!(
        "    {program} --latent-from-checkpoint <encoder_checkpoint.safetensors> <data_path> [steps] ..."
    );
    eprintln!("  Evaluation (JEPA-native):");
    eprintln!(
        "    {program} --eval-jepa <model_path> <vocab_path> <data_path|hub:dataset_id> [eval_steps] [batch] [dim] [max_seq] [num_layers] [num_heads]"
    );
    eprintln!("  World model agent:");
    eprintln!(
        "    {program} --prepare-ultrachat [output_path] [context_window] [min_tokens] [max_rows]"
    );
    eprintln!(
        "    {program} --prepare-encoder-corpus|--prepare-github-top-code|--prepare-rust-function-tasks|--prepare-rust-repair-tasks|--prepare-world-mix|--prepare-code-poc-mix|--prepare-rust-by-practice ..."
    );
    eprintln!(
        "    {program} --prepare-expert-pairs|--prepare-casual-conversation|--generate-code-eval-suite|--convert-jsonl-context-response-to-tsv ..."
    );
    eprintln!(
        "    {program} --check-dtype-discipline | --sustained-oom-probe ... | --max-vram-probe [--profile 48gb] ..."
    );
    eprintln!(
        "    {program} --train-world <encoder_model.safetensors> <encoder_vocab.txt> <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--lambda <float>] [--lr <float>] [--grad-accum <int>] [--output <path>] [--resume]"
    );
    eprintln!(
        "    {program} --train-high-world <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--macro-min-len N] [--macro-max-len N] [--output <path>] [--resume]"
    );
    eprintln!(
        "    {program} --train-orchestrator <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--lr <float>] [--grad-accum <int>] [--freeze-planner] [--output <path>] [--resume]"
    );
    eprintln!(
        "    {program} --train-decoder <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:id> [steps] ... [--decoder-kind <text|code>] [--decoder-vocab <path>] [--decoder-max-vocab <int>] [--lr <float>] [--conditioning-loss-weight <float>] [--init-decoder <path>] [--decoder-output <path>] [--resume]"
    );
    eprintln!(
        "    {program} --eval-world <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <data_path|hub:dataset_id> [eval_steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots]"
    );
    eprintln!(
        "    {program} --eval-code-assistant <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> <suite.jsonl> [max_new_tokens] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--high-world-model <override>] [--code-decoder <path>] [--code-decoder-vocab <path>] [--ablate-conditioning]"
    );
    eprintln!(
        "    {program} --eval-decoder-only <decoder.safetensors> <decoder_vocab.txt> <suite.jsonl> [max_new_tokens] [planner_dim] [num_planner_slots] [--candidates <int>]"
    );
    eprintln!(
        "    {program} --serve <encoder_model.safetensors> <encoder_vocab.txt> <world_model.safetensors> [bind] [dim] [max_seq] [num_layers] [num_heads] [planner_dim] [num_planner_slots] [--high-world-model <override>] [--debug]"
    );
}
