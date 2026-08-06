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
    eprintln!("  Training pipeline:");
    eprintln!(
        "    {program} train <minimal|48gb|80gb> [--until full] [--resume [latest|run_id|runs/path]] [--skip-trained STAGE[,STAGE...]]"
    );
    eprintln!(
        "    {program} prepare cache <minimal|48gb|80gb> [--force] [--auto-hf-upload --hf-dataset <org/dataset-name>]"
    );
    eprintln!(
        "    {program} --latent <data_path|hub:dataset_id> [steps] [batch] [dim] [max_seq] [num_layers] [num_heads] [max_vocab] [max_spans] [max_span_len] [max_masked_ratio] [lambda] [--grad-accum <int>] [--output <path>] [--resume]"
    );
    eprintln!(
        "    {program} --latent-from-checkpoint <encoder_checkpoint.safetensors> <data_path> [steps] ..."
    );
    eprintln!("  Evaluation:");
    eprintln!(
        "    {program} --eval-jepa <model_path> <vocab_path> <data_path|hub:dataset_id> [eval_steps] [batch] [dim] [max_seq] [num_layers] [num_heads]"
    );
    eprintln!(
        "    {program} --eval-bridge <qwen_dir> <bridge.safetensors> <encoder.safetensors> <vocab.txt> <world.safetensors> <suite.jsonl> [report.json]"
    );
    eprintln!("    {program} --check-bridge-logit-parity <qwen_dir> [prompt]");
    eprintln!("  Data prep:");
    eprintln!(
        "    {program} --prepare-veclab | --print-split-stats | --prepare-encoder-corpus | --prepare-pipeline-cache ..."
    );
    eprintln!(
        "    {program} --train-world-knowledge <encoder_model.safetensors> <encoder_vocab.txt> <data_path> [steps] ..."
    );
    eprintln!(
        "    {program} --train-bridge <qwen_dir> <encoder.safetensors> <encoder_vocab.txt> <world.safetensors> <tasks.txt> [steps] [batch] [output]"
    );
    eprintln!("    {program} --train-channel-probe <qwen_dir> <bridge> <encoder> <vocab> <world> <seen_tasks> <heldout_tasks> <output> [steps]");
    eprintln!(
        "    {program} --check-dtype-discipline | --sustained-oom-probe ... | --max-vram-probe [--profile 48gb|80gb] ..."
    );
}
