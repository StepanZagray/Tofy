mod cli;
mod config;
mod data;
mod model;
mod tasks;
mod util;

use anyhow::{bail, Result};
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .init();

    let args: Vec<String> = std::env::args().collect();

    if tasks::pipeline::try_run_pipeline(&args)? {
        return Ok(());
    }
    if tasks::latent::try_run_prepare_ultrachat(&args)? {
        return Ok(());
    }
    if tasks::prepare::try_run_prepare(&args)? {
        return Ok(());
    }
    if tasks::world::try_run_train(&args)? {
        return Ok(());
    }
    if tasks::world::try_run_train_high_world(&args)? {
        return Ok(());
    }
    if tasks::world::try_run_train_orchestrator(&args)? {
        return Ok(());
    }
    if tasks::world::try_run_train_decoder(&args)? {
        return Ok(());
    }
    if tasks::world::try_run_eval(&args)? {
        return Ok(());
    }
    if tasks::eval::try_run_code_eval(&args)? {
        return Ok(());
    }
    if tasks::eval::try_run_decoder_only_eval(&args)? {
        return Ok(());
    }
    if tasks::world::try_run_serve(&args)? {
        return Ok(());
    }
    if tasks::latent::try_run_eval(&args)? {
        return Ok(());
    }
    if tasks::latent::try_run_train(&args)? {
        return Ok(());
    }

    let program = args.first().map(String::as_str).unwrap_or("jepa_ai");
    cli::print_usage(program);
    bail!(
        "specify a mode: train / --prepare-ultrachat / --prepare-encoder-corpus / --prepare-github-top-code / --prepare-rust-function-tasks / --prepare-rust-repair-tasks / --prepare-world-mix / --prepare-code-poc-mix / --prepare-expert-pairs / --prepare-casual-conversation / --generate-code-eval-suite / --convert-jsonl-context-response-to-tsv / --check-dtype-discipline / --sustained-oom-probe / --max-vram-probe / --latent / --latent-from-checkpoint / --eval-jepa / --train-world / --train-high-world / --train-orchestrator / --train-decoder / --eval-world / --eval-code-assistant / --eval-decoder-only / --serve"
    );
}
