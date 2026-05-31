pub mod cli;
pub mod config;
pub mod data;
pub mod model;
pub mod tasks;
pub mod util;

use anyhow::{bail, Result};

pub fn run(args: &[String]) -> Result<()> {
    if tasks::pipeline::try_run_pipeline(args)? {
        return Ok(());
    }
    if tasks::pipeline::try_run_prepare_cache(args)? {
        return Ok(());
    }
    if tasks::latent::try_run_prepare_ultrachat(args)? {
        return Ok(());
    }
    if tasks::cache::try_run_prepare_pipeline_cache(args)? {
        return Ok(());
    }
    if tasks::prepare::try_run_prepare(args)? {
        return Ok(());
    }
    if tasks::world::try_run_train(args)? {
        return Ok(());
    }
    if tasks::world::try_run_train_high_world(args)? {
        return Ok(());
    }
    if tasks::world::try_run_train_orchestrator(args)? {
        return Ok(());
    }
    if tasks::world::try_run_train_decoder(args)? {
        return Ok(());
    }
    if tasks::world::try_run_eval(args)? {
        return Ok(());
    }
    if tasks::eval::try_run_code_eval(args)? {
        return Ok(());
    }
    if tasks::eval::try_run_decoder_only_eval(args)? {
        return Ok(());
    }
    if tasks::world::try_run_serve(args)? {
        return Ok(());
    }
    if tasks::latent::try_run_eval(args)? {
        return Ok(());
    }
    if tasks::latent::try_run_train(args)? {
        return Ok(());
    }

    let program = args.first().map(String::as_str).unwrap_or("jepa_ai");
    cli::print_usage(program);
    bail!(
        "specify a mode: train / prepare cache / --prepare-pipeline-cache / --prepare-ultrachat / --prepare-encoder-corpus / --prepare-github-top-code / --prepare-go-function-tasks / --prepare-go-algorithm-tasks / --prepare-go-semantics-tasks / --prepare-go-repair-tasks / --prepare-world-mix / --prepare-code-poc-mix / --prepare-expert-pairs / --prepare-casual-conversation / --generate-code-eval-suite / --generate-go-code-eval-suite / --convert-jsonl-context-response-to-tsv / --check-dtype-discipline / --sustained-oom-probe / --max-vram-probe / --latent / --latent-from-checkpoint / --eval-jepa / --train-world / --train-high-world / --train-orchestrator / --train-decoder / --eval-world / --eval-code-assistant / --eval-decoder-only / --serve"
    );
}
