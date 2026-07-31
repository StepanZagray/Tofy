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
    if tasks::veclab::try_run(args)? {
        return Ok(());
    }
    if tasks::probe::try_run(args)? {
        return Ok(());
    }
    if tasks::knowledge::try_run_train(args)? {
        return Ok(());
    }
    if tasks::bridge::try_run_logit_parity(args)? {
        return Ok(());
    }
    if tasks::bridge::try_run_train_bridge(args)? {
        return Ok(());
    }
    if tasks::bridge::try_run_eval_bridge(args)? {
        return Ok(());
    }
    if tasks::eval::try_run_code_eval(args)? {
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
        "specify a mode: train / prepare cache / --prepare-veclab / --print-split-stats / --prepare-pipeline-cache / --prepare-encoder-corpus / --latent / --latent-from-checkpoint / --eval-jepa / --train-world-knowledge / --train-bridge / --eval-bridge / --train-channel-probe / --run-conditioning-corruption-probe"
    );
}
