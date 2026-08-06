//! P2 recursive world-model experiment CLI.
//!
//! The P1 exact-simulator harness (`p1a` / `p1b` / `p1c` / `p1c-hard`) lives only on
//! the `p1` git branch.

use crate::p2::cli::{
    run_p2_arc3_eval, run_p2_eval, run_p2_train, P2Arc3EvalArgs, P2EvalArgs, P2TrainArgs,
};
use crate::p2::view::{run_p2_view, P2ViewArgs};
use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "tofy",
    about = "P2 recursive world-model experiments for hidden-objective discovery and planning"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// P2: train the recursive world model through synthetic curriculum lessons
    #[command(name = "p2-train")]
    P2Train(P2TrainArgs),
    /// P2: held-out synthetic world-model and PTRM evaluation
    #[command(name = "p2-eval")]
    P2Eval(P2EvalArgs),
    /// P2: held-out transfer evaluation on official-toolkit ARC recordings
    #[command(name = "p2-arc3-eval")]
    P2Arc3Eval(P2Arc3EvalArgs),
    /// P2: HTML execution-graph visualizer from `profile.jsonl`
    #[command(name = "p2-view")]
    P2View(P2ViewArgs),
}

/// Entry point used by `main`.
pub fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::P2Train(args) => run_p2_train(args)?,
        Commands::P2Eval(args) => run_p2_eval(args)?,
        Commands::P2Arc3Eval(args) => run_p2_arc3_eval(args)?,
        Commands::P2View(args) => run_p2_view(args)?,
    }
    Ok(())
}
