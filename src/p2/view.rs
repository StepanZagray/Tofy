//! P2 HTML visualizer from a candle-graph trace (`cargo p2-view`).

use anyhow::{bail, Context, Result};
use candle_graph::cli::trace_cli;
use clap::Parser;
use std::path::PathBuf;

/// Render `model.html` from a training `profile.jsonl` trace.
#[derive(Debug, Parser)]
pub struct P2ViewArgs {
    /// Trace JSONL file (`candle-graph/trace/4`), usually `<output-dir>/profile.jsonl`.
    #[arg(value_name = "TRACE")]
    pub trace: PathBuf,

    /// Output HTML file.
    #[arg(long, value_name = "FILE")]
    pub output: PathBuf,
}

pub fn run_p2_view(args: P2ViewArgs) -> Result<()> {
    if !args.trace.is_file() {
        bail!("trace not found: {}", args.trace.display());
    }
    trace_cli::run_view(&args.trace, &args.output)
        .with_context(|| format!("render HTML from {}", args.trace.display()))?;
    eprintln!("wrote {}", args.output.display());
    Ok(())
}
