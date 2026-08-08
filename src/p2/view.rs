//! Unified P2 application/GPU evidence visualizer (`cargo p2-view`).

use anyhow::{bail, Context, Result};
use candle_graph::cli::trace_cli;
use clap::Parser;
use std::path::PathBuf;

/// Render HTML from a representative-update bundle or its application trace.
#[derive(Debug, Parser)]
pub struct P2ViewArgs {
    /// Profile bundle directory or `application.jsonl` (`candle-graph/trace/6`).
    #[arg(value_name = "PROFILE")]
    pub profile: PathBuf,

    /// Output HTML file.
    #[arg(long, value_name = "FILE")]
    pub output: PathBuf,

    /// Explicit comparison baseline trace.
    #[arg(long, value_name = "TRACE")]
    pub baseline: Option<PathBuf>,

    /// Override normalized Nsight CSV directory.
    #[arg(long, value_name = "DIR")]
    pub nsight_dir: Option<PathBuf>,
}

pub fn run_p2_view(args: P2ViewArgs) -> Result<()> {
    let trace = if args.profile.is_dir() {
        args.profile.join("application.jsonl")
    } else {
        args.profile.clone()
    };
    if !trace.is_file() {
        bail!("trace not found: {}", trace.display());
    }
    let inferred_nsight = args.profile.is_dir().then(|| args.profile.join("nsight"));
    let nsight_dir = args.nsight_dir.as_deref().or(inferred_nsight.as_deref());
    trace_cli::run_view(&trace, &args.output, args.baseline.as_deref(), nsight_dir)
        .with_context(|| format!("render HTML from {}", trace.display()))?;
    eprintln!("wrote {}", args.output.display());
    Ok(())
}
