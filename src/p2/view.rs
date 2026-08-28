//! Unified P2 application/GPU evidence visualizer (`cargo p2-view`).

use anyhow::{bail, Context, Result};
use candle_graph::cli::trace_cli;
use clap::Parser;
use std::path::PathBuf;

/// Render HTML from a representative-update bundle or a readable raw trace.
#[derive(Debug, Parser)]
pub struct P2ViewArgs {
    /// Finalized profile bundle directory or raw trace; use `candle-graph protocol` for schemas.
    #[arg(value_name = "PROFILE")]
    pub profile: PathBuf,

    /// Output HTML file.
    #[arg(long, value_name = "FILE")]
    pub output: PathBuf,

    /// Nsight CSV directory for a raw-trace input.
    #[arg(long, value_name = "DIR")]
    pub nsight_dir: Option<PathBuf>,
}

pub fn run_p2_view(args: P2ViewArgs) -> Result<()> {
    let input = if args.profile.is_dir() {
        let manifest = args.profile.join("bundle.json");
        if !manifest.is_file() {
            bail!("bundle manifest not found: {}", manifest.display());
        }
        if args.nsight_dir.is_some() {
            bail!("--nsight-dir cannot augment an already-finalized bundle");
        }
        args.profile.clone()
    } else {
        args.profile.clone()
    };
    if !input.exists() {
        bail!("profile input not found: {}", input.display());
    }
    trace_cli::run_view(&input, &args.output, args.nsight_dir.as_deref())
        .with_context(|| format!("render HTML from {}", input.display()))?;
    eprintln!("wrote {}", args.output.display());
    Ok(())
}
