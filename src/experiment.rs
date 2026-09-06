//! P2 recursive world-model experiment CLI.
//!
//! The P1 exact-simulator harness (`p1a` / `p1b` / `p1c` / `p1c-hard`) lives only on
//! the `p1` git branch.

use crate::p2::cli::{
    run_p2_arc3_bridge, run_p2_arc3_eval, run_p2_arc3_live_eval, run_p2_bf16_bench,
    run_p2_bf16_drift, run_p2_context_confirmation, run_p2_context_confirmation_v2,
    run_p2_context_wiring, run_p2_eval, run_p2_fixed_batch_positive_control,
    run_p2_frozen_seam_characterization, run_p2_multibatch_frozen_diagnostic,
    run_p2_multibatch_generalization_screen, run_p2_residual_probe, run_p2_train, P2Arc3BridgeArgs,
    P2Arc3EvalArgs, P2Arc3LiveEvalArgs, P2Bf16BenchArgs, P2Bf16DriftArgs,
    P2ContextConfirmationArgs, P2ContextConfirmationV2Args, P2ContextWiringArgs, P2EvalArgs,
    P2FixedBatchPositiveControlArgs, P2FrozenSeamCharacterizationArgs,
    P2MultibatchFrozenDiagnosticArgs, P2MultibatchScreenArgs, P2ResidualProbeArgs, P2TrainArgs,
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
    P2Train(Box<P2TrainArgs>),
    /// P2: compare frozen-checkpoint F32 and recurrent-core BF16 numerics
    #[command(name = "p2-bf16-drift")]
    P2Bf16Drift(P2Bf16DriftArgs),
    /// P2: benchmark warmed synchronized full updates with BF16 off/on
    #[command(name = "p2-bf16-bench")]
    P2Bf16Bench(P2Bf16BenchArgs),
    /// P2: V6 raw-weight fixed-batch production-path positive control
    #[command(name = "p2-v6-fixed-batch-positive-control")]
    P2FixedBatchPositiveControl(Box<P2FixedBatchPositiveControlArgs>),
    /// P2: V6 multi-batch function-learning screen (selection-only single seed)
    #[command(name = "p2-v6-multibatch-generalization-screen")]
    P2MultibatchGeneralizationScreen(Box<P2MultibatchScreenArgs>),
    /// P2: read-only frozen-checkpoint diagnosis selected by failed multi-batch G
    #[command(name = "p2-v6-multibatch-frozen-diagnostic")]
    P2MultibatchFrozenDiagnostic(Box<P2MultibatchFrozenDiagnosticArgs>),
    /// P2: no-gradient localization of the frozen train/raw prediction seam
    #[command(name = "p2-v6-frozen-seam-characterization")]
    P2FrozenSeamCharacterization(Box<P2FrozenSeamCharacterizationArgs>),
    /// P2: held-out synthetic world-model and PTRM evaluation
    #[command(name = "p2-eval")]
    P2Eval(P2EvalArgs),
    /// P2: held-out transfer evaluation on official-toolkit ARC recordings
    #[command(name = "p2-arc3-eval")]
    P2Arc3Eval(P2Arc3EvalArgs),
    /// P2: held-out live evaluation on every public ARC-AGI-3 environment
    #[command(name = "p2-arc3-live-eval")]
    P2Arc3LiveEval(P2Arc3LiveEvalArgs),
    /// P2: local ARC-AGI-3 toolkit bridge over stdin/stdout JSON
    #[command(name = "p2-arc3-bridge")]
    P2Arc3Bridge(P2Arc3BridgeArgs),
    /// P2: ADR 0005 §5.3 residual-vs-reliability probe on a frozen checkpoint
    #[command(name = "p2-residual-probe")]
    P2ResidualProbe(P2ResidualProbeArgs),
    /// P2: E2W two-row context-wiring overfit diagnostic (implementation smoke)
    #[command(name = "p2-context-wiring")]
    P2ContextWiring(Box<P2ContextWiringArgs>),
    /// P2: E2C second-pair exact wiring and launch stability confirmation (implementation smoke)
    #[command(name = "p2-context-confirmation")]
    P2ContextConfirmation(Box<P2ContextConfirmationArgs>),
    /// P2: E2D canonical-singleton scoring with semantic batch invariance (implementation smoke)
    #[command(name = "p2-context-confirmation-v2")]
    P2ContextConfirmationV2(Box<P2ContextConfirmationV2Args>),
    /// P2: unified application and optional GPU evidence viewer
    #[command(name = "p2-view")]
    P2View(P2ViewArgs),
}

/// Entry point used by `main`.
pub fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::P2Train(args) => run_p2_train(*args)?,
        Commands::P2Bf16Drift(args) => run_p2_bf16_drift(args)?,
        Commands::P2Bf16Bench(args) => run_p2_bf16_bench(args)?,
        Commands::P2FixedBatchPositiveControl(args) => run_p2_fixed_batch_positive_control(*args)?,
        Commands::P2MultibatchGeneralizationScreen(args) => {
            run_p2_multibatch_generalization_screen(*args)?
        }
        Commands::P2MultibatchFrozenDiagnostic(args) => run_p2_multibatch_frozen_diagnostic(*args)?,
        Commands::P2FrozenSeamCharacterization(args) => run_p2_frozen_seam_characterization(*args)?,
        Commands::P2Eval(args) => run_p2_eval(args)?,
        Commands::P2Arc3Eval(args) => run_p2_arc3_eval(args)?,
        Commands::P2Arc3LiveEval(args) => run_p2_arc3_live_eval(args)?,
        Commands::P2Arc3Bridge(args) => run_p2_arc3_bridge(args)?,
        Commands::P2ResidualProbe(args) => run_p2_residual_probe(args)?,
        Commands::P2ContextWiring(args) => run_p2_context_wiring(*args)?,
        Commands::P2ContextConfirmation(args) => run_p2_context_confirmation(*args)?,
        Commands::P2ContextConfirmationV2(args) => run_p2_context_confirmation_v2(*args)?,
        Commands::P2View(args) => run_p2_view(args)?,
    }
    Ok(())
}
