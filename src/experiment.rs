//! P1 exact-simulator and P2 learned-world-model experiment CLI.

use crate::agents::{run_episode, AgentConfig, AgentName, AgentStats};
use crate::domain::{Goal, Scenario, Simulator, Split};
use crate::generator;
use crate::p2::cli::{
    run_p2_arc3_eval, run_p2_eval, run_p2_train, P2Arc3EvalArgs, P2EvalArgs, P2TrainArgs,
};
use crate::report::{
    aggregate_episodes, evaluate_bootstrap_gate, goal_family, oracle_normalized_efficiency,
    print_summary, strongest_by_efficiency, strongest_by_success, write_report_atomic,
    BootstrapGateSpec, EpisodeRecord, ExperimentReport, Phase, RunConfig,
    DEFAULT_BOOTSTRAP_SAMPLES, DEFAULT_BOOTSTRAP_SEED, MAX_BOOTSTRAP_SAMPLES,
};
use crate::search::shortest_path;
use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::path::PathBuf;

const AGENT_TAG_RANDOM: u64 = 0x01;
const AGENT_TAG_NOVEL: u64 = 0x02;
const AGENT_TAG_GREEDY: u64 = 0x03;
const AGENT_TAG_DISC: u64 = 0x04;
const AGENT_TAG_ORACLE_OBJ: u64 = 0x05;
const AGENT_TAG_REACTIVE: u64 = 0x06;
const AGENT_TAG_PAUSE: u64 = 0x07;
const AGENT_TAG_BESTOFK: u64 = 0x08;
const AGENT_TAG_BEAM: u64 = 0x09;
const AGENT_TAG_ORACLE_OPT: u64 = 0x0A;
const AGENT_TAG_SET_AWARE: u64 = 0x0B;
const AGENT_TAG_SHARED_PROGRESS: u64 = 0x0C;
const AGENT_TAG_BROAD_PROGRESS_NARROW_FALSIFY: u64 = 0x0D;
const AGENT_TAG_BROAD_FALSIFY_NARROW_PROGRESS: u64 = 0x0E;
const AGENT_TAG_ALTERNATING: u64 = 0x0F;
const AGENT_TAG_COST_AWARE: u64 = 0x10;
const AGENT_TAG_CAPPED_BROAD_PROGRESS: u64 = 0x11;

fn agent_tag(agent: AgentName) -> u64 {
    match agent {
        AgentName::Random => AGENT_TAG_RANDOM,
        AgentName::NovelState => AGENT_TAG_NOVEL,
        AgentName::GreedyApparentProgress => AGENT_TAG_GREEDY,
        AgentName::CandidateGoalDiscrimination => AGENT_TAG_DISC,
        AgentName::OracleObjective => AGENT_TAG_ORACLE_OBJ,
        AgentName::Reactive => AGENT_TAG_REACTIVE,
        AgentName::PauseCompute => AGENT_TAG_PAUSE,
        AgentName::BestOfK => AGENT_TAG_BESTOFK,
        AgentName::BeamSearch => AGENT_TAG_BEAM,
        AgentName::OracleOptimal => AGENT_TAG_ORACLE_OPT,
        AgentName::SetAwareParallelPlanning => AGENT_TAG_SET_AWARE,
        AgentName::SharedProgressPlanning => AGENT_TAG_SHARED_PROGRESS,
        AgentName::BroadProgressNarrowFalsify => AGENT_TAG_BROAD_PROGRESS_NARROW_FALSIFY,
        AgentName::BroadFalsifyNarrowProgress => AGENT_TAG_BROAD_FALSIFY_NARROW_PROGRESS,
        AgentName::AlternatingParallelPlanning => AGENT_TAG_ALTERNATING,
        AgentName::CostAwareParallelPlanning => AGENT_TAG_COST_AWARE,
        AgentName::CappedBroadProgressPlanning => AGENT_TAG_CAPPED_BROAD_PROGRESS,
    }
}

const P1A_AGENTS: [AgentName; 5] = [
    AgentName::Random,
    AgentName::NovelState,
    AgentName::GreedyApparentProgress,
    AgentName::CandidateGoalDiscrimination,
    AgentName::OracleObjective,
];

const P1B_AGENTS: [AgentName; 5] = [
    AgentName::Reactive,
    AgentName::PauseCompute,
    AgentName::BestOfK,
    AgentName::BeamSearch,
    AgentName::OracleOptimal,
];

/// Seven P1C strategies in declaration/aggregate order (gates sort by name).
const P1C_STRATEGIES: [AgentName; 7] = [
    AgentName::SetAwareParallelPlanning,
    AgentName::SharedProgressPlanning,
    AgentName::BroadProgressNarrowFalsify,
    AgentName::BroadFalsifyNarrowProgress,
    AgentName::AlternatingParallelPlanning,
    AgentName::CostAwareParallelPlanning,
    AgentName::CappedBroadProgressPlanning,
];

/// Standalone P1C: sequential baseline, seven strategies, oracle control.
const P1C_AGENTS: [AgentName; 9] = [
    AgentName::CandidateGoalDiscrimination,
    AgentName::SetAwareParallelPlanning,
    AgentName::SharedProgressPlanning,
    AgentName::BroadProgressNarrowFalsify,
    AgentName::BroadFalsifyNarrowProgress,
    AgentName::AlternatingParallelPlanning,
    AgentName::CostAwareParallelPlanning,
    AgentName::CappedBroadProgressPlanning,
    AgentName::OracleObjective,
];

/// P1A already runs CandidateGoalDiscrimination and OracleObjective in `all` mode.
const P1C_ADDITIONAL_AGENTS: [AgentName; 7] = P1C_STRATEGIES;

/// Hard P1C challenge: greedy sequential commitment versus research-first planning.
const P1C_HARD_AGENTS: [AgentName; 3] = [
    AgentName::CandidateGoalDiscrimination,
    AgentName::SetAwareParallelPlanning,
    AgentName::BroadFalsifyNarrowProgress,
];

const P1C_HARD_STRATEGIES: [AgentName; 2] = [
    AgentName::SetAwareParallelPlanning,
    AgentName::BroadFalsifyNarrowProgress,
];

const P1C_HARD_MIN_SEQUENTIAL_COMMITMENTS: u64 = 3;
const P1C_HARD_MAX_ATTEMPTS: u64 = 512;

const P1A_BASELINES: [AgentName; 3] = [
    AgentName::Random,
    AgentName::NovelState,
    AgentName::GreedyApparentProgress,
];

const P1B_ONE_STEP: [AgentName; 3] = [
    AgentName::Reactive,
    AgentName::PauseCompute,
    AgentName::BestOfK,
];

#[derive(Debug, Parser)]
#[command(
    name = "tofy",
    about = "Hidden-objective planning and P2 recursive world-model experiments"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// P1A: hidden-objective discovery
    #[command(name = "p1a")]
    P1a(CommonArgs),
    /// P1B: planning necessity
    #[command(name = "p1b")]
    P1b(CommonArgs),
    /// P1C: set-aware planning across hidden goal hypotheses
    #[command(name = "p1c")]
    P1c(CommonArgs),
    /// Hard P1C: repeated sequential retargeting versus research-first planning
    #[command(name = "p1c-hard")]
    P1cHard(CommonArgs),
    /// Run P1A, P1B, and P1C on identical generated scenarios
    #[command(name = "all")]
    All(CommonArgs),
    /// P2: train the recursive world model through synthetic curriculum lessons
    #[command(name = "p2-train")]
    P2Train(P2TrainArgs),
    /// P2: held-out synthetic world-model and PTRM evaluation
    #[command(name = "p2-eval")]
    P2Eval(P2EvalArgs),
    /// P2: held-out transfer evaluation on official-toolkit ARC recordings
    #[command(name = "p2-arc3-eval")]
    P2Arc3Eval(P2Arc3EvalArgs),
}

#[derive(Debug, Clone, clap::Args)]
struct CommonArgs {
    /// Master experiment seed (used when --seeds is omitted)
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Optional comma-separated seeds; when set, overrides --seed
    #[arg(long, value_delimiter = ',')]
    seeds: Option<Vec<u64>>,

    /// Episodes per split (Train and HeldOutComposition) per seed
    #[arg(long, default_value_t = 2)]
    episodes_per_split: u64,

    /// Pretty JSON report path
    #[arg(long, default_value = "runs/p1/report.json")]
    output: PathBuf,

    #[arg(long, default_value_t = 8)]
    beam_width: usize,

    #[arg(long, default_value_t = 24)]
    beam_horizon: u16,

    #[arg(long, default_value_t = 4)]
    best_of_k: usize,

    #[arg(long, default_value_t = 16)]
    pause_extra_evals: u64,

    /// Fixed bootstrap RNG seed for exploratory gates
    #[arg(long, default_value_t = DEFAULT_BOOTSTRAP_SEED)]
    bootstrap_seed: u64,

    /// Bootstrap replicate count for exploratory gates
    #[arg(long, default_value_t = DEFAULT_BOOTSTRAP_SAMPLES)]
    bootstrap_samples: u64,

    /// Exploratory minimum success-rate lift (lower CI bound)
    #[arg(long, default_value_t = 0.05)]
    min_success_lift: f64,

    /// Exploratory minimum oracle-normalized-efficiency lift (lower CI bound)
    #[arg(long, default_value_t = 0.05)]
    min_efficiency_lift: f64,
}

impl CommonArgs {
    fn resolved_seeds(&self) -> Vec<u64> {
        match &self.seeds {
            Some(s) if !s.is_empty() => s.clone(),
            _ => vec![self.seed],
        }
    }

    fn to_run_config(&self) -> RunConfig {
        let seeds = self.resolved_seeds();
        RunConfig {
            seeds,
            episodes_per_split: self.episodes_per_split,
            beam_width: self.beam_width,
            beam_horizon: self.beam_horizon,
            best_of_k: self.best_of_k,
            pause_extra_evals: self.pause_extra_evals,
            bootstrap_seed: self.bootstrap_seed,
            bootstrap_samples: self.bootstrap_samples,
            min_success_lift: self.min_success_lift,
            min_efficiency_lift: self.min_efficiency_lift,
            output: self.output.display().to_string(),
        }
    }
}

/// Entry point used by `main`.
pub fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::P1a(args) => run_phase(Phase::P1a, &args)?,
        Commands::P1b(args) => run_phase(Phase::P1b, &args)?,
        Commands::P1c(args) => run_phase(Phase::P1c, &args)?,
        Commands::P1cHard(args) => run_phase(Phase::P1cHard, &args)?,
        Commands::All(args) => run_phase(Phase::All, &args)?,
        Commands::P2Train(args) => run_p2_train(args)?,
        Commands::P2Eval(args) => run_p2_eval(args)?,
        Commands::P2Arc3Eval(args) => run_p2_arc3_eval(args)?,
    }
    Ok(())
}

fn run_phase(phase: Phase, args: &CommonArgs) -> Result<()> {
    if args.episodes_per_split == 0 {
        bail!("episodes_per_split must be > 0");
    }
    let seeds = args.resolved_seeds();
    if seeds.is_empty() {
        bail!("at least one seed is required");
    }
    let unique_seeds: BTreeSet<u64> = seeds.iter().copied().collect();
    if unique_seeds.len() != seeds.len() {
        bail!("seeds must be unique");
    }
    if args.bootstrap_samples == 0 {
        bail!("bootstrap_samples must be > 0");
    }
    if args.bootstrap_samples > MAX_BOOTSTRAP_SAMPLES {
        bail!("bootstrap_samples must be <= {MAX_BOOTSTRAP_SAMPLES}");
    }
    if !args.min_success_lift.is_finite() || !(0.0..=1.0).contains(&args.min_success_lift) {
        bail!("min_success_lift must be finite and in [0, 1]");
    }
    if !args.min_efficiency_lift.is_finite() || !(0.0..=1.15).contains(&args.min_efficiency_lift) {
        bail!("min_efficiency_lift must be finite and in [0, 1.15]");
    }
    let config = args.to_run_config();
    let scenarios = match phase {
        Phase::P1c | Phase::All => generate_p1c_paired_scenarios(&seeds, args.episodes_per_split),
        Phase::P1cHard => generate_p1c_hard_paired_scenarios(&seeds, args.episodes_per_split),
        Phase::P1a | Phase::P1b => generate_paired_scenarios(&seeds, args.episodes_per_split),
    };
    let mut report = ExperimentReport::new(phase, config);
    match phase {
        Phase::P1a => {
            report.episodes = run_suite(&scenarios, &P1A_AGENTS, args, GoalProvision::P1a);
            report.gates = p1a_gates(&report.episodes, args);
        }
        Phase::P1b => {
            report.episodes = run_suite(&scenarios, &P1B_AGENTS, args, GoalProvision::P1b);
            report.gates = p1b_gates(&report.episodes, args);
        }
        Phase::P1c => {
            report.episodes = run_suite(&scenarios, &P1C_AGENTS, args, GoalProvision::P1a);
            report.gates = p1c_gates(&report.episodes, args);
        }
        Phase::P1cHard => {
            report.episodes = run_suite(&scenarios, &P1C_HARD_AGENTS, args, GoalProvision::P1a);
            report.gates = p1c_hard_gates(&report.episodes, args);
        }
        Phase::All => {
            let mut eps = run_suite(&scenarios, &P1A_AGENTS, args, GoalProvision::P1a);
            eps.extend(run_suite(
                &scenarios,
                &P1C_ADDITIONAL_AGENTS,
                args,
                GoalProvision::P1a,
            ));
            eps.extend(run_suite(&scenarios, &P1B_AGENTS, args, GoalProvision::P1b));
            report.episodes = eps;
            let mut gates = p1a_gates(&report.episodes, args);
            gates.extend(p1b_gates(&report.episodes, args));
            gates.extend(p1c_gates(&report.episodes, args));
            report.gates = gates;
        }
    }
    report.aggregates = aggregate_episodes(&report.episodes);
    report.sort_deterministic();
    write_report_atomic(&args.output, &report)?;
    print_summary(&report);
    println!("wrote {}", args.output.display());
    Ok(())
}

/// Generate identical Train + HeldOutComposition scenarios shared by all agents.
pub fn generate_paired_scenarios(seeds: &[u64], episodes_per_split: u64) -> Vec<Scenario> {
    let mut out = Vec::with_capacity(seeds.len() * (episodes_per_split as usize) * 2);
    for &seed in seeds {
        for split in [Split::Train, Split::HeldOutComposition] {
            for episode_id in 0..episodes_per_split {
                out.push(generator::generate(seed, episode_id, split));
            }
        }
    }
    out
}

/// P1C scenarios guarantee a safe endpoint probe that tests at least two
/// equally plausible, exactly viable candidate goals. `all` uses this shared
/// scenario set so its within-report comparisons remain paired.
pub fn generate_p1c_paired_scenarios(seeds: &[u64], episodes_per_split: u64) -> Vec<Scenario> {
    let mut out = Vec::with_capacity(seeds.len() * (episodes_per_split as usize) * 2);
    for &seed in seeds {
        for split in [Split::Train, Split::HeldOutComposition] {
            for episode_id in 0..episodes_per_split {
                out.push(generator::generate_p1c(seed, episode_id, split));
            }
        }
    }
    out
}

/// Build an adversarial-but-recoverable challenge set using only the sequential
/// baseline as the selection policy. Every accepted scenario makes sequential
/// discrimination retarget after at least three falsified commitments. Success
/// is deliberately not a selection criterion, so it remains a meaningful outcome.
/// Parallel-policy outcomes are never consulted during selection.
pub fn generate_p1c_hard_paired_scenarios(seeds: &[u64], episodes_per_split: u64) -> Vec<Scenario> {
    let mut keys = Vec::with_capacity(seeds.len() * (episodes_per_split as usize) * 2);
    for &seed in seeds {
        for split in [Split::Train, Split::HeldOutComposition] {
            for episode_id in 0..episodes_per_split {
                keys.push((seed, split, episode_id));
            }
        }
    }
    keys.into_par_iter()
        .map(|(seed, split, episode_id)| {
            let family = episode_id % 6;
            for attempt in 0..P1C_HARD_MAX_ATTEMPTS {
                let source_episode_id = hard_source_episode_id(seed, episode_id, attempt, family);
                let mut scenario =
                    generator::generate_p1c_hard_candidate(seed, source_episode_id, split);
                let stats = sequential_hardness(&scenario);
                if stats.incorrect_commitments >= P1C_HARD_MIN_SEQUENTIAL_COMMITMENTS
                    && generator::p1c_falsification_probe_width(&scenario) >= 2
                    && oracle_optimal_actions(&scenario).is_some()
                {
                    // Report and pair by the requested logical episode. Source identity
                    // affects generation only and is reproducible from this function.
                    scenario.episode_id = episode_id;
                    return scenario;
                }
            }
            panic!(
                "no hard P1C scenario after {} attempts: seed={} episode={} split={:?}",
                P1C_HARD_MAX_ATTEMPTS, seed, episode_id, split
            )
        })
        .collect()
}

fn hard_source_episode_id(seed: u64, episode_id: u64, attempt: u64, family: u64) -> u64 {
    debug_assert!(family < 6);
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(episode_id.wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add(attempt.wrapping_mul(0x1656_67B1_9E37_79F9));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    let family_buckets = u64::MAX / 6;
    (x % family_buckets) * 6 + family
}

fn sequential_hardness(scenario: &Scenario) -> AgentStats {
    let config = AgentConfig::for_name(AgentName::CandidateGoalDiscrimination);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    run_episode(scenario.clone(), &config, &mut rng, None)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GoalProvision {
    /// Only OracleObjective receives `true_goal`; others get none.
    P1a,
    /// All methods know the true objective: non-oracles via `planning_goal`,
    /// OracleOptimal via the `true_goal` argument only.
    P1b,
}

fn run_suite(
    scenarios: &[Scenario],
    agents: &[AgentName],
    args: &CommonArgs,
    provision: GoalProvision,
) -> Vec<EpisodeRecord> {
    let oracle_actions: Vec<Option<u64>> =
        scenarios.par_iter().map(oracle_optimal_actions).collect();
    let pair_count = scenarios.len() * agents.len();
    (0..pair_count)
        .into_par_iter()
        .map(|pair_index| {
            let scenario_index = pair_index / agents.len();
            let sc = &scenarios[scenario_index];
            let agent = agents[pair_index % agents.len()];
            let true_goal = sc.hidden_goal().clone();
            let (config, true_goal_arg) =
                build_agent_config(agent, &true_goal, args, ProvisionKind::from(provision));
            let mut rng = rng_for_episode(sc.seed, sc.episode_id, sc.split, agent);
            let stats = run_episode(sc.clone(), &config, &mut rng, true_goal_arg.as_ref());
            episode_record(
                sc,
                agent,
                &stats,
                oracle_actions[scenario_index],
                &true_goal,
            )
        })
        .collect()
}

/// Split-aware agent RNG so Train vs HeldOut with the same episode_id are independent.
pub fn rng_for_episode(seed: u64, episode_id: u64, split: Split, agent: AgentName) -> ChaCha8Rng {
    let split_tag: u64 = match split {
        Split::Train => 0x5472_6169_6e00_0001,
        Split::HeldOutComposition => 0x4865_6c64_4f75_7402,
    };
    let tag = agent_tag(agent);
    let mut key = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(episode_id.wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    key ^= split_tag.rotate_left(17);
    key ^= tag.rotate_left(13);
    key = key.wrapping_mul(0x1656_67B1_9E37_79F9);
    key ^= tag.wrapping_mul(0x85EB_CA77_C2B2_AE63);
    ChaCha8Rng::seed_from_u64(key)
}

/// Build agent config and optional `true_goal` argument per phase rules.
fn build_agent_config(
    agent: AgentName,
    true_goal: &Goal,
    args: &CommonArgs,
    provision: ProvisionKind,
) -> (AgentConfig, Option<Goal>) {
    let mut config = AgentConfig::for_name(agent);
    config.best_of_k = args.best_of_k.max(1);
    config.beam_width = args.beam_width.max(1);
    config.beam_horizon = args.beam_horizon.max(1);
    config.pause_extra_evals = args.pause_extra_evals;
    match provision {
        ProvisionKind::P1a => {
            config.planning_goal = None;
            let tg = if agent == AgentName::OracleObjective {
                Some(true_goal.clone())
            } else {
                None
            };
            (config, tg)
        }
        ProvisionKind::P1b => {
            if agent == AgentName::OracleOptimal {
                config.planning_goal = None;
                (config, Some(true_goal.clone()))
            } else {
                config.planning_goal = Some(true_goal.clone());
                (config, None)
            }
        }
    }
}

/// Goal-provisioning mode for P1A vs P1B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProvisionKind {
    P1a,
    P1b,
}

impl From<GoalProvision> for ProvisionKind {
    fn from(g: GoalProvision) -> Self {
        match g {
            GoalProvision::P1a => ProvisionKind::P1a,
            GoalProvision::P1b => ProvisionKind::P1b,
        }
    }
}

/// Exact oracle action count, or `None` when unsolvable (never substitute budget).
fn oracle_optimal_actions(scenario: &Scenario) -> Option<u64> {
    let sim = Simulator::new(scenario.clone());
    let goal = scenario.hidden_goal();
    shortest_path(&sim, sim.state(), goal, scenario.action_budget)
        .map(|res| res.actions.len() as u64)
}

fn episode_record(
    sc: &Scenario,
    agent: AgentName,
    stats: &AgentStats,
    oracle_actions: Option<u64>,
    true_goal: &Goal,
) -> EpisodeRecord {
    let (correct_id, actions_to_id) =
        if agent == AgentName::CandidateGoalDiscrimination || agent.is_p1c_strategy() {
            let correct = stats.identified_candidate == Some(sc.hidden_goal_index);
            (Some(correct), stats.actions_to_unique_id)
        } else {
            (None, None)
        };
    let eff = match oracle_actions {
        Some(oa) => oracle_normalized_efficiency(stats.success, stats.env_actions, oa),
        None => 0.0,
    };
    let p1c = agent.is_p1c_strategy();
    EpisodeRecord {
        seed: sc.seed,
        episode_id: sc.episode_id,
        split: sc.split,
        agent,
        goal_family: goal_family(true_goal).to_string(),
        success: stats.success,
        failed: stats.failed,
        scored_env_actions: stats.env_actions,
        expansions: stats.expansions,
        evaluations: stats.evaluations,
        exhausted: stats.exhausted,
        oracle_optimal_actions: oracle_actions,
        oracle_normalized_efficiency: eff,
        correct_objective_identification: correct_id,
        actions_to_identification: actions_to_id,
        incorrect_commitments: stats.incorrect_commitments,
        shared_progress_actions: p1c.then_some(stats.shared_progress_actions),
        multi_goal_probe_actions: p1c.then_some(stats.multi_goal_probe_actions),
        goals_falsified: p1c.then_some(stats.goals_falsified),
        parallel_method_switches: p1c.then_some(stats.parallel_method_switches),
    }
}

fn p1a_gates(episodes: &[EpisodeRecord], args: &CommonArgs) -> Vec<crate::report::GateOutcome> {
    let aggs = aggregate_episodes(episodes);
    let split = Split::HeldOutComposition;
    let success_base = strongest_by_success(&aggs, &P1A_BASELINES, split);
    let eff_base = strongest_by_efficiency(&aggs, &P1A_BASELINES, split);
    vec![evaluate_bootstrap_gate(
        &BootstrapGateSpec {
            name: "p1a_discrimination_vs_best_exploration",
            candidate: AgentName::CandidateGoalDiscrimination,
            success_baseline: success_base.map(|b| b.agent),
            efficiency_baseline: eff_base.map(|b| b.agent),
            split,
            min_success_lift: args.min_success_lift,
            min_efficiency_lift: args.min_efficiency_lift,
            bootstrap_seed: args.bootstrap_seed,
            bootstrap_samples: args.bootstrap_samples,
        },
        episodes,
    )]
}

fn p1b_gates(episodes: &[EpisodeRecord], args: &CommonArgs) -> Vec<crate::report::GateOutcome> {
    let aggs = aggregate_episodes(episodes);
    let split = Split::HeldOutComposition;
    let one_step_success = strongest_by_success(&aggs, &P1B_ONE_STEP, split);
    let one_step_eff = strongest_by_efficiency(&aggs, &P1B_ONE_STEP, split);
    vec![
        evaluate_bootstrap_gate(
            &BootstrapGateSpec {
                name: "p1b_oracle_vs_reactive_planning_necessity",
                candidate: AgentName::OracleOptimal,
                success_baseline: Some(AgentName::Reactive),
                efficiency_baseline: Some(AgentName::Reactive),
                split,
                min_success_lift: args.min_success_lift,
                min_efficiency_lift: args.min_efficiency_lift,
                bootstrap_seed: args.bootstrap_seed,
                bootstrap_samples: args.bootstrap_samples,
            },
            episodes,
        ),
        evaluate_bootstrap_gate(
            &BootstrapGateSpec {
                name: "p1b_beam_vs_strongest_one_step",
                candidate: AgentName::BeamSearch,
                success_baseline: one_step_success.map(|b| b.agent),
                efficiency_baseline: one_step_eff.map(|b| b.agent),
                split,
                min_success_lift: args.min_success_lift,
                min_efficiency_lift: args.min_efficiency_lift,
                bootstrap_seed: args.bootstrap_seed,
                bootstrap_samples: args.bootstrap_samples,
            },
            episodes,
        ),
    ]
}

fn p1c_gates(episodes: &[EpisodeRecord], args: &CommonArgs) -> Vec<crate::report::GateOutcome> {
    let mut gates = Vec::with_capacity(P1C_STRATEGIES.len());
    for &candidate in &P1C_STRATEGIES {
        let name = format!("p1c_{}_vs_sequential_discrimination", candidate.as_str());
        gates.push(evaluate_bootstrap_gate(
            &BootstrapGateSpec {
                name: &name,
                candidate,
                success_baseline: Some(AgentName::CandidateGoalDiscrimination),
                efficiency_baseline: Some(AgentName::CandidateGoalDiscrimination),
                split: Split::HeldOutComposition,
                min_success_lift: args.min_success_lift,
                min_efficiency_lift: args.min_efficiency_lift,
                bootstrap_seed: args.bootstrap_seed,
                bootstrap_samples: args.bootstrap_samples,
            },
            episodes,
        ));
    }
    gates
}

fn p1c_hard_gates(
    episodes: &[EpisodeRecord],
    args: &CommonArgs,
) -> Vec<crate::report::GateOutcome> {
    let mut gates: Vec<_> = P1C_HARD_STRATEGIES
        .iter()
        .map(|&candidate| {
            let name = format!(
                "p1c_hard_{}_vs_sequential_discrimination",
                candidate.as_str()
            );
            evaluate_bootstrap_gate(
                &BootstrapGateSpec {
                    name: &name,
                    candidate,
                    success_baseline: Some(AgentName::CandidateGoalDiscrimination),
                    efficiency_baseline: Some(AgentName::CandidateGoalDiscrimination),
                    split: Split::HeldOutComposition,
                    min_success_lift: args.min_success_lift,
                    min_efficiency_lift: args.min_efficiency_lift,
                    bootstrap_seed: args.bootstrap_seed,
                    bootstrap_samples: args.bootstrap_samples,
                },
                episodes,
            )
        })
        .collect();
    gates.push(evaluate_bootstrap_gate(
        &BootstrapGateSpec {
            name: "p1c_hard_broad_falsify_narrow_progress_vs_falsification_only",
            candidate: AgentName::BroadFalsifyNarrowProgress,
            success_baseline: Some(AgentName::SetAwareParallelPlanning),
            efficiency_baseline: Some(AgentName::SetAwareParallelPlanning),
            split: Split::HeldOutComposition,
            min_success_lift: args.min_success_lift,
            min_efficiency_lift: args.min_efficiency_lift,
            bootstrap_seed: args.bootstrap_seed,
            bootstrap_samples: args.bootstrap_samples,
        },
        episodes,
    ));
    gates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{report_bytes, REPORT_VERSION};

    fn smoke_args(seed: u64, episodes: u64) -> CommonArgs {
        CommonArgs {
            seed,
            seeds: None,
            episodes_per_split: episodes,
            output: PathBuf::from("runs/p1/test_report.json"),
            beam_width: 4,
            beam_horizon: 12,
            best_of_k: 2,
            pause_extra_evals: 2,
            bootstrap_seed: 11,
            bootstrap_samples: 99,
            min_success_lift: 0.05,
            min_efficiency_lift: 0.05,
        }
    }

    fn multi_seed_args(seeds: Vec<u64>, episodes: u64) -> CommonArgs {
        let mut args = smoke_args(seeds[0], episodes);
        args.seeds = Some(seeds);
        args
    }

    #[test]
    fn agent_rng_independent_of_order() {
        let a = rng_for_episode(7, 3, Split::Train, AgentName::Random);
        let b = rng_for_episode(7, 3, Split::Train, AgentName::NovelState);
        let mut a = a;
        let mut b = b;
        use rand::Rng;
        assert_ne!(a.random::<u64>(), b.random::<u64>());
        let mut x = rng_for_episode(7, 3, Split::HeldOutComposition, AgentName::BestOfK);
        let mut y = rng_for_episode(7, 3, Split::HeldOutComposition, AgentName::BestOfK);
        assert_eq!(x.random::<u64>(), y.random::<u64>());
    }

    #[test]
    fn identical_episode_pairing_across_agents() {
        let scenarios = generate_paired_scenarios(&[11], 2);
        assert_eq!(scenarios.len(), 4);
        let again = generate_paired_scenarios(&[11], 2);
        assert_eq!(scenarios, again);

        let args = smoke_args(11, 2);
        let mut records_fwd = run_suite(&scenarios, &P1A_AGENTS, &args, GoalProvision::P1a);
        let mut rev_agents = P1A_AGENTS;
        rev_agents.reverse();
        let mut records_rev = run_suite(&scenarios, &rev_agents, &args, GoalProvision::P1a);

        records_fwd.sort_by_key(|r| {
            (
                r.seed,
                format!("{:?}", r.split),
                r.episode_id,
                format!("{:?}", r.agent),
            )
        });
        records_rev.sort_by_key(|r| {
            (
                r.seed,
                format!("{:?}", r.split),
                r.episode_id,
                format!("{:?}", r.agent),
            )
        });
        assert_eq!(records_fwd.len(), records_rev.len());
        for (a, b) in records_fwd.iter().zip(&records_rev) {
            assert_eq!(a.seed, b.seed);
            assert_eq!(a.episode_id, b.episode_id);
            assert_eq!(a.split, b.split);
            assert_eq!(a.agent, b.agent);
            assert_eq!(a.success, b.success);
            assert_eq!(a.scored_env_actions, b.scored_env_actions);
            assert_eq!(
                a.oracle_normalized_efficiency,
                b.oracle_normalized_efficiency
            );
        }
    }

    #[test]
    fn multi_seed_identity_on_episodes() {
        let seeds = vec![3u64, 9];
        let scenarios = generate_paired_scenarios(&seeds, 1);
        assert_eq!(scenarios.len(), 4);
        let args = multi_seed_args(seeds.clone(), 1);
        let records = run_suite(&scenarios, &[AgentName::Random], &args, GoalProvision::P1a);
        let mut seen_seeds: Vec<u64> = records.iter().map(|r| r.seed).collect();
        seen_seeds.sort_unstable();
        seen_seeds.dedup();
        assert_eq!(seen_seeds, seeds);
        for r in &records {
            assert!(seeds.contains(&r.seed));
            assert_eq!(
                r.seed,
                scenarios
                    .iter()
                    .find(|s| s.episode_id == r.episode_id
                        && s.split == r.split
                        && s.seed == r.seed)
                    .unwrap()
                    .seed
            );
        }
    }

    #[test]
    fn single_seed_default_matches_resolved() {
        let args = smoke_args(5, 1);
        assert_eq!(args.resolved_seeds(), vec![5]);
        let cfg = args.to_run_config();
        assert_eq!(cfg.seeds, vec![5]);
    }

    #[test]
    fn p1b_goal_provisioning() {
        let goal = Goal::ReachMarker { marker: 0 };
        let args = smoke_args(1, 1);
        for agent in P1B_AGENTS {
            let (cfg, tg) = build_agent_config(agent, &goal, &args, ProvisionKind::P1b);
            if agent == AgentName::OracleOptimal {
                assert!(cfg.planning_goal.is_none());
                assert_eq!(tg.as_ref(), Some(&goal));
            } else {
                assert_eq!(cfg.planning_goal.as_ref(), Some(&goal));
                assert!(tg.is_none(), "{agent:?} must not receive true_goal arg");
            }
        }
        for agent in P1A_AGENTS {
            let (cfg, tg) = build_agent_config(agent, &goal, &args, ProvisionKind::P1a);
            assert!(cfg.planning_goal.is_none());
            if agent == AgentName::OracleObjective {
                assert_eq!(tg.as_ref(), Some(&goal));
            } else {
                assert!(tg.is_none());
            }
        }
        for agent in P1C_AGENTS {
            let (cfg, tg) = build_agent_config(agent, &goal, &args, ProvisionKind::P1a);
            assert!(cfg.planning_goal.is_none());
            if agent == AgentName::OracleObjective {
                assert_eq!(tg.as_ref(), Some(&goal));
            } else {
                assert!(tg.is_none(), "{agent:?} must not receive true_goal arg");
            }
        }
    }

    #[test]
    fn smoke_byte_identical_full_run() {
        let args = smoke_args(42, 1);
        let scenarios = generate_paired_scenarios(&args.resolved_seeds(), args.episodes_per_split);
        let mut r1 = ExperimentReport::new(Phase::P1a, args.to_run_config());
        r1.episodes = run_suite(&scenarios, &P1A_AGENTS, &args, GoalProvision::P1a);
        r1.aggregates = aggregate_episodes(&r1.episodes);
        r1.gates = p1a_gates(&r1.episodes, &args);
        r1.sort_deterministic();

        let mut r2 = ExperimentReport::new(Phase::P1a, args.to_run_config());
        r2.episodes = run_suite(&scenarios, &P1A_AGENTS, &args, GoalProvision::P1a);
        r2.aggregates = aggregate_episodes(&r2.episodes);
        r2.gates = p1a_gates(&r2.episodes, &args);
        r2.sort_deterministic();

        assert_eq!(report_bytes(&r1).unwrap(), report_bytes(&r2).unwrap());
        assert_eq!(r1.report_version, REPORT_VERSION);
        assert!(!r1.episodes.is_empty());
        assert!(r1.gates.iter().all(|g| g.exploratory));
        for ep_id in 0..args.episodes_per_split {
            for split in [Split::Train, Split::HeldOutComposition] {
                for agent in P1A_AGENTS {
                    assert!(r1.episodes.iter().any(|e| {
                        e.seed == 42
                            && e.episode_id == ep_id
                            && e.split == split
                            && e.agent == agent
                    }));
                }
            }
        }
    }

    #[test]
    fn gate_fails_closed_on_empty() {
        let args = smoke_args(1, 1);
        let gates = p1a_gates(&[], &args);
        assert_eq!(gates.len(), 1);
        assert!(!gates[0].passed);
        assert!(gates[0].exploratory);
        assert!(gates[0].success_lift_ci_low.is_none());
        let gates_b = p1b_gates(&[], &args);
        assert!(gates_b.iter().all(|g| !g.passed && g.exploratory));
        let gates_c = p1c_gates(&[], &args);
        assert_eq!(gates_c.len(), P1C_STRATEGIES.len());
        assert!(gates_c.iter().all(|g| !g.passed && g.exploratory));
        for (gate, strategy) in gates_c.iter().zip(P1C_STRATEGIES) {
            assert_eq!(
                gate.name,
                format!("p1c_{}_vs_sequential_discrimination", strategy.as_str())
            );
        }
        let gates_hard = p1c_hard_gates(&[], &args);
        assert_eq!(gates_hard.len(), 3);
        assert!(gates_hard
            .iter()
            .all(|gate| !gate.passed && gate.exploratory));
    }

    #[test]
    fn p1c_suite_reports_diagnostics_only_for_strategy_agents() {
        let args = smoke_args(13, 1);
        let scenarios = generate_p1c_paired_scenarios(&args.resolved_seeds(), 1);
        assert!(scenarios
            .iter()
            .all(|sc| generator::p1c_falsification_probe_width(sc) >= 2));
        let records = run_suite(&scenarios, &P1C_AGENTS, &args, GoalProvision::P1a);
        assert_eq!(
            records.len(),
            scenarios.len() * P1C_AGENTS.len(),
            "standalone P1C must run all 9 agents on every paired scenario"
        );
        for record in &records {
            if record.agent.is_p1c_strategy() {
                assert!(record.shared_progress_actions.is_some());
                assert!(record.multi_goal_probe_actions.is_some());
                assert!(record.goals_falsified.is_some());
                assert!(record.parallel_method_switches.is_some());
                assert!(record.correct_objective_identification.is_some());
            } else {
                assert!(record.shared_progress_actions.is_none());
                assert!(record.multi_goal_probe_actions.is_none());
                assert!(record.goals_falsified.is_none());
                assert!(record.parallel_method_switches.is_none());
            }
        }
        let gates = p1c_gates(&records, &args);
        assert_eq!(gates.len(), 7);
        for (gate, strategy) in gates.iter().zip(P1C_STRATEGIES) {
            assert_eq!(
                gate.name,
                format!("p1c_{}_vs_sequential_discrimination", strategy.as_str())
            );
            assert_eq!(gate.candidate, strategy.as_str());
        }
    }

    #[test]
    fn p1c_reports_are_byte_identical_and_all_has_no_duplicate_agents() {
        let args = smoke_args(17, 1);
        let scenarios = generate_p1c_paired_scenarios(&args.resolved_seeds(), 1);
        let again = generate_p1c_paired_scenarios(&args.resolved_seeds(), 1);
        assert_eq!(scenarios, again);

        let build = || {
            let mut report = ExperimentReport::new(Phase::P1c, args.to_run_config());
            report.episodes = run_suite(&scenarios, &P1C_AGENTS, &args, GoalProvision::P1a);
            report.aggregates = aggregate_episodes(&report.episodes);
            report.gates = p1c_gates(&report.episodes, &args);
            report.sort_deterministic();
            report
        };
        let r1 = build();
        let r2 = build();
        assert_eq!(report_bytes(&r1).unwrap(), report_bytes(&r2).unwrap());
        assert_eq!(r1.episodes.len(), scenarios.len() * P1C_AGENTS.len());
        assert_eq!(r1.gates.len(), 7);
        for sc in &scenarios {
            for agent in P1C_AGENTS {
                assert!(r1.episodes.iter().any(|e| {
                    e.seed == sc.seed
                        && e.episode_id == sc.episode_id
                        && e.split == sc.split
                        && e.agent == agent
                }));
            }
        }

        let mut all = run_suite(&scenarios, &P1A_AGENTS, &args, GoalProvision::P1a);
        all.extend(run_suite(
            &scenarios,
            &P1C_ADDITIONAL_AGENTS,
            &args,
            GoalProvision::P1a,
        ));
        all.extend(run_suite(
            &scenarios,
            &P1B_AGENTS,
            &args,
            GoalProvision::P1b,
        ));
        for sc in &scenarios {
            let records: Vec<&EpisodeRecord> = all
                .iter()
                .filter(|ep| {
                    ep.seed == sc.seed && ep.episode_id == sc.episode_id && ep.split == sc.split
                })
                .collect();
            let unique: std::collections::HashSet<AgentName> =
                records.iter().map(|ep| ep.agent).collect();
            assert_eq!(records.len(), 17);
            assert_eq!(unique.len(), records.len());
            for strategy in P1C_STRATEGIES {
                assert!(records.iter().any(|ep| ep.agent == strategy));
            }
        }
    }

    #[test]
    fn p1c_hard_suite_guarantees_retargeting_and_only_three_agents() {
        let args = smoke_args(201, 6);
        let scenarios =
            generate_p1c_hard_paired_scenarios(&args.resolved_seeds(), args.episodes_per_split);
        assert_eq!(scenarios.len(), 12);
        for scenario in &scenarios {
            let hardness = sequential_hardness(scenario);
            assert!(
                hardness.incorrect_commitments >= P1C_HARD_MIN_SEQUENTIAL_COMMITMENTS,
                "seed={} episode={} split={:?} had only {} retargets",
                scenario.seed,
                scenario.episode_id,
                scenario.split,
                hardness.incorrect_commitments
            );
            assert!(generator::p1c_falsification_probe_width(scenario) >= 2);
            assert!(oracle_optimal_actions(scenario).is_some());
        }
        for split in [Split::Train, Split::HeldOutComposition] {
            let families: BTreeSet<_> = scenarios
                .iter()
                .filter(|scenario| scenario.split == split)
                .map(|scenario| goal_family(scenario.hidden_goal()))
                .collect();
            assert_eq!(families.len(), 6);
        }

        let records = run_suite(&scenarios, &P1C_HARD_AGENTS, &args, GoalProvision::P1a);
        assert_eq!(records.len(), scenarios.len() * 3);
        let agents: BTreeSet<_> = records.iter().map(|record| record.agent.as_str()).collect();
        assert_eq!(
            agents,
            BTreeSet::from([
                AgentName::CandidateGoalDiscrimination.as_str(),
                AgentName::SetAwareParallelPlanning.as_str(),
                AgentName::BroadFalsifyNarrowProgress.as_str(),
            ])
        );
        assert_eq!(p1c_hard_gates(&records, &args).len(), 3);
    }

    #[test]
    fn hard_source_ids_preserve_family_and_change_by_attempt() {
        for family in 0..6 {
            let first = hard_source_episode_id(7, 11, 0, family);
            let second = hard_source_episode_id(7, 11, 1, family);
            assert_eq!(first % 6, family);
            assert_eq!(second % 6, family);
            assert_ne!(first, second);
        }
    }

    #[test]
    fn oracle_actions_none_when_unsolvable() {
        use crate::domain::Pos;
        use std::collections::BTreeSet;
        let sc = Scenario {
            width: 3,
            height: 3,
            walls: BTreeSet::new(),
            markers: vec![Pos::new(0, 0)],
            collectibles: vec![],
            switches: vec![],
            hazards: vec![],
            resource_pickups: vec![],
            terminal_triggers: vec![],
            start: Pos::new(1, 1),
            initial_resource: 0,
            action_budget: 4,
            undo_enabled: false,
            candidate_goals: vec![Goal::ReachMarker { marker: 99 }],
            hidden_goal_index: 0,
            split: Split::Train,
            seed: 1,
            episode_id: 0,
        };
        assert_eq!(oracle_optimal_actions(&sc), None);
    }
}
