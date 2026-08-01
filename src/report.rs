//! Deterministic, versioned experiment reports for P1 exact-simulator gates.
//!
//! The action-efficiency proxy is **oracle-normalized efficiency**
//! (`min(1.15, (oracle_actions / scored_actions)^2)`, zero on failure). It is not RHAE.
//!
//! Gates use paired stratified bootstrap confidence intervals over episode
//! differences (stratified by goal family). P0 was skipped, so all gates remain
//! explicitly exploratory.

use crate::agents::AgentName;
use crate::domain::{Goal, Split};
use anyhow::{bail, Context, Result};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Schema version for byte-stable report JSON.
pub const REPORT_VERSION: u32 = 6;

/// Cap applied to the squared oracle/path ratio.
pub const EFFICIENCY_SCORE_CAP: f64 = 1.15;

/// Default bootstrap replicates for exploratory gates.
pub const DEFAULT_BOOTSTRAP_SAMPLES: u64 = 999;

/// Default RNG seed for bootstrap resampling (fixed in config for determinism).
pub const DEFAULT_BOOTSTRAP_SEED: u64 = 0xB007_57A9_0000_0001;

pub const MAX_BOOTSTRAP_SAMPLES: u64 = 1_000_000;

const GOAL_FAMILIES: [&str; 6] = [
    "reach_marker",
    "collect_all",
    "activate_switches_in_order",
    "preserve_resource_reach_marker",
    "avoid_hazard_reach_marker",
    "trigger_terminal",
];

/// Which P1 suite produced a report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    P1a,
    P1b,
    P1c,
    P1cHard,
    All,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::P1a => "p1a",
            Self::P1b => "p1b",
            Self::P1c => "p1c",
            Self::P1cHard => "p1c_hard",
            Self::All => "all",
        }
    }
}

/// Serializable run configuration. No wall-clock / platform fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunConfig {
    /// Deterministic run list. A single-seed CLI invocation stores one element.
    pub seeds: Vec<u64>,
    pub episodes_per_split: u64,
    pub beam_width: usize,
    pub beam_horizon: u16,
    pub best_of_k: usize,
    pub pause_extra_evals: u64,
    pub bootstrap_seed: u64,
    pub bootstrap_samples: u64,
    pub min_success_lift: f64,
    pub min_efficiency_lift: f64,
    pub output: String,
}

/// Goal-family label for reporting (not the full Goal payload).
pub fn goal_family(goal: &Goal) -> &'static str {
    match goal {
        Goal::ReachMarker { .. } => "reach_marker",
        Goal::CollectAll => "collect_all",
        Goal::ActivateSwitchesInOrder { .. } => "activate_switches_in_order",
        Goal::PreserveResourceReachMarker { .. } => "preserve_resource_reach_marker",
        Goal::AvoidHazardReachMarker { .. } => "avoid_hazard_reach_marker",
        Goal::TriggerTerminal { .. } => "trigger_terminal",
    }
}

/// Squared oracle/path-normalized action efficiency, capped at [`EFFICIENCY_SCORE_CAP`].
///
/// Returns 0 on failure. On success with zero scored actions, returns the score cap
/// (already-satisfied start). Never call this RHAE.
pub fn oracle_normalized_efficiency(
    success: bool,
    scored_env_actions: u64,
    oracle_optimal_actions: u64,
) -> f64 {
    if !success {
        return 0.0;
    }
    if scored_env_actions == 0 {
        return EFFICIENCY_SCORE_CAP;
    }
    let ratio = oracle_optimal_actions as f64 / scored_env_actions as f64;
    (ratio * ratio).min(EFFICIENCY_SCORE_CAP)
}

/// One agent × episode outcome.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpisodeRecord {
    pub seed: u64,
    pub episode_id: u64,
    pub split: Split,
    pub agent: AgentName,
    pub goal_family: String,
    pub success: bool,
    /// Irreversible terminal failure (e.g. avoid-hazard violation).
    pub failed: bool,
    pub scored_env_actions: u64,
    pub expansions: u64,
    pub evaluations: u64,
    pub exhausted: bool,
    /// `None` flags an unsolvable oracle (never substitute the action budget).
    pub oracle_optimal_actions: Option<u64>,
    /// Oracle-normalized efficiency (not RHAE).
    pub oracle_normalized_efficiency: f64,
    /// P1A/P1C: unique surviving candidate equals the hidden index.
    pub correct_objective_identification: Option<bool>,
    /// Env actions until unique hidden-goal identification.
    pub actions_to_identification: Option<u16>,
    pub incorrect_commitments: u64,
    /// P1C only: safe one-step actions advancing a strict majority of live plans.
    pub shared_progress_actions: Option<u64>,
    /// P1C only: actions spent on probes that test at least two exact-live goals.
    pub multi_goal_probe_actions: Option<u64>,
    /// P1C only: candidates removed by observed terminal/nonterminal evidence.
    pub goals_falsified: Option<u64>,
    /// P1C only: changes between the two parallel planning methods.
    pub parallel_method_switches: Option<u64>,
}

/// Aggregates for one agent on one split. Sorted deterministically in the report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aggregate {
    pub agent: AgentName,
    pub split: Split,
    pub n: usize,
    pub success_rate: f64,
    pub terminal_failure_rate: f64,
    pub mean_scored_actions: f64,
    pub mean_oracle_normalized_efficiency: f64,
    pub mean_internal_work: f64,
    /// Fraction of episodes with correct unique ID (P1A discrimination); else null.
    pub identification_accuracy: Option<f64>,
    pub incorrect_commitments: u64,
    /// P1C only: shared-progress actions divided by scored actions.
    pub shared_progress_action_rate: Option<f64>,
    /// P1C only: multi-goal probe actions divided by scored actions.
    pub multi_goal_probe_action_rate: Option<f64>,
    /// P1C only: evidence-falsified goals divided by scored actions.
    pub mean_goals_falsified_per_action: Option<f64>,
    /// P1C only: mean method changes per episode.
    pub mean_parallel_method_switches: Option<f64>,
}

/// Exploratory gate outcome (P0 statistical freezing was skipped).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateOutcome {
    pub name: String,
    pub exploratory: bool,
    pub passed: bool,
    pub candidate: String,
    pub success_baseline: String,
    pub efficiency_baseline: String,
    pub success_lift: Option<f64>,
    pub efficiency_lift: Option<f64>,
    pub success_lift_ci_low: Option<f64>,
    pub success_lift_ci_high: Option<f64>,
    pub efficiency_lift_ci_low: Option<f64>,
    pub efficiency_lift_ci_high: Option<f64>,
    pub min_success_lift: f64,
    pub min_efficiency_lift: f64,
    pub bootstrap_seed: u64,
    pub bootstrap_samples: u64,
    pub detail: String,
}

/// Full versioned report. Identical commands ⇒ byte-identical JSON.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperimentReport {
    pub report_version: u32,
    pub phase: Phase,
    pub config: RunConfig,
    pub episodes: Vec<EpisodeRecord>,
    pub aggregates: Vec<Aggregate>,
    pub gates: Vec<GateOutcome>,
}

impl ExperimentReport {
    pub fn new(phase: Phase, config: RunConfig) -> Self {
        Self {
            report_version: REPORT_VERSION,
            phase,
            config,
            episodes: Vec::new(),
            aggregates: Vec::new(),
            gates: Vec::new(),
        }
    }

    pub fn sort_deterministic(&mut self) {
        self.episodes.sort_by(|a, b| {
            a.seed
                .cmp(&b.seed)
                .then(split_ord(a.split).cmp(&split_ord(b.split)))
                .then(a.episode_id.cmp(&b.episode_id))
                .then(agent_ord(a.agent).cmp(&agent_ord(b.agent)))
        });
        self.aggregates.sort_by(|a, b| {
            agent_ord(a.agent)
                .cmp(&agent_ord(b.agent))
                .then(split_ord(a.split).cmp(&split_ord(b.split)))
        });
        self.gates.sort_by(|a, b| a.name.cmp(&b.name));
    }
}

fn split_ord(s: Split) -> u8 {
    match s {
        Split::Train => 0,
        Split::HeldOutComposition => 1,
    }
}

fn agent_ord(a: AgentName) -> u8 {
    match a {
        AgentName::Random => 0,
        AgentName::NovelState => 1,
        AgentName::GreedyApparentProgress => 2,
        AgentName::CandidateGoalDiscrimination => 3,
        AgentName::SetAwareParallelPlanning => 4,
        AgentName::SharedProgressPlanning => 5,
        AgentName::BroadProgressNarrowFalsify => 6,
        AgentName::BroadFalsifyNarrowProgress => 7,
        AgentName::AlternatingParallelPlanning => 8,
        AgentName::CostAwareParallelPlanning => 9,
        AgentName::CappedBroadProgressPlanning => 10,
        AgentName::OracleObjective => 11,
        AgentName::Reactive => 12,
        AgentName::PauseCompute => 13,
        AgentName::BestOfK => 14,
        AgentName::BeamSearch => 15,
        AgentName::OracleOptimal => 16,
    }
}

/// Aggregate episode records. Empty input ⇒ empty vec (never NaN rates).
pub fn aggregate_episodes(episodes: &[EpisodeRecord]) -> Vec<Aggregate> {
    let mut keys: Vec<(AgentName, Split)> = Vec::new();
    for ep in episodes {
        let k = (ep.agent, ep.split);
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    keys.sort_by(|a, b| {
        agent_ord(a.0)
            .cmp(&agent_ord(b.0))
            .then(split_ord(a.1).cmp(&split_ord(b.1)))
    });

    keys.into_iter()
        .map(|(agent, split)| {
            let group: Vec<&EpisodeRecord> = episodes
                .iter()
                .filter(|e| e.agent == agent && e.split == split)
                .collect();
            aggregate_group(agent, split, &group)
        })
        .collect()
}

fn aggregate_group(agent: AgentName, split: Split, group: &[&EpisodeRecord]) -> Aggregate {
    let n = group.len();
    if n == 0 {
        return Aggregate {
            agent,
            split,
            n: 0,
            success_rate: 0.0,
            terminal_failure_rate: 0.0,
            mean_scored_actions: 0.0,
            mean_oracle_normalized_efficiency: 0.0,
            mean_internal_work: 0.0,
            identification_accuracy: None,
            incorrect_commitments: 0,
            shared_progress_action_rate: None,
            multi_goal_probe_action_rate: None,
            mean_goals_falsified_per_action: None,
            mean_parallel_method_switches: None,
        };
    }
    let n_f = n as f64;
    let successes = group.iter().filter(|e| e.success).count() as f64;
    let failures = group.iter().filter(|e| e.failed).count() as f64;
    let mean_scored = group
        .iter()
        .map(|e| e.scored_env_actions as f64)
        .sum::<f64>()
        / n_f;
    let mean_eff = group
        .iter()
        .map(|e| e.oracle_normalized_efficiency)
        .sum::<f64>()
        / n_f;
    let mean_work = group
        .iter()
        .map(|e| (e.expansions + e.evaluations) as f64)
        .sum::<f64>()
        / n_f;
    let incorrect: u64 = group.iter().map(|e| e.incorrect_commitments).sum();
    let p1c_rows: Vec<&EpisodeRecord> = group
        .iter()
        .copied()
        .filter(|e| e.multi_goal_probe_actions.is_some())
        .collect();
    let (
        shared_progress_action_rate,
        multi_goal_probe_action_rate,
        mean_goals_falsified_per_action,
        mean_parallel_method_switches,
    ) = if p1c_rows.is_empty() {
        (None, None, None, None)
    } else {
        let actions: u64 = p1c_rows.iter().map(|e| e.scored_env_actions).sum();
        let shared: u64 = p1c_rows
            .iter()
            .filter_map(|e| e.shared_progress_actions)
            .sum();
        let multi: u64 = p1c_rows
            .iter()
            .filter_map(|e| e.multi_goal_probe_actions)
            .sum();
        let falsified: u64 = p1c_rows.iter().filter_map(|e| e.goals_falsified).sum();
        let switches: u64 = p1c_rows
            .iter()
            .filter_map(|e| e.parallel_method_switches)
            .sum();
        let denominator = actions.max(1) as f64;
        (
            Some(shared as f64 / denominator),
            Some(multi as f64 / denominator),
            Some(falsified as f64 / denominator),
            Some(switches as f64 / n_f),
        )
    };

    let identification_accuracy = if group
        .iter()
        .any(|e| e.correct_objective_identification.is_some())
    {
        let correct = group
            .iter()
            .filter(|e| e.correct_objective_identification == Some(true))
            .count() as f64;
        Some(correct / n_f)
    } else {
        None
    };

    Aggregate {
        agent,
        split,
        n,
        success_rate: successes / n_f,
        terminal_failure_rate: failures / n_f,
        mean_scored_actions: mean_scored,
        mean_oracle_normalized_efficiency: mean_eff,
        mean_internal_work: mean_work,
        identification_accuracy,
        incorrect_commitments: incorrect,
        shared_progress_action_rate,
        multi_goal_probe_action_rate,
        mean_goals_falsified_per_action,
        mean_parallel_method_switches,
    }
}

/// Look up an aggregate; `None` if missing or empty (`n == 0`).
pub fn find_aggregate(aggs: &[Aggregate], agent: AgentName, split: Split) -> Option<&Aggregate> {
    aggs.iter()
        .find(|a| a.agent == agent && a.split == split && a.n > 0)
}

/// Absolute lift of candidate over baseline; `None` if either group is unusable.
pub fn metric_lift(
    candidate: Option<&Aggregate>,
    baseline: Option<&Aggregate>,
) -> Option<(f64, f64)> {
    let c = candidate?;
    let b = baseline?;
    if c.n == 0 || b.n == 0 {
        return None;
    }
    let s = c.success_rate - b.success_rate;
    let e = c.mean_oracle_normalized_efficiency - b.mean_oracle_normalized_efficiency;
    if !s.is_finite() || !e.is_finite() {
        return None;
    }
    Some((s, e))
}

/// Paired per-episode differences stratified by goal family.
///
/// Keys are `(seed, episode_id)` within `split`. Returns `None` on empty,
/// mismatched pairing, or non-finite values (fail closed).
pub fn paired_metric_diffs<F>(
    episodes: &[EpisodeRecord],
    candidate: AgentName,
    baseline: AgentName,
    split: Split,
    value: F,
) -> Option<Vec<(String, f64)>>
where
    F: Fn(&EpisodeRecord) -> f64,
{
    let mut cand: BTreeMap<(u64, u64), &EpisodeRecord> = BTreeMap::new();
    let mut base: BTreeMap<(u64, u64), &EpisodeRecord> = BTreeMap::new();
    for ep in episodes {
        if ep.split != split {
            continue;
        }
        let key = (ep.seed, ep.episode_id);
        if ep.agent == candidate {
            cand.insert(key, ep);
        } else if ep.agent == baseline {
            base.insert(key, ep);
        }
    }
    if cand.is_empty() || base.is_empty() || cand.len() != base.len() {
        return None;
    }
    let mut out = Vec::with_capacity(cand.len());
    for (key, c) in &cand {
        let b = base.get(key)?;
        if c.goal_family != b.goal_family {
            return None;
        }
        let diff = value(c) - value(b);
        if !diff.is_finite() {
            return None;
        }
        out.push((c.goal_family.clone(), diff));
    }
    Some(out)
}

/// Deterministic paired stratified bootstrap CI for the mean difference.
///
/// Stratifies by the string key (goal family): each stratum is resampled with
/// replacement to its original size. Returns `(point_mean, ci_low, ci_high)` for
/// a two-sided 95% nearest-rank interval, or `None` to fail closed.
pub fn stratified_bootstrap_mean_ci(
    stratified_values: &[(String, f64)],
    bootstrap_seed: u64,
    bootstrap_samples: u64,
) -> Option<(f64, f64, f64)> {
    if stratified_values.is_empty() || bootstrap_samples == 0 {
        return None;
    }
    if stratified_values.iter().any(|(_, v)| !v.is_finite()) {
        return None;
    }

    let mut groups: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for (k, v) in stratified_values {
        groups.entry(k.as_str()).or_default().push(*v);
    }
    if groups.is_empty() {
        return None;
    }

    let n_total = stratified_values.len() as f64;
    let point = stratified_values.iter().map(|(_, v)| *v).sum::<f64>() / n_total;

    let mut rng = ChaCha8Rng::seed_from_u64(bootstrap_seed);
    let mut boots = Vec::with_capacity(bootstrap_samples as usize);
    for _ in 0..bootstrap_samples {
        let mut sum = 0.0;
        let mut count = 0usize;
        for vals in groups.values() {
            let n = vals.len();
            for _ in 0..n {
                let idx = rng.random_range(0..n);
                sum += vals[idx];
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        boots.push(sum / count as f64);
    }
    boots.sort_by(|a, b| a.partial_cmp(b).expect("finite bootstrap means"));
    let lo = nearest_rank_percentile(&boots, 0.025)?;
    let hi = nearest_rank_percentile(&boots, 0.975)?;
    if !lo.is_finite() || !hi.is_finite() || !point.is_finite() {
        return None;
    }
    Some((point, lo, hi))
}

fn nearest_rank_percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    let n = sorted.len();
    let rank = ((p * n as f64).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    Some(sorted[rank])
}

#[derive(Clone, Copy, Debug)]
pub struct BootstrapGateSpec<'a> {
    pub name: &'a str,
    pub candidate: AgentName,
    pub success_baseline: Option<AgentName>,
    pub efficiency_baseline: Option<AgentName>,
    pub split: Split,
    pub min_success_lift: f64,
    pub min_efficiency_lift: f64,
    pub bootstrap_seed: u64,
    pub bootstrap_samples: u64,
}

/// Exploratory bootstrap gate: lower CI bounds must meet predeclared lifts.
pub fn evaluate_bootstrap_gate(
    spec: &BootstrapGateSpec<'_>,
    episodes: &[EpisodeRecord],
) -> GateOutcome {
    let success_baseline_name = spec
        .success_baseline
        .map(AgentName::as_str)
        .unwrap_or("none");
    let efficiency_baseline_name = spec
        .efficiency_baseline
        .map(AgentName::as_str)
        .unwrap_or("none");
    let fail = |detail: String| GateOutcome {
        name: spec.name.to_string(),
        exploratory: true,
        passed: false,
        candidate: spec.candidate.as_str().to_string(),
        success_baseline: success_baseline_name.to_string(),
        efficiency_baseline: efficiency_baseline_name.to_string(),
        success_lift: None,
        efficiency_lift: None,
        success_lift_ci_low: None,
        success_lift_ci_high: None,
        efficiency_lift_ci_low: None,
        efficiency_lift_ci_high: None,
        min_success_lift: spec.min_success_lift,
        min_efficiency_lift: spec.min_efficiency_lift,
        bootstrap_seed: spec.bootstrap_seed,
        bootstrap_samples: spec.bootstrap_samples,
        detail,
    };

    let (Some(sb), Some(eb)) = (spec.success_baseline, spec.efficiency_baseline) else {
        return fail("missing baseline; exploratory bootstrap gate fails closed".into());
    };

    let candidate_records: Vec<&EpisodeRecord> = episodes
        .iter()
        .filter(|ep| ep.split == spec.split && ep.agent == spec.candidate)
        .collect();
    let seeds: BTreeSet<u64> = candidate_records.iter().map(|ep| ep.seed).collect();
    if seeds.len() < 2 {
        return fail("fewer than two independent seeds; exploratory gate fails closed".into());
    }
    let mut family_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for ep in &candidate_records {
        *family_counts.entry(ep.goal_family.as_str()).or_default() += 1;
    }
    if GOAL_FAMILIES
        .iter()
        .any(|family| family_counts.get(family).copied().unwrap_or(0) < 2)
    {
        return fail("incomplete six-family replication; exploratory gate fails closed".into());
    }

    let Some(success_pairs) = paired_metric_diffs(episodes, spec.candidate, sb, spec.split, |e| {
        if e.success {
            1.0
        } else {
            0.0
        }
    }) else {
        return fail(
            "empty or mismatched success pairs; exploratory bootstrap gate fails closed".into(),
        );
    };
    let Some(eff_pairs) = paired_metric_diffs(episodes, spec.candidate, eb, spec.split, |e| {
        e.oracle_normalized_efficiency
    }) else {
        return fail(
            "empty or mismatched efficiency pairs; exploratory bootstrap gate fails closed".into(),
        );
    };

    let Some((s_lift, s_lo, s_hi)) =
        stratified_bootstrap_mean_ci(&success_pairs, spec.bootstrap_seed, spec.bootstrap_samples)
    else {
        return fail("success bootstrap CI unavailable; exploratory gate fails closed".into());
    };
    let Some((e_lift, e_lo, e_hi)) = stratified_bootstrap_mean_ci(
        &eff_pairs,
        spec.bootstrap_seed ^ 0xE1F1,
        spec.bootstrap_samples,
    ) else {
        return fail("efficiency bootstrap CI unavailable; exploratory gate fails closed".into());
    };

    let passed = s_lo >= spec.min_success_lift && e_lo >= spec.min_efficiency_lift;
    GateOutcome {
        name: spec.name.to_string(),
        exploratory: true,
        passed,
        candidate: spec.candidate.as_str().to_string(),
        success_baseline: success_baseline_name.to_string(),
        efficiency_baseline: efficiency_baseline_name.to_string(),
        success_lift: Some(s_lift),
        efficiency_lift: Some(e_lift),
        success_lift_ci_low: Some(s_lo),
        success_lift_ci_high: Some(s_hi),
        efficiency_lift_ci_low: Some(e_lo),
        efficiency_lift_ci_high: Some(e_hi),
        min_success_lift: spec.min_success_lift,
        min_efficiency_lift: spec.min_efficiency_lift,
        bootstrap_seed: spec.bootstrap_seed,
        bootstrap_samples: spec.bootstrap_samples,
        detail: if passed {
            format!(
                "exploratory pass: success_ci_low={s_lo:.4} (need {}), \
                 efficiency_ci_low={e_lo:.4} (need {})",
                spec.min_success_lift, spec.min_efficiency_lift
            )
        } else {
            format!(
                "exploratory fail: success_ci_low={s_lo:.4} (need {}), \
                 efficiency_ci_low={e_lo:.4} (need {})",
                spec.min_success_lift, spec.min_efficiency_lift
            )
        },
    }
}

/// Pick the success-strongest aggregate among `candidates`.
pub fn strongest_by_success<'a>(
    aggs: &'a [Aggregate],
    candidates: &[AgentName],
    split: Split,
) -> Option<&'a Aggregate> {
    let mut best: Option<&Aggregate> = None;
    for &name in candidates {
        let Some(a) = find_aggregate(aggs, name, split) else {
            continue;
        };
        match best {
            None => best = Some(a),
            Some(b) => {
                let better = a.success_rate > b.success_rate
                    || (a.success_rate == b.success_rate
                        && a.mean_oracle_normalized_efficiency
                            > b.mean_oracle_normalized_efficiency)
                    || (a.success_rate == b.success_rate
                        && a.mean_oracle_normalized_efficiency
                            == b.mean_oracle_normalized_efficiency
                        && agent_ord(a.agent) < agent_ord(b.agent));
                if better {
                    best = Some(a);
                }
            }
        }
    }
    best
}

/// Pick the efficiency-strongest aggregate among `candidates`.
pub fn strongest_by_efficiency<'a>(
    aggs: &'a [Aggregate],
    candidates: &[AgentName],
    split: Split,
) -> Option<&'a Aggregate> {
    let mut best: Option<&Aggregate> = None;
    for &name in candidates {
        let Some(a) = find_aggregate(aggs, name, split) else {
            continue;
        };
        match best {
            None => best = Some(a),
            Some(b) => {
                let better = a.mean_oracle_normalized_efficiency
                    > b.mean_oracle_normalized_efficiency
                    || (a.mean_oracle_normalized_efficiency == b.mean_oracle_normalized_efficiency
                        && a.success_rate > b.success_rate)
                    || (a.mean_oracle_normalized_efficiency == b.mean_oracle_normalized_efficiency
                        && a.success_rate == b.success_rate
                        && agent_ord(a.agent) < agent_ord(b.agent));
                if better {
                    best = Some(a);
                }
            }
        }
    }
    best
}

/// Atomically write pretty JSON (temp beside destination, then rename).
pub fn write_report_atomic(path: &Path, report: &ExperimentReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create parent dirs for {}", path.display()))?;
        }
    }
    let mut sorted = report.clone();
    sorted.sort_deterministic();
    let json = serde_json::to_string_pretty(&sorted).context("serialize report")?;
    let tmp = temp_beside(path);
    fs::write(&tmp, &json).with_context(|| format!("write temp {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| {
        format!(
            "rename {} -> {} (atomic replace)",
            tmp.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn temp_beside(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_owned();
    os.push(".tmp");
    PathBuf::from(os)
}

/// Compact human-readable summary for stdout.
pub fn print_summary(report: &ExperimentReport) {
    println!(
        "phase={} version={} seeds={:?} episodes_per_split={} bootstrap=({},{})",
        report.phase.as_str(),
        report.report_version,
        report.config.seeds,
        report.config.episodes_per_split,
        report.config.bootstrap_seed,
        report.config.bootstrap_samples
    );
    for agg in &report.aggregates {
        let id = match agg.identification_accuracy {
            Some(v) => format!(" id_acc={v:.3}"),
            None => String::new(),
        };
        let parallel = match (
            agg.shared_progress_action_rate,
            agg.multi_goal_probe_action_rate,
            agg.mean_goals_falsified_per_action,
            agg.mean_parallel_method_switches,
        ) {
            (Some(shared), Some(probe), Some(falsified), Some(switches)) => {
                format!(
                    " shared={shared:.3} multi_probe={probe:.3} falsified/action={falsified:.3} switches/episode={switches:.2}"
                )
            }
            _ => String::new(),
        };
        println!(
            "  {:?}/{:?} n={} success={:.3} terminal_fail={:.3} actions={:.2} oneff={:.3} work={:.1}{}{} commits={}",
            agg.agent,
            agg.split,
            agg.n,
            agg.success_rate,
            agg.terminal_failure_rate,
            agg.mean_scored_actions,
            agg.mean_oracle_normalized_efficiency,
            agg.mean_internal_work,
            id,
            parallel,
            agg.incorrect_commitments
        );
    }
    for g in &report.gates {
        let mark = if g.passed { "PASS" } else { "FAIL" };
        println!(
            "  gate[{}] {} (exploratory) {} vs success_base={} efficiency_base={}: {}",
            g.name, mark, g.candidate, g.success_baseline, g.efficiency_baseline, g.detail
        );
    }
}

/// Serialize to pretty JSON bytes (sorted). Useful for determinism tests.
pub fn report_bytes(report: &ExperimentReport) -> Result<Vec<u8>> {
    let mut sorted = report.clone();
    sorted.sort_deterministic();
    let s = serde_json::to_string_pretty(&sorted)?;
    Ok(s.into_bytes())
}

/// Fail if report schema version mismatches.
pub fn validate_report(report: &ExperimentReport) -> Result<()> {
    if report.report_version != REPORT_VERSION {
        bail!(
            "unsupported report_version {} (expected {})",
            report.report_version,
            REPORT_VERSION
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ep(
        seed: u64,
        episode_id: u64,
        split: Split,
        agent: AgentName,
        success: bool,
        actions: u64,
        family: &str,
    ) -> EpisodeRecord {
        let oracle = Some(6);
        EpisodeRecord {
            seed,
            episode_id,
            split,
            agent,
            goal_family: family.into(),
            success,
            failed: false,
            scored_env_actions: actions,
            expansions: 1,
            evaluations: 2,
            exhausted: !success,
            oracle_optimal_actions: oracle,
            oracle_normalized_efficiency: oracle_normalized_efficiency(success, actions, 6),
            correct_objective_identification: None,
            actions_to_identification: None,
            incorrect_commitments: 0,
            shared_progress_actions: None,
            multi_goal_probe_actions: None,
            goals_falsified: None,
            parallel_method_switches: None,
        }
    }

    #[test]
    fn efficiency_math() {
        assert_eq!(oracle_normalized_efficiency(false, 10, 5), 0.0);
        assert!((oracle_normalized_efficiency(true, 10, 5) - 0.25).abs() < 1e-12);
        let capped = oracle_normalized_efficiency(true, 10, 100);
        assert!((capped - EFFICIENCY_SCORE_CAP).abs() < 1e-12);
        assert!((oracle_normalized_efficiency(true, 0, 0) - EFFICIENCY_SCORE_CAP).abs() < 1e-12);
    }

    #[test]
    fn aggregates_p1c_strategy_counters() {
        let mut ep = sample_ep(
            1,
            0,
            Split::Train,
            AgentName::SetAwareParallelPlanning,
            true,
            4,
            "reach_marker",
        );
        ep.shared_progress_actions = Some(1);
        ep.multi_goal_probe_actions = Some(2);
        ep.goals_falsified = Some(5);
        ep.parallel_method_switches = Some(3);
        let aggs = aggregate_episodes(&[ep]);
        assert_eq!(aggs.len(), 1);
        assert_eq!(aggs[0].shared_progress_action_rate, Some(0.25));
        assert_eq!(aggs[0].multi_goal_probe_action_rate, Some(0.5));
        assert_eq!(aggs[0].mean_goals_falsified_per_action, Some(1.25));
        assert_eq!(aggs[0].mean_parallel_method_switches, Some(3.0));
    }

    #[test]
    fn bootstrap_fail_closed_on_empty() {
        assert!(stratified_bootstrap_mean_ci(&[], DEFAULT_BOOTSTRAP_SEED, 99).is_none());
        assert!(
            stratified_bootstrap_mean_ci(&[("a".into(), 1.0)], DEFAULT_BOOTSTRAP_SEED, 0).is_none()
        );
        let gate = evaluate_bootstrap_gate(
            &BootstrapGateSpec {
                name: "empty",
                candidate: AgentName::Random,
                success_baseline: None,
                efficiency_baseline: None,
                split: Split::HeldOutComposition,
                min_success_lift: 0.05,
                min_efficiency_lift: 0.05,
                bootstrap_seed: DEFAULT_BOOTSTRAP_SEED,
                bootstrap_samples: DEFAULT_BOOTSTRAP_SAMPLES,
            },
            &[],
        );
        assert!(gate.exploratory);
        assert!(!gate.passed);
        assert!(gate.success_lift_ci_low.is_none());
    }

    #[test]
    fn bootstrap_determinism() {
        let vals = vec![
            ("reach_marker".into(), 0.2),
            ("reach_marker".into(), 0.1),
            ("reach_marker".into(), -0.05),
            ("collect_all".into(), 0.3),
            ("collect_all".into(), -0.1),
            ("collect_all".into(), 0.45),
            ("trigger_terminal".into(), 0.0),
            ("trigger_terminal".into(), 0.7),
            ("avoid_hazard_reach_marker".into(), -0.2),
            ("avoid_hazard_reach_marker".into(), 0.15),
        ];
        let a = stratified_bootstrap_mean_ci(&vals, 42, 500).unwrap();
        let b = stratified_bootstrap_mean_ci(&vals, 42, 500).unwrap();
        assert_eq!(a, b);
        let c = stratified_bootstrap_mean_ci(&vals, 43, 500).unwrap();
        // Different seeds must change at least one CI endpoint for this mix.
        assert!(
            a.1 != c.1 || a.2 != c.2,
            "expected seed-sensitive CI bounds"
        );
    }

    #[test]
    fn separate_strongest_baselines() {
        let aggs = vec![
            Aggregate {
                agent: AgentName::Random,
                split: Split::HeldOutComposition,
                n: 4,
                success_rate: 0.80,
                terminal_failure_rate: 0.0,
                mean_scored_actions: 30.0,
                mean_oracle_normalized_efficiency: 0.10,
                mean_internal_work: 1.0,
                identification_accuracy: None,
                incorrect_commitments: 0,
                shared_progress_action_rate: None,
                multi_goal_probe_action_rate: None,
                mean_goals_falsified_per_action: None,
                mean_parallel_method_switches: None,
            },
            Aggregate {
                agent: AgentName::NovelState,
                split: Split::HeldOutComposition,
                n: 4,
                success_rate: 0.50,
                terminal_failure_rate: 0.0,
                mean_scored_actions: 20.0,
                mean_oracle_normalized_efficiency: 0.40,
                mean_internal_work: 2.0,
                identification_accuracy: None,
                incorrect_commitments: 0,
                shared_progress_action_rate: None,
                multi_goal_probe_action_rate: None,
                mean_goals_falsified_per_action: None,
                mean_parallel_method_switches: None,
            },
            Aggregate {
                agent: AgentName::GreedyApparentProgress,
                split: Split::HeldOutComposition,
                n: 4,
                success_rate: 0.60,
                terminal_failure_rate: 0.0,
                mean_scored_actions: 25.0,
                mean_oracle_normalized_efficiency: 0.20,
                mean_internal_work: 3.0,
                identification_accuracy: None,
                incorrect_commitments: 0,
                shared_progress_action_rate: None,
                multi_goal_probe_action_rate: None,
                mean_goals_falsified_per_action: None,
                mean_parallel_method_switches: None,
            },
        ];
        let names = [
            AgentName::Random,
            AgentName::NovelState,
            AgentName::GreedyApparentProgress,
        ];
        let by_s = strongest_by_success(&aggs, &names, Split::HeldOutComposition).unwrap();
        let by_e = strongest_by_efficiency(&aggs, &names, Split::HeldOutComposition).unwrap();
        assert_eq!(by_s.agent, AgentName::Random);
        assert_eq!(by_e.agent, AgentName::NovelState);
        assert_ne!(by_s.agent, by_e.agent);
    }

    #[test]
    fn bootstrap_gate_records_both_baselines() {
        let split = Split::HeldOutComposition;
        let mut eps = Vec::new();
        for seed in [1, 2] {
            for (id, family) in GOAL_FAMILIES.iter().enumerate() {
                let id = id as u64;
                eps.push(sample_ep(
                    seed,
                    id,
                    split,
                    AgentName::CandidateGoalDiscrimination,
                    true,
                    8,
                    family,
                ));
                eps.push(sample_ep(
                    seed,
                    id,
                    split,
                    AgentName::Random,
                    !id.is_multiple_of(3),
                    12,
                    family,
                ));
                eps.push(sample_ep(
                    seed,
                    id,
                    split,
                    AgentName::NovelState,
                    false,
                    10,
                    family,
                ));
            }
        }
        let gate = evaluate_bootstrap_gate(
            &BootstrapGateSpec {
                name: "p1a_demo",
                candidate: AgentName::CandidateGoalDiscrimination,
                success_baseline: Some(AgentName::Random),
                efficiency_baseline: Some(AgentName::NovelState),
                split,
                min_success_lift: 0.01,
                min_efficiency_lift: 0.01,
                bootstrap_seed: 7,
                bootstrap_samples: 199,
            },
            &eps,
        );
        assert_eq!(gate.success_baseline, "random");
        assert_eq!(gate.efficiency_baseline, "novel_state");
        assert!(gate.exploratory);
        assert!(gate.success_lift_ci_low.is_some());
        assert!(gate.efficiency_lift_ci_low.is_some());
    }

    #[test]
    fn byte_identical_reports() {
        let mut a = ExperimentReport::new(
            Phase::P1a,
            RunConfig {
                seeds: vec![1],
                episodes_per_split: 2,
                beam_width: 8,
                beam_horizon: 24,
                best_of_k: 4,
                pause_extra_evals: 16,
                bootstrap_seed: DEFAULT_BOOTSTRAP_SEED,
                bootstrap_samples: DEFAULT_BOOTSTRAP_SAMPLES,
                min_success_lift: 0.05,
                min_efficiency_lift: 0.05,
                output: "runs/p1/out.json".into(),
            },
        );
        a.episodes.push(sample_ep(
            1,
            1,
            Split::Train,
            AgentName::Random,
            true,
            8,
            "reach_marker",
        ));
        a.episodes.push(sample_ep(
            1,
            0,
            Split::Train,
            AgentName::NovelState,
            false,
            10,
            "reach_marker",
        ));
        a.aggregates = aggregate_episodes(&a.episodes);
        a.gates.push(evaluate_bootstrap_gate(
            &BootstrapGateSpec {
                name: "demo",
                candidate: AgentName::Random,
                success_baseline: Some(AgentName::NovelState),
                efficiency_baseline: Some(AgentName::NovelState),
                split: Split::Train,
                min_success_lift: 0.05,
                min_efficiency_lift: 0.05,
                bootstrap_seed: DEFAULT_BOOTSTRAP_SEED,
                bootstrap_samples: 99,
            },
            &a.episodes,
        ));

        let mut b = a.clone();
        b.episodes.reverse();
        let ba = report_bytes(&a).unwrap();
        let bb = report_bytes(&b).unwrap();
        assert_eq!(ba, bb);
        assert!(!String::from_utf8_lossy(&ba).contains("timestamp"));
        assert!(!String::from_utf8_lossy(&ba).contains("RHAE"));
        assert!(String::from_utf8_lossy(&ba).contains("oracle_normalized_efficiency"));
        assert_eq!(a.report_version, REPORT_VERSION);
    }

    #[test]
    fn atomic_write_roundtrip() {
        let dir = std::env::temp_dir().join(format!("tofy_report_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nested").join("r.json");
        let mut report = ExperimentReport::new(
            Phase::P1b,
            RunConfig {
                seeds: vec![3],
                episodes_per_split: 1,
                beam_width: 4,
                beam_horizon: 8,
                best_of_k: 2,
                pause_extra_evals: 4,
                bootstrap_seed: DEFAULT_BOOTSTRAP_SEED,
                bootstrap_samples: 99,
                min_success_lift: 0.01,
                min_efficiency_lift: 0.01,
                output: path.display().to_string(),
            },
        );
        report.episodes.push(sample_ep(
            3,
            0,
            Split::Train,
            AgentName::Reactive,
            true,
            6,
            "reach_marker",
        ));
        report.aggregates = aggregate_episodes(&report.episodes);
        write_report_atomic(&path, &report).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let loaded: ExperimentReport = serde_json::from_str(&text).unwrap();
        assert_eq!(loaded.report_version, REPORT_VERSION);
        assert_eq!(loaded.episodes.len(), 1);
        assert_eq!(loaded.episodes[0].seed, 3);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn goal_family_covers_new_variants() {
        assert_eq!(
            goal_family(&Goal::AvoidHazardReachMarker {
                hazard: 0,
                marker: 1
            }),
            "avoid_hazard_reach_marker"
        );
        assert_eq!(
            goal_family(&Goal::TriggerTerminal { trigger: 0 }),
            "trigger_terminal"
        );
    }
}
