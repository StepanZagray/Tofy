//! P1A / P1B / P1C exact-simulator agent controllers (no neural model).

use crate::domain::{
    goal_satisfied, goal_terminal_failure, legal_actions, Action, Goal, Scenario, Simulator,
    StepOutcome,
};
use crate::search::{
    beam_search, goal_viable_at, greedy_action, heuristic, shortest_path, StateKey,
};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashSet};

/// Serializable agent identity for P1A, P1B, and P1C.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentName {
    // P1A
    Random,
    NovelState,
    GreedyApparentProgress,
    CandidateGoalDiscrimination,
    OracleObjective,
    // P1B
    Reactive,
    PauseCompute,
    BestOfK,
    BeamSearch,
    OracleOptimal,
    // P1C
    SetAwareParallelPlanning,
    SharedProgressPlanning,
    BroadProgressNarrowFalsify,
    BroadFalsifyNarrowProgress,
    AlternatingParallelPlanning,
    CostAwareParallelPlanning,
    CappedBroadProgressPlanning,
}

impl AgentName {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::NovelState => "novel_state",
            Self::GreedyApparentProgress => "greedy_apparent_progress",
            Self::CandidateGoalDiscrimination => "candidate_goal_discrimination",
            Self::OracleObjective => "oracle_objective",
            Self::Reactive => "reactive",
            Self::PauseCompute => "pause_compute",
            Self::BestOfK => "best_of_k",
            Self::BeamSearch => "beam_search",
            Self::OracleOptimal => "oracle_optimal",
            Self::SetAwareParallelPlanning => "set_aware_parallel_planning",
            Self::SharedProgressPlanning => "shared_progress_planning",
            Self::BroadProgressNarrowFalsify => "broad_progress_narrow_falsify",
            Self::BroadFalsifyNarrowProgress => "broad_falsify_narrow_progress",
            Self::AlternatingParallelPlanning => "alternating_parallel_planning",
            Self::CostAwareParallelPlanning => "cost_aware_parallel_planning",
            Self::CappedBroadProgressPlanning => "capped_broad_progress_planning",
        }
    }

    pub fn is_p1c_strategy(self) -> bool {
        matches!(
            self,
            Self::SetAwareParallelPlanning
                | Self::SharedProgressPlanning
                | Self::BroadProgressNarrowFalsify
                | Self::BroadFalsifyNarrowProgress
                | Self::AlternatingParallelPlanning
                | Self::CostAwareParallelPlanning
                | Self::CappedBroadProgressPlanning
        )
    }

    pub fn is_p1a(self) -> bool {
        matches!(
            self,
            Self::Random
                | Self::NovelState
                | Self::GreedyApparentProgress
                | Self::CandidateGoalDiscrimination
                | Self::OracleObjective
        ) || self.is_p1c_strategy()
    }

    pub fn is_oracle(self) -> bool {
        matches!(self, Self::OracleObjective | Self::OracleOptimal)
    }
}

/// Serializable run configuration. No wall-clock fields.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: AgentName,
    /// Known planning objective for P1B non-oracle planners (not the hidden index).
    pub planning_goal: Option<Goal>,
    pub best_of_k: usize,
    pub beam_width: usize,
    pub beam_horizon: u16,
    /// Extra non-rollout heuristic evaluations for `pause_compute`.
    pub pause_extra_evals: u64,
}

impl AgentConfig {
    pub fn for_name(name: AgentName) -> Self {
        Self {
            name,
            planning_goal: None,
            best_of_k: 4,
            beam_width: 8,
            beam_horizon: 24,
            pause_extra_evals: 16,
        }
    }
}

/// Episode statistics: environment actions vs internal compute, separately.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentStats {
    pub env_actions: u64,
    pub expansions: u64,
    pub evaluations: u64,
    pub success: bool,
    pub exhausted: bool,
    /// Irreversible terminal failure from the environment.
    pub failed: bool,
    /// Oracle search found no feasible plan under the action budget.
    pub unsolvable: bool,
    /// Env actions until the hidden-goal hypothesis set reached size 1.
    pub actions_to_unique_id: Option<u16>,
    /// Deliberate tests that falsified a targeted candidate.
    pub incorrect_commitments: u64,
    pub identified_candidate: Option<usize>,
    /// Actions spent executing probes whose endpoint tests ≥2 candidates.
    pub multi_goal_probe_actions: u64,
    /// Candidates falsified by terminal/nonterminal evidence.
    pub goals_falsified: u64,
    /// Actions taken as strict-majority shared-progress steps.
    pub shared_progress_actions: u64,
    /// Changes between progress and falsification parallel primitives.
    pub parallel_method_switches: u64,
}

/// Apply a step outcome. Returns true when the episode must stop.
fn note_outcome(stats: &mut AgentStats, out: &StepOutcome) -> bool {
    if out.failed {
        stats.failed = true;
        return true;
    }
    if out.success {
        stats.success = true;
        return true;
    }
    if out.exhausted {
        stats.exhausted = true;
        return true;
    }
    false
}

/// Fixed public proxy used by greedy apparent-progress without consulting the hidden index.
/// It can coincide with the hidden goal by chance, as any public hypothesis can.
pub fn apparent_progress_proxy(scenario: &Scenario) -> Goal {
    if let Some(g) = scenario
        .candidate_goals
        .iter()
        .find(|g| matches!(g, Goal::ReachMarker { .. }))
    {
        return g.clone();
    }
    if !scenario.markers.is_empty() {
        Goal::ReachMarker { marker: 0 }
    } else {
        Goal::CollectAll
    }
}

/// Run one episode under `config`. Only oracle agents may receive `true_goal`.
pub fn run_episode(
    scenario: Scenario,
    config: &AgentConfig,
    rng: &mut ChaCha8Rng,
    true_goal: Option<&Goal>,
) -> AgentStats {
    if config.name.is_oracle() {
        assert!(true_goal.is_some(), "oracle requires true_goal");
    } else {
        assert!(
            true_goal.is_none(),
            "non-oracle agents must not receive true_goal"
        );
    }
    let initial_sim = Simulator::new(scenario.clone());
    if let Some(outcome) = initial_sim.terminal_outcome() {
        let mut stats = AgentStats::default();
        if matches!(config.name, AgentName::CandidateGoalDiscrimination)
            || config.name.is_p1c_strategy()
        {
            let mut alive: BTreeSet<usize> = (0..scenario.candidate_goals.len()).collect();
            let falsified = update_belief_from_outcome(
                &initial_sim,
                &scenario.candidate_goals,
                &mut alive,
                &outcome,
            );
            if config.name.is_p1c_strategy() {
                stats.goals_falsified += falsified.len() as u64;
            }
            note_unique_id(&mut stats, &alive);
        }
        note_outcome(&mut stats, &outcome);
        return stats;
    }
    if config.name.is_oracle() {
        let g = true_goal.expect("oracle requires true_goal");
        return match config.name {
            AgentName::OracleObjective => run_oracle_objective(scenario, g),
            AgentName::OracleOptimal => run_oracle_optimal(scenario, g),
            _ => unreachable!(),
        };
    }
    if let Some(routing) = routing_strategy(config.name) {
        return run_exact_live_controller(scenario, routing);
    }
    match config.name {
        AgentName::Random => run_random(scenario, rng),
        AgentName::NovelState => run_novel_state(scenario, rng),
        AgentName::GreedyApparentProgress => run_greedy_apparent(scenario),
        AgentName::CandidateGoalDiscrimination => run_candidate_discrimination(scenario),
        AgentName::Reactive => run_reactive(scenario, config),
        AgentName::PauseCompute => run_pause_compute(scenario, config),
        AgentName::BestOfK => run_best_of_k(scenario, config, rng),
        AgentName::BeamSearch => run_beam_agent(scenario, config),
        AgentName::OracleObjective
        | AgentName::OracleOptimal
        | AgentName::SetAwareParallelPlanning
        | AgentName::SharedProgressPlanning
        | AgentName::BroadProgressNarrowFalsify
        | AgentName::BroadFalsifyNarrowProgress
        | AgentName::AlternatingParallelPlanning
        | AgentName::CostAwareParallelPlanning
        | AgentName::CappedBroadProgressPlanning => unreachable!(),
    }
}

fn run_random(scenario: Scenario, rng: &mut ChaCha8Rng) -> AgentStats {
    let mut sim = Simulator::new(scenario);
    let mut stats = AgentStats::default();
    loop {
        let acts = sim.legal_actions();
        let a = *acts.choose(rng).expect("actions");
        let out = sim.step(a);
        stats.env_actions += 1;
        if note_outcome(&mut stats, &out) {
            break;
        }
    }
    stats
}

fn run_novel_state(scenario: Scenario, rng: &mut ChaCha8Rng) -> AgentStats {
    let mut sim = Simulator::new(scenario);
    let mut visited = HashSet::new();
    // Novelty is defined over environment state, not path-dependent Undo
    // history; otherwise every longer walk looks novel forever.
    visited.insert(StateKey::from_state(sim.state(), false));
    let mut stats = AgentStats::default();
    loop {
        let acts = sim.legal_actions();
        let mut novel = Vec::new();
        let mut all = Vec::new();
        for &a in &acts {
            let nxt = sim.transition(sim.state(), a);
            let k = StateKey::from_state(&nxt, false);
            stats.evaluations += 1;
            all.push((a, k.clone()));
            if !visited.contains(&k) {
                novel.push((a, k));
            }
        }
        let pool = if novel.is_empty() { &all } else { &novel };
        let (a, k) = pool.choose(rng).expect("pool").clone();
        visited.insert(k);
        let out = sim.step(a);
        stats.env_actions += 1;
        if note_outcome(&mut stats, &out) {
            break;
        }
    }
    stats
}

fn run_greedy_apparent(scenario: Scenario) -> AgentStats {
    let proxy = apparent_progress_proxy(&scenario);
    let mut sim = Simulator::new(scenario);
    let mut stats = AgentStats::default();
    loop {
        let (a, evals) = greedy_action(&sim, sim.state(), &proxy);
        stats.evaluations += evals;
        let out = sim.step(a);
        stats.env_actions += 1;
        if note_outcome(&mut stats, &out) {
            break;
        }
    }
    stats
}

fn run_candidate_discrimination(scenario: Scenario) -> AgentStats {
    let candidates = scenario.candidate_goals.clone();
    let mut alive: BTreeSet<usize> = (0..candidates.len()).collect();
    let mut sim = Simulator::new(scenario);
    let mut stats = AgentStats::default();
    let mut pending_plan: Vec<Action> = Vec::new();
    let mut testing: Option<usize> = None;

    loop {
        prune_impossible(&sim, &candidates, &mut alive);

        if stats.actions_to_unique_id.is_none() && alive.len() == 1 {
            stats.actions_to_unique_id = Some(stats.env_actions as u16);
            stats.identified_candidate = alive.iter().next().copied();
        }

        if pending_plan.is_empty() {
            if let Some(idx) = choose_test_target(&sim, &candidates, &alive) {
                testing = Some(idx);
                if let Some(res) = shortest_path(
                    &sim,
                    sim.state(),
                    &candidates[idx],
                    sim.scenario().action_budget,
                ) {
                    stats.expansions += res.expanded;
                    pending_plan = res.actions;
                } else {
                    // Unreachable under budget: drop hypothesis.
                    alive.remove(&idx);
                    testing = None;
                    continue;
                }
            } else {
                // No testable candidate: deterministic legal fallback.
                let acts = sim.legal_actions();
                pending_plan.push(acts[0]);
                testing = None;
            }
        }

        let a = if pending_plan.is_empty() {
            sim.legal_actions()[0]
        } else {
            pending_plan.remove(0)
        };
        let out = sim.step(a);
        stats.env_actions += 1;

        let falsified = update_belief_from_outcome(&sim, &candidates, &mut alive, &out);
        if testing.is_some_and(|idx| falsified.contains(&idx)) {
            stats.incorrect_commitments += 1;
            pending_plan.clear();
            testing = None;
        }
        prune_impossible(&sim, &candidates, &mut alive);
        note_unique_id(&mut stats, &alive);

        if note_outcome(&mut stats, &out) {
            break;
        }

        // If current test goal already satisfied mid-plan without success, clear plan.
        if let Some(idx) = testing {
            if !alive.contains(&idx)
                || goal_satisfied(sim.scenario(), sim.state(), &candidates[idx])
            {
                pending_plan.clear();
                testing = None;
            }
        }
    }
    stats
}

/// Active P1C probe: exact-plan remainder plus endpoint discrimination width.
#[derive(Clone, Debug)]
struct ProbeCommitment {
    index: usize,
    remaining: Vec<Action>,
    /// Exact-live goals that predict success at the committed plan endpoint.
    endpoint_tests: u64,
}

/// Private P1C routing over exact-live candidates (uniform posterior 1/n).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoutingStrategy {
    FalsifyOnly,
    ProgressOnly,
    BroadProgressNarrowFalsify,
    BroadFalsifyNarrowProgress,
    Alternating,
    CostAware,
    CappedBroadProgress,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParallelMethod {
    Progress,
    Falsify,
}

fn routing_strategy(name: AgentName) -> Option<RoutingStrategy> {
    match name {
        AgentName::SetAwareParallelPlanning => Some(RoutingStrategy::FalsifyOnly),
        AgentName::SharedProgressPlanning => Some(RoutingStrategy::ProgressOnly),
        AgentName::BroadProgressNarrowFalsify => Some(RoutingStrategy::BroadProgressNarrowFalsify),
        AgentName::BroadFalsifyNarrowProgress => Some(RoutingStrategy::BroadFalsifyNarrowProgress),
        AgentName::AlternatingParallelPlanning => Some(RoutingStrategy::Alternating),
        AgentName::CostAwareParallelPlanning => Some(RoutingStrategy::CostAware),
        AgentName::CappedBroadProgressPlanning => Some(RoutingStrategy::CappedBroadProgress),
        _ => None,
    }
}

#[derive(Clone, Debug)]
struct RouterState {
    /// Next alternating preference starts with progress.
    alternate_prefer_progress: bool,
    /// Consecutive shared-progress actions for the capped router.
    progress_streak: u32,
    /// Last parallel primitive actually selected (sequential fallback excluded).
    last_parallel_method: Option<ParallelMethod>,
}

impl Default for RouterState {
    fn default() -> Self {
        Self {
            alternate_prefer_progress: true,
            progress_streak: 0,
            last_parallel_method: None,
        }
    }
}

/// Kind of action selected by the exact-live controller.
#[derive(Clone, Debug)]
enum SelectedAct {
    /// One jointly-safe majority first step (no sticky multi-step plan).
    Progress { action: Action },
    /// Sticky full exact probe (multi-goal when endpoint_tests ≥ 2).
    Probe {
        index: usize,
        actions: Vec<Action>,
        endpoint_tests: u64,
    },
    /// At most one jointly-safe sequential step.
    SingleStep { index: usize, action: Action },
}

/// Generic exact-live P1C controller. Never reads the hidden goal index.
fn run_exact_live_controller(scenario: Scenario, routing: RoutingStrategy) -> AgentStats {
    let candidates = scenario.candidate_goals.clone();
    let mut alive: BTreeSet<usize> = (0..candidates.len()).collect();
    let mut sim = Simulator::new(scenario);
    let mut stats = AgentStats::default();
    let mut commitment: Option<ProbeCommitment> = None;
    let mut router = RouterState::default();
    let mut visited = HashSet::new();
    visited.insert(StateKey::from_state(sim.state(), false));

    loop {
        prune_impossible(&sim, &candidates, &mut alive);
        let live_plans = exact_live_plans(&sim, &candidates, &mut alive, &mut stats);
        note_unique_id(&mut stats, &alive);

        if let Some(ref c) = commitment {
            let stale = !alive.contains(&c.index)
                || c.remaining.is_empty()
                || goal_satisfied(sim.scenario(), sim.state(), &candidates[c.index]);
            if stale {
                commitment = None;
            }
        }

        let acts = sim.legal_actions();
        if acts.is_empty() {
            break;
        }

        let selected = match commitment.as_mut() {
            Some(c) if !c.remaining.is_empty() => Some(c.remaining.remove(0)),
            _ => {
                commitment = None;
                match select_exact_live_action(
                    &sim,
                    &candidates,
                    &live_plans,
                    routing,
                    &mut router,
                    &visited,
                    &mut stats,
                ) {
                    Some(SelectedAct::Progress { action }) => {
                        stats.shared_progress_actions += 1;
                        Some(action)
                    }
                    Some(SelectedAct::Probe {
                        index,
                        mut actions,
                        endpoint_tests,
                    }) => {
                        if actions.is_empty() {
                            None
                        } else {
                            let a = actions.remove(0);
                            commitment = Some(ProbeCommitment {
                                index,
                                remaining: actions,
                                endpoint_tests,
                            });
                            Some(a)
                        }
                    }
                    Some(SelectedAct::SingleStep { index, action }) => {
                        commitment = Some(ProbeCommitment {
                            index,
                            remaining: Vec::new(),
                            endpoint_tests: 0,
                        });
                        Some(action)
                    }
                    None => None,
                }
            }
        };
        let Some(a) = selected else {
            // Every candidate is already satisfied or exact-unreachable. Do
            // not take an arbitrary action that could trigger a hidden failure.
            stats.exhausted = true;
            break;
        };

        let endpoint_tests = commitment.as_ref().map(|c| c.endpoint_tests).unwrap_or(0);
        if endpoint_tests >= 2 {
            stats.multi_goal_probe_actions += 1;
        }

        let committed_idx = commitment.as_ref().map(|c| c.index);
        let out = sim.step(a);
        stats.env_actions += 1;
        visited.insert(StateKey::from_state(sim.state(), false));

        let falsified = update_belief_from_outcome(&sim, &candidates, &mut alive, &out);
        stats.goals_falsified += falsified.len() as u64;
        // Deliberate falsification evidence is not an incorrect commitment.
        if committed_idx.is_some_and(|idx| falsified.contains(&idx)) {
            commitment = None;
        }
        if let Some(ref c) = commitment {
            if !alive.contains(&c.index)
                || goal_satisfied(sim.scenario(), sim.state(), &candidates[c.index])
            {
                commitment = None;
            }
        }
        prune_impossible(&sim, &candidates, &mut alive);
        note_unique_id(&mut stats, &alive);

        if note_outcome(&mut stats, &out) {
            break;
        }
    }
    stats
}

fn select_exact_live_action(
    sim: &Simulator,
    candidates: &[Goal],
    live_plans: &[LiveGoalPlan],
    routing: RoutingStrategy,
    router: &mut RouterState,
    visited: &HashSet<StateKey>,
    stats: &mut AgentStats,
) -> Option<SelectedAct> {
    if live_plans.is_empty() {
        return None;
    }
    if live_plans.len() == 1 {
        let plan = &live_plans[0];
        if plan.actions.is_empty() {
            return None;
        }
        return Some(SelectedAct::Probe {
            index: plan.index,
            actions: plan.actions.clone(),
            endpoint_tests: 1,
        });
    }

    let n = live_plans.len();
    let progress = choose_shared_progress(sim, candidates, live_plans, visited, stats);
    let falsify = choose_falsification_probe(sim, candidates, live_plans, stats);

    let chosen = match routing {
        RoutingStrategy::CostAware => {
            choose_cost_aware(sim, candidates, live_plans, progress, falsify, stats)
        }
        _ => {
            let forced_cap =
                routing == RoutingStrategy::CappedBroadProgress && router.progress_streak >= 2;
            let preferred = preferred_parallel_method(routing, n, router, forced_cap);
            let allow_other = matches!(
                routing,
                RoutingStrategy::BroadProgressNarrowFalsify
                    | RoutingStrategy::BroadFalsifyNarrowProgress
                    | RoutingStrategy::Alternating
                    | RoutingStrategy::CappedBroadProgress
            );
            let mut act = None;
            if let Some(pref) = preferred {
                match take_parallel(pref, progress, &falsify) {
                    Some(a) => {
                        act = Some(a);
                    }
                    None if allow_other => {
                        let other = match pref {
                            ParallelMethod::Progress => ParallelMethod::Falsify,
                            ParallelMethod::Falsify => ParallelMethod::Progress,
                        };
                        if let Some(a) = take_parallel(other, progress, &falsify) {
                            act = Some(a);
                        }
                    }
                    None => {}
                }
            }
            act
        }
    };

    if let Some(act) = chosen {
        let method = match &act {
            SelectedAct::Progress { .. } => ParallelMethod::Progress,
            SelectedAct::Probe { endpoint_tests, .. } if *endpoint_tests >= 2 => {
                ParallelMethod::Falsify
            }
            _ => unreachable!("parallel selection returned a sequential action"),
        };
        if routing == RoutingStrategy::Alternating {
            router.alternate_prefer_progress = !router.alternate_prefer_progress;
        }
        note_parallel_method(router, stats, method);
        return Some(act);
    }
    let fallback = sequential_fallback(sim, candidates, live_plans, stats);
    if fallback.is_some() {
        router.progress_streak = 0;
    }
    fallback
}

fn note_parallel_method(router: &mut RouterState, stats: &mut AgentStats, method: ParallelMethod) {
    if router
        .last_parallel_method
        .is_some_and(|previous| previous != method)
    {
        stats.parallel_method_switches += 1;
    }
    router.last_parallel_method = Some(method);
    match method {
        ParallelMethod::Progress => {
            router.progress_streak = router.progress_streak.saturating_add(1);
        }
        ParallelMethod::Falsify => router.progress_streak = 0,
    }
}

fn preferred_parallel_method(
    routing: RoutingStrategy,
    n: usize,
    router: &RouterState,
    forced_cap: bool,
) -> Option<ParallelMethod> {
    if forced_cap {
        return Some(ParallelMethod::Falsify);
    }
    match routing {
        RoutingStrategy::FalsifyOnly => Some(ParallelMethod::Falsify),
        RoutingStrategy::ProgressOnly => Some(ParallelMethod::Progress),
        RoutingStrategy::BroadProgressNarrowFalsify | RoutingStrategy::CappedBroadProgress => {
            if n >= 4 {
                Some(ParallelMethod::Progress)
            } else {
                Some(ParallelMethod::Falsify)
            }
        }
        RoutingStrategy::BroadFalsifyNarrowProgress => {
            if n >= 4 {
                Some(ParallelMethod::Falsify)
            } else {
                Some(ParallelMethod::Progress)
            }
        }
        RoutingStrategy::Alternating => {
            if router.alternate_prefer_progress {
                Some(ParallelMethod::Progress)
            } else {
                Some(ParallelMethod::Falsify)
            }
        }
        RoutingStrategy::CostAware => None,
    }
}

fn take_parallel(
    method: ParallelMethod,
    progress: Option<Action>,
    falsify: &Option<(usize, u64, Vec<Action>)>,
) -> Option<SelectedAct> {
    match method {
        ParallelMethod::Progress => progress.map(|action| SelectedAct::Progress { action }),
        ParallelMethod::Falsify => {
            falsify
                .as_ref()
                .map(|(index, endpoint_tests, actions)| SelectedAct::Probe {
                    index: *index,
                    actions: actions.clone(),
                    endpoint_tests: *endpoint_tests,
                })
        }
    }
}

fn choose_cost_aware(
    sim: &Simulator,
    candidates: &[Goal],
    live_plans: &[LiveGoalPlan],
    progress: Option<Action>,
    falsify: Option<(usize, u64, Vec<Action>)>,
    stats: &mut AgentStats,
) -> Option<SelectedAct> {
    let falsify_u = falsify
        .as_ref()
        .map(|(_, predicted, actions)| (*predicted as f64) / (actions.len().max(1) as f64));
    let progress_u = progress.and_then(|action| {
        shared_progress_utility(sim, candidates, live_plans, action, stats).map(|u| (action, u))
    });

    match (falsify, falsify_u, progress_u) {
        (Some((index, endpoint_tests, actions)), Some(fu), Some((action, pu))) => {
            if fu >= pu {
                Some(SelectedAct::Probe {
                    index,
                    actions,
                    endpoint_tests,
                })
            } else {
                Some(SelectedAct::Progress { action })
            }
        }
        (Some((index, endpoint_tests, actions)), Some(_), None) => Some(SelectedAct::Probe {
            index,
            actions,
            endpoint_tests,
        }),
        (None, _, Some((action, _))) => Some(SelectedAct::Progress { action }),
        _ => None,
    }
}

fn shared_progress_utility(
    sim: &Simulator,
    candidates: &[Goal],
    live_plans: &[LiveGoalPlan],
    action: Action,
    stats: &mut AgentStats,
) -> Option<f64> {
    if !first_action_jointly_safe(sim, candidates, live_plans, action) {
        return None;
    }
    let next = sim.transition(sim.state(), action);
    stats.evaluations += 1;
    let mut net = 0i64;
    for plan in live_plans {
        let before = plan.actions.len() as i64;
        let after = match shortest_path(
            sim,
            &next,
            &candidates[plan.index],
            sim.scenario().action_budget,
        ) {
            Some(res) => {
                stats.expansions += res.expanded;
                res.actions.len() as i64
            }
            None => return None,
        };
        // Majority steps reduce distance; minority regret subtracts when negative.
        net += before - after;
    }
    Some(net as f64)
}

fn sequential_fallback(
    sim: &Simulator,
    candidates: &[Goal],
    live_plans: &[LiveGoalPlan],
    stats: &mut AgentStats,
) -> Option<SelectedAct> {
    if let Some((index, actions)) =
        choose_safe_single_goal_probe(sim, candidates, live_plans, stats)
    {
        return Some(SelectedAct::Probe {
            index,
            actions,
            // Selected by a single-goal objective; incidental co-satisfaction
            // does not turn this into a falsification-method action.
            endpoint_tests: 0,
        });
    }
    let live_indices: BTreeSet<usize> = live_plans.iter().map(|plan| plan.index).collect();
    let idx = choose_commitment_target(sim, candidates, &live_indices, live_plans)?;
    let plan = live_plans.iter().find(|p| p.index == idx)?;
    let action = plan.first()?;
    Some(SelectedAct::SingleStep { index: idx, action })
}

/// Shared progress: strict-majority (≥2) safe canonical first step.
fn choose_shared_progress(
    sim: &Simulator,
    candidates: &[Goal],
    live_plans: &[LiveGoalPlan],
    visited: &HashSet<StateKey>,
    stats: &mut AgentStats,
) -> Option<Action> {
    let n = live_plans.len();
    if n < 2 {
        return None;
    }
    let mut best: Option<(Action, u64, u64, u8, u8)> = None;
    for action in sim.legal_actions() {
        let support = live_plans
            .iter()
            .filter(|p| p.first() == Some(action))
            .count() as u64;
        if support < 2 || support as usize * 2 <= n {
            continue;
        }
        if !first_action_jointly_safe(sim, candidates, live_plans, action) {
            continue;
        }
        let next = sim.transition(sim.state(), action);
        stats.evaluations += 1;
        let endpoint_tests = live_plans
            .iter()
            .filter(|live| goal_satisfied(sim.scenario(), &next, &candidates[live.index]))
            .count() as u64;
        let novelty = if visited.contains(&StateKey::from_state(&next, false)) {
            0u8
        } else {
            1u8
        };
        let rank = action_rank(action);
        let score = (action, support, endpoint_tests, novelty, rank);
        best = Some(match best {
            None => score,
            Some(prev) if progress_better(&score, &prev) => score,
            Some(prev) => prev,
        });
    }
    best.map(|(action, _, _, _, _)| action)
}

fn progress_better(a: &(Action, u64, u64, u8, u8), b: &(Action, u64, u64, u8, u8)) -> bool {
    match a.1.cmp(&b.1) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => match a.2.cmp(&b.2) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => match a.3.cmp(&b.3) {
                std::cmp::Ordering::Greater => true,
                std::cmp::Ordering::Less => false,
                std::cmp::Ordering::Equal => a.4 < b.4,
            },
        },
    }
}

fn note_unique_id(stats: &mut AgentStats, alive: &BTreeSet<usize>) {
    if stats.actions_to_unique_id.is_none() && alive.len() == 1 {
        stats.actions_to_unique_id = Some(stats.env_actions as u16);
        stats.identified_candidate = alive.iter().next().copied();
    }
}

/// Retain exactly the candidates whose public predicates predict the observed
/// terminal channel. Positive success/failure evidence narrows the set too.
fn update_belief_from_outcome(
    sim: &Simulator,
    candidates: &[Goal],
    alive: &mut BTreeSet<usize>,
    out: &StepOutcome,
) -> Vec<usize> {
    let mut falsified = Vec::new();
    for &i in alive.iter() {
        let expected_success = goal_satisfied(sim.scenario(), sim.state(), &candidates[i]);
        let expected_failure = goal_terminal_failure(sim.scenario(), sim.state(), &candidates[i]);
        let matches_observation = if out.success {
            expected_success
        } else if out.failed {
            expected_failure
        } else {
            !expected_success && !expected_failure
        };
        if !matches_observation {
            falsified.push(i);
        }
    }
    for &i in &falsified {
        alive.remove(&i);
    }
    falsified
}

fn exact_live_plans(
    sim: &Simulator,
    candidates: &[Goal],
    alive: &mut BTreeSet<usize>,
    stats: &mut AgentStats,
) -> Vec<LiveGoalPlan> {
    let mut plans = Vec::new();
    let mut unreachable = Vec::new();
    for &index in alive.iter() {
        match shortest_path(
            sim,
            sim.state(),
            &candidates[index],
            sim.scenario().action_budget,
        ) {
            Some(result) => {
                stats.expansions += result.expanded;
                plans.push(LiveGoalPlan {
                    index,
                    actions: result.actions,
                });
            }
            None => unreachable.push(index),
        }
    }
    for index in unreachable {
        alive.remove(&index);
    }
    plans
}

#[derive(Clone, Debug)]
struct LiveGoalPlan {
    index: usize,
    actions: Vec<Action>,
}

impl LiveGoalPlan {
    fn first(&self) -> Option<Action> {
        self.actions.first().copied()
    }
}

/// Safe multi-goal falsification probe: endpoint success for ≥2 live goals.
fn choose_falsification_probe(
    sim: &Simulator,
    candidates: &[Goal],
    live_plans: &[LiveGoalPlan],
    stats: &mut AgentStats,
) -> Option<(usize, u64, Vec<Action>)> {
    if live_plans.len() < 2 {
        return None;
    }
    let mut best: Option<ProbeScore> = None;
    for plan in live_plans {
        if plan.actions.is_empty() {
            continue;
        }
        let Some(predicted) =
            simulate_probe_endpoint_successes(sim, candidates, live_plans, plan, stats)
        else {
            continue;
        };
        if predicted < 2 {
            continue;
        }
        let score = ProbeScore {
            index: plan.index,
            predicted,
            len: plan.actions.len(),
        };
        best = Some(match best {
            None => score,
            Some(prev) if probe_better(&score, &prev) => score,
            Some(prev) => prev,
        });
    }
    best.map(|s| {
        let actions = live_plans
            .iter()
            .find(|p| p.index == s.index)
            .expect("probe plan")
            .actions
            .clone();
        (s.index, s.predicted, actions)
    })
}

/// Fully safe single-goal exact probe for sequential fallback.
fn choose_safe_single_goal_probe(
    sim: &Simulator,
    candidates: &[Goal],
    live_plans: &[LiveGoalPlan],
    stats: &mut AgentStats,
) -> Option<(usize, Vec<Action>)> {
    let mut safe = BTreeSet::new();
    for plan in live_plans {
        if plan.actions.is_empty() {
            continue;
        }
        let Some(_) = simulate_probe_endpoint_successes(sim, candidates, live_plans, plan, stats)
        else {
            continue;
        };
        safe.insert(plan.index);
    }
    let index = choose_test_target(sim, candidates, &safe)?;
    let actions = live_plans
        .iter()
        .find(|plan| plan.index == index)?
        .actions
        .clone();
    Some((index, actions))
}

fn probe_better(a: &ProbeScore, b: &ProbeScore) -> bool {
    match a.predicted.cmp(&b.predicted) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => match a.len.cmp(&b.len) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Greater => false,
            std::cmp::Ordering::Equal => a.index < b.index,
        },
    }
}

#[derive(Clone, Copy, Debug)]
struct ProbeScore {
    index: usize,
    predicted: u64,
    len: usize,
}

/// Simulate an exact candidate plan publicly. Reject unsafe prefixes; else count
/// exact-live goals that predict success at the endpoint.
fn simulate_probe_endpoint_successes(
    sim: &Simulator,
    candidates: &[Goal],
    live_plans: &[LiveGoalPlan],
    plan: &LiveGoalPlan,
    stats: &mut AgentStats,
) -> Option<u64> {
    let mut state = sim.state().clone();
    for &a in &plan.actions {
        let next = sim.transition(&state, a);
        stats.evaluations += 1;
        for live in live_plans {
            let g = &candidates[live.index];
            if goal_terminal_failure(sim.scenario(), &next, g) {
                return None;
            }
            if !goal_viable_at(sim, &next, g) {
                return None;
            }
        }
        state = next;
    }
    let predicted = live_plans
        .iter()
        .filter(|live| goal_satisfied(sim.scenario(), &state, &candidates[live.index]))
        .count() as u64;
    Some(predicted)
}

/// Deterministic commitment target; prefer jointly-safe first steps when any exist.
fn choose_commitment_target(
    sim: &Simulator,
    candidates: &[Goal],
    alive: &BTreeSet<usize>,
    live_plans: &[LiveGoalPlan],
) -> Option<usize> {
    let mut safe: BTreeSet<usize> = BTreeSet::new();
    for plan in live_plans {
        if !alive.contains(&plan.index) {
            continue;
        }
        match plan.first() {
            None => {
                safe.insert(plan.index);
            }
            Some(a) if first_action_jointly_safe(sim, candidates, live_plans, a) => {
                safe.insert(plan.index);
            }
            Some(_) => {}
        }
    }
    if safe.is_empty() {
        None
    } else {
        choose_test_target(sim, candidates, &safe)
    }
}

fn first_action_jointly_safe(
    sim: &Simulator,
    candidates: &[Goal],
    live_plans: &[LiveGoalPlan],
    a: Action,
) -> bool {
    let next = sim.transition(sim.state(), a);
    live_plans.iter().all(|plan| {
        let g = &candidates[plan.index];
        !goal_terminal_failure(sim.scenario(), &next, g) && goal_viable_at(sim, &next, g)
    })
}

/// Public-state viability for candidate discrimination (no hidden-goal access).
fn candidate_viable(sim: &Simulator, goal: &Goal) -> bool {
    goal_viable_at(sim, sim.state(), goal)
}

fn prune_impossible(sim: &Simulator, candidates: &[Goal], alive: &mut BTreeSet<usize>) {
    let drop: Vec<usize> = alive
        .iter()
        .copied()
        .filter(|&i| !candidate_viable(sim, &candidates[i]))
        .collect();
    for i in drop {
        alive.remove(&i);
    }
}

fn choose_test_target(
    sim: &Simulator,
    candidates: &[Goal],
    alive: &BTreeSet<usize>,
) -> Option<usize> {
    // Prefer not-yet-satisfied, possible candidates; deterministic lowest index.
    let mut best: Option<(u32, usize)> = None;
    for &i in alive {
        let g = &candidates[i];
        if goal_satisfied(sim.scenario(), sim.state(), g) {
            continue;
        }
        if !candidate_viable(sim, g) {
            continue;
        }
        let h = heuristic(sim.scenario(), sim.state(), g);
        match best {
            Some((bh, _)) if bh <= h => {}
            _ => best = Some((h, i)),
        }
    }
    best.map(|(_, i)| i)
}

fn run_oracle_plan(scenario: Scenario, true_goal: &Goal) -> AgentStats {
    let mut sim = Simulator::new(scenario);
    let mut stats = AgentStats::default();
    match shortest_path(&sim, sim.state(), true_goal, sim.scenario().action_budget) {
        Some(res) => {
            stats.expansions += res.expanded;
            if res.actions.is_empty() {
                // Empty path ⇒ start already satisfies the goal.
                stats.success = true;
                return stats;
            }
            for a in res.actions {
                let out = sim.step(a);
                stats.env_actions += 1;
                if note_outcome(&mut stats, &out) {
                    return stats;
                }
            }
            // Plan exhausted without terminal success/failure: treat as budget fail.
            stats.exhausted = true;
            stats
        }
        None => {
            // Unsolvable under budget: flag instead of substituting a heuristic walk.
            stats.unsolvable = true;
            stats
        }
    }
}

fn run_oracle_objective(scenario: Scenario, true_goal: &Goal) -> AgentStats {
    run_oracle_plan(scenario, true_goal)
}

fn planning_goal(config: &AgentConfig, scenario: &Scenario) -> Goal {
    config
        .planning_goal
        .clone()
        .unwrap_or_else(|| apparent_progress_proxy(scenario))
}

fn run_reactive(scenario: Scenario, config: &AgentConfig) -> AgentStats {
    let goal = planning_goal(config, &scenario);
    let mut sim = Simulator::new(scenario);
    let mut stats = AgentStats::default();
    loop {
        let (a, evals) = greedy_action(&sim, sim.state(), &goal);
        stats.evaluations += evals;
        let out = sim.step(a);
        stats.env_actions += 1;
        if note_outcome(&mut stats, &out) {
            break;
        }
    }
    stats
}

fn run_pause_compute(scenario: Scenario, config: &AgentConfig) -> AgentStats {
    let goal = planning_goal(config, &scenario);
    let mut sim = Simulator::new(scenario);
    let mut stats = AgentStats::default();
    loop {
        let (a, evals) = greedy_action(&sim, sim.state(), &goal);
        stats.evaluations += evals;
        // Extra non-rollout compute: re-score legal successors without committing.
        let acts = legal_actions(sim.scenario());
        let extra = config.pause_extra_evals;
        for _ in 0..extra {
            for &x in &acts {
                let nxt = sim.transition(sim.state(), x);
                let _ = heuristic(sim.scenario(), &nxt, &goal);
                stats.evaluations += 1;
            }
        }
        let out = sim.step(a);
        stats.env_actions += 1;
        if note_outcome(&mut stats, &out) {
            break;
        }
    }
    stats
}

fn run_best_of_k(scenario: Scenario, config: &AgentConfig, rng: &mut ChaCha8Rng) -> AgentStats {
    let goal = planning_goal(config, &scenario);
    let k = config.best_of_k.max(1);
    let mut sim = Simulator::new(scenario);
    let mut stats = AgentStats::default();
    loop {
        let acts = sim.legal_actions();
        let mut best_a = acts[0];
        let mut best_h = u32::MAX;
        for _ in 0..k {
            let a = *acts.choose(rng).expect("acts");
            let nxt = sim.transition(sim.state(), a);
            let h = heuristic(sim.scenario(), &nxt, &goal);
            stats.evaluations += 1;
            if h < best_h || (h == best_h && action_rank(a) < action_rank(best_a)) {
                best_h = h;
                best_a = a;
            }
        }
        let out = sim.step(best_a);
        stats.env_actions += 1;
        if note_outcome(&mut stats, &out) {
            break;
        }
    }
    stats
}

fn run_beam_agent(scenario: Scenario, config: &AgentConfig) -> AgentStats {
    let goal = planning_goal(config, &scenario);
    let mut sim = Simulator::new(scenario);
    let mut stats = AgentStats::default();
    if let Some(res) = beam_search(
        &sim,
        sim.state(),
        &goal,
        config.beam_horizon,
        config.beam_width,
    ) {
        stats.expansions += res.expanded;
        if res.actions.is_empty() {
            stats.success = true;
            return stats;
        }
        for a in res.actions {
            let out = sim.step(a);
            stats.env_actions += 1;
            if note_outcome(&mut stats, &out) {
                return stats;
            }
        }
    }
    // Residual greedy if beam did not finish.
    loop {
        let (a, evals) = greedy_action(&sim, sim.state(), &goal);
        stats.evaluations += evals;
        let out = sim.step(a);
        stats.env_actions += 1;
        if note_outcome(&mut stats, &out) {
            break;
        }
    }
    stats
}

fn run_oracle_optimal(scenario: Scenario, true_goal: &Goal) -> AgentStats {
    run_oracle_plan(scenario, true_goal)
}

fn action_rank(a: Action) -> u8 {
    match a {
        Action::Move(crate::domain::Dir::North) => 0,
        Action::Move(crate::domain::Dir::South) => 1,
        Action::Move(crate::domain::Dir::East) => 2,
        Action::Move(crate::domain::Dir::West) => 3,
        Action::Undo => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Action, Dir, Pos, Split};
    use crate::search::goal_possible;
    use std::collections::BTreeSet;

    fn tiny() -> Scenario {
        Scenario {
            width: 5,
            height: 3,
            walls: BTreeSet::from([Pos::new(2, 1)]),
            markers: vec![Pos::new(4, 1), Pos::new(0, 2)],
            collectibles: vec![Pos::new(1, 0)],
            switches: vec![Pos::new(0, 0), Pos::new(1, 2), Pos::new(3, 2)],
            hazards: vec![],
            resource_pickups: vec![],
            terminal_triggers: vec![],
            start: Pos::new(0, 1),
            initial_resource: 1,
            action_budget: 40,
            undo_enabled: false,
            candidate_goals: vec![
                Goal::ReachMarker { marker: 0 },
                Goal::ReachMarker { marker: 1 },
                Goal::CollectAll,
                Goal::ActivateSwitchesInOrder {
                    order: vec![0, 1, 2],
                },
                Goal::ActivateSwitchesInOrder {
                    order: vec![1, 0, 2],
                },
            ],
            hidden_goal_index: 0,
            split: Split::Train,
            seed: 7,
            episode_id: 0,
        }
    }

    #[test]
    fn agent_names_roundtrip_snake_case() {
        let names = [
            AgentName::Random,
            AgentName::NovelState,
            AgentName::GreedyApparentProgress,
            AgentName::CandidateGoalDiscrimination,
            AgentName::OracleObjective,
            AgentName::Reactive,
            AgentName::PauseCompute,
            AgentName::BestOfK,
            AgentName::BeamSearch,
            AgentName::OracleOptimal,
            AgentName::SetAwareParallelPlanning,
            AgentName::SharedProgressPlanning,
            AgentName::BroadProgressNarrowFalsify,
            AgentName::BroadFalsifyNarrowProgress,
            AgentName::AlternatingParallelPlanning,
            AgentName::CostAwareParallelPlanning,
            AgentName::CappedBroadProgressPlanning,
        ];
        let expected = [
            "random",
            "novel_state",
            "greedy_apparent_progress",
            "candidate_goal_discrimination",
            "oracle_objective",
            "reactive",
            "pause_compute",
            "best_of_k",
            "beam_search",
            "oracle_optimal",
            "set_aware_parallel_planning",
            "shared_progress_planning",
            "broad_progress_narrow_falsify",
            "broad_falsify_narrow_progress",
            "alternating_parallel_planning",
            "cost_aware_parallel_planning",
            "capped_broad_progress_planning",
        ];
        for (n, e) in names.iter().zip(expected) {
            assert_eq!(serde_json::to_string(n).unwrap(), format!("\"{e}\""));
            assert_eq!(n.as_str(), e);
        }
        for n in names {
            if matches!(
                n,
                AgentName::SetAwareParallelPlanning
                    | AgentName::SharedProgressPlanning
                    | AgentName::BroadProgressNarrowFalsify
                    | AgentName::BroadFalsifyNarrowProgress
                    | AgentName::AlternatingParallelPlanning
                    | AgentName::CostAwareParallelPlanning
                    | AgentName::CappedBroadProgressPlanning
            ) {
                assert!(n.is_p1c_strategy());
                assert!(n.is_p1a());
            }
        }
    }

    #[test]
    fn determinism_under_chacha() {
        let sc = tiny();
        let mut cfg = AgentConfig::for_name(AgentName::Random);
        let mut a = ChaCha8Rng::seed_from_u64(42);
        let mut b = ChaCha8Rng::seed_from_u64(42);
        let sa = run_episode(sc.clone(), &cfg, &mut a, None);
        let sb = run_episode(sc.clone(), &cfg, &mut b, None);
        assert_eq!(sa, sb);

        cfg.name = AgentName::NovelState;
        let mut a = ChaCha8Rng::seed_from_u64(9);
        let mut b = ChaCha8Rng::seed_from_u64(9);
        assert_eq!(
            run_episode(sc.clone(), &cfg, &mut a, None),
            run_episode(sc, &cfg, &mut b, None)
        );
    }

    #[test]
    fn pause_matches_reactive_action_path() {
        let sc = tiny();
        let goal = Goal::ReachMarker { marker: 0 };
        let mut cfg_r = AgentConfig::for_name(AgentName::Reactive);
        cfg_r.planning_goal = Some(goal.clone());
        let mut cfg_p = AgentConfig::for_name(AgentName::PauseCompute);
        cfg_p.planning_goal = Some(goal);
        cfg_p.pause_extra_evals = 8;
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let sim = Simulator::new(sc.clone());
        let (ar, _) = greedy_action(&sim, sim.state(), cfg_r.planning_goal.as_ref().unwrap());
        let (ap, _) = greedy_action(&sim, sim.state(), cfg_p.planning_goal.as_ref().unwrap());
        assert_eq!(ar, ap);
        let sr = run_episode(sc.clone(), &cfg_r, &mut rng, None);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let sp = run_episode(sc, &cfg_p, &mut rng, None);
        assert_eq!(sr.env_actions, sp.env_actions);
        assert_eq!(sr.success, sp.success);
        assert!(sp.evaluations > sr.evaluations);
    }

    #[test]
    fn oracle_optimal_solves_reach_marker() {
        let sc = tiny();
        let goal = sc.hidden_goal().clone();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cfg = AgentConfig::for_name(AgentName::OracleOptimal);
        let stats = run_episode(sc, &cfg, &mut rng, Some(&goal));
        assert!(stats.success);
        assert!(stats.expansions > 0);
        assert_eq!(stats.env_actions, 6);
        assert!(!stats.failed);
        assert!(!stats.unsolvable);
    }

    #[test]
    fn oracle_empty_path_is_immediate_success() {
        let mut sc = tiny();
        sc.start = Pos::new(4, 1); // already on marker 0
        sc.hidden_goal_index = 0;
        let goal = sc.hidden_goal().clone();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cfg = AgentConfig::for_name(AgentName::OracleObjective);
        let stats = run_episode(sc, &cfg, &mut rng, Some(&goal));
        assert!(stats.success);
        assert_eq!(stats.env_actions, 0);
        assert!(!stats.exhausted);
        assert!(!stats.unsolvable);
        assert!(!stats.failed);
    }

    #[test]
    fn non_oracle_observes_initial_terminal_success_without_acting() {
        let mut sc = tiny();
        sc.start = Pos::new(4, 1);
        sc.hidden_goal_index = 0;
        let mut cfg = AgentConfig::for_name(AgentName::Reactive);
        cfg.planning_goal = Some(Goal::ReachMarker { marker: 0 });
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let stats = run_episode(sc, &cfg, &mut rng, None);
        assert!(stats.success);
        assert_eq!(stats.env_actions, 0);
    }

    #[test]
    fn oracle_unsolvable_is_flagged_not_budget_substituted() {
        let mut sc = tiny();
        sc.candidate_goals = vec![Goal::ReachMarker { marker: 99 }];
        sc.hidden_goal_index = 0;
        sc.action_budget = 10;
        let goal = sc.hidden_goal().clone();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cfg = AgentConfig::for_name(AgentName::OracleOptimal);
        let stats = run_episode(sc, &cfg, &mut rng, Some(&goal));
        assert!(stats.unsolvable);
        assert!(!stats.success);
        assert_eq!(stats.env_actions, 0);
        assert!(!stats.exhausted);
    }

    #[test]
    fn candidate_discrimination_tracks_id_and_prefix() {
        let mut sc = tiny();
        sc.hidden_goal_index = 3;
        sc.action_budget = 60;
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let cfg = AgentConfig::for_name(AgentName::CandidateGoalDiscrimination);
        let stats = run_episode(sc, &cfg, &mut rng, None);
        let _ = stats.actions_to_unique_id;
        let _ = stats.incorrect_commitments;
        assert!(stats.env_actions > 0);
    }

    #[test]
    fn candidate_discrimination_reasons_about_new_goals_without_hidden() {
        let mut sc = tiny();
        sc.hazards = vec![Pos::new(1, 1)];
        sc.terminal_triggers = vec![Pos::new(3, 1)];
        sc.candidate_goals = vec![
            Goal::ReachMarker { marker: 0 },
            Goal::AvoidHazardReachMarker {
                hazard: 0,
                marker: 0,
            },
            Goal::TriggerTerminal { trigger: 0 },
        ];
        sc.hidden_goal_index = 0;
        sc.action_budget = 50;
        let mut rng = ChaCha8Rng::seed_from_u64(4);
        let cfg = AgentConfig::for_name(AgentName::CandidateGoalDiscrimination);
        // Must not receive true_goal.
        let stats = run_episode(sc, &cfg, &mut rng, None);
        assert!(stats.env_actions > 0);
        assert!(!stats.unsolvable);
    }

    #[test]
    fn candidate_viability_respects_undo_recovery() {
        let mut sc = tiny();
        sc.undo_enabled = true;
        let goal = Goal::ActivateSwitchesInOrder {
            order: vec![1, 0, 2],
        };
        let mut sim = Simulator::new(sc.clone());
        sim.step(Action::Move(Dir::North)); // activates switch 0: wrong prefix
        assert!(!goal_possible(sim.scenario(), sim.state(), &goal));
        assert!(
            candidate_viable(&sim, &goal),
            "Undo can restore the pre-activation state"
        );

        sc.undo_enabled = false;
        let mut no_undo = Simulator::new(sc);
        no_undo.step(Action::Move(Dir::North));
        assert!(!candidate_viable(&no_undo, &goal));
    }

    #[test]
    fn terminal_failure_stops_all_policies() {
        let mut sc = tiny();
        sc.hazards = vec![Pos::new(1, 1)];
        sc.start = Pos::new(0, 1);
        sc.candidate_goals = vec![
            Goal::ReachMarker { marker: 0 },
            Goal::AvoidHazardReachMarker {
                hazard: 0,
                marker: 0,
            },
        ];
        sc.hidden_goal_index = 1;
        sc.action_budget = 20;
        sc.terminal_triggers = vec![];

        // Walk east into hazard 0; domain must mark failed for avoid-goal.
        let mut sim = Simulator::new(sc.clone());
        let out = sim.step(Action::Move(Dir::East));
        assert!(
            out.failed,
            "domain must report terminal failure on hazard touch"
        );
        assert!(!out.success);

        // Public proxy is ReachMarker → greedy walks east into the hazard and must stop.
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let cfg = AgentConfig::for_name(AgentName::GreedyApparentProgress);
        let greedy = run_episode(sc.clone(), &cfg, &mut rng, None);
        assert!(greedy.failed);
        assert!(!greedy.success);
        assert_eq!(greedy.env_actions, 1);

        let avoid = Goal::AvoidHazardReachMarker {
            hazard: 0,
            marker: 0,
        };
        for name in [
            AgentName::Random,
            AgentName::NovelState,
            AgentName::CandidateGoalDiscrimination,
            AgentName::SetAwareParallelPlanning,
            AgentName::SharedProgressPlanning,
            AgentName::BroadProgressNarrowFalsify,
            AgentName::BroadFalsifyNarrowProgress,
            AgentName::AlternatingParallelPlanning,
            AgentName::CostAwareParallelPlanning,
            AgentName::CappedBroadProgressPlanning,
            AgentName::Reactive,
            AgentName::PauseCompute,
            AgentName::BestOfK,
            AgentName::BeamSearch,
        ] {
            let mut cfg = AgentConfig::for_name(name);
            if !name.is_p1a() {
                cfg.planning_goal = Some(avoid.clone());
            }
            let stats = run_episode(sc.clone(), &cfg, &mut rng, None);
            assert!(
                stats.success || stats.exhausted || stats.failed,
                "{name:?} did not terminate"
            );
        }

        let cfg = AgentConfig::for_name(AgentName::OracleOptimal);
        let stats = run_episode(sc, &cfg, &mut rng, Some(&avoid));
        assert!(stats.success || stats.unsolvable || stats.failed || stats.exhausted);
    }

    #[test]
    fn non_oracle_never_needs_true_goal() {
        let sc = tiny();
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        for name in [
            AgentName::Random,
            AgentName::NovelState,
            AgentName::GreedyApparentProgress,
            AgentName::CandidateGoalDiscrimination,
            AgentName::SetAwareParallelPlanning,
            AgentName::SharedProgressPlanning,
            AgentName::BroadProgressNarrowFalsify,
            AgentName::BroadFalsifyNarrowProgress,
            AgentName::AlternatingParallelPlanning,
            AgentName::CostAwareParallelPlanning,
            AgentName::CappedBroadProgressPlanning,
            AgentName::Reactive,
            AgentName::PauseCompute,
            AgentName::BestOfK,
            AgentName::BeamSearch,
        ] {
            let mut cfg = AgentConfig::for_name(name);
            cfg.planning_goal = Some(Goal::ReachMarker { marker: 0 });
            let _ = run_episode(sc.clone(), &cfg, &mut rng, None);
        }
    }

    #[test]
    fn novel_state_visits_keys() {
        let sc = tiny();
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let cfg = AgentConfig::for_name(AgentName::NovelState);
        let stats = run_episode(sc, &cfg, &mut rng, None);
        assert!(stats.evaluations >= stats.env_actions);
    }

    #[test]
    fn greedy_uses_public_proxy_not_hidden() {
        let mut sc = tiny();
        sc.hidden_goal_index = 3; // switches
        let proxy = apparent_progress_proxy(&sc);
        assert!(matches!(proxy, Goal::ReachMarker { .. }));
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let cfg = AgentConfig::for_name(AgentName::GreedyApparentProgress);
        let _ = run_episode(sc, &cfg, &mut rng, None);
    }

    fn shared_prefix_scenario() -> Scenario {
        // Junction at (1,1): East from start is the shared first step toward both markers.
        Scenario {
            width: 3,
            height: 3,
            walls: BTreeSet::from([Pos::new(0, 0), Pos::new(0, 2)]),
            markers: vec![Pos::new(2, 0), Pos::new(2, 2)],
            collectibles: vec![],
            switches: vec![],
            hazards: vec![],
            resource_pickups: vec![],
            terminal_triggers: vec![],
            start: Pos::new(0, 1),
            initial_resource: 1,
            action_budget: 20,
            undo_enabled: false,
            candidate_goals: vec![
                Goal::ReachMarker { marker: 0 },
                Goal::ReachMarker { marker: 1 },
            ],
            hidden_goal_index: 0,
            split: Split::Train,
            seed: 11,
            episode_id: 0,
        }
    }

    fn sequential_only_fork_scenario() -> Scenario {
        Scenario {
            width: 1,
            height: 3,
            walls: BTreeSet::new(),
            markers: vec![Pos::new(0, 0), Pos::new(0, 2)],
            collectibles: vec![],
            switches: vec![],
            hazards: vec![],
            resource_pickups: vec![],
            terminal_triggers: vec![],
            start: Pos::new(0, 1),
            initial_resource: 1,
            action_budget: 10,
            undo_enabled: false,
            candidate_goals: vec![
                Goal::ReachMarker { marker: 0 },
                Goal::ReachMarker { marker: 1 },
            ],
            hidden_goal_index: 0,
            split: Split::Train,
            seed: 23,
            episode_id: 0,
        }
    }

    fn avoid_fork_scenario() -> Scenario {
        // Direct East hits hazard; North→East→East→South reaches marker safely.
        Scenario {
            width: 3,
            height: 2,
            walls: BTreeSet::new(),
            markers: vec![Pos::new(2, 1)],
            collectibles: vec![],
            switches: vec![],
            hazards: vec![Pos::new(1, 1)],
            resource_pickups: vec![],
            terminal_triggers: vec![],
            start: Pos::new(0, 1),
            initial_resource: 1,
            action_budget: 20,
            undo_enabled: false,
            candidate_goals: vec![
                Goal::ReachMarker { marker: 0 },
                Goal::AvoidHazardReachMarker {
                    hazard: 0,
                    marker: 0,
                },
            ],
            hidden_goal_index: 1, // true avoid: stepping East would be catastrophic
            split: Split::Train,
            seed: 13,
            episode_id: 0,
        }
    }

    /// One-step marker/resource endpoint tests ≥2; three East plans form an old majority.
    fn marker_resource_probe_scenario() -> Scenario {
        Scenario {
            width: 5,
            height: 3,
            // Block North so distant markers have a unique East shortest prefix.
            walls: BTreeSet::from([Pos::new(1, 0)]),
            markers: vec![
                Pos::new(1, 2),
                Pos::new(3, 0),
                Pos::new(4, 1),
                Pos::new(3, 1),
            ],
            collectibles: vec![],
            switches: vec![],
            hazards: vec![],
            resource_pickups: vec![],
            terminal_triggers: vec![],
            start: Pos::new(1, 1),
            initial_resource: 1,
            action_budget: 30,
            undo_enabled: false,
            candidate_goals: vec![
                Goal::ReachMarker { marker: 0 },
                Goal::PreserveResourceReachMarker {
                    marker: 0,
                    min_resource: 1,
                },
                Goal::ReachMarker { marker: 1 },
                Goal::ReachMarker { marker: 2 },
                Goal::ReachMarker { marker: 3 },
            ],
            hidden_goal_index: 0,
            split: Split::Train,
            seed: 23,
            episode_id: 0,
        }
    }

    #[test]
    fn set_aware_never_receives_hidden_goal() {
        let sc = shared_prefix_scenario();
        let cfg = AgentConfig::for_name(AgentName::SetAwareParallelPlanning);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        // Non-oracle dispatch asserts true_goal is None.
        let stats = run_episode(sc, &cfg, &mut rng, None);
        assert!(stats.env_actions > 0);
        assert!(!stats.unsolvable);
    }

    #[test]
    fn set_aware_is_deterministic() {
        let sc = marker_resource_probe_scenario();
        let cfg = AgentConfig::for_name(AgentName::SetAwareParallelPlanning);
        let mut a = ChaCha8Rng::seed_from_u64(42);
        let mut b = ChaCha8Rng::seed_from_u64(99);
        let sa = run_episode(sc.clone(), &cfg, &mut a, None);
        let sb = run_episode(sc, &cfg, &mut b, None);
        assert_eq!(sa, sb);
    }

    #[test]
    fn set_aware_one_step_marker_resource_probe_beats_shared_east() {
        let sc = marker_resource_probe_scenario();
        let sim = Simulator::new(sc.clone());
        let candidates = sc.candidate_goals.clone();
        let mut alive: BTreeSet<usize> = (0..candidates.len()).collect();
        let mut stats = AgentStats::default();
        let plans = exact_live_plans(&sim, &candidates, &mut alive, &mut stats);
        assert_eq!(plans.len(), 5);
        // Old strict-majority shared progress would prefer East (3/5).
        let east_first = plans
            .iter()
            .filter(|p| p.first() == Some(Action::Move(Dir::East)))
            .count();
        let plan_starts: Vec<_> = plans
            .iter()
            .map(|p| (p.index, p.first(), p.actions.len()))
            .collect();
        assert!(
            east_first >= 3,
            "expected ≥3 East-first plans, got {east_first} from {plan_starts:?}"
        );
        let (idx, predicted, actions) =
            choose_falsification_probe(&sim, &candidates, &plans, &mut stats)
                .expect("safe marker/resource probe");
        assert_eq!(idx, 0);
        assert_eq!(predicted, 2);
        assert_eq!(actions, vec![Action::Move(Dir::South)]);

        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cfg = AgentConfig::for_name(AgentName::SetAwareParallelPlanning);
        let episode = run_episode(sc, &cfg, &mut rng, None);
        assert!(episode.success);
        assert_eq!(episode.env_actions, 1);
        assert_eq!(episode.multi_goal_probe_actions, 1);
        assert_eq!(episode.incorrect_commitments, 0);
    }

    #[test]
    fn set_aware_nonterminal_evidence_eliminates_endpoint_predictors() {
        let mut sc = marker_resource_probe_scenario();
        sc.hidden_goal_index = 2; // distant East marker; South probe is nonterminal
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cfg = AgentConfig::for_name(AgentName::SetAwareParallelPlanning);
        let episode = run_episode(sc, &cfg, &mut rng, None);
        assert!(
            episode.goals_falsified >= 2,
            "nonterminal South should falsify both marker/resource predictors: {episode:?}"
        );
        assert!(
            episode.multi_goal_probe_actions >= 1,
            "South probe endpoint tests ≥2: {episode:?}"
        );
        assert_eq!(episode.incorrect_commitments, 0);
        assert!(episode.success || episode.exhausted);
        assert!(!episode.failed);
    }

    #[test]
    fn set_aware_probe_commitment_persists_for_multiple_actions() {
        let sc = shared_prefix_scenario();
        let sim = Simulator::new(sc.clone());
        let candidates = sc.candidate_goals.clone();
        let mut alive: BTreeSet<usize> = (0..candidates.len()).collect();
        let mut stats = AgentStats::default();
        let plans = exact_live_plans(&sim, &candidates, &mut alive, &mut stats);
        assert!(
            choose_falsification_probe(&sim, &candidates, &plans, &mut stats).is_none(),
            "shared fork has no width≥2 endpoint probe"
        );
        let (idx, actions) = choose_safe_single_goal_probe(&sim, &candidates, &plans, &mut stats)
            .expect("safe single-goal sequential probe");
        let plan_len = actions.len();
        assert!(plan_len >= 2, "need a multi-step commitment");
        assert_eq!(
            plans.iter().find(|p| p.index == idx).unwrap().actions,
            actions
        );

        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cfg = AgentConfig::for_name(AgentName::SetAwareParallelPlanning);
        let episode = run_episode(sc, &cfg, &mut rng, None);
        assert!(episode.success);
        assert_eq!(
            episode.env_actions, plan_len as u64,
            "must execute the full committed probe without mid-plan replanning"
        );
        assert_eq!(episode.multi_goal_probe_actions, 0);
        assert!(!episode.exhausted);
    }

    #[test]
    fn set_aware_unique_exact_goal_commits_directly() {
        let mut sc = shared_prefix_scenario();
        sc.candidate_goals.truncate(1);
        sc.hidden_goal_index = 0;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cfg = AgentConfig::for_name(AgentName::SetAwareParallelPlanning);
        let episode = run_episode(sc, &cfg, &mut rng, None);
        assert!(episode.success);
        assert_eq!(episode.actions_to_unique_id, Some(0));
        assert_eq!(episode.multi_goal_probe_actions, 0);
    }

    #[test]
    fn set_aware_empty_exact_live_set_does_not_take_arbitrary_action() {
        let mut sc = shared_prefix_scenario();
        sc.candidate_goals = vec![Goal::ReachMarker { marker: 99 }];
        sc.hidden_goal_index = 0;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cfg = AgentConfig::for_name(AgentName::SetAwareParallelPlanning);
        let episode = run_episode(sc, &cfg, &mut rng, None);
        assert!(episode.exhausted);
        assert!(!episode.success);
        assert!(!episode.failed);
        assert_eq!(episode.env_actions, 0);
    }

    #[test]
    fn set_aware_no_jointly_safe_probe_does_not_take_unsafe_fallback() {
        let mut sc = shared_prefix_scenario();
        sc.width = 1;
        sc.height = 3;
        sc.walls.clear();
        sc.start = Pos::new(0, 1);
        sc.switches = vec![Pos::new(0, 0), Pos::new(0, 2)];
        sc.candidate_goals = vec![
            Goal::ActivateSwitchesInOrder { order: vec![0, 1] },
            Goal::ActivateSwitchesInOrder { order: vec![1, 0] },
        ];
        sc.hidden_goal_index = 0;
        sc.undo_enabled = false;

        let sim = Simulator::new(sc.clone());
        let mut alive = BTreeSet::from([0, 1]);
        let mut stats = AgentStats::default();
        let plans = exact_live_plans(&sim, &sc.candidate_goals, &mut alive, &mut stats);
        assert!(
            choose_falsification_probe(&sim, &sc.candidate_goals, &plans, &mut stats).is_none()
        );
        assert!(choose_commitment_target(&sim, &sc.candidate_goals, &alive, &plans).is_none());

        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cfg = AgentConfig::for_name(AgentName::SetAwareParallelPlanning);
        let episode = run_episode(sc, &cfg, &mut rng, None);
        assert!(episode.exhausted);
        assert!(!episode.failed);
        assert_eq!(episode.env_actions, 0);
    }

    #[test]
    fn set_aware_rejects_catastrophic_hazard_plan() {
        let sc = avoid_fork_scenario();
        let sim = Simulator::new(sc.clone());
        let candidates = sc.candidate_goals.clone();
        let mut alive: BTreeSet<usize> = BTreeSet::from([0, 1]);
        let mut stats = AgentStats::default();
        let plans = exact_live_plans(&sim, &candidates, &mut alive, &mut stats);
        assert!(
            simulate_probe_endpoint_successes(
                &sim,
                &candidates,
                &plans,
                plans.iter().find(|p| p.index == 0).unwrap(),
                &mut stats
            )
            .is_none(),
            "Reach-via-hazard must be rejected as unsafe"
        );
        let (idx, _, _) = choose_falsification_probe(&sim, &candidates, &plans, &mut stats)
            .expect("safe Avoid probe");
        assert_eq!(idx, 1);
        assert_eq!(
            plans.iter().find(|p| p.index == idx).unwrap().first(),
            Some(Action::Move(Dir::North))
        );

        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let cfg = AgentConfig::for_name(AgentName::SetAwareParallelPlanning);
        let episode = run_episode(sc, &cfg, &mut rng, None);
        assert!(episode.success, "safe detour should solve avoid goal");
        assert!(!episode.failed);
        // Avoid's endpoint also satisfies Reach, so endpoint_tests may be ≥2.
        assert_eq!(episode.incorrect_commitments, 0);
    }

    #[test]
    fn terminal_success_is_positive_evidence_for_both_hidden_goal_policies() {
        let mut sc = shared_prefix_scenario();
        sc.start = sc.markers[0];
        sc.hidden_goal_index = 0;
        for name in [
            AgentName::CandidateGoalDiscrimination,
            AgentName::SetAwareParallelPlanning,
            AgentName::SharedProgressPlanning,
        ] {
            let cfg = AgentConfig::for_name(name);
            let mut rng = ChaCha8Rng::seed_from_u64(0);
            let stats = run_episode(sc.clone(), &cfg, &mut rng, None);
            assert!(stats.success);
            assert_eq!(stats.env_actions, 0);
            assert_eq!(stats.identified_candidate, Some(0));
            assert_eq!(stats.actions_to_unique_id, Some(0));
        }
    }

    #[test]
    fn terminal_failure_retains_only_failure_predicting_candidates() {
        let sc = avoid_fork_scenario();
        let mut sim = Simulator::new(sc.clone());
        let out = sim.step(Action::Move(Dir::East));
        assert!(out.failed);
        let mut alive = BTreeSet::from([0, 1]);
        let removed = update_belief_from_outcome(&sim, &sc.candidate_goals, &mut alive, &out);
        assert_eq!(removed, vec![0]);
        assert_eq!(alive, BTreeSet::from([1]));
    }

    #[test]
    fn exact_pruning_prevents_unreachable_distractor_from_disabling_safety() {
        let mut sc = avoid_fork_scenario();
        sc.width = 4;
        sc.height = 3;
        sc.markers.push(Pos::new(3, 2));
        sc.walls.extend([Pos::new(2, 2), Pos::new(3, 1)]);
        sc.candidate_goals.push(Goal::ReachMarker { marker: 1 });

        let sim = Simulator::new(sc.clone());
        let mut alive: BTreeSet<usize> = (0..sc.candidate_goals.len()).collect();
        let mut stats = AgentStats::default();
        prune_impossible(&sim, &sc.candidate_goals, &mut alive);
        assert!(
            alive.contains(&2),
            "cheap viability keeps valid marker index"
        );
        let plans = exact_live_plans(&sim, &sc.candidate_goals, &mut alive, &mut stats);
        assert!(
            !alive.contains(&2),
            "exact search must drop enclosed marker"
        );

        let idx = choose_commitment_target(&sim, &sc.candidate_goals, &alive, &plans)
            .expect("commitment after pruning");
        assert_eq!(idx, 1);
        assert_eq!(
            plans.iter().find(|p| p.index == idx).unwrap().first(),
            Some(Action::Move(Dir::North))
        );
        assert_ne!(
            plans.iter().find(|p| p.index == 0).unwrap().first(),
            Some(Action::Move(Dir::North))
        );
    }

    fn narrow_marker_resource_scenario() -> Scenario {
        let mut sc = marker_resource_probe_scenario();
        sc.candidate_goals.truncate(3);
        sc.hidden_goal_index = 0;
        sc
    }

    fn first_parallel_decision(
        name: AgentName,
        sc: &Scenario,
    ) -> (Option<Action>, Option<(usize, u64)>, AgentStats) {
        let sim = Simulator::new(sc.clone());
        let candidates = sc.candidate_goals.clone();
        let mut alive: BTreeSet<usize> = (0..candidates.len()).collect();
        let mut stats = AgentStats::default();
        let plans = exact_live_plans(&sim, &candidates, &mut alive, &mut stats);
        let visited = HashSet::from([StateKey::from_state(sim.state(), false)]);
        let mut router = RouterState::default();
        let routing = routing_strategy(name).expect("p1c");
        let choice = select_exact_live_action(
            &sim,
            &candidates,
            &plans,
            routing,
            &mut router,
            &visited,
            &mut stats,
        );
        match choice {
            Some(SelectedAct::Progress { action }) => (Some(action), None, stats),
            Some(SelectedAct::Probe {
                index,
                actions,
                endpoint_tests,
            }) => (
                actions.first().copied(),
                Some((index, endpoint_tests)),
                stats,
            ),
            Some(SelectedAct::SingleStep { action, .. }) => (Some(action), None, stats),
            None => (None, None, stats),
        }
    }

    #[test]
    fn progress_only_vs_falsify_only_on_broad_set() {
        let sc = marker_resource_probe_scenario();
        let sim = Simulator::new(sc.clone());
        let mut alive: BTreeSet<usize> = (0..sc.candidate_goals.len()).collect();
        let mut stats = AgentStats::default();
        let plans = exact_live_plans(&sim, &sc.candidate_goals, &mut alive, &mut stats);
        let visited = HashSet::from([StateKey::from_state(sim.state(), false)]);
        let progress =
            choose_shared_progress(&sim, &sc.candidate_goals, &plans, &visited, &mut stats)
                .expect("majority East");
        assert_eq!(progress, Action::Move(Dir::East));
        let (f_idx, f_pred, f_acts) =
            choose_falsification_probe(&sim, &sc.candidate_goals, &plans, &mut stats)
                .expect("South probe");
        assert_eq!(f_idx, 0);
        assert_eq!(f_pred, 2);
        assert_eq!(f_acts, vec![Action::Move(Dir::South)]);

        let (p_act, p_probe, _) = first_parallel_decision(AgentName::SharedProgressPlanning, &sc);
        assert_eq!(p_act, Some(Action::Move(Dir::East)));
        assert!(p_probe.is_none());

        let (f_act, f_probe, _) = first_parallel_decision(AgentName::SetAwareParallelPlanning, &sc);
        assert_eq!(f_act, Some(Action::Move(Dir::South)));
        assert_eq!(f_probe, Some((0, 2)));

        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let progress_ep = run_episode(
            sc.clone(),
            &AgentConfig::for_name(AgentName::SharedProgressPlanning),
            &mut rng,
            None,
        );
        assert!(progress_ep.shared_progress_actions >= 1);
        assert_eq!(progress_ep.incorrect_commitments, 0);

        let falsify_ep = run_episode(
            sc,
            &AgentConfig::for_name(AgentName::SetAwareParallelPlanning),
            &mut rng,
            None,
        );
        assert!(falsify_ep.multi_goal_probe_actions >= 1);
        assert_eq!(falsify_ep.shared_progress_actions, 0);
        assert_eq!(falsify_ep.incorrect_commitments, 0);
    }

    #[test]
    fn broad_progress_prefers_progress_when_n_ge_4_and_falsify_when_narrow() {
        let broad = marker_resource_probe_scenario();
        let (act, probe, _) =
            first_parallel_decision(AgentName::BroadProgressNarrowFalsify, &broad);
        assert_eq!(act, Some(Action::Move(Dir::East)));
        assert!(probe.is_none(), "n≥4 should prefer shared progress");

        let narrow = narrow_marker_resource_scenario();
        let (act, probe, _) =
            first_parallel_decision(AgentName::BroadProgressNarrowFalsify, &narrow);
        assert_eq!(act, Some(Action::Move(Dir::South)));
        assert_eq!(probe, Some((0, 2)), "n=2..3 should prefer falsification");
    }

    #[test]
    fn broad_falsify_reverses_breadth_preference() {
        let broad = marker_resource_probe_scenario();
        let (act, probe, _) =
            first_parallel_decision(AgentName::BroadFalsifyNarrowProgress, &broad);
        assert_eq!(act, Some(Action::Move(Dir::South)));
        assert_eq!(probe, Some((0, 2)));

        let narrow = narrow_marker_resource_scenario();
        let (act, probe, _) =
            first_parallel_decision(AgentName::BroadFalsifyNarrowProgress, &narrow);
        assert_eq!(act, Some(Action::Move(Dir::South)));
        // n=3: South is both majority progress and the multi-goal probe action.
        // Preference is progress, so it must not open a sticky width≥2 probe.
        assert!(probe.is_none());
    }

    #[test]
    fn alternating_starts_with_progress_then_falsifies() {
        let sc = marker_resource_probe_scenario();
        let mut sim = Simulator::new(sc.clone());
        let candidates = sc.candidate_goals.clone();
        let mut alive: BTreeSet<usize> = (0..candidates.len()).collect();
        let mut stats = AgentStats::default();
        let mut visited = HashSet::from([StateKey::from_state(sim.state(), false)]);
        let mut router = RouterState::default();
        let routing = RoutingStrategy::Alternating;

        let plans = exact_live_plans(&sim, &candidates, &mut alive, &mut stats);
        let first = select_exact_live_action(
            &sim,
            &candidates,
            &plans,
            routing,
            &mut router,
            &visited,
            &mut stats,
        );
        assert!(matches!(
            first,
            Some(SelectedAct::Progress {
                action: Action::Move(Dir::East)
            })
        ));
        assert!(!router.alternate_prefer_progress);

        sim.step(Action::Move(Dir::East));
        visited.insert(StateKey::from_state(sim.state(), false));
        prune_impossible(&sim, &candidates, &mut alive);
        let plans = exact_live_plans(&sim, &candidates, &mut alive, &mut stats);
        assert!(plans.len() > 1);
        let second = select_exact_live_action(
            &sim,
            &candidates,
            &plans,
            routing,
            &mut router,
            &visited,
            &mut stats,
        );
        match second {
            Some(SelectedAct::Probe {
                endpoint_tests: t, ..
            }) if t >= 2 => {}
            other => panic!("second ambiguous decision should falsify, got {other:?}"),
        }
        assert!(router.alternate_prefer_progress);
        assert_eq!(stats.parallel_method_switches, 1);
    }

    #[test]
    fn capped_broad_forces_falsify_after_two_progress_actions() {
        let sc = marker_resource_probe_scenario();
        let sim = Simulator::new(sc.clone());
        let candidates = sc.candidate_goals.clone();
        let mut alive: BTreeSet<usize> = (0..candidates.len()).collect();
        let mut stats = AgentStats::default();
        let visited = HashSet::from([StateKey::from_state(sim.state(), false)]);
        let mut router = RouterState::default();
        let routing = RoutingStrategy::CappedBroadProgress;
        let plans = exact_live_plans(&sim, &candidates, &mut alive, &mut stats);
        assert!(plans.len() >= 4);

        let first = select_exact_live_action(
            &sim,
            &candidates,
            &plans,
            routing,
            &mut router,
            &visited,
            &mut stats,
        );
        assert!(matches!(
            first,
            Some(SelectedAct::Progress {
                action: Action::Move(Dir::East)
            })
        ));
        assert_eq!(router.progress_streak, 1);

        // Second consecutive progress decision (streak already 1 from above).
        let second = select_exact_live_action(
            &sim,
            &candidates,
            &plans,
            routing,
            &mut router,
            &visited,
            &mut stats,
        );
        assert!(matches!(second, Some(SelectedAct::Progress { .. })));
        assert_eq!(router.progress_streak, 2);

        let forced = select_exact_live_action(
            &sim,
            &candidates,
            &plans,
            routing,
            &mut router,
            &visited,
            &mut stats,
        );
        match forced {
            Some(SelectedAct::Probe {
                endpoint_tests: t, ..
            }) if t >= 2 => {
                assert_eq!(router.progress_streak, 0);
            }
            other => panic!("cap should force falsification attempt, got {other:?}"),
        }
    }

    #[test]
    fn cost_aware_prefers_cheap_multi_goal_probe_over_regretful_progress() {
        let sc = marker_resource_probe_scenario();
        let sim = Simulator::new(sc.clone());
        let mut alive: BTreeSet<usize> = (0..sc.candidate_goals.len()).collect();
        let mut stats = AgentStats::default();
        let plans = exact_live_plans(&sim, &sc.candidate_goals, &mut alive, &mut stats);
        let visited = HashSet::from([StateKey::from_state(sim.state(), false)]);
        let progress =
            choose_shared_progress(&sim, &sc.candidate_goals, &plans, &visited, &mut stats)
                .expect("East majority");
        let falsify = choose_falsification_probe(&sim, &sc.candidate_goals, &plans, &mut stats)
            .expect("probe");
        let pu = shared_progress_utility(&sim, &sc.candidate_goals, &plans, progress, &mut stats)
            .expect("progress utility");
        let fu = (falsify.1 as f64) / (falsify.2.len() as f64);
        assert!(
            fu > pu,
            "South probe utility {fu} should beat East progress {pu}"
        );

        let (act, probe, _) = first_parallel_decision(AgentName::CostAwareParallelPlanning, &sc);
        assert_eq!(act, Some(Action::Move(Dir::South)));
        assert_eq!(probe, Some((0, 2)));

        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let ep = run_episode(
            sc,
            &AgentConfig::for_name(AgentName::CostAwareParallelPlanning),
            &mut rng,
            None,
        );
        assert!(ep.multi_goal_probe_actions >= 1);
        assert_eq!(ep.shared_progress_actions, 0);
    }

    #[test]
    fn shared_progress_safety_veto_rejects_hazardous_majority() {
        let sc = avoid_fork_scenario();
        let sim = Simulator::new(sc.clone());
        let mut alive: BTreeSet<usize> = BTreeSet::from([0, 1]);
        let mut stats = AgentStats::default();
        let plans = exact_live_plans(&sim, &sc.candidate_goals, &mut alive, &mut stats);
        assert!(!first_action_jointly_safe(
            &sim,
            &sc.candidate_goals,
            &plans,
            Action::Move(Dir::East)
        ));
        let visited = HashSet::from([StateKey::from_state(sim.state(), false)]);
        let progress =
            choose_shared_progress(&sim, &sc.candidate_goals, &plans, &visited, &mut stats);
        assert_ne!(progress, Some(Action::Move(Dir::East)));

        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let ep = run_episode(
            sc,
            &AgentConfig::for_name(AgentName::SharedProgressPlanning),
            &mut rng,
            None,
        );
        assert!(!ep.failed);
        assert!(ep.success || ep.exhausted);
    }

    #[test]
    fn p1c_unique_direct_solve_and_determinism_and_no_hidden_arg() {
        for name in [
            AgentName::SharedProgressPlanning,
            AgentName::BroadProgressNarrowFalsify,
            AgentName::BroadFalsifyNarrowProgress,
            AgentName::AlternatingParallelPlanning,
            AgentName::CostAwareParallelPlanning,
            AgentName::CappedBroadProgressPlanning,
        ] {
            let mut sc = shared_prefix_scenario();
            sc.candidate_goals.truncate(1);
            sc.hidden_goal_index = 0;
            let cfg = AgentConfig::for_name(name);
            let mut rng = ChaCha8Rng::seed_from_u64(0);
            let ep = run_episode(sc.clone(), &cfg, &mut rng, None);
            assert!(ep.success, "{name:?}");
            assert_eq!(ep.actions_to_unique_id, Some(0));
            assert_eq!(ep.multi_goal_probe_actions, 0);

            let broad = marker_resource_probe_scenario();
            let mut a = ChaCha8Rng::seed_from_u64(1);
            let mut b = ChaCha8Rng::seed_from_u64(99);
            assert_eq!(
                run_episode(broad.clone(), &cfg, &mut a, None),
                run_episode(broad, &cfg, &mut b, None),
                "{name:?} must be deterministic"
            );
        }
    }

    #[test]
    fn unavailable_preference_is_not_an_actual_method_switch() {
        // n=2 shared fork: majority progress exists; multi-goal falsify does not.
        let sc = shared_prefix_scenario();
        let (act, probe, stats) =
            first_parallel_decision(AgentName::BroadProgressNarrowFalsify, &sc);
        // Prefer falsify at n=2, unavailable → switch to progress.
        assert_eq!(act, Some(Action::Move(Dir::East)));
        assert!(probe.is_none());
        assert_eq!(stats.parallel_method_switches, 0);
    }

    #[test]
    fn sequential_fallback_resets_cap_without_flipping_alternation() {
        let sc = sequential_only_fork_scenario();
        let sim = Simulator::new(sc.clone());
        let mut alive = BTreeSet::from([0, 1]);
        let mut stats = AgentStats::default();
        let plans = exact_live_plans(&sim, &sc.candidate_goals, &mut alive, &mut stats);
        let visited = HashSet::from([StateKey::from_state(sim.state(), false)]);

        let mut capped = RouterState {
            progress_streak: 2,
            ..RouterState::default()
        };
        let selected = select_exact_live_action(
            &sim,
            &sc.candidate_goals,
            &plans,
            RoutingStrategy::CappedBroadProgress,
            &mut capped,
            &visited,
            &mut stats,
        );
        assert!(matches!(
            selected,
            Some(SelectedAct::Probe {
                endpoint_tests: 0,
                ..
            })
        ));
        assert_eq!(capped.progress_streak, 0);

        let mut alternating = RouterState::default();
        let selected = select_exact_live_action(
            &sim,
            &sc.candidate_goals,
            &plans,
            RoutingStrategy::Alternating,
            &mut alternating,
            &visited,
            &mut stats,
        );
        assert!(selected.is_some());
        assert!(alternating.alternate_prefer_progress);
        assert_eq!(alternating.last_parallel_method, None);
    }
}
