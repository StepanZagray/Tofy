//! Exact bounded search over the public simulator transition.

use crate::domain::{
    goal_satisfied, legal_actions, Action, Goal, Pos, Scenario, Simulator, State, UndoFrame,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};

/// Canonical visited-set key: behaviorally relevant state, no `actions_used`.
/// When Undo is enabled, the undo history is part of the key.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateKey {
    pub pos: Pos,
    pub remaining_collectibles: u32,
    pub remaining_pickups: u32,
    pub switch_trace: Vec<u8>,
    pub resource: u8,
    pub touched_hazards: u32,
    pub undo_stack: Option<Vec<UndoFrame>>,
}

impl StateKey {
    pub fn from_state(state: &State, undo_enabled: bool) -> Self {
        Self {
            pos: state.pos,
            remaining_collectibles: state.remaining_collectibles,
            remaining_pickups: state.remaining_pickups,
            switch_trace: state.switch_trace.clone(),
            resource: state.resource,
            touched_hazards: state.touched_hazards,
            undo_stack: if undo_enabled {
                Some(state.undo_stack.clone())
            } else {
                None
            },
        }
    }
}

/// Plan plus expansion count from exact search.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchResult {
    pub actions: Vec<Action>,
    pub expanded: u64,
}

/// Exact heuristic for all six goal families (non-negative; lower is better).
pub fn heuristic(scenario: &Scenario, state: &State, goal: &Goal) -> u32 {
    match goal {
        Goal::ReachMarker { marker } => match scenario.markers.get(*marker as usize) {
            Some(m) => manhattan(state.pos, *m),
            None => u32::MAX / 4,
        },
        Goal::CollectAll => {
            if scenario.collectibles.is_empty() {
                return u32::MAX / 4;
            }
            let mut remaining = 0u32;
            let mut nearest = u32::MAX;
            for (i, p) in scenario.collectibles.iter().enumerate() {
                if i < 32 && (state.remaining_collectibles & (1u32 << i)) != 0 {
                    remaining += 1;
                    nearest = nearest.min(manhattan(state.pos, *p));
                }
            }
            if remaining == 0 {
                0
            } else {
                nearest.saturating_add(remaining.saturating_sub(1))
            }
        }
        Goal::ActivateSwitchesInOrder { order } => {
            if order.is_empty() || scenario.switches.is_empty() {
                return u32::MAX / 4;
            }
            if !switch_prefix_possible(&state.switch_trace, order) {
                return u32::MAX / 4;
            }
            if state.switch_trace.len() >= order.len() {
                return 0;
            }
            let next = order[state.switch_trace.len()] as usize;
            match scenario.switches.get(next) {
                Some(p) => manhattan(state.pos, *p),
                None => u32::MAX / 4,
            }
        }
        Goal::PreserveResourceReachMarker {
            marker,
            min_resource,
        } => {
            let Some(m) = scenario.markers.get(*marker as usize) else {
                return u32::MAX / 4;
            };
            let dist = manhattan(state.pos, *m);
            if state.resource >= *min_resource {
                dist
            } else {
                let need = u32::from(min_resource.saturating_sub(state.resource));
                dist.saturating_add(need)
            }
        }
        Goal::AvoidHazardReachMarker { hazard, marker } => {
            let h = *hazard as usize;
            if h < 32 && (state.touched_hazards & (1u32 << h)) != 0 {
                return u32::MAX / 4;
            }
            if scenario.hazards.get(h).is_none() {
                return u32::MAX / 4;
            }
            match scenario.markers.get(*marker as usize) {
                Some(m) => manhattan(state.pos, *m),
                None => u32::MAX / 4,
            }
        }
        Goal::TriggerTerminal { trigger } => {
            match scenario.terminal_triggers.get(*trigger as usize) {
                Some(t) => manhattan(state.pos, *t),
                None => u32::MAX / 4,
            }
        }
    }
}

/// Whether the current switch trace is a compatible prefix of `order`.
pub fn switch_prefix_possible(trace: &[u8], order: &[u8]) -> bool {
    let n = trace.len().min(order.len());
    trace[..n] == order[..n]
}

/// Cheap exact possibility filter (no search). False ⇒ goal cannot hold from this state.
pub fn goal_possible(scenario: &Scenario, state: &State, goal: &Goal) -> bool {
    if goal_satisfied(scenario, state, goal) {
        return true;
    }
    match goal {
        Goal::ReachMarker { marker } => scenario.markers.get(*marker as usize).is_some(),
        Goal::CollectAll => !scenario.collectibles.is_empty(),
        Goal::ActivateSwitchesInOrder { order } => {
            !order.is_empty()
                && !scenario.switches.is_empty()
                && switch_prefix_possible(&state.switch_trace, order)
                && order
                    .iter()
                    .all(|&i| scenario.switches.get(i as usize).is_some())
        }
        Goal::PreserveResourceReachMarker {
            marker,
            min_resource,
        } => {
            if scenario.markers.get(*marker as usize).is_none() {
                return false;
            }
            if state.resource >= *min_resource {
                return true;
            }
            let need = u32::from(min_resource.saturating_sub(state.resource));
            state.remaining_pickups.count_ones() >= need
        }
        Goal::AvoidHazardReachMarker { hazard, marker } => {
            if scenario.markers.get(*marker as usize).is_none() {
                return false;
            }
            if scenario.hazards.get(*hazard as usize).is_none() {
                return false;
            }
            let h = *hazard as usize;
            // Once the named hazard is touched, the goal is permanently impossible
            // without Undo (which search treats separately).
            h >= 32 || (state.touched_hazards & (1u32 << h)) == 0
        }
        Goal::TriggerTerminal { trigger } => {
            scenario.terminal_triggers.get(*trigger as usize).is_some()
        }
    }
}

/// Public-state viability using the same Undo recovery semantics as exact search.
/// False means no legal Undo prefix can restore the goal's cheap possibility.
pub fn goal_viable_at(sim: &Simulator, state: &State, goal: &Goal) -> bool {
    let structurally_valid = match goal {
        Goal::AvoidHazardReachMarker { hazard, marker } => {
            if *hazard >= 32 {
                return false;
            }
            sim.scenario().markers.get(*marker as usize).is_some()
                && sim.scenario().hazards.get(*hazard as usize).is_some()
        }
        Goal::TriggerTerminal { trigger } => sim
            .scenario()
            .terminal_triggers
            .get(*trigger as usize)
            .is_some(),
        _ => true,
    };
    if !structurally_valid {
        return false;
    }
    if goal_possible(sim.scenario(), state, goal) {
        return true;
    }
    if !sim.scenario().undo_enabled {
        return false;
    }

    let mut cur = state.clone();
    while !cur.undo_stack.is_empty() && cur.actions_used < sim.scenario().action_budget {
        cur = sim.transition(&cur, Action::Undo);
        if goal_possible(sim.scenario(), &cur, goal) {
            return true;
        }
    }
    false
}

/// One-step greedy: among legal actions, pick the successor with least heuristic.
/// Deterministic: ties broken by action index in `legal_actions` order.
pub fn greedy_action(sim: &Simulator, state: &State, goal: &Goal) -> (Action, u64) {
    let actions = legal_actions(sim.scenario());
    let mut best_a = actions[0];
    let mut best_h = u32::MAX;
    let mut evals = 0u64;
    for &a in &actions {
        let nxt = sim.transition(state, a);
        let h = heuristic(sim.scenario(), &nxt, goal);
        evals += 1;
        if h < best_h {
            best_h = h;
            best_a = a;
        }
    }
    (best_a, evals)
}

/// Exact bounded BFS shortest path to an explicit `goal`.
/// Never reads `Scenario.hidden_goal_index`. Does not mutate `sim` or `start`.
/// `max_actions_used` is an absolute cap on successor `actions_used`.
pub fn shortest_path(
    sim: &Simulator,
    start: &State,
    goal: &Goal,
    max_actions_used: u16,
) -> Option<SearchResult> {
    let sc = sim.scenario();
    if goal_satisfied(sc, start, goal) {
        return Some(SearchResult {
            actions: Vec::new(),
            expanded: 0,
        });
    }
    let actions = legal_actions(sc);
    let mut q = VecDeque::new();
    let mut seen = HashSet::new();
    let mut parent: HashMap<StateKey, (StateKey, Action)> = HashMap::new();
    // With known deterministic dynamics, moving and later undoing is strictly
    // dominated. Undo can only help by rolling back history that existed when
    // planning began, so search permits an Undo prefix followed by moves.
    let can_undo = sc.undo_enabled && !start.undo_stack.is_empty();
    let start_key = StateKey::from_state(start, can_undo);
    seen.insert(start_key.clone());
    q.push_back((start.clone(), can_undo));
    let mut expanded = 0u64;

    while let Some((cur, can_undo)) = q.pop_front() {
        expanded += 1;
        if cur.actions_used >= max_actions_used {
            continue;
        }
        let cur_key = StateKey::from_state(&cur, can_undo);
        for &a in &actions {
            if matches!(a, Action::Undo) && !can_undo {
                continue;
            }
            let nxt = sim.transition(&cur, a);
            if nxt.actions_used > max_actions_used {
                continue;
            }
            let next_can_undo = can_undo && matches!(a, Action::Undo) && !nxt.undo_stack.is_empty();
            let nk = StateKey::from_state(&nxt, next_can_undo);
            if !seen.insert(nk.clone()) {
                continue;
            }
            parent.insert(nk.clone(), (cur_key.clone(), a));
            if goal_satisfied(sc, &nxt, goal) {
                return Some(SearchResult {
                    actions: reconstruct(&parent, &start_key, &nk),
                    expanded,
                });
            }
            q.push_back((nxt, next_can_undo));
        }
    }
    None
}

fn reconstruct(
    parent: &HashMap<StateKey, (StateKey, Action)>,
    start: &StateKey,
    goal: &StateKey,
) -> Vec<Action> {
    let mut out = Vec::new();
    let mut cur = goal.clone();
    while &cur != start {
        let (prev, a) = parent.get(&cur).expect("parent chain");
        out.push(*a);
        cur = prev.clone();
    }
    out.reverse();
    out
}

/// Deterministic limited-width beam / best-first search with horizon and width.
/// Scores with [`heuristic`]. No MCTS / value head. Does not mutate inputs.
pub fn beam_search(
    sim: &Simulator,
    start: &State,
    goal: &Goal,
    horizon: u16,
    beam_width: usize,
) -> Option<SearchResult> {
    let sc = sim.scenario();
    let beam_width = beam_width.max(1);
    if goal_satisfied(sc, start, goal) {
        return Some(SearchResult {
            actions: Vec::new(),
            expanded: 0,
        });
    }
    let actions = legal_actions(sc);
    let can_undo = sc.undo_enabled && !start.undo_stack.is_empty();
    let mut beam: Vec<(u32, State, Vec<Action>, bool)> = vec![(
        heuristic(sc, start, goal),
        start.clone(),
        Vec::new(),
        can_undo,
    )];
    let mut expanded = 0u64;
    let mut best: Option<Vec<Action>> = None;

    for _depth in 0..horizon {
        let mut cand: Vec<(u32, State, Vec<Action>, bool)> = Vec::new();
        let mut seen_local = HashSet::new();
        for (_h, st, plan, can_undo) in &beam {
            expanded += 1;
            if st.actions_used >= sc.action_budget {
                continue;
            }
            for &a in &actions {
                if matches!(a, Action::Undo) && !can_undo {
                    continue;
                }
                let nxt = sim.transition(st, a);
                let mut nplan = plan.clone();
                nplan.push(a);
                if goal_satisfied(sc, &nxt, goal) {
                    match &best {
                        Some(p)
                            if p.len() < nplan.len()
                                || (p.len() == nplan.len()
                                    && action_plan_cmp(p, &nplan) != Ordering::Greater) => {}
                        _ => best = Some(nplan.clone()),
                    }
                }
                let next_can_undo =
                    *can_undo && matches!(a, Action::Undo) && !nxt.undo_stack.is_empty();
                let nk = StateKey::from_state(&nxt, next_can_undo);
                if !seen_local.insert(nk) {
                    continue;
                }
                let nh = heuristic(sc, &nxt, goal);
                cand.push((nh, nxt, nplan, next_can_undo));
            }
        }
        if let Some(ref p) = best {
            return Some(SearchResult {
                actions: p.clone(),
                expanded,
            });
        }
        if cand.is_empty() {
            break;
        }
        cand.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.2.len().cmp(&b.2.len()))
                .then_with(|| action_plan_cmp(&a.2, &b.2))
        });
        cand.truncate(beam_width);
        beam = cand;
    }
    best.map(|actions| SearchResult { actions, expanded })
}

fn action_plan_cmp(a: &[Action], b: &[Action]) -> Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        match action_ord(*x).cmp(&action_ord(*y)) {
            Ordering::Equal => {}
            o => return o,
        }
    }
    a.len().cmp(&b.len())
}

fn action_ord(a: Action) -> u8 {
    match a {
        Action::Move(crate::domain::Dir::North) => 0,
        Action::Move(crate::domain::Dir::South) => 1,
        Action::Move(crate::domain::Dir::East) => 2,
        Action::Move(crate::domain::Dir::West) => 3,
        Action::Undo => 4,
    }
}

fn manhattan(a: Pos, b: Pos) -> u32 {
    ((a.x as i16 - b.x as i16).unsigned_abs() as u32)
        + ((a.y as i16 - b.y as i16).unsigned_abs() as u32)
}

/// Whether any plan exists under the absolute `actions_used` cap (exact BFS).
pub fn reachable(
    sim: &Simulator,
    start: &State,
    goal: &Goal,
    max_actions_used: u16,
) -> (bool, u64) {
    match shortest_path(sim, start, goal, max_actions_used) {
        Some(r) => (true, r.expanded),
        None => (false, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Dir, Split};
    use std::collections::BTreeSet;

    fn obstructed_map() -> Scenario {
        // 5x3: start (0,1), marker (4,1); wall at (2,1) forces a length-6 detour.
        let mut walls = BTreeSet::new();
        walls.insert(Pos::new(2, 1));
        Scenario {
            width: 5,
            height: 3,
            walls,
            markers: vec![Pos::new(4, 1)],
            collectibles: vec![],
            switches: vec![],
            hazards: vec![],
            resource_pickups: vec![],
            terminal_triggers: vec![],
            start: Pos::new(0, 1),
            initial_resource: 0,
            action_budget: 20,
            undo_enabled: false,
            candidate_goals: vec![Goal::ReachMarker { marker: 0 }],
            hidden_goal_index: 0,
            split: Split::Train,
            seed: 0,
            episode_id: 0,
        }
    }

    fn avoid_and_trigger_map() -> Scenario {
        // Open 4x3. Hazard at (1,0) sits on the short path to marker 0 at (3,0).
        // Marker 1 at (0,2) is reachable without the hazard. Trigger at (3,2).
        Scenario {
            width: 4,
            height: 3,
            walls: BTreeSet::new(),
            markers: vec![Pos::new(3, 0), Pos::new(0, 2)],
            collectibles: vec![],
            switches: vec![],
            hazards: vec![Pos::new(1, 0)],
            resource_pickups: vec![],
            terminal_triggers: vec![Pos::new(3, 2)],
            start: Pos::new(0, 0),
            initial_resource: 2,
            action_budget: 20,
            undo_enabled: true,
            candidate_goals: vec![
                Goal::ReachMarker { marker: 0 },
                Goal::AvoidHazardReachMarker {
                    hazard: 0,
                    marker: 1,
                },
                Goal::TriggerTerminal { trigger: 0 },
            ],
            hidden_goal_index: 1,
            split: Split::Train,
            seed: 2,
            episode_id: 0,
        }
    }

    #[test]
    fn shortest_path_minimal_on_obstructed_map() {
        let sc = obstructed_map();
        let sim = Simulator::new(sc.clone());
        let start = State::initial(&sc);
        let goal = Goal::ReachMarker { marker: 0 };
        let res = shortest_path(&sim, &start, &goal, sc.action_budget).expect("path");
        assert_eq!(res.actions.len(), 6, "detour shortest must be 6");
        let mut sim2 = Simulator::new(sc);
        for a in &res.actions {
            sim2.step(*a);
        }
        assert!(sim2.goal_satisfied(sim2.state(), &goal));
    }

    #[test]
    fn no_solution_when_bound_too_tight() {
        let sc = obstructed_map();
        let sim = Simulator::new(sc.clone());
        let start = State::initial(&sc);
        let goal = Goal::ReachMarker { marker: 0 };
        assert!(shortest_path(&sim, &start, &goal, 5).is_none());
    }

    #[test]
    fn plan_execution_reaches_collect_and_switch_goals() {
        let sc = Scenario {
            width: 4,
            height: 3,
            walls: BTreeSet::new(),
            markers: vec![Pos::new(3, 2)],
            collectibles: vec![Pos::new(1, 0), Pos::new(2, 0)],
            switches: vec![Pos::new(0, 1), Pos::new(1, 1)],
            hazards: vec![],
            resource_pickups: vec![],
            terminal_triggers: vec![],
            start: Pos::new(0, 0),
            initial_resource: 1,
            action_budget: 30,
            undo_enabled: false,
            candidate_goals: vec![
                Goal::CollectAll,
                Goal::ActivateSwitchesInOrder { order: vec![0, 1] },
            ],
            hidden_goal_index: 0,
            split: Split::Train,
            seed: 1,
            episode_id: 1,
        };
        let sim = Simulator::new(sc.clone());
        let start = State::initial(&sc);
        for goal in [
            Goal::CollectAll,
            Goal::ActivateSwitchesInOrder { order: vec![0, 1] },
        ] {
            let res = shortest_path(&sim, &start, &goal, sc.action_budget).expect("solvable");
            let mut s = Simulator::new(sc.clone());
            for a in res.actions {
                s.step(a);
            }
            assert!(s.goal_satisfied(s.state(), &goal));
        }
    }

    #[test]
    fn avoid_hazard_search_avoids_named_hazard() {
        let sc = avoid_and_trigger_map();
        let sim = Simulator::new(sc.clone());
        let start = State::initial(&sc);
        let goal = Goal::AvoidHazardReachMarker {
            hazard: 0,
            marker: 1,
        };
        let res = shortest_path(&sim, &start, &goal, sc.action_budget).expect("avoid path");
        let mut s = Simulator::new(sc.clone());
        for a in &res.actions {
            let out = s.step(*a);
            assert!(!out.failed, "plan must not touch the named hazard");
        }
        assert!(s.goal_satisfied(s.state(), &goal));
        assert_eq!(s.state().touched_hazards & 1, 0);
        assert_eq!(s.state().pos, Pos::new(0, 2));
    }

    #[test]
    fn avoid_hazard_impossible_after_touch() {
        let mut sc = avoid_and_trigger_map();
        sc.undo_enabled = false;
        let sim = Simulator::new(sc.clone());
        let mut st = State::initial(&sc);
        st = sim.transition(&st, Action::Move(Dir::East)); // onto hazard (1,0)
        assert_eq!(st.touched_hazards & 1, 1);
        let goal = Goal::AvoidHazardReachMarker {
            hazard: 0,
            marker: 1,
        };
        assert!(!goal_possible(&sc, &st, &goal));
        assert!(shortest_path(&sim, &st, &goal, sc.action_budget).is_none());
    }

    #[test]
    fn trigger_terminal_shortest_path() {
        let mut sc = avoid_and_trigger_map();
        sc.hidden_goal_index = 2;
        let sim = Simulator::new(sc.clone());
        let start = State::initial(&sc);
        let goal = Goal::TriggerTerminal { trigger: 0 };
        let res = shortest_path(&sim, &start, &goal, sc.action_budget).expect("trigger path");
        let mut s = Simulator::new(sc);
        for a in res.actions {
            s.step(a);
        }
        assert!(s.goal_satisfied(s.state(), &goal));
        assert_eq!(s.state().pos, Pos::new(3, 2));
    }

    #[test]
    fn beam_horizon_matters() {
        let sc = obstructed_map();
        let sim = Simulator::new(sc.clone());
        let start = State::initial(&sc);
        let goal = Goal::ReachMarker { marker: 0 };
        assert!(beam_search(&sim, &start, &goal, 3, 8).is_none());
        let long = beam_search(&sim, &start, &goal, 10, 8).expect("horizon 10");
        assert_eq!(long.actions.len(), 6);
    }

    #[test]
    fn search_does_not_mutate_simulator_or_state() {
        let sc = obstructed_map();
        let sim = Simulator::new(sc.clone());
        let start = State::initial(&sc);
        let before_sim = sim.state().clone();
        let before_start = start.clone();
        let goal = Goal::ReachMarker { marker: 0 };
        let _ = shortest_path(&sim, &start, &goal, sc.action_budget);
        let _ = beam_search(&sim, &start, &goal, 12, 4);
        let _ = greedy_action(&sim, &start, &goal);
        assert_eq!(sim.state(), &before_sim);
        assert_eq!(start, before_start);
    }

    #[test]
    fn state_key_includes_undo_and_touched_hazards() {
        let mut sc = avoid_and_trigger_map();
        sc.undo_enabled = true;
        let mut st = State::initial(&sc);
        let k0 = StateKey::from_state(&st, true);
        assert!(k0.undo_stack.as_ref().unwrap().is_empty());
        assert_eq!(k0.touched_hazards, 0);
        let sim = Simulator::new(sc);
        st = sim.transition(&st, Action::Move(Dir::East));
        let k1 = StateKey::from_state(&st, true);
        assert_ne!(k0, k1);
        assert_eq!(k1.touched_hazards & 1, 1);
        assert_eq!(k1.undo_stack.as_ref().unwrap().len(), 1);
        let k1_no = StateKey::from_state(&st, false);
        assert!(k1_no.undo_stack.is_none());
        assert_eq!(k1_no.touched_hazards & 1, 1);
    }

    #[test]
    fn switch_prefix_and_goal_possible() {
        let sc = Scenario {
            width: 3,
            height: 3,
            walls: BTreeSet::new(),
            markers: vec![Pos::new(2, 2)],
            collectibles: vec![Pos::new(1, 0)],
            switches: vec![Pos::new(0, 1), Pos::new(1, 1), Pos::new(2, 1)],
            hazards: vec![],
            resource_pickups: vec![],
            terminal_triggers: vec![],
            start: Pos::new(0, 0),
            initial_resource: 0,
            action_budget: 20,
            undo_enabled: false,
            candidate_goals: vec![],
            hidden_goal_index: 0,
            split: Split::Train,
            seed: 0,
            episode_id: 0,
        };
        let mut st = State::initial(&sc);
        let order = vec![0u8, 1, 2];
        assert!(switch_prefix_possible(&st.switch_trace, &order));
        st.switch_trace = vec![0, 2];
        assert!(!switch_prefix_possible(&st.switch_trace, &order));
        assert!(!goal_possible(
            &sc,
            &st,
            &Goal::ActivateSwitchesInOrder { order }
        ));
    }

    #[test]
    fn heuristic_defined_for_all_families() {
        let sc = Scenario {
            width: 4,
            height: 4,
            walls: BTreeSet::new(),
            markers: vec![Pos::new(3, 3)],
            collectibles: vec![Pos::new(1, 0)],
            switches: vec![Pos::new(0, 1)],
            hazards: vec![Pos::new(2, 2)],
            resource_pickups: vec![Pos::new(2, 0)],
            terminal_triggers: vec![Pos::new(3, 1)],
            start: Pos::new(0, 0),
            initial_resource: 1,
            action_budget: 20,
            undo_enabled: false,
            candidate_goals: vec![],
            hidden_goal_index: 0,
            split: Split::Train,
            seed: 0,
            episode_id: 0,
        };
        let st = State::initial(&sc);
        for g in [
            Goal::ReachMarker { marker: 0 },
            Goal::CollectAll,
            Goal::ActivateSwitchesInOrder { order: vec![0] },
            Goal::PreserveResourceReachMarker {
                marker: 0,
                min_resource: 1,
            },
            Goal::AvoidHazardReachMarker {
                hazard: 0,
                marker: 0,
            },
            Goal::TriggerTerminal { trigger: 0 },
        ] {
            let h = heuristic(&sc, &st, &g);
            assert!(h < u32::MAX / 2, "heuristic should be finite for {g:?}");
            assert!(goal_possible(&sc, &st, &g));
        }
    }

    #[test]
    fn undo_restores_search_key_history() {
        let sc = avoid_and_trigger_map();
        let sim = Simulator::new(sc.clone());
        let start = State::initial(&sc);
        let after_hazard = sim.transition(&start, Action::Move(Dir::East));
        assert_eq!(after_hazard.touched_hazards & 1, 1);
        let undone = sim.transition(&after_hazard, Action::Undo);
        assert_eq!(undone.touched_hazards, 0);
        assert_eq!(undone.pos, start.pos);
        // After undo, avoid goal is searchable again.
        let goal = Goal::AvoidHazardReachMarker {
            hazard: 0,
            marker: 1,
        };
        assert!(goal_possible(&sc, &undone, &goal));
        assert!(shortest_path(&sim, &undone, &goal, sc.action_budget).is_some());
    }
}
