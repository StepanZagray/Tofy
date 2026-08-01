//! Exact deterministic grid-domain MDP for hidden-objective experiments.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

/// Grid coordinate. Origin is top-left; +x east, +y south.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Pos {
    pub x: i8,
    pub y: i8,
}

impl Pos {
    pub const fn new(x: i8, y: i8) -> Self {
        Self { x, y }
    }

    pub fn checked_add(self, dx: i8, dy: i8) -> Option<Self> {
        Some(Self {
            x: self.x.checked_add(dx)?,
            y: self.y.checked_add(dy)?,
        })
    }
}

/// Cardinal move directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dir {
    North,
    South,
    East,
    West,
}

impl Dir {
    pub const ALL: [Dir; 4] = [Dir::North, Dir::South, Dir::East, Dir::West];

    pub fn delta(self) -> (i8, i8) {
        match self {
            Dir::North => (0, -1),
            Dir::South => (0, 1),
            Dir::East => (1, 0),
            Dir::West => (-1, 0),
        }
    }
}

/// Agent actions. Moves that leave the board or hit a wall are deterministic no-ops.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Move(Dir),
    Undo,
}

impl Action {
    pub fn moves() -> [Action; 4] {
        [
            Action::Move(Dir::North),
            Action::Move(Dir::South),
            Action::Move(Dir::East),
            Action::Move(Dir::West),
        ]
    }
}

/// Candidate / hidden goal predicates. Evaluated exactly against State + Scenario layout.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Goal {
    /// Stand on the marker with this index into `Scenario.markers`.
    ReachMarker { marker: u8 },
    /// Collect every collectible (none remaining).
    CollectAll,
    /// Activate switches in the given index order (prefix of activation trace).
    ActivateSwitchesInOrder { order: Vec<u8> },
    /// Reach a marker while keeping resource at least `min_resource`.
    PreserveResourceReachMarker { marker: u8, min_resource: u8 },
    /// Reach a marker without ever having entered the named hazard cell.
    /// Touching that hazard while this is the hidden goal is an irreversible terminal failure.
    AvoidHazardReachMarker { hazard: u8, marker: u8 },
    /// Stand on the terminal trigger with this index into `Scenario.terminal_triggers`.
    TriggerTerminal { trigger: u8 },
}

/// Stable family label for public candidate goals (curriculum / reporting).
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

/// Dataset split label carried in scenario metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Split {
    Train,
    HeldOutComposition,
}

/// Public scenario: layout, candidate goals, and oracle-only hidden index.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Scenario {
    pub width: u8,
    pub height: u8,
    pub walls: BTreeSet<Pos>,
    pub markers: Vec<Pos>,
    pub collectibles: Vec<Pos>,
    pub switches: Vec<Pos>,
    /// Cells that decrement resource by one when entered (floor at 0).
    pub hazards: Vec<Pos>,
    /// Cells that increment resource by one when entered (once; see State).
    pub resource_pickups: Vec<Pos>,
    /// Terminal trigger pads indexed by `Goal::TriggerTerminal`.
    pub terminal_triggers: Vec<Pos>,
    pub start: Pos,
    pub initial_resource: u8,
    pub action_budget: u16,
    pub undo_enabled: bool,
    pub candidate_goals: Vec<Goal>,
    /// Oracle-only. Must never appear on State / observations.
    pub hidden_goal_index: usize,
    pub split: Split,
    pub seed: u64,
    pub episode_id: u64,
}

impl Scenario {
    pub fn hidden_goal(&self) -> &Goal {
        &self.candidate_goals[self.hidden_goal_index]
    }

    pub fn in_bounds(&self, p: Pos) -> bool {
        p.x >= 0 && p.y >= 0 && (p.x as u8) < self.width && (p.y as u8) < self.height
    }

    pub fn is_wall(&self, p: Pos) -> bool {
        self.walls.contains(&p)
    }

    pub fn is_blocked(&self, p: Pos) -> bool {
        !self.in_bounds(p) || self.is_wall(p)
    }
}

/// Compact searchable agent state. Intentionally omits `hidden_goal_index`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct State {
    pub pos: Pos,
    /// Bit i set ⇒ collectible i still present.
    pub remaining_collectibles: u32,
    /// Bit i set ⇒ resource pickup i still available.
    pub remaining_pickups: u32,
    /// Ordered switch indices activated so far (append-only on step-on).
    pub switch_trace: Vec<u8>,
    pub resource: u8,
    /// Bit i set ⇒ hazard i has been entered at least once (Undo-restorable).
    pub touched_hazards: u32,
    /// Scored actions consumed; never decreased by Undo.
    pub actions_used: u16,
    /// Prior snapshots for Undo (fields excluding undo stack / actions_used).
    pub undo_stack: Vec<UndoFrame>,
}

/// Restorable slice of State used by Undo (excludes action counter and stack).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UndoFrame {
    pub pos: Pos,
    pub remaining_collectibles: u32,
    pub remaining_pickups: u32,
    pub switch_trace: Vec<u8>,
    pub resource: u8,
    pub touched_hazards: u32,
}

impl State {
    pub fn initial(scenario: &Scenario) -> Self {
        let n_c = scenario.collectibles.len().min(32) as u32;
        let n_p = scenario.resource_pickups.len().min(32) as u32;
        Self {
            pos: scenario.start,
            remaining_collectibles: if n_c == 0 { 0 } else { (1u32 << n_c) - 1 },
            remaining_pickups: if n_p == 0 { 0 } else { (1u32 << n_p) - 1 },
            switch_trace: Vec::new(),
            resource: scenario.initial_resource,
            touched_hazards: 0,
            actions_used: 0,
            undo_stack: Vec::new(),
        }
    }

    fn to_frame(&self) -> UndoFrame {
        UndoFrame {
            pos: self.pos,
            remaining_collectibles: self.remaining_collectibles,
            remaining_pickups: self.remaining_pickups,
            switch_trace: self.switch_trace.clone(),
            resource: self.resource,
            touched_hazards: self.touched_hazards,
        }
    }

    fn apply_frame(&mut self, frame: UndoFrame) {
        self.pos = frame.pos;
        self.remaining_collectibles = frame.remaining_collectibles;
        self.remaining_pickups = frame.remaining_pickups;
        self.switch_trace = frame.switch_trace;
        self.resource = frame.resource;
        self.touched_hazards = frame.touched_hazards;
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepOutcome {
    pub noop: bool,
    /// Hidden goal became true on this step.
    pub success: bool,
    /// Hidden avoid-goal was irreversibly failed on this step (or already failed).
    pub failed: bool,
    /// Budget exhausted without success.
    pub exhausted: bool,
}

/// Exact simulator with mutating step and pure transition for search.
#[derive(Clone, Debug)]
pub struct Simulator {
    scenario: Scenario,
    state: State,
    terminal: Option<TerminalState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TerminalState {
    Success,
    Failed,
    Exhausted,
}

impl Simulator {
    pub fn new(scenario: Scenario) -> Self {
        let state = State::initial(&scenario);
        let terminal = if goal_satisfied(&scenario, &state, scenario.hidden_goal()) {
            Some(TerminalState::Success)
        } else if scenario.action_budget == 0 {
            Some(TerminalState::Exhausted)
        } else {
            None
        };
        Self {
            scenario,
            state,
            terminal,
        }
    }

    pub fn scenario(&self) -> &Scenario {
        &self.scenario
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    /// Observation-facing state clone (no hidden index — State never carries it).
    pub fn observation(&self) -> State {
        self.state.clone()
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        if self.terminal.is_some() {
            Vec::new()
        } else {
            legal_actions(&self.scenario)
        }
    }

    pub fn step(&mut self, action: Action) -> StepOutcome {
        if let Some(terminal) = self.terminal {
            return StepOutcome {
                noop: true,
                success: terminal == TerminalState::Success,
                failed: terminal == TerminalState::Failed,
                exhausted: terminal == TerminalState::Exhausted,
            };
        }
        let (next, noop) = transition_inner(&self.scenario, &self.state, action);
        self.state = next;
        let success = goal_satisfied(&self.scenario, &self.state, self.scenario.hidden_goal());
        let failed = !success && terminal_failure(&self.scenario, &self.state);
        let exhausted =
            !success && !failed && self.state.actions_used >= self.scenario.action_budget;
        self.terminal = if success {
            Some(TerminalState::Success)
        } else if failed {
            Some(TerminalState::Failed)
        } else if exhausted {
            Some(TerminalState::Exhausted)
        } else {
            None
        };
        StepOutcome {
            noop,
            success,
            failed,
            exhausted,
        }
    }

    /// Pure successor map for search. Does not mutate `self`.
    pub fn transition(&self, state: &State, action: Action) -> State {
        transition_inner(&self.scenario, state, action).0
    }

    pub fn goal_satisfied(&self, state: &State, goal: &Goal) -> bool {
        goal_satisfied(&self.scenario, state, goal)
    }

    pub fn hidden_satisfied(&self) -> bool {
        goal_satisfied(&self.scenario, &self.state, self.scenario.hidden_goal())
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal.is_some()
    }

    pub fn terminal_outcome(&self) -> Option<StepOutcome> {
        self.terminal.map(|terminal| StepOutcome {
            noop: true,
            success: terminal == TerminalState::Success,
            failed: terminal == TerminalState::Failed,
            exhausted: terminal == TerminalState::Exhausted,
        })
    }
}

pub fn legal_actions(scenario: &Scenario) -> Vec<Action> {
    let mut out = Action::moves().to_vec();
    if scenario.undo_enabled {
        out.push(Action::Undo);
    }
    out
}

/// True when the hidden avoid-goal has been irreversibly violated.
pub fn terminal_failure(scenario: &Scenario, state: &State) -> bool {
    goal_terminal_failure(scenario, state, scenario.hidden_goal())
}

/// Whether `goal` would have terminated in failure at `state`.
///
/// This is public candidate semantics, not access to the hidden goal. P1A uses it
/// to falsify an avoid hypothesis when the named hazard is touched but the real
/// environment does not terminate.
pub fn goal_terminal_failure(scenario: &Scenario, state: &State, goal: &Goal) -> bool {
    match goal {
        Goal::AvoidHazardReachMarker { hazard, .. } => {
            let h = *hazard as usize;
            scenario.hazards.get(h).is_some()
                && h < 32
                && (state.touched_hazards & (1u32 << h)) != 0
        }
        _ => false,
    }
}

pub fn goal_satisfied(scenario: &Scenario, state: &State, goal: &Goal) -> bool {
    match goal {
        Goal::ReachMarker { marker } => {
            let Some(m) = scenario.markers.get(*marker as usize) else {
                return false;
            };
            state.pos == *m
        }
        Goal::CollectAll => state.remaining_collectibles == 0 && !scenario.collectibles.is_empty(),
        Goal::ActivateSwitchesInOrder { order } => {
            if order.is_empty() || scenario.switches.is_empty() {
                return false;
            }
            state.switch_trace.len() >= order.len()
                && state.switch_trace[..order.len()] == order[..]
        }
        Goal::PreserveResourceReachMarker {
            marker,
            min_resource,
        } => {
            let Some(m) = scenario.markers.get(*marker as usize) else {
                return false;
            };
            state.pos == *m && state.resource >= *min_resource
        }
        Goal::AvoidHazardReachMarker { hazard, marker } => {
            let Some(m) = scenario.markers.get(*marker as usize) else {
                return false;
            };
            if scenario.hazards.get(*hazard as usize).is_none() {
                return false;
            }
            let h = *hazard as usize;
            let clean = h >= 32 || (state.touched_hazards & (1u32 << h)) == 0;
            state.pos == *m && clean
        }
        Goal::TriggerTerminal { trigger } => {
            let Some(t) = scenario.terminal_triggers.get(*trigger as usize) else {
                return false;
            };
            state.pos == *t
        }
    }
}

fn transition_inner(scenario: &Scenario, state: &State, action: Action) -> (State, bool) {
    let mut next = state.clone();
    // Every exposed action consumes one scored step, including no-ops and Undo.
    next.actions_used = state.actions_used.saturating_add(1);

    match action {
        Action::Undo => {
            if !scenario.undo_enabled || state.undo_stack.is_empty() {
                // Deterministic no-op Undo: still costs an action; stack unchanged.
                return (next, true);
            }
            let frame = next.undo_stack.pop().expect("non-empty");
            next.apply_frame(frame);
            (next, false)
        }
        Action::Move(dir) => {
            let before = state.to_frame();
            let (dx, dy) = dir.delta();
            let Some(dest) = state.pos.checked_add(dx, dy) else {
                next.undo_stack.push(before);
                return (next, true);
            };
            if scenario.is_blocked(dest) {
                next.undo_stack.push(before);
                return (next, true);
            }

            next.undo_stack.push(before);
            next.pos = dest;

            // Collectibles.
            if let Some(i) = scenario.collectibles.iter().position(|p| *p == dest) {
                if i < 32 {
                    next.remaining_collectibles &= !(1u32 << i);
                }
            }

            // Switches: record first activation only (stable, bounded trace).
            if let Some(i) = scenario.switches.iter().position(|p| *p == dest) {
                let idx = i as u8;
                if !next.switch_trace.contains(&idx) {
                    next.switch_trace.push(idx);
                }
            }

            // Resource pickups (once).
            if let Some(i) = scenario.resource_pickups.iter().position(|p| *p == dest) {
                if i < 32 && (next.remaining_pickups & (1u32 << i)) != 0 {
                    next.remaining_pickups &= !(1u32 << i);
                    next.resource = next.resource.saturating_add(1);
                }
            }

            // Hazards: drain resource and mark as touched (indexed).
            if let Some(i) = scenario.hazards.iter().position(|p| *p == dest) {
                next.resource = next.resource.saturating_sub(1);
                if i < 32 {
                    next.touched_hazards |= 1u32 << i;
                }
            }

            (next, false)
        }
    }
}

/// Bounded BFS reachability for a specific goal under the exact transition.
/// Returns true if some path of length ≤ `limit` satisfies the goal.
pub fn reachable_within(scenario: &Scenario, goal: &Goal, limit: u16) -> bool {
    let start = State::initial(scenario);
    if goal_satisfied(scenario, &start, goal) {
        return true;
    }
    // Move-only expansion keeps the frontier searchable; Undo is optional sugar.
    let actions: Vec<Action> = legal_actions(scenario)
        .into_iter()
        .filter(|a| !matches!(a, Action::Undo))
        .collect();
    let mut q = VecDeque::new();
    let mut seen = BTreeSet::new();
    let key = |s: &State| {
        (
            s.pos,
            s.remaining_collectibles,
            s.remaining_pickups,
            s.switch_trace.clone(),
            s.resource,
            s.touched_hazards,
        )
    };
    seen.insert(key(&start));
    q.push_back(start);
    while let Some(cur) = q.pop_front() {
        if cur.actions_used >= limit {
            continue;
        }
        for &a in &actions {
            let (nxt, _) = transition_inner(scenario, &cur, a);
            if goal_satisfied(scenario, &nxt, goal) {
                return true;
            }
            if nxt.actions_used > limit {
                continue;
            }
            if seen.insert(key(&nxt)) {
                q.push_back(nxt);
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn tiny_scenario() -> Scenario {
        Scenario {
            width: 5,
            height: 5,
            walls: BTreeSet::from([Pos::new(2, 2)]),
            markers: vec![Pos::new(4, 4), Pos::new(0, 4), Pos::new(4, 0)],
            collectibles: vec![Pos::new(1, 0), Pos::new(3, 0)],
            switches: vec![Pos::new(0, 1), Pos::new(1, 1), Pos::new(2, 1)],
            hazards: vec![Pos::new(3, 3)],
            resource_pickups: vec![Pos::new(0, 3)],
            terminal_triggers: vec![Pos::new(4, 2), Pos::new(2, 4)],
            start: Pos::new(0, 0),
            initial_resource: 2,
            action_budget: 40,
            undo_enabled: true,
            candidate_goals: vec![
                Goal::ReachMarker { marker: 0 },
                Goal::CollectAll,
                Goal::ActivateSwitchesInOrder {
                    order: vec![0, 1, 2],
                },
                Goal::PreserveResourceReachMarker {
                    marker: 0,
                    min_resource: 2,
                },
                Goal::AvoidHazardReachMarker {
                    hazard: 0,
                    marker: 0,
                },
                Goal::TriggerTerminal { trigger: 0 },
            ],
            hidden_goal_index: 0,
            split: Split::Train,
            seed: 1,
            episode_id: 0,
        }
    }

    #[test]
    fn all_goal_predicates() {
        let sc = tiny_scenario();
        let mut st = State::initial(&sc);

        assert!(!goal_satisfied(&sc, &st, &Goal::ReachMarker { marker: 0 }));
        st.pos = Pos::new(4, 4);
        assert!(goal_satisfied(&sc, &st, &Goal::ReachMarker { marker: 0 }));

        st = State::initial(&sc);
        assert!(!goal_satisfied(&sc, &st, &Goal::CollectAll));
        st.remaining_collectibles = 0;
        assert!(goal_satisfied(&sc, &st, &Goal::CollectAll));

        st = State::initial(&sc);
        let order = Goal::ActivateSwitchesInOrder {
            order: vec![0, 1, 2],
        };
        assert!(!goal_satisfied(&sc, &st, &order));
        st.switch_trace = vec![0, 1, 2];
        assert!(goal_satisfied(&sc, &st, &order));
        st.switch_trace = vec![0, 2, 1];
        assert!(!goal_satisfied(&sc, &st, &order));

        st = State::initial(&sc);
        st.pos = Pos::new(4, 4);
        st.resource = 2;
        assert!(goal_satisfied(
            &sc,
            &st,
            &Goal::PreserveResourceReachMarker {
                marker: 0,
                min_resource: 2
            }
        ));
        st.resource = 1;
        assert!(!goal_satisfied(
            &sc,
            &st,
            &Goal::PreserveResourceReachMarker {
                marker: 0,
                min_resource: 2
            }
        ));

        st = State::initial(&sc);
        st.pos = Pos::new(4, 4);
        assert!(goal_satisfied(
            &sc,
            &st,
            &Goal::AvoidHazardReachMarker {
                hazard: 0,
                marker: 0
            }
        ));
        st.touched_hazards = 1;
        assert!(!goal_satisfied(
            &sc,
            &st,
            &Goal::AvoidHazardReachMarker {
                hazard: 0,
                marker: 0
            }
        ));

        st = State::initial(&sc);
        st.pos = Pos::new(4, 2);
        assert!(goal_satisfied(
            &sc,
            &st,
            &Goal::TriggerTerminal { trigger: 0 }
        ));
        st.pos = Pos::new(2, 4);
        assert!(goal_satisfied(
            &sc,
            &st,
            &Goal::TriggerTerminal { trigger: 1 }
        ));
        assert!(!goal_satisfied(
            &sc,
            &st,
            &Goal::TriggerTerminal { trigger: 0 }
        ));
    }

    #[test]
    fn avoid_hazard_terminal_failure_when_hidden() {
        let mut sc = tiny_scenario();
        sc.hidden_goal_index = 4; // AvoidHazardReachMarker { hazard: 0, marker: 0 }
        let mut sim = Simulator::new(sc);
        // Walk toward hazard at (3,3): (0,0)->(1,0)->(2,0)->(3,0)->(3,1)->(3,2)->(3,3)
        for a in [
            Action::Move(Dir::East),
            Action::Move(Dir::East),
            Action::Move(Dir::East),
            Action::Move(Dir::South),
            Action::Move(Dir::South),
            Action::Move(Dir::South),
        ] {
            let out = sim.step(a);
            if sim.state().pos == Pos::new(3, 3) {
                assert!(out.failed, "touching named hazard must fail episode");
                assert!(!out.success);
                assert_eq!(sim.state().touched_hazards & 1, 1);
                assert!(sim.is_terminal());
                let state_at_failure = sim.state().clone();
                let repeated = sim.step(Action::Undo);
                assert!(repeated.failed, "failure must remain terminal");
                assert!(repeated.noop);
                assert_eq!(sim.state(), &state_at_failure, "Undo cannot revive failure");
                return;
            }
            assert!(!out.failed);
        }
        panic!("never reached hazard");
    }

    #[test]
    fn touching_hazard_not_failed_when_hidden_is_other_goal() {
        let mut sc = tiny_scenario();
        sc.hidden_goal_index = 0; // ReachMarker
        let mut sim = Simulator::new(sc);
        for a in [
            Action::Move(Dir::East),
            Action::Move(Dir::East),
            Action::Move(Dir::East),
            Action::Move(Dir::South),
            Action::Move(Dir::South),
            Action::Move(Dir::South),
        ] {
            let out = sim.step(a);
            if sim.state().pos == Pos::new(3, 3) {
                assert!(!out.failed);
                assert_eq!(sim.state().touched_hazards & 1, 1);
                assert!(goal_terminal_failure(
                    sim.scenario(),
                    sim.state(),
                    &Goal::AvoidHazardReachMarker {
                        hazard: 0,
                        marker: 0,
                    }
                ));
                return;
            }
        }
        panic!("never reached hazard");
    }

    #[test]
    fn undo_restores_touched_hazards() {
        let sc = tiny_scenario();
        let mut sim = Simulator::new(sc);
        // Path to hazard.
        for a in [
            Action::Move(Dir::East),
            Action::Move(Dir::East),
            Action::Move(Dir::East),
            Action::Move(Dir::South),
            Action::Move(Dir::South),
            Action::Move(Dir::South),
        ] {
            sim.step(a);
        }
        assert_eq!(sim.state().pos, Pos::new(3, 3));
        assert_eq!(sim.state().touched_hazards & 1, 1);
        let used = sim.state().actions_used;
        let out = sim.step(Action::Undo);
        assert!(!out.noop);
        assert_eq!(sim.state().pos, Pos::new(3, 2));
        assert_eq!(sim.state().touched_hazards & 1, 0);
        assert_eq!(sim.state().actions_used, used + 1);
    }

    #[test]
    fn noop_still_consumes_action() {
        let sc = tiny_scenario();
        let mut sim = Simulator::new(sc);
        // West from (0,0) is OOB ⇒ no-op.
        let before = sim.state().clone();
        let out = sim.step(Action::Move(Dir::West));
        assert!(out.noop);
        assert!(!out.failed);
        assert_eq!(sim.state().pos, before.pos);
        assert_eq!(sim.state().actions_used, before.actions_used + 1);
        let mut st = State::initial(sim.scenario());
        st.pos = Pos::new(2, 1);
        let (nxt, noop) = transition_inner(sim.scenario(), &st, Action::Move(Dir::South));
        assert!(noop);
        assert_eq!(nxt.pos, st.pos);
        assert_eq!(nxt.actions_used, st.actions_used + 1);
    }

    #[test]
    fn undo_accounting_costs_two_actions() {
        let sc = tiny_scenario();
        let mut sim = Simulator::new(sc);
        let start = sim.state().pos;
        let o1 = sim.step(Action::Move(Dir::East));
        assert!(!o1.noop);
        assert_eq!(sim.state().pos, Pos::new(1, 0));
        assert_eq!(sim.state().actions_used, 1);
        let o2 = sim.step(Action::Undo);
        assert!(!o2.noop);
        assert_eq!(sim.state().pos, start);
        assert_eq!(sim.state().actions_used, 2);
        // Undo cannot erase the counter.
        assert!(sim.state().actions_used > 0);
    }

    #[test]
    fn undo_reverts_only_the_immediately_previous_action() {
        let sc = tiny_scenario();
        let mut sim = Simulator::new(sc);
        sim.step(Action::Move(Dir::East));
        assert_eq!(sim.state().pos, Pos::new(1, 0));

        let blocked = sim.step(Action::Move(Dir::North));
        assert!(blocked.noop);
        let undone = sim.step(Action::Undo);
        assert!(!undone.noop);

        assert_eq!(sim.state().pos, Pos::new(1, 0));
        assert_eq!(sim.state().actions_used, 3);
    }

    #[test]
    fn pure_transition_matches_mutating_step() {
        let sc = tiny_scenario();
        let mut sim = Simulator::new(sc.clone());
        let actions = [
            Action::Move(Dir::East),
            Action::Move(Dir::South),
            Action::Undo,
            Action::Move(Dir::South),
            Action::Move(Dir::West),
        ];
        let mut pure = State::initial(&sc);
        for a in actions {
            let predicted = {
                let tmp = Simulator::new(sc.clone());
                tmp.transition(&pure, a)
            };
            let _ = sim.step(a);
            pure = predicted;
            assert_eq!(sim.state(), &pure);
        }
    }

    #[test]
    fn state_serialization_omits_hidden_index() {
        let sc = tiny_scenario();
        let st = State::initial(&sc);
        let json = serde_json::to_string(&st).unwrap();
        assert!(!json.contains("hidden"));
        assert!(!json.contains("hidden_goal_index"));
        // Scenario may carry it; State must not deserialize such a field as meaningful.
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("hidden_goal_index").is_none());
        assert!(v.get("touched_hazards").is_some());
    }

    #[test]
    fn episode_success_only_on_hidden_goal() {
        let mut sc = tiny_scenario();
        sc.hidden_goal_index = 1; // CollectAll
        let mut sim = Simulator::new(sc);
        // Reach marker 0 without collecting — not success for hidden CollectAll.
        sim.state.pos = Pos::new(4, 4);
        sim.state.actions_used = 5;
        assert!(!sim.hidden_satisfied());
        assert!(sim.goal_satisfied(sim.state(), &Goal::ReachMarker { marker: 0 }));
        sim.state.remaining_collectibles = 0;
        assert!(sim.hidden_satisfied());
    }

    #[test]
    fn trigger_terminal_success_on_indexed_pad() {
        let mut sc = tiny_scenario();
        sc.hidden_goal_index = 5; // TriggerTerminal { trigger: 0 }
        let mut sim = Simulator::new(sc);
        // Navigate toward (4,2).
        let path = [
            Action::Move(Dir::East),
            Action::Move(Dir::East),
            Action::Move(Dir::East),
            Action::Move(Dir::East),
            Action::Move(Dir::South),
            Action::Move(Dir::South),
        ];
        let mut done = false;
        for a in path {
            let out = sim.step(a);
            if out.success {
                assert_eq!(sim.state().pos, Pos::new(4, 2));
                assert!(!out.failed);
                done = true;
                break;
            }
            assert!(!out.failed);
        }
        assert!(done);
    }
}
