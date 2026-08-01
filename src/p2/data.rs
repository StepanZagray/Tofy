//! ARC-compatible frames, actions, goal features, and exact-simulator transitions.
//!
//! Candidate-conditioned labels are always computed against a supplied public
//! [`Goal`], never `Scenario.hidden_goal_index`.

use crate::domain::{
    goal_family, goal_satisfied, goal_terminal_failure, legal_actions, Action, Dir, Goal, Pos,
    Scenario, Simulator, Split, State,
};
use crate::generator::{
    generate, generate_p1c, generate_p1c_hard_candidate, p1c_falsification_probe_width, rng_for,
};
use crate::search::shortest_path;
use anyhow::{anyhow, bail, ensure, Result};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Official ARC-AGI-3 frame side length.
pub const FRAME_SIDE: usize = 64;

/// Fixed length of [`GoalFeatures::values`].
pub const GOAL_FEATURES_DIM: usize = 19;

/// Maximum switch-order slots packed into goal features.
pub const GOAL_ORDER_SLOTS: usize = 8;

/// Stable categorical palette for synthetic Tofy renders (values in `0..16`).
pub mod palette {
    pub const EMPTY: u8 = 0;
    pub const WALL: u8 = 1;
    pub const AGENT: u8 = 2;
    pub const MARKER_BASE: u8 = 3;
    pub const COLLECTIBLE: u8 = 6;
    pub const SWITCH_BASE: u8 = 7;
    pub const HAZARD_BASE: u8 = 10;
    pub const PICKUP: u8 = 12;
    pub const TRIGGER_BASE: u8 = 13;
    pub const PAD: u8 = 0;
}

/// Discrete ARC-like frame: row-major `width * height` categorical pixels in `0..=15`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArcFrame {
    pub width: u16,
    pub height: u16,
    pub pixels: Vec<u8>,
}

impl ArcFrame {
    pub fn new(width: u16, height: u16, pixels: Vec<u8>) -> Result<Self> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| anyhow!("frame dimensions overflow"))?;
        ensure!(
            pixels.len() == expected,
            "pixel length {} != width*height {}",
            pixels.len(),
            expected
        );
        for (i, &p) in pixels.iter().enumerate() {
            ensure!(p <= 15, "palette value {p} out of 0..=15 at index {i}");
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn pixel(&self, x: u16, y: u16) -> Option<u8> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.pixels
            .get(y as usize * self.width as usize + x as usize)
            .copied()
    }

    /// Copy pixels 1:1 into the top-left of a `64x64` canvas and pad with
    /// [`palette::PAD`]. Larger frames are rejected (no interpolation / no crop
    /// ambiguity for training).
    pub fn to_fixed_64(&self) -> Result<Self> {
        ensure!(
            self.width as usize <= FRAME_SIDE && self.height as usize <= FRAME_SIDE,
            "cannot pad frame {}x{} into {}x{} without interpolation/crop",
            self.width,
            self.height,
            FRAME_SIDE,
            FRAME_SIDE
        );
        let mut pixels = vec![palette::PAD; FRAME_SIDE * FRAME_SIDE];
        for y in 0..self.height as usize {
            for x in 0..self.width as usize {
                let src = self.pixels[y * self.width as usize + x];
                pixels[y * FRAME_SIDE + x] = src;
            }
        }
        Self::new(FRAME_SIDE as u16, FRAME_SIDE as u16, pixels)
    }
}

/// Official-like discrete action: ids `1..=6`. Coordinates allowed only for id 6.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArcAction {
    pub id: u8,
    pub x: Option<u8>,
    pub y: Option<u8>,
}

impl ArcAction {
    pub fn new(id: u8, x: Option<u8>, y: Option<u8>) -> Result<Self> {
        ensure!((1..=6).contains(&id), "action id {id} not in 1..=6");
        match id {
            6 => {
                ensure!(
                    x.is_some() && y.is_some(),
                    "ACTION6 requires x and y coordinates"
                );
                let x = x.expect("checked");
                let y = y.expect("checked");
                ensure!(x < 64 && y < 64, "ACTION6 coordinates must be in 0..64");
                Ok(Self {
                    id,
                    x: Some(x),
                    y: Some(y),
                })
            }
            _ => {
                ensure!(
                    x.is_none() && y.is_none(),
                    "coordinates only allowed for ACTION6"
                );
                Ok(Self {
                    id,
                    x: None,
                    y: None,
                })
            }
        }
    }

    pub fn from_tofy(action: Action) -> Self {
        let id = match action {
            Action::Move(Dir::North) => 1,
            Action::Move(Dir::South) => 2,
            Action::Move(Dir::East) => 3,
            Action::Move(Dir::West) => 4,
            Action::Undo => 5,
        };
        Self {
            id,
            x: None,
            y: None,
        }
    }

    pub fn to_tofy(&self) -> Result<Action> {
        match self.id {
            1 => Ok(Action::Move(Dir::North)),
            2 => Ok(Action::Move(Dir::South)),
            3 => Ok(Action::Move(Dir::East)),
            4 => Ok(Action::Move(Dir::West)),
            5 => Ok(Action::Undo),
            6 => bail!("ACTION6 has no Tofy Action mapping"),
            other => bail!("invalid action id {other}"),
        }
    }
}

/// Fixed-length public goal encoding. Never includes `hidden_goal_index`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GoalFeatures {
    pub values: [f32; GOAL_FEATURES_DIM],
}

impl GoalFeatures {
    pub fn zeros() -> Self {
        Self {
            values: [0.0; GOAL_FEATURES_DIM],
        }
    }

    pub fn encode(goal: &Goal) -> Self {
        let mut values = [0.0f32; GOAL_FEATURES_DIM];
        let family = family_index(goal);
        values[family as usize] = 1.0;
        match goal {
            Goal::ReachMarker { marker } => {
                values[6] = f32::from(*marker);
            }
            Goal::CollectAll => {}
            Goal::ActivateSwitchesInOrder { order } => {
                values[10] = order.len() as f32;
                for (i, &idx) in order.iter().take(GOAL_ORDER_SLOTS).enumerate() {
                    values[11 + i] = f32::from(idx);
                }
            }
            Goal::PreserveResourceReachMarker {
                marker,
                min_resource,
            } => {
                values[6] = f32::from(*marker);
                values[7] = f32::from(*min_resource);
            }
            Goal::AvoidHazardReachMarker { hazard, marker } => {
                values[6] = f32::from(*marker);
                values[8] = f32::from(*hazard);
            }
            Goal::TriggerTerminal { trigger } => {
                values[9] = f32::from(*trigger);
            }
        }
        Self { values }
    }
}

fn family_index(goal: &Goal) -> u8 {
    match goal {
        Goal::ReachMarker { .. } => 0,
        Goal::CollectAll => 1,
        Goal::ActivateSwitchesInOrder { .. } => 2,
        Goal::PreserveResourceReachMarker { .. } => 3,
        Goal::AvoidHazardReachMarker { .. } => 4,
        Goal::TriggerTerminal { .. } => 5,
    }
}

/// One supervised transition for world-model / event-head training.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionSample {
    pub current: ArcFrame,
    pub next: ArcFrame,
    pub action: ArcAction,
    pub goal_features: GoalFeatures,
    pub noop: Option<bool>,
    /// Whether `next` satisfies the supplied public candidate goal.
    pub goal_satisfied: Option<bool>,
    /// Whether `next` is a terminal failure under that same candidate.
    pub goal_failed: Option<bool>,
    /// Whether the action budget is exhausted at `next` without satisfaction/failure.
    pub exhausted: Option<bool>,
    pub split: Split,
    pub family: String,
    pub seed: u64,
    pub episode_id: u64,
}

/// Render `Scenario` layout + public `State` as a discrete grid (no pad).
///
/// Stable palette semantics; agent draws last. Never encodes
/// `hidden_goal_index` or candidate-goal identity.
pub fn render_state(scenario: &Scenario, state: &State) -> Result<ArcFrame> {
    let w = scenario.width as usize;
    let h = scenario.height as usize;
    let mut pixels = vec![palette::EMPTY; w * h];

    let put = |pixels: &mut [u8], p: Pos, color: u8| {
        if p.x >= 0 && p.y >= 0 && (p.x as u8) < scenario.width && (p.y as u8) < scenario.height {
            pixels[p.y as usize * w + p.x as usize] = color;
        }
    };

    for &p in &scenario.walls {
        put(&mut pixels, p, palette::WALL);
    }
    for (i, &p) in scenario.markers.iter().enumerate() {
        put(&mut pixels, p, palette::MARKER_BASE + i.min(2) as u8);
    }
    for (i, &p) in scenario.collectibles.iter().enumerate() {
        if i < 32 && (state.remaining_collectibles & (1u32 << i)) != 0 {
            put(&mut pixels, p, palette::COLLECTIBLE);
        }
    }
    for (i, &p) in scenario.switches.iter().enumerate() {
        let done = state.switch_trace.iter().any(|&t| t as usize == i);
        if !done {
            put(&mut pixels, p, palette::SWITCH_BASE + i.min(2) as u8);
        }
    }
    for (i, &p) in scenario.hazards.iter().enumerate() {
        let touched = i < 32 && (state.touched_hazards & (1u32 << i)) != 0;
        if !touched {
            put(&mut pixels, p, palette::HAZARD_BASE + i.min(1) as u8);
        }
    }
    for (i, &p) in scenario.resource_pickups.iter().enumerate() {
        if i < 32 && (state.remaining_pickups & (1u32 << i)) != 0 {
            put(&mut pixels, p, palette::PICKUP);
        }
    }
    for (i, &p) in scenario.terminal_triggers.iter().enumerate() {
        put(&mut pixels, p, palette::TRIGGER_BASE + i.min(2) as u8);
    }
    put(&mut pixels, state.pos, palette::AGENT);

    ArcFrame::new(scenario.width as u16, scenario.height as u16, pixels)
}

/// Render and pad to the official `64x64` observation size.
pub fn render_state_fixed(scenario: &Scenario, state: &State) -> Result<ArcFrame> {
    let native = render_state(scenario, state)?;
    let play_height = FRAME_SIDE - 2;
    let cell = (FRAME_SIDE / native.width as usize)
        .min(play_height / native.height as usize)
        .max(1);
    let used_width = native.width as usize * cell;
    let used_height = native.height as usize * cell;
    let origin_x = (FRAME_SIDE - used_width) / 2;
    let origin_y = (play_height - used_height) / 2;
    let mut frame = ArcFrame::new(
        FRAME_SIDE as u16,
        FRAME_SIDE as u16,
        vec![palette::PAD; FRAME_SIDE * FRAME_SIDE],
    )?;
    for y in 0..native.height as usize {
        for x in 0..native.width as usize {
            let color = native.pixels[y * native.width as usize + x];
            for dy in 0..cell {
                for dx in 0..cell {
                    let px = origin_x + x * cell + dx;
                    let py = origin_y + y * cell + dy;
                    frame.pixels[py * FRAME_SIDE + px] = color;
                }
            }
        }
    }
    // Preserve public counters as a compact visual HUD rather than feeding
    // privileged structured state to the pixel model.
    let resource = usize::from(state.resource).min(FRAME_SIDE);
    for x in 0..resource {
        frame.pixels[(FRAME_SIDE - 1) * FRAME_SIDE + x] = palette::PICKUP;
    }
    let progress = if scenario.action_budget == 0 {
        0
    } else {
        usize::from(state.actions_used).saturating_mul(FRAME_SIDE)
            / usize::from(scenario.action_budget)
    }
    .min(FRAME_SIDE);
    for x in 0..progress {
        frame.pixels[(FRAME_SIDE - 2) * FRAME_SIDE + x] = palette::WALL;
    }
    Ok(frame)
}

/// Whether `action` is a deterministic no-op under exact transition rules.
pub fn action_is_noop(scenario: &Scenario, before: &State, action: Action) -> bool {
    match action {
        Action::Undo => !scenario.undo_enabled || before.undo_stack.is_empty(),
        Action::Move(dir) => {
            let (dx, dy) = dir.delta();
            match before.pos.checked_add(dx, dy) {
                None => true,
                Some(dest) => scenario.is_blocked(dest),
            }
        }
    }
}

/// Build one candidate-conditioned sample from an exact `(before, action, after)`.
pub fn sample_from_transition(
    scenario: &Scenario,
    before: &State,
    after: &State,
    action: Action,
    goal: &Goal,
) -> Result<TransitionSample> {
    let current = render_state_fixed(scenario, before)?;
    let next = render_state_fixed(scenario, after)?;
    let goal_satisfied = goal_satisfied(scenario, after, goal);
    let goal_failed = goal_terminal_failure(scenario, after, goal);
    let exhausted = !goal_satisfied && !goal_failed && after.actions_used >= scenario.action_budget;
    Ok(TransitionSample {
        current,
        next,
        action: ArcAction::from_tofy(action),
        goal_features: GoalFeatures::encode(goal),
        noop: Some(action_is_noop(scenario, before, action)),
        goal_satisfied: Some(goal_satisfied),
        goal_failed: Some(goal_failed),
        exhausted: Some(exhausted),
        split: scenario.split,
        family: goal_family(goal).to_string(),
        seed: scenario.seed,
        episode_id: scenario.episode_id,
    })
}

fn apply_action(sim: &Simulator, state: &State, action: Action) -> State {
    sim.transition(state, action)
}

/// Random legal one-step transitions under a public goal.
pub fn generate_random_one_step(
    seed: u64,
    episode_id: u64,
    split: Split,
    n: usize,
) -> Result<Vec<TransitionSample>> {
    let scenario = generate(seed, episode_id, split);
    let public_goal =
        scenario.candidate_goals[(episode_id as usize) % scenario.candidate_goals.len()].clone();
    let sim = Simulator::new(scenario.clone());
    let mut rng = rng_for(seed ^ 0xA11C_E001, episode_id, split);
    let mut out = Vec::with_capacity(n);
    let mut state = State::initial(&scenario);
    for _ in 0..n {
        let actions = legal_actions(&scenario);
        ensure!(!actions.is_empty(), "no legal actions");
        let action = *actions.choose(&mut rng).expect("non-empty");
        let next = apply_action(&sim, &state, action);
        out.push(sample_from_transition(
            &scenario,
            &state,
            &next,
            action,
            &public_goal,
        )?);
        state = next;
        if state.actions_used >= scenario.action_budget {
            state = State::initial(&scenario);
        }
    }
    Ok(out)
}

/// ARC-style coordinate actions: ACTION6 moves the visible agent to the selected
/// public cell. This trains coordinate conditioning without using ARC recordings.
pub fn generate_coordinate_one_step(
    seed: u64,
    episode_id: u64,
    split: Split,
    n: usize,
) -> Result<Vec<TransitionSample>> {
    let mut rng = rng_for(seed ^ 0xA11C_C006, episode_id, split);
    let mut out = Vec::with_capacity(n);
    for step in 0..n {
        let start_x = 31usize;
        let start_y = 31usize;
        let (x, y) = loop {
            let candidate = (
                rng.random_range(0..FRAME_SIDE) as u8,
                rng.random_range(0..FRAME_SIDE) as u8,
            );
            if candidate != (start_x as u8, start_y as u8) {
                break candidate;
            }
        };
        let mut current_pixels = vec![palette::EMPTY; FRAME_SIDE * FRAME_SIDE];
        current_pixels[start_y * FRAME_SIDE + start_x] = palette::AGENT;
        current_pixels[y as usize * FRAME_SIDE + x as usize] = palette::MARKER_BASE;
        let mut next_pixels = current_pixels.clone();
        next_pixels[start_y * FRAME_SIDE + start_x] = palette::EMPTY;
        next_pixels[y as usize * FRAME_SIDE + x as usize] = palette::AGENT;
        out.push(TransitionSample {
            current: ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, current_pixels)?,
            next: ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, next_pixels)?,
            action: ArcAction::new(6, Some(x), Some(y))?,
            goal_features: GoalFeatures::zeros(),
            noop: Some(false),
            goal_satisfied: None,
            goal_failed: None,
            exhausted: Some(false),
            split,
            family: "coordinate_action".into(),
            seed,
            episode_id: episode_id.wrapping_mul(1_000_003).wrapping_add(step as u64),
        });
    }
    Ok(out)
}

fn interleave<T>(left: Vec<T>, right: Vec<T>) -> Vec<T> {
    let mut left = left.into_iter();
    let mut right = right.into_iter();
    let mut out = Vec::new();
    loop {
        match (left.next(), right.next()) {
            (None, None) => break,
            (l, r) => {
                out.extend(l);
                out.extend(r);
            }
        }
    }
    out
}

/// Exact shortest-path fragments for a public goal (sequential teacher forcing).
pub fn generate_plan_fragments(
    seed: u64,
    episode_id: u64,
    split: Split,
    max_actions: u16,
) -> Result<Vec<TransitionSample>> {
    let scenario = generate(seed, episode_id, split);
    let sim = Simulator::new(scenario.clone());
    let start = State::initial(&scenario);
    let offset = (episode_id as usize) % scenario.candidate_goals.len();
    let (goal, plan) = scenario
        .candidate_goals
        .iter()
        .cycle()
        .skip(offset)
        .take(scenario.candidate_goals.len())
        .find_map(|goal| shortest_path(&sim, &start, goal, max_actions).map(|plan| (goal, plan)))
        .ok_or_else(|| anyhow!("no public-candidate plan for seed={seed} episode={episode_id}"))?;
    let mut state = start;
    let mut out = Vec::with_capacity(plan.actions.len());
    for action in plan.actions {
        let next = apply_action(&sim, &state, action);
        out.push(sample_from_transition(
            &scenario, &state, &next, action, goal,
        )?);
        state = next;
    }
    Ok(out)
}

/// P1C safe multi-goal falsification probe: one-step (or short path) with labels
/// for every exact-live candidate against the same public transition.
pub fn generate_p1c_falsification_episode(
    seed: u64,
    episode_id: u64,
    split: Split,
) -> Result<Vec<TransitionSample>> {
    let scenario = generate_p1c(seed, episode_id, split);
    ensure!(
        p1c_falsification_probe_width(&scenario) >= 2,
        "P1C scenario lacks safe falsification probe"
    );
    let sim = Simulator::new(scenario.clone());
    let start = State::initial(&scenario);
    // South is the cheap multi-goal probe on the false-lead stem.
    let action = Action::Move(Dir::South);
    let next = apply_action(&sim, &start, action);
    let mut out = Vec::new();
    for goal in &scenario.candidate_goals {
        out.push(sample_from_transition(
            &scenario, &start, &next, action, goal,
        )?);
    }
    Ok(out)
}

/// P1C-hard: one public-goal fragment followed by a different viable goal.
pub fn generate_p1c_hard_retarget_multistep(
    seed: u64,
    source_episode_id: u64,
    split: Split,
    wrong_steps: usize,
) -> Result<Vec<TransitionSample>> {
    let scenario = generate_p1c_hard_candidate(seed, source_episode_id, split);
    let sim = Simulator::new(scenario.clone());
    let start = State::initial(&scenario);

    // Both commitments come from public candidate order. The oracle-only hidden
    // index is irrelevant to this world-model lesson.
    let wrong_goal = scenario
        .candidate_goals
        .iter()
        .cycle()
        .skip((source_episode_id as usize) % scenario.candidate_goals.len())
        .take(scenario.candidate_goals.len())
        .find(|goal| shortest_path(&sim, &start, goal, scenario.action_budget).is_some())
        .cloned()
        .ok_or_else(|| anyhow!("need a distractor candidate"))?;

    let mut state = start;
    let mut out = Vec::new();

    if let Some(wrong_plan) = shortest_path(&sim, &state, &wrong_goal, scenario.action_budget) {
        for action in wrong_plan.actions.into_iter().take(wrong_steps) {
            let next = apply_action(&sim, &state, action);
            // Labels stay candidate-conditioned on the wrong commitment.
            out.push(sample_from_transition(
                &scenario,
                &state,
                &next,
                action,
                &wrong_goal,
            )?);
            state = next;
            if goal_terminal_failure(&scenario, &state, &wrong_goal)
                || goal_satisfied(&scenario, &state, &wrong_goal)
            {
                break;
            }
        }
    }

    let retarget = scenario
        .candidate_goals
        .iter()
        .filter(|goal| *goal != &wrong_goal)
        .find_map(|goal| {
            shortest_path(&sim, &state, goal, scenario.action_budget)
                .map(|plan| (goal.clone(), plan))
        });
    if let Some((retarget_goal, true_plan)) = retarget {
        for action in true_plan.actions {
            let next = apply_action(&sim, &state, action);
            out.push(sample_from_transition(
                &scenario,
                &state,
                &next,
                action,
                &retarget_goal,
            )?);
            state = next;
            if goal_satisfied(&scenario, &state, &retarget_goal) {
                break;
            }
        }
    }

    ensure!(!out.is_empty(), "hard retarget produced no transitions");
    Ok(out)
}

/// Deterministic curriculum batch keyed by `(kind, seed, episode_id, split)`.
pub fn generate_curriculum(
    kind: &str,
    seed: u64,
    episode_id: u64,
    split: Split,
) -> Result<Vec<TransitionSample>> {
    match kind {
        "random_one_step" => Ok(interleave(
            generate_random_one_step(seed, episode_id, split, 4)?,
            generate_coordinate_one_step(seed, episode_id, split, 4)?,
        )),
        "plan_fragment" | "sequential" => generate_plan_fragments(seed, episode_id, split, 64),
        "p1c_falsification" => generate_p1c_falsification_episode(seed, episode_id, split),
        "p1c_hard_retarget" => generate_p1c_hard_retarget_multistep(seed, episode_id, split, 3),
        other => bail!("unknown curriculum kind {other}"),
    }
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
            markers: vec![Pos::new(4, 4), Pos::new(0, 4)],
            collectibles: vec![Pos::new(1, 0)],
            switches: vec![Pos::new(0, 1)],
            hazards: vec![Pos::new(3, 3)],
            resource_pickups: vec![Pos::new(0, 3)],
            terminal_triggers: vec![Pos::new(4, 2)],
            start: Pos::new(0, 0),
            initial_resource: 2,
            action_budget: 40,
            undo_enabled: true,
            candidate_goals: vec![
                Goal::ReachMarker { marker: 0 },
                Goal::CollectAll,
                Goal::ActivateSwitchesInOrder { order: vec![0] },
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
            seed: 7,
            episode_id: 3,
        }
    }

    #[test]
    fn palette_validation_rejects_out_of_range() {
        let err = ArcFrame::new(2, 2, vec![0, 1, 2, 16]).unwrap_err();
        assert!(err.to_string().contains("palette"));
    }

    #[test]
    fn action_mapping_nsew_undo_and_action6_coords() {
        assert_eq!(ArcAction::from_tofy(Action::Move(Dir::North)).id, 1);
        assert_eq!(ArcAction::from_tofy(Action::Move(Dir::South)).id, 2);
        assert_eq!(ArcAction::from_tofy(Action::Move(Dir::East)).id, 3);
        assert_eq!(ArcAction::from_tofy(Action::Move(Dir::West)).id, 4);
        assert_eq!(ArcAction::from_tofy(Action::Undo).id, 5);
        assert!(ArcAction::new(6, Some(10), Some(20)).is_ok());
        assert!(ArcAction::new(6, None, None).is_err());
        assert!(ArcAction::new(1, Some(0), None).is_err());
        assert!(ArcAction::new(0, None, None).is_err());
        assert!(ArcAction::new(7, None, None).is_err());
    }

    #[test]
    fn goal_features_cover_six_families_without_hidden_index() {
        let sc = tiny_scenario();
        for goal in &sc.candidate_goals {
            let feat = GoalFeatures::encode(goal);
            assert_eq!(feat.values.len(), GOAL_FEATURES_DIM);
            let json = serde_json::to_string(&feat).unwrap();
            assert!(!json.contains("hidden_goal_index"));
            assert!(!json.contains("hidden"));
        }
        let a = GoalFeatures::encode(&Goal::ReachMarker { marker: 0 });
        let b = GoalFeatures::encode(&Goal::CollectAll);
        assert_ne!(a.values, b.values);
        assert_eq!(a.values[0], 1.0);
        assert_eq!(b.values[1], 1.0);
    }

    #[test]
    fn render_has_no_hidden_index_leakage() {
        let mut sc = tiny_scenario();
        let st = State::initial(&sc);
        let f0 = render_state_fixed(&sc, &st).unwrap();
        sc.hidden_goal_index = 4;
        let f1 = render_state_fixed(&sc, &st).unwrap();
        assert_eq!(f0, f1);
        let json = serde_json::to_string(&f0).unwrap();
        assert!(!json.contains("hidden"));
        assert_eq!(f0.width, 64);
        assert_eq!(f0.height, 64);
        assert!(f0.pixels.contains(&palette::AGENT));
        assert!(f0.pixels.contains(&palette::WALL));
    }

    #[test]
    fn candidate_labels_ignore_hidden_goal() {
        let mut sc = tiny_scenario();
        sc.hidden_goal_index = 0; // ReachMarker
        let sim = Simulator::new(sc.clone());
        let before = State::initial(&sc);
        // Step onto hazard under AvoidHazard candidate.
        let mut state = before.clone();
        for a in [
            Action::Move(Dir::East),
            Action::Move(Dir::East),
            Action::Move(Dir::East),
            Action::Move(Dir::South),
            Action::Move(Dir::South),
            Action::Move(Dir::South),
        ] {
            state = apply_action(&sim, &state, a);
        }
        assert_eq!(state.pos, Pos::new(3, 3));
        let avoid = Goal::AvoidHazardReachMarker {
            hazard: 0,
            marker: 0,
        };
        let reach = Goal::ReachMarker { marker: 0 };
        let sample_avoid =
            sample_from_transition(&sc, &before, &state, Action::Move(Dir::South), &avoid).unwrap();
        let sample_reach =
            sample_from_transition(&sc, &before, &state, Action::Move(Dir::South), &reach).unwrap();
        assert_eq!(sample_avoid.goal_failed, Some(true));
        assert_eq!(sample_avoid.goal_satisfied, Some(false));
        assert_eq!(sample_reach.goal_failed, Some(false));
        assert_eq!(sample_reach.goal_satisfied, Some(false));
    }

    #[test]
    fn curriculum_generators_are_deterministic() {
        let a = generate_curriculum("random_one_step", 11, 2, Split::Train).unwrap();
        let b = generate_curriculum("random_one_step", 11, 2, Split::Train).unwrap();
        assert_eq!(a, b);
        assert!(!a.is_empty());
        assert_eq!(a[1].family, "coordinate_action");
        assert_eq!(a[1].action.id, 6);
        assert!(a[1].action.x.is_some() && a[1].action.y.is_some());

        let p = generate_curriculum("plan_fragment", 3, 0, Split::Train).unwrap();
        let q = generate_curriculum("plan_fragment", 3, 0, Split::Train).unwrap();
        assert_eq!(p, q);

        let f = generate_curriculum("p1c_falsification", 5, 1, Split::Train).unwrap();
        let g = generate_curriculum("p1c_falsification", 5, 1, Split::Train).unwrap();
        assert_eq!(f, g);
        assert!(f.len() >= 2);
        // Same transition, different candidate features.
        assert_eq!(f[0].current, f[1].current);
        assert_eq!(f[0].action, f[1].action);
        assert_ne!(f[0].goal_features, f[1].goal_features);

        let h = generate_curriculum("p1c_hard_retarget", 9, 0, Split::Train).unwrap();
        let i = generate_curriculum("p1c_hard_retarget", 9, 0, Split::Train).unwrap();
        assert_eq!(h, i);
        assert!(!h.is_empty());
    }

    #[test]
    fn pad_rejects_oversize_without_interpolation() {
        let big = ArcFrame::new(65, 1, vec![0; 65]).unwrap();
        assert!(big.to_fixed_64().is_err());
    }
}
