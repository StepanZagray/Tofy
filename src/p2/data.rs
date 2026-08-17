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
use std::collections::BTreeMap;

/// Official ARC-AGI-3 frame side length.
pub const FRAME_SIDE: usize = 64;

/// Fixed length of [`GoalFeatures::values`].
pub const GOAL_FEATURES_DIM: usize = 19;

/// Maximum switch-order slots packed into goal features.
pub const GOAL_ORDER_SLOTS: usize = 8;

/// Fixed-size simulator oracle for LeJEPA identifiability diagnostics in `p2-eval`.
pub const ORACLE_LATENT_DIM: usize = 16;

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

/// Official ARC-AGI-3 discrete action ids `1..=7`. Coordinates only for id 6.
///
/// Matches https://docs.arcprize.org/actions :
/// ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right, ACTION5=interact,
/// ACTION6=coordinate, ACTION7=undo. `RESET` is not an `ArcAction`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArcAction {
    pub id: u8,
    pub x: Option<u8>,
    pub y: Option<u8>,
}

impl ArcAction {
    pub fn new(id: u8, x: Option<u8>, y: Option<u8>) -> Result<Self> {
        ensure!((1..=7).contains(&id), "action id {id} not in 1..=7");
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
            Action::Move(Dir::West) => 3,
            Action::Move(Dir::East) => 4,
            Action::Undo => 7,
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
            3 => Ok(Action::Move(Dir::West)),
            4 => Ok(Action::Move(Dir::East)),
            5 => bail!("ACTION5 (interact) has no Tofy Action mapping"),
            6 => bail!("ACTION6 has no Tofy Action mapping"),
            7 => Ok(Action::Undo),
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionProvenance {
    /// Width of the semantic board region. Pixels outside this rectangle are padding/UI.
    pub content_width: u16,
    /// Height of the semantic board region. The ARC status row is never part of this region.
    pub content_height: u16,
    /// Generator/import population that produced the transition (stable across goal retargeting).
    pub source_kind: String,
    /// Stable trajectory identity. Unlike `family`, this does not change when a goal is retargeted.
    pub trajectory_id: String,
}

impl TransitionProvenance {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=FRAME_SIDE as u16).contains(&self.content_width),
            "content_width must be in 1..={FRAME_SIDE}"
        );
        ensure!(
            (1..FRAME_SIDE as u16).contains(&self.content_height),
            "content_height must be in 1..{}",
            FRAME_SIDE - 1
        );
        ensure!(
            !self.source_kind.is_empty(),
            "source_kind must not be empty"
        );
        ensure!(
            !self.trajectory_id.is_empty(),
            "trajectory_id must not be empty"
        );
        Ok(())
    }

    pub(crate) fn simulator(scenario: &Scenario, source_kind: impl Into<String>) -> Self {
        let source_kind = source_kind.into();
        Self {
            content_width: u16::from(scenario.width),
            content_height: u16::from(scenario.height),
            trajectory_id: format!(
                "sim/{source_kind}/{:?}/{}/{}",
                scenario.split, scenario.seed, scenario.episode_id
            ),
            source_kind,
        }
    }

    pub(crate) fn full_frame(seed: u64, episode_id: u64, split: Split, source_kind: &str) -> Self {
        Self {
            content_width: FRAME_SIDE as u16,
            content_height: (FRAME_SIDE - 1) as u16,
            source_kind: source_kind.into(),
            trajectory_id: format!("synthetic/{source_kind}/{split:?}/{seed}/{episode_id}"),
        }
    }
}

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
    /// Monotonic index within `(seed, episode_id)` for rollout ordering.
    #[serde(default)]
    pub transition_index: u64,
    pub provenance: TransitionProvenance,
    /// Exact-simulator features for identifiability eval; absent for ARC recordings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oracle_latent: Option<Vec<f32>>,
}

/// Exact board-only result of one factual action. The bottom status row is
/// deliberately excluded: it advances with the action budget even when the
/// world itself did not change.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardEffect {
    pub changed: bool,
    pub changed_cells: Vec<u16>,
    /// Collision-free outcome key, meaningful only among branches that share
    /// one current frame.
    outcome_pixels: Vec<u8>,
}

/// One confirmed transition inside a same-state action comparison.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FactualActionBranch {
    pub transition: TransitionSample,
    pub board_effect: BoardEffect,
    pub status_changed_cells: Vec<u16>,
}

/// Two or more factual actions executed from a byte-identical current frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BranchGroup {
    branches: Vec<FactualActionBranch>,
}

/// Every generated factual lesson is one four-action comparison. Keeping this
/// contract next to the data interface prevents the trainer from silently
/// accepting a truncated group as an independent batch.
pub const FACTUAL_BRANCHES_PER_GROUP: usize = 4;

/// Stable identity of one same-state factual comparison.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BranchGroupId {
    pub seed: u64,
    pub episode_id: u64,
    pub trajectory_id: String,
    pub current_fingerprint: String,
}

/// A complete factual population in canonical group/action order.
///
/// This is the only adapter from flat curriculum rows into branch learning.
/// Construction is order-independent and rejects missing or duplicated rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FactualBatch {
    groups: Vec<BranchGroup>,
    group_ids: Vec<BranchGroupId>,
    rows: Vec<TransitionSample>,
    group_ranges: Vec<std::ops::Range<usize>>,
}

fn frame_fingerprint(frame: &ArcFrame) -> String {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for byte in frame
        .width
        .to_le_bytes()
        .into_iter()
        .chain(frame.height.to_le_bytes())
        .chain(frame.pixels.iter().copied())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    format!("fnv1a64:{hash:016x}")
}

impl BranchGroupId {
    fn from_transition(transition: &TransitionSample) -> Self {
        Self {
            seed: transition.seed,
            episode_id: transition.episode_id,
            trajectory_id: transition.provenance.trajectory_id.clone(),
            current_fingerprint: frame_fingerprint(&transition.current),
        }
    }

    pub(crate) fn from_transition_for_eval(transition: &TransitionSample) -> Self {
        Self::from_transition(transition)
    }
}

impl FactualBatch {
    pub fn from_groups(mut groups: Vec<BranchGroup>) -> Result<Self> {
        ensure!(!groups.is_empty(), "factual batch is empty");
        for group in &groups {
            ensure!(
                group.branches.len() == FACTUAL_BRANCHES_PER_GROUP,
                "factual group must contain exactly {FACTUAL_BRANCHES_PER_GROUP} branches, got {}",
                group.branches.len()
            );
        }
        groups.sort_by_key(|group| BranchGroupId::from_transition(&group.branches[0].transition));

        let mut group_ids = Vec::with_capacity(groups.len());
        let mut rows = Vec::with_capacity(groups.len() * FACTUAL_BRANCHES_PER_GROUP);
        let mut group_ranges = Vec::with_capacity(groups.len());
        for group in &mut groups {
            group.branches.sort_by_key(|branch| {
                let action = &branch.transition.action;
                (action.id, action.x, action.y)
            });
            let id = BranchGroupId::from_transition(&group.branches[0].transition);
            let start = rows.len();
            rows.extend(
                group
                    .branches
                    .iter()
                    .map(|branch| branch.transition.clone()),
            );
            group_ranges.push(start..rows.len());
            group_ids.push(id);
        }
        Ok(Self {
            groups,
            group_ids,
            rows,
            group_ranges,
        })
    }

    pub fn from_rows(rows: &[TransitionSample]) -> Result<Self> {
        ensure!(!rows.is_empty(), "factual batch is empty");
        let mut grouped = BTreeMap::<BranchGroupId, Vec<FactualActionBranch>>::new();
        for transition in rows {
            ensure!(
                transition.family.starts_with("factual_"),
                "non-factual row {} cannot enter a factual batch",
                transition.family
            );
            grouped
                .entry(BranchGroupId::from_transition(transition))
                .or_default()
                .push(FactualActionBranch::try_from_transition(
                    transition.clone(),
                )?);
        }
        let groups = grouped
            .into_values()
            .map(|branches| {
                ensure!(
                    branches.len() == FACTUAL_BRANCHES_PER_GROUP,
                    "incomplete factual group: expected {FACTUAL_BRANCHES_PER_GROUP} branches, got {}",
                    branches.len()
                );
                BranchGroup::try_new(branches)
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_groups(groups)
    }

    pub fn groups(&self) -> &[BranchGroup] {
        &self.groups
    }

    pub fn group_ids(&self) -> &[BranchGroupId] {
        &self.group_ids
    }

    pub fn rows(&self) -> &[TransitionSample] {
        &self.rows
    }

    pub fn group_ranges(&self) -> &[std::ops::Range<usize>] {
        &self.group_ranges
    }
}

impl FactualActionBranch {
    pub(crate) fn try_from_transition(transition: TransitionSample) -> Result<Self> {
        ensure!(
            transition.current.width as usize == FRAME_SIDE
                && transition.current.height as usize == FRAME_SIDE
                && transition.next.width as usize == FRAME_SIDE
                && transition.next.height as usize == FRAME_SIDE,
            "factual branches require fixed {FRAME_SIDE}x{FRAME_SIDE} frames"
        );
        let status_start = (FRAME_SIDE - 1) * FRAME_SIDE;
        let mut changed_cells = Vec::new();
        let mut status_changed_cells = Vec::new();
        for (index, (&before, &after)) in transition
            .current
            .pixels
            .iter()
            .zip(&transition.next.pixels)
            .enumerate()
        {
            if before == after {
                continue;
            }
            let index = u16::try_from(index).expect("64x64 cell index fits u16");
            if usize::from(index) >= status_start {
                status_changed_cells.push(index);
            } else {
                changed_cells.push(index);
            }
        }
        let outcome_pixels = transition.next.pixels[..status_start].to_vec();
        Ok(Self {
            board_effect: BoardEffect {
                changed: !changed_cells.is_empty(),
                changed_cells,
                outcome_pixels,
            },
            status_changed_cells,
            transition,
        })
    }

    pub fn outcome_equivalent(&self, other: &Self) -> bool {
        self.board_effect.outcome_pixels == other.board_effect.outcome_pixels
    }
}

impl BranchGroup {
    pub(crate) fn try_new(branches: Vec<FactualActionBranch>) -> Result<Self> {
        ensure!(
            branches.len() >= 2,
            "a factual branch group requires at least two branches"
        );
        let first = &branches[0].transition;
        let mut actions = std::collections::BTreeSet::new();
        for branch in &branches {
            let transition = &branch.transition;
            ensure!(
                transition.current == first.current,
                "all factual branches must share a byte-identical current frame"
            );
            ensure!(
                transition.seed == first.seed && transition.episode_id == first.episode_id,
                "all factual branches must share source provenance"
            );
            ensure!(
                actions.insert((
                    transition.action.id,
                    transition.action.x,
                    transition.action.y
                )),
                "factual branch actions must be distinct"
            );
        }
        Ok(Self { branches })
    }

    pub fn branches(&self) -> &[FactualActionBranch] {
        &self.branches
    }

    /// Changed branches whose board-only outcome identifies that action within
    /// this same-state group. Status-strip changes deliberately do not enter
    /// this relation.
    pub(crate) fn unique_changed_effect_indices(&self) -> Vec<usize> {
        self.branches
            .iter()
            .enumerate()
            .filter_map(|(index, branch)| {
                (branch.board_effect.changed
                    && self
                        .branches
                        .iter()
                        .filter(|candidate| branch.outcome_equivalent(candidate))
                        .count()
                        == 1)
                    .then_some(index)
            })
            .collect()
    }

    pub fn into_transitions(self) -> impl Iterator<Item = TransitionSample> {
        self.branches.into_iter().map(|branch| branch.transition)
    }
}

fn popcount_norm(bits: u32, denom: usize) -> f32 {
    let denom = denom.max(1) as f32;
    bits.count_ones() as f32 / denom
}

fn norm_pos(scenario: &Scenario, pos: Pos) -> (f32, f32) {
    let x = (pos.x as f32 + 0.5) / scenario.width.max(1) as f32 * 2.0 - 1.0;
    let y = (pos.y as f32 + 0.5) / scenario.height.max(1) as f32 * 2.0 - 1.0;
    (x, y)
}

/// Compact oracle latent from exact simulator state (layout-independent dynamics).
pub fn oracle_latent(scenario: &Scenario, state: &State) -> Vec<f32> {
    let budget = scenario.action_budget.max(1) as f32;
    let n_switch = scenario.switches.len().max(1) as f32;
    let (px, py) = norm_pos(scenario, state.pos);
    let mut out = vec![0f32; ORACLE_LATENT_DIM];
    out[0] = px;
    out[1] = py;
    out[2] = state.resource as f32 / 255.0;
    out[3] = state.actions_used as f32 / budget;
    out[4] = popcount_norm(state.remaining_collectibles, scenario.collectibles.len());
    out[5] = popcount_norm(state.remaining_pickups, scenario.resource_pickups.len());
    out[6] = popcount_norm(state.touched_hazards, scenario.hazards.len());
    out[7] = state.switch_trace.len() as f32 / GOAL_ORDER_SLOTS as f32;
    for slot in 0..GOAL_ORDER_SLOTS {
        out[8 + slot] = if slot < state.switch_trace.len() {
            state.switch_trace[slot] as f32 / n_switch
        } else {
            -1.0
        };
    }
    out
}

/// Frame-only oracle fallback when no simulator `State` is available.
pub fn oracle_latent_from_frame(frame: &ArcFrame) -> Vec<f32> {
    let mut out = vec![0f32; ORACLE_LATENT_DIM];
    for (idx, &pixel) in frame.pixels.iter().enumerate() {
        if pixel == palette::AGENT {
            let x = idx % FRAME_SIDE;
            let y = idx / FRAME_SIDE;
            out[0] = x as f32 / (FRAME_SIDE.saturating_sub(1).max(1) as f32) * 2.0 - 1.0;
            out[1] = y as f32 / (FRAME_SIDE.saturating_sub(1).max(1) as f32) * 2.0 - 1.0;
            break;
        }
    }
    out
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

/// Render native grid, pad to official `64×64`, and draw per-episode status UI in the
/// margin (ARC-AGI-3 frames embed budget counters in pixels; placement varies by game).
pub fn render_state_padded(scenario: &Scenario, state: &State) -> Result<ArcFrame> {
    let mut frame = render_state(scenario, state)?.to_fixed_64()?;
    apply_arc_status_ui(&mut frame, scenario, state);
    Ok(frame)
}

/// Render a `(before, after)` pair padded to [`FRAME_SIDE`].
pub fn render_transition_frames(
    scenario: &Scenario,
    before: &State,
    after: &State,
) -> Result<(ArcFrame, ArcFrame)> {
    Ok((
        render_state_padded(scenario, before)?,
        render_state_padded(scenario, after)?,
    ))
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
    transition_index: u64,
) -> Result<TransitionSample> {
    let (current, next) = render_transition_frames(scenario, before, after)?;
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
        transition_index,
        provenance: TransitionProvenance::simulator(scenario, goal_family(goal)),
        oracle_latent: Some(oracle_latent(scenario, before)),
    })
}

/// Goal-free transition sample: zero goal features and masked event labels (early ARC play).
pub fn sample_from_transition_goal_free(
    scenario: &Scenario,
    before: &State,
    after: &State,
    action: Action,
    family: &str,
    transition_index: u64,
) -> Result<TransitionSample> {
    let (current, next) = render_transition_frames(scenario, before, after)?;
    Ok(TransitionSample {
        current,
        next,
        action: ArcAction::from_tofy(action),
        goal_features: GoalFeatures::zeros(),
        noop: Some(action_is_noop(scenario, before, action)),
        goal_satisfied: None,
        goal_failed: None,
        exhausted: None,
        split: scenario.split,
        family: family.into(),
        seed: scenario.seed,
        episode_id: scenario.episode_id,
        transition_index,
        provenance: TransitionProvenance::simulator(scenario, family),
        oracle_latent: Some(oracle_latent(scenario, before)),
    })
}

/// Paint remaining action-budget UI on the bottom row (common ARC-AGI-3 layout).
fn apply_arc_status_ui(frame: &mut ArcFrame, scenario: &Scenario, state: &State) {
    paint_status_ui(frame, scenario.action_budget, state.actions_used);
}

fn paint_status_ui(frame: &mut ArcFrame, action_budget: u16, actions_used: u16) {
    let budget = action_budget.max(1) as usize;
    let remaining = budget.saturating_sub(actions_used as usize);
    let filled = remaining.saturating_mul(FRAME_SIDE) / budget;
    let color = palette::WALL;
    for x in 0..filled.min(FRAME_SIDE) {
        frame.pixels[(FRAME_SIDE - 1) * FRAME_SIDE + x] = color;
    }
}

fn apply_action(sim: &Simulator, state: &State, action: Action) -> State {
    sim.transition(state, action)
}

/// Random legal one-step transitions without candidate-goal conditioning (early ARC play).
pub fn generate_random_one_step(
    seed: u64,
    episode_id: u64,
    split: Split,
    n: usize,
) -> Result<Vec<TransitionSample>> {
    let scenario = generate(seed, episode_id, split);
    let sim = Simulator::new(scenario.clone());
    let mut rng = rng_for(seed ^ 0xA11C_E001, episode_id, split);
    let mut out = Vec::with_capacity(n);
    let mut state = State::initial(&scenario);
    for step in 0..n {
        let actions = legal_actions(&scenario);
        ensure!(!actions.is_empty(), "no legal actions");
        let action = *actions.choose(&mut rng).expect("non-empty");
        let next = apply_action(&sim, &state, action);
        out.push(sample_from_transition_goal_free(
            &scenario,
            &state,
            &next,
            action,
            "dynamics",
            step as u64,
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
        let mut next_pixels = current_pixels.clone();
        next_pixels[start_y * FRAME_SIDE + start_x] = palette::EMPTY;
        next_pixels[y as usize * FRAME_SIDE + x as usize] = palette::AGENT;
        let mut current = ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, current_pixels)?;
        let mut next = ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, next_pixels)?;
        paint_status_ui(&mut current, 64, step as u16);
        paint_status_ui(&mut next, 64, step as u16 + 1);
        out.push(TransitionSample {
            next,
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
            transition_index: step as u64,
            provenance: TransitionProvenance::full_frame(
                seed,
                episode_id.wrapping_mul(1_000_003).wrapping_add(step as u64),
                split,
                "coordinate_action",
            ),
            oracle_latent: Some(oracle_latent_from_frame(&current)),
            current,
        });
    }
    Ok(out)
}

/// Train official `ACTION5` (interact) on a synthetic toggle transition.
pub fn generate_interact_one_step(
    seed: u64,
    episode_id: u64,
    split: Split,
    n: usize,
) -> Result<Vec<TransitionSample>> {
    let mut rng = rng_for(seed ^ 0xA11C_A005, episode_id, split);
    let mut out = Vec::with_capacity(n);
    for step in 0..n {
        let switch_x = rng.random_range(10..54) as u8;
        let switch_y = rng.random_range(10..54) as u8;
        let agent_x = switch_x.saturating_sub(1);
        let agent_y = switch_y;
        let mut current_pixels = vec![palette::PAD; FRAME_SIDE * FRAME_SIDE];
        current_pixels[agent_y as usize * FRAME_SIDE + agent_x as usize] = palette::AGENT;
        current_pixels[switch_y as usize * FRAME_SIDE + switch_x as usize] = palette::SWITCH_BASE;
        let mut next_pixels = current_pixels.clone();
        next_pixels[switch_y as usize * FRAME_SIDE + switch_x as usize] = palette::SWITCH_BASE + 1;
        let mut current = ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, current_pixels)?;
        let mut next = ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, next_pixels)?;
        paint_status_ui(&mut current, 64, step as u16);
        paint_status_ui(&mut next, 64, step as u16 + 1);
        out.push(TransitionSample {
            oracle_latent: Some(oracle_latent_from_frame(&current)),
            current,
            next,
            action: ArcAction::new(5, None, None)?,
            goal_features: GoalFeatures::zeros(),
            noop: Some(false),
            goal_satisfied: Some(false),
            goal_failed: Some(false),
            exhausted: Some(false),
            split,
            family: "action5_interact".into(),
            seed,
            episode_id: episode_id.wrapping_mul(1_000_003).wrapping_add(step as u64),
            transition_index: step as u64,
            provenance: TransitionProvenance::full_frame(
                seed,
                episode_id.wrapping_mul(1_000_003).wrapping_add(step as u64),
                split,
                "action5_interact",
            ),
        });
    }
    Ok(out)
}

/// Deliberate hazard-entry transitions with `goal_failed=true` for event-head training.
pub fn generate_hazard_one_step(
    seed: u64,
    episode_id: u64,
    split: Split,
    n: usize,
) -> Result<Vec<TransitionSample>> {
    let mut out = Vec::with_capacity(n);
    for step in 0..n {
        let mut scenario = generate(seed, episode_id.wrapping_add(step as u64), split);
        if scenario.hazards.is_empty() {
            scenario.hazards.push(Pos::new(2, 2));
        }
        if scenario.markers.is_empty() {
            scenario.markers.push(Pos::new(4, 4));
        }
        let hazard_pos = scenario.hazards[0];
        let west = Pos::new(hazard_pos.x - 1, hazard_pos.y);
        let east = Pos::new(hazard_pos.x + 1, hazard_pos.y);
        let start = if scenario.in_bounds(west) && !scenario.is_blocked(west) {
            west
        } else if scenario.in_bounds(east) && !scenario.is_blocked(east) {
            east
        } else {
            scenario.start
        };
        scenario.start = start;
        let sim = Simulator::new(scenario.clone());
        let state = State::initial(&scenario);
        let action = if hazard_pos.x > start.x {
            Action::Move(Dir::East)
        } else {
            Action::Move(Dir::West)
        };
        let next = apply_action(&sim, &state, action);
        out.push(sample_from_transition_goal_free(
            &scenario,
            &state,
            &next,
            action,
            "hazard_failure",
            step as u64,
        )?);
    }
    Ok(out)
}

fn generate_simulator_branch_group(
    seed: u64,
    episode_id: u64,
    split: Split,
) -> Result<BranchGroup> {
    let scenario = generate(seed, episode_id, split);
    let sim = Simulator::new(scenario.clone());
    let mut state = State::initial(&scenario);
    // Deterministically vary the shared source state without using the branch
    // action itself to construct the observation.
    let prefix_steps = (episode_id as usize) % 4;
    for prefix in 0..prefix_steps {
        let actions: Vec<_> = legal_actions(&scenario)
            .into_iter()
            .filter(|action| !matches!(action, Action::Undo))
            .collect();
        let action = actions[(episode_id as usize + prefix) % actions.len()];
        state = apply_action(&sim, &state, action);
    }
    // Freeze an explicit within-action balance schedule for held-out factual
    // evidence. Each direction is targeted in turn, alternating between a
    // traversable and blocked source cell; all four branches still share the
    // same exact observation. This removes the old ACTION1/3=no-change and
    // ACTION2/4=change confound from evaluator populations.
    if split == Split::HeldOutComposition {
        let ordinal = episode_id / 2;
        let target_dir = Dir::ALL[(ordinal as usize) % Dir::ALL.len()];
        let want_changed = (ordinal / Dir::ALL.len() as u64).is_multiple_of(2);
        let target_action = Action::Move(target_dir);
        let candidate = (0..scenario.height as i8)
            .flat_map(|y| (0..scenario.width as i8).map(move |x| Pos::new(x, y)))
            .filter(|position| !scenario.is_blocked(*position))
            .find(|position| {
                let mut candidate = state.clone();
                candidate.pos = *position;
                let next = apply_action(&sim, &candidate, target_action);
                (next.pos != candidate.pos) == want_changed
            });
        if let Some(position) = candidate {
            state.pos = position;
            state.undo_stack.clear();
        }
    }
    let branches = legal_actions(&scenario)
        .into_iter()
        .filter(|action| !matches!(action, Action::Undo))
        .take(4)
        .enumerate()
        .map(|(index, action)| {
            let next = apply_action(&sim, &state, action);
            let mut sample = sample_from_transition_goal_free(
                &scenario,
                &state,
                &next,
                action,
                "factual_branch",
                index as u64,
            )?;
            sample.episode_id = episode_id;
            FactualActionBranch::try_from_transition(sample)
        })
        .collect::<Result<Vec<_>>>()?;
    BranchGroup::try_new(branches)
}

fn generate_coordinate_branch_group(
    seed: u64,
    episode_id: u64,
    split: Split,
) -> Result<BranchGroup> {
    let mut rng = rng_for(seed ^ 0xFAC7_A006, episode_id, split);
    let start = (31u8, 31u8);
    let mut current_pixels = vec![palette::EMPTY; FRAME_SIDE * FRAME_SIDE];
    current_pixels[start.1 as usize * FRAME_SIDE + start.0 as usize] = palette::AGENT;
    let mut current = ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, current_pixels)?;
    paint_status_ui(&mut current, 64, 0);
    let mut coordinates = std::collections::BTreeSet::new();
    while coordinates.len() < 4 {
        let coordinate = (
            rng.random_range(0..FRAME_SIDE) as u8,
            rng.random_range(0..FRAME_SIDE - 1) as u8,
        );
        if coordinate != start {
            coordinates.insert(coordinate);
        }
    }
    let branches = coordinates
        .into_iter()
        .enumerate()
        .map(|(index, (x, y))| {
            let mut next_pixels = current.pixels.clone();
            next_pixels[start.1 as usize * FRAME_SIDE + start.0 as usize] = palette::EMPTY;
            next_pixels[y as usize * FRAME_SIDE + x as usize] = palette::AGENT;
            let mut next = ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, next_pixels)?;
            paint_status_ui(&mut next, 64, 1);
            FactualActionBranch::try_from_transition(TransitionSample {
                current: current.clone(),
                next,
                action: ArcAction::new(6, Some(x), Some(y))?,
                goal_features: GoalFeatures::zeros(),
                noop: Some(false),
                goal_satisfied: None,
                goal_failed: None,
                exhausted: Some(false),
                split,
                family: "factual_coordinate_branch".into(),
                seed,
                episode_id,
                transition_index: index as u64,
                provenance: TransitionProvenance::full_frame(
                    seed,
                    episode_id,
                    split,
                    "factual_coordinate_branch",
                ),
                oracle_latent: Some(oracle_latent_from_frame(&current)),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    BranchGroup::try_new(branches)
}

/// Phase-1B factual experience: four different confirmed actions from one
/// unchanged current state. Alternating groups cover simulator movement and
/// marker-free ACTION6 coordinate transitions.
pub fn generate_factual_branch_group(
    seed: u64,
    episode_id: u64,
    split: Split,
) -> Result<BranchGroup> {
    if episode_id.is_multiple_of(2) {
        generate_simulator_branch_group(seed, episode_id, split)
    } else {
        generate_coordinate_branch_group(seed, episode_id, split)
    }
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
    for (idx, action) in plan.actions.into_iter().enumerate() {
        let next = apply_action(&sim, &state, action);
        out.push(sample_from_transition(
            &scenario, &state, &next, action, goal, idx as u64,
        )?);
        state = next;
    }
    Ok(out)
}

/// Goal-free random walk on ordinary maps (early-game exploration proxy).
pub fn generate_exploration_episode(
    seed: u64,
    episode_id: u64,
    split: Split,
) -> Result<Vec<TransitionSample>> {
    let scenario = generate(seed, episode_id, split);
    let sim = Simulator::new(scenario.clone());
    let mut rng = rng_for(seed ^ 0xE1A1_0E001, episode_id, split);
    let mut state = State::initial(&scenario);
    let steps = 8 + (episode_id as usize % 5);
    let mut out = Vec::with_capacity(steps);
    for step in 0..steps {
        let actions = legal_actions(&scenario);
        ensure!(!actions.is_empty(), "no legal actions");
        let action = *actions.choose(&mut rng).expect("non-empty");
        let next = apply_action(&sim, &state, action);
        out.push(sample_from_transition_goal_free(
            &scenario,
            &state,
            &next,
            action,
            "exploration",
            step as u64,
        )?);
        state = next;
        if state.actions_used >= scenario.action_budget {
            break;
        }
    }
    ensure!(
        !out.is_empty(),
        "exploration episode produced no transitions"
    );
    Ok(out)
}

/// P1C episode: short goal-free prefix, then a safe multi-candidate probe from the
/// initial state (synthetic stand-in for “explore, then test hypotheses”).
pub fn generate_hypothesis_probe_episode(
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
    let mut rng = rng_for(seed ^ 0xE97A_7E570, episode_id, split);
    let mut state = State::initial(&scenario);
    let prefix = 8 + (episode_id as usize % 3);
    let mut out = Vec::new();
    for step in 0..prefix {
        let actions = legal_actions(&scenario);
        ensure!(!actions.is_empty(), "no legal actions");
        let action = *actions.choose(&mut rng).expect("non-empty");
        let next = apply_action(&sim, &state, action);
        out.push(sample_from_transition_goal_free(
            &scenario,
            &state,
            &next,
            action,
            "exploration",
            step as u64,
        )?);
        state = next;
        if state.actions_used >= scenario.action_budget {
            break;
        }
    }
    // Standardized safe probe from the published initial state.
    let start = State::initial(&scenario);
    let probe = Action::Move(Dir::South);
    let next = apply_action(&sim, &start, probe);
    let probe_base = out.len() as u64;
    for (gi, goal) in scenario.candidate_goals.iter().enumerate() {
        out.push(sample_from_transition(
            &scenario,
            &start,
            &next,
            probe,
            goal,
            probe_base + gi as u64,
        )?);
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
    for (gi, goal) in scenario.candidate_goals.iter().enumerate() {
        out.push(sample_from_transition(
            &scenario, &start, &next, action, goal, gi as u64,
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
        for (idx, action) in wrong_plan.actions.into_iter().take(wrong_steps).enumerate() {
            let next = apply_action(&sim, &state, action);
            // Labels stay candidate-conditioned on the wrong commitment.
            out.push(sample_from_transition(
                &scenario,
                &state,
                &next,
                action,
                &wrong_goal,
                idx as u64,
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
        let base = out.len() as u64;
        for (idx, action) in true_plan.actions.into_iter().enumerate() {
            let next = apply_action(&sim, &state, action);
            out.push(sample_from_transition(
                &scenario,
                &state,
                &next,
                action,
                &retarget_goal,
                base + idx as u64,
            )?);
            state = next;
            if goal_satisfied(&scenario, &state, &retarget_goal) {
                break;
            }
        }
    }

    for sample in &mut out {
        sample.provenance.source_kind = "p1c_hard_retarget".into();
        sample.provenance.trajectory_id =
            format!("curriculum/p1c_hard_retarget/{split:?}/{seed}/{source_episode_id}");
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
    let mut samples = match kind {
        "factual_branches" => Ok(generate_factual_branch_group(seed, episode_id, split)?
            .into_transitions()
            .collect()),
        "random_one_step" => Ok(interleave(
            interleave(
                generate_random_one_step(seed, episode_id, split, 2)?,
                generate_coordinate_one_step(seed, episode_id, split, 2)?,
            ),
            interleave(
                generate_interact_one_step(seed, episode_id, split, 2)?,
                generate_hazard_one_step(seed, episode_id, split, 2)?,
            ),
        )),
        "plan_fragment" | "sequential" => generate_plan_fragments(seed, episode_id, split, 64),
        "exploration" => generate_exploration_episode(seed, episode_id, split),
        "hypothesis_probe" => generate_hypothesis_probe_episode(seed, episode_id, split),
        "p1c_falsification" => generate_p1c_falsification_episode(seed, episode_id, split),
        "p1c_hard_retarget" => generate_p1c_hard_retarget_multistep(seed, episode_id, split, 3),
        other => bail!("unknown curriculum kind {other}"),
    }?;
    for sample in &mut samples {
        // A curriculum kind is the stable trajectory source. The mixed
        // one-step curriculum keeps its deliberately distinct movement,
        // hazard, ACTION5 and ACTION6 lanes so they cannot collide/group.
        let trajectory_source = if kind == "random_one_step" {
            sample.provenance.source_kind.as_str()
        } else {
            sample.provenance.source_kind = kind.into();
            kind
        };
        sample.provenance.trajectory_id = format!(
            "curriculum/{trajectory_source}/{:?}/{}/{}/{}",
            sample.split, sample.seed, sample.episode_id, episode_id
        );
    }
    Ok(samples)
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
        assert_eq!(ArcAction::from_tofy(Action::Move(Dir::West)).id, 3);
        assert_eq!(ArcAction::from_tofy(Action::Move(Dir::East)).id, 4);
        assert_eq!(ArcAction::from_tofy(Action::Undo).id, 7);
        assert!(ArcAction::new(5, None, None).is_ok());
        assert!(ArcAction::new(5, None, None).unwrap().to_tofy().is_err());
        assert!(ArcAction::new(6, Some(10), Some(20)).is_ok());
        assert!(ArcAction::new(6, None, None).is_err());
        assert!(ArcAction::new(1, Some(0), None).is_err());
        assert!(ArcAction::new(0, None, None).is_err());
        assert_eq!(
            ArcAction::new(7, None, None).unwrap().to_tofy().unwrap(),
            Action::Undo
        );
        assert!(ArcAction::new(8, None, None).is_err());
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
        let f0 = render_state_padded(&sc, &st).unwrap();
        sc.hidden_goal_index = 4;
        let f1 = render_state_padded(&sc, &st).unwrap();
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
            sample_from_transition(&sc, &before, &state, Action::Move(Dir::South), &avoid, 0)
                .unwrap();
        let sample_reach =
            sample_from_transition(&sc, &before, &state, Action::Move(Dir::South), &reach, 1)
                .unwrap();
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
        let coord = a
            .iter()
            .find(|s| s.family == "coordinate_action")
            .expect("coordinate sample");
        assert_eq!(coord.action.id, 6);
        assert!(coord.action.x.is_some() && coord.action.y.is_some());
        assert!(coord
            .current
            .pixels
            .iter()
            .all(|&pixel| { !(palette::MARKER_BASE..palette::COLLECTIBLE).contains(&pixel) }));
        assert!(a.iter().any(|s| s.family == "action5_interact"));
        assert!(a.iter().any(|s| s.family == "hazard_failure"));

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

        let ex = generate_curriculum("exploration", 7, 2, Split::Train).unwrap();
        assert!(ex
            .iter()
            .all(|s| s.goal_features.values == [0.0; GOAL_FEATURES_DIM]));
        assert!(ex.iter().all(|s| s.goal_satisfied.is_none()));

        let hp = generate_curriculum("hypothesis_probe", 5, 1, Split::Train).unwrap();
        assert!(hp.len() >= 3);
        assert!(hp.iter().any(|s| s.goal_satisfied.is_some()));

        let plan = generate_curriculum("sequential", 11, 2, Split::Train).unwrap();
        assert!(plan.len() >= 2);
        for pair in plan.windows(2) {
            assert_eq!(
                pair[0].next, pair[1].current,
                "sequential trace must chain rendered frames"
            );
        }
    }

    #[test]
    fn dynamics_samples_are_goal_free() {
        let samples = generate_curriculum("random_one_step", 11, 2, Split::Train).unwrap();
        assert!(samples.iter().any(|s| s.family == "dynamics"));
        assert!(samples
            .iter()
            .filter(|s| s.family == "dynamics")
            .all(|s| s.goal_features.values == [0.0; GOAL_FEATURES_DIM]));
        assert!(samples
            .iter()
            .filter(|s| s.family == "dynamics")
            .all(|s| s.goal_satisfied.is_none()));
    }

    #[test]
    fn factual_groups_preserve_shared_state_and_board_only_effects() -> Result<()> {
        let movement = generate_factual_branch_group(17, 2, Split::Train)?;
        assert_eq!(movement.branches().len(), 4);
        assert!(movement
            .branches()
            .windows(2)
            .all(|pair| pair[0].transition.current == pair[1].transition.current));
        assert!(movement
            .branches()
            .iter()
            .all(|branch| !branch.status_changed_cells.is_empty()));

        let coordinate = generate_factual_branch_group(17, 3, Split::Train)?;
        assert_eq!(coordinate.branches().len(), 4);
        let current = &coordinate.branches()[0].transition.current;
        assert_eq!(
            current
                .pixels
                .iter()
                .filter(|&&pixel| pixel == palette::AGENT)
                .count(),
            1
        );
        assert_eq!(
            current
                .pixels
                .iter()
                .filter(|&&pixel| (palette::MARKER_BASE..palette::COLLECTIBLE).contains(&pixel))
                .count(),
            0,
            "ACTION6 target coordinates must not leak through marker pixels"
        );
        assert!(coordinate
            .branches()
            .iter()
            .all(|branch| branch.board_effect.changed));
        assert_eq!(
            coordinate.unique_changed_effect_indices(),
            vec![0, 1, 2, 3],
            "distinct ACTION6 board outcomes are recoverable without status UI"
        );
        Ok(())
    }

    #[test]
    fn factual_batch_reconstructs_shuffled_complete_groups_and_rejects_halves() -> Result<()> {
        let groups = vec![
            generate_factual_branch_group(17, 2, Split::Train)?,
            generate_factual_branch_group(17, 3, Split::Train)?,
        ];
        let expected = FactualBatch::from_groups(groups)?;
        let mut shuffled = expected.rows().to_vec();
        shuffled.reverse();
        shuffled.rotate_left(3);
        let reconstructed = FactualBatch::from_rows(&shuffled)?;
        assert_eq!(reconstructed.group_ids(), expected.group_ids());
        assert_eq!(reconstructed.rows(), expected.rows());
        assert_eq!(reconstructed.group_ranges(), &[0..4, 4..8]);
        assert!(FactualBatch::from_rows(&shuffled[..2]).is_err());
        Ok(())
    }

    #[test]
    fn held_out_simulator_population_has_both_outcomes_within_each_action() -> Result<()> {
        let mut outcomes = BTreeMap::<u8, (bool, bool)>::new();
        for episode in (0..128).step_by(2) {
            for branch in
                generate_factual_branch_group(0xFA_C7_EA_11, episode, Split::HeldOutComposition)?
                    .branches()
            {
                let entry = outcomes.entry(branch.transition.action.id).or_default();
                if branch.board_effect.changed {
                    entry.0 = true;
                } else {
                    entry.1 = true;
                }
            }
        }
        assert!(
            outcomes
                .values()
                .all(|&(changed, unchanged)| changed && unchanged),
            "each evaluated simple action needs changed and unchanged examples: {outcomes:?}"
        );
        Ok(())
    }

    #[test]
    fn status_ui_preserves_native_playfield() -> Result<()> {
        let sc = tiny_scenario();
        let st = State::initial(&sc);
        let padded = render_state_padded(&sc, &st)?;
        let native = render_state(&sc, &st)?;
        let pw = sc.width as usize;
        let ph = sc.height as usize;
        for y in 0..ph {
            for x in 0..pw {
                assert_eq!(padded.pixels[y * FRAME_SIDE + x], native.pixels[y * pw + x]);
            }
        }
        Ok(())
    }

    #[test]
    fn pad_rejects_oversize_without_interpolation() {
        let big = ArcFrame::new(65, 1, vec![0; 65]).unwrap();
        assert!(big.to_fixed_64().is_err());
    }
}
