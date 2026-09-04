//! ARC-compatible frames, actions, goal features, and exact-simulator transitions.
//!
//! Candidate-conditioned labels are always computed against a supplied public
//! [`Goal`], never `Scenario.hidden_goal_index`.

use crate::domain::{
    goal_family, goal_satisfied, goal_terminal_failure, legal_actions, Action, Dir, Goal, Pos,
    Scenario, Simulator, Split, State,
};
use crate::generator::{
    generate, generate_p1c, generate_p1c_hard_candidate, generate_sized,
    p1c_falsification_probe_width, rebind_hidden_goal_family, rng_for, V5_CONTENT_SIZES,
};
use crate::search::shortest_path;
use anyhow::{anyhow, bail, ensure, Context, Result};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::sync::Arc;

/// Official ARC-AGI-3 frame side length.
pub const FRAME_SIDE: usize = 64;

/// Fixed length of [`GoalFeatures::values`].
pub const GOAL_FEATURES_DIM: usize = 19;

/// Maximum switch-order slots packed into goal features.
pub const GOAL_ORDER_SLOTS: usize = 8;

/// Fixed-size simulator oracle for LeJEPA identifiability diagnostics in `p2-eval`.
pub const ORACLE_LATENT_DIM: usize = 16;

/// Playfield rows available to synthetic content. Row 63 is status UI only.
pub const V5_PLAYFIELD_HEIGHT: usize = FRAME_SIDE - 1;

/// Goal-free queries are kept in-distribution at this fixed v5 rate.
pub const V5_GOAL_DROPOUT_PROBABILITY: f32 = 0.30;

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

/// Clone-on-write byte storage. Generated frames and content masks are cloned
/// frequently while they are immutable, so sharing here avoids copying a
/// 4096-byte buffer for every row that reuses the same state or rectangle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedBytes(Arc<Vec<u8>>);

impl SharedBytes {
    fn allocation_id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
}

impl From<Vec<u8>> for SharedBytes {
    fn from(values: Vec<u8>) -> Self {
        Self(Arc::new(values))
    }
}

impl Deref for SharedBytes {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SharedBytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.0)
    }
}

impl AsRef<[u8]> for SharedBytes {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

impl<'a> IntoIterator for &'a SharedBytes {
    type Item = &'a u8;
    type IntoIter = std::slice::Iter<'a, u8>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl Serialize for SharedBytes {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SharedBytes {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<u8>::deserialize(deserializer).map(Self::from)
    }
}

/// Discrete ARC-like frame: row-major `width * height` categorical pixels in `0..=15`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArcFrame {
    pub width: u16,
    pub height: u16,
    pub pixels: SharedBytes,
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
            pixels: pixels.into(),
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

/// ARC-AGI-3 discrete action ids `1..=7`, plus trained synthetic NULL id 0.
/// Coordinates are present only for id 6.
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
        ensure!(id <= 7, "action id {id} not in 0..=7");
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
            0 => bail!("NULL action has no Tofy Action mapping"),
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

/// The five ADR 0003 sources mixed concurrently by the foundation-v2 data
/// schedule, plus the ADR 0005 §2.1 `LearningHistories` stream and the §2.6
/// `SyntheticShards` stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MixedStreamKind {
    RandomOneStep,
    FactualBranches,
    Exploration,
    SequentialFragments,
    HazardOneStep,
    LearningHistories,
    /// ARCEngine shard rows loaded by [`crate::p2::synthetic_shards`]
    /// (ADR 0005 §2.6). Weight 0 in every shipped schedule.
    SyntheticShards,
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn is_zero_u8(value: &u8) -> bool {
    *value == 0
}

fn is_zero_u16(value: &u16) -> bool {
    *value == 0
}

fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

fn is_zero_f32(value: &f32) -> bool {
    *value == 0.0
}

/// Raw schedule weights from ADR 0003 §1.1.
///
/// The documented endpoint weights total 0.95. [`normalized`] preserves their
/// ratios when a caller needs to fill a fixed physical row budget; the raw
/// fields remain the exact percentages specified by the ADR.
/// `learning_histories` and `synthetic_shards` are zero for every legacy
/// schedule and are omitted from its serialization so legacy batch identities
/// are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MixedStreamProportions {
    pub random_one_step: f32,
    pub factual_branches: f32,
    pub exploration: f32,
    pub sequential_fragments: f32,
    pub hazard_one_step: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub learning_histories: f32,
    /// ADR 0005 §2.6 shard stream. Zero in every shipped schedule until the
    /// shard generator passes the §5.1 memorization diagnostic on its twins.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub synthetic_shards: f32,
}

impl MixedStreamProportions {
    pub fn total(self) -> f32 {
        self.random_one_step
            + self.factual_branches
            + self.exploration
            + self.sequential_fragments
            + self.hazard_one_step
            + self.learning_histories
            + self.synthetic_shards
    }

    pub fn normalized(self) -> Self {
        let total = self.total();
        if total <= f32::EPSILON {
            return self;
        }
        Self {
            random_one_step: self.random_one_step / total,
            factual_branches: self.factual_branches / total,
            exploration: self.exploration / total,
            sequential_fragments: self.sequential_fragments / total,
            hazard_one_step: self.hazard_one_step / total,
            learning_histories: self.learning_histories / total,
            synthetic_shards: self.synthetic_shards / total,
        }
    }

    fn ordered(self) -> [(MixedStreamKind, f32); 7] {
        [
            (MixedStreamKind::RandomOneStep, self.random_one_step),
            (MixedStreamKind::FactualBranches, self.factual_branches),
            (MixedStreamKind::Exploration, self.exploration),
            (
                MixedStreamKind::SequentialFragments,
                self.sequential_fragments,
            ),
            (MixedStreamKind::HazardOneStep, self.hazard_one_step),
            (MixedStreamKind::LearningHistories, self.learning_histories),
            (MixedStreamKind::SyntheticShards, self.synthetic_shards),
        ]
    }
}

/// Linear start-to-end stream schedule from ADR 0003 §1.1.
pub fn foundation_v2_stream_schedule(progress: f32) -> MixedStreamProportions {
    let progress = if progress.is_finite() {
        progress.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let lerp = |start: f32, end: f32| start + (end - start) * progress;
    MixedStreamProportions {
        random_one_step: lerp(0.35, 0.25),
        factual_branches: lerp(0.20, 0.30),
        exploration: 0.20,
        sequential_fragments: 0.15,
        hazard_one_step: lerp(0.10, 0.05),
        learning_histories: 0.0,
        synthetic_shards: 0.0,
    }
}

/// ADR 0005 §2.1 schedule: `LearningHistories` ramps linearly from 0.25 to
/// 0.50 of the physical rows; the legacy streams share the remainder in their
/// ADR 0003 ratios. Totals one at every progress. `SyntheticShards` stays at
/// weight 0 (ADR 0005 §2.6) until the shard generator passes the §5.1
/// memorization diagnostic on its own held-out twins.
pub fn adaptation_v6_stream_schedule(progress: f32) -> MixedStreamProportions {
    let progress = if progress.is_finite() {
        progress.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let learning_histories = 0.25 + 0.25 * progress;
    let legacy = foundation_v2_stream_schedule(progress).normalized();
    let share = 1.0 - learning_histories;
    MixedStreamProportions {
        random_one_step: legacy.random_one_step * share,
        factual_branches: legacy.factual_branches * share,
        exploration: legacy.exploration * share,
        sequential_fragments: legacy.sequential_fragments * share,
        hazard_one_step: legacy.hazard_one_step * share,
        learning_histories,
        synthetic_shards: 0.0,
    }
}

/// ACTION5/ACTION6 operator families used across train and held-out episodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorFamily {
    /// v4 ACTION6 behavior.
    Teleport,
    /// v4 ACTION5 behavior.
    Toggle,
    Paint,
    PushLine,
    SwapRegion,
}

impl OperatorFamily {
    pub const ALL: [Self; 5] = [
        Self::Teleport,
        Self::Toggle,
        Self::Paint,
        Self::PushLine,
        Self::SwapRegion,
    ];

    /// Stable nonzero token; token zero is reserved for UNKNOWN.
    pub const fn conditioning_token(self) -> usize {
        match self {
            Self::Teleport => 1,
            Self::Toggle => 2,
            Self::Paint => 3,
            Self::PushLine => 4,
            Self::SwapRegion => 5,
        }
    }
}

/// Episode-level operator split. Entire families, never individual rows, are
/// assigned to either train or operator-held-out evaluation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorFamilySplit {
    pub train: Vec<OperatorFamily>,
    pub held_out: Vec<OperatorFamily>,
}

impl Default for OperatorFamilySplit {
    fn default() -> Self {
        Self {
            train: vec![
                OperatorFamily::Teleport,
                OperatorFamily::Toggle,
                OperatorFamily::Paint,
                OperatorFamily::PushLine,
            ],
            held_out: vec![OperatorFamily::SwapRegion],
        }
    }
}

impl OperatorFamilySplit {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.train.is_empty(), "operator train split is empty");
        ensure!(
            self.train.contains(&OperatorFamily::Teleport)
                && self.train.contains(&OperatorFamily::Toggle),
            "the two v4 operators (teleport and toggle) must remain in-distribution"
        );
        let train = self
            .train
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let held_out = self
            .held_out
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        ensure!(
            train.len() == self.train.len() && held_out.len() == self.held_out.len(),
            "operator split contains duplicate families"
        );
        ensure!(
            train.is_disjoint(&held_out),
            "an operator family cannot be both train and held out"
        );
        let covered = train.union(&held_out).copied().collect::<Vec<_>>();
        ensure!(
            covered == OperatorFamily::ALL,
            "operator split must cover teleport, toggle, paint, push-line, and swap-region"
        );
        Ok(())
    }
}

/// Data/evaluation populations added by the v5 geometry and operator contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum V5DataSplit {
    Train,
    UnseenSeed7x7,
    Composition8x8,
    Translated7x7,
    Size16x16,
    HeldOutOperator(OperatorFamily),
}

impl V5DataSplit {
    fn generation_split(self) -> Split {
        match self {
            Self::Composition8x8 => Split::HeldOutComposition,
            _ => Split::Train,
        }
    }

    fn reported_split(self) -> Split {
        match self {
            Self::Train => Split::Train,
            _ => Split::HeldOutComposition,
        }
    }
}

/// Exact semantic content rectangle inside the 64x64 observation canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentRect {
    pub x: u8,
    pub y: u8,
    pub width: u8,
    pub height: u8,
}

impl ContentRect {
    pub fn validate(self) -> Result<()> {
        ensure!(self.width > 0 && self.height > 0, "content rect is empty");
        ensure!(
            usize::from(self.x) + usize::from(self.width) <= FRAME_SIDE,
            "content rect exceeds canvas width"
        );
        // v6 layouts may occupy row 63 (ADR 0005 §1.3); legacy layouts never
        // sample a rectangle reaching it.
        ensure!(
            usize::from(self.y) + usize::from(self.height) <= FRAME_SIDE,
            "content rect exceeds canvas height"
        );
        Ok(())
    }

    pub fn contains(self, x: u8, y: u8) -> bool {
        x >= self.x
            && y >= self.y
            && x < self.x.saturating_add(self.width)
            && y < self.y.saturating_add(self.height)
    }
}

/// Explicit PAD-vs-EMPTY discriminator consumed by every v5 loss.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentMask {
    /// Row-major 64x64 values in `{0,1}`. Row 63 is always zero for legacy
    /// rows; v6 rows are all ones (Whole-Frame Content, ADR 0005 §1.1).
    pub values: SharedBytes,
}

impl ContentMask {
    /// Whole-Frame Content mask: every pixel is board content.
    pub fn all_ones() -> Self {
        Self {
            values: vec![1; FRAME_SIDE * FRAME_SIDE].into(),
        }
    }

    fn is_all_ones(&self) -> bool {
        self.values.len() == FRAME_SIDE * FRAME_SIDE && self.values.iter().all(|&value| value == 1)
    }

    pub fn from_rect(rect: ContentRect) -> Result<Self> {
        rect.validate()?;
        let mut values = vec![0; FRAME_SIDE * FRAME_SIDE];
        for y in usize::from(rect.y)..usize::from(rect.y + rect.height) {
            let start = y * FRAME_SIDE + usize::from(rect.x);
            values[start..start + usize::from(rect.width)].fill(1);
        }
        Ok(Self {
            values: values.into(),
        })
    }

    pub fn as_f32(&self) -> Vec<f32> {
        self.values.iter().map(|&value| f32::from(value)).collect()
    }

    fn matches_rect(&self, rect: ContentRect) -> bool {
        self.values.len() == FRAME_SIDE * FRAME_SIDE
            && self.values.iter().enumerate().all(|(index, &value)| {
                let x = (index % FRAME_SIDE) as u8;
                let y = (index / FRAME_SIDE) as u8;
                value == u8::from(rect.contains(x, y))
            })
    }
}

/// Eight symmetries of a square board.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum D4Transform {
    Identity,
    Rotate90,
    Rotate180,
    Rotate270,
    ReflectVertical,
    ReflectHorizontal,
    ReflectMainDiagonal,
    ReflectAntiDiagonal,
}

impl D4Transform {
    pub const ALL: [Self; 8] = [
        Self::Identity,
        Self::Rotate90,
        Self::Rotate180,
        Self::Rotate270,
        Self::ReflectVertical,
        Self::ReflectHorizontal,
        Self::ReflectMainDiagonal,
        Self::ReflectAntiDiagonal,
    ];

    /// Transform a local coordinate in a square of side `side`.
    pub fn transform_point(self, x: u8, y: u8, side: u8) -> (u8, u8) {
        let last = side - 1;
        match self {
            Self::Identity => (x, y),
            Self::Rotate90 => (last - y, x),
            Self::Rotate180 => (last - x, last - y),
            Self::Rotate270 => (y, last - x),
            Self::ReflectVertical => (last - x, y),
            Self::ReflectHorizontal => (x, last - y),
            Self::ReflectMainDiagonal => (y, x),
            Self::ReflectAntiDiagonal => (last - y, last - x),
        }
    }

    fn transform_vector(self, dx: i8, dy: i8) -> (i8, i8) {
        match self {
            Self::Identity => (dx, dy),
            Self::Rotate90 => (-dy, dx),
            Self::Rotate180 => (-dx, -dy),
            Self::Rotate270 => (dy, -dx),
            Self::ReflectVertical => (-dx, dy),
            Self::ReflectHorizontal => (dx, -dy),
            Self::ReflectMainDiagonal => (dy, dx),
            Self::ReflectAntiDiagonal => (-dy, -dx),
        }
    }
}

/// Seeded augmentation applied consistently to both sides of a transition and
/// to every branch in a same-state group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymmetryAugmentation {
    pub d4: D4Transform,
    /// A bijection of `0..=15`. Legacy rows fix entry zero; v6 rows permute
    /// every index (free background colour, ADR 0005 §1.2).
    pub color_permutation: [u8; 16],
}

/// Color-aware parameters for applying an episode operator after augmentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeOperator {
    pub family: OperatorFamily,
    pub agent_color: u8,
    pub primary_color: u8,
    pub secondary_color: u8,
    /// Rendered colour of semantic EMPTY. Zero for every legacy row; v6 rows
    /// carry the permuted EMPTY so the operator replays on augmented frames.
    #[serde(default, skip_serializing_if = "is_zero_u8")]
    pub empty_color: u8,
}

/// V5-only provenance sidecar. It avoids changing the legacy provenance struct
/// while carrying the exact translated rectangle required by losses/evaluation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct V5SampleProvenance {
    pub source: TransitionProvenance,
    pub content_rect: ContentRect,
    pub data_split: V5DataSplit,
    pub stream: MixedStreamKind,
    pub operator: EpisodeOperator,
    pub augmentation: SymmetryAugmentation,
    pub goal_dropped: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_group_id: Option<BranchGroupId>,
    /// Row rendered under the ADR 0005 §1 interface contract: whole-frame
    /// content, free background colour, no status row, UNKNOWN conditioning.
    #[serde(default, skip_serializing_if = "is_false")]
    pub contract_v6: bool,
}

impl V5SampleProvenance {
    /// Operator the model may be conditioned on. v6 rows always condition as
    /// UNKNOWN (ADR 0005 §1.4); the sidecar `operator` remains available for
    /// counterfactual replay and censuses.
    pub fn conditioning_operator(&self) -> Option<EpisodeOperator> {
        (!self.contract_v6).then_some(self.operator)
    }
}

/// One v5 transition plus its mandatory content mask and augmentation metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct V5Sample {
    pub transition: TransitionSample,
    pub content_mask: ContentMask,
    pub provenance: V5SampleProvenance,
    #[serde(skip)]
    original_goal_nonzero: bool,
}

impl V5Sample {
    pub fn transition(&self) -> &TransitionSample {
        &self.transition
    }

    pub fn into_transition(self) -> TransitionSample {
        self.transition
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_row(true, true)
    }

    fn validate_after_unit_fields(&self) -> Result<()> {
        self.validate_row(false, false)
    }

    fn validate_row(&self, validate_mask: bool, validate_permutation: bool) -> Result<()> {
        self.provenance.source.validate()?;
        self.provenance.content_rect.validate()?;
        ensure!(
            self.transition.provenance == self.provenance.source,
            "v5 sidecar/source provenance mismatch"
        );
        ensure!(
            self.transition.provenance.operator == self.provenance.conditioning_operator(),
            "v5 transition/operator provenance mismatch"
        );
        ensure!(
            self.provenance.source.content_width == u16::from(self.provenance.content_rect.width)
                && self.provenance.source.content_height
                    == u16::from(self.provenance.content_rect.height),
            "v5 content rect size does not match source provenance"
        );
        ensure!(
            usize::from(self.provenance.source.context_len) == self.transition.context.len()
                && self.transition.context.len() <= CONTEXT_WINDOW_MAX,
            "context_len provenance does not match the context window"
        );
        if validate_mask {
            ensure!(
                if self.provenance.contract_v6 {
                    self.content_mask.is_all_ones()
                } else {
                    self.content_mask.matches_rect(self.provenance.content_rect)
                },
                "v5 content mask does not match provenance rect"
            );
        }
        if validate_permutation {
            validate_color_permutation(
                &self.provenance.augmentation.color_permutation,
                !self.provenance.contract_v6,
            )?;
        }
        for color in [
            self.provenance.operator.agent_color,
            self.provenance.operator.primary_color,
            self.provenance.operator.secondary_color,
        ] {
            ensure!(color <= 15, "operator color is outside palette");
        }
        if self.transition.action.id == 6 {
            ensure!(
                self.provenance.content_rect.contains(
                    self.transition
                        .action
                        .x
                        .ok_or_else(|| anyhow!("ACTION6 missing x"))?,
                    self.transition
                        .action
                        .y
                        .ok_or_else(|| anyhow!("ACTION6 missing y"))?
                ),
                "ACTION6 target is outside v5 content rect"
            );
        }
        Ok(())
    }
}

fn validate_color_permutation(permutation: &[u8; 16], fixed_empty: bool) -> Result<()> {
    let mut seen = [false; 16];
    for &color in permutation {
        let Some(slot) = seen.get_mut(usize::from(color)) else {
            bail!("v5 color permutation contains value outside 0..=15");
        };
        ensure!(!*slot, "v5 color permutation contains a duplicate value");
        *slot = true;
    }
    ensure!(
        (permutation[0] == 0 || !fixed_empty) && seen.into_iter().all(|value| value),
        "v5 color permutation must be a bijection with color 0 fixed"
    );
    Ok(())
}

/// Public composer configuration consumed by foundation-v2 training.
#[derive(Clone, Debug)]
pub struct MixedStreamConfig {
    pub batch_size: usize,
    pub seed: u64,
    pub schedule: fn(progress: f32) -> MixedStreamProportions,
    pub goal_dropout_probability: f32,
    pub operator_families: OperatorFamilySplit,
    pub symmetry_augmentation: bool,
    /// Render every row under the ADR 0005 §1 interface contract and allow
    /// the `LearningHistories` stream. Off keeps every legacy path
    /// byte-identical.
    pub data_contract_v6: bool,
    /// Directory of ADR 0005 §2.6 ARCEngine shards (`manifest.json` plus
    /// `*.safetensors`). Required, together with `data_contract_v6`, by any
    /// schedule that gives `SyntheticShards` non-zero weight; `None` for every
    /// shipped recipe.
    pub synthetic_shards_dir: Option<PathBuf>,
}

impl Default for MixedStreamConfig {
    fn default() -> Self {
        Self {
            batch_size: 2_048,
            seed: 0,
            schedule: foundation_v2_stream_schedule,
            goal_dropout_probability: V5_GOAL_DROPOUT_PROBABILITY,
            operator_families: OperatorFamilySplit::default(),
            symmetry_augmentation: true,
            data_contract_v6: false,
            synthetic_shards_dir: None,
        }
    }
}

impl MixedStreamConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.batch_size > FACTUAL_BRANCHES_PER_GROUP,
            "mixed batch must fit a complete factual group plus at least one \
             row from another stream; a single-group batch degenerates to \
             100% factual"
        );
        ensure!(
            (0.0..=1.0).contains(&self.goal_dropout_probability),
            "goal dropout probability must be in 0..=1"
        );
        self.operator_families.validate()?;
        let proportions = (self.schedule)(0.0);
        ensure!(
            proportions
                .ordered()
                .iter()
                .all(|(_, value)| value.is_finite() && *value >= 0.0),
            "mixed stream schedule returned invalid weights"
        );
        // The schedule is progress-dependent and intact-group rounding is
        // nonlinear, so a batch size can satisfy the tolerance at progress 0
        // yet violate it later and abort mid-run; validate across the range.
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0] {
            ensure!(
                self.data_contract_v6 || (self.schedule)(progress).learning_histories == 0.0,
                "the learning-histories stream requires data_contract_v6"
            );
            ensure!(
                (self.schedule)(progress).synthetic_shards == 0.0
                    || (self.data_contract_v6 && self.synthetic_shards_dir.is_some()),
                "the synthetic-shards stream requires data_contract_v6 and synthetic_shards_dir"
            );
            self.realized_proportions(progress)?;
        }
        Ok(())
    }

    /// Exact fixed-batch realization of the scheduled stream proportions.
    pub fn realized_proportions(&self, progress: f32) -> Result<RealizedStreamProportions> {
        let scheduled = (self.schedule)(progress);
        ensure!(
            scheduled
                .ordered()
                .iter()
                .all(|(_, weight)| weight.is_finite() && *weight >= 0.0),
            "mixed stream schedule returned invalid weights"
        );
        realized_stream_proportions(self.batch_size, scheduled)
    }
}

/// One supervised transition for world-model / event-head training.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitionProvenance {
    /// Width of the semantic board region. Pixels outside this rectangle are padding/UI.
    pub content_width: u16,
    /// Height of the semantic board region. The ARC status row is never part of this region.
    pub content_height: u16,
    /// Canvas x-origin of the content rectangle. Zero for legacy
    /// origin-aligned populations; translated V5 placements record the exact
    /// sampled origin so downstream masks never revert to top-left.
    #[serde(default)]
    pub content_x: u16,
    /// Canvas y-origin of the content rectangle (see `content_x`).
    #[serde(default)]
    pub content_y: u16,
    /// Generator/import population that produced the transition (stable across goal retargeting).
    pub source_kind: String,
    /// Stable trajectory identity. Unlike `family`, this does not change when a goal is retargeted.
    pub trajectory_id: String,
    /// Episode operator after row-level color conjugation. Legacy and real
    /// rows omit it and are conditioned as an unknown rule with neutral colors.
    /// v6 rows also omit it: rule identity reaches the model only through the
    /// context window (ADR 0005 §1.4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operator: Option<EpisodeOperator>,
    /// Hash of the Hidden Rule (ADR 0005 §2.4). Zero for legacy rows.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub rule_id: u64,
    /// Level of the meta-episode the row belongs to (ADR 0005 §2.4).
    #[serde(default, skip_serializing_if = "is_zero_u16")]
    pub level_index: u16,
    /// Bitmask of RESET, ACTION1..7 (ADR 0005 §1.6). v6 rows record all
    /// eight; legacy rows leave it unrecorded (zero).
    #[serde(default, skip_serializing_if = "is_zero_u8")]
    pub available_actions: u8,
    /// Number of context transitions carried by the row (ADR 0005 §2.4).
    #[serde(default, skip_serializing_if = "is_zero_u8")]
    pub context_len: u8,
    /// Rendered colour of semantic EMPTY (ADR 0005 §1.2). Zero for legacy rows;
    /// v6 rows carry the permuted EMPTY so no consumer treats index 0 as
    /// background.
    #[serde(default, skip_serializing_if = "is_zero_u8")]
    pub background_color: u8,
}

/// Rows of a frame that are board content: 63 under the legacy status-row
/// contract, all 64 under Whole-Frame Content (ADR 0005 §1.1).
pub const fn gameplay_rows(whole_frame: bool) -> usize {
    if whole_frame {
        FRAME_SIDE
    } else {
        FRAME_SIDE - 1
    }
}

/// All eight ARC-AGI-3 actions (RESET, ACTION1..7) available.
pub const ALL_ACTIONS_AVAILABLE: u8 = 0xFF;

impl TransitionProvenance {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=FRAME_SIDE as u16).contains(&self.content_width),
            "content_width must be in 1..={FRAME_SIDE}"
        );
        ensure!(
            (1..=FRAME_SIDE as u16).contains(&self.content_height),
            "content_height must be in 1..={FRAME_SIDE}"
        );
        // u32 arithmetic: u16 addition would wrap in release for corrupt
        // deserialized values and let an out-of-frame rectangle validate.
        ensure!(
            u32::from(self.content_x) + u32::from(self.content_width) <= FRAME_SIDE as u32,
            "content rectangle exceeds the canvas width"
        );
        ensure!(
            u32::from(self.content_y) + u32::from(self.content_height) <= FRAME_SIDE as u32,
            "content rectangle exceeds the canvas height"
        );
        ensure!(
            usize::from(self.context_len) <= CONTEXT_WINDOW_MAX,
            "context_len exceeds CONTEXT_WINDOW_MAX"
        );
        ensure!(
            !self.source_kind.is_empty(),
            "source_kind must not be empty"
        );
        ensure!(
            !self.trajectory_id.is_empty(),
            "trajectory_id must not be empty"
        );
        if let Some(operator) = self.operator {
            for color in [
                operator.agent_color,
                operator.primary_color,
                operator.secondary_color,
            ] {
                ensure!(color < 16, "operator color is outside palette");
            }
        }
        Ok(())
    }

    pub(crate) fn simulator(scenario: &Scenario, source_kind: impl Into<String>) -> Self {
        let source_kind = source_kind.into();
        Self {
            content_width: u16::from(scenario.width),
            content_height: u16::from(scenario.height),
            content_x: 0,
            content_y: 0,
            trajectory_id: format!(
                "sim/{source_kind}/{:?}/{}/{}",
                scenario.split, scenario.seed, scenario.episode_id
            ),
            source_kind,
            operator: None,
            rule_id: 0,
            level_index: 0,
            available_actions: 0,
            context_len: 0,
            background_color: 0,
        }
    }

    pub(crate) fn full_frame(seed: u64, episode_id: u64, split: Split, source_kind: &str) -> Self {
        Self {
            content_width: FRAME_SIDE as u16,
            content_height: (FRAME_SIDE - 1) as u16,
            content_x: 0,
            content_y: 0,
            source_kind: source_kind.into(),
            trajectory_id: format!("synthetic/{source_kind}/{split:?}/{seed}/{episode_id}"),
            operator: None,
            rule_id: 0,
            level_index: 0,
            available_actions: 0,
            context_len: 0,
            background_color: 0,
        }
    }
}

/// Maximum number of earlier factual transitions supplied as evidence about the
/// hidden rule (ADR 0005 §1.5).
pub const CONTEXT_WINDOW_MAX: usize = 16;

/// One earlier factual transition from the same episode, supplied to the model as
/// evidence about the Hidden Rule (ADR 0005 §1.5). Never a prediction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextTransition {
    pub current: ArcFrame,
    pub action: ArcAction,
    pub next: ArcFrame,
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
    /// Context Window: earlier factual transitions of the same episode, chronological
    /// order, at most [`CONTEXT_WINDOW_MAX`]. Empty for legacy rows (ADR 0005 §1.5).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context: Vec<ContextTransition>,
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

/// Four directional actions, ACTION5, four stratified ACTION6 coordinates, and
/// ACTION7. Keeping the fixed v5 contract here prevents physical batching from
/// silently truncating a same-state comparison.
pub const FACTUAL_BRANCHES_PER_GROUP: usize = 10;

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
    #[serde(skip, default)]
    pairwise_board_effect_labels: Vec<PairwiseBoardEffectLabel>,
}

/// One status-row-free pair label for the v5 separation/pull objectives.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairwiseBoardEffectLabel {
    pub group_index: usize,
    pub left_row: usize,
    pub right_row: usize,
    pub equivalent: bool,
    pub left_changed: bool,
    pub right_changed: bool,
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
        let pairwise_board_effect_labels = groups
            .iter()
            .zip(&group_ranges)
            .enumerate()
            .flat_map(|(group_index, (group, range))| {
                (0..group.branches.len()).flat_map(move |left| {
                    (left + 1..group.branches.len()).map(move |right| {
                        let left_branch = &group.branches[left];
                        let right_branch = &group.branches[right];
                        PairwiseBoardEffectLabel {
                            group_index,
                            left_row: range.start + left,
                            right_row: range.start + right,
                            equivalent: left_branch.outcome_equivalent(right_branch),
                            left_changed: left_branch.board_effect.changed,
                            right_changed: right_branch.board_effect.changed,
                        }
                    })
                })
            })
            .collect();
        Ok(Self {
            groups,
            group_ids,
            rows,
            group_ranges,
            pairwise_board_effect_labels,
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

    /// Pair labels are computed from board rows only; status UI never enters
    /// equivalence or changed/no-effect labels.
    pub fn pairwise_board_effect_labels(&self) -> &[PairwiseBoardEffectLabel] {
        &self.pairwise_board_effect_labels
    }
}

impl FactualActionBranch {
    /// Legacy rows: row 63 is the status strip and never enters the board effect.
    pub fn try_from_transition(transition: TransitionSample) -> Result<Self> {
        Self::try_from_transition_with_board(transition, (FRAME_SIDE - 1) * FRAME_SIDE)
    }

    /// v6 rows: every one of the 4096 pixels is board content (ADR 0005 §1.1).
    pub fn try_from_transition_whole_frame(transition: TransitionSample) -> Result<Self> {
        Self::try_from_transition_with_board(transition, FRAME_SIDE * FRAME_SIDE)
    }

    fn try_from_transition_with_board(
        transition: TransitionSample,
        status_start: usize,
    ) -> Result<Self> {
        ensure!(
            transition.current.width as usize == FRAME_SIDE
                && transition.current.height as usize == FRAME_SIDE
                && transition.next.width as usize == FRAME_SIDE
                && transition.next.height as usize == FRAME_SIDE,
            "factual branches require fixed {FRAME_SIDE}x{FRAME_SIDE} frames"
        );
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
    pub fn try_new(branches: Vec<FactualActionBranch>) -> Result<Self> {
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
                transition.split == first.split && transition.provenance == first.provenance,
                "all factual branches must share exact split/content/trajectory provenance"
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

    pub fn effect_equivalence_matrix(&self) -> Vec<Vec<bool>> {
        self.branches
            .iter()
            .map(|left| {
                self.branches
                    .iter()
                    .map(|right| left.outcome_equivalent(right))
                    .collect()
            })
            .collect()
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
fn render_state_pixels(scenario: &Scenario, state: &State, stride: usize, len: usize) -> Vec<u8> {
    let mut pixels = vec![palette::EMPTY; len];

    let put = |pixels: &mut [u8], p: Pos, color: u8| {
        if p.x >= 0 && p.y >= 0 && (p.x as u8) < scenario.width && (p.y as u8) < scenario.height {
            pixels[p.y as usize * stride + p.x as usize] = color;
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
    pixels
}

pub fn render_state(scenario: &Scenario, state: &State) -> Result<ArcFrame> {
    let width = scenario.width as usize;
    let height = scenario.height as usize;
    ArcFrame::new(
        scenario.width as u16,
        scenario.height as u16,
        render_state_pixels(scenario, state, width, width * height),
    )
}

/// Render native grid, pad to official `64×64`, and draw per-episode status UI in the
/// margin (ARC-AGI-3 frames embed budget counters in pixels; placement varies by game).
pub fn render_state_padded(scenario: &Scenario, state: &State) -> Result<ArcFrame> {
    ensure!(
        scenario.width as usize <= FRAME_SIDE && scenario.height as usize <= FRAME_SIDE,
        "cannot render scenario {}x{} into {}x{} without interpolation/crop",
        scenario.width,
        scenario.height,
        FRAME_SIDE,
        FRAME_SIDE
    );
    let mut frame = ArcFrame::new(
        FRAME_SIDE as u16,
        FRAME_SIDE as u16,
        render_state_pixels(scenario, state, FRAME_SIDE, FRAME_SIDE * FRAME_SIDE),
    )?;
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
    Ok(sample_from_rendered_transition(
        scenario,
        before,
        after,
        action,
        goal,
        transition_index,
        current,
        next,
    ))
}

fn sample_from_rendered_transition(
    scenario: &Scenario,
    before: &State,
    after: &State,
    action: Action,
    goal: &Goal,
    transition_index: u64,
    current: ArcFrame,
    next: ArcFrame,
) -> TransitionSample {
    let goal_satisfied = goal_satisfied(scenario, after, goal);
    let goal_failed = goal_terminal_failure(scenario, after, goal);
    let exhausted = !goal_satisfied && !goal_failed && after.actions_used >= scenario.action_budget;
    TransitionSample {
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
        context: Vec::new(),
    }
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
    Ok(sample_from_rendered_transition_goal_free(
        scenario,
        before,
        action,
        family,
        transition_index,
        current,
        next,
    ))
}

fn sample_from_rendered_transition_goal_free(
    scenario: &Scenario,
    before: &State,
    action: Action,
    family: &str,
    transition_index: u64,
    current: ArcFrame,
    next: ArcFrame,
) -> TransitionSample {
    TransitionSample {
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
        context: Vec::new(),
    }
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

fn split_tag(split: V5DataSplit) -> u64 {
    match split {
        V5DataSplit::Train => 0x5452_4149_4E00_0005,
        V5DataSplit::UnseenSeed7x7 => 0x554E_5345_454E_0707,
        V5DataSplit::Composition8x8 => 0x434F_4D50_0808_0005,
        V5DataSplit::Translated7x7 => 0x5452_414E_5307_0705,
        V5DataSplit::Size16x16 => 0x5349_5A45_1616_0005,
        V5DataSplit::HeldOutOperator(family) => 0x4F50_484F_4C44_0005 ^ ((family as u64) << 48),
    }
}

pub(crate) fn seeded_v5_rng(
    seed: u64,
    episode_id: u64,
    split: V5DataSplit,
    lane: u64,
) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(
        seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ episode_id.wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ split_tag(split)
            ^ lane.wrapping_mul(0x94D0_49BB_1331_11EB),
    )
}

fn sampled_content_size(split: V5DataSplit, rng: &mut ChaCha8Rng) -> u8 {
    match split {
        V5DataSplit::Train | V5DataSplit::HeldOutOperator(_) => {
            // Integer approximation to a log-skewed distribution. The weights
            // decrease monotonically with log(size), keeping small boards common
            // without starving cross-patch 16/24/32 geometry.
            const WEIGHTS: [u32; 7] = [32, 25, 18, 13, 8, 4, 2];
            let draw = rng.random_range(0..WEIGHTS.iter().sum::<u32>());
            let mut cumulative = 0;
            for (size, weight) in V5_CONTENT_SIZES.into_iter().zip(WEIGHTS) {
                cumulative += weight;
                if draw < cumulative {
                    return size;
                }
            }
            V5_CONTENT_SIZES[V5_CONTENT_SIZES.len() - 1]
        }
        V5DataSplit::UnseenSeed7x7 | V5DataSplit::Translated7x7 => 7,
        V5DataSplit::Composition8x8 => 8,
        V5DataSplit::Size16x16 => 16,
    }
}

fn sampled_content_rect(
    size: u8,
    split: V5DataSplit,
    rng: &mut ChaCha8Rng,
    whole_frame: bool,
) -> ContentRect {
    let max_x = FRAME_SIDE as u8 - size;
    // v6 layouts have no reserved status row: content may reach row 63.
    let max_y = if whole_frame {
        FRAME_SIDE as u8 - size
    } else {
        V5_PLAYFIELD_HEIGHT as u8 - size
    };
    let mut x = rng.random_range(0..=max_x);
    let mut y = rng.random_range(0..=max_y);
    if split == V5DataSplit::Translated7x7 && x == 0 && y == 0 {
        if max_x > 0 {
            x = 1;
        } else if max_y > 0 {
            y = 1;
        }
    }
    ContentRect {
        x,
        y,
        width: size,
        height: size,
    }
}

fn sampled_augmentation(
    rng: &mut ChaCha8Rng,
    enabled: bool,
    free_background: bool,
) -> SymmetryAugmentation {
    let d4 = if enabled {
        D4Transform::ALL[rng.random_range(0..D4Transform::ALL.len())]
    } else {
        D4Transform::Identity
    };
    let mut color_permutation = std::array::from_fn(|index| index as u8);
    if enabled {
        // v6: uniform permutation of all sixteen colours (ADR 0005 §1.2).
        // Legacy: colour 0 fixed.
        let lowest = if free_background { 0 } else { 1 };
        for index in (lowest + 1..16).rev() {
            let other = rng.random_range(lowest..=index);
            color_permutation.swap(index, other);
        }
    }
    SymmetryAugmentation {
        d4,
        color_permutation,
    }
}

fn permute_operator(operator: EpisodeOperator, color_permutation: &[u8; 16]) -> EpisodeOperator {
    EpisodeOperator {
        family: operator.family,
        agent_color: color_permutation[operator.agent_color as usize],
        primary_color: color_permutation[operator.primary_color as usize],
        secondary_color: color_permutation[operator.secondary_color as usize],
        empty_color: color_permutation[operator.empty_color as usize],
    }
}

fn direction_action_id(dx: i8, dy: i8) -> Result<u8> {
    match (dx, dy) {
        (0, -1) => Ok(1),
        (0, 1) => Ok(2),
        (-1, 0) => Ok(3),
        (1, 0) => Ok(4),
        _ => bail!("transformed action is not cardinal: ({dx},{dy})"),
    }
}

/// Relabel a directional action or transform an ACTION6 coordinate from one
/// square content rectangle into another.
pub fn conjugate_action(
    action: &ArcAction,
    transform: D4Transform,
    source_rect: ContentRect,
    target_rect: ContentRect,
) -> Result<ArcAction> {
    source_rect.validate()?;
    target_rect.validate()?;
    ensure!(
        source_rect.width == source_rect.height
            && target_rect.width == target_rect.height
            && source_rect.width == target_rect.width,
        "D4 action conjugation requires equal square content rectangles"
    );
    match action.id {
        1..=4 => {
            let (dx, dy) = match action.id {
                1 => (0, -1),
                2 => (0, 1),
                3 => (-1, 0),
                4 => (1, 0),
                _ => unreachable!(),
            };
            let (dx, dy) = transform.transform_vector(dx, dy);
            ArcAction::new(direction_action_id(dx, dy)?, None, None)
        }
        6 => {
            let x = action.x.ok_or_else(|| anyhow!("ACTION6 is missing x"))?;
            let y = action.y.ok_or_else(|| anyhow!("ACTION6 is missing y"))?;
            ensure!(
                source_rect.contains(x, y),
                "ACTION6 coordinate is outside source content rect"
            );
            let local_x = x - source_rect.x;
            let local_y = y - source_rect.y;
            let (x, y) = transform.transform_point(local_x, local_y, source_rect.width);
            ArcAction::new(6, Some(target_rect.x + x), Some(target_rect.y + y))
        }
        id => ArcAction::new(id, None, None),
    }
}

thread_local! {
    static FRAME_TRANSFORM_SCRATCH: RefCell<Vec<u8>> =
        RefCell::new(vec![palette::PAD; FRAME_SIDE * FRAME_SIDE]);
}

fn frame_with_transformed_content(
    source: &mut ArcFrame,
    source_rect: ContentRect,
    target_rect: ContentRect,
    augmentation: &SymmetryAugmentation,
    whole_frame: bool,
) -> Result<()> {
    ensure!(
        source.width as usize == FRAME_SIDE && source.height as usize == FRAME_SIDE,
        "v5 augmentation requires fixed-size source frames"
    );
    source_rect.validate()?;
    target_rect.validate()?;
    ensure!(
        source_rect.width == source_rect.height
            && target_rect.width == source_rect.width
            && target_rect.height == source_rect.height,
        "v5 D4 augmentation requires equal square source/target rectangles"
    );
    FRAME_TRANSFORM_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        // v6 padding is rendered as the permuted EMPTY colour so live frames
        // with arbitrary backgrounds and synthetic frames share one
        // distribution (ADR 0005 §1.2).
        scratch.fill(if whole_frame {
            augmentation.color_permutation[usize::from(palette::EMPTY)]
        } else {
            palette::PAD
        });
        for y in 0..source_rect.height {
            for x in 0..source_rect.width {
                let source_x = usize::from(source_rect.x + x);
                let source_y = usize::from(source_rect.y + y);
                let color = source.pixels[source_y * FRAME_SIDE + source_x];
                let color = augmentation.color_permutation[color as usize];
                let (target_x, target_y) = augmentation.d4.transform_point(x, y, source_rect.width);
                let target_x = usize::from(target_rect.x + target_x);
                let target_y = usize::from(target_rect.y + target_y);
                scratch[target_y * FRAME_SIDE + target_x] = color;
            }
        }
        // Legacy: status UI is copied only after spatial/color augmentation and
        // is never part of the semantic content mask or branch-effect
        // equivalence. v6 paints no status row (ADR 0005 §1.3).
        if !whole_frame {
            let status_start = V5_PLAYFIELD_HEIGHT * FRAME_SIDE;
            scratch[status_start..].copy_from_slice(&source.pixels[status_start..]);
        }
        source.pixels.copy_from_slice(&scratch);
    });
    Ok(())
}

fn transform_frame_once(
    frame: &mut ArcFrame,
    source_rect: ContentRect,
    target_rect: ContentRect,
    augmentation: &SymmetryAugmentation,
    whole_frame: bool,
    cache: &mut BTreeMap<(usize, u16, u16), ArcFrame>,
) -> Result<()> {
    let key = (frame.pixels.allocation_id(), frame.width, frame.height);
    if let Some(transformed) = cache.get(&key) {
        *frame = transformed.clone();
        return Ok(());
    }
    frame_with_transformed_content(frame, source_rect, target_rect, augmentation, whole_frame)?;
    cache.insert(key, frame.clone());
    Ok(())
}

/// Transform the agent-coordinate dimensions of an exact simulator oracle into
/// the augmented content frame. The remaining dimensions are layout-free.
pub fn transform_oracle_latent_d4(
    oracle_latent: &mut Option<Vec<f32>>,
    transform: D4Transform,
    source_rect: ContentRect,
    target_rect: ContentRect,
) -> Result<()> {
    source_rect.validate()?;
    target_rect.validate()?;
    ensure!(
        source_rect.width == source_rect.height
            && target_rect.width == target_rect.height
            && source_rect.width == target_rect.width,
        "oracle D4 transform requires equal square content rectangles"
    );
    let Some(latent) = oracle_latent else {
        return Ok(());
    };
    ensure!(
        latent.len() >= 2,
        "oracle latent is missing agent-coordinate dimensions"
    );
    let side = f32::from(source_rect.width);
    let local = |value: f32| {
        (((value + 1.0) * 0.5 * side) - 0.5)
            .round()
            .clamp(0.0, side - 1.0) as u8
    };
    let (x, y) = transform.transform_point(local(latent[0]), local(latent[1]), source_rect.width);
    latent[0] = (f32::from(x) + 0.5) / f32::from(target_rect.width) * 2.0 - 1.0;
    latent[1] = (f32::from(y) + 0.5) / f32::from(target_rect.height) * 2.0 - 1.0;
    Ok(())
}

fn find_color(frame: &ArcFrame, rect: ContentRect, color: u8) -> Option<(u8, u8)> {
    for y in rect.y..rect.y + rect.height {
        for x in rect.x..rect.x + rect.width {
            if frame.pixels[usize::from(y) * FRAME_SIDE + usize::from(x)] == color {
                return Some((x, y));
            }
        }
    }
    None
}

fn symmetric_coordinate(rect: ContentRect, x: u8, y: u8) -> (u8, u8) {
    (
        rect.x + rect.width - 1 - (x - rect.x),
        rect.y + rect.height - 1 - (y - rect.y),
    )
}

/// Apply the sampled ACTION5/ACTION6 frame operator. It is intentionally
/// status-row-free and color-parameterized so the same function validates an
/// augmented transition after color permutation.
pub fn apply_episode_operator(
    current: &ArcFrame,
    action: &ArcAction,
    content_rect: ContentRect,
    operator: EpisodeOperator,
) -> Result<ArcFrame> {
    content_rect.validate()?;
    ensure!(
        current.width as usize == FRAME_SIDE && current.height as usize == FRAME_SIDE,
        "episode operators require fixed 64x64 frames"
    );
    ensure!(
        matches!(action.id, 5 | 6),
        "episode operators only define ACTION5/ACTION6"
    );
    let coordinate = if action.id == 6 {
        let x = action.x.ok_or_else(|| anyhow!("ACTION6 is missing x"))?;
        let y = action.y.ok_or_else(|| anyhow!("ACTION6 is missing y"))?;
        ensure!(
            content_rect.contains(x, y),
            "operator ACTION6 coordinate is outside content rect"
        );
        Some((x, y))
    } else {
        None
    };
    let index = |x: u8, y: u8| usize::from(y) * FRAME_SIDE + usize::from(x);
    let mut next = current.clone();
    match operator.family {
        OperatorFamily::Teleport => {
            let Some((agent_x, agent_y)) = find_color(current, content_rect, operator.agent_color)
            else {
                return Ok(next);
            };
            let target =
                coordinate.unwrap_or_else(|| symmetric_coordinate(content_rect, agent_x, agent_y));
            if target != (agent_x, agent_y) {
                next.pixels[index(agent_x, agent_y)] = operator.empty_color;
                next.pixels[index(target.0, target.1)] = operator.agent_color;
            }
        }
        OperatorFamily::Toggle => {
            if let Some((x, y)) = coordinate {
                let value = &mut next.pixels[index(x, y)];
                *value = if *value == operator.primary_color {
                    operator.secondary_color
                } else {
                    operator.primary_color
                };
            } else {
                for y in content_rect.y..content_rect.y + content_rect.height {
                    for x in content_rect.x..content_rect.x + content_rect.width {
                        let value = &mut next.pixels[index(x, y)];
                        if *value == operator.primary_color {
                            *value = operator.secondary_color;
                        } else if *value == operator.secondary_color {
                            *value = operator.primary_color;
                        }
                    }
                }
            }
        }
        OperatorFamily::Paint => {
            if let Some((x, y)) = coordinate {
                next.pixels[index(x, y)] = operator.primary_color;
            } else if let Some((agent_x, agent_y)) =
                find_color(current, content_rect, operator.agent_color)
            {
                for (dx, dy) in [(0i8, -1i8), (0, 1), (-1, 0), (1, 0)] {
                    let x = i16::from(agent_x) + i16::from(dx);
                    let y = i16::from(agent_y) + i16::from(dy);
                    if x >= 0 && y >= 0 {
                        let (x, y) = (x as u8, y as u8);
                        if content_rect.contains(x, y)
                            && next.pixels[index(x, y)] == operator.empty_color
                        {
                            next.pixels[index(x, y)] = operator.primary_color;
                        }
                    }
                }
            }
        }
        OperatorFamily::PushLine => {
            if let Some((x, y)) = coordinate {
                let local_x2 =
                    i16::from(2 * (x - content_rect.x) + 1) - i16::from(content_rect.width);
                let local_y2 =
                    i16::from(2 * (y - content_rect.y) + 1) - i16::from(content_rect.height);
                let dx = local_x2.signum();
                let dy = local_y2.signum();
                let destination_x = i16::from(x) + dx;
                let destination_y = i16::from(y) + dy;
                if destination_x >= 0 && destination_y >= 0 {
                    let destination = (destination_x as u8, destination_y as u8);
                    if content_rect.contains(destination.0, destination.1) {
                        next.pixels[index(destination.0, destination.1)] =
                            current.pixels[index(x, y)];
                        next.pixels[index(x, y)] = operator.empty_color;
                    }
                }
            } else {
                // A half-turn pushes every board line through the center and is
                // equivariant under the complete D4 group.
                for y in content_rect.y..content_rect.y + content_rect.height {
                    for x in content_rect.x..content_rect.x + content_rect.width {
                        let symmetric = symmetric_coordinate(content_rect, x, y);
                        next.pixels[index(symmetric.0, symmetric.1)] = current.pixels[index(x, y)];
                    }
                }
            }
        }
        OperatorFamily::SwapRegion => {
            if let Some((center_x, center_y)) = coordinate {
                let symmetric = symmetric_coordinate(content_rect, center_x, center_y);
                let mut swaps = Vec::new();
                for dy in -1i16..=1 {
                    for dx in -1i16..=1 {
                        let left_x = i16::from(center_x) + dx;
                        let left_y = i16::from(center_y) + dy;
                        let right_x = i16::from(symmetric.0) - dx;
                        let right_y = i16::from(symmetric.1) - dy;
                        if [left_x, left_y, right_x, right_y]
                            .into_iter()
                            .all(|value| value >= 0)
                        {
                            let left = (left_x as u8, left_y as u8);
                            let right = (right_x as u8, right_y as u8);
                            if content_rect.contains(left.0, left.1)
                                && content_rect.contains(right.0, right.1)
                            {
                                swaps.push((left, right));
                            }
                        }
                    }
                }
                for (left, right) in swaps {
                    next.pixels
                        .swap(index(left.0, left.1), index(right.0, right.1));
                }
            } else {
                for y in content_rect.y..content_rect.y + content_rect.height {
                    for x in content_rect.x..content_rect.x + content_rect.width {
                        let symmetric = symmetric_coordinate(content_rect, x, y);
                        next.pixels[index(symmetric.0, symmetric.1)] = current.pixels[index(x, y)];
                    }
                }
            }
        }
    }
    Ok(next)
}

fn apply_action(sim: &Simulator, state: &State, action: Action) -> State {
    sim.transition(state, action)
}

fn scenario_for_v5(seed: u64, episode_id: u64, split: V5DataSplit, content_size: u8) -> Scenario {
    let mut scenario = generate_sized(seed, episode_id, split.generation_split(), content_size);
    scenario.split = split.reported_split();
    scenario
}

fn sampled_operator(
    families: &OperatorFamilySplit,
    split: V5DataSplit,
    rng: &mut ChaCha8Rng,
) -> Result<EpisodeOperator> {
    let family = match split {
        V5DataSplit::HeldOutOperator(family) => {
            ensure!(
                families.held_out.contains(&family) && !families.train.contains(&family),
                "requested operator family is not entirely held out"
            );
            family
        }
        _ => *families
            .train
            .choose(rng)
            .expect("validated non-empty operator train split"),
    };
    Ok(EpisodeOperator {
        family,
        agent_color: palette::AGENT,
        primary_color: palette::SWITCH_BASE,
        secondary_color: palette::SWITCH_BASE + 1,
        empty_color: palette::EMPTY,
    })
}

fn clear_and_paint_status(frame: &mut ArcFrame, action_budget: u16, actions_used: u16) {
    let status_start = V5_PLAYFIELD_HEIGHT * FRAME_SIDE;
    frame.pixels[status_start..].fill(palette::PAD);
    paint_status_ui(frame, action_budget, actions_used);
}

fn operator_sample_from_state(
    scenario: &Scenario,
    state: &State,
    action: ArcAction,
    operator: EpisodeOperator,
    family: &str,
    transition_index: u64,
) -> Result<TransitionSample> {
    let current = render_state_padded(scenario, state)?;
    operator_sample_from_rendered_current(
        scenario,
        state,
        current,
        action,
        operator,
        family,
        transition_index,
    )
}

fn operator_sample_from_rendered_current(
    scenario: &Scenario,
    state: &State,
    current: ArcFrame,
    action: ArcAction,
    operator: EpisodeOperator,
    family: &str,
    transition_index: u64,
) -> Result<TransitionSample> {
    let source_rect = ContentRect {
        x: 0,
        y: 0,
        width: scenario.width,
        height: scenario.height,
    };
    let mut next = apply_episode_operator(&current, &action, source_rect, operator)?;
    clear_and_paint_status(
        &mut next,
        scenario.action_budget,
        state.actions_used.saturating_add(1),
    );
    let status_start = V5_PLAYFIELD_HEIGHT * FRAME_SIDE;
    let noop = current.pixels[..status_start] == next.pixels[..status_start];
    Ok(TransitionSample {
        oracle_latent: Some(oracle_latent(scenario, state)),
        context: Vec::new(),
        current,
        next,
        action,
        goal_features: GoalFeatures::zeros(),
        noop: Some(noop),
        goal_satisfied: None,
        goal_failed: None,
        exhausted: Some(false),
        split: scenario.split,
        family: family.into(),
        seed: scenario.seed,
        episode_id: scenario.episode_id,
        transition_index,
        provenance: TransitionProvenance::simulator(scenario, family),
    })
}

fn null_sample_from_state(
    scenario: &Scenario,
    state: &State,
    family: &str,
    transition_index: u64,
) -> Result<TransitionSample> {
    let current = render_state_padded(scenario, state)?;
    Ok(TransitionSample {
        oracle_latent: Some(oracle_latent(scenario, state)),
        context: Vec::new(),
        next: current.clone(),
        current,
        action: ArcAction::new(0, None, None)?,
        goal_features: GoalFeatures::zeros(),
        noop: Some(true),
        goal_satisfied: None,
        goal_failed: None,
        exhausted: None,
        split: scenario.split,
        family: family.into(),
        seed: scenario.seed,
        episode_id: scenario.episode_id,
        transition_index,
        provenance: TransitionProvenance::simulator(scenario, family),
    })
}

fn augment_v5_transition(
    mut transition: TransitionSample,
    split: V5DataSplit,
    stream: MixedStreamKind,
    operator: EpisodeOperator,
    rect: ContentRect,
    augmentation: SymmetryAugmentation,
    content_mask: ContentMask,
    goal_dropout_probability: f32,
    contract_v6: bool,
    dropout_rng: &mut ChaCha8Rng,
    frame_cache: &mut BTreeMap<(usize, u16, u16), ArcFrame>,
) -> Result<V5Sample> {
    let source_rect = ContentRect {
        x: 0,
        y: 0,
        width: u8::try_from(transition.provenance.content_width)
            .map_err(|_| anyhow!("content width does not fit u8"))?,
        height: u8::try_from(transition.provenance.content_height)
            .map_err(|_| anyhow!("content height does not fit u8"))?,
    };
    ensure!(
        source_rect.width == rect.width && source_rect.height == rect.height,
        "augmentation rectangle does not match transition provenance"
    );
    // Learning-history units carry per-row hidden rules because twins share
    // one augmentation unit; every legacy raw row leaves the operator unset.
    let operator = transition.provenance.operator.unwrap_or(operator);
    transform_frame_once(
        &mut transition.current,
        source_rect,
        rect,
        &augmentation,
        contract_v6,
        frame_cache,
    )?;
    transform_frame_once(
        &mut transition.next,
        source_rect,
        rect,
        &augmentation,
        contract_v6,
        frame_cache,
    )?;
    transition.action = conjugate_action(&transition.action, augmentation.d4, source_rect, rect)?;
    // The context window receives the same D4/colour augmentation as the row
    // (ADR 0005 §1.5).
    for context in &mut transition.context {
        transform_frame_once(
            &mut context.current,
            source_rect,
            rect,
            &augmentation,
            contract_v6,
            frame_cache,
        )?;
        transform_frame_once(
            &mut context.next,
            source_rect,
            rect,
            &augmentation,
            contract_v6,
            frame_cache,
        )?;
        context.action = conjugate_action(&context.action, augmentation.d4, source_rect, rect)?;
    }
    // The sampled placement origin becomes part of transition provenance so
    // standalone consumers can rebuild the exact content mask; before this,
    // translated rows silently reverted to top-left masks downstream.
    transition.provenance.content_x = u16::from(rect.x);
    transition.provenance.content_y = u16::from(rect.y);
    transform_oracle_latent_d4(
        &mut transition.oracle_latent,
        augmentation.d4,
        source_rect,
        rect,
    )?;
    let original_goal_nonzero = transition
        .goal_features
        .values
        .iter()
        .any(|&value| value != 0.0);
    let goal_dropped = dropout_rng.random_bool(f64::from(goal_dropout_probability));
    if goal_dropped {
        transition.goal_features = GoalFeatures::zeros();
        // Goal-success/failure labels are candidate-dependent. Once dropout
        // removes that candidate, retaining the labels creates identical
        // observer inputs with contradictory targets.
        transition.goal_satisfied = None;
        transition.goal_failed = None;
    }
    // The V4 encoder deliberately clears the status row, and neither the
    // canonical state nor goal features carry actions-used/action-budget.
    // Per the ADR, exhaustion is therefore deliberately masked until its
    // conditioning is explicitly represented at this observer seam.
    transition.exhausted = None;
    let operator = permute_operator(operator, &augmentation.color_permutation);
    if contract_v6 {
        // Rule identity never enters conditioning (ADR 0005 §1.4); the sidecar
        // below keeps the permuted operator for counterfactual replay.
        transition.provenance.operator = None;
        // Generated rows record all eight actions; shard rows keep the game's
        // own mask (ADR 0005 §1.6).
        if transition.provenance.available_actions == 0 {
            transition.provenance.available_actions = ALL_ACTIONS_AVAILABLE;
        }
        transition.provenance.background_color = operator.empty_color;
        transition.provenance.context_len = u8::try_from(transition.context.len())
            .map_err(|_| anyhow!("context window exceeds u8"))?;
        if transition.provenance.rule_id == 0 {
            transition.provenance.rule_id = hidden_rule_id(
                operator_before_permutation(operator, &augmentation.color_permutation),
                None,
            );
        }
    } else {
        transition.provenance.operator = Some(operator);
    }
    Ok(V5Sample {
        provenance: V5SampleProvenance {
            source: transition.provenance.clone(),
            content_rect: rect,
            data_split: split,
            stream,
            operator,
            augmentation,
            goal_dropped,
            branch_group_id: None,
            contract_v6,
        },
        transition,
        content_mask,
        original_goal_nonzero,
    })
}

fn operator_before_permutation(
    operator: EpisodeOperator,
    color_permutation: &[u8; 16],
) -> EpisodeOperator {
    let mut inverse = [0u8; 16];
    for (index, &color) in color_permutation.iter().enumerate() {
        inverse[usize::from(color)] = index as u8;
    }
    permute_operator(operator, &inverse)
}

/// Stable hash of a Hidden Rule: operator family, raw colour binding, and the
/// goal family when the unit has one (ADR 0005 §2.4). Never zero.
pub fn hidden_rule_id(operator: EpisodeOperator, goal_family: Option<u8>) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for byte in [
        operator.family.conditioning_token() as u8,
        operator.agent_color,
        operator.primary_color,
        operator.secondary_color,
        operator.empty_color,
        goal_family.unwrap_or(u8::MAX),
    ] {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash.max(1)
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
        let sample_episode_id = non_meta_episode_id(episode_id, step as u64)?;
        let start_x = 31usize;
        let start_y = 31usize;
        let (x, y) = loop {
            let candidate = (
                rng.random_range(0..FRAME_SIDE) as u8,
                rng.random_range(0..V5_PLAYFIELD_HEIGHT) as u8,
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
            episode_id: sample_episode_id,
            transition_index: step as u64,
            provenance: TransitionProvenance::full_frame(
                seed,
                sample_episode_id,
                split,
                "coordinate_action",
            ),
            oracle_latent: Some(oracle_latent_from_frame(&current)),
            context: Vec::new(),
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
        let sample_episode_id = non_meta_episode_id(episode_id, step as u64)?;
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
            context: Vec::new(),
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
            episode_id: sample_episode_id,
            transition_index: step as u64,
            provenance: TransitionProvenance::full_frame(
                seed,
                sample_episode_id,
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
        let scenario_episode_id = checked_non_meta_episode_id(
            episode_id
                .checked_add(step as u64)
                .context("hazard episode id overflow")?,
        )?;
        let mut scenario = generate(seed, scenario_episode_id, split);
        if scenario.hazards.is_empty() {
            scenario.hazards.push(Pos::new(2, 2));
        }
        if scenario.markers.is_empty() {
            scenario.markers.push(Pos::new(4, 4));
        }
        let hazard_pos = scenario.hazards[0];
        let candidates = [
            (
                Pos::new(hazard_pos.x - 1, hazard_pos.y),
                Action::Move(Dir::East),
            ),
            (
                Pos::new(hazard_pos.x + 1, hazard_pos.y),
                Action::Move(Dir::West),
            ),
            (
                Pos::new(hazard_pos.x, hazard_pos.y - 1),
                Action::Move(Dir::South),
            ),
            (
                Pos::new(hazard_pos.x, hazard_pos.y + 1),
                Action::Move(Dir::North),
            ),
        ];
        let (start, action) = candidates
            .into_iter()
            .find(|(position, _)| scenario.in_bounds(*position))
            .ok_or_else(|| anyhow!("hazard has no in-bounds neighbor"))?;
        scenario.walls.remove(&start);
        scenario.walls.remove(&hazard_pos);
        scenario.start = start;
        let sim = Simulator::new(scenario.clone());
        let state = State::initial(&scenario);
        let next = apply_action(&sim, &state, action);
        ensure!(
            next.pos == hazard_pos,
            "hazard action fell back to a blocked destination"
        );
        let goal = Goal::AvoidHazardReachMarker {
            hazard: 0,
            marker: 0,
        };
        let mut sample =
            sample_from_transition(&scenario, &state, &next, action, &goal, step as u64)?;
        sample.family = "hazard_failure".into();
        sample.provenance.source_kind = "hazard_failure".into();
        out.push(sample);
    }
    Ok(out)
}

fn stratified_action6_coordinates(current: &ArcFrame, size: u8) -> Result<[(u8, u8); 4]> {
    let pixel = |x: u8, y: u8| current.pixels[usize::from(y) * FRAME_SIDE + usize::from(x)];
    let objects = (0..size)
        .flat_map(|y| (0..size).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            let value = pixel(x, y);
            value != palette::EMPTY && value != palette::AGENT
        })
        .collect::<Vec<_>>();
    let object = objects
        .iter()
        .copied()
        .find(|&(x, y)| (size - 1 - x, size - 1 - y) != (x, y))
        .or_else(|| objects.first().copied())
        .ok_or_else(|| anyhow!("factual board has no object cell"))?;
    let symmetric = (size - 1 - object.0, size - 1 - object.1);
    let boundary = (0..size)
        .flat_map(|offset| {
            [
                (offset, 0),
                (size - 1, offset),
                (size - 1 - offset, size - 1),
                (0, size - 1 - offset),
            ]
        })
        .find(|coordinate| *coordinate != object && *coordinate != symmetric)
        .ok_or_else(|| anyhow!("could not choose distinct boundary coordinate"))?;
    let empty = (0..size)
        .flat_map(|y| (0..size).map(move |x| (x, y)))
        .find(|&(x, y)| {
            pixel(x, y) == palette::EMPTY
                && (x, y) != object
                && (x, y) != symmetric
                && (x, y) != boundary
                && x > 0
                && y > 0
                && x + 1 < size
                && y + 1 < size
        })
        .or_else(|| {
            (0..size)
                .flat_map(|y| (0..size).map(move |x| (x, y)))
                .find(|&(x, y)| {
                    pixel(x, y) == palette::EMPTY
                        && (x, y) != object
                        && (x, y) != symmetric
                        && (x, y) != boundary
                })
        })
        .ok_or_else(|| anyhow!("factual board has no distinct empty coordinate"))?;
    let coordinates = [object, boundary, empty, symmetric];
    ensure!(
        coordinates
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == coordinates.len(),
        "ACTION6 strata must produce distinct coordinates"
    );
    Ok(coordinates)
}

fn generate_v5_factual_branch_group(
    seed: u64,
    episode_id: u64,
    data_split: V5DataSplit,
    content_size: u8,
    operator: EpisodeOperator,
) -> Result<BranchGroup> {
    let mut scenario = scenario_for_v5(seed, episode_id, data_split, content_size);
    scenario.undo_enabled = true;
    let scenario = Arc::new(scenario);
    let sim = Simulator::new(Arc::clone(&scenario));
    let mut state = State::initial(&scenario);
    // Give ACTION7 an applicable undo frame without conditioning the shared
    // observation on any branch action.
    let prefix = Action::Move(Dir::East);
    state = apply_action(&sim, &state, prefix);
    if data_split == V5DataSplit::Composition8x8 {
        // Preserve the frozen held-out within-action balance from ADR 0002:
        // each direction is targeted in turn, alternating traversable/blocked
        // source cells while the complete v5 group still shares one frame.
        let ordinal = episode_id / 2;
        let target_dir = Dir::ALL[ordinal as usize % Dir::ALL.len()];
        let want_changed = (ordinal / Dir::ALL.len() as u64).is_multiple_of(2);
        let target_action = Action::Move(target_dir);
        if let Some(position) = (0..scenario.height as i8)
            .flat_map(|y| (0..scenario.width as i8).map(move |x| Pos::new(x, y)))
            .filter(|position| !scenario.is_blocked(*position))
            .find(|position| {
                let mut candidate = state.clone();
                candidate.pos = *position;
                let next = apply_action(&sim, &candidate, target_action);
                (next.pos != candidate.pos) == want_changed
            })
        {
            state.pos = position;
        }
    }
    let family = if episode_id.is_multiple_of(2) {
        "factual_branch_v5"
    } else {
        "factual_coordinate_branch"
    };
    let mut transitions = Vec::with_capacity(FACTUAL_BRANCHES_PER_GROUP);
    let current = render_state_padded(&scenario, &state)?;
    for action in Action::moves() {
        let next = apply_action(&sim, &state, action);
        let next_frame = render_state_padded(&scenario, &next)?;
        transitions.push(sample_from_rendered_transition_goal_free(
            &scenario,
            &state,
            action,
            family,
            transitions.len() as u64,
            current.clone(),
            next_frame,
        ));
    }
    transitions.push(operator_sample_from_rendered_current(
        &scenario,
        &state,
        current.clone(),
        ArcAction::new(5, None, None)?,
        operator,
        family,
        transitions.len() as u64,
    )?);
    for (x, y) in stratified_action6_coordinates(&current, content_size)? {
        transitions.push(operator_sample_from_rendered_current(
            &scenario,
            &state,
            current.clone(),
            ArcAction::new(6, Some(x), Some(y))?,
            operator,
            family,
            transitions.len() as u64,
        )?);
    }
    let undo = Action::Undo;
    let next = apply_action(&sim, &state, undo);
    let next_frame = render_state_padded(&scenario, &next)?;
    transitions.push(sample_from_rendered_transition_goal_free(
        &scenario,
        &state,
        undo,
        family,
        transitions.len() as u64,
        current,
        next_frame,
    ));
    ensure!(
        transitions.len() == FACTUAL_BRANCHES_PER_GROUP,
        "v5 factual group has wrong branch count"
    );
    let trajectory_id = format!(
        "factual-v5/{data_split:?}/{seed}/{episode_id}/{:?}",
        operator.family
    );
    let branches = transitions
        .into_iter()
        .map(|mut transition| {
            transition.episode_id = episode_id;
            transition.provenance.trajectory_id = trajectory_id.clone();
            FactualActionBranch::try_from_transition(transition)
        })
        .collect::<Result<Vec<_>>>()?;
    BranchGroup::try_new(branches)
}

/// Complete v5 same-state comparison: every applicable simple action plus four
/// stratified ACTION6 coordinates (object, boundary, empty, symmetric).
pub fn generate_factual_branch_group(
    seed: u64,
    episode_id: u64,
    split: Split,
) -> Result<BranchGroup> {
    let data_split = if split == Split::Train {
        V5DataSplit::Train
    } else {
        V5DataSplit::Composition8x8
    };
    let content_size = if split == Split::Train { 7 } else { 8 };
    let families = OperatorFamilySplit::default();
    families.validate()?;
    let mut rng = seeded_v5_rng(seed, episode_id, data_split, 0xFAC7_0005);
    let operator = sampled_operator(&families, data_split, &mut rng)?;
    generate_v5_factual_branch_group(seed, episode_id, data_split, content_size, operator)
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

/// One complete stationary-schedule v5 batch. `samples` is the loss-facing
/// representation; `factual` exposes canonical group rows and pair labels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MixedStreamBatch {
    samples: Vec<V5Sample>,
    factual: Option<FactualBatch>,
    factual_group_ranges: Vec<std::ops::Range<usize>>,
    stream_counts: BTreeMap<MixedStreamKind, usize>,
    scheduled_proportions: MixedStreamProportions,
    realized_proportions: RealizedStreamProportions,
    goal_dropout_census: GoalDropoutCensus,
}

/// Exact row counts and fractions after whole factual branch groups are allocated.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealizedStreamProportions {
    pub normalized_target: MixedStreamProportions,
    pub counts: BTreeMap<MixedStreamKind, usize>,
    pub fractions: MixedStreamProportions,
}

/// Realized scope of goal dropout in one composed batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoalDropoutCensus {
    pub total: usize,
    pub eligible: usize,
    pub changed: usize,
    pub final_zero_goal: usize,
}

/// Label support in a deterministic generated stream, ordered as
/// noop/satisfied/failed/exhausted. This is a premise check for event-head
/// supervision, not a model metric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventLabelCensus {
    pub rows: usize,
    pub labeled: [usize; 4],
    pub positive: [usize; 4],
}

/// One split/family/action bucket in factual branch coverage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchCoverageStratum {
    pub split: Split,
    pub family: String,
    pub action_id: u8,
    pub eligible_rows: usize,
    pub changed_outcomes: usize,
    pub distinct_effect_classes: usize,
}

/// Missing required factual action strata, grouped by source split and family.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissingBranchActionKey {
    pub split: Split,
    pub family: String,
    pub action_id: u8,
}

/// Duplicate factual action tuple inside one same-state branch group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DuplicateBranchActionKey {
    pub split: Split,
    pub family: String,
    pub action: ArcAction,
}

/// Coverage of the complete factual branch contract.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchCoverageCensus {
    pub strata: Vec<BranchCoverageStratum>,
    pub missing_action_keys: Vec<MissingBranchActionKey>,
    pub duplicate_action_keys: Vec<DuplicateBranchActionKey>,
}

impl BranchCoverageCensus {
    /// Reject populations that cannot supervise the complete branch contract.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.missing_action_keys.is_empty(),
            "branch coverage has missing action keys: {:?}",
            self.missing_action_keys
        );
        ensure!(
            self.duplicate_action_keys.is_empty(),
            "branch coverage has duplicate action keys: {:?}",
            self.duplicate_action_keys
        );
        ensure!(
            self.strata
                .iter()
                .any(|stratum| stratum.changed_outcomes > 0),
            "branch coverage has no changed outcomes"
        );
        ensure!(
            self.strata
                .iter()
                .any(|stratum| stratum.distinct_effect_classes > 1),
            "branch coverage has no distinct effect classes"
        );
        Ok(())
    }
}

/// Census complete factual branch groups without silently repairing a malformed
/// flat population. It intentionally reports missing keys for negative controls.
pub fn census_branch_coverage(rows: &[TransitionSample]) -> BranchCoverageCensus {
    let mut groups = BTreeMap::<BranchGroupId, Vec<&TransitionSample>>::new();
    for row in rows {
        if row.family.starts_with("factual_") {
            groups
                .entry(BranchGroupId::from_transition(row))
                .or_default()
                .push(row);
        }
    }
    let mut buckets = BTreeMap::<(u8, String, u8), Vec<&TransitionSample>>::new();
    let mut missing_action_keys = Vec::new();
    let mut duplicate_action_keys = Vec::new();
    for group in groups.values() {
        let Some(first) = group.first() else {
            continue;
        };
        let action_keys = group
            .iter()
            .map(|row| (row.action.id, row.action.x, row.action.y))
            .collect::<BTreeSet<_>>();
        for action_id in [1, 2, 3, 4, 5, 7] {
            if !action_keys.iter().any(|(id, _, _)| *id == action_id) {
                missing_action_keys.push(MissingBranchActionKey {
                    split: first.split,
                    family: first.family.clone(),
                    action_id,
                });
            }
        }
        let coordinate_actions = group
            .iter()
            .filter(|row| row.action.id == 6)
            .map(|row| row.action.clone())
            .collect::<Vec<_>>();
        let coordinate_keys = coordinate_actions
            .iter()
            .map(|action| (action.x, action.y))
            .collect::<BTreeSet<_>>();
        if coordinate_actions.len() != 4 {
            missing_action_keys.push(MissingBranchActionKey {
                split: first.split,
                family: first.family.clone(),
                action_id: 6,
            });
        }
        if coordinate_keys.len() != coordinate_actions.len() {
            duplicate_action_keys.extend(
                coordinate_actions
                    .into_iter()
                    .filter(|action| {
                        group
                            .iter()
                            .filter(|candidate| {
                                candidate.action.id == 6
                                    && candidate.action.x == action.x
                                    && candidate.action.y == action.y
                            })
                            .count()
                            > 1
                    })
                    .map(|action| DuplicateBranchActionKey {
                        split: first.split,
                        family: first.family.clone(),
                        action,
                    }),
            );
        }
        for row in group {
            buckets
                .entry((
                    match row.split {
                        Split::Train => 0,
                        Split::HeldOutComposition => 1,
                    },
                    row.family.clone(),
                    row.action.id,
                ))
                .or_default()
                .push(row);
        }
    }
    let strata = buckets
        .into_iter()
        .map(|((split, family, action_id), rows)| {
            let effects = rows
                .iter()
                .map(|row| {
                    let status_start = V5_PLAYFIELD_HEIGHT * FRAME_SIDE;
                    row.next.pixels[..status_start].to_vec()
                })
                .collect::<BTreeSet<_>>();
            let changed_outcomes = rows
                .iter()
                .filter(|row| {
                    let status_start = V5_PLAYFIELD_HEIGHT * FRAME_SIDE;
                    row.current.pixels[..status_start] != row.next.pixels[..status_start]
                })
                .count();
            BranchCoverageStratum {
                split: match split {
                    0 => Split::Train,
                    1 => Split::HeldOutComposition,
                    _ => unreachable!(),
                },
                family,
                action_id,
                eligible_rows: rows.len(),
                changed_outcomes,
                distinct_effect_classes: effects.len(),
            }
        })
        .collect();
    BranchCoverageCensus {
        strata,
        missing_action_keys,
        duplicate_action_keys,
    }
}

pub fn census_event_labels<'a>(
    rows: impl IntoIterator<Item = &'a TransitionSample>,
) -> EventLabelCensus {
    let mut census = EventLabelCensus::default();
    for row in rows {
        census.rows += 1;
        for (slot, label) in [row.noop, row.goal_satisfied, row.goal_failed, row.exhausted]
            .into_iter()
            .enumerate()
        {
            if let Some(label) = label {
                census.labeled[slot] += 1;
                census.positive[slot] += usize::from(label);
            }
        }
    }
    census
}

impl MixedStreamBatch {
    pub fn samples(&self) -> &[V5Sample] {
        &self.samples
    }

    pub fn transitions(&self) -> impl ExactSizeIterator<Item = &TransitionSample> {
        self.samples.iter().map(|sample| &sample.transition)
    }

    pub fn content_masks(&self) -> impl ExactSizeIterator<Item = &ContentMask> {
        self.samples.iter().map(|sample| &sample.content_mask)
    }

    pub fn flattened_content_masks_f32(&self) -> Vec<f32> {
        self.samples
            .iter()
            .flat_map(|sample| {
                sample
                    .content_mask
                    .values
                    .iter()
                    .map(|&value| f32::from(value))
            })
            .collect()
    }

    pub fn factual(&self) -> Option<&FactualBatch> {
        self.factual.as_ref()
    }

    /// Whether every row follows the ADR 0005 §1 interface contract. Batches
    /// are homogeneous by construction.
    pub fn contract_v6(&self) -> bool {
        self.samples
            .first()
            .is_some_and(|sample| sample.provenance.contract_v6)
    }

    /// Ranges in mixed-batch row order; every range is one complete group.
    pub fn factual_group_ranges(&self) -> &[std::ops::Range<usize>] {
        &self.factual_group_ranges
    }

    pub fn stream_counts(&self) -> &BTreeMap<MixedStreamKind, usize> {
        &self.stream_counts
    }

    pub fn scheduled_proportions(&self) -> MixedStreamProportions {
        self.scheduled_proportions
    }

    pub fn realized_proportions(&self) -> &RealizedStreamProportions {
        &self.realized_proportions
    }

    pub fn goal_dropout_census(&self) -> GoalDropoutCensus {
        self.goal_dropout_census
    }

    pub fn event_label_census(&self) -> EventLabelCensus {
        census_event_labels(self.transitions())
    }

    pub fn into_samples(self) -> Vec<V5Sample> {
        self.samples
    }
}

fn realized_stream_proportions(
    batch_size: usize,
    proportions: MixedStreamProportions,
) -> Result<RealizedStreamProportions> {
    ensure!(batch_size > 0, "mixed batch size must be positive");
    let normalized = proportions.normalized();
    ensure!(
        normalized.total() > f32::EPSILON,
        "mixed stream schedule must have positive total weight"
    );
    let ordered = normalized.ordered();
    let mut counts = BTreeMap::new();
    let mut remainders = Vec::new();
    let mut assigned = 0usize;
    for (kind, weight) in ordered {
        // Legacy schedules never realize learning histories or shards;
        // leaving the kind out of `counts` keeps legacy batch identities
        // byte-identical.
        if matches!(
            kind,
            MixedStreamKind::LearningHistories | MixedStreamKind::SyntheticShards
        ) && weight == 0.0
        {
            continue;
        }
        let exact = weight as f64 * batch_size as f64;
        let floor = exact.floor() as usize;
        counts.insert(kind, floor);
        remainders.push((kind, exact - floor as f64));
        assigned += floor;
    }
    remainders.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    for (kind, _) in remainders.into_iter().take(batch_size - assigned) {
        *counts.get_mut(&kind).expect("all stream kinds inserted") += 1;
    }

    let factual = counts[&MixedStreamKind::FactualBranches];
    let complete_factual = (factual / FACTUAL_BRANCHES_PER_GROUP)
        .max(1)
        .saturating_mul(FACTUAL_BRANCHES_PER_GROUP)
        .min(batch_size);
    if complete_factual < factual {
        *counts
            .get_mut(&MixedStreamKind::RandomOneStep)
            .expect("random stream exists") += factual - complete_factual;
    } else if complete_factual > factual {
        let mut needed = complete_factual - factual;
        for kind in [
            MixedStreamKind::RandomOneStep,
            MixedStreamKind::Exploration,
            MixedStreamKind::SequentialFragments,
            MixedStreamKind::HazardOneStep,
            MixedStreamKind::LearningHistories,
            MixedStreamKind::SyntheticShards,
        ] {
            let Some(count) = counts.get_mut(&kind) else {
                continue;
            };
            let take = (*count).min(needed);
            *count -= take;
            needed -= take;
            if needed == 0 {
                break;
            }
        }
    }
    counts.insert(MixedStreamKind::FactualBranches, complete_factual);
    debug_assert_eq!(counts.values().sum::<usize>(), batch_size);
    let fractions = MixedStreamProportions {
        random_one_step: counts[&MixedStreamKind::RandomOneStep] as f32 / batch_size as f32,
        factual_branches: counts[&MixedStreamKind::FactualBranches] as f32 / batch_size as f32,
        exploration: counts[&MixedStreamKind::Exploration] as f32 / batch_size as f32,
        sequential_fragments: counts[&MixedStreamKind::SequentialFragments] as f32
            / batch_size as f32,
        hazard_one_step: counts[&MixedStreamKind::HazardOneStep] as f32 / batch_size as f32,
        learning_histories: counts
            .get(&MixedStreamKind::LearningHistories)
            .copied()
            .unwrap_or(0) as f32
            / batch_size as f32,
        synthetic_shards: counts
            .get(&MixedStreamKind::SyntheticShards)
            .copied()
            .unwrap_or(0) as f32
            / batch_size as f32,
    };
    // The intact-branch-group constraint moves stream shares in steps of one
    // whole group, so small smoke batches cannot meet the 5pp tolerance at
    // all; the enforceable tolerance is the larger of 5pp and one group's
    // share of the batch.
    let tolerance = 0.05f32.max(FACTUAL_BRANCHES_PER_GROUP as f32 / batch_size as f32);
    for ((kind, target), (_, realized)) in normalized.ordered().into_iter().zip(fractions.ordered())
    {
        ensure!(
            (target - realized).abs() <= tolerance + f32::EPSILON,
            "mixed batch size {batch_size} realizes {kind:?} at {realized:.3}, more than {tolerance:.3} from normalized target {target:.3} after intact factual-group rounding"
        );
    }
    Ok(RealizedStreamProportions {
        normalized_target: normalized,
        counts,
        fractions,
    })
}

fn raw_random_v5_sample(
    seed: u64,
    episode_id: u64,
    split: V5DataSplit,
    size: u8,
    operator: EpisodeOperator,
) -> Result<TransitionSample> {
    let scenario = Arc::new(scenario_for_v5(seed, episode_id, split, size));
    let sim = Simulator::new(Arc::clone(&scenario));
    let mut rng = seeded_v5_rng(seed, episode_id, split, 0xA11C_E005);
    let mut state = State::initial(&scenario);
    for _ in 0..episode_id as usize % 4 {
        let action = Action::moves()[rng.random_range(0..4)];
        state = apply_action(&sim, &state, action);
    }
    match episode_id % 7 {
        0..=3 => {
            let action = Action::moves()[episode_id as usize % 4];
            let next = apply_action(&sim, &state, action);
            sample_from_transition_goal_free(&scenario, &state, &next, action, "random_one_step", 0)
        }
        4 => operator_sample_from_state(
            &scenario,
            &state,
            ArcAction::new(5, None, None)?,
            operator,
            "random_one_step",
            0,
        ),
        5 => {
            let current = render_state_padded(&scenario, &state)?;
            let coordinates = stratified_action6_coordinates(&current, size)?;
            let (x, y) = coordinates[rng.random_range(0..coordinates.len())];
            operator_sample_from_rendered_current(
                &scenario,
                &state,
                current,
                ArcAction::new(6, Some(x), Some(y))?,
                operator,
                "random_one_step",
                0,
            )
        }
        _ => null_sample_from_state(&scenario, &state, "random_one_step", 0),
    }
}

fn raw_exploration_v5_fragment(
    seed: u64,
    episode_id: u64,
    split: V5DataSplit,
    size: u8,
    length: usize,
) -> Result<Vec<TransitionSample>> {
    let scenario = Arc::new(scenario_for_v5(seed, episode_id, split, size));
    let sim = Simulator::new(Arc::clone(&scenario));
    let mut rng = seeded_v5_rng(seed, episode_id, split, 0xE1A1_0005);
    let mut state = State::initial(&scenario);
    let mut current = render_state_padded(&scenario, &state)?;
    let mut samples = Vec::with_capacity(length);
    for index in 0..length {
        let action = Action::moves()[rng.random_range(0..4)];
        let next = apply_action(&sim, &state, action);
        let next_frame = render_state_padded(&scenario, &next)?;
        samples.push(sample_from_rendered_transition_goal_free(
            &scenario,
            &state,
            action,
            "exploration",
            index as u64,
            current,
            next_frame.clone(),
        ));
        state = next;
        current = next_frame;
    }
    Ok(samples)
}

fn raw_sequential_v5_fragment(
    seed: u64,
    episode_id: u64,
    split: V5DataSplit,
    size: u8,
    max_length: usize,
) -> Result<Vec<TransitionSample>> {
    let scenario = Arc::new(scenario_for_v5(seed, episode_id, split, size));
    let sim = Simulator::new(Arc::clone(&scenario));
    let state = State::initial(&scenario);
    let offset = episode_id as usize % scenario.candidate_goals.len();
    let (goal, plan) = scenario
        .candidate_goals
        .iter()
        .cycle()
        .skip(offset)
        .take(scenario.candidate_goals.len())
        .find_map(|goal| {
            shortest_path(&sim, &state, goal, scenario.action_budget)
                .filter(|plan| !plan.actions.is_empty())
                .map(|plan| (goal, plan))
        })
        .ok_or_else(|| anyhow!("no non-empty public plan for v5 sequential fragment"))?;
    let mut state = state;
    let mut current = render_state_padded(&scenario, &state)?;
    let mut samples = Vec::with_capacity(max_length.min(4));
    for (index, action) in plan.actions.into_iter().take(max_length.min(4)).enumerate() {
        let next = apply_action(&sim, &state, action);
        let next_frame = render_state_padded(&scenario, &next)?;
        let mut sample = sample_from_rendered_transition(
            &scenario,
            &state,
            &next,
            action,
            goal,
            index as u64,
            current,
            next_frame.clone(),
        );
        sample.family = "sequential_fragments".into();
        sample.provenance.source_kind = "sequential_fragments".into();
        samples.push(sample);
        state = next;
        current = next_frame;
    }
    Ok(samples)
}

fn raw_hazard_v5_sample(
    seed: u64,
    episode_id: u64,
    split: V5DataSplit,
    size: u8,
) -> Result<TransitionSample> {
    let mut scenario = scenario_for_v5(seed, episode_id, split, size);
    ensure!(!scenario.hazards.is_empty(), "v5 scenario has no hazard");
    ensure!(!scenario.markers.is_empty(), "v5 scenario has no marker");
    let hazard = scenario.hazards[0];
    let candidates = [
        (Pos::new(hazard.x - 1, hazard.y), Action::Move(Dir::East)),
        (Pos::new(hazard.x + 1, hazard.y), Action::Move(Dir::West)),
        (Pos::new(hazard.x, hazard.y - 1), Action::Move(Dir::South)),
        (Pos::new(hazard.x, hazard.y + 1), Action::Move(Dir::North)),
    ];
    let (start, action) = candidates
        .into_iter()
        .find(|(position, _)| scenario.in_bounds(*position))
        .ok_or_else(|| anyhow!("hazard has no in-bounds neighbor"))?;
    scenario.walls.remove(&start);
    scenario.walls.remove(&hazard);
    scenario.start = start;
    let scenario = Arc::new(scenario);
    let sim = Simulator::new(Arc::clone(&scenario));
    let state = State::initial(&scenario);
    let next = apply_action(&sim, &state, action);
    ensure!(
        next.pos == hazard,
        "hazard action fell back to a blocked destination"
    );
    let goal = Goal::AvoidHazardReachMarker {
        hazard: 0,
        marker: 0,
    };
    let mut sample = sample_from_transition(&scenario, &state, &next, action, &goal, 0)?;
    sample.family = "hazard_one_step".into();
    sample.provenance.source_kind = "hazard_one_step".into();
    Ok(sample)
}

pub(crate) fn augment_v5_unit(
    transitions: Vec<TransitionSample>,
    config: &MixedStreamConfig,
    split: V5DataSplit,
    stream: MixedStreamKind,
    operator: EpisodeOperator,
    episode_id: u64,
) -> Result<Vec<V5Sample>> {
    ensure!(!transitions.is_empty(), "cannot augment an empty v5 unit");
    let size = u8::try_from(transitions[0].provenance.content_width)
        .map_err(|_| anyhow!("content size does not fit u8"))?;
    let v6 = config.data_contract_v6;
    let mut augmentation_rng = seeded_v5_rng(config.seed, episode_id, split, 0xA06D_4E05);
    let rect = sampled_content_rect(size, split, &mut augmentation_rng, v6);
    let augmentation =
        sampled_augmentation(&mut augmentation_rng, config.symmetry_augmentation, v6);
    validate_color_permutation(&augmentation.color_permutation, !v6)?;
    let content_mask = if v6 {
        ContentMask::all_ones()
    } else {
        let content_mask = ContentMask::from_rect(rect)?;
        ensure!(
            content_mask.matches_rect(rect),
            "shared v5 content mask does not match sampled unit rect"
        );
        content_mask
    };
    let mut dropout_rng = seeded_v5_rng(config.seed, episode_id, split, 0xD20F_0005);
    let mut frame_cache = BTreeMap::new();
    transitions
        .into_iter()
        .map(|transition| {
            augment_v5_transition(
                transition,
                split,
                stream,
                operator,
                rect,
                augmentation.clone(),
                content_mask.clone(),
                config.goal_dropout_probability,
                v6,
                &mut dropout_rng,
                &mut frame_cache,
            )
        })
        .collect()
}

const MIXED_BATCH_EPISODE_STRIDE: u64 = 1_000_003;
const MIXED_STREAM_FRAGMENT_ROWS: usize = 4;

fn checked_non_meta_episode_id(episode_id: u64) -> Result<u64> {
    let outside_meta_domain = episode_id & META_LEVEL_EPISODE_DOMAIN == 0;
    ensure!(
        outside_meta_domain,
        "non-meta episode id enters the reserved meta-episode namespace"
    );
    debug_assert!(
        outside_meta_domain,
        "non-meta episode id enters the reserved meta-episode namespace"
    );
    Ok(episode_id)
}

fn non_meta_episode_id(batch_or_episode: u64, offset: u64) -> Result<u64> {
    let episode_id = batch_or_episode
        .checked_mul(MIXED_BATCH_EPISODE_STRIDE)
        .and_then(|base| base.checked_add(offset))
        .context("non-meta episode id overflow")?;
    checked_non_meta_episode_id(episode_id)
}

pub(crate) fn mixed_stream_episode_id(batch_index: u64, unit_index: u64) -> Result<u64> {
    non_meta_episode_id(batch_index, unit_index)
}

fn compose_nonfactual_unit(
    config: &MixedStreamConfig,
    split: V5DataSplit,
    stream: MixedStreamKind,
    episode_id: u64,
    maximum_rows: usize,
) -> Result<Vec<V5Sample>> {
    let mut rng = seeded_v5_rng(config.seed, episode_id, split, 0x51DE_0005);
    let size = sampled_content_size(split, &mut rng);
    let operator = sampled_operator(&config.operator_families, split, &mut rng)?;
    let raw = match stream {
        MixedStreamKind::RandomOneStep => vec![raw_random_v5_sample(
            config.seed,
            episode_id,
            split,
            size,
            operator,
        )?],
        MixedStreamKind::Exploration => raw_exploration_v5_fragment(
            config.seed,
            episode_id,
            split,
            size,
            maximum_rows.min(MIXED_STREAM_FRAGMENT_ROWS),
        )?,
        MixedStreamKind::SequentialFragments => {
            let mut generated = None;
            for retry in 0..16u64 {
                let retry_episode = checked_non_meta_episode_id(
                    episode_id
                        .checked_add(
                            retry
                                .checked_mul(10_000_019)
                                .context("sequential retry episode id overflow")?,
                        )
                        .context("sequential retry episode id overflow")?,
                )?;
                if let Ok(fragment) = raw_sequential_v5_fragment(
                    config.seed,
                    retry_episode,
                    split,
                    size,
                    maximum_rows.min(MIXED_STREAM_FRAGMENT_ROWS),
                ) {
                    generated = Some(fragment);
                    break;
                }
            }
            generated.ok_or_else(|| anyhow!("could not generate v5 sequential fragment"))?
        }
        MixedStreamKind::HazardOneStep => {
            vec![raw_hazard_v5_sample(config.seed, episode_id, split, size)?]
        }
        MixedStreamKind::FactualBranches => {
            bail!("factual stream must be composed as whole groups")
        }
        MixedStreamKind::LearningHistories => {
            bail!("learning histories must be composed as twin meta-episode units")
        }
        MixedStreamKind::SyntheticShards => {
            bail!("synthetic shards must be composed from a loaded shard pool")
        }
    };
    let mut unit = augment_v5_unit(raw, config, split, stream, operator, episode_id)?;
    unit.truncate(maximum_rows);
    Ok(unit)
}

fn compose_fixed_nonfactual_stream(
    config: &MixedStreamConfig,
    split: V5DataSplit,
    stream: MixedStreamKind,
    count: usize,
    batch_index: u64,
    first_unit_index: u64,
) -> Result<(Vec<V5Sample>, u64)> {
    let rows_per_unit = match stream {
        MixedStreamKind::Exploration => MIXED_STREAM_FRAGMENT_ROWS,
        MixedStreamKind::RandomOneStep | MixedStreamKind::HazardOneStep => 1,
        MixedStreamKind::SequentialFragments
        | MixedStreamKind::FactualBranches
        | MixedStreamKind::LearningHistories
        | MixedStreamKind::SyntheticShards => {
            bail!("stream does not have a fixed row count per unit")
        }
    };
    let unit_count = count.div_ceil(rows_per_unit);
    let units = (0..unit_count)
        .into_par_iter()
        .map(|offset| {
            let produced_before = offset * rows_per_unit;
            let maximum_rows = (count - produced_before).min(rows_per_unit);
            compose_nonfactual_unit(
                config,
                split,
                stream,
                mixed_stream_episode_id(
                    batch_index,
                    first_unit_index
                        .checked_add(offset as u64)
                        .context("mixed stream unit index overflow")?,
                )?,
                maximum_rows,
            )
        })
        .collect::<Vec<_>>();
    let mut samples = Vec::with_capacity(count);
    for unit in units {
        samples.extend(unit?);
    }
    ensure!(
        samples.len() == count,
        "fixed-row stream {stream:?} produced {} rows for requested {count}",
        samples.len()
    );
    Ok((
        samples,
        first_unit_index
            .checked_add(unit_count as u64)
            .context("mixed stream unit index overflow")?,
    ))
}

fn compose_sequential_stream(
    config: &MixedStreamConfig,
    split: V5DataSplit,
    count: usize,
    batch_index: u64,
    first_unit_index: u64,
) -> Result<(Vec<V5Sample>, u64)> {
    let mut samples = Vec::with_capacity(count);
    let mut next_unit_index = first_unit_index;
    while samples.len() < count {
        let remaining = count - samples.len();
        // A fragment contributes at most four rows. Generate the minimum
        // optimistic wave concurrently; if any plans are shorter, generate
        // another wave starting at the first unit the serial composer would
        // have used. Results past the exact prefix are deliberately ignored.
        let wave_units = remaining.div_ceil(MIXED_STREAM_FRAGMENT_ROWS);
        let units = (0..wave_units)
            .into_par_iter()
            .map(|offset| {
                compose_nonfactual_unit(
                    config,
                    split,
                    MixedStreamKind::SequentialFragments,
                    mixed_stream_episode_id(
                        batch_index,
                        next_unit_index
                            .checked_add(offset as u64)
                            .context("mixed stream unit index overflow")?,
                    )?,
                    MIXED_STREAM_FRAGMENT_ROWS,
                )
            })
            .collect::<Vec<_>>();
        for unit in units {
            if samples.len() == count {
                break;
            }
            let mut unit = unit?;
            ensure!(!unit.is_empty(), "sequential fragment produced no rows");
            unit.truncate(count - samples.len());
            samples.append(&mut unit);
            next_unit_index = next_unit_index
                .checked_add(1)
                .context("mixed stream unit index overflow")?;
        }
    }
    Ok((samples, next_unit_index))
}

struct ComposedFactualGroup {
    samples: Vec<V5Sample>,
    group: BranchGroup,
}

fn compose_factual_group(
    config: &MixedStreamConfig,
    split: V5DataSplit,
    episode_id: u64,
) -> Result<ComposedFactualGroup> {
    let mut rng = seeded_v5_rng(config.seed, episode_id, split, 0xFAC7_0005);
    let size = sampled_content_size(split, &mut rng);
    let operator = sampled_operator(&config.operator_families, split, &mut rng)?;
    let raw_group =
        generate_v5_factual_branch_group(config.seed, episode_id, split, size, operator)?;
    let mut raw_transitions = raw_group.into_transitions().collect::<Vec<_>>();
    raw_transitions.sort_by_key(|transition| {
        (
            transition.action.id,
            transition.action.x,
            transition.action.y,
        )
    });
    let mut samples = augment_v5_unit(
        raw_transitions,
        config,
        split,
        MixedStreamKind::FactualBranches,
        operator,
        episode_id,
    )?;
    samples.sort_by_key(|sample| {
        (
            sample.transition.action.id,
            sample.transition.action.x,
            sample.transition.action.y,
        )
    });
    let group = BranchGroup::try_new(
        samples
            .iter()
            .map(|sample| {
                if config.data_contract_v6 {
                    FactualActionBranch::try_from_transition_whole_frame(sample.transition.clone())
                } else {
                    FactualActionBranch::try_from_transition(sample.transition.clone())
                }
            })
            .collect::<Result<Vec<_>>>()?,
    )?;
    let group_id = BranchGroupId::from_transition(&samples[0].transition);
    for sample in &mut samples {
        sample.provenance.branch_group_id = Some(group_id.clone());
    }
    Ok(ComposedFactualGroup { samples, group })
}

/// Compose one deterministic, mixed foundation-v2 batch.
///
/// `progress` is clamped to `[0,1]`. Factual row allocation is rounded to a
/// whole ten-branch group and all other streams absorb the at-most-nine-row
/// difference, so `batch_size` remains exact and no group is split.
pub fn compose_mixed_stream_batch(
    config: &MixedStreamConfig,
    progress: f32,
    batch_index: u64,
    split: V5DataSplit,
) -> Result<MixedStreamBatch> {
    if let V5DataSplit::HeldOutOperator(family) = split {
        ensure!(
            config.operator_families.held_out.contains(&family),
            "operator eval split requested a non-held-out family"
        );
    }
    let scheduled_proportions = (config.schedule)(progress);
    ensure!(
        scheduled_proportions
            .ordered()
            .iter()
            .all(|(_, weight)| weight.is_finite() && *weight >= 0.0),
        "mixed stream schedule returned invalid weights"
    );
    let realized_proportions = config.realized_proportions(progress)?;
    let stream_counts = realized_proportions.counts.clone();
    let mut samples = Vec::with_capacity(config.batch_size);
    let mut factual_groups = Vec::new();
    let mut factual_group_ranges = Vec::new();
    let mut unit_index = 0u64;

    let (mut random_samples, next_unit_index) = compose_fixed_nonfactual_stream(
        config,
        split,
        MixedStreamKind::RandomOneStep,
        stream_counts[&MixedStreamKind::RandomOneStep],
        batch_index,
        unit_index,
    )?;
    samples.append(&mut random_samples);
    unit_index = next_unit_index;

    let factual_group_count =
        stream_counts[&MixedStreamKind::FactualBranches] / FACTUAL_BRANCHES_PER_GROUP;
    let composed_factual = (0..factual_group_count)
        .into_par_iter()
        .map(|offset| {
            compose_factual_group(
                config,
                split,
                mixed_stream_episode_id(
                    batch_index,
                    unit_index
                        .checked_add(offset as u64)
                        .context("mixed stream unit index overflow")?,
                )?,
            )
        })
        .collect::<Vec<_>>();
    for composed in composed_factual {
        let mut composed = composed?;
        let start = samples.len();
        samples.append(&mut composed.samples);
        factual_group_ranges.push(start..samples.len());
        factual_groups.push(composed.group);
    }
    unit_index = unit_index
        .checked_add(factual_group_count as u64)
        .context("mixed stream unit index overflow")?;

    let (mut exploration_samples, next_unit_index) = compose_fixed_nonfactual_stream(
        config,
        split,
        MixedStreamKind::Exploration,
        stream_counts[&MixedStreamKind::Exploration],
        batch_index,
        unit_index,
    )?;
    samples.append(&mut exploration_samples);
    unit_index = next_unit_index;

    let (mut sequential_samples, next_unit_index) = compose_sequential_stream(
        config,
        split,
        stream_counts[&MixedStreamKind::SequentialFragments],
        batch_index,
        unit_index,
    )?;
    samples.append(&mut sequential_samples);
    unit_index = next_unit_index;

    let (mut hazard_samples, next_unit_index) = compose_fixed_nonfactual_stream(
        config,
        split,
        MixedStreamKind::HazardOneStep,
        stream_counts[&MixedStreamKind::HazardOneStep],
        batch_index,
        unit_index,
    )?;
    samples.append(&mut hazard_samples);
    unit_index = next_unit_index;

    let learning_histories = stream_counts
        .get(&MixedStreamKind::LearningHistories)
        .copied()
        .unwrap_or(0);
    if learning_histories > 0 {
        ensure!(
            config.data_contract_v6,
            "the learning-histories stream requires data_contract_v6"
        );
        let (mut history_samples, next_unit_index) = compose_learning_histories_stream(
            config,
            split,
            learning_histories,
            batch_index,
            unit_index,
        )?;
        samples.append(&mut history_samples);
        unit_index = next_unit_index;
    }
    let synthetic_shards = stream_counts
        .get(&MixedStreamKind::SyntheticShards)
        .copied()
        .unwrap_or(0);
    if synthetic_shards > 0 {
        let shards_dir = config
            .synthetic_shards_dir
            .as_deref()
            .ok_or_else(|| anyhow!("the synthetic-shards stream requires synthetic_shards_dir"))?;
        ensure!(
            config.data_contract_v6,
            "the synthetic-shards stream requires data_contract_v6"
        );
        let pool = crate::p2::synthetic_shards::cached_pool(shards_dir)?;
        let (mut shard_samples, _) = crate::p2::synthetic_shards::compose_synthetic_shards_stream(
            &pool,
            config,
            split,
            synthetic_shards,
            batch_index,
            unit_index,
        )?;
        samples.append(&mut shard_samples);
    }
    ensure!(
        samples.len() == config.batch_size,
        "mixed composer produced {} rows for requested batch {}",
        samples.len(),
        config.batch_size
    );
    for sample in &samples {
        // Mask shape and color bijection were validated once when this row's
        // unit allocated its shared mask/augmentation.
        sample.validate_after_unit_fields()?;
        ensure!(
            sample.provenance.contract_v6 == config.data_contract_v6,
            "mixed batch mixes v6 and legacy interface contracts"
        );
    }
    let goal_dropout_census = GoalDropoutCensus {
        total: samples.len(),
        eligible: samples
            .iter()
            .filter(|sample| sample.original_goal_nonzero)
            .count(),
        changed: samples
            .iter()
            .filter(|sample| sample.original_goal_nonzero && sample.provenance.goal_dropped)
            .count(),
        final_zero_goal: samples
            .iter()
            .filter(|sample| sample.transition.goal_features.values == [0.0; GOAL_FEATURES_DIM])
            .count(),
    };
    let factual = if factual_groups.is_empty() {
        None
    } else {
        Some(FactualBatch::from_groups(factual_groups)?)
    };
    Ok(MixedStreamBatch {
        samples,
        factual,
        factual_group_ranges,
        stream_counts,
        scheduled_proportions,
        realized_proportions,
        goal_dropout_census,
    })
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

/// Per-level size knobs for one deterministic multi-level shared-rule episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetaEpisodeConfig {
    /// Number of levels. At least two, so "earlier level" information exists.
    pub levels: usize,
    /// Movement transitions per level before the operator decision points.
    pub steps_per_level: usize,
    /// Operator decision points per level: one ACTION5 plus up to four
    /// stratified ACTION6 coordinates. Zero yields a rule-independent episode
    /// whose later decisions never touch the hidden rule.
    pub operator_decisions_per_level: usize,
    /// Square content size shared by every level layout.
    pub content_size: u8,
}

impl Default for MetaEpisodeConfig {
    fn default() -> Self {
        Self {
            levels: 3,
            steps_per_level: 4,
            operator_decisions_per_level: 3,
            content_size: 7,
        }
    }
}

impl MetaEpisodeConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.levels >= 2,
            "a meta-episode needs at least two levels so earlier-level information exists"
        );
        ensure!(
            self.levels <= META_LEVEL_EPISODE_STRIDE as usize,
            "meta-episode level count exceeds the reserved episode-id namespace"
        );
        ensure!(
            self.operator_decisions_per_level <= 5,
            "at most one ACTION5 plus four stratified ACTION6 decision points per level"
        );
        ensure!(
            V5_CONTENT_SIZES.contains(&self.content_size),
            "meta-episode content size must be one of {V5_CONTENT_SIZES:?}"
        );
        Ok(())
    }
}

/// Stable identity of a same-state meta decision group.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MetaDecisionGroupId {
    pub level_index: usize,
    pub episode_id: u64,
    pub group_index: usize,
    pub current_fingerprint: String,
}

/// Counterfactual ACTION5/ACTION6 rows from one byte-identical current frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetaDecisionGroup {
    pub id: MetaDecisionGroupId,
    pub transitions: Vec<TransitionSample>,
}

/// One level of a [`MetaEpisode`]: a freshly generated layout whose trajectory
/// is chronological and whose operator alternatives are separate groups. The
/// layout regenerates from `(seed, episode_id, split, content_size)`;
/// `operator` is the rule actually applied to its decision groups.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetaLevel {
    pub level_index: usize,
    pub episode_id: u64,
    pub operator: EpisodeOperator,
    pub trajectory: Vec<TransitionSample>,
    pub decision_groups: Vec<MetaDecisionGroup>,
}

/// Multi-level episode with one stable hidden rule shared by every level.
///
/// Pure function of `(seed, meta_episode_id, split, families, config)`:
/// identical inputs reproduce identical bytes. Level boundaries are explicit
/// indices over the flattened chronological trajectory stream.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetaEpisode {
    pub seed: u64,
    pub meta_episode_id: u64,
    pub data_split: V5DataSplit,
    pub config: MetaEpisodeConfig,
    /// The stable hidden rule. In the shuffled control this is the level-0 rule.
    pub operator: EpisodeOperator,
    /// Whether later levels deliberately break the shared rule.
    pub shuffled_control: bool,
    pub levels: Vec<MetaLevel>,
    /// Level `i` owns flattened chronological rows
    /// `level_boundaries[i]..level_boundaries[i+1]`.
    pub level_boundaries: Vec<usize>,
}

/// Realized population of a shuffled-rule negative control. Counts are kept
/// separate because an independent marginal draw may legitimately repeat the
/// level-0 operator, and a genuinely changed operator may still produce the
/// same outcome for a particular state/action tuple.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShuffledRuleRealizationCensus {
    pub later_levels: usize,
    pub repeated_level0_operator_levels: usize,
    pub total_rows: usize,
    pub eligible_operator_rows: usize,
    pub genuinely_changed_operator_tuples: usize,
    pub outcome_changing_tuples: usize,
}

fn validate_meta_row(
    row: &TransitionSample,
    episode: &MetaEpisode,
    level: &MetaLevel,
    trajectory_id: &str,
    transition_index: u64,
) -> Result<()> {
    row.provenance.validate()?;
    ensure!(
        row.seed == episode.seed
            && row.episode_id == level.episode_id
            && row.transition_index == transition_index
            && row.split == episode.data_split.reported_split(),
        "meta-episode transition identity or index is inconsistent"
    );
    ensure!(
        row.family == "meta_episode_v5"
            && row.provenance.source_kind == "meta_episode_v5"
            && row.provenance.trajectory_id == trajectory_id
            && row.provenance.content_width == u16::from(episode.config.content_size)
            && row.provenance.content_height == u16::from(episode.config.content_size),
        "meta-episode transition provenance is inconsistent"
    );
    ensure!(
        row.goal_features == GoalFeatures::zeros()
            && row.goal_satisfied.is_none()
            && row.goal_failed.is_none(),
        "meta-episode rows must remain goal-free"
    );
    Ok(())
}

impl MetaEpisode {
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        ensure!(
            self.levels.len() == self.config.levels,
            "meta-episode level count does not match its config"
        );
        ensure!(
            self.level_boundaries.len() == self.levels.len() + 1 && self.level_boundaries[0] == 0,
            "level boundaries must be cumulative offsets starting at zero"
        );
        let mut offset = 0usize;
        let mut episode_ids = BTreeSet::new();
        for (index, level) in self.levels.iter().enumerate() {
            ensure!(level.level_index == index, "level indices must be ordered");
            ensure!(
                level.episode_id == meta_level_episode_id(self.meta_episode_id, index)?,
                "level episode id is outside the reserved meta-episode namespace"
            );
            ensure!(
                episode_ids.insert(level.episode_id),
                "meta-episode level ids must be unique"
            );
            ensure!(
                level.trajectory.len() == self.config.steps_per_level,
                "meta trajectory row count does not match its config"
            );
            ensure!(
                level.decision_groups.len()
                    == usize::from(self.config.operator_decisions_per_level > 0),
                "meta decision-group count does not match its config"
            );
            let trajectory_id = format!(
                "meta-v5/{:?}/{}/{}/level{}",
                self.data_split, self.seed, self.meta_episode_id, index
            );
            for (transition_index, row) in level.trajectory.iter().enumerate() {
                validate_meta_row(row, self, level, &trajectory_id, transition_index as u64)?;
                ensure!(
                    matches!(row.action.id, 1..=4)
                        && row.action.x.is_none()
                        && row.action.y.is_none(),
                    "meta trajectory action schema must be directional movement"
                );
            }
            for pair in level.trajectory.windows(2) {
                ensure!(
                    pair[0].next == pair[1].current,
                    "meta trajectory rows must be chronologically adjacent"
                );
            }
            for (group_index, group) in level.decision_groups.iter().enumerate() {
                ensure!(
                    group.id.level_index == index
                        && group.id.episode_id == level.episode_id
                        && group.id.group_index == group_index,
                    "meta decision-group id is inconsistent"
                );
                ensure!(
                    group.transitions.len() == self.config.operator_decisions_per_level,
                    "meta decision-group row count does not match its config"
                );
                let first = group
                    .transitions
                    .first()
                    .context("configured meta decision group is empty")?;
                ensure!(
                    group.id.current_fingerprint == frame_fingerprint(&first.current),
                    "meta decision-group fingerprint does not match current frame"
                );
                ensure!(
                    group
                        .transitions
                        .iter()
                        .all(|row| row.current == first.current),
                    "meta decision-group rows must share a byte-identical current frame"
                );
                if let Some(last) = level.trajectory.last() {
                    ensure!(
                        last.next == first.current,
                        "meta decision group must branch from the trajectory endpoint"
                    );
                }
                let mut coordinates = BTreeSet::new();
                for (row_index, row) in group.transitions.iter().enumerate() {
                    validate_meta_row(
                        row,
                        self,
                        level,
                        &trajectory_id,
                        (self.config.steps_per_level + group_index) as u64,
                    )?;
                    if row_index == 0 {
                        ensure!(
                            row.action == ArcAction::new(5, None, None)?,
                            "first meta decision action must be ACTION5"
                        );
                    } else {
                        ensure!(
                            row.action.id == 6
                                && row.action.x.is_some()
                                && row.action.y.is_some()
                                && coordinates.insert((
                                    row.action.x.expect("checked"),
                                    row.action.y.expect("checked")
                                )),
                            "later meta decision actions must be distinct ACTION6 coordinates"
                        );
                    }
                    let rect = ContentRect {
                        x: 0,
                        y: 0,
                        width: self.config.content_size,
                        height: self.config.content_size,
                    };
                    let replayed =
                        apply_episode_operator(&row.current, &row.action, rect, level.operator)?;
                    let status_start = V5_PLAYFIELD_HEIGHT * FRAME_SIDE;
                    ensure!(
                        replayed.pixels[..status_start] == row.next.pixels[..status_start],
                        "meta decision outcome does not replay under its operator"
                    );
                }
            }
            offset += level.trajectory.len();
            ensure!(
                self.level_boundaries[index + 1] == offset,
                "level boundary does not match its flattened transition count"
            );
        }
        if !self.shuffled_control {
            ensure!(
                self.levels
                    .iter()
                    .all(|level| level.operator == self.operator),
                "the shared rule must be stable across true meta-episode levels"
            );
        } else {
            ensure!(
                self.levels[0].operator == self.operator,
                "shuffled-control level 0 must retain the recorded operator"
            );
            let expected = generate_meta_level(
                self.seed,
                self.meta_episode_id,
                self.data_split,
                &self.config,
                0,
                self.operator,
            )?;
            ensure!(
                self.levels[0] == expected,
                "shuffled-control level 0 must equal the corresponding stable-rule level"
            );
        }
        Ok(())
    }

    /// Validate operator-family membership in addition to structural invariants.
    pub fn validate_with_families(&self, families: &OperatorFamilySplit) -> Result<()> {
        self.validate()?;
        families.validate()?;
        let allowed = match self.data_split {
            V5DataSplit::HeldOutOperator(family) => {
                ensure!(
                    families.held_out.contains(&family),
                    "meta held-out split requests a non-held-out operator"
                );
                &families.held_out
            }
            _ => &families.train,
        };
        ensure!(
            allowed.contains(&self.operator.family)
                && self
                    .levels
                    .iter()
                    .all(|level| allowed.contains(&level.operator.family)),
            "meta-episode operator is outside its split family membership"
        );
        Ok(())
    }

    /// Chronological transitions only; decision branches are intentionally excluded.
    pub fn flattened_transitions(&self) -> impl Iterator<Item = &TransitionSample> {
        self.levels.iter().flat_map(|level| level.trajectory.iter())
    }

    /// Same-state ACTION5/ACTION6 branch groups, separate from chronological rows.
    pub fn decision_groups(&self) -> impl Iterator<Item = &MetaDecisionGroup> {
        self.levels
            .iter()
            .flat_map(|level| level.decision_groups.iter())
    }

    /// Census the realized shuffled-control population without conditioning
    /// its generation on being different from level 0.
    pub fn shuffled_rule_realization_census(
        &self,
    ) -> Result<Option<ShuffledRuleRealizationCensus>> {
        self.validate()?;
        if !self.shuffled_control {
            return Ok(None);
        }
        let status_start = V5_PLAYFIELD_HEIGHT * FRAME_SIDE;
        let mut census = ShuffledRuleRealizationCensus::default();
        for level in self.levels.iter().filter(|level| level.level_index >= 1) {
            census.later_levels += 1;
            census.repeated_level0_operator_levels += usize::from(level.operator == self.operator);
            census.total_rows += level.trajectory.len()
                + level
                    .decision_groups
                    .iter()
                    .map(|group| group.transitions.len())
                    .sum::<usize>();
            for transition in level
                .decision_groups
                .iter()
                .flat_map(|group| group.transitions.iter())
            {
                census.eligible_operator_rows += 1;
                let changed = level.operator != self.operator;
                census.genuinely_changed_operator_tuples += usize::from(changed);
                if changed {
                    let rect = ContentRect {
                        x: 0,
                        y: 0,
                        width: u8::try_from(transition.provenance.content_width)
                            .map_err(|_| anyhow!("content width does not fit u8"))?,
                        height: u8::try_from(transition.provenance.content_height)
                            .map_err(|_| anyhow!("content height does not fit u8"))?,
                    };
                    let level0_outcome = apply_episode_operator(
                        &transition.current,
                        &transition.action,
                        rect,
                        self.operator,
                    )?;
                    census.outcome_changing_tuples += usize::from(
                        level0_outcome.pixels[..status_start]
                            != transition.next.pixels[..status_start],
                    );
                }
            }
        }
        Ok(Some(census))
    }
}

const META_LEVEL_EPISODE_DOMAIN: u64 = 1 << 63;
const META_LEVEL_EPISODE_STRIDE: u64 = 256;

fn meta_level_episode_id(meta_episode_id: u64, level_index: usize) -> Result<u64> {
    ensure!(
        level_index < META_LEVEL_EPISODE_STRIDE as usize,
        "meta level index exceeds reserved id stride"
    );
    let offset = meta_episode_id
        .checked_mul(META_LEVEL_EPISODE_STRIDE)
        .and_then(|base| base.checked_add(level_index as u64))
        .context("meta episode id exceeds reserved namespace")?;
    ensure!(
        offset < META_LEVEL_EPISODE_DOMAIN,
        "meta episode id exceeds reserved namespace"
    );
    Ok(META_LEVEL_EPISODE_DOMAIN | offset)
}

fn generate_meta_level(
    seed: u64,
    meta_episode_id: u64,
    split: V5DataSplit,
    config: &MetaEpisodeConfig,
    level_index: usize,
    operator: EpisodeOperator,
) -> Result<MetaLevel> {
    let episode_id = meta_level_episode_id(meta_episode_id, level_index)?;
    let scenario = scenario_for_v5(seed, episode_id, split, config.content_size);
    let sim = Simulator::new(scenario.clone());
    // Movement randomness never depends on the operator, so a true episode and
    // its shuffled control share byte-identical layouts and walks per level.
    let mut rng = seeded_v5_rng(seed, episode_id, split, 0x4D45_5441_4C56);
    let mut state = State::initial(&scenario);
    let mut trajectory = Vec::with_capacity(config.steps_per_level);
    for _ in 0..config.steps_per_level {
        let action = Action::moves()[rng.random_range(0..4)];
        let next = apply_action(&sim, &state, action);
        trajectory.push(sample_from_transition_goal_free(
            &scenario,
            &state,
            &next,
            action,
            "meta_episode_v5",
            trajectory.len() as u64,
        )?);
        state = next;
    }
    let mut decision_groups = Vec::new();
    if config.operator_decisions_per_level > 0 {
        let mut transitions = Vec::with_capacity(config.operator_decisions_per_level);
        transitions.push(operator_sample_from_state(
            &scenario,
            &state,
            ArcAction::new(5, None, None)?,
            operator,
            "meta_episode_v5",
            config.steps_per_level as u64,
        )?);
        let current = render_state_padded(&scenario, &state)?;
        for (x, y) in stratified_action6_coordinates(&current, config.content_size)?
            .into_iter()
            .take(config.operator_decisions_per_level - 1)
        {
            transitions.push(operator_sample_from_state(
                &scenario,
                &state,
                ArcAction::new(6, Some(x), Some(y))?,
                operator,
                "meta_episode_v5",
                config.steps_per_level as u64,
            )?);
        }
        decision_groups.push(MetaDecisionGroup {
            id: MetaDecisionGroupId {
                level_index,
                episode_id,
                group_index: 0,
                current_fingerprint: frame_fingerprint(&transitions[0].current),
            },
            transitions,
        });
    }
    let trajectory_id = format!("meta-v5/{split:?}/{seed}/{meta_episode_id}/level{level_index}");
    for transition in &mut trajectory {
        transition.provenance.trajectory_id = trajectory_id.clone();
    }
    for transition in decision_groups
        .iter_mut()
        .flat_map(|group| group.transitions.iter_mut())
    {
        transition.provenance.trajectory_id = trajectory_id.clone();
    }
    Ok(MetaLevel {
        level_index,
        episode_id,
        operator,
        trajectory,
        decision_groups,
    })
}

fn assemble_meta_episode(
    seed: u64,
    meta_episode_id: u64,
    split: V5DataSplit,
    config: &MetaEpisodeConfig,
    operator: EpisodeOperator,
    shuffled_control: bool,
    level_operator: impl Fn(usize) -> Result<EpisodeOperator>,
) -> Result<MetaEpisode> {
    let mut levels = Vec::with_capacity(config.levels);
    let mut level_boundaries = vec![0usize];
    for level_index in 0..config.levels {
        let level = generate_meta_level(
            seed,
            meta_episode_id,
            split,
            config,
            level_index,
            level_operator(level_index)?,
        )?;
        level_boundaries.push(level_boundaries[level_index] + level.trajectory.len());
        levels.push(level);
    }
    let episode = MetaEpisode {
        seed,
        meta_episode_id,
        data_split: split,
        config: *config,
        operator,
        shuffled_control,
        levels,
        level_boundaries,
    };
    // Families are an explicit deterministic input to the construction.
    episode.validate()?;
    Ok(episode)
}

/// Deterministic multi-level shared-rule episode: one hidden
/// [`EpisodeOperator`] sampled once and applied to every level's
/// ACTION5/ACTION6 decision points, across freshly generated per-level layouts.
pub fn generate_meta_episode(
    seed: u64,
    meta_episode_id: u64,
    split: V5DataSplit,
    families: &OperatorFamilySplit,
    config: &MetaEpisodeConfig,
) -> Result<MetaEpisode> {
    config.validate()?;
    families.validate()?;
    let mut rng = seeded_v5_rng(seed, meta_episode_id, split, 0x4D45_5441_0005);
    let operator = sampled_operator(families, split, &mut rng)?;
    let episode = assemble_meta_episode(
        seed,
        meta_episode_id,
        split,
        config,
        operator,
        false,
        |_| Ok(operator),
    )?;
    episode.validate_with_families(families)?;
    Ok(episode)
}

/// Shuffled-rule negative control for [`generate_meta_episode`].
///
/// Identical inputs produce identical layouts and movement walks (the level
/// RNG lane is operator-independent), but every later level (`level_index >=
/// 1`) independently draws from the same split marginal as level 0. Repeats
/// are allowed: conditioning on the earlier rule therefore gives no positive
/// or negative information about a later rule. `operator` records the level-0
/// rule and `shuffled_control` is set.
pub fn generate_meta_episode_shuffled_control(
    seed: u64,
    meta_episode_id: u64,
    split: V5DataSplit,
    families: &OperatorFamilySplit,
    config: &MetaEpisodeConfig,
) -> Result<MetaEpisode> {
    config.validate()?;
    families.validate()?;
    let mut rng = seeded_v5_rng(seed, meta_episode_id, split, 0x4D45_5441_0005);
    let operator = sampled_operator(families, split, &mut rng)?;
    let episode = assemble_meta_episode(
        seed,
        meta_episode_id,
        split,
        config,
        operator,
        true,
        |level_index| {
            if level_index == 0 {
                return Ok(operator);
            }
            let mut level_rng = seeded_v5_rng(
                seed,
                meta_level_episode_id(meta_episode_id, level_index)?,
                split,
                0x4D45_5348_5546,
            );
            sampled_operator(families, split, &mut level_rng)
        },
    )?;
    episode.validate_with_families(families)?;
    Ok(episode)
}

/// Counts for one rule-identifiability census bucket.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CensusCounts {
    pub later_operator_points: usize,
    pub alternative_outcome_sensitive: usize,
    /// `alternative_outcome_sensitive / later_operator_points`; zero when
    /// there are no points.
    pub alternative_outcome_sensitive_fraction: f32,
}

impl CensusCounts {
    fn add(&mut self, sensitive: bool) {
        self.later_operator_points += 1;
        self.alternative_outcome_sensitive += usize::from(sensitive);
    }

    fn finalize(&mut self) {
        self.alternative_outcome_sensitive_fraction = if self.later_operator_points == 0 {
            0.0
        } else {
            self.alternative_outcome_sensitive as f32 / self.later_operator_points as f32
        };
    }
}

/// Census bucket for one shared-rule operator family.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FamilyCensus {
    pub family: OperatorFamily,
    pub counts: CensusCounts,
}

/// Census bucket for one later-level index.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LevelCensus {
    pub level_index: usize,
    pub counts: CensusCounts,
}

/// Model-free identifiability census over generated meta-episodes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuleIdentifiabilityCensus {
    pub episodes: usize,
    /// Episodes whose level-0 operator outcomes uniquely determine the true
    /// family within the preregistered split candidate set.
    pub earlier_rule_identified_episodes: usize,
    /// Total levels across the censused episodes.
    pub levels: usize,
    pub overall: CensusCounts,
    pub per_family: Vec<FamilyCensus>,
    /// Only later levels (`level_index >= 1`) can contain decision points.
    pub per_level: Vec<LevelCensus>,
}

fn census_candidate_families(
    true_family: OperatorFamily,
    families: &OperatorFamilySplit,
) -> Result<Vec<OperatorFamily>> {
    let same_split = if families.train.contains(&true_family) {
        &families.train
    } else {
        &families.held_out
    };
    ensure!(
        same_split.len() >= 2,
        "rule identifiability is undefined for a singleton operator split"
    );
    Ok(same_split.clone())
}

/// Census whether earlier-level rule identity is relevant to later outcomes.
///
/// First, level-0 ACTION5/ACTION6 outcomes filter the same-split candidate
/// families. An episode is identified only when this leaves exactly its true
/// family. A later operator point is alternative-outcome-sensitive only when
/// the earlier rule is identified and some a-priori same-split alternative
/// produces a different board under the same fixed action. This establishes a
/// model-free opportunity for useful cross-level memory; it does not show that
/// memory changes the optimal action, task outcome, policy value, or that a
/// learned model exploits the information.
pub fn census_rule_identifiability(
    episodes: &[MetaEpisode],
    families: &OperatorFamilySplit,
) -> Result<RuleIdentifiabilityCensus> {
    families.validate()?;
    ensure!(!episodes.is_empty(), "rule census needs meta-episodes");
    let status_start = V5_PLAYFIELD_HEIGHT * FRAME_SIDE;
    let mut overall = CensusCounts::default();
    let mut per_family = BTreeMap::<OperatorFamily, CensusCounts>::new();
    let mut per_level = BTreeMap::<usize, CensusCounts>::new();
    let mut levels = 0usize;
    let mut earlier_rule_identified_episodes = 0usize;
    for episode in episodes {
        episode.validate_with_families(families)?;
        ensure!(
            !episode.shuffled_control,
            "rule-identifiability census requires stable-rule episodes"
        );
        levels += episode.levels.len();
        let true_operator = episode.operator;
        let candidates = census_candidate_families(true_operator.family, families)?;
        let first_level = episode
            .levels
            .first()
            .context("validated meta-episode has no first level")?;
        let mut posterior = candidates.clone();
        for transition in first_level
            .decision_groups
            .iter()
            .flat_map(|group| group.transitions.iter())
        {
            let rect = ContentRect {
                x: 0,
                y: 0,
                width: u8::try_from(transition.provenance.content_width)
                    .map_err(|_| anyhow!("content width does not fit u8"))?,
                height: u8::try_from(transition.provenance.content_height)
                    .map_err(|_| anyhow!("content height does not fit u8"))?,
            };
            posterior.retain(|&family| {
                apply_episode_operator(
                    &transition.current,
                    &transition.action,
                    rect,
                    EpisodeOperator {
                        family,
                        ..true_operator
                    },
                )
                .is_ok_and(|candidate| {
                    candidate.pixels[..status_start] == transition.next.pixels[..status_start]
                })
            });
        }
        ensure!(
            posterior.contains(&true_operator.family),
            "true operator was inconsistent with its own level-0 outcomes"
        );
        let earlier_rule_identified = posterior.len() == 1 && posterior[0] == true_operator.family;
        earlier_rule_identified_episodes += usize::from(earlier_rule_identified);
        let alternatives = candidates
            .into_iter()
            .filter(|family| *family != true_operator.family)
            .collect::<Vec<_>>();
        ensure!(
            !alternatives.is_empty(),
            "no alternative operator family to census against {:?}",
            true_operator.family
        );
        for level in episode.levels.iter().filter(|level| level.level_index >= 1) {
            for transition in level
                .decision_groups
                .iter()
                .flat_map(|group| group.transitions.iter())
            {
                let rect = ContentRect {
                    x: 0,
                    y: 0,
                    width: u8::try_from(transition.provenance.content_width)
                        .map_err(|_| anyhow!("content width does not fit u8"))?,
                    height: u8::try_from(transition.provenance.content_height)
                        .map_err(|_| anyhow!("content height does not fit u8"))?,
                };
                let truth = apply_episode_operator(
                    &transition.current,
                    &transition.action,
                    rect,
                    level.operator,
                )?;
                let alternative_outcome_sensitive = earlier_rule_identified
                    && alternatives.iter().any(|&family| {
                        apply_episode_operator(
                            &transition.current,
                            &transition.action,
                            rect,
                            EpisodeOperator {
                                family,
                                ..level.operator
                            },
                        )
                        .is_ok_and(|alternative| {
                            alternative.pixels[..status_start] != truth.pixels[..status_start]
                        })
                    });
                overall.add(alternative_outcome_sensitive);
                per_family
                    .entry(true_operator.family)
                    .or_default()
                    .add(alternative_outcome_sensitive);
                per_level
                    .entry(level.level_index)
                    .or_default()
                    .add(alternative_outcome_sensitive);
            }
        }
    }
    overall.finalize();
    let per_family = per_family
        .into_iter()
        .map(|(family, mut counts)| {
            counts.finalize();
            FamilyCensus { family, counts }
        })
        .collect();
    let per_level = per_level
        .into_iter()
        .map(|(level_index, mut counts)| {
            counts.finalize();
            LevelCensus {
                level_index,
                counts,
            }
        })
        .collect();
    Ok(RuleIdentifiabilityCensus {
        episodes: episodes.len(),
        earlier_rule_identified_episodes,
        levels,
        overall,
        per_family,
        per_level,
    })
}

// ---------------------------------------------------------------------------
// ADR 0005 model-side helper (added by the model/context-channel change; the
// generator owner may relocate it). Mutable access to composed rows so the
// trainer's context plumbing can be tested with hand-built context windows
// before the v6 generator emits them.
// ---------------------------------------------------------------------------
impl MixedStreamBatch {
    pub fn transitions_mut(&mut self) -> impl Iterator<Item = &mut TransitionSample> {
        self.samples.iter_mut().map(|sample| &mut sample.transition)
    }
}

/// Movement transitions per level of a TRAINING Learning History (ADR 0005
/// §2.2); see [`LearningHistoryConfig::training`].
pub const LEARNING_HISTORY_STEPS_PER_LEVEL: usize = 6;
/// Level counts a TRAINING Learning History may have (ADR 0005 §2.2).
pub const LEARNING_HISTORY_LEVELS: [usize; 3] = [2, 3, 4];
/// Movement transitions per level of the EXTENDED evaluation histories
/// ([`LearningHistoryConfig::extended_evaluation`]): long enough for the
/// production Channel B warm-up (`ADAPT_MIN_LEVEL_TRANSITIONS = 8`) to be met
/// inside one level.
pub const LEARNING_HISTORY_EVAL_STEPS_PER_LEVEL: usize = 24;
/// Level counts of the extended evaluation histories: the shortest history
/// (`3 * 25 = 75` chronological transitions) covers the preregistered §5.2
/// prefix `t = 32` plus its four scoring rows.
pub const LEARNING_HISTORY_EVAL_LEVELS: [usize; 3] = [3, 4, 5];

/// Level shape of a Learning History: movement rows per level and the level
/// counts a meta-episode may draw from. [`Self::training`] (= `Default`) is
/// the ADR 0005 §2.2 training shape and reproduces the stream byte for byte;
/// the evaluators (`p2-eval --twin-memorization`, `--adaptation-falsifier`)
/// may request longer levels through [`Self::extended_evaluation`] or an
/// explicit shape. Every draw that does not depend on the shape (content
/// size, goal family, operator, twin family, per-row context length) shares
/// its seeded lane across shapes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningHistoryConfig {
    /// Epsilon-greedy movement rows per level; one realized operator row
    /// follows them, so a level has `steps_per_level + 1` chronological
    /// transitions.
    pub steps_per_level: usize,
    /// Level counts drawn uniformly per meta-episode.
    pub levels: Vec<usize>,
}

impl Default for LearningHistoryConfig {
    fn default() -> Self {
        Self::training()
    }
}

impl LearningHistoryConfig {
    /// The training shape (`6` movement rows, levels in `{2, 3, 4}`).
    pub fn training() -> Self {
        Self {
            steps_per_level: LEARNING_HISTORY_STEPS_PER_LEVEL,
            levels: LEARNING_HISTORY_LEVELS.to_vec(),
        }
    }

    /// The extended evaluation shape (`24` movement rows, levels in
    /// `{3, 4, 5}`; at least `75` chronological transitions).
    pub fn extended_evaluation() -> Self {
        Self {
            steps_per_level: LEARNING_HISTORY_EVAL_STEPS_PER_LEVEL,
            levels: LEARNING_HISTORY_EVAL_LEVELS.to_vec(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.steps_per_level >= 1,
            "a learning history level needs at least one movement row"
        );
        ensure!(
            !self.levels.is_empty() && self.levels.iter().all(|levels| *levels >= 1),
            "learning history level counts must be non-empty and positive"
        );
        ensure!(
            u16::try_from(self.levels.iter().copied().max().unwrap_or(0)).is_ok(),
            "learning history level counts must fit u16"
        );
        Ok(())
    }

    pub fn is_training(&self) -> bool {
        *self == Self::training()
    }

    /// Chronological transitions per level (movement rows plus the realized
    /// operator row).
    pub fn transitions_per_level(&self) -> usize {
        self.steps_per_level + 1
    }

    /// Chronological transitions of the shortest history this shape can draw.
    pub fn min_chronological_transitions(&self) -> usize {
        self.levels.iter().copied().min().unwrap_or(0) * self.transitions_per_level()
    }
}
/// Epsilon-greedy schedule endpoints over the chronological transitions:
/// fully exploratory at the first transition, mostly competent at the last.
const LEARNING_HISTORY_EPSILON: (f32, f32) = (1.0, 0.2);
/// Rows forced to carry no context so the no-context prior keeps training
/// (`P(K = 0) >= 0.10`, ADR 0005 §2.2).
const LEARNING_HISTORY_NO_CONTEXT_PROBABILITY: f64 = 0.10;
/// Exact-solver lookahead (actions) for the competent arm of the policy. A
/// goal farther than this falls back to exploration, bounding search cost on
/// large boards.
const LEARNING_HISTORY_SOLVER_HORIZON: u16 = 24;
/// Realized operator row plus its same-state counterfactual alternatives:
/// ACTION5 and four stratified ACTION6 coordinates.
const LEARNING_HISTORY_OPERATOR_ACTIONS: usize = 5;

/// One Learning History: a multi-level episode under one stable Hidden Rule
/// (operator family, colour binding, goal family) whose rows carry Context
/// Windows of strictly earlier transitions from the same episode.
///
/// Each level is a fresh layout: an epsilon-decaying policy over the exact
/// shortest-path solver walks [`LearningHistoryConfig::steps_per_level`]
/// movement transitions (training: [`LEARNING_HISTORY_STEPS_PER_LEVEL`]),
/// then the level ends with one realized ACTION5/ACTION6 row whose
/// next frame is the hidden operator's outcome. The operator alternatives from
/// that same frame are emitted right after it as counterfactual sidecar rows
/// (same Context Window, gate-2 support). Operators act on frames only, so an
/// operator row can only end a level; the exact simulator has no state for a
/// painted or teleported board.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearningHistory {
    pub seed: u64,
    pub meta_episode_id: u64,
    pub data_split: V5DataSplit,
    pub content_size: u8,
    pub levels: usize,
    /// Raw (pre-augmentation) hidden operator.
    pub operator: EpisodeOperator,
    /// Hidden goal family (`0..6`), a seeded draw independent of episode ids.
    pub goal_family: u8,
    pub rule_id: u64,
    /// Emission order: chronological rows, each realized operator row followed
    /// by its counterfactual alternatives.
    pub rows: Vec<TransitionSample>,
    /// Indices into `rows` of the chronological transitions, in order.
    pub chronological: Vec<usize>,
}

struct LearningHistoryDraw {
    levels: usize,
    content_size: u8,
    goal_family: u8,
    operator: EpisodeOperator,
}

fn learning_history_draw(
    seed: u64,
    meta_episode_id: u64,
    split: V5DataSplit,
    families: &OperatorFamilySplit,
    shape: &LearningHistoryConfig,
) -> Result<LearningHistoryDraw> {
    let mut rng = seeded_v5_rng(seed, meta_episode_id, split, 0x4C48_5255_4C45);
    let levels = shape.levels[rng.random_range(0..shape.levels.len())];
    let content_size = sampled_content_size(split, &mut rng);
    let goal_family = rng.random_range(0..6u8);
    let operator = sampled_operator(families, split, &mut rng)?;
    Ok(LearningHistoryDraw {
        levels,
        content_size,
        goal_family,
        operator,
    })
}

/// Context-window length draw shared by Learning Histories and the shard
/// stream (ADR 0005 §2.2): `K ~ Uniform{0..16}` with `P(K=0) >= 0.10`, clipped
/// to the rows available before the row.
pub(crate) fn sampled_context_len(rng: &mut ChaCha8Rng, available: usize) -> usize {
    let k = if rng.random_bool(LEARNING_HISTORY_NO_CONTEXT_PROBABILITY) {
        0
    } else {
        rng.random_range(0..=CONTEXT_WINDOW_MAX)
    };
    k.min(available)
}

fn generate_learning_history(
    seed: u64,
    meta_episode_id: u64,
    split: V5DataSplit,
    draw: &LearningHistoryDraw,
    operator: EpisodeOperator,
    twin_index: u8,
    shape: &LearningHistoryConfig,
) -> Result<LearningHistory> {
    let rule_id = hidden_rule_id(operator, Some(draw.goal_family));
    // Movement, realized-operator choice and context lengths never depend on
    // the operator, so twins share byte-identical frames until the rule acts.
    let mut policy_rng = seeded_v5_rng(seed, meta_episode_id, split, 0x4C48_504F_4C49);
    let mut context_rng = seeded_v5_rng(seed, meta_episode_id, split, 0x4C48_4354_5854);
    let total = draw.levels * shape.transitions_per_level();
    let (epsilon_start, epsilon_end) = LEARNING_HISTORY_EPSILON;
    let trajectory_id =
        format!("learning-history/{split:?}/{seed}/{meta_episode_id}/twin{twin_index}");
    let mut history: Vec<ContextTransition> = Vec::with_capacity(total);
    let mut rows = Vec::new();
    let mut chronological = Vec::with_capacity(total);
    let mut finalize =
        |mut row: TransitionSample, level_index: usize, history: &[ContextTransition]| {
            let t = history.len();
            let k = sampled_context_len(&mut context_rng, t);
            row.context = history[t - k..t].to_vec();
            row.family = "learning_history".into();
            row.provenance.source_kind = "learning_history".into();
            row.provenance.trajectory_id = trajectory_id.clone();
            row.provenance.operator = Some(operator);
            row.provenance.rule_id = rule_id;
            row.provenance.level_index = u16::try_from(level_index).expect("levels fit u16");
            row.provenance.available_actions = ALL_ACTIONS_AVAILABLE;
            row.provenance.context_len = k as u8;
            row
        };
    for level_index in 0..draw.levels {
        let episode_id = meta_level_episode_id(meta_episode_id, level_index)?;
        let mut scenario = scenario_for_v5(seed, episode_id, split, draw.content_size);
        let mut goal_rng = seeded_v5_rng(seed, episode_id, split, 0x4C48_474F_414C);
        rebind_hidden_goal_family(&mut scenario, draw.goal_family, &mut goal_rng);
        let goal = scenario.hidden_goal().clone();
        let scenario = Arc::new(scenario);
        let sim = Simulator::new(Arc::clone(&scenario));
        let mut state = State::initial(&scenario);
        for _ in 0..shape.steps_per_level {
            let t = history.len();
            let epsilon =
                epsilon_start + (epsilon_end - epsilon_start) * t as f32 / (total - 1) as f32;
            let explore = policy_rng.random_bool(f64::from(epsilon));
            let random_action = Action::moves()[policy_rng.random_range(0..4)];
            let action = if explore {
                random_action
            } else {
                let horizon = state
                    .actions_used
                    .saturating_add(LEARNING_HISTORY_SOLVER_HORIZON)
                    .min(scenario.action_budget);
                shortest_path(&sim, &state, &goal, horizon)
                    .and_then(|plan| plan.actions.first().copied())
                    .unwrap_or(random_action)
            };
            let next = apply_action(&sim, &state, action);
            let row = finalize(
                sample_from_transition(&scenario, &state, &next, action, &goal, t as u64)?,
                level_index,
                &history,
            );
            history.push(ContextTransition {
                current: row.current.clone(),
                action: row.action.clone(),
                next: row.next.clone(),
            });
            chronological.push(rows.len());
            rows.push(row);
            state = next;
        }
        let current = render_state_padded(&scenario, &state)?;
        let mut actions = vec![ArcAction::new(5, None, None)?];
        for (x, y) in stratified_action6_coordinates(&current, draw.content_size)? {
            actions.push(ArcAction::new(6, Some(x), Some(y))?);
        }
        debug_assert_eq!(actions.len(), LEARNING_HISTORY_OPERATOR_ACTIONS);
        let realized = policy_rng.random_range(0..actions.len());
        actions.swap(0, realized);
        let t = history.len();
        // The realized row and its same-state alternatives share one Context
        // Window of strictly earlier transitions.
        let operator_rows = actions
            .into_iter()
            .map(|action| {
                let mut row = operator_sample_from_rendered_current(
                    &scenario,
                    &state,
                    current.clone(),
                    action,
                    operator,
                    "learning_history",
                    t as u64,
                )?;
                row.goal_features = GoalFeatures::encode(&goal);
                Ok(finalize(row, level_index, &history))
            })
            .collect::<Result<Vec<_>>>()?;
        let realized_row = &operator_rows[0];
        history.push(ContextTransition {
            current: realized_row.current.clone(),
            action: realized_row.action.clone(),
            next: realized_row.next.clone(),
        });
        chronological.push(rows.len());
        rows.extend(operator_rows);
    }
    Ok(LearningHistory {
        seed,
        meta_episode_id,
        data_split: split,
        content_size: draw.content_size,
        levels: draw.levels,
        operator,
        goal_family: draw.goal_family,
        rule_id,
        rows,
        chronological,
    })
}

/// Generate a Learning History together with its Twin Episode (ADR 0005 §2.3).
///
/// The twin shares the seed, layouts, policy and every frame up to the first
/// realized operator row, but is bound to a different operator family drawn
/// from the same split marginal; the family is chosen so that at least one
/// realized operator row maps the identical current frame to a different next
/// frame (mutual exclusivity by construction). If no alternative family
/// differs on this episode the first alternative is used and the twin census
/// reports the pair as non-divergent.
///
/// Training shape ([`LearningHistoryConfig::training`]); see
/// [`generate_learning_history_pair_with`] for an explicit shape.
pub fn generate_learning_history_pair(
    seed: u64,
    meta_episode_id: u64,
    split: V5DataSplit,
    families: &OperatorFamilySplit,
) -> Result<(LearningHistory, LearningHistory)> {
    generate_learning_history_pair_with(
        seed,
        meta_episode_id,
        split,
        families,
        &LearningHistoryConfig::training(),
    )
}

/// [`generate_learning_history_pair`] under an explicit level shape. The
/// training composer always passes [`LearningHistoryConfig::training`]; the
/// evaluators may request longer levels.
pub fn generate_learning_history_pair_with(
    seed: u64,
    meta_episode_id: u64,
    split: V5DataSplit,
    families: &OperatorFamilySplit,
    shape: &LearningHistoryConfig,
) -> Result<(LearningHistory, LearningHistory)> {
    families.validate()?;
    shape.validate()?;
    let draw = learning_history_draw(seed, meta_episode_id, split, families, shape)?;
    let primary =
        generate_learning_history(seed, meta_episode_id, split, &draw, draw.operator, 0, shape)?;
    let same_split = match split {
        V5DataSplit::HeldOutOperator(_) => &families.held_out,
        _ => &families.train,
    };
    let mut alternatives = same_split
        .iter()
        .copied()
        .filter(|family| *family != draw.operator.family)
        .collect::<Vec<_>>();
    if alternatives.is_empty() {
        // A singleton held-out split (the default holds out SwapRegion alone)
        // cannot supply a differing twin rule from within itself. The twin only
        // has to DIFFER from the primary for mutual exclusivity (ADR 0005
        // §2.3); the primary keeps the held-out rule, so no held-out rule
        // becomes trainable through this fallback.
        alternatives = families
            .train
            .iter()
            .chain(families.held_out.iter())
            .copied()
            .filter(|family| *family != draw.operator.family)
            .collect::<Vec<_>>();
    }
    ensure!(
        !alternatives.is_empty(),
        "twin episodes need at least two operator families overall"
    );
    let mut rng = seeded_v5_rng(seed, meta_episode_id, split, 0x4C48_5457_494E);
    let rotation = rng.random_range(0..alternatives.len());
    alternatives.rotate_left(rotation);
    let rect = ContentRect {
        x: 0,
        y: 0,
        width: draw.content_size,
        height: draw.content_size,
    };
    let realized_operator_rows = primary
        .chronological
        .iter()
        .map(|&index| &primary.rows[index])
        .filter(|row| matches!(row.action.id, 5 | 6))
        .collect::<Vec<_>>();
    let diverges = |family: OperatorFamily| -> Result<bool> {
        let alternative = EpisodeOperator {
            family,
            ..draw.operator
        };
        for row in &realized_operator_rows {
            let outcome = apply_episode_operator(&row.current, &row.action, rect, alternative)?;
            if outcome.pixels[..V5_PLAYFIELD_HEIGHT * FRAME_SIDE]
                != row.next.pixels[..V5_PLAYFIELD_HEIGHT * FRAME_SIDE]
            {
                return Ok(true);
            }
        }
        Ok(false)
    };
    let mut twin_family = alternatives[0];
    for family in &alternatives {
        if diverges(*family)? {
            twin_family = *family;
            break;
        }
    }
    let twin = generate_learning_history(
        seed,
        meta_episode_id,
        split,
        &draw,
        EpisodeOperator {
            family: twin_family,
            ..draw.operator
        },
        1,
        shape,
    )?;
    Ok((primary, twin))
}

/// Model-free mutual-exclusivity census over Twin Episodes (ADR 0005 §2.3, §5.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TwinExclusivityCensus {
    pub pairs: usize,
    /// Pairs whose emitted frames or actions differ before any transition has
    /// revealed the rule: the Hidden Rule would be readable from a single
    /// frame. Must be zero.
    pub single_frame_rule_identifiable: usize,
    /// Pairs with a first differing transition that maps a byte-identical
    /// current frame and action to different next frames.
    pub divergent_pairs: usize,
}

/// Index of the first differing transition of a twin pair, or `Err` when the
/// pair differs before any rule-revealing transition, or `Ok(None)` when the
/// twins never diverge.
pub fn twin_first_divergence(
    primary: &LearningHistory,
    twin: &LearningHistory,
) -> Result<Option<usize>> {
    ensure!(
        primary.rows.len() == twin.rows.len(),
        "twin episodes must emit the same number of rows"
    );
    for (index, (left, right)) in primary.rows.iter().zip(&twin.rows).enumerate() {
        ensure!(
            left.current == right.current && left.action == right.action,
            "twin episodes differ before the first rule-revealing transition (row {index})"
        );
        if left.next != right.next {
            return Ok(Some(index));
        }
    }
    Ok(None)
}

pub fn census_twin_exclusivity(
    pairs: &[(LearningHistory, LearningHistory)],
) -> TwinExclusivityCensus {
    let mut census = TwinExclusivityCensus {
        pairs: pairs.len(),
        ..TwinExclusivityCensus::default()
    };
    for (primary, twin) in pairs {
        match twin_first_divergence(primary, twin) {
            Err(_) => census.single_frame_rule_identifiable += 1,
            Ok(Some(_)) => census.divergent_pairs += 1,
            Ok(None) => {}
        }
    }
    census
}

fn compose_learning_history_unit(
    config: &MixedStreamConfig,
    split: V5DataSplit,
    meta_episode_id: u64,
) -> Result<Vec<V5Sample>> {
    // Training shape only: the stream never takes an evaluator's level shape.
    let (primary, twin) = generate_learning_history_pair(
        config.seed,
        meta_episode_id,
        split,
        &config.operator_families,
    )?;
    let operator = primary.operator;
    // Twins share one augmentation unit so their emitted frames stay
    // byte-identical until the rule acts; interleaving keeps a truncated
    // unit balanced between the two rules.
    augment_v5_unit(
        interleave(primary.rows, twin.rows),
        config,
        split,
        MixedStreamKind::LearningHistories,
        operator,
        meta_episode_id,
    )
}

/// One held-out Learning History rendered under the v6 interface contract
/// (ADR 0005 §5.2 adaptation-falsifier population): the primary rows of
/// [`generate_learning_history_pair`] after the same per-unit augmentation the
/// `LearningHistories` stream applies (content placement, D4 symmetry, colour
/// permutation, goal dropout), with the emission order preserved so
/// `chronological` still indexes `rows`.
#[derive(Clone, Debug)]
pub struct AugmentedLearningHistory {
    pub meta_episode_id: u64,
    pub levels: usize,
    /// Emission order of [`LearningHistory::rows`], augmented one-to-one.
    pub rows: Vec<V5Sample>,
    /// Indices into `rows` of the chronological transitions, in order.
    pub chronological: Vec<usize>,
}

impl AugmentedLearningHistory {
    /// The `index`-th chronological transition (0-based).
    pub fn chronological_row(&self, index: usize) -> &V5Sample {
        &self.rows[self.chronological[index]]
    }

    /// Channel A window for a decision at chronological position `index`: the
    /// last `<= CONTEXT_WINDOW_MAX` chronological transitions strictly before
    /// it, in order (episode scope).
    pub fn context_window_before(&self, index: usize) -> Vec<ContextTransition> {
        let start = index.saturating_sub(CONTEXT_WINDOW_MAX);
        (start..index)
            .map(|position| {
                let row = &self.chronological_row(position).transition;
                ContextTransition {
                    current: row.current.clone(),
                    action: row.action.clone(),
                    next: row.next.clone(),
                }
            })
            .collect()
    }
}

/// Generate the primary Learning History `meta_episode_id` of `config.seed`
/// on `split` and augment it exactly like the stream unit would (the twin is
/// generated for the family draw and discarded). Requires `data_contract_v6`.
/// Training shape; see [`augmented_learning_history_with`].
pub fn augmented_learning_history(
    config: &MixedStreamConfig,
    split: V5DataSplit,
    meta_episode_id: u64,
) -> Result<AugmentedLearningHistory> {
    augmented_learning_history_with(
        config,
        split,
        meta_episode_id,
        &LearningHistoryConfig::training(),
    )
}

/// [`augmented_learning_history`] under an explicit level shape.
pub fn augmented_learning_history_with(
    config: &MixedStreamConfig,
    split: V5DataSplit,
    meta_episode_id: u64,
    shape: &LearningHistoryConfig,
) -> Result<AugmentedLearningHistory> {
    Ok(augmented_learning_history_pair_with(config, split, meta_episode_id, shape)?.primary)
}

/// Model-free divergence facts of one Twin Episode pair (ADR 0005 §2.3),
/// computed on the raw (pre-augmentation) rows in chronological order. The
/// augmentation is one bijection shared by both twins, so every equality
/// below also holds on the augmented rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TwinPairDivergence {
    /// The pair differs in a current frame or action before any next frame
    /// differs: the Hidden Rule would be readable from a single frame. Must
    /// be false (see [`TwinExclusivityCensus`]).
    pub single_frame_rule_identifiable: bool,
    /// Chronological index of the first transition whose next frame differs
    /// between the twins; `None` when the twins never diverge.
    pub first_divergence: Option<usize>,
    /// Chronological indices whose next frame differs between the twins for
    /// a byte-identical current frame and action ("outcome-changing" rows:
    /// the realized operator rows on which the two rules disagree).
    pub outcome_changing_positions: Vec<usize>,
    /// Chronological indices whose current frame or action differs between
    /// the twins (every level restarts from its own initial layout, so this
    /// is zero by construction; reported so a generator regression cannot
    /// hide inside the exclusivity claim).
    pub state_differing_positions: usize,
}

impl TwinPairDivergence {
    pub fn of(primary: &LearningHistory, twin: &LearningHistory) -> Result<Self> {
        ensure!(
            primary.chronological == twin.chronological,
            "twin episodes must emit the same chronological layout"
        );
        let single_frame_rule_identifiable = twin_first_divergence(primary, twin).is_err();
        let mut first_divergence = None;
        let mut outcome_changing_positions = Vec::new();
        let mut state_differing_positions = 0usize;
        for (position, &index) in primary.chronological.iter().enumerate() {
            let (left, right) = (&primary.rows[index], &twin.rows[index]);
            let same_state = left.current == right.current && left.action == right.action;
            state_differing_positions += usize::from(!same_state);
            if left.next != right.next {
                first_divergence.get_or_insert(position);
                if same_state {
                    outcome_changing_positions.push(position);
                }
            }
        }
        Ok(Self {
            single_frame_rule_identifiable,
            first_divergence,
            outcome_changing_positions,
            state_differing_positions,
        })
    }
}

/// A Twin Episode pair rendered under the v6 interface contract with ONE
/// per-unit augmentation (the same content placement, D4 symmetry, colour
/// permutation and per-row goal-dropout sequence for both twins), so the
/// augmented twins stay byte-identical until the Hidden Rule acts, exactly
/// as in the `LearningHistories` stream.
#[derive(Clone, Debug)]
pub struct AugmentedTwinPair {
    pub primary: AugmentedLearningHistory,
    pub twin: AugmentedLearningHistory,
    pub divergence: TwinPairDivergence,
}

/// Generate the Twin Episode pair `meta_episode_id` of `config.seed` on
/// `split` under `shape` and augment both twins like the stream unit would.
/// Requires `data_contract_v6`.
pub fn augmented_learning_history_pair_with(
    config: &MixedStreamConfig,
    split: V5DataSplit,
    meta_episode_id: u64,
    shape: &LearningHistoryConfig,
) -> Result<AugmentedTwinPair> {
    ensure!(
        config.data_contract_v6,
        "an augmented learning history requires data_contract_v6"
    );
    let (primary, twin) = generate_learning_history_pair_with(
        config.seed,
        meta_episode_id,
        split,
        &config.operator_families,
        shape,
    )?;
    let divergence = TwinPairDivergence::of(&primary, &twin)?;
    let augment = |history: LearningHistory| -> Result<AugmentedLearningHistory> {
        let operator = history.operator;
        let expected = history.rows.len();
        // Seeded by (`config.seed`, `meta_episode_id`, `split`) only: the
        // same unit augmentation and dropout sequence for either twin.
        let rows = augment_v5_unit(
            history.rows,
            config,
            split,
            MixedStreamKind::LearningHistories,
            operator,
            meta_episode_id,
        )?;
        ensure!(
            rows.len() == expected,
            "unit augmentation must map learning-history rows one-to-one"
        );
        Ok(AugmentedLearningHistory {
            meta_episode_id,
            levels: history.levels,
            rows,
            chronological: history.chronological,
        })
    };
    Ok(AugmentedTwinPair {
        primary: augment(primary)?,
        twin: augment(twin)?,
        divergence,
    })
}

fn compose_learning_histories_stream(
    config: &MixedStreamConfig,
    split: V5DataSplit,
    count: usize,
    batch_index: u64,
    first_unit_index: u64,
) -> Result<(Vec<V5Sample>, u64)> {
    let min_rows_per_unit = 2
        * LEARNING_HISTORY_LEVELS[0]
        * (LEARNING_HISTORY_STEPS_PER_LEVEL + LEARNING_HISTORY_OPERATOR_ACTIONS);
    let mut samples = Vec::with_capacity(count);
    let mut next_unit_index = first_unit_index;
    while samples.len() < count {
        let wave_units = (count - samples.len()).div_ceil(min_rows_per_unit);
        let units = (0..wave_units)
            .into_par_iter()
            .map(|offset| {
                compose_learning_history_unit(
                    config,
                    split,
                    mixed_stream_episode_id(
                        batch_index,
                        next_unit_index
                            .checked_add(offset as u64)
                            .context("mixed stream unit index overflow")?,
                    )?,
                )
            })
            .collect::<Vec<_>>();
        for unit in units {
            if samples.len() == count {
                break;
            }
            let mut unit = unit?;
            ensure!(!unit.is_empty(), "learning-history unit produced no rows");
            unit.truncate(count - samples.len());
            samples.append(&mut unit);
            next_unit_index = next_unit_index
                .checked_add(1)
                .context("mixed stream unit index overflow")?;
        }
    }
    Ok((samples, next_unit_index))
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
        assert!(ArcAction::new(0, None, None).is_ok());
        assert!(ArcAction::new(0, None, None).unwrap().to_tofy().is_err());
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
        assert_eq!(movement.branches().len(), FACTUAL_BRANCHES_PER_GROUP);
        assert!(movement
            .branches()
            .windows(2)
            .all(|pair| pair[0].transition.current == pair[1].transition.current));
        assert!(movement
            .branches()
            .iter()
            .all(|branch| !branch.status_changed_cells.is_empty()));

        let coordinate = generate_factual_branch_group(17, 3, Split::Train)?;
        assert_eq!(coordinate.branches().len(), FACTUAL_BRANCHES_PER_GROUP);
        assert_eq!(
            coordinate
                .branches()
                .iter()
                .filter(|branch| branch.transition.action.id == 6)
                .count(),
            4
        );
        assert!(coordinate
            .branches()
            .iter()
            .any(|branch| branch.board_effect.changed));
        assert_eq!(
            coordinate.effect_equivalence_matrix().len(),
            FACTUAL_BRANCHES_PER_GROUP
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
        assert_eq!(
            reconstructed.pairwise_board_effect_labels(),
            expected.pairwise_board_effect_labels()
        );
        assert_eq!(
            reconstructed.pairwise_board_effect_labels().len(),
            2 * FACTUAL_BRANCHES_PER_GROUP * (FACTUAL_BRANCHES_PER_GROUP - 1) / 2
        );
        assert_eq!(
            reconstructed.group_ranges(),
            &[
                0..FACTUAL_BRANCHES_PER_GROUP,
                FACTUAL_BRANCHES_PER_GROUP..2 * FACTUAL_BRANCHES_PER_GROUP
            ]
        );
        assert!(FactualBatch::from_rows(&shuffled[..2]).is_err());
        Ok(())
    }

    #[test]
    fn mixed_units_share_frame_and_mask_storage() -> Result<()> {
        let config = MixedStreamConfig {
            batch_size: 20,
            seed: 0x5A4E_0001,
            schedule: foundation_v2_stream_schedule,
            ..MixedStreamConfig::default()
        };
        config.validate()?;
        let batch = compose_mixed_stream_batch(&config, 0.0, 0, V5DataSplit::Train)?;
        let factual_range = batch
            .factual_group_ranges()
            .first()
            .expect("small mixed batch contains a complete factual group");
        let factual = &batch.samples()[factual_range.clone()];
        assert!(factual.windows(2).all(|pair| {
            pair[0].transition.current.pixels.allocation_id()
                == pair[1].transition.current.pixels.allocation_id()
        }));
        assert!(factual.windows(2).all(|pair| {
            pair[0].content_mask.values.allocation_id()
                == pair[1].content_mask.values.allocation_id()
        }));
        let chained = batch
            .samples()
            .windows(2)
            .filter(|pair| {
                pair[0].provenance.stream == MixedStreamKind::Exploration
                    && pair[1].provenance.stream == MixedStreamKind::Exploration
                    && pair[0].transition.episode_id == pair[1].transition.episode_id
            })
            .collect::<Vec<_>>();
        assert!(!chained.is_empty());
        assert!(chained.iter().all(|pair| {
            pair[0].transition.next.pixels.allocation_id()
                == pair[1].transition.current.pixels.allocation_id()
        }));
        Ok(())
    }

    #[test]
    fn d4_augmentation_keeps_oracle_agent_coordinates_in_frame() -> Result<()> {
        for size in V5_CONTENT_SIZES {
            let source_rect = ContentRect {
                x: 0,
                y: 0,
                width: size,
                height: size,
            };
            let target_rect = ContentRect {
                x: (FRAME_SIDE as u8 - size) / 2,
                y: (V5_PLAYFIELD_HEIGHT as u8 - size) / 2,
                width: size,
                height: size,
            };
            let source_agent = (size / 3, size / 2);
            let mut source_pixels = vec![palette::PAD; FRAME_SIDE * FRAME_SIDE];
            source_pixels[usize::from(source_agent.1) * FRAME_SIDE + usize::from(source_agent.0)] =
                palette::AGENT;
            let source = ArcFrame::new(FRAME_SIDE as u16, FRAME_SIDE as u16, source_pixels)?;
            let mut color_permutation = std::array::from_fn(|index| index as u8);
            color_permutation.swap(palette::AGENT as usize, palette::TRIGGER_BASE as usize);
            let expected_agent = color_permutation[palette::AGENT as usize];
            for transform in D4Transform::ALL {
                let augmentation = SymmetryAugmentation {
                    d4: transform,
                    color_permutation,
                };
                let mut transformed = source.clone();
                frame_with_transformed_content(
                    &mut transformed,
                    source_rect,
                    target_rect,
                    &augmentation,
                    false,
                )?;
                let mut latent = Some(vec![
                    (f32::from(source_agent.0) + 0.5) / f32::from(size) * 2.0 - 1.0,
                    (f32::from(source_agent.1) + 0.5) / f32::from(size) * 2.0 - 1.0,
                ]);
                transform_oracle_latent_d4(&mut latent, transform, source_rect, target_rect)?;
                let mut positions = Vec::new();
                for y in target_rect.y..target_rect.y + target_rect.height {
                    for x in target_rect.x..target_rect.x + target_rect.width {
                        if transformed.pixels[usize::from(y) * FRAME_SIDE + usize::from(x)]
                            == expected_agent
                        {
                            positions.push((x, y));
                        }
                    }
                }
                assert_eq!(positions.len(), 1, "{transform:?}, size {size}");
                let (x, y) = positions[0];
                let latent = latent.expect("oracle latent remains present");
                let expected_x = (f32::from(x - target_rect.x) + 0.5) / f32::from(size) * 2.0 - 1.0;
                let expected_y = (f32::from(y - target_rect.y) + 0.5) / f32::from(size) * 2.0 - 1.0;
                assert!((latent[0] - expected_x).abs() < 1e-6);
                assert!((latent[1] - expected_y).abs() < 1e-6);
            }
        }
        Ok(())
    }

    #[test]
    fn non_meta_episode_ids_reject_the_reserved_meta_domain() {
        let colliding_episode = 9_223_344_366_822u64;
        let computed = colliding_episode
            .checked_mul(MIXED_BATCH_EPISODE_STRIDE)
            .expect("documented collision fits u64");
        assert_eq!(
            computed,
            META_LEVEL_EPISODE_DOMAIN | 1268 * META_LEVEL_EPISODE_STRIDE + 50
        );
        let rejection = std::panic::catch_unwind(|| {
            generate_coordinate_one_step(7, colliding_episode, Split::Train, 1)
        });
        assert!(matches!(rejection, Ok(Err(_)) | Err(_)));
    }

    #[test]
    fn held_out_simulator_population_has_both_outcomes_within_each_action() -> Result<()> {
        let mut outcomes = BTreeMap::<u8, (bool, bool)>::new();
        for episode in (0..128).step_by(2) {
            for branch in
                generate_factual_branch_group(0xFA_C7_EA_11, episode, Split::HeldOutComposition)?
                    .branches()
            {
                if !(1..=4).contains(&branch.transition.action.id) {
                    continue;
                }
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

    // ---- ADR 0005 v6 data contract -------------------------------------

    fn v6_config(batch_size: usize, seed: u64) -> MixedStreamConfig {
        MixedStreamConfig {
            batch_size,
            seed,
            schedule: adaptation_v6_stream_schedule,
            data_contract_v6: true,
            ..MixedStreamConfig::default()
        }
    }

    fn padding_is(frame: &ArcFrame, rect: ContentRect, color: u8) -> bool {
        frame.pixels.iter().enumerate().all(|(index, &pixel)| {
            let x = (index % FRAME_SIDE) as u8;
            let y = (index / FRAME_SIDE) as u8;
            rect.contains(x, y) || pixel == color
        })
    }

    fn approx(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }

    #[test]
    fn adaptation_v6_schedule_sums_to_one_and_ramps_learning_histories() {
        for (progress, expected) in [(0.0, 0.25), (0.5, 0.375), (1.0, 0.5)] {
            let schedule = adaptation_v6_stream_schedule(progress);
            approx(schedule.total(), 1.0);
            approx(schedule.learning_histories, expected);
            // Legacy streams keep their ADR 0003 ratios.
            let legacy = foundation_v2_stream_schedule(progress);
            approx(
                schedule.random_one_step / schedule.factual_branches,
                legacy.random_one_step / legacy.factual_branches,
            );
            approx(
                schedule.hazard_one_step / schedule.exploration,
                legacy.hazard_one_step / legacy.exploration,
            );
        }
        assert_eq!(foundation_v2_stream_schedule(0.5).learning_histories, 0.0);
    }

    #[test]
    fn legacy_config_rejects_the_learning_history_schedule() {
        let config = MixedStreamConfig {
            schedule: adaptation_v6_stream_schedule,
            ..MixedStreamConfig::default()
        };
        assert!(config.validate().is_err());
        assert!(v6_config(512, 1).validate().is_ok());
    }

    #[test]
    fn v6_rows_are_whole_frame_free_background_and_status_free() -> Result<()> {
        let config = v6_config(192, 0x6006);
        let mut saw_permuted_empty = false;
        let mut saw_row_63 = false;
        let mut saw_operator_row = false;
        for batch_index in 0..4 {
            let batch = compose_mixed_stream_batch(&config, 0.5, batch_index, V5DataSplit::Train)?;
            assert!(batch.contract_v6());
            assert!(batch.stream_counts()[&MixedStreamKind::LearningHistories] > 0);
            for sample in batch.samples() {
                sample.validate()?;
                let provenance = &sample.provenance;
                let rect = provenance.content_rect;
                let permutation = provenance.augmentation.color_permutation;
                let empty = permutation[usize::from(palette::EMPTY)];
                assert!(provenance.contract_v6);
                assert!(sample.content_mask.is_all_ones());
                assert_eq!(sample.transition.provenance.operator, None);
                assert_eq!(provenance.conditioning_operator(), None);
                assert_eq!(
                    sample.transition.provenance.available_actions,
                    ALL_ACTIONS_AVAILABLE
                );
                assert_ne!(sample.transition.provenance.rule_id, 0);
                assert_eq!(
                    usize::from(sample.transition.provenance.context_len),
                    sample.transition.context.len()
                );
                assert!(padding_is(&sample.transition.current, rect, empty));
                assert!(padding_is(&sample.transition.next, rect, empty));
                assert_eq!(provenance.operator.empty_color, empty);
                assert_eq!(sample.transition.provenance.background_color, empty);
                saw_permuted_empty |= empty != palette::EMPTY;
                saw_row_63 |= usize::from(rect.y) + usize::from(rect.height) == FRAME_SIDE;
                if matches!(sample.transition.action.id, 5 | 6) {
                    saw_operator_row = true;
                    let replayed = apply_episode_operator(
                        &sample.transition.current,
                        &sample.transition.action,
                        rect,
                        provenance.operator,
                    )?;
                    assert_eq!(
                        replayed, sample.transition.next,
                        "permuted operator must replay on the augmented whole frame"
                    );
                }
            }
        }
        assert!(saw_permuted_empty, "v6 permutation must move index 0");
        assert!(saw_row_63, "v6 content must be able to reach row 63");
        assert!(saw_operator_row);
        Ok(())
    }

    #[test]
    fn v6_factual_effect_labels_cover_row_63() -> Result<()> {
        let mut transition = generate_curriculum("factual_branches", 3, 0, Split::Train)?
            .into_iter()
            .next()
            .expect("factual row");
        transition.next = transition.current.clone();
        transition.next.pixels[FRAME_SIDE * (FRAME_SIDE - 1) + 5] ^= 1;
        let legacy = FactualActionBranch::try_from_transition(transition.clone())?;
        assert!(!legacy.board_effect.changed);
        let whole = FactualActionBranch::try_from_transition_whole_frame(transition)?;
        assert!(whole.board_effect.changed);
        assert!(whole.status_changed_cells.is_empty());
        Ok(())
    }

    fn history_time(history: &LearningHistory, index: usize) -> usize {
        let earlier = history
            .chronological
            .iter()
            .filter(|&&position| position < index)
            .count();
        if history.chronological.contains(&index) {
            earlier
        } else {
            earlier - 1
        }
    }

    #[test]
    fn learning_history_context_is_chronological_and_bounded() -> Result<()> {
        let families = OperatorFamilySplit::default();
        for meta_episode_id in 0..12u64 {
            let (primary, twin) = generate_learning_history_pair(
                0x4C48,
                meta_episode_id,
                V5DataSplit::Train,
                &families,
            )?;
            for history in [&primary, &twin] {
                assert!(LEARNING_HISTORY_LEVELS.contains(&history.levels));
                assert_eq!(
                    history.rows.len(),
                    history.levels
                        * (LEARNING_HISTORY_STEPS_PER_LEVEL + LEARNING_HISTORY_OPERATOR_ACTIONS)
                );
                assert_eq!(
                    history.chronological.len(),
                    history.levels * (LEARNING_HISTORY_STEPS_PER_LEVEL + 1)
                );
                let timeline = history
                    .chronological
                    .iter()
                    .map(|&index| {
                        let row = &history.rows[index];
                        ContextTransition {
                            current: row.current.clone(),
                            action: row.action.clone(),
                            next: row.next.clone(),
                        }
                    })
                    .collect::<Vec<_>>();
                for (index, row) in history.rows.iter().enumerate() {
                    row.provenance.validate()?;
                    let t = history_time(history, index);
                    let k = row.context.len();
                    assert!(k <= CONTEXT_WINDOW_MAX && k <= t);
                    assert_eq!(usize::from(row.provenance.context_len), k);
                    assert_eq!(row.context, timeline[t - k..t]);
                    assert_eq!(row.provenance.rule_id, history.rule_id);
                    assert_eq!(row.provenance.operator, Some(history.operator));
                    assert!(usize::from(row.provenance.level_index) < history.levels);
                    assert_eq!(row.family, "learning_history");
                    assert_eq!(row.provenance.available_actions, ALL_ACTIONS_AVAILABLE);
                }
                // Each level ends with exactly one realized operator row.
                let operator_rows = history
                    .chronological
                    .iter()
                    .filter(|&&index| matches!(history.rows[index].action.id, 5 | 6))
                    .count();
                assert_eq!(operator_rows, history.levels);
            }
        }
        Ok(())
    }

    #[test]
    fn learning_history_context_length_distribution_keeps_the_no_context_prior() -> Result<()> {
        let families = OperatorFamilySplit::default();
        let mut unclipped = 0usize;
        let mut zero = 0usize;
        let mut max_k = 0usize;
        for meta_episode_id in 0..120u64 {
            let (primary, _) = generate_learning_history_pair(
                0x4B4B,
                meta_episode_id,
                V5DataSplit::Train,
                &families,
            )?;
            for (index, row) in primary.rows.iter().enumerate() {
                if history_time(&primary, index) < CONTEXT_WINDOW_MAX {
                    continue;
                }
                unclipped += 1;
                zero += usize::from(row.context.is_empty());
                max_k = max_k.max(row.context.len());
            }
        }
        assert!(unclipped > 500, "population too small: {unclipped}");
        let zero_fraction = zero as f32 / unclipped as f32;
        assert!(
            (0.10..=0.25).contains(&zero_fraction),
            "P(K=0) = {zero_fraction}"
        );
        assert_eq!(max_k, CONTEXT_WINDOW_MAX);
        Ok(())
    }

    /// The default split holds out one operator family, so a twin cannot draw a
    /// differing rule from within it; the fallback must still produce a pair
    /// rather than failing the whole held-out evaluation population.
    #[test]
    fn twins_compose_on_a_singleton_held_out_split() -> Result<()> {
        let families = OperatorFamilySplit::default();
        assert_eq!(families.held_out.len(), 1, "fixture assumes a singleton");
        let split = V5DataSplit::HeldOutOperator(families.held_out[0]);
        let (primary, twin) = generate_learning_history_pair(7, 3, split, &families)?;
        assert_ne!(primary.operator.family, twin.operator.family);
        assert_eq!(primary.rows[0].current, twin.rows[0].current);
        let batch = compose_mixed_stream_batch(
            &MixedStreamConfig {
                batch_size: 16,
                seed: 0x5A11_0000_7E51,
                schedule: adaptation_v6_stream_schedule,
                data_contract_v6: true,
                ..MixedStreamConfig::default()
            },
            1.0,
            0,
            split,
        )?;
        assert_eq!(batch.transitions().len(), 16);
        Ok(())
    }

    #[test]
    fn twin_census_finds_no_single_frame_identifiable_rule() -> Result<()> {
        let families = OperatorFamilySplit::default();
        let pairs = (0..64u64)
            .map(|meta_episode_id| {
                generate_learning_history_pair(
                    0x5457_494E,
                    meta_episode_id,
                    V5DataSplit::Train,
                    &families,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let census = census_twin_exclusivity(&pairs);
        assert_eq!(census.pairs, 64);
        assert_eq!(census.single_frame_rule_identifiable, 0);
        assert_eq!(census.divergent_pairs, 64);
        for (primary, twin) in &pairs {
            assert_eq!(primary.rows[0].current, twin.rows[0].current);
            assert_ne!(primary.operator.family, twin.operator.family);
            assert_eq!(primary.goal_family, twin.goal_family);
            assert_ne!(primary.rule_id, twin.rule_id);
            let index = twin_first_divergence(primary, twin)?.expect("twins diverge");
            let (left, right) = (&primary.rows[index], &twin.rows[index]);
            assert_eq!(left.current, right.current);
            assert_eq!(left.action, right.action);
            assert!(matches!(left.action.id, 5 | 6));
            assert_ne!(left.next, right.next);
        }
        Ok(())
    }

    #[test]
    fn v6_learning_history_rows_keep_context_under_the_row_augmentation() -> Result<()> {
        let batch =
            compose_mixed_stream_batch(&v6_config(128, 0x4348), 1.0, 2, V5DataSplit::Train)?;
        let mut chained = 0usize;
        let mut with_context = 0usize;
        for sample in batch
            .samples()
            .iter()
            .filter(|sample| sample.provenance.stream == MixedStreamKind::LearningHistories)
        {
            let rect = sample.provenance.content_rect;
            let empty = sample.provenance.augmentation.color_permutation[0];
            for context in &sample.transition.context {
                assert!(padding_is(&context.current, rect, empty));
                assert!(padding_is(&context.next, rect, empty));
                if context.action.id == 6 {
                    assert!(rect.contains(context.action.x.unwrap(), context.action.y.unwrap()));
                }
            }
            if let Some(last) = sample.transition.context.last() {
                with_context += 1;
                // A level boundary follows every realized operator row; every
                // other row chains onto the latest context transition.
                if matches!(last.action.id, 5 | 6) {
                    continue;
                }
                assert_eq!(last.next, sample.transition.current);
                chained += 1;
            }
        }
        assert!(with_context > 0 && chained > 0);
        Ok(())
    }

    #[test]
    fn v6_mixed_stream_content_digest_snapshot() -> Result<()> {
        use crate::p2::train::{training_content_batch_digest, training_content_hash_append};
        let config = v6_config(32, 0xDA7A_0006);
        let total_steps = 3usize;
        let chain = (0..total_steps as u64).try_fold([0; 32], |chain, batch_index| {
            let batch = compose_mixed_stream_batch(
                &config,
                batch_index as f32 / total_steps as f32,
                batch_index,
                V5DataSplit::Train,
            )?;
            let digest = training_content_batch_digest(batch.transitions(), batch.content_masks())?;
            Ok::<_, anyhow::Error>(training_content_hash_append(chain, digest))
        })?;
        // Snapshot taken at 0a12f79a (before the `SyntheticShards` stream): a
        // zero-weight shard stream must leave every v6 batch byte-identical.
        assert_eq!(
            chain,
            [
                142, 133, 119, 153, 224, 225, 166, 65, 172, 117, 91, 59, 138, 253, 119, 73, 152,
                22, 85, 227, 234, 181, 135, 93, 85, 156, 221, 168, 232, 214, 19, 93,
            ]
        );
        Ok(())
    }

    #[test]
    fn provenance_v6_fields_round_trip_and_legacy_json_is_unchanged() -> Result<()> {
        let legacy = TransitionProvenance::full_frame(1, 2, Split::Train, "legacy");
        let json = serde_json::to_string(&legacy)?;
        for key in [
            "rule_id",
            "level_index",
            "available_actions",
            "context_len",
            "background_color",
            "empty_color",
        ] {
            assert!(!json.contains(key), "{key} leaked into legacy JSON");
        }
        assert_eq!(serde_json::from_str::<TransitionProvenance>(&json)?, legacy);
        let mut v6 = legacy.clone();
        v6.rule_id = 0xABCD;
        v6.level_index = 3;
        v6.available_actions = ALL_ACTIONS_AVAILABLE;
        v6.context_len = 9;
        v6.background_color = 13;
        let round_trip: TransitionProvenance = serde_json::from_str(&serde_json::to_string(&v6)?)?;
        assert_eq!(round_trip, v6);
        let operator = EpisodeOperator {
            family: OperatorFamily::Paint,
            agent_color: 2,
            primary_color: 7,
            secondary_color: 8,
            empty_color: 11,
        };
        let round_trip: EpisodeOperator = serde_json::from_str(&serde_json::to_string(&operator)?)?;
        assert_eq!(round_trip, operator);
        assert_ne!(
            hidden_rule_id(operator, Some(0)),
            hidden_rule_id(
                EpisodeOperator {
                    family: OperatorFamily::Toggle,
                    ..operator
                },
                Some(0)
            )
        );
        assert_ne!(
            hidden_rule_id(operator, Some(0)),
            hidden_rule_id(operator, Some(1))
        );
        Ok(())
    }

    /// Snapshot of the v6 training stream (`adaptation_v6_stream_schedule`,
    /// `data_contract_v6`) taken before `LearningHistoryConfig` was introduced.
    /// The training composer must keep rendering byte-identical rows under
    /// the default (training) history configuration.
    #[test]
    fn v6_training_stream_content_digest_snapshot() -> Result<()> {
        use crate::p2::train::{training_content_batch_digest, training_content_hash_append};
        let config = v6_config(24, 0xDA7A_0106);
        let total_steps = 3usize;
        let chain = (0..total_steps as u64).try_fold([0; 32], |chain, batch_index| {
            let batch = compose_mixed_stream_batch(
                &config,
                batch_index as f32 / total_steps as f32,
                batch_index,
                V5DataSplit::Train,
            )?;
            let digest = training_content_batch_digest(batch.transitions(), batch.content_masks())?;
            Ok::<_, anyhow::Error>(training_content_hash_append(chain, digest))
        })?;
        assert_eq!(
            chain,
            [
                141, 213, 94, 18, 49, 180, 221, 31, 182, 38, 138, 227, 117, 66, 81, 200, 178, 40,
                97, 243, 1, 186, 143, 232, 64, 65, 16, 6, 65, 217, 251, 135,
            ]
        );
        Ok(())
    }

    /// The explicit training shape is the default generator, row for row.
    #[test]
    fn learning_history_training_shape_reproduces_the_default_generator() -> Result<()> {
        let families = OperatorFamilySplit::default();
        let shape = LearningHistoryConfig::training();
        assert!(shape.is_training());
        assert_eq!(shape, LearningHistoryConfig::default());
        assert_eq!(shape.transitions_per_level(), 7);
        assert_eq!(shape.min_chronological_transitions(), 14);
        for meta_episode_id in 0..6u64 {
            let default_pair = generate_learning_history_pair(
                0x5348_4150,
                meta_episode_id,
                V5DataSplit::UnseenSeed7x7,
                &families,
            )?;
            let explicit = generate_learning_history_pair_with(
                0x5348_4150,
                meta_episode_id,
                V5DataSplit::UnseenSeed7x7,
                &families,
                &shape,
            )?;
            assert_eq!(default_pair, explicit);
        }
        Ok(())
    }

    /// The extended evaluation shape yields >= 36 chronological transitions
    /// (in fact >= 75) with the production level structure: 24 movement rows
    /// then one realized operator row per level, and the twins still diverge
    /// only on operator rows.
    #[test]
    fn extended_learning_history_shape_yields_long_levels() -> Result<()> {
        let families = OperatorFamilySplit::default();
        let shape = LearningHistoryConfig::extended_evaluation();
        shape.validate()?;
        assert!(!shape.is_training());
        assert_eq!(shape.transitions_per_level(), 25);
        assert_eq!(shape.min_chronological_transitions(), 75);
        assert!(shape.min_chronological_transitions() >= 36);
        let mut seen_levels = BTreeSet::new();
        for meta_episode_id in 0..6u64 {
            let (primary, twin) = generate_learning_history_pair_with(
                0x4558_5445,
                meta_episode_id,
                V5DataSplit::UnseenSeed7x7,
                &families,
                &shape,
            )?;
            seen_levels.insert(primary.levels);
            assert!(LEARNING_HISTORY_EVAL_LEVELS.contains(&primary.levels));
            assert_eq!(
                primary.chronological.len(),
                primary.levels * shape.transitions_per_level()
            );
            assert!(primary.chronological.len() >= 36);
            assert_eq!(
                primary.rows.len(),
                primary.levels * (shape.steps_per_level + LEARNING_HISTORY_OPERATOR_ACTIONS)
            );
            for (position, &index) in primary.chronological.iter().enumerate() {
                let row = &primary.rows[index];
                row.provenance.validate()?;
                let is_operator = matches!(row.action.id, 5 | 6);
                assert_eq!(
                    is_operator,
                    position % shape.transitions_per_level() == shape.steps_per_level,
                    "position {position}"
                );
                assert_eq!(
                    usize::from(row.provenance.level_index),
                    position / shape.transitions_per_level()
                );
                assert!(row.context.len() <= CONTEXT_WINDOW_MAX.min(position));
                // Movement never runs the level's action budget out.
                assert_eq!(row.exhausted, Some(false), "position {position}");
            }
            let divergence = TwinPairDivergence::of(&primary, &twin)?;
            assert!(!divergence.single_frame_rule_identifiable);
            assert_eq!(divergence.state_differing_positions, 0);
            assert_eq!(
                twin_first_divergence(&primary, &twin)?.map(|index| primary
                    .chronological
                    .iter()
                    .position(|&chronological| chronological == index)
                    .expect("first divergence is a realized row")),
                divergence.first_divergence
            );
            for position in &divergence.outcome_changing_positions {
                assert_eq!(
                    position % shape.transitions_per_level(),
                    shape.steps_per_level,
                    "twins may only diverge on operator rows"
                );
            }
            assert!(
                divergence.first_divergence.is_some(),
                "twin family is chosen to diverge"
            );
        }
        assert!(seen_levels.len() >= 2, "{seen_levels:?}");
        // The shape is a real parameter: the same seed under the training
        // shape is a different (shorter) history.
        let (short, _) =
            generate_learning_history_pair(0x4558_5445, 0, V5DataSplit::UnseenSeed7x7, &families)?;
        assert!(short.chronological.len() < 36);
        Ok(())
    }

    /// Augmented twins share one augmentation: byte-identical current frames
    /// and actions at every chronological position, next frames identical
    /// except on the outcome-changing rows.
    #[test]
    fn augmented_twin_pair_shares_one_augmentation() -> Result<()> {
        let config = v6_config(24, 0x5457_494E);
        for shape in [
            LearningHistoryConfig::training(),
            LearningHistoryConfig::extended_evaluation(),
        ] {
            let pair = augmented_learning_history_pair_with(
                &config,
                V5DataSplit::UnseenSeed7x7,
                3,
                &shape,
            )?;
            let single =
                augmented_learning_history_with(&config, V5DataSplit::UnseenSeed7x7, 3, &shape)?;
            assert_eq!(pair.primary.chronological, single.chronological);
            assert_eq!(pair.primary.chronological, pair.twin.chronological);
            assert_eq!(pair.primary.rows.len(), pair.twin.rows.len());
            assert_eq!(pair.divergence.state_differing_positions, 0);
            assert!(!pair.divergence.single_frame_rule_identifiable);
            let outcome_changing = pair
                .divergence
                .outcome_changing_positions
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            assert!(!outcome_changing.is_empty());
            for (position, (left, right)) in pair
                .primary
                .chronological
                .iter()
                .map(|&index| (&pair.primary.rows[index], &pair.twin.rows[index]))
                .enumerate()
            {
                assert_eq!(left.provenance.augmentation, right.provenance.augmentation);
                assert_eq!(left.provenance.content_rect, right.provenance.content_rect);
                assert_eq!(
                    left.transition.current, right.transition.current,
                    "{position}"
                );
                assert_eq!(
                    left.transition.action, right.transition.action,
                    "{position}"
                );
                assert_eq!(
                    left.transition.next != right.transition.next,
                    outcome_changing.contains(&position),
                    "{position}"
                );
                assert_eq!(
                    single.rows[pair.primary.chronological[position]].transition,
                    left.transition
                );
            }
        }
        Ok(())
    }
}
