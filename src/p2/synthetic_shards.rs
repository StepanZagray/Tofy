//! ADR 0005 §2.6 `SyntheticShards` stream loader.
//!
//! Reads a shard directory written by `python/tofy_arc3/synth` (`manifest.json`
//! plus `shard-*.safetensors`), validates the exact tensor contract and the
//! manifest attestation, and converts rows into [`TransitionSample`]s carrying
//! a Context Window (ADR 0005 §1.5) so the mixed composer can augment them
//! exactly like Learning Histories.
//!
//! This module is training-reachable: it must never import the live driver,
//! bridge, or recording-import modules (see the guard test in the live
//! module). Every check here fails closed: a shard directory that deviates
//! from the contract, lacks the public-game attestation, or names a public
//! game id is rejected as a whole.
//!
//! Known gaps, by design: shards carry no hidden-goal labels, so every row has
//! all-zero goal features and `goal_satisfied`/`goal_failed`/`exhausted` are
//! `None` (`level_completed`/`game_over` are validated but not consumed as
//! labels). RESET rows (action id 0) are dropped: `ArcAction` has no RESET.

use super::data::{
    augment_v5_unit, mixed_stream_episode_id, sampled_context_len, seeded_v5_rng, ArcAction,
    ArcFrame, ContextTransition, EpisodeOperator, GoalFeatures, MixedStreamConfig, MixedStreamKind,
    OperatorFamily, TransitionProvenance, TransitionSample, V5DataSplit, V5Sample,
    CONTEXT_WINDOW_MAX, FRAME_SIDE,
};
use crate::domain::Split;
use anyhow::{anyhow, bail, ensure, Context, Result};
use candle_core::safetensors::BufferedSafetensors;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// `manifest.source` value written by `tofy_arc3.synth.shard`.
pub const SYNTHETIC_SHARD_SOURCE: &str = "tofy_synth_arcengine";
/// `source_kind` and `family` of every row loaded from a shard.
pub const SYNTHETIC_SHARD_SOURCE_KIND: &str = "synthetic_shard";
/// Prefix of every generated game id (`tofy_arc3.synth.public_games`).
pub const GENERATED_GAME_ID_PREFIX: &str = "tsyn";
/// `actions[:, 1..3]` value meaning "no coordinate".
pub const SHARD_ACTION_NONE: u8 = 255;
/// Contiguous chronological rows of one episode per augmentation unit.
pub const SYNTHETIC_SHARD_UNIT_ROWS: usize = 8;

/// The 25 public ARC-AGI-3 game ids. Copied from
/// `python/tofy_arc3/synth/public_games.py`; the loader fails closed when any
/// shard game id intersects this list.
pub const PUBLIC_GAME_IDS: [&str; 25] = [
    "ar25-0c556536",
    "bp35-0a0ad940",
    "cd82-fb555c5d",
    "cn04-2fe56bfb",
    "dc22-fdcac232",
    "ft09-0d8bbf25",
    "g50t-5849a774",
    "ka59-38d34dbb",
    "lf52-271a04aa",
    "lp85-305b61c3",
    "ls20-9607627b",
    "m0r0-492f87ba",
    "r11l-495a7899",
    "re86-8af5384d",
    "s5i5-18d95033",
    "sb26-7fbdac44",
    "sc25-635fd71a",
    "sk48-d8078629",
    "sp80-589a99af",
    "su15-1944f8ab",
    "tn36-ef4dde99",
    "tr87-cd924810",
    "tu93-0768757b",
    "vc33-5430563c",
    "wa30-ee6fef47",
];

/// Exact ADR 0005 §2.6 tensor contract: name, safetensors dtype, per-row shape.
pub const SHARD_TENSORS: [(&str, &str, &[usize]); 11] = [
    ("frames", "U8", &[FRAME_SIDE, FRAME_SIDE]),
    ("next_frames", "U8", &[FRAME_SIDE, FRAME_SIDE]),
    ("actions", "U8", &[3]),
    ("available_actions", "U8", &[]),
    ("episode", "U32", &[]),
    ("level", "U16", &[]),
    ("transition_index", "U32", &[]),
    ("rule_id_lo", "U32", &[]),
    ("rule_id_hi", "U32", &[]),
    ("level_completed", "U8", &[]),
    ("game_over", "U8", &[]),
];

const CONTEXT_LANE: u64 = 0x5348_4152_4443_5458;
const UNIT_LANE: u64 = 0x5348_4152_4455_4E49;

/// Sidecar `manifest.json`. Unknown keys are ignored; the listed keys are the
/// ones the loader validates or consumes.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardManifest {
    pub source: String,
    #[serde(default)]
    pub schema: Option<String>,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub game_id_prefix: Option<String>,
    #[serde(default)]
    pub game_ids: Vec<String>,
    #[serde(default)]
    pub public_game_ids_excluded: Vec<String>,
    #[serde(default)]
    pub public_game_ids_intersection: Vec<String>,
    #[serde(default)]
    pub shards: Vec<ShardEntry>,
    #[serde(default)]
    pub episodes: Vec<EpisodeEntry>,
    #[serde(default)]
    pub total_rows: Option<usize>,
    #[serde(default)]
    pub generator: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardEntry {
    pub file: String,
    #[serde(default)]
    pub rows: Option<usize>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub episodes: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EpisodeEntry {
    pub episode: u32,
    pub game_id: String,
    #[serde(default)]
    pub rule_id: Option<u64>,
    #[serde(default)]
    pub available_actions: Option<u8>,
    /// Letterbox/background colour of the episode's frames. Not written by
    /// the current generator; when absent the loader uses the most frequent
    /// border colour of each frame.
    #[serde(default)]
    pub background: Option<u8>,
    #[serde(default)]
    pub twin_of: Option<String>,
}

impl ShardManifest {
    pub fn read(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading shard manifest {}", path.display()))?;
        let manifest: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing shard manifest {}", path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Fail closed on every attestation the ADR requires.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.source == SYNTHETIC_SHARD_SOURCE,
            "shard manifest source {:?} != {SYNTHETIC_SHARD_SOURCE:?}",
            self.source
        );
        if let Some(prefix) = &self.game_id_prefix {
            ensure!(
                prefix == GENERATED_GAME_ID_PREFIX,
                "shard manifest game_id_prefix {prefix:?} != {GENERATED_GAME_ID_PREFIX:?}"
            );
        }
        let attested: BTreeSet<&str> = self
            .public_game_ids_excluded
            .iter()
            .map(String::as_str)
            .collect();
        let missing: Vec<&str> = PUBLIC_GAME_IDS
            .iter()
            .copied()
            .filter(|id| !attested.contains(id))
            .collect();
        ensure!(
            missing.is_empty(),
            "shard manifest public_game_ids_excluded attestation is missing {missing:?}"
        );
        ensure!(
            self.public_game_ids_intersection.is_empty(),
            "shard manifest reports a public game id intersection {:?}",
            self.public_game_ids_intersection
        );
        ensure!(
            !self.episodes.is_empty(),
            "shard manifest lists no episodes"
        );
        ensure!(!self.shards.is_empty(), "shard manifest lists no shards");
        let listed: BTreeSet<&str> = self.game_ids.iter().map(String::as_str).collect();
        let mut seen = BTreeSet::new();
        for entry in &self.episodes {
            ensure!(
                seen.insert(entry.episode),
                "shard manifest lists episode {} twice",
                entry.episode
            );
            ensure!(
                self.game_ids.is_empty() || listed.contains(entry.game_id.as_str()),
                "episode {} game id {:?} is not in manifest game_ids",
                entry.episode,
                entry.game_id
            );
            if let Some(background) = entry.background {
                ensure!(
                    background <= 15,
                    "episode background colour outside palette"
                );
            }
        }
        reject_public_game_ids(
            self.game_ids
                .iter()
                .chain(self.episodes.iter().map(|e| &e.game_id)),
        )
    }

    fn episode(&self, episode: u32) -> Option<&EpisodeEntry> {
        self.episodes.iter().find(|entry| entry.episode == episode)
    }
}

/// Reject any generated id that is public or lacks the generated prefix.
pub fn reject_public_game_ids<'a>(game_ids: impl Iterator<Item = &'a String>) -> Result<()> {
    let public: BTreeSet<&str> = PUBLIC_GAME_IDS.iter().copied().collect();
    let mut clash = BTreeSet::new();
    for id in game_ids {
        if public.contains(id.as_str()) {
            clash.insert(id.clone());
            continue;
        }
        ensure!(
            id.strip_prefix(GENERATED_GAME_ID_PREFIX)
                .is_some_and(|rest| rest.starts_with('-')),
            "generated game id {id:?} lacks prefix {GENERATED_GAME_ID_PREFIX:?}"
        );
    }
    ensure!(
        clash.is_empty(),
        "shard game ids collide with public ARC-AGI-3 games: {clash:?}"
    );
    Ok(())
}

/// One tensor as stored in a shard file, before any decoding.
#[derive(Clone, Debug)]
pub struct RawTensor<'a> {
    pub name: String,
    /// safetensors dtype name (`U8`, `U16`, `U32`, ...).
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

/// Validate names, dtypes and shapes against [`SHARD_TENSORS`]; return `N`.
pub fn validate_tensor_contract(tensors: &[RawTensor<'_>]) -> Result<usize> {
    let expected: BTreeSet<&str> = SHARD_TENSORS.iter().map(|(name, _, _)| *name).collect();
    let found: BTreeSet<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
    ensure!(
        found == expected,
        "shard tensor names {found:?} != contract {expected:?}"
    );
    let n = tensors
        .iter()
        .find(|t| t.name == "frames")
        .and_then(|t| t.shape.first().copied())
        .ok_or_else(|| anyhow!("frames tensor has no leading dimension"))?;
    for (name, dtype, tail) in SHARD_TENSORS {
        let tensor = tensors
            .iter()
            .find(|t| t.name == name)
            .expect("name set matched");
        ensure!(
            tensor.dtype == dtype,
            "shard tensor {name}: dtype {} != {dtype}",
            tensor.dtype
        );
        let mut shape = vec![n];
        shape.extend_from_slice(tail);
        ensure!(
            tensor.shape == shape,
            "shard tensor {name}: shape {:?} != {shape:?}",
            tensor.shape
        );
        let width = match dtype {
            "U8" => 1,
            "U16" => 2,
            "U32" => 4,
            other => bail!("contract dtype {other} has no decoder"),
        };
        ensure!(
            tensor.data.len() == shape.iter().product::<usize>() * width,
            "shard tensor {name}: byte length {} does not match its shape",
            tensor.data.len()
        );
    }
    ensure!(n > 0, "shard has no rows");
    Ok(n)
}

fn decode_u16(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn decode_u32(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Decoded columns of one shard file.
#[derive(Clone, Debug)]
pub struct ShardColumns {
    pub rows: usize,
    pub frames: Vec<u8>,
    pub next_frames: Vec<u8>,
    pub actions: Vec<u8>,
    pub available_actions: Vec<u8>,
    pub episode: Vec<u32>,
    pub level: Vec<u16>,
    pub transition_index: Vec<u32>,
    pub rule_id_lo: Vec<u32>,
    pub rule_id_hi: Vec<u32>,
    pub level_completed: Vec<u8>,
    pub game_over: Vec<u8>,
}

impl ShardColumns {
    /// Parse and validate one `.safetensors` file against the contract.
    pub fn read(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("reading shard {}", path.display()))?;
        Self::from_bytes(bytes).with_context(|| format!("shard {}", path.display()))
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let file = BufferedSafetensors::new(bytes).context("parsing safetensors header")?;
        let views = file.tensors();
        let tensors: Vec<RawTensor<'_>> = views
            .iter()
            .map(|(name, view)| RawTensor {
                name: name.clone(),
                dtype: format!("{:?}", view.dtype()),
                shape: view.shape().to_vec(),
                data: view.data(),
            })
            .collect();
        let rows = validate_tensor_contract(&tensors)?;
        let column = |name: &str| -> &[u8] {
            tensors
                .iter()
                .find(|t| t.name == name)
                .expect("contract validated")
                .data
        };
        let columns = Self {
            rows,
            frames: column("frames").to_vec(),
            next_frames: column("next_frames").to_vec(),
            actions: column("actions").to_vec(),
            available_actions: column("available_actions").to_vec(),
            episode: decode_u32(column("episode")),
            level: decode_u16(column("level")),
            transition_index: decode_u32(column("transition_index")),
            rule_id_lo: decode_u32(column("rule_id_lo")),
            rule_id_hi: decode_u32(column("rule_id_hi")),
            level_completed: column("level_completed").to_vec(),
            game_over: column("game_over").to_vec(),
        };
        ensure!(
            columns
                .frames
                .iter()
                .chain(&columns.next_frames)
                .all(|&pixel| pixel <= 15),
            "shard frame colour index > 15"
        );
        ensure!(
            columns
                .level_completed
                .iter()
                .chain(&columns.game_over)
                .all(|&flag| flag <= 1),
            "shard level_completed/game_over flags must be 0 or 1"
        );
        Ok(columns)
    }

    fn rule_id(&self, row: usize) -> u64 {
        u64::from(self.rule_id_lo[row]) | (u64::from(self.rule_id_hi[row]) << 32)
    }

    fn frame(&self, pixels: &[u8], row: usize) -> Result<ArcFrame> {
        let side = FRAME_SIDE * FRAME_SIDE;
        ArcFrame::new(
            FRAME_SIDE as u16,
            FRAME_SIDE as u16,
            pixels[row * side..(row + 1) * side].to_vec(),
        )
    }

    /// `None` for RESET rows (action id 0).
    fn action(&self, row: usize) -> Result<Option<ArcAction>> {
        let [id, x, y] = [
            self.actions[row * 3],
            self.actions[row * 3 + 1],
            self.actions[row * 3 + 2],
        ];
        ensure!(id <= 7, "row {row}: action id {id} not in 0..=7");
        let coordinate = |value: u8| (value != SHARD_ACTION_NONE).then_some(value);
        if id == 6 {
            ensure!(
                x != SHARD_ACTION_NONE && y != SHARD_ACTION_NONE,
                "row {row}: ACTION6 without coordinates"
            );
        } else {
            ensure!(
                x == SHARD_ACTION_NONE && y == SHARD_ACTION_NONE,
                "row {row}: coordinates on a non-ACTION6 row"
            );
        }
        if id == 0 {
            return Ok(None);
        }
        ArcAction::new(id, coordinate(x), coordinate(y))
            .with_context(|| format!("row {row}"))
            .map(Some)
    }
}

/// Most frequent colour on the 64x64 frame border (lowest index on ties): the
/// ARCEngine letterbox colour when the manifest records none.
pub fn border_mode_color(frame: &ArcFrame) -> u8 {
    let mut counts = [0usize; 16];
    let side = FRAME_SIDE;
    for index in 0..side {
        for (x, y) in [(index, 0), (index, side - 1), (0, index), (side - 1, index)] {
            counts[usize::from(frame.pixels[y * side + x])] += 1;
        }
    }
    let mut best = 0u8;
    for (color, &count) in counts.iter().enumerate() {
        if count > counts[usize::from(best)] {
            best = color as u8;
        }
    }
    best
}

/// Placeholder operator handed to the shared v5/v6 augmentation so it renders
/// padding and `background_color` with the shard row's letterbox colour. Shards
/// carry no operator family; the sidecar operator of shard rows is not
/// meaningful and v6 conditioning is UNKNOWN regardless (ADR 0005 §1.4).
fn placeholder_operator(background: u8) -> EpisodeOperator {
    EpisodeOperator {
        family: OperatorFamily::Toggle,
        agent_color: 0,
        primary_color: 0,
        secondary_color: 0,
        empty_color: background,
    }
}

/// Row census of one loaded shard directory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SyntheticShardStats {
    pub shards: usize,
    pub episodes: usize,
    /// Rows kept as training rows.
    pub rows: usize,
    /// RESET rows dropped (`ArcAction` has no RESET).
    pub reset_rows_skipped: usize,
    pub rows_with_context: usize,
}

/// All rows of a shard directory, grouped by episode in chronological order,
/// each carrying its sampled Context Window. Raw (pre-augmentation) rows:
/// whole-frame content at the origin, unpermuted colours, no operator.
#[derive(Debug)]
pub struct SyntheticShardPool {
    dir: PathBuf,
    manifest: ShardManifest,
    rows: Vec<TransitionSample>,
    episodes: Vec<Range<usize>>,
    stats: SyntheticShardStats,
}

struct RawRow {
    episode: u32,
    transition_index: u32,
    level: u16,
    rule_id: u64,
    available_actions: u8,
    action: Option<ArcAction>,
    current: ArcFrame,
    next: ArcFrame,
}

impl SyntheticShardPool {
    pub fn load(dir: &Path) -> Result<Self> {
        let manifest = ShardManifest::read(&dir.join("manifest.json"))?;
        let mut raw = Vec::new();
        for entry in &manifest.shards {
            let path = dir.join(&entry.file);
            ensure!(
                entry.file.ends_with(".safetensors") && !entry.file.contains('/'),
                "shard entry {:?} is not a plain .safetensors file name",
                entry.file
            );
            let bytes = std::fs::read(&path)
                .with_context(|| format!("reading shard {}", path.display()))?;
            if let Some(expected) = &entry.sha256 {
                let digest = format!("{:x}", Sha256::digest(&bytes));
                ensure!(
                    digest.eq_ignore_ascii_case(expected),
                    "shard {} sha256 {digest} != manifest {expected}",
                    path.display()
                );
            }
            let columns = ShardColumns::from_bytes(bytes)
                .with_context(|| format!("shard {}", path.display()))?;
            if let Some(rows) = entry.rows {
                ensure!(
                    rows == columns.rows,
                    "shard {} has {} rows, manifest says {rows}",
                    path.display(),
                    columns.rows
                );
            }
            for row in 0..columns.rows {
                let episode = columns.episode[row];
                let entry = manifest.episode(episode).ok_or_else(|| {
                    anyhow!(
                        "shard {} row {row}: episode {episode} is not in the manifest",
                        path.display()
                    )
                })?;
                let rule_id = columns.rule_id(row);
                ensure!(
                    rule_id != 0,
                    "shard {} row {row}: rule_id is zero",
                    path.display()
                );
                if let Some(expected) = entry.rule_id {
                    ensure!(
                        rule_id == expected,
                        "shard {} row {row}: rule_id {rule_id} != manifest {expected}",
                        path.display()
                    );
                }
                if let Some(expected) = entry.available_actions {
                    ensure!(
                        columns.available_actions[row] == expected,
                        "shard {} row {row}: available_actions != manifest",
                        path.display()
                    );
                }
                raw.push(RawRow {
                    episode,
                    transition_index: columns.transition_index[row],
                    level: columns.level[row],
                    rule_id,
                    available_actions: columns.available_actions[row],
                    action: columns.action(row)?,
                    current: columns.frame(&columns.frames, row)?,
                    next: columns.frame(&columns.next_frames, row)?,
                });
            }
        }
        let mut stats = SyntheticShardStats {
            shards: manifest.shards.len(),
            ..SyntheticShardStats::default()
        };
        if let Some(total) = manifest.total_rows {
            ensure!(
                raw.len() == total,
                "shards hold {} rows, manifest total_rows is {total}",
                raw.len()
            );
        }
        let mut by_episode: BTreeMap<u32, Vec<RawRow>> = BTreeMap::new();
        for row in raw {
            by_episode.entry(row.episode).or_default().push(row);
        }
        let mut rows = Vec::new();
        let mut episodes = Vec::new();
        for (episode, mut episode_rows) in by_episode {
            episode_rows.sort_by_key(|row| row.transition_index);
            ensure!(
                episode_rows
                    .windows(2)
                    .all(|pair| pair[0].transition_index < pair[1].transition_index),
                "episode {episode} repeats a transition_index"
            );
            let entry = manifest.episode(episode).expect("checked per row");
            let start = rows.len();
            let mut timeline: Vec<ContextTransition> = Vec::new();
            for row in episode_rows {
                let Some(action) = row.action else {
                    stats.reset_rows_skipped += 1;
                    continue;
                };
                let background = entry
                    .background
                    .unwrap_or_else(|| border_mode_color(&row.current));
                let t = timeline.len();
                let mut rng = seeded_v5_rng(
                    manifest.seed,
                    u64::from(episode),
                    V5DataSplit::Train,
                    CONTEXT_LANE ^ u64::from(row.transition_index),
                );
                let k = sampled_context_len(&mut rng, t);
                ensure!(
                    k <= CONTEXT_WINDOW_MAX && k <= t,
                    "context draw out of range"
                );
                let context = timeline[t - k..t].to_vec();
                stats.rows_with_context += usize::from(k > 0);
                timeline.push(ContextTransition {
                    current: row.current.clone(),
                    action: action.clone(),
                    next: row.next.clone(),
                });
                rows.push(TransitionSample {
                    noop: Some(row.current == row.next),
                    current: row.current,
                    next: row.next,
                    action,
                    goal_features: GoalFeatures::zeros(),
                    goal_satisfied: None,
                    goal_failed: None,
                    exhausted: None,
                    split: Split::Train,
                    family: SYNTHETIC_SHARD_SOURCE_KIND.into(),
                    seed: manifest.seed,
                    episode_id: u64::from(episode),
                    transition_index: u64::from(row.transition_index),
                    provenance: TransitionProvenance {
                        content_width: FRAME_SIDE as u16,
                        content_height: FRAME_SIDE as u16,
                        content_x: 0,
                        content_y: 0,
                        source_kind: SYNTHETIC_SHARD_SOURCE_KIND.into(),
                        trajectory_id: format!(
                            "{SYNTHETIC_SHARD_SOURCE_KIND}/{}/{episode}",
                            entry.game_id
                        ),
                        operator: None,
                        rule_id: row.rule_id,
                        level_index: row.level,
                        available_actions: row.available_actions,
                        context_len: u8::try_from(k).expect("k <= 16"),
                        background_color: background,
                    },
                    oracle_latent: None,
                    context,
                });
            }
            if rows.len() > start {
                episodes.push(start..rows.len());
            }
        }
        ensure!(!rows.is_empty(), "shard directory holds no non-RESET rows");
        for row in &rows {
            row.provenance.validate()?;
        }
        stats.rows = rows.len();
        stats.episodes = episodes.len();
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
            rows,
            episodes,
            stats,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn manifest(&self) -> &ShardManifest {
        &self.manifest
    }

    pub fn stats(&self) -> SyntheticShardStats {
        self.stats
    }

    /// Every kept row; episodes are contiguous and chronological.
    pub fn rows(&self) -> &[TransitionSample] {
        &self.rows
    }

    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }

    /// Chronological rows of the `index`-th loaded episode.
    pub fn episode_rows(&self, index: usize) -> &[TransitionSample] {
        &self.rows[self.episodes[index].clone()]
    }
}

static POOLS: OnceLock<Mutex<BTreeMap<PathBuf, Arc<SyntheticShardPool>>>> = OnceLock::new();

/// Load a shard directory once per process; later calls share the pool.
pub fn cached_pool(dir: &Path) -> Result<Arc<SyntheticShardPool>> {
    let key = dir
        .canonicalize()
        .with_context(|| format!("shard directory {}", dir.display()))?;
    let pools = POOLS.get_or_init(Default::default);
    if let Some(pool) = pools.lock().expect("pool cache poisoned").get(&key) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(SyntheticShardPool::load(&key)?);
    Ok(Arc::clone(
        pools
            .lock()
            .expect("pool cache poisoned")
            .entry(key)
            .or_insert(pool),
    ))
}

/// One augmentation unit: up to `maximum_rows` contiguous chronological rows
/// of one deterministically chosen episode, augmented like a Learning
/// History unit (one D4 + full colour permutation shared by every row and its
/// Context Window).
fn compose_unit(
    pool: &SyntheticShardPool,
    config: &MixedStreamConfig,
    split: V5DataSplit,
    episode_id: u64,
    maximum_rows: usize,
) -> Result<Vec<V5Sample>> {
    let mut rng = seeded_v5_rng(config.seed, episode_id, split, UNIT_LANE);
    let episode = pool.episode_rows(rng.random_range(0..pool.episode_count()));
    let take = maximum_rows.min(episode.len()).max(1);
    let start = rng.random_range(0..=episode.len() - take);
    let rows: Vec<TransitionSample> = episode[start..start + take]
        .iter()
        .cloned()
        .map(|mut row| {
            row.provenance.operator = Some(placeholder_operator(row.provenance.background_color));
            row
        })
        .collect();
    let operator = placeholder_operator(rows[0].provenance.background_color);
    augment_v5_unit(
        rows,
        config,
        split,
        MixedStreamKind::SyntheticShards,
        operator,
        episode_id,
    )
}

/// Compose `count` shard rows for one mixed batch. Training population only.
pub(crate) fn compose_synthetic_shards_stream(
    pool: &SyntheticShardPool,
    config: &MixedStreamConfig,
    split: V5DataSplit,
    count: usize,
    batch_index: u64,
    first_unit_index: u64,
) -> Result<(Vec<V5Sample>, u64)> {
    ensure!(
        config.data_contract_v6,
        "the synthetic-shards stream requires data_contract_v6"
    );
    ensure!(
        split == V5DataSplit::Train,
        "the synthetic-shards stream is a training population; {split:?} is not shard-backed"
    );
    let mut samples = Vec::with_capacity(count);
    let mut next_unit_index = first_unit_index;
    while samples.len() < count {
        let wave_units = (count - samples.len()).div_ceil(SYNTHETIC_SHARD_UNIT_ROWS);
        let units = (0..wave_units)
            .into_par_iter()
            .map(|offset| {
                compose_unit(
                    pool,
                    config,
                    split,
                    mixed_stream_episode_id(
                        batch_index,
                        next_unit_index
                            .checked_add(offset as u64)
                            .context("mixed stream unit index overflow")?,
                    )?,
                    SYNTHETIC_SHARD_UNIT_ROWS,
                )
            })
            .collect::<Vec<_>>();
        for unit in units {
            if samples.len() == count {
                break;
            }
            let mut unit = unit?;
            ensure!(!unit.is_empty(), "synthetic-shard unit produced no rows");
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
    use crate::p2::data::{
        adaptation_v6_stream_schedule, compose_mixed_stream_batch, D4Transform,
        MixedStreamProportions,
    };
    use std::collections::HashMap;

    /// Minimal safetensors writer (header length, JSON header, data) so tests
    /// can emit U16 columns, which candle's own `save` cannot.
    fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                (*name).to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    struct Row {
        episode: u32,
        transition_index: u32,
        level: u16,
        action: [u8; 3],
    }

    /// Frame whose corner pixel encodes the row so contexts are traceable;
    /// the border is colour 7 (the letterbox), the interior colour 3.
    fn frame_pixels(episode: u32, transition_index: u32, next: bool) -> Vec<u8> {
        let mut pixels = vec![7u8; FRAME_SIDE * FRAME_SIDE];
        for y in 8..56 {
            for x in 8..56 {
                pixels[y * FRAME_SIDE + x] = 3;
            }
        }
        pixels[10 * FRAME_SIDE + 10] = (episode % 16) as u8;
        pixels[10 * FRAME_SIDE + 11] = (transition_index % 16) as u8;
        pixels[10 * FRAME_SIDE + 12] = ((transition_index / 16) % 16) as u8;
        pixels[10 * FRAME_SIDE + 13] = u8::from(next);
        pixels
    }

    fn columns(rows: &[Row]) -> Vec<(&'static str, &'static str, Vec<usize>, Vec<u8>)> {
        let n = rows.len();
        let u16s = |f: &dyn Fn(&Row) -> u16| rows.iter().flat_map(|r| f(r).to_le_bytes()).collect();
        let u32s = |f: &dyn Fn(&Row) -> u32| rows.iter().flat_map(|r| f(r).to_le_bytes()).collect();
        vec![
            (
                "frames",
                "U8",
                vec![n, 64, 64],
                rows.iter()
                    .flat_map(|r| frame_pixels(r.episode, r.transition_index, false))
                    .collect(),
            ),
            (
                "next_frames",
                "U8",
                vec![n, 64, 64],
                rows.iter()
                    .flat_map(|r| frame_pixels(r.episode, r.transition_index, true))
                    .collect(),
            ),
            (
                "actions",
                "U8",
                vec![n, 3],
                rows.iter().flat_map(|r| r.action).collect(),
            ),
            ("available_actions", "U8", vec![n], vec![127; n]),
            ("episode", "U32", vec![n], u32s(&|r| r.episode)),
            ("level", "U16", vec![n], u16s(&|r| r.level)),
            (
                "transition_index",
                "U32",
                vec![n],
                u32s(&|r| r.transition_index),
            ),
            ("rule_id_lo", "U32", vec![n], u32s(&|r| 0x1000 + r.episode)),
            ("rule_id_hi", "U32", vec![n], u32s(&|r| 0x2000 + r.episode)),
            ("level_completed", "U8", vec![n], vec![0; n]),
            ("game_over", "U8", vec![n], vec![0; n]),
        ]
    }

    fn tiny_rows() -> Vec<Row> {
        let mut rows = Vec::new();
        for episode in 0..2u32 {
            for t in 0..40u32 {
                let action = match t {
                    20 => [0, 255, 255],
                    t if t % 7 == 3 => [6, 5, 9],
                    t => [1 + (t % 4) as u8, 255, 255],
                };
                rows.push(Row {
                    episode,
                    transition_index: t,
                    level: (t / 20) as u16,
                    action,
                });
            }
        }
        // Shuffle the file order: the loader must sort chronologically.
        rows.reverse();
        rows
    }

    fn manifest_json(game_ids: &[&str], attest: bool) -> serde_json::Value {
        serde_json::json!({
            "source": SYNTHETIC_SHARD_SOURCE,
            "schema": "adr-0005-2.6",
            "seed": 11,
            "game_id_prefix": GENERATED_GAME_ID_PREFIX,
            "game_ids": game_ids,
            "public_game_ids_excluded": if attest { PUBLIC_GAME_IDS.to_vec() } else { vec![] },
            "public_game_ids_intersection": [],
            "shards": [{"file": "shard-00000.safetensors"}],
            "episodes": game_ids.iter().enumerate().map(|(i, id)| serde_json::json!({
                "episode": i, "game_id": id,
                "rule_id": (0x1000u64 + i as u64) | ((0x2000u64 + i as u64) << 32),
                "available_actions": 127,
            })).collect::<Vec<_>>(),
        })
    }

    fn write_dir(
        name: &str,
        tensors: &[(&str, &str, Vec<usize>, Vec<u8>)],
        manifest: &serde_json::Value,
    ) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("tofy-synth-shards-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_safetensors(&dir.join("shard-00000.safetensors"), tensors);
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_string_pretty(manifest).unwrap(),
        )
        .unwrap();
        dir
    }

    const GAMES: [&str; 2] = ["tsyn-00000001", "tsyn-00000002"];

    #[test]
    fn contract_rejects_wrong_dtype_shape_name_and_public_game_ids() {
        let rows = tiny_rows();
        let good = columns(&rows);
        assert!(
            SyntheticShardPool::load(&write_dir("good", &good, &manifest_json(&GAMES, true)))
                .is_ok()
        );

        let mut wrong_dtype = good.clone();
        let level = wrong_dtype.iter_mut().find(|t| t.0 == "level").unwrap();
        level.1 = "U32";
        level.3 = rows
            .iter()
            .flat_map(|r| u32::from(r.level).to_le_bytes())
            .collect();
        let err = SyntheticShardPool::load(&write_dir(
            "dtype",
            &wrong_dtype,
            &manifest_json(&GAMES, true),
        ))
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("level: dtype U32 != U16"),
            "{err:#}"
        );

        let mut wrong_shape = good.clone();
        let actions = wrong_shape.iter_mut().find(|t| t.0 == "actions").unwrap();
        actions.2 = vec![rows.len() * 3];
        let err = SyntheticShardPool::load(&write_dir(
            "shape",
            &wrong_shape,
            &manifest_json(&GAMES, true),
        ))
        .unwrap_err();
        assert!(format!("{err:#}").contains("actions: shape"), "{err:#}");

        let mut wrong_name = good.clone();
        wrong_name
            .iter_mut()
            .find(|t| t.0 == "game_over")
            .unwrap()
            .0 = "gameover";
        let err = SyntheticShardPool::load(&write_dir(
            "name",
            &wrong_name,
            &manifest_json(&GAMES, true),
        ))
        .unwrap_err();
        assert!(format!("{err:#}").contains("tensor names"), "{err:#}");

        let public = manifest_json(&["tsyn-00000001", "ls20-9607627b"], true);
        let err = SyntheticShardPool::load(&write_dir("public", &good, &public)).unwrap_err();
        assert!(
            format!("{err:#}").contains("collide with public"),
            "{err:#}"
        );

        let err =
            SyntheticShardPool::load(&write_dir("attest", &good, &manifest_json(&GAMES, false)))
                .unwrap_err();
        assert!(
            format!("{err:#}").contains("attestation is missing"),
            "{err:#}"
        );

        let err = SyntheticShardPool::load(&write_dir(
            "prefix",
            &good,
            &manifest_json(&["abcd-00000001", "tsyn-00000002"], true),
        ))
        .unwrap_err();
        assert!(format!("{err:#}").contains("lacks prefix"), "{err:#}");

        let mut wrong_source = manifest_json(&GAMES, true);
        wrong_source["source"] = "tofy_synth_other".into();
        assert!(SyntheticShardPool::load(&write_dir("source", &good, &wrong_source)).is_err());
    }

    fn row_marker(frame: &ArcFrame) -> (u8, u32, bool) {
        let at = |x: usize| frame.pixels[10 * FRAME_SIDE + x];
        (
            at(10),
            u32::from(at(11)) + 16 * u32::from(at(12)),
            at(13) == 1,
        )
    }

    #[test]
    fn contexts_are_chronological_bounded_same_episode_and_exclude_the_row() {
        let dir = write_dir(
            "context",
            &columns(&tiny_rows()),
            &manifest_json(&GAMES, true),
        );
        let pool = SyntheticShardPool::load(&dir).unwrap();
        let stats = pool.stats();
        assert_eq!(
            (stats.episodes, stats.rows, stats.reset_rows_skipped),
            (2, 78, 2)
        );
        let mut zero = 0usize;
        let mut full = 0usize;
        for episode in 0..pool.episode_count() {
            let rows = pool.episode_rows(episode);
            assert!(rows
                .windows(2)
                .all(|p| p[0].transition_index < p[1].transition_index));
            for (t, row) in rows.iter().enumerate() {
                let k = row.context.len();
                assert!(k <= CONTEXT_WINDOW_MAX && k <= t);
                assert_eq!(usize::from(row.provenance.context_len), k);
                zero += usize::from(k == 0);
                full += usize::from(k == CONTEXT_WINDOW_MAX);
                let (episode_marker, index_marker, _) = row_marker(&row.current);
                assert_eq!(
                    (u32::from(episode_marker), index_marker),
                    (row.episode_id as u32, row.transition_index as u32)
                );
                for (offset, context) in row.context.iter().enumerate() {
                    let earlier = &rows[t - k + offset];
                    assert_eq!(context.current, earlier.current);
                    assert_eq!(context.next, earlier.next);
                    assert_eq!(context.action, earlier.action);
                    let (ep, index, _) = row_marker(&context.current);
                    assert_eq!(u32::from(ep), row.episode_id as u32);
                    assert!(u64::from(index) < row.transition_index);
                }
                assert_ne!(row.action.id, 0);
                assert_eq!(row.provenance.background_color, 7);
                assert_eq!(row.provenance.source_kind, SYNTHETIC_SHARD_SOURCE_KIND);
                assert_eq!(row.noop, Some(false));
                assert_eq!(
                    row.provenance.rule_id,
                    (0x1000 + row.episode_id) | ((0x2000 + row.episode_id) << 32)
                );
                assert_eq!(row.goal_features, GoalFeatures::zeros());
            }
        }
        assert!(zero > 0 && full > 0, "K=0 seen {zero}, K=16 seen {full}");
    }

    fn shard_schedule(progress: f32) -> MixedStreamProportions {
        let base = adaptation_v6_stream_schedule(progress);
        MixedStreamProportions {
            synthetic_shards: 0.25,
            ..base
        }
        .normalized()
    }

    fn transformed(frame: &ArcFrame, d4: D4Transform, permutation: &[u8; 16]) -> Vec<u8> {
        let mut out = vec![0u8; FRAME_SIDE * FRAME_SIDE];
        for y in 0..FRAME_SIDE as u8 {
            for x in 0..FRAME_SIDE as u8 {
                let (tx, ty) = d4.transform_point(x, y, FRAME_SIDE as u8);
                out[usize::from(ty) * FRAME_SIDE + usize::from(tx)] = permutation
                    [usize::from(frame.pixels[usize::from(y) * FRAME_SIDE + usize::from(x)])];
            }
        }
        out
    }

    #[test]
    fn shard_stream_augments_row_and_context_consistently() -> Result<()> {
        let dir = write_dir(
            "stream",
            &columns(&tiny_rows()),
            &manifest_json(&GAMES, true),
        );
        let config = MixedStreamConfig {
            batch_size: 64,
            seed: 0x5348,
            schedule: shard_schedule,
            data_contract_v6: true,
            synthetic_shards_dir: Some(dir.clone()),
            ..MixedStreamConfig::default()
        };
        config.validate()?;
        let pool = cached_pool(&dir)?;
        let raw: HashMap<(u64, u64), &TransitionSample> = pool
            .rows()
            .iter()
            .map(|row| ((row.episode_id, row.transition_index), row))
            .collect();
        let batch = compose_mixed_stream_batch(&config, 0.5, 3, V5DataSplit::Train)?;
        let shard_rows: Vec<&V5Sample> = batch
            .samples()
            .iter()
            .filter(|sample| sample.provenance.stream == MixedStreamKind::SyntheticShards)
            .collect();
        assert_eq!(
            shard_rows.len(),
            batch.stream_counts()[&MixedStreamKind::SyntheticShards]
        );
        assert!(shard_rows.len() >= 12, "{}", shard_rows.len());
        let mut non_identity = 0usize;
        let mut with_context = 0usize;
        for sample in shard_rows {
            sample.validate()?;
            assert!(sample.provenance.contract_v6);
            assert_eq!(sample.provenance.content_rect.width, 64);
            let augmentation = &sample.provenance.augmentation;
            let permutation = &augmentation.color_permutation;
            let row = &sample.transition;
            let source = raw[&(row.episode_id, row.transition_index)];
            assert_eq!(row.provenance.source_kind, SYNTHETIC_SHARD_SOURCE_KIND);
            assert_eq!(row.provenance.operator, None);
            assert_eq!(row.provenance.rule_id, source.provenance.rule_id);
            assert_eq!(row.provenance.level_index, source.provenance.level_index);
            assert_eq!(row.provenance.available_actions, 127);
            assert_eq!(
                row.provenance.background_color,
                permutation[usize::from(source.provenance.background_color)]
            );
            assert_eq!(
                row.current.pixels.as_ref(),
                transformed(&source.current, augmentation.d4, permutation).as_slice()
            );
            assert_eq!(
                row.next.pixels.as_ref(),
                transformed(&source.next, augmentation.d4, permutation).as_slice()
            );
            assert_eq!(row.context.len(), source.context.len());
            assert_eq!(usize::from(row.provenance.context_len), row.context.len());
            for (context, raw_context) in row.context.iter().zip(&source.context) {
                assert_eq!(
                    context.current.pixels.as_ref(),
                    transformed(&raw_context.current, augmentation.d4, permutation).as_slice()
                );
                assert_eq!(
                    context.next.pixels.as_ref(),
                    transformed(&raw_context.next, augmentation.d4, permutation).as_slice()
                );
                if raw_context.action.id == 6 {
                    let (x, y) = augmentation.d4.transform_point(
                        raw_context.action.x.unwrap(),
                        raw_context.action.y.unwrap(),
                        64,
                    );
                    assert_eq!((context.action.x, context.action.y), (Some(x), Some(y)));
                }
            }
            non_identity +=
                usize::from(augmentation.d4 != D4Transform::Identity || permutation[0] != 0);
            with_context += usize::from(!row.context.is_empty());
        }
        assert!(non_identity > 0 && with_context > 0);
        // Determinism: the same batch index reproduces the same rows.
        let again = compose_mixed_stream_batch(&config, 0.5, 3, V5DataSplit::Train)?;
        assert_eq!(batch.samples(), again.samples());
        Ok(())
    }

    #[test]
    fn shard_schedule_requires_v6_and_a_directory() {
        let config = MixedStreamConfig {
            schedule: shard_schedule,
            data_contract_v6: true,
            ..MixedStreamConfig::default()
        };
        assert!(config.validate().is_err());
        assert!(MixedStreamConfig {
            data_contract_v6: false,
            synthetic_shards_dir: Some(PathBuf::from("/nonexistent")),
            ..config
        }
        .validate()
        .is_err());
        assert_eq!(adaptation_v6_stream_schedule(0.5).synthetic_shards, 0.0);
    }

    /// The committed sample shard (`runs/p2/synth-sample-20260903`). The
    /// binary is gitignored: regenerate through the repository venv when it
    /// is missing, and skip with a message when no venv is available.
    fn sample_shard_dir() -> Option<PathBuf> {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = repo.join("runs/p2/synth-sample-20260903");
        if dir.join("shard-00000.safetensors").exists() {
            return Some(dir);
        }
        let python = repo.join(".venv/bin/python");
        if !python.exists() {
            eprintln!(
                "skipping: {} is missing and {} has no venv to regenerate it",
                dir.display(),
                repo.display()
            );
            return None;
        }
        let status = std::process::Command::new(python)
            .args([
                "-m",
                "tofy_arc3.synth.generate",
                "--seed",
                "1",
                "--games",
                "20",
                "--out",
            ])
            .arg(&dir)
            .env("CUDA_VISIBLE_DEVICES", "")
            .current_dir(repo.join("python"))
            .status()
            .ok()?;
        status.success().then_some(dir)
    }

    #[test]
    fn sample_shard_loads_and_composes() -> Result<()> {
        let Some(dir) = sample_shard_dir() else {
            return Ok(());
        };
        let pool = cached_pool(&dir)?;
        let stats = pool.stats();
        assert_eq!(stats.episodes, 40);
        assert_eq!(stats.rows + stats.reset_rows_skipped, 10_000);
        assert!(stats.rows_with_context > stats.rows / 2);
        for row in pool.rows() {
            assert!(row.provenance.rule_id != 0 && row.action.id != 0);
            assert!(row.context.len() <= CONTEXT_WINDOW_MAX);
            assert!(row.context.iter().all(|c| c.action.id != 0));
        }
        let by_game = pool
            .manifest()
            .episodes
            .iter()
            .map(|e| (e.episode, e.game_id.clone()))
            .collect::<HashMap<_, _>>();
        for episode in 0..pool.episode_count() {
            let rows = pool.episode_rows(episode);
            let id = rows[0].episode_id as u32;
            assert!(rows.iter().all(|r| r
                .provenance
                .trajectory_id
                .ends_with(&format!("/{}/{id}", by_game[&id]))));
        }
        let config = MixedStreamConfig {
            batch_size: 64,
            seed: 0x5348_4152,
            schedule: shard_schedule,
            data_contract_v6: true,
            synthetic_shards_dir: Some(dir),
            ..MixedStreamConfig::default()
        };
        let batch = compose_mixed_stream_batch(&config, 0.0, 0, V5DataSplit::Train)?;
        assert!(batch.stream_counts()[&MixedStreamKind::SyntheticShards] > 0);
        for sample in batch
            .samples()
            .iter()
            .filter(|s| s.provenance.stream == MixedStreamKind::SyntheticShards)
        {
            sample.validate()?;
        }
        Ok(())
    }
}
