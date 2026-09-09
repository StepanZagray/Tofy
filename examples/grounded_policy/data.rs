//! Registered C12 populations. Only `Inputs` enters the learned computation;
//! identities, controls, labels and visible role locations are audit/scoring data.

use anyhow::{ensure, Context, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tofy::p2::looped_agent::task::{self, Inputs, Sample};
use tofy::p2::looped_agent::{ACTIONS, META_DIM, PATCH_COUNT, PATCH_PIXELS, TOKENS};

pub const SCHEMA: &str = "looped-grounded-policy-data-v1";
pub const FIT_MAPS: [usize; 16] = [0, 2, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 21, 23];
pub const HELDOUT_MAPS: [usize; 8] = [1, 3, 6, 11, 12, 17, 20, 22];
pub const TRAIN_SEED: u64 = 20260920;
pub const EVAL_SEED: u64 = 20260921;
pub const TRAIN_BASE: u64 = 0x47524f554e445452;
pub const EVAL_BASE: u64 = 0x47524f554e444556;
pub const UPDATES: usize = 1150;
pub const TRAIN_GROUPS: usize = UPDATES * 4;
pub const EVAL_GROUPS: usize = 64;
const PIXELS: usize = PATCH_COUNT * PATCH_PIXELS;
const SUPPORT_PIXELS: usize = (TOKENS - PATCH_COUNT) * PATCH_PIXELS;
const HISTORY_HASHES: [&str; 4] = [
    "09559eb4dd1b933de724da81aef3bb80547e010a00e38e39de31c9f72fab1c6d",
    "bd72fd9707759dd19074a22b9d62899419b28711722b3617c21cc277a6a25c0d",
    "09a7aa423b17f32bbae14ae9178a15d5fff8428936afdf23ebc4f09878a6ddca",
    "61ae87a267cc545b05884dc412160c0313a0c2bc1015cfa2323bde256f91eaeb",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Cohort {
    Training,
    Seen,
    Familiar,
    Heldout,
}

impl Cohort {
    pub fn name(self) -> &'static str {
        match self {
            Self::Training => "training",
            Self::Seen => "seen",
            Self::Familiar => "familiar",
            Self::Heldout => "heldout",
        }
    }

    pub fn maps(self) -> &'static [usize] {
        if self == Self::Heldout {
            &HELDOUT_MAPS
        } else {
            &FIT_MAPS
        }
    }

    pub fn group_count(self) -> usize {
        if self == Self::Training {
            TRAIN_GROUPS
        } else {
            EVAL_GROUPS
        }
    }

    pub fn row_count(self) -> usize {
        self.group_count() * self.maps().len()
    }

    /// Data seed is used directly: there is no legacy TRAIN_TAG xor in C12.
    pub fn address(self, group_index: usize) -> Result<(u64, u64, Option<usize>)> {
        ensure!(
            group_index < self.group_count(),
            "cohort group out of range"
        );
        match self {
            Self::Training => Ok((
                TRAIN_SEED,
                TRAIN_BASE + group_index as u64,
                Some(group_index),
            )),
            Self::Seen => {
                let original = group_index * (TRAIN_GROUPS - 1) / (EVAL_GROUPS - 1);
                Ok((TRAIN_SEED, TRAIN_BASE + original as u64, Some(original)))
            }
            Self::Familiar | Self::Heldout => Ok((EVAL_SEED, EVAL_BASE + group_index as u64, None)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Row {
    pub inputs: Inputs,
    pub correct_action: usize,
    pub agent_patch: usize,
    pub goal_patch: usize,
    pub audit_json: Value,
}

impl Row {
    /// Registered intervention: only support pixels change; ALL metadata stays.
    pub fn cleared(&self) -> Self {
        let mut row = self.clone();
        row.inputs.patches[..SUPPORT_PIXELS].fill(0);
        row.audit_json["condition"] = json!("cleared");
        row.audit_json["support_cleared"] = json!(true);
        row.audit_json["input_sha256"] = row.audit_json["cleared_input_sha256"].clone();
        row
    }
}

pub fn input_hash(input: &Inputs) -> String {
    let mut h = Sha256::new();
    for x in &input.patches {
        h.update(x.to_le_bytes());
    }
    for x in &input.metadata {
        h.update(x.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

pub fn query_hash(pixels: &[u32]) -> String {
    let mut h = Sha256::new();
    for x in pixels {
        h.update(x.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn metadata_hash(metadata: &[f32]) -> String {
    let mut h = Sha256::new();
    for x in metadata {
        h.update(x.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn targets_hash(sample: &Sample) -> String {
    let mut h = Sha256::new();
    for x in &sample.next {
        h.update(x.to_le_bytes());
    }
    for x in sample.policy {
        h.update(x.to_le_bytes());
    }
    for x in sample.rewards {
        h.update(x.to_le_bytes());
    }
    h.update(sample.value.to_le_bytes());
    format!("{:x}", h.finalize())
}

fn visible_cells(pixels: &[u32]) -> Result<Vec<u8>> {
    ensure!(pixels.len() == PIXELS, "wrong query pixel count");
    pixels
        .chunks_exact(PATCH_PIXELS)
        .map(|tile| {
            ensure!(
                tile[0] <= 3 && tile.iter().all(|&x| x == tile[0]),
                "nonuniform/invalid query tile"
            );
            Ok(tile[0] as u8)
        })
        .collect()
}

fn role(cells: &[u8], color: u8) -> Result<usize> {
    let found: Vec<_> = cells
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| (c == color).then_some(i))
        .collect();
    ensure!(found.len() == 1, "query must have exactly one role {color}");
    Ok(found[0])
}

fn checked_input(input: &Inputs) -> Result<()> {
    ensure!(
        input.patches.len() == TOKENS * PATCH_PIXELS && input.metadata.len() == TOKENS * META_DIM,
        "invalid public input shape"
    );
    ensure!(
        input.patches.iter().all(|&x| x < 16) && input.metadata.iter().all(|x| x.is_finite()),
        "invalid public input values"
    );
    Ok(())
}

// The only generator call is here; tests pass unrelated fixture seeds/IDs.
fn build_group(
    cohort: Cohort,
    group_index: usize,
    seed: u64,
    id: u64,
    original: Option<usize>,
) -> Result<Vec<Row>> {
    let mut rows = Vec::with_capacity(cohort.maps().len());
    let permutations = task::permutations();
    for (slot, &map) in cohort.maps().iter().enumerate() {
        let episode = task::episode_with_permutation(seed, id, map, 1, 1)?;
        let sample = task::sample(&episode)?;
        checked_input(&sample.inputs)?;
        let controls = task::inferred_controls(&episode.support)?;
        ensure!(
            controls == permutations[map],
            "support does not identify the actual controls"
        );
        let observed: Vec<_> = episode.support.iter().map(|step| step.action).collect();
        ensure!(
            observed.len() == 3 && observed.iter().copied().collect::<HashSet<_>>().len() == 3,
            "three distinct demonstrations required"
        );
        let cells = visible_cells(&sample.current)?;
        let agent = role(&cells, 2)?;
        let goal = role(&cells, 3)?;
        let dx = (goal % 8) as i32 - (agent % 8) as i32;
        let dy = (goal / 8) as i32 - (agent / 8) as i32;
        let direction = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            .iter()
            .position(|&delta| delta == (dx, dy))
            .context("query is not distance one")?;
        let action = controls
            .iter()
            .position(|&d| d == direction)
            .context("invalid controls")?;
        let (oracle, distance) = task::oracle(&episode.support, &episode.maze.render())?;
        ensure!(
            distance == 1
                && oracle == sample.policy
                && sample.policy == sample.rewards
                && sample.value == 1.0
                && (0..ACTIONS).all(|a| sample.policy[a] == f32::from(a == action)),
            "visible direction, simulator and unique policy/reward/value disagree"
        );
        let omitted = (0..ACTIONS)
            .find(|a| !observed.contains(a))
            .context("missing omitted action")?;
        let mut clear = sample.inputs.clone();
        clear.patches[..SUPPORT_PIXELS].fill(0);
        let factual_hash = input_hash(&sample.inputs);
        let mut audit = json!({
            "schema":SCHEMA,"cohort":cohort,"condition":"factual","support_cleared":false,
            "row_index":group_index*cohort.maps().len()+slot,"group_index":group_index,
            "training_group_index":original,"original_update":original.map(|i| i/4+1),
            "data_seed":seed,"episode_id":id,"permutation_id":map,"correct_action":action,
            "query_direction":direction,"agent_patch":agent,"goal_patch":goal,
            "observed_support_action_ids":observed,"omitted_action":omitted,
            "correct_action_demonstrated":observed.contains(&action),"inferred_controls":controls,
            "input_sha256":factual_hash,"factual_input_sha256":factual_hash,
            "cleared_input_sha256":input_hash(&clear),"metadata_sha256":metadata_hash(&sample.inputs.metadata),
            "query_sha256":query_hash(&sample.current),"targets_sha256":targets_hash(&sample),
            "label_sha256":format!("{:x}",Sha256::digest((action as u32).to_le_bytes()))
        });
        if cohort != Cohort::Training {
            audit["query_cells"] = json!(cells);
        }
        rows.push(Row {
            inputs: sample.inputs,
            correct_action: action,
            agent_patch: agent,
            goal_patch: goal,
            audit_json: audit,
        });
    }
    validate_group(&rows, cohort.maps())?;
    Ok(rows)
}

fn validate_group(rows: &[Row], maps: &[usize]) -> Result<()> {
    ensure!(
        rows.len() == maps.len() && !rows.is_empty(),
        "incomplete group"
    );
    let first = &rows[0];
    let mut inputs = HashSet::new();
    let mut labels = [0usize; ACTIONS];
    let mut omitted_correct = 0;
    for (row, &map) in rows.iter().zip(maps) {
        ensure!(
            row.audit_json["permutation_id"] == json!(map),
            "map order changed"
        );
        ensure!(
            row.inputs.metadata == first.inputs.metadata
                && row.inputs.patches[SUPPORT_PIXELS..] == first.inputs.patches[SUPPORT_PIXELS..]
                && row.audit_json["observed_support_action_ids"]
                    == first.audit_json["observed_support_action_ids"],
            "paired query/public metadata/action IDs differ across maps"
        );
        ensure!(
            inputs.insert(
                row.audit_json["input_sha256"]
                    .as_str()
                    .context("input hash")?
            ),
            "duplicate factual input within mapping group"
        );
        ensure!(
            row.audit_json["cleared_input_sha256"] == first.audit_json["cleared_input_sha256"],
            "cleared inputs differ across maps"
        );
        labels[row.correct_action] += 1;
        omitted_correct +=
            usize::from(row.audit_json["omitted_action"] == json!(row.correct_action));
    }
    ensure!(
        labels == [maps.len() / 4; ACTIONS] && omitted_correct * 4 == maps.len(),
        "balanced map/action-only controls failed"
    );
    Ok(())
}

pub fn group(cohort: Cohort, group_index: usize, cleared: bool) -> Result<Vec<Row>> {
    ensure!(
        !(cleared && cohort == Cohort::Training),
        "training cannot clear supports"
    );
    let (seed, id, original) = cohort.address(group_index)?;
    let rows = build_group(cohort, group_index, seed, id, original)?;
    Ok(if cleared {
        rows.iter().map(Row::cleared).collect()
    } else {
        rows
    })
}

pub fn training_batch(update_zero_based: usize) -> Result<Vec<Row>> {
    ensure!(update_zero_based < UPDATES, "training update out of range");
    let mut rows = Vec::with_capacity(64);
    for offset in 0..4 {
        rows.extend(group(
            Cohort::Training,
            update_zero_based * 4 + offset,
            false,
        )?);
    }
    Ok(rows)
}

/// Project only public task fields. Scoring identities/labels never enter tensors.
pub fn tensors(inputs: &[Inputs], device: &Device) -> Result<(Tensor, Tensor)> {
    ensure!(!inputs.is_empty(), "empty input batch");
    for input in inputs {
        checked_input(input)?;
    }
    Ok((
        Tensor::from_vec(
            inputs
                .iter()
                .flat_map(|i| i.patches.iter().copied())
                .collect::<Vec<_>>(),
            (inputs.len(), TOKENS, PATCH_PIXELS),
            device,
        )?,
        Tensor::from_vec(
            inputs
                .iter()
                .flat_map(|i| i.metadata.iter().copied())
                .collect::<Vec<_>>(),
            (inputs.len(), TOKENS, META_DIM),
            device,
        )?,
    ))
}

#[derive(Clone, Debug, Serialize)]
pub struct FileBinding {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub rows: usize,
}

pub fn file_hash(path: &Path) -> Result<String> {
    let mut h = Sha256::new();
    let mut f = File::open(path)?;
    let mut buf = [0; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// Never opens an existing stream. A partial writer cannot return a completed binding.
pub struct RowStream {
    path: PathBuf,
    writer: BufWriter<File>,
    hash: Sha256,
    cohort: Cohort,
    cleared: bool,
    rows: usize,
}

impl RowStream {
    pub fn create(path: &Path, cohort: Cohort, cleared: bool) -> Result<Self> {
        ensure!(
            !(cohort == Cohort::Training && cleared),
            "cleared training stream forbidden"
        );
        let parent = path.parent().context("stream parent")?;
        ensure!(
            path.is_absolute() && parent.canonicalize()? == parent,
            "stream parent must be canonical absolute"
        );
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        Ok(Self {
            path: path.to_owned(),
            writer: BufWriter::new(file),
            hash: Sha256::new(),
            cohort,
            cleared,
            rows: 0,
        })
    }

    pub fn write(&mut self, rows: &[Row]) -> Result<()> {
        for row in rows {
            ensure!(self.rows < self.cohort.row_count(), "too many stream rows");
            let group_index = self.rows / self.cohort.maps().len();
            let map = self.cohort.maps()[self.rows % self.cohort.maps().len()];
            let (seed, episode_id, _) = self.cohort.address(group_index)?;
            let a = &row.audit_json;
            ensure!(
                a["schema"] == SCHEMA
                    && a["cohort"] == json!(self.cohort)
                    && a["condition"] == if self.cleared { "cleared" } else { "factual" }
                    && a["support_cleared"] == self.cleared
                    && a["row_index"] == json!(self.rows)
                    && a["group_index"] == json!(group_index)
                    && a["permutation_id"] == json!(map)
                    && a["data_seed"] == json!(seed)
                    && a["episode_id"] == json!(episode_id)
                    && a["correct_action"] == json!(row.correct_action)
                    && a["input_sha256"] == input_hash(&row.inputs),
                "stream row/consumed input differs from registered order"
            );
            let mut bytes = serde_json::to_vec(a)?;
            bytes.push(b'\n');
            self.writer.write_all(&bytes)?;
            self.hash.update(&bytes);
            self.rows += 1;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<FileBinding> {
        ensure!(
            self.rows == self.cohort.row_count(),
            "incomplete stream cannot be sealed"
        );
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        let digest = format!("{:x}", self.hash.finalize());
        ensure!(
            file_hash(&self.path)? == digest,
            "stream bytes changed during publication"
        );
        Ok(FileBinding {
            bytes: self.path.metadata()?.len(),
            path: self.path,
            sha256: digest,
            rows: self.rows,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistorySource {
    pub path: PathBuf,
    pub sha256: String,
    /// Exactly Some(512) for C8; None for each complete C11 panel.
    pub first_rows: Option<usize>,
}

fn hash_field<'a>(row: &'a Value, name: &str) -> Result<&'a str> {
    let hash = row[name]
        .as_str()
        .with_context(|| format!("missing {name}"))?;
    ensure!(
        hash.len() == 64
            && hash
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid SHA256 field {name}"
    );
    Ok(hash)
}

pub fn historical_episode(row: &Value) -> Result<u64> {
    let current = row.get("episode_id");
    let legacy = row.get("id");
    if let (Some(a), Some(b)) = (current, legacy) {
        ensure!(a == b, "conflicting historical episode_id/id");
    }
    current
        .or(legacy)
        .and_then(Value::as_u64)
        .context("historical episode_id/id absent or invalid")
}

struct History {
    queries: HashSet<String>,
    bindings: Vec<Value>,
}

fn load_history(sources: &[HistorySource]) -> Result<History> {
    ensure!(
        sources.len() == 4,
        "history requires C8 fitting prefix then C11 panels 0/1/2"
    );
    let mut history = History {
        queries: HashSet::new(),
        bindings: Vec::new(),
    };
    let mut paths = HashSet::new();
    for (index, source) in sources.iter().enumerate() {
        ensure!(
            source.path.is_absolute()
                && source.path.canonicalize()? == source.path
                && source.path.is_file()
                && paths.insert(source.path.clone()),
            "noncanonical/duplicate history path"
        );
        ensure!(
            source.sha256 == HISTORY_HASHES[index] && file_hash(&source.path)? == source.sha256,
            "wrong pinned historical source"
        );
        ensure!(
            source.first_rows == if index == 0 { Some(512) } else { None },
            "wrong historical prefix"
        );
        let mut rows = 0;
        let mut selected = 0;
        let mut selected_queries = HashSet::new();
        for line in BufReader::new(File::open(&source.path)?).lines() {
            let row: Value = serde_json::from_str(&line?)?;
            let query = hash_field(&row, "query_sha256")?;
            hash_field(&row, "input_sha256")?;
            historical_episode(&row)?;
            if source.first_rows.is_none_or(|n| rows < n) {
                ensure!(
                    selected_queries.insert(query.to_owned()),
                    "duplicate query in selected historical panel"
                );
                history.queries.insert(query.to_owned());
                selected += 1;
            }
            rows += 1;
        }
        ensure!(
            rows == if index == 0 { 768 } else { 256 },
            "historical source row count differs"
        );
        history.bindings.push(json!({"path":source.path,"sha256":source.sha256,"bytes":source.path.metadata()?.len(),
            "rows":rows,"first_rows":source.first_rows,"selected_rows":selected,"selected_unique_queries":selected_queries.len()}));
    }
    ensure!(
        history.queries.len() == 1280,
        "C8 fit and C11 historical union differs"
    );
    Ok(history)
}

fn reject_overlap(
    query: &str,
    training: &HashSet<String>,
    historical: &HashSet<String>,
    new: &mut HashSet<String>,
) -> Result<()> {
    ensure!(
        !training.contains(query) && !historical.contains(query) && new.insert(query.to_owned()),
        "registered new query overlaps training/history/new population; no replacement allowed"
    );
    Ok(())
}

/// Generates no model tensors/outputs. Budget includes prior caller initialization.
/// Files are create-new and are never rewritten or silently resumed on failure.
pub fn audit(
    root: &Path,
    sources: &[HistorySource],
    started: Instant,
    max_seconds: u64,
) -> Result<Value> {
    ensure!((1..=120).contains(&max_seconds), "invalid CPU audit budget");
    let deadline = started + Duration::from_secs(max_seconds);
    ensure!(
        root.is_dir() && root.is_absolute() && root.canonicalize()? == root,
        "audit root must exist and be canonical"
    );
    let history = load_history(sources)?;
    let mut train_queries = HashSet::new();
    let mut train_inputs = HashSet::new();
    let mut train_labels = [0usize; ACTIONS];
    let mut train_maps = [0usize; 24];
    let seen_indices: HashSet<_> = (0..64).map(|i| i * 4599 / 63).collect();
    let mut seen_training = BTreeMap::new();
    let identity = |row: &Row| {
        json!([
            row.audit_json["input_sha256"],
            row.audit_json["query_sha256"],
            row.audit_json["targets_sha256"],
            row.audit_json["label_sha256"],
            row.audit_json["metadata_sha256"]
        ])
    };
    let mut train = RowStream::create(&root.join("training-audit.jsonl"), Cohort::Training, false)?;
    for index in 0..TRAIN_GROUPS {
        ensure!(Instant::now() < deadline, "CPU audit deadline exhausted");
        let rows = group(Cohort::Training, index, false)?;
        train_queries.insert(hash_field(&rows[0].audit_json, "query_sha256")?.to_owned());
        for row in &rows {
            train_inputs.insert(hash_field(&row.audit_json, "input_sha256")?.to_owned());
            train_labels[row.correct_action] += 1;
            train_maps[row.audit_json["permutation_id"]
                .as_u64()
                .context("map ID")? as usize] += 1;
        }
        if seen_indices.contains(&index) {
            seen_training.insert(index, rows.iter().map(identity).collect::<Vec<_>>());
        }
        train.write(&rows)?;
    }
    let training_seconds = started.elapsed().as_secs_f64();
    let mut artifacts = vec![train.finish()?];
    let mut new_queries = HashSet::new();
    let mut paired_queries = Vec::new();
    let mut summaries = BTreeMap::new();
    for cohort in [Cohort::Seen, Cohort::Familiar, Cohort::Heldout] {
        let mut factual = RowStream::create(
            &root.join(format!("{}-factual-audit.jsonl", cohort.name())),
            cohort,
            false,
        )?;
        let mut cleared = RowStream::create(
            &root.join(format!("{}-cleared-audit.jsonl", cohort.name())),
            cohort,
            true,
        )?;
        let mut labels = [0usize; ACTIONS];
        let mut map_counts = [0usize; 24];
        let mut unique_queries = HashSet::new();
        let mut omitted = 0;
        for index in 0..EVAL_GROUPS {
            ensure!(Instant::now() < deadline, "CPU audit deadline exhausted");
            let rows = group(cohort, index, false)?;
            let query = hash_field(&rows[0].audit_json, "query_sha256")?;
            unique_queries.insert(query.to_owned());
            match cohort {
                Cohort::Seen => {
                    let original = index * 4599 / 63;
                    ensure!(
                        train_queries.contains(query)
                            && seen_training.get(&original)
                                == Some(&rows.iter().map(identity).collect::<Vec<_>>()),
                        "seen input/query/target/label differs from selected training group"
                    );
                }
                Cohort::Familiar => {
                    reject_overlap(query, &train_queries, &history.queries, &mut new_queries)?;
                    paired_queries.push(query.to_owned());
                }
                Cohort::Heldout => ensure!(
                    paired_queries.get(index).map(String::as_str) == Some(query),
                    "heldout/familiar pairing differs"
                ),
                Cohort::Training => unreachable!(),
            }
            for row in &rows {
                labels[row.correct_action] += 1;
                map_counts[row.audit_json["permutation_id"]
                    .as_u64()
                    .context("map ID")? as usize] += 1;
                omitted += usize::from(
                    !row.audit_json["correct_action_demonstrated"]
                        .as_bool()
                        .context("demonstrated flag")?,
                );
            }
            factual.write(&rows)?;
            cleared.write(&rows.iter().map(Row::cleared).collect::<Vec<_>>())?;
        }
        ensure!(
            labels == [cohort.row_count() / 4; ACTIONS] && omitted * 4 == cohort.row_count()
                && (0..24).all(|map| map_counts[map] == if cohort.maps().contains(&map) {64} else {0}),
            "cohort controls/counts differ"
        );
        summaries.insert(
            cohort.name(),
            json!({"groups":EVAL_GROUPS,"rows_per_condition":cohort.row_count(),
            "map_ids":cohort.maps(),"map_counts":map_counts,"unique_query_images":unique_queries.len(),
            "action_counts":labels,"omitted_action_rows":omitted,
            "demonstrated_action_rows":cohort.row_count()-omitted,"oracle_accuracy":1.0,
            "constant_or_action_id_only_accuracy":0.25,"always_omitted_accuracy":0.25,
            "always_omitted_on_omitted_accuracy":1.0,"always_omitted_on_demonstrated_accuracy":0.0,
            "best_action_id_only_on_demonstrated_accuracy":1.0/3.0,
            "cleared_inputs_identical_within_group":true}),
        );
        artifacts.extend([factual.finish()?, cleared.finish()?]);
    }
    ensure!(
        new_queries.len() == EVAL_GROUPS
            && train_labels == [18400; ACTIONS]
            && (0..24).all(|map| train_maps[map] == if FIT_MAPS.contains(&map) { 4600 } else { 0 }),
        "full population count mismatch"
    );
    ensure!(Instant::now() < deadline, "CPU audit deadline exhausted");
    Ok(
        json!({"schema":SCHEMA,"status":"complete_pending_analysis","evidence_class":"data_audit",
        "optimizer_updates":0,"model_forwards":0,"training_groups":TRAIN_GROUPS,"training_rows":73600,
        "training_unique_query_images":train_queries.len(),"training_duplicate_query_groups":TRAIN_GROUPS-train_queries.len(),
        "training_unique_input_tuples":train_inputs.len(),"training_action_counts":train_labels,
        "training_map_counts":train_maps,"seen_input_target_label_parity":true,
        "historical_selected_unique_queries":history.queries.len(),"new_unique_queries":new_queries.len(),
        "new_query_overlap":0,"histories":history.bindings,"cohorts":summaries,"artifacts":artifacts,
        "training_generation_and_audit_seconds":training_seconds,
        "training_rows_per_second":73600.0/training_seconds,"elapsed_seconds":started.elapsed().as_secs_f64()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    // Fixtures deliberately never call group()/training_batch()/audit(): these
    // APIs generate registered C12 populations and are prelaunch-barrier owned.
    fn fixture(cohort: Cohort) -> Result<Vec<Row>> {
        build_group(cohort, 0, 71, 9, Some(0))
    }

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Result<Self> {
            let path = std::env::temp_dir().join(format!(
                "tofy-grounded-data-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)?
                    .as_nanos()
            ));
            std::fs::create_dir(&path)?;
            Ok(Self(path.canonicalize()?))
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).expect("remove fixture directory");
        }
    }

    #[test]
    fn explicit_map_split_is_bijective_balanced_and_contains_identity_only_in_fit() {
        let permutations = task::permutations();
        assert!(FIT_MAPS.contains(&0));
        assert!(!HELDOUT_MAPS.contains(&0));
        assert_eq!(
            FIT_MAPS
                .iter()
                .chain(&HELDOUT_MAPS)
                .copied()
                .collect::<HashSet<_>>()
                .len(),
            24
        );
        for maps in [&FIT_MAPS[..], &HELDOUT_MAPS[..]] {
            let mut counts = [[0; 4]; 4];
            for &map in maps {
                for (a, &d) in permutations[map].iter().enumerate() {
                    counts[a][d] += 1;
                }
            }
            assert_eq!(counts, [[maps.len() / 4; 4]; 4]);
        }
    }

    #[test]
    fn every_registered_address_and_seen_selection_is_exact_without_generation() -> Result<()> {
        let mut ids = HashSet::new();
        for u in 0..UPDATES {
            for j in 0..4 {
                let index = 4 * u + j;
                let (seed, id, original) = Cohort::Training.address(index)?;
                assert_eq!(
                    (seed, id, original),
                    (20260920, 0x47524f554e445452 + index as u64, Some(index))
                );
                assert!(ids.insert(id));
            }
        }
        for i in 0..64 {
            let original = i * 4599 / 63;
            assert_eq!(
                Cohort::Seen.address(i)?,
                (TRAIN_SEED, TRAIN_BASE + original as u64, Some(original))
            );
            assert_eq!(Cohort::Familiar.address(i)?, Cohort::Heldout.address(i)?);
        }
        assert_eq!(Cohort::Seen.address(63)?.2, Some(4599));
        assert!(Cohort::Familiar.address(64).is_err());
        assert_eq!(
            [
                Cohort::Training.row_count(),
                Cohort::Seen.row_count(),
                Cohort::Familiar.row_count(),
                Cohort::Heldout.row_count()
            ],
            [73600, 1024, 1024, 512]
        );
        Ok(())
    }

    #[test]
    fn visible_transition_fixtures_verify_all_maps_and_action_id_shortcut_counts() -> Result<()> {
        for cohort in [Cohort::Familiar, Cohort::Heldout] {
            let rows = fixture(cohort)?;
            let mut labels = [0; 4];
            let mut omitted = 0;
            for row in &rows {
                labels[row.correct_action] += 1;
                omitted +=
                    usize::from(row.audit_json["omitted_action"] == json!(row.correct_action));
                let cells: Vec<u8> = serde_json::from_value(row.audit_json["query_cells"].clone())?;
                assert_eq!(cells[row.agent_patch], 2);
                assert_eq!(cells[row.goal_patch], 3);
                assert_eq!(
                    (row.agent_patch % 8).abs_diff(row.goal_patch % 8)
                        + (row.agent_patch / 8).abs_diff(row.goal_patch / 8),
                    1
                );
            }
            assert_eq!(labels, [cohort.maps().len() / 4; 4]);
            assert_eq!(omitted * 4, rows.len());
        }
        Ok(())
    }

    #[test]
    fn clearing_preserves_all_public_metadata_and_query_but_removes_map_information() -> Result<()>
    {
        let rows = fixture(Cohort::Familiar)?;
        let clear: Vec<_> = rows.iter().map(Row::cleared).collect();
        for (f, c) in rows.iter().zip(&clear) {
            assert_eq!(f.inputs.metadata, c.inputs.metadata);
            assert_eq!(
                f.inputs.patches[SUPPORT_PIXELS..],
                c.inputs.patches[SUPPORT_PIXELS..]
            );
            assert!(c.inputs.patches[..SUPPORT_PIXELS].iter().all(|&x| x == 0));
            assert!(c.inputs.metadata[..(TOKENS - PATCH_COUNT) * META_DIM]
                .iter()
                .any(|&x| x != 0.0));
            assert_eq!(c.inputs.patches, clear[0].inputs.patches);
            assert_eq!(c.inputs.metadata, clear[0].inputs.metadata);
            assert_eq!(input_hash(&c.inputs), c.audit_json["input_sha256"]);
            assert_ne!(f.audit_json["input_sha256"], c.audit_json["input_sha256"]);
        }
        Ok(())
    }

    #[test]
    fn tensor_projection_cannot_include_audit_labels_or_role_indices() -> Result<()> {
        let mut row = fixture(Cohort::Heldout)?.remove(0);
        let (p, m) = tensors(&[row.inputs.clone()], &Device::Cpu)?;
        row.correct_action = 99;
        row.agent_patch = 999;
        row.goal_patch = 999;
        row.audit_json = json!({"hidden":"forbidden"});
        let (p2, m2) = tensors(&[row.inputs], &Device::Cpu)?;
        assert_eq!(
            p.flatten_all()?.to_vec1::<u32>()?,
            p2.flatten_all()?.to_vec1::<u32>()?
        );
        assert_eq!(
            m.flatten_all()?.to_vec1::<f32>()?,
            m2.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn paired_group_rejects_changed_metadata_query_map_order_and_duplicate_support() -> Result<()> {
        let rows = fixture(Cohort::Heldout)?;
        let mut bad = rows.clone();
        bad[1].inputs.metadata[0] += 0.5;
        assert!(validate_group(&bad, &HELDOUT_MAPS).is_err());
        let mut bad = rows.clone();
        bad[1].inputs.patches[SUPPORT_PIXELS] += 1;
        assert!(validate_group(&bad, &HELDOUT_MAPS).is_err());
        let mut bad = rows.clone();
        bad.swap(0, 1);
        assert!(validate_group(&bad, &HELDOUT_MAPS).is_err());
        let mut bad = rows.clone();
        bad[1].audit_json["input_sha256"] = bad[0].audit_json["input_sha256"].clone();
        assert!(validate_group(&bad, &HELDOUT_MAPS).is_err());
        Ok(())
    }

    #[test]
    fn legacy_ids_and_sha_fields_fail_closed() -> Result<()> {
        assert_eq!(historical_episode(&json!({"id":7}))?, 7);
        assert_eq!(historical_episode(&json!({"episode_id":7,"id":7}))?, 7);
        for value in [
            json!({}),
            json!({"episode_id":null,"id":7}),
            json!({"episode_id":7,"id":8}),
            json!({"id":-1}),
            json!({"id":1.0}),
            json!({"id":true}),
        ] {
            assert!(historical_episode(&value).is_err());
        }
        for value in [
            json!({}),
            json!({"q":"A".repeat(64)}),
            json!({"q":"g".repeat(64)}),
            json!({"q":"0".repeat(63)}),
        ] {
            assert!(hash_field(&value, "q").is_err());
        }
        assert!(hash_field(&json!({"q":"a".repeat(64)}), "q").is_ok());
        Ok(())
    }

    #[test]
    fn novelty_rejects_training_historical_and_internal_overlap() -> Result<()> {
        let training = HashSet::from(["train".to_owned()]);
        let history = HashSet::from(["warm-start".to_owned()]);
        let mut new = HashSet::new();
        assert!(reject_overlap("train", &training, &history, &mut new).is_err());
        assert!(reject_overlap("warm-start", &training, &history, &mut new).is_err());
        reject_overlap("fresh", &training, &history, &mut new)?;
        assert!(reject_overlap("fresh", &training, &history, &mut new).is_err());
        Ok(())
    }

    #[test]
    fn row_stream_refuses_reuse_out_of_order_and_partial_publication() -> Result<()> {
        let temp = Temp::new()?;
        let path = temp.0.join("rows.jsonl");
        let mut stream = RowStream::create(&path, Cohort::Heldout, false)?;
        assert!(RowStream::create(&path, Cohort::Heldout, false).is_err());
        // Fixture IDs are intentionally foreign, so they cannot masquerade as C12 evidence.
        assert!(stream.write(&fixture(Cohort::Heldout)?).is_err());
        assert!(stream.finish().is_err());
        assert_eq!(path.metadata()?.len(), 0);
        Ok(())
    }

    #[test]
    fn invalid_tiles_shapes_and_nonfinite_metadata_are_rejected() -> Result<()> {
        let row = fixture(Cohort::Seen)?.remove(0);
        let mut query = row.inputs.patches[SUPPORT_PIXELS..].to_vec();
        query[1] = (query[0] + 1) % 4;
        assert!(visible_cells(&query).is_err());
        let mut bad = row.inputs;
        bad.metadata[0] = f32::NAN;
        assert!(tensors(&[bad], &Device::Cpu).is_err());
        assert!(visible_cells(&[]).is_err());
        assert!(role(&[2, 2, 3], 2).is_err());
        Ok(())
    }
}
