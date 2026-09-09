//! CPU-only public panel construction, exclusion audit and sealed-input loading.
use super::{evidence, Config};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};
use tofy::p2::looped_agent::{
    task::{self, Inputs, Transition},
    META_DIM, OBSERVED_FRAMES, PALETTE, PATCH_COUNT, PATCH_PIXELS, TOKENS,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub input_index: usize,
    pub query_index: usize,
    pub episode_id: u64,
    pub data_seed: u64,
    pub permutation_id: usize,
    pub policy_label: usize,
    pub input_sha256: String,
    pub query_sha256: String,
    pub metadata_sha256: String,
    pub public_cells: Vec<Vec<u32>>,
    pub public_metadata: Vec<f32>,
}
fn hash(parts: impl Iterator<Item = [u8; 4]>) -> String {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    format!("{:x}", h.finalize())
}
impl Row {
    fn observed_actions(&self) -> Vec<&[f32]> {
        (0..3)
            .map(|step| &self.public_metadata[step * 2 * PATCH_COUNT * META_DIM + 3..][..4])
            .collect()
    }
    pub fn inputs(&self) -> Result<Inputs> {
        ensure!(
            self.public_cells.len() == OBSERVED_FRAMES
                && self
                    .public_cells
                    .iter()
                    .all(|f| f.len() == PATCH_COUNT && f.iter().all(|&p| p < PALETTE as u32))
                && self.public_metadata.len() == TOKENS * META_DIM
                && self.public_metadata.iter().all(|x| x.is_finite()),
            "invalid public input shape/values"
        );
        let patches = self
            .public_cells
            .iter()
            .flatten()
            .flat_map(|&p| std::iter::repeat_n(p, PATCH_PIXELS))
            .collect::<Vec<_>>();
        let input = Inputs {
            patches,
            metadata: self.public_metadata.clone(),
        };
        let query = &input.patches[(OBSERVED_FRAMES - 1) * PATCH_COUNT * PATCH_PIXELS..];
        ensure!(
            hash(
                input
                    .patches
                    .iter()
                    .map(|x| x.to_le_bytes())
                    .chain(input.metadata.iter().map(|x| x.to_le_bytes()))
            ) == self.input_sha256
                && hash(query.iter().map(|x| x.to_le_bytes())) == self.query_sha256
                && hash(input.metadata.iter().map(|x| x.to_le_bytes())) == self.metadata_sha256,
            "public input hashes differ"
        );
        let frames = input
            .patches
            .chunks_exact(PATCH_COUNT * PATCH_PIXELS)
            .map(task::unpatchify)
            .collect::<Result<Vec<_>>>()?;
        let mut support = Vec::new();
        for step in 0..3 {
            let action = &input.metadata[step * 2 * PATCH_COUNT * META_DIM + 3..][..4];
            ensure!(
                action.iter().all(|&a| a == 0. || a == 1.) && action.iter().sum::<f32>() == 1.,
                "invalid public action"
            );
            support.push(Transition {
                before: frames[step * 2].clone(),
                after: frames[step * 2 + 1].clone(),
                action: action.iter().position(|&a| a == 1.).unwrap(),
            });
        }
        let rebuilt = task::inputs(&support, &frames[6])?;
        ensure!(
            rebuilt.metadata == input.metadata,
            "public coordinate/frame/action encoding differs"
        );
        let (policy, distance) = task::oracle(&support, &frames[6])?;
        ensure!(
            distance == 1
                && self.policy_label < 4
                && policy[self.policy_label] == 1.
                && self.permutation_id < 24
                && task::inferred_controls(&support)? == task::permutations()[self.permutation_id],
            "audit-only label/mapping oracle differs"
        );
        Ok(input)
    }
}

fn row(seed: u64, tag: u64, query: usize, permutation: usize) -> Result<Row> {
    let episode_id = tag.checked_add(query as u64).context("episode overflow")?;
    let episode = task::episode_with_permutation(seed, episode_id, permutation, 1, 1)?;
    let sample = task::sample(&episode)?;
    let input = sample.inputs;
    let result = Row {
        input_index: query * 24 + permutation,
        query_index: query,
        episode_id,
        data_seed: seed,
        permutation_id: permutation,
        policy_label: sample
            .policy
            .iter()
            .position(|&v| v == 1.)
            .context("nonunique one-step label")?,
        input_sha256: hash(
            input
                .patches
                .iter()
                .map(|x| x.to_le_bytes())
                .chain(input.metadata.iter().map(|x| x.to_le_bytes())),
        ),
        query_sha256: hash(sample.current.iter().map(|x| x.to_le_bytes())),
        metadata_sha256: hash(input.metadata.iter().map(|x| x.to_le_bytes())),
        public_cells: input
            .patches
            .chunks_exact(PATCH_COUNT * PATCH_PIXELS)
            .map(|f| f.chunks_exact(PATCH_PIXELS).map(|p| p[0]).collect())
            .collect(),
        public_metadata: input.metadata,
    };
    result.inputs()?;
    Ok(result)
}

fn history_hash(row: &Value) -> Result<&str> {
    let a = row.get("query_sha256");
    let b = row.get("query_hash");
    ensure!(a.is_some() || b.is_some(), "history query hash missing");
    ensure!(
        a.is_none() || b.is_none() || a == b,
        "conflicting historical query hash fields"
    );
    let hash = a
        .or(b)
        .unwrap()
        .as_str()
        .context("history query hash not string")?;
    ensure!(
        hash.len() == 64
            && hash
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "malformed history query hash"
    );
    Ok(hash)
}
pub fn audit(config: &Config, started: Instant) -> Result<Value> {
    let mut excluded = HashSet::new();
    let mut histories = Vec::new();
    for source in &config.history {
        config.deadline(started)?;
        source.verify()?;
        let mut unique = HashSet::new();
        let mut parsed = 0;
        let mut occurrences = 0;
        for line in BufReader::new(File::open(&source.path)?).lines() {
            config.deadline(started)?;
            let value: Value = serde_json::from_str(&line?)?;
            let h = history_hash(&value)?.to_owned();
            occurrences += usize::from(value.get("query_sha256").is_some())
                + usize::from(value.get("query_hash").is_some());
            unique.insert(h.clone());
            excluded.insert(h);
            parsed += 1;
        }
        ensure!(parsed > 0, "empty historical exclusion");
        source.verify()?;
        histories.push(json!({"path":source.path,"sha256":source.sha256,"parsed_rows":parsed,"hash_occurrences":occurrences,"unique_query_hashes":unique.len(),"conflicts":0}));
    }
    let mut output = BufWriter::new(File::create(config.output_dir.join("panel-rows.jsonl"))?);
    let mut queries = HashSet::new();
    let mut counts = [0usize; 4];
    for q in 0..config.query_groups {
        let mut reference = None;
        let mut action_reference = None;
        let mut group_counts = [0usize; 4];
        for p in 0..24 {
            config.deadline(started)?;
            let r = row(config.panel_seed, config.panel_tag, q, p)?;
            ensure!(
                !excluded.contains(&r.query_sha256),
                "historical query collision; no repair allowed"
            );
            if let Some(ref query) = reference {
                ensure!(query == &r.query_sha256, "mapping changed query");
            } else {
                ensure!(
                    queries.insert(r.query_sha256.clone()),
                    "duplicate fresh layout"
                );
                reference = Some(r.query_sha256.clone());
            }
            counts[r.policy_label] += 1;
            group_counts[r.policy_label] += 1;
            let actions = r
                .observed_actions()
                .into_iter()
                .map(<[f32]>::to_vec)
                .collect::<Vec<_>>();
            if let Some(ref first) = action_reference {
                ensure!(first == &actions, "mapping changed observed action order");
            } else {
                action_reference = Some(actions);
            }
            serde_json::to_writer(&mut output, &r)?;
            output.write_all(b"\n")?;
        }
        ensure!(group_counts == [6; 4], "query group label balance differs");
    }
    output.flush()?;
    ensure!(
        counts == [config.query_groups * 6; 4],
        "unbalanced full mapping sweep"
    );
    Ok(
        json!({"status":"complete_pending_analysis","classification":"input_audit","model_forwards":0,"optimizer_updates":0,"input_rows":config.query_groups*24,"query_groups":queries.len(),"panel_seed":config.panel_seed,"panel_tag":config.panel_tag,"history":histories,"excluded_unique_queries":excluded.len(),"label_counts":counts,"elapsed_seconds":started.elapsed().as_secs_f64()}),
    )
}
pub fn load(config: &Config) -> Result<Vec<Row>> {
    let root = config.audit_root.as_ref().context("missing audit root")?;
    evidence::verify_root(
        root,
        config
            .audit_manifest_sha256
            .as_deref()
            .context("missing audit pin")?,
    )?;
    let report: Value = serde_json::from_reader(File::open(root.join("report.json"))?)?;
    ensure!(
        report["status"] == "complete_pending_analysis"
            && report["model_forwards"] == 0
            && report["optimizer_updates"] == 0
            && report["panel_seed"] == config.panel_seed
            && report["panel_tag"] == config.panel_tag
            && report["query_groups"] == config.query_groups,
        "audit report identity differs"
    );
    let rows = read_rows(&root.join("panel-rows.jsonl"))?;
    ensure!(rows.len() == config.query_groups * 24, "panel size differs");
    let mut unique = HashSet::new();
    let mut labels = [0usize; 4];
    for (i, row) in rows.iter().enumerate() {
        ensure!(
            row.input_index == i
                && row.query_index == i / 24
                && row.permutation_id == i % 24
                && row.data_seed == config.panel_seed
                && row.episode_id == config.panel_tag + row.query_index as u64,
            "panel row/order differs"
        );
        row.inputs()?;
        labels[row.policy_label] += 1;
        if i % 24 == 0 {
            ensure!(unique.insert(&row.query_sha256), "repeated query group");
        } else {
            ensure!(
                row.query_sha256 == rows[i / 24 * 24].query_sha256
                    && row.observed_actions() == rows[i / 24 * 24].observed_actions(),
                "mapping changed query or observed action order"
            );
        }
    }
    for group in rows.chunks_exact(24) {
        let mut counts = [0usize; 4];
        for row in group {
            counts[row.policy_label] += 1;
        }
        ensure!(counts == [6; 4], "query group label balance differs");
    }
    ensure!(
        labels == [config.query_groups * 6; 4],
        "panel label balance differs"
    );
    Ok(rows)
}
fn read_rows(path: &Path) -> Result<Vec<Row>> {
    BufReader::new(File::open(path)?)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn synthetic_full_mapping_group_preserves_queries_and_balances_labels() -> Result<()> {
        let mut hashes = HashSet::new();
        let mut inputs = HashSet::new();
        let mut labels = [0; 4];
        let mut action_order = None;
        for p in 0..24 {
            let r = row(71, 9, 0, p)?;
            r.inputs()?;
            let actions = r
                .observed_actions()
                .into_iter()
                .map(<[f32]>::to_vec)
                .collect::<Vec<_>>();
            if let Some(ref expected) = action_order {
                assert_eq!(expected, &actions);
            } else {
                action_order = Some(actions);
            }
            hashes.insert(r.query_sha256);
            inputs.insert(r.input_sha256);
            labels[r.policy_label] += 1;
        }
        assert_eq!((hashes.len(), inputs.len(), labels), (1, 24, [6; 4]));
        Ok(())
    }
    #[test]
    fn history_duplicate_fields_and_corruption_fail_closed() -> Result<()> {
        let h = "a".repeat(64);
        assert_eq!(history_hash(&json!({"query_hash":h}))?, h);
        assert_eq!(history_hash(&json!({"query_hash":h,"query_sha256":h}))?, h);
        for bad in [
            json!({}),
            json!({"query_sha256":3}),
            json!({"query_sha256":"z".repeat(64)}),
            json!({"query_sha256":h,"query_hash":"b".repeat(64)}),
        ] {
            assert!(history_hash(&bad).is_err());
        }
        let mut r = row(71, 9, 0, 0)?;
        r.public_cells[0][0] = 17;
        assert!(r.inputs().is_err());
        Ok(())
    }
}
