//! Order-only known-query replay; the generator and per-query loop depth stay fixed.
use super::*;

const QUERIES: usize = 4600;
const UPDATES: usize = 1150;
const BATCH: usize = 64;

pub(super) fn validate_args(args: &Args) -> Result<()> {
    if args.known_replay {
        ensure!(
            args.known_mapping
                && matches!(args.mode, Mode::Coverage | Mode::CoverageAudit)
                && args.coverage == Coverage::Fresh
                && !args.known_seen
                && args.checkpoint.is_none(),
            "--known-replay requires --known-mapping --coverage fresh, mode coverage or coverage-audit, no --known-seen and no checkpoint"
        );
        ensure!(
            args.data_seed == 9173
                && args.seed == 0
                && args.effective_batch == BATCH
                && args.loops == 4
                && matches!(args.updates, 3 | UPDATES),
            "--known-replay requires --data-seed 9173 --seed 0 --effective-batch 64 --loops 4 and --updates 1150 (full) or 3 (implementation smoke)"
        );
    }
    Ok(())
}

pub(super) fn episode_id(id: u64) -> u64 {
    let update = id / BATCH as u64;
    let depth = update % 3;
    let cohort_size = if depth == 0 { 1536 } else { 1532 };
    let index = (64 * (update / 3) + id % 64) % cohort_size;
    12 * (index / 4) + 4 * depth + index % 4
}

pub(super) fn annotate(args: &Args, document: &mut Value) {
    if !args.known_replay {
        return;
    }
    document["known_mapping"]["query_episode_index"] = json!(
        "u=id/64; d=u%3; n=1536 if d=0 else 1532; t=(64*(u/3)+id%64)%n; episode=12*(t/4)+4*d+t%4"
    );
    document["known_mapping"]["training_slots_per_query"] =
        json!(if args.updates == UPDATES { 16 } else { 1 });
    document["known_replay"] = json!({
        "schema":"looped-known-replay-v1","intervention":"batch composition and replay order within original loop-depth cohorts",
        "query_order_matches_contiguous":false,"requested_updates":args.updates,"implementation_smoke":args.updates==3,
        "full_schedule_updates":UPDATES,"full_schedule_rows":UPDATES*BATCH,"full_schedule_queries":QUERIES,
        "full_schedule_visits_per_query":16,"unique_queries_per_update":BATCH,
        "cohort_query_counts":[1536,1532,1532],"loop_depths":[1,2,4],
        "original_update":"episode_index/4+1","original_depth":"[1,2,4][(episode_index/4)%3]",
        "full_schedule_invariant":"same input/target/depth multiset as the contiguous known-mapping stream",
        "smoke_boundary":"three updates cover 192 distinct queries once; full multiset equality applies only to 1150 updates",
        "claim_boundary":"ordering screen only; batch composition and repetition spacing change together; no improvement or internal-cause claim"
    });
}

pub(super) fn annotate_row(id: u64, episode: u64, row: &mut Value) {
    let depth = [1, 2, 4][(id / BATCH as u64 % 3) as usize];
    let original_depth = [1, 2, 4][(episode / 4 % 3) as usize];
    row["known_replay"] = json!(true);
    row["optimizer_update"] = json!(id / BATCH as u64 + 1);
    row["training_loop_depth"] = json!(depth);
    row["original_training_update"] = json!(episode / 4 + 1);
    row["original_training_loop_depth"] = json!(original_depth);
}

pub(super) struct Audit {
    rows: usize,
    visits: Vec<[usize; 3]>,
    batch_queries: HashSet<u64>,
}

impl Default for Audit {
    fn default() -> Self {
        Self {
            rows: 0,
            visits: vec![[0; 3]; QUERIES],
            batch_queries: HashSet::new(),
        }
    }
}

impl Audit {
    pub(super) fn observe(&mut self, id: u64, episode: u64) -> Result<()> {
        ensure!(
            id == self.rows as u64 && episode == episode_id(id),
            "replay row order mismatch"
        );
        ensure!(
            episode < QUERIES as u64,
            "replay query is outside the original population"
        );
        let depth = (id / BATCH as u64 % 3) as usize;
        ensure!(
            depth == (episode / 4 % 3) as usize,
            "replay changed a query's training depth"
        );
        if self.rows.is_multiple_of(BATCH) {
            self.batch_queries.clear();
        }
        ensure!(
            self.batch_queries.insert(episode),
            "replay repeated a query within an update"
        );
        self.visits[episode as usize][depth] += 1;
        self.rows += 1;
        Ok(())
    }

    pub(super) fn finish(self, updates: usize) -> Result<Value> {
        ensure!(
            matches!(updates, 3 | UPDATES) && self.rows == updates * BATCH,
            "replay audit row count mismatch"
        );
        let full = updates == UPDATES;
        let mut queries_by_depth = [0; 3];
        let mut rows_by_depth = [0; 3];
        for (episode, visits) in self.visits.iter().enumerate() {
            let depth = episode / 4 % 3;
            let expected = if full {
                16
            } else {
                usize::from(episode < 3 * BATCH)
            };
            for (d, &count) in visits.iter().enumerate() {
                ensure!(
                    count == if d == depth { expected } else { 0 },
                    "replay multiplicity/depth mismatch for episode {episode}"
                );
                queries_by_depth[d] += usize::from(count > 0);
                rows_by_depth[d] += count;
            }
        }
        Ok(
            json!({"rows":self.rows,"queries":if full {QUERIES} else {3*BATCH},
            "visits_per_query":if full {16} else {1},"unique_queries_per_update":BATCH,
            "queries_by_original_depth":queries_by_depth,"rows_by_original_depth":rows_by_depth,
            "loop_depths":[1,2,4],"original_per_query_depth_preserved":true,
            "full_query_depth_multiset_verified":full,"query_order_matches_contiguous":false}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args::parse_from([
            "probe",
            "--mode",
            "coverage-audit",
            "--coverage",
            "fresh",
            "--known-mapping",
            "--known-replay",
            "--data-seed",
            "9173",
            "--seed",
            "0",
            "--effective-batch",
            "64",
            "--loops",
            "4",
            "--updates",
            "1150",
            "--output-dir",
            "unused",
        ])
    }

    #[test]
    fn complete_replay_matches_sorted_cohorts_and_original_depth_multiplicities() -> Result<()> {
        let cohorts: Vec<Vec<u64>> = (0..3)
            .map(|d| (0..QUERIES as u64).filter(|e| e / 4 % 3 == d).collect())
            .collect();
        assert_eq!(
            cohorts.iter().map(Vec::len).collect::<Vec<_>>(),
            [1536, 1532, 1532]
        );
        let mut visits = vec![HashSet::new(); QUERIES];
        let mut audit = Audit::default();
        for update in 0..UPDATES {
            let depth = update % 3;
            let mut batch = HashSet::new();
            for slot in 0..BATCH {
                let id = (update * BATCH + slot) as u64;
                let episode = episode_id(id);
                assert_eq!(
                    episode,
                    cohorts[depth][(BATCH * (update / 3) + slot) % cohorts[depth].len()]
                );
                assert_eq!(episode / 4 % 3, depth as u64);
                assert!(batch.insert(episode));
                assert!(visits[episode as usize].insert(update));
                audit.observe(id, episode)?;
            }
            assert_eq!(batch.len(), BATCH);
        }
        assert!(visits.iter().all(|updates| updates.len() == 16));
        let report = audit.finish(UPDATES)?;
        assert_eq!(report["rows"], 73600);
        assert_eq!(report["queries"], 4600);
        assert_eq!(
            report["queries_by_original_depth"],
            json!([1536, 1532, 1532])
        );
        assert_eq!(
            report["rows_by_original_depth"],
            json!([24576, 24512, 24512])
        );
        assert_eq!(report["full_query_depth_multiset_verified"], true);
        assert!(Audit::default().observe(0, 1).is_err());
        assert!(Audit::default().finish(3).is_err());
        Ok(())
    }

    #[test]
    fn replay_changes_only_order_and_preserves_fixed_reference_inputs_and_labels() -> Result<()> {
        let replay = args();
        let mut original = args();
        original.known_replay = false;
        for id in [
            0, 1, 63, 64, 127, 128, 191, 192, 4543, 4544, 4607, 4608, 73535, 73599,
        ] {
            let (episode, rule, sample) = coverage_sample(&replay, id)?;
            let (old_episode, old_rule, old) = coverage_sample(&original, episode * 16)?;
            assert_eq!((episode, rule), (old_episode, old_rule));
            assert_eq!(sample.inputs.patches, old.inputs.patches);
            assert_eq!(sample.inputs.metadata, old.inputs.metadata);
            assert_eq!(sample.current, old.current);
            assert_eq!(sample.next, old.next);
            assert_eq!(sample.policy, old.policy);
            assert_eq!(sample.rewards, old.rewards);
            assert_eq!(sample.value, old.value);
            assert_eq!(coverage_sample(&original, id)?.0, id / 16);
        }
        let references = fixed_samples(&replay)?;
        let old_references = fixed_samples(&original)?;
        assert_eq!(references.len(), 128);
        for ((rule, sample), (old_rule, old)) in references.iter().zip(&old_references) {
            assert_eq!(rule, old_rule);
            assert_eq!(
                coverage_row(0, 0, *rule, sample),
                coverage_row(0, 0, *old_rule, old)
            );
        }
        assert_eq!(
            known_mapping::label_controls(references.iter().map(|(_, sample)| sample))?,
            known_mapping::label_controls(old_references.iter().map(|(_, sample)| sample))?
        );
        let mut document = json!({"known_mapping":known_mapping::population()});
        let bytes = serde_json::to_vec(&document)?;
        annotate(&original, &mut document);
        assert_eq!(serde_json::to_vec(&document)?, bytes);
        annotate(&replay, &mut document);
        assert_ne!(
            document["known_mapping"]["query_episode_index"],
            "fresh=id/16; fixed=(id/16)%8"
        );
        assert_eq!(
            document["known_replay"]["query_order_matches_contiguous"],
            false
        );
        Ok(())
    }

    #[test]
    fn replay_guards_reject_unregistered_modes_seeds_counts_and_resume() -> Result<()> {
        for mode in [Mode::Coverage, Mode::CoverageAudit] {
            for updates in [3, UPDATES] {
                let mut input = args();
                input.mode = mode;
                input.updates = updates;
                known_mapping::validate_args(&input)?;
            }
        }
        for mode in [
            Mode::Smoke,
            Mode::Train,
            Mode::Evaluate,
            Mode::Inspect,
            Mode::Counterfactual,
            Mode::CounterfactualAudit,
            Mode::KnownMapping,
            Mode::KnownMappingAudit,
            Mode::Fit,
            Mode::FitSmoke,
        ] {
            let mut input = args();
            input.mode = mode;
            assert!(known_mapping::validate_args(&input).is_err());
        }
        let invalid: [fn(&mut Args); 10] = [
            |a| a.known_mapping = false,
            |a| a.known_seen = true,
            |a| a.coverage = Coverage::Fixed,
            |a| a.data_seed += 1,
            |a| a.seed = 1,
            |a| a.effective_batch = 32,
            |a| a.loops = 2,
            |a| a.updates = 2,
            |a| a.checkpoint = Some(PathBuf::from("unused")),
            |a| a.search = true,
        ];
        for change in invalid {
            let mut input = args();
            change(&mut input);
            assert!(known_mapping::validate_args(&input).is_err());
        }
        Ok(())
    }

    #[test]
    fn three_update_audit_records_consumed_order_without_full_multiset_claim() -> Result<()> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tofy-known-replay-audit-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let result = (|| -> Result<()> {
            let mut input = args();
            input.updates = 3;
            input.output_dir = root.clone();
            known_mapping::validate_args(&input)?;
            let report = known_mapping::coverage_audit(&input, Instant::now())?;
            assert_eq!(report["rows"], 192);
            assert_eq!(report["episode_ids"], 192);
            assert_eq!(report["repeated_complete_input_rows"], 0);
            assert_eq!(report["query_schedule_matches_legacy"], false);
            assert_eq!(
                report["full_input_target_depth_multiset_matches_contiguous"],
                false
            );
            assert_eq!(report["known_replay"]["implementation_smoke"], true);
            assert_eq!(report["known_mapping"]["training_slots_per_query"], 1);
            assert_eq!(
                report["known_replay_audit"]["queries_by_original_depth"],
                json!([64, 64, 64])
            );
            assert_eq!(
                report["known_replay_audit"]["complete_input_target_identity_verified"],
                true
            );
            let path = root.join("training-stream.jsonl");
            assert_eq!(report["training_stream_file_sha256"], file_hash(&path)?);
            let rows = fs::read_to_string(path)?;
            assert_eq!(rows.lines().count(), 192);
            for (id, line) in rows.lines().enumerate() {
                let row: Value = serde_json::from_str(line)?;
                assert_eq!(row["id"], id);
                assert_eq!(row["episode_index"], episode_id(id as u64));
                assert_eq!(row["optimizer_update"], id / 64 + 1);
                assert_eq!(row["training_loop_depth"], [1, 2, 4][id / 64]);
                assert_eq!(
                    row["original_training_loop_depth"],
                    row["training_loop_depth"]
                );
                assert_eq!(row["known_replay"], true);
            }
            Ok(())
        })();
        fs::remove_dir_all(&root)?;
        result
    }
}
