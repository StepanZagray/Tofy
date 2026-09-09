//! Fixed-known-control spatial prerequisite; no model, task or loss changes.
use super::counterfactual as cf;
use super::*;
use serde::Serialize;
use std::collections::BTreeMap;

const SCHEMA: &str = "looped-known-mapping-v1";
pub(super) const EVAL_SEED: u64 = 20260912;
/// ASCII KNOWN: disjoint from the legacy and counterfactual episode namespaces.
pub(super) const EVAL_EPISODE_TAG: u64 = 0x4b4e4f574e;
const EVAL_LAYOUTS: usize = 64;
const SEEN_LAYOUTS: usize = 72;
const SEEN_SEED: u64 = 9173;
const SEEN_BLOCK_STARTS: [u64; 3] = [0, 2280, 4576];
const SEEN_DEPTH_CAVEAT: &str = "all queries are scored at four loops; this is not original-depth fitting for examples trained at one or two loops";

pub(super) fn validate_args(args: &Args) -> Result<()> {
    known_features::validate_args(args)?;
    known_replay::validate_args(args)?;
    let dedicated = matches!(args.mode, Mode::KnownMapping | Mode::KnownMappingAudit);
    ensure!(
        !args.known_seen || (dedicated && args.known_mapping),
        "--known-seen requires --known-mapping and mode known-mapping or known-mapping-audit"
    );
    ensure!(
        !dedicated || args.known_mapping,
        "known-mapping modes require --known-mapping"
    );
    if args.known_mapping {
        ensure!(matches!(args.mode, Mode::Coverage | Mode::CoverageAudit | Mode::FitSmoke | Mode::KnownMapping | Mode::KnownMappingAudit | Mode::KnownFeatures | Mode::KnownFeaturesAudit | Mode::KnownFeaturesSmoke),
                "--known-mapping is scoped to coverage, coverage-audit, fit-smoke and known-mapping modes");
        ensure!(
            !args.search,
            "known-mapping prerequisite does not run search or wrong-rule stress tests"
        );
        if dedicated {
            if args.known_seen {
                ensure!(
                    args.data_seed == SEEN_SEED
                        && args.eval_episodes == SEEN_LAYOUTS
                        && args.loops == 4,
                    "--known-seen requires --data-seed 9173 --eval-episodes 72 --loops 4"
                );
            } else {
                ensure!(
                    args.data_seed == EVAL_SEED && args.eval_episodes == EVAL_LAYOUTS,
                    "known-mapping frozen population requires --data-seed 20260912 --eval-episodes 64"
                );
            }
            ensure!(
                args.batch == 1 && args.effective_batch == 1,
                "known-mapping frozen/audit modes require physical/effective batch one"
            );
        }
    }
    Ok(())
}

pub(super) fn population() -> Value {
    json!({"schema":SCHEMA,"control_population":"KnownMapping","trained_permutation_ids":[0],
        "permutation_id":0,"controls_action_to_direction":[0,1,2,3],
        "legacy_partition":"permutation zero belonged to legacy HeldOut; it is trained here",
        "factual_support":"normal public calibration frames and metadata; no rule ID feature",
        "training_slots_per_query":16,"query_episode_index":"fresh=id/16; fixed=(id/16)%8",
        "frozen_data_seed":EVAL_SEED,"frozen_episode_id_base":EVAL_EPISODE_TAG,"frozen_layouts":EVAL_LAYOUTS,
        "blind_bound_applies":false,"promotion":false})
}

pub(super) fn annotate(document: &mut Value) {
    document["known_mapping"] = population();
    document["training_rule_ids"] = json!([0]);
    document["held_out_rule_ids"] = json!([]);
}

fn panel_layouts(args: &Args) -> usize {
    if args.known_seen {
        SEEN_LAYOUTS
    } else {
        EVAL_LAYOUTS
    }
}

fn seen_episode_id(layout: usize) -> u64 {
    SEEN_BLOCK_STARTS[layout / 24] + (layout % 24) as u64
}

pub(super) fn annotate_seen(args: &Args, document: &mut Value) {
    if !args.known_seen {
        return;
    }
    let ids: Vec<_> = (0..SEEN_LAYOUTS).map(seen_episode_id).collect();
    document["known_seen"] = json!({
        "schema":"looped-known-seen-v1","training_membership":"intended; verify factual input, query and target hashes against the source training stream",
        "episode_source":"coverage/fresh","data_seed":SEEN_SEED,"episode_seed":SEEN_SEED ^ TRAIN_TAG,
        "training_episode_ids":ids,"temporal_blocks":["early","middle","late"],
        "layouts_per_temporal_block":24,"layouts_per_original_depth_per_block":8,
        "source_training_updates":1150,"source_effective_batch":64,"training_slots_per_query":16,
        "original_training_update":"episode_index / 4 + 1",
        "original_training_loop_depth":"[1,2,4][(episode_index / 4) % 3]",
        "evaluation_loops":4,"depth_caveat":SEEN_DEPTH_CAVEAT,"promotion":false
    });
    document["known_mapping"]["frozen_data_seed"] = json!(SEEN_SEED);
    document["known_mapping"]["frozen_episode_seed"] = json!(SEEN_SEED ^ TRAIN_TAG);
    document["known_mapping"]["frozen_episode_id_base"] = Value::Null;
    document["known_mapping"]["frozen_layouts"] = json!(SEEN_LAYOUTS);
}

fn controls_from_counts(counts: [usize; ACTIONS]) -> Result<Value> {
    let rows: usize = counts.iter().sum();
    ensure!(rows > 0, "known-mapping control population is empty");
    let best = *counts.iter().max().expect("four actions");
    let best_actions: Vec<_> = (0..ACTIONS).filter(|&a| counts[a] == best).collect();
    Ok(
        json!({"rows":rows,"label_counts_by_action":counts,"best_constant_actions":best_actions,
        "best_constant_action_correct":best,"best_constant_action_accuracy":best as f64/rows as f64,
        "uniform_random_expected_accuracy":0.25,"uniform_random_policy_ce":4.0f64.ln(),
        "qualifier":"uniform random is an expectation, not a factual/cleared information bound; a known mapping can use query geometry"}),
    )
}

pub(super) fn label_controls<'a>(samples: impl Iterator<Item = &'a Sample>) -> Result<Value> {
    let mut counts = [0; ACTIONS];
    for sample in samples {
        let label = argmax(&sample.policy);
        ensure!(
            sample.policy[label] == 1.0,
            "one-step known-mapping label must be unique"
        );
        counts[label] += 1;
    }
    controls_from_counts(counts)
}

pub(super) fn training_row(id: u64, episode: u64, sample: &Sample) -> Value {
    let mut row = coverage_row(id, episode, 0, sample);
    row["control_population"] = json!("KnownMapping");
    row
}

fn clear_support(input: &mut Inputs) {
    input.patches[..(TOKENS - PATCH_COUNT) * PATCH_PIXELS].fill(0);
    input.metadata[..(TOKENS - PATCH_COUNT) * META_DIM].fill(0.0);
}

fn panel_input(args: &Args, layout: usize, cleared: bool) -> Result<(Value, Sample)> {
    ensure!(
        layout < panel_layouts(args),
        "known-mapping layout out of range"
    );
    let (data_seed, episode_seed, episode_id) = if args.known_seen {
        (
            args.data_seed,
            args.data_seed ^ TRAIN_TAG,
            seen_episode_id(layout),
        )
    } else {
        (EVAL_SEED, EVAL_SEED, EVAL_EPISODE_TAG + layout as u64)
    };
    let episode = task::episode_with_permutation(episode_seed, episode_id, 0, 1, 1)?;
    let (policy, distance) = task::oracle(&episode.support, &episode.maze.render())?;
    ensure!(
        distance == 1 && task::inferred_controls(&episode.support)? == [0, 1, 2, 3],
        "known-mapping public oracle/control mismatch"
    );
    let observed: Vec<_> = episode.support.iter().map(|step| step.action).collect();
    let mut sample = task::sample(&episode)?;
    ensure!(
        sample.policy == policy && sample.value == 1.0,
        "known-mapping policy/value target mismatch"
    );
    let factual_hash = input_hash(&sample.inputs);
    if cleared {
        clear_support(&mut sample.inputs);
    }
    let mut row = json!({"schema":SCHEMA,"input_index":layout*2+usize::from(cleared),"layout_index":layout,
        "episode_id":episode_id,"data_seed":data_seed,"permutation_id":0,"split":"KnownMapping",
        "condition":if cleared {"cleared"} else {"factual"},"min_distance":1,"max_distance":1,"oracle_distance":distance,
        "factual_support_action_ids":observed,"support_cleared":cleared,
        "input_sha256":input_hash(&sample.inputs),"factual_input_sha256":factual_hash,
        "query_sha256":query_hash(&sample.current),"target_policy":sample.policy,"target_value":sample.value});
    if args.known_seen {
        let original_depth = [1, 2, 4][(episode_id / 4 % 3) as usize];
        let temporal_block = ["early", "middle", "late"][layout / 24];
        row.as_object_mut().expect("identity object").extend(json!({
            "known_seen":true,"seen_status":if cleared {"training_query_with_cleared_support"} else {"intended_training_input"},
            "episode_source":"coverage/fresh","episode_seed":episode_seed,
            "training_episode_index":episode_id,"original_training_update":episode_id/4+1,
            "original_training_loop_depth":original_depth,
            "temporal_block":temporal_block,
            "targets_sha256":coverage_row(episode_id*16,episode_id,0,&sample)["targets_sha256"],
            "evaluation_loops":4,"depth_caveat":SEEN_DEPTH_CAVEAT
        }).as_object().expect("seen identity object").clone());
    }
    Ok((row, sample))
}

pub(super) fn audit(
    args: &Args,
    started: Instant,
    training_queries: &HashSet<String>,
) -> Result<Value> {
    let path = args.output_dir.join("known-mapping-input-audit.jsonl");
    let mut writer = BufWriter::new(File::create_new(&path)?);
    let layouts = panel_layouts(args);
    let mut queries = HashSet::new();
    let mut labels = [0; ACTIONS];
    let mut overlap = 0;
    for layout in 0..layouts {
        deadline(args, started)?;
        let (factual, sample) = panel_input(args, layout, false)?;
        let query = factual["query_sha256"]
            .as_str()
            .context("missing query hash")?;
        overlap += usize::from(training_queries.contains(query));
        ensure!(
            args.known_seen || !training_queries.contains(query),
            "known-mapping frozen query overlaps training"
        );
        ensure!(
            queries.insert(query.to_owned()),
            "known-mapping frozen layouts have duplicate queries"
        );
        labels[argmax(&sample.policy)] += 1;
        let (cleared, other) = panel_input(args, layout, true)?;
        ensure!(
            sample.current == other.current
                && sample.next == other.next
                && sample.policy == other.policy,
            "known-mapping factual/cleared pairing mismatch"
        );
        for identity in [factual, cleared] {
            serde_json::to_writer(&mut writer, &identity)?;
            writer.write_all(b"\n")?;
        }
    }
    writer.flush()?;
    drop(writer);
    let mut report = json!({"schema":SCHEMA,"task_schema":task::SCHEMA,"status":"complete_pending_analysis",
        "evidence_class":"data_audit","known_mapping":population(),"optimizer_updates":0,"model_forwards":0,
        "layouts":layouts,"input_rows":layouts*2,"action_rows":layouts*2*ACTIONS,
        "factual_input_rows":layouts,"factual_action_rows":layouts*ACTIONS,
        "cleared_input_rows":layouts,"cleared_action_rows":layouts*ACTIONS,
        "policy_controls":controls_from_counts(labels)?,"unique_query_frames":queries.len(),
        "training_queries_checked":training_queries.len(),
        "evaluation_query_overlap":if training_queries.is_empty() {Value::Null} else {json!(overlap)},
        "artifacts":[{"file":"known-mapping-input-audit.jsonl","bytes":path.metadata()?.len(),"sha256":file_hash(&path)?}],
        "elapsed_seconds":started.elapsed().as_secs_f64()});
    annotate_seen(args, &mut report);
    known_replay::annotate(args, &mut report);
    if args.known_seen {
        report["expected_training_query_members"] = json!(layouts);
        report["external_training_query_members"] = if training_queries.is_empty() {
            Value::Null
        } else {
            json!(overlap)
        };
        report["external_factual_input_and_target_membership_verified"] = Value::Null;
    }
    Ok(report)
}

pub(super) fn coverage_audit(args: &Args, started: Instant) -> Result<Value> {
    let rows = args
        .updates
        .checked_mul(args.effective_batch)
        .context("training row count overflow")?;
    ensure!(
        rows > 0 && rows.is_multiple_of(16),
        "known-mapping coverage audit requires complete sixteen-row query blocks"
    );
    let mut writer = BufWriter::new(File::create_new(
        args.output_dir.join("training-stream.jsonl"),
    )?);
    let mut seen = HashMap::new();
    let mut queries = HashSet::new();
    let mut inputs = HashSet::new();
    let mut labels = [0; ACTIONS];
    let mut replay = args.known_replay.then(known_replay::Audit::default);
    for id in 0..rows as u64 {
        deadline(args, started)?;
        let (episode, rule, sample) = coverage_sample(args, id)?;
        ensure!(
            rule == 0
                && episode
                    == if args.known_replay {
                        known_replay::episode_id(id)
                    } else if args.coverage == Coverage::Fixed {
                        (id / 16) % 8
                    } else {
                        id / 16
                    },
            "known-mapping sampler differs from its declared query schedule"
        );
        let mut row = training_row(id, episode, &sample);
        if let Some(replay) = &mut replay {
            replay.observe(id, episode)?;
            known_replay::annotate_row(id, episode, &mut row);
        }
        let identity = (
            row["input_sha256"].clone(),
            row["query_sha256"].clone(),
            row["targets_sha256"].clone(),
            row["correct_action"].clone(),
        );
        if let Some(previous) = seen.insert(episode, identity.clone()) {
            ensure!(
                previous == identity,
                "repeated known-mapping query/input/targets changed"
            );
        } else if args.known_replay {
            let original = task::sample(&task::episode_with_permutation(
                args.data_seed ^ TRAIN_TAG,
                episode,
                0,
                1,
                1,
            )?)?;
            ensure!(
                sample.inputs.patches == original.inputs.patches
                    && sample.inputs.metadata == original.inputs.metadata
                    && sample.current == original.current
                    && sample.next == original.next
                    && sample.policy == original.policy
                    && sample.rewards == original.rewards
                    && sample.value == original.value,
                "replay changed the original known-mapping input or targets"
            );
        } else {
            let legacy_rule = task::permutation_ids(Split::Train)[0];
            let legacy = task::episode_with_permutation(
                args.data_seed ^ TRAIN_TAG,
                episode,
                legacy_rule,
                1,
                1,
            )?;
            ensure!(
                sample.current == task::sample(&legacy)?.current,
                "known-mapping query differs from legacy stream"
            );
        }
        queries.insert(
            row["query_sha256"]
                .as_str()
                .context("missing query")?
                .to_owned(),
        );
        inputs.insert(
            row["input_sha256"]
                .as_str()
                .context("missing input")?
                .to_owned(),
        );
        labels[argmax(&sample.policy)] += 1;
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    drop(writer);
    let expected_episodes = if args.known_replay {
        if args.updates == 3 {
            192
        } else {
            4600
        }
    } else if args.coverage == Coverage::Fixed {
        (rows / 16).min(8)
    } else {
        rows / 16
    };
    ensure!(
        seen.len() == expected_episodes,
        "known-mapping query ID coverage differs"
    );
    let replay_report = replay.map(|audit| audit.finish(args.updates)).transpose()?;
    if args.known_replay {
        ensure!(
            inputs.len() == expected_episodes && queries.len() == expected_episodes,
            "replay population has duplicate complete inputs or queries"
        );
    }
    let frozen = audit(args, started, &queries)?;
    let mut counts = [0usize; 24];
    counts[0] = rows;
    let mut report = json!({"status":"complete_pending_analysis","schema":SCHEMA,"evidence_class":"data_audit",
        "known_mapping":population(),"coverage":args.coverage,"optimizer_updates":0,"model_forwards":0,
        "rows":rows,"episode_ids":seen.len(),"unique_query_frames":queries.len(),"unique_inputs":inputs.len(),
        "repeated_complete_input_rows":rows-inputs.len(),"rule_counts":counts,
        "query_schedule_matches_legacy":!args.known_replay,"repeated_known_inputs_and_targets_identical":true,
        "policy_controls":controls_from_counts(labels)?,"frozen_panel":frozen,
        "training_stream_file_sha256":file_hash(&args.output_dir.join("training-stream.jsonl"))?,
        "elapsed_seconds":started.elapsed().as_secs_f64()});
    known_replay::annotate(args, &mut report);
    if let Some(mut replay) = replay_report {
        replay["complete_input_target_identity_verified"] = json!(true);
        report["known_replay_audit"] = replay;
        report["full_input_target_depth_multiset_matches_contiguous"] = json!(args.updates == 1150);
    }
    Ok(report)
}

#[derive(Default, Serialize)]
struct Scores {
    rows: usize,
    exact: usize,
    copy_exact: usize,
    changed: cf::Region,
    unchanged: cf::Region,
    vacated: cf::Region,
    destination: cf::Region,
    patch_inconsistency_count: usize,
    reward_positive: usize,
    reward_true_positive: usize,
    reward_false_positive: usize,
    reward_squared_error_sum: f64,
}

impl Scores {
    fn add(&mut self, metrics: &cf::Metrics, reward: f32, prediction: f32) {
        self.rows += 1;
        self.exact += usize::from(metrics.exact);
        self.copy_exact += usize::from(metrics.copy_exact);
        for (total, next) in [
            (&mut self.changed, &metrics.changed),
            (&mut self.unchanged, &metrics.unchanged),
            (&mut self.vacated, &metrics.vacated),
            (&mut self.destination, &metrics.destination),
        ] {
            total.correct += next.correct;
            total.total += next.total;
        }
        self.patch_inconsistency_count += metrics.patch_inconsistency_count;
        self.reward_positive += usize::from(reward == 1.0);
        self.reward_true_positive += usize::from(reward == 1.0 && prediction >= 0.5);
        self.reward_false_positive += usize::from(reward == 0.0 && prediction >= 0.5);
        self.reward_squared_error_sum += (f64::from(prediction) - f64::from(reward)).powi(2);
    }
}

#[derive(Default)]
struct PolicyScores {
    correct: usize,
    ce: f64,
    value_squared_error: f64,
    labels: [usize; ACTIONS],
}

pub(super) fn evaluate(
    args: &Args,
    model: &LoopedAgent,
    device: &Device,
    started: Instant,
) -> Result<Value> {
    let layouts = panel_layouts(args);
    let probability_path = args.output_dir.join("successor-probabilities.f32");
    let states_path = args.output_dir.join("successor-states.u8");
    let rows_path = args.output_dir.join("successor-rows.jsonl");
    let mut probabilities_file = BufWriter::new(File::create_new(&probability_path)?);
    let mut states_file = BufWriter::new(File::create_new(&states_path)?);
    let mut rows_file = BufWriter::new(File::create_new(&rows_path)?);
    let mut scores: BTreeMap<String, Scores> = BTreeMap::new();
    let mut policies = [PolicyScores::default(), PolicyScores::default()];
    let mut maximum_probability_error = 0.0f64;
    for layout in 0..layouts {
        for (condition_index, cleared) in [false, true].into_iter().enumerate() {
            deadline(args, started)?;
            let input_index = layout * 2 + condition_index;
            let condition = if cleared { "cleared" } else { "factual" };
            let (mut row, sample) = panel_input(args, layout, cleared)?;
            let output = if input_index == 0 && args.profile_eval {
                inspected_predict(args, model, sample.inputs.clone(), device)?
            } else {
                predict(
                    model,
                    std::slice::from_ref(&sample.inputs),
                    args.loops,
                    device,
                )?
            };
            ensure!(
                output.next_logits.dims() == [1, ACTIONS, PATCH_COUNT, PATCH_PIXELS, PALETTE]
                    && output.policy_logits.dims() == [1, ACTIONS]
                    && output.reward_logits.dims() == [1, ACTIONS]
                    && output.value.dims() == [1, 1],
                "unexpected known-mapping readout shape"
            );
            let probabilities = candle_nn::ops::softmax(&output.next_logits, D::Minus1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            maximum_probability_error =
                maximum_probability_error.max(cf::validate_probabilities(&probabilities, PALETTE)?);
            let predicted: Vec<u32> = probabilities
                .chunks_exact(PALETTE)
                .map(|pixel| {
                    (1..PALETTE).fold(0, |best, i| if pixel[i] > pixel[best] { i } else { best })
                        as u32
                })
                .collect();
            let policy_logits = output.policy_logits.flatten_all()?.to_vec1::<f32>()?;
            let policy_probabilities = candle_nn::ops::softmax(&output.policy_logits, D::Minus1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let policy_log_probabilities =
                candle_nn::ops::log_softmax(&output.policy_logits, D::Minus1)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
            let reward_logits = output.reward_logits.flatten_all()?.to_vec1::<f32>()?;
            let reward_probabilities = candle_nn::ops::sigmoid(&output.reward_logits)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let value_logit = output.value.flatten_all()?.to_vec1::<f32>()?[0];
            let value = candle_nn::ops::sigmoid(&output.value)?
                .flatten_all()?
                .to_vec1::<f32>()?[0];
            cf::validate_probabilities(&policy_probabilities, ACTIONS)?;
            ensure!(
                policy_logits
                    .iter()
                    .chain(&policy_log_probabilities)
                    .chain(&reward_logits)
                    .chain(std::iter::once(&value_logit))
                    .all(|x| x.is_finite())
                    && reward_probabilities
                        .iter()
                        .chain(std::iter::once(&value))
                        .all(|x| x.is_finite() && (0.0..=1.0).contains(x)),
                "non-finite or out-of-range known-mapping policy/reward/value"
            );
            let label = argmax(&sample.policy);
            let policy_action = argmax(&policy_logits);
            let policy_ce = -f64::from(policy_log_probabilities[label]);
            let aggregate = &mut policies[condition_index];
            aggregate.labels[label] += 1;
            aggregate.correct += usize::from(label == policy_action);
            aggregate.ce += policy_ce;
            aggregate.value_squared_error += (f64::from(value) - 1.0).powi(2);
            let offsets = cf::offsets(input_index);
            cf::write_binary(
                &mut probabilities_file,
                &mut states_file,
                &probabilities,
                &sample.current,
                &sample.next,
                &predicted,
            )?;
            let observed = row["factual_support_action_ids"]
                .as_array()
                .context("missing factual support actions")?;
            let mut actions = Vec::new();
            for action in 0..ACTIONS {
                let target = &sample.next[action * PIXELS..(action + 1) * PIXELS];
                let prediction = &predicted[action * PIXELS..(action + 1) * PIXELS];
                let metrics =
                    cf::metrics(&sample.current, target, prediction, sample.rewards[action])?;
                let demonstrated = observed.contains(&json!(action));
                for stratum in [
                    "all",
                    metrics.class,
                    if demonstrated {
                        "demonstrated_in_factual_support"
                    } else {
                        "bijection_inferred_in_factual_support"
                    },
                ] {
                    scores
                        .entry(format!("{condition}/{stratum}"))
                        .or_default()
                        .add(
                            &metrics,
                            sample.rewards[action],
                            reward_probabilities[action],
                        );
                }
                let mut action_row = serde_json::to_value(&metrics)?;
                action_row.as_object_mut().expect("metrics object").extend(json!({
                    "action":action,"direction":action,"demonstrated_in_factual_support":demonstrated,
                    "target_reward":sample.rewards[action],"target_sha256":query_hash(target),"predicted_sha256":query_hash(prediction),
                    "probabilities":cf::action_range(offsets.probabilities,action),
                    "target_state":cf::action_range(offsets.targets,action),"predicted_state":cf::action_range(offsets.predicted,action)
                }).as_object().expect("action object").clone());
                actions.push(action_row);
            }
            row.as_object_mut().expect("identity object").extend(json!({
                "policy_logits":policy_logits,"policy_probabilities":policy_probabilities,"policy_action":policy_action,"policy_ce":policy_ce,
                "reward_logits":reward_logits,"reward_probabilities":reward_probabilities,"value_logit":value_logit,"value":value,
                "byte_offsets":offsets,"actions":actions
            }).as_object().expect("readouts object").clone());
            serde_json::to_writer(&mut rows_file, &row)?;
            rows_file.write_all(b"\n")?;
        }
    }
    probabilities_file.flush()?;
    states_file.flush()?;
    rows_file.flush()?;
    drop((probabilities_file, states_file, rows_file));
    ensure!(
        probability_path.metadata()?.len()
            == (layouts * 2 * cf::PROBABILITIES_PER_INPUT * 4) as u64
            && states_path.metadata()?.len() == (layouts * 2 * cf::STATES_PER_INPUT) as u64,
        "known-mapping binary output length mismatch"
    );
    let mut arms = Vec::new();
    for (condition, policy) in ["factual", "cleared"].into_iter().zip(&policies) {
        arms.push(json!({"condition":condition,"rows":layouts,"correct":policy.correct,
            "accuracy":policy.correct as f64/layouts as f64,"policy_ce":policy.ce/layouts as f64,
            "policy_controls":controls_from_counts(policy.labels)?,
            "value":{"mse":policy.value_squared_error/layouts as f64,"constant_value_prediction":1.0,"constant_value_mse":0.0,
                "qualifier":"every query is one step from the goal, so target value is one; low MSE does not establish value generalization"}}));
        for class in ["blocked", "nonterminal", "terminal"] {
            scores.entry(format!("{condition}/{class}")).or_default();
        }
    }
    let mut artifacts = Vec::new();
    for path in [&probability_path, &states_path, &rows_path] {
        artifacts.push(
            json!({"file":path.file_name().context("missing artifact name")?.to_string_lossy(),
                             "bytes":path.metadata()?.len(),"sha256":file_hash(path)?}),
        );
    }
    let mut report = json!({"schema":SCHEMA,"task_schema":task::SCHEMA,"known_mapping":population(),
        "status":"complete_pending_analysis","optimizer_updates":0,"data_seed":if args.known_seen {args.data_seed} else {EVAL_SEED},
        "episode_id_base":if args.known_seen {Value::Null} else {json!(EVAL_EPISODE_TAG)},
        "layouts":layouts,"input_rows":layouts*2,"action_rows":layouts*2*ACTIONS,
        "factual_input_rows":layouts,"factual_action_rows":layouts*ACTIONS,
        "cleared_input_rows":layouts,"cleared_action_rows":layouts*ACTIONS,
        "loops":args.loops,"physical_batch":1,"effective_batch":1,"policy":arms,"successor_strata":scores,
        "reward_threshold":0.5,"profiled_forward_input_index":args.profile_eval.then_some(0),
        "maximum_successor_probability_normalization_error":maximum_probability_error,
        "probability_normalization_tolerance":cf::PROBABILITY_TOLERANCE,
        "binary_layout":{"input_order":"layout_index ascending, factual then cleared",
            "probability_dtype":"little-endian IEEE-754 float32","probability_shape_per_input":[ACTIONS,PATCH_COUNT,PATCH_PIXELS,PALETTE],
            "state_dtype":"uint8","state_sections_per_input":["current","targets_by_action","predicted_by_action"],
            "state_section_lengths":[PIXELS,ACTIONS*PIXELS,ACTIONS*PIXELS],
            "pixel_order":"patch-major, then within-patch row-major; never frame-raster order",
            "state_hash_encoding":"SHA-256 of patch-order palette indices encoded as little-endian u32 (query_hash convention)",
            "argmax_ties":"first palette index","offset_unit":"bytes from start of named file"},
        "artifacts":artifacts,"elapsed_seconds":started.elapsed().as_secs_f64(),
        "claim_boundary":"single-seed known spatial prerequisite; factual/cleared scores have no 25% information bound; no promotion"});
    annotate_seen(args, &mut report);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args::parse_from([
            "probe",
            "--mode",
            "coverage-audit",
            "--output-dir",
            "unused",
            "--known-mapping",
        ])
    }

    fn seen_args() -> Args {
        Args::parse_from([
            "probe",
            "--mode",
            "known-mapping-audit",
            "--output-dir",
            "unused",
            "--known-mapping",
            "--known-seen",
            "--data-seed",
            "9173",
            "--eval-episodes",
            "72",
            "--batch",
            "1",
            "--effective-batch",
            "1",
        ])
    }

    #[test]
    fn seen_panel_matches_exact_training_inputs_targets_and_temporal_depth_counts() -> Result<()> {
        let mut input = seen_args();
        input.coverage = Coverage::Fresh;
        validate_args(&input)?;
        let expected: Vec<_> = (0..24).chain(2280..2304).chain(4576..4600).collect();
        let mut counts = [[0; 3]; 3];
        let mut queries = HashSet::new();
        for (layout, &episode) in expected.iter().enumerate() {
            let (factual, sample) = panel_input(&input, layout, false)?;
            let (cleared, other) = panel_input(&input, layout, true)?;
            let (training_episode, rule, trained) = coverage_sample(&input, episode * 16)?;
            let identity = training_row(episode * 16, training_episode, &trained);
            assert_eq!(seen_episode_id(layout), episode);
            assert_eq!(training_episode, episode);
            assert_eq!(rule, 0);
            assert_eq!(factual["episode_id"], episode);
            assert_eq!(factual["training_episode_index"], episode);
            assert_eq!(factual["episode_seed"], SEEN_SEED ^ TRAIN_TAG);
            assert_eq!(factual["original_training_update"], episode / 4 + 1);
            let depth = (episode / 4 % 3) as usize;
            assert_eq!(factual["original_training_loop_depth"], [1, 2, 4][depth]);
            assert_eq!(
                factual["temporal_block"],
                ["early", "middle", "late"][layout / 24]
            );
            assert_eq!(factual["evaluation_loops"], 4);
            assert_eq!(factual["input_index"], layout * 2);
            assert_eq!(cleared["input_index"], layout * 2 + 1);
            assert_eq!(factual["input_sha256"], identity["input_sha256"]);
            assert_eq!(factual["query_sha256"], identity["query_sha256"]);
            assert_eq!(factual["targets_sha256"], identity["targets_sha256"]);
            assert_eq!(cleared["targets_sha256"], identity["targets_sha256"]);
            assert_eq!(cleared["factual_input_sha256"], identity["input_sha256"]);
            assert_ne!(cleared["input_sha256"], factual["input_sha256"]);
            assert_eq!(sample.inputs.patches, trained.inputs.patches);
            assert_eq!(sample.inputs.metadata, trained.inputs.metadata);
            assert_eq!(sample.next, trained.next);
            assert_eq!(sample.rewards, trained.rewards);
            assert_eq!(sample.policy, trained.policy);
            assert_eq!(sample.value, trained.value);
            assert_eq!(other.current, sample.current);
            assert_eq!(other.next, sample.next);
            assert_eq!(other.rewards, sample.rewards);
            assert_eq!(other.policy, sample.policy);
            assert_eq!(other.value, sample.value);
            assert!(queries.insert(query_hash(&sample.current)));
            counts[layout / 24][depth] += 1;
        }
        assert_eq!(queries.len(), SEEN_LAYOUTS);
        assert_eq!(counts, [[8; 3]; 3]);
        assert!(panel_input(&input, SEEN_LAYOUTS, false).is_err());
        Ok(())
    }

    #[test]
    fn seen_flag_requires_the_exact_frozen_scope_and_population() -> Result<()> {
        for mode in [Mode::KnownMapping, Mode::KnownMappingAudit] {
            let mut input = seen_args();
            input.mode = mode;
            validate_args(&input)?;
        }
        for mode in [
            Mode::Smoke,
            Mode::Train,
            Mode::Evaluate,
            Mode::Inspect,
            Mode::Counterfactual,
            Mode::CounterfactualAudit,
            Mode::Fit,
            Mode::FitSmoke,
            Mode::Coverage,
            Mode::CoverageAudit,
        ] {
            let mut input = seen_args();
            input.mode = mode;
            assert!(validate_args(&input).is_err());
        }
        let invalid: [fn(&mut Args); 7] = [
            |a| a.known_mapping = false,
            |a| a.data_seed += 1,
            |a| a.eval_episodes = EVAL_LAYOUTS,
            |a| a.batch = 2,
            |a| a.effective_batch = 2,
            |a| a.loops = 2,
            |a| a.search = true,
        ];
        for change in invalid {
            let mut input = seen_args();
            change(&mut input);
            assert!(validate_args(&input).is_err());
        }
        let input = args();
        assert!(!input.known_seen);
        let mut original = json!({"known_mapping":population()});
        let bytes = serde_json::to_vec(&original)?;
        annotate_seen(&input, &mut original);
        assert_eq!(serde_json::to_vec(&original)?, bytes);
        Ok(())
    }

    #[test]
    fn seen_audit_reports_intended_membership_and_preserves_pair_identities() -> Result<()> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tofy-known-seen-audit-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&root)?;
        let result = (|| -> Result<()> {
            let mut input = seen_args();
            let mut training_queries = HashSet::new();
            for (name, checked) in [("standalone", false), ("membership", true)] {
                input.output_dir = root.join(name);
                fs::create_dir(&input.output_dir)?;
                let report = audit(&input, Instant::now(), &training_queries)?;
                assert_eq!(report["layouts"], 72);
                assert_eq!(report["input_rows"], 144);
                assert_eq!(report["action_rows"], 576);
                assert_eq!(report["factual_input_rows"], 72);
                assert_eq!(report["cleared_input_rows"], 72);
                assert_eq!(report["expected_training_query_members"], 72);
                assert_eq!(report["optimizer_updates"], 0);
                assert_eq!(report["model_forwards"], 0);
                let overlap = if checked { json!(72) } else { Value::Null };
                assert_eq!(report["evaluation_query_overlap"], overlap);
                assert_eq!(report["external_training_query_members"], overlap);
                assert!(report["external_factual_input_and_target_membership_verified"].is_null());
                assert_eq!(report["known_mapping"]["frozen_layouts"], 72);
                assert!(report["known_mapping"]["frozen_episode_id_base"].is_null());
                let path = input.output_dir.join("known-mapping-input-audit.jsonl");
                assert_eq!(report["artifacts"][0]["sha256"], file_hash(&path)?);
                let rows = fs::read_to_string(path)?;
                assert_eq!(rows.lines().count(), 144);
                for (index, line) in rows.lines().enumerate() {
                    let row: Value = serde_json::from_str(line)?;
                    assert_eq!(row, panel_input(&input, index / 2, index % 2 == 1)?.0);
                    training_queries.insert(row["query_sha256"].as_str().unwrap().to_owned());
                }
            }
            Ok(())
        })();
        fs::remove_dir_all(&root)?;
        result
    }

    #[test]
    fn known_stream_retains_sixteen_slots_and_exact_legacy_query_ids() -> Result<()> {
        let mut known = args();
        let mut legacy = args();
        legacy.known_mapping = false;
        for coverage in [Coverage::Fixed, Coverage::Fresh] {
            known.coverage = coverage;
            legacy.coverage = coverage;
            let mut previous = None;
            for id in (0..160).chain([73584, 73585, 73599]) {
                let (episode, rule, sample) = coverage_sample(&known, id)?;
                let (old_episode, _, old) = coverage_sample(&legacy, id)?;
                assert_eq!(episode, old_episode);
                assert_eq!(rule, 0);
                assert_eq!(sample.current, old.current);
                assert_eq!(
                    episode,
                    if coverage == Coverage::Fixed {
                        (id / 16) % 8
                    } else {
                        id / 16
                    }
                );
                if let Some((prior_episode, hash)) = previous {
                    if episode == prior_episode {
                        assert_eq!(input_hash(&sample.inputs), hash);
                    }
                }
                previous = Some((episode, input_hash(&sample.inputs)));
            }
        }
        Ok(())
    }

    #[test]
    fn known_reference_and_controls_report_actual_labels_without_false_blind_bound() -> Result<()> {
        let samples = fixed_samples(&args())?;
        assert_eq!(samples.len(), 128);
        assert!(samples.iter().all(|(rule, _)| *rule == 0));
        assert_eq!(
            samples
                .iter()
                .map(|(_, sample)| query_hash(&sample.current))
                .collect::<HashSet<_>>()
                .len(),
            8
        );
        for block in samples.chunks_exact(16) {
            assert!(block
                .iter()
                .all(|(_, s)| input_hash(&s.inputs) == input_hash(&block[0].1.inputs)));
        }
        let control = label_controls(samples.iter().map(|(_, sample)| sample))?;
        assert_eq!(control["rows"], 128);
        let repeated = label_controls(std::iter::repeat_n(&samples[0].1, 128))?;
        assert_eq!(repeated["best_constant_action_accuracy"], 1.0);
        assert_eq!(repeated["uniform_random_expected_accuracy"], 0.25);
        let mut metadata = json!({"training_rule_ids":task::permutation_ids(Split::Train),"held_out_rule_ids":task::permutation_ids(Split::HeldOut)});
        annotate(&mut metadata);
        assert_eq!(metadata["training_rule_ids"], json!([0]));
        assert_eq!(metadata["held_out_rule_ids"], json!([]));
        assert_eq!(
            metadata["known_mapping"]["control_population"],
            "KnownMapping"
        );
        assert_eq!(metadata["known_mapping"]["blind_bound_applies"], false);
        Ok(())
    }

    #[test]
    fn known_frozen_oracle_and_cleared_pair_preserve_all_targets_and_strata() -> Result<()> {
        let mut categories = HashSet::new();
        let mut queries = HashSet::new();
        for layout in 0..EVAL_LAYOUTS {
            let (factual, sample) = panel_input(&args(), layout, false)?;
            let (cleared, other) = panel_input(&args(), layout, true)?;
            assert_eq!(factual["split"], "KnownMapping");
            assert_eq!(factual["episode_id"], EVAL_EPISODE_TAG + layout as u64);
            assert_eq!(factual["query_sha256"], cleared["query_sha256"]);
            assert_ne!(factual["input_sha256"], cleared["input_sha256"]);
            assert_eq!(sample.current, other.current);
            assert_eq!(sample.next, other.next);
            assert_eq!(sample.policy, other.policy);
            assert_eq!(sample.value, 1.0);
            assert!(queries.insert(query_hash(&sample.current)));
            assert!(
                other.inputs.patches[..(TOKENS - PATCH_COUNT) * PATCH_PIXELS]
                    .iter()
                    .all(|p| *p == 0)
            );
            assert_eq!(
                &sample.inputs.patches[(TOKENS - PATCH_COUNT) * PATCH_PIXELS..],
                &other.inputs.patches[(TOKENS - PATCH_COUNT) * PATCH_PIXELS..]
            );
            for action in 0..ACTIONS {
                let target = &sample.next[action * PIXELS..(action + 1) * PIXELS];
                let metric = cf::metrics(&sample.current, target, target, sample.rewards[action])?;
                categories.insert(metric.class);
                assert!(metric.exact);
                assert_eq!(metric.changed.correct, metric.changed.total);
                assert_eq!(
                    metric.changed.total,
                    metric.vacated.total + metric.destination.total
                );
                assert_eq!(metric.copy_exact, metric.class == "blocked");
                assert_eq!(sample.policy[action], sample.rewards[action]);
            }
        }
        assert_eq!(
            categories,
            HashSet::from(["blocked", "nonterminal", "terminal"])
        );
        Ok(())
    }

    #[test]
    fn known_flag_rejects_ambiguous_modes_and_defaults_leave_legacy_untouched() -> Result<()> {
        let mut input = args();
        validate_args(&input)?;
        for mode in [
            Mode::Smoke,
            Mode::Train,
            Mode::Evaluate,
            Mode::Inspect,
            Mode::Counterfactual,
            Mode::CounterfactualAudit,
            Mode::Fit,
        ] {
            input.mode = mode;
            assert!(validate_args(&input).is_err());
        }
        input.known_mapping = false;
        validate_args(&input)?;
        input.mode = Mode::KnownMapping;
        assert!(validate_args(&input).is_err());
        input.known_mapping = true;
        input.data_seed = EVAL_SEED;
        input.eval_episodes = EVAL_LAYOUTS;
        input.batch = 1;
        input.effective_batch = 1;
        validate_args(&input)?;
        input.data_seed += 1;
        assert!(validate_args(&input).is_err());
        Ok(())
    }

    #[test]
    fn known_coverage_audit_logs_actual_counts_and_exact_frozen_identity() -> Result<()> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("tofy-known-audit-{}-{unique}", std::process::id()));
        fs::create_dir(&root)?;
        let result = (|| -> Result<()> {
            let mut input = args();
            input.output_dir = root.clone();
            input.coverage = Coverage::Fresh;
            input.updates = 9;
            input.effective_batch = 16;
            let report = coverage_audit(&input, Instant::now())?;
            assert_eq!(report["rows"], 144);
            assert_eq!(report["episode_ids"], 9);
            assert_eq!(report["unique_inputs"], 9);
            assert_eq!(report["repeated_complete_input_rows"], 135);
            assert_eq!(report["rule_counts"][0], 144);
            assert_eq!(report["frozen_panel"]["input_rows"], 128);
            assert_eq!(report["frozen_panel"]["evaluation_query_overlap"], 0);
            let rows = fs::read_to_string(root.join("training-stream.jsonl"))?;
            for (id, line) in rows.lines().enumerate() {
                let row: Value = serde_json::from_str(line)?;
                assert_eq!(row["id"], id);
                assert_eq!(row["episode_index"], id / 16);
                assert_eq!(row["rule"], 0);
                assert_eq!(row["control_population"], "KnownMapping");
            }
            let rows = fs::read_to_string(root.join("known-mapping-input-audit.jsonl"))?;
            for (index, line) in rows.lines().enumerate() {
                let row: Value = serde_json::from_str(line)?;
                assert_eq!(row, panel_input(&input, index / 2, index % 2 == 1)?.0);
            }
            Ok(())
        })();
        fs::remove_dir_all(&root)?;
        result
    }

    #[test]
    fn known_region_reward_aggregates_distinguish_oracle_and_copy() -> Result<()> {
        let (_, sample) = panel_input(&args(), 0, false)?;
        let mut oracle = Scores::default();
        let mut copy = Scores::default();
        for action in 0..ACTIONS {
            let target = &sample.next[action * PIXELS..(action + 1) * PIXELS];
            let truth = sample.rewards[action];
            oracle.add(
                &cf::metrics(&sample.current, target, target, truth)?,
                truth,
                truth,
            );
            copy.add(
                &cf::metrics(&sample.current, target, &sample.current, truth)?,
                truth,
                0.0,
            );
        }
        assert_eq!(oracle.rows, 4);
        assert_eq!(oracle.exact, 4);
        assert_eq!(oracle.reward_positive, 1);
        assert_eq!(oracle.reward_true_positive, 1);
        assert_eq!(oracle.reward_squared_error_sum, 0.0);
        assert_eq!(oracle.changed.correct, oracle.changed.total);
        assert!(oracle.changed.total >= 128);
        assert_eq!(copy.changed.correct, 0);
        assert_eq!(copy.exact, copy.copy_exact);
        assert_eq!(copy.reward_true_positive, 0);
        assert_eq!(copy.reward_squared_error_sum, 1.0);
        Ok(())
    }
}
