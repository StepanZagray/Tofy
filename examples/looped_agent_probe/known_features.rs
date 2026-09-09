//! Frozen head-input export. No fitting, optimizer, or altered model operations.
use super::*;
use std::io::BufRead;
use tofy::p2::looped_agent::model::LoopedFeatures;

const SCHEMA: &str = "looped-known-features-v1";
const DATA_SEED: u64 = 20260915;
const EPISODE_TAG: u64 = 0x46454154555245;
const LAYOUTS: usize = 768;
const FIT_LAYOUTS: usize = 512;
const WIDTH: usize = 128;
const FRESH_SEED: u64 = 20260916;
const FRESH_TAG: u64 = 0x524541444f5554;
const FRESH_LAYOUTS: usize = 256;

fn panel(heldout: bool) -> (u64, u64, usize, usize) {
    if heldout {
        (FRESH_SEED, FRESH_TAG, FRESH_LAYOUTS, 0)
    } else {
        (DATA_SEED, EPISODE_TAG, LAYOUTS, FIT_LAYOUTS)
    }
}
const INITIAL: &str = "4cd502f7a76fe3dd9729f6693e9d707c670972f7ebe58ad1cbdde37ea2bb2802";
const FINAL: &str = "a983626a888279cbb31651c6abcd1ebf4f808c8927a29a8349d03ffc35bd4e5a";
const AUDIT_FILE: &str = "known-features-input-audit.jsonl";
const ROWS_FILE: &str = "known-features-rows.jsonl";
const ARRAYS: [(&str, &str, usize); 3] = [
    ("cls", "known-features-cls.f32", WIDTH),
    ("current", "known-features-current.f32", PATCH_COUNT * WIDTH),
    ("policy", "known-features-policy.f32", ACTIONS),
];

pub(super) fn extracts(mode: Mode) -> bool {
    matches!(mode, Mode::KnownFeatures | Mode::KnownFeaturesSmoke)
}

fn dedicated(mode: Mode) -> bool {
    extracts(mode) || mode == Mode::KnownFeaturesAudit
}

pub(super) fn validate_args(args: &Args) -> Result<()> {
    ensure!(
        !args.known_features_heldout
            || matches!(args.mode, Mode::KnownFeatures | Mode::KnownFeaturesAudit),
        "--known-features-heldout requires full known-features or known-features-audit mode"
    );
    let (data_seed, _, layouts, _) = panel(args.known_features_heldout);
    ensure!(
        dedicated(args.mode) || args.known_features_exclude.is_empty(),
        "--known-features-exclude is scoped to known-features modes"
    );
    if !dedicated(args.mode) {
        return Ok(());
    }
    ensure!(
        args.known_mapping && !args.known_seen && !args.known_replay && !args.search,
        "known-features requires --known-mapping, factual support, no --known-seen, --known-replay or --search"
    );
    ensure!(
        args.seed == 0 && args.data_seed == data_seed && args.eval_episodes == layouts
            && args.loops == 4 && args.batch == 1 && args.effective_batch == 1,
        "known-features requires registered seed/layouts (20260915/768 or heldout 20260916/256), seed 0, loops 4, physical/effective batch 1"
    );
    ensure!(
        args.hidden == WIDTH && args.heads == 4 && args.layers == 2 && args.max_loops == 8,
        "known-features requires the registered 128-wide, four-head, two-layer, max-loops-eight model"
    );
    ensure!(
        extracts(args.mode) == args.checkpoint.is_some(),
        "known-features extraction requires a checkpoint; its audit forbids one"
    );
    ensure!(
        !extracts(args.mode) || args.profile_eval,
        "known-features extraction requires first-forward profiling"
    );
    ensure!(
        args.mode == Mode::KnownFeaturesSmoke || args.known_features_exclude.len() == if args.known_features_heldout { 5 } else { 4 },
        "known-features full/audit modes require four exclusion JSONLs; heldout additionally requires all C8 queries as a fifth exclusion"
    );
    ensure!(
        args.known_features_exclude.iter().all(|p| p.is_absolute()),
        "known-features exclusion paths must be absolute"
    );
    ensure!(
        args.known_features_exclude
            .iter()
            .collect::<HashSet<_>>()
            .len()
            == args.known_features_exclude.len(),
        "known-features exclusion paths must be distinct"
    );
    Ok(())
}

fn row_count(args: &Args) -> usize {
    if args.mode == Mode::KnownFeaturesSmoke {
        1
    } else {
        panel(args.known_features_heldout).2
    }
}

pub(super) fn annotate(args: &Args, document: &mut Value) {
    if !dedicated(args.mode) {
        return;
    }
    let (data_seed, tag, layouts, fit_layouts) = panel(args.known_features_heldout);
    document["known_features"] = json!({
        "schema":SCHEMA,"data_seed":data_seed,"episode_id_base":tag,
        "layouts":layouts,"fit_layouts":fit_layouts,"eval_layouts":layouts-fit_layouts,
        "loops":4,"condition":"factual","optimizer_updates":0,
        "implementation_smoke":args.mode==Mode::KnownFeaturesSmoke,
        "smoke_boundary":"first fitting row only; excluded from evidence and probe selection",
        "feature_seam":"exact post-final-RMS CLS and current-patch tensors consumed by ordinary heads"
    });
    document["known_mapping"]["frozen_data_seed"] = json!(data_seed);
    document["known_mapping"]["frozen_episode_id_base"] = json!(tag);
    document["known_mapping"]["frozen_layouts"] = json!(layouts);
    if args.known_features_heldout {
        document["known_features_heldout"] = json!(true);
    }
}

pub(super) fn check_checkpoint(args: &Args) -> Result<String> {
    let digest = file_hash(
        args.checkpoint
            .as_ref()
            .context("missing feature checkpoint")?,
    )?;
    ensure!(
        digest == INITIAL || digest == FINAL,
        "unregistered feature checkpoint"
    );
    Ok(digest)
}

fn cells(pixels: &[u32]) -> Result<Vec<u8>> {
    ensure!(pixels.len() == PIXELS, "wrong visible state length");
    pixels
        .chunks_exact(PATCH_PIXELS)
        .map(|patch| {
            ensure!(
                patch[0] <= 4 && patch.iter().all(|&c| c == patch[0]),
                "feature panel requires uniform toy patches with colors zero through four"
            );
            Ok(patch[0] as u8)
        })
        .collect()
}

#[cfg(test)]
fn panel_input(index: usize) -> Result<(Value, Sample)> {
    panel_input_for(false, index)
}

fn panel_input_for(heldout: bool, index: usize) -> Result<(Value, Sample)> {
    let (data_seed, tag, layouts, fit_layouts) = panel(heldout);
    ensure!(index < layouts, "feature layout out of range");
    let episode_id = tag + index as u64;
    let episode = task::episode_with_permutation(data_seed, episode_id, 0, 1, 1)?;
    let sample = task::sample(&episode)?;
    let controls = task::inferred_controls(&episode.support)?;
    let (policy, distance) = task::oracle(&episode.support, &episode.maze.render())?;
    let visible = cells(&sample.current)?;
    let positions = |color| {
        visible
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| (c == color).then_some(i))
            .collect::<Vec<_>>()
    };
    let agents = positions(2);
    let goals = positions(3);
    ensure!(
        agents.len() == 1 && goals.len() == 1,
        "expected unique visible agent and goal"
    );
    let dx = (goals[0] % 8) as i32 - (agents[0] % 8) as i32;
    let dy = (goals[0] / 8) as i32 - (agents[0] / 8) as i32;
    let label = argmax(&sample.policy);
    let geometry = [(-dy) as f32, dy as f32, (-dx) as f32, dx as f32];
    ensure!(
        dx.abs() + dy.abs() == 1 && argmax(&geometry) == label,
        "visible geometry and source-owned policy disagree"
    );
    ensure!(
        controls == [0, 1, 2, 3]
            && distance == 1
            && policy == sample.policy
            && sample
                .policy
                .iter()
                .enumerate()
                .all(|(i, &p)| p == f32::from(i == label))
            && sample.value == 1.0
            && sample.rewards == sample.policy,
        "known-feature control/policy/reward/value target mismatch"
    );
    let target_cells = sample
        .next
        .chunks_exact(PIXELS)
        .map(cells)
        .collect::<Result<Vec<_>>>()?;
    ensure!(target_cells.len() == ACTIONS, "wrong successor population");
    let label_sha256 = format!("{:x}", Sha256::digest((label as u32).to_le_bytes()));
    let observed: Vec<_> = episode.support.iter().map(|step| step.action).collect();
    let row = json!({
        "schema":SCHEMA,"input_index":index,"layout_index":index,
        "partition":if heldout {"fresh_eval"} else if index<fit_layouts {"fit"} else {"eval"},
        "episode_id":episode_id,"episode_seed":data_seed,"data_seed":data_seed,
        "permutation_id":0,"split":"KnownMapping","condition":"factual","support_cleared":false,
        "min_distance":1,"max_distance":1,"oracle_distance":distance,"evaluation_loops":4,
        "observed_support_action_ids":observed,"inferred_controls":controls,
        "input_sha256":input_hash(&sample.inputs),"query_sha256":query_hash(&sample.current),
        "targets_sha256":coverage_row(index as u64,episode_id,0,&sample)["targets_sha256"],
        "label_sha256":label_sha256,"correct_action":label,"visible_cells":visible,"target_cells":target_cells,
        "target_policy":sample.policy,"target_rewards":sample.rewards,"target_value":sample.value
    });
    Ok((row, sample))
}

fn excluded_queries(args: &Args) -> Result<(HashSet<String>, Vec<Value>)> {
    let mut queries = HashSet::new();
    let mut artifacts = Vec::new();
    for path in &args.known_features_exclude {
        ensure!(
            path.is_file(),
            "missing exclusion JSONL: {}",
            path.display()
        );
        let digest = file_hash(path)?;
        let mut rows = 0;
        for line in std::io::BufReader::new(File::open(path)?).lines() {
            let row: Value = serde_json::from_str(&line?)?;
            let query = row["query_sha256"]
                .as_str()
                .context("exclusion row missing query hash")?;
            ensure!(
                query.len() == 64
                    && query
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
                "invalid exclusion query digest"
            );
            queries.insert(query.to_owned());
            rows += 1;
        }
        ensure!(
            rows > 0 && file_hash(path)? == digest,
            "empty or modified exclusion population"
        );
        artifacts
            .push(json!({"path":path,"sha256":digest,"rows":rows,"bytes":path.metadata()?.len()}));
    }
    Ok((queries, artifacts))
}

fn artifact(args: &Args, name: &str) -> Result<Value> {
    let path = args.output_dir.join(name);
    Ok(json!({"file":name,"bytes":path.metadata()?.len(),"sha256":file_hash(&path)?}))
}

pub(super) fn audit(args: &Args, started: Instant) -> Result<Value> {
    validate_args(args)?;
    let (data_seed, tag, _, fit_layouts) = panel(args.known_features_heldout);
    let (excluded, exclusions) = excluded_queries(args)?;
    let mut writer = BufWriter::new(File::create_new(args.output_dir.join(AUDIT_FILE))?);
    let mut queries = HashSet::new();
    let mut inputs = HashSet::new();
    let mut labels = [[0usize; ACTIONS]; 2];
    for index in 0..row_count(args) {
        deadline(args, started)?;
        let (row, _) = panel_input_for(args.known_features_heldout, index)?;
        let query = row["query_sha256"].as_str().context("query digest")?;
        ensure!(
            !excluded.contains(query),
            "registered feature query overlaps excluded population at row {index}"
        );
        ensure!(
            queries.insert(query.to_owned()),
            "duplicate feature query at row {index}"
        );
        ensure!(
            inputs.insert(
                row["input_sha256"]
                    .as_str()
                    .context("input digest")?
                    .to_owned()
            ),
            "duplicate feature input at row {index}"
        );
        labels[usize::from(index >= fit_layouts)]
            [row["correct_action"].as_u64().context("label")? as usize] += 1;
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    drop(writer);
    if args.mode != Mode::KnownFeaturesSmoke {
        ensure!(
            labels[usize::from(args.known_features_heldout)..]
                .iter()
                .flatten()
                .all(|&n| n > 0),
            "both feature partitions must contain all four labels"
        );
    }
    Ok(
        json!({"schema":SCHEMA,"task_schema":task::SCHEMA,"status":"complete_pending_analysis",
        "evidence_class":if args.mode==Mode::KnownFeaturesSmoke {"implementation_smoke"} else {"data_audit"},
        "optimizer_updates":0,"model_forwards":0,"layouts":row_count(args),"input_rows":row_count(args),
        "data_seed":data_seed,"episode_id_base":tag,"fit_rows":labels[0].iter().sum::<usize>(),
        "eval_rows":labels[1].iter().sum::<usize>(),"label_counts":{"fit":labels[0],"eval":labels[1]},
        "unique_queries":queries.len(),"unique_inputs":inputs.len(),"exclusions":exclusions,
        "excluded_unique_queries":excluded.len(),"query_overlap":0,
        "exclusion_scope":if args.known_features_heldout {"supplied populations; supervising analyzer must bind five registered source artifacts including all C8 queries"} else {"supplied populations; supervising analyzer must bind the four registered source artifacts"},
        "hash_layout":{"input":"patches U32 LE then metadata F32 LE","query":"4096 patch-major U32 LE pixels",
            "targets":"4*4096 U32 LE successor pixels then policy4/rewards4/value1 F32 LE","label":"correct_action U32 LE"},
        "artifacts":[artifact(args,AUDIT_FILE)?],"elapsed_seconds":started.elapsed().as_secs_f64()}),
    )
}

fn feature_forward(
    args: &Args,
    model: &LoopedAgent,
    input: Inputs,
    device: &Device,
    first: bool,
) -> Result<(LoopedOutput, LoopedFeatures)> {
    if !first {
        let (pixels, metadata) = tensors(&[input], device)?;
        return model.forward_with_features(&pixels, &metadata, args.loops);
    }
    let capture = LoopedCapture::begin_eval(
        &args
            .output_dir
            .with_extension("profiles")
            .join("evaluation-000001"),
        1,
        device,
        1,
        1,
        args.loops,
    )?;
    device.synchronize()?;
    let measured = capture.measurement();
    let phase = capture.phase("forward", Some(candle_graph::ExecutionStep::Forward));
    // Synchronize before dropping the measurement even if a forward/probe fails.
    let result: Result<(LoopedOutput, LoopedFeatures)> = (|| {
        let (pixels, metadata) = tensors(&[input], device)?;
        let (output, features) = model.forward_with_features(&pixels, &metadata, args.loops)?;
        for (name, tensor) in [
            ("readout/policy_logits", &output.policy_logits),
            ("readout/value", &output.value),
            ("readout/reward_logits", &output.reward_logits),
            ("readout/next_logits", &output.next_logits),
            ("features/cls", &features.cls),
            ("features/current", &features.current),
        ] {
            capture.record_tensor_stats(&phase, name, tensor)?;
        }
        Ok((output, features))
    })();
    let synchronized = device.synchronize();
    drop(phase);
    drop(measured);
    synchronized?;
    let result = result?;
    capture.finish()?;
    Ok(result)
}

fn finite_values(tensor: &Tensor, shape: &[usize]) -> Result<Vec<f32>> {
    ensure!(
        tensor.dtype() == DType::F32 && tensor.dims() == shape,
        "wrong feature/output dtype or shape"
    );
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "nonfinite feature or ordinary output"
    );
    Ok(values)
}

fn array_layout(index: usize) -> Value {
    let mut arrays = serde_json::Map::new();
    for (key, file, n) in ARRAYS {
        arrays.insert(
            key.into(),
            json!({"file":file,"dtype":"F32LE","byte_offset":index*n*4,
            "byte_length":n*4,"shape":if key=="current" {vec![PATCH_COUNT,WIDTH]} else {vec![n]}}),
        );
    }
    Value::Object(arrays)
}

fn write_values(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "cannot serialize nonfinite features"
    );
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

pub(super) fn extract(
    args: &Args,
    vars: &VarMap,
    config: &LoopedConfig,
    device: &Device,
    parameters: usize,
    audit: Value,
    started: Instant,
) -> Result<Value> {
    let (data_seed, tag, _, _) = panel(args.known_features_heldout);
    ensure!(parameters == 992393, "feature parameter count changed");
    let checkpoint = check_checkpoint(args)?;
    vars.save(args.output_dir.join("initial.safetensors"))?;
    ensure!(
        file_hash(&args.output_dir.join("initial.safetensors"))? == checkpoint,
        "loaded feature parameters differ from exact checkpoint bytes"
    );
    let model = frozen(vars, config, device)?;
    File::create_new(args.output_dir.join("updates.jsonl"))?;
    let mut writers = ARRAYS
        .iter()
        .map(|(_, file, _)| {
            Ok(BufWriter::new(File::create_new(
                args.output_dir.join(file),
            )?))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut rows = BufWriter::new(File::create_new(args.output_dir.join(ROWS_FILE))?);
    let mut identities =
        std::io::BufReader::new(File::open(args.output_dir.join(AUDIT_FILE))?).lines();
    for index in 0..row_count(args) {
        deadline(args, started)?;
        let (mut row, sample) = panel_input_for(args.known_features_heldout, index)?;
        let audited: Value =
            serde_json::from_str(&identities.next().context("missing feature audit row")??)?;
        ensure!(
            row == audited,
            "feature extraction input differs from audited identity"
        );
        let (output, features) = feature_forward(args, &model, sample.inputs, device, index == 0)?;
        let values = [
            finite_values(&features.cls, &[1, WIDTH])?,
            finite_values(&features.current, &[1, PATCH_COUNT, WIDTH])?,
            finite_values(&output.policy_logits, &[1, ACTIONS])?,
        ];
        finite_values(&output.value, &[1, 1])?;
        finite_values(&output.reward_logits, &[1, ACTIONS])?;
        finite_values(
            &output.next_logits,
            &[1, ACTIONS, PATCH_COUNT, PATCH_PIXELS, PALETTE],
        )?;
        for (writer, values) in writers.iter_mut().zip(values) {
            write_values(writer, &values)?;
        }
        row["arrays"] = array_layout(index);
        serde_json::to_writer(&mut rows, &row)?;
        rows.write_all(b"\n")?;
    }
    ensure!(identities.next().is_none(), "extra feature audit row");
    rows.flush()?;
    for writer in &mut writers {
        writer.flush()?;
    }
    drop(rows);
    drop(writers);
    for (_, file, n) in ARRAYS {
        ensure!(
            args.output_dir.join(file).metadata()?.len() == (row_count(args) * n * 4) as u64,
            "wrong feature output byte length"
        );
    }
    vars.save(args.output_dir.join("final.safetensors"))?;
    ensure!(
        file_hash(&args.output_dir.join("final.safetensors"))? == checkpoint
            && check_checkpoint(args)? == checkpoint,
        "frozen feature checkpoint changed"
    );
    let mut artifacts = vec![artifact(args, AUDIT_FILE)?, artifact(args, ROWS_FILE)?];
    for (_, file, _) in ARRAYS {
        artifacts.push(artifact(args, file)?);
    }
    Ok(json!({"schema":SCHEMA,"status":"complete_pending_analysis",
        "evidence_class":if args.mode==Mode::KnownFeaturesSmoke {"implementation_smoke"} else {"frozen_feature_diagnostic"},
        "included_in_evidence":args.mode!=Mode::KnownFeaturesSmoke,"optimizer_updates":0,"model_forwards":row_count(args),
        "ordinary_heads_per_forward":{"policy":1,"value":1,"reward":1,"successor":ACTIONS},
        "parameters":parameters,"physical_batch":1,"effective_batch":1,"accumulation":1,"loops":args.loops,
        "layouts":row_count(args),"input_rows":row_count(args),"fit_rows":audit["fit_rows"],"eval_rows":audit["eval_rows"],
        "data_seed":data_seed,"episode_id_base":tag,"feature_width":WIDTH,
        "checkpoint_sha256":checkpoint,"initial_checkpoint_sha256":checkpoint,"final_checkpoint_sha256":checkpoint,
        "feature_layout":"separate input-major F32 little-endian CLS[N,128], current[N,64,128], policy[N,4]; current patches row-major",
        "first_forward_profile":"evaluation-000001","audit":audit,"artifacts":artifacts,
        "elapsed_seconds":started.elapsed().as_secs_f64(),
        "claim_boundary":"frozen synthetic features for preregistered CPU probes; no model training, ARC claim or promotion"}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(mode: &str) -> Args {
        let mut args = Args::parse_from([
            "probe",
            "--mode",
            mode,
            "--known-mapping",
            "--output-dir",
            "unused",
            "--seed",
            "0",
            "--data-seed",
            "20260915",
            "--eval-episodes",
            "768",
            "--loops",
            "4",
            "--batch",
            "1",
            "--effective-batch",
            "1",
        ]);
        args.known_features_exclude = (0..4)
            .map(|i| PathBuf::from(format!("/excluded-{i}.jsonl")))
            .collect();
        if extracts(args.mode) {
            args.checkpoint = Some(PathBuf::from("unused.safetensors"));
        }
        args
    }

    #[test]
    fn feature_modes_fail_closed_without_affecting_legacy_flags() -> Result<()> {
        for mode in [
            "known-features",
            "known-features-audit",
            "known-features-smoke",
        ] {
            known_mapping::validate_args(&args(mode))?;
            let mut bad = args(mode);
            bad.known_mapping = false;
            assert!(known_mapping::validate_args(&bad).is_err());
            let mut bad = args(mode);
            bad.known_seen = true;
            assert!(known_mapping::validate_args(&bad).is_err());
            let mut bad = args(mode);
            bad.known_replay = true;
            assert!(known_mapping::validate_args(&bad).is_err());
            let mut bad = args(mode);
            bad.loops = 2;
            assert!(validate_args(&bad).is_err());
            let mut bad = args(mode);
            bad.data_seed = 20260912;
            assert!(validate_args(&bad).is_err());
            let mut bad = args(mode);
            bad.eval_episodes = 1;
            assert!(validate_args(&bad).is_err());
            let mut bad = args(mode);
            bad.seed = 1;
            assert!(validate_args(&bad).is_err());
            let mut bad = args(mode);
            bad.effective_batch = 2;
            assert!(validate_args(&bad).is_err());
            let mut bad = args(mode);
            bad.batch = 2;
            assert!(validate_args(&bad).is_err());
            let mut bad = args(mode);
            bad.hidden = 16;
            assert!(validate_args(&bad).is_err());
            let mut bad = args(mode);
            bad.known_features_exclude[0] = PathBuf::from("relative.jsonl");
            assert!(validate_args(&bad).is_err());
            let mut bad = args(mode);
            bad.checkpoint = if bad.checkpoint.is_some() {
                None
            } else {
                Some(PathBuf::from("unused"))
            };
            assert!(validate_args(&bad).is_err());
        }
        let mut bad = args("known-features");
        bad.profile_eval = false;
        assert!(validate_args(&bad).is_err());
        let mut smoke = args("known-features-smoke");
        smoke.known_features_exclude.clear();
        validate_args(&smoke)?;
        assert_eq!(row_count(&smoke), 1);
        let mut full = args("known-features");
        full.known_features_exclude.clear();
        assert!(validate_args(&full).is_err());
        let legacy = Args::parse_from(["probe", "--mode", "evaluate", "--output-dir", "unused"]);
        validate_args(&legacy)?;
        let mut bad = legacy;
        bad.known_features_exclude
            .push(PathBuf::from("/excluded.jsonl"));
        assert!(validate_args(&bad).is_err());
        Ok(())
    }

    #[test]
    fn feature_panel_has_exact_partitions_and_visible_simulator_targets() -> Result<()> {
        let mut queries = HashSet::new();
        let mut labels = [[0; ACTIONS]; 2];
        for index in 0..LAYOUTS {
            let (row, sample) = panel_input(index)?;
            assert_eq!(row["episode_id"], EPISODE_TAG + index as u64);
            assert_eq!(
                row["partition"],
                if index < FIT_LAYOUTS { "fit" } else { "eval" }
            );
            assert!(queries.insert(row["query_sha256"].as_str().unwrap().to_owned()));
            labels[usize::from(index >= FIT_LAYOUTS)][argmax(&sample.policy)] += 1;
            let visible = cells(&sample.current)?;
            let agent = visible.iter().position(|&c| c == 2).unwrap();
            let goal = visible.iter().position(|&c| c == 3).unwrap();
            for (action, (dx, dy)) in [(0, -1), (0, 1), (-1, 0), (1, 0)].into_iter().enumerate() {
                let (x, y) = ((agent % 8) as i32 + dx, (agent / 8) as i32 + dy);
                let mut expected = visible.clone();
                let mut reward = 0.0;
                if (0..8).contains(&x) && (0..8).contains(&y) {
                    let destination = (y * 8 + x) as usize;
                    if visible[destination] != 1 {
                        expected[agent] = 0;
                        expected[destination] = if destination == goal {
                            reward = 1.0;
                            4
                        } else {
                            2
                        };
                    }
                }
                assert_eq!(
                    cells(&sample.next[action * PIXELS..(action + 1) * PIXELS])?,
                    expected
                );
                assert_eq!(sample.rewards[action], reward);
            }
        }
        assert_eq!(queries.len(), 768);
        assert_eq!(labels[0].iter().sum::<usize>(), 512);
        assert_eq!(labels[1].iter().sum::<usize>(), 256);
        assert!(labels.iter().flatten().all(|&n| n > 0));
        assert!(panel_input(LAYOUTS).is_err());
        Ok(())
    }

    #[test]
    fn feature_binary_layout_roundtrips_and_rejects_nonfinite_or_wrong_shape() -> Result<()> {
        for (key, _, n) in ARRAYS {
            let mut bytes = Vec::new();
            for index in 0..2 {
                let values = (0..n)
                    .map(|j| index as f32 + j as f32 / 16384.0 - 0.5)
                    .collect::<Vec<_>>();
                write_values(&mut bytes, &values)?;
                let layout = array_layout(index);
                let offset = layout[key]["byte_offset"].as_u64().unwrap() as usize;
                let length = layout[key]["byte_length"].as_u64().unwrap() as usize;
                assert_eq!(offset, index * n * 4);
                assert_eq!(length, n * 4);
                let decoded = bytes[offset..offset + length]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(decoded, values);
            }
            assert_eq!(bytes.len(), 2 * n * 4);
        }
        let mut bytes = Vec::new();
        assert!(write_values(&mut bytes, &[f32::NAN]).is_err());
        assert!(write_values(&mut bytes, &[f32::INFINITY]).is_err());
        assert!(bytes.is_empty());
        let tensor = Tensor::zeros((1, 4), DType::F32, &Device::Cpu)?;
        assert!(finite_values(&tensor, &[4]).is_err());
        let tensor = Tensor::new(&[f32::NAN], &Device::Cpu)?;
        assert!(finite_values(&tensor, &[1]).is_err());
        Ok(())
    }

    #[test]
    fn feature_audit_has_no_model_artifacts_and_rejects_overlap() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-feature-audit-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        fs::create_dir(&root)?;
        let result = (|| -> Result<()> {
            let mut args = args("known-features-audit");
            args.output_dir = root.join("audit");
            fs::create_dir(&args.output_dir)?;
            args.known_features_exclude = (0..4)
                .map(|i| root.join(format!("excluded-{i}.jsonl")))
                .collect();
            for path in &args.known_features_exclude {
                let line = serde_json::to_string(&json!({"query_sha256":"a".repeat(64)}))?;
                fs::write(path, format!("{line}\n{line}\n"))?;
            }
            let report = audit(&args, Instant::now())?;
            assert_eq!(report["model_forwards"], 0);
            assert_eq!(report["optimizer_updates"], 0);
            assert_eq!(report["input_rows"], 768);
            assert_eq!(report["excluded_unique_queries"], 1);
            assert_eq!(
                report["artifacts"][0]["sha256"],
                "09559eb4dd1b933de724da81aef3bb80547e010a00e38e39de31c9f72fab1c6d",
                "C8 identity JSONL must remain byte-identical"
            );
            assert_eq!(fs::read_dir(&args.output_dir)?.count(), 1);
            assert!(
                audit(&args, Instant::now()).is_err(),
                "cannot overwrite audit"
            );
            let (first, _) = panel_input(0)?;
            fs::write(
                &args.known_features_exclude[0],
                format!("{}\n", serde_json::to_string(&first)?),
            )?;
            args.output_dir = root.join("collision");
            fs::create_dir(&args.output_dir)?;
            assert!(audit(&args, Instant::now()).is_err());
            Ok(())
        })();
        fs::remove_dir_all(&root)?;
        result
    }

    #[test]
    fn heldout_configuration_is_scoped_without_generating_the_future_panel() -> Result<()> {
        let mut fresh = args("known-features-audit");
        fresh.known_features_heldout = true;
        assert!(validate_args(&fresh).is_err());
        fresh.data_seed = FRESH_SEED;
        fresh.eval_episodes = FRESH_LAYOUTS;
        assert!(validate_args(&fresh).is_err());
        fresh
            .known_features_exclude
            .push(PathBuf::from("/all-c8-queries.jsonl"));
        validate_args(&fresh)?;
        assert_eq!(row_count(&fresh), 256);
        assert_eq!(panel(true), (20260916, 0x524541444f5554, 256, 0));
        let mut metadata = json!({});
        annotate(&fresh, &mut metadata);
        assert_eq!(metadata["known_features"]["fit_layouts"], 0);
        assert_eq!(metadata["known_features"]["eval_layouts"], 256);
        assert_eq!(metadata["known_features_heldout"], true);
        fresh.mode = Mode::KnownFeaturesSmoke;
        fresh.checkpoint = Some(PathBuf::from("unused"));
        assert!(validate_args(&fresh).is_err());
        // Preregistration forbids calling the future generator until all six heads seal.
        Ok(())
    }
}
