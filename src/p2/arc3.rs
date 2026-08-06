//! Official ARC-AGI toolkit recording JSONL import.
//!
//! Each line wraps `{timestamp, data}` where `data` matches the toolkit
//! `_set_last_response` recording payload. Consecutive events form transitions
//! where the later event's `action_input` caused its settled frame.

use crate::domain::Split;
use crate::p2::data::{ArcAction, ArcFrame, GoalFeatures, TransitionSample, FRAME_SIDE};
use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// One deserialized recording event (toolkit JSONL line payload).
#[derive(Clone, Debug)]
pub struct RecordingEvent {
    pub timestamp: String,
    pub game_id: String,
    pub state: String,
    pub levels_completed: i64,
    pub win_levels: i64,
    pub action: Option<ParsedActionInput>,
    pub guid: String,
    pub full_reset: bool,
    pub available_actions: Vec<i64>,
    pub frame: ArcFrame,
    pub source_path: PathBuf,
    pub line: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedActionInput {
    pub action: ArcAction,
    pub is_reset: bool,
    pub reasoning: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct LineEnvelope {
    timestamp: String,
    data: Value,
}

#[derive(Debug, Deserialize)]
struct DataFields {
    game_id: String,
    state: String,
    levels_completed: i64,
    win_levels: i64,
    action_input: Option<Value>,
    guid: Option<String>,
    #[serde(default)]
    full_reset: bool,
    #[serde(default)]
    available_actions: Vec<i64>,
    frame: Option<Value>,
}

/// Parse ACTION1..ACTION6 / RESET / numeric ids into an action descriptor.
pub fn parse_action_input(raw: &Value) -> Result<ParsedActionInput> {
    let obj = raw
        .as_object()
        .ok_or_else(|| anyhow!("action_input must be an object"))?;
    let id_val = obj
        .get("id")
        .ok_or_else(|| anyhow!("action_input missing id"))?;
    let id_name = match id_val {
        Value::String(s) => s.clone(),
        Value::Number(n) => {
            let n = n
                .as_i64()
                .ok_or_else(|| anyhow!("action id number out of range"))?;
            match n {
                0 => "RESET".to_string(),
                1..=7 => format!("ACTION{n}"),
                other => bail!("unsupported numeric action id {other}"),
            }
        }
        _ => bail!("action id must be string or number"),
    };

    let data = obj
        .get("data")
        .cloned()
        .unwrap_or(Value::Object(Default::default()));
    let (x, y) = extract_xy(&data)?;
    let reasoning = obj.get("reasoning").cloned();

    if id_name == "RESET" || id_name == "ACTION0" {
        return Ok(ParsedActionInput {
            // Placeholder; RESET transitions are skipped by the importer.
            action: ArcAction {
                id: 1,
                x: None,
                y: None,
            },
            is_reset: true,
            reasoning,
        });
    }

    let id = match id_name.as_str() {
        "ACTION1" => 1u8,
        "ACTION2" => 2,
        "ACTION3" => 3,
        "ACTION4" => 4,
        "ACTION5" => 5,
        "ACTION6" => 6,
        "ACTION7" => 7,
        other => bail!("unknown action id {other}"),
    };

    let action = if id == 6 {
        ArcAction::new(6, x, y)?
    } else {
        if x.is_some() || y.is_some() {
            bail!("coordinates only allowed for ACTION6");
        }
        ArcAction::new(id, None, None)?
    };

    Ok(ParsedActionInput {
        action,
        is_reset: false,
        reasoning,
    })
}

fn extract_xy(data: &Value) -> Result<(Option<u8>, Option<u8>)> {
    let Some(obj) = data.as_object() else {
        return Ok((None, None));
    };
    let read_u8 = |key: &str| -> Result<Option<u8>> {
        match obj.get(key) {
            None => Ok(None),
            Some(Value::Null) => Ok(None),
            Some(Value::Number(n)) => {
                let v = n.as_i64().ok_or_else(|| anyhow!("{key} not an integer"))?;
                ensure_coord(v).map(Some)
            }
            Some(other) => bail!("{key} must be an integer, got {other}"),
        }
    };
    Ok((read_u8("x")?, read_u8("y")?))
}

fn ensure_coord(v: i64) -> Result<u8> {
    if (0..64).contains(&v) {
        Ok(v as u8)
    } else {
        bail!("coordinate {v} out of 0..64");
    }
}

/// Take the last frame layer as the settled observation; validate palette/dims.
pub fn settled_frame_from_layers(frame_val: &Value) -> Result<ArcFrame> {
    let layers = frame_val
        .as_array()
        .ok_or_else(|| anyhow!("frame must be an array of layers"))?;
    ensure!(!layers.is_empty(), "frame layers empty");
    let mut parsed = Vec::with_capacity(layers.len());
    for (layer_index, layer) in layers.iter().enumerate() {
        parsed.push(
            parse_frame_layer(layer)
                .with_context(|| format!("invalid frame layer {layer_index}"))?,
        );
    }
    let first_dims = (parsed[0].width, parsed[0].height);
    ensure!(
        parsed
            .iter()
            .all(|frame| (frame.width, frame.height) == first_dims),
        "animation frame layers changed dimensions"
    );
    let frame = parsed.pop().expect("non-empty checked above");
    if frame.width as usize == FRAME_SIDE && frame.height as usize == FRAME_SIDE {
        Ok(frame)
    } else {
        frame.to_fixed_64()
    }
}

fn parse_frame_layer(layer: &Value) -> Result<ArcFrame> {
    let rows = layer
        .as_array()
        .ok_or_else(|| anyhow!("frame layer must be a 2D array"))?;
    ensure!(!rows.is_empty(), "frame layer has no rows");
    let height = rows.len();
    let width = rows[0]
        .as_array()
        .ok_or_else(|| anyhow!("frame row must be an array"))?
        .len();
    ensure!(width > 0, "frame width must be > 0");
    let mut pixels = Vec::with_capacity(width * height);
    for (y, row_val) in rows.iter().enumerate() {
        let row = row_val
            .as_array()
            .ok_or_else(|| anyhow!("frame row {y} must be an array"))?;
        ensure!(
            row.len() == width,
            "ragged frame at row {y}: width {} != {width}",
            row.len()
        );
        for (x, cell) in row.iter().enumerate() {
            let v = cell
                .as_i64()
                .ok_or_else(|| anyhow!("pixel ({x},{y}) not an integer"))?;
            ensure!(
                (0..=15).contains(&v),
                "palette value {v} out of 0..=15 at ({x},{y})"
            );
            pixels.push(v as u8);
        }
    }
    ArcFrame::new(width as u16, height as u16, pixels)
}

/// Load one JSONL recording file into ordered events.
pub fn load_recording_jsonl(path: &Path) -> Result<Vec<RecordingEvent>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();
    for (idx, line_res) in reader.lines().enumerate() {
        let line_no = idx + 1;
        let line = line_res.with_context(|| format!("{}:{line_no}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let env: LineEnvelope = serde_json::from_str(&line).with_context(|| {
            format!(
                "{}:{line_no}: invalid JSONL envelope {{timestamp,data}}",
                path.display()
            )
        })?;
        let data: DataFields = serde_json::from_value(env.data.clone()).with_context(|| {
            format!(
                "{}:{line_no}: invalid recording data payload",
                path.display()
            )
        })?;
        let frame_val = data
            .frame
            .ok_or_else(|| anyhow!("{}:{line_no}: missing frame field", path.display()))?;
        let frame = settled_frame_from_layers(&frame_val)
            .with_context(|| format!("{}:{line_no}: frame validation failed", path.display()))?;
        let action = match data.action_input {
            None => None,
            Some(Value::Null) => None,
            Some(v) => Some(parse_action_input(&v).with_context(|| {
                format!("{}:{line_no}: action_input parse failed", path.display())
            })?),
        };
        events.push(RecordingEvent {
            timestamp: env.timestamp,
            game_id: data.game_id,
            state: data.state,
            levels_completed: data.levels_completed,
            win_levels: data.win_levels,
            action,
            guid: data.guid.unwrap_or_default(),
            full_reset: data.full_reset,
            available_actions: data.available_actions,
            frame,
            source_path: path.to_path_buf(),
            line: line_no,
        });
    }
    Ok(events)
}

/// Pair consecutive events into transitions. Later `action_input` caused later frame.
/// RESET actions are skipped (not emitted as transitions).
pub fn events_to_transitions(events: &[RecordingEvent]) -> Result<Vec<TransitionSample>> {
    let mut out = Vec::new();
    for (idx, w) in events.windows(2).enumerate() {
        let prev = &w[0];
        let curr = &w[1];
        let Some(parsed) = curr.action.as_ref() else {
            continue;
        };
        if parsed.is_reset {
            continue;
        }
        let (won, goal_failed) = terminal_labels_from_public_state(&curr.state);
        let goal_satisfied = won || curr.levels_completed > prev.levels_completed;
        out.push(TransitionSample {
            current: prev.frame.clone(),
            next: curr.frame.clone(),
            action: parsed.action.clone(),
            goal_features: GoalFeatures::zeros(),
            noop: None,
            goal_satisfied: Some(goal_satisfied),
            goal_failed: Some(goal_failed),
            exhausted: None,
            split: Split::HeldOutComposition,
            family: format!("arc3:{}", curr.game_id),
            seed: 0,
            episode_id: curr.line as u64,
            transition_index: idx as u64,
            oracle_latent: None,
        });
    }
    Ok(out)
}

/// WIN / GAME_OVER labels come only from the public `state` string.
///
/// Other official states (`NOT_FINISHED`, `NOT_STARTED`) are non-terminal here.
/// See https://docs.arcprize.org/full-play-test .
pub fn terminal_labels_from_public_state(state: &str) -> (bool, bool) {
    match state {
        "WIN" => (true, false),
        "GAME_OVER" => (false, true),
        _ => (false, false),
    }
}

/// Per-run counters aligned with official scorecard `RunSummary` fields.
///
/// Derived from toolkit recordings only; human baselines are absent so RHAE
/// cannot be computed from this struct alone.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RecordingRunSummary {
    pub game_id: String,
    pub guid: String,
    pub actions: usize,
    pub resets: usize,
    pub levels_completed: i64,
    pub win_levels: i64,
    pub state: String,
    pub completed: bool,
}

/// Aggregate one JSONL recording file into per-`guid` runs.
pub fn summarize_recording_runs(events: &[RecordingEvent]) -> Vec<RecordingRunSummary> {
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Acc {
        game_id: String,
        actions: usize,
        resets: usize,
        levels_completed: i64,
        win_levels: i64,
        state: String,
    }

    let mut runs: BTreeMap<String, Acc> = BTreeMap::new();
    for event in events {
        let guid = if event.guid.is_empty() {
            event.source_path.to_string_lossy().into_owned()
        } else {
            event.guid.clone()
        };
        let acc = runs.entry(guid).or_default();
        acc.game_id = event.game_id.clone();
        acc.levels_completed = acc.levels_completed.max(event.levels_completed);
        acc.win_levels = acc.win_levels.max(event.win_levels);
        acc.state = event.state.clone();
        if event.full_reset {
            acc.resets += 1;
        }
        if let Some(parsed) = event.action.as_ref() {
            if parsed.is_reset {
                acc.resets += 1;
            } else {
                acc.actions += 1;
            }
        }
    }

    runs.into_iter()
        .map(|(guid, acc)| {
            let completed = matches!(acc.state.as_str(), "WIN" | "GAME_OVER");
            RecordingRunSummary {
                game_id: acc.game_id,
                guid,
                actions: acc.actions,
                resets: acc.resets,
                levels_completed: acc.levels_completed,
                win_levels: acc.win_levels,
                state: acc.state,
                completed,
            }
        })
        .collect()
}

/// Summarize every `*.jsonl` recording under `root`.
pub fn summarize_recordings_dir(root: &Path) -> Result<Vec<RecordingRunSummary>> {
    let mut files = Vec::new();
    collect_jsonl(root, &mut files)?;
    files.sort();
    let mut out = Vec::new();
    for path in files {
        let events = load_recording_jsonl(&path)?;
        out.extend(summarize_recording_runs(&events));
    }
    Ok(out)
}

/// Recursively import `*.jsonl` under `root`, sorted deterministically by path.
pub fn import_recordings_dir(root: &Path) -> Result<Vec<TransitionSample>> {
    let mut files = Vec::new();
    collect_jsonl(root, &mut files)?;
    files.sort();
    let mut all = Vec::new();
    for path in files {
        let events = load_recording_jsonl(&path)?;
        all.extend(events_to_transitions(&events)?);
    }
    Ok(all)
}

fn collect_jsonl(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("entry under {}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_jsonl(&path, out)?;
        } else if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_fixture(dir: &Path, relative: &str, contents: &str) -> PathBuf {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = File::create(&path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        path
    }

    fn layer(w: usize, h: usize, fill: u8) -> String {
        let row = format!(
            "[{}]",
            (0..w)
                .map(|_| fill.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let rows = (0..h).map(|_| row.clone()).collect::<Vec<_>>().join(",");
        format!("[[{rows}]]")
    }

    fn event_line(ts: &str, state: &str, action_id: &str, data: &str, frame_json: &str) -> String {
        format!(
            "{{\"timestamp\":\"{ts}\",\"data\":{{\"game_id\":\"demo\",\"state\":\"{state}\",\"levels_completed\":0,\"win_levels\":1,\"action_input\":{{\"id\":\"{action_id}\",\"data\":{data},\"reasoning\":null}},\"guid\":\"g1\",\"full_reset\":false,\"available_actions\":[1,2,3,4,5,6,7],\"frame\":{frame_json}}}}}"
        )
    }

    #[test]
    fn action_pairing_reset_skip_and_terminal_labels() {
        let dir = std::env::temp_dir().join(format!("tofy-arc3-fixture-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();

        let f0 = layer(2, 2, 1);
        let f1 = layer(2, 2, 2);
        let f2 = layer(2, 2, 3);
        let f3 = layer(2, 2, 4);

        let body = [
            event_line("t0", "NOT_FINISHED", "RESET", "{}", &f0),
            event_line("t1", "NOT_FINISHED", "ACTION1", "{}", &f1),
            event_line("t2", "NOT_FINISHED", "RESET", "{}", &f2),
            event_line(
                "t3",
                "NOT_FINISHED",
                "ACTION6",
                r#"{"x":3,"y":4}"#,
                &layer(2, 2, 8),
            ),
            event_line("t4", "WIN", "ACTION2", "{}", &f3),
        ]
        .join("\n");

        let path = write_fixture(&dir, "sub/rec.jsonl", &body);
        let events = load_recording_jsonl(&path).unwrap();
        assert_eq!(events.len(), 5);
        // Settled observation padded to 64x64; native 2x2 preserved in top-left.
        assert_eq!(events[0].frame.width, 64);
        assert_eq!(events[0].frame.pixel(0, 0), Some(1));

        let samples = events_to_transitions(&events).unwrap();
        // (0,1) ACTION1 keep; (1,2) RESET skip; (2,3) ACTION6 keep; (3,4) ACTION2/WIN keep
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].action.id, 1);
        assert_eq!(samples[0].current.pixel(0, 0), Some(1));
        assert_eq!(samples[0].next.pixel(0, 0), Some(2));

        assert_eq!(samples[1].action.id, 6);
        assert_eq!(samples[1].action.x, Some(3));
        assert_eq!(samples[1].action.y, Some(4));
        assert_eq!(samples[1].current.pixel(0, 0), Some(3));
        assert_eq!(samples[1].next.pixel(0, 0), Some(8));
        assert_eq!(samples[1].goal_satisfied, Some(false));
        assert_eq!(samples[1].goal_failed, Some(false));

        assert_eq!(samples[2].action.id, 2);
        assert_eq!(samples[2].goal_satisfied, Some(true));
        assert_eq!(samples[2].goal_failed, Some(false));

        let imported = import_recordings_dir(&dir).unwrap();
        assert_eq!(imported, samples);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn game_over_label_and_malformed_palette_rejection() {
        let dir = std::env::temp_dir().join(format!("tofy-arc3-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let ok = [
            event_line("t0", "NOT_FINISHED", "ACTION1", "{}", &layer(1, 1, 0)),
            event_line("t1", "GAME_OVER", "ACTION3", "{}", &layer(1, 1, 5)),
        ]
        .join("\n");
        let path = write_fixture(&dir, "ok.jsonl", &ok);
        let samples = events_to_transitions(&load_recording_jsonl(&path).unwrap()).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].goal_failed, Some(true));
        assert_eq!(samples[0].goal_satisfied, Some(false));

        let bad = event_line("t0", "NOT_FINISHED", "ACTION1", "{}", "[[[16]]]");
        let bad_path = write_fixture(&dir, "bad.jsonl", &bad);
        let err = load_recording_jsonl(&bad_path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("bad.jsonl"));
        assert!(msg.contains("palette") || msg.contains("frame"));

        let bad_action = event_line(
            "t0",
            "NOT_FINISHED",
            "ACTION1",
            r#"{"x":1,"y":1}"#,
            &layer(1, 1, 0),
        );
        let bad_a_path = write_fixture(&dir, "bad_action.jsonl", &bad_action);
        let err = load_recording_jsonl(&bad_a_path).unwrap_err();
        assert!(
            format!("{err:#}").contains("coordinates") || format!("{err:#}").contains("ACTION")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn level_completion_is_positive_public_event() {
        let dir =
            std::env::temp_dir().join(format!("tofy-arc3-level-complete-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let before = event_line("t0", "NOT_FINISHED", "ACTION1", "{}", &layer(1, 1, 0));
        let after = event_line("t1", "NOT_FINISHED", "ACTION2", "{}", &layer(1, 1, 1))
            .replace("\"levels_completed\":0", "\"levels_completed\":1");
        let path = write_fixture(&dir, "level.jsonl", &format!("{before}\n{after}"));
        let samples = events_to_transitions(&load_recording_jsonl(&path).unwrap()).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].goal_satisfied, Some(true));
        assert_eq!(samples[0].goal_failed, Some(false));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn action7_undo_imports() {
        let dir = std::env::temp_dir().join(format!("tofy-arc3-action7-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let body = [
            event_line("t0", "NOT_FINISHED", "ACTION1", "{}", &layer(1, 1, 0)),
            event_line("t1", "NOT_FINISHED", "ACTION7", "{}", &layer(1, 1, 1)),
        ]
        .join("\n");
        let path = write_fixture(&dir, "undo.jsonl", &body);
        let samples = events_to_transitions(&load_recording_jsonl(&path).unwrap()).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].action.id, 7);
        assert_eq!(samples[0].action.to_tofy().unwrap(), crate::domain::Action::Undo);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn recording_run_summary_counts_actions() {
        let dir = std::env::temp_dir().join(format!("tofy-arc3-runs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let body = [
            event_line("t0", "NOT_FINISHED", "RESET", "{}", &layer(1, 1, 0)),
            event_line("t1", "NOT_FINISHED", "ACTION1", "{}", &layer(1, 1, 1)),
            event_line("t2", "WIN", "ACTION2", "{}", &layer(1, 1, 2)),
        ]
        .join("\n");
        let path = write_fixture(&dir, "run.jsonl", &body);
        let events = load_recording_jsonl(&path).unwrap();
        let runs = summarize_recording_runs(&events);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].actions, 2);
        assert_eq!(runs[0].resets, 1);
        assert!(runs[0].completed);
        assert_eq!(runs[0].state, "WIN");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn last_layer_is_settled_observation() {
        // Two layers: first all 1s, last all 7s — importer must keep 7.
        let frame = "[[[1,1],[1,1]],[[7,7],[7,7]]]";
        let line = event_line("t0", "NOT_FINISHED", "ACTION1", "{}", frame);
        let dir = std::env::temp_dir().join(format!("tofy-arc3-layer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_fixture(&dir, "layers.jsonl", &line);
        let events = load_recording_jsonl(&path).unwrap();
        assert_eq!(events[0].frame.pixel(0, 0), Some(7));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
