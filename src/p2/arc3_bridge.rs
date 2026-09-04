//! Newline-delimited JSON bridge for the local ARC-AGI-3 toolkit.

use crate::gpu_lock::GpuSessionGuard;
use crate::p2::adaptation::ContextScopeKind;
use crate::p2::arc3_live::{adaptation_for, live_context_for, AdaptingPolicy};
use crate::p2::arc3_live::{
    decision_telemetry, live_evidence_class, live_run_provenance, run_public_suite, sha256_file,
    write_json_atomic, ActionDecision, AmbiguousMutation, ApiObservation, ArcApi, ArcObservation,
    LiveDriverOptions, LivePolicy, LivePolicyKind, LiveRecordingRun, LiveRunSettings, ModelPolicy,
    MutationError, MutationResult, PolicyContract, PublicGame, DEFAULT_TRIED_PENALTY,
};
use crate::p2::data::ArcAction;
use crate::p2::eval::load_model;
use crate::p2::train::{load_train_config, resolve_device};
use anyhow::{anyhow, bail, ensure, Context, Result};
use clap::ValueEnum;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

const PROTOCOL_VERSION: u8 = 1;
const PHYSICAL_BATCH: usize = 128;
const ACTION6_MAX_CANDIDATES: usize = 128;
const ACTION6_GRID_STRIDE: usize = 8;
const DRIVER_NAME: &str = "local_toolkit_bridge";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Arc3BridgeMode {
    Drive,
    Serve,
}

#[derive(Debug, Clone)]
pub struct Arc3BridgeConfig {
    pub mode: Arc3BridgeMode,
    pub device: String,
    pub checkpoint: PathBuf,
    pub train_config: PathBuf,
    pub output: Option<PathBuf>,
    pub recordings_dir: Option<PathBuf>,
    pub profile_eval: bool,
    pub seed: Option<u64>,
    pub max_actions_per_game: Option<u32>,
    pub policy: LivePolicyKind,
    pub phase_a_calibration: Option<PathBuf>,
    /// ADR 0005 §6.2 Channel B test-time adaptation.
    pub adapt: bool,
    pub adapt_carry: bool,
    /// ADR 0005 §6.1 Channel A; `None` = on for `world_core_v6` checkpoints.
    pub context_window: Option<bool>,
    /// ADR 0005 §1.5 `--context-scope`; independent of the Channel B arm.
    pub context_scope: ContextScopeKind,
}

impl Arc3BridgeConfig {
    fn validate(&self) -> Result<()> {
        if self.mode == Arc3BridgeMode::Drive {
            ensure!(self.output.is_some(), "--output is required in drive mode");
        }
        ensure!(
            self.adapt || !self.adapt_carry,
            "--adapt-carry requires --adapt"
        );
        Ok(())
    }
}

pub fn run_arc3_bridge(config: &Arc3BridgeConfig) -> Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    run_arc3_bridge_with_io(config, stdin.lock(), stdout.lock())
}

fn run_arc3_bridge_with_io<R: BufRead, W: Write>(
    config: &Arc3BridgeConfig,
    reader: R,
    writer: W,
) -> Result<()> {
    config.validate()?;
    let train_config = load_train_config(&config.train_config)?;
    let _gpu_guard = if config.device == "cuda" || config.device.starts_with("cuda:") {
        Some(GpuSessionGuard::acquire(&train_config.output_dir)?)
    } else {
        None
    };
    let checkpoint_sha256 = sha256_file(&config.checkpoint)?;
    let train_config_sha256 = sha256_file(&config.train_config)?;
    let device = resolve_device(&config.device)?;
    let (model, varmap) = load_model(&train_config, &config.checkpoint, &device)?;
    let adapter = adaptation_for(
        &model,
        &varmap,
        &device,
        config.adapt,
        config.adapt_carry,
        config.context_scope,
    )?;
    let context = live_context_for(&model, config.context_window, config.context_scope);
    let prior = adapter
        .as_ref()
        .map(crate::p2::adaptation::FastWeightAdapter::prior_weights);
    let mut policy = match config.policy {
        LivePolicyKind::Greedy => {
            let mut policy = ModelPolicy::new(
                &model,
                &device,
                PHYSICAL_BATCH,
                ACTION6_MAX_CANDIDATES,
                ACTION6_GRID_STRIDE,
                DEFAULT_TRIED_PENALTY,
            );
            policy.set_context(context);
            policy.set_prior_weights(prior);
            BridgePolicy::Greedy(Box::new(policy))
        }
        LivePolicyKind::PhaseA => {
            let calibration = crate::p2::arc3_phase_a::load_phase_a_calibration(
                config.phase_a_calibration.as_deref(),
            )?;
            let mut policy = crate::p2::arc3_phase_a::PhaseAPolicy::with_tensor_model(
                &model,
                &device,
                PHYSICAL_BATCH,
                crate::p2::latent_planning::config::PhaseAConfig::default(),
                calibration,
                ACTION6_MAX_CANDIDATES,
                ACTION6_GRID_STRIDE,
            )?;
            policy.set_context(context);
            policy.set_prior_weights(prior);
            BridgePolicy::PhaseA(Box::new(policy))
        }
    };

    match config.mode {
        Arc3BridgeMode::Drive => {
            let output = config.output.as_deref().expect("validated drive output");
            if let (true, BridgePolicy::Greedy(policy)) = (config.profile_eval, &mut policy) {
                policy.enable_eval_profile(
                    output_parent(output),
                    format!("tofy.p2.arc3.{checkpoint_sha256}"),
                    &config.device,
                );
            }
            let provenance = live_run_provenance()?;
            let driver = LiveDriverOptions {
                exploratory: provenance.git_dirty,
                max_actions_per_game: config.max_actions_per_game,
                ..LiveDriverOptions::default()
            };
            let settings = LiveRunSettings {
                checkpoint: config.checkpoint.clone(),
                checkpoint_sha256,
                train_config: config.train_config.clone(),
                train_config_sha256,
                device: config.device.clone(),
                base_url: DRIVER_NAME.into(),
                requested_games: Vec::new(),
                driver: driver.clone(),
                git_revision: provenance.git_revision,
                git_dirty: provenance.git_dirty,
                dirty_diff_sha256: provenance.dirty_diff_sha256,
                executable_sha256: provenance.executable_sha256,
                build_profile: provenance.build_profile,
                cli_args: provenance.cli_args,
                evidence_class: live_evidence_class(&driver).into(),
                recordings_dir: config.recordings_dir.clone(),
                contract: PolicyContract::for_kind(config.policy),
            };
            let mut api = StdioArcApi::new(reader, writer);
            let mut policy = AdaptingPolicy::new(policy, adapter);
            let report = run_public_suite(&mut api, &mut policy, &settings);
            if let Some(error) = api.take_fatal_error() {
                bail!("fatal stdio bridge transport error: {error}");
            }
            let report = report?;
            write_json_atomic(output, &report)?;
            eprintln!(
                "p2-arc3-bridge drive complete games={} report={}",
                report.games.len(),
                output.display()
            );
            Ok(())
        }
        Arc3BridgeMode::Serve => {
            if let (true, BridgePolicy::Greedy(policy)) = (config.profile_eval, &mut policy) {
                let anchor = config
                    .output
                    .as_deref()
                    .map(output_parent)
                    .unwrap_or(Path::new("."));
                policy.enable_streaming_eval_profile(
                    anchor,
                    format!("tofy.p2.arc3.{checkpoint_sha256}"),
                    &config.device,
                );
            }
            let _seed = config.seed.unwrap_or(train_config.seed);
            let mut policy = AdaptingPolicy::new(policy, adapter);
            run_serve_loop(
                reader,
                writer,
                &mut policy,
                config.recordings_dir.as_deref(),
            )
        }
    }
}

fn output_parent(output: &Path) -> &Path {
    output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireResponse {
    #[serde(default)]
    ok: Option<Value>,
    #[serde(default)]
    err: Option<WireError>,
}

#[derive(Debug, Deserialize)]
struct WireError {
    message: String,
    kind: WireErrorKind,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WireErrorKind {
    Failed,
    Ambiguous,
}

pub(crate) struct StdioArcApi<R, W> {
    reader: R,
    writer: W,
    fatal_error: Option<String>,
}

impl<R: BufRead, W: Write> StdioArcApi<R, W> {
    pub(crate) fn new(reader: R, writer: W) -> Self {
        Self {
            reader,
            writer,
            fatal_error: None,
        }
    }

    fn take_fatal_error(&mut self) -> Option<String> {
        self.fatal_error.take()
    }

    fn exchange(&mut self, request: &Value) -> Result<std::result::Result<Value, WireError>> {
        if let Err(error) = write_json_line(&mut self.writer, request) {
            self.fatal_error = Some(format!("write request: {error:#}"));
            return Err(error).context("write bridge request");
        }
        let mut line = String::new();
        let bytes = match self.reader.read_line(&mut line) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.fatal_error = Some(format!("read response: {error}"));
                return Err(error).context("read bridge response");
            }
        };
        if bytes == 0 {
            self.fatal_error = Some("unexpected EOF while waiting for response".into());
            bail!("unexpected EOF while waiting for bridge response");
        }
        let response: WireResponse = match serde_json::from_str(&line) {
            Ok(response) => response,
            Err(error) => {
                let message = format!("parse bridge response line {line:?}: {error}");
                self.fatal_error = Some(message.clone());
                bail!(message);
            }
        };
        match (response.ok, response.err) {
            (Some(ok), None) => Ok(Ok(ok)),
            (None, Some(error)) => Ok(Err(error)),
            _ => {
                let message = "bridge response must contain exactly one of ok or err";
                self.fatal_error = Some(message.into());
                bail!(message);
            }
        }
    }

    fn request(&mut self, request: &Value) -> Result<Value> {
        match self.exchange(request)? {
            Ok(value) => Ok(value),
            Err(error) => bail!("bridge request failed: {}", error.message),
        }
    }

    fn mutation_request(
        &mut self,
        request: &Value,
        operation: &str,
        game_id: Option<&str>,
        guid: Option<&str>,
        action: Option<&ArcAction>,
    ) -> MutationResult<Value> {
        match self.exchange(request) {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(error)) if matches!(error.kind, WireErrorKind::Ambiguous) => {
                Err(MutationError::Ambiguous(AmbiguousMutation {
                    operation: operation.into(),
                    game_id: game_id.map(str::to_owned),
                    guid: guid.map(str::to_owned),
                    action: action.cloned(),
                    cause: error.message,
                }))
            }
            Ok(Err(error)) => Err(MutationError::Failed(anyhow!(error.message))),
            Err(error) => Err(MutationError::Failed(error)),
        }
    }
}

impl<R: BufRead, W: Write> ArcApi for StdioArcApi<R, W> {
    fn list_games(&mut self) -> Result<Vec<PublicGame>> {
        #[derive(Deserialize)]
        struct Games {
            games: Vec<PublicGame>,
        }
        let value = self.request(&json!({ "op": "list_games" }))?;
        Ok(serde_json::from_value::<Games>(value)
            .context("parse list_games payload")?
            .games)
    }

    fn open_scorecard(&mut self, _metadata: &Value) -> Result<String> {
        #[derive(Deserialize)]
        struct Opened {
            card_id: String,
        }
        let value = self.request(&json!({ "op": "open_scorecard" }))?;
        let card_id = serde_json::from_value::<Opened>(value)
            .context("parse open_scorecard payload")?
            .card_id;
        ensure!(!card_id.is_empty(), "bridge returned an empty card_id");
        Ok(card_id)
    }

    fn reset(
        &mut self,
        game_id: &str,
        card_id: &str,
        guid: Option<&str>,
    ) -> MutationResult<ArcObservation> {
        let value = self.mutation_request(
            &json!({
                "op": "reset",
                "card_id": card_id,
                "game_id": game_id,
                "guid": guid,
            }),
            "reset",
            Some(game_id),
            guid,
            None,
        )?;
        let observation = parse_observation_payload(value).map_err(MutationError::Failed)?;
        if observation.game_id != game_id
            || observation.guid.is_empty()
            || guid.is_some_and(|expected| observation.guid != expected)
        {
            return Err(MutationError::Ambiguous(AmbiguousMutation {
                operation: "reset".into(),
                game_id: Some(game_id.into()),
                guid: guid.map(str::to_owned),
                action: None,
                cause: "RESET response session identifiers do not match request".into(),
            }));
        }
        Ok(observation)
    }

    fn act(
        &mut self,
        game_id: &str,
        guid: &str,
        action: &ArcAction,
        _reasoning: &Value,
    ) -> MutationResult<ArcObservation> {
        let data = if action.id == 6 {
            json!({
                "x": action.x.context("ACTION6 missing x").map_err(MutationError::Failed)?,
                "y": action.y.context("ACTION6 missing y").map_err(MutationError::Failed)?,
            })
        } else {
            json!({})
        };
        let value = self.mutation_request(
            &json!({
                "op": "act",
                "game_id": game_id,
                "guid": guid,
                "action_id": action.id,
                "data": data,
            }),
            "act",
            Some(game_id),
            Some(guid),
            Some(action),
        )?;
        let observation = parse_observation_payload(value).map_err(MutationError::Failed)?;
        if observation.game_id != game_id || observation.guid != guid {
            return Err(MutationError::Ambiguous(AmbiguousMutation {
                operation: "act".into(),
                game_id: Some(game_id.into()),
                guid: Some(guid.into()),
                action: Some(action.clone()),
                cause: "ACTION response session identifiers do not match request".into(),
            }));
        }
        Ok(observation)
    }

    fn close_scorecard(&mut self, card_id: &str) -> Result<Value> {
        let value = self.request(&json!({
            "op": "close_scorecard",
            "card_id": card_id,
        }))?;
        value
            .get("scorecard")
            .cloned()
            .context("close_scorecard payload is missing scorecard")
    }
}

fn parse_observation_payload(value: Value) -> Result<ArcObservation> {
    let observation = value
        .get("observation")
        .cloned()
        .context("bridge payload is missing observation")?;
    serde_json::from_value::<ApiObservation>(observation)
        .context("parse bridge observation")?
        .into_arc_observation(true)
}

enum PendingServeAction {
    Reset { retry: bool },
    Action { action: ArcAction, telemetry: Value },
}

struct ServeStream {
    recording: Option<LiveRecordingRun>,
    pending: Option<PendingServeAction>,
    levels_completed: u16,
    policy_started: bool,
    /// Observation the pending action was chosen from (factual transition source).
    last_observation: Option<ArcObservation>,
}

/// Either deployed controller behind one `LivePolicy` surface.
enum BridgePolicy<'a> {
    Greedy(Box<ModelPolicy<'a>>),
    PhaseA(
        Box<
            crate::p2::arc3_phase_a::PhaseAPolicy<crate::p2::arc3_phase_a::TensorPhaseAAdapter<'a>>,
        >,
    ),
}

impl LivePolicy for BridgePolicy<'_> {
    fn choose_action(&mut self, observation: &ArcObservation) -> Result<ActionDecision> {
        match self {
            Self::Greedy(p) => p.choose_action(observation),
            Self::PhaseA(p) => p.choose_action(observation),
        }
    }
    fn policy_name(&self) -> &'static str {
        match self {
            Self::Greedy(p) => p.policy_name(),
            Self::PhaseA(p) => p.policy_name(),
        }
    }
    fn on_suite_start(&mut self, games: &[PublicGame]) -> Result<()> {
        match self {
            Self::Greedy(p) => p.on_suite_start(games),
            Self::PhaseA(p) => p.on_suite_start(games),
        }
    }
    fn on_game_start(&mut self, game_id: &str) {
        match self {
            Self::Greedy(p) => p.on_game_start(game_id),
            Self::PhaseA(p) => p.on_game_start(game_id),
        }
    }
    fn on_confirmed_transition(
        &mut self,
        current: &ArcObservation,
        action: &crate::p2::data::ArcAction,
        next: &ArcObservation,
    ) {
        match self {
            Self::Greedy(p) => p.on_confirmed_transition(current, action, next),
            Self::PhaseA(p) => p.on_confirmed_transition(current, action, next),
        }
    }
    fn on_level_transition(&mut self, levels_completed: u16) {
        match self {
            Self::Greedy(p) => p.on_level_transition(levels_completed),
            Self::PhaseA(p) => p.on_level_transition(levels_completed),
        }
    }
    fn on_reset_retry(&mut self, reason: &str) {
        match self {
            Self::Greedy(p) => p.on_reset_retry(reason),
            Self::PhaseA(p) => p.on_reset_retry(reason),
        }
    }
    fn on_game_end(&mut self, outcome: &str) {
        match self {
            Self::Greedy(p) => p.on_game_end(outcome),
            Self::PhaseA(p) => p.on_game_end(outcome),
        }
    }
    fn finish_session(&mut self) -> Result<()> {
        match self {
            Self::Greedy(p) => p.finish_session(),
            Self::PhaseA(p) => p.finish_session(),
        }
    }
}

fn run_serve_loop<R: BufRead, W: Write, P: LivePolicy>(
    mut reader: R,
    mut writer: W,
    policy: &mut P,
    recordings_dir: Option<&Path>,
) -> Result<()> {
    write_json_line(
        &mut writer,
        &json!({ "ready": { "protocol": PROTOCOL_VERSION, "mode": "serve" } }),
    )?;
    let mut streams = BTreeMap::<(String, String), ServeStream>::new();
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line).context("read serve request")?;
        if bytes == 0 {
            finish_serve_session(&mut streams, policy)?;
            return Ok(());
        }
        let request: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                write_serve_error(&mut writer, format!("unparseable request: {error}"))?;
                continue;
            }
        };
        let Some(op) = request.get("op").and_then(Value::as_str) else {
            write_serve_error(&mut writer, "request is missing string op")?;
            continue;
        };
        match op {
            "observe" => {
                let parsed = request
                    .get("observation")
                    .cloned()
                    .context("observe request is missing observation")
                    .and_then(|value| {
                        serde_json::from_value::<ApiObservation>(value)
                            .context("parse serve observation")
                    })
                    .and_then(|value| value.into_arc_observation(true));
                let observation = match parsed {
                    Ok(observation) => observation,
                    Err(error) => {
                        write_serve_error(&mut writer, format!("{error:#}"))?;
                        continue;
                    }
                };
                match serve_observation(&mut streams, policy, recordings_dir, observation) {
                    Ok(response) => write_json_line(&mut writer, &response)?,
                    Err(error) => write_serve_error(&mut writer, format!("{error:#}"))?,
                }
            }
            "shutdown" => {
                if let Err(error) = finish_serve_session(&mut streams, policy) {
                    write_serve_error(&mut writer, format!("flush bridge session: {error:#}"))?;
                    return Err(error).context("flush bridge session");
                }
                write_json_line(&mut writer, &json!({ "ok": { "bye": true } }))?;
                return Ok(());
            }
            other => write_serve_error(&mut writer, format!("unknown op {other:?}"))?,
        }
    }
}

fn serve_observation<P: LivePolicy>(
    streams: &mut BTreeMap<(String, String), ServeStream>,
    policy: &mut P,
    recordings_dir: Option<&Path>,
    observation: ArcObservation,
) -> Result<Value> {
    observation.validate()?;
    let key = (observation.game_id.clone(), observation.guid.clone());
    if !streams.contains_key(&key) {
        let recording = recordings_dir
            .map(|root| LiveRecordingRun::start(root, &observation))
            .transpose()?;
        streams.insert(
            key.clone(),
            ServeStream {
                recording,
                pending: None,
                levels_completed: observation.levels_completed,
                policy_started: false,
                last_observation: None,
            },
        );
    }
    let stream = streams.get_mut(&key).expect("stream was inserted");
    if let Some(pending) = stream.pending.take() {
        match pending {
            PendingServeAction::Reset { retry } => {
                if let Some(recording) = stream.recording.as_mut() {
                    recording.push_reset(&observation)?;
                }
                if retry {
                    policy.on_reset_retry("serve_game_over");
                }
            }
            PendingServeAction::Action { action, telemetry } => {
                if let Some(recording) = stream.recording.as_mut() {
                    recording.push_action(&observation, &action, &telemetry)?;
                }
                if let Some(previous) = stream.last_observation.as_ref() {
                    let same_session = previous.guid == observation.guid
                        && !observation.full_reset
                        && observation.levels_completed >= previous.levels_completed;
                    if same_session {
                        policy.on_confirmed_transition(previous, &action, &observation);
                    }
                }
            }
        }
    }
    stream.last_observation = Some(observation.clone());
    if observation.levels_completed > stream.levels_completed {
        policy.on_level_transition(observation.levels_completed);
    }
    stream.levels_completed = observation.levels_completed;

    if !stream.policy_started && observation.state == "NOT_FINISHED" {
        policy.on_game_start(&observation.game_id);
        stream.policy_started = true;
    }

    if observation.terminal() {
        let retry = observation.state == "GAME_OVER";
        if !retry && stream.policy_started {
            policy.on_game_end(if observation.state == "WIN" {
                "completed"
            } else {
                "serve_terminal"
            });
            stream.policy_started = false;
        }
        stream.pending = Some(PendingServeAction::Reset { retry });
        return Ok(json!({
            "ok": {
                "action_id": 0,
                "x": null,
                "y": null,
                "telemetry": {
                    "policy": policy.policy_name(),
                    "terminal_reset": true,
                }
            }
        }));
    }

    let decision = policy.choose_action(&observation)?;
    ensure!(
        observation
            .available_actions
            .contains(&decision.chosen.action.id),
        "policy chose unavailable action {}",
        decision.chosen.action.id
    );
    let action = decision.chosen.action.clone();
    let telemetry = decision_telemetry(&decision);
    let response = json!({
        "ok": {
            "action_id": action.id,
            "x": action.x,
            "y": action.y,
            "telemetry": telemetry,
        }
    });
    stream.pending = Some(PendingServeAction::Action { action, telemetry });
    Ok(response)
}

fn finish_serve_session<P: LivePolicy>(
    streams: &mut BTreeMap<(String, String), ServeStream>,
    policy: &mut P,
) -> Result<()> {
    for stream in streams.values_mut() {
        if stream.policy_started {
            policy.on_game_end("serve_session_finished");
            stream.policy_started = false;
        }
    }
    policy.finish_session()?;
    for stream in streams.values_mut() {
        if let Some(recording) = stream.recording.take() {
            recording.finish()?;
        }
    }
    Ok(())
}

fn write_serve_error(writer: &mut impl Write, message: impl Into<String>) -> Result<()> {
    write_json_line(writer, &json!({ "err": { "message": message.into() } }))
}

fn write_json_line(writer: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value).context("serialize protocol response")?;
    writer.write_all(b"\n").context("write protocol newline")?;
    writer.flush().context("flush protocol response")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2::arc3::import_recordings_dir;
    use crate::p2::arc3_live::{ActionDecision, ActionScore};
    use crate::p2::model::WorldModel;
    use crate::p2::train::{reinit_varmap_deterministic, TrainConfig};
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use std::fs;
    use std::io::Cursor;

    fn wire_observation(game_id: &str, guid: &str, state: &str, actions: &[u8]) -> Value {
        json!({
            "game_id": game_id,
            "guid": guid,
            "frame": [vec![vec![0u8; 64]; 64]],
            "state": state,
            "levels_completed": if state == "WIN" { 1 } else { 0 },
            "win_levels": 1,
            "available_actions": actions,
            "full_reset": false,
        })
    }

    fn response_lines(values: &[Value]) -> Cursor<Vec<u8>> {
        let mut bytes = Vec::new();
        for value in values {
            serde_json::to_writer(&mut bytes, value).unwrap();
            bytes.push(b'\n');
        }
        Cursor::new(bytes)
    }

    #[test]
    fn stdio_arc_api_round_trips_wire_format_and_maps_errors() -> Result<()> {
        let observation = wire_observation("game", "guid", "NOT_FINISHED", &[1, 6]);
        let responses = response_lines(&[
            json!({ "ok": { "games": [{ "game_id": "game", "title": "Game" }] } }),
            json!({ "ok": { "card_id": "card" } }),
            json!({ "ok": { "observation": observation } }),
            json!({ "ok": { "observation": observation } }),
            json!({ "ok": { "observation": observation } }),
            json!({ "ok": { "scorecard": { "games": [] } } }),
        ]);
        let mut output = Vec::new();
        {
            let mut api = StdioArcApi::new(responses, &mut output);
            assert_eq!(api.list_games()?[0].game_id, "game");
            assert_eq!(api.open_scorecard(&json!({ "ignored": true }))?, "card");
            api.reset("game", "card", None)
                .map_err(anyhow::Error::new)?;
            api.act(
                "game",
                "guid",
                &ArcAction::new(6, Some(7), Some(9))?,
                &json!({}),
            )
            .map_err(anyhow::Error::new)?;
            api.act("game", "guid", &ArcAction::new(1, None, None)?, &json!({}))
                .map_err(anyhow::Error::new)?;
            assert_eq!(api.close_scorecard("card")?, json!({ "games": [] }));
        }
        let requests = String::from_utf8(output)?
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(requests[0], json!({ "op": "list_games" }));
        assert_eq!(requests[1], json!({ "op": "open_scorecard" }));
        assert_eq!(
            requests[2],
            json!({ "op": "reset", "card_id": "card", "game_id": "game", "guid": null })
        );
        assert_eq!(requests[3]["data"], json!({ "x": 7, "y": 9 }));
        assert_eq!(requests[4]["data"], json!({}));
        assert_eq!(
            requests[5],
            json!({ "op": "close_scorecard", "card_id": "card" })
        );

        let errors = b"{\"err\":{\"message\":\"nope\",\"kind\":\"failed\"}}\n{\"err\":{\"message\":\"maybe\",\"kind\":\"ambiguous\"}}\nnot-json\n";
        let mut api = StdioArcApi::new(Cursor::new(errors), Vec::new());
        assert!(matches!(
            api.reset("game", "card", None),
            Err(MutationError::Failed(_))
        ));
        assert!(matches!(
            api.act("game", "guid", &ArcAction::new(1, None, None)?, &json!({})),
            Err(MutationError::Ambiguous(_))
        ));
        assert!(api.list_games().is_err());
        assert!(api
            .take_fatal_error()
            .is_some_and(|error| error.contains("parse bridge response")));

        let empty_guid = wire_observation("game", "", "NOT_FINISHED", &[1]);
        let mut api = StdioArcApi::new(
            response_lines(&[json!({ "ok": { "observation": empty_guid } })]),
            Vec::new(),
        );
        assert!(matches!(
            api.reset("game", "card", None),
            Err(MutationError::Ambiguous(_))
        ));
        assert!(api.take_fatal_error().is_none());
        Ok(())
    }

    fn tiny_policy_fixture() -> Result<(TrainConfig, Device, VarMap, WorldModel)> {
        let train_config = TrainConfig {
            hidden_dim: 8,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            physical_batch: 4,
            ..TrainConfig::default()
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = WorldModel::new(
            train_config.model_config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        reinit_varmap_deterministic(&varmap, train_config.seed)?;
        Ok((train_config, device, varmap, model))
    }

    #[test]
    fn serve_loop_uses_model_policy_and_returns_legal_actions() -> Result<()> {
        let (_config, device, _varmap, model) = tiny_policy_fixture()?;
        let mut policy = ModelPolicy::new(&model, &device, 4, 4, 32, DEFAULT_TRIED_PENALTY);
        let profile_root = std::env::temp_dir().join(format!(
            "tofy-arc3-bridge-serve-profile-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&profile_root);
        policy.enable_streaming_eval_profile(
            &profile_root,
            "tofy.p2.arc3.bridge-test".into(),
            "cpu",
        );
        let observation = wire_observation("game", "", "NOT_FINISHED", &[1, 6]);
        let terminal = wire_observation("game", "", "WIN", &[]);
        let requests = response_lines(&[
            json!({ "op": "observe", "observation": observation }),
            json!({ "op": "observe", "observation": observation }),
            json!({ "op": "observe", "observation": terminal }),
            json!({ "op": "shutdown" }),
        ]);
        let mut input = b"not-json\n".to_vec();
        input.extend(requests.into_inner());
        let mut output = Vec::new();
        run_serve_loop(Cursor::new(input), &mut output, &mut policy, None)?;
        let lines = String::from_utf8(output)?
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(
            lines[0],
            json!({ "ready": { "protocol": 1, "mode": "serve" } })
        );
        assert!(lines[1]["err"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("unparseable request")));
        for response in &lines[2..4] {
            let ok = &response["ok"];
            let action = ok["action_id"].as_u64().context("action id")? as u8;
            assert!([1, 6].contains(&action));
            assert_eq!(ok["x"].is_number(), action == 6);
            assert_eq!(ok["y"].is_number(), action == 6);
            assert!(ok["telemetry"].is_object());
        }
        assert_eq!(lines[4]["ok"]["action_id"], 0);
        assert!(lines[4]["ok"]["x"].is_null());
        assert!(lines[4]["ok"]["y"].is_null());
        assert!(lines[4]["ok"]["telemetry"].is_object());
        assert_eq!(lines[5], json!({ "ok": { "bye": true } }));
        candle_graph::verify_bundle(&profile_root.join("profile/arc3-game"))?;
        assert!(profile_root.join("profile/arc3-campaign.json").is_file());
        fs::remove_dir_all(profile_root)?;
        Ok(())
    }

    struct FirstPolicy;

    impl LivePolicy for FirstPolicy {
        fn choose_action(&mut self, observation: &ArcObservation) -> Result<ActionDecision> {
            let action = ArcAction::new(observation.available_actions[0], None, None)?;
            Ok(ActionDecision {
                chosen: ActionScore {
                    action,
                    score: 1.0,
                    q_probability: 1.0,
                    reliability_probability: 1.0,
                    noop_probability: 0.0,
                    predicted_effect: 1.0,
                },
                candidate_count: 1,
                phase_a: None,
                adaptation: None,
                context_len: 0,
                context_scope: ContextScopeKind::default(),
            })
        }
    }

    struct LifecyclePolicy {
        events: Vec<String>,
    }

    impl LivePolicy for LifecyclePolicy {
        fn choose_action(&mut self, observation: &ArcObservation) -> Result<ActionDecision> {
            let mut first = FirstPolicy;
            first.choose_action(observation)
        }

        fn on_game_start(&mut self, game_id: &str) {
            self.events.push(format!("start:{game_id}"));
        }

        fn on_reset_retry(&mut self, reason: &str) {
            self.events.push(format!("retry:{reason}"));
        }

        fn on_game_end(&mut self, outcome: &str) {
            self.events.push(format!("end:{outcome}"));
        }
    }

    #[test]
    fn serve_loop_preserves_live_policy_lifecycle_callbacks() -> Result<()> {
        let input = response_lines(&[
            json!({
                "op": "observe",
                "observation": wire_observation("game", "guid", "NOT_STARTED", &[])
            }),
            json!({
                "op": "observe",
                "observation": wire_observation("game", "guid", "NOT_FINISHED", &[1])
            }),
            json!({
                "op": "observe",
                "observation": wire_observation("game", "guid", "GAME_OVER", &[])
            }),
            json!({
                "op": "observe",
                "observation": wire_observation("game", "guid", "NOT_FINISHED", &[1])
            }),
            json!({
                "op": "observe",
                "observation": wire_observation("game", "guid", "WIN", &[])
            }),
            json!({ "op": "shutdown" }),
        ]);
        let mut output = Vec::new();
        let mut policy = LifecyclePolicy { events: Vec::new() };
        run_serve_loop(input, &mut output, &mut policy, None)?;
        let lines = String::from_utf8(output)?
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(lines[1]["ok"]["action_id"], 0);
        assert_eq!(lines[2]["ok"]["action_id"], 1);
        assert_eq!(lines[3]["ok"]["action_id"], 0);
        assert_eq!(lines[4]["ok"]["action_id"], 1);
        assert_eq!(lines[5]["ok"]["action_id"], 0);
        assert_eq!(lines[6], json!({ "ok": { "bye": true } }));
        assert_eq!(
            policy.events,
            ["start:game", "retry:serve_game_over", "end:completed"]
        );
        Ok(())
    }

    #[derive(Default)]
    struct TransitionPolicy {
        transitions: Vec<(u16, u8, u16, bool)>,
    }

    impl LivePolicy for TransitionPolicy {
        fn choose_action(&mut self, observation: &ArcObservation) -> Result<ActionDecision> {
            let mut first = FirstPolicy;
            first.choose_action(observation)
        }

        fn on_confirmed_transition(
            &mut self,
            current: &ArcObservation,
            action: &crate::p2::data::ArcAction,
            next: &ArcObservation,
        ) {
            self.transitions.push((
                current.levels_completed,
                action.id,
                next.levels_completed,
                current.frame != next.frame,
            ));
        }
    }

    #[test]
    fn serve_loop_reports_confirmed_transitions_to_the_policy() -> Result<()> {
        let mut changed = wire_observation("game", "guid", "NOT_FINISHED", &[1]);
        changed["frame"][0][0][0] = json!(3);
        let input = response_lines(&[
            json!({
                "op": "observe",
                "observation": wire_observation("game", "guid", "NOT_FINISHED", &[1])
            }),
            json!({ "op": "observe", "observation": changed }),
            json!({
                "op": "observe",
                "observation": wire_observation("game", "guid", "GAME_OVER", &[])
            }),
            json!({
                "op": "observe",
                "observation": wire_observation("game", "guid", "NOT_FINISHED", &[1])
            }),
            json!({
                "op": "observe",
                "observation": wire_observation("game", "guid", "WIN", &[])
            }),
            json!({ "op": "shutdown" }),
        ]);
        let mut output = Vec::new();
        let mut policy = TransitionPolicy::default();
        run_serve_loop(input, &mut output, &mut policy, None)?;
        // Every ACTION response is a factual transition; the RESET after
        // GAME_OVER is not (no action was chosen from the terminal frame).
        assert_eq!(
            policy.transitions,
            vec![(0, 1, 0, true), (0, 1, 0, true), (0, 1, 1, false)]
        );
        Ok(())
    }

    #[test]
    fn bridge_recording_round_trips_through_importer() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "tofy-arc3-bridge-recording-roundtrip-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let mut changed = wire_observation("game", "guid", "WIN", &[]);
        changed["frame"][0][0][0] = json!(3);
        let input = response_lines(&[
            json!({
                "op": "observe",
                "observation": wire_observation("game", "guid", "NOT_FINISHED", &[1])
            }),
            json!({ "op": "observe", "observation": changed }),
            json!({ "op": "shutdown" }),
        ]);
        let mut output = Vec::new();
        run_serve_loop(input, &mut output, &mut FirstPolicy, Some(&root))?;
        let imported = import_recordings_dir(&root)?;
        assert_eq!(imported.len(), 1);
        assert_eq!(imported[0].action.id, 1);
        assert_eq!(imported[0].current.pixel(0, 0), Some(0));
        assert_eq!(imported[0].next.pixel(0, 0), Some(3));
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    #[ignore = "writes an orchestrator fixture when TOFY_FIXTURE_DIR is set"]
    fn emit_tiny_bridge_fixture() -> Result<()> {
        let Some(root) = std::env::var_os("TOFY_FIXTURE_DIR").map(PathBuf::from) else {
            return Ok(());
        };
        fs::create_dir_all(&root)?;
        let (mut train_config, _device, varmap, _model) = tiny_policy_fixture()?;
        train_config.output_dir = root.clone();
        varmap.save(root.join("ema.safetensors"))?;
        fs::write(
            root.join("config.json"),
            serde_json::to_vec_pretty(&train_config)?,
        )?;
        load_model(&train_config, &root.join("ema.safetensors"), &Device::Cpu)?;
        Ok(())
    }
}
