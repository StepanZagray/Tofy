//! Held-out live evaluation on every public ARC-AGI-3 environment.
//!
//! This module is deliberately downstream of training. It can load a frozen
//! checkpoint and submit actions, but exposes no samples, gradients, optimizer,
//! checkpoint selection, or curriculum hooks back to `p2::train`.

use crate::gpu_lock::GpuSessionGuard;
use crate::p2::adaptation::{
    context_batch_for, AdaptationMode, AdaptationTrace, FastWeightAdapter, LiveContext,
};
use crate::p2::agent_session::AgentSession;
use crate::p2::arc3::{first_recorded_decision_observations, ParsedActionInput, RecordingEvent};
use crate::p2::cg_profile::{
    ensure_eval_profile_campaign, EvalCaptureSpec, RepresentativeUpdateCapture,
    ARC3_PROFILE_ENTRYPOINT,
};
use crate::p2::data::ContextTransition;
use crate::p2::data::{palette, ArcAction, ArcFrame, FRAME_SIDE, GOAL_FEATURES_DIM};
use crate::p2::eval::load_model;
use crate::p2::model::{
    latent_mse_per_sample, unknown_operator_conditioning, RecursionDepth, RecursionOpts,
    WorldModel, EVENT_NOOP,
};
use crate::p2::rhae::{
    benchmark_from_scorecard_str, official_rhae_from_benchmark, ScorecardBenchmark,
};
use crate::p2::train::{frames_to_indices, load_train_config, resolve_device};
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_graph::{ExecutionStep, PlannedCapture, SpanKind};
use candle_nn::ops;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest::StatusCode;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const LIVE_REPORT_SCHEMA: &str = "p2.arc3_live_report.v4";
pub const LIVE_POLICY: &str = "model_reliable_effect_v1";

/// Which controller drives live decisions. `Greedy` is the historical
/// searchless policy and stays byte-equivalent; `PhaseA` is ADR 0004.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LivePolicyKind {
    #[default]
    Greedy,
    PhaseA,
}

/// Report-facing description of the deployed controller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyContract {
    pub name: String,
    pub limitation: String,
    pub goal_feature_contract: String,
}

impl PolicyContract {
    pub fn greedy() -> Self {
        Self {
            name: LIVE_POLICY.into(),
            limitation: POLICY_LIMITATION.into(),
            goal_feature_contract: GOAL_FEATURE_CONTRACT.into(),
        }
    }

    pub fn phase_a() -> Self {
        Self {
            name: crate::p2::arc3_phase_a::PHASE_A_POLICY.into(),
            limitation: crate::p2::arc3_phase_a::PHASE_A_POLICY_LIMITATION.into(),
            goal_feature_contract: crate::p2::arc3_phase_a::PHASE_A_GOAL_FEATURE_CONTRACT.into(),
        }
    }

    pub fn for_kind(kind: LivePolicyKind) -> Self {
        match kind {
            LivePolicyKind::Greedy => Self::greedy(),
            LivePolicyKind::PhaseA => Self::phase_a(),
        }
    }
}
const POLICY_LIMITATION: &str = "The checkpoint predicts composed next-frame accuracy, latent self-confidence, no-op probability, and latent action effect; it has no trained reward/value head. Real games provide no synthetic episode operator, so inference uses the UNKNOWN rule token with neutral colors. This exploratory policy is not a hidden-goal solver.";
const GOAL_FEATURE_CONTRACT: &str = "Live policy supplies the all-zero goal vector. Foundation-v2 trains with 30% goal dropout, so this goal-free query is in-distribution; it does not provide hidden-goal evidence.";
const TRIED_ACTION_KEY_CONTRACT: &str = "game id + session guid + levels completed + frame dimensions + visible pixels; row 63 participates only when it contains non-background gameplay content";
const MAX_HTTP_ATTEMPTS: usize = 5;
/// Default cap on guid-scoped RESET retries per level after a recoverable
/// non-WIN terminal such as GAME_OVER.
pub const DEFAULT_MAX_LEVEL_RETRIES: usize = 3;
/// Default official-style action budget for each level.
pub const DEFAULT_MAX_ACTIONS_PER_LEVEL: u32 = 512;
/// Default score penalty for already-tried actions. The policy score is a
/// convex mixture in [0, 1]; 0.25 demotes near-ties toward exploration while
/// still letting a tried action win again once its margin over every untried
/// candidate exceeds a quarter of the score range. The former hard penalty of
/// 1.0 swamped the whole range and turned "tried once" into "never again".
pub const DEFAULT_TRIED_PENALTY: f64 = 0.25;

#[derive(Debug, Clone)]
pub struct LiveEvalConfig {
    pub checkpoint: PathBuf,
    pub train_config: PathBuf,
    pub device: String,
    pub base_url: String,
    pub api_key_env: String,
    pub games: Vec<String>,
    pub physical_batch: usize,
    pub action6_max_candidates: usize,
    pub action6_grid_stride: usize,
    pub request_timeout_secs: u64,
    pub driver: LiveDriverOptions,
    pub output: PathBuf,
    pub recordings_dir: PathBuf,
    pub profile_eval: bool,
    /// Decision controller; defaults to the byte-equivalent greedy policy.
    pub policy: LivePolicyKind,
    /// ADR 0004 Phase A calibration artifact; absent => fail-closed frontier mode.
    pub phase_a_calibration: Option<PathBuf>,
    /// ADR 0005 §6.2 Channel B test-time adaptation of the fast-weight subset.
    pub adapt: bool,
    /// Preregistered carry arm: adapted fast weights persist across levels.
    pub adapt_carry: bool,
    /// ADR 0005 §6.1 Channel A (`--context-window`): `None` = default (on for
    /// `world_core_v6`); always off on checkpoints without a context channel.
    pub context_window: Option<bool>,
}

impl LiveEvalConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.physical_batch > 0, "physical_batch must be > 0");
        ensure!(
            self.adapt || !self.adapt_carry,
            "--adapt-carry requires --adapt"
        );
        ensure!(
            self.action6_max_candidates > 0,
            "action6_max_candidates must be > 0"
        );
        ensure!(
            (1..=FRAME_SIDE).contains(&self.action6_grid_stride),
            "action6_grid_stride must be in 1..={FRAME_SIDE}"
        );
        ensure!(
            self.request_timeout_secs > 0,
            "request_timeout_secs must be > 0"
        );
        ensure!(
            !self.base_url.trim().is_empty(),
            "base_url must not be empty"
        );
        ensure!(
            !self.api_key_env.trim().is_empty(),
            "api_key_env must not be empty"
        );
        ensure!(
            !self.recordings_dir.as_os_str().is_empty(),
            "recordings_dir must not be empty"
        );
        self.driver.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveDriverOptions {
    /// Competition mode: RESET always retries the current level and earlier
    /// levels cannot be revisited, so a guid-scoped retry is always safe.
    #[serde(default = "default_competition_mode")]
    pub competition_mode: bool,
    /// Maximum guid-scoped RESET retries per level after a recoverable
    /// non-WIN terminal such as GAME_OVER.
    #[serde(default = "default_max_level_retries")]
    pub max_level_retries: usize,
    /// Primary action budget, applied independently to every level.
    #[serde(default = "default_max_actions_per_level")]
    pub max_actions_per_level: u32,
    /// Optional game-wide emergency stop, not a scoring budget.
    #[serde(default)]
    pub max_actions_per_game: Option<u32>,
    /// Score penalty subtracted from already-tried actions within one
    /// tried-state key. See [`DEFAULT_TRIED_PENALTY`].
    #[serde(default = "default_tried_penalty")]
    pub tried_penalty: f64,
    /// Permit an otherwise refused dirty driver repair to open a scorecard.
    #[serde(default)]
    pub exploratory: bool,
}

fn default_max_level_retries() -> usize {
    DEFAULT_MAX_LEVEL_RETRIES
}

fn default_competition_mode() -> bool {
    true
}

fn default_max_actions_per_level() -> u32 {
    DEFAULT_MAX_ACTIONS_PER_LEVEL
}

fn default_tried_penalty() -> f64 {
    DEFAULT_TRIED_PENALTY
}

impl Default for LiveDriverOptions {
    fn default() -> Self {
        Self {
            competition_mode: true,
            max_level_retries: DEFAULT_MAX_LEVEL_RETRIES,
            max_actions_per_level: DEFAULT_MAX_ACTIONS_PER_LEVEL,
            max_actions_per_game: None,
            tried_penalty: DEFAULT_TRIED_PENALTY,
            exploratory: false,
        }
    }
}

impl LiveDriverOptions {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.max_actions_per_level > 0,
            "max_actions_per_level must be > 0"
        );
        ensure!(
            self.max_actions_per_game.is_none_or(|cap| cap > 0),
            "max_actions_per_game must be > 0 when set"
        );
        ensure!(
            self.tried_penalty.is_finite() && (0.0..=1.0).contains(&self.tried_penalty),
            "tried_penalty must be finite and in [0,1]"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicGame {
    pub game_id: String,
    pub title: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcObservation {
    pub game_id: String,
    pub guid: String,
    pub frame: ArcFrame,
    /// Every API-native animation layer in order. `frame` is the final layer
    /// padded to the fixed 64x64 model canvas. Empty for legacy serialized data.
    #[serde(default)]
    pub animation: Vec<ArcFrame>,
    /// True when RESET replaced the whole game session rather than retrying
    /// the current level. Legacy observations predate this API field.
    #[serde(default)]
    pub full_reset: bool,
    pub state: String,
    pub levels_completed: u16,
    pub win_levels: u16,
    pub available_actions: Vec<u8>,
}

impl ArcObservation {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(!self.game_id.is_empty(), "ARC response has empty game_id");
        ensure!(
            matches!(
                self.state.as_str(),
                "NOT_FINISHED" | "NOT_STARTED" | "WIN" | "GAME_OVER"
            ),
            "unknown ARC state {}",
            self.state
        );
        let unique: BTreeSet<_> = self.available_actions.iter().copied().collect();
        ensure!(
            unique.len() == self.available_actions.len(),
            "ARC response contains duplicate available actions"
        );
        ensure!(
            self.available_actions.iter().all(|id| (1..=7).contains(id)),
            "ARC response contains an action outside 1..=7"
        );
        if self.state == "NOT_FINISHED" {
            ensure!(
                !self.available_actions.is_empty(),
                "non-terminal ARC response has no available actions"
            );
        }
        Ok(())
    }

    fn won(&self) -> bool {
        self.state == "WIN" || (self.win_levels > 0 && self.levels_completed >= self.win_levels)
    }

    pub(crate) fn terminal(&self) -> bool {
        self.state != "NOT_FINISHED" || self.won()
    }
}

#[derive(Debug)]
struct PendingRecordingEvent {
    parsed: RecordingEvent,
    frame_layers: Vec<ArcFrame>,
}

#[derive(Debug)]
pub(crate) struct LiveRecordingRun {
    final_path: PathBuf,
    events: Vec<PendingRecordingEvent>,
}

impl LiveRecordingRun {
    pub(crate) fn start(root: &Path, observation: &ArcObservation) -> Result<Self> {
        let game = recording_component(&observation.game_id);
        let session = if observation.guid.is_empty() {
            format!("run-{}", unix_ms())
        } else {
            recording_component(&observation.guid)
        };
        let final_path = root.join(game).join(format!("{session}.jsonl"));
        ensure!(
            !final_path.exists(),
            "refusing to overwrite live recording {}",
            final_path.display()
        );
        let mut run = Self {
            final_path,
            events: Vec::new(),
        };
        run.push(observation, Some(reset_recording_action()))?;
        Ok(run)
    }

    pub(crate) fn push_reset(&mut self, observation: &ArcObservation) -> Result<()> {
        self.push(observation, Some(reset_recording_action()))
    }

    pub(crate) fn push_action(
        &mut self,
        observation: &ArcObservation,
        action: &ArcAction,
        reasoning: &Value,
    ) -> Result<()> {
        self.push(
            observation,
            Some(ParsedActionInput {
                action: action.clone(),
                is_reset: false,
                reasoning: Some(reasoning.clone()),
            }),
        )
    }

    fn push(
        &mut self,
        observation: &ArcObservation,
        action: Option<ParsedActionInput>,
    ) -> Result<()> {
        observation.validate()?;
        let frame_layers = if observation.animation.is_empty() {
            vec![observation.frame.clone()]
        } else {
            observation.animation.clone()
        };
        self.events.push(PendingRecordingEvent {
            parsed: RecordingEvent {
                timestamp: unix_ms().to_string(),
                game_id: observation.game_id.clone(),
                state: observation.state.clone(),
                levels_completed: i64::from(observation.levels_completed),
                win_levels: i64::from(observation.win_levels),
                action,
                guid: observation.guid.clone(),
                full_reset: observation.full_reset,
                available_actions: observation
                    .available_actions
                    .iter()
                    .map(|value| i64::from(*value))
                    .collect(),
                frame: observation.frame.clone(),
                source_path: self.final_path.clone(),
                line: self.events.len() + 1,
            },
            frame_layers,
        });
        Ok(())
    }

    #[cfg(test)]
    fn parsed_events(&self) -> Vec<RecordingEvent> {
        self.events
            .iter()
            .map(|event| event.parsed.clone())
            .collect()
    }

    pub(crate) fn finish(self) -> Result<PathBuf> {
        let parent = self
            .final_path
            .parent()
            .context("live recording path has no parent")?;
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        ensure!(
            !self.final_path.exists(),
            "refusing to overwrite live recording {}",
            self.final_path.display()
        );
        let temporary = parent.join(format!(
            ".{}.tmp-{}-{}",
            self.final_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("recording.jsonl"),
            std::process::id(),
            unix_ms()
        ));
        let write_result = (|| -> Result<()> {
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)
                .with_context(|| format!("create {}", temporary.display()))?;
            for event in &self.events {
                serde_json::to_writer(&mut file, &recording_envelope(event))
                    .with_context(|| format!("serialize {}", temporary.display()))?;
                file.write_all(b"\n")
                    .with_context(|| format!("write {}", temporary.display()))?;
            }
            file.flush()
                .with_context(|| format!("flush {}", temporary.display()))?;
            file.sync_all()
                .with_context(|| format!("sync {}", temporary.display()))?;
            Ok(())
        })();
        if let Err(error) = write_result {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        fs::rename(&temporary, &self.final_path).with_context(|| {
            format!(
                "rename {} -> {}",
                temporary.display(),
                self.final_path.display()
            )
        })?;
        fs::File::open(parent)
            .with_context(|| format!("open {} for sync", parent.display()))?
            .sync_all()
            .with_context(|| format!("sync {}", parent.display()))?;
        Ok(self.final_path)
    }
}

fn reset_recording_action() -> ParsedActionInput {
    ParsedActionInput {
        action: ArcAction {
            id: 1,
            x: None,
            y: None,
        },
        is_reset: true,
        reasoning: None,
    }
}

fn recording_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() || matches!(sanitized.as_str(), "." | "..") {
        "unnamed".into()
    } else {
        sanitized
    }
}

fn frame_layers_json(layers: &[ArcFrame]) -> Value {
    Value::Array(
        layers
            .iter()
            .map(|frame| {
                Value::Array(
                    frame
                        .pixels
                        .chunks(frame.width as usize)
                        .map(|row| Value::Array(row.iter().map(|pixel| json!(pixel)).collect()))
                        .collect(),
                )
            })
            .collect(),
    )
}

fn recording_action_json(action: &ParsedActionInput) -> Value {
    if action.is_reset {
        return json!({ "id": "RESET", "data": {} });
    }
    let mut data = json!({});
    if action.action.id == 6 {
        data["x"] = json!(action.action.x);
        data["y"] = json!(action.action.y);
    }
    let mut value = json!({
        "id": format!("ACTION{}", action.action.id),
        "data": data,
    });
    if let Some(reasoning) = &action.reasoning {
        value["reasoning"] = reasoning.clone();
    }
    value
}

fn recording_envelope(pending: &PendingRecordingEvent) -> Value {
    let event = &pending.parsed;
    json!({
        "timestamp": event.timestamp,
        "data": {
            "game_id": event.game_id,
            "state": event.state,
            "levels_completed": event.levels_completed,
            "win_levels": event.win_levels,
            "action_input": event.action.as_ref().map(recording_action_json),
            "guid": event.guid,
            "full_reset": event.full_reset,
            "available_actions": event.available_actions,
            "frame": frame_layers_json(&pending.frame_layers),
        }
    })
}

#[derive(Debug, Deserialize)]
pub(crate) struct ApiObservation {
    game_id: String,
    guid: String,
    frame: Vec<Vec<Vec<u8>>>,
    full_reset: bool,
    state: String,
    levels_completed: u16,
    win_levels: u16,
    available_actions: Vec<u8>,
}

impl ApiObservation {
    pub(crate) fn into_arc_observation(self, require_fixed_64: bool) -> Result<ArcObservation> {
        let value = self;
        ensure!(!value.frame.is_empty(), "ARC response contains no frames");
        let mut animation = Vec::with_capacity(value.frame.len());
        for layer in &value.frame {
            ensure!(!layer.is_empty(), "ARC response contains an empty frame");
            let width = layer[0].len();
            ensure!(width > 0, "ARC response contains a zero-width frame");
            ensure!(
                width <= 64 && layer.len() <= 64,
                "ARC response frame is {}x{}, exceeding the supported 64x64 canvas",
                width,
                layer.len()
            );
            if require_fixed_64 {
                ensure!(
                    width == FRAME_SIDE && layer.len() == FRAME_SIDE,
                    "bridge observation frame is {}x{}; expected {FRAME_SIDE}x{FRAME_SIDE}",
                    width,
                    layer.len()
                );
            }
            ensure!(
                layer.iter().all(|row| row.len() == width),
                "ARC response frame is ragged"
            );
            let pixels = layer.iter().flatten().copied().collect();
            animation.push(ArcFrame::new(width as u16, layer.len() as u16, pixels)?);
        }
        let frame = animation
            .last()
            .cloned()
            .context("ARC response contains no frames")?
            .to_fixed_64()?;
        let observation = ArcObservation {
            game_id: value.game_id,
            guid: value.guid,
            frame,
            animation,
            full_reset: value.full_reset,
            state: value.state,
            levels_completed: value.levels_completed,
            win_levels: value.win_levels,
            available_actions: value.available_actions,
        };
        observation.validate()?;
        Ok(observation)
    }
}

impl TryFrom<ApiObservation> for ArcObservation {
    type Error = anyhow::Error;

    fn try_from(value: ApiObservation) -> Result<Self> {
        ensure!(!value.guid.is_empty(), "ARC response has empty guid");
        value.into_arc_observation(false)
    }
}

pub trait ArcApi {
    fn list_games(&mut self) -> Result<Vec<PublicGame>>;
    fn open_scorecard(&mut self, metadata: &Value) -> Result<String>;
    /// RESET. Without `guid` this opens/wipes a game session. With `guid`
    /// (an existing session) the official semantics restart the current level
    /// when an ACTION happened since the last RESET or level transition, and
    /// otherwise reset the game; in competition mode a guid-scoped RESET
    /// always retries the current level.
    fn reset(
        &mut self,
        game_id: &str,
        card_id: &str,
        guid: Option<&str>,
    ) -> MutationResult<ArcObservation>;
    fn act(
        &mut self,
        game_id: &str,
        guid: &str,
        action: &ArcAction,
        reasoning: &Value,
    ) -> MutationResult<ArcObservation>;
    fn close_scorecard(&mut self, card_id: &str) -> Result<Value>;
}

#[derive(Debug, Clone, Copy)]
enum RetryClass {
    IdempotentRead,
    AtMostOnceMutation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguousMutation {
    pub operation: String,
    pub game_id: Option<String>,
    pub guid: Option<String>,
    pub action: Option<ArcAction>,
    pub cause: String,
}

impl std::fmt::Display for AmbiguousMutation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "ambiguous {}: {}", self.operation, self.cause)
    }
}

impl std::error::Error for AmbiguousMutation {}

#[derive(Debug)]
pub enum MutationError {
    Ambiguous(AmbiguousMutation),
    Failed(anyhow::Error),
}

impl std::fmt::Display for MutationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ambiguous(error) => error.fmt(formatter),
            Self::Failed(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for MutationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Ambiguous(error) => Some(error),
            Self::Failed(error) => error.source(),
        }
    }
}

pub type MutationResult<T> = std::result::Result<T, MutationError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HttpMethod {
    Get,
    Post,
}

#[derive(Debug, Clone)]
struct HttpRequest {
    method: HttpMethod,
    url: String,
    body: Option<Value>,
}

#[derive(Debug)]
struct HttpResponse {
    status: StatusCode,
    body: std::result::Result<String, String>,
}

trait HttpTransport {
    fn send(&self, request: HttpRequest) -> std::result::Result<HttpResponse, String>;
}

struct ReqwestTransport {
    client: Client,
}

impl HttpTransport for ReqwestTransport {
    fn send(&self, request: HttpRequest) -> std::result::Result<HttpResponse, String> {
        let builder = match request.method {
            HttpMethod::Get => self.client.get(&request.url),
            HttpMethod::Post => self
                .client
                .post(&request.url)
                .json(&request.body.expect("POST requests include a body")),
        };
        let response = builder.send().map_err(|error| error.to_string())?;
        let status = response.status();
        let body = response.text().map_err(|error| error.to_string());
        Ok(HttpResponse { status, body })
    }
}

#[derive(Debug)]
enum RequestFailure {
    Transport(String),
    ResponseBody(String),
    Status { status: StatusCode, body: String },
    Parse(String),
}

impl RequestFailure {
    fn retryable_read(&self) -> bool {
        match self {
            Self::Transport(_) | Self::ResponseBody(_) => true,
            Self::Status { status, .. } => {
                *status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
            }
            Self::Parse(_) => false,
        }
    }

    fn ambiguous_action(&self) -> bool {
        match self {
            Self::Transport(_) | Self::ResponseBody(_) | Self::Parse(_) => true,
            Self::Status { status, .. } => {
                *status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
            }
        }
    }

    fn operation_error(self, operation: &str) -> anyhow::Error {
        match self {
            Self::Transport(cause) => anyhow::anyhow!("ARC {operation}: {cause}"),
            Self::ResponseBody(cause) => {
                anyhow::anyhow!("read ARC {operation} response: {cause}")
            }
            Self::Status { status, body } => {
                anyhow::anyhow!("ARC {operation} returned HTTP {status}: {body}")
            }
            Self::Parse(cause) => anyhow::anyhow!("parse ARC {operation} response: {cause}"),
        }
    }
}

struct HttpArcApi<T = ReqwestTransport> {
    transport: T,
    base_url: String,
}

impl HttpArcApi<ReqwestTransport> {
    pub fn from_env(base_url: &str, api_key_env: &str, timeout: Duration) -> Result<Self> {
        let _ = dotenvy::dotenv();
        let mut names = vec![
            api_key_env,
            "ARC_API_KEY",
            "ARC_AGI_3_API_KEY",
            "ARC_AGI_API",
        ];
        names.dedup();
        let api_key = names
            .iter()
            .find_map(|name| env::var(name).ok().filter(|value| !value.trim().is_empty()))
            .with_context(|| {
                format!(
                    "missing ARC API key; set {} in the environment or .env",
                    names.join(" or ")
                )
            })?;
        ensure!(!api_key.trim().is_empty(), "{api_key_env} is empty");

        let mut headers = HeaderMap::new();
        let mut value =
            HeaderValue::from_str(&api_key).context("API key is not a valid header value")?;
        value.set_sensitive(true);
        headers.insert("X-API-Key", value);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let client = Client::builder()
            .cookie_store(true)
            .default_headers(headers)
            .timeout(timeout)
            .user_agent("tofy-p2-arc3-live-eval/1")
            .build()
            .context("build ARC HTTP client")?;
        Ok(Self {
            transport: ReqwestTransport { client },
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }
}

impl<T: HttpTransport> HttpArcApi<T> {
    fn request_json<R: DeserializeOwned>(
        &self,
        operation: &str,
        retry_class: RetryClass,
        request: HttpRequest,
    ) -> std::result::Result<R, RequestFailure> {
        let attempts = match retry_class {
            RetryClass::IdempotentRead => MAX_HTTP_ATTEMPTS,
            RetryClass::AtMostOnceMutation => 1,
        };
        let mut last_error = None;
        for attempt in 0..attempts {
            match self.send_json(request.clone()) {
                Ok(response) => return Ok(response),
                Err(error)
                    if matches!(retry_class, RetryClass::IdempotentRead)
                        && error.retryable_read()
                        && attempt + 1 < attempts =>
                {
                    last_error = Some(error);
                    thread::sleep(Duration::from_millis(200 * (1u64 << attempt)));
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error
            .unwrap_or_else(|| RequestFailure::Transport(format!("ARC {operation} failed"))))
    }

    fn send_json<R: DeserializeOwned>(
        &self,
        request: HttpRequest,
    ) -> std::result::Result<R, RequestFailure> {
        let response = self
            .transport
            .send(request)
            .map_err(RequestFailure::Transport)?;
        let status = response.status;
        let body = response.body.map_err(RequestFailure::ResponseBody)?;
        if status.is_success() {
            return serde_json::from_str(&body)
                .map_err(|error| RequestFailure::Parse(error.to_string()));
        }
        Err(RequestFailure::Status {
            status,
            body: body.chars().take(512).collect(),
        })
    }

    fn post<R: DeserializeOwned>(&self, path: &str, body: &Value, operation: &str) -> Result<R> {
        // No ARC POST endpoint in this client has an official idempotency guarantee.
        self.request_json(
            operation,
            RetryClass::AtMostOnceMutation,
            HttpRequest {
                method: HttpMethod::Post,
                url: format!("{}{path}", self.base_url),
                body: Some(body.clone()),
            },
        )
        .map_err(|error| error.operation_error(operation))
    }

    fn action_post<R: DeserializeOwned>(
        &self,
        path: &str,
        body: &Value,
        ambiguity: AmbiguousMutation,
    ) -> MutationResult<R> {
        match self.request_json(
            &ambiguity.operation,
            RetryClass::AtMostOnceMutation,
            HttpRequest {
                method: HttpMethod::Post,
                url: format!("{}{path}", self.base_url),
                body: Some(body.clone()),
            },
        ) {
            Ok(response) => Ok(response),
            Err(error) if error.ambiguous_action() => {
                Err(MutationError::Ambiguous(AmbiguousMutation {
                    cause: error.operation_error(&ambiguity.operation).to_string(),
                    ..ambiguity
                }))
            }
            Err(error) => Err(MutationError::Failed(
                error.operation_error(&ambiguity.operation),
            )),
        }
    }
}

impl<T: HttpTransport> ArcApi for HttpArcApi<T> {
    fn list_games(&mut self) -> Result<Vec<PublicGame>> {
        self.request_json(
            "list games",
            RetryClass::IdempotentRead,
            HttpRequest {
                method: HttpMethod::Get,
                url: format!("{}/api/games", self.base_url),
                body: None,
            },
        )
        .map_err(|error| error.operation_error("list games"))
    }

    fn open_scorecard(&mut self, metadata: &Value) -> Result<String> {
        #[derive(Deserialize)]
        struct OpenResponse {
            card_id: String,
        }
        let response: OpenResponse =
            self.post("/api/scorecard/open", metadata, "open scorecard")?;
        ensure!(
            !response.card_id.is_empty(),
            "ARC returned an empty card_id"
        );
        Ok(response.card_id)
    }

    fn reset(
        &mut self,
        game_id: &str,
        card_id: &str,
        guid: Option<&str>,
    ) -> MutationResult<ArcObservation> {
        let mut body = json!({ "game_id": game_id, "card_id": card_id });
        if let Some(guid) = guid {
            body["guid"] = json!(guid);
        }
        let ambiguity = AmbiguousMutation {
            operation: "reset game".into(),
            game_id: Some(game_id.into()),
            guid: guid.map(str::to_owned),
            action: None,
            cause: String::new(),
        };
        let response: ApiObservation =
            self.action_post("/api/cmd/RESET", &body, ambiguity.clone())?;
        let observation = ArcObservation::try_from(response).map_err(|error| {
            MutationError::Ambiguous(AmbiguousMutation {
                cause: format!("validate RESET response: {error:#}"),
                ..ambiguity.clone()
            })
        })?;
        if observation.game_id != game_id
            || guid.is_some_and(|expected| observation.guid != expected)
        {
            return Err(MutationError::Ambiguous(AmbiguousMutation {
                cause: "RESET response session identifiers do not match request".into(),
                ..ambiguity
            }));
        }
        Ok(observation)
    }

    fn act(
        &mut self,
        game_id: &str,
        guid: &str,
        action: &ArcAction,
        reasoning: &Value,
    ) -> MutationResult<ArcObservation> {
        let path = format!("/api/cmd/ACTION{}", action.id);
        let mut body = json!({
            "game_id": game_id,
            "guid": guid,
            "reasoning": reasoning,
        });
        if action.id == 6 {
            body["x"] = json!(action
                .x
                .context("ACTION6 missing x")
                .map_err(MutationError::Failed)?);
            body["y"] = json!(action
                .y
                .context("ACTION6 missing y")
                .map_err(MutationError::Failed)?);
        }
        let ambiguity = AmbiguousMutation {
            operation: "submit action".into(),
            game_id: Some(game_id.into()),
            guid: Some(guid.into()),
            action: Some(action.clone()),
            cause: String::new(),
        };
        let response: ApiObservation = self.action_post(&path, &body, ambiguity.clone())?;
        let observation = ArcObservation::try_from(response).map_err(|error| {
            MutationError::Ambiguous(AmbiguousMutation {
                cause: format!("validate ACTION response: {error:#}"),
                ..ambiguity.clone()
            })
        })?;
        if observation.game_id != game_id || observation.guid != guid {
            return Err(MutationError::Ambiguous(AmbiguousMutation {
                cause: "ACTION response session identifiers do not match request".into(),
                ..ambiguity
            }));
        }
        Ok(observation)
    }

    fn close_scorecard(&mut self, card_id: &str) -> Result<Value> {
        self.post(
            "/api/scorecard/close",
            &json!({ "card_id": card_id }),
            "close scorecard",
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionScore {
    pub action: ArcAction,
    pub score: f64,
    pub q_probability: f64,
    pub reliability_probability: f64,
    pub noop_probability: f64,
    pub predicted_effect: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionDecision {
    pub chosen: ActionScore,
    pub candidate_count: usize,
    /// Present only for the ADR 0004 Phase A controller; absent (and never
    /// serialized) for the greedy policy so its traces stay byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_a: Option<crate::p2::arc3_phase_a::PhaseADecisionTrace>,
    /// ADR 0005 §6.2 Channel B telemetry (updates, skips, reverts, prequential
    /// loss); present only under `--adapt`, never serialized otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adaptation: Option<AdaptationTrace>,
    /// ADR 0005 §6.1 Channel A: number of factual transitions in the Context
    /// Window every model call of this decision was conditioned on (0 when the
    /// window is off or empty).
    #[serde(default)]
    pub context_len: usize,
}

pub trait LivePolicy {
    fn choose_action(&mut self, observation: &ArcObservation) -> Result<ActionDecision>;

    /// Stable identifier of the deployed controller for telemetry rows.
    fn policy_name(&self) -> &'static str {
        LIVE_POLICY
    }

    /// Register the fixed public-game population before any scorecard action.
    fn on_suite_start(&mut self, _games: &[PublicGame]) -> Result<()> {
        Ok(())
    }

    /// Fires after the session-opening RESET of a game succeeds.
    fn on_game_start(&mut self, _game_id: &str) {}

    /// Fires for every confirmed factual transition, before any level
    /// transition it may have caused. Predictions never reach this hook.
    fn on_confirmed_transition(
        &mut self,
        _current: &ArcObservation,
        _action: &ArcAction,
        _next: &ArcObservation,
    ) {
    }

    /// Fires whenever a confirmed action advances `levels_completed`.
    fn on_level_transition(&mut self, _levels_completed: u16) {}

    /// Fires when the driver decides to retry the current level with a
    /// guid-scoped RESET after a recoverable non-WIN terminal.
    fn on_reset_retry(&mut self, _reason: &str) {}

    /// Fires once per attempted game with the final stop reason, even when
    /// the opening RESET failed and `on_game_start` never fired.
    fn on_game_end(&mut self, _outcome: &str) {}

    /// Flush session-scoped policy artifacts. Live evaluation publishes its
    /// fixed campaign up front; streaming callers use this to publish a plan
    /// after all game ids have been observed.
    fn finish_session(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Build the ADR 0005 §6.2 adapter when `--adapt` is set. `varmap` must be the
/// VarMap `model` was constructed from; the adapter never sees the checkpoint path.
pub fn adaptation_for<'a>(
    model: &'a WorldModel,
    varmap: &candle_nn::VarMap,
    device: &'a Device,
    adapt: bool,
    carry: bool,
) -> Result<Option<FastWeightAdapter<'a>>> {
    if !adapt {
        return Ok(None);
    }
    let mode = if carry {
        AdaptationMode::Carry
    } else {
        AdaptationMode::Reset
    };
    FastWeightAdapter::new(model, varmap, device, mode).map(Some)
}

/// Resolve `--context-window` (ADR 0005 §6.1 Channel A) against the loaded
/// checkpoint: on by default for `world_core_v6`; forced off (a no-op) for
/// checkpoints without a context channel, which reject any context.
pub fn live_context_for(model: &WorldModel, requested: Option<bool>, carry: bool) -> LiveContext {
    let v6 = model.config().world_core_v6;
    if !v6 && requested == Some(true) {
        eprintln!("--context-window ignored: the checkpoint is not world_core_v6");
    }
    let mode = if carry {
        AdaptationMode::Carry
    } else {
        AdaptationMode::Reset
    };
    LiveContext::new(v6 && requested.unwrap_or(true), mode)
}

/// Wraps any controller with Channel B adaptation (ADR 0005 §6.2). With no
/// adapter it is a transparent pass-through, so `--adapt` off stays
/// byte-identical. `maybe_update` runs before every decision; every confirmed
/// transition is observed; level boundaries reset (or carry) the fast weights;
/// game end restores theta_0. A failed restore poisons the policy so the next
/// decision fails closed instead of running a contaminated model.
pub struct AdaptingPolicy<'a, P: LivePolicy> {
    inner: P,
    adapter: Option<FastWeightAdapter<'a>>,
    poisoned: Option<String>,
}

impl<'a, P: LivePolicy> AdaptingPolicy<'a, P> {
    pub fn new(inner: P, adapter: Option<FastWeightAdapter<'a>>) -> Self {
        Self {
            inner,
            adapter,
            poisoned: None,
        }
    }

    pub fn adapter(&self) -> Option<&FastWeightAdapter<'a>> {
        self.adapter.as_ref()
    }

    fn guard(&mut self, what: &str, result: Result<()>) {
        if let Err(err) = result {
            self.poisoned = Some(format!("{what}: {err:#}"));
        }
    }
}

impl<P: LivePolicy> LivePolicy for AdaptingPolicy<'_, P> {
    fn choose_action(&mut self, observation: &ArcObservation) -> Result<ActionDecision> {
        if let Some(reason) = &self.poisoned {
            anyhow::bail!("adaptation poisoned the model: {reason}");
        }
        let adaptation = match self.adapter.as_mut() {
            Some(adapter) => adapter.maybe_update()?,
            None => None,
        };
        let mut decision = self.inner.choose_action(observation)?;
        decision.adaptation = adaptation;
        Ok(decision)
    }

    fn policy_name(&self) -> &'static str {
        self.inner.policy_name()
    }

    fn on_suite_start(&mut self, games: &[PublicGame]) -> Result<()> {
        self.inner.on_suite_start(games)
    }

    fn on_game_start(&mut self, game_id: &str) {
        self.inner.on_game_start(game_id);
        if let Some(adapter) = self.adapter.as_mut() {
            let result = adapter.begin_game();
            self.guard("begin_game", result);
        }
    }

    fn on_confirmed_transition(
        &mut self,
        current: &ArcObservation,
        action: &ArcAction,
        next: &ArcObservation,
    ) {
        self.inner.on_confirmed_transition(current, action, next);
        if let Some(adapter) = self.adapter.as_mut() {
            adapter.observe(
                &current.frame,
                action,
                &next.frame,
                current.levels_completed,
            );
        }
    }

    fn on_level_transition(&mut self, levels_completed: u16) {
        self.inner.on_level_transition(levels_completed);
        if let Some(adapter) = self.adapter.as_mut() {
            let result = adapter.on_level_transition(levels_completed);
            self.guard("level transition", result);
        }
    }

    fn on_reset_retry(&mut self, reason: &str) {
        self.inner.on_reset_retry(reason);
    }

    fn on_game_end(&mut self, outcome: &str) {
        self.inner.on_game_end(outcome);
        if let Some(adapter) = self.adapter.as_mut() {
            let result = adapter.restore_prior();
            self.guard("restore_prior", result);
        }
    }

    fn finish_session(&mut self) -> Result<()> {
        self.inner.finish_session()
    }
}

pub struct ModelPolicy<'a> {
    model: &'a WorldModel,
    device: &'a Device,
    physical_batch: usize,
    action6_max_candidates: usize,
    action6_grid_stride: usize,
    tried_penalty: f64,
    tried: BTreeMap<u64, BTreeSet<String>>,
    profile: Option<Arc3ProfileCampaign>,
    context: LiveContext,
}

struct Arc3ProfileCampaign {
    profile_dir: PathBuf,
    campaign_id: String,
    device: String,
    captures: BTreeMap<String, (u64, String)>,
    started: BTreeSet<String>,
    defer_manifest: bool,
}

fn arc3_profile_phase<T>(
    profile: Option<&RepresentativeUpdateCapture>,
    device: &Device,
    name: &str,
    kind: SpanKind,
    step: Option<ExecutionStep>,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    match profile {
        Some(profile) => profile.synchronized_phase(device, name, kind, step, f),
        None => f(),
    }
}

// TODO(ADR 0003 §7): replace this confidence/effect ranking only when the
// deferred goal/belief-conditioned Q-ranking head has a trained contract.

impl<'a> ModelPolicy<'a> {
    pub fn new(
        model: &'a WorldModel,
        device: &'a Device,
        physical_batch: usize,
        action6_max_candidates: usize,
        action6_grid_stride: usize,
        tried_penalty: f64,
    ) -> Self {
        Self {
            model,
            device,
            physical_batch,
            action6_max_candidates,
            action6_grid_stride,
            tried_penalty,
            tried: BTreeMap::new(),
            profile: None,
            context: LiveContext::disabled(),
        }
    }

    /// Channel A state (ADR 0005 §6.1); see [`live_context_for`].
    pub fn set_context(&mut self, context: LiveContext) {
        self.context = context;
    }

    pub fn enable_eval_profile(&mut self, output_dir: &Path, campaign_id: String, device: &str) {
        self.profile = Some(Arc3ProfileCampaign {
            profile_dir: output_dir.join("profile"),
            campaign_id,
            device: device.into(),
            captures: BTreeMap::new(),
            started: BTreeSet::new(),
            defer_manifest: false,
        });
    }

    pub fn enable_streaming_eval_profile(
        &mut self,
        output_dir: &Path,
        campaign_id: String,
        device: &str,
    ) {
        self.enable_eval_profile(output_dir, campaign_id, device);
        self.profile
            .as_mut()
            .expect("profile was just enabled")
            .defer_manifest = true;
    }

    fn prepare_profile_campaign(&mut self, games: &[PublicGame]) -> Result<()> {
        let Some(profile) = self.profile.as_mut() else {
            return Ok(());
        };
        let captures = games
            .iter()
            .enumerate()
            .map(|(index, game)| {
                (
                    game.game_id.clone(),
                    (
                        index as u64 + 1,
                        format!("arc3-{}", recording_component(&game.game_id)),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();
        ensure!(
            captures.len() == games.len(),
            "ARC profile plan contains duplicate game ids"
        );
        ensure_eval_profile_campaign(
            &profile.profile_dir,
            "arc3-campaign.json",
            profile.campaign_id.clone(),
            ARC3_PROFILE_ENTRYPOINT,
            captures
                .values()
                .map(|(capture_step, bundle)| PlannedCapture {
                    capture_step: *capture_step,
                    bundle: bundle.clone(),
                })
                .collect(),
        )?;
        profile.captures = captures;
        Ok(())
    }

    fn begin_decision_profile(
        &mut self,
        game_id: &str,
        candidate_count: usize,
    ) -> Result<Option<RepresentativeUpdateCapture>> {
        let Some(profile) = self.profile.as_mut() else {
            return Ok(None);
        };
        if profile.defer_manifest && !profile.captures.contains_key(game_id) {
            profile.captures.insert(
                game_id.into(),
                (
                    profile.captures.len() as u64 + 1,
                    format!("arc3-{}", recording_component(game_id)),
                ),
            );
        }
        if !profile.started.insert(game_id.into()) {
            return Ok(None);
        }
        let (capture_step, bundle) = profile
            .captures
            .get(game_id)
            .with_context(|| format!("ARC profile game {game_id} was not preregistered"))?;
        let destination = profile.profile_dir.join(bundle);
        RepresentativeUpdateCapture::begin_eval(EvalCaptureSpec {
            destination: &destination,
            capture_step: *capture_step,
            entrypoint: ARC3_PROFILE_ENTRYPOINT,
            correlation_id: format!("tofy.p2/arc3-{}", recording_component(game_id)),
            device: &profile.device,
            required_phases: &["encode", "forward", "decode", "metrics"],
            tags: &[
                ("game_id", game_id.into()),
                ("candidate_count", candidate_count.to_string()),
                ("physical_batch", self.physical_batch.to_string()),
            ],
        })
        .map(Some)
    }

    fn score_candidates(
        &self,
        frame: &ArcFrame,
        candidates: &[ArcAction],
        window: &[ContextTransition],
        profile: Option<&RepresentativeUpdateCapture>,
    ) -> Result<Vec<ActionScore>> {
        let (frames, encoded) = arc3_profile_phase(
            profile,
            self.device,
            "encode",
            SpanKind::Module,
            None,
            || {
                let frames = frames_to_indices(std::slice::from_ref(frame), self.device)?;
                let encoded = self.model.encode_state(&frames)?;
                Ok((frames, encoded))
            },
        )?;
        let (_, channels, height, width) = encoded.dims4()?;
        let mut scores = Vec::with_capacity(candidates.len());
        for (chunk_index, chunk) in candidates.chunks(self.physical_batch).enumerate() {
            let n = chunk.len();
            let selected_profile = (chunk_index == 0).then_some(profile).flatten();
            let (output, state) = arc3_profile_phase(
                selected_profile,
                self.device,
                "forward",
                SpanKind::Module,
                Some(ExecutionStep::Forward),
                || {
                    let actions = Tensor::from_vec(
                        chunk.iter().map(|a| u32::from(a.id)).collect::<Vec<_>>(),
                        n,
                        self.device,
                    )?;
                    let coords = Tensor::from_vec(
                        chunk
                            .iter()
                            .flat_map(|a| {
                                [
                                    a.x.map_or(0.0, |x| f32::from(x) / 63.0),
                                    a.y.map_or(0.0, |y| f32::from(y) / 63.0),
                                ]
                            })
                            .collect::<Vec<_>>(),
                        (n, 2),
                        self.device,
                    )?;
                    // The all-zero goal is in-distribution because
                    // foundation-v2 applies goal dropout during training.
                    let goals = Tensor::zeros((n, GOAL_FEATURES_DIM), DType::F32, self.device)?;
                    let operator_conditioning = unknown_operator_conditioning(n, self.device)?;
                    let state = encoded.broadcast_as((n, channels, height, width))?;
                    let frame_batch = frames.broadcast_as((n, 1, FRAME_SIDE, FRAME_SIDE))?;
                    let context = context_batch_for(window, n, self.device)?;
                    let output = self
                        .model
                        .forward_from_encoded_state_with_operator_conditioning_with_context(
                            &state,
                            &frame_batch,
                            &actions,
                            &coords,
                            &goals,
                            &operator_conditioning,
                            context.as_ref(),
                            RecursionDepth::from_config(self.model.config()),
                            0.0,
                            None,
                            RecursionOpts::EVAL,
                        )?;
                    Ok((output, state))
                },
            )?;
            let (q, reliability, noop, effect) = arc3_profile_phase(
                selected_profile,
                self.device,
                "decode",
                SpanKind::Module,
                None,
                || {
                    Ok((
                        ops::sigmoid(&output.q_logit)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        ops::sigmoid(&output.reliability_logit)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        ops::sigmoid(&output.event_logits.narrow(1, EVENT_NOOP, 1)?)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        latent_mse_per_sample(&output.y, &state)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                    ))
                },
            )?;
            for index in 0..n {
                let effect_scaled = f64::from(effect[index]).max(0.0);
                let effect_unit = effect_scaled / (1.0 + effect_scaled);
                // Q measures composed next-frame pixel accuracy; reliability
                // measures confidence that factual latent MSE is within its
                // threshold; no-op and effect measure predicted action impact.
                // TODO(ADR 0003 §7): retain the searchless forward-pass policy
                // until a goal/belief-conditioned ranking target is trained.
                let score = 0.25 * f64::from(q[index])
                    + 0.30 * f64::from(reliability[index])
                    + 0.30 * (1.0 - f64::from(noop[index]))
                    + 0.15 * effect_unit;
                scores.push(ActionScore {
                    action: chunk[index].clone(),
                    score,
                    q_probability: f64::from(q[index]),
                    reliability_probability: f64::from(reliability[index]),
                    noop_probability: f64::from(noop[index]),
                    predicted_effect: effect_scaled,
                });
            }
        }
        Ok(scores)
    }
}

impl LivePolicy for ModelPolicy<'_> {
    fn choose_action(&mut self, observation: &ArcObservation) -> Result<ActionDecision> {
        let candidates = enumerate_actions(
            observation,
            self.action6_max_candidates,
            self.action6_grid_stride,
        )?;
        let mut profile = self.begin_decision_profile(&observation.game_id, candidates.len())?;
        let measurement = profile
            .as_ref()
            .and_then(RepresentativeUpdateCapture::measurement);
        let window = self.context.window(observation.levels_completed);
        let mut scores =
            self.score_candidates(&observation.frame, &candidates, &window, profile.as_ref())?;
        let hash = observation_hash(observation);
        let tried = self.tried.entry(hash).or_default();
        apply_tried_penalty(&mut scores, tried, self.tried_penalty);
        scores.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.action.id.cmp(&b.action.id))
                .then_with(|| a.action.x.cmp(&b.action.x))
                .then_with(|| a.action.y.cmp(&b.action.y))
        });
        let chosen = scores
            .first()
            .cloned()
            .context("policy generated no action candidates")?;
        tried.insert(action_key(&chosen.action));
        let decision = ActionDecision {
            chosen,
            candidate_count: candidates.len(),
            phase_a: None,
            adaptation: None,
            context_len: window.len(),
        };
        if let Some(capture) = profile.as_ref() {
            let metrics = capture.phase("metrics", SpanKind::Function, None);
            let span_id = metrics.as_ref().and_then(|range| range.span_id());
            capture.record_scalar(span_id, "arc3/chosen_score", decision.chosen.score)?;
            capture.record_scalar(
                span_id,
                "arc3/chosen_q_probability",
                decision.chosen.q_probability,
            )?;
            capture.record_scalar(
                span_id,
                "arc3/chosen_reliability_probability",
                decision.chosen.reliability_probability,
            )?;
            capture.record_scalar(
                span_id,
                "arc3/chosen_noop_probability",
                decision.chosen.noop_probability,
            )?;
            capture.record_scalar(
                span_id,
                "arc3/chosen_predicted_effect",
                decision.chosen.predicted_effect,
            )?;
            drop(metrics);
        }
        drop(measurement);
        if let Some(capture) = profile.take() {
            if let Some(artifacts) = capture.finish()? {
                candle_graph::verify_bundle(&artifacts.directory).with_context(|| {
                    format!(
                        "verify ARC profile bundle {}",
                        artifacts.directory.display()
                    )
                })?;
            }
        }
        Ok(decision)
    }

    fn on_suite_start(&mut self, games: &[PublicGame]) -> Result<()> {
        self.prepare_profile_campaign(games)
    }

    fn on_game_start(&mut self, _game_id: &str) {
        self.tried.clear();
        self.context.begin_game();
    }

    fn on_confirmed_transition(
        &mut self,
        current: &ArcObservation,
        action: &ArcAction,
        next: &ArcObservation,
    ) {
        self.context.observe(
            &current.frame,
            action,
            &next.frame,
            current.levels_completed,
        );
    }

    fn on_level_transition(&mut self, _levels_completed: u16) {
        self.tried.clear();
    }

    fn on_reset_retry(&mut self, _reason: &str) {
        self.tried.clear();
    }

    fn on_game_end(&mut self, _outcome: &str) {
        self.tried.clear();
    }

    fn finish_session(&mut self) -> Result<()> {
        let Some(profile) = self.profile.as_mut() else {
            return Ok(());
        };
        if !profile.defer_manifest || profile.captures.is_empty() {
            return Ok(());
        }
        let mut planned = profile
            .captures
            .values()
            .map(|(capture_step, bundle)| PlannedCapture {
                capture_step: *capture_step,
                bundle: bundle.clone(),
            })
            .collect::<Vec<_>>();
        planned.sort_by_key(|capture| capture.capture_step);
        ensure_eval_profile_campaign(
            &profile.profile_dir,
            "arc3-campaign.json",
            profile.campaign_id.clone(),
            ARC3_PROFILE_ENTRYPOINT,
            planned,
        )?;
        profile.defer_manifest = false;
        Ok(())
    }
}

pub(crate) fn decision_telemetry(decision: &ActionDecision) -> Value {
    let policy = if decision.phase_a.is_some() {
        crate::p2::arc3_phase_a::PHASE_A_POLICY
    } else {
        LIVE_POLICY
    };
    let mut telemetry = json!({
        "policy": policy,
        "score": decision.chosen.score,
        "q_probability": decision.chosen.q_probability,
        "reliability_probability": decision.chosen.reliability_probability,
        "noop_probability": decision.chosen.noop_probability,
        "predicted_effect": decision.chosen.predicted_effect,
        "candidate_count": decision.candidate_count,
        "context_len": decision.context_len,
    });
    if let Some(trace) = &decision.phase_a {
        if let Ok(value) = serde_json::to_value(trace) {
            telemetry["phase_a"] = value;
        }
    }
    if let Some(trace) = &decision.adaptation {
        if let Ok(value) = serde_json::to_value(trace) {
            telemetry["adaptation"] = value;
        }
    }
    telemetry
}

/// Soft tried-action demotion. Once every candidate has been tried the
/// history for this key restarts; otherwise tried candidates lose `penalty`
/// score so that near-ties explore while a strongly better tried action
/// remains selectable.
fn apply_tried_penalty(scores: &mut [ActionScore], tried: &mut BTreeSet<String>, penalty: f64) {
    let all_tried = scores
        .iter()
        .all(|score| tried.contains(&action_key(&score.action)));
    if all_tried {
        tried.clear();
    }
    for score in scores {
        if tried.contains(&action_key(&score.action)) {
            score.score -= penalty;
        }
    }
}

pub fn observation_hash(observation: &ArcObservation) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    observation.game_id.hash(&mut hasher);
    observation.guid.hash(&mut hasher);
    observation.levels_completed.hash(&mut hasher);
    let frame = &observation.frame;
    frame.width.hash(&mut hasher);
    frame.height.hash(&mut hasher);
    let visible_rows = usize::from(frame.height).min(if row63_has_content(frame) {
        FRAME_SIDE
    } else {
        FRAME_SIDE - 1
    });
    let visible_len = visible_rows
        .saturating_mul(usize::from(frame.width))
        .min(frame.pixels.len());
    frame.pixels[..visible_len].hash(&mut hasher);
    hasher.finish()
}

fn action_key(action: &ArcAction) -> String {
    format!(
        "{}:{}:{}",
        action.id,
        action.x.map_or_else(|| "-".into(), |v| v.to_string()),
        action.y.map_or_else(|| "-".into(), |v| v.to_string())
    )
}

pub fn enumerate_actions(
    observation: &ArcObservation,
    action6_max_candidates: usize,
    grid_stride: usize,
) -> Result<Vec<ArcAction>> {
    observation.validate()?;
    ensure!(
        action6_max_candidates > 0,
        "ACTION6 candidate cap must be > 0"
    );
    ensure!(
        (1..=FRAME_SIDE).contains(&grid_stride),
        "ACTION6 grid stride must be in 1..={FRAME_SIDE}"
    );
    let mut candidates = Vec::new();
    for &id in &observation.available_actions {
        if id != 6 {
            candidates.push(ArcAction::new(id, None, None)?);
        }
    }
    if observation.available_actions.contains(&6) {
        let allow_row63 = row63_has_content(&observation.frame);
        for (x, y) in action6_coordinates(
            &observation.frame,
            action6_max_candidates,
            grid_stride,
            allow_row63,
        ) {
            candidates.push(ArcAction::new(6, Some(x), Some(y))?);
        }
    }
    ensure!(!candidates.is_empty(), "no valid actions available");
    Ok(candidates)
}

fn dominant_color(frame: &ArcFrame) -> u8 {
    let mut counts = [0usize; 16];
    for &pixel in &frame.pixels {
        counts[pixel as usize] += 1;
    }
    counts
        .iter()
        .enumerate()
        .max_by_key(|&(color, count)| (*count, std::cmp::Reverse(color)))
        .map(|(color, _)| color as u8)
        .unwrap_or(0)
}

/// Training synthesizes row 63 as a status row, but live games may put real
/// content there. Content is any row-63 pixel differing from both the
/// dominant background color and [`palette::PAD`] (fixed-64 padding).
pub fn row63_has_content(frame: &ArcFrame) -> bool {
    let background = dominant_color(frame);
    let row_start = (FRAME_SIDE - 1) * FRAME_SIDE;
    frame
        .pixels
        .get(row_start..)
        .is_some_and(|row| row.iter().any(|&p| p != background && p != palette::PAD))
}

fn action6_coordinates(
    frame: &ArcFrame,
    cap: usize,
    stride: usize,
    allow_row63: bool,
) -> Vec<(u8, u8)> {
    let background = dominant_color(frame);
    let mut counts = [0usize; 16];
    for &pixel in &frame.pixels {
        counts[pixel as usize] += 1;
    }
    let mut points = Vec::new();
    let mut seen = BTreeSet::new();

    for color in 0u8..16 {
        if color == background || counts[color as usize] == 0 {
            continue;
        }
        let mut min_x = 63usize;
        let mut min_y = 63usize;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut sum_x = 0usize;
        let mut sum_y = 0usize;
        let mut n = 0usize;
        for (index, &pixel) in frame.pixels.iter().enumerate() {
            let y = index / FRAME_SIDE;
            if pixel != color || (!allow_row63 && y == FRAME_SIDE - 1) {
                continue;
            }
            let x = index % FRAME_SIDE;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            sum_x += x;
            sum_y += y;
            n += 1;
        }
        let Some(n) = std::num::NonZeroUsize::new(n) else {
            continue;
        };
        for point in [
            ((sum_x / n.get()) as u8, (sum_y / n.get()) as u8),
            (((min_x + max_x) / 2) as u8, ((min_y + max_y) / 2) as u8),
            (min_x as u8, min_y as u8),
            (max_x as u8, max_y as u8),
        ] {
            if seen.insert(point) {
                points.push(point);
            }
        }
    }

    let row_limit = if allow_row63 {
        FRAME_SIDE
    } else {
        FRAME_SIDE - 1
    };
    for y in (stride / 2..row_limit).step_by(stride) {
        for x in (stride / 2..FRAME_SIDE).step_by(stride) {
            let point = (x.min(63) as u8, y.min(63) as u8);
            if seen.insert(point) {
                points.push(point);
            }
        }
    }
    if points.is_empty() {
        points.push((32, 32));
    }
    points.truncate(cap);
    points
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveActionTrace {
    pub index: usize,
    pub available_actions: Vec<u8>,
    pub decision: ActionDecision,
    pub levels_before: u16,
    pub levels_after: u16,
    pub state_after: String,
    pub frame_changed: bool,
    pub api_latency_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum MutationAttemptOutcome {
    Confirmed,
    Ambiguous { mutation: AmbiguousMutation },
    Rejected { reason: String },
    ProtocolViolation { kind: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationAttempt {
    pub index: usize,
    pub game_id: String,
    pub guid: String,
    pub action: ArcAction,
    pub levels_before: u16,
    pub api_latency_ms: u128,
    pub outcome: MutationAttemptOutcome,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LiveLevelUsage {
    pub level_index: u16,
    pub attempted_actions: usize,
    pub confirmed_actions: usize,
    pub reset_retries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveGameReport {
    pub game_id: String,
    pub title: String,
    pub actions: usize,
    pub levels_completed: u16,
    pub win_levels: u16,
    pub terminal_state: String,
    pub stop_reason: String,
    pub error: Option<String>,
    pub duration_ms: u128,
    pub trace: Vec<LiveActionTrace>,
    #[serde(default)]
    pub ambiguous_reset: Option<AmbiguousMutation>,
    #[serde(default)]
    pub attempted_actions: usize,
    #[serde(default)]
    pub confirmed_actions: usize,
    #[serde(default)]
    pub mutation_attempts: Vec<MutationAttempt>,
    #[serde(default)]
    pub reset_retries: usize,
    #[serde(default)]
    pub full_reset_detected: bool,
    #[serde(default)]
    pub level_usage: Vec<LiveLevelUsage>,
    #[serde(default)]
    pub agent_session: AgentSession,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveEvalReport {
    pub schema: String,
    pub created_at_unix_ms: u128,
    pub checkpoint: PathBuf,
    pub checkpoint_sha256: String,
    pub train_config: PathBuf,
    pub train_config_sha256: String,
    pub device: String,
    pub base_url: String,
    #[serde(default)]
    pub git_revision: String,
    #[serde(default)]
    pub git_dirty: bool,
    #[serde(default)]
    pub dirty_diff_sha256: Option<String>,
    #[serde(default)]
    pub executable_sha256: Option<String>,
    #[serde(default)]
    pub build_profile: String,
    #[serde(default)]
    pub cli_args: Vec<String>,
    #[serde(default)]
    pub evidence_class: String,
    pub policy: String,
    pub policy_limitation: String,
    pub goal_feature_contract: String,
    pub tried_action_key_contract: String,
    #[serde(default)]
    pub driver: LiveDriverOptions,
    pub held_out_only: bool,
    pub public_data_used_for_fitting: bool,
    pub discovered_games: Vec<PublicGame>,
    pub selected_game_count: usize,
    pub evaluated_all_discovered_games: bool,
    pub card_id: String,
    pub games: Vec<LiveGameReport>,
    pub scorecard: Option<Value>,
    pub scorecard_close_error: Option<String>,
    pub official_scorecard: Option<ScorecardBenchmark>,
    pub official_scorecard_parse_error: Option<String>,
    pub official_rhae: Option<f64>,
    pub research_claim: bool,
}

#[derive(Debug, Clone)]
pub struct LiveRunSettings {
    pub checkpoint: PathBuf,
    pub checkpoint_sha256: String,
    pub train_config: PathBuf,
    pub train_config_sha256: String,
    pub device: String,
    pub base_url: String,
    pub requested_games: Vec<String>,
    pub driver: LiveDriverOptions,
    pub git_revision: String,
    pub git_dirty: bool,
    pub dirty_diff_sha256: Option<String>,
    pub executable_sha256: Option<String>,
    pub build_profile: String,
    pub cli_args: Vec<String>,
    pub evidence_class: String,
    pub recordings_dir: Option<PathBuf>,
    /// Deployed controller contract recorded in the report and scorecard.
    pub contract: PolicyContract,
}

/// Evidence class at scorecard-open time. A clean worktree earns only
/// `candidate_run`: AGENTS.md reserves `completed_evidence` for sealed runs
/// whose provenance, integrity, and evaluation checks all passed, which a
/// driver cannot certify at open time. The report builder downgrades to
/// `failed_integrity_or_evaluation` when the run records a scorecard-close
/// error or an unexpected full reset.
pub(crate) fn live_evidence_class(driver: &LiveDriverOptions) -> &'static str {
    if driver.exploratory {
        "exploratory_driver_repair"
    } else {
        "candidate_run"
    }
}

pub fn run_public_suite<A: ArcApi, P: LivePolicy>(
    api: &mut A,
    policy: &mut P,
    settings: &LiveRunSettings,
) -> Result<LiveEvalReport> {
    settings.driver.validate()?;
    ensure!(
        settings.evidence_class == live_evidence_class(&settings.driver),
        "live evidence class does not match driver mode"
    );
    ensure!(
        !settings.git_dirty || settings.driver.exploratory,
        "refusing to open a scorecard from a dirty worktree without --exploratory"
    );
    let mut discovered = api.list_games().context("discover public ARC games")?;
    discovered.sort_by(|a, b| a.game_id.cmp(&b.game_id));
    ensure!(!discovered.is_empty(), "ARC API returned no public games");
    let selected = select_games(&discovered, &settings.requested_games)?;
    policy.on_suite_start(&selected)?;
    let evaluating_all = settings.requested_games.is_empty() && selected.len() == discovered.len();
    let metadata = json!({
        "tags": ["tofy", "p2", "held-out", "live-eval"],
        "competition_mode": settings.driver.competition_mode,
        "opaque": {
            "schema": LIVE_REPORT_SCHEMA,
            "checkpoint_sha256": settings.checkpoint_sha256,
            "train_config_sha256": settings.train_config_sha256,
            "policy": settings.contract.name,
            "public_data_used_for_fitting": false,
        }
    });
    let card_id = api.open_scorecard(&metadata)?;
    let mut game_reports = Vec::with_capacity(selected.len());
    let mut recording_failure = None;

    for game in &selected {
        let started = Instant::now();
        let mut trace = Vec::new();
        let mut mutation_attempts = Vec::new();
        let mut ambiguous_reset = None;
        let mut error = None;
        let mut stop_reason = "reset_failed".to_string();
        let mut agent_session = AgentSession::default();
        let mut attempted_actions = 0usize;
        let mut reset_retries = 0usize;
        let mut full_reset_detected = false;
        let mut level_usage = BTreeMap::<u16, LiveLevelUsage>::new();
        let mut action_since_reset_or_transition = false;
        let mut recording = None;
        let mut last = match api.reset(&game.game_id, &card_id, None) {
            Ok(observation) => {
                agent_session.observe(&observation)?;
                policy.on_game_start(&game.game_id);
                match settings
                    .recordings_dir
                    .as_deref()
                    .map(|root| LiveRecordingRun::start(root, &observation))
                    .transpose()
                {
                    Ok(started) => {
                        recording = started;
                        Some(observation)
                    }
                    Err(err) => {
                        error = Some(format!("recording: {err:#}"));
                        stop_reason = "recording_error".into();
                        recording_failure = Some(err);
                        None
                    }
                }
            }
            Err(MutationError::Ambiguous(mutation)) => {
                error = Some(mutation.to_string());
                stop_reason = "ambiguous_reset".into();
                ambiguous_reset = Some(mutation);
                None
            }
            Err(MutationError::Failed(err)) => {
                error = Some(format!("{err:#}"));
                None
            }
        };

        while let Some(observation) = last.clone() {
            if observation.terminal() {
                if observation.won() {
                    stop_reason = "completed".into();
                    break;
                }
                if observation.state != "GAME_OVER" {
                    stop_reason = format!("terminal_{}", observation.state.to_ascii_lowercase());
                    break;
                }
                let usage = level_usage
                    .entry(observation.levels_completed)
                    .or_insert_with(|| LiveLevelUsage {
                        level_index: observation.levels_completed,
                        ..LiveLevelUsage::default()
                    });
                if usage.attempted_actions >= settings.driver.max_actions_per_level as usize {
                    stop_reason = "level_action_cap_reached".into();
                    break;
                }
                if settings
                    .driver
                    .max_actions_per_game
                    .is_some_and(|cap| attempted_actions >= cap as usize)
                {
                    stop_reason = "max_actions_reached".into();
                    break;
                }
                let retry_is_safe =
                    settings.driver.competition_mode || action_since_reset_or_transition;
                if !retry_is_safe {
                    stop_reason = "terminal_game_over".into();
                    break;
                }
                if usage.reset_retries >= settings.driver.max_level_retries {
                    stop_reason = "level_retry_exhausted".into();
                    break;
                }

                policy.on_reset_retry("game_over");
                usage.reset_retries += 1;
                reset_retries += 1;
                let level_before = observation.levels_completed;
                let retry =
                    match api.reset(&game.game_id, &card_id, Some(observation.guid.as_str())) {
                        Ok(retry) => retry,
                        Err(MutationError::Ambiguous(mutation)) => {
                            error = Some(mutation.to_string());
                            stop_reason = "ambiguous_reset".into();
                            ambiguous_reset = Some(mutation);
                            break;
                        }
                        Err(MutationError::Failed(err)) => {
                            error = Some(format!("retry reset: {err:#}"));
                            stop_reason = "retry_reset_failed".into();
                            break;
                        }
                    };
                if let Some(recording) = recording.as_mut() {
                    if let Err(err) = recording.push_reset(&retry) {
                        error = Some(format!("recording retry RESET: {err:#}"));
                        stop_reason = "recording_error".into();
                        recording_failure = Some(err);
                        last = Some(retry);
                        break;
                    }
                }
                if retry.full_reset || retry.levels_completed != level_before {
                    full_reset_detected = retry.full_reset;
                    error = Some(format!(
                        "retry RESET changed session scope: full_reset={} levels {} -> {}",
                        retry.full_reset, level_before, retry.levels_completed
                    ));
                    stop_reason = "unsafe_retry_reset".into();
                    last = Some(retry);
                    break;
                }
                agent_session.observe(&retry)?;
                action_since_reset_or_transition = false;
                last = Some(retry);
                continue;
            }
            let usage = level_usage
                .entry(observation.levels_completed)
                .or_insert_with(|| LiveLevelUsage {
                    level_index: observation.levels_completed,
                    ..LiveLevelUsage::default()
                });
            if usage.attempted_actions >= settings.driver.max_actions_per_level as usize {
                stop_reason = "level_action_cap_reached".into();
                break;
            }
            if settings
                .driver
                .max_actions_per_game
                .is_some_and(|cap| attempted_actions >= cap as usize)
            {
                stop_reason = "max_actions_reached".into();
                break;
            }
            let decision = match policy.choose_action(&observation) {
                Ok(decision) => decision,
                Err(err) => {
                    error = Some(format!("policy: {err:#}"));
                    stop_reason = "policy_error".into();
                    break;
                }
            };
            let reasoning = decision_telemetry(&decision);
            let call_started = Instant::now();
            attempted_actions += 1;
            usage.attempted_actions += 1;
            let attempt_index = mutation_attempts.len();
            let next = match api.act(
                &game.game_id,
                &observation.guid,
                &decision.chosen.action,
                &reasoning,
            ) {
                Ok(next) => next,
                Err(MutationError::Ambiguous(mutation)) => {
                    let api_latency_ms = call_started.elapsed().as_millis();
                    mutation_attempts.push(MutationAttempt {
                        index: attempt_index,
                        game_id: game.game_id.clone(),
                        guid: observation.guid.clone(),
                        action: decision.chosen.action.clone(),
                        levels_before: observation.levels_completed,
                        api_latency_ms,
                        outcome: MutationAttemptOutcome::Ambiguous {
                            mutation: mutation.clone(),
                        },
                    });
                    agent_session.record_ambiguous(
                        &observation,
                        decision.chosen.action.clone(),
                        mutation.clone(),
                    )?;
                    error = Some(format!("action {}: {mutation}", trace.len() + 1));
                    stop_reason = "ambiguous_mutation".into();
                    break;
                }
                Err(err) => {
                    mutation_attempts.push(MutationAttempt {
                        index: attempt_index,
                        game_id: game.game_id.clone(),
                        guid: observation.guid.clone(),
                        action: decision.chosen.action.clone(),
                        levels_before: observation.levels_completed,
                        api_latency_ms: call_started.elapsed().as_millis(),
                        outcome: MutationAttemptOutcome::Rejected {
                            reason: format!("{err:#}"),
                        },
                    });
                    error = Some(format!("action {}: {err}", trace.len() + 1));
                    stop_reason = "api_error".into();
                    break;
                }
            };
            if let Some(recording) = recording.as_mut() {
                if let Err(err) = recording.push_action(&next, &decision.chosen.action, &reasoning)
                {
                    error = Some(format!("recording ACTION response: {err:#}"));
                    stop_reason = "recording_error".into();
                    recording_failure = Some(err);
                    last = Some(next);
                    break;
                }
            }
            if next.levels_completed < observation.levels_completed {
                mutation_attempts.push(MutationAttempt {
                    index: attempt_index,
                    game_id: game.game_id.clone(),
                    guid: observation.guid.clone(),
                    action: decision.chosen.action.clone(),
                    levels_before: observation.levels_completed,
                    api_latency_ms: call_started.elapsed().as_millis(),
                    outcome: MutationAttemptOutcome::ProtocolViolation {
                        kind: "level_regression".into(),
                    },
                });
                agent_session.observe(&next)?;
                error = Some(format!(
                    "ACTION response regressed levels {} -> {}",
                    observation.levels_completed, next.levels_completed
                ));
                stop_reason = "protocol_level_regression".into();
                last = Some(next);
                break;
            }
            if next.full_reset {
                mutation_attempts.push(MutationAttempt {
                    index: attempt_index,
                    game_id: game.game_id.clone(),
                    guid: observation.guid.clone(),
                    action: decision.chosen.action.clone(),
                    levels_before: observation.levels_completed,
                    api_latency_ms: call_started.elapsed().as_millis(),
                    outcome: MutationAttemptOutcome::ProtocolViolation {
                        kind: "full_reset".into(),
                    },
                });
                agent_session.observe(&next)?;
                full_reset_detected = true;
                error = Some("ACTION response unexpectedly reported full_reset=true".into());
                stop_reason = "protocol_full_reset".into();
                last = Some(next);
                break;
            }
            agent_session.record_confirmed(&observation, decision.chosen.action.clone(), &next)?;
            policy.on_confirmed_transition(&observation, &decision.chosen.action, &next);
            usage.confirmed_actions += 1;
            action_since_reset_or_transition = true;
            let api_latency_ms = call_started.elapsed().as_millis();
            mutation_attempts.push(MutationAttempt {
                index: attempt_index,
                game_id: game.game_id.clone(),
                guid: observation.guid.clone(),
                action: decision.chosen.action.clone(),
                levels_before: observation.levels_completed,
                api_latency_ms,
                outcome: MutationAttemptOutcome::Confirmed,
            });
            trace.push(LiveActionTrace {
                index: trace.len(),
                available_actions: observation.available_actions.clone(),
                decision,
                levels_before: observation.levels_completed,
                levels_after: next.levels_completed,
                state_after: next.state.clone(),
                frame_changed: observation.frame != next.frame,
                api_latency_ms,
            });
            if next.levels_completed > observation.levels_completed {
                policy.on_level_transition(next.levels_completed);
                action_since_reset_or_transition = false;
            }
            last = Some(next);
        }

        let (levels_completed, win_levels, terminal_state) = last
            .as_ref()
            .map(|observation| {
                (
                    observation.levels_completed,
                    observation.win_levels,
                    observation.state.clone(),
                )
            })
            .unwrap_or((0, 0, "RESET_FAILED".into()));
        let confirmed_actions = mutation_attempts
            .iter()
            .filter(|attempt| matches!(attempt.outcome, MutationAttemptOutcome::Confirmed))
            .count();
        ensure!(
            attempted_actions == mutation_attempts.len() && confirmed_actions == trace.len(),
            "action totals do not reconcile with the mutation ledger"
        );
        if let Some(recording) = recording {
            if let Err(err) = recording.finish() {
                error = Some(format!("persist recording: {err:#}"));
                stop_reason = "recording_error".into();
                recording_failure = Some(err);
            }
        }
        policy.on_game_end(&stop_reason);
        game_reports.push(LiveGameReport {
            game_id: game.game_id.clone(),
            title: game.title.clone(),
            actions: trace.len(),
            levels_completed,
            win_levels,
            terminal_state,
            stop_reason,
            error,
            duration_ms: started.elapsed().as_millis(),
            trace,
            ambiguous_reset,
            attempted_actions,
            confirmed_actions,
            mutation_attempts,
            reset_retries,
            full_reset_detected,
            level_usage: level_usage.into_values().collect(),
            agent_session,
        });
        if recording_failure.is_some() {
            break;
        }
    }

    let (scorecard, scorecard_close_error) = match api.close_scorecard(&card_id) {
        Ok(card) => (Some(card), None),
        Err(err) => (None, Some(format!("{err:#}"))),
    };
    if let Some(error) = recording_failure {
        return Err(error).context("live ARC recording persistence failed");
    }
    let (official_scorecard, official_scorecard_parse_error) = match scorecard.as_ref() {
        Some(card) => match benchmark_from_scorecard_str(&card.to_string()) {
            Ok(benchmark) => (Some(benchmark), None),
            Err(err) => (None, Some(format!("{err:#}"))),
        },
        None => (None, None),
    };
    let official_rhae = official_scorecard
        .as_ref()
        .and_then(official_rhae_from_benchmark);
    // Downgrade the evidence class when the run itself recorded an integrity
    // or evaluation failure; only a later sealing step may ever write
    // `completed_evidence`.
    let evidence_class = if scorecard_close_error.is_some()
        || official_scorecard_parse_error.is_some()
        || game_reports.iter().any(|game| game.full_reset_detected)
    {
        "failed_integrity_or_evaluation".to_string()
    } else {
        settings.evidence_class.clone()
    };
    Ok(LiveEvalReport {
        schema: LIVE_REPORT_SCHEMA.into(),
        created_at_unix_ms: unix_ms(),
        checkpoint: settings.checkpoint.clone(),
        checkpoint_sha256: settings.checkpoint_sha256.clone(),
        train_config: settings.train_config.clone(),
        train_config_sha256: settings.train_config_sha256.clone(),
        device: settings.device.clone(),
        base_url: settings.base_url.clone(),
        git_revision: settings.git_revision.clone(),
        git_dirty: settings.git_dirty,
        dirty_diff_sha256: settings.dirty_diff_sha256.clone(),
        executable_sha256: settings.executable_sha256.clone(),
        build_profile: settings.build_profile.clone(),
        cli_args: settings.cli_args.clone(),
        evidence_class,
        policy: settings.contract.name.clone(),
        policy_limitation: settings.contract.limitation.clone(),
        goal_feature_contract: settings.contract.goal_feature_contract.clone(),
        tried_action_key_contract: TRIED_ACTION_KEY_CONTRACT.into(),
        driver: settings.driver.clone(),
        held_out_only: true,
        public_data_used_for_fitting: false,
        discovered_games: discovered,
        selected_game_count: selected.len(),
        evaluated_all_discovered_games: evaluating_all,
        card_id,
        games: game_reports,
        scorecard,
        scorecard_close_error,
        official_scorecard,
        official_scorecard_parse_error,
        official_rhae,
        research_claim: false,
    })
}

fn select_games(discovered: &[PublicGame], requested: &[String]) -> Result<Vec<PublicGame>> {
    if requested.is_empty() {
        return Ok(discovered.to_vec());
    }
    let mut selected = Vec::new();
    for request in requested {
        let game = discovered
            .iter()
            .find(|game| {
                game.game_id == *request || game.title.eq_ignore_ascii_case(request.as_str())
            })
            .with_context(|| format!("requested game {request:?} is not public/available"))?;
        if !selected
            .iter()
            .any(|selected: &PublicGame| selected.game_id == game.game_id)
        {
            selected.push(game.clone());
        }
    }
    Ok(selected)
}

pub fn list_public_games(config: &LiveEvalConfig) -> Result<Vec<PublicGame>> {
    config.validate()?;
    let mut api = HttpArcApi::from_env(
        &config.base_url,
        &config.api_key_env,
        Duration::from_secs(config.request_timeout_secs),
    )?;
    let mut games = api.list_games()?;
    games.sort_by(|a, b| a.game_id.cmp(&b.game_id));
    Ok(games)
}

pub fn evaluate_live(config: &LiveEvalConfig) -> Result<LiveEvalReport> {
    config.validate()?;
    let provenance = live_run_provenance()?;
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
    let adapter = adaptation_for(&model, &varmap, &device, config.adapt, config.adapt_carry)?;
    let context = live_context_for(&model, config.context_window, config.adapt_carry);
    let mut api = HttpArcApi::from_env(
        &config.base_url,
        &config.api_key_env,
        Duration::from_secs(config.request_timeout_secs),
    )?;
    let settings = LiveRunSettings {
        checkpoint: config.checkpoint.clone(),
        checkpoint_sha256: checkpoint_sha256.clone(),
        train_config: config.train_config.clone(),
        train_config_sha256,
        device: config.device.clone(),
        base_url: config.base_url.clone(),
        requested_games: config.games.clone(),
        driver: config.driver.clone(),
        git_revision: provenance.git_revision,
        git_dirty: provenance.git_dirty,
        dirty_diff_sha256: provenance.dirty_diff_sha256,
        executable_sha256: provenance.executable_sha256,
        build_profile: provenance.build_profile,
        cli_args: provenance.cli_args,
        evidence_class: live_evidence_class(&config.driver).into(),
        recordings_dir: Some(config.recordings_dir.clone()),
        contract: PolicyContract::for_kind(config.policy),
    };
    let report = match config.policy {
        LivePolicyKind::Greedy => {
            let mut policy = ModelPolicy::new(
                &model,
                &device,
                config.physical_batch,
                config.action6_max_candidates,
                config.action6_grid_stride,
                config.driver.tried_penalty,
            );
            if config.profile_eval {
                let output_dir = config
                    .output
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                    .unwrap_or_else(|| Path::new("."));
                policy.enable_eval_profile(
                    output_dir,
                    format!("tofy.p2.arc3.{checkpoint_sha256}"),
                    &config.device,
                );
            }
            policy.set_context(context);
            let mut policy = AdaptingPolicy::new(policy, adapter);
            run_public_suite(&mut api, &mut policy, &settings)?
        }
        LivePolicyKind::PhaseA => {
            let calibration = crate::p2::arc3_phase_a::load_phase_a_calibration(
                config.phase_a_calibration.as_deref(),
            )?;
            let mut policy = crate::p2::arc3_phase_a::PhaseAPolicy::with_tensor_model(
                &model,
                &device,
                config.physical_batch,
                crate::p2::latent_planning::config::PhaseAConfig::default(),
                calibration,
                config.action6_max_candidates,
                config.action6_grid_stride,
            )?;
            policy.set_context(context);
            let mut policy = AdaptingPolicy::new(policy, adapter);
            run_public_suite(&mut api, &mut policy, &settings)?
        }
    };
    write_json_atomic(&config.output, &report)?;
    Ok(report)
}

pub fn profile_recorded_decisions(
    checkpoint: &Path,
    train_config_path: &Path,
    device_name: &str,
    recordings_dir: &Path,
    report_output: &Path,
    physical_batch: usize,
) -> Result<()> {
    let train_config = load_train_config(train_config_path)?;
    let _gpu_guard = if device_name == "cuda" || device_name.starts_with("cuda:") {
        Some(GpuSessionGuard::acquire(&train_config.output_dir)?)
    } else {
        None
    };
    let device = resolve_device(device_name)?;
    let (model, _varmap) = load_model(&train_config, checkpoint, &device)?;
    let observations = first_recorded_decision_observations(recordings_dir)?;
    ensure!(
        !observations.is_empty(),
        "ARC recordings contain no non-terminal observations with available actions to profile"
    );
    let games = observations
        .iter()
        .map(|observation| PublicGame {
            game_id: observation.game_id.clone(),
            title: observation.game_id.clone(),
        })
        .collect::<Vec<_>>();
    let output_dir = report_output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut policy = ModelPolicy::new(
        &model,
        &device,
        physical_batch,
        128,
        8,
        DEFAULT_TRIED_PENALTY,
    );
    policy.enable_eval_profile(
        output_dir,
        format!("tofy.p2.arc3.{}", sha256_file(checkpoint)?),
        device_name,
    );
    policy.on_suite_start(&games)?;
    for recorded in observations {
        let observation = ArcObservation {
            game_id: recorded.game_id.clone(),
            guid: recorded.guid,
            frame: recorded.frame,
            animation: Vec::new(),
            full_reset: false,
            state: recorded.state,
            levels_completed: recorded.levels_completed,
            win_levels: recorded.win_levels,
            available_actions: recorded.available_actions,
        };
        policy.on_game_start(&recorded.game_id);
        policy.choose_action(&observation)?;
        policy.on_game_end("offline_replay_profiled");
    }
    Ok(())
}

pub(crate) struct LiveRunProvenance {
    pub(crate) git_revision: String,
    pub(crate) git_dirty: bool,
    pub(crate) dirty_diff_sha256: Option<String>,
    pub(crate) executable_sha256: Option<String>,
    pub(crate) build_profile: String,
    pub(crate) cli_args: Vec<String>,
}

pub(crate) fn live_run_provenance() -> Result<LiveRunProvenance> {
    let git_revision = git_output(&["rev-parse", "HEAD"])?;
    let status = git_output(&["status", "--porcelain", "--untracked-files=all"])?;
    let git_dirty = !status.trim().is_empty();
    let dirty_diff_sha256 = git_dirty
        .then(|| {
            git_output(&["diff", "--binary", "HEAD"]).map(|diff| sha256_bytes(diff.as_bytes()))
        })
        .transpose()?;
    let executable_sha256 = env::current_exe()
        .ok()
        .and_then(|path| sha256_file(&path).ok());
    Ok(LiveRunProvenance {
        git_revision: git_revision.trim().into(),
        git_dirty,
        dirty_diff_sha256,
        executable_sha256,
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
        .into(),
        cli_args: env::args().collect(),
    })
}

fn git_output(args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .output()
        .with_context(|| format!("run git {}", args.join(" ")))?;
    ensure!(
        output.status.success(),
        "git {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    String::from_utf8(output.stdout).context("git output is not UTF-8")
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

pub(crate) fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
    }
    let json = serde_json::to_string_pretty(value).context("serialize live eval report")?;
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    fs::write(&temporary, json).with_context(|| format!("write {}", temporary.display()))?;
    fs::rename(&temporary, path)
        .with_context(|| format!("rename {} -> {}", temporary.display(), path.display()))?;
    Ok(())
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, VecDeque};
    use std::path::Path;
    use std::sync::Mutex;

    fn frame(fill: u8) -> ArcFrame {
        ArcFrame::new(64, 64, vec![fill; 64 * 64]).unwrap()
    }

    fn observation(game_id: &str, state: &str, actions: Vec<u8>) -> ArcObservation {
        ArcObservation {
            game_id: game_id.into(),
            guid: format!("guid-{game_id}"),
            frame: frame(0),
            animation: Vec::new(),
            full_reset: false,
            state: state.into(),
            levels_completed: 0,
            win_levels: 1,
            available_actions: actions,
        }
    }

    struct FakeTransport {
        responses: Mutex<VecDeque<std::result::Result<HttpResponse, String>>>,
        requests: Mutex<Vec<HttpRequest>>,
    }

    impl FakeTransport {
        fn new(responses: Vec<std::result::Result<HttpResponse, String>>) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                requests: Mutex::new(Vec::new()),
            }
        }

        fn send_count(&self) -> usize {
            self.requests.lock().unwrap().len()
        }
    }

    impl HttpTransport for FakeTransport {
        fn send(&self, request: HttpRequest) -> std::result::Result<HttpResponse, String> {
            self.requests.lock().unwrap().push(request);
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("deterministic fake transport has a response")
        }
    }

    fn response(status: StatusCode, body: &str) -> HttpResponse {
        HttpResponse {
            status,
            body: Ok(body.into()),
        }
    }

    fn observation_body(game_id: &str, guid: &str, state: &str, actions: &[u8]) -> String {
        json!({
            "game_id": game_id,
            "guid": guid,
            "frame": [[[0]]],
            "full_reset": false,
            "state": state,
            "levels_completed": 0,
            "win_levels": 1,
            "available_actions": actions,
        })
        .to_string()
    }

    fn http_api(
        responses: Vec<std::result::Result<HttpResponse, String>>,
    ) -> HttpArcApi<FakeTransport> {
        HttpArcApi {
            transport: FakeTransport::new(responses),
            base_url: "https://arc.example".into(),
        }
    }

    #[test]
    fn idempotent_get_retries_and_eventually_succeeds() {
        let mut api = http_api(vec![
            Ok(response(StatusCode::TOO_MANY_REQUESTS, "slow down")),
            Ok(response(
                StatusCode::OK,
                r#"[{"game_id":"game","title":"Game"}]"#,
            )),
        ]);

        assert_eq!(api.list_games().unwrap()[0].game_id, "game");
        assert_eq!(api.transport.send_count(), 2);
        assert_eq!(
            api.transport.requests.lock().unwrap()[0].method,
            HttpMethod::Get
        );
    }

    #[test]
    fn action_transport_failure_is_ambiguous_and_sends_once() {
        let mut api = http_api(vec![Err("connection reset".into())]);
        let action = ArcAction::new(1, None, None).unwrap();

        let error = api.act("game", "guid", &action, &json!({})).unwrap_err();
        let MutationError::Ambiguous(ambiguous) = error else {
            panic!("transport failure must be ambiguous")
        };
        assert_eq!(ambiguous.game_id.as_deref(), Some("game"));
        assert_eq!(ambiguous.guid.as_deref(), Some("guid"));
        assert_eq!(ambiguous.action, Some(action));
        assert_eq!(api.transport.send_count(), 1);
    }

    #[test]
    fn action_response_body_read_failure_is_ambiguous_and_sends_once() {
        let mut api = http_api(vec![Ok(HttpResponse {
            status: StatusCode::OK,
            body: Err("connection closed while reading".into()),
        })]);
        let action = ArcAction::new(1, None, None).unwrap();

        assert!(matches!(
            api.act("game", "guid", &action, &json!({})),
            Err(MutationError::Ambiguous(_))
        ));
        assert_eq!(api.transport.send_count(), 1);
    }

    #[test]
    fn action_invalid_json_is_ambiguous_and_sends_once() {
        let mut api = http_api(vec![Ok(response(StatusCode::OK, "not json"))]);
        let action = ArcAction::new(1, None, None).unwrap();

        assert!(matches!(
            api.act("game", "guid", &action, &json!({})),
            Err(MutationError::Ambiguous(_))
        ));
        assert_eq!(api.transport.send_count(), 1);
    }

    #[test]
    fn action_rate_limit_is_ambiguous_and_sends_once() {
        let mut api = http_api(vec![Ok(response(
            StatusCode::TOO_MANY_REQUESTS,
            "slow down",
        ))]);
        let action = ArcAction::new(1, None, None).unwrap();

        assert!(matches!(
            api.act("game", "guid", &action, &json!({})),
            Err(MutationError::Ambiguous(_))
        ));
        assert_eq!(api.transport.send_count(), 1);
    }

    #[test]
    fn action_server_error_is_ambiguous_and_sends_once() {
        let mut api = http_api(vec![Ok(response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server failed",
        ))]);
        let action = ArcAction::new(1, None, None).unwrap();

        assert!(matches!(
            api.act("game", "guid", &action, &json!({})),
            Err(MutationError::Ambiguous(_))
        ));
        assert_eq!(api.transport.send_count(), 1);
    }

    #[test]
    fn action_semantic_observation_failure_is_ambiguous_and_preserves_attempt() {
        let mut api = http_api(vec![Ok(response(
            StatusCode::OK,
            &observation_body("game", "guid", "INVALID_STATE", &[1]),
        ))]);
        let action = ArcAction::new(1, None, None).unwrap();

        let error = api.act("game", "guid", &action, &json!({})).unwrap_err();
        let MutationError::Ambiguous(ambiguous) = error else {
            panic!("semantic response failure must be ambiguous")
        };
        assert_eq!(ambiguous.game_id.as_deref(), Some("game"));
        assert_eq!(ambiguous.guid.as_deref(), Some("guid"));
        assert_eq!(ambiguous.action, Some(action));
        assert_eq!(api.transport.send_count(), 1);
    }

    #[test]
    fn action_game_id_mismatch_is_ambiguous_and_preserves_attempt() {
        let mut api = http_api(vec![Ok(response(
            StatusCode::OK,
            &observation_body("other-game", "guid", "NOT_FINISHED", &[1]),
        ))]);
        let action = ArcAction::new(1, None, None).unwrap();

        let error = api.act("game", "guid", &action, &json!({})).unwrap_err();
        let MutationError::Ambiguous(ambiguous) = error else {
            panic!("game_id mismatch must be ambiguous")
        };
        assert_eq!(ambiguous.game_id.as_deref(), Some("game"));
        assert_eq!(ambiguous.guid.as_deref(), Some("guid"));
        assert_eq!(ambiguous.action, Some(action));
        assert_eq!(api.transport.send_count(), 1);
    }

    #[test]
    fn action_guid_mismatch_is_ambiguous_and_preserves_attempt() {
        let mut api = http_api(vec![Ok(response(
            StatusCode::OK,
            &observation_body("game", "other-guid", "NOT_FINISHED", &[1]),
        ))]);
        let action = ArcAction::new(1, None, None).unwrap();

        let error = api.act("game", "guid", &action, &json!({})).unwrap_err();
        let MutationError::Ambiguous(ambiguous) = error else {
            panic!("guid mismatch must be ambiguous")
        };
        assert_eq!(ambiguous.game_id.as_deref(), Some("game"));
        assert_eq!(ambiguous.guid.as_deref(), Some("guid"));
        assert_eq!(ambiguous.action, Some(action));
        assert_eq!(api.transport.send_count(), 1);
    }

    #[test]
    fn action_missing_coordinate_fails_before_send() {
        let mut api = http_api(Vec::new());
        let action = ArcAction {
            id: 6,
            x: None,
            y: Some(1),
        };

        assert!(matches!(
            api.act("game", "guid", &action, &json!({})),
            Err(MutationError::Failed(_))
        ));
        assert_eq!(api.transport.send_count(), 0);
    }

    #[test]
    fn non_retryable_get_client_error_fails_once() {
        let mut api = http_api(vec![Ok(response(StatusCode::BAD_REQUEST, "bad request"))]);

        assert!(api.list_games().is_err());
        assert_eq!(api.transport.send_count(), 1);
    }

    #[test]
    fn reset_request_scopes_retries_with_guid_and_preserves_full_reset() {
        let mut api = http_api(vec![
            Ok(response(
                StatusCode::OK,
                &json!({
                    "game_id": "game", "guid": "guid", "frame": [[[0]]],
                    "full_reset": true, "state": "NOT_FINISHED",
                    "levels_completed": 0, "win_levels": 1, "available_actions": [1]
                })
                .to_string(),
            )),
            Ok(response(
                StatusCode::OK,
                &json!({
                    "game_id": "game", "guid": "guid", "frame": [[[0]]],
                    "full_reset": false, "state": "NOT_FINISHED",
                    "levels_completed": 0, "win_levels": 1, "available_actions": [1]
                })
                .to_string(),
            )),
        ]);
        assert!(api.reset("game", "card", None).unwrap().full_reset);
        assert!(!api.reset("game", "card", Some("guid")).unwrap().full_reset);
        let requests = api.transport.requests.lock().unwrap();
        assert!(requests[0].body.as_ref().unwrap().get("guid").is_none());
        assert_eq!(requests[1].body.as_ref().unwrap()["guid"], "guid");
    }

    #[test]
    fn mutation_response_missing_full_reset_is_ambiguous() {
        let mut api = http_api(vec![Ok(response(
            StatusCode::OK,
            &json!({
                "game_id": "game", "guid": "guid", "frame": [[[0]]],
                "state": "NOT_FINISHED", "levels_completed": 0,
                "win_levels": 1, "available_actions": [1]
            })
            .to_string(),
        ))]);
        assert!(matches!(
            api.reset("game", "card", None),
            Err(MutationError::Ambiguous(_))
        ));
    }

    #[test]
    fn settled_frame_uses_last_animation_layer() {
        let api = ApiObservation {
            game_id: "demo".into(),
            guid: "guid".into(),
            frame: vec![vec![vec![1, 1]], vec![vec![7, 7]]],
            full_reset: false,
            state: "NOT_FINISHED".into(),
            levels_completed: 0,
            win_levels: 1,
            available_actions: vec![1],
        };
        let parsed = ArcObservation::try_from(api).unwrap();
        assert_eq!(parsed.frame.pixel(0, 0), Some(7));
        assert_eq!(parsed.frame.pixel(1, 0), Some(7));
        assert_eq!(parsed.animation.len(), 2);
        assert_eq!(parsed.animation[0].pixel(0, 0), Some(1));
        assert_eq!(parsed.animation[1].pixel(0, 0), Some(7));
    }

    #[test]
    fn live_api_observation_rejects_empty_guid() {
        let api = ApiObservation {
            game_id: "demo".into(),
            guid: String::new(),
            frame: vec![vec![vec![0]]],
            full_reset: false,
            state: "NOT_FINISHED".into(),
            levels_completed: 0,
            win_levels: 1,
            available_actions: vec![1],
        };
        assert!(ArcObservation::try_from(api)
            .unwrap_err()
            .to_string()
            .contains("empty guid"));
    }

    #[test]
    fn legacy_observation_json_defaults_animation_and_full_reset() {
        let parsed: ArcObservation = serde_json::from_value(json!({
            "game_id": "game",
            "guid": "guid",
            "frame": frame(0),
            "state": "NOT_FINISHED",
            "levels_completed": 0,
            "win_levels": 1,
            "available_actions": [1]
        }))
        .unwrap();
        assert!(parsed.animation.is_empty());
        assert!(!parsed.full_reset);
    }

    #[test]
    fn settled_frame_rejects_oversize_before_dimension_casts() {
        let api = ApiObservation {
            game_id: "demo".into(),
            guid: "guid".into(),
            frame: vec![vec![vec![0; 65]]],
            full_reset: false,
            state: "NOT_FINISHED".into(),
            levels_completed: 0,
            win_levels: 1,
            available_actions: vec![1],
        };
        assert!(ArcObservation::try_from(api).is_err());
    }

    #[test]
    fn identical_frames_in_different_games_have_separate_policy_history() {
        let first = observation("first", "NOT_FINISHED", vec![1]);
        let second = observation("second", "NOT_FINISHED", vec![1]);
        assert_ne!(observation_hash(&first), observation_hash(&second));
    }

    #[test]
    fn tried_action_hash_includes_row_63_when_it_has_non_background_content() {
        let first = observation("game", "NOT_FINISHED", vec![1]);
        assert!(!row63_has_content(&first.frame));
        let mut row63_content = first.clone();
        row63_content.frame.pixels[63 * FRAME_SIDE + 7] = 9;
        assert!(row63_has_content(&row63_content.frame));
        assert_ne!(observation_hash(&first), observation_hash(&row63_content));

        let mut gameplay_changed = first.clone();
        gameplay_changed.frame.pixels[62 * FRAME_SIDE + 7] = 9;
        assert_ne!(
            observation_hash(&first),
            observation_hash(&gameplay_changed)
        );
    }

    fn scored_action(id: u8, score: f64) -> ActionScore {
        ActionScore {
            action: ArcAction::new(id, None, None).unwrap(),
            score,
            q_probability: 0.0,
            reliability_probability: 0.0,
            noop_probability: 0.0,
            predicted_effect: 0.0,
        }
    }

    #[test]
    fn tried_penalty_explores_near_ties_but_keeps_large_margins() {
        let first = ArcAction::new(1, None, None).unwrap();
        let mut near_tie = vec![scored_action(1, 1.0), scored_action(2, 0.9)];
        let mut tried = BTreeSet::from([action_key(&first)]);
        apply_tried_penalty(&mut near_tie, &mut tried, DEFAULT_TRIED_PENALTY);
        assert!(near_tie[1].score > near_tie[0].score);

        let mut large_margin = vec![scored_action(1, 1.0), scored_action(2, 0.7)];
        apply_tried_penalty(&mut large_margin, &mut tried, DEFAULT_TRIED_PENALTY);
        assert!(large_margin[0].score > large_margin[1].score);
    }

    #[test]
    fn tried_penalty_clears_an_exhausted_history_once() {
        let mut scores = vec![scored_action(1, 0.8), scored_action(2, 0.7)];
        let mut tried =
            BTreeSet::from([action_key(&scores[0].action), action_key(&scores[1].action)]);
        apply_tried_penalty(&mut scores, &mut tried, DEFAULT_TRIED_PENALTY);
        assert!(tried.is_empty());
        assert_eq!(scores[0].score, 0.8);
        apply_tried_penalty(&mut scores, &mut tried, DEFAULT_TRIED_PENALTY);
        assert!(tried.is_empty());
        assert_eq!(scores[0].score, 0.8);
    }

    #[test]
    fn tried_history_isolated_by_guid_and_level() {
        let first = observation("game", "NOT_FINISHED", vec![1]);
        let mut other_guid = first.clone();
        other_guid.guid = "other-guid".into();
        let mut other_level = first.clone();
        other_level.levels_completed = 1;
        let mut histories: BTreeMap<u64, BTreeSet<String>> = BTreeMap::new();
        histories.insert(observation_hash(&first), BTreeSet::from(["1:-:-".into()]));

        assert_ne!(observation_hash(&first), observation_hash(&other_guid));
        assert_ne!(observation_hash(&first), observation_hash(&other_level));
        assert!(histories
            .get(&observation_hash(&other_guid))
            .is_none_or(BTreeSet::is_empty));
        assert!(histories
            .get(&observation_hash(&other_level))
            .is_none_or(BTreeSet::is_empty));
    }

    #[test]
    fn action_enumeration_masks_and_bounds_action6() {
        let mut obs = observation("demo", "NOT_FINISHED", vec![1, 5, 6]);
        obs.frame.pixels[10 * 64 + 20] = 3;
        let actions = enumerate_actions(&obs, 12, 16).unwrap();
        assert!(actions.contains(&ArcAction::new(1, None, None).unwrap()));
        assert!(actions.contains(&ArcAction::new(5, None, None).unwrap()));
        let coordinate_actions: Vec<_> = actions.iter().filter(|action| action.id == 6).collect();
        assert!(!coordinate_actions.is_empty());
        assert!(coordinate_actions.len() <= 12);
        assert!(coordinate_actions
            .iter()
            .all(|action| action.x.unwrap() < 64 && action.y.unwrap() < 64));
        assert!(!actions.iter().any(|action| action.id == 2));
    }

    #[test]
    fn invalid_available_actions_fail_closed() {
        let duplicate = observation("demo", "NOT_FINISHED", vec![1, 1]);
        assert!(enumerate_actions(&duplicate, 4, 8).is_err());
        let invalid = observation("demo", "NOT_FINISHED", vec![8]);
        assert!(enumerate_actions(&invalid, 4, 8).is_err());
    }

    #[test]
    fn action6_row63_candidates_follow_the_content_classification() {
        let plain = observation("demo", "NOT_FINISHED", vec![6]);
        let plain_actions = enumerate_actions(&plain, 128, 8).unwrap();
        assert!(plain_actions
            .iter()
            .all(|action| action.y.is_none_or(|y| y < 63)));

        let mut gameplay = plain;
        gameplay.frame.pixels[63 * FRAME_SIDE + 11] = 7;
        let gameplay_actions = enumerate_actions(&gameplay, 128, 8).unwrap();
        assert!(gameplay_actions
            .iter()
            .any(|action| action.x == Some(11) && action.y == Some(63)));
    }

    #[test]
    fn model_policy_scores_available_actions_from_one_observation() -> Result<()> {
        use crate::p2::model::ModelConfig;
        use candle_nn::{VarBuilder, VarMap};

        let device = Device::Cpu;
        let config = ModelConfig {
            hidden_dim: 8,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            ..ModelConfig::default()
        };
        let varmap = VarMap::new();
        let builder = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = WorldModel::new(config, builder)?;
        let mut policy = ModelPolicy::new(&model, &device, 4, 4, 32, DEFAULT_TRIED_PENALTY);
        let decision = policy.choose_action(&observation("demo", "NOT_FINISHED", vec![1, 2, 6]))?;
        assert!([1, 2, 6].contains(&decision.chosen.action.id));
        assert!(decision.chosen.score.is_finite());
        assert_eq!(decision.candidate_count, 6);
        Ok(())
    }

    #[test]
    fn model_policy_publishes_one_profile_bundle_per_game() -> Result<()> {
        use crate::p2::model::ModelConfig;
        use candle_nn::{VarBuilder, VarMap};

        let root =
            std::env::temp_dir().join(format!("tofy-arc3-decision-profile-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let device = Device::Cpu;
        let config = ModelConfig {
            hidden_dim: 8,
            action_dim: 4,
            inner_steps: 1,
            outer_steps: 1,
            ..ModelConfig::default()
        };
        let varmap = VarMap::new();
        let model = WorldModel::new(
            config,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        let mut policy = ModelPolicy::new(&model, &device, 4, 4, 32, DEFAULT_TRIED_PENALTY);
        policy.enable_eval_profile(&root, "tofy.p2.arc3.test".into(), "cpu");
        policy.on_suite_start(&[PublicGame {
            game_id: "demo".into(),
            title: "Demo".into(),
        }])?;
        policy.choose_action(&observation("demo", "NOT_FINISHED", vec![1, 2, 6]))?;

        let bundle = root.join("profile/arc3-demo");
        candle_graph::verify_bundle(&bundle)?;
        let trace = candle_graph::parse_trace(bundle.join("trace.jsonl"))?;
        for label in [
            "arc3/chosen_q_probability",
            "arc3/chosen_reliability_probability",
            "arc3/chosen_noop_probability",
            "arc3/chosen_predicted_effect",
        ] {
            assert!(trace.tensor_stats.iter().any(|event| event.label == label));
        }
        let status = candle_graph::campaign_status(&root.join("profile/arc3-campaign.json"))?;
        assert_eq!(status.published, 1);
        assert_eq!(status.missing, 0);

        fs::remove_dir_all(root)?;
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
            })
        }
    }

    #[derive(Default)]
    struct RecordingPolicy {
        events: Vec<String>,
    }

    impl LivePolicy for RecordingPolicy {
        fn choose_action(&mut self, observation: &ArcObservation) -> Result<ActionDecision> {
            FirstPolicy.choose_action(observation)
        }

        fn on_game_start(&mut self, game_id: &str) {
            self.events.push(format!("start:{game_id}"));
        }

        fn on_level_transition(&mut self, levels_completed: u16) {
            self.events.push(format!("level:{levels_completed}"));
        }

        fn on_reset_retry(&mut self, reason: &str) {
            self.events.push(format!("retry:{reason}"));
        }

        fn on_game_end(&mut self, outcome: &str) {
            self.events.push(format!("end:{outcome}"));
        }
    }

    struct ScriptedApi {
        resets: VecDeque<MutationResult<ArcObservation>>,
        actions: VecDeque<MutationResult<ArcObservation>>,
        reset_guids: Vec<Option<String>>,
        open_metadata: Option<Value>,
        closed: bool,
    }

    impl ScriptedApi {
        fn new(
            resets: Vec<MutationResult<ArcObservation>>,
            actions: Vec<MutationResult<ArcObservation>>,
        ) -> Self {
            Self {
                resets: resets.into(),
                actions: actions.into(),
                reset_guids: Vec::new(),
                open_metadata: None,
                closed: false,
            }
        }
    }

    impl ArcApi for ScriptedApi {
        fn list_games(&mut self) -> Result<Vec<PublicGame>> {
            Ok(vec![PublicGame {
                game_id: "game".into(),
                title: "Game".into(),
            }])
        }

        fn open_scorecard(&mut self, metadata: &Value) -> Result<String> {
            self.open_metadata = Some(metadata.clone());
            Ok("card".into())
        }

        fn reset(
            &mut self,
            _game_id: &str,
            _card_id: &str,
            guid: Option<&str>,
        ) -> MutationResult<ArcObservation> {
            self.reset_guids.push(guid.map(str::to_owned));
            self.resets.pop_front().expect("scripted RESET response")
        }

        fn act(
            &mut self,
            _game_id: &str,
            _guid: &str,
            _action: &ArcAction,
            _reasoning: &Value,
        ) -> MutationResult<ArcObservation> {
            self.actions.pop_front().expect("scripted ACTION response")
        }

        fn close_scorecard(&mut self, _card_id: &str) -> Result<Value> {
            self.closed = true;
            Ok(json!({}))
        }
    }

    fn scripted_observation(state: &str, level: u16) -> ArcObservation {
        let mut obs = observation(
            "game",
            state,
            (state == "NOT_FINISHED").then_some(1).into_iter().collect(),
        );
        obs.guid = "guid-game".into();
        obs.levels_completed = level;
        obs
    }

    fn scripted_settings(driver: LiveDriverOptions) -> LiveRunSettings {
        let evidence_class = live_evidence_class(&driver).into();
        LiveRunSettings {
            checkpoint: "model.safetensors".into(),
            checkpoint_sha256: "model-hash".into(),
            train_config: "config.json".into(),
            train_config_sha256: "config-hash".into(),
            device: "cpu".into(),
            base_url: "https://example.invalid".into(),
            requested_games: Vec::new(),
            driver,
            git_revision: "test-revision".into(),
            git_dirty: false,
            dirty_diff_sha256: None,
            executable_sha256: Some("test-executable".into()),
            build_profile: "test".into(),
            cli_args: vec!["p2-arc3-live-eval".into()],
            evidence_class,
            recordings_dir: None,
            contract: PolicyContract::greedy(),
        }
    }

    #[test]
    fn live_recording_round_trips_through_offline_importer() -> Result<()> {
        use crate::p2::arc3::{events_to_transitions, import_recordings_dir};

        let root = std::env::temp_dir().join(format!(
            "tofy-arc3-live-recording-roundtrip-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let opening = scripted_observation("NOT_FINISHED", 0);
        let mut after_action = scripted_observation("NOT_FINISHED", 0);
        after_action.frame.pixels[0] = 3;
        let mut win = scripted_observation("WIN", 1);
        win.win_levels = 1;
        win.frame.pixels[1] = 4;

        let mut recording = LiveRecordingRun::start(&root, &opening)?;
        recording.push_action(
            &after_action,
            &ArcAction::new(6, Some(3), Some(4))?,
            &json!({
                "q_probability": 0.7,
                "reliability_probability": 0.8,
                "noop_probability": 0.1,
                "predicted_effect": 0.5,
            }),
        )?;
        recording.push_action(
            &win,
            &ArcAction::new(1, None, None)?,
            &json!({ "q_probability": 0.9 }),
        )?;
        let direct = events_to_transitions(&recording.parsed_events())?;
        let path = recording.finish()?;
        let imported = import_recordings_dir(&root)?;
        assert_eq!(imported, direct);
        assert_eq!(path, root.join("game/guid-game.jsonl"));
        let jsonl = fs::read_to_string(&path)?;
        assert!(jsonl.contains("ACTION6"));
        assert!(jsonl.contains("q_probability"));
        assert!(!jsonl.contains("API_KEY"));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn competition_game_over_retries_current_level_and_records_lifecycle() {
        let opening = scripted_observation("NOT_FINISHED", 0);
        let retry = scripted_observation("NOT_FINISHED", 0);
        let game_over = scripted_observation("GAME_OVER", 0);
        let mut win = scripted_observation("WIN", 1);
        win.win_levels = 1;
        let mut api = ScriptedApi::new(vec![Ok(opening), Ok(retry)], vec![Ok(game_over), Ok(win)]);
        let mut policy = RecordingPolicy::default();
        let report = run_public_suite(
            &mut api,
            &mut policy,
            &scripted_settings(LiveDriverOptions {
                competition_mode: true,
                max_level_retries: 1,
                max_actions_per_level: 2,
                ..LiveDriverOptions::default()
            }),
        )
        .unwrap();

        let game = &report.games[0];
        assert_eq!(api.reset_guids, vec![None, Some("guid-game".into())]);
        assert_eq!(game.stop_reason, "completed");
        assert_eq!(game.actions, 2);
        assert_eq!(game.attempted_actions, 2);
        assert_eq!(game.reset_retries, 1);
        assert_eq!(game.level_usage[0].attempted_actions, 2);
        assert_eq!(game.level_usage[0].reset_retries, 1);
        assert_eq!(
            policy.events,
            vec!["start:game", "retry:game_over", "level:1", "end:completed"]
        );
        assert!(api.closed);
        assert_eq!(
            api.open_metadata.as_ref().unwrap()["competition_mode"],
            Value::Bool(true)
        );
    }

    #[test]
    fn general_mode_does_not_reset_a_terminal_opening_observation() {
        let mut api = ScriptedApi::new(vec![Ok(scripted_observation("GAME_OVER", 0))], Vec::new());
        let mut policy = RecordingPolicy::default();
        let report = run_public_suite(
            &mut api,
            &mut policy,
            &scripted_settings(LiveDriverOptions {
                competition_mode: false,
                ..LiveDriverOptions::default()
            }),
        )
        .unwrap();
        assert_eq!(api.reset_guids, vec![None]);
        assert_eq!(report.games[0].stop_reason, "terminal_game_over");
        assert_eq!(policy.events, vec!["start:game", "end:terminal_game_over"]);

        let mut legacy = serde_json::to_value(&report).unwrap();
        let root = legacy.as_object_mut().unwrap();
        root.remove("driver");
        let game = root["games"][0].as_object_mut().unwrap();
        for field in [
            "ambiguous_reset",
            "attempted_actions",
            "reset_retries",
            "full_reset_detected",
            "level_usage",
        ] {
            game.remove(field);
        }
        let restored: LiveEvalReport = serde_json::from_value(legacy).unwrap();
        assert!(restored.driver.competition_mode);
        assert_eq!(restored.games[0].attempted_actions, 0);
        assert!(restored.games[0].level_usage.is_empty());
    }

    #[test]
    fn retry_full_reset_fails_closed_and_per_level_cap_survives_game_over() {
        let opening = scripted_observation("NOT_FINISHED", 0);
        let game_over = scripted_observation("GAME_OVER", 0);
        let mut unsafe_reset = scripted_observation("NOT_FINISHED", 0);
        unsafe_reset.full_reset = true;
        let mut api = ScriptedApi::new(
            vec![Ok(opening.clone()), Ok(unsafe_reset)],
            vec![Ok(game_over.clone())],
        );
        let settings = scripted_settings(LiveDriverOptions {
            competition_mode: true,
            max_level_retries: 1,
            ..LiveDriverOptions::default()
        });
        let report = run_public_suite(&mut api, &mut FirstPolicy, &settings).unwrap();
        assert_eq!(report.games[0].stop_reason, "unsafe_retry_reset");
        assert!(report.games[0].full_reset_detected);

        let mut capped_api = ScriptedApi::new(vec![Ok(opening)], vec![Ok(game_over)]);
        let capped = run_public_suite(
            &mut capped_api,
            &mut FirstPolicy,
            &scripted_settings(LiveDriverOptions {
                competition_mode: true,
                max_actions_per_level: 1,
                ..LiveDriverOptions::default()
            }),
        )
        .unwrap();
        assert_eq!(capped.games[0].stop_reason, "level_action_cap_reached");
        assert_eq!(capped.games[0].reset_retries, 0);
        assert_eq!(capped_api.reset_guids, vec![None]);
    }

    #[test]
    fn per_level_cap_does_not_impose_a_default_game_wide_stop() {
        let mut opening = scripted_observation("NOT_FINISHED", 0);
        opening.win_levels = 6;
        let mut actions = Vec::new();
        for level in 0..5 {
            for attempt in 0..100 {
                let mut next = scripted_observation("NOT_FINISHED", level);
                next.win_levels = 6;
                if attempt == 99 {
                    next.levels_completed = level + 1;
                }
                actions.push(Ok(next));
            }
        }
        for _ in 0..11 {
            let mut next = scripted_observation("NOT_FINISHED", 5);
            next.win_levels = 6;
            actions.push(Ok(next));
        }
        let mut win = scripted_observation("WIN", 6);
        win.win_levels = 6;
        actions.push(Ok(win));

        let mut api = ScriptedApi::new(vec![Ok(opening)], actions);
        let report = run_public_suite(
            &mut api,
            &mut FirstPolicy,
            &scripted_settings(LiveDriverOptions {
                max_actions_per_level: 100,
                ..LiveDriverOptions::default()
            }),
        )
        .unwrap();
        let game = &report.games[0];
        assert_eq!(game.actions, 512);
        assert_eq!(game.stop_reason, "completed");
        assert!(game
            .level_usage
            .iter()
            .all(|usage| usage.attempted_actions <= 100));
    }

    #[test]
    fn action_full_reset_fails_closed_before_confirmation() {
        let opening = scripted_observation("NOT_FINISHED", 0);
        let mut reset_action = scripted_observation("NOT_FINISHED", 0);
        reset_action.full_reset = true;
        let mut api = ScriptedApi::new(vec![Ok(opening)], vec![Ok(reset_action)]);
        let report = run_public_suite(
            &mut api,
            &mut FirstPolicy,
            &scripted_settings(LiveDriverOptions::default()),
        )
        .unwrap();
        assert_eq!(report.games[0].stop_reason, "protocol_full_reset");
        assert!(report.games[0].full_reset_detected);
        assert_eq!(report.games[0].attempted_actions, 1);
        assert_eq!(report.games[0].actions, 0);
    }

    #[test]
    fn every_action_attempt_has_one_ordered_ledger_outcome() {
        let opening = scripted_observation("NOT_FINISHED", 0);
        let mut confirmed = scripted_observation("WIN", 1);
        confirmed.win_levels = 1;
        let ambiguous = AmbiguousMutation {
            operation: "submit action".into(),
            game_id: Some("game".into()),
            guid: Some("guid-game".into()),
            action: Some(ArcAction::new(1, None, None).unwrap()),
            cause: "timeout".into(),
        };
        let mut full_reset = scripted_observation("NOT_FINISHED", 0);
        full_reset.full_reset = true;
        let cases = vec![
            (Ok(confirmed), "confirmed"),
            (Err(MutationError::Ambiguous(ambiguous)), "ambiguous"),
            (
                Err(MutationError::Failed(anyhow::anyhow!("HTTP 400"))),
                "rejected",
            ),
            (Ok(full_reset), "protocol"),
        ];

        for (action, expected) in cases {
            let mut api = ScriptedApi::new(vec![Ok(opening.clone())], vec![action]);
            let report = run_public_suite(
                &mut api,
                &mut FirstPolicy,
                &scripted_settings(LiveDriverOptions::default()),
            )
            .unwrap();
            let game = &report.games[0];
            assert_eq!(game.attempted_actions, 1);
            assert_eq!(game.mutation_attempts.len(), 1);
            let attempt = &game.mutation_attempts[0];
            assert_eq!(attempt.index, 0);
            assert_eq!(attempt.game_id, "game");
            assert_eq!(attempt.guid, "guid-game");
            assert_eq!(attempt.action.id, 1);
            assert_eq!(
                match &attempt.outcome {
                    MutationAttemptOutcome::Confirmed => "confirmed",
                    MutationAttemptOutcome::Ambiguous { .. } => "ambiguous",
                    MutationAttemptOutcome::Rejected { .. } => "rejected",
                    MutationAttemptOutcome::ProtocolViolation { .. } => "protocol",
                },
                expected
            );
            match &attempt.outcome {
                MutationAttemptOutcome::Ambiguous { mutation } => {
                    assert_eq!(mutation.cause, "timeout");
                }
                MutationAttemptOutcome::Rejected { reason } => {
                    assert!(reason.contains("HTTP 400"));
                }
                MutationAttemptOutcome::ProtocolViolation { kind } => {
                    assert_eq!(kind, "full_reset");
                    assert_eq!(game.agent_session.observations.len(), 2);
                }
                MutationAttemptOutcome::Confirmed => {}
            }
            if expected == "confirmed" {
                assert_eq!(game.confirmed_actions, 1);
                assert_eq!(game.agent_session.experience_graph.edges.len(), 1);
                assert!(game.agent_session.experience_graph.edges[0].to.is_some());
            } else {
                assert_eq!(game.confirmed_actions, 0);
                assert!(game
                    .agent_session
                    .experience_graph
                    .edges
                    .iter()
                    .all(|edge| edge.to.is_none()));
            }
        }
    }

    #[test]
    fn ambiguous_opening_reset_is_reported_and_game_end_fires_once() {
        let mutation = AmbiguousMutation {
            operation: "reset game".into(),
            game_id: Some("game".into()),
            guid: None,
            action: None,
            cause: "lost response".into(),
        };
        let mut api = ScriptedApi::new(vec![Err(MutationError::Ambiguous(mutation))], Vec::new());
        let mut policy = RecordingPolicy::default();
        let report = run_public_suite(
            &mut api,
            &mut policy,
            &scripted_settings(LiveDriverOptions::default()),
        )
        .unwrap();
        assert_eq!(report.games[0].stop_reason, "ambiguous_reset");
        assert!(report.games[0].ambiguous_reset.is_some());
        assert_eq!(policy.events, vec!["end:ambiguous_reset"]);
    }

    #[test]
    fn dirty_driver_refuses_scorecard_without_exploratory_override() {
        let mut api = ScriptedApi::new(Vec::new(), Vec::new());
        let mut settings = scripted_settings(LiveDriverOptions::default());
        settings.git_dirty = true;
        settings.dirty_diff_sha256 = Some("dirty-diff".into());

        assert!(run_public_suite(&mut api, &mut FirstPolicy, &settings).is_err());
        assert!(api.open_metadata.is_none());
    }

    #[test]
    fn exploratory_dirty_driver_records_provenance_class_and_digest() {
        let opening = scripted_observation("NOT_FINISHED", 0);
        let mut win = scripted_observation("WIN", 1);
        win.win_levels = 1;
        let mut api = ScriptedApi::new(vec![Ok(opening)], vec![Ok(win)]);
        let mut settings = scripted_settings(LiveDriverOptions {
            exploratory: true,
            ..LiveDriverOptions::default()
        });
        settings.git_dirty = true;
        settings.dirty_diff_sha256 = Some("dirty-diff".into());

        let report = run_public_suite(&mut api, &mut FirstPolicy, &settings).unwrap();
        assert_eq!(report.evidence_class, "exploratory_driver_repair");
        assert_eq!(report.dirty_diff_sha256.as_deref(), Some("dirty-diff"));
        assert_eq!(report.git_revision, "test-revision");
    }

    struct FakeApi {
        closed: bool,
        acted_games: Vec<String>,
        ambiguous_action: bool,
    }

    impl ArcApi for FakeApi {
        fn list_games(&mut self) -> Result<Vec<PublicGame>> {
            Ok(vec![
                PublicGame {
                    game_id: "b-1".into(),
                    title: "B".into(),
                },
                PublicGame {
                    game_id: "a-1".into(),
                    title: "A".into(),
                },
            ])
        }

        fn open_scorecard(&mut self, metadata: &Value) -> Result<String> {
            assert_eq!(metadata["opaque"]["public_data_used_for_fitting"], false);
            Ok("card".into())
        }

        fn reset(
            &mut self,
            game_id: &str,
            _card_id: &str,
            _guid: Option<&str>,
        ) -> MutationResult<ArcObservation> {
            Ok(observation(game_id, "NOT_FINISHED", vec![1]))
        }

        fn act(
            &mut self,
            game_id: &str,
            guid: &str,
            action: &ArcAction,
            _reasoning: &Value,
        ) -> MutationResult<ArcObservation> {
            self.acted_games.push(game_id.into());
            if self.ambiguous_action {
                return Err(MutationError::Ambiguous(AmbiguousMutation {
                    operation: "submit action".into(),
                    game_id: Some(game_id.into()),
                    guid: Some(guid.into()),
                    action: Some(action.clone()),
                    cause: "connection reset".into(),
                }));
            }
            let mut next = observation(game_id, "WIN", vec![]);
            next.levels_completed = 1;
            Ok(next)
        }

        fn close_scorecard(&mut self, card_id: &str) -> Result<Value> {
            assert_eq!(card_id, "card");
            self.closed = true;
            Ok(json!({
                "score": 100,
                "environments": [{
                    "level_count": 1,
                    "runs": [{
                        "levels_completed": 1,
                        "number_of_levels": 1,
                        "level_actions": [1],
                        "level_baseline_actions": [1]
                    }]
                }]
            }))
        }
    }

    #[test]
    fn suite_evaluates_every_discovered_game_and_closes_scorecard() {
        let mut api = FakeApi {
            closed: false,
            acted_games: Vec::new(),
            ambiguous_action: false,
        };
        let settings = scripted_settings(LiveDriverOptions::default());
        let report = run_public_suite(&mut api, &mut FirstPolicy, &settings).unwrap();
        assert!(api.closed);
        assert_eq!(api.acted_games, vec!["a-1", "b-1"]);
        assert_eq!(report.selected_game_count, 2);
        assert!(report.evaluated_all_discovered_games);
        assert!(report.held_out_only);
        assert!(!report.public_data_used_for_fitting);
        assert_eq!(report.schema, LIVE_REPORT_SCHEMA);
        assert!(report.goal_feature_contract.contains("30% goal dropout"));
        assert!(report
            .tried_action_key_contract
            .contains("row 63 participates"));
        assert_eq!(report.official_rhae, Some(100.0));
        assert!(report.official_scorecard_parse_error.is_none());
        assert!(report
            .games
            .iter()
            .all(|game| game.stop_reason == "completed"));
    }

    #[test]
    fn ambiguous_attempt_is_recorded_in_the_mutation_ledger() {
        let mut api = FakeApi {
            closed: false,
            acted_games: Vec::new(),
            ambiguous_action: true,
        };
        let mut settings = scripted_settings(LiveDriverOptions::default());
        settings.requested_games = vec!["a-1".into()];

        let report = run_public_suite(&mut api, &mut FirstPolicy, &settings).unwrap();
        let game = &report.games[0];
        assert!(api.closed);
        assert_eq!(game.stop_reason, "ambiguous_mutation");
        assert_eq!(game.actions, 0);
        assert!(game.trace.is_empty());
        assert_eq!(game.mutation_attempts.len(), 1);
        assert!(matches!(
            game.mutation_attempts[0].outcome,
            MutationAttemptOutcome::Ambiguous { .. }
        ));
    }

    #[test]
    fn semantic_action_failure_is_excluded_from_confirmed_trace_and_scorecard_closes() {
        let mut api = http_api(vec![
            Ok(response(
                StatusCode::OK,
                r#"[{"game_id":"game","title":"Game"}]"#,
            )),
            Ok(response(StatusCode::OK, r#"{"card_id":"card"}"#)),
            Ok(response(
                StatusCode::OK,
                &observation_body("game", "guid-game", "NOT_FINISHED", &[1]),
            )),
            Ok(response(
                StatusCode::OK,
                &observation_body("game", "guid-game", "INVALID_STATE", &[1]),
            )),
            Ok(response(StatusCode::OK, "{}")),
        ]);
        let mut settings = scripted_settings(LiveDriverOptions::default());
        settings.requested_games = vec!["game".into()];

        let report = run_public_suite(&mut api, &mut FirstPolicy, &settings).unwrap();
        let game = &report.games[0];
        assert_eq!(game.stop_reason, "ambiguous_mutation");
        assert_eq!(game.actions, 0);
        assert!(game.trace.is_empty());
        assert!(matches!(
            game.mutation_attempts[0].outcome,
            MutationAttemptOutcome::Ambiguous { .. }
        ));
        assert_eq!(api.transport.send_count(), 5);
    }

    fn collect_rust_sources(root: &Path, sources: &mut BTreeMap<String, String>) {
        for entry in std::fs::read_dir(root).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                collect_rust_sources(&path, sources);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                let relative = path
                    .strip_prefix(Path::new(env!("CARGO_MANIFEST_DIR")).join("src"))
                    .unwrap()
                    .display()
                    .to_string();
                sources.insert(relative, std::fs::read_to_string(path).unwrap());
            }
        }
    }

    fn contains_forbidden_training_import(source: &str) -> bool {
        [
            "crate::p2::arc3",
            "crate::p2::arc3_live",
            "import_recordings_dir",
            "RecordingRunSummary",
            "ARC_API_KEY",
            "reqwest::",
        ]
        .iter()
        .any(|forbidden| source.contains(forbidden))
    }

    fn function_source<'a>(source: &'a str, signature: &str) -> &'a str {
        let start = source.find(signature).unwrap();
        let body_start = start + source[start..].find('{').unwrap();
        let mut depth = 0usize;
        for (offset, byte) in source[body_start..].bytes().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &source[start..=body_start + offset];
                    }
                }
                _ => {}
            }
        }
        panic!("unterminated function {signature}");
    }

    #[test]
    fn training_source_cannot_depend_on_live_or_recording_modules() {
        let source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut sources = BTreeMap::new();
        collect_rust_sources(&source_root, &mut sources);
        for required in ["p2/data.rs", "p2/train.rs", "p2/eval.rs", "p2/model.rs"] {
            assert!(
                sources.contains_key(required),
                "source scan missed {required}"
            );
        }

        for (path, source) in &sources {
            if matches!(
                path.as_str(),
                // Live/eval-side modules only. `arc3_phase_a.rs` is the ADR
                // 0004 controller: it is constructed from the live driver and
                // the bridge, and no training-reachable source imports it.
                // `residual_probe.rs` is the ADR 0005 §5.3 frozen-checkpoint
                // probe CLI over imported recordings; only `cli.rs` reaches it.
                "p2/agent_session.rs"
                    | "p2/arc3.rs"
                    | "p2/arc3_bridge.rs"
                    | "p2/arc3_live.rs"
                    | "p2/arc3_phase_a.rs"
                    // ADR 0005 §5.3 frozen-checkpoint probe: reads toolkit
                    // recordings for evaluation only and is reachable solely
                    // from the CLI dispatcher, never from training code.
                    | "p2/residual_probe.rs"
                    | "p2/cli.rs"
                    | "p2/residual_probe.rs"
            ) {
                continue;
            }
            let source = if path == "p2/eval.rs" {
                function_source(source, "pub fn evaluate_gate_support_with_content_masks")
            } else {
                source
            };
            assert!(
                !contains_forbidden_training_import(source),
                "training-reachable source {path} contains a held-out dependency"
            );
        }

        let data = &sources["p2/data.rs"];
        let fixture = format!("{data}\nuse crate::p2::arc3::import_recordings_dir;");
        assert!(contains_forbidden_training_import(&fixture));
    }

    fn varmap_bits(varmap: &candle_nn::VarMap) -> Result<Vec<(String, Vec<u32>)>> {
        let data = varmap.data().lock().unwrap();
        let mut out = Vec::new();
        for (name, var) in data.iter() {
            let bits = var
                .as_tensor()
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect();
            out.push((name.clone(), bits));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    fn tiny_live_model(device: &Device, v6: bool) -> Result<(WorldModel, candle_nn::VarMap)> {
        use crate::p2::experiment::ConsumerReadoutTopology;
        use crate::p2::model::ModelConfig;
        let config = ModelConfig {
            patch_size: 4,
            hidden_dim: 8,
            action_dim: 8,
            inner_steps: 1,
            outer_steps: 1,
            spatial_action_field: true,
            world_core_v4: true,
            world_core_v5: true,
            world_core_v6: v6,
            consumer_readout: ConsumerReadoutTopology::SpatialQuery,
            ..ModelConfig::default()
        };
        let varmap = candle_nn::VarMap::new();
        let model = WorldModel::new(
            config,
            candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        if v6 {
            // Context FiLM is zero-initialised (exact v5 recovery), so make the
            // channel observable before asserting a window reached the model.
            let data = varmap.data().lock().unwrap();
            let var = data.get("context_film_gamma.weight").expect("v6 parameter");
            var.set(&var.as_tensor().ones_like()?.affine(0.5, 0.0)?)?;
        }
        Ok((model, varmap))
    }

    fn observe_transitions(policy: &mut ModelPolicy<'_>, count: u8, level: u16) {
        for i in 0..count {
            let mut current = observation("demo", "NOT_FINISHED", vec![1, 2, 6]);
            current.frame = frame(10 + i);
            current.levels_completed = level;
            let mut next = current.clone();
            next.frame = frame(11 + i);
            let action = ArcAction::new(1 + i % 5, None, None).unwrap();
            policy.on_confirmed_transition(&current, &action, &next);
        }
    }

    #[test]
    fn greedy_policy_scores_candidates_with_the_context_window() -> Result<()> {
        let device = Device::Cpu;
        let (model, _varmap) = tiny_live_model(&device, true)?;
        let mut policy = ModelPolicy::new(&model, &device, 4, 4, 32, DEFAULT_TRIED_PENALTY);
        let context = live_context_for(&model, None, false);
        assert!(context.enabled(), "v6 defaults to Channel A on");
        policy.set_context(context);
        policy.on_game_start("demo");
        let observation = observation("demo", "NOT_FINISHED", vec![1, 2, 6]);
        let candidates = enumerate_actions(&observation, 4, 32)?;
        let plain = policy.score_candidates(&observation.frame, &candidates, &[], None)?;

        observe_transitions(&mut policy, 3, 0);
        let window = policy.context.window(0);
        assert_eq!(window.len(), 3);
        let contextual = policy.score_candidates(&observation.frame, &candidates, &window, None)?;
        assert_eq!(plain.len(), contextual.len());
        assert!(
            plain
                .iter()
                .zip(&contextual)
                .any(|(a, b)| a.score != b.score),
            "the ContextBatch must reach the forward pass"
        );
        for score in &contextual {
            assert!(score.score.is_finite());
        }
        let decision = policy.choose_action(&observation)?;
        assert_eq!(decision.context_len, 3);
        assert_eq!(decision_telemetry(&decision)["context_len"], json!(3));
        // Another level sees none of level 0's transitions (default arm).
        let mut later = observation.clone();
        later.levels_completed = 1;
        assert_eq!(policy.choose_action(&later)?.context_len, 0);
        // `--context-window=false` keeps a v6 policy on the legacy path.
        policy.set_context(live_context_for(&model, Some(false), false));
        observe_transitions(&mut policy, 3, 0);
        assert_eq!(policy.choose_action(&observation)?.context_len, 0);
        Ok(())
    }

    #[test]
    fn v5_checkpoint_forces_the_context_window_off_without_error() -> Result<()> {
        let device = Device::Cpu;
        let (model, _varmap) = tiny_live_model(&device, false)?;
        let context = live_context_for(&model, Some(true), true);
        assert!(!context.enabled(), "non-v6 checkpoints reject context");
        let mut policy = ModelPolicy::new(&model, &device, 4, 4, 32, DEFAULT_TRIED_PENALTY);
        policy.set_context(context);
        policy.on_game_start("demo");
        observe_transitions(&mut policy, 4, 0);
        let decision = policy.choose_action(&observation("demo", "NOT_FINISHED", vec![1, 2, 6]))?;
        assert_eq!(decision.context_len, 0);
        assert!(decision.chosen.score.is_finite());
        Ok(())
    }

    #[test]
    fn driver_window_holds_only_transitions_confirmed_before_each_decision() -> Result<()> {
        let device = Device::Cpu;
        let (model, _varmap) = tiny_live_model(&device, true)?;
        let mut policy = ModelPolicy::new(&model, &device, 4, 4, 32, DEFAULT_TRIED_PENALTY);
        policy.set_context(live_context_for(&model, None, false));
        let opening = scripted_observation("NOT_FINISHED", 0);
        let mut mid = scripted_observation("NOT_FINISHED", 0);
        mid.frame = frame(5);
        let mut later = scripted_observation("NOT_FINISHED", 0);
        later.frame = frame(6);
        let mut win = scripted_observation("WIN", 1);
        win.win_levels = 1;
        let mut api = ScriptedApi::new(vec![Ok(opening)], vec![Ok(mid), Ok(later), Ok(win)]);
        let report = run_public_suite(
            &mut api,
            &mut policy,
            &scripted_settings(LiveDriverOptions::default()),
        )?;
        let game = &report.games[0];
        assert_eq!(game.stop_reason, "completed");
        let lens: Vec<usize> = game
            .trace
            .iter()
            .map(|step| step.decision.context_len)
            .collect();
        assert_eq!(
            lens,
            vec![0, 1, 2],
            "decision i sees exactly the i transitions confirmed before it"
        );
        let json = serde_json::to_value(&game.trace[2].decision)?;
        assert_eq!(json["context_len"], json!(2));
        Ok(())
    }

    #[test]
    fn adapting_policy_observes_transitions_and_restores_prior_at_game_end() -> Result<()> {
        use crate::p2::adaptation::AdaptationMode;
        use crate::p2::experiment::ConsumerReadoutTopology;
        use crate::p2::model::ModelConfig;
        use candle_nn::{VarBuilder, VarMap};

        let device = Device::Cpu;
        let config = ModelConfig {
            patch_size: 4,
            hidden_dim: 8,
            action_dim: 8,
            goal_dim: 6,
            inner_steps: 1,
            outer_steps: 1,
            spatial_action_field: true,
            world_core_v4: true,
            world_core_v5: true,
            consumer_readout: ConsumerReadoutTopology::SpatialQuery,
            ..ModelConfig::default()
        };
        let varmap = VarMap::new();
        let model = WorldModel::new(
            config,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
        )?;
        let theta0 = varmap_bits(&varmap)?;
        let adapter = adaptation_for(&model, &varmap, &device, true, false)?;
        assert!(adapter.is_some());
        assert!(adaptation_for(&model, &varmap, &device, false, false)?.is_none());
        let mut policy = AdaptingPolicy::new(RecordingPolicy::default(), adapter);

        let opening = scripted_observation("NOT_FINISHED", 0);
        let mid = scripted_observation("NOT_FINISHED", 0);
        let mut win = scripted_observation("WIN", 1);
        win.win_levels = 1;
        let mut api = ScriptedApi::new(vec![Ok(opening)], vec![Ok(mid), Ok(win)]);
        let report = run_public_suite(
            &mut api,
            &mut policy,
            &scripted_settings(LiveDriverOptions::default()),
        )?;
        let game = &report.games[0];
        assert_eq!(game.stop_reason, "completed");
        assert_eq!(game.trace.len(), 2);
        assert!(
            game.trace[0].decision.adaptation.is_none(),
            "no factual transition precedes the first decision"
        );
        let trace = game.trace[1]
            .decision
            .adaptation
            .as_ref()
            .expect("the confirmed transition arms maybe_update before the next decision");
        assert_eq!(trace.mode, AdaptationMode::Reset);
        assert_eq!(trace.note.as_deref(), Some("warmup"));
        assert_eq!(trace.updates, 0);
        let telemetry = decision_telemetry(&game.trace[1].decision);
        assert_eq!(
            telemetry["adaptation"]["mode"],
            Value::String("reset".into())
        );
        let first = serde_json::to_value(&game.trace[0].decision)?;
        assert!(
            first.get("adaptation").is_none(),
            "absent trace is not serialized"
        );

        // Game end: buffer discarded and every parameter is theta_0 bitwise.
        let adapter = policy.adapter().expect("adapter retained");
        assert!(adapter.buffer().is_empty());
        assert!(adapter.fast_weights_equal_prior()?);
        assert_eq!(varmap_bits(&varmap)?, theta0);
        assert_eq!(
            policy.inner.events,
            vec!["start:game", "level:1", "end:completed"],
            "inner controller lifecycle is forwarded unchanged"
        );
        Ok(())
    }
}
