//! Held-out live evaluation on every public ARC-AGI-3 environment.
//!
//! This module is deliberately downstream of training. It can load a frozen
//! checkpoint and submit actions, but exposes no samples, gradients, optimizer,
//! checkpoint selection, or curriculum hooks back to `p2::train`.

use crate::gpu_lock::GpuSessionGuard;
use crate::p2::agent_session::AgentSession;
use crate::p2::data::{ArcAction, ArcFrame, FRAME_SIDE, GOAL_FEATURES_DIM};
use crate::p2::eval::load_model;
use crate::p2::model::{
    latent_mse_per_sample, RecursionDepth, RecursionOpts, WorldModel, EVENT_NOOP,
};
use crate::p2::rhae::{
    benchmark_from_scorecard_str, official_rhae_from_benchmark, ScorecardBenchmark,
};
use crate::p2::train::{frames_to_indices, load_train_config, resolve_device};
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
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
use std::io::Read;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const LIVE_REPORT_SCHEMA: &str = "p2.arc3_live_report.v2";
pub const LIVE_POLICY: &str = "model_reliable_effect_v1";
const POLICY_LIMITATION: &str = "The checkpoint predicts transition fidelity, reliability, no-op probability, and latent action effect; it has no trained reward/value head. This exploratory policy is not a hidden-goal solver.";
const GOAL_FEATURE_CONTRACT: &str = "Live policy supplies the all-zero goal vector. Foundation-v2 trains with 30% goal dropout, so this goal-free query is in-distribution; it does not provide hidden-goal evidence.";
const TRIED_ACTION_KEY_CONTRACT: &str = "game id + frame dimensions + visible pixels in rows [0,63); row 63 is excluded because training uses it as synthetic status while live games may contain real content there";
const MAX_HTTP_ATTEMPTS: usize = 5;

#[derive(Debug, Clone)]
pub struct LiveEvalConfig {
    pub checkpoint: PathBuf,
    pub train_config: PathBuf,
    pub device: String,
    pub base_url: String,
    pub api_key_env: String,
    pub games: Vec<String>,
    pub max_actions_per_game: usize,
    pub physical_batch: usize,
    pub action6_max_candidates: usize,
    pub action6_grid_stride: usize,
    pub request_timeout_secs: u64,
    pub output: PathBuf,
}

impl LiveEvalConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.physical_batch > 0, "physical_batch must be > 0");
        ensure!(
            self.max_actions_per_game > 0,
            "max_actions_per_game must be > 0"
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
    pub state: String,
    pub levels_completed: u16,
    pub win_levels: u16,
    pub available_actions: Vec<u8>,
}

impl ArcObservation {
    fn validate(&self) -> Result<()> {
        ensure!(!self.game_id.is_empty(), "ARC response has empty game_id");
        ensure!(!self.guid.is_empty(), "ARC response has empty guid");
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

    fn terminal(&self) -> bool {
        self.state != "NOT_FINISHED"
            || (self.win_levels > 0 && self.levels_completed >= self.win_levels)
    }
}

#[derive(Debug, Deserialize)]
struct ApiObservation {
    game_id: String,
    guid: String,
    frame: Vec<Vec<Vec<u8>>>,
    state: String,
    levels_completed: u16,
    win_levels: u16,
    available_actions: Vec<u8>,
}

impl TryFrom<ApiObservation> for ArcObservation {
    type Error = anyhow::Error;

    fn try_from(value: ApiObservation) -> Result<Self> {
        let settled = value
            .frame
            .last()
            .context("ARC response contains no frames")?;
        ensure!(!settled.is_empty(), "ARC response contains an empty frame");
        let width = settled[0].len();
        ensure!(width > 0, "ARC response contains a zero-width frame");
        ensure!(
            width <= 64 && settled.len() <= 64,
            "ARC response frame is {}x{}, exceeding the supported 64x64 canvas",
            width,
            settled.len()
        );
        ensure!(
            settled.iter().all(|row| row.len() == width),
            "ARC response frame is ragged"
        );
        let pixels = settled.iter().flatten().copied().collect();
        let frame = ArcFrame::new(width as u16, settled.len() as u16, pixels)?.to_fixed_64()?;
        let observation = Self {
            game_id: value.game_id,
            guid: value.guid,
            frame,
            state: value.state,
            levels_completed: value.levels_completed,
            win_levels: value.win_levels,
            available_actions: value.available_actions,
        };
        observation.validate()?;
        Ok(observation)
    }
}

pub trait ArcApi {
    fn list_games(&mut self) -> Result<Vec<PublicGame>>;
    fn open_scorecard(&mut self, metadata: &Value) -> Result<String>;
    fn reset(&mut self, game_id: &str, card_id: &str) -> Result<ArcObservation>;
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

    fn reset(&mut self, game_id: &str, card_id: &str) -> Result<ArcObservation> {
        let response: ApiObservation = self.post(
            "/api/cmd/RESET",
            &json!({ "game_id": game_id, "card_id": card_id }),
            "reset game",
        )?;
        let observation = ArcObservation::try_from(response)?;
        ensure!(
            observation.game_id == game_id,
            "RESET response game_id does not match request"
        );
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
}

pub trait LivePolicy {
    fn choose_action(&mut self, observation: &ArcObservation) -> Result<ActionDecision>;
}

pub struct ModelPolicy<'a> {
    model: &'a WorldModel,
    device: &'a Device,
    physical_batch: usize,
    action6_max_candidates: usize,
    action6_grid_stride: usize,
    tried: BTreeMap<u64, BTreeSet<String>>,
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
    ) -> Self {
        Self {
            model,
            device,
            physical_batch,
            action6_max_candidates,
            action6_grid_stride,
            tried: BTreeMap::new(),
        }
    }

    fn score_candidates(
        &self,
        frame: &ArcFrame,
        candidates: &[ArcAction],
    ) -> Result<Vec<ActionScore>> {
        let frames = frames_to_indices(std::slice::from_ref(frame), self.device)?;
        let encoded = self.model.encode_state(&frames)?;
        let (_, channels, height, width) = encoded.dims4()?;
        let mut scores = Vec::with_capacity(candidates.len());
        for chunk in candidates.chunks(self.physical_batch) {
            let n = chunk.len();
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
            // Foundation-v2 applies 30% goal dropout during training. The live
            // goal-free query is therefore deliberately the in-distribution
            // all-zero vector, not a fabricated hidden-goal guess.
            let goals = Tensor::zeros((n, GOAL_FEATURES_DIM), DType::F32, self.device)?;
            let state = encoded.broadcast_as((n, channels, height, width))?;
            let frame_batch = frames.broadcast_as((n, 1, FRAME_SIDE, FRAME_SIDE))?;
            let output = self.model.forward_from_encoded_state(
                &state,
                &frame_batch,
                &actions,
                &coords,
                &goals,
                RecursionDepth::from_config(self.model.config()),
                0.0,
                None,
                RecursionOpts::EVAL,
            )?;
            let q = ops::sigmoid(&output.q_logit)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let reliability = ops::sigmoid(&output.reliability_logit)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let noop = ops::sigmoid(&output.event_logits.narrow(1, EVENT_NOOP, 1)?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let effect = latent_mse_per_sample(&output.y, &state)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for index in 0..n {
                let effect_scaled = f64::from(effect[index]).max(0.0);
                let effect_unit = effect_scaled / (1.0 + effect_scaled);
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
        let mut scores = self.score_candidates(&observation.frame, &candidates)?;
        let hash = observation_hash(observation);
        let tried = self.tried.entry(hash).or_default();
        let all_tried = scores
            .iter()
            .all(|score| tried.contains(&action_key(&score.action)));
        if all_tried {
            tried.clear();
        }
        for score in &mut scores {
            if tried.contains(&action_key(&score.action)) {
                score.score -= 1.0;
            }
        }
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
        Ok(ActionDecision {
            chosen,
            candidate_count: candidates.len(),
        })
    }
}

pub fn observation_hash(observation: &ArcObservation) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    observation.game_id.hash(&mut hasher);
    let frame = &observation.frame;
    frame.width.hash(&mut hasher);
    frame.height.hash(&mut hasher);
    let visible_rows = usize::from(frame.height).min(FRAME_SIDE - 1);
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
        for (x, y) in action6_coordinates(&observation.frame, action6_max_candidates, grid_stride) {
            candidates.push(ArcAction::new(6, Some(x), Some(y))?);
        }
    }
    ensure!(!candidates.is_empty(), "no valid actions available");
    Ok(candidates)
}

fn action6_coordinates(frame: &ArcFrame, cap: usize, stride: usize) -> Vec<(u8, u8)> {
    let mut counts = [0usize; 16];
    for &pixel in &frame.pixels {
        counts[pixel as usize] += 1;
    }
    let background = counts
        .iter()
        .enumerate()
        .max_by_key(|&(color, count)| (*count, std::cmp::Reverse(color)))
        .map(|(color, _)| color as u8)
        .unwrap_or(0);
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
            if pixel != color {
                continue;
            }
            let x = index % FRAME_SIDE;
            let y = index / FRAME_SIDE;
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

    for y in (stride / 2..FRAME_SIDE).step_by(stride) {
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
pub struct AmbiguousAttemptedAction {
    pub index: usize,
    pub available_actions: Vec<u8>,
    pub decision: ActionDecision,
    pub levels_before: u16,
    pub api_latency_ms: u128,
    pub mutation: AmbiguousMutation,
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
    pub ambiguous_attempted_action: Option<AmbiguousAttemptedAction>,
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
    pub policy: String,
    pub policy_limitation: String,
    pub goal_feature_contract: String,
    pub tried_action_key_contract: String,
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
    pub max_actions_per_game: usize,
}

pub fn run_public_suite<A: ArcApi, P: LivePolicy>(
    api: &mut A,
    policy: &mut P,
    settings: &LiveRunSettings,
) -> Result<LiveEvalReport> {
    let mut discovered = api.list_games().context("discover public ARC games")?;
    discovered.sort_by(|a, b| a.game_id.cmp(&b.game_id));
    ensure!(!discovered.is_empty(), "ARC API returned no public games");
    let selected = select_games(&discovered, &settings.requested_games)?;
    let evaluating_all = settings.requested_games.is_empty() && selected.len() == discovered.len();
    let metadata = json!({
        "tags": ["tofy", "p2", "held-out", "live-eval"],
        "opaque": {
            "schema": LIVE_REPORT_SCHEMA,
            "checkpoint_sha256": settings.checkpoint_sha256,
            "train_config_sha256": settings.train_config_sha256,
            "policy": LIVE_POLICY,
            "public_data_used_for_fitting": false,
        }
    });
    let card_id = api.open_scorecard(&metadata)?;
    let mut game_reports = Vec::with_capacity(selected.len());

    for game in &selected {
        let started = Instant::now();
        let mut trace = Vec::new();
        let mut ambiguous_attempted_action = None;
        let mut error = None;
        let mut stop_reason = "reset_failed".to_string();
        let mut agent_session = AgentSession::default();
        let mut last = match api.reset(&game.game_id, &card_id) {
            Ok(observation) => {
                agent_session.observe(&observation)?;
                Some(observation)
            }
            Err(err) => {
                error = Some(format!("{err:#}"));
                None
            }
        };

        while let Some(observation) = last.as_ref() {
            if observation.terminal() {
                stop_reason = if observation.state == "WIN"
                    || (observation.win_levels > 0
                        && observation.levels_completed >= observation.win_levels)
                {
                    "completed".into()
                } else {
                    format!("terminal_{}", observation.state.to_ascii_lowercase())
                };
                break;
            }
            if trace.len() >= settings.max_actions_per_game {
                stop_reason = "max_actions_reached".into();
                break;
            }
            let decision = match policy.choose_action(observation) {
                Ok(decision) => decision,
                Err(err) => {
                    error = Some(format!("policy: {err:#}"));
                    stop_reason = "policy_error".into();
                    break;
                }
            };
            let reasoning = json!({
                "policy": LIVE_POLICY,
                "score": decision.chosen.score,
                "q_probability": decision.chosen.q_probability,
                "reliability_probability": decision.chosen.reliability_probability,
                "noop_probability": decision.chosen.noop_probability,
                "predicted_effect": decision.chosen.predicted_effect,
                "candidate_count": decision.candidate_count,
            });
            let call_started = Instant::now();
            let next = match api.act(
                &game.game_id,
                &observation.guid,
                &decision.chosen.action,
                &reasoning,
            ) {
                Ok(next) => next,
                Err(MutationError::Ambiguous(mutation)) => {
                    agent_session.record_ambiguous(
                        observation,
                        decision.chosen.action.clone(),
                        mutation.clone(),
                    )?;
                    error = Some(format!("action {}: {mutation}", trace.len() + 1));
                    stop_reason = "ambiguous_mutation".into();
                    ambiguous_attempted_action = Some(AmbiguousAttemptedAction {
                        index: trace.len(),
                        available_actions: observation.available_actions.clone(),
                        decision,
                        levels_before: observation.levels_completed,
                        api_latency_ms: call_started.elapsed().as_millis(),
                        mutation,
                    });
                    break;
                }
                Err(err) => {
                    error = Some(format!("action {}: {err}", trace.len() + 1));
                    stop_reason = "api_error".into();
                    break;
                }
            };
            agent_session.record_confirmed(observation, decision.chosen.action.clone(), &next)?;
            trace.push(LiveActionTrace {
                index: trace.len(),
                available_actions: observation.available_actions.clone(),
                decision,
                levels_before: observation.levels_completed,
                levels_after: next.levels_completed,
                state_after: next.state.clone(),
                frame_changed: observation.frame != next.frame,
                api_latency_ms: call_started.elapsed().as_millis(),
            });
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
            ambiguous_attempted_action,
            agent_session,
        });
    }

    let (scorecard, scorecard_close_error) = match api.close_scorecard(&card_id) {
        Ok(card) => (Some(card), None),
        Err(err) => (None, Some(format!("{err:#}"))),
    };
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

    Ok(LiveEvalReport {
        schema: LIVE_REPORT_SCHEMA.into(),
        created_at_unix_ms: unix_ms(),
        checkpoint: settings.checkpoint.clone(),
        checkpoint_sha256: settings.checkpoint_sha256.clone(),
        train_config: settings.train_config.clone(),
        train_config_sha256: settings.train_config_sha256.clone(),
        device: settings.device.clone(),
        base_url: settings.base_url.clone(),
        policy: LIVE_POLICY.into(),
        policy_limitation: POLICY_LIMITATION.into(),
        goal_feature_contract: GOAL_FEATURE_CONTRACT.into(),
        tried_action_key_contract: TRIED_ACTION_KEY_CONTRACT.into(),
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
    let train_config = load_train_config(&config.train_config)?;
    let _gpu_guard = if config.device == "cuda" || config.device.starts_with("cuda:") {
        Some(GpuSessionGuard::acquire(&train_config.output_dir)?)
    } else {
        None
    };
    let checkpoint_sha256 = sha256_file(&config.checkpoint)?;
    let train_config_sha256 = sha256_file(&config.train_config)?;
    let device = resolve_device(&config.device)?;
    let (model, _varmap) = load_model(&train_config, &config.checkpoint, &device)?;
    let mut policy = ModelPolicy::new(
        &model,
        &device,
        config.physical_batch,
        config.action6_max_candidates,
        config.action6_grid_stride,
    );
    let mut api = HttpArcApi::from_env(
        &config.base_url,
        &config.api_key_env,
        Duration::from_secs(config.request_timeout_secs),
    )?;
    let settings = LiveRunSettings {
        checkpoint: config.checkpoint.clone(),
        checkpoint_sha256,
        train_config: config.train_config.clone(),
        train_config_sha256,
        device: config.device.clone(),
        base_url: config.base_url.clone(),
        requested_games: config.games.clone(),
        max_actions_per_game: config.max_actions_per_game,
    };
    let report = run_public_suite(&mut api, &mut policy, &settings)?;
    write_json_atomic(&config.output, &report)?;
    Ok(report)
}

fn sha256_file(path: &Path) -> Result<String> {
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

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
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
    use std::collections::VecDeque;
    use std::sync::Mutex;

    fn frame(fill: u8) -> ArcFrame {
        ArcFrame::new(64, 64, vec![fill; 64 * 64]).unwrap()
    }

    fn observation(game_id: &str, state: &str, actions: Vec<u8>) -> ArcObservation {
        ArcObservation {
            game_id: game_id.into(),
            guid: format!("guid-{game_id}"),
            frame: frame(0),
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
    fn settled_frame_uses_last_animation_layer() {
        let api = ApiObservation {
            game_id: "demo".into(),
            guid: "guid".into(),
            frame: vec![vec![vec![1, 1]], vec![vec![7, 7]]],
            state: "NOT_FINISHED".into(),
            levels_completed: 0,
            win_levels: 1,
            available_actions: vec![1],
        };
        let parsed = ArcObservation::try_from(api).unwrap();
        assert_eq!(parsed.frame.pixel(0, 0), Some(7));
        assert_eq!(parsed.frame.pixel(1, 0), Some(7));
    }

    #[test]
    fn settled_frame_rejects_oversize_before_dimension_casts() {
        let api = ApiObservation {
            game_id: "demo".into(),
            guid: "guid".into(),
            frame: vec![vec![vec![0; 65]]],
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
    fn tried_action_hash_excludes_row_63_but_keeps_visible_gameplay() {
        let first = observation("game", "NOT_FINISHED", vec![1]);
        let mut status_changed = first.clone();
        status_changed.frame.pixels[63 * FRAME_SIDE + 7] = 9;
        assert_eq!(observation_hash(&first), observation_hash(&status_changed));

        let mut gameplay_changed = first.clone();
        gameplay_changed.frame.pixels[62 * FRAME_SIDE + 7] = 9;
        assert_ne!(
            observation_hash(&first),
            observation_hash(&gameplay_changed)
        );
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
        let mut policy = ModelPolicy::new(&model, &device, 4, 4, 32);
        let decision = policy.choose_action(&observation("demo", "NOT_FINISHED", vec![1, 2, 6]))?;
        assert!([1, 2, 6].contains(&decision.chosen.action.id));
        assert!(decision.chosen.score.is_finite());
        assert_eq!(decision.candidate_count, 6);
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
            })
        }
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

        fn reset(&mut self, game_id: &str, _card_id: &str) -> Result<ArcObservation> {
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
        let settings = LiveRunSettings {
            checkpoint: "model.safetensors".into(),
            checkpoint_sha256: "model-hash".into(),
            train_config: "config.json".into(),
            train_config_sha256: "config-hash".into(),
            device: "cpu".into(),
            base_url: "https://example.invalid".into(),
            requested_games: Vec::new(),
            max_actions_per_game: 4,
        };
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
            .contains("row 63 is excluded"));
        assert_eq!(report.official_rhae, Some(100.0));
        assert!(report.official_scorecard_parse_error.is_none());
        assert!(report
            .games
            .iter()
            .all(|game| game.stop_reason == "completed"));
    }

    #[test]
    fn ambiguous_attempt_is_reported_outside_confirmed_trace_and_scorecard_closes() {
        let mut api = FakeApi {
            closed: false,
            acted_games: Vec::new(),
            ambiguous_action: true,
        };
        let settings = LiveRunSettings {
            checkpoint: "model.safetensors".into(),
            checkpoint_sha256: "model-hash".into(),
            train_config: "config.json".into(),
            train_config_sha256: "config-hash".into(),
            device: "cpu".into(),
            base_url: "https://example.invalid".into(),
            requested_games: vec!["a-1".into()],
            max_actions_per_game: 4,
        };

        let report = run_public_suite(&mut api, &mut FirstPolicy, &settings).unwrap();
        let game = &report.games[0];
        assert!(api.closed);
        assert_eq!(game.stop_reason, "ambiguous_mutation");
        assert_eq!(game.actions, 0);
        assert!(game.trace.is_empty());
        assert_eq!(
            game.ambiguous_attempted_action
                .as_ref()
                .and_then(|attempt| attempt.mutation.action.as_ref())
                .map(|action| action.id),
            Some(1)
        );
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
        let settings = LiveRunSettings {
            checkpoint: "model.safetensors".into(),
            checkpoint_sha256: "model-hash".into(),
            train_config: "config.json".into(),
            train_config_sha256: "config-hash".into(),
            device: "cpu".into(),
            base_url: "https://example.invalid".into(),
            requested_games: vec!["game".into()],
            max_actions_per_game: 4,
        };

        let report = run_public_suite(&mut api, &mut FirstPolicy, &settings).unwrap();
        let game = &report.games[0];
        assert_eq!(game.stop_reason, "ambiguous_mutation");
        assert_eq!(game.actions, 0);
        assert!(game.trace.is_empty());
        assert_eq!(
            game.ambiguous_attempted_action.as_ref().map(|_| ()),
            Some(())
        );
        assert_eq!(api.transport.send_count(), 5);
    }

    #[test]
    fn training_source_cannot_depend_on_live_or_recording_modules() {
        let training = include_str!("train.rs");
        for forbidden in [
            "crate::p2::arc3",
            "crate::p2::arc3_live",
            "ARC_API_KEY",
            "reqwest::",
        ] {
            assert!(
                !training.contains(forbidden),
                "training source contains forbidden held-out dependency {forbidden}"
            );
        }
    }
}
