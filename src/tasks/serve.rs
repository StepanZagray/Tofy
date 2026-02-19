//! OpenAI-compatible HTTP server for Tofy (world model + decoder).
//! Use with OpenCode or any client that speaks OpenAI API.

/// Model ID returned by the API. Use this in OpenCode / clients.
pub const TOFY_MODEL_ID: &str = "tofy";

/// Short description for model list and responses.
const TOFY_MODEL_DESCRIPTION: &str =
    "Tofy: JEPA-style world-model agent (encoder → transition → bridge → decoder). Local inference with optional conditioning.";

use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{sse::Event, IntoResponse, Sse},
    routing::{get, post},
    Json, Router,
};
use futures_util::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::ReceiverStream;

use super::world::AgentEngine;

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: Option<String>,
}

#[derive(Deserialize)]
#[serde(default)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    stream: Option<bool>,
}

impl Default for ChatCompletionRequest {
    fn default() -> Self {
        Self {
            model: None,
            messages: Vec::new(),
            max_tokens: Some(4096),
            temperature: None,
            stream: None,
        }
    }
}

#[derive(Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatMessageResponse,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatMessageResponse {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
}

#[derive(Serialize)]
struct ModelsListResponse {
    object: String,
    data: Vec<ModelInfo>,
}

fn make_id() -> String {
    format!(
        "chatcmpl-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    )
}

async fn chat_completions(
    State(engine): State<std::sync::Arc<Mutex<AgentEngine>>>,
    Json(body): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let prompt = body
        .messages
        .iter()
        .rev()
        .find(|m| m.role.eq_ignore_ascii_case("user"))
        .and_then(|m| m.content.as_deref())
        .unwrap_or("")
        .trim()
        .to_string();

    if prompt.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "messages must contain at least one user message with content".to_string(),
        ));
    }

    let max_tokens = body.max_tokens.unwrap_or(4096) as usize;
    if std::env::var("JEPA_DEBUG").is_ok() {
        let _ = writeln!(std::io::stderr(), "[tofy] request started (stream={})", body.stream == Some(true));
        let _ = std::io::stderr().flush();
    }
    let id = make_id();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if body.stream == Some(true) {
        // Option<String>: Some(content) = chunk, None = generation done (send finish immediately).
        let (tx, rx) = tokio::sync::mpsc::channel::<Option<String>>(32);
        let engine_clone = engine.clone();
        let prompt_clone = prompt.clone();
        tokio::task::spawn_blocking(move || {
            let guard = match engine_clone.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            let mut send = |chunk: &str| {
                let _ = tx.blocking_send(Some(chunk.to_string()));
            };
            let _ = guard.generate_stream(&prompt_clone, max_tokens, false, &mut send);
            let _ = tx.blocking_send(None); // explicit done so client gets finish_reason
        });
        let id_for_finish = id.clone();
        let role_event = stream::iter([Ok::<_, std::convert::Infallible>(
            Event::default().data(
                serde_json::json!({
                    "id": id_for_finish.clone(),
                    "choices": [{ "index": 0, "delta": { "role": "assistant" }, "finish_reason": null }],
                    "model": TOFY_MODEL_ID
                })
                .to_string(),
            ),
        )]);
        let finish_json = serde_json::json!({
            "id": id_for_finish,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
            "model": TOFY_MODEL_ID
        });
        let chunk_stream = ReceiverStream::new(rx).flat_map(move |opt| {
            let events: Vec<_> = match opt {
                Some(chunk) => vec![Ok::<_, std::convert::Infallible>(
                    Event::default().data(
                        serde_json::json!({
                            "id": id.clone(),
                            "choices": [{ "index": 0, "delta": { "content": chunk }, "finish_reason": null }],
                            "model": TOFY_MODEL_ID
                        })
                        .to_string(),
                    ),
                )],
                None => vec![
                    Ok(Event::default().data(finish_json.to_string())),
                    Ok(Event::default().data("[DONE]")),
                ],
            };
            stream::iter(events)
        });
        let sse_stream = role_event.chain(chunk_stream);
        let mut res: axum::response::Response = Sse::new(sse_stream).into_response();
        res.headers_mut()
            .insert(header::CONNECTION, header::HeaderValue::from_static("close"));
        return Ok(res);
    }

    let text = {
        let guard = engine
            .lock()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        guard
            .generate(&prompt, max_tokens, false)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    };

    let response = ChatCompletionResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model: TOFY_MODEL_ID.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageResponse {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason: "stop".to_string(),
        }],
    };

    let res: axum::response::Response = Json(response).into_response();
    Ok(res)
}

async fn models_list() -> Json<ModelsListResponse> {
    Json(ModelsListResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: TOFY_MODEL_ID.to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "Tofy".to_string(),
            description: Some(TOFY_MODEL_DESCRIPTION.to_string()),
        }],
    })
}

async fn health() -> &'static str {
    "ok"
}

pub async fn run(
    bind: &str,
    model_path: PathBuf,
    vocab_path: PathBuf,
    dim: usize,
    max_seq: usize,
    num_layers: usize,
    num_heads: usize,
    bridge_dim: usize,
    debug: bool,
) -> Result<()> {
    if debug {
        eprintln!("Tofy serve: debug mode on — t/s and decoder dumps go to this console (stderr)");
    }
    let engine = AgentEngine::load(
        &model_path,
        &vocab_path,
        dim,
        max_seq,
        num_layers,
        num_heads,
        bridge_dim,
    )
    .context("load world model for server")?;

    let state = std::sync::Arc::new(Mutex::new(engine));

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models_list))
        .route("/health", get(health))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .context("bind server")?;

    eprintln!("Tofy OpenAI-compatible server listening on http://{}", bind);
    eprintln!("  POST /v1/chat/completions  — chat completions");
    eprintln!("  GET  /v1/models            — list models (tofy)");
    eprintln!("  GET  /health               — health check");
    eprintln!("Use in OpenCode: Base URL = http://{}", bind);

    axum::serve(listener, app)
        .await
        .context("serve")?;

    Ok(())
}
