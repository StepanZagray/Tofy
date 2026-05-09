//! Orchestrator: chooses the next high-level action for the current reply.
//!
//! Tofy only exposes actions that the runtime can actually execute:
//! `text_reply`, `code`, `done`, and `fetch_docs`.

/// Actions the agent can take. Decoder prompt uses the string (e.g. "Action=code").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Plain text reply (chat).
    TextReply,
    /// Code generation (e.g. code block in message).
    Code,
    /// Terminal action: no further reply content should be produced.
    Done,
    /// Retrieve Rust documentation before generating code.
    FetchDocs,
}

impl Action {
    pub fn as_str(self) -> &'static str {
        match self {
            Action::TextReply => "text_reply",
            Action::Code => "code",
            Action::Done => "done",
            Action::FetchDocs => "fetch_docs",
        }
    }
}

/// Maps orchestrator head output index to Action.
/// Indices: 0=TextReply, 1=Code, 2=Done, 3=FetchDocs.
#[inline]
pub fn action_from_index(idx: usize) -> Action {
    match idx {
        0 => Action::TextReply,
        1 => Action::Code,
        2 => Action::Done,
        _ => Action::FetchDocs,
    }
}

fn looks_like_code_request(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    lower.contains("```")
        || lower.contains("write code")
        || lower.contains("implement")
        || lower.contains("function")
        || lower.contains("class")
        || lower.contains("rust")
        || lower.contains("python")
        || lower.contains("javascript")
        || lower.contains("typescript")
}

pub fn code_request_score(prompt: &str) -> f32 {
    let lower = prompt.to_ascii_lowercase();
    let mut score = 0.0f32;
    for needle in [
        "return only rust code",
        "implement exactly this function",
        "write the rust function",
        "complete this rust function",
        "implement",
        "write code",
        "code only",
        "rust",
        "function",
        "unit test",
        "compiler",
        "parse",
        "impl ",
        "pub fn",
        "fn ",
    ] {
        if lower.contains(needle) {
            score += if needle.len() > 10 { 0.9 } else { 0.45 };
        }
    }
    if prompt.contains("```") {
        score += 0.8;
    }
    score.min(3.0)
}

pub fn terminal_request_score(prompt: &str) -> f32 {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return 2.0;
    }
    let lower = trimmed.to_ascii_lowercase();
    let mut score = 0.0f32;
    for needle in [
        "thanks",
        "thank you",
        "that's all",
        "thats all",
        "no reply needed",
        "no response needed",
        "stop here",
        "done for now",
        "nothing else",
    ] {
        if lower.contains(needle) {
            score += 0.8;
        }
    }
    if lower.len() < 24
        && matches!(
            lower.as_str(),
            "ok" | "okay" | "thanks" | "thank you" | "done" | "stop"
        )
    {
        score += 1.2;
    }
    score.min(2.5)
}

pub fn rust_docs_request_score(prompt: &str) -> f32 {
    let lower = prompt.to_ascii_lowercase();
    if !lower.contains("rust") && !lower.contains("cargo") && !lower.contains("rustc") {
        return 0.0;
    }
    let mut score = code_request_score(prompt) * 0.35;
    for needle in [
        "std::",
        "core::",
        "alloc::",
        "iterator",
        "trait",
        "lifetime",
        "borrow",
        "ownership",
        "hashmap",
        "btree",
        "binaryheap",
        "vecdeque",
        "result<",
        "option<",
        "fromstr",
        "asref",
        "borrowchecker",
        "compiler error",
        "rustdoc",
        "docs",
        "documentation",
        "api",
    ] {
        if lower.contains(needle) {
            score += 0.7;
        }
    }
    if prompt.contains("::") {
        score += 0.8;
    }
    score.min(4.0)
}

pub fn guard_inference_action(prompt: &str, predicted: Action, logits: Option<&[f32]>) -> Action {
    let code_score = code_request_score(prompt);
    let terminal_score = terminal_request_score(prompt);
    let margin = logits
        .filter(|row| row.len() >= 3)
        .map(|row| {
            let mut sorted = [row[0], row[1], row[2]];
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            sorted[0] - sorted[1]
        })
        .unwrap_or(0.0);

    if predicted == Action::Done && terminal_score < 0.8 {
        if code_score >= 0.7 {
            return Action::Code;
        }
        return Action::TextReply;
    }

    if predicted == Action::FetchDocs && code_score < 0.4 && rust_docs_request_score(prompt) < 0.8 {
        return Action::TextReply;
    }

    if predicted == Action::TextReply && rust_docs_request_score(prompt) >= 1.4 {
        return Action::FetchDocs;
    }

    if predicted == Action::TextReply && code_score >= 1.0 {
        return Action::Code;
    }

    if predicted == Action::Done && code_score >= 0.4 && margin < 1.5 {
        return Action::Code;
    }

    if predicted == Action::Code && rust_docs_request_score(prompt) >= 2.0 && margin < 1.2 {
        return Action::FetchDocs;
    }

    if predicted == Action::Code && terminal_score >= 1.5 && margin < 0.8 {
        return Action::Done;
    }

    predicted
}

/// Fallback when no trained orchestrator head is loaded.
#[inline]
pub fn decide_next_action(prompt: &str, assistant_so_far: &str) -> Action {
    if prompt.trim().is_empty() {
        return Action::Done;
    }
    if !assistant_so_far.trim().is_empty() {
        return Action::TextReply;
    }
    if rust_docs_request_score(prompt) >= 1.4 {
        return Action::FetchDocs;
    }
    if looks_like_code_request(prompt) {
        Action::Code
    } else {
        Action::TextReply
    }
}
