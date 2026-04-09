//! Orchestrator: chooses the next high-level action for the current reply.
//!
//! Tofy only exposes actions that the runtime can actually execute:
//! `text_reply`, `code`, and `done`.

/// Actions the agent can take. Decoder prompt uses the string (e.g. "Action=code").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Plain text reply (chat).
    TextReply,
    /// Code generation (e.g. code block in message).
    Code,
    /// Terminal action: no further reply content should be produced.
    Done,
}

impl Action {
    pub fn as_str(self) -> &'static str {
        match self {
            Action::TextReply => "text_reply",
            Action::Code => "code",
            Action::Done => "done",
        }
    }
}

/// Maps orchestrator head output index to Action. Indices: 0=TextReply, 1=Code, 2=Done.
#[inline]
pub fn action_from_index(idx: usize) -> Action {
    match idx {
        0 => Action::TextReply,
        1 => Action::Code,
        _ => Action::Done,
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

/// Fallback when no trained orchestrator head is loaded.
#[inline]
pub fn decide_next_action(prompt: &str, assistant_so_far: &str) -> Action {
    if prompt.trim().is_empty() {
        return Action::Done;
    }
    if !assistant_so_far.trim().is_empty() {
        return Action::TextReply;
    }
    if looks_like_code_request(prompt) {
        Action::Code
    } else {
        Action::TextReply
    }
}

#[cfg(test)]
mod tests {
    use super::{action_from_index, decide_next_action, Action};

    #[test]
    fn action_indices_cover_supported_actions() {
        assert_eq!(action_from_index(0), Action::TextReply);
        assert_eq!(action_from_index(1), Action::Code);
        assert_eq!(action_from_index(2), Action::Done);
        assert_eq!(action_from_index(99), Action::Done);
    }

    #[test]
    fn fallback_prefers_code_for_code_requests() {
        assert_eq!(
            decide_next_action("Please implement this Rust function", ""),
            Action::Code
        );
        assert_eq!(
            decide_next_action("Explain what this code does", "already started"),
            Action::TextReply
        );
        assert_eq!(decide_next_action("Hello there", ""), Action::TextReply);
    }
}
