//! Orchestrator: decides which action the agent should take next (text, code, write file, run CLI, done).
//!
//! The orchestrator is called **per step** during a single reply, not once per user message.
//! Only one action is active at a time (easy on memory): we execute one action, append its result
//! to the assistant content, then decide the next action.
//!
//! Flow: Encoder → planner memory → orchestrator/router → decoder-specific adapter → decoder or tool.

/// Maximum number of actions per reply (text → code → write file → run CLI → done, etc.).
pub const MAX_ACTIONS_PER_REPLY: usize = 10;

/// Actions the agent can take. Decoder prompt uses the string (e.g. "Action=code").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Plain text reply (chat).
    TextReply,
    /// Code generation (e.g. code block in message).
    Code,
    /// Write generated code (or content) to a file. Tool: one action at a time.
    WriteFile,
    /// Run a CLI command (e.g. test). Tool: one action at a time.
    RunCli,
    /// No further action; finish the reply.
    Done,
}

impl Action {
    pub fn as_str(self) -> &'static str {
        match self {
            Action::TextReply => "text_reply",
            Action::Code => "code",
            Action::WriteFile => "write_file",
            Action::RunCli => "run_cli",
            Action::Done => "done",
        }
    }

    /// True if this action uses the decoder (text or code); false for tools or done.
    pub fn is_decoder(self) -> bool {
        matches!(self, Action::TextReply | Action::Code)
    }
}

/// Maps orchestrator head output index (0..5) to Action. Indices: 0=TextReply, 1=Code, 2=WriteFile, 3=RunCli, 4=Done.
#[inline]
pub fn action_from_index(idx: usize) -> Action {
    match idx {
        0 => Action::TextReply,
        1 => Action::Code,
        2 => Action::WriteFile,
        3 => Action::RunCli,
        _ => Action::Done,
    }
}

/// Returns the next action for this step (fallback when orchestrator head is not loaded). Fixed sequence.
#[inline]
pub fn decide_next_action(step: usize, _prompt: &str, _assistant_so_far: &str) -> Action {
    match step {
        0 => Action::TextReply,
        1 => Action::Code,
        2 => Action::WriteFile,
        3 => Action::RunCli,
        _ => Action::Done,
    }
}
