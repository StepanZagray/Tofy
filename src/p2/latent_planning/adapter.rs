//! Model seam for the Phase A controller (ADR 0004).
//!
//! The planning kernels in this module tree are tensor-free by design. This
//! trait is the only surface through which the controller touches the learned
//! world model; the tensor-side implementation lives in the live-eval policy
//! module and charges every transition evaluation against the per-decision budget
//! (`PhaseAConfig::max_model_evals`). Event-head fan-out across goal vectors is
//! reported separately and is not charged: A5 counts transition evaluations.

use super::graph::ActionKey;
use std::fmt;

/// Raw (uncalibrated) event-head readout for one imagined edge under one goal
/// vector. Calibration into evidence happens in `trust.rs`; the adapter never
/// interprets these numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoalEventReadout {
    pub ordinary: f32,
    pub satisfied: f32,
    pub failed: f32,
    pub exhausted: f32,
}

/// One imagined transition. `per_goal_events` has one entry per goal vector
/// passed to `step_batch`, in the same order.
#[derive(Debug, Clone)]
pub struct StepPrediction<L> {
    pub action: ActionKey,
    pub latent: L,
    pub q_raw: f32,
    pub reliability_raw: f32,
    pub noop_raw: f32,
    pub per_goal_events: Vec<GoalEventReadout>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelCallError {
    /// A whole batch was refused because it would exceed the decision budget.
    /// Partial batches are never executed.
    BudgetExhausted {
        requested: usize,
        used: usize,
        cap: usize,
    },
    Backend(String),
}

impl fmt::Display for ModelCallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BudgetExhausted {
                requested,
                used,
                cap,
            } => write!(
                f,
                "model evaluation budget exhausted: {requested} requested with {used}/{cap} used"
            ),
            Self::Backend(message) => write!(f, "model backend error: {message}"),
        }
    }
}

impl std::error::Error for ModelCallError {}

/// Budget-charged, opaque-latent access to the frozen world model.
pub trait PhaseAModel {
    type Latent: Clone;

    /// Encode a 64x64 palette-index frame into the consumer latent. 1 eval.
    fn encode(&mut self, frame_pixels: &[u8]) -> Result<Self::Latent, ModelCallError>;

    /// Roll each action once from `from` with the goal-dropout (all-zero)
    /// conditioning, then read the event head once per goal vector.
    /// Charges `actions.len()` evals; refuses the whole batch when it would
    /// exceed the cap.
    fn step_batch(
        &mut self,
        from: &Self::Latent,
        actions: &[ActionKey],
        goal_vectors: &[[f32; 19]],
    ) -> Result<Vec<StepPrediction<Self::Latent>>, ModelCallError>;

    /// Composed gameplay decode of a latent to 64x64 palette indices. 1 eval.
    fn decode(&mut self, latent: &Self::Latent) -> Result<Vec<u8>, ModelCallError>;

    /// Transition evaluations charged so far in this decision.
    fn evals_used(&self) -> usize;

    /// Event-head reads performed so far in this decision (reported, uncharged).
    fn event_head_reads(&self) -> usize;

    fn reset_decision_budget(&mut self, cap: usize);

    /// ADR 0005 §6.1 Channel A: the factual transitions observed before the
    /// decision about to be scored; every model call of that decision is
    /// conditioned on it. Adapters without a context channel ignore it.
    fn set_context_window(&mut self, _window: Vec<crate::p2::data::ContextTransition>) {}
}

/// Shared budget accounting for adapter implementations.
#[derive(Debug, Clone, Default)]
pub struct EvalBudget {
    pub cap: usize,
    pub used: usize,
    pub event_head_reads: usize,
}

impl EvalBudget {
    pub fn reset(&mut self, cap: usize) {
        self.cap = cap;
        self.used = 0;
        self.event_head_reads = 0;
    }

    /// Charge a whole batch or refuse it entirely.
    pub fn charge(&mut self, requested: usize) -> Result<(), ModelCallError> {
        if self.used + requested > self.cap {
            return Err(ModelCallError::BudgetExhausted {
                requested,
                used: self.used,
                cap: self.cap,
            });
        }
        self.used += requested;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_refuses_whole_batch_never_partial() {
        let mut budget = EvalBudget::default();
        budget.reset(10);
        assert!(budget.charge(6).is_ok());
        let refused = budget.charge(5).unwrap_err();
        assert!(matches!(
            refused,
            ModelCallError::BudgetExhausted {
                requested: 5,
                used: 6,
                cap: 10
            }
        ));
        assert_eq!(
            budget.used, 6,
            "a refused batch must not be partially charged"
        );
        assert!(budget.charge(4).is_ok());
        assert_eq!(budget.used, 10);
    }
}
