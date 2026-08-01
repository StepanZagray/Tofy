//! P2 learned recursive world model and ARC-AGI-3-compatible data boundary.
//!
//! Dynamics never receive the hidden objective. Candidate-goal features are
//! consumed only by auxiliary predicate heads so the same predicted transition
//! can be evaluated under several public hypotheses.

pub mod arc3;
pub mod cli;
pub mod data;
pub mod eval;
pub mod model;
pub mod sigreg;
pub mod train;
