//! Fixed Phase A planning limits.

use serde::{Deserialize, Serialize};
use std::fmt;

pub const MAX_MODEL_EVALS: usize = 64;
pub const MAX_HORIZON: u8 = 2;
pub const MAX_CANDIDATES: usize = 32;
pub const TARGET_DEADLINE_MILLIS: u64 = 1_000;
pub const HARD_DEADLINE_MILLIS: u64 = 2_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseADeadline {
    pub target_millis: u64,
    pub hard_millis: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhaseAConfig {
    pub max_model_evals: usize,
    pub max_horizon: u8,
    pub deadline: PhaseADeadline,
    pub max_candidates: usize,
    pub unknown_prior: f32,
    pub epsilon: f32,
    pub alpha_safe: f32,
    pub protected_mass: f32,
    pub ordinary_trust_bound: f32,
    pub irreversible_trust_bound: f32,
    pub false_safe_trust_bound: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhaseAConfigError {
    ModelEvaluationCap { requested: usize, maximum: usize },
    HorizonCap { requested: u8, maximum: u8 },
    CandidateCap { requested: usize, maximum: usize },
    InvalidDeadline,
    HardDeadlineCap { requested: u64, maximum: u64 },
    InvalidProbability { field: &'static str, value: f32 },
}

impl fmt::Display for PhaseAConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelEvaluationCap { requested, maximum } => {
                write!(f, "model evaluation cap {requested} exceeds {maximum}")
            }
            Self::HorizonCap { requested, maximum } => {
                write!(f, "horizon cap {requested} exceeds {maximum}")
            }
            Self::CandidateCap { requested, maximum } => {
                write!(f, "candidate cap {requested} exceeds {maximum}")
            }
            Self::InvalidDeadline => write!(f, "deadline must be positive and target <= hard"),
            Self::HardDeadlineCap { requested, maximum } => {
                write!(f, "hard deadline {requested}ms exceeds {maximum}ms")
            }
            Self::InvalidProbability { field, value } => {
                write!(f, "{field} must be a probability in (0, 1); got {value}")
            }
        }
    }
}

impl std::error::Error for PhaseAConfigError {}

impl Default for PhaseAConfig {
    fn default() -> Self {
        Self {
            max_model_evals: MAX_MODEL_EVALS,
            max_horizon: MAX_HORIZON,
            deadline: PhaseADeadline {
                target_millis: TARGET_DEADLINE_MILLIS,
                hard_millis: HARD_DEADLINE_MILLIS,
            },
            max_candidates: MAX_CANDIDATES,
            unknown_prior: 0.20,
            epsilon: 0.02,
            alpha_safe: 0.02,
            protected_mass: 0.95,
            ordinary_trust_bound: 0.10,
            irreversible_trust_bound: 0.02,
            false_safe_trust_bound: 0.01,
        }
    }
}

impl PhaseAConfig {
    pub fn new(max_model_evals: usize, max_horizon: u8) -> Result<Self, PhaseAConfigError> {
        let config = Self {
            max_model_evals,
            max_horizon,
            ..Self::default()
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), PhaseAConfigError> {
        if self.max_model_evals > MAX_MODEL_EVALS {
            return Err(PhaseAConfigError::ModelEvaluationCap {
                requested: self.max_model_evals,
                maximum: MAX_MODEL_EVALS,
            });
        }
        if self.max_horizon > MAX_HORIZON {
            return Err(PhaseAConfigError::HorizonCap {
                requested: self.max_horizon,
                maximum: MAX_HORIZON,
            });
        }
        if self.max_candidates > MAX_CANDIDATES {
            return Err(PhaseAConfigError::CandidateCap {
                requested: self.max_candidates,
                maximum: MAX_CANDIDATES,
            });
        }
        if self.deadline.target_millis == 0
            || self.deadline.hard_millis == 0
            || self.deadline.target_millis > self.deadline.hard_millis
        {
            return Err(PhaseAConfigError::InvalidDeadline);
        }
        if self.deadline.hard_millis > HARD_DEADLINE_MILLIS {
            return Err(PhaseAConfigError::HardDeadlineCap {
                requested: self.deadline.hard_millis,
                maximum: HARD_DEADLINE_MILLIS,
            });
        }
        for (field, value) in [
            ("unknown_prior", self.unknown_prior),
            ("epsilon", self.epsilon),
            ("alpha_safe", self.alpha_safe),
            ("protected_mass", self.protected_mass),
            ("ordinary_trust_bound", self.ordinary_trust_bound),
            ("irreversible_trust_bound", self.irreversible_trust_bound),
            ("false_safe_trust_bound", self.false_safe_trust_bound),
        ] {
            if !(value.is_finite() && value > 0.0 && value < 1.0) {
                return Err(PhaseAConfigError::InvalidProbability { field, value });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_phase_a_caps() {
        assert!(matches!(
            PhaseAConfig::new(MAX_MODEL_EVALS + 1, MAX_HORIZON),
            Err(PhaseAConfigError::ModelEvaluationCap { .. })
        ));
        assert!(matches!(
            PhaseAConfig::new(MAX_MODEL_EVALS, MAX_HORIZON + 1),
            Err(PhaseAConfigError::HorizonCap { .. })
        ));
        let too_slow = PhaseAConfig {
            deadline: PhaseADeadline {
                target_millis: TARGET_DEADLINE_MILLIS,
                hard_millis: HARD_DEADLINE_MILLIS + 1,
            },
            ..PhaseAConfig::default()
        };
        assert!(matches!(
            too_slow.validate(),
            Err(PhaseAConfigError::HardDeadlineCap { .. })
        ));
    }

    #[test]
    fn rejects_out_of_range_probability_fields() {
        let bad = PhaseAConfig {
            unknown_prior: 2.0,
            ..PhaseAConfig::default()
        };
        assert!(matches!(
            bad.validate(),
            Err(PhaseAConfigError::InvalidProbability {
                field: "unknown_prior",
                ..
            })
        ));
    }

    #[test]
    fn serde_round_trip_preserves_valid_config() {
        let config = PhaseAConfig::default();
        let decoded: PhaseAConfig =
            serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
        assert_eq!(decoded, config);
        decoded.validate().unwrap();
    }
}
