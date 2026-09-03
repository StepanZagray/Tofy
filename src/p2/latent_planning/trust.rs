//! Fail-closed Phase A trust gates and the finalist selection charge.
//!
//! Every gate here answers "may the planner rely on this model output?" and
//! defaults to *no* whenever the calibration record is missing, uncalibrated,
//! thinly supported, or looser than the configured bound.

use super::config::PhaseAConfig;
use serde::Deserialize;
use std::fmt;

/// Minimum calibration support before a bin may certify anything.
pub const MIN_SUPPORT: u64 = 64;

/// One calibrated error bin: a 95% upper bound on the error rate plus the
/// number of samples that produced it.
#[derive(Clone, Debug, Deserialize)]
pub struct CalibrationBin {
    pub upper_error_bound_95: f64,
    pub support: u64,
}

/// Phase A calibration record produced offline.
///
/// `q_direction` says whether the raw head score increases (`1`) or decreases
/// (`-1`) with reliability; `tau_unknown` is the surprise threshold fed to the
/// belief update; `score_error_bound` bounds the finalist score error and
/// drives [`selection_charge`].
#[derive(Clone, Debug, Deserialize)]
pub struct PhaseACalibration {
    pub q_direction: i8,
    pub tau_unknown: f64,
    pub score_error_bound: f64,
    #[serde(default)]
    pub ordinary: Option<CalibrationBin>,
    #[serde(default)]
    pub event_false_safe: Option<CalibrationBin>,
    #[serde(default)]
    pub satisfaction: Option<CalibrationBin>,
    #[serde(default)]
    pub ptrm: Option<CalibrationBin>,
    #[serde(default)]
    pub uncalibrated: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrustError {
    InvalidJson(String),
    InvalidTauUnknown(f64),
    InvalidScoreErrorBound(f64),
    InvalidDirection(i8),
    InvalidBinBound { bin: &'static str, value: f64 },
    ZeroSupport { bin: &'static str },
}

impl fmt::Display for TrustError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(message) => write!(f, "calibration json is invalid: {message}"),
            Self::InvalidTauUnknown(value) => {
                write!(f, "tau_unknown must be in (0, 1); got {value}")
            }
            Self::InvalidScoreErrorBound(value) => {
                write!(f, "score_error_bound must be finite and >= 0; got {value}")
            }
            Self::InvalidDirection(value) => {
                write!(f, "q_direction must be -1 or 1; got {value}")
            }
            Self::InvalidBinBound { bin, value } => {
                write!(
                    f,
                    "{bin} upper_error_bound_95 must be finite and >= 0; got {value}"
                )
            }
            Self::ZeroSupport { bin } => write!(f, "{bin} calibration bin has zero support"),
        }
    }
}

impl std::error::Error for TrustError {}

/// Whether one model edge may be trusted, and with what update strength.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeTrust {
    Trusted { eta: f64 },
    Untrusted,
}

impl PhaseACalibration {
    /// A record that certifies nothing: every gate answers fail-closed.
    pub fn fail_closed() -> Self {
        Self {
            q_direction: 1,
            tau_unknown: 0.5,
            score_error_bound: 1.0,
            ordinary: None,
            event_false_safe: None,
            satisfaction: None,
            ptrm: None,
            uncalibrated: true,
        }
    }

    /// Parse and validate a calibration record.
    pub fn from_json(text: &str) -> Result<Self, TrustError> {
        let record: Self = serde_json::from_str(text)
            .map_err(|error| TrustError::InvalidJson(error.to_string()))?;
        record.validate()?;
        Ok(record)
    }

    fn validate(&self) -> Result<(), TrustError> {
        if !(self.tau_unknown.is_finite() && self.tau_unknown > 0.0 && self.tau_unknown < 1.0) {
            return Err(TrustError::InvalidTauUnknown(self.tau_unknown));
        }
        if !(self.score_error_bound.is_finite() && self.score_error_bound >= 0.0) {
            return Err(TrustError::InvalidScoreErrorBound(self.score_error_bound));
        }
        if self.q_direction != 1 && self.q_direction != -1 {
            return Err(TrustError::InvalidDirection(self.q_direction));
        }
        for (name, bin) in [
            ("ordinary", &self.ordinary),
            ("event_false_safe", &self.event_false_safe),
            ("satisfaction", &self.satisfaction),
            ("ptrm", &self.ptrm),
        ] {
            let Some(bin) = bin else {
                continue;
            };
            if !(bin.upper_error_bound_95.is_finite() && bin.upper_error_bound_95 >= 0.0) {
                return Err(TrustError::InvalidBinBound {
                    bin: name,
                    value: bin.upper_error_bound_95,
                });
            }
            if bin.support == 0 {
                return Err(TrustError::ZeroSupport { bin: name });
            }
        }
        Ok(())
    }

    /// A bin usable for certification: present, finite, and well supported.
    fn usable_bin(&self, bin: &Option<CalibrationBin>) -> Option<f64> {
        if self.uncalibrated {
            return None;
        }
        let bin = bin.as_ref()?;
        (bin.support >= MIN_SUPPORT && bin.upper_error_bound_95.is_finite())
            .then_some(bin.upper_error_bound_95)
    }

    /// Ordinary-edge gate.
    ///
    /// Untrusted when the record is uncalibrated, the ordinary bin is missing
    /// or under-supported, its bound exceeds `cfg.ordinary_trust_bound`, or the
    /// direction-adjusted head score or the reliability score is below 0.5.
    /// Otherwise trusted with `eta = 1 - bound`.
    pub fn edge_trust(&self, q_raw: f32, reliability_raw: f32, cfg: &PhaseAConfig) -> EdgeTrust {
        let Some(bound) = self.usable_bin(&self.ordinary) else {
            return EdgeTrust::Untrusted;
        };
        if bound > f64::from(cfg.ordinary_trust_bound) {
            return EdgeTrust::Untrusted;
        }
        let q = if self.q_direction < 0 {
            1.0 - f64::from(q_raw)
        } else {
            f64::from(q_raw)
        };
        let reliability = f64::from(reliability_raw);
        if !q.is_finite() || !reliability.is_finite() || q < 0.5 || reliability < 0.5 {
            return EdgeTrust::Untrusted;
        }
        EdgeTrust::Trusted {
            eta: (1.0 - bound).clamp(0.0, 1.0),
        }
    }

    /// Prefix false-safe gate: every edge's `P(fail) + P(exhausted)` plus the
    /// calibrated bound must sum to at most `cfg.alpha_safe`.
    pub fn prefix_false_safe_ok(
        &self,
        per_edge_fail_plus_exhausted: &[f64],
        cfg: &PhaseAConfig,
    ) -> bool {
        let Some(bound) = self.usable_bin(&self.event_false_safe) else {
            return false;
        };
        if per_edge_fail_plus_exhausted
            .iter()
            .any(|value| !value.is_finite())
        {
            return false;
        }
        let total = per_edge_fail_plus_exhausted
            .iter()
            .map(|value| value + bound)
            .sum::<f64>();
        total <= f64::from(cfg.alpha_safe)
    }

    /// Lower confidence bound on a raw satisfaction score, or `None` when the
    /// satisfaction bin cannot certify anything.
    pub fn satisfaction_lcb(&self, raw: f32) -> Option<f64> {
        let bound = self.usable_bin(&self.satisfaction)?;
        let raw = f64::from(raw);
        raw.is_finite().then(|| (raw - bound).clamp(0.0, 1.0))
    }
}

/// Claim mass a finalist forfeits to cover a two-sided score error.
pub fn selection_charge(score_error_bound: f64) -> f64 {
    2.0 * score_error_bound
}

/// Accept a finalist only if its charged claim mass still meets `min_claim`.
pub fn accept_finalist(claim_mass: f64, min_claim: f64, score_error_bound: f64) -> bool {
    let charged = claim_mass - selection_charge(score_error_bound);
    charged.is_finite() && charged >= min_claim
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin(bound: f64, support: u64) -> Option<CalibrationBin> {
        Some(CalibrationBin {
            upper_error_bound_95: bound,
            support,
        })
    }

    fn calibrated() -> PhaseACalibration {
        PhaseACalibration {
            q_direction: 1,
            tau_unknown: 0.5,
            score_error_bound: 0.01,
            ordinary: bin(0.05, 1_000),
            event_false_safe: bin(0.001, 1_000),
            satisfaction: bin(0.1, 1_000),
            ptrm: bin(0.05, 1_000),
            uncalibrated: false,
        }
    }

    #[test]
    fn fail_closed_record_trusts_nothing() {
        let cfg = PhaseAConfig::default();
        let record = PhaseACalibration::fail_closed();
        for (q, r) in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (f32::NAN, 1.0)] {
            assert_eq!(record.edge_trust(q, r, &cfg), EdgeTrust::Untrusted);
        }
        assert!(!record.prefix_false_safe_ok(&[], &cfg));
        assert_eq!(record.satisfaction_lcb(1.0), None);
        let mut with_bins = calibrated();
        with_bins.uncalibrated = true;
        assert_eq!(with_bins.edge_trust(0.9, 0.9, &cfg), EdgeTrust::Untrusted);
        assert!(!with_bins.prefix_false_safe_ok(&[0.0], &cfg));
        assert_eq!(with_bins.satisfaction_lcb(1.0), None);
    }

    #[test]
    fn calibrated_record_trusts_confident_edges() {
        let cfg = PhaseAConfig::default();
        let record = calibrated();
        match record.edge_trust(0.9, 0.9, &cfg) {
            EdgeTrust::Trusted { eta } => assert!((eta - 0.95).abs() < 1e-12),
            EdgeTrust::Untrusted => panic!("expected trusted edge"),
        }
        assert_eq!(record.edge_trust(0.4, 0.9, &cfg), EdgeTrust::Untrusted);
        assert_eq!(record.edge_trust(0.9, 0.4, &cfg), EdgeTrust::Untrusted);
        assert_eq!(record.edge_trust(f32::NAN, 0.9, &cfg), EdgeTrust::Untrusted);
    }

    #[test]
    fn missing_low_support_and_loose_bins_are_untrusted() {
        let cfg = PhaseAConfig::default();
        let mut missing = calibrated();
        missing.ordinary = None;
        assert_eq!(missing.edge_trust(0.9, 0.9, &cfg), EdgeTrust::Untrusted);

        let mut thin = calibrated();
        thin.ordinary = bin(0.05, MIN_SUPPORT - 1);
        assert_eq!(thin.edge_trust(0.9, 0.9, &cfg), EdgeTrust::Untrusted);
        thin.ordinary = bin(0.05, MIN_SUPPORT);
        assert!(matches!(
            thin.edge_trust(0.9, 0.9, &cfg),
            EdgeTrust::Trusted { .. }
        ));

        let mut loose = calibrated();
        loose.ordinary = bin(f64::from(cfg.ordinary_trust_bound) + 1e-6, 1_000);
        assert_eq!(loose.edge_trust(0.9, 0.9, &cfg), EdgeTrust::Untrusted);
    }

    #[test]
    fn q_direction_is_honored() {
        let cfg = PhaseAConfig::default();
        let mut record = calibrated();
        record.q_direction = -1;
        assert!(matches!(
            record.edge_trust(0.1, 0.9, &cfg),
            EdgeTrust::Trusted { .. }
        ));
        assert_eq!(record.edge_trust(0.9, 0.9, &cfg), EdgeTrust::Untrusted);
    }

    #[test]
    fn prefix_false_safe_sums_bound_per_edge() {
        let cfg = PhaseAConfig::default();
        let record = calibrated();
        // alpha_safe = 0.02, bound = 0.001 per edge.
        assert!(record.prefix_false_safe_ok(&[0.005, 0.005], &cfg));
        assert!(!record.prefix_false_safe_ok(&[0.01, 0.01], &cfg));
        assert!(!record.prefix_false_safe_ok(&[f64::NAN], &cfg));
        let mut thin = calibrated();
        thin.event_false_safe = bin(0.001, 1);
        assert!(!thin.prefix_false_safe_ok(&[0.0], &cfg));
        thin.event_false_safe = None;
        assert!(!thin.prefix_false_safe_ok(&[0.0], &cfg));
    }

    #[test]
    fn satisfaction_lcb_subtracts_bound_and_clamps() {
        let record = calibrated();
        assert!((record.satisfaction_lcb(0.8).unwrap() - 0.7).abs() < 1e-6);
        assert_eq!(record.satisfaction_lcb(0.05), Some(0.0));
        assert_eq!(record.satisfaction_lcb(2.0), Some(1.0));
        assert_eq!(record.satisfaction_lcb(f32::NAN), None);
        let mut missing = calibrated();
        missing.satisfaction = None;
        assert_eq!(missing.satisfaction_lcb(0.8), None);
    }

    #[test]
    fn from_json_validates_fields() {
        let good = r#"{"q_direction": -1, "tau_unknown": 0.3, "score_error_bound": 0.02,
            "ordinary": {"upper_error_bound_95": 0.05, "support": 500}}"#;
        let record = PhaseACalibration::from_json(good).unwrap();
        assert!(!record.uncalibrated);
        assert_eq!(record.q_direction, -1);
        assert!(record.event_false_safe.is_none());

        let bad_tau = r#"{"q_direction": 1, "tau_unknown": 1.0, "score_error_bound": 0.02}"#;
        assert!(matches!(
            PhaseACalibration::from_json(bad_tau),
            Err(TrustError::InvalidTauUnknown(_))
        ));
        let bad_direction = r#"{"q_direction": 0, "tau_unknown": 0.5, "score_error_bound": 0.02}"#;
        assert!(matches!(
            PhaseACalibration::from_json(bad_direction),
            Err(TrustError::InvalidDirection(0))
        ));
        let bad_bound = r#"{"q_direction": 1, "tau_unknown": 0.5, "score_error_bound": -0.1}"#;
        assert!(matches!(
            PhaseACalibration::from_json(bad_bound),
            Err(TrustError::InvalidScoreErrorBound(_))
        ));
        let zero_support = r#"{"q_direction": 1, "tau_unknown": 0.5, "score_error_bound": 0.0,
            "ptrm": {"upper_error_bound_95": 0.1, "support": 0}}"#;
        assert!(matches!(
            PhaseACalibration::from_json(zero_support),
            Err(TrustError::ZeroSupport { bin: "ptrm" })
        ));
        assert!(matches!(
            PhaseACalibration::from_json("not json"),
            Err(TrustError::InvalidJson(_))
        ));
    }

    #[test]
    fn selection_charge_is_twice_epsilon() {
        for epsilon in [0.0, 0.01, 0.25, 0.5] {
            assert_eq!(selection_charge(epsilon), 2.0 * epsilon);
        }
    }

    #[test]
    fn accept_finalist_is_monotone_in_epsilon() {
        let mut previous = true;
        for step in 0..=50 {
            let epsilon = step as f64 / 100.0;
            let accepted = accept_finalist(0.6, 0.3, epsilon);
            assert!(
                previous || !accepted,
                "acceptance flipped back on at {epsilon}"
            );
            previous = accepted;
        }
        assert!(accept_finalist(0.6, 0.3, 0.15));
        assert!(!accept_finalist(0.6, 0.3, 0.1501));
        assert!(!accept_finalist(f64::NAN, 0.3, 0.0));
    }
}
