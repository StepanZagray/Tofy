//! Soft, non-destructive Phase A goal-posterior updates.

use std::fmt;

pub const LIKELIHOOD_EPSILON: f64 = 0.02;
pub const MASS_FLOOR: f64 = 1e-4;
pub const PROTECTED_MASS: f64 = 0.95;
pub const PROTECTED_WEIGHT: f64 = 0.02;

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefState {
    pub concrete_weights: Vec<f64>,
    pub unknown_mass: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BeliefError {
    EmptyCandidates,
    InvalidMass,
    InvalidLikelihoodCount,
    InvalidLikelihood,
    InvalidEta,
}

impl fmt::Display for BeliefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidates => write!(f, "belief requires a concrete candidate"),
            Self::InvalidMass => {
                write!(f, "belief masses must be finite, positive, and sum to one")
            }
            Self::InvalidLikelihoodCount => {
                write!(f, "one likelihood is required per concrete candidate")
            }
            Self::InvalidLikelihood => write!(f, "likelihoods must be finite and non-negative"),
            Self::InvalidEta => write!(f, "eta must be in [0, 1]"),
        }
    }
}

impl std::error::Error for BeliefError {}

impl BeliefState {
    pub fn new(concrete_weights: Vec<f64>, unknown_mass: f64) -> Result<Self, BeliefError> {
        let state = Self {
            concrete_weights,
            unknown_mass,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn protected_indices(&self) -> Vec<usize> {
        let concrete_total = 1.0 - self.unknown_mass;
        let mut ranked = self
            .concrete_weights
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let mut protected = Vec::new();
        let mut covered = 0.0;
        for (index, weight) in ranked {
            if covered < PROTECTED_MASS * concrete_total || weight >= PROTECTED_WEIGHT {
                protected.push(index);
                covered += weight;
            }
        }
        protected.sort_unstable();
        protected
    }

    pub fn soft_update(
        &mut self,
        likelihoods: &[f64],
        eta: f64,
        tau_unknown: f64,
    ) -> Result<(), BeliefError> {
        self.validate()?;
        if likelihoods.len() != self.concrete_weights.len() {
            return Err(BeliefError::InvalidLikelihoodCount);
        }
        if !(0.0..=1.0).contains(&eta) || !eta.is_finite() {
            return Err(BeliefError::InvalidEta);
        }
        if likelihoods
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(BeliefError::InvalidLikelihood);
        }
        if eta == 0.0 {
            return Ok(());
        }
        let likelihoods = likelihoods
            .iter()
            .map(|value| value.clamp(LIKELIHOOD_EPSILON, 1.0 - LIKELIHOOD_EPSILON))
            .collect::<Vec<_>>();
        let concrete_total = 1.0 - self.unknown_mass;
        let mixture_likelihood = self
            .concrete_weights
            .iter()
            .zip(&likelihoods)
            .map(|(weight, likelihood)| weight / concrete_total * likelihood)
            .sum::<f64>();
        let surprise = -mixture_likelihood.max(LIKELIHOOD_EPSILON).ln();
        let unknown_logit = logit(self.unknown_mass) + eta * (surprise - tau_unknown);
        let next_unknown = sigmoid(unknown_logit.clamp(logit(0.05), logit(0.95)));
        let raw = self
            .concrete_weights
            .iter()
            .zip(&likelihoods)
            .map(|(weight, likelihood)| (weight * likelihood.powf(eta)).max(MASS_FLOOR))
            .collect::<Vec<_>>();
        let raw_total = raw.iter().sum::<f64>();
        self.unknown_mass = next_unknown;
        self.concrete_weights = raw
            .into_iter()
            .map(|weight| (1.0 - next_unknown) * weight / raw_total)
            .collect();
        self.validate()
    }

    fn validate(&self) -> Result<(), BeliefError> {
        if self.concrete_weights.is_empty() {
            return Err(BeliefError::EmptyCandidates);
        }
        if !self.unknown_mass.is_finite()
            || self.unknown_mass <= 0.0
            || self.unknown_mass >= 1.0
            || self
                .concrete_weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight <= 0.0)
            || ((self.concrete_weights.iter().sum::<f64>() + self.unknown_mass) - 1.0).abs() > 1e-9
        {
            return Err(BeliefError::InvalidMass);
        }
        Ok(())
    }
}

fn logit(value: f64) -> f64 {
    (value / (1.0 - value)).ln()
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn belief() -> BeliefState {
        BeliefState::new(vec![0.4, 0.4], 0.2).unwrap()
    }

    #[test]
    fn updates_preserve_total_mass_and_eta_zero_is_identity() {
        let mut state = belief();
        let before = state.clone();
        state.soft_update(&[0.9, 0.1], 0.0, 1.0).unwrap();
        assert_eq!(state, before);
        for eta in [0.1, 0.5, 1.0] {
            let mut sample = belief();
            sample.soft_update(&[0.8, 0.2], eta, 1.0).unwrap();
            assert!(
                ((sample.concrete_weights.iter().sum::<f64>() + sample.unknown_mass) - 1.0).abs()
                    < 1e-9
            );
        }
    }

    #[test]
    fn higher_likelihood_raises_relative_mass() {
        let mut belief = belief();
        belief.soft_update(&[0.9, 0.1], 1.0, 10.0).unwrap();
        assert!(belief.concrete_weights[0] > belief.concrete_weights[1]);
    }

    #[test]
    fn repeated_surprise_raises_unknown_within_bounds_without_deletion() {
        let mut belief = belief();
        let mut previous = belief.unknown_mass;
        for _ in 0..20 {
            belief.soft_update(&[0.02, 0.02], 1.0, 0.1).unwrap();
            assert!(belief.unknown_mass >= previous);
            assert!((0.05..=0.95).contains(&belief.unknown_mass));
            assert!(belief.concrete_weights.iter().all(|weight| *weight > 0.0));
            previous = belief.unknown_mass;
        }
    }

    #[test]
    fn protected_set_covers_mass_and_keeps_threshold_members() {
        let belief = BeliefState::new(vec![0.76, 0.11, 0.10, 0.01], 0.02).unwrap();
        let protected = belief.protected_indices();
        assert!(protected.contains(&0));
        assert!(protected.contains(&1));
        assert!(protected.contains(&2));
    }

    #[test]
    fn deterministic_property_samples_match_exactly() {
        for first in 1..50 {
            let first = first as f64 / 100.0;
            let second = 0.8 - first;
            let mut left = BeliefState::new(vec![first, second], 0.2).unwrap();
            let mut right = left.clone();
            left.soft_update(&[0.31, 0.73], 0.7, 0.9).unwrap();
            right.soft_update(&[0.31, 0.73], 0.7, 0.9).unwrap();
            assert_eq!(left, right);
        }
    }
}
