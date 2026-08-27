//! Reliability calibration metrics (ECE, AUROC, risk–coverage buckets).

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;

/// Expected calibration error for binary probabilities in `[0, 1]`.
pub fn expected_calibration_error(probs: &[f32], labels: &[bool], n_bins: usize) -> Option<f64> {
    if probs.len() != labels.len() || probs.is_empty() {
        return None;
    }
    let bins = n_bins.max(1);
    let mut ece = 0f64;
    let n = probs.len() as f64;
    for b in 0..bins {
        let lo = b as f64 / bins as f64;
        let hi = (b + 1) as f64 / bins as f64;
        let mut bucket_probs = Vec::new();
        let mut bucket_labels = Vec::new();
        for (&p, &y) in probs.iter().zip(labels.iter()) {
            let in_bucket = if b + 1 == bins {
                p as f64 >= lo && p as f64 <= hi
            } else {
                p as f64 >= lo && (p as f64) < hi
            };
            if in_bucket {
                bucket_probs.push(p);
                bucket_labels.push(y);
            }
        }
        if bucket_probs.is_empty() {
            continue;
        }
        let acc = bucket_labels.iter().filter(|&&y| y).count() as f64 / bucket_probs.len() as f64;
        let conf =
            bucket_probs.iter().map(|p| f64::from(*p)).sum::<f64>() / bucket_probs.len() as f64;
        ece += (bucket_probs.len() as f64 / n) * (acc - conf).abs();
    }
    Some(ece)
}

/// ROC-AUC via Mann–Whitney rank statistic (handles ties approximately).
pub fn binary_auroc(scores: &[f32], labels: &[bool]) -> Option<f64> {
    if scores.len() != labels.len() || scores.is_empty() {
        return None;
    }
    let n_pos = labels.iter().filter(|&&y| y).count();
    let n_neg = labels.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return None;
    }
    let mut pairs: Vec<(f32, bool)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank_sum_pos = 0f64;
    let mut i = 0usize;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        let avg_rank = ((i + j + 1) as f64) / 2.0;
        for &(_, positive) in &pairs[i..j] {
            if positive {
                rank_sum_pos += avg_rank;
            }
        }
        i = j;
    }
    let auc = (rank_sum_pos - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos as f64 * n_neg as f64);
    Some(auc.clamp(0.0, 1.0))
}

/// Risk (1 - accuracy on kept set) at each coverage bucket, sorted by descending score.
pub fn risk_coverage_buckets(scores: &[f32], labels: &[bool], n_buckets: usize) -> Vec<(f64, f64)> {
    if scores.len() != labels.len() || scores.is_empty() || n_buckets == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = Vec::with_capacity(n_buckets);
    for b in 1..=n_buckets {
        let keep = ((scores.len() * b) / n_buckets).max(1);
        let kept: Vec<bool> = order.iter().take(keep).map(|&i| labels[i]).collect();
        let acc = kept.iter().filter(|&&y| y).count() as f64 / kept.len() as f64;
        out.push((keep as f64 / scores.len() as f64, 1.0 - acc));
    }
    out
}

pub const CLOPPER_PEARSON_CONFIDENCE: f64 = 0.95;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CalibrationHead {
    Event,
    Q,
    Reliability,
    Ptrm,
    InverseAction,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CalibrationStratum {
    pub head: CalibrationHead,
    pub source_family: String,
    pub horizon: u8,
    pub changed: bool,
    pub irreversible: bool,
    pub probability_bin: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationCount {
    pub total: u64,
    pub failures: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationCoverage {
    pub total: u64,
    pub failures: u64,
    pub error_lower: f64,
    pub error_upper: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinnedCalibrationTable {
    pub minimum_count: u64,
    pub bins: BTreeMap<CalibrationStratum, CalibrationCount>,
}

#[derive(Serialize, Deserialize)]
struct CalibrationBin {
    stratum: CalibrationStratum,
    count: CalibrationCount,
}

#[derive(Serialize, Deserialize)]
struct CalibrationTableWire {
    minimum_count: u64,
    bins: Vec<CalibrationBin>,
}

impl Serialize for BinnedCalibrationTable {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        CalibrationTableWire {
            minimum_count: self.minimum_count,
            bins: self
                .bins
                .iter()
                .map(|(stratum, count)| CalibrationBin {
                    stratum: stratum.clone(),
                    count: *count,
                })
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BinnedCalibrationTable {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CalibrationTableWire::deserialize(deserializer)?;
        Ok(Self {
            minimum_count: wire.minimum_count,
            bins: wire
                .bins
                .into_iter()
                .map(|bin| (bin.stratum, bin.count))
                .collect(),
        })
    }
}

impl BinnedCalibrationTable {
    pub fn new(minimum_count: u64) -> Self {
        Self {
            minimum_count,
            bins: BTreeMap::new(),
        }
    }

    pub fn record(&mut self, stratum: CalibrationStratum, failure: bool) {
        let count = self.bins.entry(stratum).or_insert(CalibrationCount {
            total: 0,
            failures: 0,
        });
        count.total += 1;
        count.failures += u64::from(failure);
    }

    /// Missing and undersampled strata have no coverage and must truncate planning.
    pub fn coverage(&self, stratum: &CalibrationStratum) -> Option<CalibrationCoverage> {
        let count = self.bins.get(stratum)?;
        if count.total < self.minimum_count {
            return None;
        }
        Some(CalibrationCoverage {
            total: count.total,
            failures: count.failures,
            error_lower: clopper_pearson_lower(count.failures, count.total)?,
            error_upper: clopper_pearson_upper(count.failures, count.total)?,
        })
    }

    pub fn accepts_error_upper(&self, stratum: &CalibrationStratum, maximum: f64) -> bool {
        self.coverage(stratum)
            .is_some_and(|coverage| coverage.error_upper <= maximum)
    }
}

/// An inverse success is a falsifier pass, never a positive trust contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InverseActionCheck {
    Success,
    Failure,
}

impl InverseActionCheck {
    pub fn trust_adjustment(self) -> Option<f64> {
        match self {
            Self::Success => Some(0.0),
            Self::Failure => None,
        }
    }
}

/// Conservative 95% Clopper--Pearson lower endpoint for a binomial error rate.
pub fn clopper_pearson_lower(failures: u64, total: u64) -> Option<f64> {
    if total == 0 || failures > total {
        return None;
    }
    if failures == 0 {
        return Some(0.0);
    }
    Some(inverse_regularized_beta(
        failures as f64,
        (total - failures + 1) as f64,
        tail_probability(),
    ))
}

/// Conservative 95% Clopper--Pearson upper endpoint for a binomial error rate.
///
/// The endpoint uses the 2.5% tail, so it remains a safe one-sided gate when
/// reported alongside its matching lower endpoint (for example, 0/1 is 0.975).
pub fn clopper_pearson_upper(failures: u64, total: u64) -> Option<f64> {
    if total == 0 || failures > total {
        return None;
    }
    if failures == total {
        return Some(1.0);
    }
    Some(inverse_regularized_beta(
        (failures + 1) as f64,
        (total - failures) as f64,
        1.0 - tail_probability(),
    ))
}

fn tail_probability() -> f64 {
    (1.0 - CLOPPER_PEARSON_CONFIDENCE) / 2.0
}

fn inverse_regularized_beta(a: f64, b: f64, probability: f64) -> f64 {
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..120 {
        let middle = (low + high) / 2.0;
        if regularized_incomplete_beta(a, b, middle) < probability {
            low = middle;
        } else {
            high = middle;
        }
    }
    (low + high) / 2.0
}

fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front =
        (log_gamma(a + b) - log_gamma(a) - log_gamma(b) + a * x.ln() + b * (-x).ln_1p()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const EPSILON: f64 = 3e-14;
    const MINIMUM: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < MINIMUM {
        d = MINIMUM;
    }
    d = 1.0 / d;
    let mut value = d;
    for m in 1..=200 {
        let m = m as f64;
        let twice_m = 2.0 * m;
        let mut aa = m * (b - m) * x / ((qam + twice_m) * (a + twice_m));
        d = 1.0 + aa * d;
        if d.abs() < MINIMUM {
            d = MINIMUM;
        }
        c = 1.0 + aa / c;
        if c.abs() < MINIMUM {
            c = MINIMUM;
        }
        d = 1.0 / d;
        value *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + twice_m) * (qap + twice_m));
        d = 1.0 + aa * d;
        if d.abs() < MINIMUM {
            d = MINIMUM;
        }
        c = 1.0 + aa / c;
        if c.abs() < MINIMUM {
            c = MINIMUM;
        }
        d = 1.0 / d;
        let delta = d * c;
        value *= delta;
        if (delta - 1.0).abs() < EPSILON {
            break;
        }
    }
    value
}

fn log_gamma(value: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let shifted = value - 1.0;
    let series = COEFFICIENTS
        .iter()
        .enumerate()
        .skip(1)
        .fold(COEFFICIENTS[0], |sum, (index, coefficient)| {
            sum + coefficient / (shifted + index as f64)
        });
    let t = shifted + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (shifted + 0.5) * t.ln() - t + series.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_calibration_has_low_ece() {
        let probs = vec![0.1, 0.1, 0.9, 0.9];
        let labels = vec![false, false, true, true];
        let ece = expected_calibration_error(&probs, &labels, 2).unwrap();
        assert!((ece - 0.1).abs() < 1e-6);
    }

    #[test]
    fn auroc_perfect_ranking() {
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        let labels = vec![false, false, true, true];
        let auc = binary_auroc(&scores, &labels).unwrap();
        assert!((auc - 1.0).abs() < 1e-6);
    }

    fn stratum(source_family: &str) -> CalibrationStratum {
        CalibrationStratum {
            head: CalibrationHead::Q,
            source_family: source_family.into(),
            horizon: 1,
            changed: true,
            irreversible: false,
            probability_bin: 9,
        }
    }

    #[test]
    fn one_zero_failure_sample_cannot_pass_irreversible_gate() {
        let mut table = BinnedCalibrationTable::new(1);
        let key = stratum("collect");
        table.record(key.clone(), false);
        let coverage = table.coverage(&key).unwrap();
        assert!((coverage.error_upper - 0.975).abs() < 1e-12);
        assert!(!table.accepts_error_upper(&key, 0.02));
    }

    #[test]
    fn clopper_pearson_matches_known_binomial_endpoints() {
        assert!((clopper_pearson_lower(1, 2).unwrap() - 0.012_579_117_093_425).abs() < 1e-6);
        assert!((clopper_pearson_upper(1, 2).unwrap() - 0.987_420_882_906_575).abs() < 1e-6);
        assert!((clopper_pearson_upper(0, 1).unwrap() - 0.975).abs() < 1e-6);
    }

    #[test]
    fn calibration_strata_never_pool_and_missing_coverage_fails_closed() {
        let mut table = BinnedCalibrationTable::new(2);
        let collect = stratum("collect");
        let avoid = stratum("avoid");
        table.record(collect.clone(), false);
        table.record(avoid.clone(), true);
        table.record(avoid.clone(), true);
        assert_eq!(table.coverage(&collect), None);
        assert_eq!(table.coverage(&avoid).unwrap().failures, 2);
    }

    #[test]
    fn calibration_serialization_is_byte_identical_for_identical_inputs() {
        let mut left = BinnedCalibrationTable::new(1);
        let mut right = BinnedCalibrationTable::new(1);
        let collect = stratum("collect");
        let avoid = stratum("avoid");
        for (key, failure) in [(collect.clone(), false), (avoid.clone(), true)] {
            left.record(key, failure);
        }
        for (key, failure) in [(avoid, true), (collect, false)] {
            right.record(key, failure);
        }
        assert_eq!(
            serde_json::to_vec(&left).unwrap(),
            serde_json::to_vec(&right).unwrap()
        );
    }

    #[test]
    fn inverse_success_adds_no_trust_and_failure_disqualifies() {
        assert_eq!(InverseActionCheck::Success.trust_adjustment(), Some(0.0));
        assert_eq!(InverseActionCheck::Failure.trust_adjustment(), None);
    }
}
