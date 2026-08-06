//! Reliability calibration metrics (ECE, AUROC, risk–coverage buckets).

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
        let conf = bucket_probs.iter().map(|p| f64::from(*p)).sum::<f64>() / bucket_probs.len() as f64;
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
        for k in i..j {
            if pairs[k].1 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j;
    }
    let auc = (rank_sum_pos - (n_pos * (n_pos + 1)) as f64 / 2.0)
        / (n_pos as f64 * n_neg as f64);
    Some(auc.clamp(0.0, 1.0))
}

/// Risk (1 - accuracy on kept set) at each coverage bucket, sorted by descending score.
pub fn risk_coverage_buckets(
    scores: &[f32],
    labels: &[bool],
    n_buckets: usize,
) -> Vec<(f64, f64)> {
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
}
