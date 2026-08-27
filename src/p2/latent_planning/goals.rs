//! Deterministic candidate-goal inventory and bounded prior construction.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

pub const FRAME_SIDE: usize = 64;
pub const FRAME_PIXELS: usize = FRAME_SIDE * FRAME_SIDE;
pub const MAX_CANDIDATES: usize = 32;
pub const UNKNOWN_PRIOR: f32 = 0.20;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoalCandidate {
    pub family: String,
    pub args: Vec<String>,
    pub g19: [f32; 19],
    pub predicate_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateProposal {
    pub candidate: GoalCandidate,
    pub salience: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DormantCandidate {
    pub candidate: GoalCandidate,
    pub salience: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateSet {
    pub candidates: Vec<GoalCandidate>,
    pub concrete_masses: Vec<f32>,
    pub unknown_mass: f32,
    pub dormant: Vec<DormantCandidate>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateSetError {
    InvalidUnknownPrior,
    InvalidFeature,
    TooManyFamilies { families: usize, cap: usize },
}

impl fmt::Display for CandidateSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUnknownPrior => write!(f, "unknown prior must be in [0, 1)"),
            Self::InvalidFeature => write!(f, "candidate feature or salience is not finite"),
            Self::TooManyFamilies { families, cap } => {
                write!(
                    f,
                    "{families} families cannot satisfy one-per-family cap {cap}"
                )
            }
        }
    }
}

impl std::error::Error for CandidateSetError {}

impl CandidateSet {
    /// Returns the bounded live set together with the exact candidates it pruned.
    pub fn restore_dormant(&self) -> Vec<CandidateProposal> {
        self.dormant
            .iter()
            .map(|dormant| CandidateProposal {
                candidate: dormant.candidate.clone(),
                salience: dormant.salience,
            })
            .collect()
    }
}

/// Deduplicate template instances and allocate equal family then instance prior mass.
pub fn build_candidate_set(
    proposals: impl IntoIterator<Item = CandidateProposal>,
    cap: usize,
    unknown_mass: f32,
) -> Result<CandidateSet, CandidateSetError> {
    let cap = cap.min(MAX_CANDIDATES);
    if !(0.0..1.0).contains(&unknown_mass) {
        return Err(CandidateSetError::InvalidUnknownPrior);
    }
    let mut unique = BTreeMap::<String, CandidateProposal>::new();
    for mut proposal in proposals {
        if !proposal.salience.is_finite()
            || proposal
                .candidate
                .g19
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(CandidateSetError::InvalidFeature);
        }
        proposal.candidate.family = normalize(&proposal.candidate.family);
        proposal.candidate.predicate_id = normalize(&proposal.candidate.predicate_id);
        let key = candidate_key(&proposal.candidate);
        match unique.get(&key) {
            Some(existing) if compare_proposals(existing, &proposal).is_le() => {}
            _ => {
                unique.insert(key, proposal);
            }
        }
    }

    let mut by_family = BTreeMap::<String, Vec<CandidateProposal>>::new();
    for proposal in unique.into_values() {
        by_family
            .entry(proposal.candidate.family.clone())
            .or_default()
            .push(proposal);
    }
    if by_family.len() > cap {
        return Err(CandidateSetError::TooManyFamilies {
            families: by_family.len(),
            cap,
        });
    }
    for proposals in by_family.values_mut() {
        proposals.sort_by(compare_proposals);
    }

    let mut selected = Vec::new();
    let mut remaining = Vec::new();
    for proposals in by_family.into_values() {
        let mut proposals = proposals.into_iter();
        if let Some(first) = proposals.next() {
            selected.push(first);
        }
        remaining.extend(proposals);
    }
    remaining.sort_by(compare_proposals);
    let open = cap.saturating_sub(selected.len());
    selected.extend(remaining.iter().take(open).cloned());
    selected.sort_by(|left, right| {
        candidate_key(&left.candidate).cmp(&candidate_key(&right.candidate))
    });

    let selected_keys = selected
        .iter()
        .map(|proposal| candidate_key(&proposal.candidate))
        .collect::<std::collections::BTreeSet<_>>();
    let mut dormant = remaining
        .into_iter()
        .filter(|proposal| !selected_keys.contains(&candidate_key(&proposal.candidate)))
        .map(|proposal| DormantCandidate {
            candidate: proposal.candidate,
            salience: proposal.salience,
        })
        .collect::<Vec<_>>();
    dormant.sort_by(|left, right| {
        compare_proposals(
            &CandidateProposal {
                candidate: left.candidate.clone(),
                salience: left.salience,
            },
            &CandidateProposal {
                candidate: right.candidate.clone(),
                salience: right.salience,
            },
        )
    });

    let families = selected
        .iter()
        .map(|proposal| proposal.candidate.family.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let family_count = families.len();
    let concrete_mass = 1.0 - unknown_mass;
    let mut instances = BTreeMap::<String, usize>::new();
    for proposal in &selected {
        *instances
            .entry(proposal.candidate.family.clone())
            .or_default() += 1;
    }
    let concrete_masses = selected
        .iter()
        .map(|proposal| {
            if family_count == 0 {
                0.0
            } else {
                concrete_mass / family_count as f32 / instances[&proposal.candidate.family] as f32
            }
        })
        .collect();
    Ok(CandidateSet {
        candidates: selected
            .into_iter()
            .map(|proposal| proposal.candidate)
            .collect(),
        concrete_masses,
        unknown_mass,
        dormant,
    })
}

fn compare_proposals(left: &CandidateProposal, right: &CandidateProposal) -> std::cmp::Ordering {
    right
        .salience
        .total_cmp(&left.salience)
        .then_with(|| candidate_key(&left.candidate).cmp(&candidate_key(&right.candidate)))
        .then_with(|| left.candidate.args.cmp(&right.candidate.args))
}

fn normalize(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn candidate_key(candidate: &GoalCandidate) -> String {
    let features = candidate
        .g19
        .iter()
        .map(|value| if *value == 0.0 { 0 } else { value.to_bits() })
        .map(|bits| format!("{bits:08x}"))
        .collect::<Vec<_>>()
        .join(":");
    format!("{}|{features}", normalize(&candidate.predicate_id))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    WrongPixelCount { actual: usize },
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongPixelCount { actual } => {
                write!(f, "expected {FRAME_PIXELS} frame pixels, got {actual}")
            }
        }
    }
}

impl std::error::Error for FrameError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlainFrame {
    pub pixels: Vec<u8>,
    pub palette: Vec<u8>,
}

impl PlainFrame {
    pub fn new(pixels: Vec<u8>, palette: Vec<u8>) -> Result<Self, FrameError> {
        if pixels.len() != FRAME_PIXELS {
            return Err(FrameError::WrongPixelCount {
                actual: pixels.len(),
            });
        }
        Ok(Self { pixels, palette })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentFeature {
    pub color: u8,
    pub area: usize,
    pub min_x: usize,
    pub min_y: usize,
    pub max_x: usize,
    pub max_y: usize,
    pub centroid_x: f32,
    pub centroid_y: f32,
    pub touches_border: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureInventory {
    pub palette_counts: BTreeMap<u8, usize>,
    pub components: Vec<ComponentFeature>,
}

/// Build a row-major, four-connected component inventory without model inputs.
pub fn feature_inventory(frame: &PlainFrame) -> FeatureInventory {
    let mut palette_counts = frame
        .palette
        .iter()
        .copied()
        .map(|color| (color, 0_usize))
        .collect::<BTreeMap<_, _>>();
    for color in &frame.pixels {
        *palette_counts.entry(*color).or_default() += 1;
    }
    let mut visited = vec![false; FRAME_PIXELS];
    let mut components = Vec::new();
    for start in 0..FRAME_PIXELS {
        if visited[start] {
            continue;
        }
        let color = frame.pixels[start];
        let mut queue = std::collections::VecDeque::from([start]);
        visited[start] = true;
        let mut area = 0usize;
        let mut min_x = FRAME_SIDE;
        let mut min_y = FRAME_SIDE;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut sum_x = 0usize;
        let mut sum_y = 0usize;
        let mut touches_border = false;
        while let Some(index) = queue.pop_front() {
            let x = index % FRAME_SIDE;
            let y = index / FRAME_SIDE;
            area += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            sum_x += x;
            sum_y += y;
            touches_border |= x == 0 || y == 0 || x + 1 == FRAME_SIDE || y + 1 == FRAME_SIDE;
            for neighbor in neighbors(index) {
                if !visited[neighbor] && frame.pixels[neighbor] == color {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        components.push(ComponentFeature {
            color,
            area,
            min_x,
            min_y,
            max_x,
            max_y,
            centroid_x: sum_x as f32 / area as f32,
            centroid_y: sum_y as f32 / area as f32,
            touches_border,
        });
    }
    FeatureInventory {
        palette_counts,
        components,
    }
}

fn neighbors(index: usize) -> impl Iterator<Item = usize> {
    let x = index % FRAME_SIDE;
    let y = index / FRAME_SIDE;
    [
        (x > 0).then(|| index - 1),
        (x + 1 < FRAME_SIDE).then(|| index + 1),
        (y > 0).then(|| index - FRAME_SIDE),
        (y + 1 < FRAME_SIDE).then(|| index + FRAME_SIDE),
    ]
    .into_iter()
    .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(family: &str, predicate: &str, feature: f32) -> GoalCandidate {
        GoalCandidate {
            family: family.into(),
            args: vec![family.into()],
            g19: [feature; 19],
            predicate_id: predicate.into(),
        }
    }

    #[test]
    fn inventory_and_candidates_are_deterministic() {
        let mut pixels = vec![0; FRAME_PIXELS];
        pixels[65] = 2;
        pixels[66] = 2;
        let frame = PlainFrame::new(pixels, vec![0, 2]).unwrap();
        assert_eq!(feature_inventory(&frame), feature_inventory(&frame));
        let proposals = vec![CandidateProposal {
            candidate: candidate("collect", "has blue", 1.0),
            salience: 1.0,
        }];
        assert_eq!(
            build_candidate_set(proposals.clone(), 32, UNKNOWN_PRIOR).unwrap(),
            build_candidate_set(proposals, 32, UNKNOWN_PRIOR).unwrap()
        );
    }

    #[test]
    fn candidates_deduplicate_by_normalized_predicate_and_features() {
        let set = build_candidate_set(
            [
                CandidateProposal {
                    candidate: candidate("collect", "  HAS   blue ", 1.0),
                    salience: 0.1,
                },
                CandidateProposal {
                    candidate: candidate("avoid", "has blue", 1.0),
                    salience: 0.9,
                },
            ],
            32,
            UNKNOWN_PRIOR,
        )
        .unwrap();
        assert_eq!(set.candidates.len(), 1);
        assert_eq!(set.candidates[0].family, "avoid");
    }

    #[test]
    fn family_mass_and_cap_preserve_a_representative() {
        let set = build_candidate_set(
            [
                CandidateProposal {
                    candidate: candidate("a", "a1", 1.0),
                    salience: 3.0,
                },
                CandidateProposal {
                    candidate: candidate("a", "a2", 2.0),
                    salience: 2.0,
                },
                CandidateProposal {
                    candidate: candidate("b", "b1", 3.0),
                    salience: 1.0,
                },
            ],
            2,
            UNKNOWN_PRIOR,
        )
        .unwrap();
        assert_eq!(set.candidates.len(), 2);
        assert!(set
            .candidates
            .iter()
            .any(|candidate| candidate.family == "a"));
        assert!(set
            .candidates
            .iter()
            .any(|candidate| candidate.family == "b"));
        assert!(((set.concrete_masses.iter().sum::<f32>() + set.unknown_mass) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pruned_candidates_remain_restorable() {
        let set = build_candidate_set(
            [
                CandidateProposal {
                    candidate: candidate("a", "a1", 1.0),
                    salience: 3.0,
                },
                CandidateProposal {
                    candidate: candidate("a", "a2", 2.0),
                    salience: 2.0,
                },
            ],
            1,
            UNKNOWN_PRIOR,
        )
        .unwrap();
        assert_eq!(set.restore_dormant().len(), 1);
        assert_eq!(set.restore_dormant()[0].candidate.predicate_id, "a2");
    }

    #[test]
    fn candidate_cap_never_exceeds_phase_a_limit() {
        let proposals = (0..MAX_CANDIDATES + 1)
            .map(|index| CandidateProposal {
                candidate: candidate("collect", &format!("goal-{index}"), index as f32),
                salience: index as f32,
            })
            .collect::<Vec<_>>();
        let set = build_candidate_set(proposals, MAX_CANDIDATES + 1, UNKNOWN_PRIOR).unwrap();
        assert_eq!(set.candidates.len(), MAX_CANDIDATES);
        assert_eq!(set.dormant.len(), 1);
    }
}
