//! A2 bridge: deterministic template instantiation from a feature inventory.
//!
//! Every proposal produced here carries a `g19` vector that byte-matches the
//! layout of `GoalFeatures::encode` in `p2::data` so that candidates can be fed
//! to the trained consumer without any tensor code on this side. The predicates
//! are evaluated exactly on frame deltas; nothing here consults a model.

use super::goals::{
    CandidateProposal, ComponentFeature, FeatureInventory, GoalCandidate, FRAME_PIXELS,
};
use std::collections::BTreeMap;

/// Length of the public goal feature vector (`GOAL_FEATURES_DIM` in `p2::data`).
pub const G19_DIM: usize = 19;

/// Family one-hot slots, mirroring `family_index` in `p2::data`.
pub const FAMILY_REACH_MARKER: usize = 0;
pub const FAMILY_COLLECT_ALL: usize = 1;
pub const FAMILY_ACTIVATE_SWITCHES_IN_ORDER: usize = 2;
pub const FAMILY_PRESERVE_RESOURCE_REACH_MARKER: usize = 3;
pub const FAMILY_AVOID_HAZARD_REACH_MARKER: usize = 4;
pub const FAMILY_TRIGGER_TERMINAL: usize = 5;

/// Argument slots, mirroring `GoalFeatures::encode` in `p2::data`.
pub const SLOT_MARKER: usize = 6;
pub const SLOT_MIN_RESOURCE: usize = 7;
pub const SLOT_HAZARD: usize = 8;
pub const SLOT_TRIGGER: usize = 9;

/// Stable family labels, mirroring `goal_family` in `domain`.
pub const REACH_MARKER: &str = "reach_marker";
pub const COLLECT_ALL: &str = "collect_all";
pub const AVOID_HAZARD_REACH_MARKER: &str = "avoid_hazard_reach_marker";
pub const TRIGGER_TERMINAL: &str = "trigger_terminal";

/// Per-family instantiation caps applied after salience ranking.
pub const MAX_REACH_INSTANCES: usize = 8;
pub const MAX_COLLECT_INSTANCES: usize = 4;
pub const MAX_AVOID_PAIRS: usize = 8;
pub const MAX_TRIGGER_INSTANCES: usize = 4;

/// Gameplay area used for the relative size thresholds below. The inventory is
/// always built from a full 64x64 frame, so the whole frame is the reference.
pub const GAMEPLAY_AREA: usize = FRAME_PIXELS;

/// Reach markers cover at most this fraction (numerator / denominator) of the
/// gameplay area: 5%.
const MARKER_MAX_AREA_NUM: usize = 5;
const MARKER_MAX_AREA_DEN: usize = 100;
/// Reach markers have at most this many same-color components.
const MARKER_MAX_COMPONENTS: usize = 2;
/// Hazard colors have at least this many components ...
const HAZARD_MIN_COMPONENTS: usize = 3;
/// ... or cover at least this fraction of the gameplay area: 15%.
const HAZARD_MIN_AREA_NUM: usize = 15;
const HAZARD_MIN_AREA_DEN: usize = 100;
/// Trigger singletons may be this small even when they do not touch the border.
const TRIGGER_MAX_SMALL_AREA: usize = 4;

/// Additive salience bonus for every candidate whose argument colors changed
/// their pixel count against the previous inventory. Applied once per
/// candidate (not once per argument) after the family salience is computed
/// and before the per-family caps are enforced, so recently-changed colors
/// survive pruning.
pub const TRANSITION_SALIENCE_BONUS: f32 = 0.5;

#[derive(Debug, Clone)]
struct ColorStats {
    color: u8,
    total_area: usize,
    components: usize,
    touches_border: bool,
}

/// Deterministic bounded template instantiation.
///
/// Colors equal to `background` never appear as arguments. Output order is
/// reach markers, collect-all, avoid/reach pairs, then triggers; within a
/// family, proposals are sorted by descending salience with ascending color
/// arguments as the tie-break, then truncated to the family cap.
pub fn propose_candidates(
    inventory: &FeatureInventory,
    previous: Option<&FeatureInventory>,
    background: u8,
) -> Vec<CandidateProposal> {
    let stats = color_stats(inventory, background);
    let changed = |color: u8| -> bool {
        previous.is_some_and(|prev| {
            prev.palette_counts.get(&color).copied().unwrap_or(0)
                != inventory.palette_counts.get(&color).copied().unwrap_or(0)
        })
    };
    let bonus = |colors: &[u8]| -> f32 {
        if colors.iter().any(|color| changed(*color)) {
            TRANSITION_SALIENCE_BONUS
        } else {
            0.0
        }
    };

    let markers = stats
        .values()
        .filter(|stat| {
            stat.components >= 1
                && stat.components <= MARKER_MAX_COMPONENTS
                && stat.total_area * MARKER_MAX_AREA_DEN <= GAMEPLAY_AREA * MARKER_MAX_AREA_NUM
        })
        .map(|stat| (stat.color, 1.0 / stat.total_area as f32))
        .collect::<Vec<_>>();
    let hazards = stats
        .values()
        .filter(|stat| {
            stat.components >= HAZARD_MIN_COMPONENTS
                || stat.total_area * HAZARD_MIN_AREA_DEN >= GAMEPLAY_AREA * HAZARD_MIN_AREA_NUM
        })
        .map(|stat| {
            (
                stat.color,
                stat.components as f32 + stat.total_area as f32 / GAMEPLAY_AREA as f32,
            )
        })
        .collect::<Vec<_>>();

    let mut out = Vec::new();

    let mut reach = markers
        .iter()
        .map(|(color, salience)| proposal(REACH_MARKER, &[*color], salience + bonus(&[*color])))
        .collect::<Vec<_>>();
    rank(&mut reach);
    reach.truncate(MAX_REACH_INSTANCES);
    out.extend(reach);

    let mut collect = stats
        .values()
        .filter(|stat| stat.components >= 2)
        .map(|stat| {
            proposal(
                COLLECT_ALL,
                &[stat.color],
                stat.components as f32 + bonus(&[stat.color]),
            )
        })
        .collect::<Vec<_>>();
    rank(&mut collect);
    collect.truncate(MAX_COLLECT_INSTANCES);
    out.extend(collect);

    let mut avoid = Vec::new();
    for (hazard, hazard_salience) in &hazards {
        for (marker, marker_salience) in &markers {
            if hazard == marker {
                continue;
            }
            avoid.push(proposal(
                AVOID_HAZARD_REACH_MARKER,
                &[*hazard, *marker],
                hazard_salience * marker_salience + bonus(&[*hazard, *marker]),
            ));
        }
    }
    rank(&mut avoid);
    avoid.truncate(MAX_AVOID_PAIRS);
    out.extend(avoid);

    let mut triggers = stats
        .values()
        .filter(|stat| {
            stat.components == 1
                && (stat.touches_border || (1..=TRIGGER_MAX_SMALL_AREA).contains(&stat.total_area))
        })
        .map(|stat| {
            let border = if stat.touches_border { 1.0 } else { 0.0 };
            proposal(
                TRIGGER_TERMINAL,
                &[stat.color],
                border + 1.0 / stat.total_area as f32 + bonus(&[stat.color]),
            )
        })
        .collect::<Vec<_>>();
    rank(&mut triggers);
    triggers.truncate(MAX_TRIGGER_INSTANCES);
    out.extend(triggers);

    out
}

/// Exact frame-delta predicate for a template candidate.
///
/// Unknown families or unparseable arguments evaluate to `false` (fail closed).
pub fn evaluate_predicate(
    candidate: &GoalCandidate,
    start: &FeatureInventory,
    end: &FeatureInventory,
) -> bool {
    let args = candidate
        .args
        .iter()
        .map(|arg| arg.trim().parse::<u8>().ok())
        .collect::<Option<Vec<_>>>();
    let Some(args) = args else {
        return false;
    };
    match (
        candidate.family.trim().to_ascii_lowercase().as_str(),
        args.as_slice(),
    ) {
        (REACH_MARKER, [marker]) => marker_reached(*marker, start, end),
        (COLLECT_ALL, [color]) => pixel_count(start, *color) > 0 && pixel_count(end, *color) == 0,
        (AVOID_HAZARD_REACH_MARKER, [hazard, marker]) => {
            marker_reached(*marker, start, end)
                && pixel_count(end, *hazard) <= pixel_count(start, *hazard)
        }
        (TRIGGER_TERMINAL, [trigger]) => {
            let before = components_of(start, *trigger);
            !before.is_empty() && before != components_of(end, *trigger)
        }
        _ => false,
    }
}

/// Marker component set shrank (fewer components or fewer pixels, which also
/// covers partial recoloring) or moved (any centroid delta strictly positive).
fn marker_reached(marker: u8, start: &FeatureInventory, end: &FeatureInventory) -> bool {
    let before = components_of(start, marker);
    if before.is_empty() {
        return false;
    }
    let after = components_of(end, marker);
    if after.len() < before.len() || pixel_count(end, marker) < pixel_count(start, marker) {
        return true;
    }
    after.len() == before.len()
        && before.iter().zip(&after).any(|(left, right)| {
            (left.centroid_x - right.centroid_x).abs() > 0.0
                || (left.centroid_y - right.centroid_y).abs() > 0.0
        })
}

/// Feature layout mirroring `GoalFeatures::encode`: family one-hot in slots
/// `0..6`, marker in slot 6, hazard in slot 8, trigger in slot 9. Argument
/// colors are stored as raw `f32::from(u8)`, exactly as the encoder does with
/// indices. Families without arguments (collect-all) only set the one-hot.
fn g19_for(family: &str, args: &[u8]) -> [f32; G19_DIM] {
    let mut values = [0.0f32; G19_DIM];
    match (family, args) {
        (REACH_MARKER, [marker]) => {
            values[FAMILY_REACH_MARKER] = 1.0;
            values[SLOT_MARKER] = f32::from(*marker);
        }
        (COLLECT_ALL, _) => {
            values[FAMILY_COLLECT_ALL] = 1.0;
        }
        (AVOID_HAZARD_REACH_MARKER, [hazard, marker]) => {
            values[FAMILY_AVOID_HAZARD_REACH_MARKER] = 1.0;
            values[SLOT_MARKER] = f32::from(*marker);
            values[SLOT_HAZARD] = f32::from(*hazard);
        }
        (TRIGGER_TERMINAL, [trigger]) => {
            values[FAMILY_TRIGGER_TERMINAL] = 1.0;
            values[SLOT_TRIGGER] = f32::from(*trigger);
        }
        _ => {}
    }
    values
}

fn proposal(family: &str, args: &[u8], salience: f32) -> CandidateProposal {
    let rendered = args.iter().map(u8::to_string).collect::<Vec<_>>();
    CandidateProposal {
        candidate: GoalCandidate {
            family: family.to_string(),
            args: rendered.clone(),
            g19: g19_for(family, args),
            predicate_id: format!("{family}:{}", rendered.join(":")),
        },
        salience,
    }
}

fn rank(proposals: &mut [CandidateProposal]) {
    proposals.sort_by(|left, right| {
        right
            .salience
            .total_cmp(&left.salience)
            .then_with(|| left.candidate.args.cmp(&right.candidate.args))
    });
}

fn color_stats(inventory: &FeatureInventory, background: u8) -> BTreeMap<u8, ColorStats> {
    let mut stats = BTreeMap::<u8, ColorStats>::new();
    for component in &inventory.components {
        if component.color == background {
            continue;
        }
        let entry = stats.entry(component.color).or_insert(ColorStats {
            color: component.color,
            total_area: 0,
            components: 0,
            touches_border: false,
        });
        entry.total_area += component.area;
        entry.components += 1;
        entry.touches_border |= component.touches_border;
    }
    stats
}

fn pixel_count(inventory: &FeatureInventory, color: u8) -> usize {
    inventory.palette_counts.get(&color).copied().unwrap_or(0)
}

/// Components of one color in a position-stable order independent of BFS
/// discovery order.
fn components_of(inventory: &FeatureInventory, color: u8) -> Vec<ComponentFeature> {
    let mut found = inventory
        .components
        .iter()
        .filter(|component| component.color == color)
        .cloned()
        .collect::<Vec<_>>();
    found.sort_by(|left, right| {
        (left.min_y, left.min_x, left.max_y, left.max_x, left.area).cmp(&(
            right.min_y,
            right.min_x,
            right.max_y,
            right.max_x,
            right.area,
        ))
    });
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{goal_family, Goal};
    use crate::p2::data::GoalFeatures;
    use crate::p2::latent_planning::goals::{feature_inventory, PlainFrame, FRAME_SIDE};

    const BACKGROUND: u8 = 0;

    struct Canvas(Vec<u8>);

    impl Canvas {
        fn new() -> Self {
            Self(vec![BACKGROUND; FRAME_PIXELS])
        }

        fn rect(mut self, x: usize, y: usize, w: usize, h: usize, color: u8) -> Self {
            for yy in y..y + h {
                for xx in x..x + w {
                    self.0[yy * FRAME_SIDE + xx] = color;
                }
            }
            self
        }

        fn inventory(self) -> FeatureInventory {
            let frame = PlainFrame::new(self.0, (0..16).collect()).unwrap();
            feature_inventory(&frame)
        }
    }

    fn by_family<'a>(
        proposals: &'a [CandidateProposal],
        family: &str,
    ) -> Vec<&'a CandidateProposal> {
        proposals
            .iter()
            .filter(|proposal| proposal.candidate.family == family)
            .collect()
    }

    fn candidate(family: &str, args: &[u8]) -> GoalCandidate {
        proposal(family, args, 1.0).candidate
    }

    /// Two small markers (colors 3, 4), a wall-like hazard of many pieces (color 1),
    /// a collectible scattered as three pieces (color 6), and a border trigger (13).
    fn scene() -> Canvas {
        Canvas::new()
            .rect(10, 10, 2, 2, 3)
            .rect(40, 40, 1, 1, 4)
            .rect(20, 5, 1, 1, 1)
            .rect(22, 5, 1, 1, 1)
            .rect(24, 5, 1, 1, 1)
            .rect(30, 30, 1, 1, 6)
            .rect(32, 30, 1, 1, 6)
            .rect(34, 30, 1, 1, 6)
            .rect(0, 50, 3, 3, 13)
    }

    #[test]
    fn family_labels_match_domain() {
        assert_eq!(goal_family(&Goal::ReachMarker { marker: 0 }), REACH_MARKER);
        assert_eq!(goal_family(&Goal::CollectAll), COLLECT_ALL);
        assert_eq!(
            goal_family(&Goal::AvoidHazardReachMarker {
                hazard: 0,
                marker: 0
            }),
            AVOID_HAZARD_REACH_MARKER
        );
        assert_eq!(
            goal_family(&Goal::TriggerTerminal { trigger: 0 }),
            TRIGGER_TERMINAL
        );
    }

    #[test]
    fn g19_matches_goal_features_encode_for_all_bridged_families() {
        assert_eq!(G19_DIM, crate::p2::data::GOAL_FEATURES_DIM);
        for value in [0u8, 3, 7, 15, 255] {
            assert_eq!(
                g19_for(REACH_MARKER, &[value]),
                GoalFeatures::encode(&Goal::ReachMarker { marker: value }).values
            );
            assert_eq!(
                g19_for(TRIGGER_TERMINAL, &[value]),
                GoalFeatures::encode(&Goal::TriggerTerminal { trigger: value }).values
            );
            for marker in [0u8, 5, 200] {
                assert_eq!(
                    g19_for(AVOID_HAZARD_REACH_MARKER, &[value, marker]),
                    GoalFeatures::encode(&Goal::AvoidHazardReachMarker {
                        hazard: value,
                        marker
                    })
                    .values
                );
            }
        }
        assert_eq!(
            g19_for(COLLECT_ALL, &[6]),
            GoalFeatures::encode(&Goal::CollectAll).values
        );
        assert_eq!(g19_for(COLLECT_ALL, &[6]), g19_for(COLLECT_ALL, &[9]));
    }

    #[test]
    fn proposals_carry_encoder_matching_features() {
        let proposals = propose_candidates(&scene().inventory(), None, BACKGROUND);
        let reach = by_family(&proposals, REACH_MARKER);
        assert!(!reach.is_empty());
        for proposal in reach {
            let marker = proposal.candidate.args[0].parse::<u8>().unwrap();
            assert_eq!(
                proposal.candidate.g19,
                GoalFeatures::encode(&Goal::ReachMarker { marker }).values
            );
        }
        let avoid = by_family(&proposals, AVOID_HAZARD_REACH_MARKER);
        assert!(!avoid.is_empty());
        for proposal in avoid {
            let hazard = proposal.candidate.args[0].parse::<u8>().unwrap();
            let marker = proposal.candidate.args[1].parse::<u8>().unwrap();
            assert_eq!(
                proposal.candidate.g19,
                GoalFeatures::encode(&Goal::AvoidHazardReachMarker { hazard, marker }).values
            );
        }
        // Colors 1 and 6 both have three pieces; both are collect-all candidates
        // sharing one argument-free g19 but distinct predicate ids.
        let collect = by_family(&proposals, COLLECT_ALL);
        assert_eq!(collect.len(), 2);
        let ids = collect
            .iter()
            .map(|proposal| proposal.candidate.predicate_id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"collect_all:1"));
        assert!(ids.contains(&"collect_all:6"));
        assert_ne!(collect[0].candidate.args, collect[1].candidate.args);
        for proposal in collect {
            assert_eq!(
                proposal.candidate.g19,
                GoalFeatures::encode(&Goal::CollectAll).values
            );
        }
    }

    #[test]
    fn proposals_are_deterministic_and_ordered() {
        let inventory = scene().inventory();
        let first = propose_candidates(&inventory, None, BACKGROUND);
        let second = propose_candidates(&inventory, None, BACKGROUND);
        assert_eq!(first, second);
        let reach = by_family(&first, REACH_MARKER);
        assert_eq!(reach.len(), 3);
        // Color 4 (area 1) outranks color 3 (area 4) outranks color 13 (area 9).
        assert_eq!(reach[0].candidate.args, vec!["4".to_string()]);
        assert_eq!(reach[1].candidate.args, vec!["3".to_string()]);
        assert_eq!(reach[2].candidate.args, vec!["13".to_string()]);
        let triggers = by_family(&first, TRIGGER_TERMINAL);
        assert_eq!(triggers[0].candidate.args, vec!["13".to_string()]);
        assert!(!first
            .iter()
            .any(|proposal| proposal.candidate.args.iter().any(|arg| arg == "0")));
    }

    #[test]
    fn family_caps_are_enforced() {
        // Twelve singleton marker colors, all touching the border row 0, and
        // one hazard color with many pieces to generate 12 avoid pairs.
        let mut canvas = Canvas::new();
        for color in 1..=12u8 {
            canvas = canvas.rect(usize::from(color) * 2, 0, 1, 1, color);
        }
        for i in 0..5 {
            canvas = canvas.rect(10 + i * 2, 20, 1, 1, 14);
        }
        for i in 0..3 {
            canvas = canvas.rect(10 + i * 2, 30, 1, 1, 15);
        }
        let proposals = propose_candidates(&canvas.inventory(), None, BACKGROUND);
        assert_eq!(
            by_family(&proposals, REACH_MARKER).len(),
            MAX_REACH_INSTANCES
        );
        assert_eq!(
            by_family(&proposals, TRIGGER_TERMINAL).len(),
            MAX_TRIGGER_INSTANCES
        );
        assert_eq!(
            by_family(&proposals, AVOID_HAZARD_REACH_MARKER).len(),
            MAX_AVOID_PAIRS
        );
        assert!(by_family(&proposals, COLLECT_ALL).len() <= MAX_COLLECT_INSTANCES);
        assert_eq!(by_family(&proposals, COLLECT_ALL).len(), 2);
    }

    #[test]
    fn large_or_fragmented_colors_are_not_markers() {
        // 5% of 4096 is 204.8 pixels: 14x15 = 210 exceeds it, 14x14 = 196 does not.
        let big = Canvas::new().rect(5, 5, 14, 15, 3).inventory();
        assert!(by_family(&propose_candidates(&big, None, BACKGROUND), REACH_MARKER).is_empty());
        let fits = Canvas::new().rect(5, 5, 14, 14, 3).inventory();
        assert_eq!(
            by_family(&propose_candidates(&fits, None, BACKGROUND), REACH_MARKER).len(),
            1
        );
        let fragmented = Canvas::new()
            .rect(1, 1, 1, 1, 3)
            .rect(3, 1, 1, 1, 3)
            .rect(5, 1, 1, 1, 3)
            .inventory();
        let proposals = propose_candidates(&fragmented, None, BACKGROUND);
        assert!(by_family(&proposals, REACH_MARKER).is_empty());
        assert_eq!(by_family(&proposals, COLLECT_ALL).len(), 1);
    }

    #[test]
    fn transition_bonus_promotes_changed_colors() {
        let previous = scene().inventory();
        let current = scene()
            .rect(40, 40, 1, 1, BACKGROUND)
            .rect(41, 41, 2, 2, 4)
            .inventory();
        let stale = propose_candidates(&current, None, BACKGROUND);
        let fresh = propose_candidates(&current, Some(&previous), BACKGROUND);
        let salience = |proposals: &[CandidateProposal], args: &[&str]| {
            proposals
                .iter()
                .find(|proposal| {
                    proposal.candidate.family == REACH_MARKER && proposal.candidate.args == args
                })
                .map(|proposal| proposal.salience)
                .unwrap()
        };
        assert!(
            (salience(&fresh, &["4"]) - salience(&stale, &["4"]) - TRANSITION_SALIENCE_BONUS).abs()
                < 1e-6
        );
        assert_eq!(salience(&fresh, &["3"]), salience(&stale, &["3"]));
    }

    #[test]
    fn reach_marker_predicate_truth_table() {
        let start = scene().inventory();
        let unchanged = scene().inventory();
        let shrunk = scene().rect(10, 10, 1, 1, 2).inventory();
        let moved = scene()
            .rect(10, 10, 2, 2, BACKGROUND)
            .rect(12, 12, 2, 2, 3)
            .inventory();
        let recolored = scene().rect(10, 10, 2, 2, 5).inventory();
        let reach = candidate(REACH_MARKER, &[3]);
        assert!(!evaluate_predicate(&reach, &start, &unchanged));
        assert!(evaluate_predicate(&reach, &start, &shrunk));
        assert!(evaluate_predicate(&reach, &start, &moved));
        assert!(evaluate_predicate(&reach, &start, &recolored));
        let other = candidate(REACH_MARKER, &[4]);
        assert!(!evaluate_predicate(&other, &start, &shrunk));
        let absent = candidate(REACH_MARKER, &[9]);
        assert!(!evaluate_predicate(&absent, &start, &shrunk));
    }

    #[test]
    fn collect_all_predicate_requires_zero_remaining_pixels() {
        let start = scene().inventory();
        let partial = scene().rect(30, 30, 1, 1, BACKGROUND).inventory();
        let cleared = scene()
            .rect(30, 30, 1, 1, BACKGROUND)
            .rect(32, 30, 1, 1, BACKGROUND)
            .rect(34, 30, 1, 1, BACKGROUND)
            .inventory();
        let collect = candidate(COLLECT_ALL, &[6]);
        assert!(!evaluate_predicate(&collect, &start, &start));
        assert!(!evaluate_predicate(&collect, &start, &partial));
        assert!(evaluate_predicate(&collect, &start, &cleared));
        assert!(!evaluate_predicate(
            &candidate(COLLECT_ALL, &[9]),
            &start,
            &cleared
        ));
    }

    #[test]
    fn avoid_hazard_predicate_requires_marker_and_non_growing_hazard() {
        let start = scene().inventory();
        let reached = scene().rect(10, 10, 1, 1, 2).inventory();
        let reached_more_hazard = scene()
            .rect(10, 10, 1, 1, 2)
            .rect(26, 5, 1, 1, 1)
            .inventory();
        let reached_less_hazard = scene()
            .rect(10, 10, 1, 1, 2)
            .rect(24, 5, 1, 1, BACKGROUND)
            .inventory();
        let avoid = candidate(AVOID_HAZARD_REACH_MARKER, &[1, 3]);
        assert!(!evaluate_predicate(&avoid, &start, &start));
        assert!(evaluate_predicate(&avoid, &start, &reached));
        assert!(!evaluate_predicate(&avoid, &start, &reached_more_hazard));
        assert!(evaluate_predicate(&avoid, &start, &reached_less_hazard));
    }

    #[test]
    fn trigger_predicate_detects_any_component_change() {
        let start = scene().inventory();
        let grown = scene().rect(0, 50, 4, 3, 13).inventory();
        let gone = scene().rect(0, 50, 3, 3, BACKGROUND).inventory();
        let recolored = scene().rect(0, 50, 1, 1, 2).inventory();
        let trigger = candidate(TRIGGER_TERMINAL, &[13]);
        assert!(!evaluate_predicate(&trigger, &start, &start));
        assert!(evaluate_predicate(&trigger, &start, &grown));
        assert!(evaluate_predicate(&trigger, &start, &gone));
        assert!(evaluate_predicate(&trigger, &start, &recolored));
        assert!(!evaluate_predicate(
            &candidate(TRIGGER_TERMINAL, &[9]),
            &start,
            &gone
        ));
    }

    #[test]
    fn malformed_candidates_fail_closed() {
        let start = scene().inventory();
        let cleared = scene().rect(10, 10, 2, 2, BACKGROUND).inventory();
        let bad_args = GoalCandidate {
            family: REACH_MARKER.into(),
            args: vec!["three".into()],
            g19: [0.0; G19_DIM],
            predicate_id: "reach_marker:three".into(),
        };
        assert!(!evaluate_predicate(&bad_args, &start, &cleared));
        let unknown = GoalCandidate {
            family: "activate_switches_in_order".into(),
            args: vec!["3".into()],
            g19: [0.0; G19_DIM],
            predicate_id: "activate:3".into(),
        };
        assert!(!evaluate_predicate(&unknown, &start, &cleared));
    }
}
