//! Deterministic scenario generation with exact solvability filtering.

use crate::domain::{
    goal_satisfied, goal_terminal_failure, reachable_within, Action, Goal, Pos, Scenario,
    Simulator, Split,
};
use crate::search::{goal_viable_at, shortest_path};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;

const MAX_ATTEMPTS: u32 = 96;
const REACH_SLACK: u16 = 8;
/// Generous episode limit for deep false-lead layouts: several recoverable
/// wrong-goal round-trips remain feasible while oracle length (not this cap)
/// still drives exact oracle-normalized efficiency.
const P1C_HARD_BUDGET: u16 = 256;
const SPLIT_TRAIN_TAG: u64 = 0x5472_6169_6e00_0001;
const SPLIT_HELD_TAG: u64 = 0x4865_6c64_4f75_7402;

/// Square board sizes used by the world-core-v5 data contract.
pub const V5_CONTENT_SIZES: [u8; 7] = [7, 8, 10, 12, 16, 24, 32];

/// Stem geometry for the shared P1C false-lead transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum P1cFalseLeadLayout {
    /// One-step south side branch at `(0,1)` (standard P1C).
    ShallowSideBranch,
    /// Isolated south corridor ending at the bottom row (hard candidate).
    DeepSouthCorridor,
}

/// How `finalize_p1c_scenario` chooses the published action budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum P1cBudgetMode {
    /// `max(inherited+stem, required+REACH_SLACK)`.
    NormalizedSlack,
    /// Same floor, then raise to [`P1C_HARD_BUDGET`] for recoverable thrash.
    GenerousRecoverable,
}

/// Derive a ChaCha8 stream from `(seed, episode_id, split)`.
pub fn rng_for(seed: u64, episode_id: u64, split: Split) -> ChaCha8Rng {
    let split_tag: u64 = match split {
        Split::Train => SPLIT_TRAIN_TAG,
        Split::HeldOutComposition => SPLIT_HELD_TAG,
    };
    let mut key = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(episode_id.wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    key ^= split_tag.rotate_left(17);
    key = key.wrapping_mul(0x1656_67B1_9E37_79F9);
    ChaCha8Rng::seed_from_u64(key)
}

/// Generate a solvable moderate-diversity scenario.
pub fn generate(seed: u64, episode_id: u64, split: Split) -> Scenario {
    let mut rng = rng_for(seed, episode_id, split);
    for attempt in 0..MAX_ATTEMPTS {
        let mut attempt_rng = ChaCha8Rng::seed_from_u64(
            rng.random::<u64>()
                .wrapping_add(u64::from(attempt).wrapping_mul(0xD1B5_4A32_D192_E5AD)),
        );
        if let Some(sc) = try_generate(seed, episode_id, split, &mut attempt_rng) {
            return sc;
        }
    }
    fallback_scenario(seed, episode_id, split, &mut rng)
}

/// Generate the existing solvable task distribution at an exact v5 board size.
///
/// The accepted 7x7/8x8 scenario is deterministically dilated to the requested
/// square. Dilation preserves object identity and goal composition while
/// spreading geometry across the larger board; the additional cells are
/// semantic empty cells, not canvas padding. The action budget is scaled with
/// side length so held-out 16x16 and large training boards remain traversable.
pub fn generate_sized(seed: u64, episode_id: u64, split: Split, content_size: u8) -> Scenario {
    assert!(
        V5_CONTENT_SIZES.contains(&content_size),
        "v5 content size must be one of {V5_CONTENT_SIZES:?}, got {content_size}"
    );
    let mut scenario = generate(seed, episode_id, split);
    let source_width = scenario.width;
    let source_height = scenario.height;
    if source_width == content_size && source_height == content_size {
        return scenario;
    }

    let scale = |position: Pos| {
        let scale_axis = |value: i8, source: u8| -> i8 {
            let source_span = u16::from(source.saturating_sub(1)).max(1);
            let target_span = u16::from(content_size.saturating_sub(1));
            let numerator =
                u16::try_from(value).expect("scenario positions are non-negative") * target_span;
            i8::try_from((numerator + source_span / 2) / source_span)
                .expect("v5 content coordinates fit i8")
        };
        Pos::new(
            scale_axis(position.x, source_width),
            scale_axis(position.y, source_height),
        )
    };

    scenario.walls = std::mem::take(&mut scenario.walls)
        .into_iter()
        .map(scale)
        .collect();
    for positions in [
        &mut scenario.markers,
        &mut scenario.collectibles,
        &mut scenario.switches,
        &mut scenario.hazards,
        &mut scenario.resource_pickups,
        &mut scenario.terminal_triggers,
    ] {
        for position in positions {
            *position = scale(*position);
        }
    }
    scenario.start = scale(scenario.start);
    scenario.width = content_size;
    scenario.height = content_size;
    let source_side = source_width.max(source_height).max(1);
    scenario.action_budget = ((u32::from(scenario.action_budget) * u32::from(content_size))
        .div_ceil(u32::from(source_side)))
    .min(u32::from(u16::MAX)) as u16;
    scenario
}

/// Generate a P1C task with a guaranteed safe multi-goal falsification probe.
///
/// All exactly viable candidates are initially equally plausible: the public
/// interface supplies no privileged prior or hidden-goal hint. At least one
/// exact shortest plan among those candidates reaches a public state that
/// satisfies two or more exact-live goals without predicting terminal failure
/// for any exact-live candidate or irreversibly ruling out any other
/// exact-live goal along the way.
pub fn generate_p1c(seed: u64, episode_id: u64, split: Split) -> Scenario {
    let mut sc = generate(seed, episode_id, split);
    // Prefix the original map with a two-cell stem, then move marker 0 into a
    // one-step side branch with a resource pickup. South is the cheap
    // multi-goal falsification probe (reach marker 0 and co-test preserve /
    // avoid siblings); East enters the main map for the remaining families.
    apply_p1c_false_lead_transform(&mut sc, P1cFalseLeadLayout::ShallowSideBranch);
    finalize_p1c_scenario(&mut sc, P1cBudgetMode::NormalizedSlack);
    sc
}

/// Deterministic deep false-lead P1C candidate derived from a source episode.
///
/// Same six goal families and simulator mechanics as [`generate_p1c`], but the
/// stem places the agent at a fork: East enters the original map, while an
/// isolated south corridor runs to the bottom of the existing grid, ending in
/// marker 0 plus a resource pickup. The corridor endpoint remains a safe
/// exact multi-goal falsification probe of width ≥ 2. The published budget is
/// raised to a generous deterministic cap so several recoverable wrong-goal
/// round-trips fit; exact oracle normalization still uses oracle path length.
pub fn generate_p1c_hard_candidate(seed: u64, source_episode_id: u64, split: Split) -> Scenario {
    let mut sc = generate(seed, source_episode_id, split);
    apply_p1c_false_lead_transform(&mut sc, P1cFalseLeadLayout::DeepSouthCorridor);
    finalize_p1c_scenario(&mut sc, P1cBudgetMode::GenerousRecoverable);
    sc
}

/// Shift the base map east and install the P1C false-lead stem geometry.
fn apply_p1c_false_lead_transform(sc: &mut Scenario, layout: P1cFalseLeadLayout) {
    let shift = 2i8;
    let shift_pos = |p: &mut Pos| p.x += shift;
    sc.width = sc.width.checked_add(shift as u8).expect("small grid width");
    sc.walls = std::mem::take(&mut sc.walls)
        .into_iter()
        .map(|mut p| {
            shift_pos(&mut p);
            p
        })
        .collect();
    // Isolate the west stem from the shifted map except at the y=0 fork row.
    for y in 1..sc.height as i8 {
        sc.walls.insert(Pos::new(1, y));
    }
    match layout {
        P1cFalseLeadLayout::ShallowSideBranch => {
            // Only (0,1) stays open as the one-step side branch.
            for y in 2..sc.height as i8 {
                sc.walls.insert(Pos::new(0, y));
            }
        }
        P1cFalseLeadLayout::DeepSouthCorridor => {
            // Leave (0,1)..(0,height-1) open as the isolated south corridor.
        }
    }
    for positions in [
        &mut sc.markers,
        &mut sc.collectibles,
        &mut sc.switches,
        &mut sc.hazards,
        &mut sc.resource_pickups,
        &mut sc.terminal_triggers,
    ] {
        for p in positions {
            shift_pos(p);
        }
    }
    let marker_y = match layout {
        P1cFalseLeadLayout::ShallowSideBranch => 1i8,
        P1cFalseLeadLayout::DeepSouthCorridor => sc.height as i8 - 1,
    };
    sc.markers[0] = Pos::new(0, marker_y);
    // Side branch / corridor remains feasible for held-out preserve-resource
    // candidates whose threshold sits one above the initial resource.
    sc.resource_pickups.push(sc.markers[0]);
    shift_pos(&mut sc.start);
    sc.start = Pos::new(0, 0);
}

/// Recompute exact hidden solvability, publish the action budget, assert probe.
fn finalize_p1c_scenario(sc: &mut Scenario, budget_mode: P1cBudgetMode) {
    let inherited_budget = sc.action_budget.saturating_add(2);
    sc.action_budget = P1C_HARD_BUDGET;
    let qualifier = Simulator::new(sc.clone());
    let required = shortest_path(
        &qualifier,
        qualifier.state(),
        sc.hidden_goal(),
        sc.action_budget,
    )
    .unwrap_or_else(|| {
        panic!(
            "P1C transform lost hidden goal: seed={} episode={} split={:?} goal={:?}",
            sc.seed,
            sc.episode_id,
            sc.split,
            sc.hidden_goal()
        )
    })
    .actions
    .len() as u16;
    let normalized = inherited_budget.max(required.saturating_add(REACH_SLACK));
    sc.action_budget = match budget_mode {
        P1cBudgetMode::NormalizedSlack => normalized,
        P1cBudgetMode::GenerousRecoverable => normalized.max(P1C_HARD_BUDGET),
    };
    // Hard candidates are rejection-sampled by the experiment. Avoid the
    // expensive all-goal probe audit on every rejected draw; the experiment
    // exact-checks the accepted scenario, and generator tests cover the
    // structural invariant. Standard P1C retains its per-scenario assertion.
    if budget_mode == P1cBudgetMode::NormalizedSlack {
        assert!(
            p1c_falsification_probe_width(sc) >= 2,
            "P1C transform lacks a safe multi-goal falsification probe: seed={} episode={} split={:?}",
            sc.seed,
            sc.episode_id,
            sc.split
        );
    }
}

/// Maximum number of exact-live goals co-satisfied by a safe falsification probe.
///
/// From the initial public state, each exactly reachable candidate's shortest
/// plan is simulated in full. A plan is rejected if any prefix predicts
/// terminal failure for any exact-live candidate or irreversibly makes any
/// other exact-live goal impossible. Surviving plans contribute the count of
/// exact-live candidates whose public goal predicate holds at the endpoint;
/// this returns the maximum such count.
pub fn p1c_falsification_probe_width(scenario: &Scenario) -> usize {
    let sim = Simulator::new(scenario.clone());
    if sim.is_terminal() {
        return 0;
    }
    let exact_live: Vec<(usize, Vec<Action>)> = scenario
        .candidate_goals
        .iter()
        .enumerate()
        .filter_map(|(i, goal)| {
            shortest_path(&sim, sim.state(), goal, scenario.action_budget)
                .map(|result| (i, result.actions))
        })
        .collect();
    if exact_live.len() < 2 {
        return 0;
    }

    let mut best = 0usize;
    for (_, actions) in &exact_live {
        let mut state = sim.state().clone();
        let mut rejected = false;
        for &action in actions {
            state = sim.transition(&state, action);
            for &(cand_idx, _) in &exact_live {
                let goal = &scenario.candidate_goals[cand_idx];
                if goal_terminal_failure(scenario, &state, goal) {
                    rejected = true;
                    break;
                }
                if !goal_viable_at(&sim, &state, goal) {
                    rejected = true;
                    break;
                }
            }
            if rejected {
                break;
            }
        }
        if rejected {
            continue;
        }
        let satisfied = exact_live
            .iter()
            .filter(|&&(i, _)| goal_satisfied(scenario, &state, &scenario.candidate_goals[i]))
            .count();
        best = best.max(satisfied);
    }
    best
}

fn try_generate(
    seed: u64,
    episode_id: u64,
    split: Split,
    rng: &mut ChaCha8Rng,
) -> Option<Scenario> {
    let (width, height) = match split {
        Split::Train => (7u8, 7u8),
        Split::HeldOutComposition => (8u8, 8u8),
    };

    let mut walls = carve_maze(width, height, rng);
    let start = Pos::new(0, 0);
    walls.remove(&start);

    let free: Vec<Pos> = cells(width, height)
        .into_iter()
        .filter(|p| !walls.contains(p) && *p != start)
        .collect();
    if free.len() < 14 {
        return None;
    }

    let markers = pick_distinct(&free, 3, rng)?;
    let rest: Vec<Pos> = free
        .iter()
        .copied()
        .filter(|p| !markers.contains(p))
        .collect();

    let n_collect = match split {
        Split::Train => 2 + rng.random_range(0..2u8) as usize,
        Split::HeldOutComposition => 3 + rng.random_range(0..2u8) as usize,
    };
    let collectibles = pick_distinct(&rest, n_collect, rng)?;
    let rest: Vec<Pos> = rest
        .iter()
        .copied()
        .filter(|p| !collectibles.contains(p))
        .collect();

    let switches = pick_distinct(&rest, 3, rng)?;
    let rest: Vec<Pos> = rest
        .iter()
        .copied()
        .filter(|p| !switches.contains(p))
        .collect();

    // Always place ≥1 hazard so AvoidHazardReachMarker is well-defined.
    let n_hazards = match split {
        Split::Train => 1 + rng.random_range(0..2u8) as usize,
        Split::HeldOutComposition => 2 + rng.random_range(0..2u8) as usize,
    };
    let mut hazards = pick_distinct(&rest, n_hazards.min(rest.len()).max(1), rng)?;
    let rest: Vec<Pos> = rest
        .iter()
        .copied()
        .filter(|p| !hazards.contains(p))
        .collect();

    let n_pickups = 1 + rng.random_range(0..2u8) as usize;
    let resource_pickups = pick_distinct(&rest, n_pickups.min(rest.len()), rng).unwrap_or_default();
    let rest: Vec<Pos> = rest
        .iter()
        .copied()
        .filter(|p| !resource_pickups.contains(p))
        .collect();

    // Terminal triggers: Train uses one pad; HeldOut uses two for multi-trigger goals.
    let n_triggers = match split {
        Split::Train => 1usize,
        Split::HeldOutComposition => 2usize,
    };
    let terminal_triggers = pick_distinct(&rest, n_triggers.min(rest.len()).max(1), rng)?;

    // Make wrong ReachMarker commitments consequential on avoid-goal episodes:
    // HeldOut places hazard 0 on a short path toward marker 0 while avoid goals
    // target a different marker (cross-index composition below).
    if matches!(split, Split::HeldOutComposition) {
        if let Some(trap) = path_cell_toward(start, markers[0], &walls, width, height) {
            if !markers.contains(&trap)
                && !collectibles.contains(&trap)
                && !switches.contains(&trap)
                && !hazards[1..].contains(&trap)
                && !resource_pickups.contains(&trap)
                && !terminal_triggers.contains(&trap)
                && trap != start
            {
                hazards[0] = trap;
            }
        }
    }

    let initial_resource = match split {
        Split::Train => 2 + rng.random_range(0..2u8),
        Split::HeldOutComposition => 1 + rng.random_range(0..2u8),
    };
    let undo_enabled = rng.random_bool(0.5);
    let action_budget: u16 = match split {
        Split::Train => 48 + u16::from(rng.random_range(0..17u8)),
        Split::HeldOutComposition => 56 + u16::from(rng.random_range(0..25u8)),
    };

    let candidate_goals = build_candidates(
        split,
        &markers,
        &switches,
        &hazards,
        &terminal_triggers,
        initial_resource,
        rng,
    );
    if !has_all_six_families(&candidate_goals) {
        return None;
    }

    let limit = action_budget.saturating_sub(REACH_SLACK).max(16);
    let mut solvable_idxs: Vec<usize> = Vec::new();
    for (i, g) in candidate_goals.iter().enumerate() {
        let draft = Scenario {
            width,
            height,
            walls: walls.clone(),
            markers: markers.clone(),
            collectibles: collectibles.clone(),
            switches: switches.clone(),
            hazards: hazards.clone(),
            resource_pickups: resource_pickups.clone(),
            terminal_triggers: terminal_triggers.clone(),
            start,
            initial_resource,
            action_budget,
            // Solvability uses move-only reachability (Undo is optional sugar).
            undo_enabled: false,
            candidate_goals: candidate_goals.clone(),
            hidden_goal_index: i,
            split,
            seed,
            episode_id,
        };
        if reachable_within(&draft, g, limit) {
            solvable_idxs.push(i);
        }
    }
    if solvable_idxs.is_empty() {
        return None;
    }

    let mut solvable_families = [false; 6];
    for &i in &solvable_idxs {
        solvable_families[family_id(&candidate_goals[i]) as usize] = true;
    }
    if !solvable_families.iter().all(|covered| *covered) {
        return None;
    }

    // Episode IDs assign hidden families round-robin. RNG only chooses between
    // multiple solvable goals within that already-fixed family.
    let hidden_goal_index = pick_hidden_for_family(
        &candidate_goals,
        &solvable_idxs,
        (episode_id % 6) as u8,
        rng,
    );

    Some(Scenario {
        width,
        height,
        walls,
        markers,
        collectibles,
        switches,
        hazards,
        resource_pickups,
        terminal_triggers,
        start,
        initial_resource,
        action_budget,
        undo_enabled,
        candidate_goals,
        hidden_goal_index,
        split,
        seed,
        episode_id,
    })
}

/// Train uses same-index avoid + single trigger; HeldOut uses cross-index avoid
/// and multi-trigger objectives — an unseen mechanic/objective composition.
fn build_candidates(
    split: Split,
    markers: &[Pos],
    switches: &[Pos],
    hazards: &[Pos],
    terminal_triggers: &[Pos],
    initial_resource: u8,
    rng: &mut ChaCha8Rng,
) -> Vec<Goal> {
    let mut goals = Vec::new();
    for i in 0..markers.len().min(3) {
        goals.push(Goal::ReachMarker { marker: i as u8 });
    }
    goals.push(Goal::CollectAll);

    match split {
        Split::Train => {
            let order: Vec<u8> = (0..switches.len() as u8).collect();
            goals.push(Goal::ActivateSwitchesInOrder {
                order: order.clone(),
            });
            if switches.len() >= 3 {
                let mut rotated = order;
                rotated.rotate_left(1);
                goals.push(Goal::ActivateSwitchesInOrder { order: rotated });
            }
            let min_res = initial_resource.max(1);
            goals.push(Goal::PreserveResourceReachMarker {
                marker: 0,
                min_resource: min_res,
            });
            if markers.len() > 1 {
                goals.push(Goal::PreserveResourceReachMarker {
                    marker: 1,
                    min_resource: 1,
                });
            }
            // Same-index avoid pairing (Train-only composition).
            if !hazards.is_empty() && !markers.is_empty() {
                goals.push(Goal::AvoidHazardReachMarker {
                    hazard: 0,
                    marker: 0,
                });
            }
            if !terminal_triggers.is_empty() {
                goals.push(Goal::TriggerTerminal { trigger: 0 });
            }
        }
        Split::HeldOutComposition => {
            // Reverse switch order — Train never emits this order pattern.
            if switches.len() >= 2 {
                let mut rev: Vec<u8> = (0..switches.len() as u8).collect();
                rev.reverse();
                goals.push(Goal::ActivateSwitchesInOrder { order: rev });
            }
            // Stricter resource floor than Train's typical min_resource=1 sibling.
            goals.push(Goal::PreserveResourceReachMarker {
                marker: 0,
                min_resource: initial_resource.saturating_add(1).max(2),
            });
            // Cross-index avoid: hazard i paired with marker j, i != j.
            if !hazards.is_empty() && markers.len() >= 2 {
                goals.push(Goal::AvoidHazardReachMarker {
                    hazard: 0,
                    marker: 1,
                });
            }
            if hazards.len() >= 2 && !markers.is_empty() {
                goals.push(Goal::AvoidHazardReachMarker {
                    hazard: 1,
                    marker: 0,
                });
            } else if !hazards.is_empty() && markers.len() >= 3 {
                goals.push(Goal::AvoidHazardReachMarker {
                    hazard: 0,
                    marker: 2,
                });
            }
            // Multi-trigger objectives (Train only exposes trigger 0).
            for t in 0..terminal_triggers.len().min(2) {
                goals.push(Goal::TriggerTerminal { trigger: t as u8 });
            }
        }
    }

    let n = goals.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        goals.swap(i, j);
    }
    goals
}

fn family_id(goal: &Goal) -> u8 {
    match goal {
        Goal::ReachMarker { .. } => 0,
        Goal::CollectAll => 1,
        Goal::ActivateSwitchesInOrder { .. } => 2,
        Goal::PreserveResourceReachMarker { .. } => 3,
        Goal::AvoidHazardReachMarker { .. } => 4,
        Goal::TriggerTerminal { .. } => 5,
    }
}

fn has_all_six_families(goals: &[Goal]) -> bool {
    let mut seen = [false; 6];
    for g in goals {
        seen[family_id(g) as usize] = true;
    }
    seen.iter().all(|&x| x)
}

fn pick_hidden_for_family(
    candidates: &[Goal],
    solvable: &[usize],
    family: u8,
    rng: &mut ChaCha8Rng,
) -> usize {
    let preferred: Vec<usize> = solvable
        .iter()
        .copied()
        .filter(|&i| family_id(&candidates[i]) == family)
        .collect();
    assert!(
        !preferred.is_empty(),
        "generator must provide a solvable goal for every family"
    );
    preferred[rng.random_range(0..preferred.len())]
}

/// First free cell one Manhattan step from `start` toward `target` (or None).
fn path_cell_toward(
    start: Pos,
    target: Pos,
    walls: &BTreeSet<Pos>,
    width: u8,
    height: u8,
) -> Option<Pos> {
    let dx = (target.x - start.x).signum();
    let dy = (target.y - start.y).signum();
    let candidates = if dx != 0 && dy != 0 {
        vec![
            Pos::new(start.x + dx, start.y),
            Pos::new(start.x, start.y + dy),
        ]
    } else if dx != 0 {
        vec![Pos::new(start.x + dx, start.y)]
    } else if dy != 0 {
        vec![Pos::new(start.x, start.y + dy)]
    } else {
        return None;
    };
    candidates
        .into_iter()
        .find(|p| in_bounds(width, height, *p) && !walls.contains(p) && *p != start && *p != target)
}

fn carve_maze(width: u8, height: u8, rng: &mut ChaCha8Rng) -> BTreeSet<Pos> {
    let mut walls: BTreeSet<Pos> = cells(width, height).into_iter().collect();
    let mut stack = vec![Pos::new(0, 0)];
    walls.remove(&Pos::new(0, 0));
    let dirs = crate::domain::Dir::ALL;
    while let Some(cur) = stack.last().copied() {
        let mut nbrs = Vec::new();
        for d in dirs {
            let (dx, dy) = d.delta();
            if let Some(mid) = cur.checked_add(dx, dy) {
                if let Some(nxt) = mid.checked_add(dx, dy) {
                    if in_bounds(width, height, nxt) && walls.contains(&nxt) {
                        nbrs.push((mid, nxt));
                    }
                }
            }
        }
        if nbrs.is_empty() {
            stack.pop();
            continue;
        }
        let &(mid, nxt) = nbrs.choose(rng).unwrap();
        walls.remove(&mid);
        walls.remove(&nxt);
        stack.push(nxt);
    }
    let extras = (width as usize * height as usize) / 5;
    let all = cells(width, height);
    for _ in 0..extras {
        if let Some(p) = all.choose(rng) {
            walls.remove(p);
        }
    }
    for x in 0..width as i8 {
        walls.remove(&Pos::new(x, 0));
        walls.remove(&Pos::new(x, height as i8 - 1));
    }
    for y in 0..height as i8 {
        walls.remove(&Pos::new(0, y));
        walls.remove(&Pos::new(width as i8 - 1, y));
    }
    walls
}

fn cells(width: u8, height: u8) -> Vec<Pos> {
    let mut v = Vec::with_capacity((width as usize) * (height as usize));
    for y in 0..height as i8 {
        for x in 0..width as i8 {
            v.push(Pos::new(x, y));
        }
    }
    v
}

fn in_bounds(width: u8, height: u8, p: Pos) -> bool {
    p.x >= 0 && p.y >= 0 && (p.x as u8) < width && (p.y as u8) < height
}

fn pick_distinct(pool: &[Pos], n: usize, rng: &mut ChaCha8Rng) -> Option<Vec<Pos>> {
    if n > pool.len() || n == 0 {
        return None;
    }
    let mut idx: Vec<usize> = (0..pool.len()).collect();
    for i in (1..idx.len()).rev() {
        let j = rng.random_range(0..=i);
        idx.swap(i, j);
    }
    Some(idx.into_iter().take(n).map(|i| pool[i]).collect())
}

fn fallback_scenario(seed: u64, episode_id: u64, split: Split, rng: &mut ChaCha8Rng) -> Scenario {
    let width = 5u8;
    let height = 5u8;
    let walls = BTreeSet::from([Pos::new(2, 2)]);
    let markers = vec![Pos::new(4, 4), Pos::new(0, 4), Pos::new(4, 0)];
    let collectibles = vec![Pos::new(1, 0), Pos::new(2, 0)];
    let switches = vec![Pos::new(0, 1), Pos::new(1, 1), Pos::new(2, 1)];
    let hazards = vec![Pos::new(3, 3), Pos::new(1, 3)];
    let resource_pickups = vec![Pos::new(0, 3)];
    let terminal_triggers = vec![Pos::new(4, 2), Pos::new(2, 4)];
    let candidate_goals = match split {
        Split::Train => vec![
            Goal::ReachMarker { marker: 0 },
            Goal::ReachMarker { marker: 1 },
            Goal::ReachMarker { marker: 2 },
            Goal::CollectAll,
            Goal::ActivateSwitchesInOrder {
                order: vec![0, 1, 2],
            },
            Goal::PreserveResourceReachMarker {
                marker: 0,
                min_resource: 1,
            },
            Goal::AvoidHazardReachMarker {
                hazard: 0,
                marker: 0,
            },
            Goal::TriggerTerminal { trigger: 0 },
        ],
        Split::HeldOutComposition => vec![
            Goal::ReachMarker { marker: 0 },
            Goal::ReachMarker { marker: 1 },
            Goal::CollectAll,
            Goal::ActivateSwitchesInOrder {
                order: vec![2, 1, 0],
            },
            Goal::PreserveResourceReachMarker {
                marker: 0,
                min_resource: 2,
            },
            Goal::AvoidHazardReachMarker {
                hazard: 0,
                marker: 1,
            },
            Goal::AvoidHazardReachMarker {
                hazard: 1,
                marker: 0,
            },
            Goal::TriggerTerminal { trigger: 0 },
            Goal::TriggerTerminal { trigger: 1 },
        ],
    };
    let action_budget = 64u16;
    let undo_enabled = true;
    let initial_resource = 2u8;
    let mut solvable = Vec::new();
    for (i, g) in candidate_goals.iter().enumerate() {
        let draft = Scenario {
            width,
            height,
            walls: walls.clone(),
            markers: markers.clone(),
            collectibles: collectibles.clone(),
            switches: switches.clone(),
            hazards: hazards.clone(),
            resource_pickups: resource_pickups.clone(),
            terminal_triggers: terminal_triggers.clone(),
            start: Pos::new(0, 0),
            initial_resource,
            action_budget,
            undo_enabled: false,
            candidate_goals: candidate_goals.clone(),
            hidden_goal_index: i,
            split,
            seed,
            episode_id,
        };
        if reachable_within(&draft, g, action_budget) {
            solvable.push(i);
        }
    }
    assert!(
        !solvable.is_empty(),
        "fallback layout must have a solvable goal"
    );
    assert!(
        has_all_six_families(&candidate_goals),
        "fallback must cover all six families"
    );
    let mut solvable_families = [false; 6];
    for &i in &solvable {
        solvable_families[family_id(&candidate_goals[i]) as usize] = true;
    }
    assert!(
        solvable_families.iter().all(|covered| *covered),
        "fallback must have a solvable goal in all six families"
    );
    let hidden_goal_index =
        pick_hidden_for_family(&candidate_goals, &solvable, (episode_id % 6) as u8, rng);
    Scenario {
        width,
        height,
        walls,
        markers,
        collectibles,
        switches,
        hazards,
        resource_pickups,
        terminal_triggers,
        start: Pos::new(0, 0),
        initial_resource,
        action_budget,
        undo_enabled,
        candidate_goals,
        hidden_goal_index,
        split,
        seed,
        episode_id,
    }
}

/// Structural composition fingerprint used by held-out tests.
pub fn composition_kind(goal: &Goal) -> &'static str {
    match goal {
        Goal::ReachMarker { .. } => "reach",
        Goal::CollectAll => "collect",
        Goal::ActivateSwitchesInOrder { order } => {
            let forward: Vec<u8> = (0..order.len() as u8).collect();
            if *order == forward {
                "switches_forward"
            } else if order.iter().eq(forward.iter().rev()) {
                "switches_reversed"
            } else {
                "switches_rotated"
            }
        }
        Goal::PreserveResourceReachMarker { .. } => "preserve",
        Goal::AvoidHazardReachMarker { hazard, marker } => {
            if hazard == marker {
                "avoid_same_index"
            } else {
                "avoid_cross_index"
            }
        }
        Goal::TriggerTerminal { trigger } => {
            if *trigger == 0 {
                "trigger_0"
            } else {
                "trigger_nonzero"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{legal_actions, Simulator, State};

    #[test]
    fn deterministic_generation() {
        let a = generate(42, 7, Split::Train);
        let b = generate(42, 7, Split::Train);
        assert_eq!(a, b);
        let c = generate(42, 8, Split::Train);
        assert_ne!(a, c);
    }

    #[test]
    fn p1c_generation_is_deterministic_with_safe_falsification_probe() {
        for seed in 1..=2u64 {
            for episode_id in 0..6u64 {
                for split in [Split::Train, Split::HeldOutComposition] {
                    let a = generate_p1c(seed, episode_id, split);
                    let b = generate_p1c(seed, episode_id, split);
                    assert_eq!(a, b);
                    assert!(
                        p1c_falsification_probe_width(&a) >= 2,
                        "missing safe multi-goal probe seed={seed} ep={episode_id} split={split:?}"
                    );
                    assert!(has_all_six_families(&a.candidate_goals));
                    assert!(reachable_within(&a, a.hidden_goal(), a.action_budget));
                }
            }
        }
    }

    #[test]
    fn p1c_marker0_side_branch_is_cheap_multi_goal_probe() {
        use crate::domain::{Action, Dir};

        let sc = generate_p1c(3, 0, Split::Train);
        assert_eq!(sc.start, Pos::new(0, 0));
        assert_eq!(sc.markers[0], Pos::new(0, 1));
        assert!(
            sc.resource_pickups.contains(&sc.markers[0]),
            "marker-0 side branch must carry the resource pickup"
        );
        let sim = Simulator::new(sc.clone());
        let marker = shortest_path(
            &sim,
            sim.state(),
            &Goal::ReachMarker { marker: 0 },
            sc.action_budget,
        )
        .expect("marker 0 path");
        assert_eq!(marker.actions.as_slice(), &[Action::Move(Dir::South)]);
        let mut state = sim.state().clone();
        for &action in &marker.actions {
            state = sim.transition(&state, action);
        }
        let exact_live: Vec<usize> = sc
            .candidate_goals
            .iter()
            .enumerate()
            .filter_map(|(i, goal)| {
                shortest_path(&sim, sim.state(), goal, sc.action_budget).map(|_| i)
            })
            .collect();
        let satisfied = exact_live
            .iter()
            .filter(|&&i| goal_satisfied(&sc, &state, &sc.candidate_goals[i]))
            .count();
        assert!(
            satisfied >= 2,
            "side-branch probe must test ≥2 equally plausible goals, got {satisfied}"
        );
        assert!(p1c_falsification_probe_width(&sc) >= satisfied);
    }

    #[test]
    fn p1c_hard_candidate_is_deterministic() {
        for seed in 1..=2u64 {
            for source_episode_id in 0..6u64 {
                for split in [Split::Train, Split::HeldOutComposition] {
                    let a = generate_p1c_hard_candidate(seed, source_episode_id, split);
                    let b = generate_p1c_hard_candidate(seed, source_episode_id, split);
                    assert_eq!(a, b);
                }
            }
        }
    }

    #[test]
    fn p1c_hard_candidate_probe_is_deeper_than_normal() {
        use crate::domain::{Action, Dir};

        for seed in 1..=2u64 {
            for source_episode_id in 0..4u64 {
                for split in [Split::Train, Split::HeldOutComposition] {
                    let normal = generate_p1c(seed, source_episode_id, split);
                    let hard = generate_p1c_hard_candidate(seed, source_episode_id, split);
                    assert_eq!(hard.start, Pos::new(0, 0));
                    assert_eq!(
                        hard.markers[0],
                        Pos::new(0, hard.height as i8 - 1),
                        "hard marker 0 must sit at the corridor end near the bottom"
                    );
                    assert!(
                        hard.resource_pickups.contains(&hard.markers[0]),
                        "hard corridor end must carry the resource pickup"
                    );
                    // Fork: East into the original map, South into the corridor.
                    let sim = Simulator::new(hard.clone());
                    let east = sim.transition(sim.state(), Action::Move(Dir::East));
                    let south = sim.transition(sim.state(), Action::Move(Dir::South));
                    assert_eq!(east.pos, Pos::new(1, 0));
                    assert_eq!(south.pos, Pos::new(0, 1));
                    for y in 1..hard.height as i8 {
                        assert!(
                            hard.walls.contains(&Pos::new(1, y)),
                            "corridor must stay isolated from the map at y={y}"
                        );
                    }

                    let normal_sim = Simulator::new(normal.clone());
                    let normal_probe = shortest_path(
                        &normal_sim,
                        normal_sim.state(),
                        &Goal::ReachMarker { marker: 0 },
                        normal.action_budget,
                    )
                    .expect("normal marker-0 probe")
                    .actions
                    .len();
                    let hard_probe = shortest_path(
                        &sim,
                        sim.state(),
                        &Goal::ReachMarker { marker: 0 },
                        hard.action_budget,
                    )
                    .expect("hard marker-0 probe")
                    .actions
                    .len();
                    assert!(
                        hard_probe > normal_probe,
                        "hard probe depth {hard_probe} must exceed normal {normal_probe} \
                         seed={seed} ep={source_episode_id} split={split:?}"
                    );
                    assert_eq!(
                        hard_probe,
                        hard.height as usize - 1,
                        "hard probe should be a pure south corridor to the bottom row"
                    );
                    assert!(hard.action_budget >= P1C_HARD_BUDGET);
                }
            }
        }
    }

    #[test]
    fn p1c_hard_candidate_preserves_families_safe_probe_and_hidden_solvability() {
        for seed in 1..=2u64 {
            for source_episode_id in 0..6u64 {
                for split in [Split::Train, Split::HeldOutComposition] {
                    let sc = generate_p1c_hard_candidate(seed, source_episode_id, split);
                    assert!(
                        has_all_six_families(&sc.candidate_goals),
                        "hard candidate lost a goal family: seed={seed} ep={source_episode_id} split={split:?}"
                    );
                    assert!(
                        p1c_falsification_probe_width(&sc) >= 2,
                        "hard candidate lacks safe multi-goal probe: seed={seed} ep={source_episode_id} split={split:?}"
                    );
                    assert!(
                        reachable_within(&sc, sc.hidden_goal(), sc.action_budget),
                        "hard candidate hidden goal unsolvable: seed={seed} ep={source_episode_id} split={split:?} goal={:?}",
                        sc.hidden_goal()
                    );
                }
            }
        }
    }

    #[test]
    fn held_out_metadata() {
        let sc = generate(9, 1, Split::HeldOutComposition);
        assert_eq!(sc.split, Split::HeldOutComposition);
        assert!(sc.markers.len() >= 3);
        assert!(!sc.collectibles.is_empty());
        assert!(sc.switches.len() >= 3);
        assert!(!sc.hazards.is_empty());
        assert!(sc.terminal_triggers.len() >= 2);
        assert!(sc.action_budget >= 40);
        assert!(sc.candidate_goals.len() >= 6);
        assert!(sc.hidden_goal_index < sc.candidate_goals.len());
    }

    #[test]
    fn six_family_coverage_deterministic() {
        for ep in 0..8u64 {
            for split in [Split::Train, Split::HeldOutComposition] {
                let sc = generate(11, ep, split);
                assert!(
                    has_all_six_families(&sc.candidate_goals),
                    "missing family ep={ep} split={split:?} goals={:?}",
                    sc.candidate_goals
                );
            }
        }
    }

    #[test]
    fn train_has_required_features() {
        let sc = generate(1, 0, Split::Train);
        assert_eq!(sc.split, Split::Train);
        assert!(sc.markers.len() >= 3);
        assert!(sc.switches.len() >= 3);
        assert!(!sc.collectibles.is_empty());
        assert_eq!(sc.terminal_triggers.len(), 1);
        assert!(has_all_six_families(&sc.candidate_goals));
        // Train composition: same-index avoid, no cross-index, no nonzero trigger.
        let mut kinds = BTreeSet::new();
        for g in &sc.candidate_goals {
            kinds.insert(composition_kind(g));
        }
        assert!(kinds.contains("avoid_same_index"));
        assert!(!kinds.contains("avoid_cross_index"));
        assert!(kinds.contains("trigger_0"));
        assert!(!kinds.contains("trigger_nonzero"));
        assert!(kinds.contains("switches_forward"));
        assert!(kinds.contains("switches_rotated") || kinds.contains("switches_forward"));
        assert!(!kinds.contains("switches_reversed"));
    }

    #[test]
    fn held_out_composition_structurally_different() {
        let train = generate(5, 0, Split::Train);
        let held = generate(5, 0, Split::HeldOutComposition);
        let train_kinds: BTreeSet<&str> =
            train.candidate_goals.iter().map(composition_kind).collect();
        let held_kinds: BTreeSet<&str> =
            held.candidate_goals.iter().map(composition_kind).collect();
        // HeldOut exposes unseen mechanic/objective compositions.
        assert!(
            held_kinds.contains("avoid_cross_index"),
            "held-out must use cross-index avoid"
        );
        assert!(
            !train_kinds.contains("avoid_cross_index"),
            "train must not use cross-index avoid"
        );
        assert!(
            held_kinds.contains("trigger_nonzero") || held.terminal_triggers.len() >= 2,
            "held-out must expose multi-trigger structure"
        );
        assert!(
            held_kinds.contains("switches_reversed"),
            "held-out must use reversed switch order"
        );
        assert!(
            !train_kinds.contains("switches_reversed"),
            "train must not use reversed switch order"
        );
        // Difference is compositional, not merely grid size.
        assert_ne!(train_kinds, held_kinds);
    }

    #[test]
    fn generated_hidden_goal_solvable() {
        for ep in 0..4u64 {
            for split in [Split::Train, Split::HeldOutComposition] {
                let sc = generate(123, ep, split);
                assert!(
                    reachable_within(&sc, sc.hidden_goal(), sc.action_budget),
                    "hidden goal unreachable ep={ep} split={split:?} goal={:?}",
                    sc.hidden_goal()
                );
                let sim = Simulator::new(sc.clone());
                let json = serde_json::to_string(sim.state()).unwrap();
                assert!(!json.contains("hidden_goal_index"));
                for (i, g) in sc.candidate_goals.iter().enumerate() {
                    if i == sc.hidden_goal_index {
                        continue;
                    }
                    let _ = goal_satisfied(&sc, sim.state(), g);
                }
            }
        }
    }

    #[test]
    fn avoid_goal_wrong_commitment_is_consequential() {
        use crate::domain::terminal_failure;

        let mut found = false;
        for ep in 0..48u64 {
            let sc = generate(77, ep, Split::HeldOutComposition);
            let Goal::AvoidHazardReachMarker { hazard, marker } = sc.hidden_goal().clone() else {
                continue;
            };
            found = true;
            let mut st = State::initial(&sc);
            // After touching the named hazard, avoid-goal success is impossible and
            // the episode is in irreversible terminal failure (hidden goal).
            st.touched_hazards |= 1u32 << hazard;
            st.pos = sc.markers[marker as usize];
            assert!(!goal_satisfied(
                &sc,
                &st,
                &Goal::AvoidHazardReachMarker { hazard, marker }
            ));
            assert!(terminal_failure(&sc, &st));
            // Competing ReachMarker of the same cell would look satisfied, but the
            // hidden avoid goal remains failed — wrong commitment is consequential.
            assert!(goal_satisfied(&sc, &st, &Goal::ReachMarker { marker }));
            break;
        }
        assert!(found, "expected an avoid-hazard hidden goal in sample");
    }

    #[test]
    fn hidden_choice_deterministic() {
        let a = generate(99, 3, Split::Train);
        let b = generate(99, 3, Split::Train);
        assert_eq!(a.hidden_goal_index, b.hidden_goal_index);
        assert_eq!(a.candidate_goals, b.candidate_goals);
    }

    #[test]
    fn legal_actions_export_used() {
        let sc = generate(2, 2, Split::Train);
        let acts = legal_actions(&sc);
        assert!(acts.len() >= 4);
        let _ = State::initial(&sc);
    }

    #[test]
    fn family_coverage_across_episodes() {
        for split in [Split::Train, Split::HeldOutComposition] {
            let mut seen = [false; 6];
            for ep in 0..6u64 {
                let sc = generate(3, ep, split);
                seen[family_id(sc.hidden_goal()) as usize] = true;
            }
            assert!(
                seen.iter().all(|covered| *covered),
                "round-robin assignment must cover all families for {split:?}: {seen:?}"
            );
        }
    }
}
