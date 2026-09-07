//! Independent synthetic control task. Oracles below are training/evaluation only.
//! The learned model receives only `Inputs`, never `Maze` or its control mapping.

use super::{
    ACTIONS, FRAME_SIDE, META_DIM, PALETTE, PATCH_COUNT, PATCH_PIXELS, PATCH_SIDE, SUPPORT_STEPS,
    TOKENS,
};
use anyhow::{bail, ensure, Result};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub const GRID: usize = FRAME_SIDE / PATCH_SIDE;
pub const SCHEMA: &str = "permuted-controls-maze-v1";
pub const DISCOUNT: f32 = 0.95;
const DIRECTIONS: [(isize, isize); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

pub type Frame = Vec<u8>;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Split {
    Train,
    HeldOut,
}

#[derive(Clone, Debug)]
pub struct Transition {
    pub before: Frame,
    pub action: usize,
    pub after: Frame,
}

#[derive(Clone, Debug)]
pub struct Inputs {
    /// [TOKENS, PATCH_PIXELS], raw categorical pixels in patch order.
    pub patches: Vec<u32>,
    /// [TOKENS, META_DIM], public coordinates, ordering and observed actions.
    pub metadata: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Maze {
    walls: [bool; GRID * GRID],
    agent: usize,
    goal: usize,
    controls: [usize; ACTIONS],
    done: bool,
}

#[derive(Clone, Debug)]
pub struct Episode {
    pub support: Vec<Transition>,
    pub maze: Maze,
    pub permutation_id: usize,
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub inputs: Inputs,
    /// [ACTIONS, PATCH_COUNT, PATCH_PIXELS], mutually exclusive next states.
    pub next: Vec<u32>,
    pub current: Vec<u32>,
    pub rewards: [f32; ACTIONS],
    /// Uniform mass on every optimal action; avoids arbitrary tie labels.
    pub policy: [f32; ACTIONS],
    pub value: f32,
}

pub fn permutations() -> Vec<[usize; ACTIONS]> {
    let mut result = Vec::new();
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let p = [a, b, c, d];
                    if (0..4).all(|i| (i + 1..4).all(|j| p[i] != p[j])) {
                        result.push(p);
                    }
                }
            }
        }
    }
    result
}

pub fn permutation_ids(split: Split) -> Vec<usize> {
    // Eight balanced permutations: every action/direction pair occurs twice.
    // The other sixteen contain each pair four times. Hold out combinations,
    // without withholding a primitive direction or introducing label imbalance.
    const HELD_OUT: [usize; 8] = [0, 5, 7, 9, 14, 16, 18, 23];
    (0..24)
        .filter(|id| HELD_OUT.contains(id) == (split == Split::HeldOut))
        .collect()
}

fn neighbor(cell: usize, direction: usize) -> Option<usize> {
    let (dx, dy) = DIRECTIONS[direction];
    let x = (cell % GRID).checked_add_signed(dx)?;
    let y = (cell / GRID).checked_add_signed(dy)?;
    (x < GRID && y < GRID).then_some(y * GRID + x)
}

fn distances(walls: &[bool; GRID * GRID], goal: usize) -> [usize; GRID * GRID] {
    let mut distance = [usize::MAX; GRID * GRID];
    distance[goal] = 0;
    let mut queue = VecDeque::from([goal]);
    while let Some(cell) = queue.pop_front() {
        for d in 0..4 {
            if let Some(next) = neighbor(cell, d) {
                if !walls[next] && distance[next] == usize::MAX {
                    distance[next] = distance[cell] + 1;
                    queue.push_back(next);
                }
            }
        }
    }
    distance
}

fn render_cells(cells: &[u8; GRID * GRID]) -> Frame {
    (0..FRAME_SIDE * FRAME_SIDE)
        .map(|i| cells[(i / FRAME_SIDE / PATCH_SIDE) * GRID + (i % FRAME_SIDE / PATCH_SIDE)])
        .collect()
}

impl Maze {
    pub fn render(&self) -> Frame {
        let mut cells = self.walls.map(u8::from);
        cells[self.goal] = 3;
        cells[self.agent] = if self.done { 4 } else { 2 };
        render_cells(&cells)
    }

    pub fn done(&self) -> bool {
        self.done
    }

    pub fn step(&mut self, action: usize) -> Result<bool> {
        ensure!(action < ACTIONS, "illegal synthetic action");
        ensure!(!self.done, "cannot step a completed synthetic maze");
        if let Some(next) = neighbor(self.agent, self.controls[action]) {
            if !self.walls[next] {
                self.agent = next;
            }
        }
        self.done = self.agent == self.goal;
        Ok(self.done)
    }
}

fn calibration(controls: [usize; ACTIONS], seed: u64, id: u64) -> Result<Vec<Transition>> {
    // Independent of query generation and its distance regime. Randomize both
    // where the evidence appears and which three distinct actions reveal it.
    let mut rng =
        ChaCha8Rng::seed_from_u64(seed ^ id.wrapping_mul(0x94d049bb133111eb) ^ 0x43414c494252);
    let mut actions = [0, 1, 2, 3];
    actions.shuffle(&mut rng);
    let mut maze = Maze {
        walls: [false; GRID * GRID],
        agent: rng.random_range(2..6) * GRID + rng.random_range(2..6),
        goal: GRID * GRID - 1,
        controls,
        done: false,
    };
    let mut support = Vec::new();
    for action in actions.into_iter().take(SUPPORT_STEPS) {
        let before = maze.render();
        maze.step(action)?;
        let after = maze.render();
        ensure!(
            before != after && !maze.done,
            "calibration must reveal a nonterminal movement"
        );
        support.push(Transition {
            before,
            action,
            after,
        });
    }
    Ok(support)
}

pub fn episode(
    seed: u64,
    id: u64,
    split: Split,
    min_distance: usize,
    max_distance: usize,
) -> Result<Episode> {
    let ids = permutation_ids(split);
    let mut rng =
        ChaCha8Rng::seed_from_u64(seed ^ id.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x4d415050494e47);
    let permutation_id = ids[rng.random_range(0..ids.len())];
    episode_with_permutation(seed, id, permutation_id, min_distance, max_distance)
}

/// Same seed/id gives an identical query maze for every control permutation.
pub fn episode_with_permutation(
    seed: u64,
    id: u64,
    permutation_id: usize,
    min_distance: usize,
    max_distance: usize,
) -> Result<Episode> {
    ensure!(
        permutation_id < 24 && min_distance > 0 && min_distance <= max_distance,
        "invalid synthetic task request"
    );
    let controls = permutations()[permutation_id];
    let mut rng =
        ChaCha8Rng::seed_from_u64(seed ^ id.wrapping_mul(0xd1b54a32d192ed03) ^ 0x4c41594f5554);
    for _ in 0..256 {
        let walls = std::array::from_fn(|i| {
            let x = i % GRID;
            let y = i / GRID;
            x == 0 || y == 0 || x + 1 == GRID || y + 1 == GRID || rng.random_bool(0.18)
        });
        let free: Vec<_> = (0..GRID * GRID).filter(|&i| !walls[i]).collect();
        if free.len() < 2 {
            continue;
        }
        let goal = free[rng.random_range(0..free.len())];
        let distance = distances(&walls, goal);
        let starts: Vec<_> = free
            .into_iter()
            .filter(|&i| (min_distance..=max_distance).contains(&distance[i]))
            .collect();
        if starts.is_empty() {
            continue;
        }
        let agent = starts[rng.random_range(0..starts.len())];
        return Ok(Episode {
            support: calibration(controls, seed, id)?,
            maze: Maze {
                walls,
                agent,
                goal,
                controls,
                done: false,
            },
            permutation_id,
        });
    }
    bail!("could not generate a connected bounded maze")
}

pub fn patchify(frame: &[u8]) -> Result<Vec<u32>> {
    ensure!(
        frame.len() == FRAME_SIDE * FRAME_SIDE && frame.iter().all(|&x| usize::from(x) < PALETTE),
        "invalid frame"
    );
    let mut result = Vec::with_capacity(frame.len());
    for patch in 0..PATCH_COUNT {
        for pixel in 0..PATCH_PIXELS {
            let x = (patch % GRID) * PATCH_SIDE + pixel % PATCH_SIDE;
            let y = (patch / GRID) * PATCH_SIDE + pixel / PATCH_SIDE;
            result.push(u32::from(frame[y * FRAME_SIDE + x]));
        }
    }
    Ok(result)
}

pub fn unpatchify(patches: &[u32]) -> Result<Frame> {
    ensure!(
        patches.len() == PATCH_COUNT * PATCH_PIXELS
            && patches.iter().all(|&x| (x as usize) < PALETTE),
        "invalid patch output"
    );
    let mut frame = vec![0; FRAME_SIDE * FRAME_SIDE];
    for patch in 0..PATCH_COUNT {
        for pixel in 0..PATCH_PIXELS {
            let x = (patch % GRID) * PATCH_SIDE + pixel % PATCH_SIDE;
            let y = (patch / GRID) * PATCH_SIDE + pixel / PATCH_SIDE;
            frame[y * FRAME_SIDE + x] = patches[patch * PATCH_PIXELS + pixel] as u8;
        }
    }
    Ok(frame)
}

pub fn inputs(support: &[Transition], current: &[u8]) -> Result<Inputs> {
    ensure!(
        support.len() == SUPPORT_STEPS,
        "expected three factual calibration transitions"
    );
    let mut patches = Vec::with_capacity(TOKENS * PATCH_PIXELS);
    let mut metadata = Vec::with_capacity(TOKENS * META_DIM);
    for (index, step) in support.iter().enumerate() {
        ensure!(step.action < ACTIONS, "invalid observed calibration action");
        for (role, frame) in [(0, &step.before), (1, &step.after)] {
            patches.extend(patchify(frame)?);
            append_metadata(
                &mut metadata,
                role,
                Some(step.action),
                index as f32 / SUPPORT_STEPS as f32,
            );
        }
    }
    patches.extend(patchify(current)?);
    append_metadata(&mut metadata, 2, None, 1.0);
    Ok(Inputs { patches, metadata })
}

fn append_metadata(out: &mut Vec<f32>, role: usize, action: Option<usize>, index: f32) {
    for patch in 0..PATCH_COUNT {
        let mut row = [0.0; META_DIM];
        row[role] = 1.0;
        if let Some(action) = action {
            row[3 + action] = 1.0;
        }
        row[7] = (patch % GRID) as f32 / (GRID - 1) as f32;
        row[8] = (patch / GRID) as f32 / (GRID - 1) as f32;
        row[9] = index;
        out.extend(row);
    }
}

/// Synthetic oracle parser. This is never called by the neural model or learned search.
fn observed_cells(frame: &[u8]) -> Result<[u8; GRID * GRID]> {
    let packed = patchify(frame)?;
    let mut cells = [0; GRID * GRID];
    for (cell, pixels) in cells.iter_mut().zip(packed.chunks_exact(PATCH_PIXELS)) {
        ensure!(
            pixels.iter().all(|x| *x == pixels[0]),
            "synthetic oracle expects uniform tiles"
        );
        *cell = pixels[0] as u8;
    }
    Ok(cells)
}

pub fn inferred_controls(support: &[Transition]) -> Result<[usize; ACTIONS]> {
    ensure!(support.len() == SUPPORT_STEPS, "incomplete observed prefix");
    let mut controls = [usize::MAX; ACTIONS];
    for transition in support {
        ensure!(
            transition.action < ACTIONS && controls[transition.action] == usize::MAX,
            "duplicate/invalid calibration action"
        );
        let before = observed_cells(&transition.before)?;
        let after = observed_cells(&transition.after)?;
        let start = before
            .iter()
            .position(|&x| x == 2)
            .ok_or_else(|| anyhow::anyhow!("no visible calibration cursor"))?;
        let end = after
            .iter()
            .position(|&x| x == 2)
            .ok_or_else(|| anyhow::anyhow!("no visible successor cursor"))?;
        let direction = (0..4)
            .find(|&d| neighbor(start, d) == Some(end))
            .ok_or_else(|| anyhow::anyhow!("non-identifying calibration move"))?;
        ensure!(
            !controls.contains(&direction),
            "observations contradict permutation assumption"
        );
        controls[transition.action] = direction;
    }
    let missing_direction = (0..4).find(|d| !controls.contains(d)).unwrap();
    let missing_action = controls.iter().position(|&d| d == usize::MAX).unwrap();
    controls[missing_action] = missing_direction;
    Ok(controls)
}

/// Uses only visible support/current frames and the known synthetic task specification.
pub fn oracle(support: &[Transition], current: &[u8]) -> Result<([f32; ACTIONS], usize)> {
    let cells = observed_cells(current)?;
    if cells.contains(&4) {
        return Ok(([0.25; ACTIONS], 0));
    }
    let start = cells
        .iter()
        .position(|&x| x == 2)
        .ok_or_else(|| anyhow::anyhow!("no visible cursor"))?;
    let goal = cells
        .iter()
        .position(|&x| x == 3)
        .ok_or_else(|| anyhow::anyhow!("no visible goal"))?;
    let walls = cells.map(|x| x == 1);
    let distance = distances(&walls, goal);
    ensure!(
        distance[start] != usize::MAX && distance[start] > 0,
        "unreachable synthetic goal"
    );
    let controls = inferred_controls(support)?;
    let mut policy = [0.0; ACTIONS];
    for action in 0..ACTIONS {
        if let Some(next) = neighbor(start, controls[action]) {
            if !walls[next] && distance[next].checked_add(1) == Some(distance[start]) {
                policy[action] = 1.0;
            }
        }
    }
    let mass: f32 = policy.iter().sum();
    ensure!(mass > 0.0, "oracle has no optimal action");
    policy.iter_mut().for_each(|p| *p /= mass);
    Ok((policy, distance[start]))
}

pub fn sample(episode: &Episode) -> Result<Sample> {
    let frame = episode.maze.render();
    let (policy, distance) = oracle(&episode.support, &frame)?;
    ensure!(distance > 0, "training sample is already terminal");
    let mut next = Vec::with_capacity(ACTIONS * FRAME_SIDE * FRAME_SIDE);
    let mut rewards = [0.0; ACTIONS];
    for (action, reward) in rewards.iter_mut().enumerate() {
        let mut branch = episode.maze.clone();
        *reward = f32::from(branch.step(action)?);
        next.extend(patchify(&branch.render())?);
    }
    Ok(Sample {
        inputs: inputs(&episode.support, &frame)?,
        next,
        current: patchify(&frame)?,
        rewards,
        policy,
        value: DISCOUNT.powi((distance - 1) as i32),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_order_preserves_every_pixel() -> Result<()> {
        let frame: Vec<_> = (0..4096).map(|i| ((i * 7 + i / 64) % 16) as u8).collect();
        assert_eq!(frame, unpatchify(&patchify(&frame)?)?);
        Ok(())
    }

    #[test]
    fn observed_prefix_identifies_all_twenty_four_rules() -> Result<()> {
        for (id, controls) in permutations().into_iter().enumerate() {
            let ep = episode_with_permutation(71, 9, id, 2, 10)?;
            assert_eq!(inferred_controls(&ep.support)?, controls);
            for pair in ep.support.windows(2) {
                assert_eq!(pair[0].after, pair[1].before);
            }
        }
        Ok(())
    }

    #[test]
    fn calibration_randomness_does_not_encode_query_distance_regime() -> Result<()> {
        let mut omitted = [0; ACTIONS];
        let mut first_positions = std::collections::BTreeSet::new();
        for id in 0..64 {
            let near = episode_with_permutation(93, id, 7, 1, 1)?;
            let far = episode_with_permutation(93, id, 7, 2, 10)?;
            for (a, b) in near.support.iter().zip(&far.support) {
                assert_eq!(a.action, b.action);
                assert_eq!(a.before, b.before);
                assert_eq!(a.after, b.after);
            }
            let missing = (0..ACTIONS)
                .find(|a| !near.support.iter().any(|t| t.action == *a))
                .unwrap();
            omitted[missing] += 1;
            first_positions.insert(near.support[0].before.iter().position(|&p| p == 2).unwrap());
        }
        assert!(omitted.iter().all(|&n| n > 0));
        assert!(first_positions.len() > 8);
        Ok(())
    }

    #[test]
    fn rules_change_labels_but_not_current_pixels() -> Result<()> {
        let mut reference = None;
        let mut action_mass = [0.0; 4];
        for id in 0..24 {
            let ep = episode_with_permutation(133, 1, id, 2, 10)?;
            let frame = ep.maze.render();
            if let Some(reference) = &reference {
                assert_eq!(&frame, reference);
            } else {
                reference = Some(frame.clone());
            }
            let (policy, _) = oracle(&ep.support, &frame)?;
            for a in 0..4 {
                action_mass[a] += policy[a];
            }
        }
        for mass in action_mass {
            assert!((mass - 6.0).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn oracle_closes_loop_and_supervised_branches_are_real_successors() -> Result<()> {
        for split in [Split::Train, Split::HeldOut] {
            for id in 0..32 {
                let mut ep = episode(45, id, split, 1, 12)?;
                let initial = sample(&ep)?;
                assert_eq!(initial.inputs.patches.len(), TOKENS * PATCH_PIXELS);
                assert_eq!(initial.inputs.metadata.len(), TOKENS * META_DIM);
                let (_, distance) = oracle(&ep.support, &ep.maze.render())?;
                for step in 0..distance {
                    let frame = ep.maze.render();
                    let (policy, remaining) = oracle(&ep.support, &frame)?;
                    assert_eq!(remaining, distance - step);
                    let action = policy.iter().position(|&p| p > 0.0).unwrap();
                    let labels = sample(&ep)?;
                    ep.maze.step(action)?;
                    assert_eq!(
                        patchify(&ep.maze.render())?,
                        labels.next[action * 4096..(action + 1) * 4096]
                    );
                }
                assert!(ep.maze.done());
            }
        }
        Ok(())
    }

    #[test]
    fn rule_splits_are_disjoint_and_exhaustive() {
        let train = permutation_ids(Split::Train);
        let test = permutation_ids(Split::HeldOut);
        assert_eq!((train.len(), test.len()), (16, 8));
        assert!(train.iter().all(|x| !test.contains(x)));
        let rules = permutations();
        for ids in [train, test] {
            for action in 0..ACTIONS {
                for direction in 0..ACTIONS {
                    assert_eq!(
                        ids.iter()
                            .filter(|&&id| rules[id][action] == direction)
                            .count(),
                        ids.len() / ACTIONS
                    );
                }
            }
        }
    }
}
