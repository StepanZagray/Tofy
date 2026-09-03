"""Learning-history rollouts (ADR 0005 §2.2) through the ARCEngine game loop.

Policy: a forced-random shared prefix (length ``SHARED_PREFIX_LEN``) followed
by an epsilon mixture of uniform-random available actions and the BFS solver,
``epsilon`` decaying linearly from 1.0 at the first transition to 0.2 at the
last.  When the game is GAME_OVER or WIN the next action is RESET.  Frames
are the 64x64 uint8 arrays ARCEngine renders (last frame of the action's
frame list).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from arcengine import ActionInput, FrameDataRaw, GameAction, GameState

from .game import SynthGame, display_coords, make_game_class
from .levelgen import SHARED_PREFIX_LEN, shared_prefix_actions
from .rules import GameSpec, split_rule_id
from .solver import SolverAction, solve

EPS_START = 1.0
EPS_END = 0.2


@dataclass
class Transition:
    frame: np.ndarray  # uint8 [64,64]
    next_frame: np.ndarray  # uint8 [64,64]
    action: tuple[int, int, int]  # id, x, y (255 = none)
    available_actions: int
    episode: int
    level: int
    transition_index: int
    rule_id: int
    level_completed: bool
    game_over: bool


def epsilon(t: int, steps: int) -> float:
    if steps <= 1:
        return EPS_START
    return EPS_START - (EPS_START - EPS_END) * (t / (steps - 1))


def _frame_of(fd: FrameDataRaw) -> np.ndarray:
    if not fd.frame:
        raise RuntimeError("engine returned an empty frame list")
    arr = np.asarray(fd.frame[-1])
    if arr.shape != (64, 64):
        raise RuntimeError(f"unexpected frame shape {arr.shape}")
    if arr.min() < 0 or arr.max() > 15:
        raise RuntimeError("frame values outside 0..15")
    return arr.astype(np.uint8)


def _perform(game: SynthGame, action: tuple[int, int, int]) -> FrameDataRaw:
    aid, x, y = action
    if aid == 6:
        ai = ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y)})
    else:
        ai = ActionInput(id=GameAction.from_id(aid))
    fd = game.perform_action(ai, raw=True)
    if not isinstance(fd, FrameDataRaw):
        raise RuntimeError("engine returned non-raw frame data (action after GAME_OVER/WIN?)")
    return fd


class EpsilonSolverPolicy:
    def __init__(self, spec: GameSpec, policy_seed: int, steps: int, prefix_len: int = SHARED_PREFIX_LEN) -> None:
        self.spec = spec
        self.steps = steps
        self.rng = random.Random(f"tofy-synth-policy:{policy_seed}")
        self.prefix = shared_prefix_actions(policy_seed, spec.available_actions, prefix_len)
        self.prefix_pos = 0
        self.plan: list[SolverAction] = []
        self.plan_level = -1
        self.expected_state = None

    def random_action(self) -> tuple[int, int, int]:
        a = self.rng.choice(self.spec.available_actions)
        if a == 6:
            return (a, self.rng.randrange(64), self.rng.randrange(64))
        return (a, 255, 255)

    def solver_action(self, game: SynthGame) -> tuple[int, int, int]:
        state = game.sim_state
        if self.plan_level != game.level_index or self.expected_state != state or not self.plan:
            plan = solve(game.level_spec, self.spec.rule, self.spec.available_actions, state)
            if plan is None or len(plan) > game.energy:
                self.plan = []
                return (0, 255, 255)  # RESET: level is not completable from here
            self.plan = plan
            self.plan_level = game.level_index
        act = self.plan.pop(0)
        from .rules import apply  # local import to keep module import graph simple

        self.expected_state, _ = apply(game.level_spec, self.spec.rule, state, act.action_id, act.click)
        if act.click is not None:
            x, y = display_coords(game, act.click)
            return (act.action_id, x, y)
        return (act.action_id, 255, 255)

    def choose(self, game: SynthGame, t: int) -> tuple[int, int, int]:
        if game._state in (GameState.GAME_OVER, GameState.WIN):
            return (0, 255, 255)
        if self.prefix_pos < len(self.prefix):
            a = self.prefix[self.prefix_pos]
            self.prefix_pos += 1
            return a
        if self.rng.random() < epsilon(t, self.steps):
            return self.random_action()
        return self.solver_action(game)


def rollout_episode(spec: GameSpec, episode_id: int, policy_seed: int, steps: int) -> list[Transition]:
    """Play ``steps`` transitions of ``spec`` and return them chronologically."""
    game_cls = make_game_class(spec)
    game = game_cls(seed=0)
    fd = _perform(game, (0, 255, 255))  # initial RESET (observation, not a row)
    frame = _frame_of(fd)
    lo, hi = split_rule_id(spec.rule.rule_id())
    rule_id = spec.rule.rule_id()
    mask = spec.mask
    policy = EpsilonSolverPolicy(spec, policy_seed, steps)
    rows: list[Transition] = []
    for t in range(steps):
        level = game.level_index
        prev_completed = game._score
        action = policy.choose(game, t)
        fd = _perform(game, action)
        nxt = _frame_of(fd)
        completed = fd.levels_completed == prev_completed + 1
        rows.append(
            Transition(
                frame=frame,
                next_frame=nxt,
                action=action,
                available_actions=mask,
                episode=episode_id,
                level=level,
                transition_index=t,
                rule_id=rule_id,
                level_completed=completed,
                game_over=fd.state == GameState.GAME_OVER,
            )
        )
        frame = nxt
    assert (lo | (hi << 32)) == rule_id
    return rows
