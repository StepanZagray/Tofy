"""Engine parity, mask semantics, level progression, reset/game-over semantics."""

from __future__ import annotations

import numpy as np
from arcengine import ActionInput, GameAction, GameState

from tofy_arc3.synth.game import display_coords, make_game_class
from tofy_arc3.synth.generate import policy_seed_for
from tofy_arc3.synth.levelgen import generate_pair
from tofy_arc3.synth.rollout import epsilon
from tofy_arc3.synth.rules import (
    HAZARD,
    HiddenRule,
    LevelSpec,
    GameSpec,
    actions_from_mask,
    initial_state,
    mask_from_actions,
    render_grid,
)
from tofy_arc3.synth.solver import solve

BASE, TWIN = generate_pair(11, 0, policy_seed_for(11, 0))


def _scaled(grid, W, rows, bg):
    scale = min(64 // W, 64 // rows)
    xo, yo = (64 - W * scale) // 2, (64 - rows * scale) // 2
    out = np.full((64, 64), bg, dtype=np.uint8)
    g = np.repeat(np.repeat(np.asarray(grid, dtype=np.uint8), scale, axis=0), scale, axis=1)
    out[yo : yo + g.shape[0], xo : xo + g.shape[1]] = g
    return out


def _reset(game):
    return np.asarray(game.perform_action(ActionInput(id=GameAction.RESET), raw=True).frame[-1]).astype(np.uint8)


def test_engine_frame_matches_simulator_render_with_letterbox():
    game = make_game_class(BASE)(seed=0)
    f = _reset(game)
    lv = BASE.levels[0]
    expected = _scaled(render_grid(lv, BASE.rule, initial_state(lv), lv.budget), lv.width, lv.rows, BASE.rule.background)
    assert f.shape == (64, 64) and f.dtype == np.uint8
    assert np.array_equal(f, expected)
    assert 0 <= BASE.rule.background <= 15


def test_mask_bits():
    assert mask_from_actions([1, 2, 3, 4]) == 0b00011111
    assert mask_from_actions([1, 2, 3, 4, 5, 6]) == 0b01111111
    assert actions_from_mask(0b01011111) == [1, 2, 3, 4, 6]
    assert BASE.mask & 1  # RESET always available
    assert set(actions_from_mask(BASE.mask)) == set(BASE.available_actions)


def test_unavailable_action_is_a_noop_without_energy_cost():
    rule = HiddenRule(("up", "down", "left", "right"), "noop", "noop", "reach_exit", tuple(range(13)))
    W, H = 6, 4
    cells = [0] * (W * (H + 1))
    for c in range(W):
        cells[W + c] = 1
        cells[H * W + c] = 1
    for r in range(1, H + 1):
        cells[r * W] = 1
        cells[r * W + W - 1] = 1
    cells[2 * W + 4] = 3  # exit
    lv = LevelSpec(W, H, tuple(cells), player=2 * W + 1, budget=6, tutorial=True)
    spec = GameSpec("tsyn-deadbeef", rule, (lv,), (1, 2, 3, 4))
    game = make_game_class(spec)(seed=0)
    f0 = _reset(game)
    fd = game.perform_action(ActionInput(id=GameAction.ACTION5), raw=True)
    assert np.array_equal(np.asarray(fd.frame[-1]), f0)
    assert game.energy == 6
    fd = game.perform_action(ActionInput(id=GameAction.ACTION4), raw=True)  # right
    assert game.energy == 5 and game.sim_state.player == 2 * W + 2
    assert not np.array_equal(np.asarray(fd.frame[-1]), f0)


def test_solver_plan_completes_levels_through_engine():
    game = make_game_class(BASE)(seed=0)
    _reset(game)
    for li, lv in enumerate(BASE.levels):
        assert game.level_index == li
        plan = solve(lv, BASE.rule, BASE.available_actions, initial_state(lv))
        assert plan is not None and len(plan) <= lv.budget
        for act in plan:
            if act.click is not None:
                x, y = display_coords(game, act.click)
                ai = ActionInput(id=GameAction.ACTION6, data={"x": x, "y": y})
            else:
                ai = ActionInput(id=GameAction.from_id(act.action_id))
            fd = game.perform_action(ai, raw=True)
        assert fd.levels_completed == li + 1
        if li + 1 < len(BASE.levels):
            # The action that completes a level yields two frames; the last is the next level's initial frame.
            assert len(fd.frame) == 2
            nxt = BASE.levels[li + 1]
            expected = _scaled(render_grid(nxt, BASE.rule, initial_state(nxt), nxt.budget), nxt.width, nxt.rows, BASE.rule.background)
            assert np.array_equal(np.asarray(fd.frame[-1]).astype(np.uint8), expected)
        else:
            assert fd.state == GameState.WIN


def test_hazard_causes_game_over_and_reset_restores_level():
    rule = HiddenRule(("up", "down", "left", "right"), "noop", "noop", "reach_exit", tuple(range(13)))
    W, H = 6, 4
    cells = [0] * (W * (H + 1))
    for c in range(W):
        cells[W + c] = 1
        cells[H * W + c] = 1
    for r in range(1, H + 1):
        cells[r * W] = 1
        cells[r * W + W - 1] = 1
    cells[2 * W + 2] = HAZARD
    cells[3 * W + 4] = 3
    lv = LevelSpec(W, H, tuple(cells), player=2 * W + 1, budget=9, tutorial=True)
    spec = GameSpec("tsyn-00000001", rule, (lv,), (1, 2, 3, 4))
    game = make_game_class(spec)(seed=0)
    f0 = _reset(game)
    fd = game.perform_action(ActionInput(id=GameAction.ACTION4), raw=True)
    assert fd.state == GameState.GAME_OVER
    assert len(fd.frame) == 1
    fd = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    assert fd.state == GameState.NOT_FINISHED
    assert np.array_equal(np.asarray(fd.frame[-1]).astype(np.uint8), f0)
    assert game.energy == 9


def test_energy_exhaustion_is_game_over():
    rule = HiddenRule(("up", "down", "left", "right"), "noop", "noop", "reach_exit", tuple(range(13)))
    W, H = 6, 4
    cells = [0] * (W * (H + 1))
    for c in range(W):
        cells[W + c] = 1
        cells[H * W + c] = 1
    for r in range(1, H + 1):
        cells[r * W] = 1
        cells[r * W + W - 1] = 1
    cells[3 * W + 4] = 3
    lv = LevelSpec(W, H, tuple(cells), player=2 * W + 1, budget=2, tutorial=True)
    spec = GameSpec("tsyn-00000002", rule, (lv,), (1, 2, 3, 4))
    game = make_game_class(spec)(seed=0)
    _reset(game)
    fd = game.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)  # blocked by wall, still costs energy
    assert fd.state == GameState.NOT_FINISHED and game.energy == 1
    fd = game.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
    assert fd.state == GameState.GAME_OVER


def test_epsilon_schedule_endpoints():
    assert epsilon(0, 100) == 1.0
    assert abs(epsilon(99, 100) - 0.2) < 1e-12
    assert epsilon(0, 1) == 1.0
    assert epsilon(50, 100) > epsilon(51, 100)
