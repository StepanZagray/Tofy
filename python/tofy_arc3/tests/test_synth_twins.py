"""Twin construction and mutual exclusivity (ADR 0005 §2.3)."""

from __future__ import annotations

import numpy as np
import pytest
from arcengine import ActionInput, GameAction

from tofy_arc3.synth.game import make_game_class
from tofy_arc3.synth.generate import policy_seed_for
from tofy_arc3.synth.levelgen import MIN_SOLUTION, generate_pair, shared_prefix_actions, twins_diverge_in_prefix
from tofy_arc3.synth.public_games import PUBLIC_GAME_IDS
from tofy_arc3.synth.rollout import rollout_episode
from tofy_arc3.synth.rules import initial_state, render_grid
from tofy_arc3.synth.solver import solve

PAIRS = [generate_pair(7, i, policy_seed_for(7, i)) for i in range(3)]


def _engine_frame0(spec):
    game = make_game_class(spec)(seed=0)
    fd = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    return np.asarray(fd.frame[-1]).astype(np.uint8)


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p[0].game_id)
def test_twin_shares_level0_frame_but_not_rule(pair):
    base, twin = pair
    assert twin.twin_of == base.game_id
    assert base.rule.rule_id() != twin.rule.rule_id()
    assert base.game_id != twin.game_id
    assert base.available_actions == twin.available_actions
    # Simulator render and engine render agree and are byte-identical across twins.
    g0 = render_grid(base.levels[0], base.rule, initial_state(base.levels[0]), base.levels[0].budget)
    g1 = render_grid(twin.levels[0], twin.rule, initial_state(twin.levels[0]), twin.levels[0].budget)
    assert g0 == g1
    assert np.array_equal(_engine_frame0(base), _engine_frame0(twin))


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p[0].game_id)
def test_twins_diverge_within_shared_prefix(pair):
    base, twin = pair
    prefix = shared_prefix_actions(policy_seed_for(7, PAIRS.index(pair)), base.available_actions)
    assert twins_diverge_in_prefix(base, twin, prefix)


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: p[0].game_id)
def test_every_level_is_solvable_within_budget(pair):
    for spec in pair:
        assert len(spec.levels) >= 2
        for li, lv in enumerate(spec.levels):
            plan = solve(lv, spec.rule, spec.available_actions, initial_state(lv))
            assert plan is not None, (spec.game_id, li)
            assert len(plan) <= lv.budget
            if li > 0:
                assert len(plan) >= MIN_SOLUTION
            else:
                assert lv.tutorial


def test_rollout_first_divergence_has_identical_frame_and_action():
    base, twin = PAIRS[0]
    ps = policy_seed_for(7, 0)
    ra = rollout_episode(base, 0, ps, 60)
    rb = rollout_episode(twin, 1, ps, 60)
    div = next(i for i, (a, b) in enumerate(zip(ra, rb)) if not np.array_equal(a.next_frame, b.next_frame))
    assert np.array_equal(ra[div].frame, rb[div].frame)
    assert ra[div].action == rb[div].action
    assert ra[div].rule_id != rb[div].rule_id
    for a, b in zip(ra[:div], rb[:div]):
        assert np.array_equal(a.frame, b.frame) and a.action == b.action


def test_generated_ids_never_collide_with_public_games():
    ids = {s.game_id for p in PAIRS for s in p}
    assert not ids & set(PUBLIC_GAME_IDS)
    assert all(i.startswith("tsyn-") for i in ids)
    assert len(ids) == 2 * len(PAIRS)
