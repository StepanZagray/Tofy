"""Procedural game/level generation and twin construction (ADR 0005 §2.3).

All randomness flows through a single ``random.Random`` per game so that the
same seed reproduces the same games byte-for-byte.  No public ARC-AGI-3 level
is read or imitated; layouts are random walls/features on small grids.
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from typing import Sequence

from .public_games import GENERATED_GAME_ID_PREFIX
from .rules import (
    ACTION5_KINDS,
    ACTION6_KINDS,
    DECOR,
    DIRECTIONS,
    DIRECTION_NAMES,
    DOOR,
    EXIT,
    FLOOR,
    GOAL_KINDS,
    HAZARD,
    ITEM,
    KEY,
    N_ROLES,
    PORTAL,
    SWAPPABLE_ROLES,
    SWITCH_OFF,
    SWITCH_ON,
    WALL,
    ClickTarget,
    GameSpec,
    HiddenRule,
    LevelSpec,
    SimState,
    apply,
    goal_feasible,
    initial_state,
    render_grid,
)
from .solver import solve

MIN_SOLUTION_TUTORIAL = 3
MAX_SOLUTION_TUTORIAL = 10
MIN_SOLUTION = 10
SHARED_PREFIX_LEN = 12  # forced-random policy prefix shared by twins (see rollout)


def game_id_for(seed: int, index: int, twin: bool) -> str:
    h = hashlib.blake2b(f"{seed}:{index}:{'twin' if twin else 'base'}".encode(), digest_size=4).hexdigest()
    return f"{GENERATED_GAME_ID_PREFIX}-{h}"


# --------------------------------------------------------------------------
# Rule drawing
# --------------------------------------------------------------------------


def draw_rule(rng: random.Random) -> HiddenRule:
    perm = list(DIRECTION_NAMES.values())
    rng.shuffle(perm)
    action5 = rng.choice(ACTION5_KINDS)
    action6 = rng.choice(ACTION6_KINDS)
    goal = rng.choice(GOAL_KINDS)
    if goal.startswith("toggle") and action5 != "toggle" and action6 != "toggle_click":
        if rng.random() < 0.5:
            action5 = "toggle"
        else:
            action6 = "toggle_click"
    colours = tuple(rng.sample(range(16), N_ROLES))
    return HiddenRule(tuple(perm), action5, action6, goal, colours)


def draw_available(rng: random.Random, rule: HiddenRule) -> tuple[int, ...]:
    avail = [1, 2, 3, 4]
    if rule.action5 != "noop" or rng.random() < 0.3:
        avail.append(5)
    if rule.action6 != "noop" or rng.random() < 0.3:
        avail.append(6)
    return tuple(avail)


# --------------------------------------------------------------------------
# Layout generation
# --------------------------------------------------------------------------


def _neighbours(W: int, H: int, idx: int):
    c, r = idx % W, idx // W
    for dc, dr in DIRECTIONS:
        nc, nr = c + dc, r + dr
        if 0 <= nc < W and 1 <= nr <= H:
            yield nr * W + nc


def _component(cells: list[int], W: int, H: int, start: int, blocked: set[int] | None = None) -> set[int]:
    blocked = blocked or set()
    seen = {start}
    q = deque([start])
    while q:
        i = q.popleft()
        for n in _neighbours(W, H, i):
            if n in seen or n in blocked or cells[n] == WALL:
                continue
            seen.add(n)
            q.append(n)
    return seen


def _shortest_path(cells: list[int], W: int, H: int, src: int, dst: int) -> list[int] | None:
    prev = {src: -1}
    q = deque([src])
    while q:
        i = q.popleft()
        if i == dst:
            path = []
            while i != -1:
                path.append(i)
                i = prev[i]
            return path[::-1]
        for n in _neighbours(W, H, i):
            if n not in prev and cells[n] != WALL:
                prev[n] = i
                q.append(n)
    return None


def _try_layout(rng: random.Random, rule: HiddenRule, available: tuple[int, ...], W: int, H: int, tutorial: bool) -> LevelSpec | None:
    rows = H + 1
    cells = [FLOOR] * (W * rows)
    for c in range(W):
        cells[1 * W + c] = WALL
        cells[H * W + c] = WALL
    for r in range(1, H + 1):
        cells[r * W] = WALL
        cells[r * W + W - 1] = WALL
    interior = [r * W + c for r in range(2, H) for c in range(1, W - 1)]
    density = rng.uniform(0.05, 0.18) if tutorial else rng.uniform(0.10, 0.30)
    for i in interior:
        if rng.random() < density:
            cells[i] = WALL
    floor = [i for i in interior if cells[i] == FLOOR]
    if len(floor) < 8:
        return None
    player = rng.choice(floor)
    comp = _component(cells, W, H, player)
    for i in floor:
        if i not in comp:
            cells[i] = WALL
    free = sorted(comp - {player})
    rng.shuffle(free)
    if len(free) < 6:
        return None

    def take() -> int | None:
        return free.pop() if free else None

    goal = rule.goal
    need_exit = goal in ("reach_exit", "collect_then_exit", "toggle_then_exit")
    need_items = goal in ("collect_all", "collect_then_exit")
    need_switches = goal in ("toggle_all", "toggle_then_exit")

    exit_cell: int | None = None
    if need_exit or rng.random() < 0.6:
        # Prefer a far cell for the exit to lengthen the solution.
        far = sorted(free, key=lambda i: -abs(i % W - player % W) - abs(i // W - player // W))
        exit_cell = far[rng.randrange(min(3, len(far)))]
        free.remove(exit_cell)
        cells[exit_cell] = EXIT

    n_items = (rng.randint(1, 2) if tutorial else rng.randint(2, 4)) if need_items or rng.random() < 0.3 else 0
    for _ in range(n_items):
        i = take()
        if i is not None:
            cells[i] = ITEM
    n_switches = (1 if tutorial else rng.randint(2, 3)) if need_switches or rng.random() < 0.15 else 0
    for k in range(n_switches):
        i = take()
        if i is not None:
            cells[i] = SWITCH_ON if (k == 0 and n_switches > 1 and rng.random() < 0.3) else SWITCH_OFF

    # Key/door: the door must be a choke point between player and exit.
    if exit_cell is not None and not tutorial and rng.random() < 0.5:
        path = _shortest_path(cells, W, H, player, exit_cell)
        if path and len(path) > 3:
            candidates = [i for i in path[1:-1] if cells[i] == FLOOR]
            rng.shuffle(candidates)
            for door in candidates[:8]:
                side = _component(cells, W, H, player, blocked={door})
                if exit_cell in side:
                    continue
                key_choices = [i for i in free if i in side and cells[i] == FLOOR]
                if not key_choices:
                    continue
                key = rng.choice(key_choices)
                free.remove(key)
                if door in free:
                    free.remove(door)
                cells[door] = DOOR
                cells[key] = KEY
                break

    if rule.action5 == "portal" or rng.random() < 0.2:
        if len(free) >= 2:
            for _ in range(2):
                i = take()
                if i is not None:
                    cells[i] = PORTAL

    n_decor = rng.randint(0, 2) if tutorial else rng.randint(0, 3)
    for _ in range(n_decor):
        i = take()
        if i is not None:
            cells[i] = DECOR

    n_hazards = rng.randint(1, 2) if tutorial else rng.randint(2, 6)
    hazards = []
    for _ in range(n_hazards):
        i = take()
        if i is not None:
            cells[i] = HAZARD
            hazards.append(i)

    level = LevelSpec(W, H, tuple(cells), player, budget=1, tutorial=tutorial)
    if not goal_feasible(level, rule):
        return None
    plan = solve(level, rule, available, initial_state(level))
    while plan is None and hazards:
        cells[hazards.pop()] = FLOOR
        level = LevelSpec(W, H, tuple(cells), player, budget=1, tutorial=tutorial)
        plan = solve(level, rule, available, initial_state(level))
    if plan is None:
        return None
    L = len(plan)
    if tutorial:
        if not MIN_SOLUTION_TUTORIAL <= L <= MAX_SOLUTION_TUTORIAL:
            return None
        budget = L + rng.randint(L, 2 * L)
    else:
        if L < MIN_SOLUTION:
            return None
        budget = L + rng.randint(1, max(2, L // 4))
    return LevelSpec(W, H, tuple(cells), player, budget=budget, tutorial=tutorial)


def generate_level(rng: random.Random, rule: HiddenRule, available: tuple[int, ...], W: int, H: int, tutorial: bool, tries: int = 60) -> LevelSpec | None:
    for _ in range(tries):
        lv = _try_layout(rng, rule, available, W, H, tutorial)
        if lv is not None:
            return lv
    return None


def budget_for(rng: random.Random, level: LevelSpec, rule: HiddenRule, available: tuple[int, ...]) -> int | None:
    """Energy budget for an existing layout under ``rule``; ``None`` if the
    layout is unsolvable or its shortest solution is too short to be a
    non-trivial level (twin levels must stay hard for random play)."""
    plan = solve(level, rule, available, initial_state(level))
    if plan is None:
        return None
    L = len(plan)
    if level.tutorial:
        if L < 2:
            return None
        return L + rng.randint(L, 2 * L)
    if L < MIN_SOLUTION:
        return None
    return L + rng.randint(1, max(2, L // 4))


# --------------------------------------------------------------------------
# Game generation
# --------------------------------------------------------------------------


def _draw_dims(rng: random.Random) -> tuple[int, int]:
    W = rng.choice((8, 10, 12, 14, 16))
    H = rng.randint(max(6, W - 4), W)
    return W, H


def generate_base_game(seed: int, index: int, n_levels: int | None = None) -> GameSpec:
    rng = random.Random(f"tofy-synth:{seed}:{index}")
    for _attempt in range(50):
        rule = draw_rule(rng)
        available = draw_available(rng, rule)
        W, H = _draw_dims(rng)
        count = n_levels or rng.choice((4, 5, 6))
        levels: list[LevelSpec] = []
        ok = True
        for li in range(count):
            lv = generate_level(rng, rule, available, W, H, tutorial=(li == 0))
            if lv is None:
                ok = False
                break
            levels.append(lv)
        if ok:
            return GameSpec(
                game_id=game_id_for(seed, index, twin=False),
                rule=rule,
                levels=tuple(levels),
                available_actions=available,
                meta={"seed": seed, "index": index, "dims": [W, H]},
            )
    raise RuntimeError(f"could not generate a game for seed={seed} index={index}")


# --------------------------------------------------------------------------
# Twins
# --------------------------------------------------------------------------


def shared_prefix_actions(policy_seed: int, available: tuple[int, ...], n: int = SHARED_PREFIX_LEN) -> list[tuple[int, int, int]]:
    """The forced-random policy prefix (action id, display x, display y).

    Both twins replay exactly this prefix (RESETs are inserted by the rollout
    when the game is over, without consuming prefix entries).
    """
    rng = random.Random(f"tofy-synth-prefix:{policy_seed}")
    out = []
    for _ in range(n):
        a = rng.choice(available)
        if a == 6:
            out.append((a, rng.randrange(64), rng.randrange(64)))
        else:
            out.append((a, 255, 255))
    return out


def _grid_click(level: LevelSpec, x: int, y: int) -> ClickTarget | None:
    scale = min(64 // level.width, 64 // level.rows)
    xo = (64 - level.width * scale) // 2
    yo = (64 - level.rows * scale) // 2
    if x < xo or y < yo:
        return None
    c, r = (x - xo) // scale, (y - yo) // scale
    if c >= level.width or r >= level.rows:
        return None
    return ClickTarget(c, r)


def _sim_prefix_frames(spec: GameSpec, level: LevelSpec, prefix: Sequence[tuple[int, int, int]]) -> list[tuple[list[list[int]], tuple[int, int, int], list[list[int]]]]:
    """Run the prefix through the fast simulator; returns (frame, action, next_frame) triples.

    Mirrors the engine semantics used by :mod:`.rollout` for level 0: death or
    energy exhaustion resets the level (next action is RESET, not consumed).
    Level completion stops the trace (later levels may differ between twins).
    """
    state = initial_state(level)
    energy = level.budget
    frame = render_grid(level, spec.rule, state, energy)
    out = []
    for a, x, y in prefix:
        if a not in spec.available_actions:
            out.append((frame, (a, x, y), frame))
            continue
        click = _grid_click(level, x, y) if a == 6 else None
        state, outcome = apply(level, spec.rule, state, a, click)
        energy -= 1
        nxt = render_grid(level, spec.rule, state, energy)
        out.append((frame, (a, x, y), nxt))
        if outcome == "complete":
            break
        if outcome == "dead" or energy <= 0:
            # RESET row
            state = initial_state(level)
            energy = level.budget
            reset_frame = render_grid(level, spec.rule, state, energy)
            out.append((nxt, (0, 255, 255), reset_frame))
            frame = reset_frame
            continue
        frame = nxt
    return out


def twins_diverge_in_prefix(base: GameSpec, twin: GameSpec, prefix: Sequence[tuple[int, int, int]]) -> bool:
    """True iff the shared prefix yields a transition with identical
    (frame, action) but different next_frame between the two games."""
    a = _sim_prefix_frames(base, base.levels[0], prefix)
    b = _sim_prefix_frames(twin, twin.levels[0], prefix)
    for (fa, aa, na), (fb, ab, nb) in zip(a, b):
        if fa != fb or aa != ab:
            return False
        if na != nb:
            return True
    return False


def _mutate_rule(rng: random.Random, spec: GameSpec, kinds: list[str]) -> tuple[HiddenRule, list[LevelSpec]] | None:
    rule = spec.rule
    levels = list(spec.levels)
    for kind in kinds:
        if kind == "move":
            for _ in range(20):
                perm = list(rule.move_perm)
                rng.shuffle(perm)
                if tuple(perm) != rule.move_perm:
                    break
            rule = HiddenRule(tuple(perm), rule.action5, rule.action6, rule.goal, rule.colours)
        elif kind == "swap":
            present = [r for r in SWAPPABLE_ROLES if r in spec.levels[0].cells]
            if len(present) < 2:
                return None
            a, b = rng.sample(present, 2)
            rule = rule.with_roles_swapped(a, b)
            levels = [lv.with_roles_swapped(a, b) for lv in levels]
        elif kind == "goal":
            options = [g for g in GOAL_KINDS if g != rule.goal]
            rng.shuffle(options)
            for g in options:
                cand = HiddenRule(rule.move_perm, rule.action5, rule.action6, g, rule.colours)
                if goal_feasible(levels[0], cand):
                    rule = cand
                    break
            else:
                return None
        elif kind == "action5":
            if 5 not in spec.available_actions:
                return None
            options = [k for k in ACTION5_KINDS if k != rule.action5]
            rule = HiddenRule(rule.move_perm, rng.choice(options), rule.action6, rule.goal, rule.colours)
        elif kind == "action6":
            if 6 not in spec.available_actions:
                return None
            options = [k for k in ACTION6_KINDS if k != rule.action6]
            rule = HiddenRule(rule.move_perm, rule.action5, rng.choice(options), rule.goal, rule.colours)
        else:
            raise AssertionError(kind)
    if rule.rule_id() == spec.rule.rule_id():
        return None
    return rule, levels


def make_twin(seed: int, index: int, base: GameSpec, policy_seed: int) -> GameSpec:
    """Construct the twin: byte-identical level-0 initial frame, different rule.

    The twin must diverge from the base within the shared random policy
    prefix so the ``single_frame_rule_identifiable`` census is 0 and the first
    divergent transition has identical (frame, action).
    """
    rng = random.Random(f"tofy-synth-twin:{seed}:{index}")
    prefix = shared_prefix_actions(policy_seed, base.available_actions)
    W, H = base.levels[0].width, base.levels[0].height
    base_frame0 = render_grid(base.levels[0], base.rule, initial_state(base.levels[0]), base.levels[0].budget)
    menu = ["move", "swap", "goal", "action5", "action6"]
    for attempt in range(200):
        n_mut = 1 if rng.random() < 0.6 else 2
        kinds = rng.sample(menu, n_mut)
        if attempt > 60 and "move" not in kinds:
            kinds.append("move")  # movement changes diverge fastest
        mutated = _mutate_rule(rng, base, kinds)
        if mutated is None:
            continue
        rule, levels = mutated
        # Level 0 keeps the layout; only the budget is recomputed.
        b0 = budget_for(rng, levels[0], rule, base.available_actions)
        if b0 is None:
            continue
        levels[0] = levels[0].with_budget(b0)
        twin_levels = [levels[0]]
        ok = True
        for li in range(1, len(levels)):
            b = budget_for(rng, levels[li], rule, base.available_actions)
            if b is not None:
                twin_levels.append(levels[li].with_budget(b))
                continue
            fresh = generate_level(rng, rule, base.available_actions, W, H, tutorial=False)
            if fresh is None:
                ok = False
                break
            twin_levels.append(fresh)
        if not ok:
            continue
        twin = GameSpec(
            game_id=game_id_for(seed, index, twin=True),
            rule=rule,
            levels=tuple(twin_levels),
            available_actions=base.available_actions,
            twin_of=base.game_id,
            meta={"seed": seed, "index": index, "dims": [W, H], "mutations": kinds},
        )
        twin_frame0 = render_grid(twin.levels[0], twin.rule, initial_state(twin.levels[0]), twin.levels[0].budget)
        if twin_frame0 != base_frame0:
            continue  # should not happen; swaps preserve rendering
        if twins_diverge_in_prefix(base, twin, prefix):
            return twin
    raise RuntimeError(f"could not construct a diverging twin for {base.game_id}")


def generate_pair(seed: int, index: int, policy_seed: int, n_levels: int | None = None) -> tuple[GameSpec, GameSpec]:
    """Generate (base, twin) for ``index``; retries the base if no twin diverges."""
    for sub in range(8):
        base = generate_base_game(seed, index * 1000 + sub if sub else index, n_levels)
        base = GameSpec(game_id_for(seed, index, twin=False), base.rule, base.levels, base.available_actions, None, base.meta)
        try:
            twin = make_twin(seed, index, base, policy_seed)
        except RuntimeError:
            continue
        return base, twin
    raise RuntimeError(f"could not generate a twin pair for seed={seed} index={index}")
