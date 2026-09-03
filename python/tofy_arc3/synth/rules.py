"""Hidden rules, level layouts and the exact grid simulator.

The simulator here is the single source of truth: the ``ARCBaseGame`` subclass
in :mod:`.game` calls :func:`apply` from its ``step()``, the BFS solver in
:mod:`.solver` searches over :func:`apply`, and the random-play census runs
:func:`apply` directly.  Rendering to a colour grid (:func:`render_grid`) is
what the game hands to the ARCEngine camera, so engine frames and simulator
frames agree by construction.

Coordinates: a level is ``W`` columns by ``H + 1`` rows.  Row 0 is the energy
HUD (board content, no reserved status row semantics), rows ``1..H`` are the
board with a wall border.  Cells are indexed ``idx = row * W + col``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, Literal, NamedTuple

# --------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------

FLOOR = 0
WALL = 1
PLAYER = 2
EXIT = 3
ITEM = 4
HAZARD = 5
KEY = 6
DOOR = 7
SWITCH_OFF = 8
SWITCH_ON = 9
PORTAL = 10
DECOR = 11
ENERGY = 12
N_ROLES = 13

ROLE_NAMES = (
    "floor", "wall", "player", "exit", "item", "hazard", "key", "door",
    "switch_off", "switch_on", "portal", "decor", "energy",
)

# Roles whose colour identity carries hidden-rule meaning and may be swapped
# between twins while the rendered frame stays identical.
SWAPPABLE_ROLES = (EXIT, ITEM, HAZARD, KEY, DECOR, PORTAL)

# Static layout roles stored in a level's cell grid.  Dynamic roles (ITEM,
# KEY, DOOR, SWITCH_*) are stored in the cell grid as their initial role and
# their live status is tracked in :class:`SimState`.
PASSABLE_STATIC = frozenset({FLOOR, EXIT, HAZARD, PORTAL, DECOR, ITEM, KEY, SWITCH_OFF, SWITCH_ON})

# Directions: (dcol, drow)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
DIRECTION_NAMES = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}
DIRECTION_BY_NAME = {v: k for k, v in DIRECTION_NAMES.items()}

Action5 = Literal["noop", "toggle", "portal"]
Action6 = Literal["noop", "teleport", "toggle_click"]
Goal = Literal["reach_exit", "collect_all", "collect_then_exit", "toggle_all", "toggle_then_exit"]

ACTION5_KINDS: tuple[str, ...] = ("noop", "toggle", "portal")
ACTION6_KINDS: tuple[str, ...] = ("noop", "teleport", "toggle_click")
GOAL_KINDS: tuple[str, ...] = ("reach_exit", "collect_all", "collect_then_exit", "toggle_all", "toggle_then_exit")

TELEPORT_RADIUS = 2

# Available-actions mask bits (ADR 0005 §1.6): bit 0 = RESET, bit i = ACTIONi.
MASK_RESET = 1 << 0


def mask_from_actions(actions: Iterable[int]) -> int:
    mask = MASK_RESET
    for a in actions:
        if not 1 <= a <= 7:
            raise ValueError(f"bad action id {a}")
        mask |= 1 << a
    return mask


def actions_from_mask(mask: int) -> list[int]:
    return [a for a in range(1, 8) if mask & (1 << a)]


# --------------------------------------------------------------------------
# Hidden rule
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HiddenRule:
    """The episode-stable rule.  Never rendered, never conditioned on."""

    move_perm: tuple[str, str, str, str]  # direction names for ACTION1..4
    action5: str
    action6: str
    goal: str
    colours: tuple[int, ...]  # role -> colour index (len N_ROLES); colours[FLOOR] is the background

    def __post_init__(self) -> None:
        if sorted(self.move_perm) != sorted(DIRECTION_NAMES.values()):
            raise ValueError(f"move_perm must be a permutation of directions: {self.move_perm}")
        if self.action5 not in ACTION5_KINDS:
            raise ValueError(self.action5)
        if self.action6 not in ACTION6_KINDS:
            raise ValueError(self.action6)
        if self.goal not in GOAL_KINDS:
            raise ValueError(self.goal)
        if len(self.colours) != N_ROLES or any(not 0 <= c <= 15 for c in self.colours):
            raise ValueError("colours must map all roles into 0..15")
        # Every role except FLOOR/DECOR-vs-nothing must be visually distinct.
        if len(set(self.colours)) != N_ROLES:
            raise ValueError("role colours must be pairwise distinct")

    @property
    def background(self) -> int:
        return self.colours[FLOOR]

    def direction(self, action_id: int) -> tuple[int, int]:
        return DIRECTION_BY_NAME[self.move_perm[action_id - 1]]

    def to_dict(self) -> dict:
        return {
            "move_perm": list(self.move_perm),
            "action5": self.action5,
            "action6": self.action6,
            "goal": self.goal,
            "colours": {ROLE_NAMES[r]: c for r, c in enumerate(self.colours)},
        }

    @staticmethod
    def from_dict(d: dict) -> "HiddenRule":
        colours = tuple(d["colours"][name] for name in ROLE_NAMES)
        return HiddenRule(tuple(d["move_perm"]), d["action5"], d["action6"], d["goal"], colours)

    def rule_id(self) -> int:
        """Stable 64-bit hash of the rule record."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return int.from_bytes(hashlib.blake2b(blob, digest_size=8).digest(), "little")

    def with_roles_swapped(self, a: int, b: int) -> "HiddenRule":
        cols = list(self.colours)
        cols[a], cols[b] = cols[b], cols[a]
        return HiddenRule(self.move_perm, self.action5, self.action6, self.goal, tuple(cols))


def split_rule_id(rule_id: int) -> tuple[int, int]:
    return rule_id & 0xFFFFFFFF, (rule_id >> 32) & 0xFFFFFFFF


# --------------------------------------------------------------------------
# Level layout
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelSpec:
    """Static layout of one level (roles per cell) plus its energy budget."""

    width: int
    height: int  # board rows (excluding HUD row 0)
    cells: tuple[int, ...]  # len == width * (height + 1); row 0 is HUD (all FLOOR)
    player: int  # cell index of the initial player position
    budget: int  # energy: number of actions allowed before the level is lost
    tutorial: bool = False

    @property
    def rows(self) -> int:
        return self.height + 1

    def idx(self, col: int, row: int) -> int:
        return row * self.width + col

    def col_row(self, idx: int) -> tuple[int, int]:
        return idx % self.width, idx // self.width

    def cells_of(self, role: int) -> tuple[int, ...]:
        return tuple(i for i, r in enumerate(self.cells) if r == role)

    def with_roles_swapped(self, a: int, b: int) -> "LevelSpec":
        cells = tuple(b if r == a else a if r == b else r for r in self.cells)
        return LevelSpec(self.width, self.height, cells, self.player, self.budget, self.tutorial)

    def with_budget(self, budget: int) -> "LevelSpec":
        return LevelSpec(self.width, self.height, self.cells, self.player, budget, self.tutorial)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "cells": list(self.cells),
            "player": self.player,
            "budget": self.budget,
            "tutorial": self.tutorial,
        }

    @staticmethod
    def from_dict(d: dict) -> "LevelSpec":
        return LevelSpec(d["width"], d["height"], tuple(d["cells"]), d["player"], d["budget"], d.get("tutorial", False))


# --------------------------------------------------------------------------
# Simulator state
# --------------------------------------------------------------------------


class SimState(NamedTuple):
    """Hashable dynamic state (energy tracked separately; see :func:`apply`)."""

    player: int
    items: frozenset  # remaining item cells
    switches_on: frozenset  # switch cells currently on
    has_key: bool  # key collected => all doors open


def initial_state(level: LevelSpec) -> SimState:
    return SimState(
        player=level.player,
        items=frozenset(level.cells_of(ITEM)),
        switches_on=frozenset(level.cells_of(SWITCH_ON)),
        has_key=False,
    )


class ClickTarget(NamedTuple):
    col: int
    row: int


Outcome = Literal["ok", "complete", "dead"]


def _passable(level: LevelSpec, state: SimState, idx: int) -> bool:
    role = level.cells[idx]
    if role == WALL:
        return False
    if role == DOOR:
        return state.has_key
    return role in PASSABLE_STATIC


def _enter(level: LevelSpec, state: SimState, idx: int) -> tuple[SimState, bool]:
    """Move the player onto ``idx`` and apply pick-up effects.  Returns (state, dead)."""
    role = level.cells[idx]
    items = state.items
    has_key = state.has_key
    if role == HAZARD:
        return SimState(idx, items, state.switches_on, has_key), True
    if role == ITEM and idx in items:
        items = items - {idx}
    elif role == KEY and not has_key:
        has_key = True
    return SimState(idx, items, state.switches_on, has_key), False


def _all_switches_on(level: LevelSpec, state: SimState) -> bool:
    switches = level.cells_of(SWITCH_OFF) + level.cells_of(SWITCH_ON)
    return all(s in state.switches_on for s in switches)


def goal_reached(level: LevelSpec, rule: HiddenRule, state: SimState) -> bool:
    on_exit = level.cells[state.player] == EXIT
    if rule.goal == "reach_exit":
        return on_exit
    if rule.goal == "collect_all":
        return not state.items
    if rule.goal == "collect_then_exit":
        return on_exit and not state.items
    if rule.goal == "toggle_all":
        return _all_switches_on(level, state)
    if rule.goal == "toggle_then_exit":
        return on_exit and _all_switches_on(level, state)
    raise AssertionError(rule.goal)


def goal_feasible(level: LevelSpec, rule: HiddenRule) -> bool:
    """Whether the level contains the elements the goal family needs."""
    if rule.goal in ("reach_exit", "collect_then_exit", "toggle_then_exit") and not level.cells_of(EXIT):
        return False
    if rule.goal in ("collect_all", "collect_then_exit") and not level.cells_of(ITEM):
        return False
    if rule.goal in ("toggle_all", "toggle_then_exit"):
        if not (level.cells_of(SWITCH_OFF) or level.cells_of(SWITCH_ON)):
            return False
        if rule.action5 != "toggle" and rule.action6 != "toggle_click":
            return False
    return True


def apply(level: LevelSpec, rule: HiddenRule, state: SimState, action_id: int, click: ClickTarget | None = None) -> tuple[SimState, Outcome]:
    """Apply one action (1..6) to ``state``.  RESET is handled by the engine.

    ``click`` is the grid cell for ACTION6 (``None`` when the click landed in
    the letterbox).  Energy accounting is done by the caller.
    """
    W = level.width
    dead = False
    if 1 <= action_id <= 4:
        dc, dr = rule.direction(action_id)
        col, row = level.col_row(state.player)
        nc, nr = col + dc, row + dr
        if 0 <= nc < W and 1 <= nr <= level.height:
            nidx = level.idx(nc, nr)
            if _passable(level, state, nidx):
                state, dead = _enter(level, state, nidx)
    elif action_id == 5:
        if rule.action5 == "toggle":
            if level.cells[state.player] in (SWITCH_OFF, SWITCH_ON):
                p = state.player
                sw = state.switches_on - {p} if p in state.switches_on else state.switches_on | {p}
                state = SimState(p, state.items, sw, state.has_key)
        elif rule.action5 == "portal":
            if level.cells[state.player] == PORTAL:
                portals = level.cells_of(PORTAL)
                others = [p for p in portals if p != state.player]
                if others:
                    # Deterministic: jump to the next portal in index order.
                    i = portals.index(state.player)
                    target = portals[(i + 1) % len(portals)]
                    state, dead = _enter(level, state, target)
    elif action_id == 6:
        if click is not None and rule.action6 != "noop":
            tc, tr = click
            if 0 <= tc < W and 1 <= tr <= level.height:
                tidx = level.idx(tc, tr)
                if rule.action6 == "teleport":
                    pc, pr = level.col_row(state.player)
                    if max(abs(tc - pc), abs(tr - pr)) <= TELEPORT_RADIUS and tidx != state.player and _passable(level, state, tidx):
                        state, dead = _enter(level, state, tidx)
                elif rule.action6 == "toggle_click":
                    if level.cells[tidx] in (SWITCH_OFF, SWITCH_ON):
                        sw = state.switches_on - {tidx} if tidx in state.switches_on else state.switches_on | {tidx}
                        state = SimState(state.player, state.items, sw, state.has_key)
    else:
        raise ValueError(f"unsupported action id {action_id}")

    if dead:
        return state, "dead"
    if goal_reached(level, rule, state):
        return state, "complete"
    return state, "ok"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def hud_fill(width: int, energy: int, budget: int) -> int:
    """Number of HUD cells lit for ``energy`` of ``budget`` (columns 1..W-2)."""
    span = max(1, width - 2)
    if energy <= 0:
        return 0
    return -(-energy * span // budget)  # ceil


def render_grid(level: LevelSpec, rule: HiddenRule, state: SimState, energy: int) -> list[list[int]]:
    """Render the level to a ``rows x width`` colour grid (ints 0..15)."""
    W = level.width
    cols = rule.colours
    bg = cols[FLOOR]
    grid = [[bg] * W for _ in range(level.rows)]
    for idx, role in enumerate(level.cells):
        c, r = idx % W, idx // W
        if r == 0:
            continue
        if role == ITEM:
            colour = cols[ITEM] if idx in state.items else bg
        elif role == KEY:
            colour = bg if state.has_key else cols[KEY]
        elif role == DOOR:
            colour = bg if state.has_key else cols[DOOR]
        elif role in (SWITCH_OFF, SWITCH_ON):
            colour = cols[SWITCH_ON] if idx in state.switches_on else cols[SWITCH_OFF]
        elif role == FLOOR:
            continue
        else:
            colour = cols[role]
        grid[r][c] = colour
    pc, pr = level.col_row(state.player)
    grid[pr][pc] = cols[PLAYER]
    lit = hud_fill(W, energy, level.budget)
    for c in range(1, 1 + lit):
        if c < W - 1:
            grid[0][c] = cols[ENERGY]
    return grid


# --------------------------------------------------------------------------
# Game spec
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GameSpec:
    """A fully determined game: id, rule, levels, available actions."""

    game_id: str
    rule: HiddenRule
    levels: tuple[LevelSpec, ...]
    available_actions: tuple[int, ...]  # subset of 1..6 (RESET implicit)
    twin_of: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def mask(self) -> int:
        return mask_from_actions(self.available_actions)

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "rule": self.rule.to_dict(),
            "rule_id": self.rule.rule_id(),
            "levels": [lv.to_dict() for lv in self.levels],
            "available_actions": list(self.available_actions),
            "twin_of": self.twin_of,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "GameSpec":
        return GameSpec(
            d["game_id"],
            HiddenRule.from_dict(d["rule"]),
            tuple(LevelSpec.from_dict(l) for l in d["levels"]),
            tuple(d["available_actions"]),
            d.get("twin_of"),
            dict(d.get("meta", {})),
        )
