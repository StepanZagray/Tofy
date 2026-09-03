"""BFS solver over the exact simulator (:func:`rules.apply`).

Used (a) by the level generator to certify solvability and measure the
shortest solution, and (b) by the learning-history rollout policy as the
competent arm of the epsilon-decaying mixture.
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

from .rules import (
    DOOR,
    PASSABLE_STATIC,
    PORTAL,
    SWITCH_OFF,
    SWITCH_ON,
    TELEPORT_RADIUS,
    ClickTarget,
    HiddenRule,
    LevelSpec,
    SimState,
    apply,
    goal_feasible,
)


class SolverAction(NamedTuple):
    action_id: int
    click: ClickTarget | None = None


def candidate_actions(level: LevelSpec, rule: HiddenRule, state: SimState, available: tuple[int, ...]) -> list[SolverAction]:
    """Actions worth trying from ``state`` (pruned but complete for solving)."""
    out: list[SolverAction] = []
    for a in available:
        if 1 <= a <= 4:
            out.append(SolverAction(a))
        elif a == 5:
            if rule.action5 == "toggle" and level.cells[state.player] in (SWITCH_OFF, SWITCH_ON):
                out.append(SolverAction(5))
            elif rule.action5 == "portal" and level.cells[state.player] == PORTAL:
                out.append(SolverAction(5))
        elif a == 6:
            if rule.action6 == "teleport":
                pc, pr = level.col_row(state.player)
                for dr in range(-TELEPORT_RADIUS, TELEPORT_RADIUS + 1):
                    for dc in range(-TELEPORT_RADIUS, TELEPORT_RADIUS + 1):
                        if dr == 0 and dc == 0:
                            continue
                        c, r = pc + dc, pr + dr
                        if 0 <= c < level.width and 1 <= r <= level.height:
                            role = level.cells[level.idx(c, r)]
                            if role in PASSABLE_STATIC or (role == DOOR and state.has_key):
                                out.append(SolverAction(6, ClickTarget(c, r)))
            elif rule.action6 == "toggle_click":
                for idx, role in enumerate(level.cells):
                    if role in (SWITCH_OFF, SWITCH_ON):
                        out.append(SolverAction(6, ClickTarget(*level.col_row(idx))))
    return out


def solve(level: LevelSpec, rule: HiddenRule, available: tuple[int, ...], start: SimState, max_states: int = 400_000) -> list[SolverAction] | None:
    """Shortest action sequence from ``start`` to level completion, or ``None``.

    Death states are pruned.  Energy is not part of the search state: BFS
    minimises action count, so the caller compares ``len(plan)`` to energy.
    """
    if not goal_feasible(level, rule):
        return None
    parent: dict[SimState, tuple[SimState, SolverAction] | None] = {start: None}
    queue: deque[SimState] = deque([start])
    while queue:
        s = queue.popleft()
        for act in candidate_actions(level, rule, s, available):
            ns, outcome = apply(level, rule, s, act.action_id, act.click)
            if outcome == "dead" or ns in parent:
                continue
            parent[ns] = (s, act)
            if outcome == "complete":
                path: list[SolverAction] = []
                cur = ns
                while parent[cur] is not None:
                    prev, a = parent[cur]  # type: ignore[misc]
                    path.append(a)
                    cur = prev
                path.reverse()
                return path
            if len(parent) > max_states:
                return None
            queue.append(ns)
    return None
