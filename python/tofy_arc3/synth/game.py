"""``ARCBaseGame`` subclass driven by a :class:`GameSpec`.

The engine owns the action loop, level progression, reset semantics, camera
scaling/letterboxing and the 64x64 frame; the game logic is delegated to
:func:`rules.apply` so that the BFS solver and the census see exactly the
dynamics the engine renders.
"""

from __future__ import annotations

from typing import Type

import numpy as np
from arcengine import ARCBaseGame, ActionInput, BlockingMode, Camera, GameAction, Level, Sprite

from .rules import ClickTarget, GameSpec, LevelSpec, SimState, apply, initial_state, render_grid

BOARD_SPRITE = "board"


def _level_sprites(level: LevelSpec) -> list[Sprite]:
    # A single opaque board sprite; pixels are rewritten every step.
    pixels = np.zeros((level.rows, level.width), dtype=np.int8)
    return [Sprite(pixels, name=BOARD_SPRITE, x=0, y=0, layer=0, blocking=BlockingMode.NOT_BLOCKED)]


class SynthGame(ARCBaseGame):
    """One generated game.  Subclasses are created per spec by :func:`make_game_class`."""

    SPEC: GameSpec

    def __init__(self, seed: int = 0) -> None:
        spec = self.SPEC
        levels = [
            Level(sprites=_level_sprites(lv), grid_size=(lv.width, lv.rows), name=f"level{i}")
            for i, lv in enumerate(spec.levels)
        ]
        bg = spec.rule.background
        camera = Camera(width=spec.levels[0].width, height=spec.levels[0].rows, background=bg, letter_box=bg)
        self._sim: SimState | None = None
        self._energy = 0
        super().__init__(
            game_id=spec.game_id,
            levels=levels,
            camera=camera,
            available_actions=list(spec.available_actions),
            seed=seed,
        )

    # -- engine hooks -------------------------------------------------------

    @property
    def level_spec(self) -> LevelSpec:
        return self.SPEC.levels[self.level_index]

    @property
    def sim_state(self) -> SimState:
        assert self._sim is not None
        return self._sim

    @property
    def energy(self) -> int:
        return self._energy

    def on_set_level(self, level: Level) -> None:
        spec = self.SPEC.levels[self._current_level_index]
        self._sim = initial_state(spec)
        self._energy = spec.budget
        self._sync_sprites()

    def _sync_sprites(self) -> None:
        spec = self.level_spec
        grid = render_grid(spec, self.SPEC.rule, self.sim_state, self._energy)
        board = self.current_level.get_sprites_by_name(BOARD_SPRITE)[0]
        board.pixels = np.asarray(grid, dtype=np.int8)

    def _click_target(self, action: ActionInput) -> ClickTarget | None:
        x = int(action.data.get("x", 0))
        y = int(action.data.get("y", 0))
        grid = self.camera.display_to_grid(x, y)
        if grid is None:
            return None
        return ClickTarget(grid[0], grid[1])

    def step(self) -> None:
        action = self.action
        aid = action.id.value if isinstance(action.id, GameAction) else int(action.id)
        if aid == GameAction.RESET.value:
            # handle_reset already rebuilt the level; just render it.
            self.complete_action()
            return
        if aid not in self.SPEC.available_actions:
            # Unavailable action: no state change, no energy cost.
            self.complete_action()
            return
        click = self._click_target(action) if aid == 6 else None
        spec = self.level_spec
        state, outcome = apply(spec, self.SPEC.rule, self.sim_state, aid, click)
        self._sim = state
        self._energy -= 1
        if outcome == "complete":
            self._sync_sprites()
            self.complete_action()
            self.next_level()
            return
        if outcome == "dead" or self._energy <= 0:
            self._sync_sprites()
            self.complete_action()
            self.lose()
            return
        self._sync_sprites()
        self.complete_action()


def make_game_class(spec: GameSpec) -> Type[SynthGame]:
    """Create a concrete ``ARCBaseGame`` subclass bound to ``spec``."""
    name = "Tsyn" + spec.game_id.split("-", 1)[1]
    return type(name, (SynthGame,), {"SPEC": spec, "__module__": __name__})


def display_coords(game: SynthGame, click: ClickTarget) -> tuple[int, int]:
    """Grid cell -> 64x64 display pixel (centre of the scaled cell)."""
    scale, x_off, y_off = game.camera._calculate_scale_and_offset()
    return click.col * scale + x_off + scale // 2, click.row * scale + y_off + scale // 2
