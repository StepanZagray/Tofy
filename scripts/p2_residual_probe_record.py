#!/usr/bin/env python3
"""ADR 0005 §5.3 residual probe: record real ARC-AGI-3 frames (EVALUATION ONLY).

Downloads the public toolkit games (local mode), drives each with a SEEDED
uniform-random policy over `available_actions` (ACTION6 with uniform x,y in
0..63) for a fixed number of actions, and lets the toolkit write its own JSONL
recordings. Nothing here trains on these frames; they feed only the
`p2-residual-probe` CLI.

The ARC API key is read from `.env` (ARC_AGI_3_API_KEY) into ARC_API_KEY, the
variable the toolkit expects, and is never printed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

from tofy_arc3.run_local import _resolve_games, _stderr_logger  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402


def _load_api_key(env_file: Path) -> bool:
    if "ARC_API_KEY" in os.environ:
        return True
    if not env_file.exists():
        return False
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("ARC_AGI_3_API_KEY=") or line.startswith("ARC_API_KEY="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                os.environ["ARC_API_KEY"] = value
                return True
    return False


def _frame_key(frame) -> tuple:
    return tuple(tuple(tuple(int(v) for v in row) for row in layer) for layer in frame)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--env-file", default=str(HERE.parent.parent.parent / "Tofy" / ".env"))
    ap.add_argument("--actions", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--games", default="all")
    args = ap.parse_args()

    out = Path(args.out)
    env_dir = out / "environment_files"
    rec_dir = out / "recordings"
    env_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RECORDINGS_DIR"] = str(rec_dir)

    logger = _stderr_logger()
    have_key = _load_api_key(Path(args.env_file))
    logger.info("api key present: %s", have_key)

    arcade, games = _resolve_games(args.games, env_dir, True, logger)
    logger.info("resolved %d games", len(games))

    summary = {
        "seed": args.seed,
        "actions_per_game": args.actions,
        "policy": "uniform over available_actions minus RESET; ACTION6 x,y ~ U{0..63}",
        "recordings_dir": str(rec_dir),
        "games": [],
    }
    for index, game in enumerate(sorted(games, key=lambda g: g["game_id"])):
        game_id = game["game_id"]
        rng = random.Random(args.seed * 1000 + index)
        row = {
            "game_id": game_id,
            "title": game["title"],
            "game_seed": args.seed * 1000 + index,
            "actions": 0,
            "transitions": 0,
            "changed": 0,
            "resets_after_terminal": 0,
            "wins": 0,
            "game_overs": 0,
            "levels_completed_max": 0,
            "error": None,
        }
        started = time.time()
        try:
            env = arcade.make(game_id, save_recording=True)
            if env is None:
                raise RuntimeError("make() returned None")
            obs = env.reset()
            if obs is None:
                raise RuntimeError("reset() returned None")
            prev = _frame_key(obs.frame)
            action_kinds: dict[str, int] = {}
            for _ in range(args.actions):
                avail = [a for a in obs.available_actions if a != 0]
                if not avail:
                    avail = [1, 2, 3, 4, 5, 6]
                aid = rng.choice(avail)
                action = GameAction.from_id(aid)
                data = None
                if action.is_complex():
                    data = {"x": rng.randrange(64), "y": rng.randrange(64)}
                obs = env.step(action, data=data)
                if obs is None:
                    raise RuntimeError(f"step({action.name}) returned None")
                row["actions"] += 1
                action_kinds[action.name] = action_kinds.get(action.name, 0) + 1
                cur = _frame_key(obs.frame)
                row["transitions"] += 1
                if cur != prev:
                    row["changed"] += 1
                row["levels_completed_max"] = max(
                    row["levels_completed_max"], int(obs.levels_completed)
                )
                if obs.state in (GameState.WIN, GameState.GAME_OVER):
                    if obs.state == GameState.WIN:
                        row["wins"] += 1
                    else:
                        row["game_overs"] += 1
                    obs = env.reset()
                    if obs is None:
                        raise RuntimeError("reset() after terminal returned None")
                    row["resets_after_terminal"] += 1
                    cur = _frame_key(obs.frame)
                prev = cur
            row["action_kinds"] = action_kinds
            row["recording_file"] = (
                str(env._recording_filename) if getattr(env, "_recording_filename", None) else None
            )
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            logger.error("game %s failed: %s", game_id, row["error"])
        row["seconds"] = round(time.time() - started, 2)
        summary["games"].append(row)
        logger.info(
            "%s actions=%d changed=%d err=%s",
            game_id, row["actions"], row["changed"], row["error"],
        )
        (out / "recordings_summary.json").write_text(json.dumps(summary, indent=2))

    try:
        arcade.close_scorecard()
    except Exception:  # noqa: BLE001
        pass
    ok = [g for g in summary["games"] if not g["error"]]
    summary["games_ok"] = len(ok)
    summary["transitions_total"] = sum(g["transitions"] for g in ok)
    summary["changed_total"] = sum(g["changed"] for g in ok)
    (out / "recordings_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "games"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
