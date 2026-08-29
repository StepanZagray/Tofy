#!/usr/bin/env python3
"""Tiny drive-mode protocol client used by test_driver.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def exchange(request: dict[str, Any]) -> dict[str, Any]:
    sys.stdout.write(json.dumps(request, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("driver closed stdin")
    response = json.loads(line)
    if "err" in response:
        raise RuntimeError(response["err"])
    return response["ok"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("subcommand", nargs="?")
    parser.add_argument("--mode")
    parser.add_argument("--device")
    parser.add_argument("--checkpoint")
    parser.add_argument("--train-config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--recordings-dir")
    parser.add_argument("--profile-eval")
    parser.add_argument("--seed")
    args, _ = parser.parse_known_args()
    if args.subcommand != "p2-arc3-bridge" or args.mode != "drive":
        raise RuntimeError("mock bridge requires p2-arc3-bridge --mode drive")

    games = exchange({"op": "list_games"})["games"]
    if not games:
        raise RuntimeError("driver returned no games")
    game = games[0]
    card_id = exchange({"op": "open_scorecard"})["card_id"]
    observation = exchange(
        {
            "op": "reset",
            "card_id": card_id,
            "game_id": game["game_id"],
            "guid": None,
        }
    )["observation"]
    guid = observation["guid"]
    actions = 0
    for action_id, data in ((6, {"x": 32, "y": 32}), (1, {}), (0, {})):
        observation = exchange(
            {
                "op": "act",
                "game_id": game["game_id"],
                "guid": guid,
                "action_id": action_id,
                "data": data,
            }
        )["observation"]
        actions += 1
    exchange({"op": "close_scorecard", "card_id": card_id})

    report = {
        "schema": "p2.arc3_live_report.v4",
        "card_id": card_id,
        "games": [
            {
                "game_id": game["game_id"],
                "title": game["title"],
                "actions": actions,
                "levels_completed": observation["levels_completed"],
                "win_levels": observation["win_levels"],
                "terminal_state": observation["state"],
            }
        ],
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
