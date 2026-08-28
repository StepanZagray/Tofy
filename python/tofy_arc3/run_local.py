#!/usr/bin/env python3
"""Drive Tofy's ARC-AGI-3 bridge against locally hosted toolkit games."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import FrameDataRaw, GameAction, GameState


class DriverError(RuntimeError):
    """A fatal bridge or toolkit protocol error."""


def _stderr_logger() -> logging.Logger:
    logger = logging.getLogger("tofy_arc3.local")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)

    # arc_agi.scorecard installs its own stdout handler at import time. Keep
    # every non-protocol diagnostic away from stdout as well.
    scorecard_logger = logging.getLogger("arc_agi.scorecard")
    scorecard_logger.handlers.clear()
    scorecard_logger.addHandler(handler)
    scorecard_logger.propagate = False
    return logger


def _arcade(
    mode: OperationMode, environments_dir: Path, logger: logging.Logger
) -> Arcade:
    # arc-agi gives OPERATION_MODE=competition precedence over its explicit
    # constructor argument. The runner's CLI is authoritative.
    os.environ.pop("OPERATION_MODE", None)
    return Arcade(
        operation_mode=mode,
        environments_dir=str(environments_dir),
        logger=logger,
    )


def _matching_game(infos: list[Any], requested: str) -> Any | None:
    if "-" in requested:
        return next((info for info in infos if info.game_id == requested), None)
    matches = [
        info for info in infos if info.game_id.split("-", 1)[0] == requested
    ]
    if not matches:
        return None
    matches.sort(
        key=lambda info: (
            getattr(info, "date_downloaded", None) is not None,
            getattr(info, "date_downloaded", None),
        ),
        reverse=True,
    )
    return matches[0]


def _download_games(
    environments_dir: Path,
    requested: list[str] | None,
    logger: logging.Logger,
) -> None:
    # NORMAL mode obtains an anonymous key only when ARC_API_KEY is absent.
    # Preserve a caller's key for the bridge while forcing toolkit bootstrap
    # to use the public anonymous-download path required by this runner.
    saved_api_key = os.environ.pop("ARC_API_KEY", None)
    try:
        normal = _arcade(OperationMode.NORMAL, environments_dir, logger)
    finally:
        if saved_api_key is not None:
            os.environ["ARC_API_KEY"] = saved_api_key
    try:
        if requested is None:
            targets = sorted({info.game_id for info in normal.get_environments()})
            public_base_ids = {game_id.split("-", 1)[0] for game_id in targets}
            if len(public_base_ids) < 25:
                raise DriverError(
                    "public-suite discovery returned fewer than the expected 25 games"
                )
        else:
            targets = requested
        if not targets:
            raise DriverError("the ARC-AGI-3 API returned no public games")
        failures: list[str] = []
        for game_id in targets:
            if normal.make(game_id) is None:
                failures.append(game_id)
        if failures:
            raise DriverError(
                "failed to download ARC-AGI-3 game(s): " + ", ".join(failures)
            )
    finally:
        # Download wrappers use the toolkit's implicit scorecard. Close it so
        # bootstrap state does not leak into the real offline evaluation.
        try:
            normal.close_scorecard()
        except Exception:
            pass


def _resolve_games(
    games_arg: str,
    environments_dir: Path,
    allow_download: bool,
    logger: logging.Logger,
) -> tuple[Arcade, list[dict[str, str]]]:
    requested: list[str] | None
    if games_arg.strip().lower() == "all":
        requested = None
    else:
        requested = [part.strip() for part in games_arg.split(",") if part.strip()]
        if not requested:
            raise DriverError("--games must be 'all' or a non-empty comma list")
        if len(set(requested)) != len(requested):
            raise DriverError("--games contains duplicate game identifiers")

    offline = _arcade(OperationMode.OFFLINE, environments_dir, logger)
    local_infos = offline.get_environments()

    if allow_download:
        if requested is None:
            _download_games(environments_dir, None, logger)
        else:
            missing = [
                game_id
                for game_id in requested
                if not _matching_game(local_infos, game_id)
            ]
            if missing:
                _download_games(environments_dir, missing, logger)
        offline = _arcade(OperationMode.OFFLINE, environments_dir, logger)
        local_infos = offline.get_environments()

    if requested is None:
        selected_infos = sorted(local_infos, key=lambda info: info.game_id)
    else:
        selected_infos = []
        missing = []
        for game_id in requested:
            info = _matching_game(local_infos, game_id)
            if info is None:
                missing.append(game_id)
            else:
                selected_infos.append(info)
        if missing:
            hint = " (rerun with --allow-download)" if not allow_download else ""
            raise DriverError(
                "requested ARC-AGI-3 game(s) are not available locally: "
                + ", ".join(missing)
                + hint
            )

    if not selected_infos:
        hint = "; rerun with --allow-download" if not allow_download else ""
        raise DriverError(
            f"no ARC-AGI-3 games found under {environments_dir}{hint}"
        )

    seen: set[str] = set()
    games: list[dict[str, str]] = []
    for info in selected_infos:
        if info.game_id in seen:
            continue
        seen.add(info.game_id)
        games.append({"game_id": info.game_id, "title": info.title or info.game_id})
    return offline, games


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DriverError(f"{field} must be an integer >= {minimum}")
    return value


def observation_json(observation: FrameDataRaw) -> dict[str, Any]:
    state_map = {
        GameState.NOT_PLAYED: "NOT_STARTED",
        GameState.NOT_FINISHED: "NOT_FINISHED",
        GameState.WIN: "WIN",
        GameState.GAME_OVER: "GAME_OVER",
    }
    try:
        wire_state = state_map[observation.state]
    except KeyError as exc:
        raise DriverError(f"unknown toolkit game state: {observation.state!r}") from exc

    if not observation.frame:
        raise DriverError(
            "toolkit returned an empty/degenerate frame; refusing to forward it"
        )
    frame: list[list[list[int]]] = []
    for index, layer in enumerate(observation.frame):
        array = np.asarray(layer)
        if array.shape != (64, 64):
            raise DriverError(
                f"observation frame layer {index} has shape {array.shape}, "
                "expected (64, 64)"
            )
        if not np.issubdtype(array.dtype, np.integer):
            raise DriverError(
                f"observation frame layer {index} is not integer-valued"
            )
        if int(array.min()) < 0 or int(array.max()) > 15:
            raise DriverError(
                f"observation frame layer {index} contains palette values outside 0..15"
            )
        frame.append(array.tolist())

    guid = observation.guid
    if not isinstance(guid, str) or not guid:
        raise DriverError("observation guid must be a non-empty string")
    if not isinstance(observation.game_id, str) or not observation.game_id:
        raise DriverError("observation game_id must be a non-empty string")
    available_actions = observation.available_actions
    if not isinstance(available_actions, list) or any(
        isinstance(action, bool)
        or not isinstance(action, int)
        or action < 0
        or action > 7
        for action in available_actions
    ):
        raise DriverError("observation available_actions must contain ids in 0..7")
    if not isinstance(observation.full_reset, bool):
        raise DriverError("observation full_reset must be a boolean")

    return {
        "game_id": observation.game_id,
        "guid": guid,
        "frame": frame,
        "state": wire_state,
        "levels_completed": _integer(
            observation.levels_completed, "levels_completed"
        ),
        "win_levels": _integer(observation.win_levels, "win_levels"),
        "available_actions": available_actions,
        "full_reset": observation.full_reset,
    }


class DriveServer:
    def __init__(self, arcade: Arcade, games: list[dict[str, str]]) -> None:
        self.arcade = arcade
        self.games = games
        self.game_ids = {game["game_id"] for game in games}
        self.card_id: str | None = None
        self.environments: dict[str, Any] = {}
        self.latest: dict[str, FrameDataRaw] = {}
        self.scorecard: dict[str, Any] | None = None

    @staticmethod
    def _require_string(request: dict[str, Any], field: str) -> str:
        value = request.get(field)
        if not isinstance(value, str) or not value:
            raise DriverError(f"{field} must be a non-empty string")
        return value

    def _require_card(self, request: dict[str, Any]) -> str:
        if self.card_id is None:
            raise DriverError("no scorecard is open")
        card_id = self._require_string(request, "card_id")
        if card_id != self.card_id:
            raise DriverError("card_id does not match the open scorecard")
        return card_id

    def _environment(self, game_id: str) -> Any:
        if game_id not in self.game_ids:
            raise DriverError(f"game_id is not in the resolved game list: {game_id}")
        if self.card_id is None:
            raise DriverError("no scorecard is open")
        environment = self.environments.get(game_id)
        if environment is None:
            environment = self.arcade.make(game_id, scorecard_id=self.card_id)
            if environment is None:
                raise DriverError(f"toolkit could not create game {game_id}")
            self.environments[game_id] = environment
        return environment

    def handle(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise DriverError("protocol request must be a JSON object")
        op = request.get("op")
        if op == "list_games":
            return {"ok": {"games": self.games}}
        if op == "open_scorecard":
            if self.card_id is not None:
                raise DriverError("a scorecard is already open")
            self.card_id = self.arcade.open_scorecard()
            return {"ok": {"card_id": self.card_id}}
        if op == "reset":
            self._require_card(request)
            game_id = self._require_string(request, "game_id")
            environment = self._environment(game_id)
            observation = environment.reset()
            if observation is None:
                raise DriverError(f"toolkit reset failed for {game_id}")
            payload = observation_json(observation)
            self.latest[game_id] = observation
            return {"ok": {"observation": payload}}
        if op == "act":
            game_id = self._require_string(request, "game_id")
            environment = self._environment(game_id)
            current = self.latest.get(game_id)
            if current is None:
                raise DriverError(f"game {game_id} has not been reset")
            if current.state in (GameState.WIN, GameState.GAME_OVER):
                raise DriverError(
                    f"refusing act after terminal state {current.state.value} "
                    f"for {game_id}"
                )
            guid = self._require_string(request, "guid")
            if guid != current.guid:
                raise DriverError(
                    "act guid does not match the current environment guid"
                )
            action_id = request.get("action_id")
            if isinstance(action_id, bool) or not isinstance(action_id, int):
                raise DriverError("action_id must be an integer")
            try:
                action = GameAction.from_id(action_id)
            except ValueError as exc:
                raise DriverError(f"unknown action_id: {action_id}") from exc
            data = request.get("data")
            if not isinstance(data, dict):
                raise DriverError("act data must be a JSON object")
            step_data: dict[str, int] | None = None
            if action_id == 6:
                if set(data) != {"x", "y"}:
                    raise DriverError("ACTION6 data must contain exactly x and y")
                x = _integer(data["x"], "ACTION6 x")
                y = _integer(data["y"], "ACTION6 y")
                if x > 63 or y > 63:
                    raise DriverError("ACTION6 x and y must be in 0..63")
                step_data = {"x": x, "y": y}
            elif data:
                raise DriverError("simple action data must be empty")
            observation = environment.step(action, data=step_data)
            if observation is None:
                raise DriverError(f"toolkit action {action_id} failed for {game_id}")
            payload = observation_json(observation)
            self.latest[game_id] = observation
            return {"ok": {"observation": payload}}
        if op == "close_scorecard":
            card_id = self._require_card(request)
            final = self.arcade.close_scorecard(card_id)
            if final is None:
                raise DriverError(f"toolkit could not close scorecard {card_id}")
            self.scorecard = final.model_dump(mode="json")
            return {"ok": {"scorecard": self.scorecard}}
        raise DriverError(f"unknown drive-mode op: {op!r}")


def _serve_bridge(process: subprocess.Popen[str], server: DriveServer) -> None:
    assert process.stdout is not None
    assert process.stdin is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        try:
            if not line:
                raise DriverError("bridge emitted an empty protocol line")
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DriverError(f"bridge emitted invalid JSON: {exc}") from exc
            response = server.handle(request)
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            response = {"err": {"message": message, "kind": "failed"}}
            try:
                process.stdin.write(json.dumps(response, separators=(",", ":")) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            raise DriverError(message) from exc
        try:
            process.stdin.write(json.dumps(response, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise DriverError("bridge closed its request pipe unexpectedly") from exc


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _score_for_game(scorecard: dict[str, Any], game_id: str) -> float:
    environments = scorecard.get("environments", [])
    if not isinstance(environments, list):
        raise DriverError("toolkit scorecard environments is not a list")
    for environment in environments:
        if isinstance(environment, dict) and environment.get("id") == game_id:
            score = environment.get("score", 0.0)
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise DriverError(f"invalid toolkit score for {game_id}")
            return float(score)
    return 0.0


def _write_outputs(
    output_dir: Path,
    report_path: Path,
    scorecard: dict[str, Any],
) -> None:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DriverError(f"could not read bridge report {report_path}: {exc}") from exc
    if not isinstance(report, dict) or not isinstance(report.get("games"), list):
        raise DriverError("bridge report must contain a games list")

    games: list[dict[str, Any]] = []
    for game in report["games"]:
        if not isinstance(game, dict):
            raise DriverError("bridge report games entries must be objects")
        game_id = game.get("game_id")
        levels_completed = game.get("levels_completed")
        win_levels = game.get("win_levels")
        actions = game.get("actions")
        terminal_state = game.get("terminal_state")
        if not isinstance(game_id, str) or not game_id:
            raise DriverError("bridge report game_id must be a non-empty string")
        if terminal_state not in {
            "NOT_STARTED",
            "NOT_FINISHED",
            "WIN",
            "GAME_OVER",
            "RESET_FAILED",
        }:
            raise DriverError(
                f"bridge report has unknown terminal_state: {terminal_state!r}"
            )
        for field, value in (
            ("levels_completed", levels_completed),
            ("win_levels", win_levels),
            ("actions", actions),
        ):
            _integer(value, f"report {field}")
        won = terminal_state == "WIN" or (
            win_levels > 0 and levels_completed >= win_levels
        )
        games.append(
            {
                "game_id": game_id,
                "won": won,
                "levels_completed": levels_completed,
                "win_levels": win_levels,
                "actions": actions,
                "toolkit_score": _score_for_game(scorecard, game_id),
            }
        )

    total_levels = scorecard.get("total_levels", 0)
    toolkit_score = scorecard.get("score", 0.0)
    _integer(total_levels, "scorecard total_levels")
    if isinstance(toolkit_score, bool) or not isinstance(toolkit_score, (int, float)):
        raise DriverError("scorecard score must be numeric")

    scorecard_path = output_dir / "toolkit_scorecard.json"
    summary_path = output_dir / "local_summary.json"
    scorecard_path.write_text(
        json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "games": games,
        "totals": {
            "games_played": len(games),
            "games_won": sum(1 for game in games if game["won"]),
            "total_levels_completed": sum(
                game["levels_completed"] for game in games
            ),
            "total_levels": total_levels,
            "toolkit_scorecard_score": float(toolkit_score),
        },
        "arc3_local_report": str(report_path),
        "toolkit_scorecard": str(scorecard_path),
        "recordings_dir": str(output_dir / "recordings"),
        "profile_dir": str(output_dir / "profile"),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", required=True, dest="binary")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--games", default="all")
    parser.add_argument("--environments-dir", default="./environment_files")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-eval", choices=("true", "false"), default="false")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logger = _stderr_logger()
    output_dir = Path(args.output_dir)
    report_path = output_dir / "arc3_local_report.json"
    recordings_dir = output_dir / "recordings"
    profile_dir = output_dir / "profile"
    process: subprocess.Popen[str] | None = None
    try:
        arcade, games = _resolve_games(
            args.games,
            Path(args.environments_dir),
            args.allow_download,
            logger,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings_dir.mkdir(parents=True, exist_ok=True)
        profile_dir.mkdir(parents=True, exist_ok=True)
        command = [
            args.binary,
            "p2-arc3-bridge",
            "--mode",
            "drive",
            "--device",
            args.device,
            "--checkpoint",
            args.checkpoint,
            "--train-config",
            args.train_config,
            "--output",
            str(report_path),
            "--recordings-dir",
            str(recordings_dir),
        ]
        if args.profile_eval == "true":
            command.extend(("--profile-eval", "true"))
        if args.seed is not None:
            command.extend(("--seed", str(args.seed)))

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        server = DriveServer(arcade, games)
        _serve_bridge(process, server)
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise DriverError(f"bridge exited with status {return_code}")
        if not report_path.is_file():
            raise DriverError(
                f"bridge exited successfully without writing {report_path}"
            )
        if server.scorecard is None:
            raise DriverError(
                "bridge exited successfully without closing the scorecard"
            )
        _write_outputs(output_dir, report_path, server.scorecard)
        logger.info("ARC-AGI-3 local evaluation complete: %s", output_dir)
        return 0
    except Exception as exc:
        logger.error("ARC-AGI-3 local evaluation failed: %s", exc)
        return 1
    finally:
        if process is not None:
            _stop_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
