"""Tofy ARC-AGI-3 Kaggle agent.

Bundle the release ``tofy`` binary, ``ema.safetensors`` (or
``model.safetensors``), and ``config.json`` in one Kaggle dataset. A
``checkpoints/best/ema.safetensors`` layout is also accepted. Add that dataset
slug to the submission's ``notebooks/kernel-metadata.json`` field, for example
``"dataset_sources": ["your-user/tofy-arc3-bridge"]``. Kaggle mounts it below
``/kaggle/input`` and this agent finds it without internet access.

For exact local starter-kit validation commands:

    make setup
    TOFY_BRIDGE_BIN=/absolute/path/to/target/release/tofy \
      TOFY_BRIDGE_CHECKPOINT=/absolute/path/to/ema.safetensors \
      TOFY_BRIDGE_TRAIN_CONFIG=/absolute/path/to/config.json \
      TOFY_BRIDGE_DEVICE=cpu make play-local GAME=ls20 STEPS=80
    make submit

Copy this file verbatim over ``agent/my_agent.py`` before running those commands.
"""

from __future__ import annotations

import atexit
import json
import os
import random
import select
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, TextIO

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent


_BRIDGE_LOCK = threading.RLock()
_BRIDGE_PROCESS: subprocess.Popen[str] | None = None
_BRIDGE_STARTS = 0
_BRIDGE_DISABLED = False
_WARNING_PRINTED = False
_READY_TIMEOUT_SECONDS = 120.0
_RESPONSE_TIMEOUT_SECONDS = 120.0


def _warning(message: str) -> None:
    global _WARNING_PRINTED
    if not _WARNING_PRINTED:
        print(
            "WARNING: Tofy ARC-AGI-3 bridge unavailable; "
            f"using random legal actions: {message}",
            file=sys.stderr,
            flush=True,
        )
        _WARNING_PRINTED = True


def _existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


def _first_file(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _bundle_for_binary(
    binary: Path,
    root: Path,
    explicit_checkpoint: Path | None,
    explicit_config: Path | None,
) -> tuple[Path, Path, Path] | None:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        return None
    checkpoint = explicit_checkpoint or _first_file(
        [
            root / "ema.safetensors",
            root / "model.safetensors",
            root / "checkpoints" / "best" / "ema.safetensors",
            root / "checkpoints" / "best" / "model.safetensors",
            binary.parent / "ema.safetensors",
            binary.parent / "model.safetensors",
            binary.parent / "checkpoints" / "best" / "ema.safetensors",
        ]
    )
    config = explicit_config or _first_file(
        [root / "config.json", binary.parent / "config.json"]
    )
    if checkpoint is None or config is None:
        return None
    return binary, checkpoint, config


def _locate_bundle() -> tuple[Path, Path, Path, str] | None:
    checkpoint = _existing_path(os.environ.get("TOFY_BRIDGE_CHECKPOINT"))
    config = _existing_path(os.environ.get("TOFY_BRIDGE_TRAIN_CONFIG"))
    explicit_binary = _existing_path(os.environ.get("TOFY_BRIDGE_BIN"))
    device = os.environ.get("TOFY_BRIDGE_DEVICE", "cpu")

    if explicit_binary is not None:
        bundle = _bundle_for_binary(
            explicit_binary, explicit_binary.parent, checkpoint, config
        )
        if bundle is not None:
            return bundle[0], bundle[1], bundle[2], device

    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        for dataset_root in sorted(kaggle_root.iterdir()):
            if not dataset_root.is_dir():
                continue
            for binary in sorted(dataset_root.rglob("tofy")):
                bundle = _bundle_for_binary(
                    binary, dataset_root, checkpoint, config
                )
                if bundle is not None:
                    return bundle[0], bundle[1], bundle[2], device

    local_binary = Path.cwd() / "target" / "release" / "tofy"
    bundle = _bundle_for_binary(local_binary, Path.cwd(), checkpoint, config)
    if bundle is not None:
        return bundle[0], bundle[1], bundle[2], device
    return None


def _read_line(stream: TextIO, timeout: float) -> str:
    readable, _, _ = select.select([stream], [], [], timeout)
    if not readable:
        raise RuntimeError(f"timed out after {timeout:.0f}s")
    line = stream.readline()
    if not line:
        status = _BRIDGE_PROCESS.poll() if _BRIDGE_PROCESS is not None else None
        raise RuntimeError(f"bridge closed stdout (status={status})")
    return line


def _stop_bridge_locked(*, graceful: bool = False) -> None:
    global _BRIDGE_PROCESS
    process = _BRIDGE_PROCESS
    _BRIDGE_PROCESS = None
    if process is None:
        return
    if graceful and process.poll() is None and process.stdin is not None:
        try:
            process.stdin.write('{"op":"shutdown"}\n')
            process.stdin.flush()
            if process.stdout is not None:
                _read_line(process.stdout, 5.0)
            process.wait(timeout=5)
        except Exception:
            pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
    if process.stdout is not None:
        try:
            process.stdout.close()
        except OSError:
            pass


def _start_bridge_locked() -> subprocess.Popen[str] | None:
    global _BRIDGE_PROCESS, _BRIDGE_STARTS, _BRIDGE_DISABLED
    if _BRIDGE_DISABLED:
        return None
    if _BRIDGE_PROCESS is not None and _BRIDGE_PROCESS.poll() is None:
        return _BRIDGE_PROCESS
    if _BRIDGE_PROCESS is not None:
        _stop_bridge_locked()
    if _BRIDGE_STARTS >= 2:
        _BRIDGE_DISABLED = True
        return None

    bundle = _locate_bundle()
    if bundle is None:
        _BRIDGE_DISABLED = True
        _warning(
            "could not find an executable tofy binary with a checkpoint and config.json"
        )
        return None
    binary, checkpoint, config, device = bundle
    _BRIDGE_STARTS += 1
    try:
        process = subprocess.Popen(
            [
                str(binary),
                "p2-arc3-bridge",
                "--mode",
                "serve",
                "--device",
                device,
                "--checkpoint",
                str(checkpoint),
                "--train-config",
                str(config),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        _BRIDGE_PROCESS = process
        assert process.stdout is not None
        ready = json.loads(_read_line(process.stdout, _READY_TIMEOUT_SECONDS))
        if ready != {"ready": {"protocol": 1, "mode": "serve"}}:
            raise RuntimeError(f"unexpected ready response: {ready!r}")
        return process
    except Exception:
        _stop_bridge_locked()
        if _BRIDGE_STARTS >= 2:
            _BRIDGE_DISABLED = True
        raise


def _json_frame(frame: Any) -> list[list[list[int]]]:
    if not isinstance(frame, list) or not frame:
        raise ValueError("observation frame must contain at least one layer")
    layers: list[list[list[int]]] = []
    for layer in frame:
        if hasattr(layer, "tolist"):
            layer = layer.tolist()
        if not isinstance(layer, list) or len(layer) != 64:
            raise ValueError("observation frame layers must be 64x64")
        rows: list[list[int]] = []
        for row in layer:
            if not isinstance(row, list) or len(row) != 64:
                raise ValueError("observation frame layers must be 64x64")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > 15
                for value in row
            ):
                raise ValueError("observation palette values must be integers in 0..15")
            rows.append(row)
        layers.append(rows)
    return layers


def _observation(frame: FrameData) -> dict[str, Any]:
    states = {
        GameState.NOT_PLAYED: "NOT_STARTED",
        GameState.NOT_FINISHED: "NOT_FINISHED",
        GameState.WIN: "WIN",
        GameState.GAME_OVER: "GAME_OVER",
    }
    if frame.state not in states:
        raise ValueError(f"unknown game state: {frame.state!r}")
    return {
        "game_id": frame.game_id,
        "guid": frame.guid or "",
        "frame": _json_frame(frame.frame),
        "state": states[frame.state],
        "levels_completed": frame.levels_completed,
        "win_levels": frame.win_levels,
        "available_actions": frame.available_actions,
        "full_reset": frame.full_reset,
    }


def _request_action(observation: dict[str, Any]) -> dict[str, Any] | None:
    global _BRIDGE_DISABLED
    last_error = "bridge unavailable"
    with _BRIDGE_LOCK:
        while not _BRIDGE_DISABLED:
            try:
                process = _start_bridge_locked()
                if process is None:
                    return None
                assert process.stdin is not None
                assert process.stdout is not None
                request = {"op": "observe", "observation": observation}
                process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                process.stdin.flush()
                response = json.loads(
                    _read_line(process.stdout, _RESPONSE_TIMEOUT_SECONDS)
                )
                if not isinstance(response, dict) or not isinstance(
                    response.get("ok"), dict
                ):
                    raise RuntimeError(f"bridge returned an error: {response!r}")
                return response["ok"]
            except Exception as exc:
                last_error = str(exc) or exc.__class__.__name__
                _stop_bridge_locked()
                if _BRIDGE_STARTS >= 2:
                    _BRIDGE_DISABLED = True
        _warning(last_error)
        return None


def _telemetry_summary(telemetry: Any) -> dict[str, Any]:
    if not isinstance(telemetry, dict):
        return {}
    summary: dict[str, Any] = {}
    for key, value in telemetry.items():
        if len(summary) >= 8:
            break
        if not isinstance(key, str):
            continue
        if value is None or isinstance(value, (bool, int, float)):
            summary[key[:48]] = value
        elif isinstance(value, str):
            summary[key[:48]] = value[:120]
    return summary


def _action_from_bridge(payload: dict[str, Any]) -> GameAction:
    action_id = payload.get("action_id")
    if isinstance(action_id, bool) or not isinstance(action_id, int):
        raise ValueError("bridge action_id is not an integer")
    action = GameAction.from_id(action_id)
    if action_id == 6:
        x, y = payload.get("x"), payload.get("y")
        if (
            isinstance(x, bool)
            or not isinstance(x, int)
            or isinstance(y, bool)
            or not isinstance(y, int)
            or not 0 <= x <= 63
            or not 0 <= y <= 63
        ):
            raise ValueError("bridge ACTION6 coordinates are not in 0..63")
        action.set_data({"x": x, "y": y})
    action.reasoning = {
        "source": "tofy",
        "telemetry": _telemetry_summary(payload.get("telemetry")),
    }
    return action


def _random_legal_action(frame: FrameData) -> GameAction:
    legal: list[GameAction] = []
    for action_id in frame.available_actions:
        if action_id == 0:
            continue
        try:
            legal.append(GameAction.from_id(action_id))
        except (TypeError, ValueError):
            continue
    if not legal:
        legal = [action for action in GameAction if action is not GameAction.RESET]
    action = random.choice(legal)
    if action.is_complex():
        action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
    action.reasoning = {"source": "uniform-random-fallback"}
    return action


def _shutdown_bridge() -> None:
    with _BRIDGE_LOCK:
        _stop_bridge_locked(graceful=True)


atexit.register(_shutdown_bridge)


class MyAgent(Agent):
    """Choose actions with Tofy's frozen bridge, with a safe random fallback."""

    MAX_ACTIONS = 80

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET
        try:
            payload = _request_action(_observation(latest_frame))
            if payload is not None:
                action = _action_from_bridge(payload)
                if (
                    latest_frame.available_actions
                    and action.value not in latest_frame.available_actions
                ):
                    raise ValueError(f"bridge chose unavailable action {action.value}")
                return action
        except Exception as exc:
            global _BRIDGE_DISABLED
            with _BRIDGE_LOCK:
                _BRIDGE_DISABLED = True
                _stop_bridge_locked()
                _warning(str(exc) or exc.__class__.__name__)
        return _random_legal_action(latest_frame)
