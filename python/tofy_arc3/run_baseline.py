# SPDX-License-Identifier: MIT-0
"""Evaluate fixed agent configurations on the official offline ARC engine.

Run with PYTHONPATH=python python -m tofy_arc3.run_baseline --config FILE
--output-dir NEW_DIRECTORY. Environment downloads are a separate preparation
step; evaluation never downloads games or invokes a remote model provider.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import selectors
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.request import ProxyHandler, build_opener

from arc_agi import OperationMode
from arcengine import GameAction

from .run_local import DriverError, _arcade, _stderr_logger, observation_json

ACTION_ACCOUNTING = "engine_charged_including_retry_reset_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DriverError(f"{name} must be a positive number")
    if not math.isfinite(value) or value <= 0:
        raise DriverError(f"{name} must be finite and positive")
    return float(value)


def validate_config(config: dict[str, Any]) -> None:
    games = config.get("games")
    if (
        not isinstance(games, list)
        or not games
        or any(
            not isinstance(game, str)
            or not re.fullmatch(r"[a-z0-9]{4}-[a-f0-9]{8}", game)
            for game in games
        )
        or len(set(games)) != len(games)
    ):
        raise DriverError("games must be a nonempty list of unique versioned IDs")
    if config.get("evidence_class") not in ("implementation_smoke", "exploratory"):
        raise DriverError("this baseline screen cannot claim promotion evidence")
    for field in ("seed", "max_level_retries"):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DriverError(f"{field} must be a nonnegative integer")
    for field in ("max_actions_per_game", "max_actions_per_level"):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DriverError(f"{field} must be a positive integer")
    for field in ("max_seconds_per_game", "max_seconds_total", "decision_timeout"):
        positive_number(config.get(field), field)
    if config.get("agent", {}).get("kind") not in ("tofy", "local_chat"):
        raise DriverError("agent.kind must be tofy or local_chat")
    if not config.get("limitations"):
        raise DriverError("record the reference agent's known limitations")
    agent = config["agent"]
    allowed = {"kind", "seed"}
    if agent["kind"] == "tofy":
        allowed |= {
            "binary",
            "checkpoint",
            "train_config",
            "device",
            "policy",
            "phase_a_calibration",
        }
    else:
        allowed |= {
            "model",
            "model_file",
            "server_binary",
            "max_tokens",
            "history_turns",
            "context_size",
            "prompt_batch",
            "physical_batch",
            "gpu_layers",
            "reasoning_budget",
        }
    if set(agent) - allowed:
        raise DriverError(f"unsupported agent fields: {sorted(set(agent) - allowed)}")
    if agent.get("seed", config["seed"]) != config["seed"]:
        raise DriverError("this screen uses one explicit environment/model seed")
    if config.get("expected_screen_contract_sha256") != screen_contract(config):
        raise DriverError(
            "screen population/budgets do not match the preregistered contract digest"
        )


def screen_contract(config: dict[str, Any]) -> str:
    fields = (
        "games",
        "seed",
        "max_level_retries",
        "max_actions_per_game",
        "max_actions_per_level",
        "max_seconds_per_game",
        "max_seconds_total",
        "decision_timeout",
    )
    encoded = json.dumps(
        {
            **{key: config[key] for key in fields},
            "action_accounting": ACTION_ACCOUNTING,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class DeadlineExceeded(TimeoutError):
    """A total operation budget expired, rather than one socket read."""


def bounded_call(call: Callable[[], Any], seconds: float) -> Any:
    """Hard total deadline, including a potentially slow HTTP response body."""

    def expired(_signum: int, _frame: Any) -> None:
        raise DeadlineExceeded("agent total response deadline exceeded")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    try:
        return call()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


class ManagedChatAgent:
    """The evaluator owns the exact model-serving process, weights and flags."""

    def __init__(self, config: dict[str, Any], directory: Path, timeout: float):
        from .local_chat import LocalChatAgent

        self.timeout = timeout
        self.process: subprocess.Popen[bytes] | None = None
        self.log = (directory / "server.stderr").open("wb")
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        endpoint = f"http://127.0.0.1:{port}"
        command = [
            config["server_binary"],
            "--model",
            config["model_file"],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--alias",
            config["model"],
            "--ctx-size",
            str(config.get("context_size", 32768)),
            "--parallel",
            "1",
            "--batch-size",
            str(config.get("prompt_batch", 1024)),
            "--ubatch-size",
            str(config.get("physical_batch", 512)),
            "--gpu-layers",
            str(config.get("gpu_layers", 99)),
            "--flash-attn",
            "on",
            "--fit",
            "off",
            "--reasoning-budget",
            str(config.get("reasoning_budget", 256)),
            "--reasoning-format",
            "deepseek",
            "--jinja",
        ]
        write_json(directory / "server-command.json", command)
        try:
            self.process = subprocess.Popen(
                command, stdout=self.log, stderr=self.log, start_new_session=True
            )
            write_json(
                directory / "server-process.json",
                {"pid": self.process.pid, "endpoint": endpoint},
            )
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic() + 120
            while True:
                if self.process.poll() is not None:
                    raise DriverError(
                        f"model server exited with status {self.process.returncode}"
                    )
                try:
                    with opener.open(endpoint + "/v1/models", timeout=1) as response:
                        models = json.loads(response.read(1024 * 1024))
                    if config["model"] not in {item["id"] for item in models["data"]}:
                        raise DriverError(
                            "server model alias does not match registered configuration"
                        )
                    write_json(directory / "server-models.json", models)
                    break
                except DeadlineExceeded:
                    raise
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "model server failed its 120-second startup deadline"
                        )
                    time.sleep(0.1)
            self.client = LocalChatAgent(
                endpoint,
                config["model"],
                seed=config.get("seed", 0),
                max_tokens=config.get("max_tokens", 1024),
                timeout=timeout,
                history_turns=config.get("history_turns", 4),
            )
        except BaseException:
            self.close()
            raise

    def choose_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.client.timeout = self.timeout
        return self.client.choose_action(observation)

    def observe_terminal(self, observation: dict[str, Any]) -> None:
        self.client.observe_terminal(observation)

    def close(self) -> None:
        if self.process is not None:
            stop_process(self.process)
            self.process = None
        self.log.close()


def validate_action(
    action: dict[str, Any], observation: dict[str, Any]
) -> dict[str, int]:
    action_id = action.get("action_id")
    if isinstance(action_id, bool) or not isinstance(action_id, int):
        raise DriverError("action_id must be an integer")
    if action_id == 0 or action_id not in observation["available_actions"]:
        raise DriverError(f"agent returned unavailable action {action_id}")
    data = {}
    if action_id == 6:
        for name in ("x", "y"):
            value = action.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 64
            ):
                raise DriverError(f"ACTION6 {name} must be an integer in 0..63")
            data[name] = value
    elif action.get("x") is not None or action.get("y") is not None:
        raise DriverError("simple actions must not contain coordinates")
    return data


class TofyAgent:
    """One serve process per game; no fallback if the bridge fails."""

    def __init__(self, config: dict[str, Any], directory: Path, timeout: float):
        self.timeout = timeout
        self.process: subprocess.Popen[bytes] | None = None
        self.buffer = bytearray()
        self.in_flight = False
        self.log = (directory / "bridge.stderr").open("wb")
        runtime = json.loads(Path(config["train_config"]).read_text())
        # GPU locks and adaptation must not write into a sealed training root.
        runtime["output_dir"] = str(directory / "runtime")
        runtime_path = directory / "runtime-config.json"
        write_json(runtime_path, runtime)
        command = [
            config["binary"],
            "p2-arc3-bridge",
            "--mode",
            "serve",
            "--checkpoint",
            config["checkpoint"],
            "--train-config",
            str(runtime_path),
            "--device",
            config.get("device", "cuda:0"),
            "--policy",
            config.get("policy", "greedy"),
            "--seed",
            str(config.get("seed", 0)),
        ]
        if config.get("phase_a_calibration"):
            command += ["--phase-a-calibration", config["phase_a_calibration"]]
        write_json(directory / "bridge-command.json", command)
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                start_new_session=True,
            )
            write_json(directory / "bridge-process.json", {"pid": self.process.pid})
            ready = self._read(timeout)
            if ready.get("ready") != {"protocol": 1, "mode": "serve"}:
                raise DriverError(f"unexpected bridge readiness: {ready}")
        except BaseException:
            self.close()
            raise

    def _read(self, timeout: float) -> dict[str, Any]:
        assert self.process and self.process.stdout
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self.buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("Tofy bridge response deadline exceeded")
                block = os.read(self.process.stdout.fileno(), 65536)
                if not block:
                    raise DriverError(f"Tofy bridge exited: {self.process.poll()}")
                self.buffer.extend(block)
                if len(self.buffer) > 8 * 1024 * 1024:
                    raise DriverError("oversized bridge response")
        line, _, rest = self.buffer.partition(b"\n")
        self.buffer = bytearray(rest)
        response = json.loads(line)
        if not isinstance(response, dict) or "err" in response:
            raise DriverError(f"bridge error: {response}")
        return response

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        assert self.process and self.process.stdin
        self.in_flight = True
        self.process.stdin.write(json.dumps(request).encode() + b"\n")
        self.process.stdin.flush()
        response = self._read(self.timeout)
        if not isinstance(response.get("ok"), dict):
            raise DriverError("bridge response has no ok object")
        self.in_flight = False
        return response["ok"]

    def choose_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self._request({"op": "observe", "observation": observation})

    def observe_terminal(self, observation: dict[str, Any]) -> None:
        # Resolve the pending factual action before reset or teardown. Any
        # returned action is deliberately not executed by this notification.
        self.choose_action(observation)

    def finish_observation(self, observation: dict[str, Any], reason: str) -> None:
        response = self._request(
            {"op": "finish_observation", "observation": observation, "reason": reason}
        )
        if response != {"finished": True}:
            raise DriverError("bridge did not acknowledge final observation")

    def close(self) -> None:
        process = self.process
        if process is not None:
            if process.poll() is None and not self.in_flight:
                try:
                    self.timeout = min(self.timeout, 5)
                    self._request({"op": "shutdown"})
                    process.wait(timeout=5)
                except (
                    OSError,
                    ValueError,
                    DriverError,
                    TimeoutError,
                    subprocess.TimeoutExpired,
                ):
                    pass
            stop_process(process)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
            self.process = None
        self.log.close()


def play_game(
    environment: Any,
    agent: Any,
    config: dict[str, Any],
    trace: Any,
    deadline: float,
    cleanup_deadline: float | None = None,
) -> dict[str, Any]:
    """The evaluator alone owns resets, budgets, terminal handling and scoring."""
    observation = observation_json(environment.observation_space)
    guid = observation["guid"]
    game_id = observation["game_id"]
    actions = resets = decisions = 0
    per_level: dict[int, int] = {}
    retries: dict[int, int] = {}
    needs_notification = False
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0}
    token_calls = 0
    incomplete_calls = 0

    def finalization_budget() -> float:
        remaining = (
            cleanup_deadline if cleanup_deadline is not None else deadline + 5
        ) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "suite budget exhausted before final observation ingestion"
            )
        return min(5, remaining)

    def record(kind: str, **fields: Any) -> None:
        trace.write(json.dumps({"kind": kind, **fields}, separators=(",", ":")) + "\n")
        trace.flush()

    record("initial", observation=observation)
    while True:
        level = observation["levels_completed"]
        state = observation["state"]
        if state in ("GAME_OVER", "WIN") and needs_notification:
            bounded_call(
                lambda observation=observation: agent.observe_terminal(
                    copy.deepcopy(observation)
                ),
                finalization_budget(),
            )
            needs_notification = False
        if state == "WIN":
            reason = "win"
            break
        if time.monotonic() >= deadline:
            reason = "time_limit"
            break
        if actions + resets >= config["max_actions_per_game"]:
            reason = "action_limit"
            break
        if per_level.get(level, 0) >= config["max_actions_per_level"]:
            reason = "level_action_limit"
            break
        if state == "GAME_OVER":
            if retries.get(level, 0) >= config["max_level_retries"]:
                reason = "retry_limit"
                break
            retries[level] = retries.get(level, 0) + 1
            result = environment.reset()
            resets += 1
            per_level[level] = per_level.get(level, 0) + 1
            kind = "reset"
            fields: dict[str, Any] = {}
        else:
            if state != "NOT_FINISHED":
                raise DriverError(f"unexpected active state {state}")
            agent.timeout = min(config["decision_timeout"], deadline - time.monotonic())
            started = time.monotonic()
            try:
                decision = bounded_call(
                    lambda observation=observation: agent.choose_action(
                        copy.deepcopy(observation)
                    ),
                    agent.timeout,
                )
            except TimeoutError:
                record(
                    "decision_timeout",
                    game_budget_exhausted=time.monotonic() >= deadline,
                )
                if time.monotonic() < deadline:
                    raise
                # No returned action was executed. The process may still be
                # computing; close it without issuing a second request.
                needs_notification = False
                incomplete_calls += 1
                reason = "time_limit"
                break
            needs_notification = False
            decisions += 1
            data = validate_action(decision, observation)
            if config["agent"]["kind"] == "local_chat":
                usage = decision.get("telemetry", {}).get("token_usage", {})
                for key in token_totals:
                    value = usage.get(key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise DriverError(f"chat response missing valid {key}")
                    token_totals[key] += value
                token_calls += 1
            if time.monotonic() >= deadline:
                record("unexecuted_decision", decision=decision)
                reason = "time_limit"
                break
            result = environment.step(
                GameAction.from_id(decision["action_id"]), data=data or None
            )
            actions += 1
            per_level[level] = per_level.get(level, 0) + 1
            kind = "action"
            fields = {
                "decision": decision,
                "decision_seconds": time.monotonic() - started,
            }
        if result is None:
            raise DriverError(f"engine returned no observation for {kind}")
        next_observation = observation_json(result)
        record(kind, **fields, observation=next_observation)
        if next_observation["guid"] != guid or next_observation["game_id"] != game_id:
            raise DriverError("engine changed game/session identity")
        if (
            next_observation["full_reset"]
            or next_observation["levels_completed"] < level
        ):
            raise DriverError("unexpected whole-game reset or level regression")
        observation = next_observation
        needs_notification = True
    if needs_notification and hasattr(agent, "finish_observation"):
        # State ingestion only; no new policy decision or environment action.
        agent.timeout = finalization_budget()
        bounded_call(
            lambda: agent.finish_observation(copy.deepcopy(observation), reason),
            agent.timeout,
        )
        record("final_bridge_flush", executed=False)
    return {
        "game_id": game_id,
        "actions": actions + resets,
        "non_reset_actions": actions,
        "resets": resets,
        "decisions": decisions,
        "levels_completed": observation["levels_completed"],
        "win_levels": observation["win_levels"],
        "won": state == "WIN",
        "stop_reason": reason,
        "actions_by_level": per_level,
        "token_usage": (
            {
                **token_totals,
                "completed_calls": token_calls,
                "incomplete_calls_without_usage": incomplete_calls,
            }
            if config["agent"]["kind"] == "local_chat"
            else "not_applicable"
        ),
    }


def make_agent(config: dict[str, Any], directory: Path, timeout: float) -> Any:
    if config["kind"] == "tofy":
        return TofyAgent(config, directory, timeout)
    return ManagedChatAgent(config, directory, timeout)


def evaluate(
    arcade: Any,
    config: dict[str, Any],
    output: Path,
    factory: Callable[..., Any] = make_agent,
) -> dict[str, Any]:
    started = time.monotonic()
    suite_deadline = started + config["max_seconds_total"]
    report: dict[str, Any] = {
        "status": "running",
        "evidence_class": config["evidence_class"],
        "screen_contract_sha256": screen_contract(config),
        "action_accounting": ACTION_ACCOUNTING,
        "limitations": config["limitations"],
        "games": [],
    }
    card_id = arcade.open_scorecard()
    failure: Exception | None = None
    try:
        for game_id in config["games"]:
            if time.monotonic() >= suite_deadline:
                raise TimeoutError("suite budget expired before all registered games")
            directory = output / game_id
            directory.mkdir()
            begin = time.monotonic()
            agent = None
            try:
                # make initializes the game. A second RESET here would create
                # another run in the engine scorecard and violate the contract.
                environment = arcade.make(
                    game_id, seed=config["seed"], scorecard_id=card_id
                )
                if environment is None:
                    raise DriverError(f"cannot load registered game {game_id}")
                agent = bounded_call(
                    lambda directory=directory: factory(
                        config["agent"], directory, config["decision_timeout"]
                    ),
                    min(120, suite_deadline - time.monotonic()),
                )
                gameplay_start = time.monotonic()
                with (directory / "trajectory.jsonl").open("w") as trace:
                    result = play_game(
                        environment,
                        agent,
                        config,
                        trace,
                        min(
                            suite_deadline,
                            gameplay_start + config["max_seconds_per_game"],
                        ),
                        suite_deadline,
                    )
                result["startup_seconds"] = gameplay_start - begin
                result["gameplay_seconds"] = time.monotonic() - gameplay_start
                report["games"].append(result)
            finally:
                if agent is not None:
                    agent.close()
            result["elapsed_seconds"] = time.monotonic() - begin
            write_json(output / "report.json", report)
        report["status"] = "complete_pending_analysis"
    except Exception as exc:  # noqa: BLE001 -- Preserve failure artifacts and close the scorecard.
        failure = exc
        report["status"] = "failed_integrity_or_evaluation"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            final = arcade.close_scorecard(card_id)
            if final is None:
                raise DriverError("engine did not close scorecard")
            scorecard = final.model_dump(mode="json")
            scorecard.pop("api_key", None)
            write_json(output / "toolkit_scorecard.json", scorecard)
            score = scorecard.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise DriverError("invalid engine score")
            report["engine_score"] = score
            if failure is None:
                charged = {
                    game["id"]: game["actions"] for game in scorecard["environments"]
                }
                for game in report["games"]:
                    if charged.get(game["game_id"]) != game["actions"]:
                        raise DriverError(
                            "reported actions disagree with engine charged actions"
                        )
                if scorecard["total_actions"] != sum(
                    game["actions"] for game in report["games"]
                ):
                    raise DriverError(
                        "engine total actions disagree with completed game reports"
                    )
            report["score_semantics"] = (
                "arc-agi root score, native 0..100 scale; local public subset"
            )
        except Exception as exc:  # noqa: BLE001 -- Preserve failure artifacts and close the scorecard.
            failure = failure or exc
            report["status"] = "failed_integrity_or_evaluation"
            report["scorecard_error"] = str(exc)
        report["elapsed_seconds"] = time.monotonic() - started
        if report["elapsed_seconds"] > config["max_seconds_total"]:
            failure = failure or TimeoutError(
                "suite elapsed time exceeded registered cap"
            )
            report["status"] = "failed_integrity_or_evaluation"
            report["budget_error"] = (
                "suite elapsed time exceeded registered cap including cleanup"
            )
        write_json(output / "report.json", report)
    if failure:
        raise DriverError(
            report.get(
                "error",
                report.get(
                    "scorecard_error", report.get("budget_error", "evaluation failed")
                ),
            )
        ) from failure
    return report


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    if dirty:
        raise DriverError("baseline requires a clean, reviewed source checkout")
    identities = {}
    for item in config.get("artifacts", []):
        path = Path(item["path"])
        actual = sha256(path)
        if actual != item["sha256"]:
            raise DriverError(f"artifact hash mismatch: {path.name}")
        identities[str(path)] = actual
    if not identities:
        raise DriverError(
            "register hashes for the model/checkpoint and inference binary"
        )
    agent = config["agent"]
    required = (
        ("binary", "checkpoint", "train_config")
        if agent["kind"] == "tofy"
        else ("model_file", "server_binary")
    )
    if agent.get("phase_a_calibration"):
        required += ("phase_a_calibration",)
    for field in required:
        if str(Path(agent[field])) not in identities:
            raise DriverError(f"agent.{field} must have a registered artifact hash")
    environment_dir = Path(config["environments_dir"])
    environment_hashes = {}
    for game in config["games"]:
        base, version = game.split("-", 1)
        directory = environment_dir / base / version
        if not (directory / "metadata.json").is_file():
            raise DriverError(f"registered public game is not cached: {game}")
        environment_hashes[game] = {
            str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        }
    return {
        "revision": revision,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("arc-agi", "arcengine", "numpy")
        },
        "artifacts": identities,
        "environment_hashes": environment_hashes,
        "screen_contract_sha256": screen_contract(config),
        "runner_transmitted_input_contract": "frames/state/available actions/level feedback only; no game source or scorecard transmitted; not an OS sandbox",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    config["agent"].setdefault("seed", config["seed"])

    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"evaluation interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    try:
        write_json(output / "provenance.json", provenance(config))
        # Public offline engine needs no account credential. Prevent inherited
        # API configuration from being serialized into a scorecard.
        saved = os.environ.pop("ARC_API_KEY", None)
        try:
            arcade = _arcade(
                OperationMode.OFFLINE,
                Path(config["environments_dir"]),
                _stderr_logger(),
            )
        finally:
            if saved is not None:
                os.environ["ARC_API_KEY"] = saved
        actual_games = {info.game_id for info in arcade.get_environments()}
        if actual_games != set(config["games"]):
            raise DriverError(
                "use an environment directory containing exactly the registered game versions"
            )
        report = evaluate(arcade, config, output)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "engine_score": report["engine_score"],
                    "games": len(report["games"]),
                    "output": str(output),
                }
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 -- Preserve failure artifacts and close the scorecard.
        if not (output / "report.json").exists():
            write_json(
                output / "report.json",
                {"status": "failed_integrity_or_evaluation", "error": str(exc)},
            )
        print(f"baseline failed: {exc}", file=sys.stderr)
        return 1
    finally:
        # No child remains alive when the finalized tree is hashed.
        entries = {
            str(p.relative_to(output)): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(output / "manifest.json", entries)


if __name__ == "__main__":
    raise SystemExit(main())
