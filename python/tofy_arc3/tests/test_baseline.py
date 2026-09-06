"""Tests for scientifically material engine/agent lifecycle boundaries."""

import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from arcengine import GameState
from tofy_arc3.run_baseline import (
    DriverError,
    evaluate,
    play_game,
    screen_contract,
    validate_action,
    validate_config,
)


def raw(state=GameState.NOT_FINISHED, level=0, *, full_reset=False):
    return SimpleNamespace(
        game_id="test-12345678",
        guid="same-session",
        state=state,
        levels_completed=level,
        win_levels=2,
        full_reset=full_reset,
        frame=[[[0] * 64 for _ in range(64)]],
        available_actions=[1, 6],
    )


def config():
    result = {
        "games": ["test-12345678"],
        "seed": 7,
        "max_level_retries": 1,
        "max_actions_per_game": 5,
        "max_actions_per_level": 5,
        "max_seconds_per_game": 10,
        "max_seconds_total": 20,
        "decision_timeout": 1,
        "evidence_class": "implementation_smoke",
        "limitations": "test double",
        "agent": {"kind": "tofy"},
    }
    result["expected_screen_contract_sha256"] = screen_contract(result)
    return result


class FakeEnvironment:
    def __init__(self, outcomes, events):
        self.observation_space = raw()
        self.outcomes = iter(outcomes)
        self.events = events
        self.reset_count = 0

    def step(self, action, data=None):
        self.events.append("step")
        self.observation_space = next(self.outcomes)
        return self.observation_space

    def reset(self):
        self.events.append("reset")
        self.reset_count += 1
        self.observation_space = raw()
        return self.observation_space


class FakeAgent:
    def __init__(self, events, decision=None):
        self.events = events
        self.decision = {"action_id": 1} if decision is None else decision

    def choose_action(self, observation):
        self.events.append("choose")
        return self.decision

    def observe_terminal(self, observation):
        self.events.append("notify_" + observation["state"])

    def close(self):
        self.events.append("agent_close")


class FakeArcade:
    def __init__(self, environment, events):
        self.environment, self.events = environment, events
        self.calls = []

    def open_scorecard(self):
        self.events.append("card_open")
        return "card"

    def make(self, game_id, seed, scorecard_id):
        self.calls.append((game_id, seed, scorecard_id))
        return self.environment

    def close_scorecard(self, card_id):
        self.events.append("card_close")
        return SimpleNamespace(
            model_dump=lambda **_: {
                "api_key": "must-not-persist",
                "score": 42.0,
                "environments": [
                    {"score": 42.0, "runs": [{"score": 0}, {"score": 42.0}]}
                ],
            }
        )


class BaselineTests(unittest.TestCase):
    def test_initial_observation_is_reused_and_terminal_feedback_precedes_reset(self):
        events = []
        environment = FakeEnvironment(
            [raw(GameState.GAME_OVER), raw(GameState.WIN, 2)], events
        )
        trace = io.StringIO()
        result = play_game(
            environment, FakeAgent(events), config(), trace, time.monotonic() + 10
        )
        self.assertEqual(
            events,
            [
                "choose",
                "step",
                "notify_GAME_OVER",
                "reset",
                "choose",
                "step",
                "notify_WIN",
            ],
        )
        self.assertEqual(environment.reset_count, 1)
        self.assertEqual(result["actions"], 2)
        self.assertEqual(result["resets"], 1)
        self.assertTrue(result["won"])
        self.assertEqual(
            [json.loads(line)["kind"] for line in trace.getvalue().splitlines()],
            ["initial", "action", "reset", "action"],
        )

    def test_action_budget_never_executes_one_extra_action(self):
        events = []
        environment = FakeEnvironment([raw(), raw()], events)
        cfg = config()
        cfg["max_actions_per_game"] = 1
        result = play_game(
            environment, FakeAgent(events), cfg, io.StringIO(), time.monotonic() + 10
        )
        self.assertEqual(events, ["choose", "step"])
        self.assertEqual(result["stop_reason"], "action_limit")

    def test_illegal_action_fails_before_touching_engine(self):
        events = []
        with self.assertRaises(DriverError):
            play_game(
                FakeEnvironment([], events),
                FakeAgent(events, {"action_id": 5}),
                config(),
                io.StringIO(),
                time.monotonic() + 10,
            )
        self.assertEqual(events, ["choose"])

    def test_final_frame_is_ingested_without_an_extra_decision(self):
        events = []
        agent = FakeAgent(events)
        agent.finish_observation = lambda observation, reason: events.append(
            ("finish", observation["levels_completed"], reason)
        )
        cfg = config()
        cfg["max_actions_per_game"] = 1
        result = play_game(
            FakeEnvironment([raw(level=1)], events),
            agent,
            cfg,
            io.StringIO(),
            time.monotonic() + 10,
        )
        self.assertEqual(events, ["choose", "step", ("finish", 1, "action_limit")])
        self.assertEqual(result["decisions"], 1)

    def test_game_budget_timeout_does_not_execute_an_action(self):
        events = []
        trace = io.StringIO()
        with (
            patch(
                "tofy_arc3.run_baseline.time.monotonic", side_effect=[0, 0, 0, 11, 11]
            ),
            patch("tofy_arc3.run_baseline.bounded_call", side_effect=TimeoutError),
        ):
            result = play_game(
                FakeEnvironment([], events), FakeAgent(events), config(), trace, 10
            )
        self.assertEqual(result["stop_reason"], "time_limit")
        self.assertEqual(result["actions"], 0)
        self.assertEqual(events, [])
        self.assertTrue(
            json.loads(trace.getvalue().splitlines()[-1])["game_budget_exhausted"]
        )

    def test_whole_game_reset_is_rejected(self):
        with self.assertRaisesRegex(DriverError, "whole-game reset"):
            play_game(
                FakeEnvironment([raw(full_reset=True)], []),
                FakeAgent([]),
                config(),
                io.StringIO(),
                time.monotonic() + 10,
            )

    def test_scorecard_is_canonical_and_closed_after_agent(self):
        events = []
        arcade = FakeArcade(FakeEnvironment([raw(GameState.WIN, 2)], events), events)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = evaluate(arcade, config(), root, lambda *_: FakeAgent(events))
            self.assertEqual(result["engine_score"], 42.0)
            self.assertEqual(result["status"], "complete_pending_analysis")
            self.assertNotIn(
                "api_key", json.loads((root / "toolkit_scorecard.json").read_text())
            )
        self.assertEqual(events[-2:], ["agent_close", "card_close"])
        self.assertEqual(arcade.calls, [("test-12345678", 7, "card")])

    def test_agent_failure_preserves_failed_report_and_closes_everything(self):
        events = []
        arcade = FakeArcade(FakeEnvironment([], events), events)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(DriverError):
                evaluate(
                    arcade,
                    config(),
                    root,
                    lambda *_: FakeAgent(events, {"action_id": True}),
                )
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["status"], "failed_integrity_or_evaluation")
            self.assertIn("integer", report["error"])
        self.assertEqual(events[-2:], ["agent_close", "card_close"])

    def test_config_rejects_unversioned_duplicate_and_traversal_ids(self):
        for games in (["test"], ["test-12345678"] * 2, ["../../other-12345678"]):
            cfg = config()
            cfg["games"] = games
            with self.assertRaises(DriverError):
                validate_config(cfg)
        validate_config(config())

    def test_action_validation_rejects_bool_coordinates_and_reset(self):
        observation = {"available_actions": [0, 1, 6]}
        for action in (
            {"action_id": 0},
            {"action_id": True},
            {"action_id": 6, "x": True, "y": 0},
            {"action_id": 6, "x": 64, "y": 0},
            {"action_id": 1, "x": 1},
        ):
            with self.assertRaises(DriverError):
                validate_action(action, observation)


if __name__ == "__main__":
    unittest.main()
