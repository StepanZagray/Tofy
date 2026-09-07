"""Tests for the deterministic loopback local-chat client."""

from __future__ import annotations

import json
import unittest
from typing import Self
from urllib.error import URLError

from tofy_arc3.frame_memory import encode_frame
from tofy_arc3.local_chat import SYSTEM_PROMPT, LocalChatAgent, LocalChatError


def observation(
    marker: int = 0, *, game_id: str = "game-secret", guid: str = "session-a"
) -> dict:
    return {
        "game_id": game_id,
        "guid": guid,
        "state": "NOT_FINISHED",
        "levels_completed": marker,
        "win_levels": 3,
        "frame": [[[marker % 16 for _ in range(64)] for _ in range(64)]],
        "available_actions": [1, 5, 6],
        "full_reset": False,
    }


class FakeResponse:
    def __init__(self, body: object, status: int = 200) -> None:
        self.status = status
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self.body[:limit]


class FakeOpen:
    def __init__(
        self,
        contents: list[str],
        *,
        usage: object | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self.contents = iter(contents)
        self.usage = (
            usage
            if usage is not None
            else {"prompt_tokens": 11, "completion_tokens": 4}
        )
        self.reasoning_content = reasoning_content
        self.calls: list[tuple[object, float]] = []

    def __call__(self, request: object, *, timeout: float) -> FakeResponse:
        self.calls.append((request, timeout))
        message = {"content": next(self.contents)}
        if self.reasoning_content is not None:
            message["reasoning_content"] = self.reasoning_content
        return FakeResponse({"choices": [{"message": message}], "usage": self.usage})


class LocalChatAgentTests(unittest.TestCase):
    def agent(
        self, contents: list[str], **kwargs: object
    ) -> tuple[LocalChatAgent, FakeOpen]:
        agent = LocalChatAgent("http://127.0.0.1:8080", "local-model", **kwargs)
        fake = FakeOpen(contents)
        agent._open = fake
        return agent, fake

    def test_constructs_deterministic_llama_server_request_without_identifiers(
        self,
    ) -> None:
        agent, fake = self.agent(
            ['```json\n{"action_id":6,"x":12,"y":63}\n```'],
            seed=7,
            timeout=2.5,
        )
        result = agent.choose_action(observation(10))

        self.assertEqual(result["action_id"], 6)
        self.assertEqual((result["x"], result["y"]), (12, 63))
        self.assertEqual(result["telemetry"]["token_usage"]["prompt_tokens"], 11)
        self.assertIn("```json", result["telemetry"]["raw_assistant_response"])
        self.assertGreaterEqual(result["telemetry"]["elapsed_seconds"], 0)
        request, timeout = fake.calls[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(timeout, 2.5)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertIsNone(request.get_header("Authorization"))
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "local-model")
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertFalse(payload["stream"])
        prompt = "\n".join(message["content"] for message in payload["messages"])
        self.assertNotIn("game-secret", prompt)
        self.assertNotIn("session-a", prompt)
        self.assertNotIn("title", prompt.lower())
        self.assertIn("ACTION1=up input", prompt)
        self.assertIn("directional inputs do not necessarily move a player", prompt)
        self.assertIn("no vision", prompt)
        self.assertIn("A" * 64, prompt)
        self.assertIsInstance(SYSTEM_PROMPT, str)

    def test_history_is_bounded_and_resets_for_a_new_session(self) -> None:
        agent, fake = self.agent(
            ['{"action_id":1}'] * 5,
            history_turns=2,
        )
        for marker in range(4):
            agent.choose_action(observation(marker))

        fourth_payload = json.loads(fake.calls[3][0].data)
        self.assertEqual(len(fourth_payload["messages"]), 6)
        user_messages = [
            message["content"]
            for message in fourth_payload["messages"]
            if message["role"] == "user"
        ]
        self.assertNotIn("levels_completed=0", "\n".join(user_messages))
        self.assertIn("levels_completed=1", user_messages[0])
        self.assertIn("levels_completed=3", user_messages[-1])

        agent.choose_action(observation(0, guid="session-b"))
        reset_payload = json.loads(fake.calls[4][0].data)
        self.assertEqual(
            [message["role"] for message in reset_payload["messages"]],
            ["system", "user"],
        )

    def test_raw_hex_remains_the_exact_default_observation_format(self) -> None:
        default_agent, default_open = self.agent(['{"action_id":1}'])
        explicit_agent, explicit_open = self.agent(
            ['{"action_id":1}'], frame_format="raw_hex"
        )
        current = observation(2)
        default_agent.choose_action(current)
        explicit_agent.choose_action(current)

        self.assertEqual(default_open.calls[0][0].data, explicit_open.calls[0][0].data)
        payload = json.loads(default_open.calls[0][0].data)
        expected_observation = (
            "Visible observation:\n"
            "state=NOT_FINISHED\n"
            "levels_completed=2\n"
            "win_levels=3\n"
            "full_reset=false\n"
            "available_actions=[1,5,6]\n"
            "Each frame row is 64 lossless hexadecimal palette symbols (0-F); x is the "
            "column and y is the row.\n"
            "layer 0:\n" + "\n".join(["2" * 64] * 64)
        )
        self.assertEqual(payload["messages"][-1]["content"], expected_observation)

    def test_compact_format_retains_all_layers_and_bottom_row(self) -> None:
        current = observation()
        first = [[0 for _ in range(64)] for _ in range(64)]
        first[63] = list(range(16)) * 4
        second = [[(x + y) % 2 for x in range(64)] for y in range(64)]
        current["frame"] = [first, second]
        agent, fake = self.agent(['{"action_id":1}'], frame_format="compact")

        agent.choose_action(current)
        prompt = json.loads(fake.calls[0][0].data)["messages"][-1]["content"]

        self.assertIn(f"layer 0 compact:\n{encode_frame(first)}\n", prompt)
        self.assertIn(f"layer 1 compact:\n{encode_frame(second)}\n", prompt)
        self.assertIn("Pn:<row> repeats a raw row n times", prompt)
        self.assertIn("Rn:<runs> repeats an RLE row n times", prompt)
        self.assertIn("components layer 0:", prompt)
        self.assertIn("components layer 1: total=4096 shown=32 truncated=4064", prompt)
        self.assertNotIn("game-secret", prompt)
        self.assertNotIn("session-a", prompt)

    def test_rejects_invalid_frame_format(self) -> None:
        for frame_format in ("raw", "compact ", "", None, []):
            with (
                self.subTest(frame_format=frame_format),
                self.assertRaisesRegex(ValueError, "frame_format"),
            ):
                LocalChatAgent(
                    "http://127.0.0.1:8080", "model", frame_format=frame_format
                )

    def test_baseline_config_accepts_compact_and_preserves_raw_default(self) -> None:
        from tofy_arc3.run_baseline import (
            DriverError,
            screen_contract,
            validate_config,
        )

        def config(frame_format: object = None) -> dict:
            agent = {"kind": "local_chat", "seed": 0}
            if frame_format is not None:
                agent["frame_format"] = frame_format
            value = {
                "games": ["ab12-01234567"],
                "evidence_class": "implementation_smoke",
                "seed": 0,
                "max_level_retries": 0,
                "max_actions_per_game": 1,
                "max_actions_per_level": 1,
                "max_seconds_per_game": 1,
                "max_seconds_total": 1,
                "decision_timeout": 1,
                "limitations": ["configuration validation fixture"],
                "agent": agent,
            }
            value["expected_screen_contract_sha256"] = screen_contract(value)
            return value

        validate_config(config())
        validate_config(config("compact"))
        with self.assertRaisesRegex(DriverError, "frame_format"):
            validate_config(config("raw"))

    def test_terminal_observation_is_retained_without_http(self) -> None:
        agent, fake = self.agent(
            ['{"action_id":1}', '{"action_id":5}'], history_turns=1
        )
        agent.choose_action(observation(0))
        terminal = observation(0)
        terminal["state"] = "GAME_OVER"
        terminal["available_actions"] = []
        agent.observe_terminal(terminal)
        self.assertEqual(len(fake.calls), 1)

        agent.choose_action(observation(0))
        payload = json.loads(fake.calls[1][0].data)
        self.assertEqual(
            [message["role"] for message in payload["messages"]],
            ["system", "user", "assistant", "user", "user"],
        )
        self.assertIn("state=GAME_OVER", payload["messages"][-2]["content"])

        with self.assertRaisesRegex(ValueError, "terminal observation"):
            agent.observe_terminal(observation())

    def test_preserves_separate_reasoning_content_as_data(self) -> None:
        agent = LocalChatAgent("http://127.0.0.1:8080", "model")
        fake = FakeOpen(['{"action_id":1}'], reasoning_content="private reasoning text")
        agent._open = fake
        result = agent.choose_action(observation())
        self.assertEqual(
            result["telemetry"]["reasoning_content"], "private reasoning text"
        )

    def test_rejects_invalid_or_illegal_actions_without_fallback(self) -> None:
        invalid = [
            '{"action_id":true}',
            '{"action_id":2}',
            '{"action_id":6}',
            '{"action_id":6,"x":true,"y":1}',
            '{"action_id":6,"x":64,"y":1}',
            '{"action_id":1,"x":0}',
            '{"action_id":1,"reason":"guess"}',
            '{"action_id":1,"action_id":5}',
            'answer: {"action_id":1}',
        ]
        for content in invalid:
            with self.subTest(content=content):
                agent, fake = self.agent([content])
                with self.assertRaises(LocalChatError):
                    agent.choose_action(observation())
                self.assertEqual(len(fake.calls), 1)
                self.assertEqual(agent._history, [])

        agent, fake = self.agent(['{"action_id":0}'])
        with self.assertRaisesRegex(LocalChatError, "unavailable action 0"):
            agent.choose_action({**observation(), "available_actions": [0, 1]})
        prompt = json.loads(fake.calls[0][0].data)["messages"][-1]["content"]
        self.assertIn("available_actions=[1]", prompt)

    def test_rejects_external_or_credentialed_endpoints(self) -> None:
        invalid = [
            "http://example.com:8080",
            "http://localhost:8080",
            "http://user:secret@127.0.0.1:8080",
            "http://localhost:8080/v1/chat/completions?token=secret",
            "http://[::1]:8080/#fragment",
            "file:///tmp/socket",
        ]
        for endpoint in invalid:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                LocalChatAgent(endpoint, "model")
        self.assertEqual(
            LocalChatAgent("http://[::1]:8080/v1/chat/completions", "model").endpoint,
            "http://[::1]:8080/v1/chat/completions",
        )

    def test_reports_http_transport_and_completion_envelope_errors(self) -> None:
        agent = LocalChatAgent("http://127.0.0.1:8080", "model")
        agent._open = lambda request, timeout: FakeResponse({}, status=503)
        with self.assertRaisesRegex(LocalChatError, "HTTP 503"):
            agent.choose_action(observation())

        def unavailable(request: object, *, timeout: float) -> FakeResponse:
            raise URLError("offline")

        agent._open = unavailable
        with self.assertRaisesRegex(LocalChatError, "chat request failed"):
            agent.choose_action(observation())

        agent._open = lambda request, timeout: FakeResponse({"choices": []})
        with self.assertRaisesRegex(LocalChatError, "invalid completion response"):
            agent.choose_action(observation())

    def test_validates_lossless_observation_shape_and_palette(self) -> None:
        agent, _ = self.agent(['{"action_id":1}'])
        bad = observation()
        bad["frame"][0][0][0] = 16
        with self.assertRaisesRegex(ValueError, "0..15"):
            agent.choose_action(bad)

    def test_transport_preserves_direct_and_wrapped_timeouts(self) -> None:
        for error in (TimeoutError("budget"), URLError(TimeoutError("socket"))):
            agent, _ = self.agent([])

            def unavailable(request, timeout, error=error):
                raise error

            agent._open = unavailable
            with self.assertRaises(TimeoutError):
                agent.choose_action(observation())
        bad = observation()
        bad["available_actions"] = [1, True]
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            agent.choose_action(bad)


if __name__ == "__main__":
    unittest.main()
